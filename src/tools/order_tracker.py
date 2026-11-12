"""
Order Tracker & Live Position Monitoring
========================================
Manages actual orders placed by the user, tracks performance over time,
toggles TRACKED / UNTRACKED states, and monitors open positions against
profit targets, stop losses, and time-stop rules.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Literal
from loguru import logger
from sqlalchemy import desc, func

from config.settings import get_settings
from src.database.connection import get_db
from src.database.models import (
    OrderStatus,
    PlacedOrder,
    TrackingStatus,
    TradeRecommendation,
    TradeStatus,
)
from src.tools.market_data import OptionChainResult, OptionContract, get_contract_mark_price, get_option_chain


def _enum_str(val: Any) -> str:
    """Safely convert Enum or string values to plain string (Python 3.11+ safe)."""
    return val.value if hasattr(val, "value") else str(val)


def _infer_short_strike(
    strike_price: float,
    structure: str | None,
    short_strike_price: float | None = None,
    contract_symbol: str | None = None,
    notes: str | None = None,
) -> float | None:
    """
    Resolve the short leg strike for vertical spreads from explicit field,
    contract_symbol (e.g. XOM_SPREAD_165_170), or notes (e.g. '165/170').
    """
    if short_strike_price is not None:
        return float(short_strike_price)
    if not structure or "SPREAD" not in structure.upper():
        return None

    # 1. Check contract_symbol like XOM_SPREAD_165_170
    if contract_symbol:
        m = re.search(r"SPREAD[_-](\d+(?:\.\d+)?)[_-](\d+(?:\.\d+)?)", contract_symbol, re.IGNORECASE)
        if m:
            s1, s2 = float(m.group(1)), float(m.group(2))
            if abs(s1 - strike_price) < 0.05:
                return s2
            if abs(s2 - strike_price) < 0.05:
                return s1

    # 2. Check notes like '165/170', '165C/170C', or 'short 170C'
    if notes:
        m_slash = re.search(r"(\d+(?:\.\d+)?)\s*[CP]?/\s*(\d+(?:\.\d+)?)\s*[CP]?", notes, re.IGNORECASE)
        if m_slash:
            s1, s2 = float(m_slash.group(1)), float(m_slash.group(2))
            if abs(s1 - strike_price) < 0.05:
                return s2
            if abs(s2 - strike_price) < 0.05:
                return s1
        m_short = re.search(r"short\s+\$?(\d+(?:\.\d+)?)", notes, re.IGNORECASE)
        if m_short:
            return float(m_short.group(1))

    return None


def _compute_spread_net_price(
    long_leg: OptionContract,
    short_leg: OptionContract,
    width: float | None = None,
) -> tuple[float, float, float, str]:
    """
    Calculate the net debit/value of a vertical debit spread:
        net_spread_price = long_leg_price - short_leg_price
    Uses consistent leg pricing (live mids if both legs have active bid/ask,
    else consistent last_price, else fallback marks) and clamps to [0, width].
    """
    if long_leg.bid > 0 and long_leg.ask > 0 and short_leg.bid > 0 and short_leg.ask > 0:
        long_px = round((long_leg.bid + long_leg.ask) / 2, 4)
        short_px = round((short_leg.bid + short_leg.ask) / 2, 4)
        source = "live_mid_spread"
    elif (long_leg.last_price or 0) > 0 and (short_leg.last_price or 0) > 0:
        long_px = round(float(long_leg.last_price), 4)
        short_px = round(float(short_leg.last_price), 4)
        source = "last_price_spread"
    else:
        long_px = get_contract_mark_price(long_leg)
        short_px = get_contract_mark_price(short_leg)
        source = "mark_spread"

    net = round(long_px - short_px, 4)
    if net < 0:
        net = 0.0
    if width is not None and width > 0 and net > width:
        net = round(width, 4)
    return net, long_px, short_px, source


def _refresh_order_live_quote(
    order: PlacedOrder,
    chain_cache: dict[tuple[str, date], OptionChainResult | None] | None = None,
) -> dict[str, Any]:
    """
    Fetch live market pricing for a placed order (supporting both single-leg options
    and vertical debit spreads) and update order.current_price, current_value, pnl, pnl_percent.
    """
    cache_key = (order.symbol.upper(), order.expiration_date)
    if chain_cache is not None and cache_key in chain_cache:
        chain = chain_cache[cache_key]
    else:
        chain = get_option_chain(order.symbol, exact_expiration=order.expiration_date)
        if chain_cache is not None:
            chain_cache[cache_key] = chain

    if not chain:
        return {"pricing_source": "unavailable"}

    contracts_list = chain.calls if order.option_type.upper() == "CALL" else chain.puts
    long_leg = next((c for c in contracts_list if abs(c.strike - order.strike_price) < 0.05), None)
    if not long_leg:
        return {"pricing_source": "long_leg_not_found", "underlying_price": chain.underlying_price}

    is_spread = "SPREAD" in (order.structure or "").upper()
    short_strike = _infer_short_strike(
        strike_price=order.strike_price,
        structure=order.structure,
        short_strike_price=order.short_strike_price,
        contract_symbol=order.contract_symbol,
        notes=order.notes,
    )
    if short_strike is not None and order.short_strike_price is None:
        order.short_strike_price = short_strike

    if is_spread:
        if short_strike is None:
            logger.warning(
                "Order #{} is {} but short_strike_price is unknown; skipping single-leg price overwrite.",
                order.id,
                order.structure,
            )
            return {"pricing_source": "missing_short_strike", "underlying_price": chain.underlying_price}

        short_leg = next((c for c in contracts_list if abs(c.strike - short_strike) < 0.05), None)
        if not short_leg:
            logger.warning(
                "Order #{} short leg (${}) not found in chain {}; skipping overwrite.",
                order.id,
                short_strike,
                order.expiration_date,
            )
            return {"pricing_source": "short_leg_not_found", "underlying_price": chain.underlying_price}

        width = abs(short_strike - order.strike_price)
        net_price, long_px, short_px, source = _compute_spread_net_price(long_leg, short_leg, width=width)
        order.current_price = round(net_price, 4)
        order.current_value = round(order.current_price * 100 * order.contracts, 2)
        order.pnl = round(order.current_value - order.total_cost, 2)
        if order.total_cost > 0:
            order.pnl_percent = round((order.pnl / order.total_cost) * 100, 2)
        return {
            "pricing_source": source,
            "long_leg_price": long_px,
            "short_leg_price": short_px,
            "underlying_price": chain.underlying_price,
        }

    # Single-leg option
    mark_px = get_contract_mark_price(long_leg)
    if mark_px > 0:
        order.current_price = round(mark_px, 4)
        order.current_value = round(order.current_price * 100 * order.contracts, 2)
        order.pnl = round(order.current_value - order.total_cost, 2)
        if order.total_cost > 0:
            order.pnl_percent = round((order.pnl / order.total_cost) * 100, 2)
        return {
            "pricing_source": "single_leg_mark",
            "long_leg_price": mark_px,
            "short_leg_price": None,
            "underlying_price": chain.underlying_price,
        }
    return {"pricing_source": "zero_quote", "underlying_price": chain.underlying_price}


def _parse_timeframe_cutoff(timeframe: str) -> datetime | None:
    """Convert timeframe string to cutoff datetime."""
    now = datetime.now()
    tf = timeframe.lower().strip()
    if tf in ("7d", "last 7 days", "7_days"):
        return now - timedelta(days=7)
    elif tf in ("14d", "last 14 days", "14_days"):
        return now - timedelta(days=14)
    elif tf in ("1m", "last month", "1_month", "30d"):
        return now - timedelta(days=30)
    elif tf in ("1q", "last quarter", "quarter", "90d"):
        return now - timedelta(days=90)
    elif tf in ("1y", "last year", "year", "365d"):
        return now - timedelta(days=365)
    elif tf in ("all", "all_time"):
        return None
    return now - timedelta(days=7)  # default to last 7 days


# ── Order Placement ────────────────────────────────────────────────────────────

def record_placed_order(
    symbol: str,
    fill_price: float,
    contracts: int = 1,
    strike_price: float | None = None,
    short_strike_price: float | None = None,
    expiration_date: date | str | None = None,
    structure: str = "LONG_CALL",
    option_type: str = "CALL",
    target_exit_price: float | None = None,
    stop_loss_price: float | None = None,
    contract_symbol: str | None = None,
    recommendation_id: int | None = None,
    notes: str | None = None,
    tracking_status: str = "TRACKED",
) -> dict[str, Any]:
    """
    Record an actual order placed by the user in the database.
    By default, it is placed in TRACKED state.
    """
    with get_db() as session:
        # If recommendation_id is provided, pull missing fields
        rec = None
        if recommendation_id:
            rec = session.query(TradeRecommendation).filter(TradeRecommendation.id == recommendation_id).first()
            if rec:
                symbol = rec.symbol
                strike_price = strike_price or rec.strike_price
                short_strike_price = short_strike_price or rec.short_strike_price
                expiration_date = expiration_date or rec.expiration_date
                structure = structure or rec.structure or "LONG_CALL"
                option_type = option_type or rec.option_type
                contract_symbol = contract_symbol or rec.contract_symbol
                # Only inherit target/stop from recommendation if structure matches and fill is close to estimate
                same_structure = (rec.structure or "").upper() == (structure or "").upper()
                close_fill = (
                    rec.entry_price_estimate > 0
                    and abs(float(fill_price) - rec.entry_price_estimate) / rec.entry_price_estimate < 0.25
                )
                if same_structure and close_fill:
                    target_exit_price = target_exit_price or rec.target_exit_price
                    stop_loss_price = stop_loss_price or rec.stop_loss_price
                rec.status = TradeStatus.EXECUTED

        if isinstance(expiration_date, str):
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        elif expiration_date is None:
            # Fallback to next Friday if not specified
            today = date.today()
            target_friday = today + timedelta(days=(4 - today.weekday()) % 7 or 7)
            expiration_date = target_friday

        strike = float(strike_price or 100.0)
        resolved_short = _infer_short_strike(
            strike_price=strike,
            structure=structure,
            short_strike_price=short_strike_price,
            contract_symbol=contract_symbol,
            notes=notes,
        )
        fill = float(fill_price)
        total_cost = round(contracts * fill * 100, 2)

        # Calculate sensible default target/stop based on actual fill price & spread width
        settings = get_settings()
        if stop_loss_price is None or stop_loss_price >= fill:
            stop_loss_price = round(fill * (1 - settings.stop_loss_percent / 100), 2)
        if target_exit_price is None or target_exit_price <= fill:
            target_exit_price = round(fill * (1 + settings.target_gain_percent / 100), 2)
        if resolved_short is not None:
            spread_width = abs(resolved_short - strike)
            if spread_width > 0 and target_exit_price > spread_width:
                target_exit_price = round(spread_width * 0.90, 2)

        order = PlacedOrder(
            recommendation_id=recommendation_id,
            symbol=symbol.upper(),
            structure=structure,
            option_type=option_type.upper(),
            strike_price=strike,
            short_strike_price=resolved_short,
            expiration_date=expiration_date,
            contract_symbol=contract_symbol,
            contracts=contracts,
            fill_price=fill,
            total_cost=total_cost,
            target_exit_price=target_exit_price,
            stop_loss_price=stop_loss_price,
            current_price=fill,
            current_value=total_cost,
            pnl=0.0,
            pnl_percent=0.0,
            status=OrderStatus.OPEN.value,
            tracking_status=TrackingStatus(tracking_status.upper()).value,
            notes=notes,
        )
        session.add(order)
        session.flush()

        strike_label = f"${order.strike_price}/{order.short_strike_price}" if order.short_strike_price else f"${order.strike_price}"
        logger.info(
            "📝 Placed Order recorded: ID={} | {} {} {} qty={} fill=${:.2f} (Cost=${:.2f}, Tracking={})",
            order.id,
            order.symbol,
            order.structure,
            strike_label,
            order.contracts,
            order.fill_price,
            order.total_cost,
            order.tracking_status,
        )

        return _order_to_dict(order)


# ── Tracking & PnL Updates ─────────────────────────────────────────────────────

def set_order_tracking_state(order_id: int, tracking_status: str) -> dict[str, Any]:
    """
    Update tracking state of an order: 'TRACKED' or 'UNTRACKED'.
    """
    ts = tracking_status.upper().strip()
    if ts not in ("TRACKED", "UNTRACKED"):
        raise ValueError(f"Invalid tracking status: '{tracking_status}'. Allowed: TRACKED, UNTRACKED")

    with get_db() as session:
        order = session.query(PlacedOrder).filter(PlacedOrder.id == order_id).first()
        if not order:
            raise ValueError(f"Order #{order_id} not found")

        order.tracking_status = ts
        order.updated_at = datetime.utcnow()
        session.flush()

        logger.info("Order #{} tracking state changed to: {}", order_id, ts)
        return _order_to_dict(order)


def update_order_pnl(
    order_id: int,
    pnl: float | None = None,
    current_price: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Update the PnL of a placed order.
    If pnl is explicitly given, updates PnL directly.
    If current_price is given without pnl, calculates pnl from fill_price and contracts.
    If neither is given, fetches the latest market price from market data feeds.
    """
    with get_db() as session:
        order = session.query(PlacedOrder).filter(PlacedOrder.id == order_id).first()
        if not order:
            raise ValueError(f"Order #{order_id} not found")

        # 1. Explicit PnL provided
        if pnl is not None:
            order.pnl = round(float(pnl), 2)
            if order.total_cost > 0:
                order.pnl_percent = round((order.pnl / order.total_cost) * 100, 2)
            if current_price is not None:
                order.current_price = round(float(current_price), 2)
                order.current_value = round(order.current_price * 100 * order.contracts, 2)

        # 2. Current price provided, calculate PnL
        elif current_price is not None:
            cp = round(float(current_price), 2)
            order.current_price = cp
            order.current_value = round(cp * 100 * order.contracts, 2)
            order.pnl = round(order.current_value - order.total_cost, 2)
            if order.total_cost > 0:
                order.pnl_percent = round((order.pnl / order.total_cost) * 100, 2)

        # 3. Automatic market data quote fetch
        else:
            try:
                _refresh_order_live_quote(order)
            except Exception as exc:
                logger.warning("Could not auto-fetch live quote for #{}: {}", order_id, exc)

        if notes:
            order.notes = f"{order.notes or ''}\n{notes}".strip()

        order.updated_at = datetime.utcnow()
        session.flush()

        logger.info(
            "Updated Order #{}: Current Price=${:.2f} | PnL=${:+.2f} ({:+.1f}%)",
            order.id,
            order.current_price or 0.0,
            order.pnl,
            order.pnl_percent or 0.0,
        )
        return _order_to_dict(order)


def close_placed_order(order_id: int, exit_price: float, notes: str | None = None) -> dict[str, Any]:
    """Close an active placed order with final realized exit price."""
    with get_db() as session:
        order = session.query(PlacedOrder).filter(PlacedOrder.id == order_id).first()
        if not order:
            raise ValueError(f"Order #{order_id} not found")

        ep = float(exit_price)
        order.status = OrderStatus.CLOSED.value
        order.exit_price = ep
        order.current_price = ep
        order.current_value = round(ep * 100 * order.contracts, 2)
        order.pnl = round(order.current_value - order.total_cost, 2)
        if order.total_cost > 0:
            order.pnl_percent = round((order.pnl / order.total_cost) * 100, 2)
        order.closed_at = datetime.utcnow()
        if notes:
            order.notes = f"{order.notes or ''}\nExit: {notes}".strip()

        session.flush()
        logger.info("Order #{} closed at ${:.2f} | Realized PnL: ${:+.2f}", order.id, ep, order.pnl)
        return _order_to_dict(order)


# ── Live Position Monitoring ───────────────────────────────────────────────────

def monitor_placed_orders(tracking_only: bool = True) -> dict[str, Any]:
    """
    Monitor open placed orders against live market quotes, profit targets,
    stop-loss limits, and time-stop rules.
    """
    with get_db() as session:
        query = session.query(PlacedOrder).filter(PlacedOrder.status == OrderStatus.OPEN.value)
        if tracking_only:
            query = query.filter(PlacedOrder.tracking_status == TrackingStatus.TRACKED.value)

        orders = query.order_by(desc(PlacedOrder.placed_at)).all()
        if not orders:
            return {
                "active_orders_count": 0,
                "total_invested": 0.0,
                "current_value": 0.0,
                "total_pnl": 0.0,
                "total_pnl_percent": 0.0,
                "positions": [],
                "alerts": [],
                "summary": "No active tracked orders to monitor.",
            }

        positions = []
        alerts = []
        total_cost = 0.0
        total_val = 0.0
        chain_cache: dict[tuple[str, date], OptionChainResult | None] = {}

        for order in orders:
            # 1. Update quote if possible (supports single-leg & vertical debit spreads)
            dte = max(0, (order.expiration_date - date.today()).days)
            quote_meta: dict[str, Any] = {}
            try:
                quote_meta = _refresh_order_live_quote(order, chain_cache=chain_cache)
            except Exception as exc:
                logger.debug("Could not refresh live quote for #{}: {}", order.id, exc)

            cur_p = order.current_price if order.current_price is not None else order.fill_price
            cost = order.total_cost
            val = order.current_value if order.current_value is not None else cost
            pnl = order.pnl
            pnl_pct = order.pnl_percent or 0.0

            total_cost += cost
            total_val += val

            # Ensure target/stop are valid relative to actual fill_price and spread width
            settings = get_settings()
            if order.stop_loss_price is not None and order.stop_loss_price >= order.fill_price:
                order.stop_loss_price = round(order.fill_price * (1 - settings.stop_loss_percent / 100), 2)
            if order.short_strike_price is not None and order.target_exit_price is not None:
                width = abs(order.short_strike_price - order.strike_price)
                if width > 0 and order.target_exit_price > width:
                    order.target_exit_price = round(min(order.fill_price * (1 + settings.target_gain_percent / 100), width * 0.90), 2)

            # Check rules & triggers
            alert_type = "NORMAL"
            action_needed = "HOLD / MONITOR"

            # A. Profit Target Trigger
            if order.target_exit_price and cur_p >= order.target_exit_price:
                alert_type = "PROFIT_TARGET_HIT"
                action_needed = f"SELL TO CLOSE: Target reached at ${cur_p:.2f} ({pnl_pct:+.1f}%)"
                alerts.append(f"🎯 [{order.symbol}] Target reached (${cur_p:.2f} >= ${order.target_exit_price:.2f}) -> {pnl:+.2f} USD")

            # B. Stop Loss Trigger
            elif order.stop_loss_price and cur_p <= order.stop_loss_price:
                alert_type = "STOP_LOSS_HIT"
                action_needed = f"CUT LOSS: Stop triggered at ${cur_p:.2f} ({pnl_pct:+.1f}%)"
                alerts.append(f"🛑 [{order.symbol}] Stop loss triggered (${cur_p:.2f} <= ${order.stop_loss_price:.2f}) -> {pnl:+.2f} USD")

            # C. Time Stop Trigger (<= 2 DTE)
            elif dte <= 2:
                alert_type = "TIME_STOP_WARNING"
                action_needed = f"TIME STOP: Only {dte} DTE remaining. Close before terminal weekend theta decay."
                alerts.append(f"⏳ [{order.symbol}] Approaching expiry ({dte} DTE) -> Prepare to exit")

            # D. Gain Milestone
            elif pnl_pct >= 40.0:
                alert_type = "NEAR_TARGET"
                action_needed = f"PROTECT PROFIT: Sitting at +{pnl_pct:.1f}% gain. Consider trailing stop."

            pos_data = _order_to_dict(order)
            pos_data["dte"] = dte
            pos_data["alert_type"] = alert_type
            pos_data["action_needed"] = action_needed
            pos_data["long_leg_price"] = quote_meta.get("long_leg_price")
            pos_data["short_leg_price"] = quote_meta.get("short_leg_price")
            pos_data["pricing_source"] = quote_meta.get("pricing_source")
            positions.append(pos_data)

        session.commit()

        total_pnl = round(total_val - total_cost, 2)
        total_pnl_pct = round((total_pnl / total_cost) * 100, 2) if total_cost > 0 else 0.0

        return {
            "active_orders_count": len(positions),
            "total_invested": round(total_cost, 2),
            "current_value": round(total_val, 2),
            "total_pnl": total_pnl,
            "total_pnl_percent": total_pnl_pct,
            "positions": positions,
            "alerts": alerts,
            "summary": f"{len(positions)} active tracked positions | Total PnL: ${total_pnl:+.2f} ({total_pnl_pct:+.1f}%) | {len(alerts)} alerts active",
        }


# ── Queries & Grouping by Date ─────────────────────────────────────────────────

def get_placed_orders_grouped_by_date(
    timeframe: str = "all",
    tracking_status: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch placed orders filtered by timeframe and tracking state,
    grouped by date (newest first).
    """
    cutoff = _parse_timeframe_cutoff(timeframe)

    with get_db() as session:
        query = session.query(PlacedOrder)
        if cutoff:
            query = query.filter(PlacedOrder.placed_at >= cutoff)
        if tracking_status and tracking_status.upper() != "ALL":
            query = query.filter(PlacedOrder.tracking_status == tracking_status.upper())
        if status and status.upper() != "ALL":
            query = query.filter(PlacedOrder.status == status.upper())

        orders = query.order_by(desc(PlacedOrder.placed_at)).all()

        # Group by date string (YYYY-MM-DD)
        groups_dict: dict[str, list[dict[str, Any]]] = {}
        for o in orders:
            d_str = o.placed_at.strftime("%Y-%m-%d") if o.placed_at else "Unknown"
            if d_str not in groups_dict:
                groups_dict[d_str] = []
            groups_dict[d_str].append(_order_to_dict(o))

        grouped: list[dict[str, Any]] = []
        for d_str, order_list in groups_dict.items():
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                display = dt.strftime("%A, %b %d, %Y")
            except Exception:
                display = d_str
            grouped.append({
                "date": d_str,
                "display_date": display,
                "total_orders": len(order_list),
                "orders": order_list,
            })

        return grouped


def get_placed_order_by_id(order_id: int) -> dict[str, Any]:
    """Fetch a single placed order by primary key."""
    with get_db() as session:
        order = session.query(PlacedOrder).filter(PlacedOrder.id == order_id).first()
        if not order:
            raise ValueError(f"Order #{order_id} not found")
        return _order_to_dict(order)


def get_recommendation_by_id(rec_id: int) -> dict[str, Any]:
    """Fetch a single trade recommendation by primary key."""
    with get_db() as session:
        rec = session.query(TradeRecommendation).filter(TradeRecommendation.id == rec_id).first()
        if not rec:
            raise ValueError(f"Recommendation #{rec_id} not found")
        return _rec_to_dict(rec)


def update_recommendation_status(rec_id: int, status: str) -> dict[str, Any]:
    """Update recommendation workflow status (e.g. APPROVED, REJECTED, EXECUTED)."""
    target_status = TradeStatus(status.upper().strip())
    with get_db() as session:
        rec = session.query(TradeRecommendation).filter(TradeRecommendation.id == rec_id).first()
        if not rec:
            raise ValueError(f"Recommendation #{rec_id} not found")
        rec.status = target_status
        session.flush()
        logger.info("Recommendation #{} status updated to: {}", rec_id, target_status.value)
        return _rec_to_dict(rec)


def get_recommendations_grouped_by_date(
    timeframe: str = "all",
    status: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch trade recommendations filtered by timeframe (and optional status),
    grouped by date (newest first).
    """
    cutoff = _parse_timeframe_cutoff(timeframe)

    with get_db() as session:
        query = session.query(TradeRecommendation)
        if cutoff:
            query = query.filter(TradeRecommendation.created_at >= cutoff)
        if status and status.upper() != "ALL":
            query = query.filter(TradeRecommendation.status == TradeStatus(status.upper()))

        recs = query.order_by(desc(TradeRecommendation.created_at)).all()

        groups_dict: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            d_str = r.created_at.strftime("%Y-%m-%d") if r.created_at else "Unknown"
            if d_str not in groups_dict:
                groups_dict[d_str] = []
            groups_dict[d_str].append(_rec_to_dict(r))

        grouped: list[dict[str, Any]] = []
        for d_str, rec_list in groups_dict.items():
            # Sort recommendations on each date by profit confidence (highest first)
            rec_list.sort(
                key=lambda r: (
                    r.get("confidence_score_pct") or 0.0,
                    -(r.get("risk_score") or 5.0),
                    r.get("estimated_profit_pct") or 0.0,
                    r.get("id") or 0,
                ),
                reverse=True,
            )
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                display = dt.strftime("%A, %b %d, %Y")
            except Exception:
                display = d_str
            grouped.append({
                "date": d_str,
                "display_date": display,
                "total_recommendations": len(rec_list),
                "recommendations": rec_list,
            })

        return grouped


# ── Serializers ────────────────────────────────────────────────────────────────

def _order_to_dict(o: PlacedOrder) -> dict[str, Any]:
    dte = (o.expiration_date - date.today()).days if o.expiration_date else None
    return {
        "id": o.id,
        "uuid": o.uuid,
        "recommendation_id": o.recommendation_id,
        "symbol": o.symbol,
        "structure": o.structure,
        "option_type": o.option_type,
        "strike_price": o.strike_price,
        "short_strike_price": o.short_strike_price,
        "expiration_date": str(o.expiration_date),
        "dte": dte,
        "contract_symbol": o.contract_symbol,
        "contracts": o.contracts,
        "fill_price": o.fill_price,
        "total_cost": o.total_cost,
        "target_exit_price": o.target_exit_price,
        "stop_loss_price": o.stop_loss_price,
        "current_price": o.current_price,
        "current_value": o.current_value,
        "pnl": o.pnl,
        "pnl_percent": o.pnl_percent,
        "status": o.status,
        "tracking_status": o.tracking_status,
        "notes": o.notes,
        "exit_price": o.exit_price,
        "placed_at": o.placed_at.strftime("%Y-%m-%d %H:%M:%S") if o.placed_at else None,
        "closed_at": o.closed_at.strftime("%Y-%m-%d %H:%M:%S") if o.closed_at else None,
    }


def _rec_to_dict(r: TradeRecommendation) -> dict[str, Any]:
    dte = (r.expiration_date - date.today()).days if r.expiration_date else None
    return {
        "id": r.id,
        "uuid": r.uuid,
        "symbol": r.symbol,
        "underlying_price": r.underlying_price,
        "option_type": _enum_str(r.option_type),
        "structure": r.structure,
        "strike_price": r.strike_price,
        "short_strike_price": r.short_strike_price,
        "expiration_date": str(r.expiration_date),
        "dte": dte,
        "contract_symbol": r.contract_symbol,
        "entry_price_estimate": r.entry_price_estimate,
        "target_exit_price": r.target_exit_price,
        "stop_loss_price": r.stop_loss_price,
        "breakeven_price": r.breakeven_price,
        "estimated_profit_usd": r.estimated_profit_usd,
        "estimated_profit_pct": r.estimated_profit_pct,
        "max_contracts": r.max_contracts,
        "max_cost": r.max_cost,
        "delta": r.delta,
        "gamma": r.gamma,
        "theta": r.theta,
        "vega": r.vega,
        "implied_volatility": r.implied_volatility,
        "iv_rank": r.iv_rank,
        "confidence_rating": r.confidence_rating,
        "confidence_score_pct": r.confidence_score_pct,
        "rank": r.rank,
        "sector_category": r.sector_category,
        "sector_name": r.sector_name,
        "sector_etf": r.sector_etf,
        "sector_7d_change_pct": r.sector_7d_change_pct,
        "relative_strength_7d_pct": r.relative_strength_7d_pct,
        "why_selected": r.why_selected,
        "when_to_buy": r.when_to_buy,
        "sell_conditions": r.sell_conditions,
        "early_warning_signs": r.early_warning_signs,
        "political_catalysts": r.political_catalysts,
        "earnings_context": r.earnings_context,
        "company_news_context": r.company_news_context,
        "thesis": r.thesis,
        "risk_score": r.risk_score,
        "risk_rating": r.risk_rating,
        "status": _enum_str(r.status),
        "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
    }
