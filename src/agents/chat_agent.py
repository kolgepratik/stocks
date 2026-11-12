"""
Interactive Agent Chat Interface
================================
Empowers the user to converse with the system via natural language:
- Log actual placed orders (into TRACKED state by default)
- Monitor active placed orders against profit targets, stop losses, and time stops
- Toggle tracking state (TRACKED vs UNTRACKED) on demand
- Update the PNL field manually or via live market quotes
- Run options market scans and discuss sector catalysts
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config.settings import get_settings
from src.agents.llm_resilience import build_llm
from src.tools.order_tracker import (
    close_placed_order,
    get_placed_orders_grouped_by_date,
    get_recommendations_grouped_by_date,
    monitor_placed_orders,
    record_placed_order,
    set_order_tracking_state,
    update_order_pnl,
)
from src.database.connection import get_db
from src.database.models import PlacedOrder


# ── Agent Tools ────────────────────────────────────────────────────────────────

def _format_strikes(strike_price: float, short_strike_price: float | None = None) -> str:
    if short_strike_price is not None:
        return f"${strike_price:g}/${short_strike_price:g}"
    return f"${strike_price:g}"


@tool
def record_placed_order_tool(
    symbol: str,
    fill_price: float,
    contracts: int = 1,
    strike_price: float | None = None,
    short_strike_price: float | None = None,
    expiration_date: str | None = None,
    structure: str = "LONG_CALL",
    recommendation_id: int | None = None,
    notes: str | None = None,
) -> str:
    """
    Log an order that the user actually placed in their brokerage into the database.
    By default, the order is placed into the 'TRACKED' state for continuous monitoring.
    Parameters:
    - symbol: Stock ticker (e.g. 'XOM', 'NVDA')
    - fill_price: Executed net debit/premium per share (e.g. 2.10 for a debit spread, 3.40 for single leg)
    - contracts: Number of option contracts/spreads bought (default 1)
    - strike_price: Long option strike price (e.g. 165.0)
    - short_strike_price: Higher short strike price if structure is CALL_DEBIT_SPREAD (e.g. 170.0)
    - expiration_date: Option expiry string in YYYY-MM-DD format
    - structure: Option structure ('LONG_CALL' or 'CALL_DEBIT_SPREAD')
    - recommendation_id: Optional ID of the trade recommendation this order originated from
    - notes: Optional notes about entry
    """
    try:
        order = record_placed_order(
            symbol=symbol,
            fill_price=fill_price,
            contracts=contracts,
            strike_price=strike_price,
            short_strike_price=short_strike_price,
            expiration_date=expiration_date,
            structure=structure,
            recommendation_id=recommendation_id,
            notes=notes,
            tracking_status="TRACKED",
        )
        strikes = _format_strikes(order["strike_price"], order.get("short_strike_price"))
        return (
            f"✅ Order successfully logged in database!\n"
            f"• Order ID: #{order['id']}\n"
            f"• Ticker: {order['symbol']} {order['structure']} {strikes}\n"
            f"• Expiry: {order['expiration_date']} ({order.get('dte')} DTE)\n"
            f"• Quantity: {order['contracts']} contract{'s' if order['contracts'] != 1 else ''}\n"
            f"• Fill Price: ${order['fill_price']:.2f}/sh (${order['total_cost']:,.2f} total cost)\n"
            f"• Tracking State: {order['tracking_status']} (Active for monitoring)"
        )
    except Exception as exc:
        return f"❌ Failed to record order: {exc}"


@tool
def monitor_orders_tool(tracking_only: bool = True) -> str:
    """
    Check and monitor all active placed orders against current market prices.
    Calculates live PnL ($ and %), pricing spreads using (long leg - short leg),
    verifies profit targets, stop loss triggers, and expiration DTE warnings.
    """
    try:
        report = monitor_placed_orders(tracking_only=tracking_only)
        positions = report.get("positions", [])
        if not positions:
            return "No active tracked orders found to monitor."

        lines = [
            f"📊 **Live Position Monitoring Report** ({len(positions)} active tracked positions)",
            f"• Total Capital Invested: ${report['total_invested']:,.2f}",
            f"• Current Portfolio Value: ${report['current_value']:,.2f}",
            f"• Overall Unrealized PnL: **${report['total_pnl']:+,.2f} ({report['total_pnl_percent']:+.1f}%)**",
            "",
            "**Active Positions:**"
        ]

        for p in positions:
            pnl_sign = "+" if p['pnl'] >= 0 else ""
            status_indicator = "🟢" if p['pnl'] >= 0 else "🔴"
            alert = f" ⚠️ **{p.get('action_needed')}**" if p.get("alert_type") != "NORMAL" else ""
            strikes = _format_strikes(p["strike_price"], p.get("short_strike_price"))
            cur_px = p.get('current_price') if p.get('current_price') is not None else p['fill_price']
            leg_info = ""
            if p.get("short_leg_price") is not None and p.get("long_leg_price") is not None:
                leg_info = f" (Long ${p['long_leg_price']:.2f} - Short ${p['short_leg_price']:.2f})"
            lines.append(
                f"{status_indicator} **Order #{p['id']}: {p['symbol']} {p['structure']} {strikes}** (Exp: {p['expiration_date']}, {p.get('dte')} DTE)\n"
                f"   • Contracts: {p['contracts']} | Fill: ${p['fill_price']:.2f} | Current Net: ${cur_px:.2f}{leg_info}\n"
                f"   • Total Cost: ${p['total_cost']:,.2f} | Current Value: ${p.get('current_value', p['total_cost']):,.2f}\n"
                f"   • PnL: **{pnl_sign}${p['pnl']:.2f} ({pnl_sign}{p.get('pnl_percent', 0.0):.1f}%)**\n"
                f"   • Targets: Profit Target ${p.get('target_exit_price') or 'N/A'} | Stop Loss ${p.get('stop_loss_price') or 'N/A'}\n"
                f"   • Tracking: `{p['tracking_status']}`{alert}"
            )

        if report.get("alerts"):
            lines.append("\n🚨 **Trigger Alerts:**")
            for a in report["alerts"]:
                lines.append(f"   • {a}")

        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Monitoring failed: {exc}"


@tool
def set_order_tracking_tool(order_id: int | None = None, symbol: str | None = None, tracking_status: str = "UNTRACKED") -> str:
    """
    Change the tracking state of a placed order to 'TRACKED' or 'UNTRACKED'.
    By default orders are TRACKED, but the user can mark them as UNTRACKED or re-enable TRACKED.
    Parameters:
    - order_id: The specific ID of the placed order (e.g. 1)
    - symbol: Optional ticker symbol to look up if order_id is not known
    - tracking_status: 'TRACKED' or 'UNTRACKED'
    """
    try:
        ts = tracking_status.upper().strip()
        if ts not in ("TRACKED", "UNTRACKED"):
            return f"❌ Invalid tracking status '{tracking_status}'. Must be TRACKED or UNTRACKED."

        # If order_id not given, lookup latest open order for symbol
        target_id = order_id
        if not target_id and symbol:
            with get_db() as session:
                found = session.query(PlacedOrder).filter(
                    PlacedOrder.symbol == symbol.upper().strip(),
                    PlacedOrder.status == "OPEN"
                ).order_by(PlacedOrder.id.desc()).first()
                if found:
                    target_id = found.id

        if not target_id:
            return "❌ Please specify an order ID (e.g. order #1) or a symbol."

        res = set_order_tracking_state(order_id=target_id, tracking_status=ts)
        return (
            f"✅ Order #{res['id']} ({res['symbol']} ${res['strike_price']}) tracking state updated!\n"
            f"• Current Tracking State: `{res['tracking_status']}`\n"
            f"• State Meaning: {'Actively included in monitoring and automated alerts' if ts == 'TRACKED' else 'Excluded from automated monitoring'}"
        )
    except Exception as exc:
        return f"❌ Failed to update tracking state: {exc}"


@tool
def update_order_pnl_tool(
    order_id: int | None = None,
    symbol: str | None = None,
    pnl: float | None = None,
    current_price: float | None = None,
    notes: str | None = None,
) -> str:
    """
    Update the PNL field and current valuation of an order to track performance of recommendations over time.
    Can be updated manually (by supplying pnl or current_price) or automatically (queries live market feeds).
    Parameters:
    - order_id: Order ID to update
    - symbol: Optional symbol to locate order if ID is not known
    - pnl: Explicit dollar PnL in USD (e.g. 45.0 for +$45 gain, or -20.0 for -$20 loss)
    - current_price: Latest contract market price per share (e.g. 4.10)
    - notes: Optional update note
    """
    try:
        target_id = order_id
        if not target_id and symbol:
            with get_db() as session:
                found = session.query(PlacedOrder).filter(
                    PlacedOrder.symbol == symbol.upper().strip(),
                    PlacedOrder.status == "OPEN"
                ).order_by(PlacedOrder.id.desc()).first()
                if found:
                    target_id = found.id

        if not target_id:
            return "❌ Please specify an order ID or a symbol to update."

        res = update_order_pnl(order_id=target_id, pnl=pnl, current_price=current_price, notes=notes)
        pnl_sign = "+" if res['pnl'] >= 0 else ""
        return (
            f"✅ PnL successfully updated for Order #{res['id']} ({res['symbol']})!\n"
            f"• Current Price: ${res.get('current_price', 0.0):.2f}\n"
            f"• Current Value: ${res.get('current_value', 0.0):.2f}\n"
            f"• Total Cost: ${res['total_cost']:.2f}\n"
            f"• Updated PnL: **{pnl_sign}${res['pnl']:.2f} ({pnl_sign}{res.get('pnl_percent', 0.0):.1f}%)**\n"
            f"• Tracking Status: `{res['tracking_status']}`"
        )
    except Exception as exc:
        return f"❌ Failed to update PnL: {exc}"


@tool
def list_placed_orders_tool(timeframe: str = "7d", tracking_status: str = "ALL") -> str:
    """
    Retrieve past placed orders grouped by dates for a given timeframe (7d, 14d, 1m, 1q, 1y, all).
    """
    try:
        grouped = get_placed_orders_grouped_by_date(timeframe=timeframe, tracking_status=tracking_status)
        if not grouped:
            return f"No placed orders found for timeframe: {timeframe}."

        lines = [f"📋 **Placed Orders ({timeframe.upper()})**"]
        for group in grouped:
            lines.append(f"\n📅 **{group['display_date']}** ({group['total_orders']} order{'s' if group['total_orders'] != 1 else ''}):")
            for o in group["orders"]:
                pnl_sign = "+" if o['pnl'] >= 0 else ""
                strikes = _format_strikes(o["strike_price"], o.get("short_strike_price"))
                lines.append(
                    f"  • Order #{o['id']}: **{o['symbol']} {o['structure']} {strikes}** (Exp: {o['expiration_date']}) "
                    f"[{o['tracking_status']}] — Fill: ${o['fill_price']:.2f} (Cost: ${o['total_cost']:,.2f}) | "
                    f"PnL: {pnl_sign}${o['pnl']:.2f} ({pnl_sign}{o.get('pnl_percent', 0.0):.1f}%)"
                )
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Failed to list orders: {exc}"


@tool
def run_market_scan_tool(
    symbols: list[str] | None = None,
    budget: float | None = None,
    max_trade_cost: float | None = None,
    notify: bool = False,
) -> str:
    """
    Execute the 3-agent LangGraph options pipeline on demand to screen tickers,
    structure defined-risk CALL strategies (LONG_CALL / CALL_DEBIT_SPREAD), run Risk Sentinel checks,
    persist ranked recommendations to the database, and return the ranked advisory results.
    Parameters:
    - symbols: Optional list of ticker symbols (e.g. ['NVDA', 'OKLO', 'XOM']). Omit to scan all 3 specialized sectors.
    - budget: Optional budget override in USD (defaults to configured weekly budget, e.g. 500.0)
    - max_trade_cost: Optional max single-trade cost cap in USD
    - notify: Whether to also fire WhatsApp/Telegram mobile notifications (default False for interactive chat)
    """
    try:
        from src.workflow.graph import run_pipeline

        clean_symbols = [s.strip().upper() for s in symbols if s and s.strip()] if symbols else None
        final_state = run_pipeline(
            symbols=clean_symbols,
            weekly_budget_usd=budget,
            max_cost_per_trade_usd=max_trade_cost,
            notify=notify,
        )
        recs = final_state.get("final_recommendations") or []
        macro = final_state.get("macro_political_context") or {}
        saved_ids = final_state.get("saved_recommendation_ids") or []

        lines = [
            f"🚀 **On-Demand Pipeline Scan Complete** ({len(recs)} approved recommendation{'s' if len(recs) != 1 else ''})"
        ]
        if macro.get("us_political_macro_summary"):
            lines.append(f"\n🏛️ **Macro & Political Backdrop:** {macro['us_political_macro_summary']}")

        if not recs:
            lines.append("\nNo candidates passed all Risk Sentinel safety and budget rules during this scan.")
            return "\n".join(lines)

        for idx, rec in enumerate(recs, 1):
            strikes = _format_strikes(rec.get("strike", 0), rec.get("short_strike"))
            db_id = saved_ids[idx - 1] if idx - 1 < len(saved_ids) else "N/A"
            cat = rec.get("sector_category") or rec.get("sector_name") or "Market"
            lines.append(
                f"\n**Rank #{idx}: {rec.get('symbol')} [{cat}] {rec.get('structure', 'LONG_CALL')} {strikes}** "
                f"(Exp: {rec.get('expiration')}, {rec.get('dte', '?')} DTE | DB ID #{db_id})\n"
                f"• Entry Mid: ${rec.get('entry_price', 0):.2f}/sh | Contracts: {rec.get('max_contracts', 1)} | Total Cost: ${rec.get('max_cost', 0):,.2f}\n"
                f"• Profit Target: ${rec.get('target_exit_price', 0):.2f} (+{rec.get('estimated_profit_pct', 50):.0f}% / +${rec.get('estimated_profit_usd', 0):,.2f}) | Stop Loss: ${rec.get('stop_loss_price', 0):.2f}\n"
                f"• Confidence: {rec.get('confidence_rating', 'MEDIUM')} ({rec.get('confidence_score_pct', 50):.0f}%) | Risk: {rec.get('risk_rating', 'MODERATE')}\n"
                f"• Why Selected: {rec.get('why_selected') or rec.get('thesis', 'N/A')}\n"
                f"• Entry Trigger: {rec.get('when_to_buy', 'N/A')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.error("run_market_scan_tool failed: {}", exc)
        return f"❌ Market scan failed: {exc}"


# ── Chat Execution Loop ────────────────────────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """You are Antigravity Options AI — the intelligent co-pilot and portfolio manager for systematic options trading.

You have access to specialized tools to:
1. `record_placed_order_tool`: Log actual orders that the user executed in their brokerage (defaults to TRACKED state).
2. `monitor_orders_tool`: Monitor active open orders, pull live market quotes, calculate current PnL, and check profit targets / stop loss / time stops.
3. `set_order_tracking_tool`: Mark placed orders as TRACKED or UNTRACKED per user request.
4. `update_order_pnl_tool`: Update the PnL and valuation of any placed order (manually or automatically via live quote).
5. `list_placed_orders_tool`: Review past placed orders grouped by dates across timeframes (7d, 14d, 1m, 1q, 1y, all).
6. `run_market_scan_tool`: Run the 3-agent LangGraph options pipeline on demand to generate fresh ranked CALL/CALL_DEBIT_SPREAD recommendations.

GUIDELINES:
- When the user tells you about an order they placed (e.g. "I placed XOM at 3.40"), immediately call `record_placed_order_tool` to persist it in `trading.db`.
- When asked to "monitor orders", call `monitor_orders_tool` and summarize position health, unrealized PnL, and any active alerts.
- When asked to "mark as untracked" or "stop tracking", use `set_order_tracking_tool` with tracking_status="UNTRACKED".
- When asked to "update pnl", use `update_order_pnl_tool`.
- When asked to "scan", "generate recommendations", or "find new trades", immediately invoke `run_market_scan_tool` with any requested symbols/budget and present the ranked output.
- Maintain a disciplined, professional quantitative trading tone. Remind the user about the 4 exit rules (profit target, stop-loss, time-stop, invalidation) when discussing open positions.
- All numbers should be clearly formatted in USD and percentages."""

ALL_TOOLS = [
    record_placed_order_tool,
    monitor_orders_tool,
    set_order_tracking_tool,
    update_order_pnl_tool,
    list_placed_orders_tool,
    run_market_scan_tool,
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def run_chat_turn(user_message: str, chat_history: list[dict[str, str]] | None = None) -> str:
    """
    Run conversation turn with the agent, supporting tool calling and multi-turn resolution.
    """
    settings = get_settings()
    llm = build_llm(model_name=settings.anthropic_model, request_timeout=45)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages: list[Any] = [SystemMessage(content=_CHAT_SYSTEM_PROMPT)]

    # Add historical messages if provided
    if chat_history:
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    # Tool calling loop (up to 4 turns)
    for _ in range(4):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return _extract_text(response.content)

        logger.info(
            "ChatAgent executing {} tool calls: {}",
            len(response.tool_calls),
            [t["name"] for t in response.tool_calls],
        )

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_fn = TOOL_MAP.get(tool_name)
            if tool_fn:
                try:
                    tool_output = tool_fn.invoke(tool_args)
                except Exception as exc:
                    tool_output = f"Tool execution error: {exc}"
            else:
                tool_output = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))

    # Final extraction if loop terminates
    return _extract_text(messages[-1].content)
