"""
Telegram Bot – Mobile Notification & Interactive Approval Handler
=================================================================
Sends formatted trade cards to your phone with inline action buttons.
Listens for approval/rejection callbacks and routes execution to the
appropriate broker.

Features
--------
- Rich trade card formatting with key metrics
- Inline keyboard: [✅ Execute Alpaca] [📱 Execute Robinhood] [❌ Pass]
- Callback handler with TELEGRAM_CHAT_ID authentication
- Database status updates on approval/rejection
- Non-blocking async notification dispatch

Usage
-----
- Called from persistence_node: `await send_trade_notifications(trades, ids)`
- Run as a persistent daemon: `python scripts/start_bot.py`
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger

from config.settings import get_settings, save_runtime_settings


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_trade_card(trade: dict[str, Any], db_id: int) -> str:
    """Format a rich, actionable trade card message for Telegram."""
    sym = trade.get("symbol", "???")
    opt_type = trade.get("option_type", "CALL")
    strike = trade.get("strike", 0)
    expiry = trade.get("expiration", "?")
    dte = trade.get("dte", "?")
    underlying = trade.get("underlying_price") or trade.get("strike", 0)
    entry = trade.get("entry_price", 0)
    target = trade.get("target_exit_price", 0)
    stop = trade.get("stop_loss_price", 0)
    est_profit_usd = trade.get("estimated_profit_usd") or round((target - entry) * 100, 2)
    est_profit_pct = trade.get("estimated_profit_pct") or (round((target / entry - 1) * 100, 0) if entry else 50.0)
    contracts = trade.get("max_contracts", 1)
    max_cost = trade.get("max_cost") or round(entry * 100 * contracts, 2)
    delta = trade.get("delta", 0)
    theta = trade.get("theta", 0)
    iv = trade.get("implied_volatility", 0)
    risk_rating = trade.get("risk_rating", "MODERATE")
    structure = trade.get("structure", f"LONG_{opt_type}")
    
    why_selected = trade.get("why_selected") or trade.get("thesis", "")
    when_to_buy = trade.get("when_to_buy", "Enter near market open using limit order at or below mid-price.")
    sell_conditions = trade.get("sell_conditions", f"Take profit at ${target:.2f} (+{est_profit_pct:.0f}%), stop loss at ${stop:.2f}.")
    warning_signs = trade.get("early_warning_signs", [])
    if isinstance(warning_signs, list):
        warning_str = "\n  • " + "\n  • ".join(warning_signs) if warning_signs else "None identified"
    else:
        warning_str = str(warning_signs)
    
    confidence = trade.get("confidence_rating", "MEDIUM")
    confidence_pct = trade.get("confidence_score_pct", 70.0)
    sec_name = trade.get("sector_name") or "Market"
    sec_etf = trade.get("sector_etf") or "SPY"
    sec_7d = trade.get("sector_7d_change_pct")
    sec_line = f"  • Sector 7d:    {sec_name} (`{sec_etf}` {sec_7d:+.1f}%)\n" if sec_7d is not None else ""

    emoji_map = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}
    risk_emoji = emoji_map.get(risk_rating, "⚪")

    political_cat = trade.get("political_catalysts")
    earnings_ctx = trade.get("earnings_context")
    company_ctx = trade.get("company_news_context")
    pol_block = f"\n🏛️ *Political & Policy Catalysts*\n  {political_cat}\n" if political_cat else ""
    earn_block = ""
    if earnings_ctx or company_ctx:
        earn_block = "\n📅 *Earnings & Company Intelligence*\n"
        if earnings_ctx:
            earn_block += f"  • {earnings_ctx}\n"
        if company_ctx:
            earn_block += f"  • {company_ctx}\n"

    card = (
        f"🚨 *WEEKLY OPTION ADVISORY* #{db_id}\n"
        f"{'─' * 34}\n"
        f"📌 *{sym}* (${underlying:.2f}) → `{structure}`\n"
        f"📋 *Contract:* `{sym} ${strike} {opt_type[:1]} exp {expiry} ({dte}d)`\n\n"
        f"💰 *Capital & Profit Forecast*\n"
        f"  • Entry Cost:   `~${entry:.2f}`/sh (${round(entry*100, 2):.2f}/contract)\n"
        f"  • Total Outlay: `{contracts} contract{'s' if contracts != 1 else ''}` = *${max_cost:,.2f}*\n"
        f"  • Est Profit:   *+${est_profit_usd:.2f}* *(+{est_profit_pct:.0f}% gain)*\n"
        f"  • Stop Loss:    `${stop:.2f}` *(-25% risk cutoff)*\n"
        f"{sec_line}"
        f"  • Greeks:       Δ={delta:.2f} | Θ={theta:.4f}/d | IV={iv:.0f}%\n"
        f"{pol_block}"
        f"{earn_block}\n"
        f"🎯 *When to Buy*\n"
        f"  {when_to_buy}\n\n"
        f"🚪 *When to Sell (Exit Rules)*\n"
        f"  {sell_conditions}\n\n"
        f"⚠️ *Early Warning Signs to Watch*\n"
        f"  {warning_str}\n\n"
        f"🎲 *Confidence & Risk*\n"
        f"  • Conviction: *{confidence}* ({confidence_pct:.0f}% confidence)\n"
        f"  • Risk Level: {risk_emoji} *{risk_rating}*\n\n"
        f"💡 *Why Selected:*\n_{why_selected[:250]}{'...' if len(why_selected) > 250 else ''}_\n"
        f"{'─' * 34}\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M ET')}"
    )
    return card


def _build_keyboard(db_id: int):
    """Build Telegram InlineKeyboardMarkup for a trade."""
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [
                InlineKeyboardButton("✅ Execute Alpaca", callback_data=f"exec_alpaca_{db_id}"),
                InlineKeyboardButton("📱 Execute Robinhood", callback_data=f"exec_robinhood_{db_id}"),
            ],
            [
                InlineKeyboardButton("❌ Pass", callback_data=f"reject_{db_id}"),
                InlineKeyboardButton("ℹ️ Details", callback_data=f"details_{db_id}"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    except ImportError:
        return None


# ── Notification Dispatch ──────────────────────────────────────────────────────

async def send_trade_notifications(
    trades: list[dict[str, Any]],
    db_ids: list[int],
) -> None:
    """
    Send trade card notifications to the configured Telegram chat.
    Called from the persistence node after DB save.
    """
    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram not configured — skipping notifications")
        return

    try:
        from telegram import Bot
        bot = Bot(token=settings.telegram_bot_token)

        for trade, db_id in zip(trades, db_ids):
            card = _fmt_trade_card(trade, db_id)
            keyboard = _build_keyboard(db_id)
            await bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=card,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
            logger.info("Telegram notification sent for trade #{}", db_id)
            await asyncio.sleep(0.5)  # rate-limit between messages

    except Exception as exc:
        logger.error("Telegram notification error: {}", exc)


# ── Callback Handlers ──────────────────────────────────────────────────────────

async def handle_callback(update, context) -> None:
    """
    Handle inline keyboard button presses from the user's phone.
    Routes to broker execution or marks as rejected.
    """
    from telegram import Update
    from telegram.ext import ContextTypes

    settings = get_settings()
    query = update.callback_query
    await query.answer()

    # Security: only accept callbacks from the authorized chat
    if str(query.from_user.id) != str(settings.telegram_chat_id):
        await query.edit_message_text("⛔ Unauthorized.")
        logger.warning("Unauthorized callback from user_id={}", query.from_user.id)
        return

    data: str = query.data
    logger.info("Callback received: {}", data)

    try:
        if data.startswith("exec_alpaca_"):
            db_id = int(data.split("_")[-1])
            await _execute_trade(db_id, broker="ALPACA", query=query)

        elif data.startswith("exec_robinhood_"):
            db_id = int(data.split("_")[-1])
            await _execute_trade(db_id, broker="ROBINHOOD", query=query)

        elif data.startswith("reject_"):
            db_id = int(data.split("_")[-1])
            await _reject_trade(db_id, query=query)

        elif data.startswith("details_"):
            db_id = int(data.split("_")[-1])
            await _send_details(db_id, query=query)

    except Exception as exc:
        logger.error("Callback handler error: {}", exc)
        await query.edit_message_text(f"❌ Error: {exc}")


async def _execute_trade(db_id: int, broker: str, query) -> None:
    """Route to broker execution and update DB."""
    from src.database.connection import get_db
    from src.database.models import TradeRecommendation, TradeStatus, ExecutionLog, BrokerName
    from src.tools.broker import execute_option_order

    logger.info("Executing trade #{} via {}", db_id, broker)
    await query.edit_message_text(
        f"⏳ Executing trade #{db_id} via {broker}...",
        parse_mode="Markdown",
    )

    try:
        with get_db() as session:
            rec = session.get(TradeRecommendation, db_id)
            if rec is None:
                await query.edit_message_text(f"❌ Trade #{db_id} not found in DB")
                return

            # Execute via broker
            opt_type_val = (
                rec.option_type.value
                if hasattr(rec.option_type, "value")
                else str(rec.option_type)
            )
            result = await execute_option_order(
                broker=broker,
                symbol=rec.symbol,
                option_type=opt_type_val,
                strike=rec.strike_price,
                expiration=str(rec.expiration_date),
                quantity=rec.max_contracts or 1,
                limit_price=rec.entry_price_estimate,
                contract_symbol=rec.contract_symbol or "",
            )

            # Log the execution
            log = ExecutionLog(
                recommendation_id=db_id,
                broker=BrokerName(broker),
                order_id=result.get("order_id", "unknown"),
                status=result.get("status", "submitted"),
                filled_avg_price=result.get("filled_price"),
                quantity=rec.max_contracts,
                raw_response=str(result),
            )
            session.add(log)
            rec.status = TradeStatus.EXECUTED

        success_msg = (
            f"✅ *Executed!* Trade #{db_id}\n"
            f"Broker: `{broker}`\n"
            f"Order ID: `{result.get('order_id', 'N/A')}`\n"
            f"Status: `{result.get('status', 'submitted')}`"
        )
        await query.edit_message_text(success_msg, parse_mode="Markdown")
        logger.info("Trade #{} executed via {}: {}", db_id, broker, result)

    except Exception as exc:
        await query.edit_message_text(f"❌ Execution failed: {exc}")
        logger.error("Execution failed for trade #{}: {}", db_id, exc)


async def _reject_trade(db_id: int, query) -> None:
    """Mark trade as rejected in DB."""
    from src.database.connection import get_db
    from src.database.models import TradeRecommendation, TradeStatus

    try:
        with get_db() as session:
            rec = session.get(TradeRecommendation, db_id)
            if rec:
                rec.status = TradeStatus.REJECTED
        await query.edit_message_text(
            f"❌ Trade #{db_id} marked as *PASSED* (rejected by user).",
            parse_mode="Markdown",
        )
        logger.info("Trade #{} rejected by user", db_id)
    except Exception as exc:
        await query.edit_message_text(f"Error rejecting: {exc}")


async def _send_details(db_id: int, query) -> None:
    """Send full trade details as a follow-up message."""
    from src.database.connection import get_db
    from src.database.models import TradeRecommendation

    try:
        with get_db() as session:
            rec = session.get(TradeRecommendation, db_id)
            if rec is None:
                await query.answer("Trade not found", show_alert=True)
                return
            detail = (
                f"📋 *Comprehensive Advisory Breakdown — Trade #{db_id}*\n\n"
                f"• *Ticker:* {rec.symbol} (${rec.underlying_price:.2f})\n"
                f"• *Structure:* `{rec.structure or rec.option_type}`\n"
                f"• *Contract:* `{rec.contract_symbol or (rec.symbol + ' $' + str(rec.strike_price))}`\n"
                f"• *Expiry:* {rec.expiration_date}\n\n"
                f"💰 *Profit Targets & Sizing:*\n"
                f"• Entry: ${rec.entry_price_estimate:.2f} | Target: ${rec.target_exit_price:.2f}\n"
                f"• Est Profit: +${rec.estimated_profit_usd or (rec.target_exit_price - rec.entry_price_estimate)*100:.2f} (+{rec.estimated_profit_pct or 50:.0f}%)\n"
                f"• Stop Loss: ${rec.stop_loss_price:.2f} (-25% cutoff)\n"
                f"• Sizing: {rec.max_contracts} contracts = ${rec.max_cost:,.2f}\n\n"
                f"🎯 *When to Buy:*\n{rec.when_to_buy or 'Enter near market open at limit order'}\n\n"
                f"🚪 *When & How to Sell:*\n{rec.sell_conditions or 'Take profit at target or cut at stop'}\n\n"
                f"⚠️ *Early Warning Signs:*\n{rec.early_warning_signs or 'Adverse volume or technical breakdown'}\n\n"
                f"🎲 *Confidence:* {rec.confidence_rating or 'MEDIUM'} ({rec.confidence_score_pct or 70:.0f}%)\n"
                f"🛡️ *Risk Score:* {rec.risk_score} ({rec.risk_rating})\n\n"
                f"💡 *Why Selected:*\n{rec.why_selected or rec.thesis}"
            )
        await query.message.reply_text(detail, parse_mode="Markdown")
    except Exception as exc:
        await query.answer(f"Error: {exc}", show_alert=True)


# ── Bot Application Builder ────────────────────────────────────────────────────

def build_bot_app():
    """Build and return a configured telegram.ext.Application instance."""
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler

    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CallbackQueryHandler(handle_callback))

    async def start_cmd(update, context):
        settings = get_settings()
        await update.message.reply_text(
            f"🤖 *Trading Agent Bot Online*\n"
            f"• Weekly Budget: `${settings.weekly_budget_usd:,.2f}`\n"
            f"• Max Cost Per Trade: `${settings.max_cost_per_trade_usd:,.2f}`\n"
            f"• Strategy: Defined-risk only (No margin selling)\n\n"
            f"Commands:\n"
            f"/status - Check pending and executed trades\n"
            f"/budget <amount> - Set a new weekly budget (e.g. `/budget 750`)\n\n"
            f"I'll notify you here when new option recommendations are ready.",
            parse_mode="Markdown",
        )

    async def budget_cmd(update, context):
        if not context.args:
            settings = get_settings()
            await update.message.reply_text(
                f"Current Weekly Budget: `${settings.weekly_budget_usd:,.2f}`\n"
                f"Max Cost Per Trade: `${settings.max_cost_per_trade_usd:,.2f}`\n"
                f"To update, use: `/budget <amount>` (e.g. `/budget 500`)",
                parse_mode="Markdown",
            )
            return
        try:
            new_budget = float(context.args[0])
            settings = save_runtime_settings(
                {
                    "weekly_budget_usd": new_budget,
                    "max_cost_per_trade_usd": new_budget,
                }
            )
            await update.message.reply_text(
                f"✅ *Budget Updated Successfully!*\n"
                f"• New Weekly Budget: `${settings.weekly_budget_usd:,.2f}`\n"
                f"• Max Per Trade: `${settings.max_cost_per_trade_usd:,.2f}`\n"
                f"All future scans will respect this investment ceiling.",
                parse_mode="Markdown",
            )
        except ValueError:
            await update.message.reply_text("❌ Please enter a valid number, e.g. `/budget 500`")

    async def status_cmd(update, context):
        settings = get_settings()
        from src.database.connection import get_db
        from src.database.models import TradeRecommendation, TradeStatus
        from sqlalchemy import select, func
        with get_db() as session:
            pending = session.scalar(
                select(func.count()).where(TradeRecommendation.status == TradeStatus.PENDING_APPROVAL)
            )
            executed = session.scalar(
                select(func.count()).where(TradeRecommendation.status == TradeStatus.EXECUTED)
            )
        await update.message.reply_text(
            f"📊 *System Status*\n"
            f"• Weekly Budget: `${settings.weekly_budget_usd:,.2f}`\n"
            f"• Max Per Trade: `${settings.max_cost_per_trade_usd:,.2f}`\n"
            f"• Pending Approvals: {pending}\n"
            f"• Executed Trades: {executed}",
            parse_mode="Markdown",
        )

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("budget", budget_cmd))

    return app


_webhook_app = None
_webhook_lock: asyncio.Lock | None = None


async def process_webhook_update(payload: dict[str, Any]) -> None:
    """
    Process an incoming Telegram webhook JSON payload inside the FastAPI server,
    avoiding the need for a separate polling daemon container on Railway.
    """
    global _webhook_app, _webhook_lock
    from telegram import Update

    if _webhook_lock is None:
        _webhook_lock = asyncio.Lock()

    async with _webhook_lock:
        if _webhook_app is None:
            _webhook_app = build_bot_app()
            await _webhook_app.initialize()

    update = Update.de_json(payload, _webhook_app.bot)
    await _webhook_app.process_update(update)
