"""
WhatsApp Notification Module (CallMeBot Gateway)
================================================
Sends real-time option trade advisory cards directly to your personal WhatsApp chat
using the lightweight, zero-setup CallMeBot API.

Setup (60 Seconds):
-------------------
1. Check the current active bot number at https://www.callmebot.com/blog/free-api-whatsapp-messages/
   Current active numbers:
   • Primary: +34 623 78 64 49
   • Backup:  +34 644 26 33 77
2. Add the number to your phone contacts as 'CallMeBot'.
3. Send this WhatsApp message to that contact: "I allow callmebot to send me messages"
4. CallMeBot will reply with your personal API Key.
5. Add to .env:
   WHATSAPP_PHONE="+1XXXXXXXXXX"
   WHATSAPP_API_KEY="XXXXXXX"
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
import urllib.parse

import httpx
from loguru import logger

from config.settings import get_settings


CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"


def _fmt_whatsapp_trade_card(trade: dict[str, Any], db_id: int) -> str:
    """Format a rich, actionable trade card formatted for WhatsApp."""
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
        f"────────────────────────────────\n"
        f"📌 *{sym}* (${underlying:.2f}) → *{structure}*\n"
        f"📋 *Contract:* `{sym} ${strike} {opt_type[:1]} exp {expiry} ({dte}d)`\n\n"
        f"💰 *Capital & Profit Forecast*\n"
        f"  • Entry Cost:   `${entry:.2f}`/sh (${round(entry*100, 2):.2f}/contract)\n"
        f"  • Total Outlay: *{contracts} contract{'s' if contracts != 1 else ''} = ${max_cost:,.2f}*\n"
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
        f"────────────────────────────────\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M ET')}"
    )
    return card


async def send_whatsapp_message(
    text: str,
    phone: str | None = None,
    api_key: str | None = None,
) -> bool:
    """
    Send a message via CallMeBot WhatsApp Gateway.
    """
    settings = get_settings()
    target_phone = phone or settings.whatsapp_phone
    target_key = api_key or settings.whatsapp_api_key

    if not target_phone or not target_key:
        logger.warning(
            "WhatsApp notification skipped: WHATSAPP_PHONE or WHATSAPP_API_KEY not configured in .env"
        )
        return False

    # Normalize phone: ensure leading '+' or strip non-digits except '+'
    clean_phone = target_phone.strip().replace(" ", "").replace("-", "")

    params = {
        "phone": clean_phone,
        "text": text,
        "apikey": target_key.strip(),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(CALLMEBOT_URL, params=params)
            if resp.status_code == 200 and ("queued" in resp.text.lower() or "success" in resp.text.lower() or "ok" in resp.text.lower() or resp.status_code == 200):
                logger.info("📱 WhatsApp alert sent successfully to {}", clean_phone)
                return True
            else:
                logger.warning(
                    "CallMeBot WhatsApp returned status {}: {}",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
    except Exception as exc:
        logger.error("Failed to send WhatsApp message via CallMeBot: {}", exc)
        return False


async def send_whatsapp_trade_notifications(
    recommendations: list[dict[str, Any]],
    saved_ids: list[int],
) -> int:
    """
    Send all final trade recommendations to WhatsApp.
    Returns the number of successfully delivered messages.
    """
    settings = get_settings()
    if not settings.whatsapp_phone or not settings.whatsapp_api_key:
        logger.debug("WhatsApp not configured — skipping mobile dispatch")
        return 0

    sent_count = 0
    for i, (rec, db_id) in enumerate(zip(recommendations, saved_ids)):
        card = _fmt_whatsapp_trade_card(rec, db_id)
        success = await send_whatsapp_message(card)
        if success:
            sent_count += 1
        # Small delay between consecutive WhatsApp messages to prevent rate limits
        if i < len(recommendations) - 1:
            await asyncio.sleep(2.0)

    logger.info("📱 Delivered {}/{} trade cards to WhatsApp", sent_count, len(recommendations))
    return sent_count
