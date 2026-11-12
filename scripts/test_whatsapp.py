"""
Test WhatsApp Notification (CallMeBot Gateway)
==============================================
Sends a test message to your personal WhatsApp to verify your CallMeBot setup.

Usage:
------
1. Add +34 644 44 20 89 to your phone contacts as 'CallMeBot'.
2. Send WhatsApp message: "I allow callmebot to send me messages"
3. Wait for CallMeBot's reply with your API Key.
4. Set in .env or pass as arguments:
   uv run python scripts/test_whatsapp.py --phone +1XXXXXXXXXX --apikey XXXXXXX
"""
import argparse
import asyncio
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import get_settings
from src.notifications.whatsapp_bot import send_whatsapp_message


async def main():
    parser = argparse.ArgumentParser(description="Test WhatsApp notification")
    parser.add_argument("--phone", help="Recipient phone number (e.g. +14155552671)")
    parser.add_argument("--apikey", help="CallMeBot personal API key")
    args = parser.parse_args()

    settings = get_settings()
    phone = args.phone or settings.whatsapp_phone
    api_key = args.apikey or settings.whatsapp_api_key

    if not phone or not api_key:
        print("\n❌ Missing WhatsApp credentials.")
        print("Please provide --phone and --apikey or add them to .env:")
        print("  WHATSAPP_PHONE='+1XXXXXXXXXX'")
        print("  WHATSAPP_API_KEY='XXXXXXX'\n")
        print("How to get your free key in 60 seconds:")
        print("1. Check current active number at: https://www.callmebot.com/blog/free-api-whatsapp-messages/")
        print("   Current active: +34 623 78 64 49 (Backup: +34 644 26 33 77)")
        print("2. Send WhatsApp message: 'I allow callmebot to send me messages'")
        print("3. CallMeBot will reply with your API key.")
        sys.exit(1)

    print(f"\n📱 Sending test trade alert to WhatsApp ({phone})...")
    sample_message = (
        "🚨 *WEEKLY OPTION ADVISORY TEST*\n"
        "────────────────────────────────\n"
        "📌 *OKLO* ($23.55) → *CALL_DEBIT_SPREAD*\n"
        "📋 *Contract:* `OKLO $24.0 C / $26.0 C exp 2026-11-20 (12d)`\n\n"
        "💰 *Capital & Profit Forecast*\n"
        "  • Entry Cost:   `$1.60`/sh ($160.00/contract)\n"
        "  • Total Outlay: *1 contract = $160.00*\n"
        "  • Est Profit:   *+$80.00 (+50% gain)*\n"
        "  • Stop Loss:    `$1.20` *(-25% risk cutoff)*\n"
        "  • Greeks:       Δ=0.50 | IV=102%\n\n"
        "🎯 *When to Buy*\n"
        "  Enter near market open via limit order at $1.60 debit.\n\n"
        "🚪 *When to Sell*\n"
        "  Take profit at $2.40 (+50%), stop loss at $1.20 (-25%).\n\n"
        "✅ *WhatsApp integration is fully active!*"
    )

    success = await send_whatsapp_message(sample_message, phone=phone, api_key=api_key)
    if success:
        print("✅ Success! Check your WhatsApp message on your phone.\n")
    else:
        print("❌ Delivery failed. Please check that phone and API key are correct.\n")


if __name__ == "__main__":
    asyncio.run(main())
