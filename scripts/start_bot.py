#!/usr/bin/env python3
"""
start_bot.py – Persistent daemon that runs the Telegram bot listener.

Keeps the bot polling for mobile approval/rejection callbacks.
Should run as a background service (Docker, launchd, systemd, etc.)

Usage
-----
    uv run python scripts/start_bot.py
    docker compose up -d trading-bot
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
    )

    Path("logs").mkdir(exist_ok=True)

    from config.settings import get_settings
    settings = get_settings()

    if not settings.telegram_bot_token:
        logger.error(
            "TELEGRAM_BOT_TOKEN is not set. Please configure .env and restart."
        )
        sys.exit(1)

    logger.info("🤖 Starting Telegram Bot daemon...")
    logger.info("   Chat ID: {}", settings.telegram_chat_id or "(not set — all users accepted!)")

    # Initialize DB
    from src.database.connection import init_db
    init_db()
    logger.info("   Database initialized.")

    # Build and run bot
    from src.notifications.telegram_bot import build_bot_app
    app = build_bot_app()

    logger.info("   Bot is online. Listening for callbacks... (Ctrl+C to stop)")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
