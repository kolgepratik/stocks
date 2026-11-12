"""
Start the Antigravity Options Web Application & REST API
========================================================
Usage:
  python scripts/start_web.py [--host 0.0.0.0] [--port 8000] [--reload]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from loguru import logger
from src.database.connection import init_db


def main():
    default_port = int(os.environ.get("PORT", "8000"))
    parser = argparse.ArgumentParser(description="Start Antigravity Options AI Web Application")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="HTTP port (defaults to $PORT env var or 8000)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local dev")
    args = parser.parse_args()

    init_db()
    logger.info("=" * 60)
    logger.info("🌐 Launching Antigravity Options Web Dashboard & REST API")
    logger.info("   Local URL:    http://127.0.0.1:{}", args.port)
    logger.info("   Network URL:  http://{}:{}", args.host, args.port)
    logger.info("=" * 60)

    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
