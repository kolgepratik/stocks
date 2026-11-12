"""
Trading Agents – Configuration
Loads all settings from environment variables / .env file using Pydantic BaseSettings.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_database_url(url: str | None) -> str:
    """
    Normalize DATABASE_URL for SQLAlchemy + psycopg v3.
    Converts Heroku/Railway `postgres://` and generic `postgresql://` URLs
    into `postgresql+psycopg://`, while leaving SQLite and explicit driver URLs intact.
    """
    raw = (url or "").strip()
    if not raw:
        return "sqlite:///./data/trading.db"
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://"):]
    return raw


MUTABLE_DB_SETTINGS: frozenset[str] = frozenset({
    "weekly_budget_usd",
    "max_cost_per_trade_usd",
    "target_gain_percent",
    "stop_loss_percent",
    "llm_provider",
    "anthropic_model",
    "gemini_model",
    "anthropic_api_key",
    "gemini_api_key",
    "notification_channel",
    "scan_universe",
    "strategy_mode",
    "target_dte",
    "max_candidates",
})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = Field(default="anthropic", description="Default LLM provider: anthropic | gemini")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(default="claude-haiku-5-5", description="Anthropic model name")
    enable_llm_fallback: bool = Field(default=False, description="Whether to cascade to fallback models on failure")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-3.5-flash", description="Gemini model name")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./data/trading.db",
        description="SQLAlchemy connection string (SQLite local or Railway PostgreSQL)",
    )

    # ── API Security & Cloud (Railway) ────────────────────────────────────────
    api_key: str = Field(
        default="",
        description="Optional API key required via X-API-Key or Bearer token for /api/* routes",
    )
    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins",
    )
    telegram_webhook_secret: str = Field(
        default="",
        description="Optional secret checked against X-Telegram-Bot-Api-Secret-Token",
    )
    cron_secret: str = Field(
        default="",
        description="Optional secret for /api/cron/* endpoints (falls back to api_key)",
    )

    # ── Notifications ─────────────────────────────────────────────────────────
    notification_channel: str = Field(
        default="WHATSAPP", description="Notification delivery: WHATSAPP | TELEGRAM | BOTH | NONE"
    )
    whatsapp_phone: str = Field(
        default="", description="WhatsApp phone number with country code, e.g. +14155552671"
    )
    whatsapp_api_key: str = Field(
        default="", description="CallMeBot personal API key"
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Authorized Telegram chat ID")

    # ── Alpaca ────────────────────────────────────────────────────────────────
    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_paper: bool = Field(default=True, description="Use Alpaca paper trading endpoint")

    # ── Robinhood ─────────────────────────────────────────────────────────────
    robinhood_username: str = Field(default="", description="Robinhood username/email")
    robinhood_password: str = Field(default="", description="Robinhood password")
    robinhood_mfa_totp: str = Field(default="", description="Robinhood TOTP secret for MFA")

    # ── Budget & Risk Controls ─────────────────────────────────────────────────
    weekly_budget_usd: float = Field(
        default=500.0,
        description="Fixed dollar budget for the entire weekly run across all trades",
    )
    max_cost_per_trade_usd: float = Field(
        default=500.0,
        description="Hard cap on cost of any single trade position ($ — defaults to max weekly budget)",
    )
    max_risk_per_trade_percent: float = Field(
        default=2.0, description="Max portfolio risk per single trade (% — used as secondary guardrail)"
    )
    max_portfolio_allocation_percent: float = Field(
        default=10.0, description="Max total options allocation (%)"
    )
    target_gain_percent: float = Field(
        default=50.0,
        description="Target premium gain (%) — 50% default for steady weekly income",
    )
    stop_loss_percent: float = Field(
        default=25.0, description="Max premium loss (%) before stopping out"
    )
    strategy_mode: str = Field(
        default="steady",
        description="'steady' = target 30-60%% gains, defined-risk only | 'aggressive' = target 100%%+ gains",
    )
    # IV rank thresholds for strategy selection
    iv_rank_spread_threshold: float = Field(
        default=50.0,
        description="IV rank above which debit spreads are preferred over single-leg longs",
    )

    # ── Scanner ───────────────────────────────────────────────────────────────
    scan_universe: str = Field(
        default="AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,AMZN,SPY,QQQ,COIN,PLTR",
        description="Comma-separated tickers to screen",
    )
    target_dte: int = Field(default=7, description="Target days-to-expiry")
    max_candidates: int = Field(default=5, description="Max candidates passed to Agent 2")

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_db_url(cls, v: object) -> str:
        """Normalize Railway postgres:// / postgresql:// URLs to postgresql+psycopg://."""
        return normalize_database_url(None if v is None else str(v))

    @field_validator("scan_universe", mode="before")
    @classmethod
    def _coerce_universe(cls, v: object) -> str:
        """Accept list or comma-string from env."""
        if isinstance(v, list):
            return ",".join(v)
        return str(v)

    @property
    def cors_origins_list(self) -> List[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["*"]

    @property
    def scan_universe_list(self) -> List[str]:
        return [t.strip().upper() for t in self.scan_universe.split(",") if t.strip()]

    @property
    def db_dir(self) -> Path:
        """Ensure the data directory exists and return its Path."""
        p = Path("data")
        p.mkdir(parents=True, exist_ok=True)
        return p


def _load_db_overrides(db_url: str) -> dict[str, str]:
    """Read DB-persisted runtime overrides without triggering get_settings() recursion."""
    try:
        from src.database.connection import read_runtime_overrides
        return read_runtime_overrides(db_url)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings with any Postgres/SQLite runtime overrides applied."""
    base = Settings()
    overrides = _load_db_overrides(base.database_url)
    if not overrides:
        return base

    merged = base.model_dump()
    for key, val in overrides.items():
        if key in MUTABLE_DB_SETTINGS and val is not None:
            merged[key] = val
    return Settings.model_validate(merged)


def save_runtime_settings(updates: dict[str, Any]) -> Settings:
    """
    Persist mutable runtime setting updates to the database (`app_settings` table)
    and also update local `.env` if present. Clears cached Settings and returns fresh config.
    """
    current = get_settings()
    clean_updates: dict[str, str] = {}
    for key, val in updates.items():
        if key in MUTABLE_DB_SETTINGS and val is not None and val != "":
            clean_updates[key] = str(val)

    if clean_updates:
        from src.database.connection import upsert_runtime_overrides
        upsert_runtime_overrides(current.database_url, clean_updates)

        # Keep local .env in sync when running outside ephemeral cloud containers
        env_path = Path(".env")
        if env_path.exists():
            try:
                env_content = env_path.read_text(encoding="utf-8")
                for key, raw_val in updates.items():
                    if key in MUTABLE_DB_SETTINGS and raw_val is not None and raw_val != "":
                        env_key = key.upper()
                        pattern = re.compile(rf"^{env_key}=.*$", flags=re.MULTILINE)
                        new_line = f'{env_key}="{raw_val}"' if isinstance(raw_val, str) else f"{env_key}={raw_val}"
                        if pattern.search(env_content):
                            env_content = pattern.sub(new_line, env_content)
                        else:
                            env_content = f"{env_content.rstrip()}\n{new_line}\n"
                env_path.write_text(env_content, encoding="utf-8")
            except OSError:
                pass

    get_settings.cache_clear()
    return get_settings()
