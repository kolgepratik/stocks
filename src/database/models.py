"""
Database Models – SQLAlchemy ORM definitions.

Tables
------
- TradeRecommendation : Full lifecycle of a proposed option trade.
- ExecutionLog        : Broker order fill record linked to a recommendation.
"""
from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Enums ──────────────────────────────────────────────────────────────────────

class OptionType(str, enum.Enum):
    CALL = "CALL"
    PUT = "PUT"


class TradeStatus(str, enum.Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    CLOSED = "CLOSED"
    REVISED = "REVISED"


class BrokerName(str, enum.Enum):
    ALPACA = "ALPACA"
    ROBINHOOD = "ROBINHOOD"
    PAPER = "PAPER"


class TrackingStatus(str, enum.Enum):
    TRACKED = "TRACKED"
    UNTRACKED = "UNTRACKED"


class OrderStatus(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


# ── Base ───────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────────────────────

class TradeRecommendation(Base):
    """
    Represents a fully analyzed and risk-reviewed option trade recommendation.
    Persists from proposal → mobile approval/rejection → execution → close.
    """
    __tablename__ = "trade_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(
        String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )

    # ── Underlying ────────────────────────────────────────────────────────────
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    underlying_price: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Option Contract ───────────────────────────────────────────────────────
    option_type: Mapped[str] = mapped_column(
        Enum(OptionType, name="option_type_enum"), nullable=False
    )
    structure: Mapped[str | None] = mapped_column(String(40), nullable=True)
    strike_price: Mapped[float] = mapped_column(Float, nullable=False)
    short_strike_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expiration_date: Mapped[date] = mapped_column(Date, nullable=False)
    contract_symbol: Mapped[str | None] = mapped_column(String(30), nullable=True)

    # ── Pricing & Risk ────────────────────────────────────────────────────────
    entry_price_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    target_exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_price: Mapped[float] = mapped_column(Float, nullable=False)
    breakeven_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_profit_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_profit_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_contracts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Greeks ────────────────────────────────────────────────────────────────
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv_rank: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Advisory Strategy & Rules ─────────────────────────────────────────────
    sector_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    sector_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    sector_etf: Mapped[str | None] = mapped_column(String(10), nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sector_7d_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_strength_7d_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    why_selected: Mapped[str | None] = mapped_column(Text, nullable=True)
    when_to_buy: Mapped[str | None] = mapped_column(Text, nullable=True)
    sell_conditions: Mapped[str | None] = mapped_column(Text, nullable=True)
    early_warning_signs: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_rating: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence_score_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    political_catalysts: Mapped[str | None] = mapped_column(Text, nullable=True)
    earnings_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    company_news_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Analysis ──────────────────────────────────────────────────────────────
    thesis: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_rating: Mapped[str | None] = mapped_column(String(20), nullable=True)
    scanner_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)

    # ── Status ────────────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        Enum(TradeStatus, name="trade_status_enum"),
        default=TradeStatus.PENDING_APPROVAL,
        nullable=False,
        index=True,
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    execution_logs: Mapped[list[ExecutionLog]] = relationship(
        "ExecutionLog", back_populates="recommendation", cascade="all, delete-orphan"
    )
    placed_orders: Mapped[list[PlacedOrder]] = relationship(
        "PlacedOrder", back_populates="recommendation"
    )

    def __repr__(self) -> str:
        return (
            f"<TradeRecommendation {self.symbol} {self.option_type} "
            f"${self.strike_price} exp={self.expiration_date} status={self.status}>"
        )


class ExecutionLog(Base):
    """
    Records the broker-level order fill for an executed recommendation.
    """
    __tablename__ = "execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("trade_recommendations.id", ondelete="CASCADE"), nullable=False
    )

    broker: Mapped[str] = mapped_column(
        Enum(BrokerName, name="broker_name_enum"), nullable=False
    )
    order_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="submitted")
    filled_avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    recommendation: Mapped[TradeRecommendation] = relationship(
        "TradeRecommendation", back_populates="execution_logs"
    )

    def __repr__(self) -> str:
        return f"<ExecutionLog {self.broker} order={self.order_id} status={self.status}>"


class PlacedOrder(Base):
    """
    Represents an actual order placed/executed by the user, tracked in the database.
    Supports live monitoring, PnL tracking over time, and TRACKED / UNTRACKED states.
    """
    __tablename__ = "placed_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(
        String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    recommendation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("trade_recommendations.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # ── Asset Details ──────────────────────────────────────────────────────────
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    structure: Mapped[str] = mapped_column(String(40), default="LONG_CALL", nullable=False)
    option_type: Mapped[str] = mapped_column(String(10), default="CALL", nullable=False)
    strike_price: Mapped[float] = mapped_column(Float, nullable=False)
    short_strike_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expiration_date: Mapped[date] = mapped_column(Date, nullable=False)
    contract_symbol: Mapped[str | None] = mapped_column(String(35), nullable=True)

    # ── Order Sizing & Entry ───────────────────────────────────────────────────
    contracts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)  # Premium per share at entry
    total_cost: Mapped[float] = mapped_column(Float, nullable=False)  # fill_price * 100 * contracts

    # ── Price Targets & Strategy ───────────────────────────────────────────────
    target_exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Live Valuation & PnL ───────────────────────────────────────────────────
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)  # Current premium per share
    current_value: Mapped[float | None] = mapped_column(Float, nullable=True)  # current_price * 100 * contracts
    pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)      # Current profit/loss in USD
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)    # Percentage return

    # ── Tracking & Order Status ────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(String(20), default="OPEN", nullable=False, index=True)  # OPEN | CLOSED | CANCELLED
    tracking_status: Mapped[str] = mapped_column(
        String(20), default="TRACKED", nullable=False, index=True
    )  # TRACKED | UNTRACKED

    # ── Execution & Notes ──────────────────────────────────────────────────────
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    placed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationship ───────────────────────────────────────────────────────────
    recommendation: Mapped[TradeRecommendation | None] = relationship(
        "TradeRecommendation", back_populates="placed_orders"
    )

    def __repr__(self) -> str:
        return (
            f"<PlacedOrder {self.symbol} {self.structure} ${self.strike_price} "
            f"qty={self.contracts} fill=${self.fill_price} pnl=${self.pnl:+.2f} "
            f"tracking={self.tracking_status} status={self.status}>"
        )


class ScanJob(Base):
    """
    Persistent background scan job tracking so async scans and progress/results
    survive restarts, worker scaling, and cloud container deployments.
    """
    __tablename__ = "scan_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    symbols_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_trade_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    notify: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    started_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    completed_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    recommendations_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    recommendations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    saved_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    macro_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    political_catalysts_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_snapshot(self) -> dict:
        import json
        import time

        end_t = self.completed_at or time.time()
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "symbols": json.loads(self.symbols_json) if self.symbols_json else None,
            "budget": self.budget,
            "max_trade_cost": self.max_trade_cost,
            "notify": self.notify,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": round(max(0.0, end_t - self.started_at), 1),
            "recommendations_count": self.recommendations_count,
            "recommendations": json.loads(self.recommendations_json) if self.recommendations_json else [],
            "saved_ids": json.loads(self.saved_ids_json) if self.saved_ids_json else [],
            "macro_summary": self.macro_summary,
            "political_catalysts": json.loads(self.political_catalysts_json) if self.political_catalysts_json else [],
            "error": self.error,
        }


class AppSetting(Base):
    """
    Database-backed mutable runtime settings (overrides env defaults so settings
    updated via /api/settings or Telegram /budget persist across cloud restarts).
    """
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
