"""
LangGraph Workflow – Assembles the 3-agent pipeline into a stateful directed graph.

Graph Structure
---------------

START
  │
  ▼
scanner_node  (Agent 1 – Opportunity Scanner)
  │
  ▼
structurer_node  (Agent 2 – Trade Structurer)
  │
  ▼
risk_node  (Agent 3 – Risk Sentinel)
  │
  ├─► (needs_revision AND iteration < 2) → structurer_node  [feedback loop]
  │
  └─► persistence_node  (save to DB + fire Telegram notification)
  │
  ▼
END

Conditional routing is handled by `should_revise()` after `risk_node`.
"""
from __future__ import annotations

import uuid
from functools import wraps
from typing import Any, Callable, Literal

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from config.settings import get_settings
from src.agents.profitability_agent import profitability_node
from src.agents.risk_agent import risk_node
from src.agents.scanner_agent import scanner_node
from src.agents.state import AgentTradingState
from src.database.connection import get_db, init_db
from src.database.models import (
    BrokerName,
    OptionType,
    TradeRecommendation,
    TradeStatus,
)


# ── Persistence Node ───────────────────────────────────────────────────────────

def persistence_node(state: AgentTradingState) -> AgentTradingState:
    """
    Saves all final_recommendations to the database and triggers notifications.
    """
    recommendations = state.get("final_recommendations") or []
    if not recommendations:
        logger.warning("Persistence: no recommendations to save")
        return state

    settings = get_settings()
    logger.info("💾 Persistence: saving {} recommendations", len(recommendations))

    saved_ids: list[int] = []
    try:
        with get_db() as session:
            for trade in recommendations:
                from datetime import datetime, date as date_type

                # Parse expiration
                exp_raw = trade.get("expiration", "")
                try:
                    exp_date = datetime.strptime(exp_raw, "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    exp_date = date_type.today()

                warning_signs_str = "; ".join(trade.get("early_warning_signs", [])) if isinstance(trade.get("early_warning_signs"), list) else trade.get("early_warning_signs")

                rec = TradeRecommendation(
                    symbol=trade["symbol"],
                    underlying_price=trade.get("underlying_price") or trade.get("strike", 0),
                    option_type=trade.get("option_type", "CALL"),
                    structure=trade.get("structure"),
                    strike_price=trade.get("strike", 0),
                    short_strike_price=trade.get("short_strike"),
                    expiration_date=exp_date,
                    contract_symbol=trade.get("contract_symbol") or "",
                    entry_price_estimate=trade.get("entry_price", 0),
                    target_exit_price=trade.get("target_exit_price", 0),
                    stop_loss_price=trade.get("stop_loss_price", 0),
                    breakeven_price=trade.get("pnl_curve", {}).get("breakeven_price"),
                    estimated_profit_usd=trade.get("estimated_profit_usd"),
                    estimated_profit_pct=trade.get("estimated_profit_pct"),
                    max_contracts=trade.get("max_contracts"),
                    max_cost=trade.get("max_cost"),
                    delta=trade.get("delta"),
                    gamma=trade.get("gamma"),
                    theta=trade.get("theta"),
                    vega=trade.get("vega"),
                    implied_volatility=trade.get("implied_volatility"),
                    sector_category=trade.get("sector_category"),
                    sector_name=trade.get("sector_name"),
                    sector_etf=trade.get("sector_etf"),
                    rank=trade.get("rank"),
                    sector_7d_change_pct=trade.get("sector_7d_change_pct"),
                    relative_strength_7d_pct=trade.get("relative_strength_7d_pct"),
                    why_selected=trade.get("why_selected"),
                    when_to_buy=trade.get("when_to_buy"),
                    sell_conditions=trade.get("sell_conditions"),
                    early_warning_signs=warning_signs_str,
                    confidence_rating=trade.get("confidence_rating"),
                    confidence_score_pct=trade.get("confidence_score_pct"),
                    political_catalysts=trade.get("political_catalysts"),
                    earnings_context=trade.get("earnings_context"),
                    company_news_context=trade.get("company_news_context"),
                    thesis=trade.get("thesis", ""),
                    risk_score=trade.get("risk_score"),
                    risk_rating=trade.get("risk_rating"),
                    scanner_rationale=trade.get("market_commentary") or trade.get("rationale"),
                    risk_feedback=trade.get("risk_feedback"),
                    iteration_count=state.get("iteration_count", 0),
                    status=TradeStatus.PENDING_APPROVAL,
                )
                session.add(rec)
                session.flush()
                saved_ids.append(rec.id)
                logger.info("  Saved {} {} @ ${} → db id={}", rec.symbol, rec.option_type, rec.strike_price, rec.id)
    except Exception as exc:
        logger.error("Persistence failed: {}", exc)

    # Fire mobile notifications (WhatsApp and/or Telegram)
    if saved_ids and state.get("notify", True):
        channel = settings.notification_channel.upper()

        # 1. WhatsApp Notifications
        if channel in ("WHATSAPP", "BOTH"):
            try:
                from src.notifications.whatsapp_bot import send_whatsapp_trade_notifications
                import asyncio

                loop = asyncio.new_event_loop()
                loop.run_until_complete(send_whatsapp_trade_notifications(recommendations, saved_ids))
                loop.close()
            except Exception as exc:
                logger.warning("WhatsApp notification failed (non-fatal): {}", exc)

        # 2. Telegram Notifications
        if channel in ("TELEGRAM", "BOTH"):
            try:
                from src.notifications.telegram_bot import send_trade_notifications
                import asyncio

                loop = asyncio.new_event_loop()
                loop.run_until_complete(send_trade_notifications(recommendations, saved_ids))
                loop.close()
            except Exception as exc:
                logger.warning("Telegram notification failed (non-fatal): {}", exc)

    return {**state, "saved_recommendation_ids": saved_ids}


# ── Conditional Router ─────────────────────────────────────────────────────────

def should_revise(state: AgentTradingState) -> Literal["structurer_node", "persistence_node"]:
    """
    After risk_node: if feedback exists and we haven't hit the iteration limit,
    loop back to Agent 2. Otherwise proceed to persistence.
    """
    feedback = state.get("review_feedback")
    iteration = state.get("iteration_count", 0)
    max_iter = 2  # matches the guardrail in risk_agent

    if feedback and iteration <= max_iter:
        logger.info("🔄 Router: revision needed (iteration {}), looping to structurer", iteration)
        return "structurer_node"

    logger.info("✅ Router: proceeding to persistence (iteration {})", iteration)
    return "persistence_node"


# ── Msgpack Safety Helpers ─────────────────────────────────────────────────────

def _sanitize_for_msgpack(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native Python types for MemorySaver."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_sanitize_for_msgpack(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            _sanitize_for_msgpack(k): _sanitize_for_msgpack(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_msgpack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_msgpack(item) for item in value)
    return value


def _wrap_node(
    node_fn: Callable[[AgentTradingState], dict[str, Any]],
) -> Callable[[AgentTradingState], dict[str, Any]]:
    """Wrap a LangGraph node so returned state diffs are msgpack-serializable."""
    @wraps(node_fn)
    def wrapped(state: AgentTradingState) -> dict[str, Any]:
        return _sanitize_for_msgpack(node_fn(state))

    return wrapped


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph(use_checkpointer: bool = True) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    use_checkpointer : bool
        Whether to attach a MemorySaver checkpointer for execution tracing.
    """
    graph = StateGraph(AgentTradingState)

    # Register nodes wrapped to sanitize NumPy scalars before MemorySaver checkpoints state
    graph.add_node("scanner_node", _wrap_node(scanner_node))
    graph.add_node("structurer_node", _wrap_node(profitability_node))
    graph.add_node("risk_node", _wrap_node(risk_node))
    graph.add_node("persistence_node", _wrap_node(persistence_node))

    # Linear edges
    graph.add_edge(START, "scanner_node")
    graph.add_edge("scanner_node", "structurer_node")
    graph.add_edge("structurer_node", "risk_node")

    # Conditional edge from risk_node: revise or persist
    graph.add_conditional_edges(
        "risk_node",
        should_revise,
        {
            "structurer_node": "structurer_node",
            "persistence_node": "persistence_node",
        },
    )

    graph.add_edge("persistence_node", END)

    if use_checkpointer:
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# ── Public Run Interface ───────────────────────────────────────────────────────

def run_pipeline(
    symbols: list[str] | None = None,
    thread_id: str | None = None,
    weekly_budget_usd: float | None = None,
    max_cost_per_trade_usd: float | None = None,
    notify: bool = True,
    progress_callback: Any | None = None,
) -> dict:
    """
    Execute the full multi-agent pipeline synchronously.

    Parameters
    ----------
    symbols : list[str] | None
        Override the scan universe. If None, reads from settings.
    thread_id : str | None
        LangGraph thread identifier for checkpointing/resumption.
    weekly_budget_usd : float | None
        Override user weekly dollar budget.
    max_cost_per_trade_usd : float | None
        Override max cost allowed per position.
    notify : bool
        Whether to send mobile alerts (WhatsApp/Telegram) upon completion.
    progress_callback : Callable[[str, dict], None] | None
        Optional callback invoked after each LangGraph node finishes.

    Returns
    -------
    dict
        Final AgentTradingState after pipeline completion.
    """
    settings = get_settings()
    init_db()

    run_id = str(uuid.uuid4())
    if thread_id is None:
        thread_id = run_id

    budget = weekly_budget_usd if weekly_budget_usd is not None else settings.weekly_budget_usd
    trade_cap = max_cost_per_trade_usd if max_cost_per_trade_usd is not None else min(settings.max_cost_per_trade_usd, budget)

    initial_state: AgentTradingState = {
        "target_symbols": symbols,
        "weekly_budget_usd": budget,
        "max_cost_per_trade_usd": trade_cap,
        "scanner_proposals": [],
        "structured_trades": [],
        "risk_reviews": [],
        "final_recommendations": [],
        "iteration_count": 0,
        "review_feedback": None,
        "workflow_error": None,
        "run_id": run_id,
        "notify": notify,
    }

    logger.info("=" * 60)
    logger.info("🚀 Starting Options Trading Pipeline | run_id={}", run_id)
    logger.info("   Universe: {}", symbols or "3 Specialized Sectors (AI/Chips, Energy/Gas, DC Nuclear Power)")
    logger.info("   Budget: ${:,.2f} total | Max ${:,.2f}/trade", initial_state["weekly_budget_usd"], initial_state["max_cost_per_trade_usd"])
    logger.info("=" * 60)

    app = build_graph(use_checkpointer=True)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    if progress_callback is not None:
        merged_state: dict = dict(initial_state)
        for chunk in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if isinstance(node_update, dict):
                    merged_state.update(node_update)
                try:
                    progress_callback(node_name, merged_state)
                except Exception as exc:
                    logger.debug("Progress callback error: {}", exc)
        final_state = merged_state
    else:
        final_state = app.invoke(initial_state, config=config)

    n_approved = len(final_state.get("final_recommendations") or [])
    logger.info("=" * 60)
    logger.info(
        "✅ Pipeline complete | {} final recommendations | run_id={}",
        n_approved,
        run_id,
    )
    logger.info("=" * 60)

    return final_state
