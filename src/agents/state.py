"""
LangGraph State – shared TypedDict that flows through the entire agent pipeline.
"""
from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class AgentTradingState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    target_symbols: list[str]           # universe to screen
    weekly_budget_usd: Optional[float]  # user-defined maximum total weekly budget ($)
    max_cost_per_trade_usd: Optional[float] # max cost per single position ($)

    # ── Agent 1 Output ────────────────────────────────────────────────────────
    macro_political_context: Optional[dict[str, Any]] # live US political, macro & market headlines
    scanner_proposals: list[dict[str, Any]]   # CandidateTicker dicts

    # ── Agent 2 Output ────────────────────────────────────────────────────────
    structured_trades: list[dict[str, Any]]   # ScoredContract + thesis dicts

    # ── Agent 3 Output ────────────────────────────────────────────────────────
    risk_reviews: list[dict[str, Any]]        # per-trade risk review decisions
    final_recommendations: list[dict[str, Any]]  # approved, ready to persist

    # ── Workflow Control ──────────────────────────────────────────────────────
    iteration_count: int                # how many Agent2→Agent3 loops so far
    review_feedback: Optional[str]      # Agent 3 → Agent 2 revision feedback
    workflow_error: Optional[str]       # any fatal error message
    run_id: Optional[str]              # unique run identifier (UUID)
    notify: Optional[bool]             # whether to dispatch mobile alerts on completion
    saved_recommendation_ids: Optional[list[int]] # persisted DB recommendation IDs
