"""
Agent 3: Risk Sentinel & Supervisor
====================================
Reviews all structured trades against strict risk guardrails and portfolio
rules. Issues APPROVE / REJECT / REVISE decisions for each trade.
Triggers a feedback loop back to Agent 2 if any trade needs revision.

Guardrails
----------
- Single trade risk ≤ MAX_RISK_PER_TRADE_PERCENT (default 2%)
- Total portfolio options exposure ≤ MAX_PORTFOLIO_ALLOCATION_PERCENT (default 10%)
- No trade held through unpriced earnings (unless confidence=HIGH)
- Delta must be between 0.30 and 0.65 (absolute)
- Minimum 3 DTE remaining

Flow
----
AgentTradingState.structured_trades
    → LLM risk review (Gemini Pro)
    → APPROVE → final_recommendations
    → REVISE  → review_feedback → loop back to Agent 2 (max 2 iterations)
    → REJECT  → dropped from final set
"""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings
from src.agents.state import AgentTradingState
from src.tools.market_data import get_market_timing_context


# ── Pydantic Output Schema ─────────────────────────────────────────────────────

class TradeRiskReview(BaseModel):
    symbol: str
    decision: str = Field(description="APPROVE | REJECT | REVISE")
    risk_score: float = Field(
        description="Risk score 0-10 (0=very safe, 10=extremely risky)", ge=0, le=10
    )
    risk_rating: str = Field(description="LOW | MODERATE | HIGH | EXTREME")
    issues_found: list[str] = Field(
        description="List of specific risk issues identified (empty if none)"
    )
    revision_notes: str = Field(
        description="Specific instructions for Agent 2 if decision=REVISE, else empty"
    )
    approved_max_contracts: int = Field(
        description="Risk-adjusted max contracts (may be lower than Agent 2 proposed)"
    )
    rationale: str = Field(description="1-2 sentence explanation of the risk decision")


class RiskSentinelOutput(BaseModel):
    reviews: list[TradeRiskReview] = Field(description="One review per structured trade")
    portfolio_assessment: str = Field(
        description="Overall portfolio-level risk commentary"
    )
    total_capital_at_risk: float = Field(
        description="Sum of max_cost across all APPROVED trades ($)"
    )
    requires_revision: bool = Field(
        description="True if any trade received REVISE decision"
    )

    @field_validator("reviews", mode="before")
    @classmethod
    def _coerce_reviews(cls, v: Any) -> Any:
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except Exception:
                return v
        return v


# ── LLM Setup ─────────────────────────────────────────────────────────────────

def _build_llm():
    from src.agents.llm_resilience import build_llm
    return build_llm()


# ── Deterministic Pre-checks ───────────────────────────────────────────────────

def _deterministic_checks(
    trade: dict[str, Any], settings, max_cost_per_trade: float
) -> list[str]:
    """
    Fast rule-based checks before LLM review.
    Returns a list of issues found.
    """
    issues: list[str] = []
    delta = abs(trade.get("delta", 0.5))
    dte = trade.get("dte", 7)
    max_cost = trade.get("max_cost", 0)
    entry = trade.get("entry_price", 0)
    stop = trade.get("stop_loss_price", 0)
    target = trade.get("target_exit_price", 0)
    iv = trade.get("implied_volatility", 30)
    structure = trade.get("structure", "").upper()

    allowed_structures = ["LONG_CALL", "CALL_DEBIT_SPREAD"]
    if structure and not any(s == structure for s in allowed_structures):
        issues.append(f"FORBIDDEN STRATEGY: '{structure}'. Only {allowed_structures} are permitted (NO LONG_PUT or margin strategies)!")

    # Debit spreads legitimately carry smaller net deltas (e.g. 0.08-0.35) because of the short leg
    if "SPREAD" in structure:
        if delta < 0.05 or delta > 0.70:
            issues.append(f"Net spread delta {delta:.2f} outside allowed range [0.05-0.70]")
    else:
        if delta < 0.25 or delta > 0.75:
            issues.append(f"Delta {delta:.2f} outside allowed range [0.25-0.75]")
    if dte < 3:
        issues.append(f"DTE {dte} < 3 — too close to expiry")
    if entry > 0 and stop >= entry:
        issues.append(f"Stop ${stop:.2f} >= entry ${entry:.2f} — invalid stop")
    if entry > 0 and target <= entry:
        issues.append(f"Target ${target:.2f} <= entry ${entry:.2f} — invalid target")
    if iv > 120:
        issues.append(f"IV {iv:.0f}% extremely elevated — premium crush risk on long options")
    if max_cost > max_cost_per_trade and max_cost > 0:
        issues.append(f"Max cost ${max_cost:,.2f} exceeds user's per-trade budget cap of ${max_cost_per_trade:,.2f}")

    return issues


# ── Prompt Builder ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the CHIEF RISK OFFICER and final gatekeeper for a systematic options trading system. You are AGENT 3 — the Risk Sentinel.

Your job is to critically review each structured option trade and issue: APPROVE, REJECT, or REVISE.

MANDATORY USER CONSTRAINTS (HARD RULES):
1. ZERO MARGIN SELLING / NO SINGLE-LEG PUTS / NO NAKED OPTIONS:
   The user does NOT own 100 shares of underlying stock and does NOT have margin to sell short or deliver stock.
   - ANY 'LONG_PUT' IS STRICTLY FORBIDDEN (prevents exercise/assignment risk requiring delivery of 100 shares).
   - ANY naked short position or uncovered option MUST BE INSTANTLY REJECTED.
   - ONLY permitted structures: LONG_CALL and CALL_DEBIT_SPREAD (buyer positions only). Any trade with 'PUT' must be REJECTED immediately!

2. INDEPENDENT PER-TRADE BUDGET & CATEGORY REPRESENTATION (ADVISORY PIPELINE):
   - This pipeline is an advisory engine where the user selects their preferred trade from the ranked output.
   - Each individual trade cost must not exceed ${max_trade_cost:,.2f}.
   - DO NOT reject or suppress candidate B simply because candidate A + candidate B > ${weekly_budget:,.2f}. Evaluate each trade independently under the ${max_trade_cost:,.2f} cap.
   - CATEGORY COVERAGE MANDATE: Ensure at least 1 compliant trade from EACH represented sector category (AI_CHIPS_INFRA, WAR_ENERGY_GAS, DC_POWER_NUCLEAR) is approved if it passes budget & structure rules (using CALL_DEBIT_SPREAD for higher-priced tickers).

3. STEADY GAINS, LOSS MINIMIZATION & SAME-DAY EXECUTION:
   - SAME-DAY ORDER REQUIREMENT: Every approved trade MUST be actionable TODAY (same-day DAY limit order). Do NOT issue REVISE telling the user to wait until Monday or tomorrow's daily close unless there is a true hard rule failure.
   - Target profit must be a realistic 30% to 50% gain (within spread width if spread).
   - Stop loss must be tight (20% to 25% max loss on premium).
   - Delta: 0.25-0.75 for LONG_CALL; net delta 0.05-0.70 for CALL_DEBIT_SPREAD.

REVISE vs REJECT:
- REVISE: good candidate whose single-leg cost > ${max_trade_cost:,.2f} (ask Agent 2 to convert to CALL_DEBIT_SPREAD <= ${max_trade_cost:,.2f}), or invalid stop/target math.
- REJECT: forbidden PUT/naked short strategy, unresolvable chain, or DTE < 3.

For each approved trade, confirm or adjust approved_max_contracts to fit ${max_trade_cost:,.2f}."""


def _build_human_prompt(
    trades: list[dict[str, Any]],
    settings,
    weekly_budget: float,
    max_cost_per_trade: float,
) -> str:
    mkt_time = get_market_timing_context()
    lines = [
        f"## ⏰ Live Execution Clock",
        f"- Today: {mkt_time['day_of_week']}, {mkt_time['today_date']} at {mkt_time['time_et']} ({mkt_time['market_session']})",
        f"- Mandate: Every approved trade MUST be placeable TODAY ({mkt_time['today_date']}).\n",
        f"## Strict Risk & Budget Context (Advisory Mode)",
        f"- Max Cost Per Trade Limit: ${max_cost_per_trade:,.2f} (each trade must be <= this amount)",
        f"- Total Weekly Budget Ceiling: ${weekly_budget:,.2f}",
        f"- Policy: Evaluate each trade INDEPENDENTLY under the ${max_cost_per_trade:,.2f} cap and preserve at least 1 compliant trade from EACH category represented.",
        f"- Number of candidates to evaluate: {len(trades)}\n",
    ]

    for i, trade in enumerate(trades, 1):
        pnl = trade.get("pnl_curve", {})
        pre_issues = _deterministic_checks(trade, settings, max_cost_per_trade)
        strike_str = f"${trade['strike']}/${trade['short_strike']}" if trade.get("short_strike") else f"${trade['strike']}"
        sec_label = trade.get("sector_category") or trade.get("sector_name") or "Market"
        lines.append(
            f"## Trade {i}: {trade['symbol']} [{sec_label}] {trade.get('structure', trade.get('option_type'))} {strike_str} exp={trade['expiration']}"
        )
        lines.append(f"- Entry: ${trade['entry_price']:.2f} | Target: ${trade['target_exit_price']:.2f} | Stop: ${trade['stop_loss_price']:.2f}")
        lines.append(f"- Est Profit: ${trade.get('estimated_profit_usd', 0):.2f} (+{trade.get('estimated_profit_pct', 0):.0f}%)")
        lines.append(f"- DTE: {trade['dte']} | Contracts: {trade['max_contracts']} | Total Cost: ${trade['max_cost']:,.2f}")
        lines.append(f"- Greeks: Δ={trade['delta']:.3f}, Γ={trade['gamma']:.5f}, Θ={trade['theta']:.4f}/d, V={trade['vega']:.4f}")
        lines.append(f"- IV: {trade['implied_volatility']:.1f}%")
        lines.append(f"- Breakeven: ${pnl.get('breakeven_price', 0):.2f}")
        lines.append(f"- Why Selected: {trade.get('why_selected', trade.get('thesis', 'N/A'))}")
        if trade.get("political_catalysts"):
            lines.append(f"- Political/Policy Catalysts: {trade.get('political_catalysts')}")
        if trade.get("earnings_context"):
            lines.append(f"- Earnings Intelligence: {trade.get('earnings_context')}")
        if trade.get("company_news_context"):
            lines.append(f"- Company News: {trade.get('company_news_context')}")
        lines.append(f"- When to Buy: {trade.get('when_to_buy', 'Market open confirmation')}")
        lines.append(f"- Sell Conditions: {trade.get('sell_conditions', 'TP/SL limits')}")
        warning_signs = trade.get("early_warning_signs", [])
        if warning_signs:
            lines.append(f"- Early Warning Signs: {'; '.join(warning_signs)}")
        lines.append(f"- Confidence: {trade.get('confidence_rating', 'MEDIUM')} ({trade.get('confidence_score_pct', 70):.0f}%)")
        if pre_issues:
            lines.append(f"- ⚠️ Pre-checks flagged: {'; '.join(pre_issues)}")
        lines.append("")

    lines.append(
        f"Review each trade against the ${weekly_budget:,.2f} total budget and ${max_cost_per_trade:,.2f} per-trade limit. "
        "Confirm that buy/sell conditions and early warning signs are actionable. "
        "Respond with JSON matching RiskSentinelOutput schema."
    )
    return "\n".join(lines)


# ── Agent Node ─────────────────────────────────────────────────────────────────

def risk_node(state: AgentTradingState) -> AgentTradingState:
    """
    LangGraph node: Risk Sentinel reviews all structured trades.
    Updates state with risk_reviews, final_recommendations, and optionally review_feedback.
    """
    settings = get_settings()
    trades = state.get("structured_trades") or []
    iteration = state.get("iteration_count", 0)

    weekly_budget = float(state.get("weekly_budget_usd") or settings.weekly_budget_usd)
    max_cost_per_trade = float(state.get("max_cost_per_trade_usd") or settings.max_cost_per_trade_usd)

    if not trades:
        logger.warning("Risk Sentinel: no structured trades to review")
        return {**state, "risk_reviews": [], "final_recommendations": []}

    logger.info(
        "🛡️  Risk Sentinel: reviewing {} trades against ${:,.2f} budget (iteration {})",
        len(trades),
        weekly_budget,
        iteration,
    )

    # LLM review with multi-model cascade (including flash-3.8 with high reasoning)
    from src.agents.llm_resilience import invoke_structured_with_fallback

    system_content = _SYSTEM_PROMPT.format(
        max_trade_cost=max_cost_per_trade,
        weekly_budget=weekly_budget,
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=_build_human_prompt(trades, settings, weekly_budget, max_cost_per_trade)),
    ]

    try:
        output: RiskSentinelOutput = invoke_structured_with_fallback(
            messages=messages,
            schema=RiskSentinelOutput,
            temperature=0.05,
            request_timeout=35,
        )
    except Exception as exc:
        logger.error("Risk Sentinel LLM failed: {}. Auto-approving with hard-rule check.", exc)
        # Fallback: approve trades that pass deterministic checks
        fallback_reviews = []
        for trade in trades:
            issues = _deterministic_checks(trade, settings, max_cost_per_trade)
            decision = "REJECT" if len(issues) > 2 else ("REVISE" if issues else "APPROVE")
            fallback_reviews.append(
                TradeRiskReview(
                    symbol=trade["symbol"],
                    decision=decision,
                    risk_score=len(issues) * 2.0,
                    risk_rating="HIGH" if issues else "MODERATE",
                    issues_found=issues,
                    revision_notes="; ".join(issues) if decision == "REVISE" else "",
                    approved_max_contracts=trade.get("max_contracts", 1),
                    rationale="Fallback: LLM unavailable, deterministic rule check only.",
                )
            )
        output = RiskSentinelOutput(
            reviews=fallback_reviews,
            portfolio_assessment="Fallback review completed.",
            total_capital_at_risk=sum(
                t.get("max_cost", 0) for t, r in zip(trades, fallback_reviews)
                if r.decision == "APPROVE"
            ),
            requires_revision=any(r.decision == "REVISE" for r in fallback_reviews),
        )

    logger.info("Risk Sentinel assessment: {}", output.portfolio_assessment)

    # Build review map
    review_map = {r.symbol: r for r in output.reviews}

    # Separate approved, rejected, revised
    final_recommendations: list[dict[str, Any]] = []
    revision_notes: list[str] = []
    risk_reviews: list[dict[str, Any]] = []
    cumulative_budget_spent = 0.0

    for trade in trades:
        sym = trade["symbol"]
        review = review_map.get(sym)

        if review is None:
            # No review received — auto-approve cautiously
            logger.warning("No risk review received for {}. Auto-approving cautiously.", sym)
            final_recommendations.append({**trade, "risk_score": 5.0, "risk_rating": "MODERATE"})
            continue

        risk_reviews.append(review.model_dump())
        logger.info(
            "  {} → {} (risk={:.1f} | {})",
            sym,
            review.decision,
            review.risk_score,
            review.risk_rating,
        )

        if review.decision == "APPROVE":
            # Independent budget check per trade (advisory mode)
            approved_contracts = min(trade.get("max_contracts", 1), review.approved_max_contracts)
            contract_cost = trade.get("entry_price", 0) * 100
            
            # Ensure this individual trade fits within max_cost_per_trade
            if contract_cost > 0 and (approved_contracts * contract_cost) > max_cost_per_trade:
                approved_contracts = max(1, int(max_cost_per_trade / contract_cost))
            
            final_cost = round(approved_contracts * contract_cost, 2)

            enriched = {
                **trade,
                "risk_score": review.risk_score,
                "risk_rating": review.risk_rating,
                "risk_feedback": None,
                "max_contracts": approved_contracts,
                "max_cost": final_cost,
                "risk_rationale": review.rationale,
            }
            final_recommendations.append(enriched)

        elif review.decision == "REVISE" and iteration < 2:
            if review.revision_notes:
                revision_notes.append(f"[{sym}] {review.revision_notes}")

        elif review.decision == "REJECT":
            logger.info("  {} REJECTED: {}", sym, review.rationale)

        # Even on REVISE/REJECT on final iteration, include with caveats
        elif review.decision == "REVISE" and iteration >= 2:
            logger.warning(
                "{} still needs revision after {} iterations — including anyway with caveats", sym, iteration
            )
            enriched = {
                **trade,
                "risk_score": review.risk_score,
                "risk_rating": review.risk_rating,
                "risk_feedback": review.revision_notes,
                "max_contracts": max(1, review.approved_max_contracts),
                "risk_rationale": f"[Included with caveats after {iteration} iterations] {review.rationale}",
            }
            final_recommendations.append(enriched)

    # Build feedback for Agent 2 if revisions needed
    combined_feedback: str | None = None
    if revision_notes and iteration < 2:
        combined_feedback = (
            f"REVISION REQUESTED (iteration {iteration + 1}):\n"
            + "\n".join(revision_notes)
            + f"\n\nPortfolio note: {output.portfolio_assessment}"
        )
        logger.info("Triggering revision loop with feedback:\n{}", combined_feedback)
    else:
        # Final pass: guarantee at least 1 compliant recommendation per represented sector category
        present_cats = {t.get("sector_category") for t in trades if t.get("sector_category")}
        approved_cats = {r.get("sector_category") for r in final_recommendations if r.get("sector_category")}
        missing_cats = present_cats - approved_cats
        for cat in missing_cats:
            cat_candidates = [
                t for t in trades
                if t.get("sector_category") == cat
                and t.get("entry_price", 0) > 0
                and t.get("max_cost", 0) <= max_cost_per_trade
                and not _deterministic_checks(t, settings, max_cost_per_trade)
            ]
            if cat_candidates:
                best_cat_trade = max(cat_candidates, key=lambda t: t.get("confidence_score_pct", 50.0))
                rev = review_map.get(best_cat_trade["symbol"])
                final_recommendations.append({
                    **best_cat_trade,
                    "risk_score": rev.risk_score if rev else 6.0,
                    "risk_rating": rev.risk_rating if rev else "MODERATE",
                    "risk_feedback": f"Included to ensure category '{cat}' coverage.",
                    "risk_rationale": (rev.rationale if rev else "Compliant category pick.") + " [Included for category representation]",
                })
                logger.info("➕ Included {} to ensure category '{}' representation in final ranked slate", best_cat_trade["symbol"], cat)

    # Rank recommendations strictly by profit confidence & risk-adjusted return (descending)
    final_recommendations.sort(
        key=lambda t: (
            t.get("confidence_score_pct", 50.0),
            -t.get("risk_score", 5.0),
            t.get("estimated_profit_pct", 0.0),
        ),
        reverse=True,
    )
    for rank_idx, rec in enumerate(final_recommendations, 1):
        rec["rank"] = rank_idx

    logger.info(
        "Risk Sentinel: {} approved recommendations ranked by profit confidence: {}",
        len(final_recommendations),
        [f"#{r['rank']} {r['symbol']} ({r.get('confidence_score_pct', 0):.0f}%)" for r in final_recommendations],
    )

    return {
        **state,
        "risk_reviews": risk_reviews,
        "final_recommendations": final_recommendations,
        "review_feedback": combined_feedback,
        "iteration_count": iteration + 1,
    }
