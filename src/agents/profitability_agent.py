"""
Agent 2: Trade Structurer & Profitability Analyst
=================================================
For each scanner proposal, fetches the live option chain, scores contracts
using Black-Scholes Greeks, and uses the LLM to select the optimal structure
(single-leg long or defined-risk debit spread) with full PnL analysis.

Flow
----
AgentTradingState.scanner_proposals
    → option chain fetch + Greeks calculation
    → LLM selects optimal contract and structures the trade
    → AgentTradingState.structured_trades
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings
from src.agents.state import AgentTradingState
from src.tools.market_data import get_market_timing_context, get_option_chain
from src.tools.options_math import (
    ScoredContract,
    breakeven,
    pnl_at_expiry,
    score_contract,
    size_position,
)


# ── LLM Output Schema ──────────────────────────────────────────────────────────

class StructuredTrade(BaseModel):
    symbol: str
    option_type: str = Field(description="CALL or PUT")
    strike: float = Field(description="Selected long strike price")
    short_strike: float | None = Field(
        default=None,
        description="Higher short call strike price when structure is CALL_DEBIT_SPREAD; None for LONG_CALL",
    )
    expiration: str = Field(description="Expiration date YYYY-MM-DD")
    dte: int = Field(description="Days to expiry")
    entry_price: float = Field(description="Estimated net debit premium per share (mid-price)")
    target_exit_price: float = Field(description="Target exit premium (aiming for steady 30% - 50% gain)")
    stop_loss_price: float = Field(description="Stop-loss premium (tight 20% - 25% loss protection)")
    breakeven_price: float = Field(description="Underlying price at which position breaks even at expiry")
    estimated_profit_usd: float = Field(description="Estimated dollar profit per contract at target exit ($)")
    estimated_profit_pct: float = Field(description="Estimated percentage gain at target exit (e.g. 50.0)")
    max_contracts: int = Field(description="Max contracts that strictly fit within the fixed dollar budget")
    max_cost: float = Field(description="Total dollar cost of the position (must stay within per-trade dollar budget)")
    delta: float = Field(description="Option delta")
    gamma: float = Field(description="Option gamma")
    theta: float = Field(description="Option theta (daily)")
    vega: float = Field(description="Option vega (per 1% IV)")
    implied_volatility: float = Field(description="Implied volatility %")
    structure: str = Field(
        description="DEFINED RISK ONLY (NO SHARES OR MARGIN NEEDED): LONG_CALL | CALL_DEBIT_SPREAD"
    )
    contract_symbol: str = Field(description="OCC contract symbol", default="")

    # ── Comprehensive Advisory Details ─────────────────────────────────────────
    why_selected: str = Field(
        description="Detailed explanation of WHY this specific trade, strike, and structure were chosen (catalyst, technical setup, IV profile, and advantage over alternatives)"
    )
    when_to_buy: str = Field(
        description="Actionable SAME-DAY entry execution plan placeable TODAY as a Good-for-Day (DAY) limit order at live market prices, including max chase limit price and intraday confirmation check. Never tell the user to wait for Monday or another day."
    )
    sell_conditions: str = Field(
        description="Explicit rules for when to sell: 1) Profit target (+30-50%), 2) Stop-loss (-25%), 3) Time stop (e.g. exit 1-2 DTE if flat), 4) Technical invalidation"
    )
    early_warning_signs: list[str] = Field(
        description="2-4 specific early warning signs indicating increasing risk before the stop-loss is triggered (e.g., underlying losing support, adverse volume, IV crush)"
    )
    political_catalysts: str = Field(
        description="Recent or expected political / regulatory catalysts affecting this trade and sector", default=""
    )
    earnings_context: str = Field(
        description="Next earnings date, days until earnings, last reported surprise %, and whether this trade crosses earnings", default=""
    )
    company_news_context: str = Field(
        description="Key company news, earnings call takeaways, or corporate developments", default=""
    )
    confidence_rating: str = Field(
        description="Overall confidence in achieving estimated profits: HIGH | MEDIUM | LOW"
    )
    confidence_score_pct: float = Field(
        description="Estimated probability/confidence score in achieving target profit (e.g. 75.0 for 75%)", ge=0, le=100
    )
    thesis: str = Field(description="2-3 sentence executive thesis")


class StructurerOutput(BaseModel):
    trades: list[StructuredTrade] = Field(description="List of fully structured trades")
    structuring_notes: str = Field(description="Brief notes on strategy selection and budget compliance")

    @field_validator("trades", mode="before")
    @classmethod
    def _coerce_trades(cls, v: Any) -> Any:
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


# ── Prompt Builder ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a master derivatives trader and quantitative options structurer. You are AGENT 2 in a multi-agent trading pipeline.

Your mission is to provide COMPLETE, ACTIONABLE, AND RIGOROUS trade advisory recommendations. Every recommendation must explain not just what to trade, but exactly WHY, WHEN TO ENTER, WHEN TO EXIT, WHAT WARNING SIGNS TO MONITOR, and THE CONFIDENCE LEVEL.

MANDATORY REQUIREMENTS:
1. COMPLETE ADVISORY DETAILS & SAME-DAY ACTIONABILITY FOR EVERY TRADE:
   - why_selected: Thoroughly explain why this ticker, strike(s), and structure were chosen over alternatives. Mention technical catalysts, momentum/reversion indicators, and IV rank.
   - estimated_profit_usd & estimated_profit_pct: State exact expected dollar gain per contract ((target_exit - entry) * 100) and expected percentage gain (e.g. 50%).
   - when_to_buy: MUST be executable TODAY (same-day as the pipeline run). Give exact DAY limit order instructions referencing today's date/session (e.g., "Place a DAY limit buy order TODAY at or below $2.45/sh...") rather than waiting for Monday or tomorrow's daily close.
   - sell_conditions: Explicitly outline the 4 exit triggers:
       * Profit Target Exit: Close at +30% to +50% gain (capped below max spread width if spread).
       * Stop-Loss Exit: Cut position immediately at -20% to -25% loss.
       * Time Stop: Exit 1-2 days before expiration to avoid weekend/terminal theta decay.
       * Technical Invalidation: Exit early if underlying breaks key technical support level.
   - early_warning_signs: Provide 2-4 tangible early warning indicators that risk is rising before the stop-loss triggers.
   - confidence_rating & confidence_score_pct: Score true profit-achievement confidence (0-100%) differentiating higher-probability setups (e.g. 60-80% on low-IV / strong RS names or defined-risk debit spreads) from speculative setups (35-55%).

2. ZERO MARGIN SELLING / NO SINGLE-LEG PUTS / STRICTLY CALLS ONLY:
   User does NOT own 100 shares of underlying stock and has NO margin for short options or delivering stock.
   STRICTLY FORBIDDEN:
   - NO LONG_PUT (prevents exercise risk where delivery of 100 shares is needed)
   - NO naked short options
   ONLY PERMITTED:
   - LONG_CALL
   - CALL_DEBIT_SPREAD (MUST populate `short_strike` with the higher call strike!)
   Net debit paid is the absolute maximum risk. Zero shares required.

3. STEADY WEEKLY PROFITS & LOSS MINIMIZATION:
   Target +30% to +50% gain; stop-loss at -20% to -25%. For single-leg LONG_CALL, delta should be 0.35-0.55. For CALL_DEBIT_SPREAD, net delta is positive and debit MUST be <= 75% of strike width.

4. POLITICAL CATALYSTS & EARNINGS TIMING INTEGRATION:
   - Carefully review upcoming earnings dates and political/regulatory catalysts.
   - If expiration date crosses an earnings release: explicitly note binary earnings risk in `earnings_context` and use a CALL_DEBIT_SPREAD to prevent IV crush. If expiration precedes earnings, note that the trade exits before earnings volatility.
   - Fully populate `political_catalysts`, `earnings_context`, and `company_news_context` with concrete intelligence.

5. MANDATORY DOLLAR BUDGET & CATEGORY COVERAGE:
   - Produce a valid trade for EVERY candidate/category presented so each sector category has representation!
   - Total cost per trade MUST be <= max_cost_per_trade_usd.
   - If single-leg CALL contracts exceed the per-trade dollar budget (e.g., expensive stocks like MU, NVDA, AMD, TSM, AVGO), YOU MUST select one of the pre-computed budget-compliant CALL_DEBIT_SPREAD pairs (which cost <= $500 per spread) instead of picking an over-budget single call!

Respond ONLY with valid JSON matching StructurerOutput schema."""


def _build_debit_spread_candidates(
    scored_calls: list[ScoredContract],
    underlying_price: float,
    max_cost_per_trade: float,
    max_results: int = 6,
) -> list[dict[str, Any]]:
    """
    Deterministically pair live scored calls into budget-compliant vertical
    CALL_DEBIT_SPREAD candidates (long strike < short strike, debit <= max_cost_per_trade).
    """
    if underlying_price <= 0 or not scored_calls:
        return []

    max_debit_per_share = max_cost_per_trade / 100.0
    # Filter plausible long strikes near-the-money and short strikes above them
    calls_sorted = sorted(scored_calls, key=lambda c: c.strike)
    long_pool = [
        c for c in calls_sorted
        if (underlying_price * 0.88 <= c.strike <= underlying_price * 1.06)
        and (0.28 <= abs(c.greeks.delta) <= 0.78)
    ]
    if not long_pool:
        long_pool = calls_sorted[:10]

    spreads: list[dict[str, Any]] = []
    for lc in long_pool:
        for sc in calls_sorted:
            if sc.strike <= lc.strike:
                continue
            width = round(sc.strike - lc.strike, 2)
            if width <= 0 or width > max(15.0, underlying_price * 0.08):
                continue
            debit = round(lc.mid - sc.mid, 2)
            if debit < 0.25 or debit > max_debit_per_share:
                continue
            if debit > width * 0.78:
                continue  # require meaningful risk/reward upside

            max_profit = round((width - debit) * 100, 2)
            max_roi_pct = round(((width - debit) / debit) * 100, 1)
            net_delta = round(lc.greeks.delta - sc.greeks.delta, 3)
            if net_delta < 0.06:
                continue  # ensure net delta clears Risk Sentinel 0.05 floor
            net_theta = round(lc.greeks.theta - sc.greeks.theta, 4)
            be = round(lc.strike + debit, 2)
            be_pct = round(((be - underlying_price) / underlying_price) * 100, 2)

            # Composite score: reward tight breakeven, solid ROI, positive delta, and budget fit
            roi_bonus = min(max_roi_pct / 50.0, 2.5)
            delta_bonus = max(0.0, net_delta * 4.0)
            be_penalty = max(0.0, abs(be_pct) * 0.15)
            score = round(roi_bonus + delta_bonus - be_penalty, 3)

            spreads.append({
                "long_strike": lc.strike,
                "short_strike": sc.strike,
                "expiration": lc.expiration,
                "dte": lc.dte,
                "debit": debit,
                "cost_per_contract": round(debit * 100, 2),
                "width": width,
                "max_profit_per_contract": max_profit,
                "max_roi_pct": max_roi_pct,
                "net_delta": net_delta,
                "net_theta": net_theta,
                "iv": lc.greeks.implied_volatility,
                "breakeven": be,
                "be_pct_move": be_pct,
                "long_symbol": lc.contract_symbol,
                "score": score,
            })

    spreads.sort(key=lambda x: x["score"], reverse=True)
    return spreads[:max_results]


def _format_scored_contracts(
    scored: list[ScoredContract], max_cost_per_trade: float = 500.0, n: int = 8
) -> str:
    """Format top N scored single-leg contracts (prioritizing budget-compliant ones first)."""
    max_debit = max_cost_per_trade / 100.0
    ordered = sorted(
        scored,
        key=lambda x: (x.mid <= max_debit, x.score),
        reverse=True,
    )[:n]
    lines: list[str] = []
    for c in ordered:
        g = c.greeks
        budget_tag = "FITS_BUDGET" if c.mid <= max_debit else "OVER_BUDGET_SINGLE_LEG"
        lines.append(
            f"  strike={c.strike} | exp={c.expiration} | DTE={c.dte} | mid=${c.mid:.2f} (${c.mid*100:.0f}/ctr [{budget_tag}]) | delta={g.delta:.2f} | "
            f"gamma={g.gamma:.5f} | theta={g.theta:.4f}/d | vega={g.vega:.4f} | "
            f"IV={g.implied_volatility:.1f}% | BE={c.breakeven:.2f} ({c.breakeven_pct_move:.1f}% move) | "
            f"symbol={c.contract_symbol} | score={c.score} | {c.rationale}"
        )
    return "\n".join(lines) if lines else "  (no contracts available)"


def _format_debit_spreads(spreads: list[dict[str, Any]]) -> str:
    """Format pre-computed CALL_DEBIT_SPREAD pairs for the LLM prompt."""
    lines: list[str] = []
    for s in spreads:
        lines.append(
            f"  SPREAD {s['long_strike']}/{s['short_strike']} | exp={s['expiration']} | DTE={s['dte']} | "
            f"net_debit=${s['debit']:.2f} (${s['cost_per_contract']:.0f}/spread [FITS_BUDGET]) | "
            f"width=${s['width']:.1f} | max_profit=${s['max_profit_per_contract']:.0f} (+{s['max_roi_pct']:.0f}% max ROI) | "
            f"net_delta={s['net_delta']:.2f} | net_theta={s['net_theta']:.4f}/d | IV={s['iv']:.1f}% | "
            f"BE={s['breakeven']:.2f} ({s['be_pct_move']:+.1f}%) | long_sym={s['long_symbol']}"
        )
    return "\n".join(lines) if lines else "  (no budget-compliant spreads found)"


def _build_human_prompt(
    proposals: list[dict[str, Any]],
    scored_chains: dict[str, dict[str, list[ScoredContract]]],
    spread_candidates: dict[str, list[dict[str, Any]]],
    portfolio_value: float,
    max_risk_pct: float,
    weekly_budget: float,
    max_cost_per_trade: float,
    revision_feedback: str | None,
) -> str:
    settings = get_settings()
    mkt_time = get_market_timing_context()
    parts: list[str] = []

    if revision_feedback:
        parts.append(
            f"## ⚠️ REVISION PASS — Address This Feedback\n{revision_feedback}\n"
            "CRITICAL: Keep previously approved trades intact in your output alongside any revised/re-structured trades so EVERY sector category remains represented.\n"
        )

    parts.append(
        f"## ⏰ LIVE EXECUTION TIMING (SAME-DAY MANDATE)\n"
        f"- Date & Time: {mkt_time['day_of_week']}, {mkt_time['today_date']} at {mkt_time['time_et']}\n"
        f"- Session: {mkt_time['market_session']}\n"
        f"- Requirement: {mkt_time['execution_guidance']}\n"
    )

    parts.append(
        f"## Strict Budget & Profit Targets\n"
        f"- Fixed Weekly Budget: ${weekly_budget:,.2f} total\n"
        f"- Max Cost Per Trade: ${max_cost_per_trade:,.2f} (Hard Cap — NO position may exceed this)\n"
        f"- Target Return: +{settings.target_gain_percent:.0f}% (Steady Weekly Gain; cap spread target <= 90% of strike width)\n"
        f"- Stop Loss: -{settings.stop_loss_percent:.0f}% (Strict Loss Minimization)\n"
        f"- Margin & Share Policy: STRICTLY CALLS ONLY. NO SHARES REQUIRED. NO MARGIN. Only BUY CALL options (LONG_CALL) or vertical CALL_DEBIT_SPREAD. NO PUTS.\n"
    )

    for proposal in proposals:
        sym = proposal["symbol"]
        opt_type = proposal.get("suggested_option_type", "CALL")
        price = proposal.get("underlying_price", 0.0)
        tech = proposal.get("technicals", {})
        iv_rank = tech.get("iv_rank") or 0
        sec_cat = proposal.get("sector_category", "OTHER")
        sec_name = proposal.get("sector_name") or tech.get("sector_name", "Broad Market")
        sec_etf = tech.get("sector_etf", "SPY")
        sec_7d = tech.get("sector_7d_change_pct")
        sec_range = tech.get("sector_7d_range", "N/A")
        stock_7d = tech.get("change_7d_pct")
        rs_7d = tech.get("relative_strength_7d_pct")

        parts.append(f"\n### {sym} [{sec_cat}: {sec_name}] @ ${price:.2f} | Direction: {opt_type} | IV Rank: {iv_rank:.0f}")
        parts.append(
            f"🌐 Sector 7-Day Context: {sec_name} ({sec_etf}) moved {f'{sec_7d:+.2f}%' if sec_7d is not None else '0.00%'} "
            f"[7d Range: {sec_range}]. Stock 7d: {f'{stock_7d:+.2f}%' if stock_7d is not None else 'N/A'}, Relative Strength vs Sector: {f'{rs_7d:+.2f}%' if rs_7d is not None else 'N/A'}"
        )
        parts.append(f"Scanner thesis: {proposal.get('thesis', 'N/A')}")
        parts.append(f"Catalyst: {proposal.get('catalyst', 'N/A')}")
        if proposal.get("political_catalysts"):
            parts.append(f"🏛️ Political/Policy Catalysts: {proposal.get('political_catalysts')}")
        if proposal.get("earnings_context"):
            parts.append(f"📅 Earnings Intelligence: {proposal.get('earnings_context')}")
        if proposal.get("company_news_context"):
            parts.append(f"📰 Company News/Developments: {proposal.get('company_news_context')}")
        if proposal.get("news_catalyst"):
            parts.append(f"Company News Driver: {proposal.get('news_catalyst')}")
        if proposal.get("political_macro_impact"):
            parts.append(f"US Political/Policy Impact: {proposal.get('political_macro_impact')}")

        sym_scored = scored_chains.get(sym, {})
        c_contracts = sym_scored.get("CALL", [])
        sym_spreads = spread_candidates.get(sym, [])

        if c_contracts:
            parts.append(f"\n**Single-Leg CALL contracts (sorted by budget fit & score):**")
            parts.append(_format_scored_contracts(c_contracts, max_cost_per_trade=max_cost_per_trade))
        if sym_spreads:
            parts.append(f"\n**Pre-Computed CALL_DEBIT_SPREAD candidates (ALL FITS BUDGET <= ${max_cost_per_trade:.0f}; set short_strike!):**")
            parts.append(_format_debit_spreads(sym_spreads))
        if not c_contracts and not sym_spreads:
            parts.append("  ⚠️ No call option chain data available — skip this ticker")

    parts.append(
        f"\nStructure one compliant, same-day actionable trade for EACH viable proposal so all sector categories remain represented. "
        f"Ensure EVERY trade is defined-risk (LONG_CALL or CALL_DEBIT_SPREAD with short_strike set) and max_cost <= ${max_cost_per_trade:.2f}. "
        "Respond with JSON matching StructurerOutput schema."
    )
    return "\n".join(parts)


# ── Option Chain Scoring ───────────────────────────────────────────────────────

def _score_chain(
    chain_result,
    opt_type: Literal["CALL", "PUT"],
    underlying_price: float,
) -> list[ScoredContract]:
    contracts = chain_result.calls if opt_type == "CALL" else chain_result.puts
    scored: list[ScoredContract] = []
    for c in contracts:
        if c.mid <= 0.05 or c.mid > underlying_price * 0.20:
            continue  # skip illiquid or unrealistically expensive
        sc = score_contract(
            contract_mid=c.mid,
            strike=c.strike,
            dte=c.dte,
            underlying_price=underlying_price,
            option_type=opt_type,  # type: ignore[arg-type]
            contract_symbol=c.contract_symbol,
            expiration_str=str(c.expiration),
            iv_override=c.implied_volatility,
        )
        if sc:
            scored.append(sc)
    return scored


# ── Agent Node ─────────────────────────────────────────────────────────────────

def profitability_node(state: AgentTradingState) -> AgentTradingState:
    """
    LangGraph node: structures option trades for all scanner proposals.
    Updates state with structured_trades.
    """
    settings = get_settings()
    proposals = state.get("scanner_proposals") or []
    revision_feedback = state.get("review_feedback")

    if not proposals:
        logger.warning("Structurer: no scanner proposals to process")
        return {**state, "structured_trades": []}

    logger.info(
        "📐 Structurer Agent: structuring {} proposals (iteration {})",
        len(proposals),
        state.get("iteration_count", 0),
    )

    # Step 1: Fetch and score option chains for each proposal
    # Budget constraints from state or settings
    weekly_budget = float(state.get("weekly_budget_usd") or settings.weekly_budget_usd)
    max_cost_per_trade = float(state.get("max_cost_per_trade_usd") or settings.max_cost_per_trade_usd)

    portfolio_value = 100_000.0  # default; will be overridden by broker query if available
    try:
        from src.tools.broker import get_buying_power
        bp = get_buying_power("PAPER")
        if bp and bp > 1000:
            portfolio_value = bp
    except Exception:
        pass
    scored_chains: dict[str, dict[str, list[ScoredContract]]] = {}
    spread_candidates: dict[str, list[dict[str, Any]]] = {}

    for proposal in proposals:
        sym = proposal["symbol"]
        opt_type = proposal.get("suggested_option_type", "CALL")
        logger.debug("Fetching option chain for {}", sym)

        chain = get_option_chain(sym, target_dte=settings.target_dte)
        if chain is None:
            logger.warning("No option chain for {}", sym)
            continue

        underlying = chain.underlying_price or proposal.get("underlying_price", 0.0)
        # Update proposal with live price
        proposal["underlying_price"] = underlying

        scored_calls = _score_chain(chain, "CALL", underlying)
        scored_puts = _score_chain(chain, "PUT", underlying)
        scored_chains[sym] = {"CALL": scored_calls, "PUT": scored_puts}
        spread_candidates[sym] = _build_debit_spread_candidates(
            scored_calls=scored_calls,
            underlying_price=underlying,
            max_cost_per_trade=max_cost_per_trade,
        )
        logger.debug(
            "{}: {} calls, {} spreads scored",
            sym,
            len(scored_calls),
            len(spread_candidates[sym]),
        )

    # Step 2: LLM structuring
    from src.agents.llm_resilience import invoke_structured_with_fallback

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_human_prompt(
                proposals=proposals,
                scored_chains=scored_chains,
                spread_candidates=spread_candidates,
                portfolio_value=portfolio_value,
                max_risk_pct=settings.max_risk_per_trade_percent,
                weekly_budget=weekly_budget,
                max_cost_per_trade=max_cost_per_trade,
                revision_feedback=revision_feedback,
            )
        ),
    ]

    try:
        output: StructurerOutput = invoke_structured_with_fallback(
            messages=messages,
            schema=StructurerOutput,
            temperature=0.1,
            request_timeout=35,
        )
        logger.info(
            "Structurer produced {} trades: {}",
            len(output.trades),
            [
                f"{t.symbol} {t.structure} ${t.strike}" + (f"/${t.short_strike}" if t.short_strike else "")
                for t in output.trades
            ],
        )
    except Exception as exc:
        logger.error("Structurer LLM failed: {}. Building fallback trades.", exc)
        mkt_time = get_market_timing_context()
        # Graceful fallback: pick best-scored budget-compliant contract or spread per proposal
        fallback_trades: list[StructuredTrade] = []
        for proposal in proposals:
            sym = proposal["symbol"]
            opt_type = "CALL"
            sym_scored = scored_chains.get(sym, {})
            contracts = [c for c in sym_scored.get("CALL", []) if c.mid * 100 <= max_cost_per_trade]
            sym_spreads = spread_candidates.get(sym, [])
            underlying = proposal.get("underlying_price", 0.0)
            profit_pct = settings.target_gain_percent

            if contracts:
                best = max(contracts, key=lambda c: c.score)
                contracts_count = max(1, int(max_cost_per_trade / (best.mid * 100)))
                actual_cost = round(contracts_count * best.mid * 100, 2)
                be = breakeven(best.strike, best.mid, "CALL")
                target_exit = round(best.mid * (1 + profit_pct / 100), 2)
                stop_exit = round(best.mid * (1 - settings.stop_loss_percent / 100), 2)
                profit_usd = round((target_exit - best.mid) * 100, 2)
                fallback_trades.append(
                    StructuredTrade(
                        symbol=sym,
                        option_type="CALL",
                        strike=best.strike,
                        short_strike=None,
                        expiration=best.expiration,
                        dte=best.dte,
                        entry_price=best.mid,
                        target_exit_price=target_exit,
                        stop_loss_price=stop_exit,
                        breakeven_price=be,
                        estimated_profit_usd=profit_usd,
                        estimated_profit_pct=profit_pct,
                        max_contracts=contracts_count,
                        max_cost=actual_cost,
                        delta=best.greeks.delta,
                        gamma=best.greeks.gamma,
                        theta=best.greeks.theta,
                        vega=best.greeks.vega,
                        implied_volatility=best.greeks.implied_volatility,
                        thesis=proposal.get("thesis", "Fallback: best-scored call contract"),
                        structure="LONG_CALL",
                        contract_symbol=best.contract_symbol,
                        why_selected=f"Selected {sym} LONG_CALL at ${best.strike} due to composite score ({best.score}) and delta ({best.greeks.delta:.2f}).",
                        when_to_buy=f"Place a DAY limit buy order TODAY ({mkt_time['today_date']}) at or below ${best.mid:.2f}/sh.",
                        sell_conditions=f"1) Profit target at ${target_exit:.2f} (+{profit_pct:.0f}%). 2) Stop loss at ${stop_exit:.2f} (-{settings.stop_loss_percent:.0f}%). 3) Time stop: exit 2 DTE before expiry. 4) Exit on technical breakdown.",
                        early_warning_signs=[
                            "Underlying breaks key intraday support on elevated volume",
                            "Implied volatility contracts rapidly without upside move",
                        ],
                        confidence_rating="MEDIUM",
                        confidence_score_pct=60.0,
                    )
                )
            elif sym_spreads:
                sp = sym_spreads[0]
                contracts_count = max(1, int(max_cost_per_trade / sp["cost_per_contract"]))
                actual_cost = round(contracts_count * sp["cost_per_contract"], 2)
                raw_target = round(sp["debit"] * (1 + profit_pct / 100), 2)
                target_exit = min(raw_target, round(sp["width"] * 0.90, 2))
                eff_pct = round(((target_exit - sp["debit"]) / sp["debit"]) * 100, 1)
                stop_exit = round(sp["debit"] * (1 - settings.stop_loss_percent / 100), 2)
                profit_usd = round((target_exit - sp["debit"]) * 100, 2)
                fallback_trades.append(
                    StructuredTrade(
                        symbol=sym,
                        option_type="CALL",
                        strike=sp["long_strike"],
                        short_strike=sp["short_strike"],
                        expiration=sp["expiration"],
                        dte=sp["dte"],
                        entry_price=sp["debit"],
                        target_exit_price=target_exit,
                        stop_loss_price=stop_exit,
                        breakeven_price=sp["breakeven"],
                        estimated_profit_usd=profit_usd,
                        estimated_profit_pct=eff_pct,
                        max_contracts=contracts_count,
                        max_cost=actual_cost,
                        delta=sp["net_delta"],
                        gamma=0.0,
                        theta=sp["net_theta"],
                        vega=0.0,
                        implied_volatility=sp["iv"],
                        thesis=proposal.get("thesis", "Fallback: budget-compliant debit spread"),
                        structure="CALL_DEBIT_SPREAD",
                        contract_symbol=f"{sym}_SPREAD_{sp['long_strike']:g}_{sp['short_strike']:g}",
                        why_selected=f"Selected {sym} CALL_DEBIT_SPREAD ${sp['long_strike']}/${sp['short_strike']} to cap outlay at ${sp['cost_per_contract']:.0f} while buffering IV.",
                        when_to_buy=f"Place a DAY limit debit spread order TODAY ({mkt_time['today_date']}) at or below ${sp['debit']:.2f}/sh net debit.",
                        sell_conditions=f"1) Profit target at ${target_exit:.2f} (+{eff_pct:.0f}%). 2) Stop loss at ${stop_exit:.2f} (-{settings.stop_loss_percent:.0f}%). 3) Time stop: exit 2 DTE before expiry. 4) Exit on technical breakdown.",
                        early_warning_signs=[
                            "Underlying loses intraday momentum below short-term VWAP",
                            "Spread bid/ask widens against long leg",
                        ],
                        confidence_rating="MEDIUM",
                        confidence_score_pct=58.0,
                    )
                )
        output = StructurerOutput(
            trades=fallback_trades,
            structuring_notes="Fallback: LLM unavailable, used best-scored budget-compliant structure per proposal.",
        )

    # Enrich & deterministically heal trades (guarantee budget compliance, valid spreads & expirations)
    structured: list[dict[str, Any]] = []
    for trade in output.trades:
        sym_scored = scored_chains.get(trade.symbol, {})
        contracts_list = sym_scored.get("CALL", [])
        sym_spreads = spread_candidates.get(trade.symbol, [])

        # Ensure CALL_DEBIT_SPREAD has valid short_strike and live spread debit
        if "SPREAD" in (trade.structure or "").upper():
            if not trade.short_strike and sym_spreads:
                sp = min(sym_spreads, key=lambda x: abs(x["long_strike"] - trade.strike))
                trade.strike = sp["long_strike"]
                trade.short_strike = sp["short_strike"]
                trade.entry_price = sp["debit"]
                trade.expiration = sp["expiration"]
                trade.dte = sp["dte"]
            elif trade.short_strike:
                lc = next((sc for sc in contracts_list if abs(sc.strike - trade.strike) < 0.05), None)
                sc_leg = next((sc for sc in contracts_list if abs(sc.strike - trade.short_strike) < 0.05), None)
                if lc and sc_leg:
                    live_debit = round(lc.mid - sc_leg.mid, 2)
                    if live_debit > 0:
                        trade.entry_price = live_debit
                    if not trade.expiration or trade.dte <= 0:
                        trade.expiration = lc.expiration
                        trade.dte = lc.dte
                    if not trade.contract_symbol:
                        trade.contract_symbol = f"{trade.symbol}_SPREAD_{trade.strike:g}_{trade.short_strike:g}"
            # Cap spread target exit within width
            if trade.short_strike:
                width = abs(trade.short_strike - trade.strike)
                if width > 0 and trade.target_exit_price > width:
                    trade.target_exit_price = round(min(trade.entry_price * 1.5, width * 0.90), 2)
                    if trade.entry_price > 0:
                        trade.estimated_profit_pct = round(((trade.target_exit_price - trade.entry_price) / trade.entry_price) * 100, 1)
                        trade.estimated_profit_usd = round((trade.target_exit_price - trade.entry_price) * 100, 2)
        else:
            # Single-leg LONG_CALL lookup
            matched = next((sc for sc in contracts_list if abs(sc.strike - trade.strike) < 0.05), None)
            if not matched and contracts_list:
                affordable = [sc for sc in contracts_list if sc.mid * 100 <= max_cost_per_trade]
                matched = min(affordable or contracts_list, key=lambda sc: abs(sc.strike - trade.strike))
            if matched:
                if not trade.expiration or trade.expiration == "" or trade.dte <= 0:
                    trade.expiration = matched.expiration
                    trade.dte = matched.dte
                if not trade.contract_symbol:
                    trade.contract_symbol = matched.contract_symbol
                if trade.entry_price <= 0:
                    trade.entry_price = matched.mid

        # Auto-heal over-budget trade: if 1 contract still exceeds max_cost_per_trade, convert to top budget-compliant spread or affordable call
        if (trade.entry_price <= 0 or trade.entry_price * 100 > max_cost_per_trade):
            if sym_spreads:
                sp = sym_spreads[0]
                logger.info(
                    "♻️ Auto-healing {} from over-budget ${:.2f}/ctr to compliant CALL_DEBIT_SPREAD {}/{} (${:.2f})",
                    trade.symbol,
                    trade.entry_price * 100,
                    sp["long_strike"],
                    sp["short_strike"],
                    sp["cost_per_contract"],
                )
                trade.structure = "CALL_DEBIT_SPREAD"
                trade.strike = sp["long_strike"]
                trade.short_strike = sp["short_strike"]
                trade.expiration = sp["expiration"]
                trade.dte = sp["dte"]
                trade.entry_price = sp["debit"]
                trade.contract_symbol = f"{trade.symbol}_SPREAD_{sp['long_strike']:g}_{sp['short_strike']:g}"
                trade.breakeven_price = sp["breakeven"]
                trade.delta = sp["net_delta"]
                trade.theta = sp["net_theta"]
                trade.target_exit_price = min(round(sp["debit"] * 1.5, 2), round(sp["width"] * 0.90, 2))
                trade.stop_loss_price = round(sp["debit"] * 0.75, 2)
                trade.estimated_profit_usd = round((trade.target_exit_price - sp["debit"]) * 100, 2)
                trade.estimated_profit_pct = round(((trade.target_exit_price - sp["debit"]) / sp["debit"]) * 100, 1)
            else:
                affordable_calls = [c for c in contracts_list if c.mid * 100 <= max_cost_per_trade]
                if affordable_calls:
                    best_c = max(affordable_calls, key=lambda c: c.score)
                    trade.structure = "LONG_CALL"
                    trade.short_strike = None
                    trade.strike = best_c.strike
                    trade.expiration = best_c.expiration
                    trade.dte = best_c.dte
                    trade.entry_price = best_c.mid
                    trade.contract_symbol = best_c.contract_symbol
                    trade.delta = best_c.greeks.delta
                    trade.theta = best_c.greeks.theta
                    trade.target_exit_price = round(best_c.mid * 1.5, 2)
                    trade.stop_loss_price = round(best_c.mid * 0.75, 2)
                    trade.estimated_profit_usd = round((trade.target_exit_price - best_c.mid) * 100, 2)
                    trade.estimated_profit_pct = 50.0

        # Clamp contracts and cost strictly to user's max_cost_per_trade
        if trade.entry_price > 0:
            max_allowed = max(1, int(max_cost_per_trade / (trade.entry_price * 100)))
            contracts = max(1, min(trade.max_contracts, max_allowed))
            trade.max_contracts = contracts
            trade.max_cost = round(contracts * trade.entry_price * 100, 2)
        pnl = pnl_at_expiry(
            option_type=trade.option_type,  # type: ignore[arg-type]
            strike=trade.strike,
            premium_paid=trade.entry_price,
            underlying_price=next(
                (p["underlying_price"] for p in proposals if p["symbol"] == trade.symbol),
                trade.strike,
            ),
        )
        trade_dict = trade.model_dump()
        trade_dict["pnl_curve"] = {
            "breakeven_price": pnl.breakeven_price,
            "max_profit": pnl.max_profit,
            "max_loss": pnl.max_loss,
            "target_exit_spot": pnl.target_exit_spot,
        }
        trade_dict["structuring_notes"] = output.structuring_notes

        # Attach sector & macro intelligence from proposal
        matching_prop = next((p for p in proposals if p["symbol"] == trade.symbol), {})
        tech_data = matching_prop.get("technicals", {})
        trade_dict["sector_category"] = matching_prop.get("sector_category")
        trade_dict["sector_name"] = matching_prop.get("sector_name") or tech_data.get("sector_name")
        trade_dict["sector_etf"] = tech_data.get("sector_etf")
        trade_dict["sector_7d_change_pct"] = tech_data.get("sector_7d_change_pct")
        trade_dict["sector_7d_range"] = tech_data.get("sector_7d_range")
        trade_dict["relative_strength_7d_pct"] = tech_data.get("relative_strength_7d_pct")
        trade_dict["political_macro_impact"] = matching_prop.get("political_macro_impact")
        trade_dict["news_catalyst"] = matching_prop.get("news_catalyst")
        if not trade_dict.get("political_catalysts"):
            trade_dict["political_catalysts"] = matching_prop.get("political_catalysts")
        if not trade_dict.get("earnings_context"):
            trade_dict["earnings_context"] = matching_prop.get("earnings_context")
        if not trade_dict.get("company_news_context"):
            trade_dict["company_news_context"] = matching_prop.get("company_news_context")

        structured.append(trade_dict)

    # Preserve already-approved or previously structured trades if omitted during revision pass
    if revision_feedback:
        prior_map: dict[str, dict[str, Any]] = {}
        for t in state.get("structured_trades") or []:
            prior_map[t["symbol"]] = t
        for r in state.get("final_recommendations") or []:
            prior_map[r["symbol"]] = r

        covered_syms = {t["symbol"] for t in structured}
        for prop in proposals:
            p_sym = prop["symbol"]
            if p_sym not in covered_syms and p_sym in prior_map:
                structured.append(prior_map[p_sym])
                covered_syms.add(p_sym)
                logger.info("♻️ Preserved previously structured/approved {} across revision pass", p_sym)

    return {**state, "structured_trades": structured, "review_feedback": None}
