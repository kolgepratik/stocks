"""
Agent 1: Opportunity Scanner & Multi-Sector Research Supervisor
==============================================================
Coordinates specialized Sector Subagents (AI/Chips, Energy/Gas, Data Center Nuclear Power),
ingests live US political & Fed macro headlines, earnings intelligence, and trailing 7-day
news, and synthesizes 3-4 high-conviction trade candidates across target sectors.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings
from src.agents.sector_subagents import (
    SectorResearchDossier,
    get_symbol_sector_info,
    run_all_sector_subagents,
)
from src.agents.state import AgentTradingState
from src.tools.market_data import (
    CandidateTicker,
    get_market_timing_context,
    get_weekly_candidates,
)
from src.tools.news_research import build_market_news_brief


# ── Pydantic Output Schema ─────────────────────────────────────────────────────

class ScannerProposal(BaseModel):
    symbol: str = Field(description="Ticker symbol (e.g. NVDA, OKLO, EQT)")
    direction: str = Field(description="BULLISH or BEARISH")
    thesis: str = Field(description="1-3 sentence trade thesis explaining why this ticker is attractive now")
    catalyst: str = Field(description="Specific technical catalyst or breakout/reversal pattern")
    news_catalyst: str = Field(description="Key company-specific headline driver or event")
    political_macro_impact: str = Field(
        description="How current US political landscape, Federal Reserve policy, trade/tariffs, or regulatory events impact this trade"
    )
    political_catalysts: str = Field(
        description="Specific recent or expected US political/regulatory catalysts affecting this sector or stock"
    )
    earnings_context: str = Field(
        description="Next earnings date, days until earnings, last EPS surprise %, and key takeaways from recent earnings calls"
    )
    company_news_context: str = Field(
        description="Recent company-related news, press releases, or corporate developments"
    )
    sector_7d_analysis: str = Field(
        description="Analysis of sector ETF performance over the previous 7 days and the stock's relative strength/weakness against its sector"
    )
    suggested_option_type: str = Field(description="CALL or PUT matching the thesis direction")
    confidence: str = Field(description="LOW | MEDIUM | HIGH")
    avoid_earnings: bool = Field(
        description="True if the agent recommends avoiding this ticker due to unpriced earnings risk"
    )


class ScannerOutput(BaseModel):
    proposals: list[ScannerProposal] = Field(
        description="Ranked list of 3-6 option trade candidates covering ALL available sector categories (at least 1 per category)",
        min_length=1,
        max_length=6,
    )
    expected_political_catalysts: list[str] = Field(
        description="Top 3-5 upcoming or recent US political & policy catalysts impacting the market this month",
        default_factory=list,
    )
    us_political_macro_summary: str = Field(
        description="Analysis of current US political landscape, Fed interest rate expectations, and national economic policy impacts on this week's market"
    )
    market_commentary: str = Field(
        description="Brief overall market context and why these sector names stand out this week"
    )

    @field_validator("proposals", mode="before")
    @classmethod
    def _coerce_proposals(cls, v: Any) -> Any:
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

_SYSTEM_PROMPT = """You are a senior quantitative options strategist and geopolitical/macro market researcher. You are AGENT 1 (Opportunity Scanner & Sector Supervisor) in a multi-agent trading pipeline.

You will receive research dossiers from 3 SPECIALIZED SECTOR SUBAGENTS:
1. AI, SEMICONDUCTORS, MEMORY & DATA CENTER HARDWARE (Benchmark: SMH)
2. WAR & GEOPOLITICAL NATURAL GAS & ENERGY (Benchmark: XLE)
3. DATA CENTER POWER: SMALL MODULAR NUCLEAR (SMRs) & GRID UTILITIES (Benchmark: XLU)

Plus:
- Real-time US political news and macro policy catalysts
- Trailing 7-day sector benchmark ETF trends and trading ranges
- Earnings intelligence (next dates, days away, surprises, earnings call headlines)
- Technical screening metrics (SMAs, RSI, ATR, volume spikes, IV rank)

YOUR OBJECTIVES:
1. MANDATORY CATEGORY COVERAGE (AT LEAST 1 STOCK PER SECTOR):
   - You MUST include at least 1 candidate (ideally 1-2 for redundancy) from EACH of the 3 specialized sector categories whenever available in the dossier:
     * AI, Chips, Memory, or Data Center Infrastructure (AI_CHIPS_INFRA: e.g. NVDA, AMD, TSM, MU, SMCI, VRT, PLTR)
     * Natural Gas & Energy War Hedge (WAR_ENERGY_GAS: e.g. EQT, LNG, OXY, XOM, CVX, COP)
     * Data Center Nuclear Power & SMRs (DC_POWER_NUCLEAR: e.g. OKLO, CEG, VST, SMR, TLN, NNE, GEV)
   - Never drop an entire sector category because of high share price; Agent 2 will use budget-compliant CALL_DEBIT_SPREADs (<= $500) on expensive names like MU, NVDA, AMD, TSM, or AVGO.

2. SAME-DAY ORDER ACTIONABILITY:
   - Every selected trade MUST be suitable for placing an order TODAY on the same day this pipeline runs.
   - Strongly prefer candidates whose next earnings release is AFTER expiry (so entry does not need to be delayed).

3. SYNTHESIZE 7-DAY NEWS & UPCOMING CATALYSTS:
   - For each candidate, explain how the past 7 days of news and upcoming expected announcements support the setup.
   - Contrast the stock's 7-day performance against its sector benchmark (Relative Strength).

4. EARNINGS & POLITICAL RISK CHECK:
   - Note if a trade's holding window crosses an earnings date.
   - Populate `political_catalysts`, `earnings_context`, and `company_news_context` with rich detail.

5. CALL OPTIONS ONLY:
   - Suggest ONLY setups for CALL options (suggested_option_type='CALL', direction='BULLISH'). The user does not own underlying shares and trades buyer call positions and call debit spreads only. No puts.

Respond ONLY with valid JSON matching the ScannerOutput schema."""


def _build_human_prompt(
    candidates: list[CandidateTicker],
    news_brief: dict[str, Any],
) -> str:
    mkt_time = get_market_timing_context()
    lines = [
        "## ⏰ LIVE MARKET DATE, TIME & EXECUTION SESSION",
        f"- Current Date & Time: {mkt_time['day_of_week']}, {mkt_time['today_date']} at {mkt_time['time_et']}",
        f"- Market Session Status: {mkt_time['market_session']}",
        f"- Guidance: {mkt_time['execution_guidance']}\n",
        "## 🏛️ LIVE US POLITICAL, ECONOMIC & POLICY NEWS DOSSIER\n",
    ]
    macro_items = news_brief.get("macro_political_news", [])
    if macro_items:
        for item in macro_items:
            lines.append(f"- [{item.get('source', 'News')}] {item.get('title', '')}")
    else:
        lines.append("- (No breaking macro headlines retrieved)")

    pol_catalysts = news_brief.get("political_catalysts", [])
    if pol_catalysts:
        lines.append("\n## ⚡ EXPECTED OR RECENT US POLITICAL & MACRO CATALYSTS\n")
        for pc in pol_catalysts:
            lines.append(f"• {pc}")

    # Format Sector Subagent Dossiers
    dossiers = news_brief.get("sector_dossiers", [])
    if dossiers:
        lines.append("\n## 🌐 SECTOR SUBAGENT RESEARCH DOSSIERS (7-Day News & Upcoming Announcements)\n")
        for d in dossiers:
            chg_str = f"{d.sector_7d_change_pct:+.2f}%" if d.sector_7d_change_pct is not None else "N/A"
            lines.append(f"### Sector: {d.sector_name} (Benchmark: {d.benchmark_etf} 7d: {chg_str} | Range: {d.sector_7d_range or 'N/A'})")
            lines.append(f"- Commentary: {d.sector_commentary}")
            lines.append("- Past 7-Day News Headlines:")
            for nh in d.news_7d_headlines[:3]:
                lines.append(f"  • {nh}")
            lines.append("- Upcoming Expected News / Announcements:")
            for ua in d.upcoming_announcements[:2]:
                lines.append(f"  • {ua}")
            lines.append("")

    lines.append("\n## 📊 CANDIDATE TICKERS: TECHNICALS, 7-DAY SECTOR DATA, EARNINGS & COMPANY CONTEXT\n")
    company_news_map = news_brief.get("company_news", {})
    company_ctx_map = news_brief.get("company_context", {})

    for c in candidates:
        t = c.technicals
        ctx = company_ctx_map.get(c.symbol, {})
        next_ed = ctx.get("next_earnings_date")
        days_to_ed = ctx.get("days_to_earnings")
        last_eps = ctx.get("last_reported_eps")
        surp = ctx.get("last_eps_surprise_pct")
        cons_eps = ctx.get("consensus_eps_estimate")

        lines.append(
            f"### {c.symbol} | Score: {c.score} | ${c.price:.2f} (7d Return: {f'{t.change_7d_pct:+.2f}%' if t.change_7d_pct is not None else 'N/A'})\n"
            f"- 🌐 Sector: {t.sector_name or 'Broad Market'} ({t.sector_etf or 'SPY'})\n"
            f"  • Sector 7-Day Performance: {f'{t.sector_7d_change_pct:+.2f}%' if t.sector_7d_change_pct is not None else 'N/A'} (Range: {t.sector_7d_range or 'N/A'})\n"
            f"  • Stock vs Sector (Relative Strength): {f'{t.relative_strength_7d_pct:+.2f}%' if t.relative_strength_7d_pct is not None else 'N/A'}\n"
            f"- 📅 Earnings Intelligence:\n"
            f"  • Next Earnings: {next_ed or 'Unknown'} ({f'in {days_to_ed} days' if days_to_ed is not None else 'N/A'}) | Consensus EPS: {f'${cons_eps:.2f}' if cons_eps else 'N/A'}\n"
            f"  • Last Quarter Reported EPS: {f'${last_eps:.2f}' if last_eps is not None else 'N/A'} (Surprise: {f'{surp:+.2f}%' if surp is not None else 'N/A'})\n"
            f"- SMA20: {f'{t.sma_20:.2f}' if t.sma_20 else 'N/A'} | SMA50: {f'{t.sma_50:.2f}' if t.sma_50 else 'N/A'}\n"
            f"- RSI-14: {f'{t.rsi_14:.1f}' if t.rsi_14 else 'N/A'} | ATR-14: {f'{t.atr_14:.2f}' if t.atr_14 else 'N/A'}\n"
            f"- Volume Ratio: {f'{t.volume_ratio:.2f}' if t.volume_ratio else 'N/A'}x avg | IV Rank: {f'{t.iv_rank:.0f}' if t.iv_rank else 'N/A'}\n"
            f"- Technical Rationale: {c.rationale}\n"
        )

        call_hl = ctx.get("earnings_call_highlights", [])
        if call_hl:
            lines.append("  📢 Earnings Call & Guidance Headlines:")
            for ch in call_hl:
                lines.append(f"  • {ch}")

        pol_news = ctx.get("political_regulatory_news", [])
        if pol_news:
            lines.append("  🏛️ Regulatory / Political Developments:")
            for pn in pol_news:
                lines.append(f"  • {pn}")

        c_news = company_news_map.get(c.symbol, [])
        if c_news:
            lines.append("  📰 Recent Company News:")
            for cn in c_news:
                lines.append(f"  • [{cn.get('publisher', 'Media')}] {cn.get('title')}")
        lines.append("")

    lines.append(
        "\nSynthesize technical setups with 7-day sector trends, US political catalysts, and earnings intelligence. "
        "Select at least 1 candidate from EACH available sector category (3-6 proposals total) that can be traded TODAY (same-day), "
        "and respond with JSON matching ScannerOutput schema."
    )
    return "\n".join(lines)


def _build_fallback_proposal(c: CandidateTicker) -> ScannerProposal:
    """Create a ScannerProposal directly from screened technical & sector data."""
    return ScannerProposal(
        symbol=c.symbol,
        direction="BULLISH",
        thesis=c.rationale,
        catalyst="Technical momentum and sector alignment",
        news_catalyst="Sector momentum headline confirmed",
        political_macro_impact="Neutral policy sensitivity",
        political_catalysts="Sector fiscal and energy/tech regulatory backdrop",
        earnings_context=f"Next earnings: {c.technicals.earnings_date or 'N/A'}",
        company_news_context="Operations steady within primary sector focus",
        sector_7d_analysis=f"Sector {c.technicals.sector_name or 'Market'} ({c.technicals.sector_etf or 'SPY'}) moved {c.technicals.sector_7d_change_pct or 0.0:+.1f}% over the last 7 days",
        suggested_option_type="CALL",
        confidence="MEDIUM",
        avoid_earnings=c.technicals.has_upcoming_earnings,
    )


# ── Agent Node ─────────────────────────────────────────────────────────────────

def scanner_node(state: AgentTradingState) -> AgentTradingState:
    """
    LangGraph node: runs specialized Sector Subagents (AI/Chips, Energy/Gas, Data Center Nuclear)
    plus live US political & company news research.
    Updates state with scanner_proposals and macro_political_context.
    """
    settings = get_settings()
    custom_symbols = state.get("target_symbols")

    if custom_symbols:
        logger.info("🔍 Scanner Agent: screening user-specified universe: {}", custom_symbols)
        candidates = get_weekly_candidates(custom_symbols, top_n=min(len(custom_symbols), 12))
        sector_dossiers = []
    else:
        logger.info("🔍 Scanner Agent: running 3 Sector Subagents (AI/Chips, Energy/Gas, Nuclear Power)")
        candidates, sector_dossiers = run_all_sector_subagents(picks_per_sector=3)

    if not candidates:
        logger.warning("Scanner: no candidates survived technical screening")
        return {**state, "scanner_proposals": [], "workflow_error": "No candidates found"}

    # Step 2: Live US political & company news research
    candidate_symbols = [c.symbol for c in candidates]
    news_brief = build_market_news_brief(candidate_symbols)
    news_brief["sector_dossiers"] = sector_dossiers
    logger.info(
        "Scanner: collected {} US macro/political headlines and news for {} candidates across {} sectors",
        len(news_brief.get("macro_political_news", [])),
        len(candidate_symbols),
        len(sector_dossiers),
    )

    # Step 3: LLM reasoning with multi-model fallback cascade
    from src.agents.llm_resilience import invoke_structured_with_fallback

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_prompt(candidates, news_brief)),
    ]

    try:
        output: ScannerOutput = invoke_structured_with_fallback(
            messages=messages,
            schema=ScannerOutput,
            temperature=0.2,
            request_timeout=35,
        )
        logger.info(
            "Scanner LLM selected {} proposals: {}",
            len(output.proposals),
            [p.symbol for p in output.proposals],
        )
        logger.info("US Political/Macro Context: {}", output.us_political_macro_summary)
        logger.info("Market commentary: {}", output.market_commentary)
    except Exception as exc:
        logger.error("Scanner LLM call failed: {}. Falling back to top sector subagent candidates.", exc)
        fallback_proposals = [_build_fallback_proposal(c) for c in candidates[:settings.max_candidates]]
        output = ScannerOutput(
            proposals=fallback_proposals,
            expected_political_catalysts=news_brief.get("political_catalysts", []),
            us_political_macro_summary="Fallback: macro feed unavailable, monitoring general market backdrop.",
            market_commentary="Fallback: LLM unavailable, using sector subagent scores.",
        )

    # Ensure every sector category represented in candidates has at least 1 proposal
    selected_syms = {p.symbol.upper() for p in output.proposals}
    represented_categories = {get_symbol_sector_info(p.symbol)[0] for p in output.proposals}
    available_by_cat: dict[str, list[CandidateTicker]] = {}
    for c in candidates:
        sec_id, _ = get_symbol_sector_info(c.symbol)
        available_by_cat.setdefault(sec_id, []).append(c)

    for sec_id, cat_candidates in available_by_cat.items():
        if sec_id not in represented_categories:
            # Prefer a candidate without imminent earnings risk; fall back to highest score
            cat_sorted = sorted(
                cat_candidates,
                key=lambda x: (not x.technicals.has_upcoming_earnings, x.score),
                reverse=True,
            )
            chosen = next((c for c in cat_sorted if c.symbol.upper() not in selected_syms), None)
            if chosen:
                output.proposals.append(_build_fallback_proposal(chosen))
                selected_syms.add(chosen.symbol.upper())
                represented_categories.add(sec_id)
                logger.info(
                    "➕ Added {} to guarantee category '{}' representation",
                    chosen.symbol,
                    sec_id,
                )

    # Merge LLM proposals with technical and news data
    tech_map = {c.symbol: c for c in candidates}
    company_ctx_map = news_brief.get("company_context", {})
    proposals: list[dict[str, Any]] = []

    for prop in output.proposals:
        tech = tech_map.get(prop.symbol)
        ctx = company_ctx_map.get(prop.symbol, {})
        sec_id, sec_name = get_symbol_sector_info(prop.symbol)
        proposals.append(
            {
                **prop.model_dump(),
                "sector_category": sec_id,
                "sector_name": (tech.technicals.sector_name if tech and tech.technicals.sector_name else sec_name),
                "underlying_price": tech.price if tech else 0.0,
                "technicals": tech.technicals.model_dump() if tech else {},
                "market_commentary": output.market_commentary,
                "us_political_macro_summary": output.us_political_macro_summary,
                "earnings_raw": ctx,
            }
        )

    return {
        **state,
        "scanner_proposals": proposals,
        "macro_political_context": {
            "us_political_macro_summary": output.us_political_macro_summary,
            "macro_headlines": news_brief.get("macro_political_news", []),
            "political_catalysts": output.expected_political_catalysts or news_brief.get("political_catalysts", []),
            "sector_dossiers": [d.model_dump() for d in sector_dossiers],
        },
    }
