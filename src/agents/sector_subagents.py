"""
Specialized Sector Research Subagents
====================================
Dedicated subagents for high-impact market sectors:
1. AI, Semiconductors, Memory & Data Center Hardware (SMH)
2. War & Geopolitical Energy & Natural Gas (XLE)
3. Data Center Power: Small Modular Nuclear (SMRs) & Grid Utilities (XLU)

Each subagent:
- Ingests news from the last 7 days for stocks in its specific sector category.
- Researches upcoming expected news, policy decisions, earnings, and announcements.
- Computes trailing 7-day sector ETF performance and stock Relative Strength.
- Vets technical setups and nominates high-conviction candidate stocks for options trading.
"""
from __future__ import annotations

from typing import Any, List
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.tools.market_data import (
    CandidateTicker,
    get_sector_7d_performance,
    get_weekly_candidates,
)
from src.tools.news_research import _fetch_rss_items, get_company_news


# ── Schemas ────────────────────────────────────────────────────────────────────

class SectorResearchDossier(BaseModel):
    sector_id: str
    sector_name: str
    benchmark_etf: str
    sector_7d_change_pct: float | None = None
    sector_7d_range: str | None = None
    news_7d_headlines: list[str] = Field(default_factory=list)
    upcoming_announcements: list[str] = Field(default_factory=list)
    sector_commentary: str = ""
    top_candidates: list[CandidateTicker] = Field(default_factory=list)


# ── Subagent Sector Definitions ────────────────────────────────────────────────

SECTOR_CONFIGS = {
    "AI_CHIPS_INFRA": {
        "name": "AI, Semiconductors, Memory & Data Center Infrastructure",
        "benchmark_etf": "SMH",
        "symbols": ["NVDA", "AMD", "TSM", "AVGO", "MU", "SMCI", "VRT", "ANET", "PLTR"],
        "news_queries": [
            "semiconductors AI chips memory HBM data center cooling NVDA MU VRT",
            "chip export restrictions TSMC Nvidia AMD Micron 2026",
            "data center liquid cooling networking Vertiv Arista Supermicro",
        ],
        "upcoming_queries": [
            "semiconductor earnings TSMC Nvidia Micron Blackwell roadmap 2026",
            "upcoming AI chip announcements GTC Computex tech conference 2026",
        ],
    },
    "WAR_ENERGY_GAS": {
        "name": "War & Geopolitical Hedge: Natural Gas & Energy",
        "benchmark_etf": "XLE",
        "symbols": ["LNG", "EQT", "XOM", "CVX", "OXY", "COP"],
        "news_queries": [
            "natural gas LNG export terminals war Ukraine Middle East Cheniere EQT",
            "crude oil natural gas geopolitical conflict supply disruption",
            "US energy policy strategic petroleum reserve export permits",
        ],
        "upcoming_queries": [
            "OPEC meeting upcoming natural gas storage EIA report 2026",
            "LNG terminal expansion approvals federal energy regulatory commission",
        ],
    },
    "DC_POWER_NUCLEAR": {
        "name": "Data Center Power: Small Nuclear (SMRs) & Grid Utilities",
        "benchmark_etf": "XLU",
        "symbols": ["CEG", "VST", "OKLO", "SMR", "TLN", "NNE", "GEV"],
        "news_queries": [
            "nuclear power AI data centers small modular reactors CEG VST OKLO SMR",
            "Constellation Vistra Talen Microsoft Amazon nuclear PPA power contract",
            "clean baseload electricity grid capacity bottleneck data centers",
        ],
        "upcoming_queries": [
            "NRC small modular reactor SMR license application approval 2026",
            "upcoming nuclear earnings Constellation Vistra Oklo NuScale 2026",
        ],
    },
}


# ── Sector Subagent Implementation ─────────────────────────────────────────────

class SectorSubagent:
    """Specialized researcher for an individual sector domain."""

    def __init__(self, sector_id: str):
        if sector_id not in SECTOR_CONFIGS:
            raise ValueError(f"Unknown sector_id: {sector_id}")
        self.sector_id = sector_id
        self.config = SECTOR_CONFIGS[sector_id]
        self.sector_name = self.config["name"]
        self.benchmark_etf = self.config["benchmark_etf"]
        self.symbols = self.config["symbols"]

    def run(self, top_n_picks: int = 2) -> SectorResearchDossier:
        """
        Execute subagent research:
        1. Fetch trailing 7-day benchmark ETF performance.
        2. Ingest past 7-day news for this sector.
        3. Ingest upcoming expected news, announcements, and policy milestones.
        4. Run quantitative technical screening across the sector's universe.
        5. Select the top N candidate stocks with sector-grounded rationales.
        """
        logger.info("🔬 [{}] Subagent researching {} (Benchmark: {})", self.sector_id, self.sector_name, self.benchmark_etf)

        # 1. 7-Day Sector ETF performance
        sector_7d = get_sector_7d_performance(self.benchmark_etf)
        chg_7d = sector_7d.get("change_7d_pct")
        range_7d = sector_7d.get("range_str")

        # 2. Ingest 7-day news for sector
        news_7d: list[str] = []
        for q in self.config["news_queries"]:
            items = _fetch_rss_items(q, limit=2)
            for it in items:
                title = it.get("title", "")
                if title and title not in news_7d:
                    news_7d.append(title)

        # 3. Ingest upcoming announcements & catalysts
        upcoming: list[str] = []
        for q in self.config["upcoming_queries"]:
            items = _fetch_rss_items(q, limit=2)
            for it in items:
                title = it.get("title", "")
                if title and title not in upcoming:
                    upcoming.append(title)

        # 4. Technical screening of sector universe
        candidates = get_weekly_candidates(self.symbols, top_n=min(len(self.symbols), 6))

        # 5. Enrich candidate rationales with sector 7-day and upcoming context
        top_candidates: list[CandidateTicker] = []
        for c in candidates[:top_n_picks]:
            t = c.technicals
            rs_str = f"{t.relative_strength_7d_pct:+.1f}%" if t.relative_strength_7d_pct is not None else "N/A"
            sector_ret_str = f"{chg_7d:+.1f}%" if chg_7d is not None else "N/A"

            # Create an enriched rationale tying technicals + sector trend
            enriched_rationale = (
                f"[{self.sector_name}] {c.rationale} | "
                f"Sector ({self.benchmark_etf}) 7d: {sector_ret_str}, Stock RS: {rs_str}. "
                f"Catalyst: {news_7d[0] if news_7d else 'Sector momentum alignment'}."
            )
            c.rationale = enriched_rationale
            top_candidates.append(c)

        commentary = (
            f"Sector benchmark {self.benchmark_etf} moved {f'{chg_7d:+.2f}%' if chg_7d is not None else '0.0%'} "
            f"over the past 7 days [Range: {range_7d or 'N/A'}]. "
            f"Key recent driver: {news_7d[0] if news_7d else 'General market positioning'}. "
            f"Upcoming catalyst: {upcoming[0] if upcoming else 'Earnings / policy calendar'}."
        )

        return SectorResearchDossier(
            sector_id=self.sector_id,
            sector_name=self.sector_name,
            benchmark_etf=self.benchmark_etf,
            sector_7d_change_pct=chg_7d,
            sector_7d_range=range_7d,
            news_7d_headlines=news_7d[:5],
            upcoming_announcements=upcoming[:4],
            sector_commentary=commentary,
            top_candidates=top_candidates,
        )


def get_symbol_sector_info(symbol: str) -> tuple[str, str]:
    """Return canonical (sector_id, sector_name) for a ticker based on SECTOR_CONFIGS."""
    sym = symbol.upper().strip()
    for sec_id, cfg in SECTOR_CONFIGS.items():
        if sym in cfg["symbols"]:
            return sec_id, cfg["name"]
    return "OTHER", "Broad Market"


def run_all_sector_subagents(
    target_sectors: list[str] | None = None,
    picks_per_sector: int = 3,
) -> tuple[list[CandidateTicker], list[SectorResearchDossier]]:
    """
    Run subagents across all three focused sectors:
    1. AI, Chips, Memory & Data Center Hardware
    2. Natural Gas & Energy (War Hedge)
    3. Small Nuclear & Renewables (Data Center Power)

    Returns:
    - combined list of top CandidateTicker objects across sectors
    - list of SectorResearchDossier objects with full 7-day and upcoming research
    """
    active_sector_ids = target_sectors or list(SECTOR_CONFIGS.keys())
    dossiers: list[SectorResearchDossier] = []
    combined_candidates: list[CandidateTicker] = []

    for sec_id in active_sector_ids:
        if sec_id in SECTOR_CONFIGS:
            subagent = SectorSubagent(sec_id)
            dossier = subagent.run(top_n_picks=picks_per_sector)
            dossiers.append(dossier)
            combined_candidates.extend(dossier.top_candidates)

    logger.info(
        "✅ Completed sector subagents: generated {} candidate tickers across {} sectors",
        len(combined_candidates),
        len(dossiers),
    )
    return combined_candidates, dossiers
