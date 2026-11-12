"""
News, Political Catalysts & Earnings Research Tool
=================================================
Fetches live company-specific news, earnings call dates & surprises,
company-level regulatory/political headlines, and broad US political,
Federal Reserve, and macroeconomic catalysts to provide full fundamental
and geopolitical context before option recommendations are structured.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Any

from loguru import logger
import yfinance as yf


def _fetch_rss_items(query: str, limit: int = 4) -> list[dict[str, str]]:
    """Helper to fetch and parse headlines from Google News RSS."""
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)

        items: list[dict[str, str]] = []
        for el in root.findall(".//item")[:limit]:
            title = el.find("title").text if el.find("title") is not None else ""
            source = el.find("source").text if el.find("source") is not None else "News"
            pub_date = el.find("pubDate").text if el.find("pubDate") is not None else ""
            if title:
                items.append({
                    "title": title.strip(),
                    "source": source.strip(),
                    "pub_date": pub_date.strip(),
                })
        return items
    except Exception as exc:
        logger.debug("Failed fetching RSS for '{}': {}", query, exc)
        return []


def get_macro_political_news(limit: int = 8) -> list[dict[str, str]]:
    """
    Fetch real-time US political, economic, Federal Reserve, and trade policy headlines.
    """
    topics = [
        "US politics stock market economy",
        "Federal Reserve interest rates inflation policy",
        "US trade tariffs regulation policy",
    ]
    all_news: list[dict[str, str]] = []
    seen: set[str] = set()

    for topic in topics:
        items = _fetch_rss_items(topic, limit=3)
        for it in items:
            norm = it["title"].lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                all_news.append({**it, "category": topic})

    return all_news[:limit]


def get_political_catalysts(limit: int = 6) -> list[str]:
    """
    Fetch expected or recent US political catalysts:
    - Upcoming elections, debt ceiling, budget battles
    - Congressional hearings on tech, energy, or market regulation
    - Federal Reserve FOMC decisions and speeches
    - Trade disputes, export restrictions, and tariff deadlines
    """
    queries = [
        "US political catalysts Congress White House economy 2026",
        "Federal Reserve FOMC interest rate decision catalyst",
        "US trade tariffs export control deadline policy",
    ]
    catalysts: list[str] = []
    seen: set[str] = set()

    for q in queries:
        items = _fetch_rss_items(q, limit=3)
        for it in items:
            title = it["title"]
            norm = title.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                catalysts.append(title)

    return catalysts[:limit]


def get_company_news(symbol: str, count: int = 4) -> list[dict[str, str]]:
    """
    Fetch top recent news articles for an individual ticker from yfinance.
    """
    try:
        t = yf.Ticker(symbol)
        raw_news = t.news or []
        articles: list[dict[str, str]] = []
        for item in raw_news[:count]:
            title = item.get("title")
            if not title and isinstance(item.get("content"), dict):
                title = item.get("content", {}).get("title")
            publisher = item.get("publisher") or (
                item.get("content", {}).get("provider", {}).get("displayName")
                if isinstance(item.get("content"), dict)
                else "Financial Media"
            )
            if title:
                articles.append({
                    "title": title.strip(),
                    "publisher": publisher.strip() if publisher else "Financial Media",
                })
        return articles
    except Exception as exc:
        logger.warning("Failed to fetch news for {}: {}", symbol, exc)
        return []


def get_company_earnings_and_catalysts(symbol: str) -> dict[str, Any]:
    """
    Fetch comprehensive earnings intelligence and company-specific political/regulatory news:
    - Next earnings date & days until earnings
    - Consensus EPS & revenue estimates
    - Last reported EPS and earnings surprise %
    - Recent earnings call & guidance commentary headlines
    - Company-specific regulatory, antitrust, or government contract headlines
    """
    t = yf.Ticker(symbol)
    result: dict[str, Any] = {
        "symbol": symbol,
        "next_earnings_date": None,
        "days_to_earnings": None,
        "consensus_eps_estimate": None,
        "last_reported_eps": None,
        "last_eps_surprise_pct": None,
        "earnings_call_highlights": [],
        "political_regulatory_news": [],
    }

    # 1. Earnings Calendar & Estimates
    try:
        cal = t.calendar
        if cal and "Earnings Date" in cal and cal["Earnings Date"]:
            ed = cal["Earnings Date"][0]
            if isinstance(ed, (date, datetime)):
                result["next_earnings_date"] = str(ed)
                target = ed if isinstance(ed, date) else ed.date()
                result["days_to_earnings"] = (target - date.today()).days
        if cal and "Earnings Average" in cal and cal["Earnings Average"] is not None:
            result["consensus_eps_estimate"] = float(cal["Earnings Average"])
    except Exception as exc:
        logger.debug("Failed reading calendar for {}: {}", symbol, exc)

    # 2. Historical Earnings Surprise
    try:
        ed_df = t.get_earnings_dates(limit=4)
        if ed_df is not None and not ed_df.empty:
            for _, row in ed_df.iterrows():
                rep = row.get("Reported EPS")
                surp = row.get("Surprise(%)")
                if rep is not None and str(rep) != "nan":
                    result["last_reported_eps"] = float(rep)
                    if surp is not None and str(surp) != "nan":
                        result["last_eps_surprise_pct"] = float(surp)
                    break
    except Exception as exc:
        logger.debug("Failed reading earnings dates for {}: {}", symbol, exc)

    # 3. Earnings call & guidance headlines
    call_items = _fetch_rss_items(f"{symbol} earnings call guidance results financial", limit=2)
    result["earnings_call_highlights"] = [it["title"] for it in call_items]

    # 4. Company-specific political & regulatory news
    pol_items = _fetch_rss_items(f"{symbol} regulation government policy antitrust export congress", limit=2)
    result["political_regulatory_news"] = [it["title"] for it in pol_items]

    return result


def build_market_news_brief(universe: list[str]) -> dict[str, Any]:
    """
    Generate a consolidated research dossier:
    - US political and macro economic news
    - Expected/recent political catalysts
    - Company-specific headlines
    - Earnings intelligence & company regulatory catalysts per ticker
    """
    logger.info("📰 Fetching live US political, macro, earnings, and company news feeds...")
    macro_news = get_macro_political_news(limit=8)
    political_catalysts = get_political_catalysts(limit=6)
    company_news: dict[str, list[dict[str, str]]] = {}
    company_context: dict[str, dict[str, Any]] = {}

    for sym in universe:
        company_news[sym] = get_company_news(sym, count=3)
        company_context[sym] = get_company_earnings_and_catalysts(sym)

    return {
        "macro_political_news": macro_news,
        "political_catalysts": political_catalysts,
        "company_news": company_news,
        "company_context": company_context,
    }
