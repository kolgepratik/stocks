"""
Market Data Provider
====================
Fetches historical price data, implied volatility, IV rank, and option chains
from yfinance (primary) with optional Alpaca fallback.

Public API
----------
- get_weekly_candidates(universe)   → ranked list of CandidateTicker
- get_option_chain(symbol, dte)     → OptionChainResult
- get_iv_rank(symbol)               → float (0-100)
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Data Models ────────────────────────────────────────────────────────────────

class TechnicalSnapshot(BaseModel):
    symbol: str
    price: float
    change_7d_pct: float | None = None         # 7-day price return (%)
    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    volume: float | None = None
    avg_volume_20: float | None = None
    volume_ratio: float | None = None          # current vol / avg vol
    iv_current: float | None = None
    iv_rank: float | None = None               # 0-100 scale
    has_upcoming_earnings: bool = False
    earnings_date: str | None = None
    
    # ── 7-Day Sector Metrics ──────────────────────────────────────────────────
    sector_name: str | None = None
    sector_etf: str | None = None
    sector_7d_change_pct: float | None = None  # 7-day sector ETF return (%)
    sector_7d_range: str | None = None         # e.g. "$445.59 - $535.32"
    relative_strength_7d_pct: float | None = None # Stock 7d return minus Sector 7d return


class OptionContract(BaseModel):
    contract_symbol: str
    option_type: str                           # "CALL" | "PUT"
    strike: float
    expiration: date
    dte: int
    bid: float
    ask: float
    mid: float
    last_price: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


class OptionChainResult(BaseModel):
    symbol: str
    underlying_price: float
    expiration: date
    dte: int
    calls: list[OptionContract] = Field(default_factory=list)
    puts: list[OptionContract] = Field(default_factory=list)


class CandidateTicker(BaseModel):
    symbol: str
    price: float
    score: float                               # composite rank score
    rationale: str
    technicals: TechnicalSnapshot


# ── Sector ETF Mapping ─────────────────────────────────────────────────────────

SECTOR_ETF_MAP: dict[str, tuple[str, str]] = {
    # ── AI, Semiconductors, Memory & Data Center Infrastructure ────────────────
    "NVDA": ("SMH", "Semiconductors & AI"),
    "AMD": ("SMH", "Semiconductors & AI"),
    "AVGO": ("SMH", "Semiconductors & AI"),
    "TSM": ("SMH", "Foundry & Chips"),
    "MU": ("SMH", "Memory & Storage"),
    "SMCI": ("SMH", "AI Data Center Servers"),
    "VRT": ("XLI", "Data Center Cooling & Power"),
    "ANET": ("XLK", "Data Center Networking"),
    "QCOM": ("SMH", "Semiconductors & Wireless"),
    "INTC": ("SMH", "Semiconductors"),
    "PLTR": ("IGV", "AI Software & Defense"),
    "MSFT": ("XLK", "Hyperscale Cloud & AI"),
    "GOOGL": ("XLC", "Hyperscale Cloud & AI"),
    "META": ("XLC", "Hyperscale Cloud & AI"),

    # ── War & Geopolitical Energy & Gas ────────────────────────────────────────
    "LNG": ("XLE", "Liquefied Natural Gas Export"),
    "EQT": ("XLE", "Natural Gas Production"),
    "XOM": ("XLE", "Integrated Oil & Gas"),
    "CVX": ("XLE", "Integrated Oil & Gas"),
    "OXY": ("XLE", "Oil & Gas Exploration"),
    "COP": ("XLE", "Oil & Gas Exploration"),

    # ── Data Center Power: Small Nuclear (SMRs) & Grid Utilities ──────────────
    "CEG": ("XLU", "Nuclear Power & Hyperscaler PPA"),
    "VST": ("XLU", "Power Generation & Data Centers"),
    "OKLO": ("NLR", "Small Modular Nuclear SMR"),
    "SMR": ("NLR", "Small Modular Nuclear SMR"),
    "TLN": ("XLU", "Nuclear Data Center Campus"),
    "NNE": ("NLR", "Micro Nuclear SMR"),
    "GEV": ("XLI", "Electrification & Turbines"),
    "FSLR": ("TAN", "Clean Energy & Solar"),

    # ── Other Benchmarks & Consumer Tech ──────────────────────────────────────
    "AAPL": ("XLK", "Technology"),
    "TSLA": ("XLY", "Consumer Discretionary & Auto"),
    "AMZN": ("XLY", "Consumer Discretionary & Cloud"),
    "SPY": ("SPY", "Broad Market Benchmark"),
    "QQQ": ("QQQ", "Nasdaq 100 Tech Benchmark"),
}

_sector_cache: dict[str, dict[str, Any]] = {}


def get_sector_7d_performance(sector_etf: str) -> dict[str, Any]:
    """
    Fetch market data for the previous 7 calendar days (5 trading bars) for a sector ETF.
    Returns return %, high-low range, and trend.
    """
    if sector_etf in _sector_cache:
        return _sector_cache[sector_etf]

    try:
        t = yf.Ticker(sector_etf)
        hist = t.history(period="1mo", auto_adjust=True)
        if hist.empty or len(hist) < 6:
            return {"etf": sector_etf, "change_7d_pct": 0.0, "range_str": "N/A"}

        bars_7d = hist.tail(6)
        p_start = float(bars_7d["Close"].iloc[0])
        p_end = float(bars_7d["Close"].iloc[-1])
        ret_pct = round(((p_end - p_start) / p_start) * 100, 2)
        hi = round(float(bars_7d["High"].max()), 2)
        lo = round(float(bars_7d["Low"].min()), 2)
        
        result = {
            "etf": sector_etf,
            "change_7d_pct": ret_pct,
            "range_str": f"${lo:.2f} - ${hi:.2f}",
            "start_price": p_start,
            "current_price": p_end,
        }
        _sector_cache[sector_etf] = result
        return result
    except Exception as exc:
        logger.warning("Failed fetching 7-day sector data for {}: {}", sector_etf, exc)
        return {"etf": sector_etf, "change_7d_pct": 0.0, "range_str": "N/A"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float | None = None) -> float | None:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ── Core Functions ─────────────────────────────────────────────────────────────

def get_technical_snapshot(symbol: str, period: str = "1y") -> TechnicalSnapshot | None:
    """
    Download OHLCV data and compute SMA-20, SMA-50, RSI-14, ATR-14,
    volume ratio, and a rough IV estimate from recent close std-dev.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        if hist.empty or len(hist) < 55:
            logger.warning("{}: insufficient price history", symbol)
            return None

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]

        sma_20 = _safe_float(close.rolling(20).mean().iloc[-1])
        sma_50 = _safe_float(close.rolling(50).mean().iloc[-1])
        rsi_14 = _safe_float(_rsi(close).iloc[-1])
        atr_14 = _safe_float(_atr(high, low, close).iloc[-1])
        avg_vol_20 = _safe_float(volume.rolling(20).mean().iloc[-1])
        cur_vol = _safe_float(volume.iloc[-1])
        vol_ratio = (cur_vol / avg_vol_20) if (cur_vol and avg_vol_20 and avg_vol_20 > 0) else None
        price = _safe_float(close.iloc[-1]) or 0.0

        # Rough historical volatility as IV proxy (annualised)
        log_ret = np.log(close / close.shift(1)).dropna()
        hv_30 = _safe_float(log_ret.tail(30).std() * np.sqrt(252) * 100)

        # IV rank: position of current HV within 52-week range
        hv_series = log_ret.rolling(21).std() * np.sqrt(252) * 100
        hv_min = _safe_float(hv_series.min())
        hv_max = _safe_float(hv_series.max())
        iv_rank: float | None = None
        if hv_min is not None and hv_max is not None and hv_max > hv_min and hv_30 is not None:
            iv_rank = round((hv_30 - hv_min) / (hv_max - hv_min) * 100, 1)

        # Earnings check
        earnings_date_str: str | None = None
        has_earnings = False
        try:
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # cal may be a DataFrame or dict depending on yfinance version
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                    ed = cal["Earnings Date"].iloc[0]
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    ed = cal["Earnings Date"][0] if isinstance(cal["Earnings Date"], list) else cal["Earnings Date"]
                else:
                    ed = None
                if ed is not None:
                    if isinstance(ed, (datetime, pd.Timestamp)):
                        ed = ed.date()
                    if isinstance(ed, date):
                        days_away = (ed - date.today()).days
                        if 0 <= days_away <= 14:
                            has_earnings = True
                            earnings_date_str = str(ed)
        except Exception:
            pass

        # Calculate stock's 7-day price change
        stock_7d_pct: float | None = None
        if len(close) >= 6:
            p_start_7d = close.iloc[-6]
            if p_start_7d and p_start_7d > 0:
                stock_7d_pct = round(((price - p_start_7d) / p_start_7d) * 100, 2)

        # Retrieve 7-day Sector ETF market data
        sector_etf, sector_name = SECTOR_ETF_MAP.get(symbol, ("SPY", "Broad Market S&P 500"))
        sector_perf = get_sector_7d_performance(sector_etf)
        sector_7d_ret = sector_perf.get("change_7d_pct")
        sector_7d_range = sector_perf.get("range_str")
        rs_7d = (
            round(stock_7d_pct - sector_7d_ret, 2)
            if stock_7d_pct is not None and sector_7d_ret is not None
            else None
        )

        return TechnicalSnapshot(
            symbol=symbol,
            price=price,
            change_7d_pct=stock_7d_pct,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi_14=rsi_14,
            atr_14=atr_14,
            volume=cur_vol,
            avg_volume_20=avg_vol_20,
            volume_ratio=round(vol_ratio, 2) if vol_ratio else None,
            iv_current=hv_30,
            iv_rank=iv_rank,
            has_upcoming_earnings=has_earnings,
            earnings_date=earnings_date_str,
            sector_name=sector_name,
            sector_etf=sector_etf,
            sector_7d_change_pct=sector_7d_ret,
            sector_7d_range=sector_7d_range,
            relative_strength_7d_pct=rs_7d,
        )
    except Exception as exc:
        logger.error("get_technical_snapshot({}) failed: {}", symbol, exc)
        return None


def _score_candidate(snap: TechnicalSnapshot) -> tuple[float, str]:
    """
    Compute a composite momentum/volatility score (higher = more attractive).
    Returns (score, rationale_string).
    """
    score = 0.0
    reasons: list[str] = []

    price = snap.price
    if snap.sma_20 and snap.sma_50:
        if price > snap.sma_20 > snap.sma_50:
            score += 2.0
            reasons.append("bullish SMA stack (price > 20 > 50)")
        elif price < snap.sma_20 < snap.sma_50:
            score += 2.0
            reasons.append("bearish SMA stack (price < 20 < 50)")

    if snap.rsi_14:
        if 55 < snap.rsi_14 < 75:
            score += 1.5
            reasons.append(f"RSI {snap.rsi_14:.1f} (bullish momentum)")
        elif 25 < snap.rsi_14 < 45:
            score += 1.5
            reasons.append(f"RSI {snap.rsi_14:.1f} (oversold bounce potential)")

    if snap.volume_ratio and snap.volume_ratio > 1.5:
        score += 1.5
        reasons.append(f"volume spike {snap.volume_ratio:.1f}x avg")

    if snap.iv_rank:
        if 20 <= snap.iv_rank <= 60:
            score += 1.0
            reasons.append(f"IV rank {snap.iv_rank:.0f} (sweet spot for buying)")
        elif snap.iv_rank > 70:
            score -= 0.5
            reasons.append(f"IV rank {snap.iv_rank:.0f} (elevated — spread may be better)")

    # Slight penalty for near-term earnings (unless overridden)
    if snap.has_upcoming_earnings:
        score -= 1.0
        reasons.append(f"⚠️ earnings {snap.earnings_date} (unpriced risk)")

    if snap.sector_7d_change_pct is not None and snap.sector_etf:
        reasons.append(
            f"Sector {snap.sector_name} ({snap.sector_etf}): {snap.sector_7d_change_pct:+.1f}% 7d change"
        )

    rationale = "; ".join(reasons) if reasons else "baseline scan"
    return round(score, 2), rationale


def get_weekly_candidates(
    universe: list[str],
    top_n: int = 5,
) -> list[CandidateTicker]:
    """
    Screen a universe of tickers and return the top-N ranked candidates
    for further options analysis.
    """
    results: list[CandidateTicker] = []
    for symbol in universe:
        snap = get_technical_snapshot(symbol)
        if snap is None:
            continue
        score, rationale = _score_candidate(snap)
        results.append(
            CandidateTicker(
                symbol=symbol,
                price=snap.price,
                score=score,
                rationale=rationale,
                technicals=snap,
            )
        )
        logger.debug("{}: score={} | {}", symbol, score, rationale)

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_n]


def _target_expiry(dte: int) -> date:
    """Return the nearest option expiry date at or beyond `dte` calendar days."""
    target = date.today() + timedelta(days=dte)
    # Roll to Friday if needed (options expire on Fridays for weeklies)
    while target.weekday() != 4:  # 4 = Friday
        target += timedelta(days=1)
    return target


def get_contract_mark_price(contract: OptionContract) -> float:
    """
    Compute the best fair mark price for an option contract.
    Uses live (bid + ask) / 2 when active two-sided quotes exist;
    otherwise falls back cleanly to last_price (e.g., at market open or off-hours).
    """
    if contract.bid > 0 and contract.ask > 0:
        # Guard against obviously broken ultra-wide off-hours quotes when last_price is inside spread
        if contract.ask > contract.bid * 5 and contract.last_price and contract.bid <= contract.last_price <= contract.ask:
            return round(contract.last_price, 4)
        return round((contract.bid + contract.ask) / 2, 4)
    if contract.last_price and contract.last_price > 0:
        return round(contract.last_price, 4)
    if contract.mid > 0:
        return round(contract.mid, 4)
    if contract.ask > 0:
        return round(contract.ask / 2, 4)
    return 0.0


def get_option_chain(
    symbol: str,
    target_dte: int = 7,
    exact_expiration: date | str | None = None,
) -> OptionChainResult | None:
    """
    Fetch the option chain for an exact expiration (if provided) or the nearest
    weekly expiry at or beyond target_dte. Returns calls and puts sorted by strike.
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options  # tuple of expiry strings "YYYY-MM-DD"
        if not expirations:
            logger.warning("{}: no option expirations available", symbol)
            return None

        chosen_str: str | None = None

        if exact_expiration is not None:
            exact_date = (
                datetime.strptime(exact_expiration, "%Y-%m-%d").date()
                if isinstance(exact_expiration, str)
                else exact_expiration
            )
            exact_str = exact_date.strftime("%Y-%m-%d")
            if exact_str in expirations:
                chosen_str = exact_str
            else:
                # Pick closest expiration date if exact string is unavailable
                chosen_str = min(
                    expirations,
                    key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - exact_date).days),
                )
        else:
            target = _target_expiry(target_dte)
            # Pick the nearest expiry >= target
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= target:
                    chosen_str = exp_str
                    break

            if not chosen_str:
                chosen_str = expirations[-1]

        exp_date = datetime.strptime(chosen_str, "%Y-%m-%d").date()
        dte_actual = (exp_date - date.today()).days

        # Underlying price
        info = ticker.fast_info
        underlying_price = _safe_float(getattr(info, "last_price", None)) or _safe_float(
            getattr(info, "regular_market_price", None)
        ) or 0.0

        chain = ticker.option_chain(chosen_str)

        def _parse_df(df: pd.DataFrame, opt_type: str) -> list[OptionContract]:
            contracts: list[OptionContract] = []
            for _, row in df.iterrows():
                try:
                    bid = _safe_float(row.get("bid"), 0.0) or 0.0
                    ask = _safe_float(row.get("ask"), 0.0) or 0.0
                    last_px = _safe_float(row.get("lastPrice"))
                    if bid > 0 and ask > 0:
                        mid = round((bid + ask) / 2, 4)
                    elif last_px and last_px > 0:
                        mid = round(last_px, 4)
                    else:
                        mid = round((bid + ask) / 2, 4)
                    contracts.append(
                        OptionContract(
                            contract_symbol=str(row.get("contractSymbol", "")),
                            option_type=opt_type,
                            strike=_safe_float(row.get("strike"), 0.0) or 0.0,
                            expiration=exp_date,
                            dte=dte_actual,
                            bid=bid,
                            ask=ask,
                            mid=mid,
                            last_price=last_px,
                            volume=int(row["volume"]) if pd.notna(row.get("volume")) else None,
                            open_interest=int(row["openInterest"]) if pd.notna(row.get("openInterest")) else None,
                            implied_volatility=_safe_float(row.get("impliedVolatility")),
                        )
                    )
                except Exception:
                    continue
            return contracts

        return OptionChainResult(
            symbol=symbol,
            underlying_price=underlying_price,
            expiration=exp_date,
            dte=dte_actual,
            calls=_parse_df(chain.calls, "CALL"),
            puts=_parse_df(chain.puts, "PUT"),
        )
    except Exception as exc:
        logger.error("get_option_chain({}) failed: {}", symbol, exc)
        return None


def get_iv_rank(symbol: str) -> float | None:
    """Return the IV rank (0-100) for a symbol using 1-year HV data."""
    snap = get_technical_snapshot(symbol)
    return snap.iv_rank if snap else None


def get_market_timing_context() -> dict[str, Any]:
    """
    Return the exact current US Eastern date, weekday, time, and market session status
    so agents generate accurate SAME-DAY execution instructions rather than stale weekdays.
    """
    from zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    weekday_idx = now_et.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    day_name = now_et.strftime("%A")
    date_str = now_et.strftime("%Y-%m-%d")
    time_str = now_et.strftime("%H:%M ET")

    if weekday_idx >= 5:
        session = "WEEKEND_CLOSED"
        guidance = (
            f"Today is {day_name}, {date_str} ({time_str}). US equity/options markets are closed for the weekend; "
            "structure limit orders ready for the next market open (Monday 9:30 AM ET)."
        )
    else:
        mins = now_et.hour * 60 + now_et.minute
        if 570 <= mins < 960:  # 9:30 AM - 4:00 PM ET
            session = "REGULAR_HOURS (MARKET OPEN)"
            guidance = (
                f"Today is {day_name}, {date_str} at {time_str}. US markets are OPEN RIGHT NOW. "
                f"ALL recommendations MUST be executable TODAY ({day_name}, {date_str}) via Good-for-Day (DAY) limit orders at live market prices. "
                "Do NOT say 'wait until Monday' or delay entry until tomorrow."
            )
        elif mins < 570:
            session = "PRE_MARKET"
            guidance = (
                f"Today is {day_name}, {date_str} at {time_str} (Pre-Market). "
                f"ALL recommendations MUST be executable TODAY ({day_name}, {date_str}) using a DAY limit order queued for today's 9:30 AM ET regular session open."
            )
        else:
            session = "AFTER_HOURS"
            guidance = (
                f"Today is {day_name}, {date_str} at {time_str} (After-Hours). "
                f"Structure immediate DAY limit orders ready for next open using today's closing option marks ({date_str})."
            )

    return {
        "today_date": date_str,
        "day_of_week": day_name,
        "time_et": time_str,
        "market_session": session,
        "execution_guidance": guidance,
    }
