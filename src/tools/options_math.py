"""
Options Math – Black-Scholes Greeks, breakeven, PnL simulation, and position sizing.

All calculations are deterministic and unit-tested in isolation.

Public API
----------
- black_scholes_greeks(S, K, T, r, sigma, option_type)  → GreeksResult
- breakeven(strike, premium, option_type)                → float
- pnl_at_expiry(contracts, premium_paid, spot_range)     → PnLCurve
- size_position(portfolio_value, max_risk_pct, premium)  → PositionSize
- score_contract(contract, underlying_price, settings)   → ScoredContract
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import norm  # type: ignore[import]

from config.settings import get_settings


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GreeksResult:
    delta: float
    gamma: float
    theta: float      # per-day
    vega: float       # per 1% IV move
    rho: float
    theoretical_price: float
    implied_volatility: float


@dataclass
class PnLCurve:
    spot_prices: list[float]
    pnl_values: list[float]           # $ per contract (100 multiplier applied)
    pnl_pct_values: list[float]       # % of premium paid
    breakeven_price: float
    max_profit: float
    max_loss: float
    target_exit_spot: float | None = None


@dataclass
class PositionSize:
    max_contracts: int
    max_cost: float                   # total cost of position ($)
    cost_per_contract: float
    portfolio_risk_pct: float


@dataclass
class ScoredContract:
    contract_symbol: str
    option_type: str
    strike: float
    expiration: str
    dte: int
    mid: float
    greeks: GreeksResult
    breakeven: float
    breakeven_pct_move: float         # % underlying must move to break even
    score: float
    rationale: str


# ── Black-Scholes ──────────────────────────────────────────────────────────────

_RISK_FREE_RATE = 0.045   # approximate current US risk-free rate


def _round_float(val: object, ndigits: int) -> float:
    """Convert numeric scalars (including NumPy float64) to rounded native Python float."""
    return float(round(float(val), ndigits))  # type: ignore[arg-type]


def black_scholes_greeks(
    S: float,            # underlying spot price
    K: float,            # strike price
    T: float,            # time to expiry in years
    r: float,            # risk-free rate (annualised)
    sigma: float,        # implied volatility (annualised, e.g. 0.30 for 30%)
    option_type: Literal["CALL", "PUT"] = "CALL",
) -> GreeksResult:
    """
    Compute full Black-Scholes option price and all first-order Greeks.

    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError(f"Invalid inputs: S={S}, K={K}, T={T}, sigma={sigma}")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_neg_d1 = norm.cdf(-d1)
    n_neg_d2 = norm.cdf(-d2)
    phi_d1 = norm.pdf(d1)   # standard normal PDF at d1

    disc = math.exp(-r * T)

    if option_type == "CALL":
        price = S * nd1 - K * disc * nd2
        delta = nd1
    else:  # PUT
        price = K * disc * n_neg_d2 - S * n_neg_d1
        delta = nd1 - 1.0

    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega = S * phi_d1 * sqrt_T / 100          # per 1% IV move
    theta_call = (-(S * phi_d1 * sigma) / (2 * sqrt_T) - r * K * disc * nd2) / 365
    theta_put = (-(S * phi_d1 * sigma) / (2 * sqrt_T) + r * K * disc * n_neg_d2) / 365
    theta = theta_call if option_type == "CALL" else theta_put
    rho_call = K * T * disc * nd2 / 100
    rho_put = -K * T * disc * n_neg_d2 / 100
    rho = rho_call if option_type == "CALL" else rho_put

    return GreeksResult(
        delta=_round_float(delta, 4),
        gamma=_round_float(gamma, 6),
        theta=_round_float(theta, 4),
        vega=_round_float(vega, 4),
        rho=_round_float(rho, 4),
        theoretical_price=_round_float(price, 4),
        implied_volatility=_round_float(sigma * 100, 2),
    )


def enrich_greeks(
    contract_mid: float,
    underlying_price: float,
    strike: float,
    dte: int,
    iv_override: float | None = None,
    option_type: Literal["CALL", "PUT"] = "CALL",
    r: float = _RISK_FREE_RATE,
) -> GreeksResult | None:
    """
    Given a market mid-price and option parameters, back-solve for IV
    (or use iv_override) and compute Greeks.
    Returns None on failure.
    """
    T = max(dte / 365.0, 1 / 365.0)
    sigma = iv_override if iv_override else _implied_vol(contract_mid, underlying_price, strike, T, r, option_type)
    if sigma is None or sigma <= 0.01:
        return None
    try:
        return black_scholes_greeks(underlying_price, strike, T, r, sigma, option_type)
    except Exception:
        return None


def _implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float | None:
    """Newton-Raphson IV solver."""
    sigma = 0.30  # initial guess
    for _ in range(max_iter):
        try:
            g = black_scholes_greeks(S, K, T, r, sigma, option_type)  # type: ignore[arg-type]
            price = g.theoretical_price
            vega = g.vega * 100  # undo the /100 we applied
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            if abs(vega) < 1e-10:
                break
            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 20.0))
        except Exception:
            break
    return sigma if 0.001 < sigma < 20.0 else None


# ── Breakeven ──────────────────────────────────────────────────────────────────

def breakeven(strike: float, premium: float, option_type: Literal["CALL", "PUT"]) -> float:
    """
    Breakeven_call = K + Premium
    Breakeven_put  = K - Premium
    """
    if option_type == "CALL":
        return _round_float(strike + premium, 4)
    return _round_float(strike - premium, 4)


# ── PnL Simulation ─────────────────────────────────────────────────────────────

def pnl_at_expiry(
    option_type: Literal["CALL", "PUT"],
    strike: float,
    premium_paid: float,     # per share (e.g. $2.15 mid)
    spot_range: tuple[float, float] | None = None,
    underlying_price: float | None = None,
    n_points: int = 50,
) -> PnLCurve:
    """
    Compute PnL at expiry for a long single-leg option over a spot price range.
    PnL is in $ per contract (×100 multiplier).
    """
    if spot_range is None and underlying_price is not None:
        low = underlying_price * 0.70
        high = underlying_price * 1.30
    elif spot_range is not None:
        low, high = spot_range
    else:
        raise ValueError("Provide spot_range or underlying_price")

    spots = np.linspace(low, high, n_points).tolist()
    pnl_vals: list[float] = []
    pnl_pct_vals: list[float] = []
    cost = premium_paid * 100

    for s in spots:
        if option_type == "CALL":
            intrinsic = max(s - strike, 0.0)
        else:
            intrinsic = max(strike - s, 0.0)
        pnl_per_share = intrinsic - premium_paid
        pnl_dollar = _round_float(pnl_per_share * 100, 2)
        pnl_pct = _round_float((pnl_per_share / premium_paid) * 100, 1) if premium_paid else 0.0
        pnl_vals.append(pnl_dollar)
        pnl_pct_vals.append(pnl_pct)

    be = breakeven(strike, premium_paid, option_type)
    max_profit = max(pnl_vals)
    max_loss = min(pnl_vals)

    # Spot price that achieves target gain
    settings = get_settings()
    target_pnl_pct = settings.target_gain_percent
    target_exit_spot: float | None = None
    for s, p in zip(spots, pnl_pct_vals):
        if p >= target_pnl_pct:
            target_exit_spot = _round_float(s, 2)
            break

    return PnLCurve(
        spot_prices=[_round_float(s, 4) for s in spots],
        pnl_values=pnl_vals,
        pnl_pct_values=pnl_pct_vals,
        breakeven_price=be,
        max_profit=_round_float(max_profit, 2),
        max_loss=_round_float(max_loss, 2),
        target_exit_spot=target_exit_spot,
    )


# ── Position Sizing ────────────────────────────────────────────────────────────

def size_position(
    portfolio_value: float,
    max_risk_pct: float,
    premium_per_share: float,
    fixed_max_cost: float | None = None,
) -> PositionSize:
    """
    Determine maximum number of option contracts respecting either a fixed dollar cap
    or a portfolio percentage risk budget. Each contract controls 100 shares.
    """
    if fixed_max_cost is not None and fixed_max_cost > 0:
        max_cost_allowed = fixed_max_cost
    else:
        max_cost_allowed = portfolio_value * (max_risk_pct / 100.0)

    cost_per_contract = premium_per_share * 100
    if cost_per_contract <= 0:
        return PositionSize(
            max_contracts=0, max_cost=0.0, cost_per_contract=0.0, portfolio_risk_pct=0.0
        )
    contracts = int(max_cost_allowed / cost_per_contract)
    actual_cost = contracts * cost_per_contract
    actual_pct = _round_float((actual_cost / portfolio_value) * 100, 2) if portfolio_value else 0.0
    return PositionSize(
        max_contracts=contracts,
        max_cost=_round_float(actual_cost, 2),
        cost_per_contract=_round_float(cost_per_contract, 2),
        portfolio_risk_pct=actual_pct,
    )


# ── Contract Scorer ────────────────────────────────────────────────────────────

def score_contract(
    contract_mid: float,
    strike: float,
    dte: int,
    underlying_price: float,
    option_type: Literal["CALL", "PUT"],
    contract_symbol: str = "",
    expiration_str: str = "",
    iv_override: float | None = None,
) -> ScoredContract | None:
    """
    Enrich a raw option quote with Greeks, breakeven, and a selection score.
    Score rewards delta in [0.35, 0.55] and penalizes extreme OTM/ITM.
    Returns None if Greeks calculation fails.
    """
    if contract_mid <= 0:
        return None

    greeks = enrich_greeks(
        contract_mid=contract_mid,
        underlying_price=underlying_price,
        strike=strike,
        dte=dte,
        iv_override=iv_override,
        option_type=option_type,
    )
    if greeks is None:
        return None

    be = breakeven(strike, contract_mid, option_type)  # type: ignore[arg-type]
    be_pct = abs((be - underlying_price) / underlying_price * 100)

    # Score: prefer delta 0.35-0.55, penalise excessive breakeven % needed
    delta_abs = abs(greeks.delta)
    delta_score = 2.0 - abs(delta_abs - 0.45) * 10   # peaks at delta=0.45
    be_penalty = max(0, (be_pct - 5.0) * 0.2)        # penalise >5% move needed
    score = _round_float(delta_score - be_penalty, 3)

    reasons: list[str] = []
    if 0.35 <= delta_abs <= 0.55:
        reasons.append(f"ideal delta {greeks.delta:.2f}")
    if greeks.theta < -0.05:
        reasons.append(f"theta decay {greeks.theta:.3f}/day")
    reasons.append(f"breakeven {be_pct:.1f}% move required")

    return ScoredContract(
        contract_symbol=contract_symbol,
        option_type=option_type,
        strike=float(strike),
        expiration=expiration_str,
        dte=int(dte),
        mid=float(contract_mid),
        greeks=greeks,
        breakeven=be,
        breakeven_pct_move=_round_float(be_pct, 2),
        score=score,
        rationale="; ".join(reasons),
    )
