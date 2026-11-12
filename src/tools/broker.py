"""
Broker Abstraction Layer
========================
Unified interface for submitting option orders across Alpaca and Robinhood.
Provides a common async execute_option_order() function that dispatches
to the correct broker implementation.

Brokers
-------
- Alpaca  : Full options API via `alpaca-py` (paper + live)
- Robinhood: Via `robin_stocks`, MFA TOTP authentication

All order submissions return a standardized OrderResult dict:
{
    "order_id": str,
    "status": str,       # "submitted" | "filled" | "rejected" | "error"
    "filled_price": float | None,
    "broker": str,
    "raw": dict,
}
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from loguru import logger

from config.settings import get_settings


# ── Type Alias ─────────────────────────────────────────────────────────────────

OrderResult = dict[str, Any]


# ── Alpaca Broker ──────────────────────────────────────────────────────────────

class AlpacaBroker:
    """
    Alpaca options order submission via alpaca-py.
    Paper trading by default (settings.alpaca_paper=True).
    """

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.alpaca_api_key
        self._secret = settings.alpaca_secret_key
        self._paper = settings.alpaca_paper
        self._client = None

    def _get_client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret,
                paper=self._paper,
            )
        return self._client

    def get_buying_power(self) -> float:
        """Return available buying power in USD."""
        try:
            client = self._get_client()
            account = client.get_account()
            return float(account.buying_power)
        except Exception as exc:
            logger.error("Alpaca get_buying_power failed: {}", exc)
            return 0.0

    async def submit_option_order(
        self,
        symbol: str,
        option_type: str,          # "CALL" | "PUT"
        strike: float,
        expiration: str,           # "YYYY-MM-DD"
        quantity: int,
        limit_price: float,
        contract_symbol: str = "",
    ) -> OrderResult:
        """
        Submit a limit buy order for an option contract.
        Uses the OCC contract symbol when available, else constructs it.
        """
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

            client = self._get_client()

            # Use provided contract symbol or construct OCC symbol
            occ_symbol = contract_symbol or _build_occ_symbol(
                symbol, expiration, option_type, strike
            )

            order_data = LimitOrderRequest(
                symbol=occ_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                type="limit",
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )

            # Run blocking SDK call in executor
            order = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.submit_order(order_data)
            )

            logger.info(
                "Alpaca order submitted: {} {} x{} @ ${} → id={}",
                occ_symbol, "BUY", quantity, limit_price, order.id,
            )
            return {
                "order_id": str(order.id),
                "status": str(order.status),
                "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "broker": "ALPACA",
                "raw": {"id": str(order.id), "status": str(order.status)},
            }
        except Exception as exc:
            logger.error("Alpaca submit_option_order failed: {}", exc)
            return {
                "order_id": "error",
                "status": "error",
                "filled_price": None,
                "broker": "ALPACA",
                "raw": {"error": str(exc)},
            }


# ── Robinhood Broker ───────────────────────────────────────────────────────────

class RobinhoodBroker:
    """
    Robinhood options order submission via robin_stocks.
    Authenticates with username/password + TOTP MFA on first use.
    """

    _logged_in: bool = False

    def _login(self) -> None:
        if self._logged_in:
            return
        settings = get_settings()
        try:
            import robin_stocks.robinhood as rh
            import pyotp

            totp = pyotp.TOTP(settings.robinhood_mfa_totp).now() if settings.robinhood_mfa_totp else None
            rh.login(
                username=settings.robinhood_username,
                password=settings.robinhood_password,
                mfa_code=totp,
            )
            RobinhoodBroker._logged_in = True
            logger.info("Robinhood: authenticated successfully")
        except Exception as exc:
            logger.error("Robinhood login failed: {}", exc)
            raise

    def get_buying_power(self) -> float:
        """Return Robinhood cash buying power."""
        try:
            self._login()
            import robin_stocks.robinhood as rh
            profile = rh.load_account_profile()
            return float(profile.get("buying_power", 0))
        except Exception as exc:
            logger.error("Robinhood get_buying_power failed: {}", exc)
            return 0.0

    async def submit_option_order(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        quantity: int,
        limit_price: float,
        contract_symbol: str = "",
    ) -> OrderResult:
        """
        Submit a limit buy order for an options contract on Robinhood.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._login)
            import robin_stocks.robinhood as rh

            opt_type_rh = "call" if option_type.upper() == "CALL" else "put"

            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: rh.order_buy_option_limit(
                    positionEffect="open",
                    creditOrDebit="debit",
                    price=limit_price,
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration,
                    strike=strike,
                    optionType=opt_type_rh,
                    timeInForce="gfd",
                ),
            )

            if order is None or "detail" in (order or {}):
                raise RuntimeError(f"Robinhood rejected order: {order}")

            order_id = order.get("id", "unknown")
            logger.info(
                "Robinhood order submitted: {} {} ${} exp={} x{} → id={}",
                symbol, opt_type_rh, strike, expiration, quantity, order_id,
            )
            return {
                "order_id": str(order_id),
                "status": order.get("state", "submitted"),
                "filled_price": None,
                "broker": "ROBINHOOD",
                "raw": order,
            }
        except Exception as exc:
            logger.error("Robinhood submit_option_order failed: {}", exc)
            return {
                "order_id": "error",
                "status": "error",
                "filled_price": None,
                "broker": "ROBINHOOD",
                "raw": {"error": str(exc)},
            }


# ── Paper / Simulation Broker ──────────────────────────────────────────────────

class PaperBroker:
    """
    Local paper trading simulation — no real orders placed.
    Useful for testing the full execution pipeline without credentials.
    """

    def get_buying_power(self) -> float:
        return 25_000.0

    async def submit_option_order(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        quantity: int,
        limit_price: float,
        contract_symbol: str = "",
    ) -> OrderResult:
        import uuid
        fake_id = str(uuid.uuid4())[:8].upper()
        total = limit_price * 100 * quantity
        logger.info(
            "[PAPER] BUY {} {} ${} exp={} x{} @ ${:.2f} = ${:.2f} total → id={}",
            symbol, option_type, strike, expiration, quantity, limit_price, total, fake_id,
        )
        return {
            "order_id": fake_id,
            "status": "filled",
            "filled_price": limit_price,
            "broker": "PAPER",
            "raw": {"paper": True, "total_cost": total},
        }


# ── Dispatch ───────────────────────────────────────────────────────────────────

_broker_instances: dict[str, Any] = {}


def _get_broker(broker: str):
    if broker not in _broker_instances:
        if broker == "ALPACA":
            _broker_instances[broker] = AlpacaBroker()
        elif broker == "ROBINHOOD":
            _broker_instances[broker] = RobinhoodBroker()
        else:
            _broker_instances[broker] = PaperBroker()
    return _broker_instances[broker]


async def execute_option_order(
    broker: Literal["ALPACA", "ROBINHOOD", "PAPER"],
    symbol: str,
    option_type: str,
    strike: float,
    expiration: str,
    quantity: int,
    limit_price: float,
    contract_symbol: str = "",
) -> OrderResult:
    """
    Unified entry point for option order submission.
    Dispatches to the requested broker implementation.
    """
    b = _get_broker(broker)
    return await b.submit_option_order(
        symbol=symbol,
        option_type=option_type,
        strike=strike,
        expiration=expiration,
        quantity=quantity,
        limit_price=limit_price,
        contract_symbol=contract_symbol,
    )


def get_buying_power(broker: Literal["ALPACA", "ROBINHOOD", "PAPER"]) -> float:
    """Return available buying power for the specified broker."""
    b = _get_broker(broker)
    return b.get_buying_power()


# ── Utilities ──────────────────────────────────────────────────────────────────

def _build_occ_symbol(symbol: str, expiration: str, option_type: str, strike: float) -> str:
    """
    Construct an OCC-format option symbol.
    Format: {symbol padded to 6}{YYMMDD}{C/P}{strike * 1000 zero-padded to 8}
    Example: AAPL  261121C00150000
    """
    sym = symbol.upper().ljust(6)
    date_str = expiration.replace("-", "")[2:]  # YYMMDD
    opt_char = "C" if option_type.upper() == "CALL" else "P"
    strike_str = str(int(strike * 1000)).zfill(8)
    return f"{sym}{date_str}{opt_char}{strike_str}"
