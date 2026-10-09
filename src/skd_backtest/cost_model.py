"""Historical, direction-aware execution cost quotes."""

from bisect import bisect_right
from datetime import date as iso_date
from math import isfinite
from numbers import Real

from .config import CostConfig, FeeScheduleEntry
from .contracts import CostQuote, CostRequest, Topic
from .runtime_cache import CacheView


class CostModel:
    """Quote execution values and explicit fees without changing account state."""

    def __init__(self, config: CostConfig):
        if not isinstance(config, CostConfig):
            raise TypeError("config must be CostConfig")

        self._commission_rate = self._nonnegative_rate(
            config.commission_rate, "commission_rate"
        )
        self._minimum_commission = self._nonnegative_rate(
            config.minimum_commission, "minimum_commission"
        )
        self._slippage = self._nonnegative_rate(config.slippage, "slippage")
        if self._slippage >= 1:
            raise ValueError("slippage must be less than 1")

        try:
            entries = tuple(config.fee_schedule)
        except TypeError as exc:
            raise ValueError("fee_schedule must be an iterable of FeeScheduleEntry") from exc

        schedule = []
        previous_date = None
        for entry in entries:
            if not isinstance(entry, FeeScheduleEntry):
                raise TypeError("fee_schedule entries must be FeeScheduleEntry")
            effective_date = self._iso_date(entry.effective_date, "effective_date")
            if previous_date is not None and effective_date <= previous_date:
                raise ValueError("fee_schedule dates must be strictly increasing")
            stamp_tax_rate = self._nonnegative_rate(
                entry.stamp_tax_rate, "stamp_tax_rate"
            )
            transfer_fee_rate = self._nonnegative_rate(
                entry.transfer_fee_rate, "transfer_fee_rate"
            )
            schedule.append((effective_date, stamp_tax_rate, transfer_fee_rate))
            previous_date = effective_date

        self._schedule = tuple(schedule)
        self._schedule_dates = tuple(item[0] for item in self._schedule)

    def calculate(self, *, date: str, request_id: int, cache: CacheView) -> None:
        key = (date, request_id)
        request = cache.read(Topic.COST_REQUEST, key)
        if not isinstance(request, CostRequest):
            raise TypeError("cost.request must contain a CostRequest")
        self._validate_request(request, date, request_id)

        stamp_tax_rate, transfer_fee_rate = self._rates_for(date)
        if request.price_mode == "raw_price":
            base_price = self._positive_number(request.base_price, "base_price")
            shares = self._positive_shares(request.shares)
            direction = 1.0 if request.side == "BUY" else -1.0
            execution_price = base_price * (1.0 + direction * self._slippage)
            position_value = base_price * shares
            trade_value = execution_price * shares
        else:
            position_value = self._positive_number(
                request.position_value, "position_value"
            )
            direction = 1.0 if request.side == "BUY" else -1.0
            execution_price = None
            trade_value = position_value * (1.0 + direction * self._slippage)

        commission = max(
            self._commission_rate * trade_value,
            self._minimum_commission,
        )
        stamp_tax = (
            stamp_tax_rate * trade_value if request.side == "SELL" else 0.0
        )
        other_cost = transfer_fee_rate * trade_value
        total_cost = commission + stamp_tax + other_cost
        cash_delta = (
            -(trade_value + total_cost)
            if request.side == "BUY"
            else trade_value - total_cost
        )

        values = (
            execution_price,
            position_value,
            trade_value,
            commission,
            stamp_tax,
            other_cost,
            total_cost,
            cash_delta,
        )
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("cost quote values must be finite")

        quote = CostQuote(
            request.request_id,
            request.order_id,
            date,
            request.side,
            request.price_mode,
            execution_price,
            position_value,
            trade_value,
            commission,
            stamp_tax,
            other_cost,
            total_cost,
            cash_delta,
        )
        cache.publish(Topic.COST_RESULT, key, quote)
        cache.log(
            level="DEBUG",
            message="quoted execution costs",
            details={"request_id": request_id, "total_cost": total_cost},
        )

    def _rates_for(self, trading_date: str) -> tuple[float, float]:
        # An empty schedule means the configured default of zero stamp and transfer fees.

        if not self._schedule:
            return 0.0, 0.0
        index = bisect_right(self._schedule_dates, trading_date) - 1
        if index < 0:
            raise ValueError(
                f"no fee schedule entry applies on {trading_date}"
            )
        _, stamp_tax_rate, transfer_fee_rate = self._schedule[index]
        return stamp_tax_rate, transfer_fee_rate

    @classmethod
    def _validate_request(
        cls, request: CostRequest, date: str, request_id: int
    ) -> None:
        if request.request_id != request_id or request.date != date:
            raise ValueError("cost request does not match its cache key")
        cls._iso_date(date, "date")
        if request.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if request.price_mode == "raw_price":
            if request.base_price is None or request.shares is None:
                raise ValueError("raw_price requests require base_price and shares")

            if request.position_value is not None:
                raise ValueError("raw_price requests must omit position_value")
        elif request.price_mode == "adjusted_return":
            if request.position_value is None:
                raise ValueError("adjusted_return requests require position_value")
            if request.base_price is not None or request.shares is not None:
                raise ValueError(
                    "adjusted_return requests must omit base_price and shares"
                )
        else:
            raise ValueError("price_mode must be raw_price or adjusted_return")

    @staticmethod
    def _iso_date(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must use YYYY-MM-DD")
        try:
            parsed = iso_date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must use YYYY-MM-DD") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"{name} must use YYYY-MM-DD")
        return value

    @staticmethod
    def _nonnegative_rate(value: Real, name: str) -> float:
        if (
            (type(value) not in (float, int) and (not isinstance(value, Real) or isinstance(value, bool)))
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
        return float(value)

    @classmethod
    def _positive_number(cls, value: Real | None, name: str) -> float:
        if value is None:
            raise ValueError(f"{name} is required")
        number = cls._nonnegative_rate(value, name)
        if number <= 0:
            raise ValueError(f"{name} must be positive")
        return number

    @staticmethod
    def _positive_shares(value: int | None) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError("shares must be a positive integer")
        return value
