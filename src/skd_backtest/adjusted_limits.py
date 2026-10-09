"""Temporary raw-open bridge for mandatory adjusted-return price limits."""

from decimal import Decimal, ROUND_HALF_UP
from math import isfinite
from pathlib import Path

import pandas as pd


_CENT = Decimal("0.01")


def _positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) and value > 0 else None


def _limit_rate(date, code, is_st):
    if pd.isna(is_st) or is_st not in (0, 1):
        raise ValueError(f"invalid is_st for adjusted price limits: {date} {code}")
    if code.startswith("SH688") or (code.startswith("SZ30") and date >= "2020-08-24"):
        return Decimal("0.20")
    if code.startswith(("SH60", "SZ00", "SZ30")):
        return Decimal("0.05") if is_st else Decimal("0.10")
    raise ValueError(f"unsupported security for adjusted price limits: {date} {code}")


def _price_limits(adjusted_open, raw_open, previous_close, rate):
    # Restore the reference quote to cents before percentage arithmetic. Never
    # round the adjustment ratio or apply the raw tick size to adjusted prices.
    factor = Decimal(str(adjusted_open)) / Decimal(str(raw_open))
    reference = (Decimal(str(previous_close)) / factor).quantize(_CENT, rounding=ROUND_HALF_UP)
    if reference <= 0:
        raise ValueError("estimated raw reference price must be positive")
    upper = (reference * (1 + rate)).quantize(_CENT, rounding=ROUND_HALF_UP)
    lower = (reference * (1 - rate)).quantize(_CENT, rounding=ROUND_HALF_UP)
    upper = max(upper, reference + _CENT)
    lower = max(_CENT, min(lower, reference - _CENT))
    values = float(upper * factor), float(lower * factor)
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("estimated adjusted price limits must be positive and finite")
    return values


class AdjustedLimitProvider:
    """Read only raw open and daily ST flags; cache one month outside research."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.close()

    def close(self):
        self._month = None
        self._table = None
        self._inputs = {}
        self.files_read = 0

    def apply(self, date: str, market: pd.DataFrame) -> pd.DataFrame:
        month = date[:7]
        if month != self._month:
            year, number = month.split("-")
            frames = []
            for dataset, columns in (
                ("MarketDataRawOpen", ["日期", "代码", "open"]),
                ("Factor33_winsor", ["日期", "代码", "is_st"]),
            ):
                path = self.data_dir / dataset / year / number / f"{year}{number}.parquet"
                table = pd.read_parquet(path, columns=columns)
                if table.duplicated(["日期", "代码"]).any():
                    raise ValueError(f"duplicate date/code for adjusted price limits in {path}")
                frames.append(table)
            self._table = frames[0].merge(frames[1], on=["日期", "代码"], how="outer",
                                           validate="one_to_one").set_index(["日期", "代码"])
            self._inputs = {}
            table = self._table.reset_index()
            for day, code, raw, flag in zip(*(table[name].to_numpy() for name in ("日期", "代码", "open", "is_st"))):
                self._inputs.setdefault(day, {})[code] = (raw, flag)
            self._month = month
            self.files_read += 2
        day = int(date.replace("-", ""))
        inputs = self._inputs.get(day, {})
        limits = []
        columns = ("code", "adjusted_open", "is_missing", "is_suspended", "previous_close")
        for code, adjusted_open, is_missing, is_suspended, previous_close in zip(*(market[name].to_numpy() for name in columns)):
            raw, is_st = inputs.get(code, (float("nan"), float("nan")))
            opening = _positive(adjusted_open)
            if is_missing or pd.isna(is_suspended) or is_suspended or opening is None:
                limits.append((None, None))
                continue
            raw = _positive(raw)
            if raw is None:
                raise ValueError(f"MarketDataRawOpen requires positive finite open: {date} {code}")
            rate = _limit_rate(date, code, is_st)
            previous = _positive(previous_close)
            limits.append((None, None) if previous is None else _price_limits(opening, raw, previous, rate))
        result = market.copy()
        result["upper_limit"] = [upper for upper, _ in limits]
        result["lower_limit"] = [lower for _, lower in limits]
        return result
