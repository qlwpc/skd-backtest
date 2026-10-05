"""Bounded monthly batches, one background reader, and isolated as-of copies."""

import logging

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .adjusted_limits import AdjustedLimitProvider
from .config import BacktestConfig
from .schemas import SOURCE_COLUMNS


logger = logging.getLogger(__name__)


def raw_price_columns(path: Path, fields) -> tuple[list[str], bool]:
    """Choose observed raw prices or adjustment factors from this file's schema."""
    columns = set(pq.read_schema(path).names)
    raw = [f"raw_{name}" for name in fields]
    if set(raw).issubset(columns):
        return raw, False
    adjusted = [*fields, "adjustment_factor"]
    if set(adjusted).issubset(columns):
        return adjusted, True
    raise ValueError(f"raw_price requires columns {raw} or {adjusted} in {path}")


@dataclass
class DailyData:
    """Engine-only daily inputs; only research is passed to the user model."""

    date: str
    open_market: pd.DataFrame
    close_market: pd.DataFrame
    research: dict[str, pd.DataFrame] | None
    portfolio: dict[str, pd.DataFrame] | None
    execution_date: str | None


def _price_history(market, dates, previous, close_column="close"):
    """Align observed/previously seen securities and carry only past valid prices."""
    codes = sorted(set(market["代码"]) | set(previous.index))
    index = pd.MultiIndex.from_product([dates, codes], names=["日期", "代码"])
    result = market.assign(is_missing=False).set_index(["日期", "代码"]).reindex(index)
    result["is_missing"] = result["is_missing"].isna()
    valid = result[close_column].where(
        result[close_column].between(0, float("inf"), inclusive="neither")
        & result["is_suspend"].eq(False).fillna(False))
    code_index = index.get_level_values("代码")
    days = pd.Series(index.get_level_values("日期"), index=index).where(valid.notna())
    for values, name, previous_name in (
        (valid, "reference_close", "previous_close"),
        (days, "reference_date", "previous_close_date"),
    ):
        seed = pd.Series(code_index.map(previous[name]), index=index)
        result[name] = values.groupby(level="代码").ffill().fillna(seed)
        result[previous_name] = result[name].groupby(level="代码").shift().fillna(seed)
    result["has_valid_close"] = valid.notna()
    result["is_stale"] = result["reference_date"].notna() & result["reference_date"].ne(
        index.get_level_values("日期"))
    for name in ("reference_date", "previous_close_date"):
        result[name] = result[name].astype("Int64")
    # Future batch constituents do not appear before their first source observation.
    seen = (~result["is_missing"]).groupby(level="代码").cummax() | code_index.isin(previous.index)
    result = result.loc[seen]
    state = (result.groupby(level="代码").tail(1).droplevel("日期")[["reference_close", "reference_date"]]
             if not result.empty else previous)
    return result.reset_index(), state


class DataProvider:
    def __init__(self, config: BacktestConfig, *, load_research: bool = True, load_portfolio: bool = False):
        self.config = config
        self.load_research = load_research
        self._adjusted_limits = (AdjustedLimitProvider(config.data_dir)
                                 if config.price_mode == "adjusted_return" else None)
        self._datasets = dict(SOURCE_COLUMNS) if load_research else {
            "MarketData": ("日期", "代码", "open", "close", "is_suspend"),
        }
        if load_portfolio and not load_research:
            self._datasets["Barra_factor"] = SOURCE_COLUMNS["Barra_factor"]
        market_columns = list(self._datasets["MarketData"])
        if config.price_mode == "raw_price":
            market_columns.extend(("raw_open", "raw_high", "raw_low", "raw_close"))
            market_columns.extend(("upper_limit", "lower_limit"))
        self._datasets["MarketData"] = tuple(dict.fromkeys(market_columns))
        self._market_open_column = "raw_open" if config.price_mode == "raw_price" else "open"
        self._market_close_column = "raw_close" if config.price_mode == "raw_price" else "close"
        self._market_open_label = "raw_open" if config.price_mode == "raw_price" else "adjusted_open"
        self._market_close_label = "raw_close" if config.price_mode == "raw_price" else "adjusted_close"
        self._executor = None
        self._pending = None
        self._tables = {}
        self._price_state = None
        self.stats = {}
        self._prepared_dates = None
        self._playing = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        """Join the reader even on inference/read errors; release cached frames."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None
        self._pending = None
        self._tables = {}
        self._price_state = None
        if self._adjusted_limits is not None:
            self._adjusted_limits.close()
        self._prepared_dates = None
        self._playing = False

    def monthly_path(self, dataset: str, year: int, month: int) -> Path:
        return self.config.data_dir / dataset / str(year) / f"{month:02d}" / f"{year}{month:02d}.parquet"

    def _read_parquet(self, path: Path, columns, **kwargs):
        raw_fields = [name.removeprefix("raw_") for name in columns if name.startswith("raw_")]
        use_factor = False
        if raw_fields:
            source, use_factor = raw_price_columns(path, raw_fields)
            columns = list(dict.fromkeys([name for name in columns if not name.startswith("raw_")] + source))
        try:
            table = pd.read_parquet(path, columns=list(columns), **kwargs)
            if use_factor:
                factor = self._adjustment_factor(table, path)
                for name in raw_fields:
                    table[f"raw_{name}"] = pd.to_numeric(table[name], errors="coerce") / factor
            return table
        except (pa.ArrowInvalid, KeyError) as exc:
            requested = ", ".join(columns)
            raise ValueError(f"cannot read required columns ({requested}) from {path}: {exc}") from exc

    @staticmethod
    def _adjustment_factor(table: pd.DataFrame, path: Path) -> pd.Series:
        factor = pd.to_numeric(table["adjustment_factor"], errors="coerce")
        if not (factor.gt(0) & factor.lt(float("inf"))).all():
            raise ValueError(f"adjustment_factor must be positive and finite in {path}")
        return factor

    def _restore_raw_prices(self, table: pd.DataFrame, path: Path) -> pd.DataFrame:
        if self.config.price_mode == "raw_price" and {"upper_limit", "lower_limit"}.issubset(table):
            upper = pd.to_numeric(table["upper_limit"], errors="coerce")
            lower = pd.to_numeric(table["lower_limit"], errors="coerce")
            valid = (upper.gt(0) & upper.lt(float("inf")) & lower.gt(0)
                     & lower.lt(float("inf")) & lower.lt(upper))
            if not valid.all():
                raise ValueError(f"upper_limit/lower_limit must be finite, positive, and ordered in {path}")
            table["upper_limit"] = upper
            table["lower_limit"] = lower
        return table

    def _read_seed_prices(self, path: Path) -> pd.DataFrame:
        close_column = self._market_close_column
        columns = ["日期", "代码", close_column, "is_suspend"]
        seed = self._read_parquet(
            path, columns, filters=[("日期", "<", self._first_date)],
        )
        seed = self._restore_raw_prices(seed, path)
        if self.config.price_mode == "raw_price":
            close_column = "raw_close"
        seed = seed.loc[seed["is_suspend"].eq(False).fillna(False)]
        seed = seed.rename(columns={close_column: "reference_close", "日期": "reference_date"})
        return seed[["代码", "reference_close", "reference_date"]]

    def prepare(self) -> list[str]:
        """Read the real calendar and enough available history for lookback."""
        if self._playing:
            raise RuntimeError("cannot prepare during active playback")
        self.close()
        self.stats = {
            "calendar_files": 0, "calendar_seconds": 0.0,
            "data_files": 0, "data_bytes": 0, "batches": 0,
            "read_seconds": 0.0, "wait_seconds": 0.0, "as_of_seconds": 0.0,
            "source_rows": {name: 0 for name in SOURCE_COLUMNS},
            "delivered_rows": {name: 0 for name in SOURCE_COLUMNS},
            "warmup_days": 0,
            "seed_files": 0, "seed_rows": 0, "limit_files": 0,
            "open_rows": 0, "close_rows": 0, "market_seconds": 0.0,
        }
        self._batch_index = -1
        self._current_date = None
        self._batches = []
        started = perf_counter()
        start = int(self.config.start_date.replace("-", ""))
        end = int(self.config.end_date.replace("-", ""))
        months = list(pd.period_range(self.config.start_date, self.config.end_date, freq="M"))
        month_first_dates = {}

        def read_dates(month):
            path = self.monthly_path("MarketData", month.year, month.month)
            values = self._read_parquet(
                path, ["日期"], filters=[("日期", "<=", end)],
            )["日期"].drop_duplicates()
            self.stats["calendar_files"] += 1
            month_first_dates[month] = values.min() if not values.empty else None
            return values.tolist()

        dates = set()
        for month in months:
            dates.update(read_dates(month))
        trading = sorted(day for day in dates if start <= day <= end)
        if trading:
            # Stop at the dataset's beginning, but do not silently skip holes in history.
            available = sorted((self.config.data_dir / "MarketData").glob("*/*/*.parquet"))
            earliest = pd.Period(available[0].stem, freq="M")
            previous = months[0] - 1
            while sum(day < trading[0] for day in dates) < self.config.lookback - 1 and previous >= earliest:
                dates.update(read_dates(previous))
                previous -= 1
            history = sorted(day for day in dates if day < trading[0])
            history = history[-(self.config.lookback - 1):] if self.config.lookback > 1 else []
            self.stats["warmup_days"] = len(history)
            self._calendar = history + trading
            self._positions = {day: index for index, day in enumerate(self._calendar)}
            self._playback_dates = set(trading)
            self._first_date, self._last_date = self._calendar[0], trading[-1]
            first_month = pd.Period(str(self._first_date), freq="M")
            self._seed_paths = [self.monthly_path("MarketData", month.year, month.month)
                                for month in pd.period_range(earliest, first_month, freq="M")
                                if month < first_month or month_first_dates[first_month] < self._first_date]
            months = list(pd.period_range(str(self._first_date), str(self._last_date), freq="M"))
            size = self.config.read_batch_months
            self._batches = [months[index:index + size] for index in range(0, len(months), size)]
            if self.config.prefetch:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skd-data")
                self._pending = self._executor.submit(self._read_batch, self._batches[0])
        else:
            self._calendar, self._positions, self._playback_dates = [], {}, set()
        self.stats["calendar_seconds"] = perf_counter() - started
        trading_dates = pd.to_datetime(trading, format="%Y%m%d").strftime("%Y-%m-%d").tolist()
        logger.debug("[DataProvider.prepare] %s .. %s; %s trading days, %s warmup days",
                     self.config.start_date, self.config.end_date, len(trading_dates), self.stats["warmup_days"])
        self._prepared_dates = tuple(trading_dates)
        return trading_dates

    def _read_batch(self, months):
        """Only the reader builds these frames; the foreground owns them afterwards."""
        started = perf_counter()
        tables, rows = {}, {}
        byte_count = 0
        for name, columns in self._datasets.items():
            parts = []
            for month in months:
                path = self.monthly_path(name, month.year, month.month)
                table = self._read_parquet(
                    path, columns,
                    filters=[("日期", ">=", self._first_date), ("日期", "<=", self._last_date)],
                )
                if name == "MarketData":
                    table = self._restore_raw_prices(table, path)
                parts.append(table)
                byte_count += path.stat().st_size
            tables[name] = pd.concat(parts, ignore_index=True)
            rows[name] = len(tables[name])
        # Preserve the annual universe only for the engine (including exited holdings).
        market = tables["MarketData"]
        seed_rows = 0
        seed_files = 0
        if self._price_state is None:
            state = pd.DataFrame(columns=["reference_close", "reference_date"])
            state.index.name = "代码"
            for path in self._seed_paths:
                seed = self._read_seed_prices(path)
                seed_rows += len(seed)
                seed_files += 1
                known_codes = state.index.union(seed["代码"].drop_duplicates())
                seed = seed[seed["reference_close"].between(0, float("inf"), inclusive="neither")]
                latest = seed.sort_values("reference_date").drop_duplicates("代码", keep="last").set_index("代码")
                state = pd.concat([state, latest]).loc[lambda frame: ~frame.index.duplicated(keep="last")]
                state = state.reindex(known_codes)
            self._price_state = state
        first = int(months[0].start_time.strftime("%Y%m%d"))
        last = int(months[-1].end_time.strftime("%Y%m%d"))
        dates = [day for day in self._calendar if first <= day <= last]
        tables["_market"], self._price_state = _price_history(
            market, dates, self._price_state, self._market_close_column)
        if self.load_research and self.config.price_mode == "raw_price":
            tables["MarketData"] = tables["MarketData"].loc[:, list(SOURCE_COLUMNS["MarketData"])]
        if "Barra_factor" in tables:
            membership = tables["Barra_factor"][["日期", "代码"]]
            for name in ("Factor33_winsor", "MarketData"):
                if name not in tables:
                    continue
                tables[name] = tables[name].merge(membership, on=["日期", "代码"], how="inner", sort=False)
        tables = {
            name: table.sort_values(["日期", "代码"], ignore_index=True)
            for name, table in tables.items()
        }
        return tables, rows, byte_count, perf_counter() - started, seed_files, seed_rows

    def _ensure_loaded(self, date: str) -> int:
        day = int(date.replace("-", ""))
        if day not in self._playback_dates:
            raise ValueError(f"{date} is not a playback trading day")
        if self._current_date is not None and day < self._current_date:
            raise ValueError("DataProvider playback must move forward; call prepare() to restart")
        self._current_date = day
        month = pd.Period(date, freq="M")
        while self._batch_index < 0 or month > self._batches[self._batch_index][-1]:
            index = self._batch_index + 1
            started = perf_counter()
            result = (self._pending.result() if self._pending is not None
                      else self._read_batch(self._batches[index]))
            self.stats["wait_seconds"] += perf_counter() - started
            tables, rows, byte_count, elapsed, seed_files, seed_rows = result
            self._pending = None
            # One future at most; submit before joining history to overlap that work too.
            if self._executor is not None and index + 1 < len(self._batches):
                self._pending = self._executor.submit(self._read_batch, self._batches[index + 1])
            lower = self._calendar[max(0, self._positions[day] - self.config.lookback + 1)]
            for name, table in tables.items():
                if name in self._tables:
                    previous = self._tables[name]
                    tail = previous.iloc[previous["日期"].searchsorted(lower):]
                    table = pd.concat([tail, table], ignore_index=True)
                self._tables[name] = table
            self._batch_index = index
            self.stats["batches"] += 1
            self.stats["data_files"] += len(self._batches[index]) * len(self._datasets)
            self.stats["seed_files"] += seed_files
            self.stats["seed_rows"] += seed_rows
            self.stats["data_bytes"] += byte_count
            self.stats["read_seconds"] += elapsed
            for name, count in rows.items():
                self.stats["source_rows"][name] += count
        return day

    @staticmethod
    def _slice(table, first, last, columns=None):
        begin = table["日期"].searchsorted(first)
        end = table["日期"].searchsorted(last, side="right")
        result = table.iloc[begin:end]
        if columns is not None:
            result = result.loc[:, columns]
        # Never expose a view of prefetched future rows or mutable provider state.
        return result.copy(deep=True).reset_index(drop=True)

    def as_of(self, as_of_date: str) -> dict[str, pd.DataFrame]:
        day = self._ensure_loaded(as_of_date)
        started = perf_counter()
        first = self._calendar[max(0, self._positions[day] - self.config.lookback + 1)]
        data = {name: self._slice(self._tables[name], first, day) for name in SOURCE_COLUMNS}
        for name, table in data.items():
            self.stats["delivered_rows"][name] += len(table)
        self.stats["as_of_seconds"] += perf_counter() - started
        logger.debug("[DataProvider.as_of] %s; history starts at %s", as_of_date, first)
        return data

    def portfolio_inputs(self, as_of_date: str) -> dict[str, pd.DataFrame]:
        day = self._ensure_loaded(as_of_date)
        return {
            "barra_exposures": self._slice(self._tables["Barra_factor"], day, day),
        }

    def open_market(self, date: str) -> pd.DataFrame:
        day = self._ensure_loaded(date)
        started = perf_counter()
        logger.debug("[DataProvider.open_market] %s; %s prices", date, self._market_open_label)
        columns = [
            "代码", self._market_open_column, "is_suspend", "is_missing",
            "previous_close", "previous_close_date",
        ]
        if self.config.price_mode == "raw_price":
            columns.extend(("upper_limit", "lower_limit"))
        result = self._slice(self._tables["_market"], day, day, columns).rename(
            columns={"代码": "code", self._market_open_column: self._market_open_label,
                     "is_suspend": "is_suspended"},
        ).assign(date=date)
        if self._adjusted_limits is not None:
            result = self._adjusted_limits.apply(date, result)
            self.stats["limit_files"] = self._adjusted_limits.files_read
        self.stats["open_rows"] += len(result)
        self.stats["market_seconds"] += perf_counter() - started
        return result

    def close_market(self, date: str) -> pd.DataFrame:
        day = self._ensure_loaded(date)
        started = perf_counter()
        logger.debug("[DataProvider.close_market] %s; %s prices", date, self._market_close_label)
        result = self._slice(self._tables["_market"], day, day, [
            "代码", self._market_close_column, "is_suspend", "is_missing", "has_valid_close",
            "previous_close", "previous_close_date", "reference_close", "reference_date", "is_stale",
        ]).rename(
            columns={"代码": "code", self._market_close_column: self._market_close_label,
                     "is_suspend": "is_suspended"},
        ).assign(date=date)
        self.stats["close_rows"] += len(result)
        self.stats["market_seconds"] += perf_counter() - started
        return result

    def playback(self) -> Iterator[DailyData]:
        """Read and deliver daily inputs without running any financial modules.

        Use contextlib.closing() when stopping early or when a consumer can fail.
        """
        if self._playing:
            raise RuntimeError("playback is already active")
        with self:
            dates = (list(self._prepared_dates)
                     if self._prepared_dates is not None and self._current_date is None
                     else self.prepare())
            self._playing = True
            for index, date in enumerate(dates):
                execution_date = dates[index + 1] if index + 1 < len(dates) else None
                signal = ("Barra_factor" in self._datasets and index % self.config.rebalance_interval == 0
                          and execution_date is not None)
                yield DailyData(
                    date=date, open_market=self.open_market(date), close_market=self.close_market(date),
                    research=self.as_of(date) if signal and self.load_research else None,
                    portfolio=self.portfolio_inputs(date) if signal else None,
                    execution_date=execution_date,
                )

    def valuation_inputs(self, codes: Sequence[str] | None = None) -> pd.DataFrame:
        """Load a whole interval once for independent batch accounting; no P&L calculation.

        Only MarketData is read. This uses a fresh reader and does not change playback state.
        Missing requested securities remain explicit rows with unknown prices.
        """
        columns = ["code", self._market_close_label, "is_suspended", "is_missing", "has_valid_close",
                   "previous_close", "previous_close_date", "reference_close", "reference_date", "is_stale", "date"]
        if codes is not None:
            codes = list(dict.fromkeys(codes))
        parts = []
        with DataProvider(replace(self.config, lookback=1), load_research=False) as reader:
            for date in reader.prepare():
                market = reader.close_market(date)
                if codes is not None:
                    market = market.set_index("code").reindex(codes).reset_index()
                    market["date"] = date
                    market["is_missing"] = market["is_missing"].fillna(True).astype(bool)
                    for name in ("has_valid_close", "is_stale"):
                        market[name] = market[name].fillna(False).astype(bool)
                parts.append(market)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
