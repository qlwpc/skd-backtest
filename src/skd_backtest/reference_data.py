"""Point-in-time benchmark and portfolio reference data."""

from datetime import date as Date
from math import fsum, isfinite
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .config import BacktestConfig, REFERENCE_DATASETS
from .contracts import BenchmarkDay, Dataset, PortfolioInputs, Topic
from .runtime_cache import CacheView


_REQUIRED_COLUMNS = {
    "benchmark_returns": ("date", "benchmark_return"),
    "benchmark_weights": ("date", "code", "benchmark_weight"),
    "industries": ("date", "code", "industry"),
}
_KEY_COLUMNS = {
    "benchmark_returns": ("date",),
    "benchmark_weights": ("date", "code"),
    "industries": ("date", "code"),
}


class _DateSlices(Mapping):
    """Index a validated, sorted table without constructing every daily frame."""
    def __init__(self, frame):
        self.frame = frame
        values = frame["date"].to_numpy()
        self.bounds = {}
        self.materialized = {}
        if len(values):
            starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
            self.bounds = {values[start]: (start, stop) for start, stop in zip(starts[:-1], starts[1:])}

    def __getitem__(self, date):
        start, stop = self.bounds[date]
        if date not in self.materialized:
            self.materialized[date] = self.frame.iloc[start:stop]
        return self.materialized[date]

    def __iter__(self):
        return iter(self.bounds)

    def __len__(self):
        return len(self.bounds)


class ReferenceDataProvider:
    """Load each configured source once per run and expose exact-date slices."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._tables = {}
        self._slices = {}

    def _load(self, name: str) -> Mapping[str, pd.DataFrame] | None:
        if name in self._slices:
            return self._slices[name]

        path = self.config.data_dir / REFERENCE_DATASETS[name]
        if not path.exists():
            return None
        if not path.is_dir():
            raise ValueError(f"{name} source must be a directory: {path}")

        frame = _read_source(Path(path), name)

        self._validate(name, frame)
        if name == "benchmark_weights":
            totals = frame.groupby("date")["benchmark_weight"].transform("sum")
            frame["benchmark_weight"] = frame["benchmark_weight"] / totals
        frame = frame.sort_values(list(_KEY_COLUMNS[name]), kind="stable", ignore_index=True)
        slices = _DateSlices(frame)

        self._tables[name] = frame
        self._slices[name] = slices
        return slices

    @staticmethod
    def _validate(name: str, frame: pd.DataFrame) -> None:
        required = _REQUIRED_COLUMNS[name]
        if not frame.columns.is_unique or any(column not in frame.columns for column in required):
            raise ValueError(f"{name} source is missing required columns or has duplicate columns")

        dates = frame["date"]
        if dates.isna().any() or not all(
            isinstance(value, str) and _is_iso_date(value) for value in dates.drop_duplicates()
        ):
            raise ValueError(f"{name} dates must be YYYY-MM-DD strings")
        if frame.duplicated(list(_KEY_COLUMNS[name])).any():
            raise ValueError(f"{name} source contains duplicate keys")

        if "code" in required:
            codes = frame["code"]
            if codes.isna().any() or not all(
                isinstance(value, str) and value for value in codes.drop_duplicates()
            ):
                raise ValueError(f"{name} codes must be nonempty strings")

        if name == "industries":
            values = frame["industry"]
            if values.isna().any() or not all(
                isinstance(value, str) and value for value in values.drop_duplicates()
            ):
                raise ValueError("industries must contain nonempty strings")
            return

        value_column = {
            "benchmark_returns": "benchmark_return", "benchmark_weights": "benchmark_weight",
        }[name]
        values = frame[value_column]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
            raise ValueError(f"{value_column} values must be numeric")
        try:
            numeric = (np.asarray([float(value) for value in values]) if values.dtype.kind == "c"
                       else values.to_numpy(dtype=float, na_value=np.nan))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{value_column} values must be finite numbers") from None
        if not np.isfinite(numeric).all():
            raise ValueError(f"{value_column} values must be finite numbers")
        if name == "benchmark_returns" and (numeric < -1).any():
            raise ValueError("benchmark returns must be at least -1")
        if name == "benchmark_weights":
            if "snapshot_date" in frame:
                snapshots = frame["snapshot_date"]
                if snapshots.isna().any() or not all(
                    isinstance(value, str) and _is_iso_date(value)
                    for value in snapshots.drop_duplicates()
                ):
                    raise ValueError("weight snapshot dates must be YYYY-MM-DD strings")
                if (snapshots > dates).any():
                    raise ValueError("weight snapshot date must not be after its record date")
                if frame.groupby("date").snapshot_date.nunique().gt(1).any():
                    raise ValueError("weights must use one snapshot date per record date")
            if (numeric < 0).any():
                raise ValueError("benchmark weights must be nonnegative")
            for indices in frame.groupby("date", sort=False).indices.values():
                try:
                    total = fsum(float(value) for value in numeric[indices])
                except OverflowError:
                    total = float("inf")
                if not isfinite(total) or total <= 0:
                    raise ValueError("benchmark weights must have a positive finite sum for each date")

    def _external(self, name: str, date: str) -> Dataset:
        slices = self._load(name)
        if slices is None:
            return Dataset("unavailable", None, f"{name} directory is missing: {self.config.data_dir / REFERENCE_DATASETS[name]}")
        data = slices.get(date)
        if data is None:
            raise ValueError(f"{name} source has no data for {date}")
        return Dataset("available", data)

    def prepare_close(self, *, date: str, cache: CacheView) -> None:
        source = self._external("benchmark_returns", date)
        if source.status == "unavailable":
            raise ValueError(source.reason)
        value = float(source.data["benchmark_return"].iloc[0])
        data = Dataset("available", BenchmarkDay(date, value))
        cache.publish(Topic.REFERENCE_BENCHMARK, date, data)

    def prepare_signal(self, *, date: str, cache: CacheView) -> None:
        market = cache.read(Topic.MARKET_CONTEXT, date)
        universe = market.universe
        legal_codes = set(universe["code"])

        weights = self._external("benchmark_weights", date)
        if weights.status == "available" and set(weights.data.code) != legal_codes:
            raise ValueError(f"benchmark weights must cover the legal universe exactly once on {date}")

        industries = self._external("industries", date)
        if industries.status == "available":
            industry_codes = set(industries.data.code)
            if not legal_codes.issubset(industry_codes):
                raise ValueError("industries are missing codes from the legal universe")
            if industry_codes != legal_codes:
                industries = Dataset(
                    "available", industries.data.loc[industries.data.code.isin(legal_codes)],
                )

        cache.publish(Topic.REFERENCE_PORTFOLIO, date, PortfolioInputs(
            date,
            universe,
            Dataset("available", market.barra_exposures),
            weights,
            industries,
        ))

    def close(self) -> None:
        self._tables.clear()
        self._slices.clear()


def _is_iso_date(value: str) -> bool:
    try:
        return Date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _read_source(path: Path, name: str) -> pd.DataFrame:
    """Read canonical references or native HS300 index/weight/industry exports."""
    if path.is_dir():
        files = sorted(file for file in path.rglob("*")
                       if file.is_file() and file.suffix.lower() in (".csv", ".parquet"))
        if not files:
            raise ValueError(f"{name} directory contains no CSV or Parquet files: {path}")
        return pd.concat([_read_source(file, name) for file in files], ignore_index=True)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        types = {"date": "string", "code": "string", "代码": "string"}
        try:
            frame = pd.read_csv(path, dtype=types, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, dtype=types, encoding="gb18030")
    else:
        raise ValueError(f"{name} source must be a CSV or Parquet file")
    if "日期" in frame and name in _REQUIRED_COLUMNS:
        names = {"日期": "date", "代码": "code", "名称": "name"}
        if name == "benchmark_returns":
            required = {"日期", "代码", "涨跌幅"}
            names["涨跌幅"] = "benchmark_return"
        elif name == "benchmark_weights":
            required = {"日期", "代码", "权重", "指数成份日"}
            names.update({"权重": "benchmark_weight", "指数成份日": "snapshot_date", "权重来源": "source"})
        else:
            required = {"日期", "代码", "行业代码"}
            names.update({"行业代码": "industry", "行业名称": "industry_name"})
        if not required.issubset(frame.columns):
            raise ValueError(f"native {name} source is missing required columns: {sorted(required)}")
        frame = frame.rename(columns=names)
        date_columns = ("date", "snapshot_date") if name == "benchmark_weights" else ("date",)
        for column in date_columns:
            values = frame[column].astype("string")
            if values.isna().any() or not values.str.fullmatch(r"[0-9]{8}").all():
                raise ValueError(f"native {name} {column} must use YYYYMMDD")
            # Format each distinct session once instead of repeating it per security.
            unique = values.drop_duplicates()
            formatted = pd.to_datetime(unique, format="%Y%m%d").dt.strftime("%Y-%m-%d")
            frame[column] = values.map(dict(zip(unique, formatted)))
        if name == "benchmark_returns":
            if frame["code"].isna().any() or not frame["code"].eq("SH000300").all():
                raise ValueError("native benchmark_returns index code must be SH000300")
            values = frame["benchmark_return"]
            if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
                raise ValueError("native benchmark_returns 涨跌幅 must be numeric percentages")
            frame["benchmark_return"] = values / 100.0
    return frame
