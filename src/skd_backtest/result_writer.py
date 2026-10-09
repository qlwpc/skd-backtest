"""Write deterministic run metrics and audit tables, plus the run log."""

from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import tempfile

import pandas as pd

from .contracts import OutputReceipt, Topic
from .runtime_cache import CacheView
from .schemas import METRIC_NAMES, RESULT_COLUMNS


class ResultWriter:
    def __init__(self, output_dir: Path | None):
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self._log = None
        self._cursor = 0
        self._opened = False
        self._marker_path: Path | None = None

    def open(self, *, cache: CacheView) -> None:
        if self._opened:
            raise RuntimeError("writer is already open")
        self._cursor = 0
        self._marker_path = None
        try:
            if self.output_dir is not None:
                names = ["metrics.json", "run.log", *(name + ".csv" for name in RESULT_COLUMNS)]
                if any((self.output_dir / name).exists() for name in names):
                    raise FileExistsError("output directory already contains run results")
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._log = (self.output_dir / "run.log").open("x", encoding="utf-8", newline="\n")
            self._opened = True
            context = cache.read(Topic.RUN_CONTEXT)
            cache.log(
                level="INFO",
                message="run configuration",
                details=json.loads(json.dumps(asdict(context), default=str, allow_nan=False)),
            )
            self.flush_log(cache=cache)
        except BaseException as failure:
            self._opened = False
            log, self._log = self._log, None
            if log is not None:
                try:
                    log.close()
                except BaseException as cleanup_error:
                    raise cleanup_error from failure
            raise
    def flush_log(self, *, cache: CacheView) -> None:
        if self._log is None:
            return
        records = cache.log_records(after_seq=self._cursor)
        if not records:
            return
        for record in records:
            # Logs contain JSON values validated at publication. Serializing
            # these fields directly avoids recursively copying every cost quote.
            line = json.dumps({"seq": record.seq, "date": record.date, "phase": record.phase,
                               "component": record.component, "level": record.level,
                               "message": record.message, "details": record.details},
                              ensure_ascii=False, allow_nan=False) + "\n"
            if self._log.write(line) != len(line):
                raise OSError("short write while flushing run log")
        self._log.flush()
        self._cursor = records[-1].seq

    def write(self, *, cache: CacheView) -> None:
        if not self._opened:
            raise RuntimeError("writer is not open")
        if self.output_dir is None:
            cache.log(level="INFO", message="file output disabled: no output directory configured")
            self.flush_log(cache=cache)
            cache.publish(Topic.OUTPUT_RECEIPT, None, OutputReceipt("disabled", None, {}))
            return
        if self._log is None:
            raise RuntimeError("run log is not open")

        metrics = cache.read(Topic.EVALUATION_METRICS)
        if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
            raise ValueError("metrics must contain exactly the 15 protocol fields")
        metrics_json = json.dumps(
            {name: _json_value(metrics[name]) for name in METRIC_NAMES},
            ensure_ascii=False, allow_nan=False, indent=2,
        ) + "\n"
        tables = cache.result_tables()
        if set(tables) != set(RESULT_COLUMNS):
            raise ValueError("result tables do not match the seven protocol tables")

        cache.log(level="INFO", message="writing result files")
        self.flush_log(cache=cache)
        marker = self.output_dir / "metrics.json"
        try:
            with tempfile.TemporaryDirectory(prefix=".skd-backtest-", dir=self.output_dir) as temporary:
                temporary_dir = Path(temporary)
                for name, columns in RESULT_COLUMNS.items():
                    tables[name].to_csv(
                        temporary_dir / (name + ".csv"),
                        columns=columns,
                        index=False,
                        na_rep="",
                        encoding="utf-8",
                        lineterminator="\n",
                    )
                with (temporary_dir / "metrics.json").open(
                    "w", encoding="utf-8", newline="\n",
                ) as stream:
                    stream.write(metrics_json)

                for name in RESULT_COLUMNS:
                    (temporary_dir / (name + ".csv")).replace(self.output_dir / (name + ".csv"))
                (temporary_dir / "metrics.json").replace(marker)
                self._marker_path = marker

            names = ["metrics.json", *(name + ".csv" for name in RESULT_COLUMNS), "run.log"]
            files = {name: str((self.output_dir / name).resolve()) for name in names}
            cache.publish(
                Topic.OUTPUT_RECEIPT, None,
                OutputReceipt("written", str(self.output_dir), files),
            )
        except BaseException:
            if self._marker_path == marker:
                self._remove_marker()
            raise

    def close(self, *, cache: CacheView) -> None:
        failure = None
        try:
            self.flush_log(cache=cache)
        except BaseException as exc:
            failure = exc
        log, self._log = self._log, None
        if log is not None:
            try:
                log.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        self._opened = False
        if failure is not None:
            try:
                self._remove_marker()
            except BaseException as cleanup_error:
                raise cleanup_error from failure
            raise failure
        self._marker_path = None

    def _remove_marker(self) -> None:
        marker, self._marker_path = self._marker_path, None
        if marker is not None:
            try:
                marker.unlink()
            except FileNotFoundError:
                pass


def _json_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
