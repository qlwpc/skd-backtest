"""Single-run, main-thread packet exchange with explicit publication stages."""

from copy import deepcopy
from dataclasses import is_dataclass
from datetime import date as iso_date
import json
from math import isfinite
from numbers import Real
from threading import get_ident
from typing import NamedTuple

import pandas as pd

from .contracts import (
    AccountState, BenchmarkDay, CacheKey, CloseSnapshot, ComponentRole as Role,
    CostQuote, CostRequest, Dataset, ExecutionResult, InitialAccount, LogRecord, MarketContext,
    OpenSnapshot, OutputReceipt, Phase, PortfolioInputs, PredictionResult, RunCalendar,
    RunContext, TargetPlan, Topic,
)
from .schemas import (
    LABEL_COLUMNS, LOCK_COLUMNS, METRIC_NAMES, RESULT_COLUMNS,
    SCORE_COLUMNS, STATE_COLUMNS, VALUE_COLUMNS, WEIGHT_COLUMNS,
)


class _Rule(NamedTuple):
    owner: Role
    phase: Phase
    payload: type
    readers: tuple[Role, ...]
    key: str = "date"


_RULES = {
    Topic.RUN_CONTEXT: _Rule(Role.ENGINE, Phase.INITIALIZE, RunContext, tuple(Role), "run"),
    Topic.RUN_CALENDAR: _Rule(Role.ENGINE, Phase.INITIALIZE, RunCalendar, tuple(Role), "run"),
    Topic.ACCOUNT_INITIAL: _Rule(Role.ENGINE, Phase.INITIALIZE, InitialAccount,
                                (Role.BROKER, Role.ACCOUNTING), "run"),
    Topic.MARKET_CONTEXT: _Rule(Role.ENGINE, Phase.SIGNAL, MarketContext, (Role.REFERENCE_DATA,)),
    Topic.REFERENCE_BENCHMARK: _Rule(Role.REFERENCE_DATA, Phase.CLOSE_VALUE, Dataset, (Role.ACCOUNTING,)),
    Topic.REFERENCE_PORTFOLIO: _Rule(Role.REFERENCE_DATA, Phase.SIGNAL, PortfolioInputs,
                                   (Role.RUNNER, Role.OPTIMIZER)),
    Topic.ACCOUNT_SETTLED: _Rule(Role.BROKER, Phase.SETTLEMENT, AccountState, (Role.ACCOUNTING,)),
    Topic.ACCOUNT_OPEN: _Rule(Role.ACCOUNTING, Phase.OPEN_VALUE, OpenSnapshot, (Role.BROKER, Role.ACCOUNTING)),
    Topic.EXECUTION_DAY: _Rule(Role.BROKER, Phase.EXECUTION, ExecutionResult, (Role.ACCOUNTING,)),
    Topic.ACCOUNT_CLOSE: _Rule(Role.ACCOUNTING, Phase.CLOSE_VALUE, CloseSnapshot,
                              (Role.BROKER, Role.ACCOUNTING, Role.OPTIMIZER)),
    Topic.SIGNAL_SCORES: _Rule(Role.RUNNER, Phase.SIGNAL, pd.DataFrame,
                             (Role.OPTIMIZER, Role.LABEL_PROVIDER, Role.EVALUATOR)),
    Topic.SIGNAL_TARGETS: _Rule(Role.OPTIMIZER, Phase.SIGNAL, TargetPlan, (Role.BROKER,), "execution"),
    Topic.COST_REQUEST: _Rule(Role.BROKER, Phase.EXECUTION, CostRequest, (Role.COST_MODEL,), "quote"),
    Topic.COST_RESULT: _Rule(Role.COST_MODEL, Phase.EXECUTION, CostQuote, (Role.BROKER,), "quote"),
    Topic.EVALUATION_LABELS: _Rule(Role.LABEL_PROVIDER, Phase.EVALUATION, pd.DataFrame, (Role.EVALUATOR,), "run"),
    Topic.EVALUATION_PREDICTION: _Rule(Role.EVALUATOR, Phase.EVALUATION, PredictionResult,
                                    (Role.METRICS, Role.WRITER), "run"),
    Topic.EVALUATION_METRICS: _Rule(Role.METRICS, Phase.METRICS, dict, (Role.WRITER,), "run"),
    Topic.OUTPUT_RECEIPT: _Rule(Role.WRITER, Phase.OUTPUT, OutputReceipt, (), "run"),
}
_DEPENDENCIES = {
    Topic.REFERENCE_PORTFOLIO: (Topic.MARKET_CONTEXT,),
    Topic.ACCOUNT_CLOSE: (Topic.REFERENCE_BENCHMARK,),
    Topic.SIGNAL_SCORES: (Topic.REFERENCE_PORTFOLIO,),
    Topic.SIGNAL_TARGETS: (Topic.SIGNAL_SCORES,),
    Topic.EVALUATION_PREDICTION: (Topic.EVALUATION_LABELS,),
}
_REQUIRED = {
    Phase.INITIALIZE: (Topic.RUN_CALENDAR,),
    Phase.SETTLEMENT: (Topic.ACCOUNT_SETTLED,),
    Phase.OPEN_VALUE: (Topic.ACCOUNT_OPEN,),
    Phase.EXECUTION: (Topic.EXECUTION_DAY,),
    Phase.CLOSE_VALUE: (Topic.REFERENCE_BENCHMARK, Topic.ACCOUNT_CLOSE),
    Phase.SIGNAL: (Topic.MARKET_CONTEXT, Topic.REFERENCE_PORTFOLIO, Topic.SIGNAL_SCORES, Topic.SIGNAL_TARGETS),
    Phase.EVALUATION: (Topic.EVALUATION_LABELS, Topic.EVALUATION_PREDICTION),
    Phase.METRICS: (Topic.EVALUATION_METRICS,),
    Phase.OUTPUT: (Topic.OUTPUT_RECEIPT,),
}


def _frame(value, columns):
    """Check a fixed table schema at its first publication in a run."""
    if not isinstance(value, pd.DataFrame) or not value.columns.is_unique:
        raise ValueError("expected a DataFrame with unique columns")
    if not set(columns).issubset(value.columns):
        raise ValueError(f"missing required columns: {sorted(set(columns) - set(value.columns))}")


def _combine(parts, columns):
    if not parts:
        return pd.DataFrame(columns=columns)
    result = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    return result if list(result.columns) == list(columns) else result.loc[:, columns]


def _number(value, *, nonnegative=False):
    if (type(value) not in (float, int) and (isinstance(value, bool) or not isinstance(value, Real))) or not isfinite(value):
        raise ValueError("expected a finite number")
    if nonnegative and value < 0:
        raise ValueError("expected a nonnegative number")


class RuntimeCache:
    def __init__(self, *, context: RunContext):
        if not isinstance(context, RunContext):
            raise TypeError("context must be RunContext")
        self._thread = get_ident()
        self._phase = Phase.INITIALIZE
        self._date = None
        self._context = context
        state = AccountState(context.price_mode, context.initial_cash,
                             pd.DataFrame(columns=STATE_COLUMNS[context.price_mode]),
                             pd.DataFrame(columns=LOCK_COLUMNS))
        initial = InitialAccount(state, context.initial_cash, 1.0, 1.0)
        self._live = {(Topic.RUN_CONTEXT, None): self._context, (Topic.ACCOUNT_INITIAL, None): initial}
        self._calendar = None
        self._day_index = -1
        self._signals = {}
        self._executions = {}
        self._day_finished = False
        self._quote_id = 0
        self._active_quote = None
        self._audit = {name: [] for name in RESULT_COLUMNS}
        self._score_history = []
        self._checked_schemas = set()
        self._results = None
        self._scores = None
        self._logs = []

    @property
    def phase(self):
        return self._phase

    @property
    def date(self):
        return self._date

    def _check(self):
        if get_ident() != self._thread:
            raise RuntimeError("RuntimeCache belongs to the run's main thread")
        if self._phase == Phase.CLOSED:
            raise RuntimeError("cache is closed")

    def for_component(self, role: Role):
        self._check()
        return CacheView(self, Role(role))

    def _key_for(self, topic):
        kind = _RULES[topic].key
        if kind == "run":
            return None
        if kind == "execution":
            return self._signals[self._date]
        return self._date

    def _complete_phase(self):
        for topic in _REQUIRED.get(self._phase, ()):
            if (topic, self._key_for(topic)) not in self._live:
                raise RuntimeError(f"unfinished {self._phase}: missing {topic}")
        if self._active_quote is not None:
            raise RuntimeError("cannot leave a phase with an outstanding cost request")

    def advance(self, *, date: str | None, phase: Phase):
        self._check()
        phase = Phase(phase)
        if phase == Phase.FAILED:
            self._phase = phase
            return
        if self._phase == Phase.FAILED:
            raise RuntimeError("failed runs cannot resume")
        if self._day_finished:
            dates = self._calendar.trading_dates
            next_index = self._day_index + 1
            expected = (Phase.SETTLEMENT, dates[next_index]) if next_index < len(dates) else (Phase.EVALUATION, None)
        elif self._phase == Phase.INITIALIZE:
            self._complete_phase()
            dates = self._calendar.trading_dates
            expected = (Phase.SETTLEMENT, dates[0]) if dates else (Phase.EVALUATION, None)
        else:
            successors = {Phase.SETTLEMENT: Phase.OPEN_VALUE, Phase.OPEN_VALUE: Phase.EXECUTION,
                          Phase.EXECUTION: Phase.CLOSE_VALUE, Phase.EVALUATION: Phase.METRICS,
                          Phase.METRICS: Phase.OUTPUT}
            if self._phase == Phase.CLOSE_VALUE and self._date in self._signals:
                expected = (Phase.SIGNAL, self._date)
            elif self._phase in successors:
                following = successors[self._phase]
                expected = (following, self._date if following in (Phase.OPEN_VALUE, Phase.EXECUTION, Phase.CLOSE_VALUE) else None)
            else:
                raise RuntimeError("finish_day or close is required")
            self._complete_phase()
        if (phase, date) != expected:
            raise RuntimeError(f"expected {expected}, got {(phase, date)}")
        self._phase, self._date = phase, date
        self._day_finished = False
        if phase == Phase.SETTLEMENT:
            self._day_index += 1
            self._quote_id = 0

    def finish_day(self, *, date: str):
        self._check()
        expected = Phase.SIGNAL if self._date in self._signals else Phase.CLOSE_VALUE
        if date != self._date or self._phase != expected or self._day_finished:
            raise RuntimeError("day is not ready to finish")
        self._complete_phase()
        # Keep only the last closing account and targets due on a future day.
        for topic, key in list(self._live):
            if _RULES[topic].key == "run":
                continue
            if topic == Topic.ACCOUNT_CLOSE and key == date:
                continue
            if topic == Topic.SIGNAL_TARGETS and key > date:
                continue
            del self._live[topic, key]
        self._day_finished = True

    def close(self):
        if get_ident() != self._thread:
            raise RuntimeError("cache must close on its owning thread")
        self._live.clear()
        self._audit.clear()
        self._score_history.clear()
        self._checked_schemas.clear()
        self._results = None
        self._scores = None
        self._logs.clear()
        self._calendar = None
        self._context = None
        self._signals.clear()
        self._executions.clear()
        self._active_quote = None
        self._phase = Phase.CLOSED

    def _read_allowed(self, role, topic, key):
        self._check()
        if role != Role.ENGINE and role not in _RULES[topic].readers:
            raise PermissionError(f"{role} cannot read {topic}")
        if topic == Topic.EVALUATION_LABELS and role != Role.EVALUATOR:
            raise PermissionError("labels are evaluator-only")
        kind = _RULES[topic].key
        if kind == "run":
            if key is not None:
                raise ValueError("run topics use key=None")
            return
        if role == Role.ENGINE:
            return
        if kind == "quote":
            if self._phase != Phase.EXECUTION or key != self._active_quote:
                raise RuntimeError("cost reads require the active request")
            return
        if topic == Topic.ACCOUNT_CLOSE and role in (Role.ACCOUNTING, Role.BROKER):
            previous = self._calendar.trading_dates[self._day_index - 1] if self._day_index > 0 else None
            if key not in (self._date, previous) or key is None:
                raise PermissionError("only current or previous closing snapshot is visible")
            if role == Role.BROKER and key != previous:
                raise PermissionError("settlement may only read the previous close")
        elif key != self._date:
            raise PermissionError("only the exact current date is visible")

    def _publish(self, role, topic, key, value):
        self._check()
        rule = _RULES[topic]
        if role != rule.owner or topic in (Topic.RUN_CONTEXT, Topic.ACCOUNT_INITIAL):
            raise PermissionError(f"{role} cannot publish {topic}")
        if self._phase != rule.phase or self._day_finished:
            raise RuntimeError(f"{topic} must be published during {rule.phase}")
        if rule.key == "quote":
            if (not isinstance(key, tuple) or len(key) != 2 or key[0] != self._date
                    or type(key[1]) is not int or key[1] < 1):
                raise ValueError("cost key must be (current date, positive integer request ID)")
        elif key != self._key_for(topic):
            raise ValueError(f"incorrect key for {topic}")
        if (topic, key) in self._live:
            raise RuntimeError(f"duplicate publication: {topic}, {key}")
        if not isinstance(value, rule.payload):
            raise TypeError(f"{topic} requires {rule.payload.__name__}")
        for dependency in _DEPENDENCIES.get(topic, ()):
            if (dependency, self._key_for(dependency)) not in self._live:
                raise RuntimeError(f"{topic} requires {dependency}")
        self._validate(topic, key, value)
        # Publication transfers ownership; all subsequent access borrows a read-only reference.
        packet = value
        self._live[topic, key] = packet
        if topic == Topic.RUN_CALENDAR:
            self._calendar = packet
            self._signals = dict(packet.signal_calendar.itertuples(index=False, name=None))
            self._executions = {execution: signal for signal, execution in self._signals.items()}
        elif topic == Topic.COST_REQUEST:
            self._active_quote = key
            self._quote_id = key[1]
        elif topic == Topic.SIGNAL_SCORES:
            self._score_history.append(packet)
        elif topic == Topic.SIGNAL_TARGETS:
            self._archive("target_weights", packet.weights)
        elif topic == Topic.EXECUTION_DAY:
            for name in ("orders", "trades"):
                self._archive(name, getattr(packet, name))
        elif topic == Topic.ACCOUNT_CLOSE:
            for name in ("positions", "equity_curve"):
                self._archive(name, getattr(packet, name))
        elif topic == Topic.EVALUATION_PREDICTION:
            self._audit["predictions"] = [packet.predictions]
            self._audit["rankic"] = [packet.rankic]

    def _archive(self, name, frame):
        if not frame.empty:
            self._audit[name].append(frame)

    def _schema(self, name, value, columns):
        if name not in self._checked_schemas:
            _frame(value, columns)
            self._checked_schemas.add(name)

    def _account(self, account):
        if not isinstance(account, AccountState) or account.price_mode != self._context.price_mode:
            raise ValueError("account price mode must match the run")
        _number(account.cash, nonnegative=True)
        self._schema("account.positions", account.positions, STATE_COLUMNS[account.price_mode])
        self._schema("account.locked_lots", account.locked_lots, LOCK_COLUMNS)
        if account.price_mode == "adjusted_return" and not account.locked_lots.empty:
            raise ValueError("adjusted-return accounts cannot contain share locks")

    def _dataset(self, name, dataset, columns):
        if not isinstance(dataset, Dataset):
            raise TypeError("expected Dataset")
        if dataset.status == "available":
            self._schema(name, dataset.data, columns)

    def _validate(self, topic, key, value):
        day = self._date
        if is_dataclass(value) and hasattr(value, "date") and value.date != day:
            raise ValueError("packet date must match the current day")
        if is_dataclass(value) and hasattr(value, "account"):
            self._account(value.account)
        if topic == Topic.RUN_CALENDAR:
            dates = value.trading_dates
            if not isinstance(dates, tuple) or tuple(sorted(set(dates))) != dates:
                raise ValueError("calendar must contain sorted unique dates")
            for date in dates:
                if iso_date.fromisoformat(date).isoformat() != date:
                    raise ValueError("calendar dates must use YYYY-MM-DD")
                if not self._context.backtest.start_date <= date <= self._context.backtest.end_date:
                    raise ValueError("calendar date outside configured interval")
            expected = RunCalendar.from_dates(dates, self._context.backtest.rebalance_interval)
            if not value.signal_calendar.equals(expected.signal_calendar):
                raise ValueError("signal calendar must use the next actual trading day")
        elif topic == Topic.ACCOUNT_SETTLED:
            self._account(value)
        elif topic == Topic.MARKET_CONTEXT:
            self._schema("universe", value.universe, ("code",))
            self._schema("barra", value.barra_exposures, ("date", "code"))
        elif topic == Topic.REFERENCE_BENCHMARK:
            if value.status != "available":
                raise ValueError("benchmark data must be available")
            if not isinstance(value.data, BenchmarkDay) or value.data.date != day:
                raise ValueError("benchmark must belong to the current close")
            _number(value.data.benchmark_return)
        elif topic == Topic.REFERENCE_PORTFOLIO:
            self._schema("universe", value.universe, ("code",))
            self._dataset("barra", value.barra_exposures, ("date", "code"))
            self._dataset("benchmark_weights", value.benchmark_weights, ("date", "code", "benchmark_weight"))
            self._dataset("industries", value.industries, ("date", "code", "industry"))
        elif topic == Topic.ACCOUNT_OPEN:
            self._schema("open.values", value.values, VALUE_COLUMNS)
            _number(value.market_value, nonnegative=True)
            _number(value.portfolio_value, nonnegative=True)
        elif topic == Topic.EXECUTION_DAY:
            if self._active_quote is not None:
                raise RuntimeError("finish the cost exchange before publishing execution")
            if value.executed_signal_date != self._executions.get(day):
                raise ValueError("execution must consume exactly the signal scheduled for today")
            self._schema("orders", value.orders, RESULT_COLUMNS["orders"])
            self._schema("trades", value.trades, RESULT_COLUMNS["trades"])
            _number(value.trade_value, nonnegative=True)
            _number(value.total_cost, nonnegative=True)
        elif topic == Topic.ACCOUNT_CLOSE:
            self._schema("positions", value.positions, RESULT_COLUMNS["positions"])
            self._schema("equity_curve", value.equity_curve, RESULT_COLUMNS["equity_curve"])
            self._schema("actual_weights", value.actual_weights, WEIGHT_COLUMNS)
            if len(value.equity_curve) != 1:
                raise ValueError("close must publish exactly one equity row")
        elif topic == Topic.SIGNAL_TARGETS:
            if value.signal_date != day or value.execution_date != key:
                raise ValueError("target signal/execution dates do not match the calendar")
            self._schema("target_weights", value.weights, RESULT_COLUMNS["target_weights"])
        elif topic in (Topic.COST_REQUEST, Topic.COST_RESULT):
            self._validate_cost(topic, key, value)
        elif topic == Topic.EVALUATION_LABELS:
            self._schema("labels", value, LABEL_COLUMNS)
            if not value.date.isin(self._signals).all():
                raise ValueError("labels must refer to signal dates")
            if not value.label_price_basis.eq(self._context.label_price_basis).all():
                raise ValueError("label price basis must match the run")
        elif topic == Topic.EVALUATION_PREDICTION:
            self._schema("predictions", value.predictions, RESULT_COLUMNS["predictions"])
            self._schema("rankic", value.rankic, RESULT_COLUMNS["rankic"])
            if not value.predictions.date.isin(self._signals).all() or not value.rankic.date.isin(self._signals).all():
                raise ValueError("evaluation contains a non-signal date")
        elif topic == Topic.EVALUATION_METRICS:
            if set(value) != set(METRIC_NAMES):
                raise ValueError("metrics must contain exactly the 15 protocol keys")
            for number in value.values():
                if number is not None:
                    _number(number)
        elif topic == Topic.OUTPUT_RECEIPT:
            if value.status == "disabled":
                if value.output_dir is not None or value.files:
                    raise ValueError("disabled output cannot contain paths")
            elif value.status == "written":
                names = {"metrics.json", "run.log", *(name + ".csv" for name in RESULT_COLUMNS)}
                if value.output_dir is None or set(value.files) != names:
                    raise ValueError("written receipt must list every result file")
            else:
                raise ValueError("invalid output status")

    def _validate_cost(self, topic, key, value):
        if (type(value.request_id) is not int or value.request_id != key[1]
                or not isinstance(value.order_id, str) or not value.order_id or value.side not in ("BUY", "SELL")):
            raise ValueError("invalid cost identity or side")
        if value.price_mode != self._context.price_mode:
            raise ValueError("cost price mode must match the run")
        if topic == Topic.COST_REQUEST:
            if self._active_quote is not None or value.request_id <= self._quote_id:
                raise RuntimeError("cost requests must be sequential and strictly increasing")
            if value.price_mode == "raw_price":
                _number(value.base_price)
                if (value.base_price <= 0 or type(value.shares) is not int or value.shares <= 0
                        or value.position_value is not None):
                    raise ValueError("raw cost requests require positive price/shares only")
            else:
                _number(value.position_value)
                if value.position_value <= 0 or value.base_price is not None or value.shares is not None:
                    raise ValueError("adjusted cost requests require positive position value only")
        else:
            if key != self._active_quote:
                raise RuntimeError("quote does not match the outstanding request")
            request = self._live[Topic.COST_REQUEST, key]
            for name in ("request_id", "order_id", "date", "side", "price_mode"):
                if getattr(value, name) != getattr(request, name):
                    raise ValueError("quote/request identity mismatch")
            for name in ("position_value", "trade_value", "commission", "stamp_tax", "other_cost", "total_cost"):
                _number(getattr(value, name), nonnegative=True)
            _number(value.cash_delta)
            if value.price_mode == "adjusted_return":
                if value.execution_price is not None:
                    raise ValueError("adjusted quotes cannot invent an execution price")
            else:
                _number(value.execution_price)
                if value.execution_price <= 0:
                    raise ValueError("execution price must be positive")


class CacheView:
    """A component's fixed role; this view does not expose lifecycle operations."""

    __slots__ = ("__cache", "__role")

    def __init__(self, cache: RuntimeCache, role: Role):
        self.__cache = cache
        self.__role = role

    def contains(self, topic: Topic, key: CacheKey = None) -> bool:
        topic = Topic(topic)
        self.__cache._read_allowed(self.__role, topic, key)
        return (topic, key) in self.__cache._live

    def read(self, topic: Topic, key: CacheKey = None):
        topic = Topic(topic)
        self.__cache._read_allowed(self.__role, topic, key)
        return self.__cache._live[topic, key]

    def publish(self, topic: Topic, key: CacheKey, value):
        self.__cache._publish(self.__role, Topic(topic), key, value)

    def history(self, topic: Topic, *, date: str | None = None) -> pd.DataFrame:
        cache, role, topic = self.__cache, self.__role, Topic(topic)
        cache._check()
        if topic == Topic.SIGNAL_SCORES:
            if role not in (Role.LABEL_PROVIDER, Role.EVALUATOR) or cache._phase != Phase.EVALUATION:
                raise PermissionError("score history is evaluator-only after playback")
            if cache._scores is None:
                cache._scores = _combine(cache._score_history, SCORE_COLUMNS)
            return cache._scores if date is None else cache._scores.loc[cache._scores.date == date]
        raise PermissionError("no history projection for this topic")

    def result_tables(self) -> dict[str, pd.DataFrame]:
        cache = self.__cache
        cache._check()
        if self.__role not in (Role.METRICS, Role.WRITER, Role.ENGINE):
            raise PermissionError("result tables are only available to metrics, writer and engine")
        if cache._phase not in (Phase.EVALUATION, Phase.METRICS, Phase.OUTPUT):
            raise RuntimeError("result tables are unavailable during playback")
        if (Topic.EVALUATION_PREDICTION, None) not in cache._live:
            raise RuntimeError("prediction evaluation is unfinished")
        if cache._results is None:
            cache._results = {name: _combine(parts, RESULT_COLUMNS[name])
                              for name, parts in cache._audit.items()}
        return cache._results

    def finish_quote(self, *, date: str, request_id: int):
        cache = self.__cache
        cache._check()
        if self.__role != Role.BROKER:
            raise PermissionError("only Broker can finish a cost exchange")
        key = (date, request_id)
        if cache._phase != Phase.EXECUTION or key != cache._active_quote or (Topic.COST_RESULT, key) not in cache._live:
            raise RuntimeError("no completed quote to finish")
        del cache._live[Topic.COST_REQUEST, key]
        del cache._live[Topic.COST_RESULT, key]
        cache._active_quote = None

    def log(self, *, level: str, message: str, details: dict | None = None):
        cache = self.__cache
        cache._check()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR") or not isinstance(message, str):
            raise ValueError("invalid log level/message")
        if details is not None and not isinstance(details, dict):
            raise TypeError("log details must be a dictionary")
        json.dumps(details, allow_nan=False)
        cache._logs.append(LogRecord(len(cache._logs) + 1, cache._date, cache._phase,
                                     self.__role, level, message, deepcopy(details or {})))

    def log_records(self, *, after_seq: int) -> list[LogRecord]:
        cache = self.__cache
        cache._check()
        if self.__role != Role.WRITER:
            raise PermissionError("only Writer can read run logs")
        if type(after_seq) is not int or after_seq < 0:
            raise ValueError("after_seq must be a nonnegative integer")
        return cache._logs[after_seq:]
