"""Market-flow orchestration; all non-market handoffs use the runtime protocol."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import pandas as pd

from .accounting import PortfolioAccounting
from .broker import Broker
from .config import BacktestConfig, CostConfig, OptimizerConfig
from .console import ConsoleReporter
from .contracts import ComponentRole as Role, MarketContext, Phase, RunCalendar, RunContext, Topic
from .cost_model import CostModel
from .data_provider import DataProvider
from .inference_pipeline import inference_days
from .label_provider import LabelProvider
from .metrics import Metrics
from .portfolio_optimizer import PortfolioOptimizer
from .prediction_evaluator import PredictionEvaluator
from .reference_data import ReferenceDataProvider
from .result_writer import ResultWriter
from .runtime_cache import RuntimeCache
from .submission_runner import Inference, SubmissionRunner


class BacktestEngine:
    """Coordinate market playback, asynchronous inference, and financial components."""

    def __init__(
        self, *, data_dir: str | Path, start_date: str, end_date: str,
        inference: Inference | None = None, initial_cash: float = 1_000_000.0,
        submission_dir: str | Path | None = None,
        rebalance_interval: int = 5, holding_period: int = 5, lookback: int = 252,
        price_mode: Literal["adjusted_return", "raw_price"] = "adjusted_return",
        optimizer_config: OptimizerConfig | None = None,
        cost_config: CostConfig | None = None,
        trading_days_per_year: int = 252, risk_free_rate: float = 0.0,
        output_dir: str | Path | None = None,
        friendly_output: bool = True,
        read_batch_months: int = 12, prefetch: bool = True, async_inference: bool = True,
        random_seed: int = 0,
        label_price_basis: Literal["adjusted_open", "raw_open"] = "adjusted_open",
        precomputed_scores: pd.DataFrame | None = None,
    ):
        self.config = BacktestConfig(
            data_dir=Path(data_dir), start_date=start_date, end_date=end_date,
            initial_cash=initial_cash, rebalance_interval=rebalance_interval,
            holding_period=holding_period, lookback=lookback, price_mode=price_mode,
            trading_days_per_year=trading_days_per_year, risk_free_rate=risk_free_rate,
            output_dir=Path(output_dir) if output_dir is not None else None,
            friendly_output=friendly_output,
            read_batch_months=read_batch_months, prefetch=prefetch, async_inference=async_inference,
            random_seed=random_seed,
            label_price_basis=label_price_basis,
        )
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.cost_config = cost_config or CostConfig()
        # Business state belongs to the cache; readers own per-run source resources.
        if sum(value is not None for value in (inference, submission_dir, precomputed_scores)) != 1:
            raise ValueError("provide exactly one of inference, submission_dir or precomputed_scores")
        self._precomputed_scores = None
        if precomputed_scores is not None:
            if (not isinstance(precomputed_scores, pd.DataFrame)
                    or not precomputed_scores.columns.is_unique
                    or not {"date", "code", "score"}.issubset(precomputed_scores.columns)):
                raise ValueError("precomputed_scores must be a DataFrame with date/code/score")
            if precomputed_scores.date.isna().any():
                raise ValueError("precomputed score dates must be nonmissing")
            self._precomputed_scores = {
                date: frame.loc[:, ["date", "code", "score"]].copy(deep=True)
                for date, frame in precomputed_scores.groupby("date", sort=False)
            }

            def frozen_inference(*, as_of_date, data):
                if as_of_date not in self._precomputed_scores:
                    raise ValueError(f"precomputed_scores missing signal date {as_of_date}")
                return self._precomputed_scores[as_of_date].copy(deep=True)

            inference = frozen_inference
        self.submission_runner = (
            SubmissionRunner.from_submission(submission_dir, random_seed) if submission_dir is not None
            else SubmissionRunner(inference, random_seed)
        )
        self._submission_dir = submission_dir
        self._submission_has_run = False
        self.data_provider = (DataProvider(replace(self.config, lookback=1), load_research=False, load_portfolio=True)
                              if precomputed_scores is not None else DataProvider(self.config))
        self.reference_data = ReferenceDataProvider(self.config)
        self.label_provider = LabelProvider(self.config)
        self.prediction_evaluator = PredictionEvaluator()
        self.optimizer = PortfolioOptimizer(self.optimizer_config)
        self.broker = Broker(self.config)
        self.cost_model = CostModel(self.cost_config)
        self.accounting = PortfolioAccounting(self.config)
        self.metrics_calculator = Metrics(trading_days_per_year, risk_free_rate)
        self.result_writer = ResultWriter(self.config.output_dir)
        self.metrics = None
        self.tables = {}
        self.account = {}
        self.trading_dates = []
        self.performance = {}

    def run(self) -> dict[str, float | int | None]:
        self.metrics, self.tables, self.account = None, {}, {}
        if self._submission_dir is not None:
            if self._submission_has_run:
                self.submission_runner = SubmissionRunner.from_submission(
                    self._submission_dir, self.config.random_seed)
            self._submission_has_run = True
        else:
            self.submission_runner.reset_random_state()
        self.trading_dates, self.performance = [], {}
        started = perf_counter()
        console = ConsoleReporter(self.config.friendly_output)
        cache = RuntimeCache(context=RunContext(self.config, self.optimizer_config, self.cost_config))
        views = {role: cache.for_component(role) for role in Role}
        control = views[Role.ENGINE]
        active_component = "engine"
        failed = False
        inference_seconds, inference_wait_seconds = 0.0, 0.0
        inference_calls, market_rows = 0, 0

        def call(role, method, **kwargs):
            nonlocal active_component
            active_component = f"{role}.{getattr(method, '__name__', type(method).__name__)}"
            control.log(level="DEBUG", message=active_component)
            return method(cache=views[role], **kwargs)

        try:
            console.status("准备回测数据…")
            call(Role.WRITER, self.result_writer.open)
            self.config.validate_data(self.optimizer_config)
            active_component = "data_provider.prepare"
            dates = self.data_provider.prepare()
            calendar = RunCalendar.from_dates(dates, self.config.rebalance_interval)
            control.publish(Topic.RUN_CALENDAR, None, calendar)
            console.progress(0, len(dates))
            playback_started = perf_counter()
            with closing(inference_days(
                self.data_provider.playback(), self.submission_runner,
                rebalance_interval=self.config.rebalance_interval,
                enabled=self.config.async_inference and self._precomputed_scores is None,
            )) as days:
                for completed, (day, prediction) in enumerate(days, start=1):
                    date = day.date
                    self.trading_dates.append(date)
                    market_rows += len(day.open_market)
                    cache.advance(date=date, phase=Phase.SETTLEMENT)
                    call(Role.BROKER, self.broker.start_day, date=date)
                    cache.advance(date=date, phase=Phase.OPEN_VALUE)
                    call(Role.ACCOUNTING, self.accounting.mark_at_open, date=date, market=day.open_market)
                    cache.advance(date=date, phase=Phase.EXECUTION)
                    with closing(call(Role.BROKER, self.broker.execute, date=date, market=day.open_market)) as execution:
                        while True:
                            active_component = "broker.execute"
                            try:
                                request_id = next(execution)
                            except StopIteration:
                                break
                            if type(request_id) is not int or not control.contains(Topic.COST_REQUEST, (date, request_id)):
                                raise RuntimeError("Broker must publish the cost request before yielding its ID")
                            call(Role.COST_MODEL, self.cost_model.calculate, date=date, request_id=request_id)
                            if not control.contains(Topic.COST_RESULT, (date, request_id)):
                                raise RuntimeError("Cost Model did not publish a quote")
                    cache.advance(date=date, phase=Phase.CLOSE_VALUE)
                    call(Role.REFERENCE_DATA, self.reference_data.prepare_close, date=date)
                    call(Role.ACCOUNTING, self.accounting.mark_to_market, date=date, market=day.close_market)
                    if day.portfolio is not None:
                        cache.advance(date=date, phase=Phase.SIGNAL)
                        barra = day.portfolio["barra_exposures"].rename(columns={"日期": "date", "代码": "code"})
                        barra["date"] = pd.to_datetime(barra["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
                        control.publish(Topic.MARKET_CONTEXT, date, MarketContext(date, barra[["code"]], barra))
                        call(Role.REFERENCE_DATA, self.reference_data.prepare_signal, date=date)
                        if prediction is None:
                            inference_started = perf_counter()
                            call(Role.RUNNER, self.submission_runner.predict, as_of_date=date,
                                 data={} if self._precomputed_scores is not None else day.research)
                            inference_seconds += perf_counter() - inference_started
                        else:
                            active_component = "runner.predict"
                            wait_started = perf_counter()
                            scores, duration = prediction.result()
                            inference_wait_seconds += perf_counter() - wait_started
                            inference_seconds += duration
                            call(Role.RUNNER, self.submission_runner.publish_scores,
                                 as_of_date=date, scores=scores)
                        inference_calls += 1
                        call(Role.OPTIMIZER, self.optimizer.optimize, signal_date=date)
                    call(Role.WRITER, self.result_writer.flush_log)
                    cache.finish_day(date=date)
                    console.progress(completed, len(dates), date)
                    active_component = "data_provider.playback"
            playback_seconds = perf_counter() - playback_started
            console.status("计算标签与评价指标…")
            cache.advance(date=None, phase=Phase.EVALUATION)
            call(Role.LABEL_PROVIDER, self.label_provider.build)
            call(Role.EVALUATOR, self.prediction_evaluator.evaluate)
            cache.advance(date=None, phase=Phase.METRICS)
            call(Role.METRICS, self.metrics_calculator.calculate)
            cache.advance(date=None, phase=Phase.OUTPUT)
            if self.config.output_dir is not None:
                console.status("保存回测结果…")
            call(Role.WRITER, self.result_writer.write)
            control.read(Topic.OUTPUT_RECEIPT)  # Require a completed output/disabled receipt.
            metrics = control.read(Topic.EVALUATION_METRICS)
            tables = control.result_tables()
            if dates:
                closing_snapshot = control.read(Topic.ACCOUNT_CLOSE, dates[-1])
                state = closing_snapshot.account
                equity = closing_snapshot.equity_curve.iloc[0]
                market_value, portfolio_value = equity["market_value"], equity["portfolio_value"]
            else:
                initial = control.read(Topic.ACCOUNT_INITIAL)
                state, market_value, portfolio_value = initial.account, 0.0, initial.portfolio_value
            positions = state.positions.set_index("code")
            account = {
                "price_mode": state.price_mode, "cash": state.cash,
                "total_shares": positions.total_shares.to_dict() if state.price_mode == "raw_price" else {},
                "sellable_shares": positions.sellable_shares.to_dict() if state.price_mode == "raw_price" else {},
                "market_value": market_value, "portfolio_value": portfolio_value,
            }
            if state.price_mode == "adjusted_return":
                account["position_values"] = positions.position_value.to_dict()
        except BaseException as exc:
            failed = True
            failed_phase = cache.phase
            cache.advance(date=cache.date, phase=Phase.FAILED)
            control.log(level="ERROR", message=str(exc),
                        details={"component": active_component, "exception": type(exc).__name__, "failed_phase": failed_phase})
            raise
        finally:
            cleanup_error = None
            # Keep the log open while releasing the reader, so cleanup failures are recorded.
            for component, cleanup in (("data_provider", self.data_provider.close),
                                       ("reference_data", self.reference_data.close),
                                       ("writer", lambda: self.result_writer.close(cache=views[Role.WRITER]))):
                try:
                    cleanup()
                    if component == "reference_data" and not failed and cleanup_error is None:
                        control.log(level="INFO", message="run completed")
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
                    cache.advance(date=cache.date, phase=Phase.FAILED)
                    control.log(level="ERROR", message=str(exc), details={"component": component + ".close"})
            cache.close()
            console.close()
            if cleanup_error is not None and not failed:
                raise cleanup_error
        # Publish public success state only after resources have closed successfully.
        self.metrics, self.tables, self.account = metrics, tables, account
        elapsed = perf_counter() - started
        self.performance = {
            "score_source": "precomputed" if self._precomputed_scores is not None else "inference",
            "elapsed_seconds": elapsed, "playback_seconds": playback_seconds,
            "inference_seconds": inference_seconds, "inference_wait_seconds": inference_wait_seconds,
            "inference_calls": inference_calls,
            "trading_days": len(self.trading_dates), "market_rows": market_rows,
            "days_per_second": len(self.trading_dates) / elapsed,
            "source_rows_per_second": sum(self.data_provider.stats["source_rows"].values()) / elapsed,
            "data": self.data_provider.stats.copy(),
        }
        console.results(self.metrics, trading_days=len(self.trading_dates), elapsed=elapsed,
                        output_dir=self.config.output_dir)
        return self.metrics
