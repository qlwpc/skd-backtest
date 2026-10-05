"""End-to-end acceptance with hand-computed prices and real components."""

from contextlib import redirect_stdout
from io import StringIO
from statistics import stdev
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pandas as pd

from market_fixtures import write_raw_open

from skd_backtest import BacktestEngine, CostConfig, FeeScheduleEntry, OptimizerConfig
from skd_backtest.schemas import METRIC_NAMES, RESULT_COLUMNS, SOURCE_COLUMNS


class FinancialIntegrationTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dates = [20180102, 20180103, 20180104, 20180105, 20180108, 20180109]
        prices = {
            "SH600000": [(10, 10), (10, 11), (12, 12), (12, 15), (13, 13), (14, 14)],
            "SZ000001": [(20, 20), (20, 20), (20, 20), (20, 22), (25, 25), (26, 26)],
        }
        for name, columns in SOURCE_COLUMNS.items():
            rows = []
            for i, day in enumerate(self.dates):
                for code, values in prices.items():
                    opening, close = values[i]
                    row = dict.fromkeys(columns, 1.0)
                    row.update({"日期": day, "代码": code, "名称": code})
                    if name == "Factor33_winsor":
                        row["is_st"] = 0
                    if name == "MarketData":
                        row.update(open=opening, close=close, high=max(opening, close),
                                   low=min(opening, close), is_suspend=False)
                    rows.append(row)
            folder = self.root / name / "2018" / "01"
            folder.mkdir(parents=True)
            pd.DataFrame(rows).to_parquet(folder / "201801.parquet", index=False)
        write_raw_open(self.root)
        self.model_dates = []
        benchmark = self.root / "HS300_index" / "benchmark.csv"
        benchmark.parent.mkdir()
        pd.DataFrame({"date": pd.to_datetime(pd.Series(self.dates).astype(str)).dt.strftime("%Y-%m-%d"),
                      "benchmark_return": 0.0}).to_csv(benchmark, index=False)

    def inference(self, *, as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.model_dates.append(as_of_date)
        self.assertNotIn("raw_open", data["MarketData"])
        self.assertNotIn("upper_limit", data["MarketData"])
        for table in data.values():
            self.assertLessEqual(table["日期"].max(), int(as_of_date.replace("-", "")))
        return pd.DataFrame({
            "date": as_of_date, "code": ["SH600000", "SZ000001"],
            "score": [2.0, 1.0] if as_of_date == "2018-01-02" else [1.0, 2.0],
        })

    def engine(self, **options):
        config = dict(data_dir=self.root, start_date="2018-01-02", end_date="2018-01-05",
                      initial_cash=1000.0, inference=self.inference,
                      lookback=1, holding_period=1, rebalance_interval=2,
                      optimizer_config=OptimizerConfig(top_k=1))
        config.update(options)
        return BacktestEngine(**config)

    def test_native_weights_industries_and_suspended_constituent_exit(self):
        for name in SOURCE_COLUMNS:
            path = self.root / name / "2018" / "01" / "201801.parquet"
            frame = pd.read_parquet(path)
            if name == "Barra_factor":
                change = (frame["日期"] >= 20180104) & frame["代码"].eq("SH600000")
                frame.loc[change, "代码"] = "SH600002"
            else:
                added = frame.loc[frame["代码"].eq("SH600000")].copy()
                added["代码"] = "SH600002"
                frame = pd.concat([frame, added], ignore_index=True)
                if name == "MarketData":
                    frame.loc[(frame["日期"] == 20180105) & frame["代码"].eq("SH600000"), "is_suspend"] = True
            frame.to_parquet(path, index=False)
        write_raw_open(self.root)
        source = self.root / "HS300_weight" / "2018"
        source.mkdir(parents=True)
        pd.DataFrame([
            (day, code, code, weight, "天软预估", 20171229)
            for day in (20180102, 20180104)
            for code, weight in ((("SH600000", 59.99), ("SZ000001", 40.)) if day == 20180102
                                 else (("SH600002", 20.), ("SZ000001", 80.)))
        ], columns=["日期", "代码", "名称", "权重", "权重来源", "指数成份日"]).to_csv(
            source / "hs300_weight_2018.csv", encoding="gbk", index=False)
        industries = self.root / "HS300_industry" / "2018"
        industries.mkdir(parents=True)
        pd.DataFrame([
            (day, code, "申万电子" if code.startswith("SH") else "申万银行",
             "SW801080" if code.startswith("SH") else "SW801780")
            for day in (20180102, 20180104)
            for code in (("SH600000", "SZ000001") if day == 20180102 else ("SH600002", "SZ000001"))
        ], columns=["日期", "代码", "行业名称", "行业代码"]).to_csv(
            industries / "hs300_industry_2018.csv", encoding="gbk", index=False)

        def inference(*, as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            today = data["Barra_factor"].loc[lambda f: f["日期"].eq(int(as_of_date.replace("-", "")))]
            return pd.DataFrame({"date": as_of_date, "code": today["代码"],
                                 "score": today["代码"].str.startswith("SH").astype(float)})

        engine = self.engine(
            inference=inference, output_dir=self.root / "native_output", friendly_output=False,

            optimizer_config=OptimizerConfig(method="barra",
                single_name_weight_limit=.8, active_weight_limit=.6,
                industry_exposure_limit=.05, barra_style_exposure_limit=.01, turnover_limit=2.),
        )
        metrics = engine.run()
        targets = engine.tables["target_weights"]
        last = targets.loc[targets.signal_date.eq("2018-01-04")].set_index("code")
        self.assertEqual(last.loc["SH600000", "target_weight"], 0.)
        self.assertEqual(last.loc["SH600002", "benchmark_weight"], .2)
        self.assertAlmostEqual(last.loc["SH600002", "target_weight"], .25)
        self.assertAlmostEqual(last.benchmark_weight.sum(), 1.)
        orders = engine.tables["orders"]
        self.assertTrue(((orders.code == "SH600000") & (orders.reject_reason == "SUSPENDED")).any())
        last_positions = engine.tables["positions"].loc[lambda f: f.date.eq("2018-01-05")]
        self.assertIn("SH600000", last_positions.code.tolist())
        self.assertTrue(engine.tables["equity_curve"].cash.ge(-1e-8).all())
        self.assertEqual(engine.tables["rankic"].n_stocks.tolist(), [2, 2])
        self.assertIsNotNone(metrics["tracking_error"])
        # Both async and sequential inference must reproduce all audit results.
        engine_sync = self.engine(
            inference=inference, friendly_output=False, async_inference=False,
            optimizer_config=engine.optimizer_config,
        )
        self.assertEqual(metrics, engine_sync.run())
        for name in engine.tables:
            pd.testing.assert_frame_equal(engine.tables[name], engine_sync.tables[name])

    def test_real_components_match_hand_computed_adjusted_portfolio(self):
        engine = self.engine(output_dir=self.root / "output")
        metrics = engine.run()
        equity = engine.tables["equity_curve"]
        self.assertEqual(equity.date.tolist(),
                         ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"])
        for actual, expected in zip(equity.portfolio_value, [1000.0, 1100.0, 1200.0, 1320.0]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(metrics["total_return"], 0.32)
        self.assertAlmostEqual(metrics["turnover"], 3.0)
        self.assertEqual(metrics["transaction_cost"], 0.0)
        self.assertEqual(metrics["failed_orders"], 0)
        self.assertEqual(set(metrics), set(METRIC_NAMES))
        self.assertEqual(metrics["annualized_excess_return"], metrics["annualized_return"])
        self.assertTrue(equity.benchmark_nav.eq(1.0).all())
        self.assertEqual(engine.account["total_shares"], {})
        self.assertAlmostEqual(engine.account["position_values"]["SZ000001"], 1320.0)
        self.assertAlmostEqual(engine.account["cash"], 0.0)
        self.assertEqual(self.model_dates, ["2018-01-02", "2018-01-04"])

        trades = engine.tables["trades"]
        self.assertEqual(list(zip(trades.date, trades.side, trades.code)), [
            ("2018-01-03", "BUY", "SH600000"),
            ("2018-01-05", "SELL", "SH600000"),
            ("2018-01-05", "BUY", "SZ000001"),
        ])
        self.assertTrue(trades.shares.isna().all())
        self.assertTrue(trades.price.isna().all())
        self.assertEqual(engine.tables["orders"].status.tolist(), ["FILLED"] * 3)
        predictions = engine.tables["predictions"].set_index(["date", "code"])
        for key, expected in {
            ("2018-01-02", "SH600000"): 0.2,
            ("2018-01-02", "SZ000001"): 0.0,
            ("2018-01-04", "SH600000"): 13 / 12 - 1,
            ("2018-01-04", "SZ000001"): 0.25,
        }.items():
            self.assertAlmostEqual(predictions.loc[key, "future_return"], expected)
        self.assertAlmostEqual(metrics["mean_rankic"], 1.0)
        self.assertAlmostEqual(metrics["rankic_std"], 0.0)
        self.assertIsNone(metrics["rankic_ir"])
        output = self.root / "output"
        expected_files = {"metrics.json", "run.log", *(name + ".csv" for name in RESULT_COLUMNS)}
        self.assertEqual({p.name for p in output.iterdir()}, expected_files)
        self.assertEqual(json.loads((output / "metrics.json").read_text(encoding="utf-8")), metrics)
        for name, columns in RESULT_COLUMNS.items():
            exported = pd.read_csv(output / (name + ".csv"))
            self.assertEqual(exported.columns.tolist(), list(columns))
            self.assertEqual(len(exported), len(engine.tables[name]))

    def test_adjusted_run_enforces_limits_across_a_factor_change_and_failed_sale(self):
        path = self.root / "MarketData/2018/01/201801.parquet"
        market = pd.read_parquet(path).astype({column: float for column in ("open", "high", "low", "close")})
        for day, price in ((20180102, 20.0), (20180103, 22.0), (20180104, 22.0), (20180105, 19.8)):
            mask = market["日期"].eq(day) & market["代码"].eq("SH600000")
            market.loc[mask, ["open", "high", "low", "close"]] = price
        market.to_parquet(path, index=False)
        raw = market[["日期", "代码", "open"]].copy()
        raw["open"] /= raw["日期"].map(lambda day: 2.0 if day == 20180102 else 4.0)
        raw.iloc[::-1].to_parquet(self.root / "MarketDataRawOpen/2018/01/201801.parquet", index=False)

        def inference(*, as_of_date, data):
            self.assertEqual(set(data), set(SOURCE_COLUMNS))
            self.assertNotIn("upper_limit", data["MarketData"])
            return pd.DataFrame({"date": as_of_date, "code": ["SH600000", "SZ000001"],
                                 "score": [2., 1.] if as_of_date < "2018-01-04" else [1., 2.]})

        baseline = None
        for prefetch, asynchronous in ((False, False), (True, True)):
            engine = self.engine(inference=inference, rebalance_interval=1, prefetch=prefetch,
                                 async_inference=asynchronous, read_batch_months=1, friendly_output=False)
            engine.run()
            orders = engine.tables["orders"]
            self.assertEqual(orders.reject_reason.iloc[0], "LIMIT_UP")
            sale = orders.loc[orders.side.eq("SELL")].iloc[0]
            self.assertEqual(sale.reject_reason, "LIMIT_DOWN")
            trades = engine.tables["trades"]
            self.assertEqual(trades.code.tolist(), ["SH600000"])
            self.assertEqual(trades.date.tolist(), ["2018-01-04"])
            self.assertEqual(engine.tables["equity_curve"].portfolio_value.tolist(), [1000., 1000., 1000., 900.])
            self.assertGreaterEqual(engine.account["cash"], 0.0)
            self.assertEqual(engine.account["position_values"], {"SH600000": 900.0})
            self.assertTrue(trades.shares.isna().all())
            self.assertTrue(trades.price.isna().all())
            if baseline is not None:
                for name in engine.tables:
                    pd.testing.assert_frame_equal(engine.tables[name], baseline.tables[name])
            baseline = engine

    def test_adjusted_run_allows_buy_at_limit_down_and_sell_at_limit_up(self):
        path = self.root / "MarketData/2018/01/201801.parquet"
        market = pd.read_parquet(path).astype({column: float for column in ("open", "high", "low", "close")})
        for day, price in ((20180102, 20.0), (20180103, 18.0), (20180104, 19.8)):
            mask = market["日期"].eq(day) & market["代码"].eq("SH600000")
            market.loc[mask, ["open", "high", "low", "close"]] = price
        market.to_parquet(path, index=False)
        write_raw_open(self.root)
        engine = self.engine(end_date="2018-01-04", rebalance_interval=1, friendly_output=False)
        engine.run()
        trades = engine.tables["trades"]
        self.assertEqual(list(zip(trades.date, trades.side, trades.code)), [
            ("2018-01-03", "BUY", "SH600000"), ("2018-01-04", "SELL", "SH600000"),
            ("2018-01-04", "BUY", "SZ000001"),
        ])
        self.assertTrue(engine.tables["orders"].status.eq("FILLED").all())
        self.assertAlmostEqual(engine.account["portfolio_value"], 1100.0)

    def test_constant_scores_count_as_zero_in_rankic_metrics_and_exports(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                def inference(as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
                    scores = [1.0, 2.0] if mixed and as_of_date == "2018-01-04" else [0.0, 0.0]
                    return pd.DataFrame({"date": as_of_date, "code": ["SH600000", "SZ000001"],
                                         "score": scores})

                output = self.root / ("mixed-scores" if mixed else "constant-scores")
                engine = self.engine(inference=inference, friendly_output=False, output_dir=output)
                metrics = engine.run()
                expected = [0.0, 1.0] if mixed else [0.0, 0.0]
                for actual, wanted in zip(engine.tables["rankic"].rankic, expected):
                    self.assertAlmostEqual(actual, wanted)
                self.assertEqual(engine.tables["rankic"].n_stocks.tolist(), [2, 2])
                self.assertAlmostEqual(metrics["mean_rankic"], 0.5 if mixed else 0.0)
                self.assertAlmostEqual(metrics["rankic_std"], 0.5 ** 0.5 if mixed else 0.0)
                self.assertEqual(metrics["positive_rankic_ratio"], 0.5 if mixed else 0.0)
                if mixed:
                    self.assertAlmostEqual(metrics["rankic_ir"], 0.5 ** 0.5)
                else:
                    self.assertIsNone(metrics["rankic_ir"])
                saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, metrics)
                exported = pd.read_csv(output / "rankic.csv")
                for actual, wanted in zip(exported.rankic, expected):
                    self.assertAlmostEqual(actual, wanted)

    def test_rankic_and_rankicir_use_full_universe_independently_of_top_k(self):
        for name in SOURCE_COLUMNS:
            path = self.root / name / "2018/01/201801.parquet"
            frame = pd.read_parquet(path)
            third = frame.loc[frame["代码"].eq("SZ000001")].copy()
            third["代码"], third["名称"] = "SZ000002", "SZ000002"
            if name == "MarketData":
                for column in ("open", "high", "low", "close"):
                    third[column] = [10., 10., 11., 20., 23., 23.]
            pd.concat([frame, third], ignore_index=True).to_parquet(path, index=False)
        write_raw_open(self.root)

        def inference(as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            return pd.DataFrame({
                "date": as_of_date, "code": ["SH600000", "SZ000001", "SZ000002"],
                "score": [3., 1., 2.] if as_of_date == "2018-01-02" else [1., 2., 3.],
            })

        baseline = None
        for top_k in (1, 2, 3):
            with self.subTest(top_k=top_k):
                engine = self.engine(inference=inference, friendly_output=False,
                                     optimizer_config=OptimizerConfig(top_k=top_k))
                metrics = engine.run()
                rankic = engine.tables["rankic"]
                targets = engine.tables["target_weights"]
                selected = targets.loc[targets.target_weight.gt(0)].groupby("signal_date").size()
                self.assertEqual(selected.tolist(), [top_k, top_k])
                self.assertEqual(rankic.n_stocks.tolist(), [3, 3])
                self.assertEqual(len(engine.tables["predictions"]), 6)
                # Full-universe ranks give 1 and 0.5; selecting the top two gives 1 and -1.
                for actual, expected in zip(rankic.rankic, [1., .5]):
                    self.assertAlmostEqual(actual, expected)
                self.assertAlmostEqual(metrics["mean_rankic"], .75)
                self.assertAlmostEqual(metrics["rankic_std"], .125 ** .5)
                self.assertAlmostEqual(metrics["rankic_ir"], .75 / (.125 ** .5))
                if baseline is None:
                    baseline = engine
                else:
                    for name in ("predictions", "rankic"):
                        pd.testing.assert_frame_equal(baseline.tables[name], engine.tables[name])
                    for name in ("mean_rankic", "rankic_std", "rankic_ir", "positive_rankic_ratio"):
                        self.assertEqual(baseline.metrics[name], metrics[name])

    def test_fees_and_slippage_match_cash_and_asset_accounting(self):
        costs = CostConfig(
            commission_rate=0.01, slippage=0.01,
            fee_schedule=(FeeScheduleEntry("2018-01-01", 0.02, 0.03),),
        )
        engine = self.engine(cost_config=costs)
        engine.run()
        first_asset = 1000 / (1.01 * 1.04)
        sold_asset = first_asset * 1.2
        sell_cash = sold_asset * 0.99 * 0.94
        final_asset = sell_cash / (1.01 * 1.04)
        self.assertAlmostEqual(engine.account["portfolio_value"], final_asset * 1.1)
        self.assertGreaterEqual(engine.account["cash"], 0.0)
        self.assertAlmostEqual(engine.account["cash"], 0.0, places=7)
        trades = engine.tables["trades"]
        expected_assets = [first_asset, sold_asset, final_asset]
        expected_cash_values = [first_asset * 1.01, sold_asset * 0.99, final_asset * 1.01]
        for actual, expected in zip(trades.position_value, expected_assets):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(trades.trade_value, expected_cash_values):
            self.assertAlmostEqual(actual, expected)
        expected_cost = (expected_cash_values[0] * 0.04
                         + expected_cash_values[1] * 0.06
                         + expected_cash_values[2] * 0.04)
        self.assertAlmostEqual(engine.metrics["transaction_cost"], expected_cost)
        self.assertAlmostEqual(trades.total_cost.sum(), expected_cost)
        self.assertEqual(engine.metrics["failed_orders"], 0)

    def test_raw_execution_uses_real_prices_and_share_locks(self):
        market_path = self.root / "MarketData" / "2018" / "01" / "201801.parquet"
        market = pd.read_parquet(market_path)
        for price in ("open", "high", "low", "close"):
            market["raw_" + price] = market[price] / 2
        market["upper_limit"] = market.raw_open * 1.1
        market["lower_limit"] = market.raw_open * 0.9
        market.to_parquet(market_path, index=False)
        engine = self.engine(
            initial_cash=10000.0, price_mode="raw_price",

        )
        engine.run()
        self.assertAlmostEqual(engine.account["portfolio_value"], 13200.0)
        self.assertAlmostEqual(engine.account["cash"], 0.0)
        self.assertEqual(engine.account["total_shares"], {"SZ000001": 1200})
        self.assertEqual(engine.account["sellable_shares"], {"SZ000001": 0})
        trades = engine.tables["trades"]
        self.assertEqual(trades.shares.tolist(), [2000, 2000, 1200])
        self.assertEqual(trades.price.tolist(), [5.0, 6.0, 10.0])
        self.assertAlmostEqual(engine.metrics["total_return"], 0.32)
        self.assertAlmostEqual(engine.metrics["turnover"], 3.0)

    def test_precomputed_scores_match_both_price_modes_and_own_their_input(self):
        path = self.root / "MarketData" / "2018" / "01" / "201801.parquet"
        market = pd.read_parquet(path)
        for price in ("open", "high", "low", "close"):
            market["raw_" + price] = market[price] / 2
        market["upper_limit"] = market.raw_open * 1.1
        market["lower_limit"] = market.raw_open * .9
        market.to_parquet(path, index=False)
        for mode in ("adjusted_return", "raw_price"):
            baseline = self.engine(price_mode=mode, initial_cash=10000., lookback=4, friendly_output=False)
            metrics = baseline.run()
            scores = baseline.tables["predictions"][["date", "code", "score"]].copy()
            frozen = self.engine(inference=None, precomputed_scores=scores, price_mode=mode,
                                 initial_cash=10000., lookback=4, friendly_output=False)
            scores["score"] = 999.
            self.assertEqual(metrics, frozen.run())
            self.assertEqual(baseline.account, frozen.account)
            for name in baseline.tables:
                pd.testing.assert_frame_equal(baseline.tables[name], frozen.tables[name], check_exact=True)
            self.assertEqual(frozen.performance["data"]["as_of_seconds"], 0.)
            self.assertEqual(frozen.performance["score_source"], "precomputed")
            self.assertEqual(metrics, frozen.run())

    def test_precomputed_scores_reject_bad_coverage_finiteness_and_dates(self):
        baseline = self.engine(friendly_output=False)
        baseline.run()
        scores = baseline.tables["predictions"][["date", "code", "score"]]
        cases = [(scores.iloc[1:].copy(), "legal universe"),
                 (scores.loc[scores.date.ne("2018-01-02")].copy(), "missing signal date"),
                 (scores.assign(score=float("inf")), "finite numeric"),
                 (pd.concat([scores, scores.iloc[[0]]]), "legal universe")]
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.engine(inference=None, precomputed_scores=invalid, friendly_output=False).run()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.engine(precomputed_scores=scores)

    def test_native_index_reaches_metrics_audit_files_and_friendly_output(self):
        path = self.root / "HS300_index" / "benchmark.csv"
        pd.DataFrame({"日期": self.dates[:4], "代码": ["SH000300"] * 4,
                      "涨跌幅": [-5., 2., -1., 3.]}).to_csv(path, encoding="gbk", index=False)
        engine = self.engine(trading_days_per_year=4,
                             output_dir=self.root / "index-output")
        with redirect_stdout(StringIO()) as output:
            metrics = engine.run()
        equity = engine.tables["equity_curve"]
        expected_returns = [-.05, .02, -.01, .03]
        expected_active = [.05, .08, 1 / 11 + .01, .07]
        pd.testing.assert_series_equal(equity.benchmark_return,
                                       pd.Series(expected_returns, name="benchmark_return"))
        pd.testing.assert_series_equal(equity.active_return,
                                       pd.Series(expected_active, name="active_return"))
        self.assertAlmostEqual(equity.benchmark_nav.iloc[0], .95)
        self.assertAlmostEqual(equity.benchmark_nav.iloc[-1], .9880893)
        self.assertAlmostEqual(metrics["annualized_excess_return"], .3319107)
        self.assertAlmostEqual(metrics["tracking_error"], stdev(expected_active) * 2)
        self.assertAlmostEqual(metrics["information_ratio"], .3319107 / (stdev(expected_active) * 2))
        for label in ("年化超额收益率", "跟踪误差", "信息比率"):
            row = next(line for line in output.getvalue().splitlines() if label in line)
            self.assertNotIn("N/A", row)
        saved = json.loads((engine.config.output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, metrics)
        exported = pd.read_csv(engine.config.output_dir / "equity_curve.csv")
        pd.testing.assert_series_equal(exported.benchmark_nav, equity.benchmark_nav)
        pd.testing.assert_series_equal(equity.portfolio_nav,
                                       pd.Series([1., 1.1, 1.2, 1.32], name="portfolio_nav"))

    def test_benchmark_directory_reloads_each_run(self):
        path = self.root / "HS300_index" / "benchmark.csv"
        benchmark = pd.DataFrame({
            "date": ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
            "benchmark_return": [0.01] * 4,
        })
        benchmark.to_csv(path, index=False)
        engine = self.engine()
        engine.run()
        first = engine.tables["equity_curve"]
        self.assertAlmostEqual(first.iloc[-1].benchmark_nav, 1.01 ** 4)
        self.assertAlmostEqual(first.iloc[0].active_return, -0.01)
        benchmark.loc[0, "benchmark_return"] = 0.02
        benchmark.to_csv(path, index=False)
        engine.run()
        second = engine.tables["equity_curve"]
        self.assertAlmostEqual(second.iloc[-1].benchmark_nav, 1.02 * 1.01 ** 3)
        self.assertAlmostEqual(second.iloc[0].active_return, -0.02)
        self.assertAlmostEqual(first.iloc[-1].benchmark_nav, 1.01 ** 4)

    def test_empty_range_exports_complete_empty_results(self):
        output = self.root / "empty-output"
        engine = self.engine(start_date="2018-01-06", end_date="2018-01-07", output_dir=output)
        metrics = engine.run()
        self.assertEqual(engine.trading_dates, [])
        self.assertEqual(engine.account["portfolio_value"], 1000.0)
        for key, value in metrics.items():
            if key in ("turnover", "transaction_cost", "failed_orders"):
                self.assertEqual(value, 0)
            else:
                self.assertIsNone(value, key)
        for name, columns in RESULT_COLUMNS.items():
            self.assertTrue(engine.tables[name].empty, name)
            self.assertEqual(pd.read_csv(output / (name + ".csv")).columns.tolist(), list(columns))
        self.assertEqual(len(list(output.iterdir())), 9)

    def test_single_day_has_cash_nav_without_signal_or_sample_variance(self):
        engine = self.engine(end_date="2018-01-02")
        metrics = engine.run()
        self.assertEqual(self.model_dates, [])
        self.assertEqual(engine.tables["equity_curve"].portfolio_nav.tolist(), [1.0])
        self.assertTrue(engine.tables["predictions"].empty)
        self.assertEqual(metrics["total_return"], 0.0)
        self.assertEqual(metrics["annualized_return"], 0.0)
        self.assertEqual(metrics["maximum_drawdown"], 0.0)
        self.assertIsNone(metrics["annualized_volatility"])
        self.assertIsNone(metrics["sharpe_ratio"])

    def test_real_financial_outputs_are_identical_in_all_async_modes(self):
        baseline = self.engine(prefetch=False, async_inference=False)
        baseline.run()
        for prefetch, asynchronous in ((False, True), (True, False), (True, True)):
            with self.subTest(prefetch=prefetch, async_inference=asynchronous):
                actual = self.engine(prefetch=prefetch, async_inference=asynchronous)
                actual.run()
                self.assertEqual(actual.metrics, baseline.metrics)
                self.assertEqual(actual.account, baseline.account)
                for name in RESULT_COLUMNS:
                    pd.testing.assert_frame_equal(actual.tables[name], baseline.tables[name])


if __name__ == "__main__":
    unittest.main()
