"""Data visibility, batch boundaries, read failures, and actual reader overlap."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, current_thread, enumerate as threads
import unittest
from unittest.mock import patch

import pandas as pd

from market_fixtures import write_raw_open
from pandas.testing import assert_frame_equal

from skd_backtest import BacktestEngine
from skd_backtest.schemas import SOURCE_COLUMNS
from protocol_support import protocol_components


class DataFlowTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.days = [
            20161129, 20161130, 20161201, 20161202, 20161230,
            20170103, 20170104, 20170105, 20170201, 20170202, 20170301, 20170302,
        ]
        self.source = {}
        for name, columns in SOURCE_COLUMNS.items():
            rows = []
            for index, day in enumerate(self.days):
                members = ("SH600000", "SZ000001") if day < 20170104 else ("SZ000001", "SH600002")
                for code in ("SH600000", "SZ000001", "SH600002"):
                    if name == "Barra_factor" and code not in members:
                        continue
                    row = dict.fromkeys(columns, float(index + 1))
                    row.update(日期=day, 代码=code, 名称=code)
                    if name == "MarketData":
                        row.update(open=float(index + 10), close=float(index + 11), is_suspend=0)
                    if name == "Factor33_winsor":
                        row["pe_ttm"] = float("nan")
                        row["is_st"] = 0
                    rows.append(row)
            table = pd.DataFrame(rows, columns=columns)
            self.source[name] = table
            for month, part in table.groupby(table["日期"] // 100):
                folder = self.root / name / str(month)[:4] / str(month)[4:]
                folder.mkdir(parents=True)
                part.iloc[::-1].to_parquet(folder / f"{month}.parquet", index=False)

        write_raw_open(self.root)
        benchmark = self.root / "HS300_index" / "benchmark.csv"
        benchmark.parent.mkdir()
        pd.DataFrame({"date": pd.to_datetime(pd.Series(self.days).astype(str)).dt.strftime("%Y-%m-%d"),
                      "benchmark_return": 0.0}).to_csv(benchmark, index=False)

    def make_engine(self, **kwargs):
        def inference(as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            return pd.DataFrame({
                "date": as_of_date, "score": 0.0,
                "code": data["Barra_factor"].loc[
                    lambda frame: frame["日期"] == int(as_of_date.replace("-", "")), "代码"],
            })
        config = dict(
            data_dir=self.root, start_date="2016-12-01", end_date="2017-03-02",
            inference=inference,
            lookback=4, read_batch_months=1, rebalance_interval=1,
        )
        config.update(kwargs)
        return BacktestEngine(**config)

    def test_frozen_scores_preserve_membership_and_batch_boundaries(self):
        baseline = self.make_engine(friendly_output=False, async_inference=False, prefetch=False)
        metrics = baseline.run()
        scores = baseline.tables["predictions"][["date", "code", "score"]]
        for prefetch in (False, True):
            frozen = self.make_engine(inference=None, precomputed_scores=scores,
                                     friendly_output=False, prefetch=prefetch)
            self.assertEqual(metrics, frozen.run())
            for name in baseline.tables:
                assert_frame_equal(baseline.tables[name], frozen.tables[name], check_exact=True)
            self.assertEqual(frozen.performance["data"]["delivered_rows"]["MarketData"], 0)

    def test_replay_matches_source_and_keeps_only_bounded_history(self):
        for prefetch in (False, True):
            engine = self.make_engine(prefetch=prefetch)
            provider = engine.data_provider
            with self.subTest(prefetch=prefetch), redirect_stdout(StringIO()), provider:
                dates = provider.prepare()
                with patch("pandas.read_parquet", wraps=pd.read_parquet) as read:
                    for date in dates:
                        day = int(date.replace("-", ""))
                        position = self.days.index(day)
                        window = self.days[max(0, position - 3):position + 1]
                        market = provider.open_market(date)
                        expected_market = self.source["MarketData"].query("日期 == @day").sort_values("代码")
                        self.assertEqual(market["code"].tolist(), expected_market["代码"].tolist())
                        self.assertEqual(market["adjusted_open"].tolist(), expected_market["open"].tolist())
                        self.assertEqual(set(market), {"date", "code", "adjusted_open", "is_suspended",
                                                       "is_missing", "previous_close", "previous_close_date",
                                                       "upper_limit", "lower_limit"})
                        self.assertEqual(
                            provider.close_market(date)["adjusted_close"].tolist(),
                            expected_market["close"].tolist(),
                        )
                        actual = provider.as_of(date)
                        members = self.source["Barra_factor"][["日期", "代码"]]
                        for name, source in self.source.items():
                            expected = source[source["日期"].isin(window)]
                            if name != "Barra_factor":
                                expected = expected.merge(members, on=["日期", "代码"])
                            expected = expected.sort_values(["日期", "代码"]).reset_index(drop=True)
                            assert_frame_equal(actual[name], expected)
                        exposure = provider.portfolio_inputs(date)["barra_exposures"]
                        assert_frame_equal(exposure, actual["Barra_factor"].query("日期 == @day").reset_index(drop=True))
                        # Mutating all delivered frames cannot poison later calls or engine data.
                        for table in actual.values():
                            table.iloc[0, 0] = 20990101
                        self.assertTrue((provider.as_of(date)["MarketData"]["日期"] <= day).all())
                        self.assertEqual(provider.open_market(date)["adjusted_open"].tolist(), market["adjusted_open"].tolist())
                        # At a boundary, old cached rows are only the lookback tail.
                        current_month = provider._batches[provider._batch_index][0]
                        month_start = int(current_month.start_time.strftime("%Y%m%d"))
                        old = provider._tables["_market"].query("日期 < @month_start")
                        self.assertLessEqual(old["日期"].nunique(), engine.config.lookback - 1)
                # Every full monthly dataset is read exactly once; no daily reads.
                self.assertEqual(provider.stats["data_files"], 15)
                self.assertEqual(provider.stats["batches"], 5)
                self.assertEqual(provider.stats["limit_files"], 8)
                paths = [str(call.args[0]) for call in read.call_args_list
                         if call.kwargs.get("columns") != ["日期", "代码", "is_st"]]
                self.assertEqual(len(paths), len(set(paths)))
            self.assertIsNone(provider._executor)
            self.assertFalse(provider._tables)

    def test_warmup_before_start_including_more_than_one_batch(self):
        provider = self.make_engine(start_date="2017-02-01", lookback=8).data_provider
        with redirect_stdout(StringIO()), provider:
            provider.prepare()
            data = provider.as_of("2017-02-01")
            self.assertEqual(data["MarketData"]["日期"].drop_duplicates().tolist(), self.days[1:9])
            self.assertEqual(provider.stats["warmup_days"], 7)

    def test_lookback_one_partial_month_and_no_backward_replay(self):
        provider = self.make_engine(
            start_date="2017-01-04", end_date="2017-01-05", lookback=1,
        ).data_provider
        with redirect_stdout(StringIO()), provider:
            self.assertEqual(provider.prepare(), ["2017-01-04", "2017-01-05"])
            for date in ("2017-01-04", "2017-01-05"):
                self.assertEqual(provider.as_of(date)["MarketData"]["日期"].unique().tolist(),
                                 [int(date.replace("-", ""))])
            with self.assertRaisesRegex(ValueError, "move forward"):
                provider.as_of("2017-01-04")
            with self.assertRaisesRegex(ValueError, "not a playback"):
                provider.as_of("2017-01-06")

    def test_background_read_overlaps_inference_and_is_joined_on_error(self):
        entered, release, finished = Event(), Event(), Event()
        engine = self.make_engine(start_date="2016-12-01", lookback=1)
        original = engine.data_provider._read_batch
        reader_names = []

        def load(months):
            reader_names.append(current_thread().name)
            if str(months[0]) == "2017-01":
                entered.set()
                if not release.wait(5):
                    raise AssertionError("foreground never released the reader")
            result = original(months)
            if str(months[0]) == "2017-01":
                finished.set()
            return result

        def inference(*, as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            self.assertTrue(entered.wait(5), "next batch was not started during current inference")
            release.set()
            self.assertTrue(finished.wait(5), "reader cannot progress concurrently with inference")
            raise RuntimeError("model failed during prefetch")

        engine.submission_runner.inference = inference
        try:
            with (protocol_components(engine), redirect_stdout(StringIO()),
                  patch.object(engine.data_provider, "_read_batch", side_effect=load),
                  self.assertRaisesRegex(RuntimeError, "model failed")):
                engine.run()
        finally:
            release.set()
        self.assertTrue(all(name.startswith("skd-data") for name in reader_names))
        self.assertFalse(any(thread.name.startswith("skd-data") for thread in threads()))
        self.assertIsNone(engine.metrics)
        with protocol_components(engine), redirect_stdout(StringIO()):
            engine.submission_runner.inference = self.make_engine().submission_runner.inference
            engine.run()
        self.assertEqual(engine.performance["data"]["batches"], 4)

    def test_read_errors_propagate_from_first_and_prefetched_batches(self):
        for month in ("201612", "201701"):
            for prefetch in (False, True):
                engine = self.make_engine(lookback=1, prefetch=prefetch)
                original = pd.read_parquet

                def read(path, **kwargs):
                    if Path(path).name == f"{month}.parquet" and Path(path).parts[-4] == "Factor33_winsor":
                        raise OSError("broken parquet")
                    return original(path, **kwargs)

                with (self.subTest(month=month, prefetch=prefetch), protocol_components(engine), redirect_stdout(StringIO()),
                      patch("pandas.read_parquet", side_effect=read),
                      self.assertRaisesRegex(OSError, "broken parquet")):
                    engine.run()
                self.assertIsNone(engine.data_provider._executor)
                self.assertFalse(engine.data_provider._tables)

    def test_missing_prices_cross_year_seed_and_independent_valuation_api(self):
        # An exited security disappears from annual files; another first appears in January.
        market = self.source["MarketData"].copy()
        market = market[~((market["代码"] == "SH600000") & (market["日期"] >= 20170103))]
        market = market[~((market["代码"] == "SH600002") & (market["日期"] < 20170104))]
        market.loc[(market["代码"] == "SZ000001") & (market["日期"] == 20170104), "close"] = float("nan")
        market.loc[(market["代码"] == "SZ000001") & (market["日期"] == 20170105), "close"] = float("inf")
        market.loc[(market["代码"] == "SZ000001") & (market["日期"] == 20170201), ["close", "is_suspend"]] = [16.0, 1]
        market = pd.concat([market, market.iloc[:1].assign(代码="SH601999", close=float("nan"))])
        for month, part in market.groupby(market["日期"] // 100):
            part.to_parquet(self.root / "MarketData" / str(month)[:4] / str(month)[4:] / f"{month}.parquet",
                            index=False)
        for prefetch in (False, True):
            engine = self.make_engine(start_date="2017-01-03", lookback=1, prefetch=prefetch)
            provider = engine.data_provider
            with self.subTest(prefetch=prefetch), provider:
                provider.prepare()
                first = provider.close_market("2017-01-03").set_index("code")
                exited = first.loc["SH600000"]
                self.assertTrue(exited["is_missing"])
                self.assertTrue(exited["is_stale"])
                self.assertTrue(pd.isna(exited["adjusted_close"]))
                self.assertEqual(exited["reference_close"], 15.0)
                self.assertEqual(exited["reference_date"], 20161230)
                self.assertEqual(exited["previous_close_date"], 20161230)
                self.assertNotIn("SH600002", first.index)  # No prefetched constituent leakage.
                self.assertTrue(first.loc["SH601999", "is_missing"])
                self.assertTrue(pd.isna(first.loc["SH601999", "reference_close"]))
                self.assertEqual(provider.stats["warmup_days"], 0)
                self.assertEqual(provider.stats["seed_files"], 2)

                # Offline API reads only MarketData; the active reader can subsequently continue.
                if provider._pending is not None:
                    provider._pending.result()  # Exclude the active reader's concurrent I/O from this spy.
                with patch("pandas.read_parquet", wraps=pd.read_parquet) as read:
                    valuation = provider.valuation_inputs(["SH600000", "SZ000001", "UNKNOWN"])
                self.assertTrue(all("MarketData" in str(call.args[0]) for call in read.call_args_list))
                unknown = valuation.query("code == 'UNKNOWN'")
                self.assertTrue(unknown["is_missing"].all())
                self.assertTrue(unknown["reference_close"].isna().all())
                longer = self.make_engine(start_date="2017-01-03", lookback=8).data_provider
                assert_frame_equal(valuation, longer.valuation_inputs(["SH600000", "SZ000001", "UNKNOWN"]))
                missing = provider.close_market("2017-01-04").set_index("code").loc["SZ000001"]
                self.assertFalse(missing["has_valid_close"])
                self.assertEqual(missing["reference_close"], 16.0)
                self.assertEqual(missing["reference_date"], 20170103)
                suspended = provider.close_market("2017-02-01").set_index("code").loc["SZ000001"]
                self.assertEqual(suspended["is_suspended"], 1)
                self.assertEqual(suspended["previous_close"], 16.0)
                self.assertEqual(suspended["previous_close_date"], 20170103)
                self.assertEqual(suspended["reference_close"], 16.0)
                last = provider.close_market("2017-03-02").set_index("code")
                assert_frame_equal(
                    last.loc[["SH600000", "SZ000001"]].reset_index(),
                    valuation.query("date == '2017-03-02' and code != 'UNKNOWN'").reset_index(drop=True),
                )
                first["reference_close"] = -999
                self.assertEqual(last.loc["SH600000", "reference_close"], 15.0)

    def test_public_playback_has_no_financial_calls_and_closes_on_early_stop(self):
        from contextlib import closing

        engine = self.make_engine()
        with (patch.object(engine.accounting, "mark_to_market", side_effect=AssertionError("valuation")),
              patch.object(engine.accounting, "mark_at_open", side_effect=AssertionError("open valuation")),
              patch.object(engine.broker, "start_day", side_effect=AssertionError("broker")),
              closing(engine.data_provider.playback()) as days):
            first = next(days)
            self.assertEqual(first.date, "2016-12-01")
            self.assertEqual(first.execution_date, "2016-12-02")
            self.assertIsNotNone(first.research)
            self.assertIsNotNone(first.portfolio)
            self.assertNotIn("adjusted_close", first.open_market)
            self.assertIn("reference_close", first.close_market)
        self.assertIsNone(engine.data_provider._executor)
        self.assertFalse(engine.data_provider._tables)
        with closing(engine.data_provider.playback()) as days:
            last = list(days)[-1]
        self.assertIsNone(last.research)
        self.assertIsNone(last.execution_date)

    def test_empty_month_keeps_price_history_and_empty_interval_valuation_schema(self):
        for name, table in self.source.items():
            path = self.root / name / "2017" / "02" / "201702.parquet"
            table.iloc[:0].to_parquet(path, index=False)
        provider = self.make_engine(start_date="2017-01-03", lookback=1).data_provider
        with provider:
            provider.prepare()
            last = provider.close_market("2017-03-01")
            self.assertEqual(last["previous_close_date"].tolist(), [20170105] * 3)
        empty = self.make_engine(start_date="2017-02-01", end_date="2017-02-02").data_provider.valuation_inputs()
        self.assertTrue(empty.empty)
        self.assertIn("reference_date", empty)

    def test_missing_month_and_invalid_configuration_fail_explicitly(self):
        provider = self.make_engine(end_date="2017-04-01").data_provider
        with redirect_stdout(StringIO()), provider, self.assertRaises(FileNotFoundError):
            provider.prepare()
        for field in ("read_batch_months", "lookback", "rebalance_interval"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.make_engine(**{field: 0})
        with self.assertRaises(ValueError):
            self.make_engine(start_date="2017-04-01")
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "raw_price"):
            self.make_engine(price_mode="raw_price").run()


if __name__ == "__main__":
    unittest.main()
