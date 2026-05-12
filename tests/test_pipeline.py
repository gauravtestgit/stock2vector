import pytest
import json
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.cleaner import StandardCleaner
from src.implementations.pipeline.downloader import IncrementalDownloader
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.status import TrainingStatus


# ── Cache Tests ──────────────────────────────────────────────

class TestParquetCache:
    def test_write_and_read(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        df = pd.DataFrame({"AAPL": [100, 101], "MSFT": [200, 201]},
                          index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        cache.write("us", df)
        result = cache.read("us")
        assert result is not None
        assert list(result.columns) == ["AAPL", "MSFT"]
        assert len(result) == 2

    def test_read_missing_returns_none(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        assert cache.read("nonexistent") is None

    def test_exists(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        assert cache.exists("us") is False
        df = pd.DataFrame({"AAPL": [100]}, index=pd.to_datetime(["2026-01-01"]))
        cache.write("us", df)
        assert cache.exists("us") is True

    def test_get_last_date(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        df = pd.DataFrame({"AAPL": [100, 101, 102]},
                          index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
        cache.write("us", df)
        last = cache.get_last_date("us")
        assert str(last) == "2026-01-03"

    def test_get_last_date_missing_returns_none(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        assert cache.get_last_date("missing") is None

    def test_delete(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        df = pd.DataFrame({"AAPL": [100]}, index=pd.to_datetime(["2026-01-01"]))
        cache.write("us", df)
        assert cache.exists("us") is True
        cache.delete("us")
        assert cache.exists("us") is False


# ── Cleaner Tests ────────────────────────────────────────────

class TestStandardCleaner:
    def _make_prices(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="B")
        return pd.DataFrame({
            "GOOD": np.random.uniform(50, 150, 100),
            "LOW_PRICE": np.random.uniform(0.1, 0.5, 100),
            "SPARSE": [np.nan] * 90 + list(np.random.uniform(50, 100, 10)),
        }, index=dates)

    def test_drops_low_price_tickers(self):
        cleaner = StandardCleaner(min_history_pct=0.5, min_price=1.0)
        prices = self._make_prices()
        result = cleaner.clean(prices)
        assert "LOW_PRICE" not in result.columns

    def test_drops_sparse_tickers(self):
        cleaner = StandardCleaner(min_history_pct=0.8, min_price=0.01)
        prices = self._make_prices()
        result = cleaner.clean(prices)
        assert "SPARSE" not in result.columns

    def test_keeps_good_tickers(self):
        cleaner = StandardCleaner(min_history_pct=0.5, min_price=1.0)
        prices = self._make_prices()
        result = cleaner.clean(prices)
        assert "GOOD" in result.columns

    def test_forward_fills(self):
        cleaner = StandardCleaner(min_history_pct=0.5, min_price=1.0, max_ffill_days=3)
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"A": [100, np.nan, np.nan, np.nan, 104,
                                      105, np.nan, 107, 108, 109]}, index=dates)
        result = cleaner.clean(prices)
        # First 3 NaNs should be forward filled
        assert result["A"].iloc[1] == 100
        assert result["A"].iloc[2] == 100
        assert result["A"].iloc[3] == 100

    def test_drops_all_nan_tickers(self):
        cleaner = StandardCleaner(min_history_pct=0.5, min_price=0.01)
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        prices = pd.DataFrame({
            "GOOD": range(10),
            "ALL_NAN": [np.nan] * 10
        }, index=dates)
        result = cleaner.clean(prices)
        assert "ALL_NAN" not in result.columns
        assert "GOOD" in result.columns

    def test_logs_anomalies(self, tmp_path):
        log_path = str(tmp_path / "anomalies.csv")
        cleaner = StandardCleaner(min_history_pct=0.1, min_price=0.01,
                                  anomaly_threshold=0.25, anomaly_log_path=log_path)
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        # 50% jump on day 3
        prices = pd.DataFrame({"A": [100, 100, 150, 150, 150]}, index=dates)
        cleaner.clean(prices)
        assert os.path.exists(log_path)

    def test_empty_dataframe(self):
        cleaner = StandardCleaner(min_history_pct=0.5, min_price=1.0)
        result = cleaner.clean(pd.DataFrame())
        assert result.empty


# ── Downloader Tests ─────────────────────────────────────────

class TestIncrementalDownloader:
    def _mock_source(self, data):
        source = MagicMock()
        source.fetch.return_value = data
        return source

    def test_first_download(self, tmp_path):
        df = pd.DataFrame({"AAPL": [100, 101]},
                          index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        source = self._mock_source(df)
        cache = ParquetCache(str(tmp_path))
        downloader = IncrementalDownloader(source, cache)

        result = downloader.download("us", ["AAPL"], start="2026-01-01")
        assert len(result) == 2
        assert cache.exists("us")

    def test_incremental_download(self, tmp_path):
        # Seed cache with initial data
        cache = ParquetCache(str(tmp_path))
        initial = pd.DataFrame({"AAPL": [100, 101]},
                               index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        cache.write("us", initial)

        # New data
        new = pd.DataFrame({"AAPL": [102, 103]},
                           index=pd.to_datetime(["2026-01-03", "2026-01-04"]))
        source = self._mock_source(new)
        downloader = IncrementalDownloader(source, cache)

        result = downloader.download("us", ["AAPL"], start="2026-01-01")
        assert len(result) == 4
        # Source should be called with last cached date as start
        source.fetch.assert_called_once()

    def test_force_refresh(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        initial = pd.DataFrame({"AAPL": [100]},
                               index=pd.to_datetime(["2026-01-01"]))
        cache.write("us", initial)

        fresh = pd.DataFrame({"AAPL": [200, 201]},
                             index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        source = self._mock_source(fresh)
        downloader = IncrementalDownloader(source, cache)

        result = downloader.download("us", ["AAPL"], start="2026-01-01", force_refresh=True)
        assert len(result) == 2
        assert result["AAPL"].iloc[0] == 200  # fresh data, not old

    def test_writes_meta(self, tmp_path):
        df = pd.DataFrame({"AAPL": [100]},
                          index=pd.to_datetime(["2026-01-01"]))
        source = self._mock_source(df)
        cache = ParquetCache(str(tmp_path))
        downloader = IncrementalDownloader(source, cache)

        downloader.download("us", ["AAPL", "MSFT"], start="2026-01-01")
        meta_path = tmp_path / "us" / "download_meta.json"
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["ticker_count"] == 1
        assert "MSFT" in meta["failed_tickers"]

    def test_empty_fetch_returns_cached(self, tmp_path):
        cache = ParquetCache(str(tmp_path))
        initial = pd.DataFrame({"AAPL": [100]},
                               index=pd.to_datetime(["2026-01-01"]))
        cache.write("us", initial)

        source = self._mock_source(pd.DataFrame())
        downloader = IncrementalDownloader(source, cache)

        result = downloader.download("us", ["AAPL"], start="2026-01-01")
        assert len(result) == 1


# ── Status Tests ─────────────────────────────────────────────

class TestTrainingStatus:
    def test_write_and_read_status(self, tmp_path):
        status = TrainingStatus(str(tmp_path))
        status.write_status("us", "full", epoch=10, total_epochs=300,
                            loss=1.5, started_at="2026-01-01T00:00:00")
        result = status.read_status("us")
        assert result["status"] == "running"
        assert result["current_epoch"] == 10

    def test_write_complete(self, tmp_path):
        status = TrainingStatus(str(tmp_path))
        status.write_complete("us", final_loss=0.5)
        result = status.read_status("us")
        assert result["status"] == "complete"
        assert result["final_loss"] == 0.5

    def test_write_failed(self, tmp_path):
        status = TrainingStatus(str(tmp_path))
        status.write_failed("us", "out of memory")
        result = status.read_status("us")
        assert result["status"] == "failed"
        assert result["error"] == "out of memory"

    def test_read_missing_returns_none(self, tmp_path):
        status = TrainingStatus(str(tmp_path))
        assert status.read_status("nonexistent") is None

    def test_reload_signal(self, tmp_path):
        status = TrainingStatus(str(tmp_path))
        assert status.check_reload_signal("us") is False
        status.write_reload_signal("us", "v1_20260101_full")
        assert status.check_reload_signal("us") is True


# ── YFinancePriceSource Integration Test ─────────────────────

class TestYFinancePriceSource:
    def test_fetch_real_data(self):
        source = YFinancePriceSource(
            batch_size=10, retry_count=2,
            retry_wait_secs=2, rate_limit_secs=0.5
        )
        result = source.fetch(["AAPL", "MSFT"], start="2026-04-01", end="2026-04-13")
        assert not result.empty
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_failed_ticker_omitted(self):
        source = YFinancePriceSource(
            batch_size=10, retry_count=1,
            retry_wait_secs=1, rate_limit_secs=0.5
        )
        result = source.fetch(["AAPL", "INVALIDTICKER999"], start="2026-04-01", end="2026-04-13")
        assert "AAPL" in result.columns
        assert "INVALIDTICKER999" not in result.columns
