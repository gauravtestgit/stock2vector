import pytest
from src.universe.scrapers import TickerLoader
from src.universe.universe import UniverseManager


@pytest.fixture
def loader():
    return TickerLoader()


@pytest.fixture
def manager():
    return UniverseManager()


class TestTickerLoader:
    def test_load_nzx(self, loader):
        tickers = loader.load_nzx()
        assert len(tickers) == 50
        assert all(t.endswith(".NZ") for t in tickers)
        assert "IFT.NZ" in tickers

    def test_load_sp500(self, loader):
        tickers = loader.load_sp500()
        assert len(tickers) > 400
        assert "AAPL" in tickers

    def test_load_nasdaq100(self, loader):
        tickers = loader.load_nasdaq100()
        assert len(tickers) > 90
        assert "NVDA" in tickers

    def test_load_asx200(self, loader):
        tickers = loader.load_asx200()
        assert len(tickers) > 150
        assert all(t.endswith(".AX") for t in tickers)

    def test_missing_file_returns_empty(self, loader):
        assert loader._load_csv_tickers("nonexistent.csv") == []
        assert loader._load_yaml_tickers("nonexistent.yaml") == []


class TestBuildUniverse:
    def test_single_csv(self, manager):
        tickers = manager.build_universe(["sp_500.csv"])
        assert len(tickers) > 400
        assert "AAPL" in tickers

    def test_single_yaml(self, manager):
        tickers = manager.build_universe(["sample.yaml"])
        assert len(tickers) == 7
        assert "AAPL" in tickers

    def test_multiple_files(self, manager):
        tickers = manager.build_universe(["sp_500.csv", "nasdaq_100.csv"])
        assert len(tickers) > 400
        assert "AAPL" in tickers
        assert "NVDA" in tickers

    def test_csv_and_yaml_mixed(self, manager):
        tickers = manager.build_universe(["nasdaq_100.csv", "sample.yaml"])
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_deduplicates(self, manager):
        tickers = manager.build_universe(["sp_500.csv", "nasdaq_100.csv"])
        assert len(tickers) == len(set(tickers))

    def test_sorted(self, manager):
        tickers = manager.build_universe(["sample.yaml"])
        assert tickers == sorted(tickers)

    def test_with_anchors(self, manager):
        tickers = manager.build_universe(["sample.yaml"], anchor_groups=["sector_etfs"])
        assert "AAPL" in tickers
        assert "XLK" in tickers

    def test_with_multiple_anchor_groups(self, manager):
        tickers = manager.build_universe(
            ["sample.yaml"],
            anchor_groups=["sector_etfs", "macro", "thematic"]
        )
        assert "XLK" in tickers
        assert "GLD" in tickers
        assert "SOXX" in tickers

    def test_anchors_only(self, manager):
        tickers = manager.build_universe([], anchor_groups=["market_etfs"])
        assert "SPY" in tickers
        assert "QQQ" in tickers

    def test_unknown_file_skipped(self, manager):
        tickers = manager.build_universe(["sample.yaml", "nonexistent.csv"])
        assert len(tickers) == 7

    def test_cross_market(self, manager):
        tickers = manager.build_universe(["sp_500.csv", "asx_200.csv"])
        assert "AAPL" in tickers
        assert "BHP.AX" in tickers


class TestAnchors:
    def test_get_us_sector_anchors(self, manager):
        anchors = manager.get_anchors(["sector_etfs"])
        assert "XLK" in anchors
        assert "XLE" in anchors
        assert len(anchors) == 11

    def test_get_multiple_groups(self, manager):
        anchors = manager.get_anchors(["market_etfs", "factor_etfs"])
        assert "SPY" in anchors
        assert "VUG" in anchors

    def test_get_asx_anchors(self, manager):
        anchors = manager.get_anchors(["commodities_and_fx"])
        assert "GC=F" in anchors

    def test_unknown_group_skipped(self, manager):
        anchors = manager.get_anchors(["sector_etfs", "nonexistent_group"])
        assert "XLK" in anchors

    def test_anchor_descriptions(self, manager):
        desc = manager.get_anchor_descriptions(["sector_etfs", "macro"])
        assert desc["XLK"] == "Technology"
        assert desc["GLD"] == "Gold"


class TestNzx:
    def test_get_nzx_stocks(self, manager):
        tickers = manager.get_nzx_stocks()
        assert len(tickers) == 50
        assert "IFT.NZ" in tickers

    def test_nzx_in_universe(self, manager):
        nzx = manager.get_nzx_stocks()
        tickers = manager.build_universe(["sample.yaml"])
        # NZX not included unless explicitly added
        assert "IFT.NZ" not in tickers


class TestDiscovery:
    def test_list_stock_files(self, manager):
        files = manager.list_stock_files()
        assert "sp_500.csv" in files
        assert "sample.yaml" in files

    def test_list_anchor_groups(self, manager):
        groups = manager.list_anchor_groups()
        assert "sector_etfs" in groups
        assert "macro" in groups
        assert "thematic" in groups
