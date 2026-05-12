import pytest
import yaml
from src.implementations.config_providers.vocab_stocks_yaml_config_provider import VocabStocksConfigProvider


@pytest.fixture
def provider():
    return VocabStocksConfigProvider()


@pytest.fixture
def single_yaml(tmp_path):
    data = {"tickers": ["AAPL", "MSFT", "GOOGL"]}
    f = tmp_path / "test.yaml"
    f.write_text(yaml.dump(data))
    return f


@pytest.fixture
def multi_yaml(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.dump({"tickers": ["AAPL", "MSFT"]}))
    (tmp_path / "b.yaml").write_text(yaml.dump({"tickers": ["GOOGL", "AAPL"]}))
    return tmp_path


class TestReadConfig:
    def test_returns_tickers(self, provider, single_yaml):
        result = provider._read_config(str(single_yaml))
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_missing_tickers_key_returns_empty(self, provider, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text(yaml.dump({"source": "test"}))
        assert provider._read_config(str(f)) == []

    def test_file_not_found_raises(self, provider):
        with pytest.raises(FileNotFoundError):
            provider._read_config("nonexistent.yaml")


class TestGetConfig:
    def test_reads_all_yaml_files(self, provider, multi_yaml):
        result = provider.get_config(str(multi_yaml))
        assert set(result) == {"AAPL", "MSFT", "GOOGL"}

    def test_deduplicates_tickers(self, provider, multi_yaml):
        result = provider.get_config(str(multi_yaml))
        assert len(result) == len(set(result))

    def test_empty_directory(self, provider, tmp_path):
        assert provider.get_config(str(tmp_path)) == []

    def test_ignores_non_yaml_files(self, provider, tmp_path):
        (tmp_path / "stocks.yaml").write_text(yaml.dump({"tickers": ["AAPL"]}))
        (tmp_path / "notes.txt").write_text("not yaml")
        result = provider.get_config(str(tmp_path))
        assert result == ["AAPL"]
    
    def test_real_files_read_config(self, provider):
        result = provider._read_config("config/vocabulary/stocks/asx300.yaml")
        assert len(result) == 30

    def test_real_files_get_config(self, provider):
        result = provider.get_config("config/vocabulary/stocks/")
        assert len(result) == 0
