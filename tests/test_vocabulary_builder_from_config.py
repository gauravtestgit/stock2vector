import pytest
import os
import pandas as pd
from src.implementations.vocabulary_builder.vocabulary_builder_from_config import VocabularyBuilderFromConfig
from src.implementations.config_providers.vocab_stocks_yaml_config_provider import VocabStocksYAMLConfigProvider
from src.implementations.data_providers.yahoo_provider import YahooFinanceProvider
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import FixedThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator


@pytest.fixture
def get_source_dir():
    return "config/vocabulary/stocks/"


@pytest.fixture
def get_source_file():
    return "config/vocabulary/stocks/sample.yaml"


@pytest.fixture
def config_provider():
    return VocabStocksYAMLConfigProvider()


@pytest.fixture
def data_provider():
    return YahooFinanceProvider()


@pytest.fixture
def builder(config_provider, data_provider):
    return VocabularyBuilderFromConfig(
        retrieve_config=config_provider,
        data_provider=data_provider,
        returns_processor=LogReturnsProcessor(),
        threshold_strategy=FixedThresholdStrategy(0.005),
        pair_generator=CoMovementPairGenerator()
    )


class TestBuildVocabFromDirectory:
    def test_returns_tickers(self, builder, get_source_dir):
        result = builder.build_vocabulary(get_source_dir)
        assert len(result) > 0

    def test_returns_list(self, builder, get_source_dir):
        result = builder.build_vocabulary(get_source_dir)
        assert isinstance(result, list)

    def test_tickers_are_strings(self, builder, get_source_dir):
        result = builder.build_vocabulary(get_source_dir)
        assert all(isinstance(t, str) for t in result)

    def test_no_duplicates(self, builder, get_source_dir):
        result = builder.build_vocabulary(get_source_dir)
        assert len(result) == len(set(result))

    def test_empty_directory_returns_empty(self, builder):
        result = builder.build_vocabulary("nonexistent/path/")
        assert result == []


class TestBuildVocabFromFile:
    def test_returns_tickers(self, builder, get_source_file):
        result = builder.build_vocabulary(get_source_file)
        assert len(result) == 7
        sample_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        for s in sample_stocks:
            assert s in result

    def test_returns_list(self, builder, get_source_file):
        result = builder.build_vocabulary(get_source_file)
        assert isinstance(result, list)

    def test_contains_known_ticker(self, builder, get_source_file):
        result = builder.build_vocabulary(get_source_file)
        assert "AAPL" in result

    def test_file_count_less_than_dir(self, builder, get_source_file, get_source_dir):
        file_result = builder.build_vocabulary(get_source_file)
        dir_result = builder.build_vocabulary(get_source_dir)
        assert len(file_result) <= len(dir_result)

    def test_build_training_pairs(self, builder, get_source_file, tmp_path):
        result = builder.build_training_pairs(get_source_file, start="2026-04-01", end="2026-04-13", output_dir=str(tmp_path))
        assert result is not None
        assert result["pair_count"] > 0
        assert result["pairs_file"] is not None
        assert result["vocab_file"] is not None
        assert os.path.exists(result["pairs_file"])
        assert os.path.exists(result["vocab_file"])

    def test_training_pairs_parquet_content(self, builder, get_source_file, tmp_path):
        result = builder.build_training_pairs(get_source_file, start="2026-04-01", end="2026-04-13", output_dir=str(tmp_path))
        pairs_df = pd.read_parquet(result["pairs_file"])
        assert "center" in pairs_df.columns
        assert "target" in pairs_df.columns
        assert len(pairs_df) == result["pair_count"]

    def test_training_pairs_vocab_parquet(self, builder, get_source_file, tmp_path):
        result = builder.build_training_pairs(get_source_file, start="2026-04-01", end="2026-04-13", output_dir=str(tmp_path))
        vocab_df = pd.read_parquet(result["vocab_file"])
        assert "index" in vocab_df.columns
        assert "ticker" in vocab_df.columns
        assert len(vocab_df) == len(result["t2i"])

    def test_training_pairs_has_mappings(self, builder, get_source_file, tmp_path):
        result = builder.build_training_pairs(get_source_file, start="2026-04-01", end="2026-04-13", output_dir=str(tmp_path))
        assert "t2i" in result
        assert "i2t" in result
        assert "AAPL" in result["t2i"]

    def test_training_pairs_has_threshold(self, builder, get_source_file, tmp_path):
        result = builder.build_training_pairs(get_source_file, start="2026-04-01", end="2026-04-13", output_dir=str(tmp_path))
        assert "threshold" in result
        assert result["threshold"] == 0.005


class TestBuildVocabEdgeCases:
    def test_returns_none_when_no_config_provider(self):
        b = VocabularyBuilderFromConfig(retrieve_config=None)
        assert b.build_vocabulary("any/path") is None
