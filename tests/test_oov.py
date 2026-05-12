import pytest
import numpy as np
import pandas as pd
from src.implementations.embeddings.oov import WeightedAverageOOV
from src.implementations.embeddings.oov_factory import OOVStrategyFactory
from src.implementations.embeddings.metrics import CosineMetric
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.training.word2vec import Word2VecModel
from src.implementations.training.trainer import Trainer
from src.implementations.training.clipping import GradientClipper
from src.implementations.training.lr_strategy import FixedLRStrategy
from src.implementations.training.stopping import PatienceEarlyStopper
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator


@pytest.fixture
def oov_strategy():
    return WeightedAverageOOV(high_confidence_days=60, medium_confidence_days=20)


@pytest.fixture
def synthetic_data():
    """Create synthetic returns where OOV stock co-moves with AAPL and MSFT."""
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    np.random.seed(42)

    # Vocab returns
    vocab_returns = pd.DataFrame({
        "AAPL": np.random.normal(0.01, 0.02, 50),
        "MSFT": np.random.normal(0.01, 0.02, 50),
        "GOOGL": np.random.normal(-0.005, 0.02, 50),
        "TSLA": np.random.normal(0, 0.03, 50),
    }, index=dates)

    # OOV stock that co-moves with AAPL and MSFT (similar positive returns)
    oov_returns = vocab_returns["AAPL"] * 0.8 + np.random.normal(0, 0.005, 50)
    oov_returns = pd.Series(oov_returns.values, index=dates, name="AMD")

    # Embeddings (pretend trained)
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # AAPL at index 0
        [0.9, 0.1, 0.0, 0.0],  # MSFT at index 1
        [0.0, 1.0, 0.0, 0.0],  # GOOGL at index 2
        [0.0, 0.0, 1.0, 0.0],  # TSLA at index 3
    ])

    t2i = {"AAPL": 0, "MSFT": 1, "GOOGL": 2, "TSLA": 3}

    return {
        "vocab_returns": vocab_returns,
        "oov_returns": oov_returns,
        "embeddings": embeddings,
        "t2i": t2i,
        "threshold": 0.005,
    }


class TestWeightedAverageOOV:
    def test_returns_embedding_and_metadata(self, oov_strategy, synthetic_data):
        result = oov_strategy.estimate(
            ticker="AMD",
            new_returns=synthetic_data["oov_returns"],
            embeddings=synthetic_data["embeddings"],
            t2i=synthetic_data["t2i"],
            vocab_returns=synthetic_data["vocab_returns"],
            threshold=synthetic_data["threshold"],
        )
        assert result is not None
        embedding, metadata = result
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (4,)
        assert metadata.method == "weighted_average"

    def test_embedding_is_unit_length(self, oov_strategy, synthetic_data):
        result = oov_strategy.estimate(
            ticker="AMD",
            new_returns=synthetic_data["oov_returns"],
            embeddings=synthetic_data["embeddings"],
            t2i=synthetic_data["t2i"],
            vocab_returns=synthetic_data["vocab_returns"],
            threshold=synthetic_data["threshold"],
        )
        embedding, _ = result
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_oov_closer_to_comover(self, oov_strategy, synthetic_data):
        """OOV stock should be more similar to AAPL (its co-mover) than TSLA."""
        result = oov_strategy.estimate(
            ticker="AMD",
            new_returns=synthetic_data["oov_returns"],
            embeddings=synthetic_data["embeddings"],
            t2i=synthetic_data["t2i"],
            vocab_returns=synthetic_data["vocab_returns"],
            threshold=synthetic_data["threshold"],
        )
        embedding, _ = result
        metric = CosineMetric()
        sim_aapl = metric.compute(embedding, synthetic_data["embeddings"][0])
        sim_tsla = metric.compute(embedding, synthetic_data["embeddings"][3])
        assert sim_aapl > sim_tsla

    def test_metadata_has_comovers(self, oov_strategy, synthetic_data):
        result = oov_strategy.estimate(
            ticker="AMD",
            new_returns=synthetic_data["oov_returns"],
            embeddings=synthetic_data["embeddings"],
            t2i=synthetic_data["t2i"],
            vocab_returns=synthetic_data["vocab_returns"],
            threshold=synthetic_data["threshold"],
        )
        _, metadata = result
        assert metadata.co_movement_days > 0
        assert len(metadata.top_5_comovers) > 0
        assert metadata.data_days_used == 50

    def test_confidence_low_with_few_days(self):
        """With very few days, confidence should be low."""
        strategy = WeightedAverageOOV(high_confidence_days=60, medium_confidence_days=20)
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        vocab_returns = pd.DataFrame({"A": [0.02, 0.03, -0.01, 0.02, 0.01]}, index=dates)
        oov_returns = pd.Series([0.02, 0.03, -0.01, 0.02, 0.01], index=dates)
        embeddings = np.array([[1.0, 0.0]])
        t2i = {"A": 0}

        result = strategy.estimate("X", oov_returns, embeddings, t2i, vocab_returns, 0.005)
        assert result is not None
        _, metadata = result
        assert metadata.confidence == "low"

    def test_returns_none_no_overlap(self, oov_strategy):
        """Returns None when no overlapping dates."""
        dates_vocab = pd.date_range("2026-01-01", periods=10, freq="B")
        dates_oov = pd.date_range("2026-06-01", periods=10, freq="B")
        vocab_returns = pd.DataFrame({"A": range(10)}, index=dates_vocab)
        oov_returns = pd.Series(range(10), index=dates_oov)
        embeddings = np.array([[1.0, 0.0]])
        t2i = {"A": 0}

        result = oov_strategy.estimate("X", oov_returns, embeddings, t2i, vocab_returns, 0.005)
        assert result is None

    def test_returns_none_no_comovement(self, oov_strategy):
        """Returns None when OOV stock never co-moves with any vocab stock."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        # OOV always flat (within threshold), vocab always moving
        vocab_returns = pd.DataFrame({"A": [0.05] * 10}, index=dates)
        oov_returns = pd.Series([0.001] * 10, index=dates)  # below threshold
        embeddings = np.array([[1.0, 0.0]])
        t2i = {"A": 0}

        result = oov_strategy.estimate("X", oov_returns, embeddings, t2i, vocab_returns, 0.01)
        assert result is None


class TestOOVStrategyFactory:
    def test_create_default(self):
        factory = OOVStrategyFactory()
        strategy = factory.create()
        assert isinstance(strategy, WeightedAverageOOV)

    def test_create_with_kwargs(self):
        factory = OOVStrategyFactory()
        strategy = factory.create("weighted_average", high_confidence_days=100)
        assert strategy._high_days == 100

    def test_unknown_method_raises(self):
        factory = OOVStrategyFactory()
        with pytest.raises(ValueError):
            factory.create("nonexistent_method")

    def test_list_methods(self):
        factory = OOVStrategyFactory()
        methods = factory.list_methods()
        assert "weighted_average" in methods

    def test_register_new_strategy(self):
        factory = OOVStrategyFactory()
        factory.register("custom", WeightedAverageOOV)
        assert "custom" in factory.list_methods()
        strategy = factory.create("custom")
        assert isinstance(strategy, WeightedAverageOOV)


class TestOOVIntegration:
    """Integration test: train model on sample stocks, estimate OOV for a real stock."""

    def test_oov_with_real_data(self, tmp_path):
        # Train on sample stocks
        source = YFinancePriceSource(batch_size=10, retry_count=2,
                                     retry_wait_secs=2, rate_limit_secs=0.5)
        vocab_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        oov_ticker = "AMD"

        # Fetch prices for vocab + OOV
        all_tickers = vocab_tickers + [oov_ticker]
        prices = source.fetch(all_tickers, start="2026-04-01", end="2026-04-13")

        if prices.empty or oov_ticker not in prices.columns:
            pytest.skip("Could not fetch price data")

        # Split vocab and OOV
        vocab_prices = prices[vocab_tickers].dropna(axis=1)
        oov_prices = prices[[oov_ticker]].dropna()

        if vocab_prices.empty or oov_prices.empty:
            pytest.skip("Insufficient price data")

        # Compute returns
        returns_proc = LogReturnsProcessor()
        vocab_returns = returns_proc.compute(vocab_prices)
        oov_returns = returns_proc.compute(oov_prices)[oov_ticker]

        # Generate pairs and train
        t2i = {t: i for i, t in enumerate(vocab_prices.columns)}
        i2t = {i: t for i, t in enumerate(vocab_prices.columns)}
        threshold_strategy = VolatilityThresholdStrategy(0.5)
        threshold = threshold_strategy.compute(vocab_returns)
        pair_gen = CoMovementPairGenerator()
        pairs = pair_gen.generate(vocab_returns, t2i, threshold)

        if not pairs:
            pytest.skip("No training pairs generated")

        model = Word2VecModel()
        model.initialise(vocab_size=len(t2i), embed_dim=8, seed=42)
        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=NumpyPersistence(),
        )
        trainer.train(pairs, epochs=100)

        # Estimate OOV
        embeddings = model.get_embeddings()
        oov_strategy = WeightedAverageOOV()
        result = oov_strategy.estimate(
            ticker=oov_ticker,
            new_returns=oov_returns,
            embeddings=embeddings,
            t2i=t2i,
            vocab_returns=vocab_returns,
            threshold=threshold,
        )

        if result is None:
            pytest.skip("No co-movement found (too few days)")

        embedding, metadata = result
        assert embedding.shape == (8,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6
        assert metadata.method == "weighted_average"
        assert metadata.co_movement_days > 0

        # OOV embedding should be queryable via CosineMetric
        metric = CosineMetric()
        # Compare AMD to each vocab stock
        similarities = []
        for ticker, idx in t2i.items():
            sim = metric.compute(embedding, embeddings[idx])
            similarities.append((ticker, sim))
        similarities.sort(key=lambda x: -x[1])

        # Should have valid similarity scores
        assert all(-1.0 <= s <= 1.0 for _, s in similarities)
        assert len(similarities) == len(t2i)
