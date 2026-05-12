import pytest
import numpy as np
import pandas as pd
from src.implementations.embeddings.metrics import CosineMetric
from src.implementations.embeddings.oov import WeightedAverageOOV
from src.implementations.embeddings.oov_factory import OOVStrategyFactory
from src.implementations.training.word2vec import Word2VecModel
from src.implementations.training.trainer import Trainer
from src.implementations.training.clipping import GradientClipper
from src.implementations.training.lr_strategy import FixedLRStrategy
from src.implementations.training.stopping import PatienceEarlyStopper
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.vocabulary_builder.vocabulary_builder_from_config import VocabularyBuilderFromConfig
from src.implementations.config_providers.vocab_stocks_yaml_config_provider import VocabStocksYAMLConfigProvider
from src.implementations.data_providers.yahoo_provider import YahooFinanceProvider
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import FixedThresholdStrategy, VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator


@pytest.fixture
def metric():
    return CosineMetric()


@pytest.fixture
def builder():
    return VocabularyBuilderFromConfig(
        retrieve_config=VocabStocksYAMLConfigProvider(),
        data_provider=YahooFinanceProvider(),
        returns_processor=LogReturnsProcessor(),
        threshold_strategy=FixedThresholdStrategy(0.005),
        pair_generator=CoMovementPairGenerator()
    )


@pytest.fixture
def real_training_data(builder, tmp_path):
    result = builder.build_training_pairs(
        "config/vocabulary/stocks/sample.yaml",
        start="2026-04-01", end="2026-04-13",
        output_dir=str(tmp_path)
    )
    pairs_df = pd.read_parquet(result["pairs_file"])
    all_pairs = list(zip(pairs_df["center"], pairs_df["target"]))
    return {"pairs": all_pairs, "t2i": result["t2i"], "i2t": result["i2t"]}


class TestComputePairwise:
    def test_identical_vectors(self, metric):
        a = np.array([1.0, 2.0, 3.0])
        assert metric.compute(a, a) == pytest.approx(1.0)

    def test_opposite_vectors(self, metric):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert metric.compute(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self, metric):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert metric.compute(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self, metric):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert metric.compute(a, b) == 0.0
        assert metric.compute(b, a) == 0.0

    def test_symmetry(self, metric):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert metric.compute(a, b) == pytest.approx(metric.compute(b, a))

    def test_result_bounded(self, metric):
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.standard_normal(32)
            b = rng.standard_normal(32)
            sim = metric.compute(a, b)
            assert -1.0 <= sim <= 1.0


class TestComputeMatrix:
    def test_shape(self, metric):
        embeddings = np.random.randn(5, 8)
        matrix = metric.compute_matrix(embeddings)
        assert matrix.shape == (5, 5)

    def test_diagonal_is_one(self, metric):
        embeddings = np.random.randn(5, 8)
        matrix = metric.compute_matrix(embeddings)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)

    def test_symmetric(self, metric):
        embeddings = np.random.randn(5, 8)
        matrix = metric.compute_matrix(embeddings)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)

    def test_values_bounded(self, metric):
        embeddings = np.random.randn(10, 8)
        matrix = metric.compute_matrix(embeddings)
        assert matrix.min() >= -1.0 - 1e-6
        assert matrix.max() <= 1.0 + 1e-6

    def test_consistent_with_pairwise(self, metric):
        embeddings = np.random.randn(5, 8)
        matrix = metric.compute_matrix(embeddings)
        for i in range(5):
            for j in range(5):
                expected = metric.compute(embeddings[i], embeddings[j])
                assert matrix[i, j] == pytest.approx(expected, abs=1e-6)


class TestMostSimilar:
    def test_returns_list(self, metric):
        embeddings = np.random.randn(5, 8)
        t2i = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        i2t = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        result = metric.most_similar("A", t2i, i2t, embeddings, top_n=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_excludes_self(self, metric):
        embeddings = np.random.randn(5, 8)
        t2i = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        i2t = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        result = metric.most_similar("A", t2i, i2t, embeddings, top_n=10)
        tickers = [r[0] for r in result]
        assert "A" not in tickers

    def test_sorted_descending(self, metric):
        embeddings = np.random.randn(5, 8)
        t2i = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        i2t = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        result = metric.most_similar("A", t2i, i2t, embeddings, top_n=4)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_ticker_returns_empty(self, metric):
        embeddings = np.random.randn(5, 8)
        t2i = {"A": 0, "B": 1}
        i2t = {0: "A", 1: "B"}
        assert metric.most_similar("UNKNOWN", t2i, i2t, embeddings) == []

    def test_top_n_limits_results(self, metric):
        embeddings = np.random.randn(10, 8)
        t2i = {f"S{i}": i for i in range(10)}
        i2t = {i: f"S{i}" for i in range(10)}
        result = metric.most_similar("S0", t2i, i2t, embeddings, top_n=3)
        assert len(result) == 3


class TestCosineAfterTraining:
    """Integration: train model on real data, then check cosine similarity."""

    def test_trained_model_similarity(self, real_training_data, metric, tmp_path):
        data = real_training_data
        vocab_size = len(data["t2i"])
        pairs = data["pairs"]

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=32, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=NumpyPersistence()
        )

        trainer.train(pairs, epochs=300)
        embeddings = model.get_embeddings()

        # Check similarity for each ticker
        i2t = data["i2t"]
        t2i = data["t2i"]

        for ticker in t2i:
            similar = metric.most_similar(ticker, t2i, i2t, embeddings, top_n=3)
            assert len(similar) > 0
            assert all(isinstance(s[0], str) for s in similar)
            assert all(-1.0 <= s[1] <= 1.0 for s in similar)

    def test_similarity_matrix_after_training(self, real_training_data, metric, tmp_path):
        data = real_training_data
        vocab_size = len(data["t2i"])

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=32, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=NumpyPersistence()
        )

        trainer.train(data["pairs"], epochs=100)
        embeddings = model.get_embeddings()

        matrix = metric.compute_matrix(embeddings)
        assert matrix.shape == (vocab_size, vocab_size)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)

    def test_comoving_stocks_more_similar(self, real_training_data, metric, tmp_path):
        """Stocks that co-move frequently should have higher cosine similarity."""
        data = real_training_data
        vocab_size = len(data["t2i"])
        pairs = data["pairs"]

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=32, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=NumpyPersistence()
        )

        trainer.train(pairs, epochs=300)
        embeddings = model.get_embeddings()

        from collections import Counter
        pair_counts = Counter(pairs)
        most_common = pair_counts.most_common(1)[0][0]
        a_idx, b_idx = most_common

        paired_with_a = {p[1] for p in pairs if p[0] == a_idx}
        unpaired = [i for i in range(vocab_size) if i != a_idx and i not in paired_with_a]

        if unpaired:
            sim_paired = metric.compute(embeddings[a_idx], embeddings[b_idx])
            sim_unpaired = metric.compute(embeddings[a_idx], embeddings[unpaired[0]])
            assert sim_paired > sim_unpaired


class TestOOVEstimation:
    """OOV estimation tests using synthetic and real data."""

    def test_oov_returns_embedding(self, metric):
        """OOV estimation returns a valid unit-length embedding."""
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        np.random.seed(42)
        vocab_returns = pd.DataFrame({
            "AAPL": np.random.normal(0.01, 0.02, 30),
            "MSFT": np.random.normal(0.01, 0.02, 30),
            "TSLA": np.random.normal(0, 0.03, 30),
        }, index=dates)
        oov_returns = pd.Series(vocab_returns["AAPL"] * 0.9 + np.random.normal(0, 0.003, 30),
                                index=dates)
        embeddings = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 0.0, 1.0]])
        t2i = {"AAPL": 0, "MSFT": 1, "TSLA": 2}

        strategy = WeightedAverageOOV()
        result = strategy.estimate("AMD", oov_returns, embeddings, t2i, vocab_returns, 0.005)

        assert result is not None
        embedding, metadata = result
        assert embedding.shape == (3,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6
        assert metadata.method == "weighted_average"
        assert metadata.co_movement_days > 0

    def test_oov_closer_to_comover(self, metric):
        """OOV stock should be more similar to its co-mover than to unrelated stocks."""
        dates = pd.date_range("2026-01-01", periods=50, freq="B")
        np.random.seed(42)
        vocab_returns = pd.DataFrame({
            "AAPL": np.random.normal(0.01, 0.02, 50),
            "MSFT": np.random.normal(0.01, 0.02, 50),
            "TSLA": np.random.normal(-0.005, 0.03, 50),
        }, index=dates)
        # OOV co-moves with AAPL
        oov_returns = pd.Series(vocab_returns["AAPL"] * 0.8 + np.random.normal(0, 0.005, 50),
                                index=dates)
        embeddings = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
        t2i = {"AAPL": 0, "MSFT": 1, "TSLA": 2}

        strategy = WeightedAverageOOV()
        result = strategy.estimate("AMD", oov_returns, embeddings, t2i, vocab_returns, 0.005)
        embedding, _ = result

        sim_aapl = metric.compute(embedding, embeddings[0])
        sim_tsla = metric.compute(embedding, embeddings[2])
        assert sim_aapl > sim_tsla

    def test_oov_returns_none_no_overlap(self):
        """Returns None when no overlapping dates."""
        dates_vocab = pd.date_range("2026-01-01", periods=10, freq="B")
        dates_oov = pd.date_range("2026-06-01", periods=10, freq="B")
        vocab_returns = pd.DataFrame({"A": range(10)}, index=dates_vocab)
        oov_returns = pd.Series(range(10), index=dates_oov)
        embeddings = np.array([[1.0, 0.0]])
        t2i = {"A": 0}

        strategy = WeightedAverageOOV()
        assert strategy.estimate("X", oov_returns, embeddings, t2i, vocab_returns, 0.005) is None

    def test_oov_returns_none_no_comovement(self):
        """Returns None when OOV never co-moves with vocab."""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        vocab_returns = pd.DataFrame({"A": [0.05] * 10}, index=dates)
        oov_returns = pd.Series([0.001] * 10, index=dates)  # below threshold
        embeddings = np.array([[1.0, 0.0]])
        t2i = {"A": 0}

        strategy = WeightedAverageOOV()
        assert strategy.estimate("X", oov_returns, embeddings, t2i, vocab_returns, 0.01) is None

    def test_oov_factory(self):
        """Factory creates strategy by name."""
        factory = OOVStrategyFactory()
        strategy = factory.create("weighted_average")
        assert isinstance(strategy, WeightedAverageOOV)
        assert "weighted_average" in factory.list_methods()

    def test_oov_with_trained_model(self, real_training_data, metric):
        """Train on sample stocks, estimate OOV for AMD using real Yahoo data."""
        data = real_training_data
        vocab_size = len(data["t2i"])
        pairs = data["pairs"]

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=32, seed=42)
        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=NumpyPersistence()
        )
        trainer.train(pairs, epochs=100)
        embeddings = model.get_embeddings()

        # Fetch OOV stock returns
        source = YFinancePriceSource(batch_size=1, retry_count=2,
                                     retry_wait_secs=2, rate_limit_secs=0.5)
        oov_prices = source.fetch(["AMD"], start="2026-04-01", end="2026-04-13")
        if oov_prices.empty:
            pytest.skip("Could not fetch AMD prices")

        # Get vocab returns from same period
        vocab_tickers = list(data["t2i"].keys())
        vocab_prices = source.fetch(vocab_tickers, start="2026-04-01", end="2026-04-13")
        if vocab_prices.empty:
            pytest.skip("Could not fetch vocab prices")

        returns_proc = LogReturnsProcessor()
        vocab_returns = returns_proc.compute(vocab_prices)
        oov_returns = returns_proc.compute(oov_prices)["AMD"]

        threshold_strategy = VolatilityThresholdStrategy(0.5)
        threshold = threshold_strategy.compute(vocab_returns)

        strategy = WeightedAverageOOV()
        result = strategy.estimate("AMD", oov_returns, embeddings, data["t2i"],
                                   vocab_returns, threshold)

        if result is None:
            pytest.skip("No co-movement found for AMD")

        embedding, metadata = result
        assert embedding.shape == (32,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6
        assert metadata.co_movement_days > 0

        # OOV embedding should produce valid similarity scores
        for ticker, idx in data["t2i"].items():
            sim = metric.compute(embedding, embeddings[idx])
            assert -1.0 <= sim <= 1.0
