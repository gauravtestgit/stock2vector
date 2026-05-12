import pytest
import os
import numpy as np
import pandas as pd
from src.implementations.training.word2vec import Word2VecModel
from src.implementations.training.trainer import Trainer
from src.implementations.training.clipping import GradientClipper
from src.implementations.training.lr_strategy import FixedLRStrategy
from src.implementations.training.stopping import PatienceEarlyStopper
from src.implementations.training.observers import LossCsvObserver, StatusFileObserver, CheckpointObserver, DashboardNotifierObserver
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.vocabulary_builder.vocabulary_builder_from_config import VocabularyBuilderFromConfig
from src.implementations.config_providers.vocab_stocks_yaml_config_provider import VocabStocksYAMLConfigProvider
from src.implementations.data_providers.yahoo_provider import YahooFinanceProvider
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import FixedThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator


@pytest.fixture
def model():
    m = Word2VecModel()
    m.initialise(vocab_size=7, embed_dim=8, seed=42)
    return m


@pytest.fixture
def sample_pairs():
    return [(0, 1), (1, 0), (0, 2), (2, 0), (3, 4), (4, 3)]


@pytest.fixture
def builder():
    return VocabularyBuilderFromConfig(
        retrieve_config=VocabStocksYAMLConfigProvider(),
        data_provider=YahooFinanceProvider(),
        returns_processor=LogReturnsProcessor(),
        threshold_strategy=FixedThresholdStrategy(0.005),
        pair_generator=CoMovementPairGenerator()
    )


def _build_training_data(builder, source, tmp_path):
    result = builder.build_training_pairs(
        source, start="2026-04-01", end="2026-04-13",
        output_dir=str(tmp_path)
    )
    pairs_df = pd.read_parquet(result["pairs_file"])
    all_pairs = list(zip(pairs_df["center"], pairs_df["target"]))
    return {"pairs": all_pairs, "t2i": result["t2i"], "i2t": result["i2t"], "source": source}


@pytest.fixture
def real_training_data(builder, tmp_path):
    return _build_training_data(builder, "config/vocabulary/stocks/sample.yaml", tmp_path)


class TestInitialise:
    def test_weight_shapes(self, model):
        emb = model.get_embeddings()
        assert emb.shape == (7, 8)
        assert model._W2.shape == (8, 7)

    def test_weights_near_zero(self, model):
        emb = model.get_embeddings()
        assert np.abs(emb).max() < 0.1

    def test_seed_reproducibility(self):
        m1 = Word2VecModel()
        m1.initialise(7, 8, seed=42)
        m2 = Word2VecModel()
        m2.initialise(7, 8, seed=42)
        np.testing.assert_array_equal(m1.get_embeddings(), m2.get_embeddings())


class TestForward:
    def test_returns_three_arrays(self, model):
        h, scores, probs = model.forward(0)
        assert isinstance(h, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(probs, np.ndarray)

    def test_h_shape(self, model):
        h, _, _ = model.forward(0)
        assert h.shape == (8,)

    def test_probs_sum_to_one(self, model):
        _, _, probs = model.forward(0)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_probs_all_positive(self, model):
        _, _, probs = model.forward(0)
        assert (probs > 0).all()


class TestBackward:
    def test_returns_two_arrays(self, model):
        h, _, probs = model.forward(0)
        dW2, dh = model.backward(h, probs, target_idx=1)
        assert isinstance(dW2, np.ndarray)
        assert isinstance(dh, np.ndarray)

    def test_dW2_shape(self, model):
        h, _, probs = model.forward(0)
        dW2, _ = model.backward(h, probs, target_idx=1)
        assert dW2.shape == (8, 7)

    def test_dh_shape(self, model):
        h, _, probs = model.forward(0)
        _, dh = model.backward(h, probs, target_idx=1)
        assert dh.shape == (8,)

    def test_critical_subtract_not_assign(self, model):
        """CRITICAL: dscores[target_idx] -= 1, never = -1."""
        h, _, probs = model.forward(0)
        target_idx = 1
        prob_at_target = probs[target_idx]
        dW2, _ = model.backward(h, probs, target_idx)
        expected_dscore_at_target = prob_at_target - 1.0
        actual_dscore_at_target = dW2[:, target_idx] / h
        np.testing.assert_allclose(actual_dscore_at_target[0], expected_dscore_at_target, atol=1e-7)

    def test_probs_not_mutated(self, model):
        h, _, probs = model.forward(0)
        probs_copy = probs.copy()
        model.backward(h, probs, target_idx=1)
        np.testing.assert_array_equal(probs, probs_copy)


class TestUpdate:
    def test_weights_change_after_update(self, model):
        emb_before = model.get_embeddings()
        h, _, probs = model.forward(0)
        dW2, dh = model.backward(h, probs, target_idx=1)
        model.update(0, dW2, dh, lr=0.01)
        emb_after = model.get_embeddings()
        assert not np.array_equal(emb_before[0], emb_after[0])

    def test_only_center_row_changes(self, model):
        emb_before = model.get_embeddings()
        h, _, probs = model.forward(0)
        dW2, dh = model.backward(h, probs, target_idx=1)
        model.update(0, dW2, dh, lr=0.01)
        emb_after = model.get_embeddings()
        np.testing.assert_array_equal(emb_before[1:], emb_after[1:])


class TestTrainingLoopUnit:
    def test_loss_decreases_manual_loop(self, model, sample_pairs):
        first_epoch_loss = 0
        last_epoch_loss = 0
        for epoch in range(200):
            total_loss = 0
            for center_idx, target_idx in sample_pairs:
                h, scores, probs = model.forward(center_idx)
                loss = -np.log(probs[target_idx] + 1e-9)
                total_loss += loss
                dW2, dh = model.backward(h, probs, target_idx)
                model.update(center_idx, dW2, dh, lr=0.01)
            avg_loss = total_loss / len(sample_pairs)
            if epoch == 0:
                first_epoch_loss = avg_loss
            if epoch == 199:
                last_epoch_loss = avg_loss
        assert last_epoch_loss < first_epoch_loss

    def test_loss_not_used_in_update(self, model):
        h, scores, probs = model.forward(0)
        dW2, dh = model.backward(h, probs, target_idx=1)
        model.update(0, dW2, dh, lr=0.01)


class TestTrainer:
    """Tests for the Trainer orchestrator with injected dependencies."""

    def test_trainer_loss_decreases(self, sample_pairs, tmp_path):
        model = Word2VecModel()
        model.initialise(vocab_size=7, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[LossCsvObserver(str(tmp_path))],
            persistence=NumpyPersistence()
        )

        loss_history = trainer.train(sample_pairs, epochs=200)
        assert loss_history[-1] < loss_history[0]

    def test_trainer_writes_loss_csv(self, sample_pairs, tmp_path):
        model = Word2VecModel()
        model.initialise(vocab_size=7, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[LossCsvObserver(str(tmp_path))],
            persistence=NumpyPersistence()
        )

        trainer.train(sample_pairs, epochs=10)
        assert os.path.exists(tmp_path / "loss_history.csv")

    def test_trainer_writes_status_file(self, sample_pairs, tmp_path):
        model = Word2VecModel()
        model.initialise(vocab_size=7, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[StatusFileObserver(str(tmp_path), market="test", total_epochs=10)],
            persistence=NumpyPersistence()
        )

        trainer.train(sample_pairs, epochs=10)
        status_path = tmp_path / "training_status.json"
        assert os.path.exists(status_path)
        import json
        with open(status_path) as f:
            status = json.load(f)
        assert status["status"] == "complete"

    def test_trainer_writes_checkpoints(self, sample_pairs, tmp_path):
        model = Word2VecModel()
        model.initialise(vocab_size=7, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[CheckpointObserver(str(tmp_path), checkpoint_every=5)],
            persistence=NumpyPersistence()
        )

        trainer.train(sample_pairs, epochs=10, checkpoint_every=5)
        assert os.path.exists(tmp_path / "checkpoints" / "W1_epoch_0000.npy")
        assert os.path.exists(tmp_path / "checkpoints" / "W1_epoch_0005.npy")

    def test_trainer_writes_reload_signal(self, sample_pairs, tmp_path):
        model = Word2VecModel()
        model.initialise(vocab_size=7, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[DashboardNotifierObserver(str(tmp_path))],
            persistence=NumpyPersistence()
        )

        trainer.train(sample_pairs, epochs=10)
        assert os.path.exists(tmp_path / "reload_signal.json")

    def test_trainer_early_stops(self):
        stopper = PatienceEarlyStopper(patience=3)
        # Loss increasing for 3 consecutive epochs
        assert stopper.should_stop([1.0, 1.1, 1.2, 1.3]) is True
        # Loss not consistently increasing
        assert stopper.should_stop([1.0, 0.9, 1.1, 1.0]) is False
        # Not enough history
        assert stopper.should_stop([1.0, 1.1]) is False

    def test_trainer_gradient_clipping(self):
        clipper = GradientClipper(1.0)
        grad = np.array([2.0, -3.0, 0.5, -0.1])
        clipped = clipper.clip(grad)
        np.testing.assert_array_equal(clipped, [1.0, -1.0, 0.5, -0.1])

    def test_trainer_all_observers(self, sample_pairs, tmp_path):
        model = Word2VecModel()
        model.initialise(vocab_size=7, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[
                LossCsvObserver(str(tmp_path)),
                StatusFileObserver(str(tmp_path), market="test", total_epochs=50),
                CheckpointObserver(str(tmp_path), checkpoint_every=10),
                DashboardNotifierObserver(str(tmp_path)),
            ],
            persistence=NumpyPersistence()
        )

        loss_history = trainer.train(sample_pairs, epochs=50, checkpoint_every=10)
        assert loss_history[-1] < loss_history[0]
        assert os.path.exists(tmp_path / "loss_history.csv")
        assert os.path.exists(tmp_path / "training_status.json")
        assert os.path.exists(tmp_path / "reload_signal.json")
        assert os.path.exists(tmp_path / "checkpoints" / "W1_epoch_0000.npy")

    def test_persistence_save_load(self, model, tmp_path):
        persistence = NumpyPersistence()
        vocab = {"AAPL": 0, "MSFT": 1}
        metadata = {"embed_dim": 8, "epochs": 100}

        persistence.save(model, vocab, metadata, tmp_path / "model")
        assert os.path.exists(tmp_path / "model" / "W1.npy")
        assert os.path.exists(tmp_path / "model" / "vocab.json")
        assert os.path.exists(tmp_path / "model" / "metadata.json")

        emb, loaded_vocab, loaded_meta = persistence.load(tmp_path / "model")
        np.testing.assert_array_almost_equal(emb, model.get_embeddings(), decimal=5)
        assert loaded_vocab == vocab
        assert loaded_meta == metadata


class TestTrainingLoopIntegration:
    """Integration tests using Trainer with real training pairs from Yahoo Finance."""

    def test_trainer_loss_decreases_with_real_pairs(self, real_training_data, tmp_path):
        data = real_training_data
        vocab_size = len(data["t2i"])
        pairs = data["pairs"]

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[LossCsvObserver(str(tmp_path))],
            persistence=NumpyPersistence()
        )

        loss_history = trainer.train(pairs, epochs=100)
        assert loss_history[-1] < loss_history[0]

    def test_trainer_embeddings_shape(self, real_training_data, tmp_path):
        data = real_training_data
        vocab_size = len(data["t2i"])

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=8, seed=42)

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=NumpyPersistence()
        )

        trainer.train(data["pairs"], epochs=10)
        assert model.get_embeddings().shape == (vocab_size, 8)

    def test_trainer_similar_stocks_higher_cosine(self, real_training_data, tmp_path):
        """After training via Trainer, co-moving stocks should have more similar embeddings."""
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
        emb = model.get_embeddings()

        def cosine_sim(a, b):
            dot = np.dot(a, b)
            norms = np.linalg.norm(a) * np.linalg.norm(b)
            return dot / norms if norms > 0 else 0.0

        from collections import Counter
        pair_counts = Counter(pairs)
        most_common_pair = pair_counts.most_common(1)[0][0]
        a_idx, b_idx = most_common_pair

        paired_with_a = {p[1] for p in pairs if p[0] == a_idx}
        unpaired = [i for i in range(vocab_size) if i != a_idx and i not in paired_with_a]

        if unpaired:
            sim_paired = cosine_sim(emb[a_idx], emb[b_idx])
            sim_unpaired = cosine_sim(emb[a_idx], emb[unpaired[0]])
            assert sim_paired > sim_unpaired

    def test_trainer_save_and_load_model(self, real_training_data, tmp_path):
        data = real_training_data
        vocab_size = len(data["t2i"])

        model = Word2VecModel()
        model.initialise(vocab_size=vocab_size, embed_dim=8, seed=42)
        persistence = NumpyPersistence()

        trainer = Trainer(
            model=model,
            lr_strategy=FixedLRStrategy(0.01),
            gradient_clipper=GradientClipper(1.0),
            early_stopper=PatienceEarlyStopper(20),
            observers=[],
            persistence=persistence
        )

        trainer.train(data["pairs"], epochs=50)
        trained_emb = model.get_embeddings()

        persistence.save(model, data["t2i"], {"epochs": 50}, tmp_path / "model")
        loaded_emb, loaded_vocab, loaded_meta = persistence.load(tmp_path / "model")

        np.testing.assert_array_almost_equal(loaded_emb, trained_emb, decimal=5)
        assert loaded_vocab == data["t2i"]
        assert loaded_meta["epochs"] == 50



class TestGetEmbeddings:
    def test_returns_copy(self, model):
        emb1 = model.get_embeddings()
        emb2 = model.get_embeddings()
        assert emb1 is not emb2
        np.testing.assert_array_equal(emb1, emb2)

    def test_modifying_copy_doesnt_affect_model(self, model):
        emb = model.get_embeddings()
        emb[0] = 999.0
        original = model.get_embeddings()
        assert original[0][0] != 999.0
