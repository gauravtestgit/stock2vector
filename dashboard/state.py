"""Shared model state for the dashboard. Loaded once at startup."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric

MODEL_DIR = os.environ.get("STOCK2VEC_MODEL", "data/models/nasdaq_100/current")

persistence = NumpyPersistence()
embeddings, vocab, metadata = persistence.load(MODEL_DIR)
t2i = vocab
i2t = {int(i): t for t, i in vocab.items()}
metric = CosineMetric()
