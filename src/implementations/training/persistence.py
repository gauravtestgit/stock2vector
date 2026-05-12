import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from ...interfaces.model import IModelPersistence, IEmbeddingModel

logger = logging.getLogger(__name__)


class NumpyPersistence(IModelPersistence):
    """Saves and loads model using numpy .npy and JSON files."""

    def save(self, model: IEmbeddingModel, vocab: dict, metadata: dict, path: Path) -> None:
        """Save W1.npy, vocab.json, metadata.json to path directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "W1.npy", model.get_embeddings().astype(np.float32))

        with open(path / "vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> Tuple[np.ndarray, dict, dict]:
        """Load W1.npy, vocab.json, metadata.json from path directory."""
        path = Path(path)

        embeddings = np.load(path / "W1.npy")

        with open(path / "vocab.json", "r") as f:
            vocab = json.load(f)

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        logger.info("Model loaded from %s", path)
        return embeddings, vocab, metadata
