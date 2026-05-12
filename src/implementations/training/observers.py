import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...interfaces.model import ITrainingObserver

logger = logging.getLogger(__name__)


class LossCsvObserver(ITrainingObserver):
    """Appends epoch and avg_loss to a CSV file. Single responsibility: CSV loss logging."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._csv_path = os.path.join(output_dir, "loss_history.csv")

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        """Append epoch and loss to CSV."""
        os.makedirs(self._output_dir, exist_ok=True)
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "avg_loss"])
            writer.writerow([epoch, f"{avg_loss:.6f}"])

    def on_training_complete(self, final_loss: float) -> None:
        pass

    def on_checkpoint(self, epoch: int, W1: np.ndarray) -> None:
        pass

    def on_error(self, error: Exception) -> None:
        pass


class StatusFileObserver(ITrainingObserver):
    """Writes training_status.json after every epoch. Single responsibility: status file."""

    def __init__(self, output_dir: str, market: str, total_epochs: int):
        self._output_dir = output_dir
        self._market = market
        self._total_epochs = total_epochs
        self._status_path = os.path.join(output_dir, "training_status.json")
        self._started_at = None

    def _write_status(self, status_data: dict) -> None:
        """Write status dict to JSON file."""
        os.makedirs(self._output_dir, exist_ok=True)
        status_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self._status_path, "w") as f:
            json.dump(status_data, f, indent=2)

    def on_epoch_start(self, epoch: int) -> None:
        if epoch == 0:
            self._started_at = datetime.now(timezone.utc).isoformat()

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        """Update training_status.json with current progress."""
        self._write_status({
            "status": "running",
            "market": self._market,
            "current_epoch": epoch,
            "total_epochs": self._total_epochs,
            "current_loss": avg_loss,
            "started_at": self._started_at
        })

    def on_training_complete(self, final_loss: float) -> None:
        """Mark training as complete."""
        self._write_status({
            "status": "complete",
            "market": self._market,
            "total_epochs": self._total_epochs,
            "final_loss": final_loss,
            "started_at": self._started_at
        })

    def on_checkpoint(self, epoch: int, W1: np.ndarray) -> None:
        pass

    def on_error(self, error: Exception) -> None:
        """Mark training as failed."""
        self._write_status({
            "status": "failed",
            "market": self._market,
            "error": str(error),
            "started_at": self._started_at
        })


class CheckpointObserver(ITrainingObserver):
    """Saves W1 as .npy checkpoint. Single responsibility: checkpoint saving."""

    def __init__(self, output_dir: str, checkpoint_every: int):
        self._checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self._checkpoint_every = checkpoint_every

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        pass

    def on_training_complete(self, final_loss: float) -> None:
        pass

    def on_checkpoint(self, epoch: int, W1: np.ndarray) -> None:
        """Save W1 checkpoint if epoch is a checkpoint interval."""
        if epoch % self._checkpoint_every == 0:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            path = os.path.join(self._checkpoint_dir, f"W1_epoch_{epoch:04d}.npy")
            np.save(path, W1)
            logger.info("Checkpoint saved: %s", path)

    def on_error(self, error: Exception) -> None:
        pass


class DashboardNotifierObserver(ITrainingObserver):
    """Writes reload_signal.json on training completion. Single responsibility: notify dashboard."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._signal_path = os.path.join(output_dir, "reload_signal.json")

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        pass

    def on_training_complete(self, final_loss: float) -> None:
        """Write reload signal for dashboard to pick up."""
        os.makedirs(self._output_dir, exist_ok=True)
        signal = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "new_version": "latest"
        }
        with open(self._signal_path, "w") as f:
            json.dump(signal, f, indent=2)
        logger.info("Reload signal written: %s", self._signal_path)

    def on_checkpoint(self, epoch: int, W1: np.ndarray) -> None:
        pass

    def on_error(self, error: Exception) -> None:
        pass
