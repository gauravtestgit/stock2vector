"""Training status file management."""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TrainingStatus:
    """Manages training_status.json and reload_signal.json files."""

    def __init__(self, models_dir: str):
        self._models_dir = models_dir

    def _status_path(self, market: str) -> str:
        return os.path.join(self._models_dir, market, "training_status.json")

    def _reload_path(self, market: str) -> str:
        return os.path.join(self._models_dir, market, "reload_signal.json")

    def write_status(self, market: str, mode: str, epoch: int,
                     total_epochs: int, loss: float, started_at: str) -> None:
        """Write running status during training."""
        self._write(self._status_path(market), {
            "status": "running",
            "mode": mode,
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "current_loss": loss,
            "started_at": started_at,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    def write_complete(self, market: str, final_loss: float,
                       validation_results: Dict = None) -> None:
        """Mark training as complete."""
        self._write(self._status_path(market), {
            "status": "complete",
            "final_loss": final_loss,
            "validation_results": validation_results,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    def write_failed(self, market: str, error_message: str) -> None:
        """Mark training as failed."""
        self._write(self._status_path(market), {
            "status": "failed",
            "error": error_message,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    def read_status(self, market: str) -> Optional[Dict]:
        """Read current training status."""
        path = self._status_path(market)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def write_reload_signal(self, market: str, version: str) -> None:
        """Write reload signal for dashboard to pick up."""
        self._write(self._reload_path(market), {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "new_version": version,
        })

    def check_reload_signal(self, market: str) -> bool:
        """Check if a reload signal exists."""
        return os.path.exists(self._reload_path(market))

    def _write(self, path: str, data: Dict) -> None:
        """Write JSON data to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
