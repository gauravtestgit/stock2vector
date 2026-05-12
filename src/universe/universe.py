"""Universe manager — builds stock universes from user-selected config files.

Users choose which stock files and anchor groups to include.
No hardcoded market definitions.
"""
import logging
from typing import Dict, List

import yaml
from pathlib import Path

from .scrapers import TickerLoader

logger = logging.getLogger(__name__)


class UniverseManager:
    """Builds stock universes from config files and anchor groups.

    Usage:
        manager = UniverseManager()

        # Build from one or more stock files
        tickers = manager.build_universe(["sp_500.csv", "nasdaq_100.csv"])

        # Build with anchors
        tickers = manager.build_universe(
            stock_files=["sp_500.csv"],
            anchor_groups=["sector_etfs", "macro"]
        )

        # List available files and anchor groups
        manager.list_stock_files()
        manager.list_anchor_groups()
    """

    def __init__(self, universe_config_path: str = None, vocab_dir: str = None):
        if universe_config_path is None:
            universe_config_path = str(Path(__file__).parent.parent.parent / "config" / "universe.yaml")
        if vocab_dir is None:
            vocab_dir = str(Path(__file__).parent.parent.parent / "config" / "vocabulary" / "stocks")

        with open(universe_config_path, "r") as f:
            self._config = yaml.safe_load(f)
        self._loader = TickerLoader(universe_config_path, vocab_dir)
        self._vocab_dir = Path(vocab_dir)

    def build_universe(self, stock_files: List[str],
                       anchor_groups: List[str] = None) -> List[str]:
        """Build a universe from stock files and optional anchor groups.

        Args:
            stock_files: list of filenames in config/vocabulary/stocks/
                         e.g., ["sp_500.csv", "nasdaq_100.csv", "sample.yaml"]
            anchor_groups: optional list of anchor group names from universe.yaml
                          e.g., ["sector_etfs", "macro", "thematic"]

        Returns:
            Sorted, deduplicated list of ticker strings.
        """
        tickers = set()

        for filename in stock_files:
            loaded = self._load_file(filename)
            tickers.update(loaded)
            logger.info("Loaded %d tickers from %s", len(loaded), filename)

        if anchor_groups:
            anchors = self.get_anchors(anchor_groups)
            tickers.update(anchors)
            logger.info("Added %d anchor tickers from %s", len(anchors), anchor_groups)

        result = sorted(tickers)
        logger.info("Universe built: %d unique tickers", len(result))
        return result

    def _load_file(self, filename: str) -> List[str]:
        """Load tickers from a file, detecting format by extension."""
        if filename.endswith(".csv"):
            return self._loader._load_csv_tickers(filename)
        elif filename.endswith(".yaml") or filename.endswith(".yml"):
            return self._loader._load_yaml_tickers(filename)
        else:
            logger.warning("Unknown file format: %s", filename)
            return []

    def get_anchors(self, groups: List[str]) -> List[str]:
        """Get anchor tickers from specified groups.

        Groups are looked up across all anchor sections in universe.yaml
        (us_tier_a_anchors, asx_anchors). If the same group name exists in
        multiple sections, tickers from ALL sections are included.
        """
        tickers = []
        for section in ["us_tier_a_anchors", "asx_anchors"]:
            section_data = self._config.get(section, {})
            for group in groups:
                if group in section_data:
                    tickers.extend(section_data[group].keys())

        if not tickers:
            for group in groups:
                logger.warning("Unknown anchor group: '%s'", group)

        return tickers

    def get_anchor_descriptions(self, groups: List[str]) -> Dict[str, str]:
        """Get ticker → description mapping for specified anchor groups."""
        descriptions = {}
        for section in ["us_tier_a_anchors", "asx_anchors"]:
            section_data = self._config.get(section, {})
            for group in groups:
                if group in section_data:
                    descriptions.update(section_data[group])
        return descriptions

    def get_nzx_stocks(self) -> List[str]:
        """Get hardcoded NZX stock list from universe.yaml."""
        return self._loader.load_nzx()

    def list_stock_files(self) -> List[str]:
        """List available stock files in the vocab directory."""
        files = []
        for ext in ["*.csv", "*.yaml", "*.yml"]:
            files.extend([f.name for f in self._vocab_dir.glob(ext)])
        return sorted(files)

    def list_anchor_groups(self) -> List[str]:
        """List available anchor group names from universe.yaml."""
        groups = []
        for section in ["us_tier_a_anchors", "asx_anchors"]:
            groups.extend(self._config.get(section, {}).keys())
        return groups
