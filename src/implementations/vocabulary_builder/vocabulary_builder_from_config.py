import os
import pandas as pd
from ...interfaces.vocabulary_builder import IVocabularyBuilder
from ...interfaces.data_provider import IDataProvider
from ...interfaces.retrieve_config import IRetrieveConfig
from ...interfaces.pipeline import IReturnsProcessor, IThresholdStrategy, IPairGenerator
from typing import List, Dict


class VocabularyBuilderFromConfig(IVocabularyBuilder):
    """Builds vocabulary and training pairs from config-driven stock lists.

    Delegates returns computation, threshold calculation, and pair generation
    to injected strategy interfaces.
    """

    def __init__(self, retrieve_config: IRetrieveConfig = None,
                 data_provider: IDataProvider = None,
                 returns_processor: IReturnsProcessor = None,
                 threshold_strategy: IThresholdStrategy = None,
                 pair_generator: IPairGenerator = None):
        self.retrieve_config = retrieve_config
        self.data_provider = data_provider
        self.returns_processor = returns_processor
        self.threshold_strategy = threshold_strategy
        self.pair_generator = pair_generator

    def build_vocabulary(self, source: str) -> List:
        """Build vocabulary list from config source."""
        if self.retrieve_config is None:
            return None
        return self.retrieve_config.get_config(source)

    def build_training_pairs(self, source: str, start: str, end: str, output_dir: str = "data") -> Dict:
        """Build training pairs using injected pipeline strategies."""
        vocab_tickers = self.build_vocabulary(source)
        if not vocab_tickers or self.data_provider is None:
            return None

        t2i = {v: i for i, v in enumerate(vocab_tickers)}
        i2t = {i: v for i, v in enumerate(vocab_tickers)}

        result = self.data_provider.get_daily_close_prices(vocab_tickers, start, end)
        prices = result["prices"]

        if prices.empty:
            return {"pairs_file": None, "vocab_file": None, "t2i": t2i, "i2t": i2t, "failed": result["failed"], "pair_count": 0}

        returns = self.returns_processor.compute(prices)
        threshold = self.threshold_strategy.compute(returns)
        all_pairs = self.pair_generator.generate(returns, t2i, threshold)

        os.makedirs(output_dir, exist_ok=True)
        pairs_file = os.path.join(output_dir, f"training_pairs_{start}_{end}.parquet")
        vocab_file = os.path.join(output_dir, f"vocab_{start}_{end}.parquet")

        pairs_df = pd.DataFrame(all_pairs, columns=["center", "target"])
        if not pairs_df.empty:
            pairs_df.to_parquet(pairs_file, index=False)

        vocab_df = pd.DataFrame([{"index": i, "ticker": t} for t, i in t2i.items()])
        vocab_df.to_parquet(vocab_file, index=False)

        return {
            "pairs_file": pairs_file if not pairs_df.empty else None,
            "vocab_file": vocab_file,
            "t2i": t2i,
            "i2t": i2t,
            "failed": result["failed"],
            "pair_count": len(all_pairs),
            "threshold": threshold
        }
