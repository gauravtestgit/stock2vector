import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator

cache = ParquetCache('data/raw')
prices = cache.read('nasdaq_100_11_may_3')
returns = LogReturnsProcessor().compute(prices)

print(f"{'Mult':>5} {'Threshold':>10} {'Pairs':>8} {'Unique':>7} {'Coverage':>9} {'Avg UP/day':>11}")
print("-" * 60)

for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    threshold = VolatilityThresholdStrategy(mult).compute(returns)
    t2i = {t: i for i, t in enumerate(sorted(prices.columns))}
    pairs = CoMovementPairGenerator(max_pairs_per_day=10000, seed=42).generate(returns, t2i, threshold)
    unique = len(set(pairs))
    max_possible = len(t2i) * (len(t2i) - 1)

    up_per_day = []
    down_per_day = []
    for date, row in returns.iterrows():
        up = sum(1 for t in row.index if row[t] > threshold and not np.isnan(row[t]))
        down = sum(1 for t in row.index if row[t] < -threshold and not np.isnan(row[t]))
        up_per_day.append(up)
        down_per_day.append(down)

    print(f"{mult:>5.1f} {threshold:>10.5f} {len(pairs):>8,} {unique:>5}/{max_possible:<5} {unique/max_possible*100:>7.0f}%  {np.mean(up_per_day):>5.1f}/{np.mean(down_per_day):.1f}")
