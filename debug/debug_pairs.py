import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from collections import Counter
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator
from config.settings import settings

cache = ParquetCache(settings.raw_dir)

# Try to find the cached prices
for name in ['nasdaq_100_11_may_3', 'nasdaq_100_11_may_2', 'nasdaq_100_11_may', 'nasdaq_100']:
    prices = cache.read(name)
    if prices is not None:
        print(f"Found cache: {name}")
        break

print(f"Prices shape: {prices.shape}")
print(f"Trading days: {len(prices)}")

returns = LogReturnsProcessor().compute(prices)
threshold = VolatilityThresholdStrategy(0.5).compute(returns)
print(f"Threshold: {threshold:.6f}")

t2i = {t: i for i, t in enumerate(sorted(prices.columns))}
pairs = CoMovementPairGenerator(max_pairs_per_day=10000, seed=42).generate(returns, t2i, threshold)

print(f"\nTotal pairs: {len(pairs)}")
print(f"Unique pairs: {len(set(pairs))}")
print(f"Max possible unique pairs: {len(t2i) * (len(t2i)-1)}")
print(f"Coverage: {len(set(pairs)) / (len(t2i) * (len(t2i)-1)) * 100:.1f}%")

# How many stocks move per day?
up_counts = []
down_counts = []
for date, row in returns.iterrows():
    up = sum(1 for t in row.index if row[t] > threshold and not np.isnan(row[t]))
    down = sum(1 for t in row.index if row[t] < -threshold and not np.isnan(row[t]))
    up_counts.append(up)
    down_counts.append(down)

print(f"\nAvg stocks UP per day: {np.mean(up_counts):.1f}")
print(f"Avg stocks DOWN per day: {np.mean(down_counts):.1f}")
print(f"Avg total moving: {np.mean(up_counts) + np.mean(down_counts):.1f} / {len(t2i)}")

# Pair distribution
pair_counts = Counter(pairs)
counts = list(pair_counts.values())
print(f"\nPair frequency stats:")
print(f"  Min occurrences: {min(counts)}")
print(f"  Max occurrences: {max(counts)}")
print(f"  Mean occurrences: {np.mean(counts):.1f}")
print(f"  Median occurrences: {np.median(counts):.1f}")
print(f"  Pairs appearing 1-5 times: {sum(1 for c in counts if c <= 5)}")
print(f"  Pairs appearing 50+ times: {sum(1 for c in counts if c >= 50)}")

print(f"\nMost common pairs:")
for pair, count in pair_counts.most_common(10):
    i2t = {i: t for t, i in t2i.items()}
    print(f"  {i2t[pair[0]]:6s} - {i2t[pair[1]]:6s}: {count}")
