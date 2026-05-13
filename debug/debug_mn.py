import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import Counter
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.returns import MarketNeutralReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator
from src.implementations.training.word2vec import Word2VecModel

# Find the latest cache with anchors
cache = ParquetCache('data/raw')
for name in ['nasdaq_100_11_may_14', 'nasdaq_100_11_may_13', 'nasdaq_100_11_may_12']:
    prices = cache.read(name)
    if prices is not None:
        print(f"Using cache: {name}, shape: {prices.shape}")
        break

# Market neutral returns
returns_proc = MarketNeutralReturnsProcessor("SPY")
returns = returns_proc.compute(prices)
print(f"Returns shape: {returns.shape}")
print(f"SPY in returns: {'SPY' in returns.columns}")

threshold = VolatilityThresholdStrategy(1.0).compute(returns)
print(f"Threshold: {threshold:.6f}")

t2i = {t: i for i, t in enumerate(sorted(returns.columns))}
pairs = CoMovementPairGenerator(max_pairs_per_day=10000, seed=42).generate(returns, t2i, threshold)

print(f"Total pairs: {len(pairs)}")
print(f"Unique pairs: {len(set(pairs))}")
print(f"Max possible: {len(t2i) * (len(t2i)-1)}")
print(f"Coverage: {len(set(pairs)) / (len(t2i) * (len(t2i)-1)) * 100:.1f}%")

pair_counts = Counter(pairs)
counts = list(pair_counts.values())
print(f"Min freq: {min(counts)}, Max freq: {max(counts)}, Mean: {np.mean(counts):.1f}")

# Quick training test - raw loop, no clipping, subsample pairs
subsample = pairs[:2000]  # Use first 2000 pairs for speed
print(f"\nTraining test (50 epochs, LR=0.1, 5 negatives, {len(subsample)} pairs):")
model = Word2VecModel(mode="negative_sampling", num_negatives=5)
model.initialise(vocab_size=len(t2i), embed_dim=32, seed=42)

for epoch in range(50):
    indices = np.random.permutation(len(subsample))
    total_loss = 0
    for idx in indices:
        c, t = subsample[idx]
        h, _, probs = model.forward(c)
        pos_score = h @ model._W2[:, t]
        sig = 1.0 / (1.0 + np.exp(-np.clip(pos_score, -10, 10)))
        total_loss += -np.log(sig + 1e-9)
        dW2, dh = model.backward(h, probs, t)
        model.update(c, dW2, dh, 0.1)
    avg = total_loss / len(subsample)
    if epoch in [0, 1, 5, 10, 25, 49]:
        print(f"  Epoch {epoch}: loss={avg:.4f}")
