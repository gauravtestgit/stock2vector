import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator
from src.implementations.training.word2vec import Word2VecModel

cache = ParquetCache('data/raw')
prices = cache.read('nasdaq_100_11_may_3')
returns = LogReturnsProcessor().compute(prices)
threshold = VolatilityThresholdStrategy(2.0).compute(returns)
t2i = {t: i for i, t in enumerate(sorted(prices.columns))}
pairs = CoMovementPairGenerator(max_pairs_per_day=10000, seed=42).generate(returns, t2i, threshold)

print(f'Pairs: {len(pairs)}, Unique: {len(set(pairs))}')
print(f'Vocab: {len(t2i)}')
print()

for lr in [0.1, 0.5, 1.0, 2.0, 5.0]:
    model = Word2VecModel()
    model.initialise(vocab_size=len(t2i), embed_dim=32, seed=42)
    
    first_loss = 0
    last_loss = 0
    for epoch in range(20):
        indices = np.random.permutation(len(pairs))
        total_loss = 0
        for idx in indices:
            c, t = pairs[idx]
            h, s, probs = model.forward(c)
            total_loss += -np.log(probs[t] + 1e-9)
            dW2, dh = model.backward(h, probs, t)
            model.update(c, dW2, dh, lr)
        avg = total_loss / len(pairs)
        if epoch == 0:
            first_loss = avg
        last_loss = avg
    print(f'LR={lr:<4}: {first_loss:.4f} -> {last_loss:.4f}  (delta={first_loss-last_loss:.4f})')
