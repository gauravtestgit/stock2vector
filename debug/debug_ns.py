import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.implementations.training.word2vec import Word2VecModel
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator
from src.implementations.embeddings.metrics import CosineMetric

cache = ParquetCache('data/raw')
prices = cache.read('nasdaq_100_11_may_3')
returns = LogReturnsProcessor().compute(prices)
threshold = VolatilityThresholdStrategy(2.0).compute(returns)
t2i = {t: i for i, t in enumerate(sorted(prices.columns))}
i2t = {i: t for t, i in t2i.items()}
pairs = CoMovementPairGenerator(max_pairs_per_day=10000, seed=42).generate(returns, t2i, threshold)

print(f"Pairs: {len(pairs)}, Unique: {len(set(pairs))}, Vocab: {len(t2i)}")
print()

# Test negative sampling with different LRs
for lr in [0.1, 0.5, 1.0, 2.0]:
    model = Word2VecModel(mode="negative_sampling", num_negatives=10)
    model.initialise(vocab_size=len(t2i), embed_dim=32, seed=42)
    
    # Check gradient magnitude on first pair
    c, t = pairs[0]
    h, _, probs = model.forward(c)
    dW2, dh = model.backward(h, probs, t)
    print(f"LR={lr}: dW2 max={np.abs(dW2).max():.6f}, dh max={np.abs(dh).max():.6f}")
    
    # Train 20 epochs, track softmax loss
    losses = []
    for epoch in range(20):
        indices = np.random.permutation(len(pairs))
        total_loss = 0
        for idx in indices:
            c, t = pairs[idx]
            h, _, probs = model.forward(c)
            total_loss += -np.log(probs[t] + 1e-9)
            dW2, dh = model.backward(h, probs, t)
            model.update(c, dW2, dh, lr)
        losses.append(total_loss / len(pairs))
    print(f"  Softmax loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    # Also check the negative sampling loss (the actual objective)
    ns_loss = 0
    for idx in range(min(1000, len(pairs))):
        c, t = pairs[idx]
        h = model._W1[c]
        # Positive
        score_pos = h @ model._W2[:, t]
        sig_pos = 1 / (1 + np.exp(-np.clip(score_pos, -10, 10)))
        ns_loss += -np.log(sig_pos + 1e-9)
        # Negatives (just check a few)
        for _ in range(5):
            neg = np.random.randint(len(t2i))
            if neg != t:
                score_neg = h @ model._W2[:, neg]
                sig_neg = 1 / (1 + np.exp(-np.clip(score_neg, -10, 10)))
                ns_loss += -np.log(1 - sig_neg + 1e-9)
    print(f"  NS loss (sample): {ns_loss/1000:.4f}")
    
    # Check cosine similarity of a known pair
    emb = model.get_embeddings()
    metric = CosineMetric()
    # Most common pair from debug_pairs.py: STX-WDC (indices may differ)
    if "STX" in t2i and "WDC" in t2i:
        sim = metric.compute(emb[t2i["STX"]], emb[t2i["WDC"]])
        # Random pair for comparison
        sim_rand = metric.compute(emb[t2i["AAPL"]], emb[t2i["XEL"]])
        print(f"  Cosine STX-WDC: {sim:.4f}, AAPL-XEL: {sim_rand:.4f}")
    print()
