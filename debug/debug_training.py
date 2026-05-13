"""Debug: test if training works with simple pairs vs real data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.implementations.training.word2vec import Word2VecModel
from src.implementations.training.trainer import Trainer
from src.implementations.training.clipping import GradientClipper
from src.implementations.training.lr_strategy import FixedLRStrategy
from src.implementations.training.stopping import PatienceEarlyStopper
from src.implementations.training.persistence import NumpyPersistence

print("=" * 60)
print("TEST 1: Simple fixed pairs (should learn easily)")
print("=" * 60)

model = Word2VecModel()
model.initialise(vocab_size=7, embed_dim=8, seed=42)
pairs = [(0,1),(1,0),(2,3),(3,2),(4,5),(5,4),(0,2),(2,0)]

trainer = Trainer(
    model=model,
    lr_strategy=FixedLRStrategy(0.1),
    gradient_clipper=GradientClipper(5.0),
    early_stopper=PatienceEarlyStopper(1000),
    observers=[],
    persistence=NumpyPersistence(),
)
history = trainer.train(pairs, epochs=200, checkpoint_every=1000)
print(f"  First: {history[0]:.4f}")
print(f"  Last:  {history[-1]:.4f}")
print(f"  Learned: {'YES' if history[-1] < history[0] * 0.5 else 'NO'}")

print("\n" + "=" * 60)
print("TEST 2: Real sample pairs from parquet")
print("=" * 60)

# Load real pairs
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy
from src.implementations.pipeline.pairs import CoMovementPairGenerator
from config.settings import settings

cache = ParquetCache(settings.raw_dir)
prices = cache.read("sample_11_may_2")
if prices is None:
    prices = cache.read("sample_11May")
if prices is None:
    prices = cache.read("sample")

if prices is not None:
    returns_proc = LogReturnsProcessor()
    returns = returns_proc.compute(prices)
    
    threshold_strategy = VolatilityThresholdStrategy(settings.threshold_multiplier)
    threshold = threshold_strategy.compute(returns)
    
    vocab_tickers = sorted(prices.columns.tolist())
    t2i = {t: i for i, t in enumerate(vocab_tickers)}
    
    pair_gen = CoMovementPairGenerator(max_pairs_per_day=settings.max_pairs_per_day, seed=42)
    real_pairs = pair_gen.generate(returns, t2i, threshold)
    
    print(f"  Vocab: {len(t2i)} stocks")
    print(f"  Pairs: {len(real_pairs)}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Unique pairs: {len(set(real_pairs))}")
    
    # Check pair distribution
    from collections import Counter
    pair_counts = Counter(real_pairs)
    print(f"  Most common pair: {pair_counts.most_common(1)}")
    print(f"  Least common pair: {pair_counts.most_common()[-1]}")
    
    # Train
    model2 = Word2VecModel()
    model2.initialise(vocab_size=len(t2i), embed_dim=8, seed=42)
    
    trainer2 = Trainer(
        model=model2,
        lr_strategy=FixedLRStrategy(0.1),
        gradient_clipper=GradientClipper(5.0),
        early_stopper=PatienceEarlyStopper(1000),
        observers=[],
        persistence=NumpyPersistence(),
    )
    history2 = trainer2.train(real_pairs, epochs=50, checkpoint_every=1000)
    print(f"  First: {history2[0]:.4f}")
    print(f"  Last:  {history2[-1]:.4f}")
    print(f"  Learned: {'YES' if history2[-1] < history2[0] * 0.9 else 'NO'}")
    
    # Test without clipping
    print("\n" + "=" * 60)
    print("TEST 3: Real pairs WITHOUT gradient clipping")
    print("=" * 60)
    
    model3 = Word2VecModel()
    model3.initialise(vocab_size=len(t2i), embed_dim=8, seed=42)
    
    # Manual training loop without clipping
    lr = 0.1
    for epoch in range(50):
        indices = np.random.permutation(len(real_pairs))
        total_loss = 0
        for idx in indices:
            c, t = real_pairs[idx]
            h, s, probs = model3.forward(c)
            loss = -np.log(probs[t] + 1e-9)
            total_loss += loss
            dW2, dh = model3.backward(h, probs, t)
            model3.update(c, dW2, dh, lr)
        avg = total_loss / len(real_pairs)
        if epoch == 0:
            print(f"  Epoch 0: {avg:.4f}")
        if epoch == 49:
            print(f"  Epoch 49: {avg:.4f}")
    print(f"  Learned: {'YES' if avg < 1.9 else 'NO'}")
else:
    print("  No cached prices found")
