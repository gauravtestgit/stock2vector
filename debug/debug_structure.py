import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric
import glob

# Find the latest model (pass as arg or edit here)
model_dir = sys.argv[1] if len(sys.argv) > 1 else "data/models/nasdaq_100_12_may_4/current"
print(f"Model: {model_dir}")

p = NumpyPersistence()
emb, vocab, meta = p.load(model_dir)
t2i = vocab
i2t = {int(i): t for t, i in vocab.items()}
m = CosineMetric()

print(f"Vocab: {len(t2i)}, Loss: {meta.get('final_loss', '?')}")
print()

# Check key relationships
print("=== All stocks top 3 similar ===")
for ticker in sorted(t2i.keys()):
    if ticker not in ['SPY','QQQ','IWM','DIA','VTI']:
        similar = m.most_similar(ticker, t2i, i2t, emb, top_n=3)
        result = '  '.join([f'{t}({sc:.3f})' for t, sc in similar])
        print(f"  {ticker:6s} -> {result}")

print()
print("=== Sector checks ===")
sectors = {
    "Semiconductors": ["NVDA", "AMD", "AVGO", "MRVL", "QCOM", "TXN"],
    "Software/SaaS": ["ADBE", "INTU", "ADSK", "TEAM", "CRWD", "ZS"],
    "Consumer Staples": ["PEP", "MDLZ", "KHC", "MNST", "KDP", "WMT"],
    "Cybersecurity": ["PANW", "CRWD", "FTNT", "ZS"],
    "Industrials/Broad": ["XLI", "VTI", "XLF", "DIA"],
    "Energy": ["BKR", "FANG", "XLE"],
}

for sector_name, tickers in sectors.items():
    present = [t for t in tickers if t in t2i]
    if len(present) < 2:
        continue
    print(f"{sector_name}: {', '.join(present)}")
    for i in range(len(present)):
        for j in range(i+1, len(present)):
            a, b = present[i], present[j]
            sim = m.compute(emb[t2i[a]], emb[t2i[b]])
            print(f"  {a}-{b}: {sim:.3f}")

print()
print("=== Cross-sector (should be LOW) ===")
cross_pairs = [
    ("NVDA", "PEP"), ("NVDA", "BKR"), ("NVDA", "WMT"),
    ("ADBE", "FANG"), ("PEP", "AMD"), ("TXN", "MDLZ"),
]
for a, b in cross_pairs:
    if a in t2i and b in t2i:
        sim = m.compute(emb[t2i[a]], emb[t2i[b]])
        print(f"  {a}-{b}: {sim:.3f}")
