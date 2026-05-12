import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric

p = NumpyPersistence()
emb, vocab, meta = p.load('data/models/nasdaq_100_11_may_12/current')
t2i = vocab
i2t = {int(i): t for t, i in vocab.items()}
m = CosineMetric()

print("Sector ETF top 5 similar:")
print()
for etf in ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU']:
    if etf in t2i:
        similar = m.most_similar(etf, t2i, i2t, emb, top_n=5)
        result = '  '.join([f'{t}({s:.3f})' for t, s in similar])
        print(f"  {etf:6s} -> {result}")

print()
print("NVDA vs each sector ETF:")
for etf in ['XLK','XLE','XLF','XLV','XLI','XLY','XLP','XLU','XLB','XLRE','XLC']:
    if etf in t2i:
        sim = m.compute(emb[t2i['NVDA']], emb[t2i[etf]])
        print(f"  NVDA vs {etf:5s}: {sim:+.4f}")

print()
print("AAPL vs each sector ETF:")
for etf in ['XLK','XLE','XLF','XLV','XLI','XLY','XLP','XLU','XLB','XLRE','XLC']:
    if etf in t2i:
        sim = m.compute(emb[t2i['AAPL']], emb[t2i[etf]])
        print(f"  AAPL vs {etf:5s}: {sim:+.4f}")
