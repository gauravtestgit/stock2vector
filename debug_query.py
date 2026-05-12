import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric

persistence = NumpyPersistence()
embeddings, vocab, metadata = persistence.load('data/models/nasdaq_100_11_may_11/current')
t2i = vocab
i2t = {int(i): t for t, i in vocab.items()}
metric = CosineMetric()

print(f"Vocab: {len(t2i)}, Epochs: {metadata.get('epochs')}, Loss: {metadata.get('final_loss'):.4f}")
print(f"Anchors: {metadata.get('anchor_groups')}")
print()

for query in ['NVDA', 'AAPL', 'MSFT', 'XLK', 'XLE', 'GLD', 'TLT', 'TSLA']:
    if query in t2i:
        similar = metric.most_similar(query, t2i, i2t, embeddings, top_n=5)
        result = '  '.join([f'{t}({s:.3f})' for t, s in similar])
        print(f"  {query:6s} -> {result}")
    else:
        print(f"  {query:6s} -> NOT IN VOCAB")
