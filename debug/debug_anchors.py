import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.universe.universe import UniverseManager

m = UniverseManager()
print("Available anchor groups:", m.list_anchor_groups())
print()

for group in ['sector_etfs', 'macro']:
    anchors = m.get_anchors([group])
    print(f"{group}: {anchors}")
    print()

# Full universe with anchors
tickers = m.build_universe(["nasdaq_100.csv"], ["sector_etfs", "macro"])
print(f"Total universe: {len(tickers)}")

# Check if US sector ETFs are in there
us_sector = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
for etf in us_sector:
    print(f"  {etf}: {'YES' if etf in tickers else 'NO'}")
