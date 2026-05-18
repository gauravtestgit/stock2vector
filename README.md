# Stock2Vec

A Word2Vec-inspired embedding system that discovers behavioral relationships between stocks from price co-movement data. Trains skip-gram models on stock price correlations to produce vector representations where similar-behaving stocks are close in embedding space.

## What It Does

- Trains embeddings on rolling correlation patterns between stocks
- Discovers sector/factor relationships purely from price data (no fundamental data needed)
- Queries similar stocks via cosine similarity
- Estimates embeddings for out-of-vocabulary stocks using correlation-based OOV
- Provides a REST API and interactive dashboard

**Example output (NASDAQ-100 model):**
```
NVDA → AMD (0.94), AVGO (0.84), QQQ (0.78)
TXN  → MCHP (0.87), NXPI (0.80), QCOM (0.77)
PEP  → MDLZ (0.75), KHC (0.74), COST (0.72)
BKR  → FANG (0.98), XLE (0.79)
```

## Architecture

```
server.py          ← FastAPI REST API (port 8000)
dashboard/app.py   ← Dash interactive UI (port 8050)
run_full_flow.py   ← CLI training pipeline
src/
├── api.py         ← Core API class (training, querying, model management)
├── implementations/
│   ├── pipeline/  ← Download, clean, returns, pair generation
│   ├── training/  ← Word2Vec model, trainer, observers
│   └── embeddings/← Cosine similarity, OOV estimation
├── interfaces/    ← Abstract base classes (SOLID)
└── universe/      ← Stock universe management
config/
├── hyperparams.yaml    ← Training hyperparameters
├── universe.yaml       ← Anchor ETFs and stock groups
├── run.yaml.example    ← Run configuration template
└── vocabulary/stocks/  ← Stock list files (CSV/YAML)
```

## Quick Start

### Prerequisites

- Python 3.11+
- Internet connection (for downloading price data from Yahoo Finance)

### Installation

```bash
git clone https://github.com/<your-username>/stock2vec.git
cd stock2vec
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Configure a Run

Copy the example config:
```bash
cp config/run.yaml.example config/run.yaml
```

Edit `config/run.yaml`:
```yaml
run_name: my_first_model
stock_files:
  - nasdaq_100.csv
anchor_groups:
  - market_etfs
  - sector_etfs
start: "2024-04-01"
end: "2026-05-10"
```

### Train a Model

**Option 1: CLI**
```bash
# Uses config/run.yaml
python run_full_flow.py

# Or with CLI overrides
python run_full_flow.py --run-name my_model --stock-file nasdaq_100.csv --start 2024-01-01 --end 2026-05-10
```

**Option 2: REST API**
```bash
python server.py

curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"run_name": "my_model", "stock_file": "nasdaq_100.csv", "anchor_groups": ["market_etfs", "sector_etfs"], "start": "2024-04-01", "end": "2026-05-10", "epochs": 200, "lr": 0.01, "min_corr": 0.85}'
```

**Option 3: Dashboard UI**
```bash
python server.py &
python dashboard/app.py
# Navigate to http://localhost:8050/train
```

### Query a Model

**CLI:**
```bash
python debug/query_model.py --model data/models/my_model/current --ticker NVDA
python debug/query_model.py --model data/models/my_model/current --ticker CRM  # OOV estimation
```

**REST API:**
```bash
# Similar stocks
curl http://localhost:8000/api/models/my_model/similar/NVDA?top_n=10

# Compare two stocks
curl http://localhost:8000/api/models/my_model/compare/NVDA/AMD

# OOV estimation (stock not in training vocab)
curl http://localhost:8000/api/models/my_model/oov/CRM?top_n=10

# List vocab
curl http://localhost:8000/api/models/my_model/vocab
```

**Dashboard:**
```
http://localhost:8050/query
```

## Running the Full Stack

```bash
# Terminal 1: API server
python server.py

# Terminal 2: Dashboard
python dashboard/app.py
```

- Dashboard: http://localhost:8050
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all trained models |
| GET | `/api/models/{name}` | Model metadata |
| GET | `/api/models/{name}/similar/{ticker}?top_n=10` | Top N similar stocks |
| GET | `/api/models/{name}/compare/{a}/{b}` | Compare two stocks |
| GET | `/api/models/{name}/oov/{ticker}?top_n=10` | OOV estimation |
| GET | `/api/models/{name}/vocab` | List vocab tickers |
| POST | `/api/train` | Start training |
| GET | `/api/train/{run_name}/status` | Training progress |
| GET | `/api/train/{run_name}/loss` | Loss history |
| GET | `/api/config/stock-files` | Available stock files |
| GET | `/api/config/anchor-groups` | Available anchor groups |

## How It Works

### Training Pipeline

1. **Universe** — Select stocks from CSV/YAML files + anchor ETFs (SPY, sector ETFs)
2. **Download** — Fetch daily prices from Yahoo Finance (cached as Parquet)
3. **Clean** — Remove stocks with insufficient history or low price
4. **Returns** — Compute market-neutral log returns (subtract SPY)
5. **Pairs** — Generate training pairs from rolling 30-day correlation windows (min_correlation=0.85)
6. **Train** — Skip-gram with negative sampling (5 negatives, 200 epochs)
7. **Save** — Embeddings + vocab + metadata as NumPy/JSON

### Key Design Decisions

- **Correlation-based pairing** over threshold co-movement — produces asymmetric pair distributions needed for learning
- **Market-neutral returns** — subtracts SPY to remove market-wide noise, isolating sector-specific signal
- **Negative sampling** over full softmax — scales to 100+ stocks without gradient dilution
- **2-year training window** — balances stability with recency (5-year captures factor/momentum artifacts)

### OOV Estimation

For stocks not in the training vocabulary:
1. Fetch price data for the same date range
2. Compute rolling correlation with each vocab stock (market-neutral)
3. Weight vocab embeddings by correlation strength (min 0.5)
4. Normalize to unit length

Works well for stocks that genuinely correlate with the vocab (e.g., CRM → TEAM, ADSK, INTU). Reports "low confidence" for stocks with no strong correlations.

## Configuration

### Hyperparameters (`config/hyperparams.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 32 | Embedding dimensions |
| epochs | 200 | Training epochs |
| lr | 0.01 | Learning rate (0.01 for ~100 stocks, 0.1+ for larger) |
| model_mode | negative_sampling | `softmax` or `negative_sampling` |
| num_negatives | 5 | Negative samples per pair |
| returns_mode | market_neutral | `log` or `market_neutral` |
| pair_mode | correlation | `comovement` or `correlation` |
| correlation_min | 0.85 | Minimum correlation to form a pair |
| correlation_window_days | 30 | Rolling window size |
| correlation_step_days | 5 | Step between windows |

### Stock Files (`config/vocabulary/stocks/`)

- `nasdaq_100.csv` — NASDAQ-100 constituents
- `diverse_sample.yaml` — 15 stocks from 5 sectors (for testing)
- Add your own CSV (one ticker per line) or YAML files

### Anchor Groups (`config/universe.yaml`)

- `market_etfs` — SPY, QQQ, IWM, DIA, VTI
- `sector_etfs` — XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLY, XLC, XLB, XLRE

## Deployment (EC2)

```bash
# On EC2 instance
sudo mkdir -p /opt/stock2vec
sudo chown ec2-user:ec2-user /opt/stock2vec
cd /opt/stock2vec
git clone https://github.com/<your-username>/stock2vec.git .
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create run config
cp config/run.yaml.example config/run.yaml

# Run in production
export STOCK2VEC_ENV=production
nohup python server.py > server.log 2>&1 &
nohup python dashboard/app.py > dashboard.log 2>&1 &
```

Security group: open ports 8000 (API) and 8050 (dashboard).

## Project Structure

```
stock2vec/
├── server.py                 # FastAPI REST server
├── run_full_flow.py          # CLI training pipeline
├── dashboard/
│   └── app.py                # Dash UI (calls API via HTTP)
├── src/
│   ├── api.py                # Core API class
│   ├── interfaces/           # Abstract base classes
│   ├── implementations/
│   │   ├── pipeline/         # Data pipeline (download, clean, returns, pairs)
│   │   ├── training/         # Model, trainer, observers, persistence
│   │   └── embeddings/       # Metrics, OOV estimation
│   └── universe/             # Universe management
├── config/
│   ├── hyperparams.yaml      # Training parameters
│   ├── universe.yaml         # ETF/anchor definitions
│   ├── settings.py           # Settings loader
│   ├── run.yaml.example      # Run config template
│   └── vocabulary/stocks/    # Stock list files
├── debug/                    # Debug/analysis scripts
├── data/                     # Generated (gitignored)
│   ├── raw/                  # Cached price data (Parquet)
│   └── models/               # Trained models (NumPy)
└── requirements.txt
```

## License

MIT
