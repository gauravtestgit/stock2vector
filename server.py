"""Stock2Vec REST API — FastAPI layer on top of src/api.py.

Endpoints:
    GET  /api/models                              — list available models
    GET  /api/models/{name}                       — model metadata
    GET  /api/models/{name}/similar/{ticker}      — top N similar stocks
    GET  /api/models/{name}/compare/{a}/{b}       — compare two stocks
    GET  /api/models/{name}/oov/{ticker}          — OOV estimation
    POST /api/train                               — start training
    GET  /api/train/{run_name}/status             — poll training progress
    GET  /api/train/{run_name}/loss               — loss history
    GET  /api/config/stock-files                  — available stock files
    GET  /api/config/anchor-groups                — available anchor groups

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
    # Or alongside dashboard: python server.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from src.api import api, TrainConfig, TrainStatus

app = FastAPI(title="Stock2Vec API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache loaded models in memory ────────────────────────────
_model_cache = {}


def _get_model(name: str):
    """Load model by name, with caching."""
    if name not in _model_cache:
        models = api.list_models()
        match = next((m for m in models if m["name"] == name), None)
        if not match:
            raise HTTPException(404, f"Model '{name}' not found")
        embeddings, vocab, metadata = api.load_model(match["path"])
        i2t = {int(i): t for t, i in vocab.items()}
        _model_cache[name] = {
            "embeddings": embeddings,
            "t2i": vocab,
            "i2t": i2t,
            "metadata": metadata,
            "path": match["path"],
        }
    return _model_cache[name]


# ── Request/Response models ──────────────────────────────────

class TrainRequest(BaseModel):
    run_name: str
    stock_file: str
    anchor_groups: List[str] = ["market_etfs", "sector_etfs"]
    start: str = "2024-04-01"
    end: str = "2026-05-10"
    epochs: int = 200
    lr: float = 0.01
    min_corr: float = 0.85


class SimilarStock(BaseModel):
    ticker: str
    similarity: float


class OOVResponse(BaseModel):
    ticker: str
    confidence: str
    method: str
    co_movement_days: int
    data_days_used: int
    top_comovers: List[dict]
    similar: List[SimilarStock]


# ── Model endpoints ──────────────────────────────────────────

@app.get("/api/models")
def list_models():
    return api.list_models()


@app.get("/api/models/{name}")
def get_model_info(name: str):
    model = _get_model(name)
    return model["metadata"]


@app.get("/api/models/{name}/similar/{ticker}")
def get_similar(name: str, ticker: str, top_n: int = 10):
    model = _get_model(name)
    ticker = ticker.upper()

    if ticker not in model["t2i"]:
        raise HTTPException(404, f"Ticker '{ticker}' not in model vocab. Use /oov/ endpoint.")

    results = api.most_similar(
        ticker, model["embeddings"], model["t2i"], model["i2t"], top_n=top_n
    )
    return [{"ticker": t, "similarity": round(s, 4)} for t, s in results]


@app.get("/api/models/{name}/compare/{ticker_a}/{ticker_b}")
def compare_stocks(name: str, ticker_a: str, ticker_b: str):
    model = _get_model(name)
    ticker_a, ticker_b = ticker_a.upper(), ticker_b.upper()

    for t in [ticker_a, ticker_b]:
        if t not in model["t2i"]:
            raise HTTPException(404, f"Ticker '{t}' not in model vocab")

    emb_a = model["embeddings"][model["t2i"][ticker_a]]
    emb_b = model["embeddings"][model["t2i"][ticker_b]]
    sim = api.compute_similarity(emb_a, emb_b)

    return {"ticker_a": ticker_a, "ticker_b": ticker_b, "similarity": round(float(sim), 4)}


@app.get("/api/models/{name}/oov/{ticker}")
def estimate_oov(name: str, ticker: str, top_n: int = 10):
    model = _get_model(name)
    ticker = ticker.upper()

    if ticker in model["t2i"]:
        raise HTTPException(400, f"Ticker '{ticker}' is in vocab. Use /similar/ endpoint.")

    result = api.estimate_oov(ticker, model["embeddings"], model["t2i"], model["metadata"])
    if result is None:
        raise HTTPException(404, f"Could not estimate '{ticker}' — no price data or insufficient correlation")

    oov_embedding, oov_meta = result

    # Compute similarities
    similar = []
    for t, idx in model["t2i"].items():
        sim = api.compute_similarity(oov_embedding, model["embeddings"][idx])
        similar.append((t, float(sim)))
    similar.sort(key=lambda x: -x[1])
    similar = similar[:top_n]

    return {
        "ticker": ticker,
        "confidence": oov_meta.confidence,
        "method": oov_meta.method,
        "co_movement_days": oov_meta.co_movement_days,
        "data_days_used": oov_meta.data_days_used,
        "top_comovers": [{"ticker": t, "correlation": round(c, 4)} for t, c in oov_meta.top_5_comovers],
        "similar": [{"ticker": t, "similarity": round(s, 4)} for t, s in similar],
    }


@app.get("/api/models/{name}/vocab")
def get_vocab(name: str):
    model = _get_model(name)
    return sorted(model["t2i"].keys())


# ── Training endpoints ───────────────────────────────────────

@app.post("/api/train")
def start_training(req: TrainRequest):
    if api.is_training:
        raise HTTPException(409, "Training already in progress")

    config = TrainConfig(
        run_name=req.run_name,
        stock_file=req.stock_file,
        anchor_groups=req.anchor_groups,
        start=req.start,
        end=req.end,
        epochs=req.epochs,
        lr=req.lr,
        min_corr=req.min_corr,
    )
    api.start_training(config)
    return {"status": "started", "run_name": req.run_name}


@app.get("/api/train/{run_name}/status")
def get_train_status(run_name: str):
    status = api.get_train_status(run_name)
    return {
        "status": status.status,
        "epoch": status.epoch,
        "total_epochs": status.total_epochs,
        "current_loss": round(status.current_loss, 4),
        "final_loss": round(status.final_loss, 4),
        "error": status.error,
    }


@app.get("/api/train/{run_name}/loss")
def get_loss_history(run_name: str):
    df = api.get_loss_history(run_name)
    if df is None:
        raise HTTPException(404, f"No loss history for '{run_name}'")
    return df.to_dict(orient="records")


# ── Config endpoints ─────────────────────────────────────────

@app.get("/api/config/stock-files")
def get_stock_files():
    return api.list_stock_files()


@app.get("/api/config/anchor-groups")
def get_anchor_groups():
    return api.list_anchor_groups()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
