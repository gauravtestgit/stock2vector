"""Page 2: Stock Query — ticker input, top N similar, OOV estimation."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from dashboard.state import embeddings, t2i, i2t, metric, metadata
from config.settings import settings
from src.implementations.embeddings.oov import WeightedAverageOOV
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.returns import LogReturnsProcessor
from src.implementations.pipeline.cache import ParquetCache
from src.implementations.pipeline.threshold import VolatilityThresholdStrategy

tickers_sorted = sorted(t2i.keys())

layout = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Stock Query"),
                dbc.InputGroup([
                    dbc.Input(id="ticker-input", placeholder="Enter ticker (e.g., NVDA, AMD)",
                              type="text", value=""),
                    dbc.Button("Search", id="search-btn", color="primary", n_clicks=0),
                ]),
                html.Div(id="ticker-badge", className="mt-2"),
                html.Small(f"Vocab: {len(t2i)} stocks. OOV tickers estimated automatically.",
                           className="text-muted"),
            ])),
        ], md=4),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Top N"),
                dcc.Slider(id="top-n-slider", min=3, max=20, step=1, value=10,
                           marks={i: str(i) for i in [3, 5, 10, 15, 20]}),
            ])),
        ], md=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dcc.Loading(dcc.Graph(id="similar-chart", figure=go.Figure())),
        ], md=8),
        dbc.Col([
            html.Div(id="oov-info"),
        ], md=4),
    ]),
])


@callback(
    Output("similar-chart", "figure"),
    Output("ticker-badge", "children"),
    Output("oov-info", "children"),
    Input("search-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("top-n-slider", "value"),
)
def search_ticker(n_clicks, ticker, top_n):
    if not ticker:
        return go.Figure(), "", ""

    ticker = ticker.strip().upper()
    oov_info = ""

    if ticker in t2i:
        # In-vocab query
        badge = dbc.Badge("IN VOCAB", color="success", className="fs-6")
        similar = metric.most_similar(ticker, t2i, i2t, embeddings, top_n=top_n)
    else:
        # OOV estimation
        badge = dbc.Badge("OOV ESTIMATED", color="warning", className="fs-6")
        result = _estimate_oov(ticker)
        if result is None:
            badge = dbc.Badge("OOV FAILED", color="danger", className="fs-6")
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              title=f"Could not estimate '{ticker}' — no co-movement data")
            return fig, badge, "No price data or co-movement found."

        oov_embedding, oov_meta = result
        similar = []
        for t, idx in t2i.items():
            sim = metric.compute(oov_embedding, embeddings[idx])
            similar.append((t, sim))
        similar.sort(key=lambda x: -x[1])
        similar = similar[:top_n]

        oov_info = dbc.Card(dbc.CardBody([
            html.H6("OOV Estimation Details"),
            html.P(f"Method: {oov_meta.method}"),
            html.P(f"Confidence: {oov_meta.confidence}"),
            html.P(f"Co-movement days: {oov_meta.co_movement_days}"),
            html.P(f"Data days used: {oov_meta.data_days_used}"),
            html.Hr(),
            html.H6("Top Co-movers"),
            html.Ul([html.Li(f"{t}: {c:.0f} days") for t, c in oov_meta.top_5_comovers]),
        ]))

    if not similar:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"No results for '{ticker}'")
        return fig, badge, oov_info

    # Build horizontal bar chart
    sim_tickers = [s[0] for s in similar][::-1]
    sim_scores = [s[1] for s in similar][::-1]
    colors = ["#2ecc71" if s > 0.5 else "#f39c12" if s > 0 else "#e74c3c" for s in sim_scores]

    fig = go.Figure(go.Bar(
        x=sim_scores,
        y=sim_tickers,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in sim_scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} most similar to {ticker}",
        xaxis_title="Cosine Similarity",
        xaxis_range=[-1, 1],
        template="plotly_dark",
        height=max(300, top_n * 35),
    )

    return fig, badge, oov_info


def _estimate_oov(ticker):
    """Estimate OOV embedding by fetching prices and computing co-movement."""
    try:
        run_name = metadata.get("run_name", "")
        cache = ParquetCache(settings.raw_dir)
        cached_prices = cache.read(run_name)

        if cached_prices is None or cached_prices.empty:
            return None

        returns_proc = LogReturnsProcessor()
        vocab_returns = returns_proc.compute(cached_prices)

        start = metadata.get("start", "2026-04-01")
        end = metadata.get("end", "2026-04-13")
        source = YFinancePriceSource(batch_size=1, retry_count=2,
                                     retry_wait_secs=2, rate_limit_secs=0.5)
        oov_prices = source.fetch([ticker], start=start, end=end)
        if oov_prices.empty or ticker not in oov_prices.columns:
            return None

        oov_returns = returns_proc.compute(oov_prices)[ticker]

        threshold_strategy = VolatilityThresholdStrategy(settings.threshold_multiplier)
        threshold = threshold_strategy.compute(vocab_returns)

        strategy = WeightedAverageOOV(
            high_confidence_days=settings.oov_high_confidence_days,
            medium_confidence_days=settings.oov_medium_confidence_days,
        )
        return strategy.estimate(ticker, oov_returns, embeddings, t2i,
                                 vocab_returns, threshold)
    except Exception:
        return None
