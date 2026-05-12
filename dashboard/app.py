"""Stock2Vec Dashboard — polished version.

Page 1: Model Overview — clustered heatmap, PCA scatter, loss curve
Page 2: Stock Query — autocomplete, top N similar, OOV support

Run: python dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from config.settings import settings
from src.implementations.training.persistence import NumpyPersistence
from src.implementations.embeddings.metrics import CosineMetric
from src.implementations.embeddings.oov import CorrelationOOV
from src.implementations.pipeline.sources import YFinancePriceSource
from src.implementations.pipeline.returns import LogReturnsProcessor, MarketNeutralReturnsProcessor
from src.implementations.pipeline.cache import ParquetCache

persistence = NumpyPersistence()
metric = CosineMetric()


def discover_models():
    """Find models that trained successfully (have current/ with W1.npy)."""
    models_dir = settings.models_dir
    options = []
    if os.path.exists(models_dir):
        for name in sorted(os.listdir(models_dir)):
            current_path = os.path.join(models_dir, name, "current")
            if os.path.exists(os.path.join(current_path, "W1.npy")):
                # Load metadata to show useful info in dropdown
                try:
                    _, _, meta = persistence.load(current_path)
                    vocab_size = meta.get("vocab_size", "?")
                    loss = meta.get("final_loss", 0)
                    label = f"{name} ({vocab_size} stocks, loss={loss:.3f})"
                except Exception:
                    label = name
                options.append({"label": label, "value": current_path})
    return options


def load_model(model_path):
    """Load model and compute derived data."""
    embeddings, vocab, metadata = persistence.load(model_path)
    t2i = vocab
    i2t = {int(i): t for t, i in vocab.items()}
    sim_matrix = metric.compute_matrix(embeddings)

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)

    # Load loss history if available
    model_root = os.path.dirname(model_path)
    loss_path = os.path.join(model_root, "loss_history.csv")
    loss_df = None
    if os.path.exists(loss_path):
        try:
            loss_df = pd.read_csv(loss_path)
        except Exception:
            pass

    return embeddings, t2i, i2t, metadata, sim_matrix, coords_2d, pca, loss_df


# Initial load
available_models = discover_models()
default_model = available_models[0]["value"] if available_models else "data/models/sample/current"
embeddings, t2i, i2t, metadata, sim_matrix, coords_2d, pca, loss_df = load_model(default_model)

# ── App ──────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="model-path-store", data=default_model),

    # Navbar
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/")),
            dbc.NavItem(dbc.NavLink("Query", href="/query")),
            dbc.NavItem(html.Div([
                dcc.Dropdown(
                    id="model-selector",
                    options=available_models,
                    value=default_model,
                    clearable=False,
                    style={"width": "380px", "color": "#000"},
                ),
            ], className="ms-3 d-flex align-items-center")),
        ],
        brand="Stock2Vec",
        brand_href="/",
        color="primary",
        dark=True,
    ),

    # Model info bar
    html.Div(id="model-info-bar", className="container-fluid mt-2"),

    # Page content
    html.Div(id="page-content", className="container-fluid mt-3"),
])


@app.callback(
    Output("model-path-store", "data"),
    Output("model-info-bar", "children"),
    Input("model-selector", "value"),
)
def switch_model(model_path):
    global embeddings, t2i, i2t, metadata, sim_matrix, coords_2d, pca, loss_df
    embeddings, t2i, i2t, metadata, sim_matrix, coords_2d, pca, loss_df = load_model(model_path)

    info = dbc.Alert([
        html.Strong(f"{metadata.get('run_name', '?')}"),
        f" | {len(t2i)} stocks | dim={metadata.get('embed_dim', '?')}",
        f" | {metadata.get('epochs', '?')} epochs | loss={metadata.get('final_loss', 0):.4f}",
        f" | {metadata.get('start', '?')} → {metadata.get('end', '?')}",
        f" | {metadata.get('pair_count', '?')} pairs",
    ], color="info", className="py-1 mb-0")

    return model_path, info


@app.callback(Output("page-content", "children"),
              Input("url", "pathname"),
              Input("model-path-store", "data"))
def display_page(pathname, model_path):
    if pathname == "/query":
        return _build_query_layout()
    return _build_overview_layout()


# ── Overview Page ────────────────────────────────────────────

def _cluster_order(sim_mat):
    """Hierarchical clustering order for heatmap."""
    dist = 1 - sim_mat
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return leaves_list(Z)


def _build_overview_layout():
    n = len(t2i)

    # Clustered heatmap
    order = _cluster_order(sim_matrix)
    ordered_tickers = [i2t[i] for i in order]
    ordered_matrix = sim_matrix[np.ix_(order, order)]

    heatmap_fig = px.imshow(
        ordered_matrix,
        x=ordered_tickers, y=ordered_tickers,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    heatmap_fig.update_layout(
        title=f"Similarity Matrix (clustered, {n} stocks)",
        template="plotly_dark",
        height=max(550, n * 10),
    )
    # Hide tick labels if too many stocks
    if n > 50:
        heatmap_fig.update_xaxes(tickfont=dict(size=7))
        heatmap_fig.update_yaxes(tickfont=dict(size=7))

    # PCA scatter — hover only for large vocab
    ticker_labels = [i2t[i] for i in range(len(embeddings))]
    text_mode = "markers+text" if n <= 30 else "markers"
    scatter_fig = go.Figure(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1],
        mode=text_mode,
        text=ticker_labels,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=7, color=coords_2d[:, 0], colorscale="Viridis"),
        hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
    ))
    scatter_fig.update_layout(
        title=f"PCA Embedding Space ({pca.explained_variance_ratio_[0]:.0%} + {pca.explained_variance_ratio_[1]:.0%} variance)",
        xaxis_title="PC1", yaxis_title="PC2",
        template="plotly_dark", height=550,
    )

    # Loss curve
    loss_chart = html.Div()
    if loss_df is not None and not loss_df.empty:
        loss_fig = go.Figure(go.Scatter(
            x=loss_df["epoch"], y=loss_df["avg_loss"],
            mode="lines", line=dict(color="#2ecc71", width=2),
        ))
        loss_fig.update_layout(
            title="Training Loss",
            xaxis_title="Epoch", yaxis_title="Loss",
            template="plotly_dark", height=300,
        )
        loss_chart = dcc.Graph(figure=loss_fig)

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=scatter_fig), md=6),
            dbc.Col(dcc.Graph(figure=heatmap_fig), md=6),
        ]),
        dbc.Row([
            dbc.Col(loss_chart, md=12),
        ], className="mt-2"),
    ])


# ── Query Page ───────────────────────────────────────────────

def _build_query_layout():
    ticker_options = [{"label": t, "value": t} for t in sorted(t2i.keys())]

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Stock Query"),
                dcc.Dropdown(
                    id="ticker-input",
                    options=ticker_options,
                    placeholder="Type or select ticker (e.g., NVDA)",
                    style={"color": "#000"},
                ),
                html.Small("Type any ticker — OOV stocks estimated automatically.",
                           className="text-muted mt-1"),
                html.Div(id="ticker-badge", className="mt-2"),
            ])), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Top N"),
                dcc.Slider(id="top-n-slider", min=3, max=20, step=1, value=10,
                           marks={i: str(i) for i in [3, 5, 10, 15, 20]}),
            ])), md=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("OOV Ticker (manual)"),
                dbc.InputGroup([
                    dbc.Input(id="oov-input", placeholder="e.g., TSMC, BABA",
                              type="text", value="", debounce=True),
                    dbc.Button("Estimate", id="oov-btn", color="warning", n_clicks=0),
                ]),
            ])), md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="similar-chart", figure=go.Figure())), md=7),
            dbc.Col(html.Div(id="oov-info"), md=5),
        ]),
    ])


@app.callback(
    Output("similar-chart", "figure"),
    Output("ticker-badge", "children"),
    Output("oov-info", "children"),
    Input("ticker-input", "value"),
    Input("oov-btn", "n_clicks"),
    State("oov-input", "value"),
    State("top-n-slider", "value"),
)
def search_ticker(ticker_dropdown, oov_clicks, oov_input, top_n):
    # Determine which input triggered
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "oov-btn" in triggered and oov_input:
        ticker = oov_input.strip().upper()
        force_oov = True
    elif ticker_dropdown:
        ticker = ticker_dropdown.strip().upper()
        force_oov = False
    else:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Select a ticker to search")
        return fig, "", ""

    oov_info = ""

    if ticker in t2i and not force_oov:
        badge = dbc.Badge("IN VOCAB", color="success", className="fs-6")
        similar = metric.most_similar(ticker, t2i, i2t, embeddings, top_n=top_n)
    else:
        badge = dbc.Badge("OOV — ESTIMATING...", color="warning", className="fs-6")
        result = _estimate_oov(ticker)
        if result is None:
            badge = dbc.Badge("OOV FAILED", color="danger", className="fs-6")
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              title=f"Could not estimate '{ticker}' — no price data or co-movement")
            return fig, badge, "Failed to download prices or find co-movement with vocab stocks."

        oov_embedding, oov_meta = result
        badge = dbc.Badge(f"OOV — {oov_meta.confidence.upper()} CONFIDENCE", color="warning", className="fs-6")
        similar = []
        for t, idx in t2i.items():
            sim = metric.compute(oov_embedding, embeddings[idx])
            similar.append((t, sim))
        similar.sort(key=lambda x: -x[1])
        similar = similar[:top_n]

        oov_info = dbc.Card(dbc.CardBody([
            html.H6("OOV Estimation"),
            html.P([html.Strong("Method: "), oov_meta.method]),
            html.P([html.Strong("Confidence: "), oov_meta.confidence]),
            html.P([html.Strong("Co-movement days: "), str(oov_meta.co_movement_days)]),
            html.P([html.Strong("Data days: "), str(oov_meta.data_days_used)]),
            html.Hr(),
            html.H6("Top Co-movers (by days)"),
            html.Ul([html.Li(f"{t}: {c:.0f}d") for t, c in oov_meta.top_5_comovers]),
        ]))

    if not similar:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"No results for '{ticker}'")
        return fig, badge, oov_info

    sim_tickers = [s[0] for s in similar][::-1]
    sim_scores = [s[1] for s in similar][::-1]
    colors = ["#2ecc71" if s > 0.7 else "#f1c40f" if s > 0.4 else "#e67e22" if s > 0 else "#e74c3c"
              for s in sim_scores]

    fig = go.Figure(go.Bar(
        x=sim_scores, y=sim_tickers,
        orientation="h", marker_color=colors,
        text=[f"{s:.3f}" for s in sim_scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} most similar to {ticker}",
        xaxis_title="Cosine Similarity",
        xaxis_range=[-0.5, 1.05],
        template="plotly_dark",
        height=max(350, top_n * 38),
        margin=dict(l=80),
    )
    return fig, badge, oov_info


def _estimate_oov(ticker):
    """Estimate OOV embedding."""
    try:
        run_name = metadata.get("run_name", "")
        cache = ParquetCache(settings.raw_dir)
        cached_prices = cache.read(run_name)
        if cached_prices is None or cached_prices.empty:
            return None

        if settings.returns_mode == "market_neutral":
            returns_proc = MarketNeutralReturnsProcessor(settings.benchmark_ticker)
        else:
            returns_proc = LogReturnsProcessor()
        vocab_returns = returns_proc.compute(cached_prices)

        start = metadata.get("start", "2024-04-01")
        end = metadata.get("end", "2026-05-10")
        source = YFinancePriceSource(batch_size=1, retry_count=2,
                                     retry_wait_secs=2, rate_limit_secs=0.5)
        tickers_to_fetch = [ticker]
        if settings.returns_mode == "market_neutral":
            tickers_to_fetch.append(settings.benchmark_ticker)
        oov_prices = source.fetch(tickers_to_fetch, start=start, end=end)
        if oov_prices.empty or ticker not in oov_prices.columns:
            return None

        oov_returns = returns_proc.compute(oov_prices)[ticker]

        strategy = CorrelationOOV(
            window_days=settings.correlation_window_days,
            min_correlation=0.5,
            step_days=settings.correlation_step_days,
        )
        return strategy.estimate(ticker, oov_returns, embeddings, t2i,
                                 vocab_returns)
    except Exception:
        return None


if __name__ == "__main__":
    print("Starting Stock2Vec Dashboard")
    print(f"  Models: {len(available_models)} available")
    print(f"  Default: {default_model}")
    print(f"  URL: http://{settings.host}:{settings.port}")
    app.run(host=settings.host, port=settings.port, debug=True)
