"""Page 1: Model Overview — similarity heatmap + PCA scatter."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA

from dashboard.state import embeddings, t2i, i2t, metric, metadata

# Pre-compute
tickers = sorted(t2i.keys())
sim_matrix = metric.compute_matrix(embeddings)

# PCA for scatter
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(embeddings)


def build_heatmap():
    """Build similarity heatmap figure."""
    # Reorder by ticker name for readability
    indices = [t2i[t] for t in tickers]
    ordered_matrix = sim_matrix[np.ix_(indices, indices)]

    fig = px.imshow(
        ordered_matrix,
        x=tickers,
        y=tickers,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        title=f"Cosine Similarity Matrix ({len(tickers)} stocks)",
        template="plotly_dark",
        height=max(500, len(tickers) * 12),
    )
    return fig


def build_scatter():
    """Build 2D PCA scatter plot."""
    ticker_labels = [i2t[i] for i in range(len(embeddings))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode="markers+text",
        text=ticker_labels,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=8, color=coords_2d[:, 0], colorscale="Viridis"),
        hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="2D PCA Embedding Space",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        template="plotly_dark",
        height=600,
    )
    return fig


# Model info card
info_card = dbc.Card(dbc.CardBody([
    html.H5("Model Info"),
    html.P(f"Run: {metadata.get('run_name', '?')}"),
    html.P(f"Vocab: {len(t2i)} stocks | Embed dim: {metadata.get('embed_dim', '?')}"),
    html.P(f"Epochs: {metadata.get('epochs', '?')} | Final loss: {metadata.get('final_loss', 0):.4f}"),
    html.P(f"Pairs: {metadata.get('pair_count', '?')} | Date range: {metadata.get('start', '?')} → {metadata.get('end', '?')}"),
]), className="mb-3")

layout = html.Div([
    dbc.Row([dbc.Col(info_card)]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="pca-scatter", figure=build_scatter()),
        ], md=6),
        dbc.Col([
            dcc.Graph(id="heatmap", figure=build_heatmap()),
        ], md=6),
    ]),
])
