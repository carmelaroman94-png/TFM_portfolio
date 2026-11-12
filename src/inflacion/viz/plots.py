from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from inflacion.config import Paths
from inflacion.viz.theme import CATEGORICAL, DIMENSION_COLORS, INK_MUTED, INK_SECONDARY

CLUSTER_COLORS = CATEGORICAL[4:7] + [CATEGORICAL[7]]


def save(fig: go.Figure, name: str, width: int = 900, height: int = 520, png: bool = True) -> go.Figure:
    """Guarda HTML interactivo (dashboard) y PNG estático (repositorio/informe)."""
    Paths.figures.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=width, height=height)
    fig.write_html(Paths.figures / f"{name}.html", include_plotlyjs="cdn")
    if png:
        fig.write_image(Paths.figures / f"{name}.png", width=width, height=height, scale=2)
    return fig


def show(fig: go.Figure, name: str, width: int = 900, height: int = 520) -> None:
    """Guarda la figura y la muestra como imagen estática (legible en GitHub sin JavaScript)."""
    from IPython.display import Image, display

    save(fig, name, width, height)
    display(Image(filename=str(Paths.figures / f"{name}.png"), width=width))


def subtitle(title: str, sub: str) -> str:
    return f"{title}<br><sup style='color:{INK_SECONDARY}'>{sub}</sup>"


def heatmap(df: pd.DataFrame, title: str, colorscale, zmid=None, fmt=".2f", zmin=None, zmax=None) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale=colorscale,
                               zmid=zmid, zmin=zmin, zmax=zmax, texttemplate="%{z:" + fmt + "}",
                               hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>", xgap=2, ygap=2,
                               colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(title=title, xaxis=dict(showgrid=False, side="bottom"), yaxis=dict(showgrid=False, autorange="reversed"),
                      margin=dict(l=150))
    return fig


def note(fig: go.Figure, text: str) -> go.Figure:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0, y=-0.18, showarrow=False, xanchor="left",
                       font=dict(size=10, color=INK_MUTED))
    return fig
