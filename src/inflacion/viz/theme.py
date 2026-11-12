"""Tema gráfico único para notebooks, informes y dashboard.

Paleta categórica validada para daltonismo (orden fijo, no se cicla). Los gráficos
de dispersión con más de tres series deben facetarse o agrupar en "Otros".
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

CATEGORICAL = [
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aqua
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#6250d6",  # violeta
    "#e34948",  # rojo
]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#184f95"], [0.5, NEUTRAL], [1.0, "#b8302f"]]

DIMENSION_COLORS = {
    "MONETARIA": CATEGORICAL[0],
    "INSTITUCIONAL": CATEGORICAL[1],
    "ESTRUCTURAL": CATEGORICAL[2],
    "SOCIODEMOGRAFICA": CATEGORICAL[3],
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _template() -> go.layout.Template:
    axis = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
        title_font=dict(color=INK_SECONDARY, size=12),
        tickfont=dict(color=INK_MUTED, size=11),
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family=FONT, color=INK, size=12),
            title=dict(font=dict(size=16, color=INK), x=0.01, xanchor="left", xref="container", y=0.97, yanchor="top", yref="container"),
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            colorway=CATEGORICAL,
            xaxis=axis, yaxis={**axis, "showline": False},
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text="",
                        font=dict(color=INK_SECONDARY)),
            hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
            margin=dict(l=64, r=32, t=100, b=64),
            bargap=0.25,
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=1, color=SURFACE)))],
            bar=[go.Bar(marker=dict(line=dict(width=0)))],
            histogram=[go.Histogram(marker=dict(line=dict(width=1, color=SURFACE)))],
        ),
    )


def apply_theme() -> None:
    pio.templates["tfm"] = _template()
    pio.templates.default = "tfm"

DIM_LABELS = {
    "MONETARIA": "Monetaria",
    "INSTITUCIONAL": "Institucional",
    "ESTRUCTURAL": "Estructural",
    "SOCIODEMOGRAFICA": "Sociodemográfica",
}
