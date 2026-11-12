"""Tema gráfico único para notebooks, informes y dashboard.

Paleta categórica validada para daltonismo (orden fijo, no se cicla). Los gráficos
de dispersión con más de tres series deben facetarse o agrupar en "Otros".
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Identidad visual del portfolio: carbón #26352F, musgo #315B4B, rosa #D88EA3, crema #F8F3ED, arena #D8C7B5.
SURFACE = "#FFFDFA"
INK = "#1E2A25"
INK_SECONDARY = "#4E5A54"
INK_MUTED = "#8A8F88"
GRID = "#E8E0D5"
AXIS = "#CFC4B6"
NEUTRAL = "#F1EBE3"

CATEGORICAL = [
    "#315B4B",  # musgo (serie principal)
    "#C2577A",  # rosa (destacado)
    "#3D7EA6",  # azul
    "#C08A2E",  # ocre
    "#7B5EA7",  # ciruela
    "#7FA88F",  # salvia
    "#B05D3B",  # terracota
    "#B03A48",  # rojo ladrillo (excluidos)
]

SEQUENTIAL_BLUE = ["#E4EDE8", "#C2D6CB", "#97B9A8", "#6B9A84", "#477A65", "#315B4B", "#1E3A2F"]
DIVERGING = [[0.0, "#315B4B"], [0.5, NEUTRAL], [1.0, "#C2577A"]]

DIMENSION_COLORS = {
    "MONETARIA": CATEGORICAL[0],
    "INSTITUCIONAL": CATEGORICAL[1],
    "ESTRUCTURAL": CATEGORICAL[2],
    "SOCIODEMOGRAFICA": CATEGORICAL[3],
}

FONT = '"Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif'


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
