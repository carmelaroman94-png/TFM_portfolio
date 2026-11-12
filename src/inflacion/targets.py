"""Variable dependiente: trayectorias inflacionarias con el criterio de cobertura temporal aplicado."""
from __future__ import annotations

import pandas as pd

from inflacion.config import Paths

STATS = ["media", "tendencia", "volatilidad", "mediana", "tendencia_theil_sen"]


def load(name: str = "principal") -> pd.DataFrame:
    """Target por país; los países sin años suficientes quedan como faltantes (no se imputan)."""
    t = pd.read_csv(Paths.processed / f"target_{name}.csv", index_col="iso3")
    cols = [c for c in STATS if c in t]
    t.loc[~t["valido"].fillna(False).astype(bool), cols] = float("nan")
    return t
