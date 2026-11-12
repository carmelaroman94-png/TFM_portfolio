"""Cobertura de indicadores y reducción de cada serie país-indicador a estadísticos de trayectoria."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def coverage(panel: pd.DataFrame, n_countries: int, years: range) -> pd.DataFrame:
    """% de países con algún dato y % medio de años cubiertos entre los países con dato."""
    p = panel[panel["anio"].between(years.start, years.stop - 1)]
    per_country = p.groupby(["codigo", "iso3"]).size().rename("n_anios").reset_index()
    out = per_country.groupby("codigo").agg(n_paises_con_dato=("iso3", "nunique"), anios_medios=("n_anios", "mean"))
    out["pct_paises"] = 100 * out["n_paises_con_dato"] / n_countries
    out["pct_anios_promedio"] = 100 * out["anios_medios"] / len(years)
    out["n_obs"] = p.groupby("codigo").size()
    return out.reset_index()


def trajectories(panel: pd.DataFrame, years: range, min_years: int) -> pd.DataFrame:
    """Media, pendiente OLS (cambio por año) y desvío estándar de los residuos (ddof=1) por país-indicador.

    Se exige un mínimo de años con dato para los tres estadísticos: la media de una serie de uno o dos
    años no es comparable con la de una serie completa.
    """
    p = panel[panel["anio"].between(years.start, years.stop - 1)][["iso3", "codigo", "anio", "valor"]].copy()
    p["x"] = p["anio"] - years.start
    p["xx"], p["xy"], p["yy"] = p["x"] ** 2, p["x"] * p["valor"], p["valor"] ** 2
    g = p.groupby(["iso3", "codigo"]).agg(n=("x", "size"), sx=("x", "sum"), sy=("valor", "sum"),
                                         sxx=("xx", "sum"), sxy=("xy", "sum"), syy=("yy", "sum"))
    g = g[g["n"] >= min_years]
    n = g["n"]
    sxx_c = g["sxx"] - g["sx"] ** 2 / n
    sxy_c = g["sxy"] - g["sx"] * g["sy"] / n
    syy_c = g["syy"] - g["sy"] ** 2 / n
    slope = sxy_c / sxx_c
    rss = (syy_c - slope * sxy_c).clip(lower=0)
    return pd.DataFrame({
        "n_anios": n,
        "media": g["sy"] / n,
        "tendencia": slope,
        "volatilidad": np.sqrt(rss / (n - 1)),
    }).reset_index()


def theil_sen_slopes(panel: pd.DataFrame, codigo: str, years: range, min_years: int) -> pd.Series:
    """Pendiente robusta de Theil-Sen, para contrastar la sensibilidad de la tendencia a episodios extremos."""
    p = panel[(panel["codigo"] == codigo) & panel["anio"].between(years.start, years.stop - 1)]
    out = {}
    for iso, grp in p.groupby("iso3"):
        if len(grp) >= min_years:
            out[iso] = stats.theilslopes(grp["valor"], grp["anio"])[0]
    return pd.Series(out, name="tendencia_theil_sen")


def wide_features(traj: pd.DataFrame, statistics: list[str]) -> pd.DataFrame:
    w = traj.pivot(index="iso3", columns="codigo", values=statistics)
    w.columns = [f"{c}__{s}" for s, c in w.columns]
    return w.sort_index(axis=1)
