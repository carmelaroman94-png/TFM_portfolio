"""Exclusión de países con información insuficiente e imputación KNN sobre features estandarizadas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def country_missingness(X: pd.DataFrame) -> pd.Series:
    return 100 * X.isna().mean(axis=1)


def standardize(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu, sd = X.mean(), X.std(ddof=0).replace(0, np.nan)
    return (X - mu) / sd, mu, sd


def knn_cv(Z: pd.DataFrame, k_values: list[int], n_rep: int, frac: float, seed: int) -> pd.DataFrame:
    """Enmascara una fracción de celdas observadas y mide el RMSE de reconstrucción (en unidades estándar)."""
    rng = np.random.default_rng(seed)
    observed = np.argwhere(Z.notna().to_numpy())
    rows = []
    for rep in range(n_rep):
        pick = observed[rng.choice(len(observed), int(frac * len(observed)), replace=False)]
        Zm = Z.to_numpy(copy=True)
        truth = Zm[pick[:, 0], pick[:, 1]].copy()
        Zm[pick[:, 0], pick[:, 1]] = np.nan
        for k in k_values:
            imp = KNNImputer(n_neighbors=k).fit_transform(Zm)
            err = imp[pick[:, 0], pick[:, 1]] - truth
            rows.append({"k": k, "rep": rep, "rmse": float(np.sqrt(np.mean(err ** 2)))})
    res = pd.DataFrame(rows)
    return res.groupby("k")["rmse"].agg(["mean", "std"]).reset_index().sort_values("mean")


def knn_impute(Z: pd.DataFrame, k: int) -> pd.DataFrame:
    return pd.DataFrame(KNNImputer(n_neighbors=k).fit_transform(Z), index=Z.index, columns=Z.columns)
