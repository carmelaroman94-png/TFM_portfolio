"""Clustering de perfiles monetario-fiscales y medidas de heterogeneidad inflacionaria intra-grupo (H1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture


def tukey_outliers(X: pd.DataFrame) -> tuple[pd.Series, float]:
    """Distancia euclídea al centroide; atípico si supera Q3 + 1.5·RIC."""
    d = pd.Series(np.linalg.norm(X - X.mean(), axis=1), index=X.index, name="distancia_centroide")
    q1, q3 = d.quantile([0.25, 0.75])
    return d, float(q3 + 1.5 * (q3 - q1))


def fit_labels(X: np.ndarray, algo: str, param: str, k: int, seed: int) -> np.ndarray:
    if algo == "KMeans":
        return KMeans(k, n_init=50, random_state=seed).fit_predict(X)
    if algo == "Jerarquico":
        return AgglomerativeClustering(n_clusters=k, linkage=param).fit_predict(X)
    if algo == "GMM":
        return GaussianMixture(k, covariance_type=param, n_init=10, random_state=seed).fit(X).predict(X)
    raise ValueError(algo)


def grid(X: pd.DataFrame, k_grid, linkages, covariances, seed: int) -> pd.DataFrame:
    configs = [("KMeans", "-", k) for k in k_grid]
    configs += [("Jerarquico", l, k) for l in linkages for k in k_grid]
    configs += [("GMM", c, k) for c in covariances for k in k_grid]
    rows = []
    for algo, param, k in configs:
        lab = fit_labels(X.to_numpy(), algo, param, k, seed)
        sizes = np.bincount(lab, minlength=k)
        n_eff = int((sizes > 0).sum())
        rows.append({"algoritmo": algo, "parametro": param, "k": k, "k_efectivo": n_eff,
                     "silhouette": silhouette_score(X, lab) if n_eff > 1 else np.nan,
                     "davies_bouldin": davies_bouldin_score(X, lab) if n_eff > 1 else np.nan,
                     "calinski_harabasz": calinski_harabasz_score(X, lab) if n_eff > 1 else np.nan,
                     "tam_min": int(sizes[sizes > 0].min()), "tamanios": sorted(sizes.tolist(), reverse=True)})
    return pd.DataFrame(rows)


def bootstrap_stability(X: pd.DataFrame, algo: str, param: str, k: int, seed: int,
                        n_boot: int = 50, frac: float = 0.8) -> float:
    """ARI medio entre la partición completa y las obtenidas en submuestras (sobre los países comunes)."""
    rng = np.random.default_rng(seed)
    full = fit_labels(X.to_numpy(), algo, param, k, seed)
    aris = []
    for b in range(n_boot):
        idx = np.sort(rng.choice(len(X), int(frac * len(X)), replace=False))
        lab = fit_labels(X.to_numpy()[idx], algo, param, k, seed + b + 1)
        aris.append(adjusted_rand_score(full[idx], lab))
    return float(np.mean(aris))


def within_dispersion(y: pd.DataFrame, labels: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Dispersión de cada estadístico de inflación dentro de cada cluster y en el total."""
    rows = []
    groups = list(labels.groupby(labels).groups.items()) + [("Total", labels.index)]
    for g, idx in groups:
        for c in cols:
            v = y.loc[idx, c].dropna()
            q1, q3 = v.quantile([0.25, 0.75])
            rows.append({"cluster": g, "estadistico": c, "n": len(v), "mediana": v.median(), "media": v.mean(),
                         "desvio": v.std(), "ric": q3 - q1, "min": v.min(), "max": v.max(),
                         "cv": v.std() / abs(v.mean()) if c != "tendencia" and v.mean() != 0 else np.nan})
    return pd.DataFrame(rows)


def epsilon_squared(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Kruskal-Wallis y ε² (proporción de variación en rangos asociada al grupo)."""
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return float(h), float(p), float(h / ((n ** 2 - 1) / (n + 1)))


def permutation_eta2(y: np.ndarray, labels: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    """η² observado de la partición sobre y, y p-valor por permutación de etiquetas."""
    def eta2(lab):
        grand = y.mean()
        ssb = sum(len(y[lab == g]) * (y[lab == g].mean() - grand) ** 2 for g in np.unique(lab))
        return ssb / ((y - grand) ** 2).sum()
    rng = np.random.default_rng(seed)
    obs = eta2(labels)
    perm = np.array([eta2(rng.permutation(labels)) for _ in range(n_perm)])
    return float(obs), float((1 + (perm >= obs).sum()) / (n_perm + 1))
