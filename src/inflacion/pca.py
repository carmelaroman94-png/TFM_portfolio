"""PCA dentro de cada bloque dimensión × estadístico de trayectoria, con poda previa por redundancia y MSA."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy import stats
from sklearn.decomposition import PCA


@dataclass
class BlockResult:
    name: str
    n_inicial: int
    n_post_correlacion: int
    umbral_correlacion: float | None
    variables: list[str]
    protegidas: list[str]
    kmo: float
    bartlett_p: float
    n_componentes: int
    varianza: np.ndarray
    scores: pd.DataFrame
    cargas: pd.DataFrame
    eliminadas: list[tuple[str, str]] = field(default_factory=list)


def _kmo(X: pd.DataFrame) -> tuple[np.ndarray, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calculate_kmo(X)


def bartlett_sphericity(X: pd.DataFrame) -> tuple[float, float]:
    """Test de esfericidad de Bartlett con log-determinante estable (evita underflow con muchas variables)."""
    n, p = X.shape
    _, logdet = np.linalg.slogdet(np.corrcoef(X.to_numpy(), rowvar=False))
    stat = -(n - 1 - (2 * p + 5) / 6) * logdet
    return float(stat), float(stats.chi2.sf(stat, p * (p - 1) / 2))


def prune_correlated(X: pd.DataFrame, protected: list[str], threshold: float) -> tuple[list[str], list[str]]:
    """Elimina, de cada par con |r| > umbral, la variable no protegida con mayor correlación media absoluta."""
    names = list(X.columns)
    corr = np.abs(np.corrcoef(X.to_numpy(), rowvar=False))
    np.fill_diagonal(corr, 0)
    is_prot = np.array([c in protected for c in names])
    alive = np.ones(len(names), dtype=bool)
    dropped = []
    while True:
        sub = np.where(np.outer(alive, alive), corr, 0)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i, j] <= threshold:
            break
        cand = [c for c in (i, j) if not is_prot[c]]
        if not cand:
            corr[i, j] = corr[j, i] = 0
            continue
        victim = max(cand, key=lambda c: sub[c].sum() / max(alive.sum() - 1, 1))
        alive[victim] = False
        dropped.append(names[victim])
    return [n for n, a in zip(names, alive) if a], dropped


def prune_msa(X: pd.DataFrame, protected: list[str], kmo_target: float, min_vars: int) -> tuple[list[str], list[str], float]:
    """Elimina iterativamente la variable no protegida de menor MSA hasta alcanzar el KMO objetivo."""
    cols, dropped = list(X.columns), []
    while True:
        msa, kmo = _kmo(X[cols])
        if kmo >= kmo_target or len(cols) <= min_vars:
            return cols, dropped, float(kmo)
        msa = pd.Series(msa, index=cols).drop(protected, errors="ignore")
        if msa.isna().all() or msa.empty:
            return cols, dropped, float(kmo)
        worst = msa.idxmin()
        cols.remove(worst)
        dropped.append(worst)


def run_block(Z: pd.DataFrame, name: str, protected: list[str], var_target: float, kmo_target: float,
              min_vars: int, thresholds=(0.95, 0.90, 0.85, 0.80)) -> BlockResult:
    n_obs, n0 = Z.shape
    # Duplicados casi exactos (p. ej. el mismo índice WGI en dos escalas) se eliminan siempre.
    cols, dropped = prune_correlated(Z, protected, 0.99)
    used_thr, removed = None, [(c, "duplicado|r|>0.99") for c in dropped]
    for thr in thresholds:
        if len(cols) <= n_obs // 2:
            break
        cols, dropped = prune_correlated(Z[cols], protected, thr)
        removed += [(c, f"correlacion>{thr}") for c in dropped]
        used_thr = thr
    n_corr = len(cols)
    cols, dropped, kmo = prune_msa(Z[cols], protected, kmo_target, min_vars)
    removed += [(c, "msa") for c in dropped]

    X = Z[cols]
    _, bart_p = bartlett_sphericity(X)
    pca = PCA(random_state=0).fit(X)
    var = pca.explained_variance_ratio_
    k = int(np.searchsorted(np.cumsum(var), var_target) + 1)
    comp_names = [f"{name}__PC{i + 1}" for i in range(k)]
    # Orientación: el signo de cada componente se fija para que su variable de mayor carga absoluta sea positiva.
    sign = np.sign(pca.components_[np.arange(k), np.abs(pca.components_[:k]).argmax(axis=1)])
    scores = pd.DataFrame(pca.transform(X)[:, :k] * sign, index=Z.index, columns=comp_names)
    loadings = pd.DataFrame(pca.components_[:k].T * sign * np.sqrt(pca.explained_variance_[:k]), index=cols, columns=comp_names)
    return BlockResult(name, n0, n_corr, used_thr, cols, [c for c in cols if c in protected], kmo, float(bart_p),
                       k, var, scores, loadings, removed)
