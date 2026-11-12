"""Regresiones anidadas (H2/H3): contribución de las dimensiones no monetarias sobre un modelo monetario."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fit(y: pd.Series, X: pd.DataFrame, robust: bool = True):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3" if robust else "nonrobust")


def nested_f(y: pd.Series, X_full: pd.DataFrame, drop: list[str]) -> dict:
    """F clásico anidado y Wald robusto HC3 para la restricción conjunta de que `drop` vale cero."""
    full_ols = fit(y, X_full, robust=False)
    restricted = fit(y, X_full.drop(columns=drop), robust=False)
    f, p, df_diff = full_ols.compare_f_test(restricted)
    full_hc3 = fit(y, X_full, robust=True)
    R = np.zeros((len(drop), full_hc3.params.size))
    for i, c in enumerate(drop):
        R[i, list(full_hc3.params.index).index(c)] = 1
    w = full_hc3.f_test(R)
    return {"F": float(f), "p": float(p), "gl": int(df_diff), "F_HC3": float(np.squeeze(w.fvalue)),
            "p_HC3": float(w.pvalue), "r2adj_completo": full_ols.rsquared_adj,
            "r2adj_restringido": restricted.rsquared_adj,
            "delta_r2adj": full_ols.rsquared_adj - restricted.rsquared_adj}


def effect_size(y: pd.Series, X_full: pd.DataFrame, drop: list[str]) -> dict:
    """f² de Cohen y R² parcial del bloque: varianza que explica sobre la que deja sin explicar el modelo restringido."""
    full = fit(y, X_full, robust=False)
    restricted = fit(y, X_full.drop(columns=drop), robust=False)
    delta = full.rsquared - restricted.rsquared
    return {"r2_completo": full.rsquared, "r2_restringido": restricted.rsquared, "delta_r2": delta,
            "f2": delta / (1 - full.rsquared), "r2_parcial": delta / (1 - restricted.rsquared)}


def cv_delta_r2(y: pd.Series, X_full: pd.DataFrame, drop: list[str], n_splits: int, n_rep: int, seed: int,
                ridge: bool = False) -> pd.DataFrame:
    """R² fuera de muestra (K-fold repetido) del modelo completo y del restringido, sobre las mismas particiones.

    Con ridge=True ambos modelos se estiman con ridge y penalización elegida por validación cruzada interna,
    lo que permite usar todos los componentes retenidos sin sobreajuste.
    """
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model():
        if ridge:
            return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        return LinearRegression()

    yv, Xf, Xr = y.to_numpy(), X_full.to_numpy(), X_full.drop(columns=drop).to_numpy()
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_rep, random_state=seed)
    pred_f, pred_r = np.zeros((n_rep, len(yv))), np.zeros((n_rep, len(yv)))
    for i, (tr, te) in enumerate(rkf.split(Xf)):
        rep = i // n_splits
        pred_f[rep, te] = model().fit(Xf[tr], yv[tr]).predict(Xf[te])
        pred_r[rep, te] = model().fit(Xr[tr], yv[tr]).predict(Xr[te])
    return pd.DataFrame({"r2_oos_completo": [r2_score(yv, p) for p in pred_f],
                         "r2_oos_restringido": [r2_score(yv, p) for p in pred_r]}).assign(
        delta_r2_oos=lambda d: d["r2_oos_completo"] - d["r2_oos_restringido"])


def diagnostics(y: pd.Series, X: pd.DataFrame) -> dict:
    m = fit(y, X, robust=False)
    Xc = sm.add_constant(X)
    vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    _, bp_p, _, _ = het_breuschpagan(m.resid, Xc)
    return {"n": int(m.nobs), "k": X.shape[1], "r2": m.rsquared, "r2adj": m.rsquared_adj,
            "F_global_p": float(m.f_pvalue), "vif_max": float(np.max(vif)), "breusch_pagan_p": float(bp_p),
            "shapiro_p": float(stats.shapiro(m.resid).pvalue), "jarque_bera_p": float(stats.jarque_bera(m.resid).pvalue)}


def bootstrap_nested(y: pd.Series, X_full: pd.DataFrame, drop: list[str], n_boot: int, seed: int, alpha: float) -> dict:
    """Bootstrap de pares: IC percentil 95% del ΔR² ajustado y % de remuestras con F significativo."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas, signif = [], []
    yv, Xf, Xr = y.to_numpy(), sm.add_constant(X_full).to_numpy(), sm.add_constant(X_full.drop(columns=drop)).to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mf, mr = sm.OLS(yv[idx], Xf[idx]).fit(), sm.OLS(yv[idx], Xr[idx]).fit()
        deltas.append(mf.rsquared_adj - mr.rsquared_adj)
        signif.append(mf.compare_f_test(mr)[1] < alpha)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_ic95_inf": float(lo), "delta_ic95_sup": float(hi), "pct_significativo": 100 * float(np.mean(signif))}


def permutation_block(y: pd.Series, X_full: pd.DataFrame, drop: list[str], n_perm: int, seed: int) -> float:
    """Permuta conjuntamente las filas del bloque evaluado (manteniendo y y el resto fijos); p-valor del F."""
    rng = np.random.default_rng(seed)
    keep = X_full.drop(columns=drop)
    f_obs = nested_f(y, X_full, drop)["F"]
    yv, Xr = y.to_numpy(), sm.add_constant(keep).to_numpy()
    mr = sm.OLS(yv, Xr).fit()
    B = X_full[drop].to_numpy()
    count = 0
    for _ in range(n_perm):
        Xp = np.column_stack([Xr, B[rng.permutation(len(B))]])
        f = sm.OLS(yv, Xp).fit().compare_f_test(mr)[0]
        count += f >= f_obs
    return float((count + 1) / (n_perm + 1))
