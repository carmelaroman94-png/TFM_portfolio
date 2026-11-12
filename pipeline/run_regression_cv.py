"""Validación predictiva de H3: R² fuera de muestra del modelo con las cuatro dimensiones frente al monetario.

Dos especificaciones: (1) OLS con el mismo número de componentes por dimensión que la inferencia principal;
(2) ridge con todos los componentes retenidos por el PCA (70% de varianza por bloque), que pone a prueba H3 frente
a un control monetario completo sin depender de cuántos componentes se elijan.
"""
import pickle

import numpy as np
import pandas as pd

from inflacion import regression as rg
from inflacion import targets
from inflacion.config import Paths, load_params

P = load_params()
R = P["regresion"]
OUT = Paths.processed
DIMS = P["pca"]["dimensiones"]
NON_MON = [d for d in DIMS if d != "MONETARIA"]
TRANSFORM = {"media": np.log, "tendencia": np.arcsinh, "volatilidad": np.log}
N_SPLITS, N_REP = 10, 50

pca = pickle.load(open(OUT / "pca_resultados.pkl", "rb"))
scores = pd.read_parquet(OUT / "pca_scores.parquet")
sample = pd.read_csv(OUT / "clusters.csv", index_col=0).index
target = targets.load("principal")

rows = []
for stat in P["trayectorias"]["estadisticos"]:
    y = TRANSFORM[stat](target.loc[sample, stat].dropna())
    for spec, ridge in [("ols_k_balanceado", False), ("ridge_todos_los_componentes", True)]:
        blocks = {}
        for d in DIMS:
            res = pca[f"{d}_{stat}"]
            k = res.n_componentes if ridge else min(R["componentes_por_bloque"], res.n_componentes)
            blocks[d] = [f"{d}_{stat}__PC{i + 1}" for i in range(k)]
        X = scores.loc[y.index, sum(blocks.values(), [])]
        drop = sum((blocks[d] for d in NON_MON), [])
        cv = rg.cv_delta_r2(y, X, drop, N_SPLITS, N_REP, P["seed"], ridge=ridge)
        rows.append({"estadistico": stat, "especificacion": spec, "n": len(y),
                     "comp_monetarios": len(blocks["MONETARIA"]), "comp_no_monetarios": len(drop),
                     "r2_oos_monetario": cv["r2_oos_restringido"].mean(), "r2_oos_completo": cv["r2_oos_completo"].mean(),
                     "delta_r2_oos": cv["delta_r2_oos"].mean(),
                     "delta_p05": cv["delta_r2_oos"].quantile(0.05), "delta_p95": cv["delta_r2_oos"].quantile(0.95),
                     "pct_reps_mejora": 100 * (cv["delta_r2_oos"] > 0).mean()})

res = pd.DataFrame(rows)
res.to_csv(OUT / "reg_cv.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
