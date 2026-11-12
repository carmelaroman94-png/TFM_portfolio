"""H2/H3: modelo monetario (A) vs. modelo con las cuatro dimensiones (B), por estadístico de trayectoria.

Cada dimensión entra con el mismo número de componentes principales, para que las pruebas de H2 comparen
bloques con los mismos grados de libertad y el control monetario no quede en desventaja frente al resto.

Uso: run_regression.py [variante] [--k N]
  variantes: principal, deflactor, theil_sen, mediana, con_atipicos, tasas_nominales
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from statsmodels.stats.multitest import multipletests

from inflacion import preprocessing as pp
from inflacion import regression as rg
from inflacion import targets
from inflacion.config import Paths, load_params

P = load_params()
R = P["regresion"]
OUT = Paths.processed
SEED = P["seed"]
DIMS = P["pca"]["dimensiones"]
NON_MON = [d for d in DIMS if d != "MONETARIA"]

ap = argparse.ArgumentParser()
ap.add_argument("variant", nargs="?", default="principal",
                choices=["principal", "deflactor", "theil_sen", "mediana", "con_atipicos", "tasas_nominales"])
ap.add_argument("--k", type=int, default=R["componentes_por_bloque"])
args = ap.parse_args()
variant, K = args.variant, args.k
main_run = variant == "principal" and K == R["componentes_por_bloque"]
N_BOOT = R["n_bootstrap"] if main_run else 500
N_PERM = 2000 if main_run else 500

pca = pickle.load(open(OUT / "pca_resultados.pkl", "rb"))
scores = pd.read_parquet(OUT / "pca_scores.parquet")
sample = pd.read_csv(OUT / "clusters.csv", index_col=0).index
if variant == "con_atipicos":
    sample = scores.index
target = targets.load("secundario" if variant == "deflactor" else "principal")
if variant == "theil_sen":
    target["tendencia"] = target["tendencia_theil_sen"]
if variant == "mediana":
    target["media"] = target["mediana"]

extra_controls = pd.DataFrame(index=scores.index)
if variant == "tasas_nominales":
    # Tasas nominales estandarizadas e imputadas con los mismos vecinos que el resto de las features.
    Z = pd.read_parquet(OUT / "features_imputadas_z.parquet")
    rates, _, _ = pp.standardize(pd.read_parquet(OUT / "features_tasas_nominales.parquet").reindex(Z.index))
    k_knn = int(pd.read_csv(OUT / "imputacion_cv.csv").iloc[0]["k"])
    joint = KNNImputer(n_neighbors=k_knn).fit_transform(pd.concat([Z, rates], axis=1))
    extra_controls = pd.DataFrame(joint[:, Z.shape[1]:], index=Z.index, columns=rates.columns)

TRANSFORM = {"media": np.log, "tendencia": np.arcsinh, "volatilidad": np.log}

rows_diag, rows_h2, rows_h3, rows_blocks, coefs = [], [], [], [], []
for stat in P["trayectorias"]["estadisticos"]:
    y_raw = target.loc[sample, stat].dropna()
    transform = TRANSFORM[stat]
    if stat == "media" and not (y_raw > 0).all():
        transform = np.arcsinh
        print(f"media: {(y_raw <= 0).sum()} valores no positivos, se usa asinh en lugar de log")
    y = transform(y_raw).rename(stat)

    blocks = {}
    for d in DIMS:
        res = pca[f"{d}_{stat}"]
        k = min(K, res.n_componentes)
        blocks[d] = [f"{d}_{stat}__PC{i + 1}" for i in range(k)]
        rows_blocks.append({"estadistico": stat, "dimension": d, "n_comp": k,
                            "varianza_capturada": float(np.sum(res.varianza[:k])), "n_variables": len(res.variables)})
    controls = [c for c in extra_controls.columns if c.endswith(f"__{stat}")]
    X_B = pd.concat([scores.loc[y.index, sum(blocks.values(), [])], extra_controls.loc[y.index, controls]], axis=1)
    X_A = X_B.drop(columns=sum((blocks[d] for d in NON_MON), []))
    obs_pred = len(y) / X_B.shape[1]
    if obs_pred < R["obs_por_predictor"]:
        print(f"AVISO {stat}: {obs_pred:.1f} observaciones por predictor (< {R['obs_por_predictor']})")

    for name, X in [("A", X_A), ("B", X_B)]:
        rows_diag.append({"estadistico": stat, "modelo": name, **rg.diagnostics(y, X)})
    mB = rg.fit(y, X_B, robust=True)
    coefs.append(pd.DataFrame({"estadistico": stat, "termino": mB.params.index, "coef": mB.params.values,
                               "se_HC3": mB.bse.values, "p_HC3": mB.pvalues.values}))

    tests = [("H2", d, blocks[d]) for d in NON_MON] + [("H3", "NO_MONETARIAS", sum((blocks[d] for d in NON_MON), []))]
    for hyp, label, drop in tests:
        res = {"estadistico": stat, "bloque": label, "n_comp": len(drop), "n": len(y),
               **rg.nested_f(y, X_B, drop), **rg.effect_size(y, X_B, drop)}
        res.update(rg.bootstrap_nested(y, X_B, drop, N_BOOT, SEED, R["alpha"]))
        res["p_permutacion"] = rg.permutation_block(y, X_B, drop, N_PERM, SEED)
        (rows_h2 if hyp == "H2" else rows_h3).append(res)

diag, h2, h3, coefs = pd.DataFrame(rows_diag), pd.DataFrame(rows_h2), pd.DataFrame(rows_h3), pd.concat(coefs)
for df in (h2, h3):
    df["p_HC3_holm"] = multipletests(df["p_HC3"], method="holm")[1]
    df["solido"] = (df["p_HC3_holm"] < R["alpha"]) & (df["p_permutacion"] < R["alpha"]) & (df["delta_ic95_inf"] > 0)

suffix = ("" if variant == "principal" else f"_{variant}") + ("" if K == R["componentes_por_bloque"] else f"_k{K}")
diag.to_csv(OUT / f"reg_diagnosticos{suffix}.csv", index=False)
h2.to_csv(OUT / f"reg_h2{suffix}.csv", index=False)
h3.to_csv(OUT / f"reg_h3{suffix}.csv", index=False)
coefs.to_csv(OUT / f"reg_coeficientes{suffix}.csv", index=False)
pd.DataFrame(rows_blocks).to_csv(OUT / f"reg_bloques{suffix}.csv", index=False)

pd.set_option("display.width", 250)
cols = ["estadistico", "bloque", "n_comp", "p_HC3", "p_HC3_holm", "p_permutacion", "delta_r2adj", "f2",
        "delta_ic95_inf", "delta_ic95_sup", "solido"]
print(f"== {variant} · k = {K}")
print(pd.DataFrame(rows_blocks).round(3).to_string(index=False))
print(diag.round(4).to_string(index=False))
print(h2[cols].round(4).to_string(index=False))
print(h3[cols + ["r2adj_restringido", "r2adj_completo"]].round(4).to_string(index=False))
