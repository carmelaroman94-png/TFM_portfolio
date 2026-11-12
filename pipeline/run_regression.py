"""H2/H3: modelo monetario (A) vs. modelo con las cuatro dimensiones (B), por estadístico de trayectoria."""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from inflacion import regression as rg
from inflacion.config import Paths, load_params

P = load_params()
R = P["regresion"]
OUT = Paths.processed
SEED = P["seed"]
DIMS = P["pca"]["dimensiones"]
NON_MON = [d for d in DIMS if d != "MONETARIA"]

# Variantes de robustez: target secundario (deflactor), tendencia Theil-Sen, muestra con atípicos.
variant = sys.argv[1] if len(sys.argv) > 1 else "principal"
N_BOOT = R["n_bootstrap"] if variant == "principal" else 500
N_PERM = 2000 if variant == "principal" else 500

pca = pickle.load(open(OUT / "pca_resultados.pkl", "rb"))
scores = pd.read_parquet(OUT / "pca_scores.parquet")
sample = pd.read_csv(OUT / "clusters.csv", index_col=0).index
if variant == "con_atipicos":
    sample = scores.index
target = pd.read_csv(OUT / ("target_secundario.csv" if variant == "deflactor" else "target_principal.csv"), index_col="iso3")
if variant == "theil_sen":
    target["tendencia"] = target["tendencia_theil_sen"]

TRANSFORM = {"media": np.log, "tendencia": np.arcsinh, "volatilidad": np.log}

rows_diag, rows_h2, rows_h3, coefs = [], [], [], []
for stat in P["trayectorias"]["estadisticos"]:
    y_raw = target.loc[sample, stat].dropna()
    transform = TRANSFORM[stat]
    if stat == "media" and not (y_raw > 0).all():
        transform = np.arcsinh
        print(f"media: {(y_raw <= 0).sum()} valores no positivos, se usa asinh en lugar de log")
    y = transform(y_raw).rename(stat)
    blocks = {}
    for d in DIMS:
        k = rg.n_components_floor(pca[f"{d}_{stat}"].varianza, R["piso_varianza_componentes"])
        blocks[d] = [f"{d}_{stat}__PC{i + 1}" for i in range(k)]
    X_B = scores.loc[y.index, sum(blocks.values(), [])]
    X_A = X_B[blocks["MONETARIA"]]
    assert len(y) / X_B.shape[1] >= R["obs_por_predictor"], f"{stat}: menos de {R['obs_por_predictor']} obs/predictor"

    for name, X in [("A", X_A), ("B", X_B)]:
        rows_diag.append({"estadistico": stat, "modelo": name, **rg.diagnostics(y, X)})
    mB = rg.fit(y, X_B, robust=True)
    coefs.append(pd.DataFrame({"estadistico": stat, "termino": mB.params.index, "coef": mB.params.values,
                               "se_HC3": mB.bse.values, "p_HC3": mB.pvalues.values}))

    tests = [("H2", d, blocks[d]) for d in NON_MON] + [("H3", "NO_MONETARIAS", sum((blocks[d] for d in NON_MON), []))]
    for hyp, label, drop in tests:
        res = {"estadistico": stat, "bloque": label, "n_comp": len(drop), **rg.nested_f(y, X_B, drop)}
        res.update(rg.bootstrap_nested(y, X_B, drop, N_BOOT, SEED, R["alpha"]))
        res["p_permutacion"] = rg.permutation_block(y, X_B, drop, N_PERM, SEED)
        (rows_h2 if hyp == "H2" else rows_h3).append(res)

diag, h2, h3, coefs = pd.DataFrame(rows_diag), pd.DataFrame(rows_h2), pd.DataFrame(rows_h3), pd.concat(coefs)
for df in (h2, h3):
    df["p_HC3_holm"] = multipletests(df["p_HC3"], method="holm")[1]
    df["solido"] = (df["p_HC3"] < R["alpha"]) & (df["p_permutacion"] < R["alpha"]) & (df["delta_ic95_inf"] > 0)

suffix = "" if variant == "principal" else f"_{variant}"
diag.to_csv(OUT / f"reg_diagnosticos{suffix}.csv", index=False)
h2.to_csv(OUT / f"reg_h2{suffix}.csv", index=False)
h3.to_csv(OUT / f"reg_h3{suffix}.csv", index=False)
coefs.to_csv(OUT / f"reg_coeficientes{suffix}.csv", index=False)

pd.set_option("display.width", 250)
cols = ["estadistico", "bloque", "n_comp", "F", "p", "p_HC3", "p_HC3_holm", "p_permutacion", "delta_r2adj",
        "delta_ic95_inf", "delta_ic95_sup", "pct_significativo", "solido"]
print(f"== {variant}")
print(diag.round(4).to_string(index=False))
print(h2[cols].round(4).to_string(index=False))
print(h3[cols + ["r2adj_restringido", "r2adj_completo"]].round(4).to_string(index=False))
