"""PCA por bloque dimensión × estadístico sobre las features imputadas y estandarizadas."""
import pickle

import numpy as np
import pandas as pd

from inflacion.config import Paths, load_params
from inflacion.pca import run_block

P = load_params()
OUT = Paths.processed
Z = pd.read_parquet(OUT / "features_imputadas_z.parquet")
dims = pd.read_csv(OUT / "cobertura_indicadores.csv").set_index("codigo")["dimension"]
protected = P["clasificacion"]["variables_protegidas"]

results, rows = {}, []
for dim in P["pca"]["dimensiones"]:
    for stat in P["trayectorias"]["estadisticos"]:
        cols = [c for c in Z.columns if c.endswith(f"__{stat}") and dims.get(c.rsplit("__", 1)[0]) == dim]
        prot = [f"{c}__{stat}" for c in protected[dim] if f"{c}__{stat}" in cols]
        r = run_block(Z[cols], f"{dim}_{stat}", prot, P["pca"]["varianza_objetivo"], P["pca"]["kmo_objetivo"],
                      P["pca"]["min_variables"])
        results[r.name] = r
        rows.append({"bloque": r.name, "dimension": dim, "estadistico": stat, "n_inicial": r.n_inicial,
                     "umbral_corr": r.umbral_correlacion, "n_post_corr": r.n_post_correlacion,
                     "n_final": len(r.variables), "protegidas": len(r.protegidas), "kmo": round(r.kmo, 3),
                     "bartlett_p": r.bartlett_p, "n_comp_70": r.n_componentes,
                     "var_pc1": round(r.varianza[0], 3), "var_acum": round(float(np.sum(r.varianza[:r.n_componentes])), 3)})

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "pca_resumen.csv", index=False)
scores = pd.concat([r.scores for r in results.values()], axis=1)
scores.to_parquet(OUT / "pca_scores.parquet")
with open(OUT / "pca_resultados.pkl", "wb") as fh:
    pickle.dump(results, fh)
pd.set_option("display.width", 220)
print(summary.to_string(index=False))
