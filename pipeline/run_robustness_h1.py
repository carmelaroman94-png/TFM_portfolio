"""Sensibilidad de H1 a la configuración de clustering: η² intra/entre grupos para configuraciones alternativas."""
import numpy as np
import pandas as pd

from inflacion import clustering as cl
from inflacion import targets
from inflacion.config import Paths, load_params

P = load_params()
OUT = Paths.processed
scores = pd.read_parquet(OUT / "pca_scores.parquet")
lab0 = pd.read_csv(OUT / "clusters.csv", index_col=0)["cluster"]
X = scores.loc[lab0.index, [c for c in scores.columns if c.startswith("MONETARIA_")]]
target = targets.load("principal").loc[X.index]
grid = pd.read_csv(OUT / "clustering_grid.csv").dropna(subset=["ari_bootstrap"])
configs = grid.sort_values("silhouette", ascending=False)
configs = pd.concat([configs, pd.DataFrame([{"algoritmo": "GMM", "parametro": "tied", "k": 4}])]).drop_duplicates(
    ["algoritmo", "parametro", "k"])

rows = []
for r in configs.itertuples():
    lab = pd.Series(cl.fit_labels(X.to_numpy(), r.algoritmo, r.parametro, int(r.k), P["seed"]), index=X.index)
    row = {"algoritmo": r.algoritmo, "parametro": r.parametro, "k": int(r.k),
           "silhouette": getattr(r, "silhouette", np.nan), "ari_bootstrap": getattr(r, "ari_bootstrap", np.nan)}
    for stat, f in [("media", np.log), ("tendencia", np.arcsinh), ("volatilidad", np.log)]:
        v = target[stat].dropna()
        eta, p = cl.permutation_eta2(f(v).to_numpy(), lab.loc[v.index].to_numpy(), 2000, P["seed"])
        row[f"eta2_{stat}"], row[f"p_{stat}"] = eta, p
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(OUT / "h1_sensibilidad.csv", index=False)
print(res.round(3).to_string(index=False))
