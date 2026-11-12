"""H1: clustering sobre el perfil monetario-fiscal y heterogeneidad de trayectorias inflacionarias intra-grupo."""
import json

import numpy as np
import pandas as pd

from inflacion import clustering as cl
from inflacion.config import Paths, load_params

P = load_params()
OUT = Paths.processed
SEED = P["seed"]
C = P["clustering"]

scores = pd.read_parquet(OUT / "pca_scores.parquet")
X_all = scores[[c for c in scores.columns if c.startswith("MONETARIA_")]]

dist, thr = cl.tukey_outliers(X_all)
outliers = dist[dist > thr].sort_values(ascending=False)
X = X_all.drop(outliers.index)

g = cl.grid(X, C["k_grid"], C["linkages"], C["covarianzas_gmm"], SEED)
eligible = g[(g["tam_min"] >= C["tamanio_minimo"]) & (g["k_efectivo"] == g["k"])].copy()
top = eligible.sort_values("silhouette", ascending=False).head(12).copy()
top["ari_bootstrap"] = [cl.bootstrap_stability(X, r.algoritmo, r.parametro, r.k, SEED) for r in top.itertuples()]
g = g.merge(top[["algoritmo", "parametro", "k", "ari_bootstrap"]], how="left", on=["algoritmo", "parametro", "k"])

stable = top[top["ari_bootstrap"] >= C["ari_minimo"]].sort_values("silhouette", ascending=False)
best = (stable if len(stable) else top).iloc[0]
labels = pd.Series(cl.fit_labels(X.to_numpy(), best.algoritmo, best.parametro, int(best.k), SEED), index=X.index)
# Etiquetas ordenadas por tamaño para que sean estables entre ejecuciones.
order = {old: new for new, old in enumerate(labels.value_counts().index)}
labels = labels.map(order).rename("cluster")

target = pd.read_csv(OUT / "target_principal.csv", index_col="iso3")
y = target.loc[labels.index]
disp = cl.within_dispersion(y, labels, ["media", "tendencia", "volatilidad"])

tests = []
for stat in ["media", "tendencia", "volatilidad"]:
    v = y[stat].dropna()
    lab = labels.loc[v.index]
    h, p, eps = cl.epsilon_squared([v[lab == k].to_numpy() for k in sorted(lab.unique())])
    yv = np.log(v) if stat == "media" else (np.log(v) if stat == "volatilidad" else v)
    eta, p_perm = cl.permutation_eta2(yv.to_numpy(), lab.to_numpy(), 5000, SEED)
    tests.append({"estadistico": stat, "n": len(v), "kruskal_H": h, "kruskal_p": p, "epsilon2": eps,
                  "eta2_transformado": eta, "p_permutacion": p_perm, "no_explicado_pct": 100 * (1 - eta)})
tests = pd.DataFrame(tests)

g.to_csv(OUT / "clustering_grid.csv", index=False)
labels.to_csv(OUT / "clusters.csv")
dist.to_frame().assign(atipico=dist > thr).to_csv(OUT / "clustering_distancias.csv")
disp.to_csv(OUT / "h1_dispersion.csv", index=False)
tests.to_csv(OUT / "h1_tests.csv", index=False)
cent = X.groupby(labels).mean()
cent.to_csv(OUT / "clusters_centroides.csv")
res = {"umbral_tukey": thr, "atipicos": outliers.index.tolist(), "n_cluster": int(len(X)),
       "config": {"algoritmo": best.algoritmo, "parametro": best.parametro, "k": int(best.k),
                  "silhouette": float(best.silhouette), "ari_bootstrap": float(best.ari_bootstrap)}}
(OUT / "clustering_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

pd.set_option("display.width", 220)
print(json.dumps(res, indent=1, ensure_ascii=False))
print(top[["algoritmo", "parametro", "k", "silhouette", "davies_bouldin", "tam_min", "tamanios", "ari_bootstrap"]].round(3).to_string(index=False))
print(labels.value_counts().sort_index().to_dict())
print(disp.round(2).to_string(index=False))
print(tests.round(4).to_string(index=False))
