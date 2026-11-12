"""H1: clustering sobre el perfil monetario-fiscal y heterogeneidad de trayectorias inflacionarias intra-grupo."""
import json

import numpy as np
import pandas as pd

from inflacion import clustering as cl
from inflacion.config import Paths, load_params
from inflacion.sources.countries import un_members

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

# Mismas transformaciones que en las regresiones: la tendencia cruda tiene colas extremas.
TRANSFORM = {"media": np.log, "tendencia": np.arcsinh, "volatilidad": np.log}
region = un_members().set_index("iso3")["region"]
tests, ref = [], []
for stat in ["media", "tendencia", "volatilidad"]:
    v = y[stat].dropna()
    lab = labels.loc[v.index]
    h, p, eps = cl.epsilon_squared([v[lab == k].to_numpy() for k in sorted(lab.unique())])
    yv = TRANSFORM[stat](v)
    eta, p_perm = cl.permutation_eta2(yv.to_numpy(), lab.to_numpy(), 5000, SEED)
    tests.append({"estadistico": stat, "n": len(v), "kruskal_H": h, "kruskal_p": p, "epsilon2": eps,
                  "eta2_transformado": eta, "p_permutacion": p_perm, "no_explicado_pct": 100 * (1 - eta)})
    # Referencia: cuánto explica la región ONU, con el mismo número de países.
    eta_r, p_r = cl.permutation_eta2(yv.to_numpy(), region.loc[v.index].to_numpy(), 5000, SEED)
    ref.append({"estadistico": stat, "eta2_perfil": eta, "eta2_region": eta_r, "p_region": p_r})
tests = pd.DataFrame(tests)
ct = pd.crosstab(labels, region.loc[labels.index])
chi2 = float(((ct - np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum()) ** 2 /
              (np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum())).values.sum())
cramer_v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
ref = pd.DataFrame(ref).assign(cramer_v_perfil_region=cramer_v)

g.to_csv(OUT / "clustering_grid.csv", index=False)
labels.to_csv(OUT / "clusters.csv")
dist.to_frame().assign(atipico=dist > thr).to_csv(OUT / "clustering_distancias.csv")
disp.to_csv(OUT / "h1_dispersion.csv", index=False)
tests.to_csv(OUT / "h1_tests.csv", index=False)
ref.to_csv(OUT / "h1_region.csv", index=False)
ct.to_csv(OUT / "h1_perfil_por_region.csv")
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
print(ref.round(3).to_string(index=False))
print(ct)
