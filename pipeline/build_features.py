"""Filtro de cobertura, trayectorias país-indicador, exclusión de países e imputación."""
import json

import pandas as pd

from inflacion import features as ft
from inflacion import preprocessing as pp
from inflacion.config import Paths, load_params
from inflacion.sources.countries import un_members

P = load_params()
YEARS = range(P["periodo"]["inicio"], P["periodo"]["fin"] + 1)
STATS = P["trayectorias"]["estadisticos"]
OUT = Paths.processed

panel = pd.read_parquet(OUT / "panel_largo.parquet")
members = un_members()
targets = {"principal": P["targets"]["principal"], "secundario": P["targets"]["secundario"]}

cov = ft.coverage(panel, P["universo"]["n_paises"], YEARS)
dims = panel.drop_duplicates("codigo").set_index("codigo")[["fuente", "dimension"]]
cov = cov.join(dims, on="codigo")
cov["cumple_umbral"] = (cov["pct_paises"] >= P["cobertura"]["umbral_pct_paises"]) & \
                       (cov["pct_anios_promedio"] >= P["cobertura"]["umbral_pct_anios"])
cov.to_csv(OUT / "cobertura_indicadores.csv", index=False)

traj = ft.trajectories(panel, YEARS, P["trayectorias"]["min_anios"])
traj.to_parquet(OUT / "trayectorias_largo.parquet", index=False)

for name, code in targets.items():
    t = traj[traj["codigo"] == code].set_index("iso3")[["n_anios"] + STATS]
    if name == "principal":
        t = t.join(ft.theil_sen_slopes(panel, code, YEARS, P["trayectorias"]["min_anios"]))
    t.reindex(members["iso3"]).to_csv(OUT / f"target_{name}.csv", index_label="iso3")

keep = cov.loc[cov["cumple_umbral"] & (cov["dimension"] != "TARGET"), "codigo"]
X = ft.wide_features(traj[traj["codigo"].isin(keep)], STATS).reindex(members["iso3"])

miss = pp.country_missingness(X)
excluded = miss[miss > P["paises"]["umbral_missingness_exclusion"]]
X = X.drop(excluded.index)
X = X.loc[:, X.notna().any()]
Z, mu, sd = pp.standardize(X)
Z = Z.loc[:, sd.notna()]

cv = pp.knn_cv(Z, P["imputacion"]["k_candidatos"], P["imputacion"]["n_repeticiones"],
               P["imputacion"]["fraccion_enmascarada"], P["seed"])
k = int(cv.iloc[0]["k"])
Zi = pp.knn_impute(Z, k)

X.to_parquet(OUT / "features_crudas.parquet")
Zi.to_parquet(OUT / "features_imputadas_z.parquet")
cv.to_csv(OUT / "imputacion_cv.csv", index=False)
miss.rename("pct_missing").to_csv(OUT / "missingness_pais.csv", index_label="iso3")

summary = {
    "indicadores_candidatos": int(cov["dimension"].ne("TARGET").sum()),
    "indicadores_cumplen_umbral": int(len(keep)),
    "features": int(Z.shape[1]),
    "paises_excluidos": excluded.round(1).to_dict(),
    "paises_analisis": int(Zi.shape[0]),
    "missing_medio_pct_antes": round(float(pp.country_missingness(ft.wide_features(traj[traj["codigo"].isin(keep)], STATS).reindex(members["iso3"])).mean()), 2),
    "missing_medio_pct_despues": round(float(X.isna().mean().mean() * 100), 2),
    "k_knn": k,
    "rmse_cv": round(float(cv.iloc[0]["mean"]), 3),
}
(OUT / "resumen_features.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(cov[cov["dimension"] != "TARGET"].groupby("dimension")["cumple_umbral"].agg(["size", "sum"]))
print(cv)
