"""Casos extremos: detección por regla, influencia sobre H3 y sensibilidad de los resultados a su tratamiento.

Reglas (cada una tiene su tratamiento; ninguna recorta ni modifica valores de la inflación):
  sin_ipc            < 5 años de IPC en 2000-2020                → fuera de los modelos (la inflación no se imputa)
  serie_corta        5-12 años de IPC (< 60% del período)        → fuera de los modelos principales
  excl_faltantes     > 40% de features sin dato                  → fuera de todo el análisis
  atipico_perfil     distancia del perfil monetario > Q3 + 1,5·RIC → fuera del clustering y de los modelos principales;
                                                                    se reincorporan en la variante «con atípicos»
  inflacion_extrema  log(media) > Q3 + 1,5·RIC                   → se mantiene; log/asinh, HC3 y permutaciones
  episodio_40        algún año con inflación > 40%               → se mantiene; mediana y Theil-Sen como variantes
  brecha_deflactor   media IPC y deflactor difieren más de 2×    → se mantiene; variante con el deflactor
  influyente         distancia de Cook > 4/n en el modelo de H3  → se mantiene; reestimación sin ellos y jackknife
"""
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from inflacion import regression as rg
from inflacion import targets
from inflacion.config import Paths, load_params
from inflacion.sources.countries import un_members

P = load_params()
R = P["regresion"]
OUT = Paths.processed
YEARS = range(P["periodo"]["inicio"], P["periodo"]["fin"] + 1)
DIMS = P["pca"]["dimensiones"]
NON_MON = [d for d in DIMS if d != "MONETARIA"]
TRANSFORM = {"media": np.log, "tendencia": np.arcsinh, "volatilidad": np.log, "mediana": np.arcsinh}
K = R["componentes_por_bloque"]

members = un_members().set_index("iso3")
raw = pd.read_csv(OUT / "target_principal.csv", index_col="iso3")
defl = pd.read_csv(OUT / "target_secundario.csv", index_col="iso3")
miss = pd.read_csv(OUT / "missingness_pais.csv", index_col="iso3")["pct_missing"]
dist = pd.read_csv(OUT / "clustering_distancias.csv", index_col=0)
labels = pd.read_csv(OUT / "clusters.csv", index_col=0)["cluster"]
scores = pd.read_parquet(OUT / "pca_scores.parquet")
pca = pickle.load(open(OUT / "pca_resultados.pkl", "rb"))
panel = pd.read_parquet(OUT / "panel_largo.parquet")
cpi = panel[(panel["codigo"] == P["targets"]["principal"]) & panel["anio"].between(YEARS.start, YEARS.stop - 1)]
peak = cpi.groupby("iso3")["valor"].max()

# ------------------------------------------------------------- influencia sobre H3 (especificación principal)
target = targets.load("principal")


def design(stat: str, idx) -> tuple[pd.DataFrame, list[str]]:
    blocks = {d: [f"{d}_{stat}__PC{i + 1}" for i in range(min(K, pca[f"{d}_{stat}"].n_componentes))] for d in DIMS}
    return scores.loc[idx, sum(blocks.values(), [])], sum((blocks[d] for d in NON_MON), [])


def h3(y: pd.Series, stat: str) -> dict:
    X, drop = design(stat, y.index)
    r = rg.nested_f(y, X, drop)
    return {"delta_r2adj": r["delta_r2adj"], "p_HC3": r["p_HC3"], "n": len(y)}


cook, sens = {}, []
for stat in ["media", "volatilidad"]:
    y = TRANSFORM[stat](target.loc[labels.index, stat].dropna())
    X, drop = design(stat, y.index)
    cook[stat] = pd.Series(sm.OLS(y, sm.add_constant(X)).fit().get_influence().cooks_distance[0], index=y.index)
    base = h3(y, stat)
    infl = cook[stat][cook[stat] > 4 / len(y)].index
    jk = pd.DataFrame([h3(y.drop(c), stat) for c in y.index], index=y.index)
    sens += [{"estadistico": stat, "especificacion": "principal", **base},
             {"estadistico": stat, "especificacion": "sin_influyentes", **h3(y.drop(infl), stat)},
             {"estadistico": stat, "especificacion": "jackknife_min", "delta_r2adj": jk["delta_r2adj"].min(),
              "p_HC3": jk.loc[jk["delta_r2adj"].idxmin(), "p_HC3"], "n": len(y) - 1,
              "pais": jk["delta_r2adj"].idxmin()},
             {"estadistico": stat, "especificacion": "jackknife_max", "delta_r2adj": jk["delta_r2adj"].max(),
              "p_HC3": jk.loc[jk["delta_r2adj"].idxmax(), "p_HC3"], "n": len(y) - 1,
              "pais": jk["delta_r2adj"].idxmax()},
             {"estadistico": stat, "especificacion": "jackknife_p_max", "delta_r2adj": np.nan,
              "p_HC3": jk["p_HC3"].max(), "n": len(y) - 1, "pais": jk["p_HC3"].idxmax()}]
sens = pd.DataFrame(sens)

# ------------------------------------------------------------- detección por país
logm = np.log(raw["media"].where(raw["media"] > 0))
q1, q3 = logm.quantile([0.25, 0.75])
lim_infl = float(np.exp(q3 + 1.5 * (q3 - q1)))

df = members[["pais", "region"]].copy()
df["n_anios_ipc"] = raw["n_anios"]
for c in ["media", "mediana", "tendencia", "volatilidad"]:
    df[c] = raw[c]
df["media_deflactor"] = defl["media"]
df["pico_ipc"] = peak
df["pct_features_faltantes"] = miss
df["distancia_perfil"] = dist.iloc[:, 0]
df["cook_media"] = cook["media"]
df["cook_volatilidad"] = cook["volatilidad"]

df["excl_faltantes"] = df["pct_features_faltantes"] > P["paises"]["umbral_missingness_exclusion"]
df["sin_ipc"] = df["n_anios_ipc"].isna() & ~df["excl_faltantes"]
df["serie_corta"] = df["n_anios_ipc"].lt(P["trayectorias"]["min_anios_target"]) & ~df["sin_ipc"] & df["n_anios_ipc"].notna()
df["atipico_perfil"] = dist["atipico"].reindex(df.index).fillna(False).astype(bool)
df["inflacion_extrema"] = df["media"] > lim_infl
df["episodio_40"] = df["pico_ipc"] > 40
ratio = (df["media_deflactor"] / df["media"]).where((df["media"] > 0) & (df["media_deflactor"] > 0))
df["brecha_deflactor"] = (ratio > 2) | (ratio < 0.5)
n_media = cook["media"].size
df["influyente"] = (df["cook_media"] > 4 / n_media) | (df["cook_volatilidad"] > 4 / cook["volatilidad"].size)
df["en_modelos_principales"] = df.index.isin(labels.index) & target["media"].reindex(df.index).notna()

TREAT = {"excl_faltantes": "Excluido de todo el análisis", "sin_ipc": "Fuera de los modelos: la inflación no se imputa",
         "serie_corta": "Fuera de los modelos principales (< 60% de los años)",
         "atipico_perfil": "Fuera del clustering; incluido en la variante «con atípicos»"}


def treatment(r):
    out = [v for k, v in TREAT.items() if r[k]]
    if out:
        return "; ".join(out)
    flags = [k for k in ["inflacion_extrema", "episodio_40", "brecha_deflactor", "influyente"] if r[k]]
    return "Se mantiene; cubierto por variantes de robustez" if flags else ""


df["tratamiento"] = df.apply(treatment, axis=1)
flag_cols = ["excl_faltantes", "sin_ipc", "serie_corta", "atipico_perfil", "inflacion_extrema", "episodio_40",
             "brecha_deflactor", "influyente"]
df["n_reglas"] = df[flag_cols].sum(axis=1)

# ------------------------------------------------------------- distribuciones: asimetría antes y después de transformar
dist_rows = []
for stat in ["media", "tendencia", "volatilidad", "mediana"]:
    v = target.loc[labels.index, stat].dropna()
    tv = TRANSFORM[stat](v) if stat != "mediana" or (v > 0).all() else np.arcsinh(v)
    for esc, s in [("original", v), ("transformada", tv)]:
        dist_rows.append({"estadistico": stat, "escala": esc, "n": len(s), "asimetria": stats.skew(s),
                          "curtosis_exceso": stats.kurtosis(s), "shapiro_p": stats.shapiro(s).pvalue})

df.to_csv(OUT / "extremos_paises.csv", index_label="iso3")
sens.to_csv(OUT / "extremos_sensibilidad.csv", index=False)
pd.DataFrame(dist_rows).to_csv(OUT / "distribuciones_target.csv", index=False)

pd.set_option("display.width", 220)
print(f"Umbral de inflación extrema (Q3 + 1,5·RIC en log): {lim_infl:.1f}%")
print(df[flag_cols].sum().to_string())
print(df[df["n_reglas"] > 0].sort_values("n_reglas", ascending=False)[["pais"] + flag_cols + ["tratamiento"]].head(40).to_string())
print(sens.round(4).to_string(index=False))
print(pd.DataFrame(dist_rows).round(3).to_string(index=False))
