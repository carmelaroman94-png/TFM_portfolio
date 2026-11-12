"""Consolida las dos corridas de clasificación y marca los indicadores inestables para revisión."""
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from inflacion.classification.llm import parse_results
from inflacion.config import Paths

CACHE = Paths.interim / "classification"
OUT = Paths.processed / "classification"
OUT.mkdir(parents=True, exist_ok=True)


def load(label: str, run: str) -> pd.DataFrame:
    recs = [json.loads(l) for l in (CACHE / f"{label}_{run}_v2.jsonl").read_text(encoding="utf-8").splitlines()]
    return parse_results(recs).drop_duplicates("codigo", keep="last").set_index("codigo")


frames = []
for label, meta_path in [("wdi", sorted((Paths.raw / "wdi").glob("wdi_metadata_*.csv"))[-1]),
                         ("vdem", Paths.raw / "vdem" / "vdem_79_metadata_v16.csv")]:
    meta = pd.read_csv(meta_path).set_index("codigo")
    r1, r2 = load(label, "r1"), load(label, "r2")
    df = meta[["fuente", "nombre_indicador"]].join(
        r1[["dimension_final", "confianza", "justificacion"]].add_suffix("_r1")).join(
        r2[["dimension_final", "confianza"]].add_suffix("_r2"))
    df["estable"] = df["dimension_final_r1"] == df["dimension_final_r2"]
    df["dimension"] = df["dimension_final_r1"].where(df["estable"], "REVISAR")
    df["confianza_media"] = df[["confianza_r1", "confianza_r2"]].mean(axis=1)
    frames.append(df)
    print(f"{label}: n={len(df)}  acuerdo={df['estable'].mean():.3f}  "
          f"kappa={cohen_kappa_score(df['dimension_final_r1'], df['dimension_final_r2']):.3f}")

mc = pd.DataFrame({"fuente": ["MC"], "nombre_indicador": ["Herfindahl-Hirschman Market Concentration Index (export)"],
                   "dimension": ["ESTRUCTURAL"], "estable": [True]}, index=pd.Index(["hh_market_concentration"], name="codigo"))
full = pd.concat(frames + [mc])
full.to_csv(OUT / "clasificacion_universo_v2.csv")
print(pd.crosstab(full["fuente"], full["dimension"], margins=True))

from inflacion.classification.review_decisions import apply as apply_review
from inflacion.classification.selection import finalize

final = finalize(apply_review(full))
final.to_csv(OUT / "clasificacion_final.csv")
print("\nMétodo:", final["metodo"].value_counts().to_dict())
print("Exclusiones:", final["motivo_exclusion"].replace("", "incluido").value_counts().to_dict())
print(pd.crosstab(final.loc[final.incluir, "fuente"], final.loc[final.incluir, "dimension"], margins=True))

cols = ["fuente", "nombre_indicador", "dimension", "metodo", "revisor", "justificacion_revision", "dimension_final_r1",
        "dimension_final_r2", "confianza_media", "protegida", "incluir", "motivo_exclusion"]
Paths.tables.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(Paths.tables / "clasificacion_final_v2.xlsx") as w:
    final[cols].to_excel(w, sheet_name="universo")
    final.loc[final["metodo"] == "clasificacion manual", cols].to_excel(w, sheet_name="clasificacion_manual")
