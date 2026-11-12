"""Evalúa versiones del codebook contra la muestra de 100 indicadores clasificados manualmente."""
import re
import sys

import pandas as pd

from inflacion.classification import evaluation as ev
from inflacion.classification.llm import classify, load_codebook
from inflacion.config import Paths

VERSIONS = sys.argv[1:] or ["v0", "v1", "v2"]

sample = pd.read_excel(Paths.raw / "classification" / "muestra_100_clasificacion_manual.xlsx")
sample["codigo"] = sample["codigo"].str.strip()


def cited_in(version: str) -> set[str]:
    text = load_codebook(version)
    return {c for c in sample["codigo"] if re.search(r"(?<![\w.])" + re.escape(c) + r"(?![\w.])", text)}


cited_any = set().union(*(cited_in(v) for v in ["v0", "v1", "v2"]))
clean = ~sample["codigo"].isin(cited_any)

preds, rows = {}, []
for v in VERSIONS:
    res = classify(sample, v, "muestra100").set_index("codigo").loc[sample["codigo"]].reset_index()
    preds[v] = res
    y, yhat = sample["clasificacion_humana"], res["dimension_final"]
    for subset, mask in [("100", slice(None)), (f"{clean.sum()}_sin_citados", clean)]:
        rows.append({"codebook": v, "subconjunto": subset, **ev.summary(y[mask], yhat[mask])})

out = Paths.processed / "classification"
out.mkdir(parents=True, exist_ok=True)
metrics = pd.DataFrame(rows)
metrics.to_csv(out / "muestra100_metricas.csv", index=False)
wide = sample.copy()
for v, res in preds.items():
    wide[f"llm_{v}"] = res["dimension_final"].values
    wide[f"conf_{v}"] = res["confianza"].values
wide["citado_en_codebook"] = ~clean
# Segunda corrida independiente del codebook elegido, para medir la variabilidad entre ejecuciones.
rep = classify(sample, "v2", "muestra100_rep2").set_index("codigo").loc[sample["codigo"]]
wide["llm_v2_rep"] = rep["dimension_final"].values
wide.to_csv(out / "muestra100_predicciones.csv", index=False)

pd.set_option("display.width", 200)
print(metrics.round(3).to_string(index=False))
for v, res in preds.items():
    print(f"\n### {v}"); print(ev.per_class(sample["clasificacion_humana"], res["dimension_final"]).round(2))
    print(ev.confusion(sample["clasificacion_humana"], res["dimension_final"]))
