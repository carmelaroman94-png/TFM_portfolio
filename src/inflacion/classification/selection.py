from __future__ import annotations

import pandas as pd

from inflacion.config import load_params


def is_absolute_level(codigo: str, nombre: str, params: dict | None = None) -> bool:
    cfg = (params or load_params())["comparabilidad"]
    by_code = any(t in set(cfg["tokens_codigo_absoluto"]) for t in codigo.split("."))
    by_name = pd.Series([nombre]).str.contains(cfg["patron_nombre_absoluto"], regex=True).iloc[0]
    relative = pd.Series([nombre]).str.contains(cfg["patron_nombre_relativo"], regex=True).iloc[0]
    return bool((by_code or by_name) and not relative)


def finalize(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Aplica variables protegidas y criterio de comparabilidad; define la columna `incluir`."""
    params = params or load_params()
    out = df.copy()
    out["dimension_llm_o_revision"] = out["dimension"]
    out["motivo_exclusion"] = ""

    for dim, codes in params["clasificacion"]["variables_protegidas"].items():
        present = out.index.intersection(codes)
        changed = present[out.loc[present, "dimension"] != dim]
        out.loc[changed, "metodo"] = out.loc[changed, "metodo"] + " | sobrescrito por variable protegida"
        out.loc[present, "dimension"] = dim
        out.loc[present, "protegida"] = True
    out["protegida"] = out["protegida"].fillna(False).astype(bool)

    absolute = pd.Series([is_absolute_level(c, n, params) for c, n in zip(out.index, out["nombre_indicador"].fillna(""))],
                         index=out.index)
    out.loc[absolute & ~out["protegida"], "motivo_exclusion"] = "nivel_absoluto_no_comparable"
    prices = out.index.isin(params["targets"]["excluir_de_features"])
    out.loc[prices, "motivo_exclusion"] = "medida_de_precios"
    if params["clasificacion"]["excluir_ambiguo"]:
        amb = (out["dimension"] == "AMBIGUO") & (out["motivo_exclusion"] == "")
        out.loc[amb, "motivo_exclusion"] = "ambiguo"
    out["incluir"] = out["motivo_exclusion"] == ""
    return out
