from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests

from inflacion.config import Paths

API = "https://api.worldbank.org/v2"
BULK_URL = "https://databankfiles.worldbank.org/public/ddpext_download/WDI_CSV.zip"


def panel_from_bulk(zip_path: Path, codes: set[str], iso3: set[str], years: range) -> pd.DataFrame:
    """Lee WDICSV.csv del zip masivo y devuelve el panel largo (iso3, anio, codigo, valor)."""
    import zipfile

    year_cols = [str(y) for y in years]
    chunks = []
    with zipfile.ZipFile(zip_path) as z, z.open("WDICSV.csv") as fh:
        for chunk in pd.read_csv(fh, chunksize=200_000):
            chunk = chunk[chunk["Indicator Code"].isin(codes) & chunk["Country Code"].isin(iso3)]
            if len(chunk):
                chunks.append(chunk[["Country Code", "Indicator Code"] + [c for c in year_cols if c in chunk]])
    wide = pd.concat(chunks)
    long = wide.melt(id_vars=["Country Code", "Indicator Code"], var_name="anio", value_name="valor").dropna()
    long["anio"] = long["anio"].astype(int)
    return long.rename(columns={"Country Code": "iso3", "Indicator Code": "codigo"})[["iso3", "anio", "codigo", "valor"]]
WDI_SOURCE_ID = 2


def fetch_metadata(out_dir: Path = Paths.raw / "wdi", timeout: int = 120) -> pd.DataFrame:
    """Descarga el catálogo de indicadores de WDI (fuente 2) y lo guarda con fecha de descarga."""
    r = requests.get(f"{API}/indicator", params={"source": WDI_SOURCE_ID, "format": "json", "per_page": 5000},
                     timeout=timeout)
    r.raise_for_status()
    header, items = r.json()
    if header["total"] != len(items):
        raise RuntimeError(f"Catálogo incompleto: {len(items)} de {header['total']}")

    df = pd.DataFrame({
        "codigo": [i["id"] for i in items],
        "nombre_indicador": [i["name"] for i in items],
        "descripcion_metadata_original": [(i.get("sourceNote") or "").strip() for i in items],
        "topicos": ["; ".join(t["value"].strip() for t in i.get("topics", []) if t.get("value")) for i in items],
        "organizacion_fuente": [(i.get("sourceOrganization") or "").strip() for i in items],
        "unidad": [(i.get("unit") or "").strip() for i in items],
    })
    df["fuente"] = "WDI"

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"wdi_metadata_{date.today():%Y%m%d}.csv", index=False)
    return df
