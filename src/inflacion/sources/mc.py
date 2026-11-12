"""Índice Herfindahl-Hirschman de concentración de mercado de exportaciones (WITS, Banco Mundial)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from inflacion.config import Paths

URL = ("https://wits.worldbank.org/API/V1/SDMX/V21/datasource/tradestats-trade/"
       "reporter/all/year/all/indicator/HH-MKT-CNCNTRTN-NDX?format=JSON")
CODE = "hh_market_concentration"
# WITS conserva códigos históricos para mantener la continuidad de las series.
HISTORICAL_TO_ISO3 = {"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "MNT": "MNE", "SER": "SRB", "SUD": "SDN"}


def download(path: Path = Paths.raw / "mc_wits_raw.json") -> Path:
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def to_long(path: Path = Paths.raw / "mc_wits_raw.json") -> pd.DataFrame:
    data = json.loads(path.read_text())
    dims = data["structure"]["dimensions"]
    rep = next(d for d in dims["series"] if d["id"] == "REPORTER")
    time = next(d for d in dims["observation"] if d["id"] == "TIME_PERIOD")
    rep_codes = [v["id"].upper() for v in rep["values"]]
    years = [int(v["id"]) for v in time["values"]]

    rows = []
    for key, content in data["dataSets"][0]["series"].items():
        reporter = rep_codes[int(key.split(":")[rep["keyPosition"]])]
        for idx, obs in content.get("observations", {}).items():
            rows.append((reporter, years[int(idx)], obs[0]))
    df = pd.DataFrame(rows, columns=["codigo_origen", "anio", "valor"])
    df["iso3"] = df["codigo_origen"].replace(HISTORICAL_TO_ISO3)
    # SUD (Sudán pre-2011) solo completa años sin dato de SDN.
    df["prioridad"] = (df["codigo_origen"] == "SUD").astype(int)
    df = df.sort_values("prioridad").drop_duplicates(["iso3", "anio"], keep="first")
    df["codigo"] = CODE
    return df[["iso3", "anio", "codigo", "valor"]]
