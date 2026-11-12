from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from inflacion.config import Paths

CODEBOOK_URL = "https://v-dem.net/documents/70/codebook_v16.pdf"
_SECTION_START = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S")
_HEADER = re.compile(r"^\s*\d+(?:\.\d+)+\s+(?P<name>.+?)\s+\([A-Z]\)\s+\((?P<code>\w+)\)\s*$")
_CODE_TAIL = re.compile(r"^\s+(?:.*\s)?\(\w+\)\s*$")
_PAGE_HEADER = re.compile(r"^TOC\s+\d+\s*$")
_FIELD = re.compile(r"^(?P<indent>\s{2,8})(?P<field>[A-Z][A-Za-z()\s–-]{1,40}?):\s?(?P<text>.*)$")
_KEEP = ("Question", "Clarification")


def _clean_lines(raw: list[str]) -> list[str]:
    """Une encabezados partidos en dos líneas y elimina los encabezados de página (TOC + sección corriente)."""
    out, i, in_page_header = [], 0, False
    while i < len(raw):
        line = raw[i]
        if _PAGE_HEADER.match(line):
            in_page_header = True
            i += 1
            continue
        if in_page_header:
            in_page_header = bool(line.strip())
            i += 1
            continue
        if (i + 1 < len(raw) and _SECTION_START.match(line) and not _HEADER.match(line)
                and _CODE_TAIL.match(raw[i + 1])):
            line = line.rstrip() + " " + raw[i + 1].strip()
            i += 1
        out.append(line)
        i += 1
    return out


def parse_codebook(txt_path: Path) -> dict[str, dict]:
    """Extrae nombre, pregunta y aclaración de cada variable del codebook de V-Dem (pdftotext -layout)."""
    entries, current, field, indent = {}, None, None, 0
    for line in _clean_lines(txt_path.read_text(encoding="utf-8").split("\n")):
        h = _HEADER.match(line)
        if h and "." * 4 not in line:
            name = re.sub(r"\s+", " ", h["name"]).strip()
            current = entries.setdefault(h["code"], {"nombre_indicador": name, "Question": "", "Clarification": ""})
            field = None
            continue
        if current is None:
            continue
        f = _FIELD.match(line)
        if f:
            field, indent = f["field"].strip(), len(f["indent"])
            if field in _KEEP and not current[field]:
                current[field] = f["text"].strip()
            elif field in _KEEP:
                field = None
            continue
        if field in _KEEP and line.strip() and len(line) - len(line.lstrip()) > indent:
            current[field] += " " + line.strip()
    return entries


def panel_from_rdata(rdata_path: Path, codes: list[str], iso3: set[str], years: range) -> pd.DataFrame:
    """Panel largo desde el dataset V-Dem v16 (paquete oficial vdemdata). country_text_id coincide con ISO3."""
    import pyreadr

    df = next(iter(pyreadr.read_r(str(rdata_path)).values()))
    df = df[df["country_text_id"].isin(iso3) & df["year"].between(years.start, years.stop - 1)]
    long = df[["country_text_id", "year"] + codes].melt(id_vars=["country_text_id", "year"],
                                                        var_name="codigo", value_name="valor").dropna()
    long["anio"] = long["year"].astype(int)
    return long.rename(columns={"country_text_id": "iso3"})[["iso3", "anio", "codigo", "valor"]]


def build_metadata(codes: list[str], txt_path: Path = Paths.raw / "vdem" / "codebook_v16.txt") -> pd.DataFrame:
    entries = parse_codebook(txt_path)
    missing = [c for c in codes if c not in entries]
    if missing:
        raise KeyError(f"Códigos no encontrados en el codebook de V-Dem: {missing}")
    rows = []
    for c in codes:
        e = entries[c]
        desc = " | ".join(re.sub(r"\s+", " ", t) for t in (e["Question"], e["Clarification"]) if t)
        rows.append({"codigo": c, "fuente": "V-Dem", "nombre_indicador": e["nombre_indicador"],
                     "descripcion_metadata_original": desc})
    return pd.DataFrame(rows)
