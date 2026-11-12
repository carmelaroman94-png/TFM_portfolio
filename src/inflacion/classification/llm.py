"""Clasificación de indicadores en dimensiones teóricas con un LLM servido por Groq.

Reproducibilidad: modelo de pesos abiertos fijado, temperatura 0, semilla fija y
caché por (versión de codebook, código de indicador). Toda respuesta cruda se guarda,
de modo que la clasificación puede auditarse y re-evaluarse sin volver a llamar a la API.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from inflacion.config import Paths

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CODEBOOK_DIR = Path(__file__).parent / "codebooks"
CATEGORIES = ["MONETARIA", "SOCIODEMOGRAFICA", "INSTITUCIONAL", "ESTRUCTURAL", "AMBIGUO"]
CONFIDENCE_FLOOR = 0.60

SYSTEM_TEMPLATE = """Sos un codificador experto que clasifica indicadores estadísticos en dimensiones teóricas de la inflación.
Aplicá estrictamente el siguiente codebook. Respondé SOLO con un objeto JSON válido.

===== CODEBOOK =====
{codebook}
===== FIN CODEBOOK =====

Formato de salida (JSON):
{{
  "codigo": str,
  "dimension_principal": one of {categories},
  "dimension_secundaria": one of {categories} o null,
  "confianza": número entre 0 y 1,
  "mecanismo_conceptual": str,
  "justificacion_conceptual": str (máx. 60 palabras),
  "necesita_revision_humana": bool
}}"""


@dataclass(frozen=True)
class LLMConfig:
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    seed: int = 42
    reasoning_effort: str = "medium"
    max_tokens: int = 4096
    max_workers: int = 6
    max_retries: int = 5


def load_codebook(version: str) -> str:
    return (CODEBOOK_DIR / f"codebook_{version}.md").read_text(encoding="utf-8").replace("\\_", "_")


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        env = Paths.root / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY no definida (variable de entorno o .env).")
    return key


def _user_message(row: pd.Series) -> str:
    payload = {
        "codigo": row["codigo"],
        "fuente": row["fuente"],
        "nombre": row["nombre_indicador"],
        "descripcion": row["descripcion_metadata_original"],
    }
    return "Clasificá este indicador:\n" + json.dumps(payload, ensure_ascii=False)


def _call(system: str, user: str, cfg: LLMConfig, key: str) -> dict:
    body = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "reasoning_effort": cfg.reasoning_effort,
        "max_completion_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(cfg.max_retries):
        try:
            r = requests.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            data = r.json()
            return {"content": data["choices"][0]["message"]["content"], "usage": data.get("usage", {})}
        if r.status_code in (429, 500, 502, 503) or (r.status_code == 400 and "json_validate_failed" in r.text):
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Groq falló tras {cfg.max_retries} intentos")


def classify(df: pd.DataFrame, codebook_version: str, run_name: str, cfg: LLMConfig = LLMConfig()) -> pd.DataFrame:
    """Clasifica cada fila de `df`; reanuda desde caché si existe."""
    cache = Paths.interim / "classification" / f"{run_name}_{codebook_version}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["codigo"]] = rec

    system = SYSTEM_TEMPLATE.format(codebook=load_codebook(codebook_version), categories=CATEGORIES)
    pending = df[~df["codigo"].isin(done)]
    key = _api_key() if len(pending) else None

    with ThreadPoolExecutor(cfg.max_workers) as pool, open(cache, "a", encoding="utf-8") as fh:
        futures = {pool.submit(_call, system, _user_message(row), cfg, key): row["codigo"] for _, row in pending.iterrows()}
        for fut in as_completed(futures):
            codigo = futures[fut]
            res = fut.result()
            rec = {"codigo": codigo, "codebook": codebook_version, "model": cfg.model,
                   "reasoning_effort": cfg.reasoning_effort, "raw": res["content"], "usage": res["usage"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[codigo] = rec

    return parse_results([done[c] for c in df["codigo"]])


def parse_results(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        try:
            out = json.loads(rec["raw"])
        except json.JSONDecodeError:
            out = {}
        dim = str(out.get("dimension_principal", "")).upper().strip()
        dim = dim if dim in CATEGORIES else "AMBIGUO"
        conf = pd.to_numeric(out.get("confianza"), errors="coerce")
        rows.append({
            "codigo": rec["codigo"],
            "codebook": rec["codebook"],
            "dimension_llm": dim,
            "confianza": conf,
            "dimension_final": "AMBIGUO" if (pd.isna(conf) or conf < CONFIDENCE_FLOOR) else dim,
            "dimension_secundaria": out.get("dimension_secundaria"),
            "necesita_revision_humana": out.get("necesita_revision_humana"),
            "justificacion": out.get("justificacion_conceptual"),
            "parse_ok": bool(out),
        })
    return pd.DataFrame(rows)
