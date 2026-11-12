from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Paths:
    root = ROOT
    config = ROOT / "config" / "params.yaml"
    raw = ROOT / "data" / "raw"
    interim = ROOT / "data" / "interim"
    processed = ROOT / "data" / "processed"
    figures = ROOT / "reports" / "figures"
    tables = ROOT / "reports" / "tables"


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(Paths.config, encoding="utf-8") as f:
        return yaml.safe_load(f)
