"""Clasifica el universo de indicadores con el codebook elegido, en dos corridas independientes."""
import sys

import pandas as pd

from inflacion.classification.llm import LLMConfig, classify
from inflacion.config import Paths

CODEBOOK = "v2"
source_csv = sys.argv[1]
label = sys.argv[2]

universe = pd.read_csv(source_csv)
cfg = LLMConfig(max_workers=8)
for run in ["r1", "r2"]:
    res = classify(universe, CODEBOOK, f"{label}_{run}", cfg)
    print(run, res["dimension_final"].value_counts().to_dict(), flush=True)
