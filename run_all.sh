#!/usr/bin/env bash
# Pipeline completo. La clasificación usa las respuestas de Groq en caché (data/interim/classification):
# sin caché, requiere GROQ_API_KEY. Los datos crudos se obtienen con pipeline/download_data.py.
set -euo pipefail
cd "$(dirname "$0")"
python3 pipeline/classify_sample.py
python3 pipeline/classify_universe.py data/raw/wdi/wdi_metadata_20261111.csv wdi
python3 pipeline/classify_universe.py data/raw/vdem/vdem_79_metadata_v16.csv vdem
python3 pipeline/consolidate_classification.py
python3 pipeline/build_dataset.py
python3 pipeline/build_features.py
python3 pipeline/run_pca.py
python3 pipeline/run_clustering.py
python3 pipeline/run_robustness_h1.py
for v in principal deflactor theil_sen mediana con_atipicos tasas_nominales; do python3 pipeline/run_regression.py "$v"; done
for k in 2 3; do python3 pipeline/run_regression.py principal --k "$k"; done
python3 pipeline/run_regression_cv.py
python3 pipeline/build_dashboards.py
