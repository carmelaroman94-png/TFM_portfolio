"""Descarga los datos crudos: catálogo y base masiva de WDI, V-Dem v16 (paquete oficial vdemdata),
codebook de V-Dem y el índice de concentración de exportaciones (WITS).

Uso: python pipeline/download_data.py
"""
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import requests

from inflacion.config import Paths
from inflacion.sources import mc, wdi

RAW = Paths.raw


def fetch(url: str, dest: Path, timeout: int = 1800) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(f"  {dest.relative_to(Paths.root)} ({dest.stat().st_size / 1e6:.1f} MB)")


print("WDI · catálogo de indicadores")
wdi.fetch_metadata()
print("WDI · base masiva")
fetch(wdi.BULK_URL, RAW / "wdi" / f"WDI_CSV_{date.today():%Y%m%d}.zip")

print("V-Dem v16 · dataset (paquete oficial vdemdata en GitHub) y codebook")
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/vdeminstitute/vdemdata.git", tmp], check=True)
    (RAW / "vdem").mkdir(parents=True, exist_ok=True)
    (RAW / "vdem" / "vdem_v16.RData").write_bytes((Path(tmp) / "data" / "vdem.RData").read_bytes())
fetch("https://v-dem.net/documents/70/codebook_v16.pdf", RAW / "vdem" / "codebook_v16.pdf")
subprocess.run(["pdftotext", "-layout", str(RAW / "vdem" / "codebook_v16.pdf"), str(RAW / "vdem" / "codebook_v16.txt")], check=True)

print("WITS · concentración de mercado de exportaciones")
mc.download()
print("Listo. Siguiente paso: ./run_all.sh")
