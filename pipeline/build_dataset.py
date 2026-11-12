"""Construye el panel largo país-año-indicador (WDI + V-Dem + MC) para los 193 miembros de la ONU."""
import pandas as pd

from inflacion.config import Paths, load_params
from inflacion.sources import countries, mc, vdem, wdi

P = load_params()
YEARS = range(1990, 2025)

iso3 = set(countries.un_members()["iso3"])
clasif = pd.read_csv(Paths.processed / "classification" / "clasificacion_final.csv")
targets = [P["targets"]["principal"], P["targets"]["secundario"]]
features = clasif.loc[clasif["incluir"]]

wdi_codes = set(features.loc[features["fuente"] == "WDI", "codigo"]) | set(targets)
vdem_codes = list(features.loc[features["fuente"] == "V-Dem", "codigo"])

zip_path = sorted((Paths.raw / "wdi").glob("WDI_CSV_*.zip"))[-1]
panel = pd.concat([
    wdi.panel_from_bulk(zip_path, wdi_codes, iso3, YEARS).assign(fuente="WDI"),
    vdem.panel_from_rdata(Paths.raw / "vdem" / "vdem_v16.RData", vdem_codes, iso3, YEARS).assign(fuente="V-Dem"),
    mc.to_long().query("iso3 in @iso3 and anio >= @YEARS.start and anio < @YEARS.stop").assign(fuente="MC"),
], ignore_index=True)

dim = features.set_index("codigo")["dimension"].to_dict()
panel["dimension"] = panel["codigo"].map(dim).fillna("TARGET")
panel.loc[panel["codigo"].isin(targets), "dimension"] = "TARGET"
panel.to_parquet(Paths.processed / "panel_largo.parquet", index=False)

print(panel.groupby("fuente").agg(obs=("valor", "size"), indicadores=("codigo", "nunique"), paises=("iso3", "nunique")))
missing = sorted(wdi_codes - set(panel["codigo"]))
print(f"Códigos WDI sin datos en el bulk: {len(missing)}", missing[:10])
