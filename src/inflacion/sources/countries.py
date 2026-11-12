from __future__ import annotations

import pandas as pd

from inflacion.config import Paths


def un_members() -> pd.DataFrame:
    """193 Estados miembros de la ONU con códigos ISO3, M49, COW y región (tabla de elaboración propia)."""
    df = pd.read_excel(Paths.raw / "df_regiones_miembros_onu.xlsx")
    return df.rename(columns={"ISO-alpha3": "iso3", "Country or Area": "pais",
                              "Region Name": "region", "Sub-region Name": "subregion"})[
        ["iso3", "pais", "region", "subregion", "M49_country", "cow_code_countryVdem"]]
