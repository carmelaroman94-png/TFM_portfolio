"""Genera los dashboards interactivos (HTML estático + Plotly.js local) a partir de los resultados procesados."""
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from inflacion import targets
from inflacion.classification.evaluation import confusion
from inflacion.config import Paths, load_params
from inflacion.dashboard import card, fig_json, kpis, page, seg
from inflacion.sources.countries import un_members
from inflacion.viz import CATEGORICAL, DIM_LABELS, DIMENSION_COLORS, apply_theme
from inflacion.viz.theme import DIVERGING, INK_MUTED, INK_SECONDARY, NEUTRAL, SEQUENTIAL_BLUE

apply_theme()
P = load_params()
D = Paths.processed
OUT = Paths.root / "dashboards"
(OUT / "assets").mkdir(parents=True, exist_ok=True)
(OUT / "assets" / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
REPO = P["portfolio"]["repo_url"]
SOURCES = ("Fuentes: Banco Mundial, World Development Indicators (versión 8-oct-2026); V-Dem v16 (Coppedge et al., 2026); "
           "Banco Mundial, WITS (índice Herfindahl-Hirschman de exportaciones).")

M = un_members().set_index("iso3")
REG_ES = {"Africa": "África", "Americas": "América", "Asia": "Asia", "Europe": "Europa", "Oceania": "Oceanía"}
M["region_es"] = M["region"].map(REG_ES)
M["corto"] = M["pais"].str.split(" \\(").str[0].replace({"Democratic Republic of the Congo": "DR Congo", "United Republic of Tanzania": "Tanzania",
    "Democratic People's Republic of Korea": "North Korea", "Bosnia and Herzegovina": "Bosnia-Herzegovina", "Brunei Darussalam": "Brunei"})
# Nombres en castellano para los países que se citan en los textos de lectura.
ES = {"VEN": "Venezuela", "BRN": "Brunéi", "BIH": "Bosnia y Herzegovina", "MNE": "Montenegro", "AGO": "Angola", "JPN": "Japón", "BLR": "Belarús",
      "COD": "Congo RD", "COG": "Congo", "SSD": "Sudán del Sur", "ZWE": "Zimbabue", "SDN": "Sudán", "SUR": "Surinam", "DJI": "Yibuti", "CHE": "Suiza",
      "ECU": "Ecuador", "BHR": "Baréin", "MDV": "Maldivas", "GUY": "Guyana", "SRB": "Serbia", "QAT": "Catar", "TUN": "Túnez", "URY": "Uruguay",
      "NPL": "Nepal", "CPV": "Cabo Verde", "SAU": "Arabia Saudita", "OMN": "Omán", "LBY": "Libia", "IRN": "Irán", "TKM": "Turkmenistán"}
es = lambda i: ES.get(i, M.loc[i, "corto"])
STAT = {"media": "Media", "tendencia": "Tendencia", "volatilidad": "Volatilidad"}
NAMES = {2: "Avanzadas", 0: "Ingreso medio", 3: "Rentistas de recursos", 4: "Pequeñas con Estado grande", 1: "Baja profundidad financiera"}
ORDER = [2, 0, 3, 4, 1]
SHORT = {2: "Avanzadas", 0: "Ingreso<br>medio", 3: "Rentistas<br>de recursos", 4: "Pequeñas con<br>Estado grande", 1: "Baja profundidad<br>financiera"}
GCOL = {2: CATEGORICAL[0], 0: CATEGORICAL[4], 3: CATEGORICAL[2], 4: CATEGORICAL[3], 1: CATEGORICAL[6]}
TRF = {"media": np.log, "tendencia": np.arcsinh, "volatilidad": np.log}


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


panel = pd.read_parquet(D / "panel_largo.parquet")
cov = pd.read_csv(D / "cobertura_indicadores.csv")
cand = cov[cov["dimension"] != "TARGET"]
final = pd.read_csv(D / "classification" / "clasificacion_final.csv")
miss = pd.read_csv(D / "missingness_pais.csv", index_col="iso3")["pct_missing"]
target = pd.read_csv(D / "target_principal.csv", index_col="iso3")
labels = pd.read_csv(D / "clusters.csv", index_col=0)["cluster"]
scores = pd.read_parquet(D / "pca_scores.parquet")
summ_feat = json.loads((D / "resumen_features.json").read_text())
summ_cl = json.loads((D / "clustering_resumen.json").read_text())
cpi = panel[panel["codigo"] == P["targets"]["principal"]].pivot(index="iso3", columns="anio", values="valor")

# ---------------------------------------------------------------- 1 · Datos
figs = {}
n_ind = panel[panel["dimension"] != "TARGET"].groupby("fuente")["codigo"].nunique()
by_year = panel[panel["dimension"] != "TARGET"].groupby(["fuente", "anio"]).size().div(n_ind * 193, level="fuente").mul(100).reset_index(name="pct")
f = go.Figure()
f.add_vrect(x0=1999.5, x1=2020.5, fillcolor=NEUTRAL, line_width=0, layer="below")
for i, s in enumerate(["WDI", "V-Dem", "MC"]):
    d = by_year[by_year["fuente"] == s]
    f.add_scatter(x=d["anio"], y=d["pct"], name=s, mode="lines", line=dict(color=CATEGORICAL[i]), hovertemplate=s + " %{x}: %{y:.1f}%<extra></extra>")
f.add_annotation(x=2010, y=4, text="período de estudio", showarrow=False, font=dict(size=11, color=INK_SECONDARY))
f.update_layout(yaxis=dict(title="% de celdas país-indicador con dato", range=[0, 100]), hovermode="x unified")
figs["cov_year"] = fig_json(f, 380)

steps = [("Catálogo inicial", len(final)),
         ("Niveles absolutos no comparables", -int((final["motivo_exclusion"] == "nivel_absoluto_no_comparable").sum())),
         ("Ambiguos (LLM)", -int((final["motivo_exclusion"] == "ambiguo").sum())),
         ("Medidas de precios", -int((final["motivo_exclusion"] == "medida_de_precios").sum())),
         ("Relación mecánica con la inflación", -int(final["motivo_exclusion"].isin(["fuga_hacia_inflacion", "tasa_nominal_sensibilidad"]).sum())),
         ("Sin datos en la descarga", -(int(final["incluir"].sum()) - len(cand))),
         ("Cobertura insuficiente", -int((~cand["cumple_umbral"]).sum()))]
vals = [v for _, v in steps]
f = go.Figure(go.Waterfall(orientation="h", y=[s for s, _ in steps] + ["Universo de análisis"], x=vals + [None],
                           measure=["absolute"] + ["relative"] * (len(vals) - 1) + ["total"],
                           text=[f"{abs(v):,}".replace(",", ".") for v in vals] + [str(sum(vals))], textposition="outside",
                           connector=dict(line=dict(color=INK_MUTED, width=1)), decreasing=dict(marker=dict(color=CATEGORICAL[7])),
                           increasing=dict(marker=dict(color=CATEGORICAL[0])), totals=dict(marker=dict(color=CATEGORICAL[0]))))
f.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(title="Indicadores", range=[0, 1800]), margin=dict(l=210))
figs["waterfall"] = fig_json(f, 380)
figs["waterfall"]["layout"]["margin"]["l"] = 210

rows = []
for dim, g in cand.groupby("dimension"):
    ok = g["pct_anios_promedio"] >= P["cobertura"]["umbral_pct_anios"]
    for t in range(0, 101, 2):
        rows.append({"dim": dim, "t": t, "r": 100 * ((g["pct_paises"] >= t) & ok).mean(), "n": int(((g["pct_paises"] >= t) & ok).sum())})
sens = pd.DataFrame(rows)
f = go.Figure()
for dim, c in DIMENSION_COLORS.items():
    d = sens[sens["dim"] == dim]
    f.add_scatter(x=d["t"], y=d["r"], name=DIM_LABELS[dim], line=dict(color=c), customdata=d["n"],
                  hovertemplate=DIM_LABELS[dim] + " · umbral %{x}%: %{y:.0f}% (%{customdata} indicadores)<extra></extra>")
f.add_vline(x=P["cobertura"]["umbral_pct_paises"], line=dict(color=INK_SECONDARY, dash="dash", width=1))
f.update_layout(xaxis_title="Umbral mínimo de % de países con dato", yaxis=dict(title="% de indicadores retenidos", range=[0, 100]))
figs["retention"] = fig_json(f, 380)

ms = miss.sort_values()
thr = P["paises"]["umbral_missingness_exclusion"]
f = go.Figure(go.Bar(x=[M.loc[i, "pais"] for i in ms.index], y=ms.values, marker_color=np.where(ms > thr, CATEGORICAL[7], CATEGORICAL[0]),
                     hovertemplate="%{x}: %{y:.1f}% sin dato<extra></extra>"))
f.add_hline(y=thr, line=dict(color=INK_SECONDARY, dash="dash", width=1))
f.update_layout(xaxis=dict(showticklabels=False, showgrid=False, ticks="", title="183 + 10 países, ordenados"), yaxis_title="% de features sin dato", bargap=0.1)
figs["missing"] = fig_json(f, 340)

excl = miss[miss > thr].sort_values(ascending=False)
excl_rows = "".join(f"<tr><td>{M.loc[i, 'pais']}</td><td class='n'>{fmt(v)}%</td></tr>" for i, v in excl.items())

body = kpis([("193", "Estados miembros de la ONU"), ("3", "fuentes internacionales"), (f"{len(final):,}".replace(",", "."), "indicadores catalogados"),
             (str(len(cand[cand["cumple_umbral"]])), "indicadores en el análisis"), (str(summ_feat["paises_analisis"]), "países analizados"), ("2000–2020", "ventana de estudio")])
body += '<div class="grid">'
body += card("cov_year", "Cobertura de datos por año y fuente",
             "Porcentaje de celdas país-indicador con dato en cada año, sobre los 193 países y todos los indicadores candidatos de cada fuente.",
             read="<b>Lectura.</b> Las tres fuentes se estabilizan a partir de 2000 y su cobertura cae después de 2020 por demoras en la publicación y por los ciclos electorales de V-Dem. La ventana 2000-2020 es el tramo más largo con cobertura alta y simultánea.",
             foot="La franja gris marca el período de estudio. MC no tiene datos publicados para 2024.", height=380, full=True)
body += card("waterfall", "Del catálogo al universo de análisis",
             "Cuántos indicadores elimina cada filtro, desde el catálogo completo hasta los que entran en el análisis.",
             read="<b>Lectura.</b> El filtro decisivo es el de comparabilidad (totales en moneda o conteos que crecen con el tamaño del país), seguido por la cobertura temporal. Las medidas de precios se excluyen porque son la variable a explicar, y las tasas de interés porque contienen la inflación por construcción (la real se calcula con el deflactor; las nominales incorporan la inflación esperada).",
             foot="Catálogo: WDI (1.498), V-Dem (79 índices de nivel intermedio) y MC (1). Ambiguo: confianza del LLM menor a 0,60 o sin dimensión dominante.", height=380)
body += card("retention", "Sensibilidad del umbral de cobertura por dimensión",
             "Porcentaje de indicadores que conserva cada dimensión según el mínimo de países con dato exigido (umbral de años fijo en 60%).",
             read=f"<b>Lectura.</b> El umbral elegido (70%, línea punteada) queda antes de la caída pronunciada de la dimensión monetaria. La sociodemográfica es la más penalizada en proporción ({fmt(100 * cand[cand['dimension'] == 'SOCIODEMOGRAFICA']['cumple_umbral'].mean(), 0)}%), pero sigue siendo la que más indicadores aporta.",
             foot="Pasá el cursor por las líneas para ver el número de indicadores retenidos.", height=380)
body += card("missing", "Información faltante por país",
             "Porcentaje de features (indicador × estadístico) sin dato para cada país, ordenado de menor a mayor.",
             read=f"<b>Lectura.</b> La mediana es {fmt(miss.median())}%. Se excluyen los {len(excl)} países con más del 40% de features vacías (en rojo): microestados y Corea del Norte, cuya ausencia de datos es estructural y volvería poco confiable la imputación.",
             foot="Umbral de exclusión: 40% (línea punteada). Los faltantes restantes se imputan con KNN (k = 8, elegido por validación cruzada).",
             full=True, height=340,
             body=f'<div id="missing" class="chart" data-fig="missing" style="height:340px"></div><details><summary style="cursor:pointer;font-size:14px;margin-top:8px">Ver países excluidos</summary><table><tr><th>País</th><th class="n">% sin dato</th></tr>{excl_rows}</table></details>')
body += "</div>"
(OUT / "01_datos.html").write_text(page("01_datos.html", "Datos: fuentes, cobertura y universo de análisis",
    "Cómo se construyó la base país-año que sostiene todo el análisis.",
    "Mostrar de dónde provienen los datos, qué cobertura tienen y qué decisiones de selección (ventana temporal, umbrales de cobertura, exclusión de países e indicadores) definen el universo analizado, para que cualquier lector pueda evaluar su alcance y sus límites.",
    body, figs, REPO, sources=SOURCES), encoding="utf-8")

# ---------------------------------------------------------------- 2 · Clasificación
figs = {}
CL = D / "classification"
metrics = pd.read_csv(CL / "muestra100_metricas.csv")
pred = pd.read_csv(CL / "muestra100_predicciones.csv")
f = make_subplots(rows=1, cols=3, subplot_titles=("Accuracy", "F1 macro", "Kappa de Cohen"), shared_yaxes=True, horizontal_spacing=0.05)
for i, (sub, lab) in enumerate([("100", "Muestra completa (100)"), ("89_sin_citados", "Sin indicadores citados en el codebook (89)")]):
    g = metrics[metrics["subconjunto"] == sub]
    for j, met in enumerate(["accuracy", "macro_f1", "kappa"], start=1):
        f.add_bar(x=g["codebook"].str.upper(), y=g[met], name=lab, legendgroup=lab, showlegend=j == 1, marker_color=CATEGORICAL[i * 6],
                  hovertemplate=lab + " · %{x}: %{y:.3f}<extra></extra>", row=1, col=j)
f.update_yaxes(range=[0.6, 1], row=1, col=1, title_text="Valor")
f.update_annotations(font_size=12)
f.update_layout(barmode="group", bargap=0.3, legend=dict(orientation="h", y=-0.12, yanchor="top", x=0))
figs["codebooks"] = fig_json(f, 380)

cm = confusion(pred["clasificacion_humana"], pred["llm_v2"]).drop(index="AMBIGUO")
lab_cm = [DIM_LABELS.get(i, "Ambiguo") for i in cm.columns]
lab_x = [l.replace("Sociodemográfica", "Socio-<br>demográfica") for l in lab_cm]
f = go.Figure(go.Heatmap(z=cm.values, x=lab_x, y=lab_cm[:-1], colorscale=[[0, "#fcfcfb"], [1, SEQUENTIAL_BLUE[5]]], texttemplate="%{z}",
                         hovertemplate="Manual %{y} · LLM %{x}: %{z}<extra></extra>", xgap=2, ygap=2, showscale=False))
f.update_layout(xaxis=dict(title="Clasificación del LLM", showgrid=False), yaxis=dict(title="Clasificación manual", autorange="reversed", showgrid=False), margin=dict(l=140))
figs["confusion"] = fig_json(f, 380)
figs["confusion"]["layout"]["margin"]["l"] = 150
figs["confusion"]["layout"]["xaxis"]["tickangle"] = 0

wv = final[final["fuente"].isin(["WDI", "V-Dem"])]
f = go.Figure()
for i, (lab, g) in enumerate([("Coinciden las dos corridas", wv[wv["estable"]]), ("No coinciden", wv[~wv["estable"]])]):
    f.add_histogram(x=g["confianza_media"], name=f"{lab} (n={len(g)})", histnorm="percent", xbins=dict(start=0.3, end=1.0001, size=0.025),
                    marker_color=CATEGORICAL[i * 7], opacity=0.85)
f.add_vline(x=0.6, line=dict(color=INK_SECONDARY, dash="dash", width=1))
f.update_layout(barmode="overlay", xaxis_title="Confianza media declarada por el LLM", yaxis_title="% de indicadores", bargap=0.05)
figs["stability"] = fig_json(f, 380)

inc = cand[cand["cumple_umbral"]]
tab = inc.groupby(["dimension", "fuente"]).size().unstack(fill_value=0)
f = go.Figure()
for i, s in enumerate(["WDI", "V-Dem", "MC"]):
    if s in tab:
        f.add_bar(y=[DIM_LABELS[d] for d in tab.index], x=tab[s], name=s, orientation="h", marker_color=CATEGORICAL[i],
                  hovertemplate=s + " · %{y}: %{x}<extra></extra>")
f.update_layout(barmode="stack", xaxis_title="Indicadores", yaxis=dict(categoryorder="total ascending", showgrid=False), bargap=0.35)
figs["universe"] = fig_json(f, 300)
figs["universe"]["layout"]["margin"]["l"] = 140

meta = pd.concat([pd.read_csv(Paths.raw / "wdi" / "wdi_metadata_20261111.csv"), pd.read_csv(Paths.raw / "vdem" / "vdem_79_metadata_v16.csv")]
                 ).drop_duplicates("codigo").set_index("codigo")["nombre_indicador"]
rows = final.set_index("codigo").loc[inc["codigo"]].reset_index()
indicators = [{"c": r.codigo, "n": str(meta.get(r.codigo, r.nombre_indicador)), "s": r.fuente, "d": DIM_LABELS[r.dimension],
               "m": "Revisión" if r.metodo.startswith("clasificacion manual") else ("Protegida" if "protegida" in r.metodo else "LLM")}
              for r in rows.itertuples()]
table_js = """
const IND=%s;const tb=document.getElementById('indbody'),q=document.getElementById('q'),fd=document.getElementById('fd'),cnt=document.getElementById('cnt');
function renderT(){const s=q.value.toLowerCase(),d=fd.value;const r=IND.filter(x=>(!d||x.d===d)&&(x.n.toLowerCase().includes(s)||x.c.toLowerCase().includes(s)));
 cnt.textContent=r.length+' indicadores';tb.innerHTML=r.map(x=>`<tr><td>${x.c}</td><td>${x.n}</td><td>${x.s}</td><td>${x.d}</td><td>${x.m}</td></tr>`).join('');}
q.addEventListener('input',renderT);fd.addEventListener('change',renderT);renderT();
""" % json.dumps(indicators, ensure_ascii=False)
opts = "".join(f"<option>{v}</option>" for v in DIM_LABELS.values())
table_html = (f'<div class="controls"><input id="q" type="search" placeholder="Buscar por nombre o código…" style="min-width:260px">'
              f'<select id="fd"><option value="">Todas las dimensiones</option>{opts}</select><span id="cnt" style="color:#898781"></span></div>'
              '<div class="tablebox"><table><thead><tr><th>Código</th><th>Indicador</th><th>Fuente</th><th>Dimensión</th><th>Método</th></tr></thead><tbody id="indbody"></tbody></table></div>')

agree = wv.groupby("fuente")["estable"].mean()
body = kpis([("0,88", "accuracy frente a la clasificación manual"), ("0,83", "kappa de Cohen"), (fmt(100 * agree["WDI"]) + "%", "acuerdo entre corridas (WDI)"),
             (fmt(100 * agree["V-Dem"]) + "%", "acuerdo entre corridas (V-Dem)"), ("95%", "coincidencia con la clasificación de la tesis")])
body += '<div class="grid">'
body += card("codebooks", "¿Qué versión del codebook clasifica mejor?",
             "Métricas de las tres versiones del codebook frente a 100 indicadores clasificados manualmente por la autora.",
             read="<b>Lectura.</b> Las tres versiones rinden prácticamente igual (McNemar p = 1,0 entre cada par). Se eligió V2 por tener las reglas más completas y documentadas, no por un mejor desempeño. Sin los indicadores citados como ejemplo en el codebook, las métricas bajan apenas.",
             foot="Modelo: openai/gpt-oss-120b vía Groq, temperatura 0, semilla 42. Una segunda corrida de V2 coincide consigo misma en el 96% de los casos.")
body += card("confusion", "Matriz de confusión del codebook V2",
             "Cada celda cuenta indicadores según su clasificación manual (filas) y la del LLM (columnas).",
             read="<b>Lectura.</b> El error dominante es sociodemográfica → estructural: indicadores de empleo que describen a la vez una característica del agente y la estructura ocupacional. Es una frontera difusa del propio codebook más que un error del modelo.",
             foot="Ambiguo: el LLM no asigna dimensión (confianza menor a 0,60).")
body += card("stability", "Estabilidad de la clasificación sobre el universo",
             "Confianza declarada por el LLM en los indicadores clasificados igual y distinto en dos corridas independientes.",
             read="<b>Lectura.</b> Los desacuerdos se concentran en los casos que el propio modelo declara dudosos (confianza mediana 0,70 frente a 0,92). Los 162 indicadores sin acuerdo se resolvieron uno a uno aplicando las reglas del codebook.",
             foot="Línea punteada: umbral de 0,60 por debajo del cual el indicador se considera ambiguo.")
body += card("universe", "Universo de análisis por dimensión y fuente",
             "Indicadores que superan todos los filtros, según dimensión teórica y fuente de origen.",
             read="<b>Lectura.</b> El universo está desbalanceado: domina la sociodemográfica y la monetaria es la más acotada. El PCA posterior resume cada bloque en unos pocos componentes, lo que neutraliza el tamaño desigual de los bloques.",
             foot="Cada indicador entra con tres estadísticos de trayectoria (media, tendencia y volatilidad 2000-2020).", height=300)
body += card("indtable", "Explorador de indicadores", "Buscá cualquier indicador del universo final y filtrá por dimensión.",
             foot="Método: LLM (dos corridas concordantes), Revisión (sin acuerdo entre corridas, resuelto por revisión documentada) o Protegida (asignación teórica fija).",
             full=True, body=table_html)
body += "</div>"
(OUT / "02_clasificacion.html").write_text(page("02_clasificacion.html", "Clasificación de indicadores con un modelo de lenguaje",
    "Cómo se asignó cada indicador a una dimensión teórica de la inflación y qué tan confiable es esa asignación.",
    "Evaluar la clasificación automática de 1.578 indicadores en las cuatro dimensiones del marco teórico (monetaria, institucional, estructural y sociodemográfica): cuánto coincide con el criterio experto, qué tan estable es entre ejecuciones y cómo queda compuesto el universo final.",
    body, figs, REPO, extra_js=table_js, sources=SOURCES), encoding="utf-8")

# ---------------------------------------------------------------- 3 · Inflación
figs = {}
q = cpi.loc[:, 1990:2024].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T
f = go.Figure()
f.add_vrect(x0=1999.5, x1=2020.5, fillcolor=NEUTRAL, line_width=0, layer="below")
f.add_scatter(x=list(q.index) + list(q.index[::-1]), y=list(q[0.9]) + list(q[0.1][::-1]), fill="toself", fillcolor="rgba(42,120,214,0.12)",
              line=dict(width=0), name="Percentiles 10–90", hoverinfo="skip")
f.add_scatter(x=list(q.index) + list(q.index[::-1]), y=list(q[0.75]) + list(q[0.25][::-1]), fill="toself", fillcolor="rgba(42,120,214,0.28)",
              line=dict(width=0), name="Percentiles 25–75", hoverinfo="skip")
f.add_scatter(x=q.index, y=q[0.5], name="Mediana", line=dict(color=CATEGORICAL[0]), hovertemplate="%{x}: mediana %{y:.1f}%<extra></extra>")
f.update_layout(yaxis=dict(title="Inflación anual (%)", range=[-2, 45]))
figs["bands"] = fig_json(f, 380)

t = target.dropna(subset=["media"]).join(M[["pais", "region_es"]])
f = go.Figure()
for i, (reg, g) in enumerate(t.groupby("region_es")):
    f.add_scatter(x=g["media"], y=g["volatilidad"], mode="markers", name=reg, customdata=g[["pais"]].values,
                  marker=dict(size=8, color=[CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[3], CATEGORICAL[6]][i],
                              symbol=["circle", "square", "diamond", "triangle-up", "x"][i], line=dict(color="#fcfcfb", width=1)),
                  hovertemplate="%{customdata[0]}<br>media %{x:.1f}% · volatilidad %{y:.1f}<extra></extra>")
ticks = [0.1, 0.3, 1, 3, 10, 30, 100]
f.update_layout(xaxis=dict(type="log", title="Inflación media 2000-2020 (%)", tickvals=ticks, ticktext=[fmt(v, 1).rstrip("0").rstrip(",") for v in ticks]),
                yaxis=dict(type="log", title="Volatilidad (p.p.)", tickvals=[0.5, 1, 2, 5, 10, 20, 50, 100], ticktext=["0,5", "1", "2", "5", "10", "20", "50", "100"]))
figs["scatter"] = fig_json(f, 380)

world = go.Figure(go.Choropleth(locations=t.index, z=np.log10(t["media"]), customdata=np.c_[t["pais"], t["media"]],
                                colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [0.75, "#eb6834"], [1, "#8a2a1f"]], zmin=np.log10(0.5), zmax=np.log10(60),
                                hovertemplate="%{customdata[0]}: %{customdata[1]:.1f}%<extra></extra>", marker_line_color="#fcfcfb", marker_line_width=0.4,
                                colorbar=dict(title="%", tickvals=np.log10([0.5, 1, 2, 5, 10, 20, 50]), ticktext=["0,5", "1", "2", "5", "10", "20", "50"], thickness=10)))
world.update_layout(geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)", landcolor="#efeee9", showland=True))
figs["map"] = fig_json(world, 480)
figs["map"]["layout"]["margin"] = dict(l=0, r=0, t=10, b=0)

H = cpi.loc[:, 2000:2020]
hi = H.index[(H > 40).any(axis=1)]
H = H.loc[H.loc[hi].max(axis=1).sort_values(ascending=False).index]
sl = lambda x: np.sign(x) * np.log10(1 + np.abs(x))
f = go.Figure(go.Heatmap(z=sl(H.values), x=list(H.columns), y=[M.loc[i, "corto"] for i in H.index], customdata=H.values, zmin=0, zmax=sl(600),
                         colorscale=[[0, "#cde2fb"], [0.35, "#6da7ec"], [0.6, "#eb6834"], [1, "#8a2a1f"]], xgap=1, ygap=1,
                         hovertemplate="%{y} %{x}: %{customdata:.1f}%<extra></extra>",
                         colorbar=dict(title="%", tickvals=sl(np.array([0, 10, 40, 100, 500])), ticktext=["0", "10", "40", "100", "500"], thickness=10)))
f.update_layout(xaxis=dict(showgrid=False, dtick=2), yaxis=dict(showgrid=False, autorange="reversed"), margin=dict(l=170))
figs["episodes"] = fig_json(f, 560)
figs["episodes"]["layout"]["margin"]["l"] = 170

ranks = target[["media", "tendencia", "volatilidad"]].rank(pct=True).mul(100)
series = {}
for iso in M.index:
    if iso in cpi.index:
        s = cpi.loc[iso].dropna()
        s = s[(s.index >= 1998) & (s.index <= 2024)]
        series[iso] = {"n": M.loc[iso, "pais"], "r": M.loc[iso, "region_es"], "y": s.index.tolist(), "v": [round(float(v), 2) for v in s.values]}
        if iso in target.index and pd.notna(target.loc[iso, "media"]):
            series[iso].update({k: round(float(target.loc[iso, k]), 2) for k in ["media", "tendencia", "volatilidad"]})
            series[iso].update({f"p_{k}": int(round(ranks.loc[iso, k])) for k in ["media", "tendencia", "volatilidad"]})
        if iso in labels.index:
            series[iso]["g"] = NAMES[int(labels[iso])]
opts = "".join(f'<option value="{k}"{" selected" if k == "BRA" else ""}>{v["n"]}</option>'
               for k, v in sorted(series.items(), key=lambda kv: kv[1]["n"]))
explorer_html = (f'<div class="controls"><label for="cty">País</label><select id="cty">{opts}</select></div>'
                 '<div style="display:grid;grid-template-columns:2fr 1fr;gap:16px;align-items:start">'
                 '<div id="ctychart" class="chart" style="height:400px"></div><div id="ctycard"></div></div>')
explorer_js = """
const SER=%s;
function ols(x,y){const n=x.length,mx=x.reduce((a,b)=>a+b)/n,my=y.reduce((a,b)=>a+b)/n;let sxy=0,sxx=0;
 for(let i=0;i<n;i++){sxy+=(x[i]-mx)*(y[i]-my);sxx+=(x[i]-mx)**2}const b=sxy/sxx;return [b,my-b*mx];}
const nf=(v,d=1)=>v.toFixed(d).replace('.',',');
function renderC(){const k=document.getElementById('cty').value,s=SER[k];
 const wy=[],wv=[];s.y.forEach((y,i)=>{if(y>=2000&&y<=2020){wy.push(y);wv.push(s.v[i])}});
 const tr=[{x:s.y,y:s.v,mode:'lines+markers',name:'Inflación anual',line:{color:'#2a78d6',width:2},marker:{size:5},hovertemplate:'%%{x}: %%{y:.1f}%%<extra></extra>'}];
 if(wy.length>=5){const [b,a]=ols(wy,wv);tr.push({x:[2000,2020],y:[a+b*2000,a+b*2020],mode:'lines',name:'Tendencia OLS 2000-2020',line:{color:'#eb6834',dash:'dash',width:2},hoverinfo:'skip'})}
 const lay={height:400,margin:{l:56,r:16,t:20,b:40},paper_bgcolor:'#fcfcfb',plot_bgcolor:'#fcfcfb',font:{family:'system-ui,sans-serif',size:12},
  shapes:[{type:'rect',xref:'x',yref:'paper',x0:1999.5,x1:2020.5,y0:0,y1:1,fillcolor:'#f0efec',line:{width:0},layer:'below'}],
  xaxis:{range:[1997.5,2024.5],gridcolor:'#e1e0d9'},yaxis:{title:{text:'Inflación anual (%%)'},gridcolor:'#e1e0d9',zerolinecolor:'#c3c2b7'},legend:{orientation:'h',y:1.08,x:0}};
 Plotly.react('ctychart',tr,lay,CFG);
 const c=document.getElementById('ctycard');
 if(s.media===undefined){c.innerHTML=`<div class="kpi"><b>${s.n}</b><span>${s.r}</span></div><p style="font-size:14px;color:#52514e">Menos de 5 años con dato en 2000-2020: no se calcula la trayectoria. ${s.g?'Grupo monetario-fiscal: '+s.g+'.':''}</p>`;return;}
 c.innerHTML=`<div class="kpi" style="margin-bottom:10px"><b style="font-size:20px">${s.n}</b><span>${s.r}${s.g?' · grupo '+s.g:' · fuera del clustering (atípico o excluido)'}</span></div>
 <div class="kpi" style="margin-bottom:10px"><b>${nf(s.media)}%%</b><span>inflación media 2000-2020 · percentil ${s.p_media}</span></div>
 <div class="kpi" style="margin-bottom:10px"><b>${s.tendencia>0?'+':''}${nf(s.tendencia,2)}</b><span>tendencia (p.p. por año) · percentil ${s.p_tendencia}</span></div>
 <div class="kpi"><b>${nf(s.volatilidad)}</b><span>volatilidad (p.p.) · percentil ${s.p_volatilidad}</span></div>`;}
document.getElementById('cty').addEventListener('change',renderC);renderC();
""" % json.dumps(series, ensure_ascii=False)

annual = cpi.loc[:, 2000:2020].stack().dropna()
rho_t = t[["media", "tendencia", "volatilidad"]].corr(method="spearman")
body = kpis([(fmt(annual.median()) + "%", "inflación anual mediana 2000-2020"), (str((annual > 40).sum()), "años-país con inflación > 40%"),
             (str(len(t)), "países con trayectoria calculable"), (fmt(rho_t.loc["media", "volatilidad"], 2), "correlación nivel–volatilidad (ρ)")])
body += '<div class="grid">'
body += card("cty", "Explorador de trayectorias por país",
             "Elegí un país para ver su inflación anual (1998-2024), la tendencia lineal del período de estudio y sus tres estadísticos de trayectoria, con el percentil que ocupa entre los países.",
             read="<b>Cómo se resume cada trayectoria.</b> <b>Media</b>: nivel promedio 2000-2020. <b>Tendencia</b>: pendiente de la regresión lineal valor ~ año. <b>Volatilidad</b>: desvío estándar de los residuos alrededor de esa tendencia. Son las tres variables dependientes de todo el análisis.",
             foot="La franja gris marca el período de estudio. Dentro del período de estudio, Argentina tiene IPC en WDI solo para 2018-2020, por lo que queda fuera del análisis principal (se incluye en la especificación con el deflactor del PIB).",
             full=True, body=explorer_html)
body += card("bands", "La inflación entre países a lo largo del tiempo",
             "Mediana y bandas de percentiles de la inflación anual entre los países miembros de la ONU, 1990-2024.",
             read="<b>Lectura.</b> El período de estudio empieza después de la gran desinflación de los 90 y termina antes del shock de 2022. La inflación típica es baja (mediana entre 1,5% y 5%), pero la dispersión entre países nunca desaparece: el percentil 90 se mantiene entre 9% y 20%.",
             foot="Cobertura anual: entre 159 (2000) y 180 (2010) países con dato.", height=380)
body += card("scatter", "Nivel y volatilidad de la inflación",
             "Cada punto es un país: inflación media frente a volatilidad alrededor de su tendencia (ejes logarítmicos). Hacé clic en la leyenda para mostrar u ocultar regiones.",
             read=f"<b>Lectura.</b> Nivel y volatilidad están fuertemente asociados (ρ = {fmt(rho_t.loc['media', 'volatilidad'], 2)}): los países con más inflación son también los de inflación más errática. La tendencia, en cambio, se relaciona débilmente con ambos (ρ = {fmt(rho_t.loc['media', 'tendencia'], 2)} con el nivel), por eso los tres estadísticos se modelan por separado.",
             foot=f"{len(t)} países con al menos {P['trayectorias']['min_anios']} años de IPC (descriptivo; los modelos exigen {P['trayectorias']['min_anios_target']}). ρ: correlación de rangos de Spearman.", height=380)
body += card("map", "Inflación media 2000-2020 por país",
             "Mapa de la inflación media del período (escala logarítmica de color).",
             read="<b>Lectura.</b> Los niveles más altos se concentran en África subsahariana, Venezuela, Irán y Asia Central; los más bajos, en Europa occidental, Japón y Brunéi.",
             foot="Gris: países sin trayectoria calculable (menos de 5 años de IPC en 2000-2020).", full=True, height=480)
body += card("episodes", "Episodios de alta inflación",
             f"Inflación anual de los {len(hi)} países que superaron el 40% en al menos un año entre 2000 y 2020, ordenados por su pico.",
             read="<b>Lectura.</b> Hay dos grupos temporales: colas de desinflaciones de los 90 al inicio del período (Angola, Congo RD, Belarús, Serbia, Turquía) y episodios posteriores a 2011 asociados a conflicto o colapso macroeconómico (Sudán del Sur, Venezuela, Sudán, Líbano, Zimbabue). Un episodio al inicio produce tendencias muy negativas, y uno al final, muy positivas.",
             foot="Celdas vacías: sin dato en WDI. Color en escala log simétrica.", full=True, height=560)
body += "</div>"
(OUT / "03_inflacion.html").write_text(page("03_inflacion.html", "La variable a explicar: trayectorias inflacionarias 2000-2020",
    "Exploración de la inflación de los países miembros de la ONU y de los tres estadísticos que resumen cada trayectoria.",
    "Describir cómo se comportó la inflación entre países y a lo largo del tiempo, identificar los episodios extremos y mostrar cómo se resume la trayectoria de cada país en tres medidas (nivel, tendencia y volatilidad), que son las variables dependientes del análisis.",
    body, figs, REPO, extra_js=explorer_js, sources=SOURCES), encoding="utf-8")

# ---------------------------------------------------------------- 4 · Distribuciones y casos extremos
figs = {}
EX = pd.read_csv(D / "extremos_paises.csv", index_col="iso3")
SENS = pd.read_csv(D / "extremos_sensibilidad.csv")
DST = pd.read_csv(D / "distribuciones_target.csv")
tgt = targets.load("principal")
model_idx = EX.index[EX["en_modelos_principales"]]
RULES = [("excl_faltantes", "Más de 40% de features sin dato", "excluye"),
         ("sin_ipc", "Menos de 5 años de IPC", "excluye"),
         ("serie_corta", "Serie de IPC corta (< 60% de los años)", "excluye"),
         ("atipico_perfil", "Perfil monetario atípico (Tukey)", "excluye"),
         ("inflacion_extrema", "Inflación media extrema (Tukey en log)", "mantiene"),
         ("episodio_40", "Algún año con inflación > 40%", "mantiene"),
         ("brecha_deflactor", "IPC y deflactor difieren más de 2×", "mantiene"),
         ("influyente", "Alta influencia en H3 (Cook > 4/n)", "mantiene")]
EXCL_COLOR, KEEP_COLOR = CATEGORICAL[7], CATEGORICAL[0]

# Distribuciones: escala original frente a transformada
TT = {"media": ("Media", "log(media)"), "tendencia": ("Tendencia", "asinh(tendencia)"), "volatilidad": ("Volatilidad", "log(volatilidad)")}
for esc in ["original", "transformada"]:
    f = make_subplots(rows=1, cols=3, subplot_titles=[TT[s][0 if esc == "original" else 1] for s in TT], horizontal_spacing=0.07)
    for j, s in enumerate(TT):
        v = tgt.loc[model_idx, s].dropna()
        v = TRF[s](v) if esc == "transformada" else v
        sk = DST.query("estadistico == @s and escala == @esc")["asimetria"].iloc[0]
        f.add_histogram(x=v, nbinsx=35, marker_color=CATEGORICAL[0], marker_line=dict(color="#fcfcfb", width=0.5), showlegend=False,
                        hovertemplate="%{x}: %{y} países<extra></extra>", row=1, col=j + 1)
        f.add_annotation(text=f"asimetría {fmt(sk, 2)}", xref=f"x{j + 1 if j else ''} domain", yref=f"y{j + 1 if j else ''} domain",
                         x=0.98, y=0.95, showarrow=False, xanchor="right", font=dict(size=11, color=INK_SECONDARY))
    f.update_yaxes(title_text="Países", row=1, col=1)
    figs[f"dist_{esc}"] = fig_json(f, 360)
sk = DST.set_index(["estadistico", "escala"])["asimetria"]

# Correlaciones: estadísticos de inflación y primer componente de cada dimensión
cmat = pd.concat([TRF[s](tgt.loc[model_idx, s]).rename(STAT[s]) for s in TT] +
                 [scores.loc[model_idx, f"{d}_media__PC1"].rename(f"PC1 {DIM_LABELS[d].lower()}") for d in DIMENSION_COLORS], axis=1).dropna()
rho = cmat.corr(method="spearman")
f = go.Figure(go.Heatmap(z=rho.values, x=rho.columns, y=rho.index, zmin=-1, zmax=1, colorscale=DIVERGING, text=np.round(rho.values, 2),
                         texttemplate="%{text}", hovertemplate="%{y} · %{x}: ρ = %{z:.2f}<extra></extra>", colorbar=dict(thickness=10)))
f.update_layout(xaxis=dict(showgrid=False, tickangle=-30), yaxis=dict(showgrid=False, autorange="reversed"), margin=dict(l=150, b=110))
figs["corr"] = fig_json(f, 480)
figs["corr"]["layout"]["margin"].update(l=150, b=110)
pc1 = rho.iloc[3:, 3:].where(~np.eye(4, dtype=bool)).abs()

# Reglas: cuántos países marca cada una y qué tratamiento reciben
counts = [(lab, int(EX[k].sum()), tr) for k, lab, tr in RULES]
f = go.Figure(go.Bar(y=[c[0] for c in counts], x=[c[1] for c in counts], orientation="h", text=[c[1] for c in counts], textposition="outside",
                     marker_color=[EXCL_COLOR if c[2] == "excluye" else KEEP_COLOR for c in counts], hovertemplate="%{y}: %{x} países<extra></extra>"))
f.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis_title="Países marcados", margin=dict(l=280))
figs["rules"] = fig_json(f, 360)
figs["rules"]["layout"]["margin"]["l"] = 280

# Matriz país × regla
flag = EX[EX[[k for k, _, _ in RULES]].any(axis=1)].copy()
flag = flag.assign(_e=flag[[k for k, _, tr in RULES if tr == "excluye"]].any(axis=1)).sort_values(["_e", "n_reglas"], ascending=[False, False])
z = np.array([[(1 if tr == "excluye" else 0.5) if flag.loc[i, k] else 0 for k, _, tr in RULES] for i in flag.index])
f = go.Figure(go.Heatmap(z=z, x=[lab for _, lab, _ in RULES], y=[M.loc[i, "corto"] for i in flag.index], zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                         colorscale=[[0, "#efeee9"], [0.5, KEEP_COLOR], [0.75, KEEP_COLOR], [0.76, EXCL_COLOR], [1, EXCL_COLOR]],
                         customdata=np.array([[flag.loc[i, "tratamiento"]] * len(RULES) for i in flag.index]),
                         hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>"))
f.update_layout(xaxis=dict(side="top", tickangle=-35, showgrid=False), yaxis=dict(autorange="reversed", showgrid=False, dtick=1), margin=dict(l=170, t=190))
figs["matrix"] = fig_json(f, max(500, 17 * len(flag) + 200))
figs["matrix"]["layout"]["margin"].update(l=170, t=190)

# Perfil monetario atípico: distancia al centroide y umbral de Tukey
dd = EX["distancia_perfil"].dropna().sort_values(ascending=False)
thr = summ_cl["umbral_tukey"]
f = go.Figure(go.Bar(x=[M.loc[i, "corto"] for i in dd.index], y=dd.values, marker_color=[EXCL_COLOR if v > thr else "#c3c2b7" for v in dd.values],
                     hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
f.add_hline(y=thr, line=dict(color=INK_SECONDARY, dash="dash", width=1), annotation_text="Q3 + 1,5·RIC", annotation_position="top right")
f.update_layout(xaxis=dict(showticklabels=False, title=f"{len(dd)} países, ordenados"), yaxis_title="Distancia al centroide (componentes monetarios)", bargap=0.1)
figs["tukey"] = fig_json(f, 360)

# Cobertura temporal del IPC
yrs = EX["n_anios_ipc"].fillna(0).sort_values()
min_t = P["trayectorias"]["min_anios_target"]
f = go.Figure(go.Bar(x=[M.loc[i, "corto"] for i in yrs.index], y=yrs.values, hovertemplate="%{x}: %{y} años<extra></extra>",
                     marker_color=[EXCL_COLOR if v < min_t else "#c3c2b7" for v in yrs.values]))
f.add_hline(y=min_t, line=dict(color=INK_SECONDARY, dash="dash", width=1), annotation_text=f"mínimo: {min_t} años (60%)", annotation_position="top left")
f.update_layout(xaxis=dict(showticklabels=False, title="193 países, ordenados"), yaxis=dict(title="Años con IPC en 2000-2020", range=[0, 22]), bargap=0.1)
figs["years"] = fig_json(f, 340)

# IPC frente a deflactor
g = EX.dropna(subset=["media", "media_deflactor"])
g = g[(g["media"] > 0) & (g["media_deflactor"] > 0)]
f = go.Figure()
lo, hi = 0.05, max(g["media_deflactor"].max(), g["media"].max()) * 1.5
for fac, dash in [(1, "solid"), (2, "dot"), (0.5, "dot")]:
    f.add_scatter(x=[lo, hi], y=[lo * fac, hi * fac], mode="lines", line=dict(color=INK_MUTED, width=1, dash=dash), showlegend=False, hoverinfo="skip")
g = g.assign(ratio=np.abs(np.log(g["media_deflactor"] / g["media"])))
top_gap = set(g[g["brecha_deflactor"]].nlargest(5, "ratio").index)
for gap, col, name in [(False, "#c3c2b7", "Coinciden (dentro de 2×)"), (True, EXCL_COLOR, "Difieren más de 2×")]:
    h = g[g["brecha_deflactor"] == gap]
    f.add_scatter(x=h["media"], y=h["media_deflactor"], mode="markers+text" if gap else "markers", name=name,
                  text=[es(i) if i in top_gap else "" for i in h.index] if gap else None, textposition=["top center", "bottom center", "middle left", "top right", "bottom right"] * 4 if gap else "top center", textfont=dict(size=10),
                  marker=dict(color=col, size=8, line=dict(color="#fcfcfb", width=1)), customdata=[M.loc[i, "pais"] for i in h.index],
                  hovertemplate="%{customdata}<br>IPC %{x:.1f}% · deflactor %{y:.1f}%<extra></extra>")
tk = [0.1, 1, 10, 100, 1000, 10000]
f.update_layout(xaxis=dict(type="log", title="Inflación media según IPC (%)", tickvals=tk, ticktext=["0,1", "1", "10", "100", "1.000", "10.000"]),
                yaxis=dict(type="log", title="Inflación media según deflactor (%)", tickvals=tk, ticktext=["0,1", "1", "10", "100", "1.000", "10.000"]),
                legend=dict(orientation="h", y=1.08, x=0))
figs["gap"] = fig_json(f, 440)

# Influencia: distancia de Cook en el modelo de H3
for s in ["media", "volatilidad"]:
    c = EX[f"cook_{s}"].dropna().sort_values(ascending=False)
    lim = 4 / len(c)
    f = go.Figure(go.Bar(x=[M.loc[i, "corto"] for i in c.index], y=c.values, marker_color=[CATEGORICAL[2] if v > lim else "#c3c2b7" for v in c.values],
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    f.add_hline(y=lim, line=dict(color=INK_SECONDARY, dash="dash", width=1), annotation_text="4/n", annotation_position="top right")
    f.update_layout(xaxis=dict(tickangle=-60, range=[-0.5, 30.5]), yaxis_title="Distancia de Cook", bargap=0.15)
    figs[f"cook_{s}"] = fig_json(f, 380)
    figs[f"cook_{s}"]["layout"]["margin"]["b"] = 110

# Sensibilidad de H3 al tratamiento de los extremos
SPECS = [("principal", "Especificación principal"), ("con_atipicos", "Con atípicos del perfil"), ("sin_influyentes", "Sin países influyentes"),
         ("mediana", "Mediana anual en lugar de media"), ("deflactor", "Deflactor en lugar de IPC")]
rows = []
for s in ["media", "volatilidad"]:
    for key, lab in SPECS:
        if key == "sin_influyentes":
            r = SENS.query("estadistico == @s and especificacion == @key").iloc[0]
            rows.append({"s": s, "lab": lab, "d": r["delta_r2adj"], "lo": np.nan, "hi": np.nan, "p": r["p_HC3"]})
            continue
        if key == "mediana" and s != "media":
            continue
        h = pd.read_csv(D / ("reg_h3.csv" if key == "principal" else f"reg_h3_{key}.csv")).set_index("estadistico").loc[s]
        rows.append({"s": s, "lab": lab, "d": h["delta_r2adj"], "lo": h["delta_ic95_inf"], "hi": h["delta_ic95_sup"], "p": h["p_HC3_holm"]})
    jk = SENS.query("estadistico == @s").set_index("especificacion")
    rows.append({"s": s, "lab": "Jackknife (rango al quitar cada país)", "d": np.nan, "lo": jk.loc["jackknife_min", "delta_r2adj"],
                 "hi": jk.loc["jackknife_max", "delta_r2adj"], "p": jk.loc["jackknife_p_max", "p_HC3"]})
fr = pd.DataFrame(rows)
f = make_subplots(rows=1, cols=2, subplot_titles=["Nivel (media)", "Volatilidad"], shared_yaxes=True, horizontal_spacing=0.04)
for j, s in enumerate(["media", "volatilidad"]):
    d = fr[fr["s"] == s]
    f.add_scatter(x=d["d"], y=d["lab"], mode="markers", marker=dict(color=CATEGORICAL[0], size=10), showlegend=False,
                  error_x=dict(type="data", symmetric=False, array=(d["hi"] - d["d"]).fillna(0), arrayminus=(d["d"] - d["lo"]).fillna(0), color=CATEGORICAL[0]),
                  customdata=d["p"], hovertemplate="%{y}: ΔR² ajustado %{x:.3f} · p = %{customdata:.4f}<extra></extra>", row=1, col=j + 1)
    jkr = d[d["d"].isna()]
    f.add_scatter(x=[jkr["lo"].iloc[0], jkr["hi"].iloc[0]], y=[jkr["lab"].iloc[0]] * 2, mode="lines+markers", line=dict(color=CATEGORICAL[2], width=6),
                  marker=dict(size=8, color=CATEGORICAL[2]), showlegend=False, hovertemplate="jackknife: %{x:.3f}<extra></extra>", row=1, col=j + 1)
    f.add_vline(x=0, line=dict(color=INK_MUTED, width=1), row=1, col=j + 1)
f.update_yaxes(autorange="reversed", showgrid=False)
f.update_xaxes(title_text="Mejora del R² ajustado (IC 95% bootstrap)")
f.update_layout(margin=dict(l=260))
figs["sens"] = fig_json(f, 380)
figs["sens"]["layout"]["margin"]["l"] = 260

# Tabla de pruebas y tratamiento
TESTS = [("Datos faltantes en las features", f"&gt; {P['paises']['umbral_missingness_exclusion']}% de las features sin dato", "excl_faltantes",
          "Se excluye del análisis", "—"),
         ("Inflación sin dato suficiente", f"&lt; {P['trayectorias']['min_anios']} años de IPC", "sin_ipc", "Fuera de los modelos (la inflación nunca se imputa)", "Variante con el deflactor"),
         ("Serie de inflación corta", f"&lt; {min_t} años de IPC (60% del período, igual que las features)", "serie_corta", "Fuera de los modelos principales", "Variante con el deflactor"),
         ("Perfil monetario atípico", "Distancia al centroide &gt; Q3 + 1,5·RIC (Tukey)", "atipico_perfil", "Fuera del clustering y de los modelos principales", "Variante «con atípicos»"),
         ("Inflación media extrema", "log(media) &gt; Q3 + 1,5·RIC", "inflacion_extrema", "Se mantiene; transformación logarítmica", "Mediana anual, HC3, permutaciones"),
         ("Episodios de inflación alta", "Algún año &gt; 40% en 2000-2020", "episodio_40", "Se mantiene", "Mediana anual; tendencia de Theil-Sen"),
         ("Medición inconsistente", "Media del IPC y del deflactor difieren más de 2×", "brecha_deflactor", "Se mantiene", "Variante con el deflactor"),
         ("Influencia en el modelo", "Distancia de Cook &gt; 4/n en el modelo de H3", "influyente", "Se mantiene", "Reestimación sin ellos; jackknife")]
trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td class='n'>{int(EX[k].sum())}</td><td>{d}</td><td>{e}</td></tr>" for a, b, k, d, e in TESTS)
table_tests = ('<div class="tablebox" style="max-height:none"><table><thead><tr><th>Problema</th><th>Criterio de detección</th><th class="n">Países</th>'
               f"<th>Tratamiento</th><th>Prueba de que no cambia el resultado</th></tr></thead><tbody>{trs}</tbody></table></div>")

# Textos calculados
bad = fr[(fr["lo"] <= 0) & fr["d"].notna()]
sens_txt = ("La mejora es positiva en todas las especificaciones y su intervalo nunca incluye el cero." if bad.empty else
            "La mejora es positiva en todas las especificaciones; su intervalo excluye el cero en todas salvo " +
            ", ".join(f"{r.lab[0].lower() + r.lab[1:]} ({'nivel' if r.s == 'media' else 'volatilidad'})" for r in bad.itertuples()) + ".")
n_flag, n_out = len(flag), int((~EX["en_modelos_principales"] & EX.index.isin(flag.index)).sum())
sm_ = SENS.set_index(["estadistico", "especificacion"])
gap_cases = ", ".join(es(i) for i in g[g["brecha_deflactor"]].nlargest(4, "ratio").index)
top_cook = ", ".join(es(i) for i in EX["cook_media"].dropna().sort_values(ascending=False).index[:4])
lim_inf = float(np.exp(np.log(EX["media"][EX["media"] > 0]).quantile(0.75) + 1.5 * np.subtract(*np.log(EX["media"][EX["media"] > 0]).quantile([0.75, 0.25]))))
r_mv = rho.loc["Media", "Volatilidad"]

body = kpis([(str(n_flag), "países marcados por al menos una regla"), (str(int(EX["en_modelos_principales"].sum())), "países en los modelos principales"),
             (fmt(sk[("media", "original")], 1) + " → " + fmt(sk[("media", "transformada")], 1), "asimetría de la media: original → log"),
             (fmt(sm_.loc[("media", "jackknife_p_max"), "p_HC3"], 3), "peor p de H3 (nivel) al quitar cualquier país")])
body += '<div class="grid">'
body += card("dist", "Distribución de los tres estadísticos de trayectoria",
             "Histogramas de la media, la tendencia y la volatilidad de la inflación en los países de los modelos principales, en escala original y transformada.",
             controls=seg("dist", [("dist_original", "Escala original"), ("dist_transformada", "Escala transformada")]), fig_key="dist_original",
             read=(f"<b>Lectura.</b> En escala original las tres distribuciones son muy asimétricas: unos pocos países concentran valores muy altos "
                   f"(asimetría de {fmt(sk[('media', 'original')], 1)} en la media y {fmt(sk[('volatilidad', 'original')], 1)} en la volatilidad). "
                   f"El logaritmo deja la media casi simétrica ({fmt(sk[('media', 'transformada')], 2)}) y reduce mucho la de la volatilidad "
                   f"({fmt(sk[('volatilidad', 'transformada')], 2)}). La tendencia conserva colas pesadas aun con asinh "
                   f"(asimetría {fmt(sk[('tendencia', 'transformada')], 1)}): por eso se contrasta con la pendiente robusta de Theil-Sen."),
             foot="Ninguna distribución es normal (Shapiro-Wilk p &lt; 0,01 en todas): la inferencia usa errores HC3, permutaciones y bootstrap, que no lo requieren.",
             full=True, height=360)
body += card("corr", "Relaciones entre la inflación y las cuatro dimensiones",
             "Correlación de rangos (Spearman) entre los estadísticos de inflación y el primer componente principal de cada dimensión (estadístico: media).",
             read=(f"<b>Lectura.</b> El nivel y la volatilidad de la inflación están muy asociados (ρ = {fmt(r_mv, 2)}), mientras que la tendencia va por otro lado. "
                   f"Los primeros componentes de las cuatro dimensiones están muy correlacionados entre sí (|ρ| hasta {fmt(pc1.max().max(), 2)}): comparten un gradiente "
                   "de desarrollo. Esto anticipa que será difícil atribuir un efecto a una sola dimensión, y justifica contrastar los bloques en conjunto (H3)."),
             height=480)
body += card("rules", "Ocho reglas para detectar casos extremos",
             "Cantidad de países que marca cada regla. En naranja, las reglas que excluyen al país de los modelos principales; en azul, las que lo mantienen con pruebas de robustez.",
             read=("<b>Lectura.</b> Solo se excluye un país cuando falta información para medirlo (features o IPC insuficientes) o cuando su perfil monetario está tan alejado "
                   "del resto que distorsionaría los grupos. La inflación alta, por sí sola, nunca es motivo de exclusión: es parte del fenómeno a explicar."),
             height=360)
body += card("matrix", "Mapa de casos extremos: qué regla marca a cada país",
             "Cada fila es un país marcado por al menos una regla. Pasá el cursor para ver el tratamiento que recibe.",
             read=(f"<b>Lectura.</b> {n_flag} países activan alguna regla, pero solo {n_out} quedan fuera de los modelos principales. Los casos más extremos acumulan varias "
                   "reglas a la vez: Sudán del Sur y Zimbabue tienen inflación extrema, series cortas y un perfil monetario atípico; Venezuela tiene una serie corta que "
                   "termina justo antes de su hiperinflación."),
             full=True, height=max(500, 17 * len(flag) + 200))
body += card("years", "Cobertura temporal de la inflación",
             "Años con dato de IPC entre 2000 y 2020 por país. En naranja, los que no alcanzan el mínimo exigido.",
             read=(f"<b>Lectura.</b> La gran mayoría de los países tiene la serie completa. Se exige el mismo 60% de los años que a las features ({min_t} de 21): con menos, "
                   "la media no describe la trayectoria del período. El caso que motiva la regla es Venezuela: su IPC cubre solo 2009-2016, los años previos a la hiperinflación, de modo que su "
                   "media (73%) no refleja lo que ocurrió después."),
             foot="Argentina tiene IPC en WDI solo para 2018-2020; con el deflactor se recupera en la variante correspondiente.", height=340)
body += card("tukey", "Perfil monetario atípico",
             "Distancia de cada país al centro de los componentes monetarios. Por encima de la línea (Q3 + 1,5·RIC), el perfil se considera atípico.",
             read=(f"<b>Lectura.</b> {int(EX['atipico_perfil'].sum())} países tienen perfiles monetarios muy alejados del resto. Se dejan fuera del clustering porque, con "
                   "K-means, un punto lejano puede formar un grupo propio o arrastrar a otro. La regla mira solo las variables explicativas, no la inflación; aun así, "
                   "incluye a dos de los países con inflación más alta, por lo que se reincorporan en una variante de los modelos."),
             height=360)
body += card("gap", "Dos medidas de inflación: IPC frente a deflactor",
             "Inflación media según el IPC (eje horizontal) y según el deflactor del PIB (eje vertical). Las líneas punteadas marcan una diferencia de 2×.",
             read=(f"<b>Lectura.</b> Las dos medidas coinciden en casi todos los países. Las brechas grandes ({gap_cases}) señalan problemas de medición o series que "
                   "cubren períodos distintos. Por eso toda la estimación se repite con el deflactor como variable dependiente."),
             foot="Ejes logarítmicos. Solo países con ambas medias positivas.", height=440)
body += card("cook", "Influencia de cada país en el modelo de H3",
             "Distancia de Cook de los 30 países más influyentes en el modelo con las cuatro dimensiones. Por encima de 4/n, el país influye de forma apreciable en las estimaciones.",
             controls=seg("cook", [("cook_media", "Nivel (media)"), ("cook_volatilidad", "Volatilidad")]), fig_key="cook_media",
             read=(f"<b>Lectura.</b> Los países más influyentes en el nivel son {top_cook}: casos en los extremos de la distribución, tanto por inflación alta como por "
                   "inflación muy baja. Que influyan no significa que deban excluirse; significa que hay que comprobar que el resultado no depende de ellos."),
             height=380)
body += card("sens", "¿Depende H3 del tratamiento de los extremos?",
             "Mejora del R² ajustado al sumar las dimensiones no monetarias, en cada especificación (punto e intervalo bootstrap del 95%). En verde, el rango al quitar los países de a uno.",
             read=(f"<b>Resultado.</b> {sens_txt} Sin los países influyentes, la mejora en el nivel "
                   f"pasa de {fmt(sm_.loc[('media', 'principal'), 'delta_r2adj'], 3)} a {fmt(sm_.loc[('media', 'sin_influyentes'), 'delta_r2adj'], 3)}; al quitar cualquier país "
                   f"va de {fmt(sm_.loc[('media', 'jackknife_min'), 'delta_r2adj'], 3)} a {fmt(sm_.loc[('media', 'jackknife_max'), 'delta_r2adj'], 3)}. Los extremos no sostienen el "
                   "resultado; si algo, lo atenúan."),
             foot="Sin países influyentes: reestimación sin intervalo bootstrap (p HC3 &lt; 0,001). p de cada especificación en el cursor (corregido por Holm salvo jackknife).",
             full=True, height=380)
body += card("tests", "Pruebas realizadas y tratamiento de cada caso", "Resumen de las reglas, el criterio exacto, cuántos países afecta y cómo se comprobó que la decisión no determina los resultados.",
             read=("<b>Principio.</b> Ningún valor de la inflación se recorta, se winsoriza ni se imputa. Los casos extremos se conservan siempre que puedan medirse, y su "
                   "efecto se controla con transformaciones, inferencia robusta y especificaciones alternativas."),
             full=True, body=table_tests)
body += "</div>"
(OUT / "04_extremos.html").write_text(page("04_extremos.html", "Distribuciones y casos extremos",
    "Cómo se distribuyen las variables del análisis, qué países son extremos y por qué, y cómo se tratan.",
    "Mostrar la forma de las distribuciones que se modelan, detectar los casos extremos con reglas explícitas, decidir su tratamiento y comprobar que las conclusiones no dependen de esa decisión.",
    body, figs, REPO, sources=SOURCES), encoding="utf-8")

# ---------------------------------------------------------------- 5 · H1
figs = {}
tgt = targets.load("principal")
sizes = labels.value_counts()
m = go.Figure(go.Choropleth(locations=labels.index, z=[ORDER.index(int(k)) for k in labels], text=[NAMES[int(k)] for k in labels],
                            colorscale=[[i / 4, GCOL[k]] for i, k in enumerate(ORDER)], zmin=0, zmax=4, showscale=False,
                            hovertemplate="%{location} · %{text}<extra></extra>", marker_line_color="#fcfcfb", marker_line_width=0.4))
for k in ORDER:
    m.add_scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=GCOL[k], symbol="square"), name=f"{NAMES[k]} ({sizes[k]})")
m.update_layout(geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)", landcolor="#efeee9", showland=True),
                xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.02))
figs["groupmap"] = fig_json(m, 460)
figs["groupmap"]["layout"]["margin"] = dict(l=0, r=0, t=10, b=10)

X = pd.read_parquet(D / "features_crudas.parquet")
ind = {"FS.AST.PRVT.GD.ZS": "Crédito al sector privado (% PIB)", "FM.LBL.BMNY.GD.ZS": "Dinero amplio (% PIB)", "FM.LBL.BMNY.ZG": "Crecimiento del dinero amplio",
       "GC.XPN.INTP.RV.ZS": "Pago de intereses (% ingresos)", "GC.REV.XGRT.GD.ZS": "Ingresos fiscales (% PIB)", "GC.TAX.TOTL.GD.ZS": "Recaudación tributaria (% PIB)",
       "NE.CON.GOVT.ZS": "Consumo del gobierno (% PIB)", "SH.XPD.GHED.GD.ZS": "Gasto público en salud (% PIB)", "NY.GNS.ICTR.ZS": "Ahorro bruto (% PIB)",
       "FI.RES.TOTL.MO": "Reservas (meses de importación)"}
ind = {k: v for k, v in ind.items() if f"{k}__media" in X}
R = X[[f"{c}__media" for c in ind]].rank(pct=True).mul(100)
prof = R.loc[labels.index].groupby(labels).median().T[ORDER]
f = go.Figure(go.Heatmap(z=prof.values, x=[SHORT[k] for k in ORDER], y=list(ind.values()), colorscale=DIVERGING, zmid=50, zmin=0, zmax=100,
                         texttemplate="%{z:.0f}", hovertemplate="%{y} · %{x}: percentil %{z:.0f}<extra></extra>", xgap=2, ygap=2,
                         colorbar=dict(thickness=10, title="pct.")))
f.update_layout(xaxis=dict(side="top", showgrid=False, tickangle=-35, tickfont=dict(size=10)), yaxis=dict(autorange="reversed", showgrid=False))
figs["profiles"] = fig_json(f, 540)
figs["profiles"]["layout"]["margin"] = dict(l=210, r=10, t=110, b=10)
low = prof[1].loc[[f"{c}__media" for c in ["FS.AST.PRVT.GD.ZS", "FM.LBL.BMNY.GD.ZS", "GC.TAX.TOTL.GD.ZS"] if c in ind]]

y = tgt.loc[labels.index]
TIT = {"media": "Inflación media (%)", "tendencia": "Tendencia (p.p./año)", "volatilidad": "Volatilidad (p.p.)"}
TICK = {"media": [0.3, 1, 3, 10, 30], "volatilidad": [0.5, 1, 3, 10, 30, 100], "tendencia": [-10, -1, 0, 1, 10]}
for stat in STAT:
    f = go.Figure()
    for k in ORDER:
        v = y.loc[labels[labels == k].index, stat].dropna()
        f.add_box(y=TRF[stat](v), name=SHORT[k], boxpoints="all", jitter=0.35, pointpos=0, marker=dict(size=5, color=GCOL[k]),
                  line=dict(color=GCOL[k], width=1.5), fillcolor="rgba(0,0,0,0)", customdata=np.c_[[M.loc[i, "pais"] for i in v.index], v.values],
                  hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}<extra></extra>", showlegend=False)
    f.update_layout(yaxis=dict(title=TIT[stat], tickvals=TRF[stat](np.array(TICK[stat], dtype=float)), ticktext=[str(v).replace(".", ",") for v in TICK[stat]]),
                    xaxis=dict(tickangle=0, tickfont=dict(size=11)))
    figs[f"box_{stat}"] = fig_json(f, 440)
rent = y.loc[labels[labels == 3].index, "media"].dropna()
cv_grp = y["media"].groupby(labels).apply(lambda v: v.std() / v.mean())

tests = pd.read_csv(D / "h1_tests.csv")
f = go.Figure()
f.add_bar(y=[TIT[s] for s in tests["estadistico"]], x=100 * tests["eta2_transformado"], orientation="h", name="Entre grupos", marker_color=CATEGORICAL[0],
          text=[f"{fmt(100 * v, 0)}%" for v in tests["eta2_transformado"]], textposition="inside", hovertemplate="%{y}: %{x:.1f}% entre grupos<extra></extra>")
f.add_bar(y=[TIT[s] for s in tests["estadistico"]], x=100 * (1 - tests["eta2_transformado"]), orientation="h", name="Dentro de los grupos", marker_color="#c3c2b7",
          text=[f"{fmt(100 * (1 - v), 0)}%" for v in tests["eta2_transformado"]], textposition="inside", hovertemplate="%{y}: %{x:.1f}% dentro de los grupos<extra></extra>")
f.update_layout(barmode="stack", xaxis=dict(range=[0, 100], title="% de la variación total"), yaxis=dict(autorange="reversed", showgrid=False), bargap=0.4, margin=dict(l=170))
figs["decomp"] = fig_json(f, 300)
figs["decomp"]["layout"]["margin"]["l"] = 170

reg_ref = pd.read_csv(D / "h1_region.csv").set_index("estadistico")
f = go.Figure()
for j, (col, lab) in enumerate([("eta2_perfil", "Perfil monetario-fiscal (5 grupos)"), ("eta2_region", "Región ONU (5 regiones)")]):
    f.add_bar(x=[STAT[s] for s in reg_ref.index], y=100 * reg_ref[col], name=lab, marker_color=[CATEGORICAL[0], "#c3c2b7"][j],
              text=[f"{fmt(100 * v, 0)}%" for v in reg_ref[col]], textposition="outside", hovertemplate="%{x} · " + lab + ": %{y:.1f}%<extra></extra>")
f.update_layout(barmode="group", yaxis=dict(title="% de la variación explicada (η²)", range=[0, 50]), legend=dict(orientation="h", y=1.1, x=0))
figs["region"] = fig_json(f, 340)

sens = pd.read_csv(D / "h1_sensibilidad.csv")
sens["config"] = sens["algoritmo"].str.replace("Jerarquico", "Jerárquico") + " " + sens["parametro"].replace("-", "") + " · k=" + sens["k"].astype(str)
f = go.Figure()
for i, s in enumerate(STAT):
    f.add_scatter(x=100 * sens[f"eta2_{s}"], y=sens["config"], mode="markers", name=STAT[s], marker=dict(size=10, color=CATEGORICAL[i], symbol=["circle", "square", "diamond"][i]),
                  hovertemplate="%{y} · " + STAT[s] + ": %{x:.1f}%<extra></extra>")
f.update_layout(xaxis=dict(title="% de la variación explicada por el grupo", range=[0, 50]), yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=190))
figs["sens"] = fig_json(f, 460)
figs["sens"]["layout"]["margin"]["l"] = 190
stable = sens[sens["ari_bootstrap"] >= P["clustering"]["ari_minimo"]]
rng = {s: (100 * stable[f"eta2_{s}"].min(), 100 * stable[f"eta2_{s}"].max()) for s in STAT}

grid = pd.read_csv(D / "clustering_grid.csv").dropna(subset=["ari_bootstrap"])
grid["config"] = grid["algoritmo"].str.replace("Jerarquico", "Jerárq.") + " " + grid["parametro"].replace("-", "") + " k=" + grid["k"].astype(str)
ch = (grid["algoritmo"] == summ_cl["config"]["algoritmo"]) & (grid["k"] == summ_cl["config"]["k"])
f = go.Figure()
f.add_hrect(y0=P["clustering"]["ari_minimo"], y1=1.02, fillcolor=NEUTRAL, line_width=0, layer="below")
show_lab = ch | (grid["silhouette"] == grid["silhouette"].max())
f.add_scatter(x=grid["silhouette"], y=grid["ari_bootstrap"], mode="markers+text", text=np.where(show_lab, grid["config"], ""), customdata=grid["config"],
              textposition="top left", textfont=dict(size=11, color=INK_SECONDARY),
              marker=dict(size=np.where(ch, 16, 10), color=np.where(ch, CATEGORICAL[1], CATEGORICAL[0]), line=dict(color="#fcfcfb", width=2)),
              hovertemplate="%{customdata}<br>silhouette %{x:.3f} · ARI %{y:.2f}<extra></extra>", showlegend=False)
f.update_layout(xaxis_title="Silhouette (separación entre grupos)", yaxis=dict(title="ARI bootstrap (estabilidad)", range=[0, 1.05]))
figs["stab"] = fig_json(f, 420)
gmm4 = grid[(grid["algoritmo"] == "GMM") & (grid["parametro"] == "tied") & (grid["k"] == 4)]

e = tests.set_index("estadistico")
n_mon = len([c for c in scores.columns if c.startswith("MONETARIA_")])
eta_txt = f"{fmt(100 * e.loc['media', 'eta2_transformado'], 0)}% de la variación de la media, {fmt(100 * e.loc['tendencia', 'eta2_transformado'], 0)}% de la tendencia y {fmt(100 * e.loc['volatilidad', 'eta2_transformado'], 0)}% de la volatilidad"
within = (100 * (1 - e["eta2_transformado"].max()), 100 * (1 - e["eta2_transformado"].min()))
body = kpis([(str(summ_cl["config"]["k"]), "grupos (K-means)"), (str(summ_cl["n_cluster"]), "países agrupados"),
             (f"{fmt(100 * e.loc['media', 'eta2_transformado'], 0)}%", "de la variación de la inflación media explicada por el grupo"),
             (f"{fmt(100 * e.loc['tendencia', 'eta2_transformado'], 0)}%", "de la tendencia"), (f"{fmt(100 * e.loc['volatilidad', 'eta2_transformado'], 0)}%", "de la volatilidad")])
body += '<div class="grid">'
body += card("groupmap", "Cinco perfiles monetario-fiscales",
             f"Países agrupados por su perfil monetario-fiscal 2000-2020 ({n_mon} componentes principales de la dimensión monetaria). La inflación no interviene en la formación de los grupos.",
             read="<b>Lectura.</b> Los grupos tienen una geografía reconocible: economías avanzadas, un amplio grupo de ingreso medio, rentistas de recursos naturales, pequeñas economías con Estado grande y economías de baja profundidad financiera, sobre todo en África subsahariana.",
             foot=f"En blanco: los {len(summ_cl['atipicos'])} países atípicos en el espacio monetario (excluidos del ajuste) y los {len(summ_feat['paises_excluidos'])} excluidos por información insuficiente.", full=True, height=460)
body += card("profiles", "¿Qué distingue a cada grupo?",
             f"Mediana de cada grupo en indicadores monetario-fiscales seleccionados, expresada como percentil entre los {summ_feat['paises_analisis']} países (50 = país típico).",
             read=f"<b>Lectura.</b> Las avanzadas combinan la mayor profundidad financiera y presión fiscal. Las rentistas ahorran mucho y recaudan poco: el Estado se financia con la renta de los recursos. En las de baja profundidad financiera, crédito, dinero y recaudación están entre los percentiles {fmt(low.min(), 0)} y {fmt(low.max(), 0)}.",
             foot="Se usan percentiles porque algunos países tienen valores extremos (por ejemplo, el crecimiento monetario en hiperinflaciones) que distorsionarían una escala en desvíos estándar.", height=480)
body += card("box", "La inflación dentro de cada grupo",
             "Distribución de los estadísticos de inflación en cada grupo: cada punto es un país. Elegí el estadístico.",
             read=f"<b>Lectura.</b> Las diferencias entre grupos existen, sobre todo en volatilidad, pero domina la dispersión interna: en «Rentistas de recursos» la inflación media va de {fmt(rent.min())}% ({es(rent.idxmin())}) a {fmt(rent.max())}% ({es(rent.idxmax())}). Las avanzadas son el grupo más homogéneo (coeficiente de variación {fmt(cv_grp[2], 2)}, frente a {fmt(cv_grp.drop(2).min(), 2)}-{fmt(cv_grp.drop(2).max(), 2)} en el resto).",
             foot="Ejes en escala logarítmica (media, volatilidad) y asinh (tendencia).", height=440,
             controls=seg("box", [("box_media", "Media"), ("box_tendencia", "Tendencia"), ("box_volatilidad", "Volatilidad")]), fig_key="box_media")
body += card("decomp", "Prueba de H1: ¿cuánta variación queda dentro de los grupos?",
             "Descomposición de la variación de cada estadístico (escala transformada) entre grupos y dentro de los grupos (η²).",
             read=f"<b>Resultado.</b> El perfil monetario-fiscal explica una parte significativa pero minoritaria: entre el {fmt(within[0], 0)}% y el {fmt(within[1], 0)}% de la variación ocurre <b>dentro</b> de los grupos. <b>H1 se sostiene</b>: compartir perfil no implica compartir trayectoria inflacionaria.",
             foot=f"Permutación (5.000): p &lt; 0,001 para media y volatilidad; p = {fmt(e.loc['tendencia', 'p_permutacion'], 2)} para la tendencia. Muestra: {int(e.loc['media', 'n'])} países con al menos {P['trayectorias']['min_anios_target']} años de IPC.",
             full=True, height=300)
body += card("region", "¿Son los perfiles solo regiones?",
             "Variación de la inflación explicada por el perfil monetario-fiscal frente a la explicada por la región ONU, con la misma cantidad de grupos.",
             read=f"<b>Lectura.</b> Los perfiles tienen una geografía marcada (V de Cramér con la región = {fmt(reg_ref['cramer_v_perfil_region'].iloc[0], 2)}), pero explican {fmt((reg_ref['eta2_perfil'] / reg_ref['eta2_region']).loc[['media', 'volatilidad']].min(), 1)} a {fmt((reg_ref['eta2_perfil'] / reg_ref['eta2_region']).loc[['media', 'volatilidad']].max(), 1)} veces más variación que la región en la media y la volatilidad: capturan algo más que la ubicación geográfica.",
             foot="η² con las mismas transformaciones; significancia por permutación.", height=340)
body += card("sens", "¿Depende el resultado del algoritmo elegido?",
             f"Variación explicada por el grupo en {len(sens)} configuraciones alternativas de clustering.",
             read=f"<b>Lectura.</b> En las configuraciones estables, el grupo explica {fmt(rng['media'][0], 0)}-{fmt(rng['media'][1], 0)}% de la media, {fmt(rng['tendencia'][0], 0)}-{fmt(rng['tendencia'][1], 0)}% de la tendencia y {fmt(rng['volatilidad'][0], 0)}-{fmt(rng['volatilidad'][1], 0)}% de la volatilidad. En ninguna supera el 40%: la conclusión de H1 no depende de la configuración.",
             foot="Los valores más bajos corresponden a soluciones jerárquicas inestables.", height=460)
body += card("stab", "Calidad vs. estabilidad: cómo se eligió la configuración",
             "Las 12 mejores configuraciones según silhouette, frente a su estabilidad cuando se vuelve a agrupar con submuestras del 80% de los países.",
             read=f"<b>Lectura.</b> Todas las silhouettes son bajas (≤ {fmt(grid['silhouette'].max(), 2)}): los perfiles forman un continuo más que tipos discretos. La regla, fijada antes de ver los resultados, exige estabilidad (ARI ≥ {fmt(P['clustering']['ari_minimo'], 2)}, zona gris) y grupos de al menos {P['clustering']['tamanio_minimo']} países, y elige K-means con k = {summ_cl['config']['k']} (naranja, ARI {fmt(summ_cl['config']['ari_bootstrap'], 2)}).",
             foot=(f"GMM tied con k = 4, la configuración de la versión original de la tesis, tiene ARI {fmt(gmm4['ari_bootstrap'].iloc[0], 2)}." if len(gmm4) else ""), height=420)
body += "</div>"
(OUT / "05_h1_perfiles.html").write_text(page("05_h1_perfiles.html", "H1 · Perfiles monetario-fiscales y heterogeneidad inflacionaria",
    "¿Los países con perfiles monetario-fiscales similares tienen trayectorias inflacionarias similares?",
    "Agrupar a los países según su perfil monetario-fiscal y medir cuánta de la variación inflacionaria queda dentro de esos grupos. Si la mayor parte de la variación es interna, compartir un perfil monetario-fiscal no basta para explicar la trayectoria (Hipótesis 1).",
    body, figs, REPO, sources=SOURCES), encoding="utf-8")

# ---------------------------------------------------------------- 6 · H2/H3
figs = {}
K = P["regresion"]["componentes_por_bloque"]
diag = pd.read_csv(D / "reg_diagnosticos.csv")
f = go.Figure()
for stat, g in diag.groupby("estadistico", sort=False):
    g = g.set_index("modelo")
    f.add_scatter(x=[g.loc["A", "r2adj"], g.loc["B", "r2adj"]], y=[STAT[stat]] * 2, mode="lines", line=dict(color="#c3c2b7", width=4), showlegend=False, hoverinfo="skip")
for j, (mod, lab) in enumerate([("A", "Modelo A · solo monetario"), ("B", "Modelo B · cuatro dimensiones")]):
    g = diag[diag["modelo"] == mod]
    f.add_scatter(x=g["r2adj"], y=g["estadistico"].map(STAT), mode="markers+text", name=lab, text=[fmt(v, 2) for v in g["r2adj"]],
                  textposition="top center" if mod == "B" else "bottom center", marker=dict(size=15, color=CATEGORICAL[j], line=dict(color="#fcfcfb", width=2)),
                  hovertemplate="%{y} · " + lab + ": %{x:.3f}<extra></extra>")
f.update_layout(xaxis=dict(title="R² ajustado", range=[0, 0.5]), yaxis=dict(autorange="reversed", showgrid=False))
figs["r2"] = fig_json(f, 340)
figs["r2"]["layout"]["margin"].update(l=100, t=40)
dg = diag.set_index(["estadistico", "modelo"])

VAR = {"": "Principal", "_k2": "2 comp. por bloque", "_k3": "3 comp. por bloque", "_deflactor": "Deflactor del PIB", "_mediana": "Mediana anual",
       "_theil_sen": "Tendencia Theil-Sen", "_con_atipicos": "Con atípicos", "_tasas_nominales": "Control + tasas nominales"}
rows = []
for suf, nm in VAR.items():
    for hip in ["h2", "h3"]:
        rows.append(pd.read_csv(D / f"reg_{hip}{suf}.csv").assign(spec=nm, hip=hip.upper()))
RR = pd.concat(rows)
RR["prueba"] = RR["hip"] + " · " + RR["estadistico"].map(STAT) + " · " + RR["bloque"].map(lambda b: DIM_LABELS.get(b, "No monetarias"))
RR["nivel"] = np.where(RR["solido"], 2, np.where(RR[["p", "p_HC3", "p_permutacion"]].min(axis=1) < 0.05, 1, 0))
order = RR.drop_duplicates("prueba")["prueba"].tolist()
Mv = RR.pivot(index="prueba", columns="spec", values="nivel").loc[order, list(VAR.values())]
Pp = RR.pivot(index="prueba", columns="spec", values="p_HC3_holm").loc[order, list(VAR.values())]
cs = [[0, "#fcfcfb"], [0.333, "#fcfcfb"], [0.334, "#cde2fb"], [0.666, "#cde2fb"], [0.667, "#184f95"], [1, "#184f95"]]
f = go.Figure(go.Heatmap(z=Mv.values, x=list(Mv.columns), y=list(Mv.index), zmin=-0.5, zmax=2.5, colorscale=cs, text=Pp.map(lambda v: fmt(v, 3)).values,
                         texttemplate="%{text}", xgap=2, ygap=2, hovertemplate="%{y} · %{x}<br>p HC3 corregido por Holm: %{text}<extra></extra>",
                         colorbar=dict(tickvals=[0, 1, 2], ticktext=["Sin evidencia", "Indicio", "Sólido"], thickness=10)))
f.update_layout(xaxis=dict(side="top", showgrid=False, tickangle=-30), yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=270, t=90))
figs["robust"] = fig_json(f, 600)
figs["robust"]["layout"]["margin"] = dict(l=270, r=20, t=90, b=20)

main_tests = RR[RR["spec"] == "Principal"]
f = go.Figure()
for i, (hip, g) in enumerate(main_tests.groupby("hip")):
    f.add_scatter(x=g["delta_r2adj"], y=g["prueba"], mode="markers", name=hip, marker=dict(size=11, color=CATEGORICAL[i], symbol=["circle", "diamond"][i]),
                  error_x=dict(type="data", symmetric=False, array=g["delta_ic95_sup"] - g["delta_r2adj"], arrayminus=g["delta_r2adj"] - g["delta_ic95_inf"],
                               color=INK_MUTED, thickness=1.2, width=4),
                  customdata=np.c_[g["p_HC3_holm"], g["p_permutacion"], g["f2"]],
                  hovertemplate="%{y}<br>ΔR² aj. %{x:.3f} · f² %{customdata[2]:.2f}<br>p HC3 (Holm) %{customdata[0]:.3f} · p perm. %{customdata[1]:.3f}<extra></extra>")
f.add_vline(x=0, line=dict(color=INK_SECONDARY, width=1))
f.update_layout(xaxis=dict(title="ΔR² ajustado", zeroline=False), yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=270))
figs["delta"] = fig_json(f, 500)
figs["delta"]["layout"]["margin"]["l"] = 270

cvr = pd.read_csv(D / "reg_cv.csv")
SPEC_CV = {"ols_k_balanceado": f"OLS · {K} componentes por bloque", "ridge_todos_los_componentes": "Ridge · todos los componentes"}
f = go.Figure()
for j, (sp, lab) in enumerate(SPEC_CV.items()):
    g = cvr[cvr["especificacion"] == sp]
    f.add_scatter(x=g["delta_r2_oos"], y=g["estadistico"].map(STAT), mode="markers", name=lab, marker=dict(size=12, color=CATEGORICAL[j], symbol=["circle", "diamond"][j]),
                  error_x=dict(type="data", symmetric=False, array=g["delta_p95"] - g["delta_r2_oos"], arrayminus=g["delta_r2_oos"] - g["delta_p05"], color=CATEGORICAL[j]),
                  customdata=g["pct_reps_mejora"], hovertemplate="%{y} · " + lab + ": %{x:.3f}<br>mejora en %{customdata:.0f}% de las repeticiones<extra></extra>")
f.add_vline(x=0, line=dict(color=INK_SECONDARY, width=1))
f.update_layout(xaxis_title="Cambio en el R² fuera de muestra (percentiles 5-95)", yaxis=dict(autorange="reversed", showgrid=False), legend=dict(orientation="h", y=1.12, x=0))
figs["oos"] = fig_json(f, 340)
figs["oos"]["layout"]["margin"]["l"] = 100
cvm = cvr.set_index(["estadistico", "especificacion"])

pca = pickle.load(open(D / "pca_resultados.pkl", "rb"))
import statsmodels.api as sm
ym = np.log(tgt.loc[labels.index, "media"].dropna())
cols = [f"{d}_media__PC{i + 1}" for d in DIMENSION_COLORS for i in range(min(K, pca[f"{d}_media"].n_componentes))]
fit = sm.OLS(ym, sm.add_constant(scores.loc[ym.index, cols])).fit(cov_type="HC3")
cf = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues}).drop("const")
cf["dim"] = cf.index.str.split("_").str[0]
f = go.Figure()
for dim, c in DIMENSION_COLORS.items():
    g = cf[cf["dim"] == dim]
    f.add_scatter(x=g["coef"], y=[f"{DIM_LABELS[dim]} PC{s.split('PC')[1]}" for s in g.index], mode="markers", name=DIM_LABELS[dim], marker=dict(size=11, color=c),
                  error_x=dict(type="data", array=1.96 * g["se"], color=c, thickness=1.5, width=0), customdata=g["p"],
                  hovertemplate="%{y}: %{x:.3f} (p = %{customdata:.3f})<extra></extra>")
f.add_vline(x=0, line=dict(color=INK_SECONDARY, width=1))
f.update_layout(xaxis_title="Coeficiente sobre log(inflación media)", yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=170))
figs["coef"] = fig_json(f, 520)
figs["coef"]["layout"]["margin"]["l"] = 170
sig = cf[cf["p"] < 0.05]

h3 = main_tests[main_tests["hip"] == "H3"].set_index("estadistico")
h2_solid = main_tests[(main_tests["hip"] == "H2") & main_tests["solido"]]
h2_any = RR[(RR["hip"] == "H2") & RR["solido"] & (RR["spec"] != "Control + tasas nominales")]
h2_txt = ("Ninguna dimensión no monetaria aporta por sí sola en la especificación principal, una vez corregidas las comparaciones múltiples (Holm) y con los bloques en igualdad de condiciones."
          + (" Solo aparece " + "; ".join(f"la {DIM_LABELS[r.bloque].lower()} sobre la {STAT[r.estadistico].lower()} con {r.spec.lower().replace('ipc', 'IPC').replace('pib', 'PIB')}" for r in h2_any.itertuples())
             + f", en {h2_any['spec'].nunique()} de {len(VAR) - 1} especificaciones." if len(h2_any) else ""))
h3_solid = [STAT[s].lower() for s in h3.index if h3.loc[s, "solido"]]
tn = RR[(RR["spec"] == "Control + tasas nominales") & (RR["hip"] == "H3")].set_index("estadistico")
spec_ok = RR[(RR["hip"] == "H3") & (RR["estadistico"] == "media")]
verdicts = f"""<div class="cards3">
<div class="hyp"><span class="tag yes">● Respaldada</span><h4>H1 · Perfil compartido, trayectorias distintas</h4>
<p>El grupo monetario-fiscal explica {eta_txt}: la mayor parte de la variación ocurre dentro de los grupos.</p></div>
<div class="hyp"><span class="tag {'partial' if len(h2_solid) else 'no'}">● {'Evidencia parcial' if len(h2_solid) else 'No respaldada'}</span><h4>H2 · Aporte de cada dimensión</h4>
<p>{h2_txt}</p></div>
<div class="hyp"><span class="tag partial">● Parcial: asociación, no predicción</span><h4>H3 · Modelo multidimensional</h4>
<p>Las tres dimensiones juntas mejoran el R² ajustado del nivel de {fmt(h3.loc['media', 'r2adj_restringido'], 2)} a {fmt(h3.loc['media', 'r2adj_completo'], 2)} (f² = {fmt(h3.loc['media', 'f2'], 2)}), de forma robusta en {int(spec_ok['solido'].sum())} de {len(spec_ok)} especificaciones (falla solo si se agregan tasas nominales al control); también la volatilidad. Pero la mejora casi no se traslada a la predicción fuera de muestra.</p></div></div>"""
body = verdicts + '<div class="grid">'
body += card("r2", "Modelo monetario frente a modelo de cuatro dimensiones",
             f"R² ajustado de cada estadístico con solo componentes monetarios (A) y con las cuatro dimensiones (B), {K} componentes por dimensión.",
             read=f"<b>Lectura.</b> El perfil monetario explica parte del nivel ({fmt(dg.loc[('media', 'A'), 'r2adj'], 2)}) y de la volatilidad ({fmt(dg.loc[('volatilidad', 'A'), 'r2adj'], 2)}), y casi nada de la tendencia ({fmt(dg.loc[('tendencia', 'A'), 'r2adj'], 2)}). Agregar las tres dimensiones no monetarias suma entre {fmt(h3['delta_r2adj'].min(), 2)} y {fmt(h3['delta_r2adj'].max(), 2)} de R² ajustado.",
             foot=f"{int(dg.loc[('media', 'B'), 'n'])} países. Variables dependientes: log(media), asinh(tendencia), log(volatilidad). Cada dimensión entra con {K} componentes principales, el máximo que respeta 10 países por predictor, para comparar bloques con los mismos grados de libertad.",
             full=True, height=320)
body += card("delta", "Aporte de cada bloque (especificación principal)",
             "Mejora del R² ajustado al agregar cada dimensión (H2) o las tres juntas (H3), con su intervalo de confianza bootstrap.",
             read=f"<b>Lectura.</b> Ninguna dimensión aporta por sí sola de forma sólida. Juntas, en cambio, mejoran el ajuste del {' y la '.join(h3_solid)}: el aporte es <b>conjunto</b>. Con f² = {fmt(h3.loc['media', 'f2'], 2)} en el nivel, el efecto es mediano según la convención de Cohen (0,02 pequeño; 0,15 mediano; 0,35 grande).",
             foot="Pasá el cursor para ver f², el p-valor HC3 corregido por Holm y el de permutación. IC 95% bootstrap percentil (2.000 remuestras).", full=True, height=500)
body += card("oos", "¿Mejora la capacidad de predecir?",
             "Cambio en el R² fuera de muestra al sumar las dimensiones no monetarias (validación cruzada de 10 particiones, 50 repeticiones).",
             read=f"<b>Lectura.</b> La mejora dentro de muestra casi no se traslada a países no vistos: en el nivel, el R² fuera de muestra cambia {fmt(cvm.loc[('media', 'ols_k_balanceado'), 'delta_r2_oos'], 3)} con OLS y {fmt(cvm.loc[('media', 'ridge_todos_los_componentes'), 'delta_r2_oos'], 3)} con ridge. Solo con ridge y todos los componentes aparece una ganancia pequeña pero consistente en la volatilidad (mejora en el {fmt(cvm.loc[('volatilidad', 'ridge_todos_los_componentes'), 'pct_reps_mejora'], 0)}% de las repeticiones). Las dimensiones no monetarias describen diferencias entre países, pero no son una herramienta de predicción.",
             foot="Ridge con penalización elegida por validación cruzada interna, usando todos los componentes retenidos por el PCA (70% de la varianza de cada bloque).", full=True, height=340)
body += card("robust", f"Robustez en {len(VAR)} especificaciones",
             "Nivel de evidencia de cada prueba bajo especificaciones alternativas. El número es el p-valor HC3 corregido por Holm.",
             read=f"<b>Lectura.</b> H3 en el nivel es sólida en todas las especificaciones salvo una: si se agregan las tasas de interés nominales al control monetario, la mejora desaparece (p Holm = {fmt(tn.loc['media', 'p_HC3_holm'], 2)}). Las tasas nominales incorporan la inflación esperada (efecto Fisher), por eso quedan fuera de la especificación principal; pero el resultado obliga a leer H3 como una asociación condicionada a esa decisión.",
             foot=f"Sólido: p HC3 corregido por Holm &lt; 0,05, p de permutación &lt; 0,05 e IC del ΔR² por encima de 0. Indicio: alguna prueba sin corregir con p &lt; 0,05. La mediana solo cambia el nivel y Theil-Sen solo la tendencia: en las demás filas repiten la especificación principal. Con tasas nominales hay {fmt(dg.loc[('media', 'B'), 'n'] / (4 * K + 2), 1)} países por predictor.", full=True, height=600)
body += card("coef", "¿Qué hay detrás del resultado para el nivel de inflación?",
             "Coeficientes del modelo completo para log(inflación media), con intervalos del 95% (errores HC3).",
             read=f"<b>Lectura.</b> {('Ningún componente es significativo por separado' if sig.empty else f'Solo {len(sig)} de {len(cf)} componentes son significativos por separado (' + ', '.join(DIM_LABELS[c.split('_')[0]].lower() + ' PC' + c.split('PC')[1] for c in sig.index) + ')')}. El aporte de las dimensiones no monetarias es sobre todo <b>conjunto</b>, coherente con un fenómeno multicausal y con que las dimensiones comparten un gradiente de desarrollo.",
             foot=f"Asociaciones, no efectos causales: cada componente resume decenas de indicadores correlacionados (VIF máximo {fmt(dg.loc[('media', 'B'), 'vif_max'], 1)}). La colinealidad entre bloques impide interpretar coeficientes individuales, pero no afecta a las pruebas conjuntas.", full=True, height=520)
body += "</div>"
(OUT / "06_h2_h3_modelos.html").write_text(page("06_h2_h3_modelos.html", "H2 y H3 · ¿Qué aportan las dimensiones no monetarias?",
    "Modelos de regresión anidados que controlan por el perfil monetario-fiscal.",
    "Medir si las dimensiones institucional, estructural y sociodemográfica explican diferencias en las trayectorias inflacionarias más allá del perfil monetario-fiscal, cada una por separado (H2) y en conjunto (H3), comprobar si las conclusiones resisten especificaciones alternativas y si se traducen en capacidad predictiva.",
    body, figs, REPO, sources=SOURCES + " Inferencia: F clásico, Wald robusto HC3, permutación del bloque (2.000), bootstrap de pares (2.000), corrección de Holm y validación cruzada repetida."), encoding="utf-8")

# ---------------------------------------------------------------- Inicio
n_ind = summ_feat["indicadores_cumplen_umbral"]
tiles = [("01_datos.html", "01", "Datos", f"Fuentes, cobertura y cómo se construyó el universo de {n_ind} indicadores y {summ_feat['paises_analisis']} países."),
         ("02_clasificacion.html", "02", "Clasificación con LLM", "Validación de la asignación de indicadores a las cuatro dimensiones teóricas."),
         ("03_inflacion.html", "03", "Inflación", "Explorador por país, episodios de inflación alta y los tres estadísticos de trayectoria."),
         ("04_extremos.html", "04", "Distribuciones y extremos", "Forma de las distribuciones, detección de casos extremos, pruebas y tratamiento."),
         ("05_h1_perfiles.html", "05", "H1 · Perfiles", "Grupos monetario-fiscales y heterogeneidad de la inflación dentro de ellos."),
         ("06_h2_h3_modelos.html", "06", "H2–H3 · Modelos", "Aporte de las dimensiones no monetarias, robustez y validación fuera de muestra.")]
tile_html = lambda ts: '<div class="tiles">' + "".join(f'<a class="tile" href="{h}"><span class="num">{n}</span><h4>{t}</h4><p>{d}</p></a>' for h, n, t, d in ts) + "</div>"
body = (f'<div class="objective"><h2>Pregunta de investigación</h2><p>¿En qué medida compartir un perfil monetario-fiscal explica que los países tengan trayectorias inflacionarias similares, y qué aportan las dimensiones institucional, estructural y sociodemográfica a las diferencias que persisten? Se analizan {summ_feat["paises_analisis"]} países miembros de la ONU entre 2000 y 2020, combinando {n_ind} indicadores clasificados con un modelo de lenguaje, reducción de dimensionalidad (PCA), clustering y regresiones anidadas.</p></div>'
        + kpis([(str(summ_feat["paises_analisis"]), "países"), (str(n_ind), "indicadores"), ("3", "fuentes"), ("21", "años"), ("3", "hipótesis")]) + verdicts
        + '<h2 style="font-size:20px;margin:28px 0 12px">Análisis exploratorio</h2>' + tile_html(tiles[:4])
        + '<h2 style="font-size:20px;margin:28px 0 12px">Modelado</h2>' + tile_html(tiles[4:])
        + '<div class="card" style="margin-top:24px"><h3>Método en cinco pasos</h3><ol style="margin:6px 0 0;padding-left:20px;font-size:14px;color:#52514e">'
          '<li>Construcción de una base país-año con WDI, V-Dem y WITS, estandarizada a los 193 miembros de la ONU, sin variables con relación mecánica con la inflación.</li>'
          '<li>Clasificación de cada indicador en una dimensión teórica con un LLM de pesos abiertos (Groq), validada contra una muestra manual.</li>'
          '<li>Resumen de cada serie 2000-2020 en nivel, tendencia y volatilidad; imputación KNN; PCA por dimensión; análisis de distribuciones y casos extremos.</li>'
          '<li>H1: clustering del perfil monetario-fiscal y descomposición de la variación inflacionaria.</li>'
          f'<li>H2-H3: regresiones anidadas con {K} componentes por dimensión, inferencia robusta, permutaciones, bootstrap, corrección de Holm, {len(VAR)} especificaciones y validación fuera de muestra.</li></ol></div>')
hero_sub = (f'Trabajo Final de Máster en Ciencia de Datos · B. Carmela Roman. Proyecto reproducible: código modular, pipeline de punta a punta y 8 notebooks documentados.'
            f'<br><br><a class="btn" href="01_datos.html">Explorar los dashboards</a><a class="btn ghost" href="{REPO}">Ver el repositorio en GitHub</a>')
(OUT / "index.html").write_text(page("index.html", "Multicausalidad de la inflación: un estudio comparativo con Machine Learning", hero_sub, "", body, {}, REPO,
                                     sources=SOURCES), encoding="utf-8")
print("Dashboards generados en", OUT)
for p in sorted(OUT.glob("*.html")):
    print(f"  {p.name}  {p.stat().st_size / 1e3:.0f} kB")
