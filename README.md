# Multicausalidad de la inflación: perfiles monetario-fiscales y heterogeneidad de trayectorias (2000-2020)

Análisis comparado de 183 países miembros de la ONU que pregunta en qué medida compartir un perfil monetario-fiscal explica la convergencia inflacionaria entre países, y qué aportan las dimensiones institucional, estructural y sociodemográfica a la heterogeneidad restante.

Reconstrucción reproducible del Trabajo Final de Máster *"Multicausalidad de la inflación: un estudio comparativo con técnicas de Machine Learning"* (B. Carmela Roman). Los resultados de este repositorio se recalcularon por completo y difieren en parte de la versión entregada (ver [Cambios respecto de la tesis](#cambios-respecto-de-la-tesis-v1)).

## Hipótesis y resultados

| | Hipótesis | Resultado |
|---|---|---|
| **H1** | Compartir perfil monetario-fiscal no implica trayectoria inflacionaria homogénea | **Respaldada.** El grupo explica el 25% de la variación de la inflación media, el 1% de la tendencia y el 35% de la volatilidad; entre el 65% y el 99% queda dentro de los grupos. Robusta a la configuración de clustering, y el perfil explica unas 3 veces más que la región. |
| **H2** | Cada dimensión no monetaria aporta, controlando lo monetario | **No respaldada.** Con los bloques en igualdad de condiciones, ninguna dimensión aporta por sí sola después de la corrección de Holm (un caso aislado con el deflactor, en 1 de 7 especificaciones). |
| **H3** | Las cuatro dimensiones caracterizan mejor que el modelo monetario | **Parcial: asociación, no predicción.** Mejora sólida dentro de muestra en el nivel (R² ajustado 0,27 → 0,33; f² = 0,17; Holm p < 0,001; 7 de 8 especificaciones) y en la volatilidad (0,31 → 0,36), pero casi no mejora la predicción fuera de muestra, y desaparece si el control monetario incluye las tasas de interés nominales. |

Lectura de conjunto: más allá del perfil monetario-fiscal, las configuraciones no monetarias se **asocian** con cuánta inflación tiene un país y con cuán estable es. El efecto es de tamaño mediano y robusto a las pruebas, pero es conjunto (no se atribuye a una sola dimensión) y describe diferencias entre países sin servir para pronosticar.

## Datos

| Fuente | Contenido | Acceso |
|---|---|---|
| World Development Indicators (Banco Mundial) | 1.498 indicadores; versión del 8-oct-2026 | API y descarga masiva |
| V-Dem v16 | 79 índices de nivel intermedio | Paquete oficial [`vdemdata`](https://github.com/vdeminstitute/vdemdata) |
| WITS (Banco Mundial) | Índice Herfindahl-Hirschman de concentración de exportaciones | API SDMX |
| Tabla de países (elaboración propia) | 193 miembros ONU con códigos ISO3, M49, COW y región | `data/raw/df_regiones_miembros_onu.xlsx` |

Variable dependiente: inflación anual del IPC (`FP.CPI.TOTL.ZG`). Alternativa: deflactor del PIB (`NY.GDP.DEFL.KD.ZG`).

## Estructura

```
config/params.yaml            todos los parámetros (umbrales, semillas, criterios de selección)
src/inflacion/
  sources/                    extracción y estandarización: wdi, vdem, mc, países
  classification/             clasificación por LLM (Groq), codebooks V0-V2, evaluación, revisión, selección
  features.py                 cobertura y estadísticos de trayectoria (media, tendencia, volatilidad)
  preprocessing.py            exclusión de países e imputación KNN con validación cruzada
  pca.py                      PCA por bloque con poda por redundancia y MSA
  clustering.py               grilla de algoritmos, estabilidad bootstrap, pruebas de H1
  regression.py               modelos anidados, HC3, bootstrap, permutación, f², validación fuera de muestra
  targets.py                  variable dependiente con el criterio de cobertura temporal
  viz/                        tema gráfico único (paleta validada para daltonismo)
pipeline/                     scripts ejecutables, en orden (ver run_all.sh)
notebooks/                    análisis documentado con visualizaciones e interpretaciones
  01-04                       Exploratory Data Analysis (incluye distribuciones y casos extremos, 03 §5)
  05-08                       Statistical Analysis & Modeling
data/raw | interim | processed
reports/figures               figuras en PNG (estáticas) y HTML (interactivas, base para dashboards)
reports/tables                tabla de clasificación final
dashboards/                   portada y 6 dashboards interactivos (HTML + Plotly.js local)
```

## Reproducir

```bash
conda env create -f environment.yml && conda activate tfm-inflacion
python pipeline/download_data.py      # ~320 MB de datos crudos
./run_all.sh                          # clasificación (desde caché), features, PCA, clustering, regresiones, casos extremos, dashboards
jupyter lab notebooks/
```

Las respuestas del LLM están guardadas en `data/interim/classification/*.jsonl`, así que `run_all.sh` no llama a la API. Para reclasificar desde cero hace falta definir `GROQ_API_KEY` y borrar esos archivos; el modelo (`openai/gpt-oss-120b`, temperatura 0) no es completamente determinista, por lo que una reclasificación puede diferir levemente.

## Dashboards

`dashboards/index.html` es la portada interactiva del proyecto: pregunta de investigación, veredicto de cada hipótesis y acceso a seis dashboards en dos bloques. **Análisis exploratorio**: datos, clasificación, inflación y distribuciones y casos extremos. **Modelado**: H1 y H2-H3. Publicados en [carmelaroman94-png.github.io/TFM_portfolio/dashboards](https://carmelaroman94-png.github.io/TFM_portfolio/dashboards/index.html). Cada uno declara su objetivo y acompaña cada gráfico con una descripción, una lectura y notas al pie. Son HTML estáticos con Plotly.js incluido en `dashboards/assets/`, así que funcionan sin servidor y se pueden publicar en GitHub Pages. Se regeneran con `python pipeline/build_dashboards.py` (última línea de `run_all.sh`). Los mapas necesitan conexión a internet para cargar las geometrías de los países.

## Decisiones metodológicas principales

- **Clasificación de indicadores**: Groq (`gpt-oss-120b`) con el codebook V2, elegido entre tres versiones evaluadas contra 100 indicadores clasificados a mano (accuracy 0,88; kappa 0,83; las versiones no difieren significativamente, McNemar p = 1). El universo se clasificó dos veces: 89% de acuerdo en WDI y 97,5% en V-Dem. Los 162 indicadores sin acuerdo se resolvieron por revisión documentada (`metodo = "clasificacion manual"`, con revisor identificado). Coincide en un 95% con la clasificación de la tesis.
- **Exclusiones previas al análisis**: niveles absolutos no comparables entre países (totales en US$ o en moneda local, series per cápita en moneda local, poblaciones, conteos), medidas de precios (IPC, deflactores, índices de precios, porque son la variable dependiente), tasas de interés (la real se calcula con el deflactor; las nominales incorporan la inflación esperada y solo se usan como prueba de estrés) y ambiguos.
- **Ventana y cobertura**: 2000-2020; indicadores con dato en ≥70% de los países y ≥60% de los años; países con ≤40% de features faltantes; para la inflación, el mismo 60% de los años (13 de 21) que a las features.
- **Trayectorias**: media, pendiente OLS y desvío de residuos. Variables dependientes transformadas: log(media), asinh(tendencia) y log(volatilidad). Variantes robustas: mediana anual y pendiente de Theil-Sen.
- **Imputación**: KNN sobre features estandarizadas; k = 8 por validación cruzada (meseta entre k = 5 y k = 10). La inflación nunca se imputa.
- **Casos extremos**: ocho reglas explícitas (`pipeline/run_extremos.py`). Solo se excluye un país cuando falta información para medirlo o cuando su perfil monetario distorsionaría el clustering; ningún valor de inflación se recorta ni se winsoriza. El resultado de H3 se verifica sin los países influyentes (Cook > 4/n) y con jackknife.
- **PCA por dimensión y estadístico**: duplicados con |r| > 0,99 eliminados, poda por redundancia en bloques grandes, KMO ≥ 0,60, 70% de varianza retenida.
- **Clustering**: regla de selección fijada antes de ver los resultados (grupos de ≥5 países, ARI bootstrap ≥ 0,60, máxima silhouette), que eligió K-means con k = 5. La región ONU se usa como referencia.
- **Regresiones**: el mismo número de componentes por dimensión (4, el máximo con ≥10 países por predictor), para que los bloques compitan con los mismos grados de libertad; inferencia con HC3, permutación y bootstrap; corrección de Holm; f² de Cohen; R² fuera de muestra (validación cruzada repetida, OLS y ridge); ocho especificaciones de robustez.

## Cambios respecto de la tesis (v1)

| Cambio | Motivo |
|---|---|
| Se excluyen el IPC, los deflactores y los índices de precios de las features | En la v1 formaban parte del bloque monetario: fuga de información hacia la variable dependiente |
| Se excluyen las tasas de interés | La real se calcula con el deflactor y las nominales contienen la inflación esperada: también es fuga |
| Se excluyen los niveles absolutos y las series en moneda local en todas las dimensiones | En la v1 solo se excluían en el bloque monetario; las series en moneda local corriente crecen con la inflación |
| Se eliminan duplicados casi exactos antes del PCA | Los índices WGI en dos escalas pesaban doble |
| log(media) en lugar de log(1 + media/100) | La transformación original no corregía la asimetría (4,7 → 4,2; ahora 0,3) |
| Estabilidad bootstrap como criterio del clustering | Con la regla, K-means k = 5 es más estable que GMM tied k = 4, la elección de la v1 |
| Mismo número de componentes por dimensión en las regresiones | Con un piso de varianza, el control monetario quedaba en desventaja y la estructural, favorecida |
| Pruebas por permutación, IC bootstrap del ΔR², corrección de Holm, f² y validación fuera de muestra | Residuos no normales, comparaciones múltiples y distinguir asociación de capacidad predictiva |
| Clasificación íntegramente con Groq (sin el agente de relevancia) | Reproducibilidad del procedimiento |

Con estos cambios, H1 se mantiene; H2 sigue sin respaldo; y H3 pasa de confirmarse solo en la volatilidad a una asociación sólida en el nivel y la volatilidad, acotada por la falta de mejora predictiva.

## Limitaciones

- La serie del IPC excluye los casos más extremos: Argentina solo tiene 2018-2020 en WDI, la de Venezuela termina en 2016, y Zimbabue y Sudán del Sur tienen series cortas. La especificación con el deflactor los recupera parcialmente.
- H3 es una asociación dentro de muestra: casi no mejora la predicción fuera de muestra, y desaparece si el control monetario incluye las tasas de interés nominales.
- Resumir 21 años en tres estadísticos descarta la dinámica: los resultados describen asociaciones entre perfiles de largo plazo, no efectos causales.
- Las dimensiones comparten un gradiente de desarrollo (|r| hasta 0,9 entre primeros componentes; VIF máximo 19,6), lo que vuelve imprecisos los coeficientes individuales y frágil la atribución a una dimensión específica.
- La regla de "variables fiscales en sentido amplio" del codebook lleva el gasto público en salud al bloque monetario.

## Citas de datos

World Bank (2026). *World Development Indicators*. · Coppedge, M. et al. (2026). *V-Dem Country-Year Dataset v16*. Varieties of Democracy Project. · World Bank. *World Integrated Trade Solution (WITS)*.
