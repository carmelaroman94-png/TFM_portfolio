"""Resolución de los indicadores sin acuerdo entre las dos corridas del LLM (codebook v2).

Cada decisión cita la regla del codebook aplicada. Revisión realizada por Claude (asistente IA)
a pedido de la autora, aplicando el codebook v2; queda etiquetada como "clasificacion manual"
en la columna `metodo` y con el revisor identificado en `revisor`.
"""
from __future__ import annotations

import re

import pandas as pd

REVISOR = "Claude (asistente IA) - codebook v2"

RULES: list[tuple[str, str, str]] = [
    # (regex sobre código, dimensión, justificación)
    (r"^(BM\.GSR|BX\.GSR|BM\.TRF|BX\.TRF|BN\.TRF|NY\.TRF|BX\.GRT|ST\.INT|TM\.VAL)\.",
     "ESTRUCTURAL", "Cuenta corriente/transferencias externas: posición en la estructura internacional (3.4b)."),
    (r"^BN\.KLT\.DINV", "ESTRUCTURAL", "IED: inserción externa del aparato productivo (3.4b); coherente con variable protegida BX.KLT.DINV.WD.GD.ZS."),
    (r"^BN\.KLT\.PTXL", "MONETARIA", "Flujos de cartera: condiciones financieras (3.1)."),
    (r"^(DC\.DAC|DT\.ODA)\.", "ESTRUCTURAL", "Ayuda oficial recibida: dependencia de financiamiento externo (3.4b, análogo a 4.3 caso externo)."),
    (r"^(DT\.DOD\.DSTC\.ZS|DT\.NFL|DT\.TDS)", "MONETARIA", "Deuda/financiamiento externo no expresado sobre exportaciones o reservas: monetaria por defecto (4.3)."),
    (r"^(EN\.|ER\.|AG\.LND|NY\.ADJ\.DPEM)", "AMBIGUO", "Indicador ambiental/geográfico sin mecanismo inflacionario en el marco teórico."),
    (r"^FB\.ATM", "MONETARIA", "Profundidad del sistema financiero: condiciones financieras (3.1)."),
    (r"^GC\.XPN\.COMP", "MONETARIA", "Gasto público corriente: variables fiscales en sentido amplio (3.1)."),
    (r"^GF\.XPD\.BUDG", "INSTITUCIONAL", "Ejecución vs. presupuesto aprobado: calidad de gestión presupuestaria (4.3)."),
    (r"^IC\.BRE\.(BE\.P3|MC\.P2)", "ESTRUCTURAL", "Barreras de entrada y competencia de mercado (3.4a)."),
    (r"^IC\.BRE\.P2\.PS", "INSTITUCIONAL", "Calidad de servicios públicos que enmarcan la actividad: diseño institucional (3.3)."),
    (r"^(IC\.CUS|IC\.FRM\.FREG|IC\.MNG|IE\.PP[IN])", "ESTRUCTURAL", "Estructura productiva/empresarial e integración comercial (3.4a)."),
    (r"^IC\.FRM\.METG", "INSTITUCIONAL", "Relación Estado-empresa en fiscalización: diseño y práctica institucional (3.3)."),
    (r"^(IQ\.CPA|IQ\.SPI|SP\.REG)", "INSTITUCIONAL", "Evaluación de políticas/capacidad estatal (estadística, registro civil): institucional (3.3)."),
    (r"^(IT\.NET\.SECR|MS\.MIL|SG\.VAW|SH\.ALC|SH\.CON|SH\.STA\.FGMS|SH\.STA\.SUIC)", "AMBIGUO",
     "Sin vínculo plausible con expectativas/canasta (3.2) ni con otra dimensión; gasto sectorial excluido de 3.1."),
    (r"^NE\.CON\.PRVT", "SOCIODEMOGRAFICA", "Consumo de hogares: bienestar material y canasta de los agentes (3.2)."),
    (r"^(NE\.CON\.TOTL|NE\.DAB|NY\.GDP\.MKTP\.KD\.ZG)", "MONETARIA", "Demanda agregada/producto real: términos de la teoría cuantitativa (3.1)."),
    (r"^(NE\.GDI|NY\.ADJ\.DKAP)", "ESTRUCTURAL", "Formación y desgaste de capital: estructura productiva (3.4a)."),
    (r"^(NY\.ADJ\.(ICTR|SVNG|SVNX)|NY\.GNS)", "MONETARIA", "Ahorro nacional: condiciones financieras agregadas (3.1)."),
    (r"^(NY\.ADJ\.NNTY|NY\.GNP\.MKTP)", "SOCIODEMOGRAFICA", "Ingreso nacional: bienestar material agregado (3.2); coherente con la etiqueta humana del INB."),
    (r"^PA\.NUS\.PPP", "MONETARIA", "Factor PPA: nivel de precios relativo (3.1)."),
    (r"^per_(allsp|lm|si)", "SOCIODEMOGRAFICA", "Cobertura de protección social sobre la población: condición del agente; coherente con la etiqueta humana de incidencia (3.2)."),
    (r"^SE\.XPD", "SOCIODEMOGRAFICA", "Educación, aun describiendo el sistema educativo (ancla SE.SEC.TCAQ.UP.ZS, 3.2)."),
    (r"^(SH\.DTH|SH\.DYN|SH\.MLR|SH\.STA|SH\.TBS|SP\.DYN\.CDRT)", "SOCIODEMOGRAFICA", "Condiciones de salud de la población (3.2)."),
    (r"^SH\.XPD\.(CHEX|OOPC)", "SOCIODEMOGRAFICA", "Gasto en salud de los hogares/per cápita: canasta de consumo (3.2)."),
    (r"^SH\.XPD\.EHEX", "ESTRUCTURAL", "Financiamiento externo del sistema de salud: dependencia externa (3.4b)."),
    (r"^(SL\.EMP\.MPYR|SL\.EMP\.VULN|SL\.FAM)", "ESTRUCTURAL", "Estructura ocupacional/informalidad; coherente con la etiqueta humana de cuentapropismo (3.4a)."),
    (r"^(SL\.EMP\.SMGT|SL\.TLF\.PART)", "SOCIODEMOGRAFICA", "Característica laboral del agente; coherente con la etiqueta humana (3.2)."),
    (r"^SM\.POP", "SOCIODEMOGRAFICA", "Composición demográfica (3.2); coherente con la etiqueta humana."),
    (r"^SP\.POP\.TECH", "ESTRUCTURAL", "Capacidad tecnológica del aparato productivo (3.4a); coherente con I+D (etiqueta humana)."),
    (r"^(TM\.TAX|TM\.UVI)", "ESTRUCTURAL", "Protección comercial/términos de intercambio (3.4b); coherente con la etiqueta humana de aranceles."),
    (r"^v2xpe_exlecon$", "ESTRUCTURAL", "Exclusión por grupo socioeconómico: mismo criterio que v2xeg_eqaccess (3.4)."),
    (r"^v2xpe_exlgender$", "INSTITUCIONAL", "Caso mixto (3.4 nota): la descripción v16 define exclusión política; prima poder político-formal (4.4)."),
]


def resolve(codigo: str) -> tuple[str, str]:
    for pattern, dim, why in RULES:
        if re.search(pattern, codigo):
            return dim, why
    raise KeyError(f"Sin regla de revisión para {codigo}")


def apply(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza 'REVISAR' por la decisión revisada y documenta método y revisor."""
    out = df.copy()
    out["metodo"] = "LLM (2 corridas concordantes)"
    out["revisor"] = ""
    out["justificacion_revision"] = ""
    mask = out["dimension"] == "REVISAR"
    decisions = [resolve(c) for c in out.index[mask]]
    out.loc[mask, "dimension"] = [d for d, _ in decisions]
    out.loc[mask, "justificacion_revision"] = [w for _, w in decisions]
    out.loc[mask, "metodo"] = "clasificacion manual"
    out.loc[mask, "revisor"] = REVISOR
    return out
