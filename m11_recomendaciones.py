"""
===============================================================================
 MÓDULO 11 — RECOMENDACIONES PARA MEJORAR EL SHARPE (screening de candidatos)
===============================================================================
Responde a la pregunta: "¿qué activo, de los disponibles en Yahoo Finance,
complementaría mejor mi cartera actual?"

No es una lista arbitraria: se apoya en un resultado estándar de teoría de
carteras (criterio de Treynor-Black / Elton-Gruber). Un activo candidato *c*
mejora el Sharpe de la cartera óptima resultante al añadirlo si y solo si:

    SR_c > rho(c, P) * SR_P

donde SR_x = (retorno anual esperado - Rf) / volatilidad anual, SR_P es el
Sharpe de tu cartera actual, y rho(c, P) es la correlación de los retornos
diarios del candidato con los de tu cartera. Intuición: un activo con Sharpe
mediocre puede seguir mereciendo la pena si su correlación con lo que ya
tienes es muy baja (diversifica); un activo con Sharpe alto pero muy
correlacionado con tu cartera aporta poco nuevo.

Esto es un cribado cuantitativo sobre datos HISTÓRICOS, no una recomendación
de inversión personalizada.

Uso rápido
----------
    from m11_recomendaciones import evaluar_candidatos, simular_incorporacion

    res = evaluar_candidatos(retornos_cartera, pesos_cartera, retornos_candidatos, rf=0.04)
    print(res.tabla)
===============================================================================
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("Falta 'yfinance'. Instálalo con: pip install yfinance") from exc

try:
    import plotly.graph_objects as go
    PLOTLY_DISPONIBLE = True
except ImportError:
    PLOTLY_DISPONIBLE = False

logger = logging.getLogger("quant.recomendaciones")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

DIAS_HABILES_ANIO: int = 252


class ErrorDeRecomendacion(Exception):
    """Se lanza cuando no hay datos suficientes para evaluar candidatos."""


# =============================================================================
# UNIVERSO DE CANDIDATOS
# =============================================================================
# ETFs y activos líquidos de Yahoo Finance que cubren clases de activo y
# geografías que NO están representadas en una cartera típica de renta
# variable europea/tecnológica: bonos, oro, materias primas, inmobiliario,
# emergentes, otros sectores y regiones.

CANDIDATOS_DEFECTO: dict[str, dict[str, str]] = {
    "TLT":     {"nombre": "iShares 20+ Year Treasury Bond",  "categoria": "Bonos largo plazo EE.UU."},
    "IEF":     {"nombre": "iShares 7-10 Year Treasury Bond", "categoria": "Bonos medio plazo EE.UU."},
    "TIP":     {"nombre": "iShares TIPS Bond",                "categoria": "Bonos ligados a inflación"},
    "SHY":     {"nombre": "iShares 1-3 Year Treasury Bond",   "categoria": "Bonos corto plazo (cuasi-cash)"},
    "GLD":     {"nombre": "SPDR Gold Shares",                 "categoria": "Oro"},
    "SLV":     {"nombre": "iShares Silver Trust",             "categoria": "Plata"},
    "DBC":     {"nombre": "Invesco DB Commodity Index",       "categoria": "Materias primas diversificadas"},
    "USO":     {"nombre": "United States Oil Fund",           "categoria": "Petróleo"},
    "VNQ":     {"nombre": "Vanguard Real Estate ETF",         "categoria": "Inmobiliario (REITs) EE.UU."},
    "EEM":     {"nombre": "iShares MSCI Emerging Markets",    "categoria": "Renta variable emergente"},
    "IWM":     {"nombre": "iShares Russell 2000",             "categoria": "Small caps EE.UU."},
    "XLE":     {"nombre": "Energy Select Sector SPDR",        "categoria": "Sector energía EE.UU."},
    "XLV":     {"nombre": "Health Care Select Sector SPDR",   "categoria": "Sector salud EE.UU."},
    "XLF":     {"nombre": "Financial Select Sector SPDR",     "categoria": "Sector financiero EE.UU."},
    "XLU":     {"nombre": "Utilities Select Sector SPDR",     "categoria": "Sector utilities (defensivo)"},
    "EWJ":     {"nombre": "iShares MSCI Japan",                "categoria": "Renta variable Japón"},
    "FXI":     {"nombre": "iShares China Large-Cap",           "categoria": "Renta variable China"},
    "INDA":    {"nombre": "iShares MSCI India",                "categoria": "Renta variable India"},
    "EWZ":     {"nombre": "iShares MSCI Brazil",               "categoria": "Renta variable Brasil"},
    "BTC-USD": {"nombre": "Bitcoin",                           "categoria": "Cripto (muy alta volatilidad)"},
}


def listar_candidatos(catalogo: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    """Tabla legible del universo de candidatos disponible."""
    catalogo = catalogo or CANDIDATOS_DEFECTO
    return pd.DataFrame([
        {"Ticker": t, "Nombre": i["nombre"], "Categoría": i["categoria"]}
        for t, i in catalogo.items()
    ]).set_index("Ticker")


# =============================================================================
# DESCARGA INDEPENDIENTE POR CANDIDATO
# =============================================================================
# A diferencia del Módulo 1 (que exige un calendario común y un mínimo de 2
# activos, porque construye una cartera conjunta), aquí cada candidato se
# evalúa de forma INDEPENDIENTE contra tu cartera. Es imprescindible: mezclar
# cripto (cotiza 7 días/semana) con ETFs (5 días) en una descarga conjunta
# hace que el control de calidad del Módulo 1 descarte casi todo por
# "huecos" que en realidad son solo fines de semana de un mercado distinto.

def _descargar_precio_individual(ticker: str, anios: float) -> pd.Series | None:
    """Descarga el histórico de cierre ajustado de UN ticker. None si falla."""
    fin = pd.Timestamp.today().normalize()
    inicio = fin - pd.Timedelta(days=int(round(anios * 365.25)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bruto = yf.download(
                ticker, start=inicio.strftime("%Y-%m-%d"), end=fin.strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True, progress=False,
            )
    except Exception as exc:
        logger.warning("Descarga de %s falló: %s", ticker, exc)
        return None

    if bruto is None or bruto.empty:
        return None

    columnas = bruto.columns
    if isinstance(columnas, pd.MultiIndex):
        if "Close" not in columnas.get_level_values(0):
            return None
        serie = bruto["Close"].iloc[:, 0]
    else:
        if "Close" not in columnas:
            return None
        serie = bruto["Close"]

    serie = pd.to_numeric(serie, errors="coerce").dropna()
    serie = serie[serie > 0]
    if len(serie) < 60:
        return None
    return serie.rename(ticker).sort_index()


def descargar_precios_candidatos(
    tickers: Sequence[str], anios: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Descarga cada candidato por separado y los junta en un único DataFrame
    (con NaN donde un ticker no cotiza ese día — es intencional, ver arriba).

    Returns
    -------
    (precios, fallidos) donde fallidos es {ticker: motivo}.
    """
    series: dict[str, pd.Series] = {}
    fallidos: dict[str, str] = {}
    for ticker in tickers:
        serie = _descargar_precio_individual(ticker, anios)
        if serie is None:
            fallidos[ticker] = "Sin datos suficientes en Yahoo Finance para este ticker."
        else:
            series[ticker] = serie

    if not series:
        raise ErrorDeRecomendacion(f"No se pudo descargar ningún candidato. Errores: {fallidos}")

    precios = pd.DataFrame(series)
    logger.info("Candidatos descargados: %d OK, %d fallidos", len(series), len(fallidos))
    return precios, fallidos


def calcular_retornos_candidatos(precios: pd.DataFrame) -> pd.DataFrame:
    """Retornos diarios simples, SIN eliminar filas donde falte algún otro
    candidato (cada columna conserva sus propios NaN — se filtran más tarde,
    por candidato, en evaluar_candidatos)."""
    return precios.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoRecomendaciones:
    tabla: pd.DataFrame                # una fila por candidato, ordenada por Score
    fallidos: dict[str, str]           # candidato -> motivo de exclusión
    sharpe_cartera: float
    alertas: list[str] = field(default_factory=list)


# =============================================================================
# EVALUACIÓN
# =============================================================================

def _retorno_cartera(retornos: pd.DataFrame, pesos: pd.Series) -> pd.Series:
    pesos_alineados = pesos.reindex(retornos.columns).fillna(0.0)
    suma = pesos_alineados.sum()
    if suma <= 0:
        raise ErrorDeRecomendacion("Los pesos de la cartera suman 0 o menos.")
    pesos_alineados = pesos_alineados / suma
    return (retornos * pesos_alineados).sum(axis=1)


def evaluar_candidatos(
    retornos_cartera: pd.DataFrame,
    pesos_cartera: pd.Series,
    retornos_candidatos: pd.DataFrame,
    rf: float = 0.04,
    catalogo: dict[str, dict[str, str]] | None = None,
    minimo_sesiones_comunes: int = 60,
) -> ResultadoRecomendaciones:
    """Rankea candidatos por su potencial de mejorar el Sharpe de la cartera.

    Ver docstring del módulo para el criterio (Treynor-Black / Elton-Gruber).
    """
    catalogo = catalogo or CANDIDATOS_DEFECTO
    r_p = _retorno_cartera(retornos_cartera, pesos_cartera)

    media_p = r_p.mean() * DIAS_HABILES_ANIO
    vol_p = r_p.std(ddof=1) * np.sqrt(DIAS_HABILES_ANIO)
    if vol_p <= 1e-12:
        raise ErrorDeRecomendacion("La volatilidad de tu cartera es ~0; no se puede evaluar el Sharpe.")
    sharpe_p = (media_p - rf) / vol_p

    filas = []
    fallidos: dict[str, str] = {}

    for ticker in retornos_candidatos.columns:
        serie = retornos_candidatos[ticker].dropna()
        comunes = pd.concat([serie.rename("candidato"), r_p.rename("cartera")], axis=1, join="inner").dropna()

        if len(comunes) < minimo_sesiones_comunes:
            fallidos[ticker] = f"Solo {len(comunes)} sesiones comunes con la cartera (mínimo {minimo_sesiones_comunes})."
            continue

        media_c = comunes["candidato"].mean() * DIAS_HABILES_ANIO
        vol_c = comunes["candidato"].std(ddof=1) * np.sqrt(DIAS_HABILES_ANIO)
        if vol_c <= 1e-12:
            fallidos[ticker] = "Volatilidad ~0 (activo sin variación)."
            continue
        sharpe_c = (media_c - rf) / vol_c

        rho = comunes["candidato"].corr(comunes["cartera"])
        hurdle = rho * sharpe_p
        score = sharpe_c - hurdle

        info = catalogo.get(ticker, {})
        filas.append({
            "Ticker": ticker,
            "Nombre": info.get("nombre", ticker),
            "Categoría": info.get("categoria", "—"),
            "Retorno anual %": media_c * 100,
            "Volatilidad anual %": vol_c * 100,
            "Sharpe individual": sharpe_c,
            "Correlación con cartera": rho,
            "Sharpe mínimo exigido": hurdle,
            "Score (mejora esperada)": score,
        })

    if not filas:
        raise ErrorDeRecomendacion(f"Ningún candidato pudo evaluarse. Errores: {fallidos}")

    tabla = (
        pd.DataFrame(filas)
        .set_index("Ticker")
        .sort_values("Score (mejora esperada)", ascending=False)
        .round(3)
    )

    alertas = generar_alertas_recomendaciones(tabla, sharpe_p)

    logger.info(
        "Recomendaciones evaluadas · %d candidatos OK · %d fallidos · Sharpe cartera=%.3f",
        len(tabla), len(fallidos), sharpe_p,
    )

    return ResultadoRecomendaciones(
        tabla=tabla, fallidos=fallidos, sharpe_cartera=sharpe_p, alertas=alertas,
    )


def generar_alertas_recomendaciones(tabla: pd.DataFrame, sharpe_p: float) -> list[str]:
    alertas: list[str] = []
    mejores = tabla[tabla["Score (mejora esperada)"] > 0]

    if mejores.empty:
        alertas.append(
            f"✓ Ningún candidato del universo analizado bate el listón de tu cartera actual "
            f"(Sharpe {sharpe_p:.2f}). Tu cartera ya está bien construida frente a este universo."
        )
        return alertas

    top = mejores.index[0]
    fila = mejores.loc[top]
    alertas.append(
        f"⭐ {top} ({fila['Nombre']}) es el candidato con mayor score: Sharpe individual "
        f"{fila['Sharpe individual']:.2f} frente a un mínimo exigido de "
        f"{fila['Sharpe mínimo exigido']:.2f} (correlación {fila['Correlación con cartera']:+.2f} "
        "con tu cartera)."
    )

    baja_corr = mejores[mejores["Correlación con cartera"] < 0.2]
    if not baja_corr.empty:
        alertas.append(
            "ℹ Candidatos con score positivo Y correlación muy baja (<0.20) — la combinación "
            f"ideal para diversificar de verdad: {', '.join(baja_corr.index)}."
        )

    negativos = tabla[tabla["Correlación con cartera"] < -0.15]
    positivos_score = negativos[negativos["Score (mejora esperada)"] > 0]
    if not positivos_score.empty:
        alertas.append(
            "ℹ Candidatos con correlación NEGATIVA y score positivo — actúan como cobertura, no "
            f"solo como diversificación: {', '.join(positivos_score.index)}."
        )

    return alertas


def peso_optimo_dos_activos(
    retornos_cartera: pd.DataFrame,
    pesos_cartera: pd.Series,
    retornos_candidato: pd.Series,
    rf: float = 0.04,
) -> float:
    """Peso (0-1) que MAXIMIZA el Sharpe de la combinación {tu cartera, candidato},
    tratando tu cartera actual como un único activo y resolviendo el problema de
    tangencia de Markowitz para 2 "activos" (cartera y candidato).

    Importante: el criterio de Treynor-Black (`evaluar_candidatos`) solo garantiza
    que EXISTE un peso que mejora el Sharpe — no que cualquier peso lo haga. Un
    candidato con score positivo puede seguir empeorando tu Sharpe si le asignas
    un peso arbitrario (por ejemplo, muy por encima o por debajo del óptimo). Esta
    función calcula justo ese peso óptimo, para usarlo como referencia.

    Se recorta a [0, 1] porque aquí no se permiten posiciones cortas ni apalancamiento;
    el óptimo matemático sin restricciones podría caer fuera de ese rango.
    """
    r_p = _retorno_cartera(retornos_cartera, pesos_cartera)
    comunes = pd.concat(
        [r_p.rename("cartera"), retornos_candidato.rename("candidato")], axis=1, join="inner",
    ).dropna()
    if len(comunes) < 30:
        raise ErrorDeRecomendacion("Muy pocas sesiones comunes para calcular el peso óptimo.")

    mu = comunes.mean() * DIAS_HABILES_ANIO
    cov = comunes.cov() * DIAS_HABILES_ANIO
    exceso = (mu - rf).values

    try:
        inversa = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError:
        return 0.0

    x = inversa @ exceso
    suma = x.sum()
    if abs(suma) < 1e-9:
        return 0.0

    w_candidato = x[1] / suma
    return float(np.clip(w_candidato, 0.0, 1.0))


# =============================================================================
# SIMULACIÓN "QUÉ PASARÍA SI..."
# =============================================================================

def simular_incorporacion(
    retornos_cartera: pd.DataFrame,
    pesos_cartera: pd.Series,
    retornos_candidato: pd.Series,
    peso_nuevo: float,
    rf: float = 0.04,
) -> dict[str, float]:
    """Recalcula retorno/vol/Sharpe de la cartera si se destina `peso_nuevo`
    (0-1) al candidato, reescalando proporcionalmente el resto de posiciones.

    No es una reoptimización de Markowitz: es la respuesta directa a "¿y si
    meto un X% en este activo, quitándoselo proporcionalmente a lo demás?".
    """
    if not 0 <= peso_nuevo <= 1:
        raise ErrorDeRecomendacion("peso_nuevo debe estar entre 0 y 1.")

    r_p_actual = _retorno_cartera(retornos_cartera, pesos_cartera)
    comunes = pd.concat(
        [r_p_actual.rename("cartera"), retornos_candidato.rename("candidato")],
        axis=1, join="inner",
    ).dropna()

    if len(comunes) < 30:
        raise ErrorDeRecomendacion("Muy pocas sesiones comunes para simular la incorporación.")

    r_nueva = (1 - peso_nuevo) * comunes["cartera"] + peso_nuevo * comunes["candidato"]

    def _metricas(serie: pd.Series) -> tuple[float, float, float]:
        media = serie.mean() * DIAS_HABILES_ANIO
        vol = serie.std(ddof=1) * np.sqrt(DIAS_HABILES_ANIO)
        sharpe = (media - rf) / vol if vol > 1e-12 else np.nan
        return media, vol, sharpe

    media_a, vol_a, sharpe_a = _metricas(comunes["cartera"])
    media_d, vol_d, sharpe_d = _metricas(r_nueva)

    return {
        "retorno_antes": media_a, "vol_antes": vol_a, "sharpe_antes": sharpe_a,
        "retorno_despues": media_d, "vol_despues": vol_d, "sharpe_despues": sharpe_d,
    }


# =============================================================================
# GRÁFICOS
# =============================================================================

def plot_ranking_candidatos(tabla: pd.DataFrame, top_n: int = 15):
    """Barras horizontales del score de cada candidato, verde si mejora el
    Sharpe de la cartera, rojo si no. Requiere plotly.
    """
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeRecomendacion("plotly no está instalado. Instálalo con: pip install plotly")

    datos = tabla.head(top_n).sort_values("Score (mejora esperada)")
    colores = ["#22C55E" if v > 0 else "#EF4444" for v in datos["Score (mejora esperada)"]]

    fig = go.Figure(go.Bar(
        x=datos["Score (mejora esperada)"], y=datos.index, orientation="h",
        marker_color=colores,
        customdata=datos["Nombre"],
        hovertemplate="%{y} — %{customdata}<br>Score: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#8892A6", line_dash="dash")
    fig.update_layout(
        xaxis_title="Score (> 0 = mejora esperada)", yaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
        height=max(320, 28 * len(datos)),
    )
    return fig


def plot_correlacion_vs_sharpe(tabla: pd.DataFrame, sharpe_p: float):
    """Dispersión Sharpe individual vs correlación con la cartera, con la
    línea de "listón mínimo" (hurdle = rho * SR_p): todo lo que queda por
    ENCIMA de esa línea mejora el Sharpe de la cartera. Requiere plotly.
    """
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeRecomendacion("plotly no está instalado. Instálalo con: pip install plotly")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tabla["Correlación con cartera"], y=tabla["Sharpe individual"],
        mode="markers+text", text=tabla.index, textposition="top center",
        marker=dict(
            size=11,
            color=["#22C55E" if v > 0 else "#EF4444" for v in tabla["Score (mejora esperada)"]],
            line=dict(width=1, color="white"),
        ),
        customdata=tabla["Nombre"],
        hovertemplate="%{text} — %{customdata}<br>Correlación: %{x:.2f}<br>Sharpe: %{y:.2f}<extra></extra>",
        name="Candidatos",
    ))

    x_linea = np.linspace(min(tabla["Correlación con cartera"].min(), -0.2),
                           max(tabla["Correlación con cartera"].max(), 1.0), 50)
    fig.add_trace(go.Scatter(
        x=x_linea, y=x_linea * sharpe_p, mode="lines",
        line=dict(color="#F59E0B", width=1.6, dash="dash"), name="Listón mínimo (ρ·SR_cartera)",
    ))

    fig.update_layout(
        xaxis_title="Correlación con la cartera (ρ)", yaxis_title="Sharpe individual del candidato",
        legend=dict(
            x=0.015, y=0.985, xanchor="left", yanchor="top",
            bgcolor="rgba(20,27,46,0.88)", bordercolor="#2A3450", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig
