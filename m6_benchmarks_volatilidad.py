"""
===============================================================================
 MÓDULO 6 — BENCHMARKS DE MERCADO + VISUALIZACIÓN DE VOLATILIDAD HISTÓRICA
===============================================================================
Dos funciones complementarias:

  A) BENCHMARKS: descarga índices de referencia (S&P 500 vía SPY, KOSPI,
     EuroStoxx 50, sector defensa europea) para poder contestar la pregunta
     que de verdad importa: ¿la volatilidad de MIS activos es alta en términos
     absolutos, o es que TODO el mercado está nervioso?

  B) GRÁFICOS: convierte las tablas del Módulo 5 en el panel visual de
     volatilidad histórica (rolling, mapa de calor temporal, régimen,
     y comparación cartera vs benchmarks).

-------------------------------------------------------------------------------
 NOTA IMPORTANTE SOBRE EL ÍNDICE DE DEFENSA EUROPEA
-------------------------------------------------------------------------------
El ETF de referencia (WisdomTree Europe Defence, WDEF.MI) se lanzó en marzo de
2025: NO tiene 3 años de histórico. Si lo pides con `anios=3`, el Módulo 1 lo
descartará por falta de datos, y con razón.

Solución implementada: `construir_cesta_defensa_europea()` replica el índice
sintéticamente con sus 10 mayores componentes (Thales, BAE, Rheinmetall,
Airbus, Rolls-Royce, Safran, Leonardo, Saab, Dassault, Kongsberg), que sí
cotizan desde hace décadas. Así obtienes el histórico completo.

Ojo: es una réplica aproximada (pesos fijos, sin rebalanceo ni conversión de
divisa GBP/SEK/NOK→EUR), útil para medir VOLATILIDAD y CORRELACIÓN del sector,
no para replicar la rentabilidad exacta del ETF.

-------------------------------------------------------------------------------
 NOTA SOBRE DIVISAS Y CALENDARIOS
-------------------------------------------------------------------------------
Cada índice cotiza en su divisa y su calendario (SPY en USD, KOSPI en KRW,
EuroStoxx en EUR). Para VOLATILIDAD esto es correcto y estándar: mides el
nerviosismo de cada mercado en su propia moneda. Para comparar RENTABILIDADES
haría falta convertir divisa, cosa que este módulo NO hace a propósito.

Uso rápido
----------
    from m6_benchmarks_volatilidad import descargar_benchmarks, panel_volatilidad

    bench = descargar_benchmarks(["SPY", "KOSPI", "EUROSTOXX", "DEFENSA_EU"], anios=3)
    panel_volatilidad(vol, retornos, benchmarks=bench, guardar="panel_volatilidad.png")
===============================================================================
"""

from __future__ import annotations

import logging
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

try:
    import plotly.graph_objects as go
    PLOTLY_DISPONIBLE = True
except ImportError:
    PLOTLY_DISPONIBLE = False

logger = logging.getLogger("quant.benchmarks")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

DIAS_HABILES_ANIO: int = 252
sns.set_theme(style="whitegrid", font_scale=0.95)


class ErrorDeBenchmark(Exception):
    """Se lanza cuando un benchmark no se puede descargar o construir."""


# =============================================================================
# CATÁLOGO DE BENCHMARKS
# =============================================================================

BENCHMARKS: dict[str, dict[str, str]] = {
    "SPY": {
        "ticker": "SPY",
        "nombre": "S&P 500 (SPDR ETF)",
        "divisa": "USD",
        "descripcion": "Renta variable EE.UU. El mercado más líquido del mundo y el que marca el tono global.",
    },
    "KOSPI": {
        "ticker": "^KS11",
        "nombre": "KOSPI Composite",
        "divisa": "KRW",
        "descripcion": "Bolsa de Corea del Sur. Muy sensible al ciclo de semiconductores y a la tensión geopolítica en Asia.",
    },
    "EUROSTOXX": {
        "ticker": "^STOXX50E",
        "nombre": "EURO STOXX 50",
        "divisa": "EUR",
        "descripcion": "Las 50 mayores cotizadas de la eurozona. El benchmark natural para RHM, IDR y MC.",
    },
    "DEFENSA_EU_ETF": {
        "ticker": "WDEF.MI",
        "nombre": "WisdomTree Europe Defence UCITS ETF",
        "divisa": "EUR",
        "descripcion": "ETF puro de defensa europea. ⚠ Cotiza desde marzo de 2025: histórico MUY corto.",
    },
    "DEFENSA_GLOBAL_ETF": {
        "ticker": "DFNS.MI",
        "nombre": "VanEck Defense ETF",
        "divisa": "USD",
        "descripcion": "Defensa global (incluye EE.UU.). Histórico algo más largo que el europeo puro.",
    },
    "VIX": {
        "ticker": "^VIX",
        "nombre": "CBOE Volatility Index (VIX)",
        "divisa": "índice",
        "descripcion": "El 'índice del miedo': volatilidad implícita a 30 días del S&P 500. Es un anticipo del Módulo 7.",
    },
    "MSCI_WORLD": {
        "ticker": "EUNL.DE",
        "nombre": "iShares Core MSCI World UCITS ETF",
        "divisa": "EUR",
        "descripcion": "Renta variable global desarrollada. Ya lo tienes en cartera.",
    },
}

# Componentes de la cesta sintética de defensa europea, con sus pesos
# aproximados según la composición del ETF WisdomTree (WDEF).
CESTA_DEFENSA_EUROPEA: dict[str, float] = {
    "HO.PA":      0.1279,  # Thales (Francia)
    "BA.L":       0.1239,  # BAE Systems (Reino Unido)
    "RHM.DE":     0.1218,  # Rheinmetall (Alemania) — ¡ya lo tienes en cartera!
    "AIR.PA":     0.0850,  # Airbus (Francia/paneuropeo)
    "RR.L":       0.0771,  # Rolls-Royce (Reino Unido)
    "SAF.PA":     0.0740,  # Safran (Francia)
    "LDO.MI":     0.0653,  # Leonardo (Italia)
    "SAAB-B.ST":  0.0634,  # Saab (Suecia)
    "AM.PA":      0.0543,  # Dassault Aviation (Francia)
    "KOG.OL":     0.0368,  # Kongsberg Gruppen (Noruega)
}


def listar_benchmarks() -> pd.DataFrame:
    """Tabla legible con todos los benchmarks disponibles y sus tickers."""
    filas = [
        {"Clave": clave, "Ticker": info["ticker"], "Nombre": info["nombre"],
         "Divisa": info["divisa"], "Descripción": info["descripcion"]}
        for clave, info in BENCHMARKS.items()
    ]
    return pd.DataFrame(filas).set_index("Clave")


# =============================================================================
# A) DESCARGA DE BENCHMARKS
# =============================================================================

def descargar_benchmarks(
    claves: Sequence[str] = ("SPY", "KOSPI", "EUROSTOXX"),
    anios: float = 3.0,
    umbral_cobertura: float = 0.70,
    cache_dir: str | None = "./cache_datos",
) -> pd.DataFrame:
    """Descarga los índices de referencia solicitados.

    A diferencia del Módulo 1, aquí el umbral de cobertura es más laxo (70%)
    porque los mercados de EE.UU., Europa y Corea tienen calendarios de
    festivos distintos: nunca coincidirán al 100% de las sesiones. Los huecos
    se rellenan arrastrando el último precio conocido (ffill).

    Parameters
    ----------
    claves : claves del catálogo BENCHMARKS (no tickers). Usa listar_benchmarks()
        para ver las disponibles.
    """
    from m1_datos import descargar_precios  # import local para no crear dependencia circular

    desconocidas = [c for c in claves if c not in BENCHMARKS]
    if desconocidas:
        raise ErrorDeBenchmark(
            f"Claves de benchmark desconocidas: {desconocidas}. "
            f"Disponibles: {list(BENCHMARKS.keys())}"
        )

    tickers = [BENCHMARKS[c]["ticker"] for c in claves]
    mapa_inverso = {BENCHMARKS[c]["ticker"]: c for c in claves}

    precios = descargar_precios(
        tickers, anios=anios, umbral_cobertura=umbral_cobertura, cache_dir=cache_dir,
    )
    # Renombramos las columnas del ticker técnico a la clave legible
    precios = precios.rename(columns=mapa_inverso)

    perdidos = [c for c in claves if c not in precios.columns]
    if perdidos:
        logger.warning(
            "Estos benchmarks no superaron el control de calidad y se descartaron: %s. "
            "Causa habitual: histórico más corto que el periodo pedido.", perdidos,
        )

    logger.info("Benchmarks descargados: %s", list(precios.columns))
    return precios


def construir_cesta_defensa_europea(
    anios: float = 3.0,
    componentes: dict[str, float] | None = None,
    cache_dir: str | None = "./cache_datos",
    min_componentes: int = 5,
) -> pd.Series:
    """Construye sintéticamente un índice de defensa europea con histórico completo.

    Por qué: el ETF real (WDEF.MI) solo cotiza desde marzo de 2025, así que no
    sirve para comparar 3 años de volatilidad. Esta función reconstruye el
    sector combinando sus componentes reales, que sí tienen histórico largo.

    Método: índice de retornos ponderados con pesos fijos, normalizado a base
    100 en la primera sesión. Los pesos se renormalizan automáticamente si
    algún componente no se puede descargar.

    Returns
    -------
    pd.Series con el "precio" del índice sintético (base 100), lista para
    concatenar con el resto de precios.
    """
    from m1_datos import calcular_retornos, descargar_precios

    componentes = componentes or CESTA_DEFENSA_EUROPEA

    logger.info("Construyendo cesta sintética de defensa europea (%d componentes)...", len(componentes))
    precios = descargar_precios(
        list(componentes.keys()), anios=anios,
        umbral_cobertura=0.80, cache_dir=cache_dir,
    )

    disponibles = [t for t in componentes if t in precios.columns]
    if len(disponibles) < min_componentes:
        raise ErrorDeBenchmark(
            f"Solo se pudieron descargar {len(disponibles)} de {len(componentes)} componentes "
            f"de la cesta de defensa ({disponibles}). Mínimo requerido: {min_componentes}."
        )
    if len(disponibles) < len(componentes):
        faltantes = [t for t in componentes if t not in disponibles]
        logger.warning(
            "Componentes no descargados (se renormalizan los pesos del resto): %s", faltantes,
        )

    pesos = pd.Series({t: componentes[t] for t in disponibles})
    pesos = pesos / pesos.sum()

    retornos = calcular_retornos(precios[disponibles])
    retornos_cesta = (retornos * pesos).sum(axis=1)

    # Reconstruimos un "precio" acumulando los retornos desde base 100
    indice = 100 * (1 + retornos_cesta).cumprod()
    indice.name = "DEFENSA_EU"

    logger.info(
        "Cesta de defensa construida · %d componentes · %d sesiones · base 100",
        len(disponibles), len(indice),
    )
    return indice


def alinear_con_cartera(
    retornos_cartera: pd.DataFrame,
    precios_benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    """Alinea los retornos de los benchmarks con el calendario de tu cartera.

    Necesario porque Corea, EE.UU. y Europa no tienen los mismos festivos: sin
    alinear, cualquier comparación día a día estaría desfasada.
    """
    from m1_datos import calcular_retornos

    retornos_bench = calcular_retornos(precios_benchmarks)
    # Reindexamos al calendario de la cartera, arrastrando el último dato conocido
    alineado = retornos_bench.reindex(retornos_cartera.index).ffill()
    return alineado.dropna(how="all")


# =============================================================================
# COMPARATIVA CARTERA vs BENCHMARKS
# =============================================================================

def comparar_volatilidad_benchmarks(
    retornos_cartera: pd.DataFrame,
    retornos_benchmarks: pd.DataFrame,
    pesos_cartera: pd.Series | None = None,
    ventana: int = 60,
) -> pd.DataFrame:
    """Tabla que sitúa la volatilidad de tus activos EN CONTEXTO de mercado.

    Sin esto, saber que un activo tiene un 35% de volatilidad no dice nada:
    ¿es mucho? Depende de si el mercado entero está al 12% o al 40%.

    La columna 'vs EuroStoxx' (o el benchmark que uses) es la lectura clave:
    un valor de 2.0 significa que ese activo se mueve el doble que su mercado
    de referencia.
    """
    from m5_volatilidad_historica import volatilidad_cartera_rolling, volatilidad_rolling

    vol_activos = volatilidad_rolling(retornos_cartera, ventana)
    vol_bench = volatilidad_rolling(retornos_benchmarks, ventana)

    filas = []
    for activo in vol_activos.columns:
        serie = vol_activos[activo].dropna()
        if serie.empty:
            continue
        filas.append({
            "Serie": activo,
            "Tipo": "Activo cartera",
            f"Vol actual {ventana}d %": serie.iloc[-1] * 100,
            "Vol media %": serie.mean() * 100,
            "Vol máx %": serie.max() * 100,
            "Percentil actual": (serie < serie.iloc[-1]).mean() * 100,
        })

    if pesos_cartera is not None:
        vol_cart = volatilidad_cartera_rolling(retornos_cartera, pesos_cartera, ventana)
        filas.append({
            "Serie": "★ TU CARTERA",
            "Tipo": "Cartera completa",
            f"Vol actual {ventana}d %": vol_cart.iloc[-1] * 100,
            "Vol media %": vol_cart.mean() * 100,
            "Vol máx %": vol_cart.max() * 100,
            "Percentil actual": (vol_cart < vol_cart.iloc[-1]).mean() * 100,
        })

    for bench in vol_bench.columns:
        serie = vol_bench[bench].dropna()
        if serie.empty:
            continue
        filas.append({
            "Serie": bench,
            "Tipo": "Benchmark",
            f"Vol actual {ventana}d %": serie.iloc[-1] * 100,
            "Vol media %": serie.mean() * 100,
            "Vol máx %": serie.max() * 100,
            "Percentil actual": (serie < serie.iloc[-1]).mean() * 100,
        })

    tabla = pd.DataFrame(filas).set_index("Serie")
    return tabla.sort_values(f"Vol actual {ventana}d %", ascending=False).round(2)


def beta_vs_benchmarks(
    retornos_cartera: pd.DataFrame,
    retornos_benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    """Beta de cada activo frente a CADA benchmark, no solo frente a uno.

    Muy revelador: RHM puede tener una Beta baja frente al EuroStoxx (porque el
    sector defensa se ha desacoplado del mercado general) pero muy alta frente
    al índice de defensa europea. Esas dos cifras juntas cuentan la historia
    completa de dónde viene su riesgo.
    """
    from m3_optimizacion import calcular_beta

    filas = []
    for activo in retornos_cartera.columns:
        fila: dict[str, object] = {"Activo": activo}
        for bench in retornos_benchmarks.columns:
            try:
                fila[f"β vs {bench}"] = round(
                    calcular_beta(retornos_cartera[activo], retornos_benchmarks[bench]), 3
                )
            except Exception as exc:  # un benchmark con pocos datos no debe tumbar la tabla
                logger.warning("No se pudo calcular β de %s vs %s: %s", activo, bench, exc)
                fila[f"β vs {bench}"] = np.nan
        filas.append(fila)

    return pd.DataFrame(filas).set_index("Activo")


# =============================================================================
# B) GRÁFICOS DE VOLATILIDAD HISTÓRICA
# =============================================================================

def plot_volatilidad_rolling(
    rolling_corta: pd.DataFrame,
    rolling_larga: pd.DataFrame,
    ventana_corta: int = 21,
    ventana_larga: int = 60,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Evolución de la volatilidad de cada activo: línea fina = corto plazo
    (21d, reactiva), línea gruesa = medio plazo (60d, estable).

    Cuando la fina se dispara por encima de la gruesa, el riesgo está
    acelerando: es la señal visual del 'ratio corto/largo' del Módulo 5.
    """
    if rolling_larga.empty:
        raise ErrorDeBenchmark("No hay datos de volatilidad rolling para dibujar.")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    paleta = sns.color_palette("tab10", rolling_larga.shape[1])
    for i, activo in enumerate(rolling_larga.columns):
        ax.plot(rolling_larga.index, rolling_larga[activo] * 100,
                color=paleta[i], linewidth=2.0, label=f"{activo} ({ventana_larga}d)")
        if activo in rolling_corta.columns:
            ax.plot(rolling_corta.index, rolling_corta[activo] * 100,
                    color=paleta[i], linewidth=0.8, alpha=0.45, linestyle="-")

    ax.set_ylabel("Volatilidad anualizada (%)")
    ax.set_title(
        f"Volatilidad histórica · línea gruesa = {ventana_larga}d, fina = {ventana_corta}d",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_volatilidad_rolling_interactivo(
    rolling_corta: pd.DataFrame,
    rolling_larga: pd.DataFrame,
    ventana_corta: int = 21,
    ventana_larga: int = 60,
):
    """Versión interactiva (Plotly) de plot_volatilidad_rolling: zoom, hover con
    el valor exacto por fecha, y activos que se pueden ocultar haciendo clic en
    la leyenda. Requiere plotly (pip install plotly).
    """
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeBenchmark("plotly no está instalado. Instálalo con: pip install plotly")
    if rolling_larga.empty:
        raise ErrorDeBenchmark("No hay datos de volatilidad rolling para dibujar.")

    from m0_estilo import GRIS_BORDE, PALETA_CATEGORICA

    paleta = PALETA_CATEGORICA

    fig = go.Figure()
    for i, activo in enumerate(rolling_larga.columns):
        color = paleta[i % len(paleta)]
        fig.add_trace(go.Scatter(
            x=rolling_larga.index, y=rolling_larga[activo] * 100,
            mode="lines", name=f"{activo} ({ventana_larga}d)",
            line=dict(color=color, width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>%{fullData.name}</extra>",
        ))
        if activo in rolling_corta.columns:
            fig.add_trace(go.Scatter(
                x=rolling_corta.index, y=rolling_corta[activo] * 100,
                mode="lines", name=f"{activo} ({ventana_corta}d)",
                line=dict(color=color, width=1, dash="dot"), opacity=0.5,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>%{fullData.name}</extra>",
            ))

    fig.update_layout(
        xaxis_title="Fecha", yaxis_title="Volatilidad anualizada (%)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def plot_mapa_calor_volatilidad_interactivo(
    rolling: pd.DataFrame,
    frecuencia: str = "ME",
):
    """Versión interactiva del mapa de calor: hover con el valor exacto de cada
    celda (activo × mes). Requiere plotly.
    """
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeBenchmark("plotly no está instalado. Instálalo con: pip install plotly")
    if rolling.empty:
        raise ErrorDeBenchmark("No hay datos para el mapa de calor.")

    mensual = rolling.resample(frecuencia).mean() * 100
    mensual.index = mensual.index.strftime("%Y-%m")

    fig = go.Figure(data=go.Heatmap(
        z=mensual.T.values, x=mensual.index, y=mensual.columns,
        colorscale="YlOrRd",
        colorbar=dict(
            title=dict(text="Vol. anualizada (%)", side="right", font=dict(size=11)),
            thickness=12, outlinewidth=0, tickfont=dict(size=10),
        ),
        hovertemplate="%{x}<br>%{y}<br>%{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def plot_cartera_vs_benchmarks_interactivo(
    retornos_cartera: pd.DataFrame,
    pesos_cartera: pd.Series,
    retornos_benchmarks: pd.DataFrame,
    ventana: int = 60,
):
    """Versión interactiva de tu cartera vs benchmarks. Requiere plotly."""
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeBenchmark("plotly no está instalado. Instálalo con: pip install plotly")

    from m0_estilo import PALETA_CATEGORICA
    from m5_volatilidad_historica import volatilidad_cartera_rolling, volatilidad_rolling

    vol_bench = volatilidad_rolling(retornos_benchmarks, ventana)
    paleta = PALETA_CATEGORICA

    fig = go.Figure()
    for i, bench in enumerate(vol_bench.columns):
        fig.add_trace(go.Scatter(
            x=vol_bench.index, y=vol_bench[bench] * 100,
            mode="lines", name=bench, line=dict(color=paleta[i % len(paleta)], width=1.6),
            opacity=0.85,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>%{fullData.name}</extra>",
        ))

    vol_cart = volatilidad_cartera_rolling(retornos_cartera, pesos_cartera, ventana)
    fig.add_trace(go.Scatter(
        x=vol_cart.index, y=vol_cart * 100,
        mode="lines", name="★ TU CARTERA", line=dict(color="#F2F4F8", width=3),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>TU CARTERA</extra>",
    ))

    fig.update_layout(
        xaxis_title="Fecha", yaxis_title="Volatilidad anualizada (%)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def plot_mapa_calor_volatilidad(
    rolling: pd.DataFrame,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
    frecuencia: str = "ME",
) -> Figure:
    """Mapa de calor tiempo × activo: cada celda es la volatilidad media de ese
    activo en ese mes. Permite ver de un golpe de vista qué periodos fueron
    turbulentos y qué activos los sufrieron más.
    """
    if rolling.empty:
        raise ErrorDeBenchmark("No hay datos para el mapa de calor.")

    mensual = rolling.resample(frecuencia).mean() * 100
    mensual.index = mensual.index.strftime("%Y-%m")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        mensual.T, ax=ax, cmap="YlOrRd", annot=False,
        cbar_kws={"label": "Vol. anualizada (%)"}, linewidths=0.3, linecolor="white",
    )
    ax.set_title("Mapa de calor de volatilidad (media mensual)", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_cartera_vs_benchmarks(
    retornos_cartera: pd.DataFrame,
    pesos_cartera: pd.Series,
    retornos_benchmarks: pd.DataFrame,
    ventana: int = 60,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Tu cartera (línea negra gruesa) contra los índices de referencia.

    Es el gráfico que responde: ¿mi cartera es más nerviosa que el mercado, o
    simplemente estoy viviendo el mismo régimen que todo el mundo?
    """
    from m5_volatilidad_historica import volatilidad_cartera_rolling, volatilidad_rolling

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    vol_bench = volatilidad_rolling(retornos_benchmarks, ventana)
    paleta = sns.color_palette("Set2", vol_bench.shape[1])
    for i, bench in enumerate(vol_bench.columns):
        ax.plot(vol_bench.index, vol_bench[bench] * 100,
                color=paleta[i], linewidth=1.5, alpha=0.85, label=bench)

    vol_cart = volatilidad_cartera_rolling(retornos_cartera, pesos_cartera, ventana)
    ax.plot(vol_cart.index, vol_cart * 100, color="black", linewidth=2.6,
            label="★ TU CARTERA", zorder=5)

    ax.set_ylabel("Volatilidad anualizada (%)")
    ax.set_title(f"Tu cartera vs benchmarks de mercado (rolling {ventana}d)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_regimen_volatilidad(
    resumen: pd.DataFrame,
    regimen: pd.DataFrame,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> Figure:
    """Barras horizontales del percentil de volatilidad actual de cada activo,
    coloreadas por régimen (verde = calma, ámbar = normal, rojo = estrés).
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colores_regimen = {"CALMA": "#00B050", "NORMAL": "#FFC000", "ESTRÉS": "#C00000"}
    datos = regimen.sort_values("Percentil")
    colores = [colores_regimen.get(r, "gray") for r in datos["Régimen"]]

    barras = ax.barh(datos.index, datos["Percentil"], color=colores, edgecolor="white")
    for barra, (activo, fila) in zip(barras, datos.iterrows()):
        ax.text(
            min(fila["Percentil"] + 2, 96), barra.get_y() + barra.get_height() / 2,
            f"{fila['Tendencia']}", va="center", fontsize=8,
        )

    ax.axvline(25, color="gray", linestyle="--", linewidth=0.9)
    ax.axvline(75, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Percentil de volatilidad dentro de su propio histórico")
    ax.set_title("Régimen de volatilidad actual por activo", fontsize=11, fontweight="bold")

    if fig is not None:
        fig.tight_layout()
    return fig


def panel_volatilidad(
    resultado_vol,                       # ResultadoVolatilidad del Módulo 5
    retornos_cartera: pd.DataFrame,
    pesos_cartera: pd.Series | None = None,
    retornos_benchmarks: pd.DataFrame | None = None,
    guardar: str | None = "panel_volatilidad.png",
    dpi: int = 150,
) -> Figure:
    """Panel compuesto completo de volatilidad histórica (Fase 2).

    Composición: rolling arriba, mapa de calor en medio, régimen y comparación
    con benchmarks abajo (esta última solo si se pasan `retornos_benchmarks`
    y `pesos_cartera`).
    """
    ventana_corta, ventana_larga = resultado_vol.ventanas[0], resultado_vol.ventanas[-1]
    hay_comparativa = retornos_benchmarks is not None and pesos_cartera is not None
    n_filas = 3 if hay_comparativa else 2

    fig = plt.figure(figsize=(14, 4.5 * n_filas))
    gs = fig.add_gridspec(n_filas, 2 if hay_comparativa else 1)

    ax_roll = fig.add_subplot(gs[0, :])
    plot_volatilidad_rolling(
        resultado_vol.rolling[ventana_corta], resultado_vol.rolling[ventana_larga],
        ventana_corta, ventana_larga, ax=ax_roll,
    )

    ax_mapa = fig.add_subplot(gs[1, :])
    plot_mapa_calor_volatilidad(resultado_vol.rolling[ventana_larga], ax=ax_mapa)

    if hay_comparativa:
        ax_regimen = fig.add_subplot(gs[2, 0])
        plot_regimen_volatilidad(resultado_vol.resumen, resultado_vol.regimen, ax=ax_regimen)

        ax_bench = fig.add_subplot(gs[2, 1])
        plot_cartera_vs_benchmarks(
            retornos_cartera, pesos_cartera, retornos_benchmarks,
            ventana=ventana_larga, ax=ax_bench,
        )

    fig.suptitle(
        "Fase 2 — Mapa de volatilidad histórica (realizada)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.tight_layout()

    if guardar is not None:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Panel de volatilidad guardado en '%s'", guardar)

    return fig


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m6_benchmarks_volatilidad.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 25)

    print("\n--- Benchmarks disponibles ---")
    print(listar_benchmarks().to_string())

    try:
        from m1_datos import calcular_retornos, descargar_precios
        from m5_volatilidad_historica import analizar_volatilidad

        # 👇 tus tickers
        mis_tickers = ["RHM.DE", "IDR.MC", "MC.PA", "0P0001KGI5.F", "EUNL.DE"]
        precios = descargar_precios(mis_tickers, anios=3, cache_dir="./cache_datos")
        retornos = calcular_retornos(precios)

        # Benchmarks + cesta sintética de defensa
        precios_bench = descargar_benchmarks(["SPY", "KOSPI", "EUROSTOXX"], anios=3)
        cesta = construir_cesta_defensa_europea(anios=3)
        precios_bench = precios_bench.join(cesta, how="outer").ffill().dropna()

        ret_bench = alinear_con_cartera(retornos, precios_bench)

        # 👇 pesos reales de tu cartera (a partir de los importes en euros)
        mis_importes = {
            "RHM.DE": 1955.0,
            "IDR.MC": 1670.0,
            "MC.PA": 1095.0,
            "0P0001KGI5.F": 2176.06,
            "EUNL.DE": 531.0,
        }
        total = sum(mis_importes.values())
        pesos = pd.Series({t: v / total for t, v in mis_importes.items()})

        vol = analizar_volatilidad(retornos, ventanas=(21, 60), pesos_cartera=pesos)

        print("\n--- Comparativa de volatilidad con el mercado ---")
        print(comparar_volatilidad_benchmarks(retornos, ret_bench, pesos, ventana=60).to_string())
        print("\n--- Beta frente a cada benchmark ---")
        print(beta_vs_benchmarks(retornos, ret_bench).to_string())

        panel_volatilidad(vol, retornos, pesos, ret_bench, guardar="panel_volatilidad.png")
        print("\n[OK] Módulo 6 ejecutado. Panel guardado en 'panel_volatilidad.png'.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
