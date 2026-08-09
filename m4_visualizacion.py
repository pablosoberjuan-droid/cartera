"""
===============================================================================
 MÓDULO 4 — VISUALIZACIÓN GRÁFICA PROFESIONAL
===============================================================================
Genera el panel compuesto pedido en las especificaciones:

  a) Heatmaps comparativos de correlación: Pearson (lineal) vs Spearman (rangos)
  b) Frontera Eficiente: nube de Monte Carlo + curva exacta + Línea del Mercado
     de Capitales (CML) + marcadores únicos para Máximo Sharpe y Mínima Varianza
  c) Gráfico de barras comparando los pesos de cada activo entre estrategias

Cada pieza también se puede pedir por separado (útil si quieres un gráfico
grande de uno solo, en vez del panel compuesto en miniatura).

Uso rápido
----------
    from m4_visualizacion import panel_completo

    fig = panel_completo(
        pearson=res_corr.pearson, spearman=res_corr.spearman,
        nube=res_opt.nube_montecarlo, frontera=res_opt.frontera_eficiente,
        max_sharpe=res_opt.maximo_sharpe, min_var=res_opt.minima_varianza,
        rf=0.04, carteras_barras=[res_opt.maximo_sharpe, res_opt.minima_varianza,
                                   res_opt.paridad_riesgo],
        guardar="panel_cartera.png",
    )
===============================================================================
"""

from __future__ import annotations

import logging
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla: funciona igual guardando a PNG,
                        # y evita errores si se ejecuta sin entorno gráfico.
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

logger = logging.getLogger("quant.visualizacion")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Paleta consistente en todo el módulo
_COLOR_NUBE = "#B4C7E7"
_COLOR_FRONTERA = "#1F4E79"
_COLOR_CML = "#C00000"
_COLOR_MAX_SHARPE = "#FFC000"
_COLOR_MIN_VAR = "#00B050"
sns.set_theme(style="whitegrid", font_scale=0.95)


class ErrorDeVisualizacion(Exception):
    """Se lanza cuando los datos de entrada no permiten dibujar el gráfico."""


# =============================================================================
# a) HEATMAPS DE CORRELACIÓN
# =============================================================================

def plot_heatmaps_correlacion(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    ax_pearson: plt.Axes | None = None,
    ax_spearman: plt.Axes | None = None,
    figsize: tuple[float, float] = (11, 4.5),
) -> Figure:
    """Dos heatmaps lado a lado: Pearson (correlación lineal, la que usa
    Markowitz) vs Spearman (correlación por rangos, robusta a valores extremos).

    Si se pasan ax_pearson/ax_spearman, dibuja sobre esos ejes existentes (para
    componer un panel); si no, crea su propia figura independiente.
    """
    if pearson.empty or spearman.empty:
        raise ErrorDeVisualizacion("Las matrices de correlación están vacías.")

    fig = None
    if ax_pearson is None or ax_spearman is None:
        fig, (ax_pearson, ax_spearman) = plt.subplots(1, 2, figsize=figsize)

    for ax, matriz, titulo in (
        (ax_pearson, pearson, "Correlación de Pearson (lineal)"),
        (ax_spearman, spearman, "Correlación de Spearman (rangos)"),
    ):
        sns.heatmap(
            matriz, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={"shrink": 0.8, "label": "ρ"},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_correlacion_rolling(
    rolling: pd.DataFrame,
    ventana: int = 60,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 4),
    max_pares: int = 6,
) -> Figure:
    """Evolución temporal de la correlación (Módulo 2). No es uno de los 3
    gráficos obligatorios del panel, pero complementa muy bien el análisis de
    correlación dinámica que pide la especificación.
    """
    if rolling.empty:
        raise ErrorDeVisualizacion("La matriz de correlación rolling está vacía.")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    columnas = rolling.columns[:max_pares]  # evita saturar el gráfico si hay muchos activos
    for columna in columnas:
        ax.plot(rolling.index, rolling[columna], label=columna, linewidth=1.3, alpha=0.85)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Correlación dinámica (rolling {ventana} días)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlación")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)

    if fig is not None:
        fig.tight_layout()
    return fig


# =============================================================================
# b) FRONTERA EFICIENTE + CML
# =============================================================================

def plot_frontera_eficiente(
    nube: pd.DataFrame,
    frontera: pd.DataFrame,
    max_sharpe,          # CarteraOptima (import evitado a propósito para no acoplar módulos)
    min_var,              # CarteraOptima
    rf: float,
    cartera_actual=None,  # CarteraOptima opcional: tu cartera real, si la tienes
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """El gráfico central de Markowitz: la nube de Monte Carlo coloreada por
    Sharpe, la curva exacta de la Frontera Eficiente, la Línea del Mercado de
    Capitales (CML, del CAPM) y marcadores ÚNICOS para Máximo Sharpe y Mínima
    Varianza — y, si se lo pasas, también tu cartera actual, para que veas
    visualmente dónde caes tú respecto a lo óptimo.
    """
    if nube.empty:
        raise ErrorDeVisualizacion("La nube de Monte Carlo está vacía.")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # --- Nube de Monte Carlo, coloreada por Sharpe ---
    disp = ax.scatter(
        nube["volatilidad"], nube["retorno"], c=nube["sharpe"],
        cmap="viridis", s=6, alpha=0.35, linewidths=0,
    )
    cbar = plt.colorbar(disp, ax=ax, shrink=0.85)
    cbar.set_label("Ratio de Sharpe", fontsize=9)

    # --- Curva exacta de la Frontera Eficiente ---
    if not frontera.empty:
        ax.plot(
            frontera["volatilidad"], frontera["retorno_objetivo"],
            color=_COLOR_FRONTERA, linewidth=2.2, label="Frontera Eficiente (exacta)",
        )

    # --- Línea del Mercado de Capitales (CML): de Rf a la cartera de Máximo Sharpe,
    #     y extendida más allá para representar el apalancamiento teórico ---
    vol_max_plot = max(nube["volatilidad"].max(), max_sharpe.volatilidad) * 1.15
    pendiente_cml = (max_sharpe.retorno_esperado - rf) / max_sharpe.volatilidad
    x_cml = np.array([0, vol_max_plot])
    y_cml = rf + pendiente_cml * x_cml
    ax.plot(x_cml, y_cml, color=_COLOR_CML, linewidth=1.6, linestyle="--",
            label=f"CML (Rf={rf:.1%})")

    # --- Marcadores únicos ---
    ax.scatter(
        max_sharpe.volatilidad, max_sharpe.retorno_esperado,
        color=_COLOR_MAX_SHARPE, marker="*", s=400, edgecolors="black",
        linewidths=0.8, zorder=5, label="Máximo Sharpe (tangencia)",
    )
    ax.scatter(
        min_var.volatilidad, min_var.retorno_esperado,
        color=_COLOR_MIN_VAR, marker="D", s=130, edgecolors="black",
        linewidths=0.8, zorder=5, label="Mínima Varianza",
    )
    if cartera_actual is not None and pd.notna(cartera_actual.retorno_esperado):
        ax.scatter(
            cartera_actual.volatilidad, cartera_actual.retorno_esperado,
            color="#C00000", marker="P", s=220, edgecolors="black",
            linewidths=0.8, zorder=6, label=cartera_actual.nombre,
        )

    ax.scatter(0, rf, color="black", marker="o", s=50, zorder=5)
    ax.annotate("Rf", (0, rf), textcoords="offset points", xytext=(6, -4), fontsize=9)

    ax.set_xlabel("Volatilidad anual (riesgo)")
    ax.set_ylabel("Retorno anual esperado")
    ax.set_title("Frontera Eficiente de Markowitz", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_xlim(left=0)

    if fig is not None:
        fig.tight_layout()
    return fig


def _texto_pesos(pesos, etiquetas: Sequence[str], top_n: int = 6) -> str:
    """Formatea una composición de cartera para el tooltip: 'SPY 42% · GLD 31%...'

    Ordena de mayor a menor y omite los pesos < 0.5%, que solo añaden ruido.
    """
    pares = sorted(zip(etiquetas, pesos), key=lambda p: -p[1])
    partes = [f"{t} {w * 100:.1f}%" for t, w in pares[:top_n] if w >= 0.005]
    return " · ".join(partes) if partes else "—"


def plot_frontera_eficiente_interactivo(
    nube: pd.DataFrame,
    frontera: pd.DataFrame,
    max_sharpe,
    min_var,
    rf: float,
    cartera_actual=None,
    max_puntos_nube: int = 4000,
):
    """Versión INTERACTIVA de la Frontera Eficiente: al pasar el cursor (o hacer
    clic) sobre cualquier punto se ve la composición exacta de esa cartera.

    Cada punto de la nube de Monte Carlo y de la curva exacta lleva asociados
    sus pesos en `customdata`, de modo que el tooltip puede mostrar qué
    combinación de activos produce ese par (riesgo, retorno) concreto.

    `max_puntos_nube` submuestrea la nube: 20.000 puntos con tooltip individual
    hacen el gráfico pesado en el navegador sin aportar información nueva.

    Sin título propio (queda a cargo del `st.subheader` que lo antecede en el
    dashboard) y con la leyenda dentro de una caja discreta en la esquina
    superior izquierda — ahí la nube de Monte Carlo siempre deja hueco libre,
    porque esa zona (bajo riesgo, alto retorno) es matemáticamente inalcanzable.
    """
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeVisualizacion("plotly no está instalado. Instálalo con: pip install plotly")
    if nube.empty:
        raise ErrorDeVisualizacion("La nube de Monte Carlo está vacía.")

    from m0_estilo import BLANCO, GRIS_BORDE, NARANJA_ACENTO, ROJO_NEGATIVO

    color_frontera = "#60A5FA"     # azul claro: se lee bien sobre la nube Viridis
    color_max_sharpe = "#FCD34D"   # ámbar
    color_min_var = "#34D399"      # verde azulado

    columnas_peso = [c for c in nube.columns if c.startswith("peso_")]
    tickers = [c.replace("peso_", "") for c in columnas_peso]

    muestra = nube.sample(min(max_puntos_nube, len(nube)), random_state=42) if len(nube) > max_puntos_nube else nube

    fig = go.Figure()

    # --- Nube de Monte Carlo, coloreada por Sharpe -------------------------
    textos_nube = [
        _texto_pesos(fila, tickers) for fila in muestra[columnas_peso].values
    ] if columnas_peso else ["—"] * len(muestra)

    fig.add_trace(go.Scattergl(
        x=muestra["volatilidad"], y=muestra["retorno"],
        mode="markers",
        marker=dict(
            size=3.5, opacity=0.5, color=muestra["sharpe"], colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Sharpe", side="right", font=dict(size=11)),
                thickness=12, len=0.7, x=1.0, tickfont=dict(size=10),
                outlinewidth=0,
            ),
            showscale=True,
        ),
        customdata=np.array([textos_nube]).T,
        hovertemplate=(
            "<b>Cartera simulada</b><br>Volatilidad: %{x:.2%}<br>Retorno: %{y:.2%}"
            "<br>%{customdata[0]}<extra></extra>"
        ),
        name="Monte Carlo",
    ))

    # --- Curva exacta de la Frontera Eficiente -----------------------------
    if not frontera.empty:
        cols_peso_frontera = [c for c in frontera.columns if c.startswith("peso_")]
        if cols_peso_frontera:
            tickers_frontera = [c.replace("peso_", "") for c in cols_peso_frontera]
            textos_frontera = [
                _texto_pesos(fila, tickers_frontera)
                for fila in frontera[cols_peso_frontera].values
            ]
        else:
            textos_frontera = ["(pesos no disponibles)"] * len(frontera)

        fig.add_trace(go.Scatter(
            x=frontera["volatilidad"], y=frontera["retorno_objetivo"],
            mode="lines+markers",
            line=dict(color=color_frontera, width=3),
            marker=dict(size=4, color=color_frontera, opacity=0),  # invisibles: solo activan el hover
            customdata=np.array([textos_frontera]).T,
            hovertemplate=(
                "<b>Frontera Eficiente</b><br>Volatilidad: %{x:.2%}<br>Retorno: %{y:.2%}"
                "<br>%{customdata[0]}<extra></extra>"
            ),
            name="Frontera Eficiente",
        ))

    # --- CML — se corta poco más allá de la tangencia, no cruza todo el gráfico ---
    vol_max = min(nube["volatilidad"].max(), max_sharpe.volatilidad * 1.4)
    pendiente = (max_sharpe.retorno_esperado - rf) / max_sharpe.volatilidad
    fig.add_trace(go.Scatter(
        x=[0, vol_max], y=[rf, rf + pendiente * vol_max],
        mode="lines", line=dict(color=NARANJA_ACENTO, width=1.6, dash="dot"),
        opacity=0.8, name=f"CML (Rf={rf:.1%})", hoverinfo="skip",
    ))

    # --- Carteras destacadas ---------------------------------------------------
    # Contorno BLANCO en todas: sin él, un marcador de color intermedio se
    # camufla contra la nube Viridis justo en la zona de mayor Sharpe (que es,
    # por definición, donde suelen caer estas carteras destacadas).
    def _añadir_marcador(cartera, color, simbolo, tamano, nombre):
        if cartera is None or pd.isna(cartera.retorno_esperado):
            return
        fig.add_trace(go.Scatter(
            x=[cartera.volatilidad], y=[cartera.retorno_esperado],
            mode="markers",
            marker=dict(size=tamano, color=color, symbol=simbolo,
                        line=dict(width=2, color=BLANCO)),
            customdata=[[_texto_pesos(cartera.pesos.values, list(cartera.pesos.index), top_n=10)]],
            hovertemplate=(
                f"<b>{nombre}</b><br>Volatilidad: %{{x:.2%}}<br>Retorno: %{{y:.2%}}"
                f"<br>Sharpe: {cartera.sharpe:.2f}<br>%{{customdata[0]}}<extra></extra>"
            ),
            name=nombre,
        ))

    _añadir_marcador(max_sharpe, color_max_sharpe, "star", 19, "Máximo Sharpe")
    _añadir_marcador(min_var, color_min_var, "diamond", 13, "Mínima Varianza")
    _añadir_marcador(cartera_actual, ROJO_NEGATIVO, "cross", 15,
                     cartera_actual.nombre if cartera_actual is not None else "Tu cartera")

    fig.update_layout(
        xaxis_title="Volatilidad anual (riesgo)",
        yaxis_title="Retorno anual esperado",
        xaxis=dict(tickformat=".0%", rangemode="tozero"),
        yaxis=dict(tickformat=".0%"),
        hovermode="closest",
        legend=dict(
            x=0.015, y=0.985, xanchor="left", yanchor="top",
            bgcolor=f"rgba(20,27,46,0.88)", bordercolor=GRIS_BORDE, borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=560,
    )
    return fig


# =============================================================================
# c) COMPARATIVA DE PESOS ENTRE ESTRATEGIAS
# =============================================================================

def plot_pesos_comparativa_interactivo(carteras: Sequence):
    """Versión interactiva de la comparativa de pesos: barras agrupadas por
    activo con el peso exacto de cada estrategia en el tooltip.

    Sin título propio (lo pone el `st.subheader` del dashboard) para dejar
    toda la altura disponible al gráfico.
    """
    if not PLOTLY_DISPONIBLE:
        raise ErrorDeVisualizacion("plotly no está instalado. Instálalo con: pip install plotly")
    if not carteras:
        raise ErrorDeVisualizacion("No se pasó ninguna cartera para comparar.")

    from m0_estilo import PALETA_CATEGORICA

    tabla = pd.DataFrame({c.nombre: c.pesos for c in carteras}).fillna(0.0)

    fig = go.Figure()
    for i, estrategia in enumerate(tabla.columns):
        fig.add_trace(go.Bar(
            x=tabla.index, y=tabla[estrategia] * 100, name=estrategia,
            marker_color=PALETA_CATEGORICA[i % len(PALETA_CATEGORICA)],
            hovertemplate="%{x} — %{y:.1f}%<extra>" + estrategia + "</extra>",
        ))

    fig.update_layout(
        xaxis_title="", yaxis_title="Peso en la cartera (%)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
    )
    return fig


def plot_pesos_comparativa(
    carteras: Sequence,  # Sequence[CarteraOptima]
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> Figure:
    """Barras agrupadas: para cada activo, cuánto peso le asigna cada estrategia.
    Deja ver de un vistazo si, por ejemplo, Máximo Sharpe concentra todo en un
    activo mientras que Risk Parity lo reparte de forma mucho más homogénea.
    """
    if not carteras:
        raise ErrorDeVisualizacion("No se pasó ninguna cartera para comparar.")

    tabla = pd.DataFrame({c.nombre: c.pesos for c in carteras})
    if tabla.isna().any().any():
        # Si alguna cartera no invierte en un activo que otra sí, se rellena con 0
        tabla = tabla.fillna(0.0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_estrategias = tabla.shape[1]
    n_activos = tabla.shape[0]
    ancho_barra = 0.8 / n_estrategias
    posiciones = np.arange(n_activos)
    paleta = sns.color_palette("Set2", n_estrategias)

    for i, estrategia in enumerate(tabla.columns):
        ax.bar(
            posiciones + i * ancho_barra, tabla[estrategia].values * 100,
            width=ancho_barra, label=estrategia, color=paleta[i], edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(posiciones + ancho_barra * (n_estrategias - 1) / 2)
    ax.set_xticklabels(tabla.index, rotation=30, ha="right")
    ax.set_ylabel("Peso en la cartera (%)")
    ax.set_title("Comparativa de pesos por estrategia", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)

    if fig is not None:
        fig.tight_layout()
    return fig


# =============================================================================
# PANEL COMPUESTO (a + b + c)
# =============================================================================

def panel_completo(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    nube: pd.DataFrame,
    frontera: pd.DataFrame,
    max_sharpe,
    min_var,
    rf: float,
    carteras_barras: Sequence,
    cartera_actual=None,
    rolling: pd.DataFrame | None = None,
    ventana_rolling: int = 60,
    guardar: str | None = "panel_cartera.png",
    dpi: int = 150,
) -> Figure:
    """Genera el panel compuesto completo en una sola figura, con la
    disposición: heatmaps arriba, frontera eficiente y barras abajo (y, si se
    pasa `rolling`, una fila extra con la correlación dinámica).

    Guarda automáticamente a PNG si `guardar` no es None (por defecto sí).
    """
    incluye_rolling = rolling is not None and not rolling.empty
    n_filas = 3 if incluye_rolling else 2

    fig = plt.figure(figsize=(15, 6 * n_filas / 2 + (2.5 if incluye_rolling else 0)))
    gs = fig.add_gridspec(n_filas, 2, height_ratios=([1, 1.3, 0.8] if incluye_rolling else [1, 1.3]))

    ax_pearson = fig.add_subplot(gs[0, 0])
    ax_spearman = fig.add_subplot(gs[0, 1])
    plot_heatmaps_correlacion(pearson, spearman, ax_pearson=ax_pearson, ax_spearman=ax_spearman)

    ax_frontera = fig.add_subplot(gs[1, 0])
    plot_frontera_eficiente(
        nube, frontera, max_sharpe, min_var, rf,
        cartera_actual=cartera_actual, ax=ax_frontera,
    )

    ax_barras = fig.add_subplot(gs[1, 1])
    plot_pesos_comparativa(carteras_barras, ax=ax_barras)

    if incluye_rolling:
        ax_rolling = fig.add_subplot(gs[2, :])
        plot_correlacion_rolling(rolling, ventana=ventana_rolling, ax=ax_rolling)

    fig.suptitle(
        "Panel de análisis de cartera — MPT · CAPM · Risk Parity",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if guardar is not None:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Panel guardado en '%s' (%d dpi)", guardar, dpi)

    return fig


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m4_visualizacion.py
# =============================================================================

if __name__ == "__main__":
    try:
        from m1_datos import calcular_retornos, descargar_precios
        from m2_correlaciones import analizar_correlaciones
        from m3_optimizacion import cartera_desde_importes, optimizar_cartera

        # 👇 tus tickers
        mis_tickers = ["RHM.DE", "IDR.MC", "MC.PA", "0P0001KGI5.F", "EUNL.DE"]

        # 👇 lo que tienes invertido hoy en cada uno (valor actual, en euros)
        mis_importes = {
            "RHM.DE": 1955.0,
            "IDR.MC": 1670.0,
            "MC.PA": 1095.0,
            "0P0001KGI5.F": 2176.06,
            "EUNL.DE": 531.0,
        }

        precios = descargar_precios(mis_tickers, anios=3, cache_dir="./cache_datos")
        retornos = calcular_retornos(precios)
        res_corr = analizar_correlaciones(retornos, ventana=60)
        res_opt = optimizar_cartera(retornos, rf=0.04, ticker_mercado="EUNL.DE")
        cartera_actual = cartera_desde_importes(mis_importes, res_opt.mu_anual, res_opt.cov_anual, rf=0.04)

        panel_completo(
            pearson=res_corr.pearson,
            spearman=res_corr.spearman,
            nube=res_opt.nube_montecarlo,
            frontera=res_opt.frontera_eficiente,
            max_sharpe=res_opt.maximo_sharpe,
            min_var=res_opt.minima_varianza,
            rf=0.04,
            cartera_actual=cartera_actual,
            carteras_barras=[cartera_actual, res_opt.maximo_sharpe, res_opt.minima_varianza, res_opt.paridad_riesgo],
            rolling=res_corr.rolling,
            guardar="panel_cartera.png",
        )
        print("\n--- Tu cartera actual ---")
        print(cartera_actual)
        print(cartera_actual.extra["importe"])
        print("\n[OK] Módulo 4 ejecutado sin errores. Panel guardado en 'panel_cartera.png'.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
