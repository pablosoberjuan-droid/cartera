"""
===============================================================================
 MÓDULO 9 — DASHBOARD WEB LOCAL (Fase 4, parte A)
===============================================================================
Convierte todos los módulos anteriores en una web local con Streamlit: en vez
de ejecutar scripts y abrir PNGs, abres una página en el navegador, pulsas
"Actualizar" y ves todo recalculado con datos en vivo de Yahoo Finance.

CÓMO EJECUTARLO
---------------
    pip3 install streamlit plotly
    streamlit run m9_dashboard.py

Se abrirá solo en http://localhost:8501

Para pararlo: Ctrl+C en la terminal.

NOTA: este archivo NO se ejecuta con `python m9_dashboard.py`. Streamlit tiene
su propio lanzador (`streamlit run`), que es el que arranca el servidor web.

CACHÉ
-----
Las descargas se cachean 15 minutos con @st.cache_data. Sin esto, cada vez que
mueves un control (un slider, una casilla) Streamlit re-ejecuta TODO el script
y volverías a descargar de Yahoo, lo que sería lentísimo y acabaría bloqueado
por rate limiting.
===============================================================================
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime

import pandas as pd

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Falta 'streamlit'. Instálalo con:  pip3 install streamlit\n"
        "Y ejecuta este archivo con:  streamlit run m9_dashboard.py"
    ) from exc

from m0_estilo import aplicar_tema_matplotlib, aplicar_tema_plotly, barra_superior, inyectar_css
from m1_datos import calcular_retornos, descargar_precios, resumen_activos
from m2_correlaciones import analizar_correlaciones
from m3_optimizacion import (
    calcular_pnl_posiciones,
    cartera_desde_importes,
    comparar_carteras,
    optimizar_cartera,
)
from m4_visualizacion import (
    plot_frontera_eficiente_interactivo,
    plot_heatmaps_correlacion,
    plot_pesos_comparativa_interactivo,
)
from m5_volatilidad_historica import analizar_volatilidad, beneficio_diversificacion
from m6_benchmarks_volatilidad import (
    alinear_con_cartera,
    beta_vs_benchmarks,
    comparar_volatilidad_benchmarks,
    construir_cesta_defensa_europea,
    descargar_benchmarks,
    plot_cartera_vs_benchmarks_interactivo,
    plot_mapa_calor_volatilidad_interactivo,
    plot_volatilidad_rolling_interactivo,
)
from m11_recomendaciones import (
    CANDIDATOS_DEFECTO,
    calcular_retornos_candidatos,
    descargar_precios_candidatos,
    evaluar_candidatos,
    peso_optimo_dos_activos,
    plot_correlacion_vs_sharpe,
    plot_ranking_candidatos,
    simular_incorporacion,
)
from m11_energia import (
    FUTUROS_ENERGIA,
    analizar_complejo_energetico,
    calcular_diferenciales,
    generar_alertas_energia,
    listar_tickers_energia,
    plot_diferenciales,
    plot_estacionalidad,
    plot_precios_normalizados,
)
from m12_fuentes import crisis_energeticas_historicas, energia_historica_larga
from m13_gas_europa import (
    ErrorDeGIE,
    PAISES_GAS,
    analizar_gas_europeo,
    plot_estacionalidad_llenado,
    plot_flujos,
    plot_llenado,
)

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Panel Quant de Cartera",
    page_icon="📊",
    layout="wide",
)
inyectar_css(st)
aplicar_tema_matplotlib()


def _pro(fig):
    """Aplica el tema visual profesional a una figura Plotly antes de mostrarla."""
    return aplicar_tema_plotly(fig)

# Cartera por defecto (edítala aquí o desde la barra lateral)
CARTERA_DEFECTO: dict[str, float] = {
    "RHM.DE": 1955.0,
    "IDR.MC": 1670.0,
    "MC.PA": 1095.0,
    "0P0001KGI5.F": 2176.06,
    "EUNL.DE": 531.0,
}

# Lo que pagaste por cada posición (precio de compra y nº de títulos). Solo se
# usa para la tabla de rentabilidad real de la pestaña "Resumen"; si editas la
# cartera en la barra lateral con tickers que no están aquí, esa tabla se omite.
COMPRAS_DEFECTO: dict[str, dict[str, float]] = {
    "RHM.DE": {"precio_compra": 497.04, "titulos": 1.69},
    "IDR.MC": {"precio_compra": 35.13, "titulos": 24.2},
    "MC.PA": {"precio_compra": 573.67, "titulos": 2.17},
    "0P0001KGI5.F": {"precio_compra": 205.55, "titulos": 9.875},
    "EUNL.DE": {"precio_compra": 11.55, "titulos": 43.29},
}


# =============================================================================
# FUNCIONES CACHEADAS
# =============================================================================
# El TTL de 15 minutos es un compromiso: suficientemente fresco para datos
# diarios (que solo cambian al cierre) y suficientemente largo para no saturar
# la API de Yahoo mientras trasteas con los controles.

@st.cache_data(ttl=900, show_spinner="Descargando precios...")
def cargar_precios(tickers: tuple[str, ...], anios: float) -> pd.DataFrame:
    return descargar_precios(list(tickers), anios=anios)


@st.cache_data(ttl=900, show_spinner="Descargando benchmarks...")
def cargar_benchmarks(claves: tuple[str, ...], anios: float, con_defensa: bool) -> pd.DataFrame:
    precios = descargar_benchmarks(list(claves), anios=anios)
    if con_defensa:
        try:
            cesta = construir_cesta_defensa_europea(anios=anios)
            precios = precios.join(cesta, how="outer").ffill().dropna()
        except Exception as exc:
            st.warning(f"No se pudo construir la cesta de defensa europea: {exc}")
    return precios


@st.cache_data(ttl=900, show_spinner="Descargando complejo energético...")
def cargar_energia(claves: tuple[str, ...], anios: float):
    return analizar_complejo_energetico(list(claves), anios=anios)


@st.cache_data(ttl=3600, show_spinner="Descargando histórico largo de FRED (puede tardar)...")
def cargar_energia_larga(anios: float):
    largo = energia_historica_larga(anios=anios)
    crisis = crisis_energeticas_historicas(largo)
    return largo, crisis


@st.cache_data(ttl=900, show_spinner="Descargando almacenamiento de gas europeo (GIE)...")
def cargar_gas_europa(paises: tuple[str, ...], anios: float, api_key: str):
    return analizar_gas_europeo(list(paises), anios=anios, api_key=api_key)


@st.cache_data(ttl=900, show_spinner="Descargando universo de candidatos (puede tardar un minuto)...")
def cargar_candidatos(tickers: tuple[str, ...], anios: float):
    precios, fallidos = descargar_precios_candidatos(list(tickers), anios=anios)
    retornos = calcular_retornos_candidatos(precios)
    return retornos, fallidos


@st.cache_data(ttl=900, show_spinner="Descargando cadena de opciones...")
def cargar_superficie(ticker: str, max_vencimientos: int):
    """Cacheamos solo los datos serializables, no el objeto completo con mallas."""
    from m7_superficie_iv import (
        calcular_metricas,
        construir_superficie,
        descargar_cadena_opciones,
        estructura_temporal,
        generar_alertas_iv,
        limpiar_cadena,
    )

    bruto, spot = descargar_cadena_opciones(ticker, max_vencimientos=max_vencimientos)
    cadena = limpiar_cadena(bruto, spot)
    malla_x, malla_y, malla_z = construir_superficie(cadena)
    resumen = estructura_temporal(cadena)
    metricas = calcular_metricas(cadena, resumen)
    alertas = generar_alertas_iv(metricas)
    return spot, cadena, malla_x, malla_y, malla_z, resumen, metricas, alertas


# =============================================================================
# BARRA LATERAL — CONTROLES
# =============================================================================

st.sidebar.title("⚙️ Configuración")

st.sidebar.subheader("Tu cartera")
texto_cartera = st.sidebar.text_area(
    "Un activo por línea, formato `TICKER: importe`",
    value="\n".join(f"{t}: {v:.0f}" for t, v in CARTERA_DEFECTO.items()),
    height=140,
    help="Los tickers deben ser los de Yahoo Finance (p. ej. RHM.DE, IDR.MC, MC.PA).",
)


def parsear_cartera(texto: str) -> dict[str, float]:
    """Convierte el texto de la barra lateral en un diccionario de importes."""
    importes: dict[str, float] = {}
    for numero, linea in enumerate(texto.strip().splitlines(), start=1):
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        if ":" not in linea:
            raise ValueError(f"Línea {numero} sin ':' → «{linea}». Formato correcto: TICKER: importe")
        ticker, importe = linea.split(":", 1)
        ticker = ticker.strip().upper()
        try:
            importes[ticker] = float(importe.strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"Línea {numero}: «{importe.strip()}» no es un número válido.") from exc
    if not importes:
        raise ValueError("No se ha introducido ningún activo.")
    return importes


anios = st.sidebar.slider("Años de histórico", 1.0, 10.0, 3.0, 0.5)
rf = st.sidebar.number_input("Tasa libre de riesgo (Rf) %", 0.0, 15.0, 4.0, 0.25) / 100
ventana_corta = st.sidebar.selectbox("Ventana volatilidad corta (días)", [10, 21, 30], index=1)
ventana_larga = st.sidebar.selectbox("Ventana volatilidad larga (días)", [60, 90, 120], index=0)

st.sidebar.subheader("Benchmarks")
benchmarks_elegidos = st.sidebar.multiselect(
    "Índices de referencia",
    ["SPY", "KOSPI", "EUROSTOXX", "VIX", "MSCI_WORLD"],
    default=["SPY", "KOSPI", "EUROSTOXX"],
)
incluir_defensa = st.sidebar.checkbox("Incluir cesta de defensa europea", value=True)

st.sidebar.subheader("Volatilidad implícita")
proxy_iv = st.sidebar.selectbox(
    "Subyacente para la superficie",
    ["SPY", "ITA", "EZU", "URTH", "QQQ", "EWY"],
    index=0,
    help="yfinance solo tiene opciones de EE.UU. ITA≈defensa, EZU≈eurozona, QQQ≈tecnología.",
)
n_vencimientos = st.sidebar.slider("Vencimientos a descargar", 4, 12, 8)

if st.sidebar.button("🔄 Forzar actualización de datos"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Última carga: {datetime.now():%Y-%m-%d %H:%M:%S}")


# =============================================================================
# CUERPO PRINCIPAL
# =============================================================================

st.title("📊 Panel Quant de Cartera")
st.caption(
    "MPT · CAPM · Black-Litterman · Risk Parity · Kelly · Volatilidad realizada e implícita"
)

try:
    importes = parsear_cartera(texto_cartera)
except ValueError as exc:
    st.error(f"Error en la definición de la cartera: {exc}")
    st.stop()

tickers = tuple(importes.keys())

try:
    precios = cargar_precios(tickers, anios)
    retornos = calcular_retornos(precios)
except Exception as exc:
    st.error(f"No se pudieron descargar los precios: {exc}")
    st.stop()

# Avisamos si algún ticker se cayó por el control de calidad del Módulo 1
descartados = [t for t in tickers if t not in precios.columns]
if descartados:
    st.warning(
        f"Estos activos se descartaron por falta de datos históricos: {', '.join(descartados)}. "
        "El análisis continúa con el resto."
    )
importes_validos = {t: v for t, v in importes.items() if t in precios.columns}

# --- Barra superior tipo terminal financiera ---------------------------------
_pesos_kpi = pd.Series(importes_validos) / sum(importes_validos.values())
_r_cartera_kpi = (retornos * _pesos_kpi.reindex(retornos.columns).fillna(0.0)).sum(axis=1)
_media_kpi = _r_cartera_kpi.mean() * 252
_vol_kpi = _r_cartera_kpi.std(ddof=1) * (252 ** 0.5)
_sharpe_kpi = (_media_kpi - rf) / _vol_kpi if _vol_kpi > 1e-12 else float("nan")

_compras_kpi = {t: c for t, c in COMPRAS_DEFECTO.items() if t in importes_validos}
_celdas_kpi = [
    ("Invertido", f"{sum(importes_validos.values()):,.0f} €", None),
    ("Retorno anual (histórico 3a)", f"{_media_kpi * 100:+.2f} %", "positivo" if _media_kpi >= 0 else "negativo"),
    ("Volatilidad anual", f"{_vol_kpi * 100:.2f} %", None),
    ("Sharpe", f"{_sharpe_kpi:.2f}", "positivo" if _sharpe_kpi >= 1 else None),
]
if _compras_kpi:
    _tabla_kpi = calcular_pnl_posiciones(_compras_kpi, importes_validos)
    _pnl_kpi = _tabla_kpi.loc["TOTAL", "P&L €"]
    _pnl_pct_kpi = _tabla_kpi.loc["TOTAL", "P&L %"]
    _celdas_kpi.append((
        "P&L real (compra)", f"{_pnl_kpi:+,.0f} € ({_pnl_pct_kpi:+.1f}%)",
        "positivo" if _pnl_kpi >= 0 else "negativo",
    ))
barra_superior(st, _celdas_kpi)

pestanas = st.tabs([
    "📈 Resumen", "🔗 Correlaciones", "🎯 Optimización",
    "📉 Volatilidad histórica", "🌐 Superficie implícita", "🧭 Recomendaciones",
    "⚡ Energía", "🌍 Gas Europa",
])


# ----------------------------------------------------------------- Resumen ---
with pestanas[0]:
    st.subheader("Ficha de los activos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Activos", len(precios.columns))
    col2.metric("Sesiones", len(precios))
    col3.metric("Invertido", f"{sum(importes_validos.values()):,.0f} €")

    st.dataframe(resumen_activos(precios, retornos, rf), use_container_width=True)
    st.caption(
        "⚠️ Esta ficha usa el histórico de MERCADO de los últimos 3 años (precio de hace 3 años "
        "→ precio de hoy), no tu precio de compra real ni tu fecha de entrada. "
        "'Rent. total %' es un dato de contexto del activo, no tu rentabilidad personal. "
        "Para TU rentabilidad real (con tu precio de compra y tus títulos), baja hasta "
        "**'Rentabilidad real (precio de compra vs valor actual)'**, unas líneas más abajo."
    )

    st.subheader("Evolución (base 100)")
    st.line_chart(precios / precios.iloc[0] * 100)

    compras_disponibles = {t: c for t, c in COMPRAS_DEFECTO.items() if t in importes_validos}
    if compras_disponibles:
        st.subheader("Rentabilidad real (precio de compra vs valor actual)")
        st.latex(r"\text{P\&L}_i = V_{actual,i} - (P_{compra,i} \times N_{títulos,i})")
        st.caption(
            "Coste de cada posición = precio de compra × nº de títulos. La plusvalía/minusvalía "
            "es la diferencia entre ese coste y el valor actual que has indicado — a diferencia "
            "de las secciones anteriores (basadas en precios históricos de mercado), esta es TU "
            "rentabilidad real, incluyendo el momento exacto en que compraste."
        )
        tabla_pnl = calcular_pnl_posiciones(compras_disponibles, importes_validos)
        st.dataframe(tabla_pnl, use_container_width=True)

        pnl_total = tabla_pnl.loc["TOTAL", "P&L €"]
        pnl_pct_total = tabla_pnl.loc["TOTAL", "P&L %"]
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("P&L total", f"{pnl_total:,.2f} €", f"{pnl_pct_total:+.2f}%")
        col_p2.metric("Coste total", f"{tabla_pnl.loc['TOTAL', 'Coste €']:,.2f} €")


# ------------------------------------------------------------ Correlaciones ---
with pestanas[1]:
    try:
        corr = analizar_correlaciones(retornos, ventana=ventana_larga)
        st.subheader("Pearson vs Spearman")
        st.latex(
            r"\rho^{Pearson}_{X,Y} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X \, \sigma_Y}"
            r"\qquad\qquad"
            r"\rho^{Spearman}_{X,Y} = \rho^{Pearson}_{\,\operatorname{rango}(X),\,\operatorname{rango}(Y)}"
        )
        st.caption(
            "**Pearson** mide la relación LINEAL entre los retornos diarios de dos activos "
            "(de -1 a +1); es la que alimenta la matriz de covarianzas del optimizador (pestaña "
            "Optimización), pero es sensible a valores extremos. **Spearman** aplica la misma "
            "fórmula pero sobre los RANGOS de los retornos, no sobre sus valores: captura "
            "relaciones monótonas no lineales y es robusta frente a crashes. "
            "**Interpretación:** si Spearman >> Pearson hay dependencia oculta que Markowitz no "
            "ve; si Pearson >> Spearman, la correlación la están generando unos pocos días extremos."
        )
        st.pyplot(plot_heatmaps_correlacion(corr.pearson, corr.spearman))

        st.subheader("Comparativa par a par")
        st.dataframe(corr.comparativa, use_container_width=True)

        st.subheader(f"Correlación dinámica ({ventana_larga} días)")
        st.latex(
            r"\rho_t^{(n)} = \operatorname{corr}\big(\{r_{X,i}\}_{i=t-n+1}^{t},\ "
            r"\{r_{Y,i}\}_{i=t-n+1}^{t}\big)"
        )
        st.caption(
            "La misma correlación de Pearson, pero calculada solo con los últimos *n* días "
            "(la ventana larga elegida en la barra lateral) y recalculada cada día, formando una "
            "serie temporal. **Interpretación:** la correlación NO es constante — en las crisis "
            "tiende a subir hacia 1 justo cuando más falta hace la diversificación "
            "('correlation breakdown'). Un rango amplio entre el mínimo y el máximo de la serie "
            "es la señal de que no puedes confiar en la correlación media de 3 años para ese par."
        )
        st.line_chart(corr.rolling)

        st.subheader("Diagnóstico")
        for alerta in corr.alertas:
            st.write(alerta)
    except Exception as exc:
        st.error(f"Error en el análisis de correlaciones: {exc}")


# -------------------------------------------------------------- Optimización ---
with pestanas[2]:
    try:
        with st.spinner("Optimizando (Monte Carlo + scipy)..."):
            opt = optimizar_cartera(
                retornos, rf=rf,
                ticker_mercado=precios.columns[-1],
                n_carteras_mc=20_000,
            )
            mi_cartera = cartera_desde_importes(importes_validos, opt.mu_anual, opt.cov_anual, rf)

        st.latex(
            r"\text{maximizar}\ \ \frac{w^\top \mu - R_f}{\sqrt{w^\top \Sigma w}}"
            r"\qquad \text{sujeto a}\ \ \textstyle\sum_i w_i = 1,\ \ w_i \geq 0"
        )
        st.caption(
            "**Ratio de Sharpe de la cartera**, con *μ* el vector de retornos anuales esperados, "
            "*Σ* la matriz de covarianzas y *w* los pesos. La nube de puntos es Monte Carlo "
            "(20.000 carteras con pesos aleatorios); la curva es la Frontera Eficiente exacta "
            "(resuelta con `scipy.optimize`); la línea discontinua es la CML (Capital Market "
            "Line) desde *Rf*. **Interpretación:** ninguna cartera puede quedar por encima de la "
            "curva — es el límite teórico riesgo/retorno. La ⭐ es la de máximo Sharpe (tangencia "
            "con la CML); la cruz roja es TU cartera real: cuanto más cerca de la curva y más a "
            "la derecha de tu posición actual esté la ⭐, más margen de mejora tienes."
        )
        st.subheader("Frontera Eficiente")
        st.caption(
            "🖱️ **Pasa el cursor por cualquier punto** (nube gris-verde o curva azul) para ver "
            "la composición exacta de esa cartera: qué % en cada activo produce ese par "
            "riesgo/retorno. Usa la barra de herramientas del gráfico para hacer zoom, y haz "
            "clic en la leyenda para ocultar series."
        )
        st.plotly_chart(_pro(plot_frontera_eficiente_interactivo(
            opt.nube_montecarlo, opt.frontera_eficiente,
            opt.maximo_sharpe, opt.minima_varianza, rf,
            cartera_actual=mi_cartera,
        )), use_container_width=True)

        st.subheader("Pesos por estrategia")
        st.plotly_chart(_pro(plot_pesos_comparativa_interactivo([
            mi_cartera, opt.maximo_sharpe, opt.minima_varianza, opt.paridad_riesgo,
        ])), use_container_width=True)

        st.subheader("Comparativa de carteras")
        st.dataframe(comparar_carteras([
            mi_cartera, opt.maximo_sharpe, opt.minima_varianza,
            opt.paridad_riesgo, opt.black_litterman.cartera_resultante,
        ]), use_container_width=True)

        col_a, col_b = st.columns(2)
        col_a.subheader("CAPM")
        col_a.latex(r"E(R_i) = R_f + \beta_i \big(E(R_m) - R_f\big) \qquad \beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}")
        col_a.caption(
            "El retorno que 'debería' tener un activo según su Beta (sensibilidad al mercado "
            "elegido). **Interpretación:** si el retorno histórico está muy por encima del CAPM, "
            "el activo se ha comportado mejor de lo que su riesgo sistemático justificaría en "
            "este periodo (o el mercado elegido no es el benchmark adecuado para él)."
        )
        col_a.dataframe(opt.capm, use_container_width=True)
        col_b.subheader("Kelly multiactivo")
        col_b.latex(r"f^{*} = \Sigma^{-1}(\mu - R_f)")
        col_b.dataframe(
            opt.kelly_multiactivo.rename("f* (fracción óptima)").to_frame().round(3),
            use_container_width=True,
        )
        col_b.caption(
            "Fracción del capital a apalancar/asignar a cada activo que maximiza el crecimiento "
            "geométrico esperado a largo plazo, dado *μ*, *Σ* y *Rf*. "
            "**Interpretación:** Kelly completo es agresivo y muy sensible a errores de "
            "estimación de *μ* — un valor negativo sugiere ir corto (no aplicable long-only). "
            "En la práctica se suele usar 1/2 o 1/4 de estas fracciones."
        )
    except Exception as exc:
        st.error(f"Error en la optimización: {exc}")
        st.code(traceback.format_exc())


# ------------------------------------------------------ Volatilidad histórica ---
with pestanas[3]:
    try:
        mi_cartera_pesos = pd.Series(importes_validos) / sum(importes_validos.values())
        vol = analizar_volatilidad(
            retornos, ventanas=(ventana_corta, ventana_larga),
            pesos_cartera=mi_cartera_pesos,
        )

        st.subheader("Régimen actual")
        st.dataframe(vol.regimen, use_container_width=True)

        st.subheader("Evolución de la volatilidad")
        st.latex(
            r"\sigma_t^{(n)} = \sqrt{252 \cdot \frac{1}{n-1}"
            r"\sum_{i=t-n+1}^{t} \left(r_i - \bar{r}_n\right)^2}"
        )
        st.caption(
            "**Desviación estándar móvil, anualizada.** Se calcula la desviación típica "
            "de los últimos *n* retornos diarios (n = 21 ó 60 en este panel) y se multiplica "
            "por √252 (sesiones bursátiles/año) para expresarla en términos anuales. "
            "**Interpretación:** cuanto más alta, más ha oscilado el precio en esa ventana. "
            "Si la línea punteada (corto plazo) sube por encima de la gruesa (largo plazo), "
            "el riesgo del activo se está ACELERANDO ahora mismo; si está por debajo, se enfría."
        )
        st.plotly_chart(_pro(plot_volatilidad_rolling_interactivo(
            vol.rolling[ventana_corta], vol.rolling[ventana_larga],
            ventana_corta, ventana_larga,
        )), use_container_width=True)

        st.subheader("Mapa de calor mensual")
        st.latex(
            r"V_{a,m} = \frac{1}{|D_m|}\sum_{t \in D_m} \sigma_t^{(n_{\text{larga}}),\,a}"
        )
        st.caption(
            "**Media mensual de la volatilidad rolling.** Para cada activo *a* y cada mes *m*, "
            "se promedian todos los valores diarios de σ (la misma fórmula del gráfico anterior, "
            "con la ventana larga) que caen dentro de ese mes. "
            "**Interpretación:** las celdas más oscuras/rojas marcan los meses y activos más "
            "turbulentos; permite detectar de un vistazo si un episodio de estrés fue general "
            "(toda la fila/columna oscura) o específico de un solo activo."
        )
        st.plotly_chart(
            _pro(plot_mapa_calor_volatilidad_interactivo(vol.rolling[ventana_larga])),
            use_container_width=True,
        )

        if benchmarks_elegidos:
            precios_bench = cargar_benchmarks(tuple(benchmarks_elegidos), anios, incluir_defensa)
            ret_bench = alinear_con_cartera(retornos, precios_bench)

            st.subheader("Tu cartera vs el mercado")
            st.latex(
                r"r_{cartera,t} = \sum_i w_i \, r_{i,t} "
                r"\qquad \sigma_{cartera,t}^{(n)} = \sqrt{252 \cdot \frac{1}{n-1}"
                r"\sum_{k=t-n+1}^{t} \left(r_{cartera,k} - \bar{r}_{cartera,n}\right)^2}"
            )
            st.caption(
                "**Volatilidad de la cartera COMPLETA, no la media de sus activos.** Primero se "
                "construye el retorno diario de la cartera como la suma de los retornos de cada "
                "activo ponderados por su peso *wᵢ* (importe invertido / total), y después se le "
                "aplica la misma volatilidad rolling anualizada. "
                "**Interpretación:** gracias a que tus activos no están perfectamente "
                "correlacionados, la línea negra (tu cartera) casi siempre queda por DEBAJO de "
                "la media ponderada de los benchmarks — esa diferencia es el beneficio real de "
                "diversificar. Compárala con los benchmarks de color para saber si tu riesgo "
                "actual es alto en términos absolutos o solo porque todo el mercado está nervioso."
            )
            st.plotly_chart(_pro(plot_cartera_vs_benchmarks_interactivo(
                retornos, mi_cartera_pesos, ret_bench, ventana=ventana_larga,
            )), use_container_width=True)
            st.dataframe(
                comparar_volatilidad_benchmarks(
                    retornos, ret_bench, mi_cartera_pesos, ventana=ventana_larga
                ),
                use_container_width=True,
            )
            st.subheader("Beta frente a cada benchmark")
            st.dataframe(beta_vs_benchmarks(retornos, ret_bench), use_container_width=True)

        st.subheader("Beneficio de diversificar")
        st.dataframe(
            beneficio_diversificacion(retornos, mi_cartera_pesos, ventana_larga).tail(10),
            use_container_width=True,
        )

        st.subheader("Diagnóstico")
        for alerta in vol.alertas:
            st.write(alerta)
    except Exception as exc:
        st.error(f"Error en el análisis de volatilidad: {exc}")
        st.code(traceback.format_exc())


# ------------------------------------------------------ Superficie implícita ---
with pestanas[4]:
    st.info(
        "yfinance solo sirve cadenas de opciones de EE.UU. Ninguno de tus activos europeos "
        "las tiene, así que se usa un proxy: **ITA**≈defensa (RHM), **EZU**≈eurozona (IDR/MC), "
        "**QQQ**≈tecnología, **URTH**≈MSCI World."
    )
    try:
        with st.spinner(f"Descargando cadena de opciones de {proxy_iv}..."):
            spot, cadena, mx, my, mz, resumen_iv, metricas_iv, alertas_iv = cargar_superficie(
                proxy_iv, n_vencimientos
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Spot", f"{spot:,.2f}")
        col2.metric("IV ATM", f"{metricas_iv.get('IV ATM referencia %', float('nan')):.1f}%")
        col3.metric("Skew", f"{metricas_iv.get('Skew (put0.90 - call1.10)', float('nan')):+.1f}")

        st.latex(
            r"C = S\,N(d_1) - K e^{-r\tau} N(d_2) \qquad "
            r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}},\ \ "
            r"d_2 = d_1 - \sigma\sqrt{\tau}"
        )
        st.caption(
            "La superficie no se mide directamente: cada punto es la **volatilidad implícita** "
            "(σ) que, insertada en Black-Scholes, reproduce el precio de mercado de ESE contrato "
            "concreto (strike y vencimiento). Yahoo Finance ya la calcula y la publica junto al "
            "precio; aquí se interpola entre contratos (`scipy.interpolate.griddata`) para formar "
            "una superficie continua en (días al vencimiento × moneyness). "
            "**Interpretación — Skew:** si la IV es más alta en strikes BAJOS (puts OTM) que en "
            "altos, el mercado paga más por protegerse de caídas que por apostar a subidas — es "
            "la forma normal en renta variable desde 1987, y cuanto más pronunciado, más miedo. "
            "**Estructura temporal:** si el corto plazo cotiza MÁS IV que el largo (curva "
            "invertida), el mercado descuenta un evento inminente (resultados, tipos, geopolítica)."
        )

        try:
            import plotly.graph_objects as go

            from m0_estilo import GRIS_BORDE, GRIS_TEXTO, NAVY_PANEL

            figura = go.Figure(data=[go.Surface(x=mx, y=my, z=mz, colorscale="Viridis")])
            eje_3d = dict(
                backgroundcolor=NAVY_PANEL, gridcolor=GRIS_BORDE,
                showbackground=True, zerolinecolor=GRIS_BORDE, color=GRIS_TEXTO,
            )
            figura.update_layout(
                scene=dict(
                    xaxis=dict(title="Días al vencimiento", **eje_3d),
                    yaxis=dict(title="Moneyness", **eje_3d),
                    zaxis=dict(title="IV (%)", **eje_3d),
                    bgcolor=NAVY_PANEL,
                ),
                paper_bgcolor=NAVY_PANEL,
                font=dict(color=GRIS_TEXTO),
                height=650, margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(figura, use_container_width=True)
        except ImportError:
            st.warning("Instala plotly (`pip3 install plotly`) para ver la superficie 3D interactiva.")

        col_a, col_b = st.columns(2)
        col_a.subheader("Estructura temporal")
        col_a.dataframe(resumen_iv, use_container_width=True)
        col_b.subheader("Métricas")
        col_b.dataframe(
            pd.Series(metricas_iv, name="Valor").to_frame(), use_container_width=True
        )

        st.subheader("Diagnóstico")
        for alerta in alertas_iv:
            st.write(alerta)
    except Exception as exc:
        st.error(f"No se pudo construir la superficie de {proxy_iv}: {exc}")


# ------------------------------------------------------------ Recomendaciones ---
with pestanas[5]:
    st.info(
        "Cribado cuantitativo sobre un universo fijo de ~20 ETFs/activos líquidos de Yahoo "
        "Finance (bonos, oro, materias primas, inmobiliario, emergentes, otros sectores y "
        "regiones) que NO están en tu cartera. No es asesoramiento financiero personalizado."
    )
    st.latex(r"SR_c > \rho(c, P) \cdot SR_P")
    st.caption(
        "**Criterio de Treynor-Black / Elton-Gruber.** Un candidato *c* mejora el Sharpe de la "
        "cartera óptima resultante si y solo si su propio Sharpe (*SR_c*) supera el 'listón' "
        "*ρ(c,P)·SR_P* — su correlación con tu cartera actual multiplicada por el Sharpe que ya "
        "tienes. **Interpretación:** un activo mediocre en solitario puede merecer la pena si "
        "apenas correlaciona con lo que ya tienes (diversifica); un activo brillante pero muy "
        "correlacionado con tu cartera aporta poco que no tuvieras ya."
    )

    try:
        with st.spinner("Evaluando universo de candidatos..."):
            retornos_candidatos, fallidos_dl = cargar_candidatos(
                tuple(CANDIDATOS_DEFECTO.keys()), anios,
            )
            pesos_actuales = pd.Series(importes_validos) / sum(importes_validos.values())
            res_rec = evaluar_candidatos(retornos, pesos_actuales, retornos_candidatos, rf=rf)

        col1, col2 = st.columns(2)
        col1.metric("Sharpe de tu cartera actual", f"{res_rec.sharpe_cartera:.2f}")
        col2.metric(
            "Candidatos con score positivo",
            int((res_rec.tabla["Score (mejora esperada)"] > 0).sum()),
            f"de {len(res_rec.tabla)} evaluados",
        )

        st.subheader("Ranking de candidatos")
        st.plotly_chart(_pro(plot_ranking_candidatos(res_rec.tabla)), use_container_width=True)
        st.plotly_chart(
            _pro(plot_correlacion_vs_sharpe(res_rec.tabla, res_rec.sharpe_cartera)),
            use_container_width=True,
        )
        st.dataframe(res_rec.tabla, use_container_width=True)

        if res_rec.fallidos or fallidos_dl:
            with st.expander(f"Candidatos descartados ({len(res_rec.fallidos) + len(fallidos_dl)})"):
                for t, motivo in {**fallidos_dl, **res_rec.fallidos}.items():
                    st.write(f"**{t}**: {motivo}")

        st.subheader("Diagnóstico")
        for alerta in res_rec.alertas:
            st.write(alerta)

        st.subheader("Simulación: ¿y si incorporo un candidato?")
        st.caption(
            "Elige un candidato y qué porcentaje de tu cartera destinarías a él (se resta "
            "proporcionalmente del resto de posiciones) para ver el efecto inmediato en "
            "retorno, volatilidad y Sharpe — sin esperar a una reoptimización completa."
        )
        candidato_elegido = st.selectbox(
            "Candidato", res_rec.tabla.index.tolist(),
            format_func=lambda t: f"{t} — {res_rec.tabla.loc[t, 'Nombre']}",
        )

        peso_optimo = peso_optimo_dos_activos(
            retornos, pesos_actuales, retornos_candidatos[candidato_elegido], rf=rf,
        )
        st.caption(
            f"📐 Peso que **maximiza** el Sharpe combinado para {candidato_elegido}: "
            f"**{peso_optimo * 100:.1f}%** (resolviendo Markowitz para 2 'activos': tu cartera "
            "y el candidato). Pesos muy alejados de este óptimo pueden EMPEORAR tu Sharpe "
            "aunque el candidato tenga score positivo — el criterio de arriba solo garantiza "
            "que existe un peso que mejora, no que cualquier peso lo haga."
        )
        peso_optimo_recortado = round(min(peso_optimo, 0.50), 2)
        usar_optimo = st.checkbox(
            f"Usar el peso óptimo ({peso_optimo_recortado * 100:.0f} %)",
            value=True, key=f"usar_optimo_{candidato_elegido}",
        )
        if usar_optimo:
            peso_nuevo = peso_optimo_recortado
            st.slider(
                "Peso a asignar", 0.0, 0.50, peso_nuevo, step=0.01, disabled=True,
                key=f"peso_fijo_{candidato_elegido}",
            )
        else:
            peso_nuevo = st.slider(
                "Peso a asignar", 0.0, 0.50, peso_optimo_recortado, step=0.01,
                key=f"peso_manual_{candidato_elegido}",
            )

        sim = simular_incorporacion(
            retornos, pesos_actuales, retornos_candidatos[candidato_elegido], peso_nuevo, rf=rf,
        )
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric(
            "Retorno anual", f"{sim['retorno_despues'] * 100:.2f} %",
            f"{(sim['retorno_despues'] - sim['retorno_antes']) * 100:+.2f} pp",
        )
        col_r2.metric(
            "Volatilidad anual", f"{sim['vol_despues'] * 100:.2f} %",
            f"{(sim['vol_despues'] - sim['vol_antes']) * 100:+.2f} pp",
            delta_color="inverse",
        )
        col_r3.metric(
            "Sharpe", f"{sim['sharpe_despues']:.2f}",
            f"{sim['sharpe_despues'] - sim['sharpe_antes']:+.2f}",
        )
        st.caption(
            f"Comparación sobre las **sesiones comunes** entre tu cartera y {candidato_elegido} "
            f"(calendarios distintos: bolsa europea vs EE.UU./cripto). Sobre esa muestra tu "
            f"cartera parte de un Sharpe de **{sim['sharpe_antes']:.2f}**, retorno "
            f"{sim['retorno_antes'] * 100:.2f} % y volatilidad {sim['vol_antes'] * 100:.2f} % — "
            "puede diferir ligeramente del Sharpe de cabecera, que usa todo el histórico. "
            "Lo válido aquí es la DIFERENCIA (antes → después), no el nivel absoluto."
        )
    except Exception as exc:
        st.error(f"No se pudo completar el análisis de recomendaciones: {exc}")
        st.code(traceback.format_exc())


# ------------------------------------------------------------------ Energía ---
with pestanas[6]:
    st.info(
        "Complejo energético (crudo y gas): volatilidad, diferenciales entre referencias "
        "(Brent-WTI, TTF/Henry Hub) y estacionalidad. Relevante para IDR.MC y MC.PA, "
        "expuestas a costes energéticos de la industria europea."
    )

    claves_energia = st.multiselect(
        "Referencias a analizar",
        list(FUTUROS_ENERGIA.keys()),
        default=["WTI", "BRENT", "GAS_US", "GAS_EU"],
        format_func=lambda c: f"{c} ({FUTUROS_ENERGIA[c]['ticker']})",
    )

    if not claves_energia:
        st.warning("Elige al menos una referencia energética.")
    else:
        try:
            energia = cargar_energia(tuple(claves_energia), anios)

            st.subheader("Evolución (base 100)")
            st.pyplot(plot_precios_normalizados(energia.precios))

            col_e1, col_e2 = st.columns(2)
            col_e1.subheader("Volatilidad histórica")
            col_e1.dataframe(energia.resumen_volatilidad, use_container_width=True)
            col_e2.subheader("Régimen actual")
            col_e2.dataframe(energia.regimen, use_container_width=True)

            if not energia.diferenciales.empty:
                st.subheader("Diferenciales entre referencias")
                st.latex(
                    r"\text{Brent-WTI} = P_{Brent} - P_{WTI} \qquad "
                    r"\text{TTF/HH} = \dfrac{P_{TTF}}{P_{HenryHub}}"
                )
                st.caption(
                    "Brent-WTI se resta porque ambos cotizan en \\$/barril (el diferencial "
                    "atlántico: transporte y calidad del crudo). TTF/Henry Hub se calcula como "
                    "RATIO, no como resta, porque cotizan en unidades y divisas distintas "
                    "(EUR/MWh vs USD/MMBtu) — restarlos no tendría significado físico."
                )
                st.pyplot(plot_diferenciales(energia.diferenciales))

            st.subheader("Estacionalidad de la volatilidad")
            st.caption(
                "Media histórica de la volatilidad rolling de 21 días, agrupada por mes del año. "
                "El gas natural tiene picos estructurales en invierno (riesgo de ola de frío) y "
                "en verano en EE.UU. (demanda eléctrica de aire acondicionado)."
            )
            st.pyplot(plot_estacionalidad(energia.estacionalidad))

            st.subheader("Diagnóstico")
            for alerta in energia.alertas:
                st.write(alerta)

            with st.expander("📜 Histórico largo (FRED, hasta 25 años) y peores episodios de volatilidad"):
                st.caption(
                    "yfinance solo da 3-5 años de futuros. FRED tiene el Brent y el WTI desde los "
                    "80 y el Henry Hub desde 1997 — suficiente para ver dónde queda la volatilidad "
                    "actual frente a 2008, la crisis de precios negativos de 2020 o 2022."
                )
                anios_largo = st.slider("Años de histórico (FRED)", 5, 25, 25, 5, key="anios_fred")
                if st.button("Cargar histórico largo"):
                    st.session_state["mostrar_historico_largo"] = True
                # Guardamos la intención en session_state, no solo en el resultado del
                # botón: un st.button solo es True en el rerun exacto del clic, así que
                # sin esto el gráfico desaparecería en cuanto cambiaras de pestaña o
                # tocaras cualquier otro control (el resto de la app reruns el script
                # entero). Los datos en sí ya están cacheados por @st.cache_data, así
                # que volver a pedirlos aquí es prácticamente instantáneo.
                if st.session_state.get("mostrar_historico_largo"):
                    try:
                        largo, crisis = cargar_energia_larga(float(anios_largo))
                        st.line_chart(largo)
                        st.subheader("Mayores episodios de volatilidad por año")
                        st.dataframe(crisis, use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.error(f"No se pudo descargar el histórico largo: {exc}")
        except Exception as exc:
            st.error(f"No se pudo completar el análisis energético: {exc}")
            st.code(traceback.format_exc())


# -------------------------------------------------------------- Gas Europa ---
with pestanas[7]:
    st.info(
        "Almacenamiento de gas europeo (GIE AGSI+): el % de llenado es la CAUSA del precio del "
        "TTF, no al revés. Requiere una clave de API gratuita — regístrate en "
        "[agsi.gie.eu](https://agsi.gie.eu/) (botón 'API' arriba a la derecha)."
    )

    clave_gie = os.environ.get("GIE_API_KEY") or st.session_state.get("gie_api_key", "")
    with st.expander("🔑 Clave de API de GIE", expanded=not clave_gie):
        entrada_clave = st.text_input(
            "GIE_API_KEY", value=clave_gie, type="password",
            help="Se usa solo en esta sesión del navegador; no se guarda en ningún archivo.",
        )
        if entrada_clave:
            st.session_state["gie_api_key"] = entrada_clave
        clave_gie = entrada_clave

    if not clave_gie:
        st.warning("Introduce tu clave de GIE arriba para ver el almacenamiento real.")
    else:
        paises_elegidos = st.multiselect(
            "Países", list(PAISES_GAS.keys()),
            default=["EU", "DE", "IT", "NL", "ES"],
            format_func=lambda p: f"{p} — {PAISES_GAS[p]['nombre']}",
        )
        pais_detalle = st.selectbox(
            "País para el detalle estacional", paises_elegidos or ["EU"],
            index=0,
        )

        if not paises_elegidos:
            st.warning("Elige al menos un país.")
        else:
            try:
                gas = cargar_gas_europa(tuple(paises_elegidos), anios, clave_gie)

                st.subheader("Estado actual")
                st.dataframe(gas.resumen, use_container_width=True)

                st.subheader("Evolución del llenado")
                st.pyplot(plot_llenado(gas.llenado))

                if pais_detalle in gas.llenado.columns:
                    st.subheader(f"Llenado vs patrón estacional — {PAISES_GAS.get(pais_detalle, {}).get('nombre', pais_detalle)}")
                    st.caption(
                        "La banda gris es el rango histórico para cada día del año; la línea roja, "
                        "el año en curso. Lo que importa no es el nivel absoluto, sino la desviación "
                        "frente a lo normal para esa fecha."
                    )
                    from m13_gas_europa import patron_estacional_llenado
                    patron = patron_estacional_llenado(gas.llenado[pais_detalle].dropna())
                    st.pyplot(plot_estacionalidad_llenado(gas.llenado[pais_detalle].dropna(), patron))

                if not gas.flujos.empty:
                    st.subheader("Flujos netos de almacenamiento")
                    st.caption("Positivo = Europa acumula reservas; negativo = las está consumiendo.")
                    st.pyplot(plot_flujos(gas.flujos))

                st.subheader("Diagnóstico")
                for alerta in gas.alertas:
                    st.write(alerta)
            except ErrorDeGIE as exc:
                st.error(f"GIE: {exc}")
            except Exception as exc:
                st.error(f"No se pudo completar el análisis de gas europeo: {exc}")
                st.code(traceback.format_exc())


st.divider()
st.caption(
    "Este panel es una herramienta de análisis cuantitativo, no asesoramiento financiero. "
    "Las estimaciones se basan en datos históricos, que no predicen resultados futuros."
)
