"""
===============================================================================
 MÓDULO 12 — FUENTES ALTERNATIVAS DE DATOS (FRED · EIA · Stooq)
===============================================================================
Resuelve dos problemas de yfinance:

  1) PROFUNDIDAD: Yahoo da 3-5 años cómodos de futuros. FRED tiene el Brent
     diario desde 1987 y el Henry Hub desde 1997. Para estudiar cómo se comportó
     la volatilidad en 2008 o en la crisis de 2022 hacen falta esos históricos.

  2) FIABILIDAD: Yahoo se cae, cambia su API sin avisar y aplica rate limiting.
     Una tarea automatizada que dependa de una sola fuente falla tarde o temprano.
     Este módulo implementa una CADENA DE RESPALDO: si Yahoo falla, prueba Stooq;
     si Stooq falla, prueba FRED.

-------------------------------------------------------------------------------
 LAS TRES FUENTES
-------------------------------------------------------------------------------
FRED (Reserva Federal de St. Louis)
  · Gratis, SIN clave de API (se usa el endpoint CSV público)
  · Históricos de décadas, calidad institucional
  · Limitación: son precios SPOT publicados con 1-4 días de retraso, no
    cotizaciones en tiempo real. Perfecto para análisis, inútil para operar.

EIA (Energy Information Administration, Dpto. de Energía de EE.UU.)
  · Gratis CON clave (registro instantáneo en eia.gov/opendata/register.php)
  · Lo que ninguna API financiera tiene: inventarios, producción, capacidad de
    refino, importaciones. Los FUNDAMENTALES que mueven el precio.
  · Limitación: centrado en EE.UU.; para el gas europeo hay que ir a GIE AGSI+.

Stooq
  · Gratis, sin clave, CSV directo
  · Mejor cobertura europea que Yahoo (incluidas acciones españolas)
  · Limitación: sin documentación oficial y el formato de símbolos es peculiar.

Uso rápido
----------
    from m12_fuentes import descargar_fred, catalogo_fred, descargar_con_respaldo

    print(catalogo_fred())
    brent = descargar_fred("DCOILBRENTEU", anios=25)     # Brent desde 2001
    precios = descargar_con_respaldo("SPY", anios=3)     # Yahoo → Stooq → FRED
===============================================================================
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger("quant.fuentes")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

TIEMPO_ESPERA = 30  # segundos
TIEMPO_ESPERA_REQUESTS = 12  # más corto: si va a fallar, que falle rápido y probemos curl


class ErrorDeFuente(Exception):
    """Se lanza cuando una fuente de datos no responde o devuelve datos inválidos."""


def _get_texto(url: str, headers: dict[str, str] | None = None) -> str:
    """GET que devuelve el cuerpo como texto, con fallback a `curl`.

    Algunas fuentes (FRED en particular, tras un WAF tipo Akamai) responden con
    normalidad a `curl` pero cuelgan en silencio ante `requests`/urllib3: el
    handshake se completa pero el cuerpo nunca llega, hasta agotar el timeout.
    Es un fingerprint TLS/HTTP2, no un problema de red ni de credenciales — así
    que en vez de fallar, probamos `curl` (disponible en macOS/Linux) antes de
    rendirnos. Si tampoco hay curl, se propaga el error original de requests.
    """
    try:
        respuesta = requests.get(url, timeout=TIEMPO_ESPERA_REQUESTS, headers=headers or {})
        respuesta.raise_for_status()
        return respuesta.text
    except requests.RequestException as exc_requests:
        try:
            resultado = subprocess.run(
                ["curl", "-sS", "-L", "-m", str(TIEMPO_ESPERA), url],
                capture_output=True, text=True, timeout=TIEMPO_ESPERA + 5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            raise exc_requests  # no hay curl disponible: el error de requests es el único dato que tenemos
        if resultado.returncode != 0 or not resultado.stdout:
            raise exc_requests
        logger.info("'%s' obtenido vía curl (requests se quedó colgado)", url.split("?")[0])
        return resultado.stdout


# =============================================================================
# CATÁLOGO DE SERIES DE FRED
# =============================================================================

SERIES_FRED: dict[str, dict[str, str]] = {
    # --- Energía: precios diarios -------------------------------------------
    "DCOILBRENTEU": {"nombre": "Brent (Europa), spot diario", "desde": "1987",
                     "frecuencia": "Diaria", "unidad": "USD/barril"},
    "DCOILWTICO": {"nombre": "WTI (Cushing), spot diario", "desde": "1986",
                   "frecuencia": "Diaria", "unidad": "USD/barril"},
    "DHHNGSP": {"nombre": "Henry Hub gas natural, spot diario", "desde": "1997",
                "frecuencia": "Diaria", "unidad": "USD/MMBtu"},
    # --- Energía: series mensuales (históricos aún más largos) ---------------
    "POILBREUSDM": {"nombre": "Brent, precio global mensual", "desde": "1990",
                    "frecuencia": "Mensual", "unidad": "USD/barril"},
    "PNGASEUUSDM": {"nombre": "Gas natural EUROPA, precio global mensual",
                    "desde": "1990", "frecuencia": "Mensual", "unidad": "USD/MMBtu"},
    "PNGASUSUSDM": {"nombre": "Gas natural EE.UU., precio global mensual",
                    "desde": "1990", "frecuencia": "Mensual", "unidad": "USD/MMBtu"},
    "GASREGW": {"nombre": "Gasolina regular EE.UU., media semanal", "desde": "1990",
                "frecuencia": "Semanal", "unidad": "USD/galón"},
    # --- Macro relevante para la energía y tu cartera ------------------------
    "CPIENGSL": {"nombre": "IPC de energía EE.UU.", "desde": "1957",
                 "frecuencia": "Mensual", "unidad": "Índice"},
    "T10YIE": {"nombre": "Inflación implícita a 10 años (breakeven)", "desde": "2003",
               "frecuencia": "Diaria", "unidad": "%"},
    "DTWEXBGS": {"nombre": "Índice del dólar (amplio)", "desde": "2006",
                 "frecuencia": "Diaria", "unidad": "Índice"},
    "DGS10": {"nombre": "Bono EE.UU. 10 años", "desde": "1962",
              "frecuencia": "Diaria", "unidad": "%"},
    "DFF": {"nombre": "Tipo de los fondos federales (Rf real)", "desde": "1954",
            "frecuencia": "Diaria", "unidad": "%"},
    "VIXCLS": {"nombre": "VIX (cierre)", "desde": "1990",
               "frecuencia": "Diaria", "unidad": "%"},
    "SP500": {"nombre": "S&P 500", "desde": "2015 (solo 10 años)",
              "frecuencia": "Diaria", "unidad": "Índice"},
}


def catalogo_fred() -> pd.DataFrame:
    """Series de FRED preconfiguradas, con su histórico disponible."""
    return pd.DataFrame(SERIES_FRED).T.rename_axis("Serie")


# =============================================================================
# FRED — DESCARGA (sin clave de API)
# =============================================================================

def _parsear_csv_fred(texto: str, serie_id: str) -> pd.Series:
    """Convierte el CSV de fredgraph en una Serie limpia.

    Separado de la descarga a propósito: así la lógica de parseo se puede
    probar sin red, que es donde se esconden los errores de formato.

    FRED marca los días sin dato con un punto ('.') — festivos, fines de semana
    o simplemente publicación pendiente. Hay que convertirlos a NaN y eliminarlos,
    o contaminarían todos los cálculos posteriores.
    """
    if not texto or not texto.strip():
        raise ErrorDeFuente(f"FRED devolvió una respuesta vacía para '{serie_id}'.")

    try:
        datos = pd.read_csv(io.StringIO(texto))
    except Exception as exc:
        raise ErrorDeFuente(f"No se pudo parsear el CSV de FRED para '{serie_id}': {exc}") from exc

    if datos.shape[1] < 2:
        raise ErrorDeFuente(
            f"El CSV de FRED para '{serie_id}' no tiene el formato esperado. "
            f"Columnas recibidas: {list(datos.columns)}. ¿Existe esa serie?"
        )

    # La primera columna es la fecha ('DATE' u 'observation_date' según versión)
    columna_fecha, columna_valor = datos.columns[0], datos.columns[1]
    datos[columna_fecha] = pd.to_datetime(datos[columna_fecha], errors="coerce")
    # '.' = dato no disponible en la convención de FRED
    datos[columna_valor] = pd.to_numeric(
        datos[columna_valor].astype(str).str.strip().replace(".", pd.NA), errors="coerce",
    )

    serie = (
        datos.dropna(subset=[columna_fecha])
        .set_index(columna_fecha)[columna_valor]
        .dropna()
        .sort_index()
    )
    serie.name = serie_id

    if serie.empty:
        raise ErrorDeFuente(f"La serie '{serie_id}' de FRED no contiene datos válidos.")
    return serie


def descargar_fred(
    series: str | list[str],
    anios: float | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
) -> pd.DataFrame:
    """Descarga una o varias series de FRED. NO necesita clave de API.

    Usa el endpoint público `fredgraph.csv`, el mismo que alimenta los gráficos
    de la web de FRED. Es estable y no tiene cuota documentada, pero conviene no
    abusar: para descargas masivas, regístrate y usa la API oficial.

    Parameters
    ----------
    series : ID(s) de FRED (ver `catalogo_fred()` o buscar en fred.stlouisfed.org)
    anios : profundidad del histórico. None = todo el histórico disponible.

    Returns
    -------
    DataFrame con índice de fechas y una columna por serie.
    """
    if isinstance(series, str):
        series = [series]

    if fecha_inicio is None and anios is not None:
        fecha_inicio = (datetime.today() - timedelta(days=int(anios * 365.25))).strftime("%Y-%m-%d")
    fecha_fin = fecha_fin or datetime.today().strftime("%Y-%m-%d")

    resultados: dict[str, pd.Series] = {}
    fallidas: dict[str, str] = {}

    for serie_id in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie_id}"
        if fecha_inicio:
            url += f"&cosd={fecha_inicio}&coed={fecha_fin}"

        try:
            texto = _get_texto(url, headers={"User-Agent": "Mozilla/5.0"})
            resultados[serie_id] = _parsear_csv_fred(texto, serie_id)
            logger.info(
                "FRED %s: %d observaciones (%s → %s)",
                serie_id, len(resultados[serie_id]),
                resultados[serie_id].index[0].date(), resultados[serie_id].index[-1].date(),
            )
        except requests.RequestException as exc:
            fallidas[serie_id] = f"Error de red: {exc}"
            logger.warning("FRED %s falló: %s", serie_id, exc)
        except ErrorDeFuente as exc:
            fallidas[serie_id] = str(exc)
            logger.warning("FRED %s: %s", serie_id, exc)

    if not resultados:
        raise ErrorDeFuente(
            f"No se pudo descargar ninguna serie de FRED. Errores: {fallidas}"
        )

    return pd.DataFrame(resultados).sort_index()


# =============================================================================
# EIA — DESCARGA (requiere clave gratuita)
# =============================================================================

RUTAS_EIA: dict[str, dict[str, str]] = {
    "precios_spot_petroleo": {
        "ruta": "petroleum/pri/spt/data",
        "descripcion": "Precios spot de crudo y derivados (WTI, Brent, gasolina, diésel)",
    },
    "inventarios_petroleo": {
        "ruta": "petroleum/stoc/wstk/data",
        "descripcion": "Inventarios semanales de crudo. El dato que mueve el precio los miércoles.",
    },
    "inventarios_gas": {
        "ruta": "natural-gas/stor/wkly/data",
        "descripcion": "Inventarios semanales de gas natural. Publicación de los jueves.",
    },
    "produccion_petroleo": {
        "ruta": "petroleum/crd/crpdn/data",
        "descripcion": "Producción de crudo en EE.UU. por estado y mes.",
    },
    "precios_gas": {
        "ruta": "natural-gas/pri/sum/data",
        "descripcion": "Precios del gas natural por sector de consumo.",
    },
}


def catalogo_eia() -> pd.DataFrame:
    """Rutas de la API v2 de EIA preconfiguradas."""
    return pd.DataFrame(RUTAS_EIA).T.rename_axis("Clave")


def _parsear_json_eia(carga: dict) -> pd.DataFrame:
    """Extrae el DataFrame de la respuesta JSON de la API v2 de EIA.

    La estructura es {'response': {'data': [...], 'total': N}}. Si la clave es
    inválida, EIA devuelve un 200 con un objeto 'error' dentro, no un código
    HTTP de error — por eso hay que comprobarlo explícitamente.
    """
    if "error" in carga:
        raise ErrorDeFuente(f"EIA devolvió un error: {carga['error']}")

    respuesta = carga.get("response", {})
    datos = respuesta.get("data", [])

    if not datos:
        raise ErrorDeFuente(
            "La consulta a EIA no devolvió datos. Revisa la ruta y los filtros; "
            "las rutas de la API v2 son muy específicas."
        )

    tabla = pd.DataFrame(datos)
    if "period" in tabla.columns:
        tabla["period"] = pd.to_datetime(tabla["period"], errors="coerce")
        tabla = tabla.set_index("period").sort_index()
    if "value" in tabla.columns:
        tabla["value"] = pd.to_numeric(tabla["value"], errors="coerce")

    return tabla


def descargar_eia(
    ruta: str,
    api_key: str | None = None,
    frecuencia: str = "daily",
    longitud: int = 5000,
    parametros_extra: dict | None = None,
) -> pd.DataFrame:
    """Descarga datos de la API v2 de EIA.

    La clave se lee de la variable de entorno EIA_API_KEY si no se pasa. Es
    gratuita e instantánea: eia.gov/opendata/register.php

    Parameters
    ----------
    ruta : ruta de la API (ver `catalogo_eia()`), p.ej. "petroleum/pri/spt/data"
    frecuencia : 'daily', 'weekly', 'monthly', 'annual' — debe existir para esa ruta.
    longitud : máximo de filas (tope de la API: 5000 por petición).
    """
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ErrorDeFuente(
            "Falta la clave de EIA. Consíguela gratis en "
            "https://www.eia.gov/opendata/register.php y luego:\n"
            '  export EIA_API_KEY="tu_clave"\n'
            "O pásala como argumento: descargar_eia(ruta, api_key='...')"
        )

    ruta = RUTAS_EIA.get(ruta, {}).get("ruta", ruta)  # acepta clave del catálogo o ruta directa
    url = f"https://api.eia.gov/v2/{ruta}"

    parametros = {
        "api_key": api_key,
        "frequency": frecuencia,
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": longitud,
    }
    if parametros_extra:
        parametros.update(parametros_extra)

    try:
        respuesta = requests.get(url, params=parametros, timeout=TIEMPO_ESPERA)
        respuesta.raise_for_status()
        carga = respuesta.json()
    except requests.RequestException as exc:
        raise ErrorDeFuente(f"Error de red al consultar EIA: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ErrorDeFuente(f"EIA devolvió una respuesta no válida: {exc}") from exc

    tabla = _parsear_json_eia(carga)
    logger.info("EIA %s: %d filas descargadas", ruta, len(tabla))
    return tabla


# =============================================================================
# STOOQ — DESCARGA (sin clave, buena cobertura europea)
# =============================================================================

def _parsear_csv_stooq(texto: str, simbolo: str) -> pd.DataFrame:
    """Convierte el CSV de Stooq en un DataFrame OHLCV.

    Cuando el símbolo no existe, Stooq no devuelve un error HTTP: devuelve el
    texto 'No data' con código 200. Hay que detectarlo a mano.
    """
    if not texto or "No data" in texto[:200]:
        raise ErrorDeFuente(
            f"Stooq no tiene datos para '{simbolo}'. Sus símbolos usan un formato "
            "propio: acciones EE.UU. → 'spy.us', alemanas → 'rhm.de', "
            "españolas → 'san.es'. Compruébalo en stooq.com."
        )

    try:
        datos = pd.read_csv(io.StringIO(texto))
    except Exception as exc:
        raise ErrorDeFuente(f"No se pudo parsear el CSV de Stooq: {exc}") from exc

    if "Date" not in datos.columns or "Close" not in datos.columns:
        raise ErrorDeFuente(
            f"Formato inesperado de Stooq para '{simbolo}'. Columnas: {list(datos.columns)}"
        )

    datos["Date"] = pd.to_datetime(datos["Date"], errors="coerce")
    datos = datos.dropna(subset=["Date"]).set_index("Date").sort_index()

    if datos.empty:
        raise ErrorDeFuente(f"Stooq devolvió un histórico vacío para '{simbolo}'.")
    return datos


def descargar_stooq(simbolo: str, anios: float | None = None) -> pd.DataFrame:
    """Descarga el histórico OHLCV de Stooq. Sin clave, sin registro.

    Formato de símbolos (no es el mismo que Yahoo):
        EE.UU.:   spy.us, aapl.us
        Alemania: rhm.de
        España:   san.es
        Índices:  ^spx, ^dax
        Divisas:  eurusd

    Útil sobre todo como RESPALDO de Yahoo y para valores europeos que Yahoo
    cubre mal.
    """
    url = f"https://stooq.com/q/d/l/?s={simbolo.lower()}&i=d"

    try:
        respuesta = requests.get(url, timeout=TIEMPO_ESPERA,
                                 headers={"User-Agent": "Mozilla/5.0"})
        respuesta.raise_for_status()
    except requests.RequestException as exc:
        raise ErrorDeFuente(f"Error de red al consultar Stooq: {exc}") from exc

    datos = _parsear_csv_stooq(respuesta.text, simbolo)

    if anios is not None:
        corte = pd.Timestamp.today() - pd.Timedelta(days=int(anios * 365.25))
        datos = datos[datos.index >= corte]

    logger.info("Stooq %s: %d sesiones", simbolo, len(datos))
    return datos


# =============================================================================
# CADENA DE RESPALDO
# =============================================================================

# Traducción de tickers entre fuentes: cada una usa su propio formato
EQUIVALENCIAS: dict[str, dict[str, str]] = {
    "SPY":    {"stooq": "spy.us", "fred": "SP500"},
    "CL=F":   {"stooq": "cl.f", "fred": "DCOILWTICO"},
    "BZ=F":   {"stooq": "cb.f", "fred": "DCOILBRENTEU"},
    "NG=F":   {"stooq": "ng.f", "fred": "DHHNGSP"},
    "^VIX":   {"stooq": "^vix", "fred": "VIXCLS"},
    "RHM.DE": {"stooq": "rhm.de", "fred": None},
    "IDR.MC": {"stooq": "idr.es", "fred": None},
    "MC.PA":  {"stooq": "mc.fr", "fred": None},
}


def descargar_con_respaldo(
    ticker: str,
    anios: float = 3.0,
    orden: tuple[str, ...] = ("yahoo", "stooq", "fred"),
) -> pd.Series:
    """Intenta descargar el precio de cierre probando varias fuentes en orden.

    Existe porque en una tarea automatizada que corre sin supervisión, una caída
    de Yahoo no debería significar quedarse sin informe. Devuelve la primera
    fuente que responda con datos válidos y registra cuál fue.

    Returns
    -------
    pd.Series de precios de cierre. El atributo `.attrs['fuente']` indica de
    dónde salieron los datos.
    """
    errores: dict[str, str] = {}
    equivalencia = EQUIVALENCIAS.get(ticker.upper(), {})

    for fuente in orden:
        try:
            if fuente == "yahoo":
                from m1_datos import descargar_precios
                precios = descargar_precios([ticker], anios=anios)
                serie = precios[precios.columns[0]]

            elif fuente == "stooq":
                simbolo = equivalencia.get("stooq")
                if not simbolo:
                    errores["stooq"] = f"Sin equivalencia de Stooq para '{ticker}'"
                    continue
                serie = descargar_stooq(simbolo, anios=anios)["Close"]

            elif fuente == "fred":
                serie_fred = equivalencia.get("fred")
                if not serie_fred:
                    errores["fred"] = f"Sin equivalencia de FRED para '{ticker}'"
                    continue
                serie = descargar_fred(serie_fred, anios=anios).iloc[:, 0]

            else:
                errores[fuente] = "Fuente desconocida"
                continue

            if serie is None or serie.empty:
                errores[fuente] = "Devolvió una serie vacía"
                continue

            serie = serie.dropna()
            serie.name = ticker
            serie.attrs["fuente"] = fuente
            logger.info("'%s' obtenido de %s (%d observaciones)", ticker, fuente, len(serie))
            return serie

        except Exception as exc:
            errores[fuente] = f"{type(exc).__name__}: {str(exc)[:100]}"
            logger.warning("Fuente '%s' falló para '%s': %s", fuente, ticker, exc)

    raise ErrorDeFuente(
        f"Ninguna fuente pudo entregar datos de '{ticker}'. Detalle: {errores}"
    )


# =============================================================================
# ANÁLISIS ENERGÉTICO DE LARGO PLAZO (lo que yfinance no permite)
# =============================================================================

def energia_historica_larga(anios: float = 25.0) -> pd.DataFrame:
    """Brent, WTI y Henry Hub desde FRED, con décadas de histórico.

    Con esto sí se puede contestar a preguntas que 3 años de Yahoo no permiten:
    ¿cuál fue la volatilidad del crudo en 2008? ¿en la caída a precios negativos
    de abril de 2020? ¿en la crisis energética de 2022?
    """
    datos = descargar_fred(["DCOILBRENTEU", "DCOILWTICO", "DHHNGSP"], anios=anios)
    datos = datos.rename(columns={
        "DCOILBRENTEU": "BRENT", "DCOILWTICO": "WTI", "DHHNGSP": "GAS_US",
    })
    return datos.ffill().dropna(how="all")


def crisis_energeticas_historicas(precios: pd.DataFrame, ventana: int = 60) -> pd.DataFrame:
    """Identifica los episodios de mayor volatilidad del histórico largo.

    Devuelve, para cada activo, los 5 periodos con la volatilidad rolling más
    alta, agrupados por año. Sirve para poner en contexto la volatilidad actual:
    saber que el crudo está al 35% significa poco hasta que ves que en 2020
    llegó al 200%.
    """
    from m5_volatilidad_historica import volatilidad_rolling

    retornos = precios.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
    vol = volatilidad_rolling(retornos, ventana)

    filas = []
    for activo in vol.columns:
        serie = vol[activo].dropna()
        if serie.empty:
            continue
        por_anio = serie.groupby(serie.index.year).max().sort_values(ascending=False)
        for anio, valor in por_anio.head(5).items():
            filas.append({
                "Activo": activo, "Año": int(anio),
                "Vol. máxima %": round(valor * 100, 1),
            })

    if not filas:
        return pd.DataFrame(columns=["Activo", "Año", "Vol. máxima %"])
    return pd.DataFrame(filas).sort_values(["Activo", "Vol. máxima %"], ascending=[True, False])


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m12_fuentes.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 25)

    print("\n--- Catálogo de series de FRED ---")
    print(catalogo_fred().to_string())
    print("\n--- Rutas de EIA ---")
    print(catalogo_eia().to_string())

    try:
        print("\n--- Energía histórica larga (FRED, 25 años) ---")
        largo = energia_historica_larga(anios=25)
        print(f"Periodo: {largo.index[0].date()} → {largo.index[-1].date()} "
              f"({len(largo)} observaciones)")
        print(largo.tail().round(2).to_string())

        print("\n--- Mayores episodios de volatilidad de la historia reciente ---")
        print(crisis_energeticas_historicas(largo).to_string(index=False))

        print("\n--- Cadena de respaldo ---")
        serie = descargar_con_respaldo("BZ=F", anios=3)
        print(f"Brent obtenido de: {serie.attrs.get('fuente')} · {len(serie)} observaciones")

        print("\n[OK] Módulo 12 ejecutado.")

    except ErrorDeFuente as exc:
        print(f"\n[ERROR DE FUENTE] {exc}")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
