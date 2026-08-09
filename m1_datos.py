"""
===============================================================================
 MÓDULO 1 — DESCARGA Y PREPARACIÓN DE DATOS
===============================================================================
Responsabilidad única: obtener precios de cierre AJUSTADOS, validarlos y
convertirlos en una matriz de retornos limpia y utilizable por el resto de
módulos (correlaciones, optimización, gráficos).

Diseño defensivo: yfinance cambia su API con frecuencia (MultiIndex de
columnas, `auto_adjust` por defecto, tickers inexistentes que devuelven NaN...).
Este módulo absorbe esas diferencias para que los módulos posteriores reciban
siempre el mismo contrato: un DataFrame de precios indexado por fecha,
una columna por ticker, sin NaN.

Uso rápido
----------
    from m1_datos import descargar_precios, calcular_retornos, resumen_activos

    precios  = descargar_precios(["SPY", "IEV", "TLT", "GLD", "USO"], anios=3)
    retornos = calcular_retornos(precios)
    print(resumen_activos(precios, retornos))

Requisitos: pip install yfinance pandas numpy
===============================================================================
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

# --- Dependencia externa opcional: se importa con mensaje de error claro -----
try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Falta la librería 'yfinance'. Instálala con:  pip install yfinance"
    ) from exc


# =============================================================================
# CONSTANTES GLOBALES DEL PROYECTO
# =============================================================================

TICKERS_DEFECTO: tuple[str, ...] = ("SPY", "IEV", "TLT", "GLD", "USO")
"""Cartera multiactivo por defecto:
   SPY = Renta variable EE.UU. | IEV = Renta variable Europa | TLT = Bonos 20y+
   GLD = Oro (refugio)         | USO = Petróleo (materia prima / inflación)"""

DIAS_HABILES_ANIO: int = 252   # Días de cotización al año (anualización)
TASA_LIBRE_RIESGO: float = 0.04  # Rf anual = 4.0% (usada en CAPM/Sharpe, módulo 3)

# Logger propio del módulo (no contamina el root logger del usuario)
logger = logging.getLogger("quant.datos")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =============================================================================
# EXCEPCIONES PROPIAS
# =============================================================================

class ErrorDeDatos(Exception):
    """Se lanza cuando los datos descargados no son utilizables."""


# =============================================================================
# FUNCIONES AUXILIARES (privadas)
# =============================================================================

def _normalizar_tickers(tickers: Sequence[str] | str) -> list[str]:
    """Acepta 'SPY', 'SPY GLD', 'SPY,GLD' o una lista, y devuelve lista limpia.

    Elimina duplicados preservando el orden y pasa todo a mayúsculas.
    """
    if isinstance(tickers, str):
        tickers = tickers.replace(",", " ").split()

    limpios: list[str] = []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in limpios:
            limpios.append(t)

    if not limpios:
        raise ValueError("La lista de tickers está vacía.")
    return limpios


def _extraer_cierres(bruto: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extrae la columna de cierre ajustado sea cual sea el formato de yfinance.

    yfinance puede devolver:
      * Columnas planas: ['Open','High','Low','Close','Volume']  (1 ticker)
      * MultiIndex (campo, ticker): ('Close','SPY'), ('Close','GLD'), ...
      * MultiIndex (ticker, campo): ('SPY','Close'), ...  (si group_by='ticker')
    """
    if bruto is None or bruto.empty:
        raise ErrorDeDatos(
            "yfinance devolvió un DataFrame vacío. Causas habituales: (a) sin conexión "
            "a internet o firewall/proxy bloqueando query1.finance.yahoo.com; "
            f"(b) tickers inexistentes en Yahoo Finance {tickers}; "
            "(c) rango de fechas sin sesiones bursátiles."
        )

    columnas = bruto.columns

    # --- Caso A: columnas planas -------------------------------------------
    if not isinstance(columnas, pd.MultiIndex):
        for campo in ("Adj Close", "Close"):
            if campo in columnas:
                serie = bruto[campo].copy()
                df = serie.to_frame(tickers[0]) if isinstance(serie, pd.Series) else serie
                return df
        raise ErrorDeDatos(f"No se encontró columna de cierre. Columnas: {list(columnas)}")

    # --- Caso B: MultiIndex -------------------------------------------------
    nivel0 = set(columnas.get_level_values(0))
    nivel1 = set(columnas.get_level_values(1))

    for campo in ("Adj Close", "Close"):
        if campo in nivel0:                       # formato (campo, ticker)
            return bruto[campo].copy()
        if campo in nivel1:                       # formato (ticker, campo)
            return bruto.xs(campo, axis=1, level=1).copy()

    raise ErrorDeDatos(f"No se encontró 'Close' en el MultiIndex: {columnas[:5].tolist()}")


def _validar_calidad(
    precios: pd.DataFrame,
    umbral_cobertura: float,
) -> pd.DataFrame:
    """Descarta tickers con demasiados huecos y avisa de los problemas."""
    if precios.empty:
        raise ErrorDeDatos("No se descargó ninguna fila de precios.")

    cobertura = precios.notna().mean()  # % de sesiones con dato, por ticker
    validos = cobertura[cobertura >= umbral_cobertura].index.tolist()
    descartados = [t for t in precios.columns if t not in validos]

    for t in descartados:
        logger.warning(
            "Ticker '%s' descartado: solo %.1f%% de datos válidos (mínimo %.0f%%). "
            "¿Ticker mal escrito, deslistado o con histórico más corto?",
            t, cobertura[t] * 100, umbral_cobertura * 100,
        )

    if not validos:
        raise ErrorDeDatos(
            "Ningún ticker superó el control de calidad. Revisa los símbolos "
            "(deben ser los de Yahoo Finance, p.ej. 'SAN.MC' para Santander)."
        )
    if len(validos) < 2:
        raise ErrorDeDatos(
            f"Solo sobrevivió el ticker {validos}. La teoría de carteras exige "
            "al menos 2 activos para diversificar."
        )

    return precios[validos]


# =============================================================================
# API PÚBLICA
# =============================================================================

def descargar_precios(
    tickers: Sequence[str] | str = TICKERS_DEFECTO,
    anios: float = 3.0,
    fecha_fin: str | datetime | None = None,
    umbral_cobertura: float = 0.90,
    cache_dir: str | Path | None = None,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """Descarga precios de cierre AJUSTADOS diarios y devuelve una matriz limpia.

    Parameters
    ----------
    tickers : lista de símbolos de Yahoo Finance (o string separado por espacios/comas).
    anios : profundidad del histórico en años (3.0 por defecto, según especificación).
    fecha_fin : fecha final del histórico; None = hoy.
    umbral_cobertura : fracción mínima de sesiones con dato para aceptar un ticker.
    cache_dir : si se indica, guarda/lee un CSV para no volver a llamar a la API.
    forzar_descarga : ignora la caché y vuelve a descargar.

    Returns
    -------
    pd.DataFrame
        Índice = fechas (DatetimeIndex), columnas = tickers, valores = precio
        ajustado. Sin NaN y ordenado cronológicamente.

    Raises
    ------
    ErrorDeDatos : si la descarga falla o los datos no son utilizables.
    """
    tickers = _normalizar_tickers(tickers)

    fin = pd.Timestamp(fecha_fin) if fecha_fin is not None else pd.Timestamp.today().normalize()
    inicio = fin - timedelta(days=int(round(anios * 365.25)))

    # ---------------- 1. Caché en disco (opcional) --------------------------
    ruta_cache: Path | None = None
    if cache_dir is not None:
        ruta_cache = Path(cache_dir) / f"precios_{'_'.join(tickers)}_{inicio:%Y%m%d}_{fin:%Y%m%d}.csv"
        if ruta_cache.exists() and not forzar_descarga:
            logger.info("Leyendo precios desde caché: %s", ruta_cache)
            cacheado = pd.read_csv(ruta_cache, index_col=0, parse_dates=True)
            return cacheado.sort_index()

    # ---------------- 2. Descarga -------------------------------------------
    logger.info("Descargando %s | %s → %s", tickers, inicio.date(), fin.date())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bruto = yf.download(
                tickers=tickers,
                start=inicio.strftime("%Y-%m-%d"),
                end=fin.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,     # 'Close' ya viene ajustado por dividendos/splits
                progress=False,
                threads=True,
                group_by="column",
            )
    except Exception as exc:
        raise ErrorDeDatos(
            f"Fallo de red o de la API de Yahoo Finance: {exc}. "
            "Comprueba tu conexión o reintenta en unos segundos."
        ) from exc

    # ---------------- 3. Extracción, validación y limpieza ------------------
    precios = _extraer_cierres(bruto, tickers)

    # Si solo se pidió 1 ticker y la columna quedó sin nombre útil
    if precios.shape[1] == 1 and precios.columns[0] not in tickers:
        precios.columns = [tickers[0]]

    precios = precios.reindex(columns=[t for t in tickers if t in precios.columns])
    precios = precios.sort_index()
    precios = precios.replace([np.inf, -np.inf], np.nan)
    precios[precios <= 0] = np.nan          # un precio <= 0 es un error de datos

    precios = _validar_calidad(precios, umbral_cobertura)

    # Huecos puntuales (festivos locales de un mercado) -> arrastrar último precio
    precios = precios.ffill().dropna(how="any")

    if len(precios) < 60:
        raise ErrorDeDatos(
            f"Solo hay {len(precios)} sesiones comunes. Insuficiente para estimar "
            "covarianzas de forma fiable (mínimo recomendado: 250)."
        )

    logger.info(
        "OK · %d sesiones · %d activos: %s",
        len(precios), precios.shape[1], list(precios.columns),
    )

    # ---------------- 4. Guardar caché --------------------------------------
    if ruta_cache is not None:
        ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        precios.to_csv(ruta_cache)
        logger.info("Caché guardada en %s", ruta_cache)

    return precios


def calcular_retornos(
    precios: pd.DataFrame,
    metodo: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Convierte precios en retornos diarios.

    'simple' = P_t/P_{t-1} - 1  → correcto para agregar EN CARTERA (usa este para
                                  optimizar pesos: el retorno de la cartera es la
                                  suma ponderada de los retornos simples).
    'log'    = ln(P_t/P_{t-1}) → correcto para agregar EN EL TIEMPO (Kelly, CAGR).
    """
    if not isinstance(precios, pd.DataFrame) or precios.empty:
        raise ErrorDeDatos("Se esperaba un DataFrame de precios no vacío.")

    if metodo == "simple":
        ret = precios.pct_change()
    elif metodo == "log":
        ret = np.log(precios / precios.shift(1))
    else:
        raise ValueError("metodo debe ser 'simple' o 'log'.")

    ret = ret.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if ret.empty:
        raise ErrorDeDatos("La matriz de retornos quedó vacía tras la limpieza.")
    return ret


def resumen_activos(
    precios: pd.DataFrame,
    retornos: pd.DataFrame | None = None,
    rf: float = TASA_LIBRE_RIESGO,
) -> pd.DataFrame:
    """Ficha estadística por activo: rentabilidad, riesgo, Sharpe y drawdown.

    Sirve como control de sanidad ANTES de optimizar: si un activo muestra una
    volatilidad absurda (>100%) o un retorno imposible, casi siempre es un
    problema de datos, no una oportunidad de inversión.
    """
    if retornos is None:
        retornos = calcular_retornos(precios)

    n = len(retornos)
    anios_efectivos = n / DIAS_HABILES_ANIO

    # CAGR real, calculado sobre precios (no sobre la media de retornos)
    total = precios.iloc[-1] / precios.iloc[0]
    cagr = total ** (1 / anios_efectivos) - 1

    media_anual = retornos.mean() * DIAS_HABILES_ANIO
    vol_anual = retornos.std(ddof=1) * np.sqrt(DIAS_HABILES_ANIO)
    sharpe = (media_anual - rf) / vol_anual

    # Máxima caída desde máximos (Max Drawdown)
    curva = precios / precios.iloc[0]
    max_dd = (curva / curva.cummax() - 1).min()

    resumen = pd.DataFrame({
        "Precio inicio": precios.iloc[0],
        "Precio fin": precios.iloc[-1],
        "Rent. total %": (total - 1) * 100,
        "CAGR %": cagr * 100,
        "Rent. anual media %": media_anual * 100,
        "Volatilidad anual %": vol_anual * 100,
        f"Sharpe (Rf={rf:.1%})": sharpe,
        "Max Drawdown %": max_dd * 100,
        "Asimetría": retornos.skew(),
        "Curtosis": retornos.kurtosis(),
    })
    return resumen.round(3)


def matriz_covarianzas(
    retornos: pd.DataFrame,
    anualizada: bool = True,
) -> pd.DataFrame:
    """Matriz de covarianzas (Σ), pieza central de Markowitz.

    σ²_cartera = wᵀ · Σ · w
    """
    cov = retornos.cov(ddof=1)
    return cov * DIAS_HABILES_ANIO if anualizada else cov


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m1_datos.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    try:
        precios = descargar_precios(TICKERS_DEFECTO, anios=3, cache_dir="./cache_datos")
        retornos = calcular_retornos(precios)

        print("\n" + "=" * 78)
        print("MÓDULO 1 · DATOS DESCARGADOS")
        print("=" * 78)
        print(f"Periodo    : {precios.index[0]:%Y-%m-%d} → {precios.index[-1]:%Y-%m-%d}")
        print(f"Sesiones   : {len(precios)}   |   Activos: {list(precios.columns)}")
        print("\n--- Últimos 5 precios ajustados ---")
        print(precios.tail().round(2))
        print("\n--- Ficha estadística por activo ---")
        print(resumen_activos(precios, retornos))
        print("\n--- Matriz de covarianzas anualizada (Σ) ---")
        print(matriz_covarianzas(retornos).round(5))
        print("\n[OK] Módulo 1 ejecutado sin errores.")

    except ErrorDeDatos as exc:
        print(f"\n[ERROR DE DATOS] {exc}")
    except Exception as exc:  # red de seguridad final
        print(f"\n[ERROR INESPERADO] {type(exc).__name__}: {exc}")
