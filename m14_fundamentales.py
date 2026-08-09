"""
===============================================================================
 MÓDULO 14 — ANÁLISIS FUNDAMENTAL (Financial Modeling Prep)
===============================================================================
Cierra el único hueco real del proyecto: hasta ahora todo era precio, riesgo y
correlación. Nada respondía a "¿es buena empresa?".

La optimización de Markowitz no distingue entre una empresa sólida y una que
está a punto de quebrar: solo ve la media y la varianza de sus retornos
históricos. Este módulo aporta la capa que falta, para filtrar candidatos ANTES
de meterlos en la frontera eficiente.

-------------------------------------------------------------------------------
 ⚠ LIMITACIÓN CRÍTICA DEL PLAN GRATUITO
-------------------------------------------------------------------------------
El plan gratuito de FMP (250 peticiones/día) cubre SOLO EMPRESAS
ESTADOUNIDENSES, con hasta 5 años de estados anuales.

Para tu cartera esto significa:
    RHM.DE, IDR.MC, MC.PA     → NO cubiertos en el plan gratuito
    0P0001KGI5.F (fondo)      → nunca cubierto (los fondos no publican balances)
    EUNL.DE (ETF)             → nunca cubierto (un ETF no tiene estados financieros)

O sea: en gratuito te sirve para ESTUDIAR CANDIDATOS estadounidenses, no para
analizar tu cartera actual. Es honesto saberlo antes de invertir tiempo.

Alternativas para valores europeos:
  · Plan de pago de FMP (cubre internacional)
  · Los informes anuales de la propia empresa (rheinmetall.com/investor-relations)
  · SimplyWall.st o Morningstar (web, no API)

-------------------------------------------------------------------------------
 CONSUMO DE PETICIONES
-------------------------------------------------------------------------------
Una ficha completa de empresa gasta ~5 peticiones (perfil, resultados, balance,
flujos, ratios). Con 250/día puedes analizar unas 50 empresas diarias. El módulo
cachea en disco para no repetir llamadas.

CLAVE DE API
------------
    1. Regístrate en https://site.financialmodelingprep.com/developer/docs
    2. export FMP_API_KEY="tu_clave"

Uso rápido
----------
    from m14_fundamentales import ficha_empresa, comparar_empresas

    ficha = ficha_empresa("AAPL")
    print(ficha.metricas)
    print(ficha.puntuacion)

    print(comparar_empresas(["AAPL", "MSFT", "LMT", "RTX"]))
===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("quant.fundamentales")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

URL_ESTABLE = "https://financialmodelingprep.com/stable"
URL_LEGADO = "https://financialmodelingprep.com/api/v3"
TIEMPO_ESPERA = 30
PAUSA_ENTRE_LLAMADAS = 0.25  # cortesía: evita ráfagas que disparen el rate limiting


class ErrorFundamental(Exception):
    """Se lanza cuando FMP no responde o los datos no son utilizables."""


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class FichaEmpresa:
    """Toda la información fundamental de una empresa, ya procesada."""
    ticker: str
    perfil: dict
    resultados: pd.DataFrame        # cuenta de resultados, un año por fila
    balance: pd.DataFrame
    flujos: pd.DataFrame
    metricas: pd.Series             # ratios calculados del último ejercicio
    tendencias: pd.DataFrame        # evolución de las métricas clave
    puntuacion: dict[str, float]    # 0-100 por dimensión
    alertas: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        nombre = self.perfil.get("companyName", self.ticker)
        total = self.puntuacion.get("TOTAL", float("nan"))
        return f"<FichaEmpresa {self.ticker} — {nombre} · Puntuación: {total:.0f}/100>"


# =============================================================================
# LLAMADAS A LA API
# =============================================================================

def _obtener_clave(api_key: str | None) -> str:
    clave = api_key or os.environ.get("FMP_API_KEY")
    if not clave:
        raise ErrorFundamental(
            "Falta la clave de FMP. Consíguela gratis en\n"
            "  https://site.financialmodelingprep.com/developer/docs\n"
            'y luego:  export FMP_API_KEY="tu_clave"'
        )
    return clave


def _llamar_fmp(
    endpoint: str,
    parametros: dict | None = None,
    api_key: str | None = None,
    usar_legado: bool = False,
    cache_dir: str | Path | None = "./cache_fmp",
    ttl_horas: float = 24.0,
) -> list | dict:
    """Llamada genérica a FMP, con caché en disco.

    La caché es importante aquí: los estados financieros cambian cuatro veces
    al año, así que volver a pedirlos en cada ejecución desperdicia el cupo
    diario. Con TTL de 24h, un dashboard que se recarga veinte veces gasta
    una sola petición.

    FMP tiene dos familias de rutas:
      · 'stable' (actual):  /stable/income-statement?symbol=AAPL
      · 'legacy' (antigua): /api/v3/income-statement/AAPL
    Se usa la estable por defecto y se puede caer a la antigua si hiciera falta.
    """
    clave = _obtener_clave(api_key)
    parametros = dict(parametros or {})

    # --- Caché -------------------------------------------------------------
    ruta_cache: Path | None = None
    if cache_dir is not None:
        firma = f"{endpoint}_{'_'.join(f'{k}-{v}' for k, v in sorted(parametros.items()))}"
        firma = firma.replace("/", "_").replace(" ", "")
        ruta_cache = Path(cache_dir) / f"{firma}.json"
        if ruta_cache.exists():
            edad_horas = (time.time() - ruta_cache.stat().st_mtime) / 3600
            if edad_horas < ttl_horas:
                logger.debug("Caché usada para %s (%.1f h)", endpoint, edad_horas)
                return json.loads(ruta_cache.read_text())

    # --- Petición ----------------------------------------------------------
    if usar_legado:
        simbolo = parametros.pop("symbol", "")
        url = f"{URL_LEGADO}/{endpoint}/{simbolo}" if simbolo else f"{URL_LEGADO}/{endpoint}"
    else:
        url = f"{URL_ESTABLE}/{endpoint}"

    parametros["apikey"] = clave

    try:
        respuesta = requests.get(url, params=parametros, timeout=TIEMPO_ESPERA)
    except requests.RequestException as exc:
        raise ErrorFundamental(f"Error de red al consultar FMP: {exc}") from exc

    if respuesta.status_code == 401:
        raise ErrorFundamental("FMP rechazó la clave (401). Comprueba FMP_API_KEY.")
    if respuesta.status_code == 403:
        raise ErrorFundamental(
            f"FMP denegó el acceso a '{endpoint}' (403). Ese endpoint suele requerir plan de pago, "
            "o el símbolo no está cubierto por el plan gratuito (que es solo EE.UU.)."
        )
    if respuesta.status_code == 429:
        raise ErrorFundamental(
            "Cupo diario de FMP agotado (429). El plan gratuito son 250 peticiones/día."
        )

    try:
        respuesta.raise_for_status()
        carga = respuesta.json()
    except Exception as exc:
        raise ErrorFundamental(f"Respuesta inválida de FMP ({respuesta.status_code}): {exc}") from exc

    # FMP devuelve errores como dict con la clave 'Error Message', no como HTTP error
    if isinstance(carga, dict) and ("Error Message" in carga or "error" in carga):
        mensaje = carga.get("Error Message") or carga.get("error")
        raise ErrorFundamental(f"FMP devolvió un error para '{endpoint}': {mensaje}")

    if ruta_cache is not None:
        ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        ruta_cache.write_text(json.dumps(carga))

    time.sleep(PAUSA_ENTRE_LLAMADAS)
    return carga


def _a_dataframe(carga: list, ticker: str, nombre: str) -> pd.DataFrame:
    """Convierte la lista de ejercicios de FMP en un DataFrame indexado por fecha."""
    if not carga:
        raise ErrorFundamental(
            f"FMP no devolvió {nombre} para '{ticker}'. Causas habituales: símbolo no "
            "cubierto por el plan gratuito (solo EE.UU.), o es un ETF/fondo (no publican "
            "estados financieros)."
        )
    if isinstance(carga, dict):
        carga = [carga]

    tabla = pd.DataFrame(carga)
    if "date" in tabla.columns:
        tabla["date"] = pd.to_datetime(tabla["date"], errors="coerce")
        tabla = tabla.dropna(subset=["date"]).set_index("date").sort_index()
    return tabla


# =============================================================================
# DESCARGAS ESPECÍFICAS
# =============================================================================

def descargar_perfil(ticker: str, **kwargs) -> dict:
    """Perfil de la empresa: sector, industria, capitalización, descripción."""
    carga = _llamar_fmp("profile", {"symbol": ticker}, **kwargs)
    if not carga:
        raise ErrorFundamental(f"FMP no tiene perfil para '{ticker}'.")
    return carga[0] if isinstance(carga, list) else carga


def descargar_estados(
    ticker: str, periodo: str = "annual", limite: int = 5, **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Descarga los tres estados financieros de golpe.

    Returns
    -------
    (cuenta de resultados, balance, flujos de caja)
    """
    parametros = {"symbol": ticker, "period": periodo, "limit": limite}

    resultados = _a_dataframe(
        _llamar_fmp("income-statement", parametros, **kwargs), ticker, "cuenta de resultados")
    balance = _a_dataframe(
        _llamar_fmp("balance-sheet-statement", parametros, **kwargs), ticker, "balance")
    flujos = _a_dataframe(
        _llamar_fmp("cash-flow-statement", parametros, **kwargs), ticker, "flujos de caja")

    return resultados, balance, flujos


def descargar_ratios(ticker: str, limite: int = 5, **kwargs) -> pd.DataFrame:
    """Ratios ya calculados por FMP (útiles como contraste de los propios)."""
    return _a_dataframe(
        _llamar_fmp("ratios", {"symbol": ticker, "limit": limite}, **kwargs),
        ticker, "ratios",
    )


# =============================================================================
# CÁLCULO DE MÉTRICAS
# =============================================================================

def _seguro(numerador, denominador, multiplicador: float = 1.0) -> float:
    """División que devuelve NaN en vez de reventar cuando el denominador es 0.

    Necesario porque en los estados financieros reales hay ceros por todas
    partes: empresas sin deuda, sin I+D, con patrimonio negativo...
    """
    try:
        num = float(numerador)
        den = float(denominador)
    except (TypeError, ValueError):
        return np.nan
    if den == 0 or pd.isna(den) or pd.isna(num):
        return np.nan
    return (num / den) * multiplicador


def calcular_metricas(
    resultados: pd.DataFrame,
    balance: pd.DataFrame,
    flujos: pd.DataFrame,
    perfil: dict,
) -> pd.Series:
    """Calcula los ratios fundamentales del último ejercicio disponible.

    Se calculan a partir de los estados brutos en vez de tomar los ratios ya
    hechos de FMP: así se sabe exactamente qué contiene cada número y no
    dependemos de su metodología ni de sus cambios.
    """
    if resultados.empty or balance.empty:
        raise ErrorFundamental("Faltan estados financieros para calcular métricas.")

    r = resultados.iloc[-1]
    b = balance.iloc[-1]
    f = flujos.iloc[-1] if not flujos.empty else pd.Series(dtype=float)

    def campo(fuente, *nombres):
        """Los nombres de campo de FMP han cambiado entre versiones de la API."""
        for nombre in nombres:
            if nombre in fuente.index and pd.notna(fuente[nombre]):
                return fuente[nombre]
        return np.nan

    ingresos = campo(r, "revenue")
    beneficio_neto = campo(r, "netIncome")
    ebitda = campo(r, "ebitda")
    ebit = campo(r, "operatingIncome", "ebit")
    gastos_intereses = campo(r, "interestExpense")

    patrimonio = campo(b, "totalStockholdersEquity", "totalEquity")
    activos = campo(b, "totalAssets")
    deuda_total = campo(b, "totalDebt")
    efectivo = campo(b, "cashAndCashEquivalents", "cashAndShortTermInvestments")
    activo_corriente = campo(b, "totalCurrentAssets")
    pasivo_corriente = campo(b, "totalCurrentLiabilities")

    flujo_operativo = campo(f, "operatingCashFlow", "netCashProvidedByOperatingActivities")
    capex = campo(f, "capitalExpenditure")
    flujo_libre = campo(f, "freeCashFlow")
    if pd.isna(flujo_libre) and pd.notna(flujo_operativo) and pd.notna(capex):
        flujo_libre = flujo_operativo + capex  # capex viene en negativo

    capitalizacion = perfil.get("marketCap") or perfil.get("mktCap")
    precio = perfil.get("price")

    deuda_neta = (deuda_total - efectivo) if pd.notna(deuda_total) and pd.notna(efectivo) else np.nan
    valor_empresa = (
        capitalizacion + deuda_neta
        if capitalizacion is not None and pd.notna(deuda_neta) else np.nan
    )

    metricas = pd.Series({
        # --- Rentabilidad ---
        "Margen neto %": _seguro(beneficio_neto, ingresos, 100),
        "Margen EBITDA %": _seguro(ebitda, ingresos, 100),
        "ROE %": _seguro(beneficio_neto, patrimonio, 100),
        "ROA %": _seguro(beneficio_neto, activos, 100),
        # --- Solidez financiera ---
        "Deuda/Patrimonio": _seguro(deuda_total, patrimonio),
        "Deuda neta/EBITDA": _seguro(deuda_neta, ebitda),
        "Ratio corriente": _seguro(activo_corriente, pasivo_corriente),
        "Cobertura de intereses": _seguro(ebit, abs(gastos_intereses) if pd.notna(gastos_intereses) else np.nan),
        # --- Generación de caja ---
        "Margen FCF %": _seguro(flujo_libre, ingresos, 100),
        "FCF yield %": _seguro(flujo_libre, capitalizacion, 100),
        "Conversión caja %": _seguro(flujo_operativo, beneficio_neto, 100),
        # --- Valoración ---
        "PER": _seguro(capitalizacion, beneficio_neto),
        "EV/EBITDA": _seguro(valor_empresa, ebitda),
        "P/Ventas": _seguro(capitalizacion, ingresos),
        # --- Referencia ---
        "Ingresos (M)": ingresos / 1e6 if pd.notna(ingresos) else np.nan,
        "Capitalización (M)": capitalizacion / 1e6 if capitalizacion else np.nan,
        "Precio": precio,
    })

    # --- Crecimiento: exige al menos 2 ejercicios --------------------------
    if len(resultados) >= 2:
        años = len(resultados) - 1
        ingresos_inicial = resultados["revenue"].iloc[0] if "revenue" in resultados.columns else np.nan
        if pd.notna(ingresos_inicial) and ingresos_inicial > 0 and pd.notna(ingresos):
            metricas["Crecimiento ingresos CAGR %"] = ((ingresos / ingresos_inicial) ** (1 / años) - 1) * 100
        if "netIncome" in resultados.columns:
            beneficio_inicial = resultados["netIncome"].iloc[0]
            # El CAGR no tiene sentido si se parte de pérdidas
            if pd.notna(beneficio_inicial) and beneficio_inicial > 0 and pd.notna(beneficio_neto) and beneficio_neto > 0:
                metricas["Crecimiento beneficio CAGR %"] = (
                    (beneficio_neto / beneficio_inicial) ** (1 / años) - 1
                ) * 100

    return metricas.round(3)


def calcular_tendencias(resultados: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Evolución año a año de las magnitudes clave.

    Mirar solo el último ejercicio engaña: una empresa con ROE del 20% que
    viene del 35% cuenta una historia muy distinta a otra que viene del 8%.
    """
    filas = []
    for fecha in resultados.index:
        fila = {"Año": fecha.year}
        r = resultados.loc[fecha]

        if "revenue" in resultados.columns:
            fila["Ingresos (M)"] = round(r["revenue"] / 1e6, 1) if pd.notna(r["revenue"]) else np.nan
        if "netIncome" in resultados.columns:
            fila["Beneficio (M)"] = round(r["netIncome"] / 1e6, 1) if pd.notna(r["netIncome"]) else np.nan
        if "revenue" in resultados.columns and "netIncome" in resultados.columns:
            fila["Margen neto %"] = round(_seguro(r["netIncome"], r["revenue"], 100), 2)

        if fecha in balance.index:
            b = balance.loc[fecha]
            patrimonio = b.get("totalStockholdersEquity", b.get("totalEquity", np.nan))
            if "netIncome" in resultados.columns:
                fila["ROE %"] = round(_seguro(r["netIncome"], patrimonio, 100), 2)
            fila["Deuda/Patrimonio"] = round(_seguro(b.get("totalDebt", np.nan), patrimonio), 3)

        filas.append(fila)

    return pd.DataFrame(filas).set_index("Año")


# =============================================================================
# PUNTUACIÓN DE CALIDAD
# =============================================================================

# Umbrales orientativos. NO son universales: una utility apalancada al 2x deuda/EBITDA
# es normal, en una tecnológica sería preocupante. Ajústalos por sector.
UMBRALES: dict[str, tuple[str, float, float]] = {
    # métrica: (dirección, malo, bueno)
    "ROE %": ("alto_mejor", 5.0, 20.0),
    "Margen neto %": ("alto_mejor", 2.0, 15.0),
    "Margen FCF %": ("alto_mejor", 0.0, 12.0),
    "Deuda neta/EBITDA": ("bajo_mejor", 4.0, 1.0),
    "Ratio corriente": ("alto_mejor", 1.0, 2.0),
    "Cobertura de intereses": ("alto_mejor", 2.0, 10.0),
    "Crecimiento ingresos CAGR %": ("alto_mejor", 0.0, 12.0),
    "FCF yield %": ("alto_mejor", 1.0, 7.0),
    "PER": ("bajo_mejor", 40.0, 12.0),
    "EV/EBITDA": ("bajo_mejor", 20.0, 7.0),
}

DIMENSIONES: dict[str, list[str]] = {
    "Rentabilidad": ["ROE %", "Margen neto %", "Margen FCF %"],
    "Solidez": ["Deuda neta/EBITDA", "Ratio corriente", "Cobertura de intereses"],
    "Crecimiento": ["Crecimiento ingresos CAGR %"],
    "Valoración": ["PER", "EV/EBITDA", "FCF yield %"],
}


def _puntuar_metrica(valor: float, direccion: str, malo: float, bueno: float) -> float:
    """Normaliza una métrica a una escala 0-100 mediante interpolación lineal."""
    if pd.isna(valor):
        return np.nan
    if direccion == "alto_mejor":
        if valor <= malo:
            return 0.0
        if valor >= bueno:
            return 100.0
        return (valor - malo) / (bueno - malo) * 100
    else:  # bajo_mejor
        if valor >= malo:
            return 0.0
        if valor <= bueno:
            return 100.0
        return (malo - valor) / (malo - bueno) * 100


def puntuar_empresa(metricas: pd.Series) -> dict[str, float]:
    """Puntuación 0-100 por dimensión y total.

    ⚠ Es una HEURÍSTICA de cribado, no una valoración. Sirve para ordenar
    candidatos y detectar señales de alarma, no para decidir una inversión.
    Ignora el sector, la calidad del equipo gestor, el foso competitivo, la
    regulación y cualquier factor cualitativo. Un 85/100 no significa "compra".
    """
    puntuaciones: dict[str, float] = {}

    for dimension, metricas_dim in DIMENSIONES.items():
        valores = []
        for nombre in metricas_dim:
            if nombre in UMBRALES and nombre in metricas.index:
                direccion, malo, bueno = UMBRALES[nombre]
                punto = _puntuar_metrica(metricas[nombre], direccion, malo, bueno)
                if pd.notna(punto):
                    valores.append(punto)
        puntuaciones[dimension] = round(float(np.mean(valores)), 1) if valores else np.nan

    validas = [v for v in puntuaciones.values() if pd.notna(v)]
    puntuaciones["TOTAL"] = round(float(np.mean(validas)), 1) if validas else np.nan

    return puntuaciones


def generar_alertas_fundamentales(metricas: pd.Series, tendencias: pd.DataFrame) -> list[str]:
    """Señales de alarma y puntos fuertes, en lenguaje llano."""
    alertas: list[str] = []

    deuda_ebitda = metricas.get("Deuda neta/EBITDA")
    if pd.notna(deuda_ebitda):
        if deuda_ebitda > 4:
            alertas.append(
                f"⚠ Deuda neta/EBITDA de {deuda_ebitda:.1f}×: apalancamiento elevado. "
                "Vulnerable a subidas de tipos o a una caída de resultados."
            )
        elif deuda_ebitda < 0:
            alertas.append("✓ Caja neta positiva: la empresa tiene más efectivo que deuda.")

    cobertura = metricas.get("Cobertura de intereses")
    if pd.notna(cobertura) and cobertura < 2:
        alertas.append(
            f"⚠ Cobertura de intereses de solo {cobertura:.1f}×: los beneficios operativos "
            "apenas cubren el coste de la deuda. Señal de riesgo financiero serio."
        )

    conversion = metricas.get("Conversión caja %")
    if pd.notna(conversion) and conversion < 60:
        alertas.append(
            f"⚠ Conversión de caja del {conversion:.0f}%: el beneficio contable no se está "
            "traduciendo en caja real. Merece revisar circulante y política contable."
        )

    fcf = metricas.get("Margen FCF %")
    if pd.notna(fcf) and fcf < 0:
        alertas.append(f"⚠ Flujo de caja libre NEGATIVO ({fcf:.1f}% de los ingresos): quema caja.")

    per = metricas.get("PER")
    if pd.notna(per):
        if per < 0:
            alertas.append("⚠ PER negativo: la empresa está en pérdidas.")
        elif per > 40:
            alertas.append(
                f"ℹ PER de {per:.1f}: valoración exigente. El mercado descuenta un crecimiento "
                "alto que tendrá que materializarse."
            )

    # Tendencia del margen: lo importante no es el nivel, sino la dirección
    if "Margen neto %" in tendencias.columns and len(tendencias) >= 3:
        margenes = tendencias["Margen neto %"].dropna()
        if len(margenes) >= 3:
            cambio = margenes.iloc[-1] - margenes.iloc[0]
            if cambio < -3:
                alertas.append(
                    f"⚠ El margen neto ha caído {abs(cambio):.1f} puntos en {len(margenes)} años "
                    f"({margenes.iloc[0]:.1f}% → {margenes.iloc[-1]:.1f}%): deterioro sostenido."
                )
            elif cambio > 3:
                alertas.append(
                    f"✓ Margen neto en expansión: {margenes.iloc[0]:.1f}% → {margenes.iloc[-1]:.1f}%."
                )

    if not alertas:
        alertas.append("✓ Sin señales de alarma evidentes en los fundamentales.")
    return alertas


# =============================================================================
# ORQUESTADORES
# =============================================================================

def ficha_empresa(
    ticker: str, periodo: str = "annual", limite: int = 5, **kwargs,
) -> FichaEmpresa:
    """Ficha fundamental completa de una empresa. Consume ~4 peticiones de FMP."""
    logger.info("Analizando %s...", ticker)

    perfil = descargar_perfil(ticker, **kwargs)
    resultados, balance, flujos = descargar_estados(ticker, periodo, limite, **kwargs)

    metricas = calcular_metricas(resultados, balance, flujos, perfil)
    tendencias = calcular_tendencias(resultados, balance)
    puntuacion = puntuar_empresa(metricas)
    alertas = generar_alertas_fundamentales(metricas, tendencias)

    logger.info(
        "%s (%s) · Puntuación %.0f/100",
        ticker, perfil.get("companyName", "?"), puntuacion.get("TOTAL", float("nan")),
    )

    return FichaEmpresa(
        ticker=ticker, perfil=perfil, resultados=resultados, balance=balance,
        flujos=flujos, metricas=metricas, tendencias=tendencias,
        puntuacion=puntuacion, alertas=alertas,
    )


def comparar_empresas(
    tickers: list[str], continuar_si_falla: bool = True, **kwargs,
) -> pd.DataFrame:
    """Tabla comparativa de varias empresas, ordenada por puntuación total.

    Consume ~4 peticiones por empresa: con el cupo gratuito de 250/día, unas
    60 empresas diarias.
    """
    filas = []
    fallidas: dict[str, str] = {}

    for ticker in tickers:
        try:
            ficha = ficha_empresa(ticker, **kwargs)
            fila = {
                "Ticker": ticker,
                "Empresa": ficha.perfil.get("companyName", ticker),
                "Sector": ficha.perfil.get("sector", "—"),
                **{k: v for k, v in ficha.puntuacion.items()},
                "ROE %": ficha.metricas.get("ROE %"),
                "Margen neto %": ficha.metricas.get("Margen neto %"),
                "Deuda neta/EBITDA": ficha.metricas.get("Deuda neta/EBITDA"),
                "PER": ficha.metricas.get("PER"),
                "FCF yield %": ficha.metricas.get("FCF yield %"),
                "Nº alertas": sum(1 for a in ficha.alertas if a.startswith("⚠")),
            }
            filas.append(fila)
        except ErrorFundamental as exc:
            fallidas[ticker] = str(exc)[:120]
            logger.warning("%s descartada: %s", ticker, str(exc)[:120])
            if not continuar_si_falla:
                raise

    if not filas:
        raise ErrorFundamental(f"No se pudo analizar ninguna empresa. Errores: {fallidas}")

    if fallidas:
        logger.warning("Empresas no analizadas: %s", list(fallidas))

    tabla = pd.DataFrame(filas).set_index("Ticker")
    return tabla.sort_values("TOTAL", ascending=False).round(2)


def cribar_candidatos(
    tickers: list[str],
    puntuacion_minima: float = 60.0,
    max_deuda_ebitda: float = 3.0,
    max_per: float = 30.0,
    **kwargs,
) -> pd.DataFrame:
    """Filtra una lista de candidatos según criterios fundamentales mínimos.

    Pensado para usarse ANTES del Módulo 3: reduces el universo a empresas
    financieramente sanas y luego optimizas la cartera solo con esas. Optimizar
    sobre empresas en deterioro produce carteras que parecen buenas en el
    histórico y decepcionan después.
    """
    tabla = comparar_empresas(tickers, **kwargs)

    filtro = tabla["TOTAL"] >= puntuacion_minima
    if "Deuda neta/EBITDA" in tabla.columns:
        filtro &= (tabla["Deuda neta/EBITDA"] <= max_deuda_ebitda) | tabla["Deuda neta/EBITDA"].isna()
    if "PER" in tabla.columns:
        filtro &= (tabla["PER"] > 0) & (tabla["PER"] <= max_per) | tabla["PER"].isna()

    aprobadas = tabla[filtro]
    logger.info("Criba: %d de %d empresas superan los filtros", len(aprobadas), len(tabla))

    if aprobadas.empty:
        logger.warning(
            "Ninguna empresa superó los filtros. Considera relajarlos: "
            "puntuacion_minima=%.0f, max_deuda_ebitda=%.1f, max_per=%.0f",
            puntuacion_minima, max_deuda_ebitda, max_per,
        )
    return aprobadas


def resumen_para_analisis(ficha: FichaEmpresa) -> str:
    """Genera un resumen en texto listo para pegarlo en una conversación conmigo.

    Los números por sí solos no cuentan la historia: el contexto sectorial, la
    posición competitiva o los riesgos regulatorios requieren interpretación.
    Esta función empaqueta los datos duros para que puedas pedir esa lectura.
    """
    perfil = ficha.perfil
    lineas = [
        f"EMPRESA: {perfil.get('companyName', ficha.ticker)} ({ficha.ticker})",
        f"Sector: {perfil.get('sector', '—')} | Industria: {perfil.get('industry', '—')}",
        f"País: {perfil.get('country', '—')} | Capitalización: {ficha.metricas.get('Capitalización (M)', float('nan')):,.0f} M",
        "",
        "PUNTUACIÓN (heurística de cribado, 0-100):",
    ]
    lineas += [f"  {k:15s} {v}" for k, v in ficha.puntuacion.items()]
    lineas += ["", "MÉTRICAS DEL ÚLTIMO EJERCICIO:"]
    lineas += [f"  {k:32s} {v}" for k, v in ficha.metricas.items() if pd.notna(v)]
    lineas += ["", "EVOLUCIÓN:", ficha.tendencias.to_string()]
    lineas += ["", "SEÑALES DETECTADAS:"]
    lineas += [f"  {a}" for a in ficha.alertas]
    return "\n".join(lineas)


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m14_fundamentales.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 78)
    print("MÓDULO 14 · ANÁLISIS FUNDAMENTAL (FMP)")
    print("=" * 78)
    print("\n⚠ El plan gratuito de FMP cubre SOLO empresas estadounidenses.")
    print("  Para RHM.DE / IDR.MC / MC.PA necesitarías el plan de pago.\n")

    try:
        # Ejemplo con defensa estadounidense: el sector más cercano a tu RHM.DE
        candidatos = ["LMT", "RTX", "NOC", "GD", "LHX"]

        print("--- Comparativa del sector defensa EE.UU. ---")
        tabla = comparar_empresas(candidatos)
        print(tabla.to_string())

        print("\n--- Criba (puntuación ≥ 60, deuda/EBITDA ≤ 3, PER ≤ 30) ---")
        print(cribar_candidatos(candidatos).to_string())

        print("\n--- Ficha detallada de la mejor puntuada ---")
        mejor = tabla.index[0]
        ficha = ficha_empresa(mejor)
        print(resumen_para_analisis(ficha))

        print("\n[OK] Módulo 14 ejecutado.")

    except ErrorFundamental as exc:
        print(f"\n[ERROR FMP] {exc}")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
