"""
===============================================================================
 MÓDULO 7 — SUPERFICIE DE VOLATILIDAD IMPLÍCITA 3D (Fase 3)
===============================================================================
Mientras el Módulo 5 mide lo que YA pasó (volatilidad realizada), este módulo
extrae lo que el mercado de opciones ESPERA que pase. Es el "radar predictivo".

  Eje X → Días hasta el vencimiento del contrato
  Eje Y → Moneyness (Strike / Precio spot). 1.00 = At-The-Money
  Eje Z → Volatilidad implícita (%)

Se usa `scipy.interpolate.griddata` para rellenar los huecos entre contratos
cotizados y generar una superficie continua.

QUÉ REVELA LA FORMA DE LA SUPERFICIE
------------------------------------
  · SONRISA (smile): la IV sube tanto en strikes bajos como altos. Típico de
    divisas y materias primas: el mercado teme movimientos bruscos en ambos sentidos.
  · SKEW / MUECA (smirk): la IV es mucho más alta en strikes BAJOS (puts OTM).
    Es la forma normal en renta variable desde 1987: los inversores pagan una
    prima por protegerse de caídas. Cuanto más pronunciado el skew, más miedo.
  · ESTRUCTURA TEMPORAL: si los vencimientos cortos tienen MÁS IV que los
    largos (inversión de la curva), hay un evento a corto plazo descontado
    (resultados, elecciones, decisión de tipos). Es una señal de estrés agudo.

-------------------------------------------------------------------------------
 LIMITACIÓN IMPORTANTE PARA TU CARTERA
-------------------------------------------------------------------------------
yfinance solo sirve cadenas de opciones de mercados estadounidenses. De tus
activos, NINGUNO tiene cadena disponible por esta vía (RHM.DE e IDR.MC cotizan
opciones en Eurex/MEFF, pero yfinance no las cubre; el fondo Polar Capital no
tiene opciones en absoluto).

Por eso este módulo trabaja sobre PROXIES líquidos: SPY (S&P 500), o sectoriales
como ITA (defensa EE.UU.) y EWY (Corea del Sur), que sí replican parcialmente
los riesgos de tu cartera. Estás midiendo el clima general del mercado, no el
de tus posiciones concretas — pero es exactamente lo que hacen los
institucionales cuando no hay opciones líquidas de un subyacente.

Uso rápido
----------
    from m7_superficie_iv import analizar_superficie

    sup = analizar_superficie("SPY", max_vencimientos=8)
    print(sup.resumen)
    print(sup.metricas)
    # → guarda superficie_iv_SPY.png y (si tienes plotly) superficie_iv_SPY.html

Requisitos: yfinance, scipy, matplotlib. Opcional: plotly (versión interactiva).
===============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.interpolate import griddata

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("Falta 'yfinance'. Instálalo con: pip install yfinance") from exc

# Plotly es OPCIONAL: sin él funciona todo salvo el 3D interactivo
try:
    import plotly.graph_objects as go
    PLOTLY_DISPONIBLE = True
except ImportError:
    PLOTLY_DISPONIBLE = False

logger = logging.getLogger("quant.superficie_iv")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


class ErrorDeSuperficie(Exception):
    """Se lanza cuando no hay datos de opciones suficientes para la superficie."""


# =============================================================================
# PROXIES RECOMENDADOS PARA TU CARTERA
# =============================================================================

PROXIES_OPCIONES: dict[str, dict[str, str]] = {
    "SPY": {
        "nombre": "S&P 500 ETF",
        "cubre": "Riesgo de mercado global. El subyacente con opciones más líquido del mundo.",
    },
    "ITA": {
        "nombre": "iShares U.S. Aerospace & Defense ETF",
        "cubre": "Sector defensa. El proxy más cercano a RHM.DE con opciones líquidas.",
    },
    "EWY": {
        "nombre": "iShares MSCI South Korea ETF",
        "cubre": "Bolsa coreana (KOSPI) cotizada en EE.UU., con cadena de opciones.",
    },
    "EZU": {
        "nombre": "iShares MSCI Eurozone ETF",
        "cubre": "Eurozona. Proxy con opciones para el EURO STOXX / tus activos europeos.",
    },
    "URTH": {
        "nombre": "iShares MSCI World ETF",
        "cubre": "MSCI World. Equivalente con opciones a tu EUNL.DE.",
    },
    "QQQ": {
        "nombre": "Invesco QQQ (Nasdaq 100)",
        "cubre": "Tecnología. Proxy para tu fondo Polar Capital Global Tech.",
    },
}


def listar_proxies() -> pd.DataFrame:
    """Tabla de subyacentes con opciones líquidas y qué parte de tu cartera cubre cada uno."""
    return pd.DataFrame([
        {"Ticker": t, "Nombre": i["nombre"], "Qué cubre de tu cartera": i["cubre"]}
        for t, i in PROXIES_OPCIONES.items()
    ]).set_index("Ticker")


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoSuperficie:
    """Empaqueta la superficie y todas sus métricas derivadas."""
    ticker: str
    spot: float
    cadena: pd.DataFrame            # contratos limpios: strike, iv, dias, moneyness, tipo
    malla_x: np.ndarray             # días (grid)
    malla_y: np.ndarray             # moneyness (grid)
    malla_z: np.ndarray             # IV interpolada (grid)
    resumen: pd.DataFrame           # IV ATM por vencimiento (estructura temporal)
    metricas: dict[str, float]      # skew, pendiente temporal, IV ATM 30d...
    alertas: list[str] = field(default_factory=list)
    fecha: datetime = field(default_factory=datetime.now)


# =============================================================================
# 1) DESCARGA DE LA CADENA DE OPCIONES
# =============================================================================

def obtener_spot(ticker: str) -> float:
    """Precio actual del subyacente. Necesario para calcular el moneyness."""
    activo = yf.Ticker(ticker)
    try:
        info_rapida = activo.fast_info
        spot = float(info_rapida["lastPrice"])
        if spot > 0:
            return spot
    except Exception:
        pass  # fast_info falla a menudo; caemos al histórico

    try:
        historico = activo.history(period="5d")
        if not historico.empty:
            return float(historico["Close"].dropna().iloc[-1])
    except Exception as exc:
        raise ErrorDeSuperficie(f"No se pudo obtener el precio spot de {ticker}: {exc}") from exc

    raise ErrorDeSuperficie(f"No se pudo obtener el precio spot de {ticker}.")


def _seleccionar_vencimientos(
    vencimientos: list[str],
    hoy: pd.Timestamp,
    max_vencimientos: int,
    dias_min: int,
    dias_max: int,
) -> list[tuple[str, int]]:
    """Elige vencimientos REPARTIDOS por toda la curva temporal, no los primeros.

    Por qué importa: subyacentes muy líquidos como SPY tienen vencimientos
    semanales (a veces diarios). Si coges simplemente los 8 primeros, TODOS caen
    dentro del mes siguiente y la superficie sale plana en el eje temporal:
    pierdes justamente la estructura de plazos, que es la mitad de la información.

    Estrategia: se fijan objetivos logarítmicamente espaciados (≈7, 14, 30, 60,
    90, 180, 270, 365 días) y se escoge el vencimiento real más cercano a cada uno.
    """
    candidatos = []
    for fecha_str in vencimientos:
        dias = (pd.Timestamp(fecha_str) - hoy).days
        if dias_min <= dias <= dias_max:
            candidatos.append((fecha_str, dias))

    if not candidatos:
        return []
    if len(candidatos) <= max_vencimientos:
        return candidatos

    # Objetivos espaciados logarítmicamente: mucha resolución en el corto plazo
    # (donde la superficie cambia rápido) y menos en el largo.
    dias_disponibles = np.array([d for _, d in candidatos])
    objetivos = np.geomspace(
        max(dias_disponibles.min(), dias_min),
        dias_disponibles.max(),
        max_vencimientos,
    )

    seleccionados: list[tuple[str, int]] = []
    usados: set[int] = set()
    for objetivo in objetivos:
        idx = int(np.argmin(np.abs(dias_disponibles - objetivo)))
        # Si ese vencimiento ya está cogido, buscamos el siguiente más cercano libre
        if idx in usados:
            orden = np.argsort(np.abs(dias_disponibles - objetivo))
            for alternativa in orden:
                if int(alternativa) not in usados:
                    idx = int(alternativa)
                    break
            else:
                continue
        usados.add(idx)
        seleccionados.append(candidatos[idx])

    return sorted(seleccionados, key=lambda par: par[1])


def descargar_cadena_opciones(
    ticker: str,
    max_vencimientos: int = 8,
    dias_min: int = 5,
    dias_max: int = 400,
) -> tuple[pd.DataFrame, float]:
    """Descarga la cadena de opciones completa de un subyacente.

    Parameters
    ----------
    max_vencimientos : cuántas fechas de vencimiento descargar. Cada una es una
        llamada a la API, así que más de 10-12 hace la descarga muy lenta.
        Se reparten por toda la curva (ver `_seleccionar_vencimientos`), no se
        cogen los primeros del calendario.
    dias_min : descarta vencimientos inminentes (< 5 días). Su IV está
        distorsionada por el efecto "pin risk" y el decaimiento temporal extremo.
    dias_max : descarta vencimientos muy lejanos (LEAPS), normalmente ilíquidos.

    Returns
    -------
    (DataFrame de contratos en bruto, precio spot)
    """
    activo = yf.Ticker(ticker)

    try:
        vencimientos = list(activo.options)
    except Exception as exc:
        raise ErrorDeSuperficie(
            f"No se pudo leer la lista de vencimientos de '{ticker}': {exc}. "
            "Comprueba tu conexión."
        ) from exc

    if not vencimientos:
        raise ErrorDeSuperficie(
            f"'{ticker}' no tiene cadena de opciones en Yahoo Finance. "
            f"yfinance solo cubre mercados de EE.UU. Prueba con un proxy: {list(PROXIES_OPCIONES)}"
        )

    spot = obtener_spot(ticker)
    hoy = pd.Timestamp.today().normalize()

    elegidos = _seleccionar_vencimientos(vencimientos, hoy, max_vencimientos, dias_min, dias_max)
    if not elegidos:
        raise ErrorDeSuperficie(
            f"'{ticker}' no tiene vencimientos entre {dias_min} y {dias_max} días. "
            f"Vencimientos disponibles: {vencimientos[:5]}..."
        )

    logger.info(
        "Vencimientos seleccionados para %s (días): %s",
        ticker, [d for _, d in elegidos],
    )

    filas = []
    for fecha_str, dias in elegidos:
        try:
            cadena = activo.option_chain(fecha_str)
        except Exception as exc:
            logger.warning("Vencimiento %s no descargado: %s", fecha_str, exc)
            continue

        for tipo, tabla in (("call", cadena.calls), ("put", cadena.puts)):
            if tabla is None or tabla.empty:
                continue
            parcial = tabla.copy()
            parcial["tipo"] = tipo
            parcial["vencimiento"] = fecha_str
            parcial["dias"] = dias
            filas.append(parcial)

    if not filas:
        raise ErrorDeSuperficie(
            f"No se descargó ningún contrato válido de '{ticker}' entre {dias_min} y "
            f"{dias_max} días. Prueba a ampliar el rango de días."
        )

    bruto = pd.concat(filas, ignore_index=True)
    logger.info(
        "Cadena de %s descargada · %d contratos · %d vencimientos · spot=%.2f",
        ticker, len(bruto), len(elegidos), spot,
    )
    return bruto, spot


# =============================================================================
# 2) LIMPIEZA DE LA CADENA
# =============================================================================

def limpiar_cadena(
    bruto: pd.DataFrame,
    spot: float,
    iv_min: float = 0.01,
    iv_max: float = 3.00,
    moneyness_min: float = 0.70,
    moneyness_max: float = 1.30,
    volumen_min: int = 0,
    open_interest_min: int = 1,
    solo_otm: bool = True,
) -> pd.DataFrame:
    """Filtra la cadena en bruto para quedarse con contratos fiables.

    Esta limpieza NO es opcional: las cadenas de Yahoo vienen llenas de
    contratos sin negociar cuya IV es 0, absurda (500%) o simplemente heredada
    de la última cotización de hace semanas. Si no se filtran, la superficie
    sale llena de picos artificiales.

    Filtros aplicados:
      · IV dentro de un rango razonable (1%-300%)
      · Moneyness entre 0.70 y 1.30 (fuera de ahí no hay liquidez real)
      · Open interest mínimo (contratos que alguien tiene abiertos de verdad)
      · solo_otm=True → usa puts por debajo del spot y calls por encima.
        Es el estándar de mercado: las opciones OTM son las líquidas, y así
        se evita duplicar el mismo punto de la superficie con dos IV distintas.
    """
    if bruto.empty:
        raise ErrorDeSuperficie("La cadena de opciones está vacía.")

    columnas_necesarias = {"strike", "impliedVolatility", "dias", "tipo"}
    faltantes = columnas_necesarias - set(bruto.columns)
    if faltantes:
        raise ErrorDeSuperficie(f"Faltan columnas en la cadena: {faltantes}")

    df = bruto.copy()
    df = df.rename(columns={"impliedVolatility": "iv", "openInterest": "oi", "volume": "volumen"})

    if "oi" not in df.columns:
        df["oi"] = np.nan
    if "volumen" not in df.columns:
        df["volumen"] = np.nan

    n_inicial = len(df)

    df["moneyness"] = df["strike"] / spot
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)
    df["volumen"] = pd.to_numeric(df["volumen"], errors="coerce").fillna(0)

    df = df[df["iv"].between(iv_min, iv_max)]
    df = df[df["moneyness"].between(moneyness_min, moneyness_max)]
    df = df[(df["oi"] >= open_interest_min) | (df["volumen"] > volumen_min)]

    if solo_otm:
        es_otm = ((df["tipo"] == "put") & (df["moneyness"] <= 1.0)) | \
                 ((df["tipo"] == "call") & (df["moneyness"] > 1.0))
        df = df[es_otm]

    # Si aun así queda algún duplicado (mismo día y strike), nos quedamos con el
    # de mayor open interest, que es el contrato con precio más fiable.
    df = df.sort_values("oi", ascending=False).drop_duplicates(subset=["dias", "strike"], keep="first")

    columnas = ["dias", "vencimiento", "strike", "moneyness", "iv", "tipo", "oi", "volumen"]
    df = df[[c for c in columnas if c in df.columns]].sort_values(["dias", "strike"]).reset_index(drop=True)

    if len(df) < 20:
        raise ErrorDeSuperficie(
            f"Tras limpiar solo quedan {len(df)} contratos (de {n_inicial}). "
            "Insuficiente para interpolar una superficie. Prueba con un subyacente "
            "más líquido (SPY, QQQ) o relaja los filtros de moneyness."
        )
    if df["dias"].nunique() < 3:
        raise ErrorDeSuperficie(
            f"Solo hay {df['dias'].nunique()} vencimientos distintos tras limpiar. "
            "Se necesitan al menos 3 para construir el eje temporal de la superficie."
        )

    logger.info(
        "Cadena limpiada · %d → %d contratos · %d vencimientos · IV media %.1f%%",
        n_inicial, len(df), df["dias"].nunique(), df["iv"].mean() * 100,
    )
    return df


# =============================================================================
# 3) CONSTRUCCIÓN DE LA SUPERFICIE (scipy.interpolate.griddata)
# =============================================================================

def construir_superficie(
    cadena: pd.DataFrame,
    n_grid_dias: int = 50,
    n_grid_moneyness: int = 50,
    metodo: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola una superficie continua a partir de los contratos discretos.

    Los contratos cotizan en strikes y vencimientos concretos (puntos sueltos).
    `griddata` rellena los huecos entre ellos para obtener una superficie suave.

    Método:
      · 'cubic'  → suave y realista, pero puede oscilar en zonas con pocos datos
      · 'linear' → más conservador, sin oscilaciones artificiales
      · 'nearest'→ escalonado, solo como último recurso

    Estrategia de dos pasadas: primero interpola con el método pedido y luego
    rellena los NaN restantes (bordes del grid) con 'nearest', de modo que la
    superficie no tenga agujeros.
    """
    if cadena.empty:
        raise ErrorDeSuperficie("No hay contratos para construir la superficie.")

    puntos = cadena[["dias", "moneyness"]].values
    valores = cadena["iv"].values * 100  # a porcentaje, más legible en el eje Z

    dias_grid = np.linspace(cadena["dias"].min(), cadena["dias"].max(), n_grid_dias)
    money_grid = np.linspace(cadena["moneyness"].min(), cadena["moneyness"].max(), n_grid_moneyness)
    malla_x, malla_y = np.meshgrid(dias_grid, money_grid)

    malla_z = griddata(puntos, valores, (malla_x, malla_y), method=metodo)

    # Segunda pasada: tapar los huecos que deja el método principal
    huecos = np.isnan(malla_z)
    if huecos.any():
        relleno = griddata(puntos, valores, (malla_x, malla_y), method="nearest")
        malla_z[huecos] = relleno[huecos]
        logger.info("Rellenados %d puntos del grid con interpolación 'nearest'.", huecos.sum())

    # Recorte de seguridad: la interpolación cúbica puede generar valores absurdos
    malla_z = np.clip(malla_z, 0.5, 300.0)

    return malla_x, malla_y, malla_z


# =============================================================================
# 4) MÉTRICAS DERIVADAS DE LA SUPERFICIE
# =============================================================================

def estructura_temporal(cadena: pd.DataFrame, banda_atm: float = 0.03) -> pd.DataFrame:
    """IV At-The-Money por vencimiento: el "corte" de la superficie en moneyness=1.

    Es la lectura más directa: ¿el mercado espera más turbulencia a 1 mes o a 1 año?
    """
    atm = cadena[(cadena["moneyness"] - 1.0).abs() <= banda_atm]
    if atm.empty:
        # Ampliamos la banda si no hay contratos suficientemente cerca del dinero
        atm = cadena[(cadena["moneyness"] - 1.0).abs() <= banda_atm * 3]
    if atm.empty:
        raise ErrorDeSuperficie("No hay contratos cerca del dinero (ATM) para la estructura temporal.")

    resumen = atm.groupby("dias").agg(
        IV_ATM=("iv", "mean"),
        contratos=("iv", "size"),
    ).reset_index()
    resumen["IV_ATM %"] = (resumen["IV_ATM"] * 100).round(2)
    return resumen[["dias", "IV_ATM %", "contratos"]]


def calcular_metricas(
    cadena: pd.DataFrame,
    resumen_temporal: pd.DataFrame,
    dias_referencia: int = 30,
) -> dict[str, float]:
    """Métricas cuantitativas que resumen la forma de la superficie.

    · 'skew_aprox': diferencia de IV entre moneyness 0.90 (puts de protección)
      y 1.10 (calls especulativas). Positivo = el mercado paga más por
      protegerse de caídas que por apostar a subidas. Es lo normal; lo
      relevante es CUÁNTO.
    · 'pendiente_temporal': IV del vencimiento más largo menos la del más corto.
      Negativa = curva invertida = estrés a corto plazo.

    El skew se ancla al vencimiento más cercano a `dias_referencia` (30 días por
    defecto, el mismo horizonte que usa el VIX) y NO al más corto disponible: a
    7 días el skew es siempre extremo por el propio decaimiento temporal, y
    tomarlo de ahí produciría falsas alarmas sistemáticas.
    """
    metricas: dict[str, float] = {}

    # Vencimiento más próximo a los 30 días (referencia estándar de mercado)
    dias_disponibles = resumen_temporal["dias"].values
    vencimiento_ref = int(dias_disponibles[np.argmin(np.abs(dias_disponibles - dias_referencia))])
    corte = cadena[cadena["dias"] == vencimiento_ref]

    def iv_en(objetivo: float, tolerancia: float = 0.03) -> float:
        seleccion = corte[(corte["moneyness"] - objetivo).abs() <= tolerancia]
        return float(seleccion["iv"].mean() * 100) if not seleccion.empty else np.nan

    iv_put_otm = iv_en(0.90)
    iv_atm = iv_en(1.00)
    iv_call_otm = iv_en(1.10)

    metricas["Vencimiento referencia (días)"] = float(vencimiento_ref)
    metricas["IV ATM referencia %"] = iv_atm
    metricas["IV put OTM (0.90) %"] = iv_put_otm
    metricas["IV call OTM (1.10) %"] = iv_call_otm
    metricas["Skew (put0.90 - call1.10)"] = (
        iv_put_otm - iv_call_otm if pd.notna(iv_put_otm) and pd.notna(iv_call_otm) else np.nan
    )

    if len(resumen_temporal) >= 2:
        iv_corto = resumen_temporal.iloc[0]["IV_ATM %"]
        iv_largo = resumen_temporal.iloc[-1]["IV_ATM %"]
        metricas["Pendiente temporal (largo - corto)"] = iv_largo - iv_corto
        metricas["Días corto"] = float(resumen_temporal.iloc[0]["dias"])
        metricas["Días largo"] = float(resumen_temporal.iloc[-1]["dias"])

    metricas["IV media superficie %"] = float(cadena["iv"].mean() * 100)
    metricas["Contratos usados"] = float(len(cadena))

    return {k: (round(v, 3) if pd.notna(v) else np.nan) for k, v in metricas.items()}


def generar_alertas_iv(
    metricas: dict[str, float],
    vol_realizada: float | None = None,
) -> list[str]:
    """Interpreta las métricas en lenguaje natural.

    Si se pasa `vol_realizada` (la del Módulo 5, en %), añade la comparación
    más valiosa del análisis: la prima de riesgo de volatilidad.
    """
    alertas: list[str] = []

    skew = metricas.get("Skew (put0.90 - call1.10)")
    if pd.notna(skew):
        if skew > 8:
            alertas.append(
                f"⚠ Skew muy pronunciado ({skew:+.1f} puntos): el mercado está pagando una prima "
                "elevada por protegerse de caídas. Señal clásica de nerviosismo institucional."
            )
        elif skew > 2:
            alertas.append(f"✓ Skew normal ({skew:+.1f} puntos): forma habitual en renta variable.")
        elif skew < -2:
            alertas.append(
                f"ℹ Skew invertido ({skew:+.1f}): se paga más por calls que por puts. Poco común "
                "en índices; típico de materias primas en escasez o de manías especulativas."
            )

    pendiente = metricas.get("Pendiente temporal (largo - corto)")
    if pd.notna(pendiente):
        if pendiente < -2:
            alertas.append(
                f"⚠ Estructura temporal INVERTIDA ({pendiente:+.1f} puntos): el corto plazo cotiza "
                "más volatilidad que el largo. El mercado descuenta un evento inminente "
                "(resultados, tipos, geopolítica)."
            )
        elif pendiente > 2:
            alertas.append(
                f"✓ Estructura temporal normal ({pendiente:+.1f} puntos): más incertidumbre a largo "
                "plazo que a corto, como corresponde a un mercado en calma."
            )

    iv_atm = metricas.get("IV ATM referencia %")
    if vol_realizada is not None and pd.notna(iv_atm):
        prima = iv_atm - vol_realizada
        if prima > 3:
            alertas.append(
                f"ℹ Prima de riesgo de volatilidad: +{prima:.1f} puntos (IV {iv_atm:.1f}% vs "
                f"realizada {vol_realizada:.1f}%). El mercado espera MÁS turbulencia de la que ha "
                "habido. Vender volatilidad ha sido históricamente rentable en este régimen, "
                "pero con riesgo de cola muy elevado."
            )
        elif prima < -3:
            alertas.append(
                f"⚠ IV por DEBAJO de la volatilidad realizada ({iv_atm:.1f}% vs {vol_realizada:.1f}%). "
                "El mercado de opciones está infravalorando el movimiento que ya está ocurriendo: "
                "la protección está barata."
            )
        else:
            alertas.append(
                f"✓ IV ({iv_atm:.1f}%) alineada con la volatilidad realizada ({vol_realizada:.1f}%)."
            )

    if not alertas:
        alertas.append("✓ Superficie sin anomalías destacables.")
    return alertas


# =============================================================================
# 5) GRÁFICOS
# =============================================================================

def plot_superficie_3d(
    resultado: ResultadoSuperficie,
    guardar: str | None = None,
    dpi: int = 150,
    elevacion: int = 25,
    azimut: int = -60,
) -> Figure:
    """Superficie 3D estática con matplotlib. Siempre disponible."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    superficie = ax.plot_surface(
        resultado.malla_x, resultado.malla_y, resultado.malla_z,
        cmap="viridis", edgecolor="none", alpha=0.92,
        rstride=1, cstride=1, antialiased=True,
    )

    # Contratos reales, como puntos rojos sobre la superficie interpolada
    ax.scatter(
        resultado.cadena["dias"], resultado.cadena["moneyness"], resultado.cadena["iv"] * 100,
        color="red", s=8, alpha=0.45, label="Contratos cotizados",
    )

    ax.set_xlabel("Días al vencimiento", labelpad=10)
    ax.set_ylabel("Moneyness (Strike / Spot)", labelpad=10)
    ax.set_zlabel("Volatilidad implícita (%)", labelpad=10)
    ax.set_title(
        f"Superficie de Volatilidad Implícita · {resultado.ticker}\n"
        f"Spot = {resultado.spot:.2f} · {resultado.fecha:%Y-%m-%d %H:%M}",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.view_init(elev=elevacion, azim=azimut)
    fig.colorbar(superficie, ax=ax, shrink=0.55, aspect=18, label="IV (%)")
    ax.legend(loc="upper left", fontsize=8)

    if guardar:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Superficie 3D guardada en '%s'", guardar)
    return fig


def plot_superficie_interactiva(
    resultado: ResultadoSuperficie,
    guardar: str | None = None,
) -> object | None:
    """Superficie 3D INTERACTIVA (rotable, con zoom) usando plotly.

    Devuelve None si plotly no está instalado, sin romper el flujo. Para
    habilitarla: pip install plotly
    """
    if not PLOTLY_DISPONIBLE:
        logger.warning(
            "plotly no está instalado: se omite la superficie interactiva. "
            "Instálalo con 'pip install plotly' para obtener el HTML rotable."
        )
        return None

    figura = go.Figure(data=[
        go.Surface(
            x=resultado.malla_x, y=resultado.malla_y, z=resultado.malla_z,
            colorscale="Viridis", colorbar=dict(title="IV (%)"), opacity=0.95,
        )
    ])
    figura.add_trace(go.Scatter3d(
        x=resultado.cadena["dias"], y=resultado.cadena["moneyness"],
        z=resultado.cadena["iv"] * 100,
        mode="markers", marker=dict(size=2, color="red"), name="Contratos cotizados",
    ))
    figura.update_layout(
        title=f"Superficie de Volatilidad Implícita · {resultado.ticker} · Spot {resultado.spot:.2f}",
        scene=dict(
            xaxis_title="Días al vencimiento",
            yaxis_title="Moneyness (Strike/Spot)",
            zaxis_title="Volatilidad implícita (%)",
        ),
        width=1000, height=750,
    )

    if guardar:
        figura.write_html(guardar)
        logger.info("Superficie interactiva guardada en '%s' (ábrela en el navegador)", guardar)
    return figura


def plot_smile_y_estructura(
    resultado: ResultadoSuperficie,
    guardar: str | None = None,
    dpi: int = 150,
    max_vencimientos_smile: int = 5,
) -> Figure:
    """Los dos cortes 2D de la superficie, que a menudo se leen mejor que el 3D:

    Izquierda: la SONRISA/SKEW — IV vs moneyness, una curva por vencimiento.
    Derecha: la ESTRUCTURA TEMPORAL — IV At-The-Money vs días al vencimiento.
    """
    fig, (ax_smile, ax_termino) = plt.subplots(1, 2, figsize=(14, 5))

    vencimientos = sorted(resultado.cadena["dias"].unique())[:max_vencimientos_smile]
    colores = plt.cm.viridis(np.linspace(0, 0.85, len(vencimientos)))

    for color, dias in zip(colores, vencimientos):
        corte = resultado.cadena[resultado.cadena["dias"] == dias].sort_values("moneyness")
        ax_smile.plot(corte["moneyness"], corte["iv"] * 100, "o-",
                      color=color, markersize=3, linewidth=1.5, label=f"{dias}d")

    ax_smile.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_smile.annotate("ATM", (1.0, ax_smile.get_ylim()[1]), fontsize=8,
                      ha="center", va="top", color="gray")
    ax_smile.set_xlabel("Moneyness (Strike / Spot)")
    ax_smile.set_ylabel("Volatilidad implícita (%)")
    ax_smile.set_title("Sonrisa / Skew de volatilidad", fontsize=11, fontweight="bold")
    ax_smile.legend(fontsize=8, title="Vencimiento")
    ax_smile.grid(alpha=0.3)

    ax_termino.plot(resultado.resumen["dias"], resultado.resumen["IV_ATM %"],
                    "o-", color="#1F4E79", linewidth=2, markersize=6)
    ax_termino.set_xlabel("Días al vencimiento")
    ax_termino.set_ylabel("IV At-The-Money (%)")
    ax_termino.set_title("Estructura temporal de la volatilidad", fontsize=11, fontweight="bold")
    ax_termino.grid(alpha=0.3)

    fig.suptitle(
        f"Cortes 2D de la superficie · {resultado.ticker} (spot {resultado.spot:.2f})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Cortes 2D guardados en '%s'", guardar)
    return fig


# =============================================================================
# ORQUESTADOR DEL MÓDULO
# =============================================================================

def analizar_superficie(
    ticker: str = "SPY",
    max_vencimientos: int = 8,
    metodo_interpolacion: str = "cubic",
    vol_realizada: float | None = None,
    guardar_png: bool = True,
    guardar_html: bool = True,
) -> ResultadoSuperficie:
    """Flujo completo: descarga → limpieza → interpolación → métricas → gráficos.

    Parameters
    ----------
    vol_realizada : volatilidad histórica del activo en % (del Módulo 5). Si se
        pasa, se calcula la prima de riesgo de volatilidad (IV - realizada),
        que es la comparación más informativa entre las Fases 2 y 3.
    """
    bruto, spot = descargar_cadena_opciones(ticker, max_vencimientos=max_vencimientos)
    cadena = limpiar_cadena(bruto, spot)
    malla_x, malla_y, malla_z = construir_superficie(cadena, metodo=metodo_interpolacion)
    resumen = estructura_temporal(cadena)
    metricas = calcular_metricas(cadena, resumen)
    alertas = generar_alertas_iv(metricas, vol_realizada)

    resultado = ResultadoSuperficie(
        ticker=ticker, spot=spot, cadena=cadena,
        malla_x=malla_x, malla_y=malla_y, malla_z=malla_z,
        resumen=resumen, metricas=metricas, alertas=alertas,
    )

    if guardar_png:
        plot_superficie_3d(resultado, guardar=f"superficie_iv_{ticker}.png")
        plot_smile_y_estructura(resultado, guardar=f"smile_iv_{ticker}.png")
    if guardar_html:
        plot_superficie_interactiva(resultado, guardar=f"superficie_iv_{ticker}.html")

    return resultado


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m7_superficie_iv.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 25)

    print("\n--- Subyacentes con opciones líquidas (proxies para tu cartera) ---")
    print(listar_proxies().to_string())

    try:
        sup = analizar_superficie("SPY", max_vencimientos=8)

        print("\n" + "=" * 78)
        print(f"MÓDULO 7 · SUPERFICIE DE VOLATILIDAD IMPLÍCITA · {sup.ticker}")
        print("=" * 78)
        print(f"\nSpot: {sup.spot:.2f} · Contratos usados: {len(sup.cadena)}")
        print("\n--- Estructura temporal (IV At-The-Money) ---")
        print(sup.resumen.to_string(index=False))
        print("\n--- Métricas de forma ---")
        for clave, valor in sup.metricas.items():
            print(f"  {clave:38s} {valor}")
        print("\n--- Diagnóstico ---")
        for a in sup.alertas:
            print("  " + a)
        print(f"\n[OK] Módulo 7 ejecutado. Archivos: superficie_iv_{sup.ticker}.png, "
              f"smile_iv_{sup.ticker}.png" + (f", superficie_iv_{sup.ticker}.html" if PLOTLY_DISPONIBLE else ""))

    except ErrorDeSuperficie as exc:
        print(f"\n[ERROR DE SUPERFICIE] {exc}")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
