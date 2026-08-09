"""
===============================================================================
 MÓDULO 5 — VOLATILIDAD PASADA (HISTÓRICA / REALIZADA)
===============================================================================
Mide cómo se ha comportado el riesgo de cada activo EN EL TIEMPO, no como un
número único. La volatilidad de la matriz de covarianzas del Módulo 3 es un
promedio de 3 años; este módulo la descompone día a día para responder:

  · ¿Está cada activo hoy MÁS o MENOS nervioso que su media histórica?
  · ¿Cómo reaccionó en las caídas del mercado? (¿amplificó o amortiguó?)
  · ¿En qué régimen de volatilidad estamos: calma, normal o estrés?

Métricas implementadas
----------------------
  1) Volatilidad móvil (rolling) de 21 días  → "corto plazo" (~1 mes bursátil)
  2) Volatilidad móvil (rolling) de 60 días  → "medio plazo" (~1 trimestre)
  3) Volatilidad EWMA (λ=0.94, estándar RiskMetrics) → da más peso a lo reciente,
     reacciona antes que la rolling simple a un cambio de régimen
  4) Percentil actual de volatilidad → dónde estás hoy dentro de tu propio histórico
  5) Ratio corto/largo → detector de aceleración del riesgo
  6) Volatilidad condicional en caídas (downside) → el riesgo que de verdad duele

IMPORTANTE — límite conceptual: TODO esto es retrospectivo. Mide lo que YA pasó.
No predice. El componente predictivo (volatilidad implícita de opciones) es el
Módulo 6, y ambos deben leerse juntos: la histórica dice dónde has estado, la
implícita dice dónde cree el mercado que vas a estar.

Uso rápido
----------
    from m5_volatilidad_historica import analizar_volatilidad
    vol = analizar_volatilidad(retornos, ventanas=(21, 60))
    print(vol.resumen)
    print(vol.regimen)
===============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("quant.volatilidad")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

DIAS_HABILES_ANIO: int = 252
LAMBDA_RISKMETRICS: float = 0.94  # factor de decaimiento estándar de J.P. Morgan


class ErrorDeVolatilidad(Exception):
    """Se lanza cuando no hay datos suficientes para estimar volatilidad."""


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoVolatilidad:
    """Empaqueta todas las salidas para el módulo gráfico y para inspección."""
    rolling: dict[int, pd.DataFrame]   # {21: DataFrame, 60: DataFrame} anualizadas
    ewma: pd.DataFrame                 # volatilidad EWMA anualizada, por activo
    resumen: pd.DataFrame              # ficha por activo: actual, media, percentil, ratio
    regimen: pd.DataFrame              # clasificación calma/normal/estrés por activo
    downside: pd.DataFrame             # volatilidad solo de los días negativos
    vol_cartera: pd.Series | None      # volatilidad rolling de TU cartera, si se pasaron pesos
    ventanas: tuple[int, ...]
    alertas: list[str] = field(default_factory=list)


# =============================================================================
# VALIDACIÓN
# =============================================================================

def _validar(retornos: pd.DataFrame, ventana_max: int) -> pd.DataFrame:
    if not isinstance(retornos, pd.DataFrame) or retornos.empty:
        raise ErrorDeVolatilidad("Se esperaba un DataFrame de retornos no vacío.")

    limpio = retornos.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(limpio) <= ventana_max:
        raise ErrorDeVolatilidad(
            f"Solo hay {len(limpio)} observaciones y la ventana mayor es de {ventana_max} días. "
            "Amplía el histórico o reduce las ventanas."
        )

    constantes = limpio.columns[limpio.std(ddof=1) < 1e-12].tolist()
    if constantes:
        logger.warning("Activos sin variación (volatilidad cero), se excluyen: %s", constantes)
        limpio = limpio.drop(columns=constantes)
        if limpio.empty:
            raise ErrorDeVolatilidad("Todos los activos tienen volatilidad cero.")
    return limpio


# =============================================================================
# 1-2) VOLATILIDAD MÓVIL (ROLLING)
# =============================================================================

def volatilidad_rolling(
    retornos: pd.DataFrame,
    ventana: int = 21,
    anualizada: bool = True,
) -> pd.DataFrame:
    """Desviación estándar móvil de los retornos, anualizada.

    σ_anual = σ_diaria · √252

    Ventana 21 ≈ 1 mes bursátil: reacciona rápido, pero es ruidosa.
    Ventana 60 ≈ 1 trimestre: más estable, pero tarda más en detectar cambios.
    Usar ambas y compararlas es lo que revela si el riesgo está ACELERANDO.
    """
    min_periodos = max(5, int(ventana * 0.8))
    vol = retornos.rolling(window=ventana, min_periods=min_periodos).std(ddof=1)
    if anualizada:
        vol = vol * np.sqrt(DIAS_HABILES_ANIO)
    return vol.dropna(how="all")


# =============================================================================
# 3) VOLATILIDAD EWMA (RiskMetrics)
# =============================================================================

def volatilidad_ewma(
    retornos: pd.DataFrame,
    lambda_decay: float = LAMBDA_RISKMETRICS,
    anualizada: bool = True,
) -> pd.DataFrame:
    """Volatilidad con media móvil exponencialmente ponderada.

        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

    A diferencia de la rolling simple (que trata igual al dato de hace 1 día y
    al de hace 60), la EWMA pondera más lo reciente. Con λ=0.94, la "vida media"
    de la información es de ~11 días: detecta un cambio de régimen mucho antes,
    que es justo lo que se necesita para gestión de riesgo real.
    """
    if not 0 < lambda_decay < 1:
        raise ErrorDeVolatilidad("lambda_decay debe estar entre 0 y 1 (típico: 0.94).")

    # alpha de pandas = 1 - lambda
    vol = retornos.ewm(alpha=1 - lambda_decay, adjust=False).std(bias=False)
    if anualizada:
        vol = vol * np.sqrt(DIAS_HABILES_ANIO)
    return vol.dropna(how="all")


# =============================================================================
# 6) VOLATILIDAD DOWNSIDE (semi-desviación)
# =============================================================================

def volatilidad_downside(
    retornos: pd.DataFrame,
    umbral: float = 0.0,
    anualizada: bool = True,
) -> pd.DataFrame:
    """Volatilidad calculada SOLO con los días por debajo del umbral (normalmente 0).

    La desviación estándar clásica penaliza igual una subida del 3% que una
    caída del 3%, pero al inversor solo le duele la segunda. Esta métrica es la
    base del Ratio de Sortino y suele ser más informativa que la vol total
    para activos con retornos asimétricos.
    """
    filas = []
    for activo in retornos.columns:
        r = retornos[activo]
        malos = r[r < umbral]

        if len(malos) < 10:
            semi_desv = np.nan
        else:
            # Semi-desviación respecto al umbral, no respecto a la media
            semi_desv = np.sqrt(np.mean((malos - umbral) ** 2))
            if anualizada:
                semi_desv *= np.sqrt(DIAS_HABILES_ANIO)

        vol_total = r.std(ddof=1) * (np.sqrt(DIAS_HABILES_ANIO) if anualizada else 1)

        filas.append({
            "Activo": activo,
            "Vol. total %": vol_total * 100,
            "Vol. downside %": semi_desv * 100 if pd.notna(semi_desv) else np.nan,
            "Ratio downside/total": semi_desv / vol_total if pd.notna(semi_desv) and vol_total > 0 else np.nan,
            "Días negativos %": len(malos) / len(r) * 100,
        })

    return pd.DataFrame(filas).set_index("Activo").round(3)


# =============================================================================
# 4-5) RESUMEN, PERCENTILES Y RÉGIMEN
# =============================================================================

def resumen_volatilidad(
    rolling: dict[int, pd.DataFrame],
    ewma: pd.DataFrame,
    ventana_corta: int,
    ventana_larga: int,
) -> pd.DataFrame:
    """Ficha por activo cruzando corto plazo, largo plazo y posición histórica.

    Columnas clave:
      · 'Percentil actual': si es 90, la volatilidad de hoy es más alta que el
        90% de los días de los últimos 3 años → estás en la cola de estrés.
      · 'Ratio corto/largo': >1.2 significa que el riesgo se está ACELERANDO
        (el corto plazo se ha despegado del medio plazo).
    """
    vol_corta = rolling[ventana_corta]
    vol_larga = rolling[ventana_larga]

    filas = []
    for activo in vol_corta.columns:
        serie_corta = vol_corta[activo].dropna()
        serie_larga = vol_larga[activo].dropna()
        if serie_corta.empty or serie_larga.empty:
            continue

        actual_corta = serie_corta.iloc[-1]
        actual_larga = serie_larga.iloc[-1]
        actual_ewma = ewma[activo].dropna().iloc[-1] if activo in ewma.columns else np.nan

        # Percentil de la vol actual dentro de su propio histórico
        percentil = (serie_larga < actual_larga).mean() * 100

        filas.append({
            "Activo": activo,
            f"Vol {ventana_corta}d %": actual_corta * 100,
            f"Vol {ventana_larga}d %": actual_larga * 100,
            "Vol EWMA %": actual_ewma * 100,
            f"Media {ventana_larga}d %": serie_larga.mean() * 100,
            f"Mín {ventana_larga}d %": serie_larga.min() * 100,
            f"Máx {ventana_larga}d %": serie_larga.max() * 100,
            "Percentil actual": percentil,
            "Ratio corto/largo": actual_corta / actual_larga if actual_larga > 1e-12 else np.nan,
        })

    if not filas:
        raise ErrorDeVolatilidad("No se pudo calcular el resumen: series de volatilidad vacías.")
    return pd.DataFrame(filas).set_index("Activo").round(3)


def clasificar_regimen(
    resumen: pd.DataFrame,
    umbral_calma: float = 25.0,
    umbral_estres: float = 75.0,
) -> pd.DataFrame:
    """Traduce el percentil de volatilidad a un régimen legible.

    · CALMA   (percentil < 25): volatilidad baja frente a su propio histórico.
              Ojo: la calma prolongada suele preceder a los repuntes bruscos.
    · NORMAL  (25-75): rango habitual del activo.
    · ESTRÉS  (> 75): el mercado está descontando o sufriendo algo.
    """
    filas = []
    for activo, fila in resumen.iterrows():
        p = fila["Percentil actual"]
        if p < umbral_calma:
            regimen = "CALMA"
        elif p > umbral_estres:
            regimen = "ESTRÉS"
        else:
            regimen = "NORMAL"

        ratio = fila["Ratio corto/largo"]
        if pd.isna(ratio):
            tendencia = "—"
        elif ratio > 1.20:
            tendencia = "ACELERANDO ↑"
        elif ratio < 0.83:
            tendencia = "ENFRIANDO ↓"
        else:
            tendencia = "ESTABLE →"

        filas.append({
            "Activo": activo,
            "Régimen": regimen,
            "Percentil": round(p, 1),
            "Tendencia": tendencia,
            "Ratio corto/largo": round(ratio, 2) if pd.notna(ratio) else np.nan,
        })

    return pd.DataFrame(filas).set_index("Activo")


# =============================================================================
# VOLATILIDAD DE TU CARTERA (no de los activos sueltos)
# =============================================================================

def volatilidad_cartera_rolling(
    retornos: pd.DataFrame,
    pesos: pd.Series,
    ventana: int = 60,
) -> pd.Series:
    """Volatilidad móvil de la CARTERA COMPLETA con unos pesos fijos.

    No es la media de las volatilidades individuales: gracias a la
    diversificación, la volatilidad de la cartera es (casi siempre) MENOR que
    la media ponderada de las volatilidades de sus componentes. La diferencia
    entre ambas es, literalmente, el beneficio que te está dando diversificar.
    """
    pesos_alineados = pesos.reindex(retornos.columns).fillna(0.0)
    suma = pesos_alineados.sum()
    if suma <= 0:
        raise ErrorDeVolatilidad("Los pesos de la cartera suman 0 o menos.")
    pesos_alineados = pesos_alineados / suma

    retornos_cartera = (retornos * pesos_alineados).sum(axis=1)
    min_periodos = max(5, int(ventana * 0.8))
    vol = retornos_cartera.rolling(window=ventana, min_periods=min_periodos).std(ddof=1)
    return (vol * np.sqrt(DIAS_HABILES_ANIO)).dropna().rename(f"Cartera ({ventana}d)")


def beneficio_diversificacion(
    retornos: pd.DataFrame, pesos: pd.Series, ventana: int = 60,
) -> pd.DataFrame:
    """Compara, día a día, la volatilidad REAL de la cartera contra la media
    ponderada de las volatilidades individuales (el caso hipotético en que
    todos los activos estuvieran perfectamente correlacionados).

    La columna 'Ahorro %' es el porcentaje de riesgo que te ahorra diversificar.
    """
    pesos_alineados = pesos.reindex(retornos.columns).fillna(0.0)
    pesos_alineados = pesos_alineados / pesos_alineados.sum()

    vol_real = volatilidad_cartera_rolling(retornos, pesos_alineados, ventana)
    vol_individuales = volatilidad_rolling(retornos, ventana)
    vol_ponderada = (vol_individuales * pesos_alineados).sum(axis=1)

    comparativa = pd.DataFrame({
        "Vol. real cartera %": vol_real * 100,
        "Vol. sin diversificar %": vol_ponderada.reindex(vol_real.index) * 100,
    })
    comparativa["Ahorro %"] = (
        (1 - comparativa["Vol. real cartera %"] / comparativa["Vol. sin diversificar %"]) * 100
    )
    return comparativa.dropna().round(3)


# =============================================================================
# DIAGNÓSTICO AUTOMÁTICO
# =============================================================================

def generar_alertas_volatilidad(resumen: pd.DataFrame, regimen: pd.DataFrame) -> list[str]:
    """Convierte las tablas numéricas en avisos en lenguaje natural."""
    alertas: list[str] = []

    for activo, fila in regimen.iterrows():
        if fila["Régimen"] == "ESTRÉS":
            alertas.append(
                f"⚠ {activo}: volatilidad en el percentil {fila['Percentil']:.0f} de su histórico "
                "→ régimen de estrés. Las estimaciones de riesgo del Módulo 3 (basadas en la "
                "media de 3 años) están INFRAESTIMANDO su riesgo actual."
            )
        elif fila["Régimen"] == "CALMA":
            alertas.append(
                f"ℹ {activo}: volatilidad en el percentil {fila['Percentil']:.0f} (calma). "
                "Cuidado: los periodos de calma prolongada suelen preceder repuntes bruscos."
            )
        if fila["Tendencia"] == "ACELERANDO ↑":
            alertas.append(
                f"⚠ {activo}: la volatilidad de corto plazo se ha despegado de la de medio plazo "
                f"(ratio {fila['Ratio corto/largo']:.2f}) → el riesgo está acelerando AHORA."
            )

    # Activo más y menos volátil
    columna_vol = [c for c in resumen.columns if c.startswith("Vol ") and c.endswith("d %")]
    if columna_vol:
        col = columna_vol[-1]
        mas = resumen[col].idxmax()
        menos = resumen[col].idxmin()
        alertas.append(
            f"ℹ Rango de riesgo actual: {mas} es el más volátil ({resumen.loc[mas, col]:.1f}%) "
            f"y {menos} el más tranquilo ({resumen.loc[menos, col]:.1f}%). "
            f"Proporción: {resumen.loc[mas, col] / resumen.loc[menos, col]:.1f}x — "
            "esta es exactamente la ratio que Risk Parity usa para repartir capital."
        )

    if not alertas:
        alertas.append("✓ Todos los activos en régimen de volatilidad normal y estable.")
    return alertas


# =============================================================================
# ORQUESTADOR DEL MÓDULO
# =============================================================================

def analizar_volatilidad(
    retornos: pd.DataFrame,
    ventanas: tuple[int, ...] = (21, 60),
    pesos_cartera: pd.Series | None = None,
    lambda_ewma: float = LAMBDA_RISKMETRICS,
) -> ResultadoVolatilidad:
    """Ejecuta el análisis completo de volatilidad histórica.

    Parameters
    ----------
    ventanas : ventanas rolling a calcular. La primera se trata como "corto plazo"
        y la última como "medio/largo plazo" para los ratios y percentiles.
    pesos_cartera : si se pasan (p.ej. mi_cartera.pesos del Módulo 3), calcula
        también la volatilidad rolling de TU cartera completa.
    """
    if len(ventanas) < 2:
        raise ErrorDeVolatilidad("Se necesitan al menos 2 ventanas (p.ej. 21 y 60) para comparar.")

    ventanas = tuple(sorted(ventanas))
    ret = _validar(retornos, max(ventanas))

    rolling = {v: volatilidad_rolling(ret, v) for v in ventanas}
    ewma = volatilidad_ewma(ret, lambda_ewma)
    resumen = resumen_volatilidad(rolling, ewma, ventanas[0], ventanas[-1])
    regimen = clasificar_regimen(resumen)
    downside = volatilidad_downside(ret)
    alertas = generar_alertas_volatilidad(resumen, regimen)

    vol_cartera = None
    if pesos_cartera is not None:
        vol_cartera = volatilidad_cartera_rolling(ret, pesos_cartera, ventanas[-1])

    logger.info(
        "Volatilidad histórica calculada · %d activos · ventanas %s · EWMA λ=%.2f",
        ret.shape[1], ventanas, lambda_ewma,
    )

    return ResultadoVolatilidad(
        rolling=rolling,
        ewma=ewma,
        resumen=resumen,
        regimen=regimen,
        downside=downside,
        vol_cartera=vol_cartera,
        ventanas=ventanas,
        alertas=alertas,
    )


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m5_volatilidad_historica.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 25)

    try:
        from m1_datos import calcular_retornos, descargar_precios

        # 👇 tus tickers
        mis_tickers = ["RHM.DE", "IDR.MC", "MC.PA", "0P0001KGI5.F", "EUNL.DE"]

        # 👇 pesos reales de tu cartera (calculados a partir de los importes en euros)
        mis_importes = {
            "RHM.DE": 1955.0,
            "IDR.MC": 1670.0,
            "MC.PA": 1095.0,
            "0P0001KGI5.F": 2176.06,
            "EUNL.DE": 531.0,
        }
        total = sum(mis_importes.values())
        pesos_cartera = pd.Series({t: v / total for t, v in mis_importes.items()})

        precios = descargar_precios(mis_tickers, anios=3, cache_dir="./cache_datos")
        retornos = calcular_retornos(precios)
        vol = analizar_volatilidad(retornos, ventanas=(21, 60), pesos_cartera=pesos_cartera)

        print("\n" + "=" * 78)
        print("MÓDULO 5 · VOLATILIDAD HISTÓRICA")
        print("=" * 78)
        print("\n--- Resumen por activo ---")
        print(vol.resumen)
        print("\n--- Régimen actual ---")
        print(vol.regimen)
        print("\n--- Volatilidad downside (la que duele) ---")
        print(vol.downside)
        print("\n--- Últimos 5 días de volatilidad rolling 21d ---")
        print((vol.rolling[21].tail() * 100).round(2))
        print("\n--- Diagnóstico ---")
        for a in vol.alertas:
            print("  " + a)
        if vol.vol_cartera is not None:
            print("\n--- Volatilidad rolling de TU CARTERA (60d, últimos 5 días) ---")
            print((vol.vol_cartera.tail() * 100).round(2))
        print("\n[OK] Módulo 5 ejecutado sin errores.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
