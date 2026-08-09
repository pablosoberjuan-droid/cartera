"""
===============================================================================
 MÓDULO 2 — ANÁLISIS DE CORRELACIÓN (estática y dinámica)
===============================================================================
La diversificación de Markowitz solo funciona si las correlaciones son bajas.
Este módulo mide esa correlación desde tres ángulos complementarios:

  1) PEARSON  (ρ)  → relación LINEAL. Es la que alimenta la matriz de
                     covarianzas del optimizador. Sensible a valores extremos.
  2) SPEARMAN (ρs) → relación MONÓTONA (no lineal), calculada sobre rangos.
                     Robusta frente a crashes y colas gruesas.
     · Si |Spearman| >> |Pearson| hay dependencia no lineal que Markowitz NO ve.
     · Si |Pearson| >> |Spearman| la correlación la están creando 4 días extremos.

  3) ROLLING 60 días → la correlación NO es constante. En las crisis tiende a 1
                     justo cuando más falta hace la diversificación
                     ("correlation breakdown"). Este gráfico lo demuestra.

Depende del Módulo 1 solo por la constante DIAS_HABILES_ANIO (import opcional).

Uso rápido
----------
    from m2_correlaciones import analizar_correlaciones
    corr = analizar_correlaciones(retornos, ventana=60)
    print(corr.pearson)
    print(corr.comparativa)
===============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("quant.correlaciones")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoCorrelacion:
    """Empaqueta todas las salidas del módulo para pasarlas al módulo gráfico."""
    pearson: pd.DataFrame           # matriz NxN lineal
    spearman: pd.DataFrame          # matriz NxN por rangos
    rolling: pd.DataFrame           # una columna por par: 'SPY-TLT', ...
    correlacion_media: pd.Series    # media de todos los pares, día a día
    comparativa: pd.DataFrame       # tabla par a par ordenada
    ventana: int = 60
    alertas: list[str] = field(default_factory=list)

    def par(self, a: str, b: str) -> pd.Series:
        """Devuelve la serie rolling de un par concreto en cualquier orden."""
        for clave in (f"{a}-{b}", f"{b}-{a}"):
            if clave in self.rolling.columns:
                return self.rolling[clave]
        raise KeyError(f"El par {a}-{b} no existe. Disponibles: {list(self.rolling.columns)}")


# =============================================================================
# VALIDACIÓN
# =============================================================================

def _validar_retornos(retornos: pd.DataFrame, ventana: int) -> pd.DataFrame:
    if not isinstance(retornos, pd.DataFrame):
        raise TypeError("Se esperaba un DataFrame de retornos (filas=fechas, cols=activos).")
    if retornos.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 activos para calcular correlaciones.")

    limpio = retornos.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if limpio.empty:
        raise ValueError("La matriz de retornos quedó vacía tras eliminar NaN/inf.")
    if len(limpio) <= ventana:
        raise ValueError(
            f"Solo hay {len(limpio)} observaciones y la ventana rolling es de {ventana}. "
            "Reduce la ventana o amplía el histórico."
        )

    # Un activo de varianza cero (p.ej. un fondo monetario mal ajustado) rompe la correlación
    constantes = limpio.columns[limpio.std(ddof=1) < 1e-12].tolist()
    if constantes:
        logger.warning("Activos sin variación, se excluyen del análisis: %s", constantes)
        limpio = limpio.drop(columns=constantes)
        if limpio.shape[1] < 2:
            raise ValueError("Tras excluir activos constantes quedan menos de 2 activos.")
    return limpio


# =============================================================================
# MATRICES ESTÁTICAS
# =============================================================================

def matriz_correlacion(
    retornos: pd.DataFrame,
    metodo: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """Matriz de correlación NxN. 'pearson' = lineal, 'spearman' = por rangos."""
    return retornos.corr(method=metodo)


# =============================================================================
# CORRELACIÓN DINÁMICA (ROLLING)
# =============================================================================

def correlacion_rolling(
    retornos: pd.DataFrame,
    ventana: int = 60,
    min_periodos: int | None = None,
) -> pd.DataFrame:
    """Correlación de Pearson móvil de `ventana` sesiones para CADA par de activos.

    Returns
    -------
    DataFrame con índice de fechas y una columna por par ('SPY-TLT', 'SPY-GLD'...).
    Con N activos hay N*(N-1)/2 pares.
    """
    if min_periodos is None:
        min_periodos = max(10, int(ventana * 0.8))

    series: dict[str, pd.Series] = {}
    for a, b in combinations(retornos.columns, 2):
        series[f"{a}-{b}"] = (
            retornos[a].rolling(window=ventana, min_periods=min_periodos).corr(retornos[b])
        )

    roll = pd.DataFrame(series).dropna(how="all")
    if roll.empty:
        raise ValueError("La correlación rolling no produjo datos. Reduce la ventana.")
    return roll


# =============================================================================
# TABLA COMPARATIVA PEARSON vs SPEARMAN
# =============================================================================

def comparar_pearson_spearman(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    rolling: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tabla par a par: Pearson, Spearman, diferencia, y rango histórico rolling.

    La columna 'Δ (Sp-Pe)' es la clave interpretativa:
      Δ > +0.10 → dependencia no lineal: se mueven juntos casi siempre, pero
                  Pearson lo diluye. El riesgo real es MAYOR del que ve Markowitz.
      Δ < -0.10 → la correlación lineal la generan unos pocos días extremos.
    """
    filas = []
    for a, b in combinations(pearson.columns, 2):
        fila = {
            "Activo A": a,
            "Activo B": b,
            "Pearson": pearson.loc[a, b],
            "Spearman": spearman.loc[a, b],
            "Δ (Sp-Pe)": spearman.loc[a, b] - pearson.loc[a, b],
        }
        if rolling is not None:
            clave = f"{a}-{b}"
            if clave in rolling.columns:
                s = rolling[clave].dropna()
                fila.update({
                    "Rolling mín": s.min(),
                    "Rolling máx": s.max(),
                    "Rolling actual": s.iloc[-1],
                })
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    # Ordenamos de menor a mayor Pearson: arriba, los mejores diversificadores
    return tabla.sort_values("Pearson").reset_index(drop=True).round(3)


# =============================================================================
# DIAGNÓSTICO AUTOMÁTICO
# =============================================================================

def generar_alertas(comparativa: pd.DataFrame, umbral_alta: float = 0.80) -> list[str]:
    """Traduce la tabla numérica a avisos en lenguaje natural."""
    alertas: list[str] = []

    for _, f in comparativa.iterrows():
        par = f"{f['Activo A']}-{f['Activo B']}"

        if abs(f["Pearson"]) >= umbral_alta:
            alertas.append(
                f"⚠ {par}: correlación {f['Pearson']:+.2f} (muy alta). "
                "Aportan poca diversificación; el optimizador tenderá a elegir uno u otro "
                "de forma inestable."
            )
        if abs(f["Δ (Sp-Pe)"]) >= 0.10:
            signo = "no lineal oculta" if f["Δ (Sp-Pe)"] > 0 else "concentrada en días extremos"
            alertas.append(
                f"ℹ {par}: Δ(Spearman-Pearson) = {f['Δ (Sp-Pe)']:+.2f} → dependencia {signo}."
            )
        if "Rolling máx" in f and pd.notna(f.get("Rolling máx")):
            recorrido = f["Rolling máx"] - f["Rolling mín"]
            if recorrido >= 1.0:
                alertas.append(
                    f"ℹ {par}: la correlación móvil ha oscilado entre {f['Rolling mín']:+.2f} "
                    f"y {f['Rolling máx']:+.2f}. La diversificación de este par NO es estable."
                )

    if not alertas:
        alertas.append("✓ Sin correlaciones extremas ni anomalías relevantes detectadas.")
    return alertas


# =============================================================================
# FUNCIÓN ORQUESTADORA DEL MÓDULO
# =============================================================================

def analizar_correlaciones(
    retornos: pd.DataFrame,
    ventana: int = 60,
) -> ResultadoCorrelacion:
    """Ejecuta el análisis completo y devuelve un objeto con todos los resultados."""
    ret = _validar_retornos(retornos, ventana)

    pearson = matriz_correlacion(ret, "pearson")
    spearman = matriz_correlacion(ret, "spearman")
    rolling = correlacion_rolling(ret, ventana=ventana)
    media = rolling.mean(axis=1).rename(f"Correlación media ({ventana}d)")
    comparativa = comparar_pearson_spearman(pearson, spearman, rolling)
    alertas = generar_alertas(comparativa)

    logger.info(
        "Correlaciones calculadas · %d activos · %d pares · ventana %dd",
        ret.shape[1], rolling.shape[1], ventana,
    )

    return ResultadoCorrelacion(
        pearson=pearson,
        spearman=spearman,
        rolling=rolling,
        correlacion_media=media,
        comparativa=comparativa,
        ventana=ventana,
        alertas=alertas,
    )


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m2_correlaciones.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 25)

    try:
        from m1_datos import TICKERS_DEFECTO, calcular_retornos, descargar_precios

        precios = descargar_precios(TICKERS_DEFECTO, anios=3, cache_dir="./cache_datos")
        retornos = calcular_retornos(precios)
        res = analizar_correlaciones(retornos, ventana=60)

        print("\n" + "=" * 78)
        print("MÓDULO 2 · CORRELACIONES")
        print("=" * 78)
        print("\n--- Pearson (lineal) ---")
        print(res.pearson.round(3))
        print("\n--- Spearman (no lineal / rangos) ---")
        print(res.spearman.round(3))
        print(f"\n--- Comparativa par a par (rolling {res.ventana}d) ---")
        print(res.comparativa.to_string(index=False))
        print("\n--- Correlación media de la cartera (últimos 5 días) ---")
        print(res.correlacion_media.tail().round(3))
        print("\n--- Diagnóstico ---")
        for a in res.alertas:
            print("  " + a)
        print("\n[OK] Módulo 2 ejecutado sin errores.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
