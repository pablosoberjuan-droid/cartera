"""
===============================================================================
 MÓDULO 8 — COMPARATIVA DE VOLATILIDAD IMPLÍCITA ENTRE PROXIES
===============================================================================
El Módulo 7 analiza UN subyacente. Este ejecuta el análisis sobre VARIOS
proxies a la vez y los pone en una sola tabla y un solo gráfico, que es donde
aparece la información realmente accionable.

POR QUÉ IMPORTA
---------------
Saber que SPY cotiza una IV del 13% no dice nada por sí solo. Lo que dice algo es:

  · ITA (defensa) al 28% vs SPY al 13% → el mercado espera que el sector defensa
    se mueva MÁS DEL DOBLE que el mercado general. Ese es el riesgo real de tu
    posición en Rheinmetall, y no lo ves mirando solo el S&P.
  · EZU (eurozona) por encima de SPY → el riesgo percibido está en Europa, no
    en EE.UU. Relevante para IDR.MC y MC.PA.
  · QQQ vs SPY → cuánta prima extra de riesgo tiene la tecnología (tu fondo Polar).

MAPEO PROXY → TU CARTERA
------------------------
  SPY  → riesgo de mercado global (todos)
  ITA  → RHM.DE (defensa)
  EZU  → IDR.MC, MC.PA (eurozona)
  URTH → EUNL.DE (MSCI World)
  QQQ  → Polar Capital Global Tech
  EWY  → KOSPI (benchmark que pediste)

Uso rápido
----------
    from m8_comparativa_proxies import analizar_todos_los_proxies

    comp = analizar_todos_los_proxies(["SPY", "ITA", "EZU", "URTH", "QQQ"])
    print(comp.tabla)
    print(comp.mapeo_cartera)
===============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from m7_superficie_iv import (
    PROXIES_OPCIONES,
    ErrorDeSuperficie,
    ResultadoSuperficie,
    analizar_superficie,
)

logger = logging.getLogger("quant.comparativa_iv")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

sns.set_theme(style="whitegrid", font_scale=0.95)


# Qué activo de TU cartera cubre cada proxy
MAPEO_CARTERA: dict[str, list[str]] = {
    "SPY":  ["(riesgo de mercado global — afecta a todos)"],
    "ITA":  ["RHM.DE"],
    "EZU":  ["IDR.MC", "MC.PA"],
    "URTH": ["EUNL.DE"],
    "QQQ":  ["0P0001KGI5.F (Polar Capital Global Tech)"],
    "EWY":  ["(benchmark KOSPI)"],
}


@dataclass
class ResultadoComparativa:
    """Empaqueta los resultados de todos los proxies analizados."""
    superficies: dict[str, ResultadoSuperficie]
    tabla: pd.DataFrame                  # una fila por proxy con sus métricas clave
    mapeo_cartera: pd.DataFrame          # qué activo tuyo cubre cada proxy
    fallidos: dict[str, str]             # proxy -> motivo del fallo
    alertas: list[str] = field(default_factory=list)


# =============================================================================
# EJECUCIÓN MULTI-PROXY
# =============================================================================

def analizar_todos_los_proxies(
    proxies: list[str] | None = None,
    max_vencimientos: int = 8,
    vol_realizada: dict[str, float] | None = None,
    guardar_individuales: bool = False,
    continuar_si_falla: bool = True,
) -> ResultadoComparativa:
    """Ejecuta el análisis de superficie sobre varios proxies y los compara.

    Parameters
    ----------
    proxies : lista de tickers. Por defecto, todos los del catálogo del Módulo 7.
    vol_realizada : dict opcional {proxy: vol_histórica_%} para calcular la prima
        de riesgo de volatilidad de cada uno.
    guardar_individuales : si True, guarda además el PNG/HTML de cada superficie.
        Con 5-6 proxies genera muchos archivos; por defecto está desactivado.
    continuar_si_falla : si un proxy falla (sin opciones, ilíquido, error de red),
        se registra y se sigue con los demás en vez de abortar todo el análisis.
        Recomendado dejarlo en True.
    """
    proxies = proxies or list(PROXIES_OPCIONES.keys())
    vol_realizada = vol_realizada or {}

    superficies: dict[str, ResultadoSuperficie] = {}
    fallidos: dict[str, str] = {}

    for ticker in proxies:
        logger.info("Analizando %s...", ticker)
        try:
            resultado = analizar_superficie(
                ticker,
                max_vencimientos=max_vencimientos,
                vol_realizada=vol_realizada.get(ticker),
                guardar_png=guardar_individuales,
                guardar_html=guardar_individuales,
            )
            superficies[ticker] = resultado
        except ErrorDeSuperficie as exc:
            fallidos[ticker] = str(exc)
            logger.warning("%s descartado: %s", ticker, str(exc)[:120])
            if not continuar_si_falla:
                raise
        except Exception as exc:
            fallidos[ticker] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s falló inesperadamente: %s", ticker, exc)
            if not continuar_si_falla:
                raise

    if not superficies:
        raise ErrorDeSuperficie(
            f"Ningún proxy se pudo analizar. Errores: {fallidos}"
        )

    tabla = construir_tabla_comparativa(superficies)
    mapeo = construir_mapeo_cartera(superficies)
    alertas = generar_alertas_comparativa(tabla)

    logger.info(
        "Comparativa completada · %d proxies OK · %d fallidos",
        len(superficies), len(fallidos),
    )

    return ResultadoComparativa(
        superficies=superficies, tabla=tabla, mapeo_cartera=mapeo,
        fallidos=fallidos, alertas=alertas,
    )


# =============================================================================
# TABLAS COMPARATIVAS
# =============================================================================

def construir_tabla_comparativa(superficies: dict[str, ResultadoSuperficie]) -> pd.DataFrame:
    """Una fila por proxy con sus métricas clave, ordenada por IV ATM descendente.

    La columna 'vs SPY' es la lectura más directa: un valor de 2.1 significa que
    el mercado espera que ese subyacente se mueva 2,1 veces más que el S&P 500.
    """
    filas = []
    for ticker, sup in superficies.items():
        met = sup.metricas
        filas.append({
            "Proxy": ticker,
            "Nombre": PROXIES_OPCIONES.get(ticker, {}).get("nombre", ticker),
            "Spot": round(sup.spot, 2),
            "IV ATM %": met.get("IV ATM referencia %"),
            "Skew": met.get("Skew (put0.90 - call1.10)"),
            "Pendiente temporal": met.get("Pendiente temporal (largo - corto)"),
            "IV media %": met.get("IV media superficie %"),
            "Contratos": int(met.get("Contratos usados", 0)),
        })

    tabla = pd.DataFrame(filas).set_index("Proxy")

    # Ratio frente a SPY: el "cuántas veces más nervioso que el mercado"
    if "SPY" in tabla.index and pd.notna(tabla.loc["SPY", "IV ATM %"]):
        iv_spy = tabla.loc["SPY", "IV ATM %"]
        tabla["vs SPY (×)"] = (tabla["IV ATM %"] / iv_spy).round(2)

    return tabla.sort_values("IV ATM %", ascending=False).round(2)


def construir_mapeo_cartera(superficies: dict[str, ResultadoSuperficie]) -> pd.DataFrame:
    """Traduce cada proxy al activo de TU cartera cuyo riesgo aproxima."""
    filas = []
    for ticker, sup in superficies.items():
        filas.append({
            "Proxy": ticker,
            "Cubre en tu cartera": ", ".join(MAPEO_CARTERA.get(ticker, ["—"])),
            "IV ATM %": sup.metricas.get("IV ATM referencia %"),
            "Lectura": _interpretar_nivel_iv(sup.metricas.get("IV ATM referencia %")),
        })
    return pd.DataFrame(filas).set_index("Proxy")


def _interpretar_nivel_iv(iv: float | None) -> str:
    """Traduce un nivel de IV a lenguaje llano.

    Referencias aproximadas para renta variable desarrollada: por debajo del 15%
    es un mercado tranquilo; por encima del 30% hay estrés real descontado.
    """
    if iv is None or pd.isna(iv):
        return "—"
    if iv < 12:
        return "Muy tranquilo (complacencia)"
    if iv < 18:
        return "Normal"
    if iv < 28:
        return "Elevado"
    if iv < 40:
        return "Estrés"
    return "Pánico / evento binario"


def generar_alertas_comparativa(tabla: pd.DataFrame) -> list[str]:
    """Diagnóstico automático de la comparativa entre proxies."""
    alertas: list[str] = []

    if tabla.empty:
        return ["No hay datos suficientes para comparar."]

    mas_nervioso = tabla["IV ATM %"].idxmax()
    mas_tranquilo = tabla["IV ATM %"].idxmin()
    alertas.append(
        f"ℹ El mercado espera más movimiento en {mas_nervioso} "
        f"({tabla.loc[mas_nervioso, 'IV ATM %']:.1f}%) y menos en {mas_tranquilo} "
        f"({tabla.loc[mas_tranquilo, 'IV ATM %']:.1f}%)."
    )

    if "vs SPY (×)" in tabla.columns:
        for proxy, fila in tabla.iterrows():
            ratio = fila.get("vs SPY (×)")
            if proxy == "SPY" or pd.isna(ratio):
                continue
            if ratio >= 1.8:
                activos = ", ".join(MAPEO_CARTERA.get(proxy, ["—"]))
                alertas.append(
                    f"⚠ {proxy} cotiza {ratio:.1f}× la volatilidad implícita del S&P 500. "
                    f"Si tienes {activos}, tu riesgo real es muy superior al de un índice global."
                )

    invertidas = tabla[tabla["Pendiente temporal"] < -2]
    for proxy, fila in invertidas.iterrows():
        alertas.append(
            f"⚠ {proxy}: estructura temporal invertida ({fila['Pendiente temporal']:+.1f}). "
            "El mercado descuenta un evento a corto plazo en ese activo."
        )

    skew_alto = tabla[tabla["Skew"] > 10]
    for proxy, fila in skew_alto.iterrows():
        alertas.append(
            f"⚠ {proxy}: skew de {fila['Skew']:+.1f} puntos. Demanda intensa de protección "
            "frente a caídas en ese subyacente."
        )

    if len(alertas) == 1:
        alertas.append("✓ Sin anomalías destacables entre los proxies analizados.")
    return alertas


# =============================================================================
# GRÁFICOS COMPARATIVOS
# =============================================================================

def plot_comparativa_estructura_temporal(
    superficies: dict[str, ResultadoSuperficie],
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 5.5),
) -> Figure:
    """Estructura temporal de TODOS los proxies en un mismo eje.

    La separación vertical entre curvas es la prima de riesgo relativa: cuánto
    más caro es protegerse en un sector que en otro, a cada plazo.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    paleta = sns.color_palette("tab10", len(superficies))
    for color, (ticker, sup) in zip(paleta, superficies.items()):
        ax.plot(
            sup.resumen["dias"], sup.resumen["IV_ATM %"], "o-",
            color=color, linewidth=2, markersize=5, label=ticker,
        )

    ax.set_xlabel("Días al vencimiento")
    ax.set_ylabel("IV At-The-Money (%)")
    ax.set_title("Estructura temporal comparada entre proxies", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, title="Subyacente")
    ax.grid(alpha=0.3)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_comparativa_smile(
    superficies: dict[str, ResultadoSuperficie],
    dias_objetivo: int = 30,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 5.5),
) -> Figure:
    """Sonrisa/skew de todos los proxies al mismo plazo (≈30 días).

    Compararlos al MISMO vencimiento es imprescindible: el skew se aplana con el
    tiempo, así que cruzar el skew a 7 días de uno con el de 90 días de otro no
    tiene ningún significado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    paleta = sns.color_palette("tab10", len(superficies))
    for color, (ticker, sup) in zip(paleta, superficies.items()):
        dias_disp = sup.cadena["dias"].unique()
        if len(dias_disp) == 0:
            continue
        dias_elegido = int(dias_disp[np.argmin(np.abs(dias_disp - dias_objetivo))])
        corte = sup.cadena[sup.cadena["dias"] == dias_elegido].sort_values("moneyness")
        ax.plot(
            corte["moneyness"], corte["iv"] * 100, "-",
            color=color, linewidth=2, label=f"{ticker} ({dias_elegido}d)",
        )

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Moneyness (Strike / Spot)")
    ax.set_ylabel("Volatilidad implícita (%)")
    ax.set_title(f"Skew comparado (≈{dias_objetivo} días)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_barras_iv(
    tabla: pd.DataFrame,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 5),
) -> Figure:
    """Barras horizontales de IV ATM por proxy, con la línea de referencia de SPY."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    datos = tabla.sort_values("IV ATM %")
    colores = ["#C00000" if v >= 28 else "#FFC000" if v >= 18 else "#00B050"
               for v in datos["IV ATM %"]]

    ax.barh(datos.index, datos["IV ATM %"], color=colores, edgecolor="white")
    for i, (proxy, fila) in enumerate(datos.iterrows()):
        etiqueta = f"{fila['IV ATM %']:.1f}%"
        if "vs SPY (×)" in datos.columns and proxy != "SPY" and pd.notna(fila.get("vs SPY (×)")):
            etiqueta += f"  ({fila['vs SPY (×)']:.1f}× SPY)"
        ax.text(fila["IV ATM %"] + 0.4, i, etiqueta, va="center", fontsize=8)

    if "SPY" in tabla.index:
        ax.axvline(tabla.loc["SPY", "IV ATM %"], color="black",
                   linestyle="--", linewidth=1.2, label="SPY (mercado)")
        ax.legend(fontsize=8)

    ax.set_xlabel("Volatilidad implícita At-The-Money (%)")
    ax.set_title("Expectativa de movimiento por subyacente", fontsize=11, fontweight="bold")
    ax.set_xlim(0, datos["IV ATM %"].max() * 1.35)

    if fig is not None:
        fig.tight_layout()
    return fig


def panel_comparativa_proxies(
    resultado: ResultadoComparativa,
    dias_objetivo: int = 30,
    guardar: str | None = "panel_proxies_iv.png",
    dpi: int = 150,
) -> Figure:
    """Panel compuesto con las tres vistas comparativas."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    ax_barras = fig.add_subplot(gs[0, :])
    plot_barras_iv(resultado.tabla, ax=ax_barras)

    ax_termino = fig.add_subplot(gs[1, 0])
    plot_comparativa_estructura_temporal(resultado.superficies, ax=ax_termino)

    ax_smile = fig.add_subplot(gs[1, 1])
    plot_comparativa_smile(resultado.superficies, dias_objetivo=dias_objetivo, ax=ax_smile)

    fig.suptitle(
        "Fase 3 — Volatilidad implícita comparada entre proxies",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Panel comparativo guardado en '%s'", guardar)
    return fig


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m8_comparativa_proxies.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 25)

    try:
        comp = analizar_todos_los_proxies(
            ["SPY", "ITA", "EZU", "URTH", "QQQ", "EWY"],
            max_vencimientos=7,
        )

        print("\n" + "=" * 78)
        print("MÓDULO 8 · COMPARATIVA DE VOLATILIDAD IMPLÍCITA")
        print("=" * 78)
        print("\n--- Tabla comparativa ---")
        print(comp.tabla.to_string())
        print("\n--- Qué cubre cada proxy de tu cartera ---")
        print(comp.mapeo_cartera.to_string())

        if comp.fallidos:
            print("\n--- Proxies no analizados ---")
            for ticker, motivo in comp.fallidos.items():
                print(f"  {ticker}: {motivo[:110]}")

        print("\n--- Diagnóstico ---")
        for a in comp.alertas:
            print("  " + a)

        panel_comparativa_proxies(comp, guardar="panel_proxies_iv.png")
        print("\n[OK] Módulo 8 ejecutado. Panel guardado en 'panel_proxies_iv.png'.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
