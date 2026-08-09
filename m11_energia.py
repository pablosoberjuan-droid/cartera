"""
===============================================================================
 MÓDULO 11 — COMPLEJO ENERGÉTICO: VOLATILIDAD Y SUPERFICIES
===============================================================================
Analiza crudo (Brent/WTI) y gas (Henry Hub / TTF europeo) con las mismas
herramientas de los módulos anteriores, más lo específico de las materias
primas: diferenciales entre referencias y el desacople gas EE.UU. vs Europa.

-------------------------------------------------------------------------------
 EL PROBLEMA CENTRAL: FUTUROS vs ETFs
-------------------------------------------------------------------------------
Los futuros (CL=F, BZ=F, NG=F, TTF=F) sirven para el PRECIO y la volatilidad
histórica, pero yfinance NO ofrece sus cadenas de opciones. Las opciones de
crudo se negocian en CME/ICE, fuera del alcance de esta API.

Para la superficie de volatilidad implícita hay que usar los ETFs cotizados en
EE.UU. que replican esas materias primas y que SÍ tienen opciones líquidas:

    Futuro  →  ETF con opciones      Qué mide
    CL=F    →  USO                   WTI
    BZ=F    →  BNO                   Brent
    NG=F    →  UNG                   Gas natural EE.UU. (Henry Hub)
    TTF=F   →  (no existe)           ⚠ el gas europeo NO tiene ETF con opciones

⚠ ADVERTENCIA SOBRE LOS ETFs DE MATERIAS PRIMAS: USO y UNG no replican el precio
spot, sino una cesta de futuros que hay que renovar cada mes (roll). En contango
esa renovación destruye valor de forma sistemática: UNG ha perdido más del 99%
desde su lanzamiento pese a que el gas no ha caído tanto. Para VOLATILIDAD son
un proxy válido; para RENTABILIDAD, no lo son en absoluto.

-------------------------------------------------------------------------------
 POR QUÉ EL GAS EUROPEO ES UN CASO APARTE
-------------------------------------------------------------------------------
TTF=F cotiza en EUR y es el índice de referencia del gas europeo. Desde 2021 se
ha desacoplado brutalmente del Henry Hub estadounidense: Europa depende del GNL
importado y de la geopolítica, EE.UU. produce en exceso. El ratio TTF/HH es,
por sí solo, un termómetro del estrés energético europeo — y afecta directamente
a la inflación y a los márgenes industriales de la eurozona (donde están IDR.MC
y MC.PA de tu cartera).

Uso rápido
----------
    from m11_energia import analizar_complejo_energetico, verificar_opciones

    print(verificar_opciones())          # qué tickers tienen cadena AHORA MISMO
    energia = analizar_complejo_energetico(anios=3)
    print(energia.resumen_volatilidad)
    print(energia.diferenciales.tail())
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

logger = logging.getLogger("quant.energia")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

DIAS_HABILES_ANIO: int = 252
sns.set_theme(style="whitegrid", font_scale=0.95)


class ErrorDeEnergia(Exception):
    """Se lanza cuando no se pueden obtener o procesar los datos energéticos."""


# =============================================================================
# CATÁLOGO DE TICKERS ENERGÉTICOS
# =============================================================================

FUTUROS_ENERGIA: dict[str, dict[str, str]] = {
    "WTI": {
        "ticker": "CL=F", "nombre": "WTI Crude Oil (NYMEX)", "divisa": "USD",
        "notas": "Referencia de crudo en EE.UU. Entrega física en Cushing, Oklahoma.",
    },
    "BRENT": {
        "ticker": "BZ=F", "nombre": "Brent Crude Oil (ICE)", "divisa": "USD",
        "notas": "Referencia mundial: fija el precio de ~2/3 del crudo global. Marino, más ligado a la geopolítica.",
    },
    "GAS_US": {
        "ticker": "NG=F", "nombre": "Henry Hub Natural Gas (NYMEX)", "divisa": "USD",
        "notas": "Gas natural EE.UU. Muy estacional (calefacción invierno / aire acondicionado verano).",
    },
    "GAS_EU": {
        "ticker": "TTF=F", "nombre": "Dutch TTF Natural Gas", "divisa": "EUR",
        "notas": "Referencia del gas europeo. ⚠ Cotiza en EUR y con menos volumen en Yahoo: revisa los huecos.",
    },
    "GASOLINA": {
        "ticker": "RB=F", "nombre": "RBOB Gasoline", "divisa": "USD",
        "notas": "Gasolina. Su diferencial con el crudo es el margen de refino (crack spread).",
    },
    "DIESEL": {
        "ticker": "HO=F", "nombre": "Heating Oil / Diesel", "divisa": "USD",
        "notas": "Destilados medios. El termómetro real de la actividad industrial y el transporte.",
    },
}

# ETFs con cadena de opciones en yfinance → sirven para la superficie de IV
ETFS_ENERGIA_CON_OPCIONES: dict[str, dict[str, str]] = {
    "USO": {
        "nombre": "United States Oil Fund", "replica": "WTI (CL=F)",
        "notas": "El ETF de crudo con opciones más líquido. Sufre roll yield en contango.",
    },
    "BNO": {
        "nombre": "United States Brent Oil Fund", "replica": "Brent (BZ=F)",
        "notas": "Equivalente para Brent. Menos líquido que USO; puede tener pocos strikes.",
    },
    "UNG": {
        "nombre": "United States Natural Gas Fund", "replica": "Henry Hub (NG=F)",
        "notas": "Gas EE.UU. La IV suele ser la más alta de todo el complejo energético.",
    },
    "XLE": {
        "nombre": "Energy Select Sector SPDR", "replica": "Petroleras del S&P 500",
        "notas": "Acciones energéticas (Exxon, Chevron), no la materia prima. Muy líquido.",
    },
    "XOP": {
        "nombre": "SPDR S&P Oil & Gas Exploration", "replica": "Exploración y producción",
        "notas": "Equiponderado y con más small caps: bastante más volátil que XLE.",
    },
    "OIH": {
        "nombre": "VanEck Oil Services ETF", "replica": "Servicios petroleros",
        "notas": "Servicios (SLB, Halliburton). El eslabón más cíclico de la cadena.",
    },
    "FCG": {
        "nombre": "First Trust Natural Gas ETF", "replica": "Empresas de gas natural",
        "notas": "Acciones gasistas. Alternativa a UNG sin el problema del roll.",
    },
}

INDICES_VOLATILIDAD_ENERGIA: dict[str, str] = {
    "^OVX": "CBOE Crude Oil Volatility Index — el 'VIX del petróleo' (IV a 30 días de USO)",
}


def listar_tickers_energia() -> pd.DataFrame:
    """Catálogo completo con qué sirve para qué."""
    filas = []
    for clave, info in FUTUROS_ENERGIA.items():
        filas.append({
            "Clave": clave, "Ticker": info["ticker"], "Nombre": info["nombre"],
            "Divisa": info["divisa"], "Tipo": "Futuro", "¿Opciones?": "NO",
            "Notas": info["notas"],
        })
    for ticker, info in ETFS_ENERGIA_CON_OPCIONES.items():
        filas.append({
            "Clave": ticker, "Ticker": ticker, "Nombre": info["nombre"],
            "Divisa": "USD", "Tipo": "ETF", "¿Opciones?": "SÍ",
            "Notas": f"Replica {info['replica']}. {info['notas']}",
        })
    for ticker, desc in INDICES_VOLATILIDAD_ENERGIA.items():
        filas.append({
            "Clave": ticker, "Ticker": ticker, "Nombre": desc.split("—")[0].strip(),
            "Divisa": "índice", "Tipo": "Índice vol.", "¿Opciones?": "—",
            "Notas": desc,
        })
    return pd.DataFrame(filas).set_index("Clave")


# =============================================================================
# VERIFICACIÓN EN VIVO DE DISPONIBILIDAD DE OPCIONES
# =============================================================================

def verificar_opciones(tickers: list[str] | None = None) -> pd.DataFrame:
    """Comprueba EN VIVO qué tickers tienen cadena de opciones ahora mismo.

    Merece la pena ejecutarla antes de construir superficies: la disponibilidad
    en Yahoo cambia con el tiempo y algunos ETFs pequeños dejan de tener strikes
    cotizados en periodos de poco interés. Es preferible comprobarlo a que falle
    la superficie a mitad del análisis.
    """
    import yfinance as yf

    tickers = tickers or (
        list(ETFS_ENERGIA_CON_OPCIONES.keys())
        + [info["ticker"] for info in FUTUROS_ENERGIA.values()]
    )

    filas = []
    for ticker in tickers:
        try:
            activo = yf.Ticker(ticker)
            vencimientos = list(activo.options)
            if vencimientos:
                # Comprobamos que el primer vencimiento trae contratos de verdad
                cadena = activo.option_chain(vencimientos[0])
                n_contratos = len(cadena.calls) + len(cadena.puts)
                estado = "OK"
            else:
                n_contratos, estado = 0, "Sin cadena"
        except Exception as exc:
            vencimientos, n_contratos = [], 0
            estado = f"Error: {type(exc).__name__}"

        filas.append({
            "Ticker": ticker,
            "¿Opciones?": "SÍ" if vencimientos else "NO",
            "Vencimientos": len(vencimientos),
            "Contratos (1er venc.)": n_contratos,
            "Estado": estado,
        })
        logger.info("%s → %s (%d vencimientos)", ticker, estado, len(vencimientos))

    return pd.DataFrame(filas).set_index("Ticker")


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoEnergia:
    precios: pd.DataFrame               # precios de los futuros energéticos
    retornos: pd.DataFrame
    resumen_volatilidad: pd.DataFrame   # ficha por materia prima
    regimen: pd.DataFrame               # calma / normal / estrés
    correlaciones: pd.DataFrame         # Pearson entre energéticos
    diferenciales: pd.DataFrame         # Brent-WTI, TTF/HH...
    estacionalidad: pd.DataFrame        # volatilidad media por mes
    alertas: list[str] = field(default_factory=list)


# =============================================================================
# DESCARGA
# =============================================================================

def descargar_energia(
    claves: list[str] | None = None,
    anios: float = 3.0,
    umbral_cobertura: float = 0.75,
    cache_dir: str | None = "./cache_datos",
) -> pd.DataFrame:
    """Descarga los futuros energéticos solicitados.

    El umbral de cobertura es laxo (75%) porque los futuros tienen calendarios
    de negociación irregulares y TTF=F en concreto presenta huecos frecuentes
    en Yahoo. Con el umbral estándar del 90% se descartaría casi siempre.
    """
    from m1_datos import descargar_precios

    claves = claves or ["WTI", "BRENT", "GAS_US", "GAS_EU"]

    desconocidas = [c for c in claves if c not in FUTUROS_ENERGIA]
    if desconocidas:
        raise ErrorDeEnergia(
            f"Claves desconocidas: {desconocidas}. Disponibles: {list(FUTUROS_ENERGIA)}"
        )

    tickers = [FUTUROS_ENERGIA[c]["ticker"] for c in claves]
    mapa = {FUTUROS_ENERGIA[c]["ticker"]: c for c in claves}

    precios = descargar_precios(
        tickers, anios=anios, umbral_cobertura=umbral_cobertura, cache_dir=cache_dir,
    )
    precios = precios.rename(columns=mapa)

    perdidas = [c for c in claves if c not in precios.columns]
    if perdidas:
        logger.warning(
            "No se pudieron descargar: %s. Es habitual con TTF=F, que tiene "
            "cobertura irregular en Yahoo Finance.", perdidas,
        )

    return precios


# =============================================================================
# DIFERENCIALES ENTRE REFERENCIAS
# =============================================================================

def calcular_diferenciales(precios: pd.DataFrame) -> pd.DataFrame:
    """Diferenciales clave del complejo energético.

    · BRENT-WTI: el diferencial atlántico. Se amplía cuando hay cuellos de
      botella logísticos en EE.UU. o tensión geopolítica marítima (Ormuz, Suez).
      Históricamente ronda los 3-5 $; por encima de 8 $ hay una dislocación.

    · TTF/HENRY HUB (ratio): el termómetro del estrés energético europeo. Antes
      de 2021 rondaba 2-3×; en el pico de la crisis de 2022 superó 10×. Se
      calcula como RATIO y no como resta porque cotizan en unidades y divisas
      distintas (EUR/MWh vs USD/MMBtu), así que restarlos no significaría nada.

    · CRACK SPREAD (gasolina - crudo): el margen bruto de refino.
    """
    diferenciales = pd.DataFrame(index=precios.index)

    if "BRENT" in precios.columns and "WTI" in precios.columns:
        diferenciales["Brent-WTI ($)"] = precios["BRENT"] - precios["WTI"]

    if "GAS_EU" in precios.columns and "GAS_US" in precios.columns:
        # Ratio, no diferencia: unidades y divisas distintas
        diferenciales["TTF/HenryHub (ratio)"] = precios["GAS_EU"] / precios["GAS_US"]

    if "GASOLINA" in precios.columns and "WTI" in precios.columns:
        # RBOB cotiza en $/galón; el crudo en $/barril (42 galones)
        diferenciales["Crack spread ($/bbl)"] = precios["GASOLINA"] * 42 - precios["WTI"]

    if diferenciales.empty:
        logger.warning("No se pudo calcular ningún diferencial con los activos disponibles.")

    return diferenciales.dropna(how="all")


# =============================================================================
# ESTACIONALIDAD
# =============================================================================

def volatilidad_estacional(retornos: pd.DataFrame, ventana: int = 21) -> pd.DataFrame:
    """Volatilidad media por mes del año.

    En energía la estacionalidad no es un adorno estadístico: el gas natural
    tiene picos estructurales de volatilidad en invierno (riesgo de ola de frío
    con inventarios ajustados) y en verano en EE.UU. (demanda eléctrica para
    aire acondicionado). Ignorarlo lleva a infraestimar el riesgo justo en los
    meses en que más importa.
    """
    from m5_volatilidad_historica import volatilidad_rolling

    vol = volatilidad_rolling(retornos, ventana)
    if vol.empty:
        raise ErrorDeEnergia("No hay datos suficientes para la estacionalidad.")

    tabla = vol.groupby(vol.index.month).mean() * 100
    tabla.index = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
    ][:len(tabla)]
    tabla.index.name = "Mes"
    return tabla.round(2)


# =============================================================================
# DIAGNÓSTICO
# =============================================================================

def generar_alertas_energia(
    resumen: pd.DataFrame,
    diferenciales: pd.DataFrame,
    regimen: pd.DataFrame,
) -> list[str]:
    """Traduce los números a lecturas de mercado."""
    alertas: list[str] = []

    columna_vol = [c for c in resumen.columns if c.startswith("Vol ") and c.endswith("d %")]
    if columna_vol:
        col = columna_vol[-1]
        mas = resumen[col].idxmax()
        alertas.append(
            f"ℹ El activo energético más volátil ahora es {mas} ({resumen.loc[mas, col]:.1f}% anualizado)."
        )

    for activo, fila in regimen.iterrows():
        if fila["Régimen"] == "ESTRÉS":
            alertas.append(
                f"⚠ {activo}: volatilidad en el percentil {fila['Percentil']:.0f} de su histórico."
            )

    if "Brent-WTI ($)" in diferenciales.columns:
        actual = diferenciales["Brent-WTI ($)"].dropna().iloc[-1]
        media = diferenciales["Brent-WTI ($)"].mean()
        if abs(actual) > 8:
            alertas.append(
                f"⚠ Diferencial Brent-WTI en {actual:+.2f} $ (media del periodo: {media:+.2f} $). "
                "Un diferencial tan amplio suele indicar cuellos de botella logísticos "
                "o prima geopolítica sobre el crudo marino."
            )
        else:
            alertas.append(
                f"✓ Diferencial Brent-WTI en {actual:+.2f} $, dentro del rango normal."
            )

    if "TTF/HenryHub (ratio)" in diferenciales.columns:
        serie = diferenciales["TTF/HenryHub (ratio)"].dropna()
        if not serie.empty:
            actual = serie.iloc[-1]
            percentil = (serie < actual).mean() * 100
            if actual > 6:
                alertas.append(
                    f"⚠ El gas europeo cotiza {actual:.1f}× el estadounidense (percentil {percentil:.0f} "
                    "del periodo). Estrés energético en Europa: presión al alza sobre la inflación "
                    "y sobre los costes de la industria de la eurozona."
                )
            else:
                alertas.append(
                    f"ℹ Ratio TTF/Henry Hub en {actual:.1f}× (percentil {percentil:.0f}). "
                    "Recuerda que son unidades y divisas distintas: lo relevante es su evolución, "
                    "no el nivel absoluto."
                )

    if not alertas:
        alertas.append("✓ Sin anomalías destacables en el complejo energético.")
    return alertas


# =============================================================================
# GRÁFICOS
# =============================================================================

def plot_precios_normalizados(
    precios: pd.DataFrame, ax: plt.Axes | None = None, figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Precios en base 100 para poder compararlos pese a estar en unidades distintas."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    normalizado = precios / precios.iloc[0] * 100
    for columna in normalizado.columns:
        ax.plot(normalizado.index, normalizado[columna], linewidth=1.8, label=columna)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Base 100 = inicio del periodo")
    ax.set_title("Evolución comparada del complejo energético", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_diferenciales(
    diferenciales: pd.DataFrame, figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Un panel por diferencial, con su media histórica marcada."""
    if diferenciales.empty:
        raise ErrorDeEnergia("No hay diferenciales que dibujar.")

    n = diferenciales.shape[1]
    fig, ejes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), squeeze=False)

    for ax, columna in zip(ejes[0], diferenciales.columns):
        serie = diferenciales[columna].dropna()
        ax.plot(serie.index, serie, color="#1F4E79", linewidth=1.5)
        ax.axhline(serie.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"Media: {serie.mean():.2f}")
        ax.set_title(columna, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Diferenciales del complejo energético", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_estacionalidad(
    estacionalidad: pd.DataFrame, ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (11, 4.5),
) -> Figure:
    """Mapa de calor de volatilidad media por mes: revela los picos estacionales."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        estacionalidad.T, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
        cbar_kws={"label": "Vol. anualizada (%)"}, linewidths=0.4, linecolor="white",
    )
    ax.set_title("Estacionalidad de la volatilidad (media por mes)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")

    if fig is not None:
        fig.tight_layout()
    return fig


def panel_energia(
    resultado: ResultadoEnergia,
    guardar: str | None = "panel_energia.png",
    dpi: int = 150,
) -> Figure:
    """Panel compuesto del complejo energético."""
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    plot_precios_normalizados(resultado.precios, ax=ax1)

    ax2 = fig.add_subplot(gs[1])
    from m5_volatilidad_historica import volatilidad_rolling
    vol60 = volatilidad_rolling(resultado.retornos, 60)
    for columna in vol60.columns:
        ax2.plot(vol60.index, vol60[columna] * 100, linewidth=1.8, label=columna)
    ax2.set_ylabel("Volatilidad anualizada (%)")
    ax2.set_title("Volatilidad histórica (rolling 60d)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    plot_estacionalidad(resultado.estacionalidad, ax=ax3)

    ax4 = fig.add_subplot(gs[3])
    if not resultado.diferenciales.empty:
        primera = resultado.diferenciales.columns[0]
        serie = resultado.diferenciales[primera].dropna()
        ax4.plot(serie.index, serie, color="#1F4E79", linewidth=1.5)
        ax4.axhline(serie.mean(), color="red", linestyle="--", linewidth=1)
        ax4.set_title(f"Diferencial: {primera}", fontsize=11, fontweight="bold")

    fig.suptitle("Complejo energético — volatilidad y diferenciales",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Panel energético guardado en '%s'", guardar)
    return fig


# =============================================================================
# SUPERFICIES DE VOLATILIDAD IMPLÍCITA DEL COMPLEJO ENERGÉTICO
# =============================================================================

def superficies_energia(
    etfs: list[str] | None = None,
    max_vencimientos: int = 7,
    guardar_panel: str | None = "panel_iv_energia.png",
):
    """Construye y compara las superficies de IV de los ETFs energéticos.

    Reutiliza el Módulo 8, así que la tabla comparativa y los gráficos tienen
    exactamente el mismo formato que los de tu cartera: puedes cruzarlos.

    Lo esperable: UNG (gas) con la IV más alta de todo el complejo — el gas es
    la materia prima cotizada más volátil del mundo, con picos históricos por
    encima del 100% anualizado.
    """
    from m8_comparativa_proxies import analizar_todos_los_proxies, panel_comparativa_proxies

    etfs = etfs or ["USO", "UNG", "XLE", "XOP", "BNO"]

    resultado = analizar_todos_los_proxies(
        etfs, max_vencimientos=max_vencimientos, continuar_si_falla=True,
    )
    if guardar_panel:
        panel_comparativa_proxies(resultado, guardar=guardar_panel)
    return resultado


# =============================================================================
# ORQUESTADOR
# =============================================================================

def analizar_complejo_energetico(
    claves: list[str] | None = None,
    anios: float = 3.0,
    ventanas: tuple[int, int] = (21, 60),
) -> ResultadoEnergia:
    """Análisis completo del complejo energético (parte histórica)."""
    from m1_datos import calcular_retornos
    from m5_volatilidad_historica import analizar_volatilidad

    precios = descargar_energia(claves, anios=anios)
    retornos = calcular_retornos(precios)

    vol = analizar_volatilidad(retornos, ventanas=ventanas)
    correlaciones = retornos.corr()
    diferenciales = calcular_diferenciales(precios)
    estacionalidad = volatilidad_estacional(retornos)
    alertas = generar_alertas_energia(vol.resumen, diferenciales, vol.regimen)

    logger.info("Complejo energético analizado · %d activos", precios.shape[1])

    return ResultadoEnergia(
        precios=precios, retornos=retornos,
        resumen_volatilidad=vol.resumen, regimen=vol.regimen,
        correlaciones=correlaciones, diferenciales=diferenciales,
        estacionalidad=estacionalidad, alertas=alertas,
    )


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m11_energia.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 190)
    pd.set_option("display.max_columns", 30)

    print("\n--- Catálogo de tickers energéticos ---")
    print(listar_tickers_energia()[["Ticker", "Tipo", "¿Opciones?", "Divisa"]].to_string())

    try:
        print("\n--- Verificando opciones disponibles (en vivo) ---")
        print(verificar_opciones(["USO", "UNG", "XLE", "XOP", "BNO", "CL=F", "NG=F"]).to_string())

        energia = analizar_complejo_energetico(anios=3)

        print("\n" + "=" * 78)
        print("MÓDULO 11 · COMPLEJO ENERGÉTICO")
        print("=" * 78)
        print("\n--- Volatilidad ---")
        print(energia.resumen_volatilidad.to_string())
        print("\n--- Régimen ---")
        print(energia.regimen.to_string())
        print("\n--- Correlaciones ---")
        print(energia.correlaciones.round(3).to_string())
        print("\n--- Diferenciales (últimos 5 días) ---")
        print(energia.diferenciales.tail().round(3).to_string())
        print("\n--- Estacionalidad de la volatilidad ---")
        print(energia.estacionalidad.to_string())
        print("\n--- Diagnóstico ---")
        for a in energia.alertas:
            print("  " + a)

        panel_energia(energia, guardar="panel_energia.png")

        print("\n--- Superficies de volatilidad implícita ---")
        comp = superficies_energia(["USO", "UNG", "XLE", "XOP"])
        print(comp.tabla.to_string())
        for a in comp.alertas:
            print("  " + a)

        print("\n[OK] Módulo 11 ejecutado.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
