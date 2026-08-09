"""
===============================================================================
 MÓDULO 13 — GAS EUROPEO: INVENTARIOS Y FUNDAMENTALES (GIE AGSI+ / ALSI)
===============================================================================
Este módulo cierra el hueco que dejaba el Módulo 11: el precio del TTF se puede
sacar de Yahoo (mal), pero lo que REALMENTE mueve ese precio son los inventarios
de gas europeos, y eso no está en ninguna API financiera.

GIE (Gas Infrastructure Europe) publica esos datos gratis:

  · AGSI+ → almacenamiento subterráneo de gas, por país e instalación
  · ALSI  → terminales de GNL (regasificación), la vía por la que Europa
            sustituyó el gas ruso por importaciones marítimas

POR QUÉ IMPORTA MÁS QUE EL PRECIO
---------------------------------
El TTF es el resultado; los inventarios son la causa. Un almacenamiento europeo
al 95% en noviembre significa un invierno tranquilo; al 65%, significa que
cualquier ola de frío dispara el precio. Esta relación es tan fuerte que el
nivel de llenado frente a su media histórica explica buena parte de la
volatilidad del TTF.

Además hay un ancla regulatoria: la UE obliga a alcanzar el 90% de llenado
antes del 1 de noviembre de cada año. Ese objetivo condiciona las compras de
toda Europa durante el verano y es una fecha de referencia para el mercado.

CLAVE DE API (gratuita)
-----------------------
    1. Regístrate en https://agsi.gie.eu/  (arriba a la derecha, "API")
    2. Copia tu clave
    3. export GIE_API_KEY="tu_clave"

DETALLES TÉCNICOS QUE CONVIENE SABER
------------------------------------
  · Los datos se publican a diario a las 19:30 CET, con una segunda pasada a
    las 23:00. Reflejan el día de gas ANTERIOR, no el actual.
  · Histórico desde 2011 en AGSI+ y desde 2012 en ALSI.
  · La API pagina: hay que recorrer las páginas para históricos largos.
  · Unidades: TWh para volúmenes, GWh/día para flujos, % para el llenado.

Uso rápido
----------
    from m13_gas_europa import analizar_gas_europeo

    gas = analizar_gas_europeo(paises=["EU", "DE", "ES"], anios=3)
    print(gas.resumen)
    print(gas.alertas)
===============================================================================
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger("quant.gas_europa")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

sns.set_theme(style="whitegrid", font_scale=0.95)

URL_AGSI = "https://agsi.gie.eu/api"
URL_ALSI = "https://alsi.gie.eu/api"
TIEMPO_ESPERA = 30
PAUSA_ENTRE_PAGINAS = 0.35  # cortesía con la API: evita que nos limiten

# Objetivo regulatorio de la UE: 90% de llenado antes del 1 de noviembre
OBJETIVO_UE_LLENADO = 90.0


class ErrorDeGIE(Exception):
    """Se lanza cuando la API de GIE no responde o devuelve datos inválidos."""


# =============================================================================
# CATÁLOGO DE PAÍSES
# =============================================================================

PAISES_GAS: dict[str, dict[str, str]] = {
    "EU":  {"nombre": "Unión Europea (agregado)", "relevancia": "La cifra que mira el mercado del TTF."},
    "DE":  {"nombre": "Alemania", "relevancia": "El mayor almacenamiento de Europa (~24% del total). Marca el tono."},
    "IT":  {"nombre": "Italia", "relevancia": "Segundo mayor. Muy dependiente de importaciones."},
    "NL":  {"nombre": "Países Bajos", "relevancia": "Sede física del hub TTF. Groningen ya cerrado."},
    "FR":  {"nombre": "Francia", "relevancia": "Su demanda depende del estado del parque nuclear."},
    "ES":  {"nombre": "España", "relevancia": "Poca conexión con el resto de la UE, pero mucha capacidad de GNL."},
    "AT":  {"nombre": "Austria", "relevancia": "Nudo de tránsito histórico del gas ruso (Baumgarten)."},
    "UA":  {"nombre": "Ucrania", "relevancia": "Enorme capacidad de almacenamiento. Riesgo geopolítico directo."},
    "GB":  {"nombre": "Reino Unido", "relevancia": "Muy poco almacenamiento: depende del flujo diario."},
    "PL":  {"nombre": "Polonia", "relevancia": "Clave desde el cierre del Yamal."},
}


def listar_paises() -> pd.DataFrame:
    """Países disponibles en AGSI+ y por qué importa cada uno."""
    return pd.DataFrame(PAISES_GAS).T.rename_axis("Código")


# =============================================================================
# CONTENEDOR DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoGasEuropeo:
    datos: dict[str, pd.DataFrame]     # {país: DataFrame diario}
    llenado: pd.DataFrame              # % de llenado, una columna por país
    resumen: pd.DataFrame              # ficha actual por país
    estacional: pd.DataFrame           # llenado medio por día del año (patrón típico)
    flujos: pd.DataFrame               # inyección neta diaria
    alertas: list[str] = field(default_factory=list)


# =============================================================================
# PARSEO (separado de la red para poder probarlo sin conexión)
# =============================================================================

# La API ha cambiado nombres de campos entre versiones; los normalizamos.
CAMPOS_FECHA = ("gasDayStart", "gasDayStartedOn")

CAMPOS_NUMERICOS = (
    "gasInStorage",        # TWh almacenados
    "consumption",         # TWh consumidos
    "consumptionFull",     # % del consumo cubierto por el almacenamiento
    "full",                # % de llenado — EL CAMPO CLAVE
    "trend",               # variación diaria del llenado
    "injection",           # GWh/día inyectados
    "withdrawal",          # GWh/día retirados
    "workingGasVolume",    # TWh de capacidad total
    "injectionCapacity",   # GWh/día de capacidad de inyección
    "withdrawalCapacity",  # GWh/día de capacidad de retirada
)


def _parsear_respuesta_gie(carga: dict, pais: str) -> pd.DataFrame:
    """Convierte la respuesta JSON de GIE en un DataFrame limpio.

    La API devuelve TODOS los valores como cadenas de texto, incluidos los
    numéricos, y usa '-' para los días sin dato. Sin esta conversión, cualquier
    cálculo posterior fallaría o daría resultados absurdos.
    """
    if not isinstance(carga, dict):
        raise ErrorDeGIE(f"Respuesta inesperada de GIE para '{pais}': {type(carga)}")

    if "error" in carga:
        raise ErrorDeGIE(f"GIE devolvió un error para '{pais}': {carga['error']}")

    registros = carga.get("data", [])
    if not registros:
        raise ErrorDeGIE(
            f"GIE no devolvió datos para '{pais}'. Comprueba el código de país "
            f"(disponibles: {list(PAISES_GAS)}) y el rango de fechas."
        )

    tabla = pd.DataFrame(registros)

    # --- Fecha: el nombre del campo depende de la versión de la API ---------
    columna_fecha = next((c for c in CAMPOS_FECHA if c in tabla.columns), None)
    if columna_fecha is None:
        raise ErrorDeGIE(
            f"No se encontró la columna de fecha en la respuesta. "
            f"Columnas recibidas: {list(tabla.columns)[:10]}"
        )
    tabla["fecha"] = pd.to_datetime(tabla[columna_fecha], errors="coerce")

    # --- Numéricos: vienen como texto, y '-' significa "sin dato" -----------
    for campo in CAMPOS_NUMERICOS:
        if campo in tabla.columns:
            tabla[campo] = pd.to_numeric(
                tabla[campo].astype(str).str.strip().replace({"-": None, "": None, "N/A": None}),
                errors="coerce",
            )

    tabla = tabla.dropna(subset=["fecha"]).set_index("fecha").sort_index()

    if tabla.empty:
        raise ErrorDeGIE(f"Tras limpiar, no quedaron datos válidos para '{pais}'.")

    return tabla


# =============================================================================
# DESCARGA CON PAGINACIÓN
# =============================================================================

def descargar_agsi(
    pais: str = "EU",
    anios: float = 3.0,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    api_key: str | None = None,
    max_paginas: int = 40,
    tamano_pagina: int = 300,
) -> pd.DataFrame:
    """Descarga el histórico de almacenamiento de gas de un país desde AGSI+.

    Parameters
    ----------
    pais : código de 2 letras ('DE', 'ES'...) o 'EU' para el agregado europeo.
    anios : profundidad del histórico (hay datos desde 2011).
    api_key : si no se pasa, se lee de la variable de entorno GIE_API_KEY.
    max_paginas : tope de seguridad. Con 300 registros por página, 40 páginas
        cubren unos 32 años: más que suficiente y evita bucles infinitos si la
        API devuelve una paginación inconsistente.

    Returns
    -------
    DataFrame diario con el % de llenado, volúmenes, inyecciones y retiradas.
    """
    api_key = api_key or os.environ.get("GIE_API_KEY")
    if not api_key:
        raise ErrorDeGIE(
            "Falta la clave de GIE. Es gratuita:\n"
            "  1. Regístrate en https://agsi.gie.eu/ (botón 'API' arriba a la derecha)\n"
            "  2. Copia tu clave\n"
            '  3. export GIE_API_KEY="tu_clave"\n'
            "O pásala directamente: descargar_agsi('DE', api_key='...')"
        )

    if fecha_inicio is None:
        fecha_inicio = (datetime.today() - timedelta(days=int(anios * 365.25))).strftime("%Y-%m-%d")
    fecha_fin = fecha_fin or datetime.today().strftime("%Y-%m-%d")

    cabeceras = {"x-key": api_key, "Accept": "application/json"}
    paginas: list[pd.DataFrame] = []
    pagina_actual = 1

    while pagina_actual <= max_paginas:
        parametros = {
            "from": fecha_inicio,
            "to": fecha_fin,
            "page": pagina_actual,
            "size": tamano_pagina,
        }
        # 'EU' es el agregado: se obtiene NO enviando el parámetro country
        if pais.upper() != "EU":
            parametros["country"] = pais.upper()

        try:
            respuesta = requests.get(
                URL_AGSI, params=parametros, headers=cabeceras, timeout=TIEMPO_ESPERA,
            )
        except requests.RequestException as exc:
            raise ErrorDeGIE(f"Error de red al consultar AGSI+: {exc}") from exc

        if respuesta.status_code == 401:
            raise ErrorDeGIE(
                "GIE rechazó la clave de API (401). Comprueba que GIE_API_KEY es correcta "
                "y que tu cuenta está activada."
            )
        if respuesta.status_code == 429:
            raise ErrorDeGIE(
                "GIE ha limitado las peticiones (429). Espera unos minutos o reduce "
                "el número de países/años solicitados."
            )
        try:
            respuesta.raise_for_status()
            carga = respuesta.json()
        except Exception as exc:
            raise ErrorDeGIE(f"Respuesta inválida de AGSI+ ({respuesta.status_code}): {exc}") from exc

        paginas.append(_parsear_respuesta_gie(carga, pais))

        ultima = int(carga.get("last_page", 1) or 1)
        if pagina_actual >= ultima:
            break
        pagina_actual += 1
        time.sleep(PAUSA_ENTRE_PAGINAS)

    else:
        logger.warning(
            "Se alcanzó el límite de %d páginas para '%s'. El histórico puede estar incompleto.",
            max_paginas, pais,
        )

    datos = pd.concat(paginas).sort_index()
    datos = datos[~datos.index.duplicated(keep="first")]

    logger.info(
        "AGSI+ %s: %d días (%s → %s)",
        pais, len(datos), datos.index[0].date(), datos.index[-1].date(),
    )
    return datos


def descargar_alsi(
    pais: str = "EU",
    anios: float = 3.0,
    api_key: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Igual que `descargar_agsi` pero para terminales de GNL (plataforma ALSI).

    El GNL es la variable que salvó a Europa en 2022: cuando el gas ruso por
    tubo desapareció, el hueco lo llenaron los barcos metaneros. El nivel de
    llenado de los tanques y el ritmo de regasificación indican cuánta oferta
    alternativa está entrando de verdad.
    """
    global URL_AGSI
    url_original = URL_AGSI
    try:
        URL_AGSI = URL_ALSI  # reutilizamos la lógica de paginación
        return descargar_agsi(pais=pais, anios=anios, api_key=api_key, **kwargs)
    finally:
        URL_AGSI = url_original


# =============================================================================
# ANÁLISIS
# =============================================================================

def patron_estacional_llenado(llenado: pd.Series) -> pd.DataFrame:
    """Nivel de llenado típico para cada día del año, con su banda histórica.

    Es la referencia con la que el mercado juzga la situación actual: no importa
    tanto estar al 70% como estar al 70% cuando lo normal para esa fecha es 85%.
    """
    serie = llenado.dropna()
    if serie.empty:
        raise ErrorDeGIE("No hay datos de llenado para calcular el patrón estacional.")

    por_dia = serie.groupby(serie.index.dayofyear)
    patron = pd.DataFrame({
        "media": por_dia.mean(),
        "mínimo": por_dia.min(),
        "máximo": por_dia.max(),
        "p25": por_dia.quantile(0.25),
        "p75": por_dia.quantile(0.75),
    })
    patron.index.name = "día del año"
    return patron.round(2)


def resumen_actual(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ficha del estado actual de cada país frente a su propia historia."""
    filas = []
    for pais, tabla in datos.items():
        if "full" not in tabla.columns or tabla["full"].dropna().empty:
            continue

        llenado = tabla["full"].dropna()
        actual = llenado.iloc[-1]
        fecha = llenado.index[-1]

        # Comparación con la misma fecha de años anteriores (±3 días)
        dia_anio = fecha.dayofyear
        mismos_dias = llenado[abs(llenado.index.dayofyear - dia_anio) <= 3]
        historico = mismos_dias[mismos_dias.index.year < fecha.year]
        media_historica = historico.mean() if not historico.empty else np.nan

        fila = {
            "País": PAISES_GAS.get(pais, {}).get("nombre", pais),
            "Fecha": fecha.date(),
            "Llenado %": round(actual, 2),
            "Media histórica (misma fecha) %": round(media_historica, 2) if pd.notna(media_historica) else np.nan,
            "Desviación (pp)": round(actual - media_historica, 2) if pd.notna(media_historica) else np.nan,
        }

        if "gasInStorage" in tabla.columns:
            fila["Almacenado (TWh)"] = round(tabla["gasInStorage"].dropna().iloc[-1], 2)
        if "workingGasVolume" in tabla.columns:
            fila["Capacidad (TWh)"] = round(tabla["workingGasVolume"].dropna().iloc[-1], 2)
        if "trend" in tabla.columns and not tabla["trend"].dropna().empty:
            tendencia = tabla["trend"].dropna().iloc[-1]
            fila["Tendencia diaria (pp)"] = round(tendencia, 3)
            fila["Fase"] = "Inyectando" if tendencia > 0 else "Retirando" if tendencia < 0 else "Estable"

        filas.append(fila)

    if not filas:
        raise ErrorDeGIE("Ningún país tenía datos de llenado utilizables.")
    return pd.DataFrame(filas).set_index("País")


def flujos_netos(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inyección neta diaria (inyección - retirada) en GWh/día, por país.

    Positivo = Europa está acumulando reservas (típico de abril a octubre).
    Negativo = está consumiéndolas (típico de noviembre a marzo).
    Un cambio de signo fuera de temporada es una señal relevante.
    """
    columnas = {}
    for pais, tabla in datos.items():
        if "injection" in tabla.columns and "withdrawal" in tabla.columns:
            columnas[pais] = tabla["injection"].fillna(0) - tabla["withdrawal"].fillna(0)
    return pd.DataFrame(columnas).sort_index() if columnas else pd.DataFrame()


def generar_alertas_gas(resumen: pd.DataFrame, llenado: pd.DataFrame) -> list[str]:
    """Interpretación en lenguaje natural del estado del almacenamiento."""
    alertas: list[str] = []
    hoy = datetime.today()

    for pais, fila in resumen.iterrows():
        actual = fila["Llenado %"]
        desviacion = fila.get("Desviación (pp)")

        if pd.notna(desviacion):
            if desviacion < -10:
                alertas.append(
                    f"⚠ {pais}: llenado al {actual:.1f}%, {abs(desviacion):.1f} puntos POR DEBAJO "
                    "de su media histórica para esta fecha. Situación de tensión: cualquier ola "
                    "de frío tendría un impacto desproporcionado en el precio del TTF."
                )
            elif desviacion > 10:
                alertas.append(
                    f"✓ {pais}: llenado al {actual:.1f}%, {desviacion:.1f} puntos por encima de su "
                    "media histórica. Situación holgada, presión bajista sobre el precio."
                )
            else:
                alertas.append(
                    f"ℹ {pais}: llenado al {actual:.1f}% ({desviacion:+.1f} pp frente a su media "
                    "histórica para esta fecha). Dentro de lo normal."
                )

    # Comprobación del objetivo regulatorio de la UE (90% antes del 1 de noviembre)
    if "Unión Europea (agregado)" in resumen.index:
        llenado_ue = resumen.loc["Unión Europea (agregado)", "Llenado %"]
        objetivo = datetime(hoy.year, 11, 1)
        if hoy > objetivo:
            objetivo = datetime(hoy.year + 1, 11, 1)
        dias_restantes = (objetivo - hoy).days

        if llenado_ue < OBJETIVO_UE_LLENADO and dias_restantes < 120:
            faltan = OBJETIVO_UE_LLENADO - llenado_ue
            alertas.append(
                f"⚠ La UE está al {llenado_ue:.1f}% y quedan {dias_restantes} días para el objetivo "
                f"regulatorio del {OBJETIVO_UE_LLENADO:.0f}% (1 de noviembre). Faltan {faltan:.1f} "
                "puntos: compras forzadas de gas en verano suelen sostener el precio."
            )
        elif llenado_ue >= OBJETIVO_UE_LLENADO:
            alertas.append(
                f"✓ La UE ya supera el objetivo regulatorio del {OBJETIVO_UE_LLENADO:.0f}% "
                f"(está al {llenado_ue:.1f}%). Menos presión compradora."
            )

    if not alertas:
        alertas.append("✓ Sin anomalías destacables en el almacenamiento europeo.")
    return alertas


# =============================================================================
# CORRELACIÓN CON EL PRECIO DEL TTF
# =============================================================================

def relacion_llenado_precio(
    llenado: pd.Series,
    precio_ttf: pd.Series,
    ventana: int = 60,
) -> pd.DataFrame:
    """Cruza el nivel de almacenamiento con el precio del TTF.

    La relación esperada es NEGATIVA: más inventarios, menos precio. Cuando esa
    correlación se rompe (se vuelve positiva), suele significar que el mercado
    está descontando algo que los inventarios actuales no reflejan — una
    interrupción de suministro, un invierno anticipado, un shock geopolítico.

    Requiere el precio del TTF, que puedes obtener del Módulo 11 (TTF=F).
    """
    conjunto = pd.DataFrame({
        "llenado_%": llenado,
        "precio_ttf": precio_ttf,
    }).dropna()

    if len(conjunto) < ventana:
        raise ErrorDeGIE(
            f"Solo hay {len(conjunto)} días con ambas series. Se necesitan al menos {ventana}. "
            "¿Coinciden los periodos descargados de GIE y de Yahoo?"
        )

    conjunto["correlacion_rolling"] = (
        conjunto["llenado_%"].rolling(ventana).corr(conjunto["precio_ttf"])
    )
    conjunto["cambio_llenado_5d"] = conjunto["llenado_%"].diff(5)
    conjunto["cambio_precio_5d_%"] = conjunto["precio_ttf"].pct_change(5) * 100

    return conjunto.dropna()


# =============================================================================
# GRÁFICOS
# =============================================================================

def plot_llenado(
    llenado: pd.DataFrame, ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Evolución del % de llenado por país, con el objetivo de la UE marcado."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    paleta = sns.color_palette("tab10", llenado.shape[1])
    for color, pais in zip(paleta, llenado.columns):
        etiqueta = PAISES_GAS.get(pais, {}).get("nombre", pais)
        ax.plot(llenado.index, llenado[pais], color=color, linewidth=1.8, label=etiqueta)

    ax.axhline(OBJETIVO_UE_LLENADO, color="red", linestyle="--", linewidth=1.2,
               label=f"Objetivo UE ({OBJETIVO_UE_LLENADO:.0f}%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Llenado (%)")
    ax.set_title("Almacenamiento de gas europeo", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_estacionalidad_llenado(
    llenado: pd.Series, patron: pd.DataFrame,
    ax: plt.Axes | None = None, figsize: tuple[float, float] = (11, 5),
) -> Figure:
    """Situación actual superpuesta al patrón estacional histórico.

    La banda gris es el rango normal para cada fecha; la línea negra, el año en
    curso. Es la forma en que los analistas de gas leen el mercado: no el nivel
    absoluto, sino la desviación respecto a lo normal para esa época del año.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(patron.index, patron["mínimo"], patron["máximo"],
                    color="lightgray", alpha=0.5, label="Rango histórico")
    ax.fill_between(patron.index, patron["p25"], patron["p75"],
                    color="gray", alpha=0.35, label="Rango intercuartílico")
    ax.plot(patron.index, patron["media"], color="black", linestyle="--",
            linewidth=1.2, label="Media histórica")

    anio_actual = llenado.index.year.max()
    actual = llenado[llenado.index.year == anio_actual]
    if not actual.empty:
        ax.plot(actual.index.dayofyear, actual.values, color="#C00000",
                linewidth=2.4, label=f"{anio_actual}")

    ax.axhline(OBJETIVO_UE_LLENADO, color="blue", linestyle=":", linewidth=1.2,
               label=f"Objetivo UE {OBJETIVO_UE_LLENADO:.0f}%")
    ax.set_xlabel("Día del año")
    ax.set_ylabel("Llenado (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Llenado actual frente al patrón estacional", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower center", ncol=3)

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_flujos(
    flujos: pd.DataFrame, ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 4.5),
) -> Figure:
    """Inyección neta diaria: por encima de cero Europa acumula, por debajo consume."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for pais in flujos.columns:
        etiqueta = PAISES_GAS.get(pais, {}).get("nombre", pais)
        ax.plot(flujos.index, flujos[pais], linewidth=1.3, label=etiqueta, alpha=0.85)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Inyección neta (GWh/día)")
    ax.set_title("Flujos netos de almacenamiento (positivo = acumulando)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    if fig is not None:
        fig.tight_layout()
    return fig


def panel_gas_europeo(
    resultado: ResultadoGasEuropeo,
    pais_detalle: str = "EU",
    guardar: str | None = "panel_gas_europa.png",
    dpi: int = 150,
) -> Figure:
    """Panel compuesto del gas europeo."""
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 1)

    ax1 = fig.add_subplot(gs[0])
    plot_llenado(resultado.llenado, ax=ax1)

    ax2 = fig.add_subplot(gs[1])
    if pais_detalle in resultado.llenado.columns:
        plot_estacionalidad_llenado(
            resultado.llenado[pais_detalle].dropna(), resultado.estacional, ax=ax2,
        )

    ax3 = fig.add_subplot(gs[2])
    if not resultado.flujos.empty:
        plot_flujos(resultado.flujos, ax=ax3)

    fig.suptitle("Gas europeo — inventarios y flujos (GIE AGSI+)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=dpi, bbox_inches="tight")
        logger.info("Panel de gas europeo guardado en '%s'", guardar)
    return fig


# =============================================================================
# ORQUESTADOR
# =============================================================================

def analizar_gas_europeo(
    paises: list[str] | None = None,
    anios: float = 3.0,
    api_key: str | None = None,
    pais_estacional: str = "EU",
) -> ResultadoGasEuropeo:
    """Descarga y analiza el almacenamiento de gas de varios países europeos.

    Si un país falla, se registra y se continúa con el resto: en una tarea
    automatizada es preferible un informe parcial a ningún informe.
    """
    paises = paises or ["EU", "DE", "IT", "NL", "ES"]

    datos: dict[str, pd.DataFrame] = {}
    for pais in paises:
        try:
            datos[pais] = descargar_agsi(pais, anios=anios, api_key=api_key)
        except ErrorDeGIE as exc:
            logger.warning("No se pudo descargar '%s': %s", pais, str(exc)[:120])

    if not datos:
        raise ErrorDeGIE(
            "No se pudo descargar ningún país. Comprueba tu GIE_API_KEY y tu conexión."
        )

    llenado = pd.DataFrame({
        pais: tabla["full"] for pais, tabla in datos.items() if "full" in tabla.columns
    }).sort_index()

    if llenado.empty:
        raise ErrorDeGIE("Ningún país devolvió el campo 'full' (% de llenado).")

    referencia = pais_estacional if pais_estacional in llenado.columns else llenado.columns[0]
    estacional = patron_estacional_llenado(llenado[referencia])

    resumen = resumen_actual(datos)
    flujos = flujos_netos(datos)
    alertas = generar_alertas_gas(resumen, llenado)

    logger.info("Gas europeo analizado · %d países · %d días", len(datos), len(llenado))

    return ResultadoGasEuropeo(
        datos=datos, llenado=llenado, resumen=resumen,
        estacional=estacional, flujos=flujos, alertas=alertas,
    )


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m13_gas_europa.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 190)
    pd.set_option("display.max_columns", 30)

    print("\n--- Países disponibles en AGSI+ ---")
    print(listar_paises().to_string())

    try:
        gas = analizar_gas_europeo(paises=["EU", "DE", "IT", "NL", "ES"], anios=3)

        print("\n" + "=" * 78)
        print("MÓDULO 13 · GAS EUROPEO (GIE AGSI+)")
        print("=" * 78)
        print("\n--- Estado actual ---")
        print(gas.resumen.to_string())
        print("\n--- Llenado (últimos 5 días) ---")
        print(gas.llenado.tail().round(2).to_string())
        print("\n--- Diagnóstico ---")
        for a in gas.alertas:
            print("  " + a)

        panel_gas_europeo(gas, guardar="panel_gas_europa.png")

        # Cruce con el precio del TTF, si está disponible
        try:
            from m11_energia import descargar_energia
            precios = descargar_energia(["GAS_EU"], anios=3)
            relacion = relacion_llenado_precio(gas.llenado["EU"], precios["GAS_EU"])
            print("\n--- Relación llenado vs precio TTF (últimos 5 días) ---")
            print(relacion.tail().round(3).to_string())
            print(f"\nCorrelación actual (60d): {relacion['correlacion_rolling'].iloc[-1]:+.3f} "
                  "(lo normal es NEGATIVA: más inventarios → menos precio)")
        except Exception as exc:
            print(f"\n[Aviso] No se pudo cruzar con el precio del TTF: {exc}")

        print("\n[OK] Módulo 13 ejecutado.")

    except ErrorDeGIE as exc:
        print(f"\n[ERROR DE GIE] {exc}")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
