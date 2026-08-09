"""
===============================================================================
 MÓDULO 10 — INFORME AUTOMÁTICO DIARIO (Fase 4, parte B)
===============================================================================
Genera un PDF con todos los gráficos y tablas, y lo envía por email o Telegram.
Pensado para ejecutarse solo cada mañana antes de la apertura, mediante cron
(en tu Mac) o GitHub Actions (en la nube, gratis).

EJECUCIÓN MANUAL
----------------
    python3 m10_informe_automatico.py                    # solo genera el PDF
    python3 m10_informe_automatico.py --email            # genera y envía por email
    python3 m10_informe_automatico.py --telegram         # genera y envía por Telegram

CONFIGURACIÓN (variables de entorno, NUNCA en el código)
--------------------------------------------------------
Email (Gmail):
    export EMAIL_ORIGEN="tucorreo@gmail.com"
    export EMAIL_PASSWORD="xxxx xxxx xxxx xxxx"   # contraseña de aplicación, NO la normal
    export EMAIL_DESTINO="tucorreo@gmail.com"

    ⚠ Gmail exige una "contraseña de aplicación" (16 caracteres), que se genera en
    myaccount.google.com → Seguridad → Verificación en 2 pasos → Contraseñas de
    aplicaciones. Tu contraseña habitual NO funcionará por SMTP.

Telegram:
    export TELEGRAM_TOKEN="123456:ABC-DEF..."     # te lo da @BotFather
    export TELEGRAM_CHAT_ID="123456789"           # te lo da @userinfobot

AUTOMATIZACIÓN EN TU MAC (cron)
--------------------------------
    crontab -e
    # cada día laborable a las 8:00
    0 8 * * 1-5 cd /Users/Pablo/Desktop/inversion && /usr/bin/python3 m10_informe_automatico.py --email >> informe.log 2>&1

AUTOMATIZACIÓN EN LA NUBE (GitHub Actions)
-------------------------------------------
Ver la plantilla al final de este archivo (función `plantilla_github_actions`).
Ventaja: funciona aunque tu Mac esté apagado.

SEGURIDAD: este script solo LEE datos de mercado y envía un informe. No opera,
no accede a tu bróker y no necesita credenciales de ninguna cuenta de inversión.
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger("quant.informe")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =============================================================================
# CONFIGURACIÓN DE TU CARTERA
# =============================================================================

CARTERA: dict[str, float] = {
    "RHM.DE": 1955.0,
    "IDR.MC": 1670.0,
    "MC.PA": 1095.0,
    "0P0001KGI5.F": 2176.06,
    "EUNL.DE": 531.0,
}

# Precio de compra y nº de títulos de cada posición, para la página de
# rentabilidad real (P&L) del informe.
COMPRAS: dict[str, dict[str, float]] = {
    "RHM.DE": {"precio_compra": 497.04, "titulos": 1.69},
    "IDR.MC": {"precio_compra": 35.13, "titulos": 24.2},
    "MC.PA": {"precio_compra": 573.67, "titulos": 2.17},
    "0P0001KGI5.F": {"precio_compra": 205.55, "titulos": 9.875},
    "EUNL.DE": {"precio_compra": 11.55, "titulos": 43.29},
}

BENCHMARKS = ["SPY", "KOSPI", "EUROSTOXX"]
PROXY_IV = "SPY"
RF = 0.04
ANIOS = 3.0
VENTANAS_VOL = (21, 60)


class ErrorDeInforme(Exception):
    """Se lanza cuando el informe no se puede generar o enviar."""


# =============================================================================
# GENERACIÓN DEL INFORME
# =============================================================================

def _limpiar_glifos(texto: str) -> str:
    """Sustituye símbolos que la fuente monoespaciada del PDF no puede dibujar.

    matplotlib usa DejaVu Sans Mono, que no incluye ℹ ⚠ ✓ ★ ni flechas. Sin esta
    conversión el PDF sale con cuadraditos vacíos y un aviso por cada línea.
    """
    reemplazos = {
        "⚠": "[!]", "ℹ": "[i]", "✓": "[OK]", "★": "*",
        "→": "->", "↑": "^", "↓": "v", "—": "-", "×": "x",
        "Δ": "D", "σ": "sigma", "β": "beta", "μ": "mu", "Σ": "Sigma",
        "λ": "lambda", "Π": "Pi", "ρ": "rho", "π": "pi", "τ": "tau", "δ": "delta",
    }
    for original, sustituto in reemplazos.items():
        texto = texto.replace(original, sustituto)
    return texto


def _pagina_texto(pdf: PdfPages, titulo: str, lineas: list[str], tamano: int = 9) -> None:
    """Añade una página de solo texto al PDF (para tablas y diagnósticos)."""
    titulo = _limpiar_glifos(titulo)
    lineas = [_limpiar_glifos(str(linea)) for linea in lineas]

    fig = plt.figure(figsize=(11.7, 8.3))  # A4 apaisado
    fig.text(0.05, 0.95, titulo, fontsize=15, fontweight="bold", va="top")

    y = 0.88
    for linea in lineas:
        # Las líneas muy largas se cortan para que no se salgan de la página
        if len(linea) > 155:
            linea = linea[:152] + "..."
        if y < 0.05:  # si no cabe, abrimos página nueva
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(0.05, 0.95, f"{titulo} (cont.)", fontsize=15, fontweight="bold", va="top")
            y = 0.88
        fig.text(0.05, y, linea, fontsize=tamano, family="monospace", va="top")
        y -= 0.022

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generar_informe(
    cartera: dict[str, float] = None,
    ruta_salida: str | Path = None,
    incluir_superficie: bool = True,
) -> Path:
    """Ejecuta todos los módulos y compone un PDF multipágina.

    Cada sección va en su propio try/except: si un módulo falla (por ejemplo,
    la cadena de opciones no está disponible), el informe se genera igualmente
    con el resto de secciones y una nota del error. En una tarea automatizada
    esto es preferible a que un fallo puntual cancele el informe entero.
    """
    from m1_datos import calcular_retornos, descargar_precios, resumen_activos
    from m2_correlaciones import analizar_correlaciones
    from m3_optimizacion import (
        calcular_pnl_posiciones, cartera_desde_importes, comparar_carteras, optimizar_cartera,
    )
    from m4_visualizacion import (
        plot_frontera_eficiente, plot_heatmaps_correlacion, plot_pesos_comparativa,
    )
    from m5_volatilidad_historica import analizar_volatilidad
    from m6_benchmarks_volatilidad import (
        alinear_con_cartera, comparar_volatilidad_benchmarks,
        plot_cartera_vs_benchmarks, plot_volatilidad_rolling,
    )

    cartera = cartera or CARTERA
    hoy = datetime.now()
    ruta_salida = Path(ruta_salida or f"informe_cartera_{hoy:%Y%m%d}.pdf")

    logger.info("Generando informe para %d activos...", len(cartera))

    precios = descargar_precios(list(cartera.keys()), anios=ANIOS)
    retornos = calcular_retornos(precios)
    cartera_valida = {t: v for t, v in cartera.items() if t in precios.columns}
    pesos = pd.Series(cartera_valida) / sum(cartera_valida.values())

    with PdfPages(ruta_salida) as pdf:
        # --- Portada -------------------------------------------------------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.5, 0.62, "INFORME DIARIO DE CARTERA", fontsize=26,
                 fontweight="bold", ha="center")
        fig.text(0.5, 0.53, f"{hoy:%A, %d de %B de %Y — %H:%M}", fontsize=13, ha="center")
        fig.text(0.5, 0.44, f"{len(cartera_valida)} activos · "
                            f"{sum(cartera_valida.values()):,.0f} € invertidos",
                 fontsize=12, ha="center")
        fig.text(0.5, 0.36, " · ".join(cartera_valida.keys()), fontsize=10,
                 ha="center", color="gray")
        fig.text(0.5, 0.08, "Análisis cuantitativo automatizado — no constituye "
                            "asesoramiento financiero", fontsize=8, ha="center", color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Resumen de activos --------------------------------------------
        try:
            resumen = resumen_activos(precios, retornos, RF)
            _pagina_texto(pdf, "1. Ficha de los activos", resumen.to_string().splitlines())
        except Exception as exc:
            _pagina_texto(pdf, "1. Ficha de los activos", [f"ERROR: {exc}"])

        # --- Rentabilidad real (compra vs valor actual) -----------------------
        try:
            compras_validas = {t: c for t, c in COMPRAS.items() if t in cartera_valida}
            if compras_validas:
                pnl = calcular_pnl_posiciones(compras_validas, cartera_valida)
                _pagina_texto(pdf, "1b. Rentabilidad real (precio de compra vs valor actual)",
                              pnl.to_string().splitlines() + [
                                  "",
                                  "P&L_i = Valor_actual_i - (Precio_compra_i x Titulos_i)",
                                  "Coste de cada posicion = precio de compra x numero de titulos.",
                                  "A diferencia del resto del informe (precios historicos de mercado),",
                                  "esta es TU rentabilidad real, incluyendo el momento exacto de compra.",
                              ])
        except Exception as exc:
            _pagina_texto(pdf, "1b. Rentabilidad real", [f"ERROR: {exc}"])

        # --- Correlaciones ---------------------------------------------------
        try:
            corr = analizar_correlaciones(retornos, ventana=VENTANAS_VOL[-1])
            pdf.savefig(plot_heatmaps_correlacion(corr.pearson, corr.spearman), bbox_inches="tight")
            plt.close("all")
            _pagina_texto(pdf, "2. Correlaciones",
                          corr.comparativa.to_string(index=False).splitlines() + [""] + corr.alertas)
        except Exception as exc:
            _pagina_texto(pdf, "2. Correlaciones", [f"ERROR: {exc}"])

        # --- Optimización ----------------------------------------------------
        try:
            opt = optimizar_cartera(retornos, rf=RF, ticker_mercado=precios.columns[-1])
            mi_cartera = cartera_desde_importes(cartera_valida, opt.mu_anual, opt.cov_anual, RF)

            pdf.savefig(plot_frontera_eficiente(
                opt.nube_montecarlo, opt.frontera_eficiente, opt.maximo_sharpe,
                opt.minima_varianza, RF, cartera_actual=mi_cartera,
            ), bbox_inches="tight")
            plt.close("all")

            pdf.savefig(plot_pesos_comparativa([
                mi_cartera, opt.maximo_sharpe, opt.minima_varianza, opt.paridad_riesgo,
            ]), bbox_inches="tight")
            plt.close("all")

            tabla = comparar_carteras([
                mi_cartera, opt.maximo_sharpe, opt.minima_varianza, opt.paridad_riesgo,
            ])
            _pagina_texto(pdf, "3. Optimización de cartera",
                          tabla.to_string().splitlines() + ["", "CAPM:"] +
                          opt.capm.to_string().splitlines())
        except Exception as exc:
            _pagina_texto(pdf, "3. Optimización", [f"ERROR: {exc}", traceback.format_exc()[:800]])

        # --- Volatilidad histórica -------------------------------------------
        vol_actual = None
        try:
            vol = analizar_volatilidad(retornos, ventanas=VENTANAS_VOL, pesos_cartera=pesos)
            pdf.savefig(plot_volatilidad_rolling(
                vol.rolling[VENTANAS_VOL[0]], vol.rolling[VENTANAS_VOL[1]], *VENTANAS_VOL,
            ), bbox_inches="tight")
            plt.close("all")

            if vol.vol_cartera is not None and not vol.vol_cartera.empty:
                vol_actual = float(vol.vol_cartera.iloc[-1] * 100)

            _pagina_texto(pdf, "4. Volatilidad histórica",
                          vol.regimen.to_string().splitlines() + [""] + vol.alertas)
        except Exception as exc:
            _pagina_texto(pdf, "4. Volatilidad histórica", [f"ERROR: {exc}"])

        # --- Benchmarks --------------------------------------------------------
        try:
            from m6_benchmarks_volatilidad import descargar_benchmarks
            precios_bench = descargar_benchmarks(BENCHMARKS, anios=ANIOS)
            ret_bench = alinear_con_cartera(retornos, precios_bench)

            pdf.savefig(plot_cartera_vs_benchmarks(
                retornos, pesos, ret_bench, ventana=VENTANAS_VOL[-1],
            ), bbox_inches="tight")
            plt.close("all")

            comp = comparar_volatilidad_benchmarks(retornos, ret_bench, pesos, VENTANAS_VOL[-1])
            _pagina_texto(pdf, "5. Tu cartera frente al mercado", comp.to_string().splitlines())
        except Exception as exc:
            _pagina_texto(pdf, "5. Benchmarks", [f"ERROR: {exc}"])

        # --- Superficie de volatilidad implícita -------------------------------
        if incluir_superficie:
            try:
                from m7_superficie_iv import analizar_superficie, plot_smile_y_estructura

                sup = analizar_superficie(
                    PROXY_IV, vol_realizada=vol_actual,
                    guardar_png=False, guardar_html=False,
                )
                pdf.savefig(plot_smile_y_estructura(sup), bbox_inches="tight")
                plt.close("all")

                lineas = [f"{k:38s} {v}" for k, v in sup.metricas.items()] + [""] + sup.alertas
                _pagina_texto(pdf, f"6. Volatilidad implícita ({PROXY_IV})", lineas)
            except Exception as exc:
                _pagina_texto(pdf, "6. Volatilidad implícita", [f"ERROR: {exc}"])

        # --- Metadatos del PDF -------------------------------------------------
        metadatos = pdf.infodict()
        metadatos["Title"] = f"Informe de cartera {hoy:%Y-%m-%d}"
        metadatos["Author"] = "Panel Quant"
        metadatos["CreationDate"] = hoy

    logger.info("Informe generado: %s (%.1f KB)", ruta_salida, ruta_salida.stat().st_size / 1024)
    return ruta_salida


# =============================================================================
# ENVÍO POR EMAIL
# =============================================================================

def enviar_por_email(
    ruta_pdf: Path,
    asunto: str | None = None,
    cuerpo: str | None = None,
) -> None:
    """Envía el PDF adjunto por email usando SMTP de Gmail.

    Las credenciales se leen SIEMPRE de variables de entorno, nunca del código:
    un token en el código acaba en el historial de git y de ahí no se borra fácil.
    """
    origen = os.environ.get("EMAIL_ORIGEN")
    password = os.environ.get("EMAIL_PASSWORD")
    destino = os.environ.get("EMAIL_DESTINO", origen)

    if not origen or not password:
        raise ErrorDeInforme(
            "Faltan las variables de entorno EMAIL_ORIGEN y/o EMAIL_PASSWORD.\n"
            "Configúralas con:\n"
            '  export EMAIL_ORIGEN="tucorreo@gmail.com"\n'
            '  export EMAIL_PASSWORD="contraseña de aplicación de 16 caracteres"\n'
            "Recuerda: Gmail exige una contraseña de APLICACIÓN, no la habitual."
        )

    mensaje = EmailMessage()
    mensaje["Subject"] = asunto or f"📊 Informe de cartera — {datetime.now():%d/%m/%Y}"
    mensaje["From"] = origen
    mensaje["To"] = destino
    mensaje.set_content(
        cuerpo or
        "Adjunto el informe diario de tu cartera.\n\n"
        "Incluye: ficha de activos, correlaciones, frontera eficiente, "
        "volatilidad histórica y superficie de volatilidad implícita.\n\n"
        "Generado automáticamente. No constituye asesoramiento financiero."
    )

    with open(ruta_pdf, "rb") as fichero:
        mensaje.add_attachment(
            fichero.read(), maintype="application", subtype="pdf", filename=ruta_pdf.name,
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as servidor:
            servidor.login(origen, password)
            servidor.send_message(mensaje)
    except smtplib.SMTPAuthenticationError as exc:
        raise ErrorDeInforme(
            f"Gmail rechazó las credenciales: {exc}. Comprueba que EMAIL_PASSWORD es una "
            "contraseña de APLICACIÓN (16 caracteres, generada en myaccount.google.com) "
            "y no tu contraseña normal."
        ) from exc
    except Exception as exc:
        raise ErrorDeInforme(f"No se pudo enviar el email: {exc}") from exc

    logger.info("Informe enviado por email a %s", destino)


# =============================================================================
# ENVÍO POR TELEGRAM
# =============================================================================

def enviar_por_telegram(ruta_pdf: Path, mensaje: str | None = None) -> None:
    """Envía el PDF por Telegram usando la API de bots.

    Requiere `requests` (pip3 install requests). Para obtener las credenciales:
      1. Habla con @BotFather en Telegram → /newbot → te da el TOKEN
      2. Habla con @userinfobot → te da tu CHAT_ID
      3. Escribe algo a tu bot al menos una vez (si no, no puede escribirte él)
    """
    try:
        import requests
    except ImportError as exc:
        raise ErrorDeInforme("Falta 'requests'. Instálalo con: pip3 install requests") from exc

    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ErrorDeInforme(
            "Faltan TELEGRAM_TOKEN y/o TELEGRAM_CHAT_ID.\n"
            "  1. @BotFather → /newbot → copia el token\n"
            "  2. @userinfobot → copia tu chat id\n"
            '  3. export TELEGRAM_TOKEN="..." && export TELEGRAM_CHAT_ID="..."'
        )

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    texto = mensaje or f"📊 Informe de cartera — {datetime.now():%d/%m/%Y}"

    try:
        with open(ruta_pdf, "rb") as fichero:
            respuesta = requests.post(
                url,
                data={"chat_id": chat_id, "caption": texto},
                files={"document": (ruta_pdf.name, fichero, "application/pdf")},
                timeout=60,
            )
        respuesta.raise_for_status()
    except Exception as exc:
        raise ErrorDeInforme(f"No se pudo enviar por Telegram: {exc}") from exc

    logger.info("Informe enviado por Telegram al chat %s", chat_id)


# =============================================================================
# PLANTILLA DE GITHUB ACTIONS
# =============================================================================

def plantilla_github_actions() -> str:
    """Devuelve el YAML para automatizar el informe en la nube, gratis.

    Guárdalo en tu repositorio como .github/workflows/informe.yml y añade
    EMAIL_ORIGEN, EMAIL_PASSWORD y EMAIL_DESTINO en:
    Settings → Secrets and variables → Actions → New repository secret.
    """
    return """\
name: Informe diario de cartera

on:
  schedule:
    # 06:30 UTC = 08:30 en España (horario de verano). Cron usa SIEMPRE UTC.
    - cron: '30 6 * * 1-5'
  workflow_dispatch:        # permite lanzarlo a mano desde la pestaña Actions

jobs:
  informe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Instalar dependencias
        run: pip install yfinance pandas numpy scipy matplotlib seaborn requests

      - name: Generar y enviar informe
        env:
          EMAIL_ORIGEN: ${{ secrets.EMAIL_ORIGEN }}
          EMAIL_PASSWORD: ${{ secrets.EMAIL_PASSWORD }}
          EMAIL_DESTINO: ${{ secrets.EMAIL_DESTINO }}
        run: python m10_informe_automatico.py --email

      - name: Guardar el PDF como artefacto
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: informe-cartera
          path: '*.pdf'
          retention-days: 30
"""


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Genera y envía el informe diario de cartera.")
    parser.add_argument("--email", action="store_true", help="Enviar por email")
    parser.add_argument("--telegram", action="store_true", help="Enviar por Telegram")
    parser.add_argument("--sin-superficie", action="store_true",
                        help="Omitir la superficie de volatilidad implícita (más rápido)")
    parser.add_argument("--salida", type=str, default=None, help="Ruta del PDF de salida")
    parser.add_argument("--github-actions", action="store_true",
                        help="Imprimir la plantilla YAML de GitHub Actions y salir")
    args = parser.parse_args()

    if args.github_actions:
        print(plantilla_github_actions())
        return 0

    try:
        ruta = generar_informe(
            ruta_salida=args.salida,
            incluir_superficie=not args.sin_superficie,
        )
    except Exception as exc:
        logger.error("No se pudo generar el informe: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    codigo = 0
    if args.email:
        try:
            enviar_por_email(ruta)
        except ErrorDeInforme as exc:
            logger.error("%s", exc)
            codigo = 1

    if args.telegram:
        try:
            enviar_por_telegram(ruta)
        except ErrorDeInforme as exc:
            logger.error("%s", exc)
            codigo = 1

    if not args.email and not args.telegram:
        logger.info("PDF generado en '%s'. Usa --email o --telegram para enviarlo.", ruta)

    return codigo


if __name__ == "__main__":
    sys.exit(main())
