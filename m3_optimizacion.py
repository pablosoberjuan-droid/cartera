"""
===============================================================================
 MÓDULO 3 — OPTIMIZACIÓN DE CARTERAS
===============================================================================
Implementa y compara los 5 marcos teóricos pedidos:

  1) MPT (Markowitz)     → simular_montecarlo() + frontera_eficiente()
  2) CAPM (Sharpe)       → calcular_capm()  (Beta y E(Ri) por activo)
  3) Black-Litterman     → black_litterman()  (equilibrio + vistas subjetivas)
  4) Risk Parity (Dalio) → cartera_paridad_riesgo()
  5) Kelly (Thorp)       → kelly_fraccion_multiactivo() + kelly_fraccion_simple()

Todas las funciones trabajan sobre retornos DIARIOS (formato del Módulo 1) y
anualizan internamente. Los pesos son siempre "long-only" (sin cortos) y
suman 1, salvo Kelly, que por definición puede pedir apalancamiento (>100%)
o infra-inversión.

Uso rápido
----------
    from m1_datos import descargar_precios, calcular_retornos
    from m3_optimizacion import optimizar_cartera

    precios  = descargar_precios(["RHM.DE","IDR.MC","MC.PA","EUNL.DE"])
    retornos = calcular_retornos(precios)

    res = optimizar_cartera(retornos, rf=0.04, ticker_mercado="EUNL.DE")
    print(res.maximo_sharpe)
    print(res.paridad_riesgo)
    print(res.capm)
===============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("quant.optimizacion")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

DIAS_HABILES_ANIO: int = 252
TASA_LIBRE_RIESGO_DEFECTO: float = 0.04


class ErrorDeOptimizacion(Exception):
    """Se lanza cuando un optimizador no converge o los datos son inválidos."""


# =============================================================================
# CONTENEDORES DE RESULTADOS
# =============================================================================

@dataclass
class CarteraOptima:
    """Una cartera resultante de cualquier método (Sharpe, MinVar, Risk Parity...)."""
    nombre: str
    pesos: pd.Series               # índice = tickers, suma 1.0 (salvo Kelly)
    retorno_esperado: float        # anualizado
    volatilidad: float             # anualizada
    sharpe: float
    extra: dict = field(default_factory=dict)  # info adicional específica del método

    def __repr__(self) -> str:
        pesos_str = ", ".join(f"{k}={v:.1%}" for k, v in self.pesos.items())
        return (
            f"<{self.nombre}> Retorno={self.retorno_esperado:.2%} "
            f"Vol={self.volatilidad:.2%} Sharpe={self.sharpe:.2f} | {pesos_str}"
        )


@dataclass
class ResultadoOptimizacion:
    """Empaqueta todos los resultados del Módulo 3 para pasar al Módulo 4 (gráficos)."""
    nube_montecarlo: pd.DataFrame       # 20.000 carteras aleatorias (retorno, vol, sharpe, pesos)
    maximo_sharpe: CarteraOptima
    minima_varianza: CarteraOptima
    frontera_eficiente: pd.DataFrame    # curva exacta (retorno objetivo -> vol mínima)
    capm: pd.DataFrame                  # Beta y E(Ri) por activo
    paridad_riesgo: CarteraOptima
    kelly_simple: pd.DataFrame          # aproximación binaria de Thorp, por activo
    kelly_multiactivo: pd.Series        # fracciones óptimas (pueden exceder 100%)
    black_litterman: "ResultadoBlackLitterman"
    rf: float
    mu_anual: pd.Series
    cov_anual: pd.DataFrame


@dataclass
class ResultadoBlackLitterman:
    pesos_equilibrio: pd.Series      # w_mkt supuesto (equilibrio de partida)
    retornos_implicitos: pd.Series   # Π: lo que el mercado "opina" implícitamente
    retornos_posteriores: pd.Series  # E[R] tras mezclar con las vistas del inversor
    cartera_resultante: CarteraOptima


@dataclass
class Vista:
    """Una opinión subjetiva del inversor sobre uno o más activos (para Black-Litterman).

    Ejemplos
    --------
    Vista absoluta:  "Creo que RHM.DE rendirá un 15% anual"
        Vista(activos={"RHM.DE": 1.0}, retorno_esperado=0.15, confianza=0.6)

    Vista relativa:  "Creo que IDR.MC batirá a EUNL.DE en un 5% anual"
        Vista(activos={"IDR.MC": 1.0, "EUNL.DE": -1.0}, retorno_esperado=0.05, confianza=0.4)
    """
    activos: dict[str, float]
    retorno_esperado: float
    confianza: float = 0.5  # 0 = sin confianza (se ignora), 1 = certeza absoluta


# =============================================================================
# ESTADÍSTICOS BASE
# =============================================================================

def _estadisticos_anuales(retornos: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Retorno medio anualizado (μ) y matriz de covarianzas anualizada (Σ)."""
    if retornos.empty or retornos.shape[1] < 2:
        raise ErrorDeOptimizacion("Se necesitan al menos 2 activos con retornos válidos.")
    mu = retornos.mean() * DIAS_HABILES_ANIO
    cov = retornos.cov(ddof=1) * DIAS_HABILES_ANIO
    return mu, cov


def _metricas_cartera(
    pesos: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float,
) -> tuple[float, float, float]:
    """Retorno, volatilidad y Sharpe de una cartera dados sus pesos."""
    retorno = float(pesos @ mu.values)
    varianza = float(pesos @ cov.values @ pesos)
    if varianza < 0:  # solo puede pasar por errores numéricos de coma flotante
        varianza = 0.0
    vol = float(np.sqrt(varianza))
    sharpe = (retorno - rf) / vol if vol > 1e-12 else np.nan
    return retorno, vol, sharpe


def _restricciones_basicas(n: int) -> tuple[list[dict], list[tuple[float, float]], np.ndarray]:
    """Restricciones estándar: pesos entre 0 y 1 (long-only), suma = 1."""
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    limites = [(0.0, 1.0)] * n
    x0 = np.full(n, 1.0 / n)  # punto de partida: equiponderado
    return restricciones, limites, x0


# =============================================================================
# 1) MPT — SIMULACIÓN DE MONTE CARLO
# =============================================================================

def simular_montecarlo(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_carteras: int = 20_000,
    rf: float = TASA_LIBRE_RIESGO_DEFECTO,
    semilla: int | None = 42,
) -> pd.DataFrame:
    """Genera `n_carteras` combinaciones aleatorias de pesos (long-only, suman 1)
    y calcula su retorno/volatilidad/Sharpe. Es la "nube de puntos" que dibuja
    la Frontera Eficiente de Markowitz.

    Usamos una distribución de Dirichlet (alpha=1) para muestrear pesos que
    suman exactamente 1 sin sesgo hacia el centro del simplex, a diferencia de
    normalizar uniformes independientes.
    """
    if n_carteras < 1000:
        logger.warning("n_carteras=%d es bajo; la nube de la Frontera Eficiente saldrá poco densa.", n_carteras)

    rng = np.random.default_rng(semilla)
    n_activos = len(mu)
    pesos_muestreados = rng.dirichlet(np.ones(n_activos), size=n_carteras)  # (n_carteras, n_activos)

    retornos = pesos_muestreados @ mu.values
    # varianza de cada cartera: w Σ w^T, vectorizado
    varianzas = np.einsum("ij,jk,ik->i", pesos_muestreados, cov.values, pesos_muestreados)
    varianzas = np.clip(varianzas, 0, None)
    vols = np.sqrt(varianzas)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(vols > 1e-12, (retornos - rf) / vols, np.nan)

    columnas_pesos = {f"peso_{t}": pesos_muestreados[:, i] for i, t in enumerate(mu.index)}
    nube = pd.DataFrame({
        "retorno": retornos,
        "volatilidad": vols,
        "sharpe": sharpes,
        **columnas_pesos,
    })

    logger.info("Monte Carlo: %d carteras simuladas · %d activos", n_carteras, n_activos)
    return nube


# =============================================================================
# 2) MPT — OPTIMIZACIÓN EXACTA (Máximo Sharpe, Mínima Varianza, Frontera)
# =============================================================================

def cartera_maximo_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float) -> CarteraOptima:
    """Cartera de tangencia: maximiza (E[Rp]-Rf)/σp. Es la cartera de mercado
    óptima según CAPM, el punto donde la Línea del Mercado de Capitales (CML)
    toca la Frontera Eficiente.
    """
    n = len(mu)
    restricciones, limites, x0 = _restricciones_basicas(n)

    def neg_sharpe(w: np.ndarray) -> float:
        ret, vol, _ = _metricas_cartera(w, mu, cov, rf)
        return -(ret - rf) / vol if vol > 1e-12 else 1e6

    resultado = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=limites, constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not resultado.success:
        raise ErrorDeOptimizacion(f"Máximo Sharpe no convergió: {resultado.message}")

    pesos = pd.Series(resultado.x, index=mu.index).clip(lower=0)
    pesos /= pesos.sum()  # renormaliza por si acaso hay ruido numérico
    ret, vol, sharpe = _metricas_cartera(pesos.values, mu, cov, rf)
    return CarteraOptima("Máximo Sharpe (Tangencia CAPM)", pesos, ret, vol, sharpe)


def cartera_minima_varianza(mu: pd.Series, cov: pd.DataFrame, rf: float) -> CarteraOptima:
    """Cartera del extremo izquierdo de la Frontera Eficiente: la de menor
    riesgo posible, sin importar el retorno.
    """
    n = len(mu)
    restricciones, limites, x0 = _restricciones_basicas(n)

    def varianza(w: np.ndarray) -> float:
        return float(w @ cov.values @ w)

    resultado = minimize(
        varianza, x0, method="SLSQP", bounds=limites, constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not resultado.success:
        raise ErrorDeOptimizacion(f"Mínima Varianza no convergió: {resultado.message}")

    pesos = pd.Series(resultado.x, index=mu.index).clip(lower=0)
    pesos /= pesos.sum()
    ret, vol, sharpe = _metricas_cartera(pesos.values, mu, cov, rf)
    return CarteraOptima("Mínima Varianza", pesos, ret, vol, sharpe)


def frontera_eficiente(
    mu: pd.Series, cov: pd.DataFrame, n_puntos: int = 50,
) -> pd.DataFrame:
    """Curva EXACTA de la Frontera Eficiente (no la nube de Monte Carlo):
    para cada nivel de retorno objetivo entre el mínimo y el máximo posible,
    encuentra la cartera de MENOR varianza que lo consigue.

    Además de 'retorno_objetivo' y 'volatilidad', devuelve una columna
    'peso_TICKER' por activo con la composición exacta de cada punto de la
    curva (mismo convenio que la nube de Monte Carlo), para poder inspeccionar
    en el gráfico interactivo qué cartera hay detrás de cada punto.
    """
    n = len(mu)
    ret_min, ret_max = float(mu.min()), float(mu.max())
    objetivos = np.linspace(ret_min, ret_max, n_puntos)

    filas = []
    x0 = np.full(n, 1.0 / n)
    for objetivo in objetivos:
        restricciones = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, obj=objetivo: w @ mu.values - obj},
        ]
        limites = [(0.0, 1.0)] * n

        resultado = minimize(
            lambda w: w @ cov.values @ w, x0, method="SLSQP",
            bounds=limites, constraints=restricciones,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if resultado.success:
            vol = float(np.sqrt(max(resultado.fun, 0)))
            fila = {"retorno_objetivo": objetivo, "volatilidad": vol}
            fila.update({f"peso_{t}": float(w) for t, w in zip(mu.index, resultado.x)})
            filas.append(fila)
            x0 = resultado.x  # siguiente punto arranca cerca del anterior (más rápido y estable)

    curva = pd.DataFrame(filas)
    if curva.empty:
        raise ErrorDeOptimizacion("Ningún punto de la Frontera Eficiente convergió.")
    return curva


# =============================================================================
# 3) CAPM — BETA Y RETORNO ESPERADO
# =============================================================================

def calcular_beta(retornos_activo: pd.Series, retornos_mercado: pd.Series) -> float:
    """β = Cov(Ri, Rm) / Var(Rm). Mide el riesgo SISTEMÁTICO: cuánto se mueve
    el activo por cada 1% que se mueve el mercado. β=1 → se mueve como el
    mercado; β>1 → amplifica sus movimientos; β<1 → los amortigua.
    """
    conjunto = pd.concat([retornos_activo, retornos_mercado], axis=1).dropna()
    if len(conjunto) < 30:
        raise ErrorDeOptimizacion("Muy pocas observaciones comunes para estimar Beta (mínimo 30).")
    cov = conjunto.cov(ddof=1).iloc[0, 1]
    var_mercado = conjunto.iloc[:, 1].var(ddof=1)
    if var_mercado < 1e-12:
        raise ErrorDeOptimizacion("La varianza del mercado es prácticamente cero; no se puede calcular Beta.")
    return float(cov / var_mercado)


def calcular_capm(
    retornos: pd.DataFrame,
    ticker_mercado: str | None,
    rf: float = TASA_LIBRE_RIESGO_DEFECTO,
) -> pd.DataFrame:
    """Calcula Beta y el retorno esperado CAPM de cada activo:

        E(Ri) = Rf + βi · [E(Rm) - Rf]

    Si `ticker_mercado` es None, usa como proxy de "mercado" la cartera
    equiponderada de todos los activos (razonable si no tienes un índice
    de referencia explícito, pero MENOS preciso que usar un ETF de mercado
    real como MSCI World).
    """
    if ticker_mercado is not None:
        if ticker_mercado not in retornos.columns:
            raise ErrorDeOptimizacion(
                f"'{ticker_mercado}' no está en las columnas de retornos: {list(retornos.columns)}"
            )
        r_mercado = retornos[ticker_mercado]
        fuente_mercado = ticker_mercado
    else:
        r_mercado = retornos.mean(axis=1)
        fuente_mercado = "Cartera equiponderada (proxy)"

    retorno_mercado_anual = r_mercado.mean() * DIAS_HABILES_ANIO
    prima_riesgo_mercado = retorno_mercado_anual - rf

    filas = []
    for activo in retornos.columns:
        if activo == ticker_mercado:
            beta = 1.0  # el mercado tiene Beta 1 respecto a sí mismo, por definición
        else:
            beta = calcular_beta(retornos[activo], r_mercado)
        e_ri_capm = rf + beta * prima_riesgo_mercado
        retorno_historico = retornos[activo].mean() * DIAS_HABILES_ANIO
        filas.append({
            "Activo": activo,
            "Beta": beta,
            "E(Ri) CAPM %": e_ri_capm * 100,
            "Retorno histórico %": retorno_historico * 100,
            "Diferencia (Hist-CAPM) %": (retorno_historico - e_ri_capm) * 100,
        })

    tabla = pd.DataFrame(filas).set_index("Activo").round(3)
    logger.info(
        "CAPM · mercado=%s · E(Rm)=%.2f%% · Rf=%.2f%%",
        fuente_mercado, retorno_mercado_anual * 100, rf * 100,
    )
    return tabla


# =============================================================================
# 4) RISK PARITY (Dalio / Bridgewater)
# =============================================================================

def _contribuciones_riesgo(pesos: np.ndarray, cov: pd.DataFrame) -> np.ndarray:
    """Contribución de cada activo al riesgo TOTAL de la cartera.

    RC_i = w_i · (Σw)_i / σ_p     donde σ_p = sqrt(wᵀΣw)

    La suma de todas las RC_i es igual a σ_p (se reparte el 100% del riesgo).
    """
    cov_w = cov.values @ pesos
    varianza = pesos @ cov_w
    vol = np.sqrt(max(varianza, 1e-16))
    return pesos * cov_w / vol


def cartera_paridad_riesgo(cov: pd.DataFrame) -> CarteraOptima:
    """Encuentra los pesos donde CADA activo aporta la MISMA contribución de
    riesgo a la cartera — no el mismo capital. Un activo 3 veces más volátil
    recibirá aproximadamente 1/3 del capital.

    Se resuelve minimizando la suma de diferencias al cuadrado entre todas
    las contribuciones de riesgo (el óptimo es cuando todas son iguales).
    """
    n = cov.shape[0]
    objetivo_por_activo = 1.0 / n  # cada activo debería aportar 1/n del riesgo total

    def dispersión_de_contribuciones(w: np.ndarray) -> float:
        rc = _contribuciones_riesgo(w, cov)
        rc_relativa = rc / rc.sum()  # normalizado a fracción del riesgo total
        return float(np.sum((rc_relativa - objetivo_por_activo) ** 2))

    restricciones, limites, x0 = _restricciones_basicas(n)
    # Evitamos pesos exactamente en 0 como punto de partida (rompería la RC de ese activo)
    limites = [(1e-4, 1.0)] * n

    resultado = minimize(
        dispersión_de_contribuciones, x0, method="SLSQP",
        bounds=limites, constraints=restricciones,
        options={"maxiter": 2000, "ftol": 1e-16},
    )
    if not resultado.success:
        raise ErrorDeOptimizacion(f"Risk Parity no convergió: {resultado.message}")

    pesos = pd.Series(resultado.x, index=cov.columns).clip(lower=0)
    pesos /= pesos.sum()

    rc_final = _contribuciones_riesgo(pesos.values, cov)
    rc_pct = pd.Series(rc_final / rc_final.sum(), index=cov.columns)

    # El retorno esperado no es parte del objetivo de Risk Parity; lo calculamos
    # aparte solo con fines informativos si se dispone de mu fuera de esta función.
    vol = float(np.sqrt(pesos.values @ cov.values @ pesos.values))

    return CarteraOptima(
        "Paridad de Riesgo (All Weather)",
        pesos,
        retorno_esperado=np.nan,  # se rellena en optimizar_cartera() si hay mu disponible
        volatilidad=vol,
        sharpe=np.nan,
        extra={"contribucion_riesgo_%": (rc_pct * 100).round(2)},
    )


# =============================================================================
# 5) CRITERIO DE KELLY (Edward Thorp)
# =============================================================================

def kelly_fraccion_simple(retornos: pd.DataFrame) -> pd.DataFrame:
    """Aplica la fórmula CLÁSICA de Kelly f* = (b·p - q) / b a cada activo,
    tratando cada día como una "apuesta" binaria de ganar/perder.

    Aquí:
      p = probabilidad histórica de un día positivo
      q = 1 - p
      b = ratio beneficio/pérdida = |ganancia media en días buenos| / |pérdida media en días malos|

    Es una SIMPLIFICACIÓN fuerte (los mercados no son apuestas binarias), útil
    como referencia intuitiva, pero para asignación real de cartera es
    preferible `kelly_fraccion_multiactivo`, que sí modela la distribución
    continua de retornos y las correlaciones entre activos.
    """
    filas = []
    for activo in retornos.columns:
        r = retornos[activo]
        positivos = r[r > 0]
        negativos = r[r < 0]

        p = len(positivos) / len(r)
        q = 1 - p
        ganancia_media = positivos.mean() if len(positivos) > 0 else 0.0
        perdida_media = abs(negativos.mean()) if len(negativos) > 0 else np.nan

        if not perdida_media or perdida_media < 1e-12:
            f_kelly = np.nan
            b = np.nan
        else:
            b = ganancia_media / perdida_media
            f_kelly = (b * p - q) / b

        filas.append({
            "Activo": activo,
            "P(día positivo)": p,
            "Ratio b (ganancia/pérdida)": b,
            "Kelly f* (simple)": f_kelly,
            "Kelly f*/2 (medio-Kelly, más prudente)": f_kelly / 2 if pd.notna(f_kelly) else np.nan,
        })

    return pd.DataFrame(filas).set_index("Activo").round(4)


def kelly_fraccion_multiactivo(
    mu: pd.Series, cov: pd.DataFrame, rf: float = TASA_LIBRE_RIESGO_DEFECTO,
) -> pd.Series:
    """Criterio de Kelly MULTIACTIVO (generalización continua de Thorp):

        f* = Σ⁻¹ (μ - Rf)

    Maximiza la tasa de crecimiento geométrico esperado del capital a largo
    plazo bajo el supuesto de retornos ~ Normal. A diferencia de las carteras
    anteriores, estas fracciones:
      · NO están acotadas a sumar 1 (Kelly puede pedir apalancamiento >100%,
        o incluso posiciones cortas si el signo de f* es negativo).
      · Kelly "completo" es agresivo y sensible a errores de estimación de μ;
        en la práctica los profesionales usan una fracción de Kelly (1/2 o 1/4)
        para reducir el riesgo de ruina por errores de estimación.
    """
    try:
        inversa_cov = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError as exc:
        raise ErrorDeOptimizacion(
            "La matriz de covarianzas es singular (activos perfectamente "
            "correlacionados o redundantes); Kelly multiactivo no se puede calcular."
        ) from exc

    exceso_retorno = (mu - rf).values
    f_estrella = inversa_cov @ exceso_retorno
    return pd.Series(f_estrella, index=mu.index)


# =============================================================================
# BLACK-LITTERMAN (Goldman Sachs)
# =============================================================================

def pesos_equilibrio_mercado(
    cov: pd.DataFrame, capitalizaciones: pd.Series | None = None,
) -> pd.Series:
    """Pesos de la cartera de "equilibrio" de partida para Black-Litterman.

    Lo correcto es usar capitalización bursátil real (cuanto más grande la
    empresa, más pesa "el mercado" en ella). Si no la tienes a mano, se usa
    equiponderación como aproximación razonable — pero AVISA de que es una
    aproximación, porque cambia los retornos implícitos resultantes.
    """
    activos = cov.columns
    if capitalizaciones is None:
        logger.warning(
            "Black-Litterman sin capitalizaciones de mercado: se usa equiponderación "
            "como proxy del equilibrio. Para mayor precisión, pasa `capitalizaciones`."
        )
        return pd.Series(1.0 / len(activos), index=activos)

    capitalizaciones = capitalizaciones.reindex(activos)
    if capitalizaciones.isna().any():
        faltantes = capitalizaciones[capitalizaciones.isna()].index.tolist()
        raise ErrorDeOptimizacion(f"Faltan capitalizaciones para: {faltantes}")
    return capitalizaciones / capitalizaciones.sum()


def retornos_implicitos(
    cov: pd.DataFrame, pesos_mercado: pd.Series, rf: float, delta: float = 2.5,
) -> pd.Series:
    """Retornos de equilibrio implícitos (Π), obtenidos por OPTIMIZACIÓN INVERSA:
    en vez de partir de μ para hallar los pesos óptimos (como en Markowitz),
    partimos de los pesos "de mercado" observados y preguntamos: ¿qué μ haría
    que ESTOS pesos fueran los óptimos?

        Π = δ · Σ · w_mercado  + Rf

    δ (delta) es el coeficiente de aversión al riesgo del mercado agregado;
    2.5 es el valor típico usado en la literatura (Black-Litterman, 1992).
    """
    pi = delta * (cov.values @ pesos_mercado.values) + rf
    return pd.Series(pi, index=cov.columns)


def _matriz_vistas(vistas: Sequence[Vista], activos: pd.Index, cov: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte la lista de Vista(...) del usuario a las matrices P, Q, Ω
    que exige la fórmula bayesiana de Black-Litterman."""
    k = len(vistas)
    n = len(activos)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    varianzas_vista = np.zeros(k)

    for i, vista in enumerate(vistas):
        for activo, peso in vista.activos.items():
            if activo not in activos:
                raise ErrorDeOptimizacion(f"La vista incluye un activo desconocido: '{activo}'")
            P[i, activos.get_loc(activo)] = peso
        Q[i] = vista.retorno_esperado

        confianza = min(max(vista.confianza, 1e-4), 1.0)  # evita división por 0
        # Incertidumbre estándar de Black-Litterman: proporcional a τ·P·Σ·Pᵀ,
        # escalada por (1/confianza - 1) para que confianza=1 → varianza≈0
        varianza_base = float(P[i] @ (tau * cov.values) @ P[i])
        varianzas_vista[i] = varianza_base * (1.0 / confianza - 1.0 + 1e-6)

    omega = np.diag(varianzas_vista)
    return P, Q, omega


def combinar_black_litterman(
    pi: pd.Series, cov: pd.DataFrame, vistas: Sequence[Vista], tau: float = 0.05,
) -> pd.Series:
    """Fórmula bayesiana de Black-Litterman: combina el equilibrio de mercado (Π)
    con las vistas subjetivas del inversor, ponderando cada una según su
    confianza declarada.

        E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

    Si no hay vistas, devuelve directamente Π (el equilibrio de mercado, sin
    ninguna opinión propia mezclada) — es el comportamiento correcto y
    documentado, no un caso de error.
    """
    if not vistas:
        logger.info("Black-Litterman sin vistas del inversor: se devuelve el equilibrio de mercado (Π).")
        return pi.copy()

    activos = cov.index
    P, Q, omega = _matriz_vistas(vistas, activos, cov, tau)

    tau_sigma_inv = np.linalg.inv(tau * cov.values)
    try:
        omega_inv = np.linalg.inv(omega)
    except np.linalg.LinAlgError as exc:
        raise ErrorDeOptimizacion(
            "No se pudo invertir Ω; revisa que ninguna vista tenga confianza=1.0 exacta "
            "combinada con otras vistas contradictorias."
        ) from exc

    termino_precision = tau_sigma_inv + P.T @ omega_inv @ P
    termino_media = tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q

    posterior = np.linalg.solve(termino_precision, termino_media)
    return pd.Series(posterior, index=activos)


def black_litterman(
    cov: pd.DataFrame,
    rf: float,
    vistas: Sequence[Vista] | None = None,
    capitalizaciones: pd.Series | None = None,
    delta: float = 2.5,
    tau: float = 0.05,
) -> ResultadoBlackLitterman:
    """Orquesta el flujo completo de Black-Litterman: equilibrio → vistas →
    retornos posteriores → cartera óptima resultante (máximo Sharpe con esos
    retornos posteriores, en vez de con la media histórica).
    """
    w_mkt = pesos_equilibrio_mercado(cov, capitalizaciones)
    pi = retornos_implicitos(cov, w_mkt, rf, delta)
    posterior = combinar_black_litterman(pi, cov, vistas or [], tau)
    cartera = cartera_maximo_sharpe(posterior, cov, rf)
    cartera.nombre = "Black-Litterman (equilibrio + vistas)"

    return ResultadoBlackLitterman(
        pesos_equilibrio=w_mkt,
        retornos_implicitos=pi,
        retornos_posteriores=posterior,
        cartera_resultante=cartera,
    )


# =============================================================================
# TU CARTERA REAL (a partir de los importes que tienes invertidos)
# =============================================================================

def cartera_desde_importes(
    importes: dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float,
) -> CarteraOptima:
    """Construye una CarteraOptima a partir de lo que TÚ tienes invertido realmente
    en cada activo — no una cartera optimizada, sino la tuya tal cual es hoy.

    Parameters
    ----------
    importes : dict {ticker: importe invertido}, en la MISMA moneda para todos
        (p. ej. euros). No hace falta que sumen nada en concreto: se normalizan
        automáticamente para calcular los pesos (%).
        Ejemplo: {"RHM.DE": 4000, "IDR.MC": 2500, "MC.PA": 3000,
                  "0P0001KGI5.F": 1500, "EUNL.DE": 5000}

    Returns
    -------
    CarteraOptima con nombre="Tu cartera actual". El campo `.extra` incluye
    además el importe en euros de cada posición y su contribución al riesgo.
    """
    faltantes = [t for t in importes if t not in mu.index]
    if faltantes:
        raise ErrorDeOptimizacion(
            f"Estos tickers de tu cartera no están entre los activos descargados: {faltantes}. "
            f"Activos disponibles: {list(mu.index)}. ¿Coinciden exactamente los símbolos?"
        )
    if any(v < 0 for v in importes.values()):
        raise ErrorDeOptimizacion(
            "Los importes no pueden ser negativos (esta función no admite posiciones cortas)."
        )

    total = sum(importes.values())
    if total <= 0:
        raise ErrorDeOptimizacion("La suma de tus importes debe ser mayor que 0.")

    # Cualquier activo descargado que NO esté en tus importes se asume con peso 0
    pesos = pd.Series(0.0, index=mu.index)
    for ticker, importe in importes.items():
        pesos[ticker] = importe / total

    ret, vol, sharpe = _metricas_cartera(pesos.values, mu, cov, rf)

    rc = _contribuciones_riesgo(pesos.values, cov)
    suma_rc = rc.sum()
    rc_pct = pd.Series(rc / suma_rc * 100 if suma_rc > 1e-12 else np.nan, index=mu.index)

    importes_completos = pd.Series(0.0, index=mu.index)
    for ticker, importe in importes.items():
        importes_completos[ticker] = importe

    logger.info(
        "Tu cartera actual · %d activos · %.2f invertido · Retorno=%.2f%% Vol=%.2f%% Sharpe=%.2f",
        len(importes), total, ret * 100, vol * 100, sharpe,
    )

    return CarteraOptima(
        "Tu cartera actual",
        pesos,
        ret, vol, sharpe,
        extra={
            "importe": importes_completos.round(2),
            "importe_total": total,
            "contribucion_riesgo_%": rc_pct.round(2),
        },
    )


def calcular_pnl_posiciones(
    compras: dict[str, dict[str, float]],
    valor_actual: dict[str, float],
) -> pd.DataFrame:
    """Plusvalía/minusvalía real de cada posición: precio de compra × títulos
    frente al valor actual de mercado.

    Parameters
    ----------
    compras : dict {ticker: {"precio_compra": float, "titulos": float}}.
        El coste de cada posición es precio_compra * titulos.
    valor_actual : dict {ticker: importe actual en euros} (p.ej. lo que ya
        pasas a cartera_desde_importes).

    Returns
    -------
    Una fila por posición (Títulos, Precio compra, Coste, Valor actual, P&L €,
    P&L %) más una fila 'TOTAL' con la cartera agregada.
    """
    faltantes = [t for t in compras if t not in valor_actual]
    if faltantes:
        raise ErrorDeOptimizacion(
            f"Faltan valores actuales para: {faltantes}. "
            f"Disponibles en valor_actual: {list(valor_actual)}"
        )

    filas = []
    for ticker, datos in compras.items():
        coste = datos["precio_compra"] * datos["titulos"]
        actual = valor_actual[ticker]
        pnl_eur = actual - coste
        pnl_pct = pnl_eur / coste * 100 if coste > 0 else np.nan
        filas.append({
            "Ticker": ticker,
            "Títulos": datos["titulos"],
            "Precio compra €": datos["precio_compra"],
            "Coste €": coste,
            "Valor actual €": actual,
            "P&L €": pnl_eur,
            "P&L %": pnl_pct,
        })

    tabla = pd.DataFrame(filas).set_index("Ticker").round(2)

    coste_total = tabla["Coste €"].sum()
    actual_total = tabla["Valor actual €"].sum()
    fila_total = pd.DataFrame([{
        "Títulos": np.nan,
        "Precio compra €": np.nan,
        "Coste €": round(coste_total, 2),
        "Valor actual €": round(actual_total, 2),
        "P&L €": round(actual_total - coste_total, 2),
        "P&L %": round((actual_total - coste_total) / coste_total * 100, 2) if coste_total > 0 else np.nan,
    }], index=["TOTAL"])

    return pd.concat([tabla, fila_total])


def comparar_carteras(carteras: Sequence[CarteraOptima]) -> pd.DataFrame:
    """Tabla resumen para poner "tu cartera actual" junto a las óptimas
    (Máximo Sharpe, Mínima Varianza, Risk Parity, Black-Litterman...) y ver
    de un vistazo cuánto te dejas sobre la mesa en rentabilidad ajustada a riesgo.
    """
    filas = []
    for c in carteras:
        filas.append({
            "Cartera": c.nombre,
            "Retorno %": c.retorno_esperado * 100,
            "Volatilidad %": c.volatilidad * 100,
            "Sharpe": c.sharpe,
        })
    return pd.DataFrame(filas).set_index("Cartera").round(3)


# =============================================================================
# ORQUESTADOR DEL MÓDULO
# =============================================================================

def optimizar_cartera(
    retornos: pd.DataFrame,
    rf: float = TASA_LIBRE_RIESGO_DEFECTO,
    ticker_mercado: str | None = None,
    vistas_bl: Sequence[Vista] | None = None,
    capitalizaciones_bl: pd.Series | None = None,
    n_carteras_mc: int = 20_000,
    n_puntos_frontera: int = 50,
) -> ResultadoOptimizacion:
    """Ejecuta las 5 teorías sobre el mismo conjunto de activos y devuelve
    todo empaquetado para el Módulo 4 (gráficos) y para inspección directa.

    Parameters
    ----------
    ticker_mercado : el ticker que hace de "mercado" para el CAPM (recomendado:
        tu ETF/índice más amplio, p.ej. "EUNL.DE" si tienes MSCI World en la cartera).
    vistas_bl : lista opcional de objetos Vista(...) para Black-Litterman.
    capitalizaciones_bl : capitalización bursátil de cada activo (index=tickers),
        para un equilibrio de Black-Litterman más realista que la equiponderación.
    """
    mu, cov = _estadisticos_anuales(retornos)

    logger.info("Optimizando cartera · %d activos · Rf=%.2f%%", len(mu), rf * 100)

    nube = simular_montecarlo(mu, cov, n_carteras=n_carteras_mc, rf=rf)
    max_sharpe = cartera_maximo_sharpe(mu, cov, rf)
    min_var = cartera_minima_varianza(mu, cov, rf)
    frontera = frontera_eficiente(mu, cov, n_puntos=n_puntos_frontera)
    capm = calcular_capm(retornos, ticker_mercado, rf)

    risk_parity = cartera_paridad_riesgo(cov)
    # Completamos retorno/Sharpe de Risk Parity ahora que sí tenemos mu:
    ret_rp, vol_rp, sharpe_rp = _metricas_cartera(risk_parity.pesos.values, mu, cov, rf)
    risk_parity.retorno_esperado, risk_parity.volatilidad, risk_parity.sharpe = ret_rp, vol_rp, sharpe_rp

    kelly_simple = kelly_fraccion_simple(retornos)
    kelly_multi = kelly_fraccion_multiactivo(mu, cov, rf)

    bl = black_litterman(cov, rf, vistas_bl, capitalizaciones_bl)

    return ResultadoOptimizacion(
        nube_montecarlo=nube,
        maximo_sharpe=max_sharpe,
        minima_varianza=min_var,
        frontera_eficiente=frontera,
        capm=capm,
        paridad_riesgo=risk_parity,
        kelly_simple=kelly_simple,
        kelly_multiactivo=kelly_multi,
        black_litterman=bl,
        rf=rf,
        mu_anual=mu,
        cov_anual=cov,
    )


# =============================================================================
# PRUEBA AUTÓNOMA — ejecutar:  python m3_optimizacion.py
# =============================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 25)

    try:
        from m1_datos import calcular_retornos, descargar_precios

        # 👇 tus tickers
        mis_tickers = ["RHM.DE", "IDR.MC", "MC.PA", "0P0001KGI5.F", "EUNL.DE"]

        precios = descargar_precios(mis_tickers, anios=3, cache_dir="./cache_datos")
        retornos = calcular_retornos(precios)

        vistas = [
            Vista(activos={"EUNL.DE": 1.0}, retorno_esperado=0.10, confianza=0.5),
            Vista(activos={"RHM.DE": 1.0, "MC.PA": -1.0}, retorno_esperado=0.05, confianza=0.3),
        ]

        res = optimizar_cartera(retornos, rf=0.04, ticker_mercado="EUNL.DE", vistas_bl=vistas)

        print("\n" + "=" * 78)
        print("MÓDULO 3 · OPTIMIZACIÓN DE CARTERAS")
        print("=" * 78)
        print("\n--- Nube Monte Carlo (5 primeras filas) ---")
        print(res.nube_montecarlo.head().round(4))
        print(f"\n--- {res.maximo_sharpe.nombre} ---")
        print(res.maximo_sharpe)
        print(f"\n--- {res.minima_varianza.nombre} ---")
        print(res.minima_varianza)
        print("\n--- Frontera Eficiente (primeros 5 puntos) ---")
        print(res.frontera_eficiente.head().round(4))
        print("\n--- CAPM ---")
        print(res.capm)
        print(f"\n--- {res.paridad_riesgo.nombre} ---")
        print(res.paridad_riesgo)
        print(res.paridad_riesgo.extra["contribucion_riesgo_%"])
        print("\n--- Kelly simple (aprox. binaria) ---")
        print(res.kelly_simple)
        print("\n--- Kelly multiactivo (f* = Σ⁻¹(μ-Rf)) ---")
        print(res.kelly_multiactivo.round(3))
        print("\n--- Black-Litterman: pesos de equilibrio ---")
        print(res.black_litterman.pesos_equilibrio.round(3))
        print("\n--- Black-Litterman: retornos implícitos (Π) ---")
        print((res.black_litterman.retornos_implicitos * 100).round(2))
        print("\n--- Black-Litterman: retornos posteriores (tras vistas) ---")
        print((res.black_litterman.retornos_posteriores * 100).round(2))
        print(f"\n--- {res.black_litterman.cartera_resultante.nombre} ---")
        print(res.black_litterman.cartera_resultante)

        print("\n[OK] Módulo 3 ejecutado sin errores.")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
