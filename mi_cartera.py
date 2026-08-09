import pandas as pd

from m1_datos import descargar_precios, calcular_retornos, resumen_activos
from m2_correlaciones import analizar_correlaciones
from m3_optimizacion import calcular_pnl_posiciones

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

# 👇 tus tickers van aquí
mis_tickers = ["RHM.DE", "IDR.MC", "MC.PA", "0P0001KGI5.F", "EUNL.DE"]

# 👇 lo que pagaste por cada posición (precio de compra y nº de títulos)
mis_compras = {
    "RHM.DE": {"precio_compra": 497.04, "titulos": 1.69},
    "IDR.MC": {"precio_compra": 35.13, "titulos": 24.2},
    "MC.PA": {"precio_compra": 573.67, "titulos": 2.17},
    "0P0001KGI5.F": {"precio_compra": 205.55, "titulos": 9.875},
    "EUNL.DE": {"precio_compra": 11.55, "titulos": 43.29},
}

# 👇 valor actual de cada posición, en euros
mi_valor_actual = {
    "RHM.DE": 1955.0,
    "IDR.MC": 1670.0,
    "MC.PA": 1095.0,
    "0P0001KGI5.F": 2176.06,
    "EUNL.DE": 531.0,
}

precios = descargar_precios(mis_tickers, anios=3, cache_dir="./cache_datos")
retornos = calcular_retornos(precios)

print(resumen_activos(precios, retornos))

res = analizar_correlaciones(retornos, ventana=60)
print(res.pearson)
print(res.comparativa)

print("\n--- Rentabilidad real de tu cartera (compra vs valor actual) ---")
print(calcular_pnl_posiciones(mis_compras, mi_valor_actual))
