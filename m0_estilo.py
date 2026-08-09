"""
===============================================================================
 MÓDULO 0 — ESTILO PROFESIONAL COMPARTIDO (tema tipo terminal financiera)
===============================================================================
Paleta, CSS y plantillas de gráfico usadas por el dashboard (Módulo 9) y por
el resto de módulos que dibujan con matplotlib/plotly, para que todo el panel
tenga una apariencia consistente de terminal profesional (estilo IBKR/Bloomberg):
fondo oscuro, tipografía compacta, colores de acento discretos, y verde/rojo
reservados exclusivamente para ganancias/pérdidas.

Uso rápido
----------
    from m0_estilo import inyectar_css, aplicar_tema_matplotlib, aplicar_tema_plotly

    inyectar_css(st)              # una vez, justo tras st.set_page_config()
    aplicar_tema_matplotlib()     # una vez, antes de generar ningún gráfico
    st.plotly_chart(aplicar_tema_plotly(mi_figura), use_container_width=True)
===============================================================================
"""

from __future__ import annotations

# =============================================================================
# PALETA
# =============================================================================

NAVY_FONDO = "#0B1220"
NAVY_PANEL = "#141B2E"
GRIS_BORDE = "#2A3450"
GRIS_TEXTO = "#B8C2D9"
BLANCO = "#F2F4F8"
AZUL_ACENTO = "#3B82F6"
NARANJA_ACENTO = "#F59E0B"
VERDE_POSITIVO = "#22C55E"
ROJO_NEGATIVO = "#EF4444"

# Colorway para series categóricas (activos, benchmarks...). Verde/rojo se
# dejan fuera a propósito: en un panel financiero esos dos colores deben
# significar SIEMPRE "sube/baja", nunca "es la serie número 3".
PALETA_CATEGORICA = [
    "#3B82F6", "#F59E0B", "#A78BFA", "#06B6D4", "#F472B6",
    "#84CC16", "#FB923C", "#64748B", "#EAB308", "#38BDF8",
]


# =============================================================================
# CSS DE LA APP (Streamlit)
# =============================================================================

CSS_PROFESIONAL = f"""
<style>
    .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }}
    html, body, [class*="css"] {{
        font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    h1 {{ font-size: 1.55rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }}
    h2, .stMarkdown h2 {{
        font-size: 1.0rem !important; font-weight: 700 !important;
        text-transform: uppercase; letter-spacing: 0.05em; color: {GRIS_TEXTO} !important;
        border-bottom: 1px solid {GRIS_BORDE}; padding-bottom: 0.35rem; margin-top: 1.6rem;
    }}
    h3 {{ font-size: 0.95rem !important; font-weight: 700 !important; color: {BLANCO} !important; }}
    [data-testid="stMetric"] {{
        background-color: {NAVY_PANEL};
        border: 1px solid {GRIS_BORDE};
        border-radius: 3px;
        padding: 0.65rem 0.9rem;
    }}
    [data-testid="stMetricLabel"] {{
        text-transform: uppercase; font-size: 0.68rem !important;
        letter-spacing: 0.06em; color: {GRIS_TEXTO} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-variant-numeric: tabular-nums; font-size: 1.3rem !important;
    }}
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
        border: 1px solid {GRIS_BORDE}; border-radius: 3px;
    }}
    div[data-testid="stDataFrame"] * {{ font-variant-numeric: tabular-nums; }}
    button[data-baseweb="tab"] {{
        font-size: 0.82rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    [data-testid="stSidebar"] {{ border-right: 1px solid {GRIS_BORDE}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        text-transform: uppercase; font-size: 0.78rem !important; letter-spacing: 0.05em;
        color: {GRIS_TEXTO} !important; border: none !important; margin-top: 1.1rem;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {GRIS_TEXTO} !important; }}
    div[data-testid="stAlert"] {{ border-radius: 3px; }}
    hr {{ border-color: {GRIS_BORDE}; }}

    /* Barra superior tipo "ticker tape" */
    .barra-superior {{
        display: flex; gap: 0; border: 1px solid {GRIS_BORDE}; border-radius: 3px;
        overflow: hidden; margin-bottom: 1.1rem;
    }}
    .barra-superior .celda {{
        flex: 1; padding: 0.55rem 1rem; background-color: {NAVY_PANEL};
        border-right: 1px solid {GRIS_BORDE};
    }}
    .barra-superior .celda:last-child {{ border-right: none; }}
    .barra-superior .etiqueta {{
        font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: {GRIS_TEXTO}; margin-bottom: 0.15rem;
    }}
    .barra-superior .valor {{
        font-size: 1.15rem; font-weight: 700; color: {BLANCO};
        font-variant-numeric: tabular-nums;
    }}
    .barra-superior .valor.positivo {{ color: {VERDE_POSITIVO}; }}
    .barra-superior .valor.negativo {{ color: {ROJO_NEGATIVO}; }}
</style>
"""


def inyectar_css(st_module) -> None:
    """Inyecta el CSS profesional en la app. Pasa el módulo `streamlit` (st)."""
    st_module.markdown(CSS_PROFESIONAL, unsafe_allow_html=True)


def barra_superior(st_module, celdas: list[tuple[str, str, str | None]]) -> None:
    """Renderiza una fila de KPIs estilo 'ticker tape' de terminal financiera.

    celdas: lista de (etiqueta, valor, signo) donde signo es 'positivo',
    'negativo' o None (color neutro).
    """
    html = ['<div class="barra-superior">']
    for etiqueta, valor, signo in celdas:
        clase = f"valor {signo}" if signo else "valor"
        html.append(
            f'<div class="celda"><div class="etiqueta">{etiqueta}</div>'
            f'<div class="{clase}">{valor}</div></div>'
        )
    html.append("</div>")
    st_module.markdown("".join(html), unsafe_allow_html=True)


# =============================================================================
# PLOTLY
# =============================================================================

def construir_template_plotly():
    import plotly.graph_objects as go

    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=NAVY_PANEL,
            plot_bgcolor=NAVY_PANEL,
            font=dict(color=GRIS_TEXTO, family="Helvetica, Arial, sans-serif", size=12),
            colorway=PALETA_CATEGORICA,
            xaxis=dict(gridcolor=GRIS_BORDE, zerolinecolor=GRIS_BORDE, linecolor=GRIS_BORDE),
            yaxis=dict(gridcolor=GRIS_BORDE, zerolinecolor=GRIS_BORDE, linecolor=GRIS_BORDE),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=50, b=10),
            hoverlabel=dict(bgcolor=NAVY_FONDO, font_color=BLANCO),
        )
    )


def aplicar_tema_plotly(fig):
    """Aplica el tema profesional a una figura Plotly ya construida y la devuelve."""
    fig.update_layout(template=construir_template_plotly())
    return fig


# =============================================================================
# MATPLOTLIB
# =============================================================================

def aplicar_tema_matplotlib() -> None:
    """Configura rcParams globales para que TODOS los gráficos matplotlib
    generados en el proceso (Módulos 4, 6, 7, 8) usen el mismo tema oscuro.

    Se basa en rcParams (estado global de matplotlib), así que basta con
    llamarla una vez al arrancar el dashboard, después de importar los
    módulos que hacen `sns.set_theme(...)` — esta llamada debe ir DESPUÉS
    para que sus valores prevalezcan sobre los de seaborn.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": NAVY_PANEL,
        "axes.facecolor": NAVY_PANEL,
        "axes.edgecolor": GRIS_BORDE,
        "axes.labelcolor": GRIS_TEXTO,
        "axes.grid": True,
        "grid.color": GRIS_BORDE,
        "grid.alpha": 0.6,
        "text.color": GRIS_TEXTO,
        "xtick.color": GRIS_TEXTO,
        "ytick.color": GRIS_TEXTO,
        "axes.prop_cycle": plt.cycler(color=PALETA_CATEGORICA),
        "legend.facecolor": NAVY_PANEL,
        "legend.edgecolor": GRIS_BORDE,
        "legend.labelcolor": GRIS_TEXTO,
        "savefig.facecolor": NAVY_PANEL,
        "figure.edgecolor": NAVY_PANEL,
    })
