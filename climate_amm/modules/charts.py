"""
charts.py
=========
Módulo de visualizaciones para el dashboard climático AMM.
Todas las funciones retornan figuras de Plotly listas para st.plotly_chart().

Uso:
    from modules.charts import (
        chart_convergencia_mc,
        chart_ks_distribution,
        chart_series_runs,
        chart_promedios,
        chart_pollutants,
        chart_humidity_pressure,
        chart_forecast_radar,
        chart_amm_heatmap,
    )
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# ─── Paleta consistente ───────────────────────────────────────────
C_BLUE   = "#378ADD"
C_PURPLE = "#7F77DD"
C_TEAL   = "#1D9E75"
C_AMBER  = "#EF9F27"
C_RED    = "#E24B4A"
C_GRAY   = "#888780"
C_GREEN  = "#639922"

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, sans-serif", size=11, color="#73726c"),
    margin=dict(l=40, r=20, t=36, b=36),
    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                font=dict(size=10)),
    hoverlabel=dict(bgcolor="white", font_size=11),
)


def _fig(title="", height=280):
    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=title, font=dict(size=12, color="#444441"), x=0, xref="paper"
    ), height=height)
    return fig


# ─────────────────────────────────────────────────────────────────
#  1. CONVERGENCIA MONTE CARLO
# ─────────────────────────────────────────────────────────────────

def chart_convergencia_mc(convergence_curve: np.ndarray, rain_prob: float,
                           ci: tuple, municipio: str) -> go.Figure:
    """
    Muestra cómo la probabilidad de lluvia converge a medida que
    aumentan las simulaciones Monte Carlo.
    Incluye banda de confianza 95% y línea final.
    """
    n = len(convergence_curve)
    x = np.arange(1, n + 1)

    # Banda de confianza aproximada
    se = np.sqrt(convergence_curve * (1 - convergence_curve + 1e-9) / x)
    upper = np.clip(convergence_curve + 1.96 * se, 0, 1)
    lower = np.clip(convergence_curve - 1.96 * se, 0, 1)

    fig = _fig(f"Convergencia Monte Carlo — {municipio}", height=300)

    # Banda IC 95%
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(55,138,221,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 95%", hoverinfo="skip"
    ))
    # Curva de convergencia
    fig.add_trace(go.Scatter(
        x=x, y=convergence_curve,
        line=dict(color=C_BLUE, width=2),
        name="Prob. acumulada"
    ))
    # Valor final
    fig.add_hline(
        y=rain_prob, line_dash="dash",
        line_color=C_RED, line_width=1.5,
        annotation_text=f"Final: {rain_prob:.1%}",
        annotation_font_size=10
    )
    fig.update_xaxes(title_text="Número de simulaciones", gridcolor="#f0ede8")
    fig.update_yaxes(title_text="P(lluvia)", tickformat=".0%", gridcolor="#f0ede8",
                     range=[0, 1])
    return fig


# ─────────────────────────────────────────────────────────────────
#  2. DISTRIBUCIÓN KS vs UNIFORME
# ─────────────────────────────────────────────────────────────────

def chart_ks_distribution(numbers: np.ndarray, ks_stat: float,
                           p_value: float, passed: bool) -> go.Figure:
    """
    Compara la distribución empírica (ECDF) de los números GCL
    contra la distribución uniforme teórica.
    Resalta la distancia máxima D del estadístico KS.
    """
    sorted_n = np.sort(numbers)
    n = len(sorted_n)
    ecdf = np.arange(1, n + 1) / n
    theoretical = sorted_n  # CDF uniforme = x

    # Encontrar punto de máxima distancia
    diff = np.abs(ecdf - theoretical)
    idx_max = np.argmax(diff)

    fig = _fig("Prueba KS — ECDF vs Distribución Uniforme", height=300)

    # Uniforme teórica
    fig.add_trace(go.Scatter(
        x=sorted_n, y=theoretical,
        line=dict(color=C_GRAY, width=1.5, dash="dash"),
        name="Uniforme teórica"
    ))
    # ECDF empírica
    fig.add_trace(go.Scatter(
        x=sorted_n, y=ecdf,
        line=dict(color=C_BLUE, width=2),
        name="ECDF empírica"
    ))
    # Distancia máxima D
    x_d = sorted_n[idx_max]
    fig.add_shape(type="line",
        x0=x_d, y0=theoretical[idx_max],
        x1=x_d, y1=ecdf[idx_max],
        line=dict(color=C_RED if not passed else C_TEAL, width=2)
    )
    fig.add_annotation(
        x=x_d, y=(theoretical[idx_max] + ecdf[idx_max]) / 2,
        text=f"D={ks_stat:.4f}<br>p={p_value:.3f}",
        showarrow=True, arrowhead=2, font=dict(size=10),
        bgcolor="white", bordercolor="#ddd"
    )
    color_result = C_TEAL if passed else C_RED
    fig.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text=f"{'✓ H₀ Aceptada' if passed else '✗ H₀ Rechazada'}",
        showarrow=False, font=dict(size=11, color=color_result),
        bgcolor="white", bordercolor=color_result, borderwidth=1
    )
    fig.update_xaxes(title_text="Valor", gridcolor="#f0ede8", range=[0, 1])
    fig.update_yaxes(title_text="Probabilidad acumulada", gridcolor="#f0ede8", range=[0, 1])
    return fig


# ─────────────────────────────────────────────────────────────────
#  3. PRUEBA DE SERIES — RACHAS
# ─────────────────────────────────────────────────────────────────

def chart_series_runs(numbers: np.ndarray, runs: int,
                      expected_runs: float, z_stat: float,
                      passed: bool) -> go.Figure:
    """
    Visualiza la secuencia binaria (sobre/bajo mediana) y
    compara rachas observadas vs esperadas.
    """
    sample = numbers[:80]  # mostrar primeros 80 valores
    median = np.median(numbers)
    binary = (sample > median).astype(int)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=["Secuencia binaria (muestra 80 obs.)",
                                        "Rachas: observado vs esperado"])

    # Barras de la secuencia binaria
    colors = [C_BLUE if v else C_AMBER for v in binary]
    fig.add_trace(go.Bar(
        x=list(range(len(binary))), y=binary,
        marker_color=colors, name="Sobre/Bajo mediana",
        showlegend=True, width=0.8
    ), row=1, col=1)

    # Comparación rachas
    fig.add_trace(go.Bar(
        x=["Observadas", "Esperadas"],
        y=[runs, expected_runs],
        marker_color=[C_TEAL if passed else C_RED, C_GRAY],
        text=[str(runs), f"{expected_runs:.1f}"],
        textposition="outside",
        showlegend=False,
        name="Rachas"
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Prueba de Series (Runs Test)", font=dict(size=12), x=0),
        height=280,
        showlegend=False
    )
    fig.add_annotation(
        x=0.82, y=0.97, xref="paper", yref="paper",
        text=f"Z = {z_stat:.3f}<br>{'✓ Aleatoria' if passed else '✗ No aleatoria'}",
        showarrow=False, font=dict(size=10,
        color=C_TEAL if passed else C_RED),
        bgcolor="white", bordercolor=C_TEAL if passed else C_RED, borderwidth=1
    )
    fig.update_xaxes(gridcolor="#f0ede8")
    fig.update_yaxes(gridcolor="#f0ede8")
    return fig


# ─────────────────────────────────────────────────────────────────
#  4. PRUEBA DE PROMEDIOS — CONVERGENCIA DE LA MEDIA
# ─────────────────────────────────────────────────────────────────

def chart_promedios(numbers: np.ndarray, sample_mean: float,
                    z_stat: float, passed: bool) -> go.Figure:
    """
    Muestra la convergencia de la media muestral hacia 0.5
    a medida que aumenta el tamaño de muestra.
    """
    n = len(numbers)
    x = np.arange(1, n + 1)
    running_mean = np.cumsum(numbers) / x

    fig = _fig("Prueba de Promedios — Convergencia a μ = 0.5", height=280)

    # Zona de aceptación (±2σ teórica)
    sigma_upper = 0.5 + 2 * np.sqrt(1/12 / x)
    sigma_lower = 0.5 - 2 * np.sqrt(1/12 / x)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([sigma_upper, sigma_lower[::-1]]),
        fill="toself", fillcolor="rgba(29,158,117,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Zona aceptación ±2σ", hoverinfo="skip"
    ))
    # Media teórica
    fig.add_hline(y=0.5, line_color=C_TEAL, line_width=1.5, line_dash="dot",
                  annotation_text="μ teórica = 0.5", annotation_font_size=9)
    # Media corrida
    fig.add_trace(go.Scatter(
        x=x, y=running_mean,
        line=dict(color=C_PURPLE, width=2),
        name=f"Media muestral (final: {sample_mean:.4f})"
    ))

    fig.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text=f"μ̂ = {sample_mean:.4f}  Z = {z_stat:.3f}<br>{'✓ Converge' if passed else '✗ No converge'}",
        showarrow=False, font=dict(size=10, color=C_TEAL if passed else C_RED),
        bgcolor="white", bordercolor=C_TEAL if passed else C_RED, borderwidth=1
    )
    fig.update_xaxes(title_text="n", gridcolor="#f0ede8")
    fig.update_yaxes(title_text="Media acumulada", gridcolor="#f0ede8",
                     range=[0.3, 0.7])
    return fig


# ─────────────────────────────────────────────────────────────────
#  5. CONTAMINANTES — BARRAS HORIZONTALES
# ─────────────────────────────────────────────────────────────────

def chart_pollutants(pm25: float, pm10: float, no2: float,
                     co: float, o3: float, so2: float,
                     municipio: str) -> go.Figure:
    """Barras horizontales de contaminantes con umbrales OMS."""
    # Umbrales OMS (µg/m³ anual / 24h)
    limits = {"PM2.5": 15, "PM10": 45, "NO₂": 40, "CO": 4000,
              "O₃": 100, "SO₂": 40}
    values = {"PM2.5": pm25, "PM10": pm10, "NO₂": no2,
              "CO": co, "O₃": o3, "SO₂": so2}

    names  = list(values.keys())
    vals   = list(values.values())
    limits_list = [limits[k] for k in names]

    colors = [C_RED if v > l else C_AMBER if v > l * 0.7 else C_TEAL
              for v, l in zip(vals, limits_list)]

    fig = _fig(f"Contaminantes del aire — {municipio}", height=280)
    fig.add_trace(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
        name="Concentración"
    ))
    # Líneas de umbral OMS
    for name, limit in limits.items():
        idx = names.index(name)
        fig.add_shape(type="line",
            x0=limit, y0=idx - 0.4, x1=limit, y1=idx + 0.4,
            line=dict(color="#E24B4A", width=1.5, dash="dot")
        )
    fig.update_xaxes(title_text="µg/m³ (o mg/m³ CO)", gridcolor="#f0ede8")
    fig.update_yaxes(gridcolor="#f0ede8")
    return fig


# ─────────────────────────────────────────────────────────────────
#  6. SERIE DE TIEMPO — HUMEDAD & PRESIÓN
# ─────────────────────────────────────────────────────────────────

def chart_humidity_pressure(humidity: float, pressure: float,
                              municipio: str) -> go.Figure:
    """
    Genera una serie de tiempo simulada de 24h con los valores actuales
    como ancla, útil para mostrar la tendencia histórica-estimada.
    """
    rng = np.random.default_rng(abs(hash(municipio)) % (2**31))
    hours = list(range(-12, 13))
    hum_series  = np.clip(humidity  + np.cumsum(rng.normal(0, 1.2, 25)), 20, 100)
    pres_series = np.clip(pressure  + np.cumsum(rng.normal(0, 0.4, 25)), 990, 1030)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=hours, y=hum_series,
        line=dict(color=C_BLUE, width=2),
        fill="toself", fillcolor="rgba(55,138,221,0.08)",
        name="Humedad (%)"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=hours, y=pres_series,
        line=dict(color=C_PURPLE, width=2, dash="dot"),
        name="Presión (hPa)"
    ), secondary_y=True)
    fig.add_vline(x=0, line_color="#ddd", line_width=1, annotation_text="Ahora",
                  annotation_font_size=9)
    fig.update_layout(**LAYOUT_BASE,
        title=dict(text=f"Humedad & Presión 24h — {municipio}",
                   font=dict(size=12), x=0),
        height=280
    )
    fig.update_xaxes(title_text="Horas (0 = ahora)", gridcolor="#f0ede8",
                     tickvals=[-12, -6, 0, 6, 12])
    fig.update_yaxes(title_text="Humedad %", gridcolor="#f0ede8",
                     secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="hPa", secondary_y=True)
    return fig


# ─────────────────────────────────────────────────────────────────
#  7. RADAR DE VARIABLES CLIMÁTICAS
# ─────────────────────────────────────────────────────────────────

def chart_radar(humidity: float, pressure: float, wind_speed: float,
                pm25: float, cloud_cover: float,
                municipio: str) -> go.Figure:
    """
    Radar chart de las 5 variables predictoras normalizadas.
    Facilita comparación visual rápida entre municipios.
    """
    categories = ["Humedad", "Presión\n(inv.)", "Viento", "PM2.5", "Nubosidad"]
    # Normalizar a [0,1]
    hum_n  = humidity / 100
    pres_n = 1 - (pressure - 990) / 40
    wind_n = min(wind_speed / 50, 1)
    pm_n   = min(pm25 / 80, 1)
    cloud_n= cloud_cover / 100

    values = [hum_n, pres_n, wind_n, pm_n, cloud_n]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself", fillcolor="rgba(55,138,221,0.15)",
        line=dict(color=C_BLUE, width=2),
        name=municipio
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=9), gridcolor="#e8e5e0"),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        title=dict(text=f"Variables predictoras — {municipio}",
                   font=dict(size=12), x=0),
        height=300
    )
    return fig


# ─────────────────────────────────────────────────────────────────
#  8. HEATMAP AMM — COMPARATIVA DE MUNICIPIOS
# ─────────────────────────────────────────────────────────────────

def chart_amm_heatmap(all_data: dict, variable: str = "humidity") -> go.Figure:
    """
    Heatmap de una variable climática para todos los municipios del AMM.
    variable: "humidity" | "pressure" | "pm25" | "wind_speed" | "rain_prob_24h"
    """
    labels_map = {
        "humidity":      ("Humedad (%)", "%"),
        "pressure":      ("Presión atm. (hPa)", " hPa"),
        "pm25":          ("PM2.5 (µg/m³)", " µg/m³"),
        "wind_speed":    ("Velocidad viento (km/h)", " km/h"),
        "rain_prob_24h": ("Prob. lluvia 24h", "%"),
    }

    title, unit = labels_map.get(variable, (variable, ""))
    municipios  = list(all_data.keys())
    values      = [getattr(all_data[m], variable) * (100 if variable == "rain_prob_24h" else 1)
                   for m in municipios]

    colors = {
        "humidity":      "Blues",
        "pressure":      "Purp",
        "pm25":          "Reds",
        "wind_speed":    "Teal",
        "rain_prob_24h": "Blues",
    }.get(variable, "Viridis")

    fig = go.Figure(go.Bar(
        x=municipios, y=values,
        marker=dict(
            color=values,
            colorscale=colors,
            showscale=True,
            colorbar=dict(title=unit, thickness=12, len=0.8)
        ),
        text=[f"{v:.1f}{unit}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"AMM — {title}", font=dict(size=12), x=0),
        height=320,
        xaxis=dict(tickangle=-35, gridcolor="#f0ede8"),
        yaxis=dict(gridcolor="#f0ede8")
    )
    return fig


# ─────────────────────────────────────────────────────────────────
#  9. PANEL RESUMEN DE PRUEBAS ESTADÍSTICAS (4 subplots)
# ─────────────────────────────────────────────────────────────────

def chart_tests_summary(suite) -> go.Figure:
    """
    Panel de 4 mini-gráficas que resume el estado de todas las
    pruebas estadísticas para un municipio en una sola figura.
    """
    from modules.statistical_tests import StatTestSuite

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "KS — ECDF vs Uniforme",
            "Convergencia Monte Carlo",
            "Prueba de Series",
            "Prueba de Promedios"
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    numbers = suite.gcl.numbers
    n = len(numbers)
    x_idx = np.arange(1, n + 1)

    # ── KS ──
    sorted_n = np.sort(numbers)
    ecdf = np.arange(1, n + 1) / n
    fig.add_trace(go.Scatter(x=sorted_n, y=sorted_n, line=dict(color=C_GRAY, dash="dash", width=1), showlegend=False, name="Teórica"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sorted_n, y=ecdf, line=dict(color=C_BLUE, width=1.5), showlegend=False, name="ECDF"), row=1, col=1)

    # ── Monte Carlo convergencia ──
    fig.add_trace(go.Scatter(
        x=x_idx, y=suite.monte_carlo.convergence_curve,
        line=dict(color=C_PURPLE, width=1.5), showlegend=False
    ), row=1, col=2)
    fig.add_hline(y=suite.monte_carlo.rain_probability, line_dash="dash",
                  line_color=C_RED, line_width=1, row=1, col=2)

    # ── Series (media corrida) ──
    running = np.cumsum(numbers) / x_idx
    binary_sample = (numbers[:80] > np.median(numbers)).astype(int)
    colors_b = [C_BLUE if v else C_AMBER for v in binary_sample]
    fig.add_trace(go.Bar(
        x=list(range(80)), y=binary_sample,
        marker_color=colors_b, showlegend=False, width=0.8
    ), row=2, col=1)

    # ── Promedios ──
    sigma_u = 0.5 + 2 * np.sqrt(1/12 / x_idx)
    sigma_l = 0.5 - 2 * np.sqrt(1/12 / x_idx)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_idx, x_idx[::-1]]),
        y=np.concatenate([sigma_u, sigma_l[::-1]]),
        fill="toself", fillcolor="rgba(29,158,117,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=x_idx, y=running,
        line=dict(color=C_PURPLE, width=1.5), showlegend=False
    ), row=2, col=2)
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_TEAL, line_width=1, row=2, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"Resumen de Pruebas Estadísticas — {suite.municipio}",
            font=dict(size=13), x=0
        ),
        height=500,
        showlegend=False
    )
    return fig
