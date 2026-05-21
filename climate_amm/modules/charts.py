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
                     municipio: str,
                     mode: str = "pct") -> go.Figure:
    """
    Barras horizontales de contaminantes.

    mode="pct" (default):
        Muestra % del límite normativo (OMS / NOM-025).
        Todos los contaminantes en escala comparable 0-100%+.
        Ideal para ver qué contaminante está más cerca de la contingencia.

    mode="abs":
        Valores absolutos con eje logarítmico.
        Permite ver la magnitud real aunque haya diferencias de 1000x.
    """
    # Umbrales normativos
    limits = {"PM2.5": 15, "PM10": 45, "NO₂": 40, "CO": 4000,
              "O₃": 100, "SO₂": 40}
    units  = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO₂": "µg/m³",
              "CO": "µg/m³", "O₃": "µg/m³", "SO₂": "µg/m³"}
    values = {"PM2.5": pm25, "PM10": pm10, "NO₂": no2,
              "CO": co, "O₃": o3, "SO₂": so2}

    names  = list(values.keys())
    vals   = [values[k] for k in names]
    lims   = [limits[k] for k in names]

    if mode == "pct":
        # % del límite normativo
        pct_vals = [(v / l) * 100 if l else 0 for v, l in zip(vals, lims)]
        colors = [C_RED if p > 100 else C_AMBER if p > 70 else C_TEAL
                  for p in pct_vals]
        x_vals = pct_vals
        x_title = "% del límite normativo"
        text_vals = [f"{p:.0f}% ({v:.1f} {units[n]})" for p, v, n in zip(pct_vals, vals, names)]
        hover_tmpl = "%{y}<br>% del límite: %{x:.1f}%<br>Valor: %{customdata:.1f}<extra></extra>"
        show_thresh = True
        x_range = [0, max(max(pct_vals) * 1.25, 110)]
    else:
        # Valores absolutos con escala logarítmica
        colors = [C_RED if v > l else C_AMBER if v > l * 0.7 else C_TEAL
                  for v, l in zip(vals, lims)]
        x_vals = vals
        x_title = "Concentración (µg/m³) — escala logarítmica"
        text_vals = [f"{v:.1f}" for v in vals]
        hover_tmpl = "%{y}<br>%{x:.2f} µg/m³<extra></extra>"
        show_thresh = False
        x_range = None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals, y=names, orientation="h",
        marker_color=colors,
        text=text_vals,
        textposition="outside",
        textfont=dict(size=12, color="#FFFFFF"),
        customdata=vals if mode == "pct" else None,
        hovertemplate=hover_tmpl,
        name="Contaminante",
    ))

    if show_thresh:
        # Línea de umbral al 100%
        fig.add_vline(
            x=100, line_dash="dash", line_color=C_RED, line_width=2,
            annotation_text="Límite normativo (100%)",
            annotation_font_size=10, annotation_font_color=C_RED,
        )
    else:
        # Líneas de umbral individuales (solo modo absoluto)
        for name, limit in limits.items():
            idx = names.index(name)
            fig.add_shape(type="line",
                x0=limit, y0=idx - 0.4, x1=limit, y1=idx + 0.4,
                line=dict(color="#E24B4A", width=1.5, dash="dot")
            )

    base_no_margin = {k: v for k, v in LAYOUT_BASE.items() if k != "margin"}
    fig.update_layout(
        **base_no_margin,
        title=dict(
            text=f"Contaminantes — {municipio} ({'% límite' if mode == 'pct' else 'valores absolutos'})",
            font=dict(size=13, color="#444441"), x=0, xref="paper"
        ),
        height=400,
        margin=dict(l=40, r=120, t=46, b=40),
    )
    fig.update_xaxes(
        title_text=x_title, gridcolor="#f0ede8",
        type="log" if mode == "abs" else "linear",
        range=[0, np.log10(max(vals) * 2)] if mode == "abs" and max(vals) > 0 else x_range,
    )
    fig.update_yaxes(gridcolor="#f0ede8", tickfont=dict(size=12))
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


# ─────────────────────────────────────────────────────────────────
#  10. TRAYECTORIA PM2.5 — SIMULACIÓN DINÁMICA
# ─────────────────────────────────────────────────────────────────

def chart_pm25_trajectory(
    sim_result,
    M_star: float = 35.0,
    municipio: str = "",
    show_rain: bool = False,
    rain_start: float = 0.0,
    rain_duration: float = 0.0,
) -> go.Figure:
    """
    Gráfica de la trayectoria proyectada de PM2.5 con:
      • Línea de concentración simulada
      • Línea crítica M* (NOM-025)
      • Zonas de riesgo coloreadas
      • Anotación del Tipping Point si existe
    """
    t = sim_result.t
    M = sim_result.M

    fig = _fig(f"Proyección PM2.5 — {municipio}", height=360)

    # Zona segura (verde) y de riesgo (rojo) como bandas de fondo
    fig.add_hrect(y0=0, y1=M_star * 0.7, line_width=0,
                  fillcolor="rgba(29,158,117,0.06)", layer="below")
    fig.add_hrect(y0=M_star * 0.7, y1=M_star, line_width=0,
                  fillcolor="rgba(239,159,39,0.08)", layer="below")
    fig.add_hrect(y0=M_star, y1=max(np.max(M) * 1.15, M_star * 1.3), line_width=0,
                  fillcolor="rgba(228,75,74,0.06)", layer="below")

    # Trayectoria principal
    fig.add_trace(go.Scatter(
        x=t, y=M,
        line=dict(color=C_BLUE, width=2.5),
        name="PM2.5 proyectado",
        hovertemplate="t = %{x:.2f} h<br>PM2.5 = %{y:.1f} µg/m³<extra></extra>",
    ))

    # Línea crítica
    fig.add_hline(
        y=M_star, line_dash="dash", line_color=C_RED, line_width=1.8,
        annotation_text=f"M* = {M_star:.0f} µg/m³ (NOM-025)",
        annotation_font_size=10, annotation_font_color=C_RED,
    )

    # Tipping Point
    if sim_result.tipping_time is not None:
        idx = np.argmin(np.abs(t - sim_result.tipping_time))
        fig.add_vline(
            x=sim_result.tipping_time, line_dash="dot",
            line_color=C_RED, line_width=1.5,
        )
        fig.add_annotation(
            x=sim_result.tipping_time, y=M[idx],
            text=f"⚠ Tipping Point<br>{sim_result.tipping_time:.2f} h",
            showarrow=True, arrowhead=2, font=dict(size=10, color=C_RED),
            bgcolor="white", bordercolor=C_RED, borderwidth=1,
            ay=-40
        )

    # Ventana de lluvia
    if show_rain and rain_duration > 0:
        fig.add_vrect(
            x0=rain_start, x1=rain_start + rain_duration,
            fillcolor="rgba(55,138,221,0.12)", line_width=0,
            annotation_text="🌧 Lluvia", annotation_position="top left",
            annotation_font_size=9,
        )

    fig.update_xaxes(title_text="Tiempo (h)", gridcolor="#f0ede8", range=[0, np.max(t)])
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", gridcolor="#f0ede8",
                     range=[0, max(np.max(M) * 1.15, M_star * 1.2)])

    # Leyenda de zonas
    fig.add_annotation(
        x=0.99, y=0.99, xref="paper", yref="paper",
        text="<b>Zonas:</b> 🟢 Segura  🟡 Precaución  🔴 Contingencia",
        showarrow=False, font=dict(size=9, color="#555"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="#ddd", borderwidth=1,
        align="right",
    )
    return fig


# ─────────────────────────────────────────────────────────────────
#  11. SENSIBILIDAD PM2.5 — MÚLTIPLES ESCENARIOS DE VIENTO
# ─────────────────────────────────────────────────────────────────

def chart_pm25_sensitivity(
    sens_result,
    M_star: float = 35.0,
    municipio: str = "",
) -> go.Figure:
    """
    Panel comparativo de trayectorias PM2.5 para distintas variaciones
    de velocidad del viento (análisis de sensibilidad paramétrica).
    """
    fig = _fig(f"Análisis de sensibilidad — Viento vs PM2.5 ({municipio})", height=380)

    colors = px.colors.sequential.Viridis
    n = len(sens_result.simulations)

    for i, (sim, label) in enumerate(zip(sens_result.simulations, sens_result.labels)):
        color = colors[int(i * (len(colors) - 1) / max(n - 1, 1))]
        width = 3.0 if label == "0%" else 1.5
        dash = "solid" if label == "0%" else "dot"

        fig.add_trace(go.Scatter(
            x=sim.t, y=sim.M,
            line=dict(color=color, width=width, dash=dash),
            name=f"Viento {label}",
            hovertemplate=f"Viento {label}<br>t = %{{x:.2f}} h<br>PM2.5 = %{{y:.1f}} µg/m³<extra></extra>",
        ))

    # Línea crítica
    fig.add_hline(
        y=M_star, line_dash="dash", line_color=C_RED, line_width=1.5,
        annotation_text=f"M* = {M_star:.0f}", annotation_font_size=9,
    )

    fig.update_xaxes(title_text="Tiempo (h)", gridcolor="#f0ede8")
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", gridcolor="#f0ede8",
                     range=[0, max(max(s.max_M for s in sens_result.simulations) * 1.1, M_star * 1.15)])

    fig.update_layout(legend=dict(
        orientation="h", y=-0.22, x=0.5, xanchor="center",
        font=dict(size=10), title=dict(text="Variación viento:", font=dict(size=10))
    ))
    return fig


# ─────────────────────────────────────────────────────────────────
#  12. PRONÓSTICO POST-LLUVIA — TEMPERATURA ANTES/DESPUÉS
# ─────────────────────────────────────────────────────────────────

def chart_post_rain(hourly: list, event: dict, municipio: str = "") -> go.Figure:
    """
    Gráfica de temperatura horaria que resalta un evento de lluvia
    y muestra la tendencia térmica antes y después.
    """
    from datetime import datetime

    times = []
    temps = []
    probs = []
    for h in hourly[:48]:  # primeras 48h
        t = h.time_iso
        # Parsear ISO
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        times.append(dt.strftime("%H:%h"))
        temps.append(h.temp)
        probs.append(h.precipitation_prob)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Temperatura
    fig.add_trace(go.Scatter(
        x=list(range(len(times))), y=temps,
        line=dict(color=C_AMBER, width=2.5),
        name="Temperatura (°C)",
        hovertemplate="%{y:.1f}°C<extra></extra>",
    ), secondary_y=False)

    # Probabilidad de lluvia (barras de fondo)
    fig.add_trace(go.Bar(
        x=list(range(len(times))), y=probs,
        marker_color="rgba(55,138,221,0.25)", width=0.6,
        name="Prob. lluvia (%)",
        hovertemplate="%{y:.0f}%<extra></extra>",
    ), secondary_y=True)

    # Ventana de lluvia detectada
    start_i = event.get("start_idx", 0)
    end_i = event.get("end_idx", 0)
    if 0 <= start_i < len(times) and 0 <= end_i < len(times):
        fig.add_vrect(
            x0=start_i, x1=end_i,
            fillcolor="rgba(55,138,221,0.18)", line_width=0,
            annotation_text="🌧 Lluvia", annotation_position="top left",
            annotation_font_size=9,
        )

        # Temperatura antes del evento (pico)
        pre_start = max(0, start_i - 6)
        pre_temps = temps[pre_start:start_i]
        if pre_temps:
            max_pre = max(pre_temps)
            max_idx = pre_start + pre_temps.index(max_pre)
            fig.add_annotation(
                x=max_idx, y=max_pre,
                text=f"↑ {max_pre:.1f}°C",
                showarrow=True, arrowhead=2, font=dict(size=10, color=C_AMBER),
                ay=-30
            )

        # Temperatura después del evento (valle)
        post_end = min(len(temps), end_i + 7)
        post_temps = temps[end_i + 1:post_end]
        if post_temps:
            min_post = min(post_temps)
            min_idx = end_i + 1 + post_temps.index(min_post)
            fig.add_annotation(
                x=min_idx, y=min_post,
                text=f"↓ {min_post:.1f}°C",
                showarrow=True, arrowhead=2, font=dict(size=10, color=C_BLUE),
                ay=30
            )

    base_no_legend = {k: v for k, v in LAYOUT_BASE.items() if k != "legend"}
    fig.update_layout(
        **base_no_legend,
        title=dict(
            text=f"Pronóstico post-lluvia — {municipio}",
            font=dict(size=12), x=0
        ),
        height=320,
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
    )
    fig.update_xaxes(
        title_text="Hora", gridcolor="#f0ede8",
        tickvals=list(range(0, len(times), 6)),
        ticktext=[times[i] for i in range(0, len(times), 6)],
    )
    fig.update_yaxes(title_text="Temperatura (°C)", gridcolor="#f0ede8", secondary_y=False)
    fig.update_yaxes(title_text="Prob. lluvia (%)", range=[0, 105], gridcolor="#f0ede8", secondary_y=True)
    return fig
