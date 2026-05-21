"""
dashboard.py
============
Dashboard principal del Sistema de Pronóstico Climático AMM.

Ejecutar:
    streamlit run dashboard.py

Estructura de la app:
  ├── Sidebar: selección de municipio + parámetros del modelo
  ├── Tab 1: Condiciones actuales (KPIs + gráficas climáticas)
  ├── Tab 2: Pruebas estadísticas (GCL, KS, Series, Promedios, MC)
  └── Tab 3: Comparativa AMM (todos los municipios)
"""

import streamlit as st
import time
import numpy as np

# Importar módulos del proyecto
from modules import (
    WeatherClient,
    MUNICIPIOS_AMM,
    StatisticalEngine,
    chart_convergencia_mc,
    chart_ks_distribution,
    chart_series_runs,
    chart_promedios,
    chart_pollutants,
    chart_humidity_pressure,
    chart_radar,
    chart_amm_heatmap,
    chart_tests_summary,
)

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Pronóstico Climático AMM",
    page_icon="⛈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS global mínimo
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
  .block-container { padding-top: 1.2rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 13px; }
  div[data-testid="stExpander"] details { border: 0.5px solid #e8e5e0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CACHÉ DE DATOS (evita re-peticiones en cada interacción)
# ─────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner="Obteniendo datos climáticos…")
def load_all_data(api_key: str) -> dict:
    """Carga datos de todos los municipios. Caché 10 min."""
    client = WeatherClient(api_key=api_key if api_key else None)
    return client.get_all_municipios()


@st.cache_data(ttl=600, show_spinner="Ejecutando pruebas estadísticas…")
def run_stat_tests(municipio: str, humidity: float, pressure: float,
                   wind_speed: float, pm25: float,
                   n_sim: int, seed: int):
    """
    Ejecuta el pipeline estadístico completo para un municipio.
    Caché 10 min — mismo resultado para mismos inputs.
    """
    engine = StatisticalEngine(seed=seed, n_simulations=n_sim)
    return engine.run_all(
        municipio=municipio,
        humidity=humidity,
        pressure=pressure,
        wind_speed=wind_speed,
        pm25=pm25,
    )


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⛈ Pronóstico Climático AMM")
    st.markdown("---")

    # API Key
    with st.expander("🔑 API Key (OpenWeather)", expanded=False):
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Pegar aquí tu key…",
            help="Sin key, el sistema usa datos simulados realistas."
        )
        if api_key:
            st.success("Key configurada ✓")
        else:
            st.info("Modo demo — datos simulados")

    st.markdown("---")

    # Selector de municipio
    st.markdown("**Municipio**")
    municipio_sel = st.selectbox(
        "Seleccionar municipio",
        options=list(MUNICIPIOS_AMM.keys()),
        index=0,          # General Escobedo por defecto
        label_visibility="collapsed"
    )
    meta = MUNICIPIOS_AMM[municipio_sel]
    st.caption(f"📍 {meta['lat']}° N, {abs(meta['lon'])}° W  ·  {meta['cat'].title()}")

    st.markdown("---")

    # Parámetros del modelo estadístico
    st.markdown("**Parámetros del modelo**")
    n_simulations = st.slider(
        "Simulaciones Monte Carlo", 200, 5000, 2000, 100,
        help="Más simulaciones = mayor precisión, mayor tiempo de cómputo."
    )
    gcl_seed = st.number_input(
        "Semilla GCL", min_value=1, max_value=99999, value=42,
        help="Semilla del Generador Congruencial Lineal."
    )
    alpha = st.select_slider(
        "Nivel de significancia α",
        options=[0.01, 0.05, 0.10], value=0.05
    )

    st.markdown("---")

    # Botón de actualización
    refresh = st.button("🔄 Actualizar datos", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Actualizado: {time.strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────
#  CARGA DE DATOS
# ─────────────────────────────────────────────

with st.spinner("Cargando datos…"):
    all_data = load_all_data(api_key if api_key else "")

data = all_data[municipio_sel]

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    src_badge = "🟢 API en vivo" if data.source == "openweather" else "🟡 Datos simulados"
    st.markdown(f"## {municipio_sel}  &nbsp; `{src_badge}`")
    st.caption(f"Zona Metropolitana de Monterrey · NL · México")
with col_h2:
    diagnosis_text = {
        "lluvia_alta":     "⛈ Lluvia alta",
        "lluvia_moderada": "🌦 Lluvia moderada",
        "despejado":       "☀️ Despejado",
    }

# ─────────────────────────────────────────────
#  TABS PRINCIPALES
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🌡 Condiciones actuales",
    "📊 Pruebas estadísticas",
    "🗺 Comparativa AMM"
])


# ════════════════════════════════════════════════════════════════
#  TAB 1 — CONDICIONES ACTUALES
# ════════════════════════════════════════════════════════════════

with tab1:

    # KPIs principales
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("💧 Humedad",     f"{data.humidity:.0f}%",
              f"{data.humidity - 65:+.0f}% vs prom.")
    k2.metric("🌡 Temperatura", f"{data.temperature:.1f}°C",
              f"ST {data.feels_like:.1f}°C")
    k3.metric("📊 Presión",     f"{data.pressure:.0f} hPa",
              "▼ Bajando" if data.pressure < 1009 else "→ Estable")
    k4.metric("💨 Viento",      f"{data.wind_speed:.1f} km/h",
              f"{data.wind_deg:.0f}° dir.")
    k5.metric("☁️ Nubosidad",   f"{data.cloud_cover:.0f}%",
              f"UV: {data.uvi:.1f}")
    k6.metric("🏭 AQI",         str(data.aqi),
              f"PM2.5: {data.pm25:.1f} µg")

    st.markdown("---")

    # Gráficas climáticas
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(
            chart_humidity_pressure(data.humidity, data.pressure, municipio_sel),
            use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            chart_radar(data.humidity, data.pressure, data.wind_speed,
                        data.pm25, data.cloud_cover, municipio_sel),
            use_container_width=True
        )

    # Contaminantes
    st.plotly_chart(
        chart_pollutants(data.pm25, data.pm10, data.no2,
                         data.co, data.o3, data.so2, municipio_sel),
        use_container_width=True
    )

    # Pronóstico 3 días
    if data.forecast_3d:
        st.markdown("#### Pronóstico 3 días")
        f1, f2, f3 = st.columns(3)
        for col, day in zip([f1, f2, f3], data.forecast_3d):
            prob = day["rain_prob"]
            icon = "⛈" if prob > 0.7 else "🌦" if prob > 0.4 else "🌤"
            label = "Hoy" if day == data.forecast_3d[0] else \
                    "Mañana" if day == data.forecast_3d[1] else "Pasado"
            col.metric(
                f"{icon} {label}",
                f"{prob*100:.0f}% lluvia",
                f"↑{day['temp_max']:.1f}° ↓{day['temp_min']:.1f}°"
            )
            col.caption(day["description"].capitalize())


# ════════════════════════════════════════════════════════════════
#  TAB 2 — PRUEBAS ESTADÍSTICAS
# ════════════════════════════════════════════════════════════════

with tab2:

    st.markdown(f"**Parámetros:** GCL Hull-Dobell · N = {n_simulations:,} sim · "
                f"Semilla = {gcl_seed} · α = {alpha}")

    # Ejecutar pruebas (con caché)
    suite = run_stat_tests(
        municipio=municipio_sel,
        humidity=data.humidity,
        pressure=data.pressure,
        wind_speed=data.wind_speed,
        pm25=data.pm25,
        n_sim=n_simulations,
        seed=gcl_seed,
    )

    # ── Diagnóstico principal ────────────────────────────────────
    diag_label = diagnosis_text.get(suite.diagnosis, "—")
    diag_color = {"lluvia_alta": "error",
                  "lluvia_moderada": "warning",
                  "despejado": "success"}.get(suite.diagnosis, "info")

    getattr(st, diag_color)(
        f"{diag_label} — Probabilidad: **{suite.rain_probability:.1%}** · "
        f"Confianza: **{suite.confidence:.1%}** · "
        f"Pruebas OK: **{'✓ Todas' if suite.all_passed else '⚠ Revisar'}**"
    )

    # ── Tabla resumen de pruebas ─────────────────────────────────
    col_t = st.columns(4)
    tests_summary = [
        ("GCL", suite.gcl.passed,
         f"Período: {suite.gcl.period:,}",
         f"a={suite.gcl.params['a']:,} · m=2³²"),
        ("KS", suite.ks.passed,
         f"D = {suite.ks.statistic:.4f}",
         f"p = {suite.ks.p_value:.4f}"),
        ("Series", suite.series.passed,
         f"Z = {suite.series.z_statistic:.3f}",
         f"Rachas: {suite.series.runs} obs. / {suite.series.expected_runs:.1f} esp."),
        ("Promedios", suite.promedios.passed,
         f"μ̂ = {suite.promedios.sample_mean:.4f}",
         f"p = {suite.promedios.p_value:.4f}"),
    ]
    for col, (name, passed, val, detail) in zip(col_t, tests_summary):
        icon = "✅" if passed else "⚠️"
        col.metric(f"{icon} {name}", val, detail)

    st.markdown("---")

    # ── Gráficas de pruebas ──────────────────────────────────────
    st.markdown("#### Visualización de pruebas")

    # Fila 1: KS + Monte Carlo
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(
            chart_ks_distribution(suite.gcl.numbers, suite.ks.statistic,
                                   suite.ks.p_value, suite.ks.passed),
            use_container_width=True
        )
    with col_g2:
        st.plotly_chart(
            chart_convergencia_mc(
                suite.monte_carlo.convergence_curve,
                suite.monte_carlo.rain_probability,
                suite.monte_carlo.confidence_interval,
                municipio_sel
            ),
            use_container_width=True
        )

    # Fila 2: Series + Promedios
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.plotly_chart(
            chart_series_runs(
                suite.gcl.numbers,
                suite.series.runs,
                suite.series.expected_runs,
                suite.series.z_statistic,
                suite.series.passed
            ),
            use_container_width=True
        )
    with col_g4:
        st.plotly_chart(
            chart_promedios(suite.gcl.numbers, suite.promedios.sample_mean,
                            suite.promedios.z_statistic, suite.promedios.passed),
            use_container_width=True
        )

    # Panel resumen 4-en-1
    with st.expander("📋 Panel resumen — 4 pruebas en una vista", expanded=False):
        st.plotly_chart(chart_tests_summary(suite), use_container_width=True)

    # Interpretación detallada
    with st.expander("📖 Interpretación técnica de resultados", expanded=False):
        st.markdown(f"""
**Generador Congruencial Lineal (GCL)**
- Parámetros: a = {suite.gcl.params['a']:,} · c = {suite.gcl.params['c']:,} · m = 2³² = {suite.gcl.params['m']:,}
- Período detectado: **{suite.gcl.period:,}** · Estado: {'✓ Período máximo alcanzado' if suite.gcl.passed else '⚠ Período reducido'}

**Prueba de Kolmogorov-Smirnov**
- {suite.ks.interpretation}
- D = {suite.ks.statistic:.4f} · Valor crítico α={alpha}: {suite.ks.critical_value:.4f}

**Prueba de Series (Runs Test)**
- {suite.series.interpretation}

**Prueba de Promedios**
- {suite.promedios.interpretation}

**Simulación Monte Carlo**
- N = {suite.monte_carlo.n_simulations:,} simulaciones
- P(lluvia) = {suite.monte_carlo.rain_probability:.4f}
- IC 95%: [{suite.monte_carlo.confidence_interval[0]:.4f}, {suite.monte_carlo.confidence_interval[1]:.4f}]
- Error estándar: {suite.monte_carlo.std_error:.5f} · {'✓ Varianza aceptable' if suite.monte_carlo.passed else '⚠ Aumentar N'}
        """)

    # Números GCL
    with st.expander("🔢 Primeros 20 números GCL generados", expanded=False):
        nums = suite.gcl.numbers[:20]
        cols = st.columns(10)
        for i, (col, n) in enumerate(zip(cols * 2, nums)):
            col.code(f"{n:.5f}")


# ════════════════════════════════════════════════════════════════
#  TAB 3 — COMPARATIVA AMM
# ════════════════════════════════════════════════════════════════

with tab3:

    st.markdown("#### Comparativa de municipios — Zona Metropolitana de Monterrey")

    var_options = {
        "Humedad (%)": "humidity",
        "Presión atm. (hPa)": "pressure",
        "PM2.5 (µg/m³)": "pm25",
        "Velocidad viento (km/h)": "wind_speed",
        "Probabilidad lluvia 24h": "rain_prob_24h",
    }
    var_sel = st.selectbox("Variable a comparar", list(var_options.keys()))

    st.plotly_chart(
        chart_amm_heatmap(all_data, var_options[var_sel]),
        use_container_width=True
    )

    # Tabla resumen AMM
    st.markdown("#### Resumen completo — todos los municipios")
    table_data = []
    for muni, d in all_data.items():
        table_data.append({
            "Municipio": muni,
            "Tipo": d.categoria.title(),
            "Hum. %": f"{d.humidity:.0f}",
            "Presión hPa": f"{d.pressure:.0f}",
            "Viento km/h": f"{d.wind_speed:.1f}",
            "PM2.5 µg": f"{d.pm25:.1f}",
            "AQI": str(d.aqi),
            "P(lluvia) 24h": f"{d.rain_prob_24h*100:.0f}%",
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Ejecutar pruebas para todos y mostrar resumen
    st.markdown("#### Diagnóstico estadístico — todos los municipios")
    with st.spinner("Ejecutando pruebas para todos los municipios…"):
        engine = StatisticalEngine(seed=int(gcl_seed), n_simulations=int(n_simulations))
        all_suites = engine.run_batch([
            {
                "nombre": muni,
                "humidity": d.humidity,
                "pressure": d.pressure,
                "wind_speed": d.wind_speed,
                "pm25": d.pm25,
            }
            for muni, d in all_data.items()
        ])

    diag_icons = {
        "lluvia_alta": "⛈",
        "lluvia_moderada": "🌦",
        "despejado": "☀️"
    }
    diag_rows = []
    for muni, s in all_suites.items():
        diag_rows.append({
            "Municipio": muni,
            "Diagnóstico": f"{diag_icons.get(s.diagnosis, '')} {diagnosis_text.get(s.diagnosis, s.diagnosis)}",
            "P(lluvia)": f"{s.rain_probability:.1%}",
            "Confianza": f"{s.confidence:.1%}",
            "KS": "✓" if s.ks.passed else "✗",
            "Series": "✓" if s.series.passed else "✗",
            "Promedios": "✓" if s.promedios.passed else "✗",
            "MC error": f"{s.monte_carlo.std_error:.4f}",
        })
    st.dataframe(diag_rows, use_container_width=True, hide_index=True)
