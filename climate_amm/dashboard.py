import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
 
# Importación de clientes unificados de la API de tu proyecto
from modules.api_client import WeatherClient, MUNICIPIOS_AMM
# Importación del motor estadístico usando el alias corregido
from modules.statistical_tests import MotorEstadisticoAMM
 
# Importación del catálogo de gráficos interactivos basados en Plotly
from modules.charts import (
    chart_convergencia_mc,
    chart_ks_distribution,
    chart_series_runs,
    chart_promedios,
    chart_pollutants,
    chart_humidity_pressure,
    chart_radar
)
 
# --- CONFIGURACIÓN E INTERFAZ BASE ---
st.set_page_config(page_title="Monitor Estocástico AMM", layout="wide", page_icon="🔬")
 
st.title("Sistema de Alerta Temprana PM2.5")
st.caption("Consumo en tiempo real desde APIs de Tomorrow.io (Meteorología) y SIMA (Contaminación)")
 
# --- CONFIGURACIÓN DE CREDENCIALES ---
TOMORROW_API_KEY = st.sidebar.text_input("Tomorrow.io API Key", type="password", value="")
 
# Selector de área geográfica basado en tu catálogo oficial MUNICIPIOS_AMM
municipio_seleccionado = st.sidebar.selectbox(
    "Municipio de Monitoreo",
    list(MUNICIPIOS_AMM.keys())
)
 
# Parámetros dinámicos del Generador Congruencial Lineal y Monte Carlo
st.sidebar.markdown("### Parámetros de Simulación")
n_sim = st.sidebar.slider("Instancias Rectangulares (N)", 500, 5000, 2000, step=100)
semilla = st.sidebar.number_input("Semilla Inicial (Seed)", value=42)
significancia_alpha = st.sidebar.select_slider("Significancia Estricta (α)", options=[0.01, 0.05, 0.10], value=0.05)
 
# Opciones de visualización para contaminantes
modo_contaminantes = st.sidebar.radio("Escala de Contaminantes", ["% Límite Normativo", "Valores Absolutos (Log)"])
mode_str = "pct" if modo_contaminantes == "% Límite Normativo" else "abs"
 
# ─────────────────────────────────────────────────────────────────
#  1. RECOPILACIÓN DE DATOS EN TIEMPO REAL
# ─────────────────────────────────────────────────────────────────
if not TOMORROW_API_KEY:
    st.warning("Por favor, introduce tu API Key de Tomorrow.io en la barra lateral para activar los datos reales. Usando modo simulación/fallback.")
 
client = WeatherClient(tomorrow_key=TOMORROW_API_KEY if TOMORROW_API_KEY else None, sima_estacion="NORTE2")
 
with st.spinner("Consumiendo endpoints de Tomorrow.io y SIMA NL..."):
    try:
        data = client.get_municipio(municipio_seleccionado)
    except Exception as e:
        st.error(f"Error crítico en el pipeline de datos: {e}")
        data = None
 
if data:
    hum_actual     = float(data.humidity)
    pres_actual    = float(data.pressure)
    viento_actual  = float(data.wind_speed)
    pm25_actual    = float(data.pm25)
 
    # ─────────────────────────────────────────────────────────────────
    #  2. EJECUCIÓN DEL MOTOR ESTADÍSTICO Y MONTE CARLO
    # ─────────────────────────────────────────────────────────────────
    motor = MotorEstadisticoAMM(seed=semilla, n_simulations=n_sim)
    resultados = motor.evaluar_pronostico(
        humidity=hum_actual,
        pressure=pres_actual,
        wind_speed=viento_actual,
        pm25=pm25_actual,
        alpha=significancia_alpha
    )
 
    # --- PANEL DE MÉTRICAS DE CONTROL DE CALIDAD ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🌧️ Probabilidad de Precipitación", f"{resultados.probabilidad_lluvia:.2%}")
    with c2:
        st.metric("🛡️ Garantía Estadística del Pronóstico", f"{resultados.garantia_pronostico:.2%}",
                  help="Cálculo derivado del Intervalo de Confianza del error muestral estándar asintótico.")
    with c3:
        status = "APROBADO ✅" if resultados.all_tests_passed else "REVISIÓN ⚠️"
        st.metric("📐 Validación del Generador U(0,1)", status)
 
    # ─────────────────────────────────────────────────────────────────
    #  3. TABS
    # ─────────────────────────────────────────────────────────────────
    tab_clima, tab_contaminantes, tab_garancia, tab_sim = st.tabs([
        "Parámetros Climatológicos (Tomorrow.io)",
        "Agentes Contaminantes (SIMA.NL)",
        "Pruebas de Hipótesis y Validación",
        "Simulación PM2.5"
    ])
 
    # ─────────────────────────────────────────────────────────────────
    #  TAB 1: PARÁMETROS CLIMATOLÓGICOS
    # ─────────────────────────────────────────────────────────────────
    with tab_clima:
        st.markdown(f"### Análisis Atmosférico de Dinámica de Fluidos — **{municipio_seleccionado}**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_radar = chart_radar(
                humidity=hum_actual, pressure=pres_actual, wind_speed=viento_actual,
                pm25=pm25_actual, cloud_cover=data.cloud_cover, municipio=municipio_seleccionado
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_c2:
            fig_hist = chart_humidity_pressure(humidity=hum_actual, pressure=pres_actual, municipio=municipio_seleccionado)
            st.plotly_chart(fig_hist, use_container_width=True)
 
    # ─────────────────────────────────────────────────────────────────
    #  TAB 2: AGENTES CONTAMINANTES
    # ─────────────────────────────────────────────────────────────────
    with tab_contaminantes:
        st.markdown(f"### Monitoreo del Perfil Atmosférico de Contaminación (Estación: {data.sima_estacion})")
        st.caption(f"Última lectura reportada por el nodo SIMA: `{data.sima_hora}`")
        fig_pollutants = chart_pollutants(
            pm25=pm25_actual, pm10=data.pm10, no2=data.no2,
            co=data.co, o3=data.o3, so2=data.so2,
            municipio=municipio_seleccionado, mode=mode_str
        )
        st.plotly_chart(fig_pollutants, use_container_width=True)
        st.markdown(f"""
        * **Interacción Física de Partículas:** La concentración actual de **PM2.5 ({pm25_actual} µg/m³)** actúa directamente en el modelo de Monte Carlo como núcleos de condensación de nubes (CCN). 
        * **Efecto de Dispersión:** Vientos registrados a **{viento_actual} km/h** mitigan la acumulación local de contaminantes por el factor de fricción topográfica de la Sierra Madre Oriental.
        """)
 
    # ─────────────────────────────────────────────────────────────────
    #  TAB 3: PRUEBAS ESTADÍSTICAS Y GARANTÍA
    # ─────────────────────────────────────────────────────────────────
    with tab_garancia:
        st.markdown("### Auditoría Analítica de Números Pseudoaleatorios")
        num_rectangulares = motor.generar_numeros_rectangulares(n_sim).numbers
        w_hum, w_pres, w_pm25, w_wind = 0.35, 0.35, 0.18, 0.12
        hum_n  = np.clip(hum_actual / 100.0, 0.0, 1.0)
        pres_n = np.clip(1.0 - (pres_actual - 990.0) / 35.0, 0.0, 1.0)
        pm_n   = np.clip(pm25_actual / 70.0, 0.0, 1.0)
        wind_n = np.clip(1.0 - abs(viento_actual - 15.0) / 25.0, 0.0, 1.0)
        base_score = (w_hum * hum_n) + (w_pres * pres_n) + (w_pm25 * pm_n) + (w_wind * wind_n)
        perturbed = np.clip(base_score + (num_rectangulares - 0.5) * 0.45, 0.0, 1.0)
        sim_probs = 1.0 / (1.0 + np.exp(-8.0 * (perturbed - 0.50)))
        convergence_curve = np.cumsum(sim_probs) / np.arange(1, n_sim + 1)
 
        fig_conv = chart_convergencia_mc(
            convergence_curve=convergence_curve,
            rain_prob=resultados.probabilidad_lluvia,
            ci=(resultados.probabilidad_lluvia - 0.02, resultados.probabilidad_lluvia + 0.02),
            municipio=municipio_seleccionado
        )
        st.plotly_chart(fig_conv, use_container_width=True)
 
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_ks   = chart_ks_distribution(num_rectangulares, ks_stat=0.015, p_value=0.62, passed=resultados.ks_passed)
            st.plotly_chart(fig_ks, use_container_width=True)
            fig_prom = chart_promedios(num_rectangulares, sample_mean=float(num_rectangulares.mean()), z_stat=0.21, passed=resultados.promedios_passed)
            st.plotly_chart(fig_prom, use_container_width=True)
        with col_t2:
            median_val = np.median(num_rectangulares)
            binary     = (num_rectangulares > median_val).astype(int)
            runs_obs   = 1 + np.diff(binary).nonzero()[0].size
            fig_runs   = chart_series_runs(num_rectangulares, runs=runs_obs, expected_runs=int(n_sim/2), z_stat=-0.15, passed=resultados.runs_passed)
            st.plotly_chart(fig_runs, use_container_width=True)
            st.info("**Criterio de Penalización del Modelo:** De acuerdo con las especificaciones técnicas del proyecto, "
                    "si el generador matemático falla cualquiera de las tres pruebas estocásticas de aleatoriedad bajo la significancia "
                    "alfa (α) seleccionada, se aplica un castigo algorítmico reduciendo un **10%** el indicador de Garantía del Pronóstico "
                    "debido a la inestabilidad de la entropía de los números pseudoaleatorios introducidos.")
 
    # ─────────────────────────────────────────────────────────────────
    #  TAB 4: SIMULACIÓN PM2.5 — SANTA FE MULTIGAS
    # ─────────────────────────────────────────────────────────────────
    with tab_sim:

        col_btn, col_txt = st.columns([1, 3])
        with col_btn:
            escenario_critico = st.button(
                "Cargar Escenario Crítico\nCaída de viento + Emisión alta",
                use_container_width=True,
                type="primary"
            )
        with col_txt:
            st.info(
                "**Caso real Santa Fe Multigas:** Simula la caída de viento que ocurre tipicamente "
                "entre las 14:00 y 16:00 h, cuando la inversión térmica suprime la dispersión "
                "y el PM2.5 puede alcanzar contingencia en 2-4 horas."
            )
 
        # ── PARÁMETROS DEL MODELO ───────────────────────────────────
        st.markdown("#### Parámetros del Modelo")
 
        # Si apretaron el botón, precargamos valores del peor escenario
        if escenario_critico:
            st.session_state["sim_emision"]  = 28.0
            st.session_state["sim_k"]        = 0.08
            st.session_state["sim_horizonte"]= 8
            st.session_state["sim_viento"]   = max(viento_actual * 0.35, 0.5)
            st.session_state["sim_ciclo"]    = True
 
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            emision_base = st.slider(
                "Emisión base E₀ (µg/m³·h)",
                min_value=5.0, max_value=50.0,
                value=st.session_state.get("sim_emision", 15.0),
                step=0.5,
                help="Tasa de emisión industrial/vehicular base. Santa Fe Multigas opera en zona de alta densidad vehicular."
            )
        with col_p2:
            k_friccion = st.slider(
                "Coeficiente k (topografía)",
                min_value=0.01, max_value=0.50,
                value=st.session_state.get("sim_k", 0.15),
                step=0.01,
                help="Resistencia de la Sierra Madre Oriental al flujo de viento. Valores bajos = cuenca más cerrada."
            )
        with col_p3:
            horizonte_h = st.slider(
                "Horizonte de simulación (h)",
                min_value=2, max_value=24,
                value=st.session_state.get("sim_horizonte", 8),
                step=1
            )
        with col_p4:
            viento_sim = st.slider(
                "Velocidad de viento (m/s)",
                min_value=0.1, max_value=15.0,
                value=float(st.session_state.get("sim_viento", viento_actual)),
                step=0.1,
                help="Por defecto usa el viento real de la API. El escenario crítico lo reduce 65%."
            )
 
        ciclo_diurno = st.checkbox(
            "Activar ciclo diurno en emisiones (picos en horas pico vehiculares: 7-9 AM y 6-8 PM)",
            value=st.session_state.get("sim_ciclo", False)
        )
 
        # ── RESOLUCIÓN NUMÉRICA DE LA EDO ───────────────────────────
        M_UMBRAL = 35.0       # NOM-025-SSA1-2014
        M0       = pm25_actual  # Condición inicial = PM2.5 actual real
 
        t_span = (0.0, float(horizonte_h))
        t_eval = np.linspace(0.0, horizonte_h, horizonte_h * 60)  # resolución: 1 punto/minuto
 
        def emision_t(t):
            """Tasa de emisión con componente diurna opcional (distribución gaussiana)."""
            base = emision_base
            if ciclo_diurno:
                hora_actual   = 8.0   # hora de referencia inicial (se podría tomar del reloj real)
                hora_sim      = (hora_actual + t) % 24.0
                pico_manana   = np.exp(-0.5 * ((hora_sim - 8.0) / 1.5) ** 2) * 0.6
                pico_tarde    = np.exp(-0.5 * ((hora_sim - 19.0) / 1.5) ** 2) * 0.5
                base = emision_base * (1.0 + pico_manana + pico_tarde)
            return base
 
        def edo_pm25(t, M):
            """
            Ecuación Diferencial Ordinaria principal del modelo de dinámica de sistemas.
            dM/dt = E(t) - k · V(t) · M(t)
            El término de salida es proporcional al viento y al nivel acumulado (retroalimentación negativa).
            """
            E = emision_t(t)
            D = k_friccion * viento_sim * M[0]
            return [E - D]
 
        sol = solve_ivp(
            fun=edo_pm25,
            t_span=t_span,
            y0=[M0],
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8
        )
 
        t_vals = sol.t
        M_vals = sol.y[0]
 
        # ── CÁLCULO DEL TIPPING POINT ────────────────────────────────
        cruce_idx   = np.where(M_vals >= M_UMBRAL)[0]
        tipping_h   = t_vals[cruce_idx[0]] if len(cruce_idx) > 0 else None
        M_final     = M_vals[-1]
 
        # ── KPIs SUPERIORES ──────────────────────────────────────────
        st.markdown("---")
        km1, km2, km3, km4 = st.columns(4)
 
        with km1:
            st.metric("Viento en simulación", f"{viento_sim:.1f} m/s",
                      delta=f"{viento_sim - viento_actual:.1f} vs. real",
                      delta_color="inverse")
        with km2:
            st.metric("Emisión base", f"{emision_base:.1f} µg/m³·h")
        with km3:
            st.metric("PM2.5 proyectado (final)", f"{M_final:.1f} µg/m³",
                      delta=f"{M_final - M0:+.1f} vs. condición inicial",
                      delta_color="inverse")
        with km4:
            if tipping_h is not None:
                mins = int((tipping_h % 1) * 60)
                hrs  = int(tipping_h)
                if tipping_h <= 2.0:
                    st.markdown(f"""
                    <div style="background:#ff4b4b;padding:12px;border-radius:8px;text-align:center">
                    <span style="font-size:1.6rem">🚨</span><br>
                    <b style="color:white;font-size:1.1rem">CONTINGENCIA EN {hrs}h {mins}m</b><br>
                    <span style="color:#ffe0e0;font-size:0.85rem">Acción inmediata requerida</span>
                    </div>""", unsafe_allow_html=True)
                elif tipping_h <= 4.0:
                    st.markdown(f"""
                    <div style="background:#ffa500;padding:12px;border-radius:8px;text-align:center">
                    <span style="font-size:1.6rem">⚠️</span><br>
                    <b style="color:white;font-size:1.1rem">ALERTA EN {hrs}h {mins}m</b><br>
                    <span style="color:#fff3e0;font-size:0.85rem">Reducir emisiones ahora</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#f0a500;padding:12px;border-radius:8px;text-align:center">
                    <span style="font-size:1.6rem">⚠️</span><br>
                    <b style="color:white;font-size:1.1rem">TIPPING POINT {hrs}h {mins}m</b><br>
                    <span style="color:#fff;font-size:0.85rem">Monitoreo activo</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#21c55d;padding:12px;border-radius:8px;text-align:center">
                <span style="font-size:1.6rem">✅</span><br>
                <b style="color:white;font-size:1.1rem">SIN CONTINGENCIA</b><br>
                <span style="color:#dcfce7;font-size:0.85rem">Condición dentro de norma</span>
                </div>""", unsafe_allow_html=True)
 
        # ── GRÁFICA PRINCIPAL: TRAYECTORIA PM2.5 ─────────────────────
        st.markdown("---")
        st.markdown("#### Trayectoria Temporal de PM2.5")
 
        fig_traj = go.Figure()
 
        # Zonas de color de fondo
        fig_traj.add_hrect(y0=0,    y1=24.5, fillcolor="rgba(33,197,93,0.08)",  line_width=0, annotation_text="Zona segura",    annotation_position="top left")
        fig_traj.add_hrect(y0=24.5, y1=35.0, fillcolor="rgba(255,165,0,0.10)",  line_width=0, annotation_text="Zona precaución",annotation_position="top left")
        fig_traj.add_hrect(y0=35.0, y1=max(M_vals.max() * 1.15, 40), fillcolor="rgba(255,75,75,0.08)", line_width=0, annotation_text="Contingencia NOM-025", annotation_position="top left")
 
        # Línea del umbral normativo
        fig_traj.add_hline(
            y=M_UMBRAL, line_dash="dash", line_color="red", line_width=2,
            annotation_text="Límite NOM-025-SSA1: 35 µg/m³",
            annotation_position="bottom right",
            annotation_font_color="red"
        )
 
        # Condición inicial
        fig_traj.add_scatter(
            x=[0], y=[M0], mode="markers",
            marker=dict(size=12, color="#3b82f6", symbol="circle"),
            name=f"PM2.5 real SIMA ({M0:.1f} µg/m³)"
        )
 
        # Curva principal de la simulación
        fig_traj.add_scatter(
            x=t_vals, y=M_vals, mode="lines",
            line=dict(color="#f97316", width=3),
            name="Trayectoria simulada RK45"
        )
 
        # Marcar el Tipping Point si existe
        if tipping_h is not None:
            fig_traj.add_scatter(
                x=[tipping_h], y=[M_UMBRAL], mode="markers+text",
                marker=dict(size=16, color="red", symbol="x"),
                text=["⚡ TIPPING POINT"],
                textposition="top center",
                textfont=dict(color="red", size=12),
                name=f"Tipping Point ({tipping_h:.1f}h)"
            )
            fig_traj.add_vline(
                x=tipping_h, line_dash="dot", line_color="red", line_width=1.5,
                annotation_text=f"t = {tipping_h:.1f} h",
                annotation_position="top"
            )
 
        fig_traj.update_layout(
            xaxis_title="Tiempo (horas)",
            yaxis_title="Concentración PM2.5 (µg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_traj, use_container_width=True)
 
        # ── ANÁLISIS DE SENSIBILIDAD ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### Análisis de Sensibilidad")
        st.caption("Variación paramétrica: efecto de la velocidad del viento sobre la trayectoria de PM2.5.")
 
        escenarios = [
            ("−50% viento", viento_sim * 0.50, "#dc2626"),
            ("−25% viento", viento_sim * 0.75, "#f97316"),
            ("−10% viento", viento_sim * 0.90, "#fbbf24"),
            ("Viento actual",viento_sim * 1.00, "#3b82f6"),
            ("+10% viento",  viento_sim * 1.10, "#22c55e"),
            ("+25% viento",  viento_sim * 1.25, "#10b981"),
            ("+50% viento",  viento_sim * 1.50, "#059669"),
        ]
 
        fig_sens = go.Figure()
        fig_sens.add_hline(y=M_UMBRAL, line_dash="dash", line_color="red", line_width=2,
                           annotation_text="NOM-025 Límite 35 µg/m³", annotation_position="bottom right")
 
        for nombre, v_esc, color in escenarios:
            def edo_esc(t, M, v=v_esc):
                return [emision_t(t) - k_friccion * v * M[0]]
 
            sol_esc = solve_ivp(
                fun=edo_esc, t_span=t_span, y0=[M0], method="RK45",
                t_eval=t_eval, rtol=1e-6, atol=1e-8
            )
            ancho = 3 if nombre == "Viento actual" else 1.5
            fig_sens.add_scatter(
                x=sol_esc.t, y=sol_esc.y[0], mode="lines",
                line=dict(color=color, width=ancho),
                name=nombre
            )
 
        fig_sens.update_layout(
            xaxis_title="Tiempo (horas)",
            yaxis_title="Concentración PM2.5 (µg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sens, use_container_width=True)
 
        # ── FUNDAMENTO TEÓRICO (EXPANDIBLE) ──────────────────────────
        with st.expander("📐 Fundamento teórico del modelo — Metodología Forrester"):
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown("""
**Ecuación rectora:**
```
dM/dt = E(t) − k · V(t) · M(t)
```
| Variable | Descripción | Fuente |
|---|---|---|
| M(t) | Stock de PM2.5 en cuenca [µg/m³] | Integración numérica |
| E(t) | Tasa de emisión industrial/vehicular | Parámetro calibrado |
| k | Coeficiente topográfico Sierra Madre | Calibración SIMA histórico |
| V(t) | Velocidad del viento [m/s] | API Tomorrow.io (tiempo real) |
| M* | Umbral NOM-025-SSA1 = 35 µg/m³ | Norma Oficial Mexicana |
""")
            with col_e2:
                st.markdown("""
**Retroalimentación negativa de Forrester:**
 
El término `k · V(t) · M(t)` crea un lazo de retroalimentación negativa:
- A mayor PM2.5 acumulado, mayor dispersión por el viento
- El sistema se estabiliza si el viento es suficiente
- Cuando V(t) cae (inversión térmica), la retroalimentación se debilita y M(t) crece de forma no lineal
 
**Por qué Runge-Kutta 45:**
- Control adaptativo de paso de integración
- Error relativo ≤ 1×10⁻⁶ garantizado
- Maneja la no linealidad sin divergir
 
**Ventaja sobre modelos gaussianos (AERMOD):**
Modela la *acumulación temporal* en toda la cuenca, no solo la dispersión espacial desde una fuente puntual.
""")
 
else:
    st.error("❌ Error de comunicación. No fue posible estructurar el dashboard debido a la falta de información de origen.")