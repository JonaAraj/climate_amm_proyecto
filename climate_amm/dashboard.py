import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.statistical_tests import MotorEstadisticoAMM

st.title("🔬 Monitor de Garantía y Simulación Estocástica AMM")
st.caption("Consumo en tiempo real desde APIs de Tomorrow.io (Meteorología) y SIMA (Contaminación)")

# --- CARGA DE DATOS REALES DESDE EL CLIENTE INTEGRADO EN EL DASHBOARD ---
# El objeto `data` ya expone las mediciones actuales recopiladas por tu backend
hum_actual = float(data.humidity)
pres_actual = float(data.pressure)
viento_actual = float(data.wind_speed)
pm25_actual = float(data.pm25)

# Parámetros dinámicos configurables en interfaz
st.sidebar.markdown("### Parámetros de Simulación")
n_sim = st.sidebar.slider("Instancias Rectangulares Simuladas", 500, 5000, 2000, step=100)
semilla = st.sidebar.number_input("Semilla Inicial Rectangular", value=42)
significancia_alpha = st.sidebar.select_slider("Significancia Estricta (α)", options=[0.01, 0.05, 0.10], value=0.05)

# --- INSTANCIACIÓN Y INYECCIÓN DE DATOS DE API EN EL MOTOR ---
motor = MotorEstadisticoAMM(seed=semilla, n_simulations=n_sim)
resultados = motor.evaluar_pronostico(
    humidity=hum_actual,
    pressure=pres_actual,
    wind_speed=viento_actual,
    pm25=pm25_actual,
    alpha=significancia_alpha
)

# --- PANEL METRICAS COMPUESTAS DE GARANTÍA ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("🌧️ Probabilidad de Precipitación", f"{resultados.probabilidad_lluvia:.2%}")
with c2:
    st.metric("🛡️ Garantía Estadística del Pronóstico", f"{resultados.garantia_pronostico:.2%}", 
              help="Calculada en función del error estándar de convergencia y el cumplimiento de las hipótesis estocásticas.")
with c3:
    status = "APROBADO ✅" if resultados.all_tests_passed else "REVISIÓN ⚠️"
    st.metric("📐 Estado de Validadores U(0,1)", status)

# Detalle técnico de los validadores de números rectangulares
with st.expander("🔎 Estatus de Pruebas Estadísticas para Generadores Rectangulares"):
    st.markdown(f"""
    * **Prueba de Kolmogorov-Smirnov (Uniformidad):** {"`PASADO` ✅" if resultados.ks_passed else "`FALLADO` ❌"}
    * **Prueba de Promedios (Tendencia Central a 0.5):** {"`PASADO` ✅" if resultados.promedios_passed else "`FALLADO` ❌"}
    * **Prueba de Rachas (Independencia Serial):** {"`PASADO` ✅" if resultados.runs_passed else "`FALLADO` ❌"}
    
    *Los números rectangulares generados deben pasar estos filtros analíticos para asegurar la validez matemática del Monte Carlo.*
    """)

# --- RENDERIZADO DE GRÁFICOS MATPLOTLIB EN STREAMLIT ---
st.markdown("#### Análisis Sistemático de Distribución y Convergencia")

# Generamos datos locales para graficar el comportamiento del generador seleccionado
numeros_grafica = motor.generar_numeros_rectangulares(n_sim).numbers

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Verificación Física de Densidad Rectangular Uniforme
ax1.hist(numeros_grafica, bins=20, density=True, color="#3498db", alpha=0.75, edgecolor="white", label="Muestra Simulada")
ax1.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=2, label="Densidad Rectangular Teórica")
ax1.set_title("Densidad Rectangular del Generador U(0,1)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Intervalo de la Variable Aleatoria")
ax1.set_ylabel("Densidad Probabilística")
ax1.set_ylim(0, 1.5)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Estabilidad Asintótica del Error Estándar vs Número de Muestras
errores = []
pasos = np.linspace(10, n_sim, 50, dtype=int)
for p in pasos:
    sub_muestra = numeros_grafica[:p]
    # Representa la estabilidad y disminución de varianza en el cálculo de la probabilidad
    errores.append(sub_muestra.std() / np.sqrt(p))

ax2.plot(pasos, errores, color="#2ecc71", linewidth=2.5, marker="o", markersize=4, label="Error Estándar Muestral")
ax2.axhline(0.01, color="#7f8c8d", linestyle=":", label="Umbral Crítico de Estabilidad (0.01)")
ax2.set_title("Convergencia del Error y Garantía del Modelo", fontsize=11, fontweight="bold")
ax2.set_xlabel("Número de Instancias Evaluadas ($N$)")
ax2.set_ylabel("Magnitud del Error Estándar")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)