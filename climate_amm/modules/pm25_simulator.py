"""
pm25_simulator.py
=================
Simulador de dinámica de sistemas para acumulación/dispersión de PM2.5
en la cuenca atmosférica del AMM (Área Metropolitana de Monterrey).

Basado en la ecuación diferencial de Forrester:
    dM/dt = E(t) – k · V(t) · M(t)

Donde:
    M(t)  = concentración de PM2.5 [µg/m³]
    E(t)  = tasa de emisión antropogénica [µg/m³·h]
    k     = coeficiente de fricción atmosférica / topografía [adimensional]
    V(t)  = velocidad del viento [m/s]

Resuelto numéricamente con Runge-Kutta 4-5 (scipy.integrate.solve_ivp).
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
#  ESTRUCTURAS DE RESULTADO
# ─────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Resultado de una simulación individual."""
    t: np.ndarray              # vector de tiempo [h]
    M: np.ndarray              # concentración PM2.5 [µg/m³]
    wind_speed: float          # viento usado [km/h]
    E_base: float              # emisión base usada [µg/m³·h]
    k: float                   # coef. fricción usado
    tipping_time: Optional[float]  # hora donde cruza M* (None si no cruza)
    final_M: float             # M al final del horizonte
    max_M: float               # M máxima en la ventana
    safe: bool                 # True si nunca supera M*


@dataclass
class SensitivityResult:
    """Resultado del análisis de sensibilidad paramétrica."""
    variations: np.ndarray     # factores de variación del viento
    labels: list[str]          # etiquetas legibles
    simulations: list[SimulationResult]
    tipping_times: list[Optional[float]]


# ─────────────────────────────────────────────
#  SIMULADOR PRINCIPAL
# ─────────────────────────────────────────────

class PM25Simulator:
    """
    Motor de simulación de dinámica de PM2.5 para el AMM.

    Parámetros por defecto calibrados para cuenca de Monterrey, NL:
        M_star = 35.0 µg/m³  (límite NOM-025-SSA1-2014, promedio 24h)
        k      = 0.12        (fricción + barrera Sierra Madre Oriental)
    """

    M_STAR = 35.0   # µg/m³ — umbral normativo

    def __init__(
        self,
        k: float = 0.12,
        M_star: float = 35.0,
        use_diurnal_cycle: bool = True,
        current_hour: Optional[int] = None,
    ):
        self.k = k
        self.M_star = M_star
        self.use_diurnal_cycle = use_diurnal_cycle
        self.current_hour = current_hour or 12  # default mediodía

    # ── 1. TASA DE EMISIÓN E(t) ───────────────────────────────────

    def emission_rate(self, t: float, E_base: float) -> float:
        """
        Tasa de emisión con ciclo diurno opcional.
        
        Picos:
          • Matutino ~ 08:00 h (tráfico industrial/vehicular)
          • Vespertino ~ 19:00 h (regreso + actividad industrial)
        
        Parámetros
        ----------
        t : float
            horas transcurridas desde el inicio de la simulación
        E_base : float
            emisión base [µg/m³·h]
        """
        if not self.use_diurnal_cycle:
            return E_base

        hour = (self.current_hour + t) % 24

        # Pico matutino (gaussiano centrado en 8h, σ≈1.5h)
        morning = 1.0 + 0.45 * np.exp(-((hour - 8.0) ** 2) / 4.5)
        # Pico vespertino (gaussiano centrado en 19h, σ≈2h)
        evening = 1.0 + 0.35 * np.exp(-((hour - 19.0) ** 2) / 8.0)

        # Factor combinado: base + excesos de los picos
        factor = morning + evening - 1.0
        return E_base * max(factor, 0.3)  # mínimo 30% de emisión nocturna

    # ── 2. RESOLUCIÓN DE LA EDO ───────────────────────────────────

    def simulate(
        self,
        M0: float,
        wind_speed_kmh: float,
        E_base: float,
        horizon: float = 4.0,
        dt_eval: float = 0.05,
    ) -> SimulationResult:
        """
        Resuelve la EDO dM/dt = E(t) – k·V(t)·M(t) con RK45.

        Parámetros
        ----------
        M0 : float
            Concentración inicial de PM2.5 [µg/m³]
        wind_speed_kmh : float
            Velocidad del viento [km/h]
        E_base : float
            Tasa de emisión base [µg/m³·h]
        horizon : float
            Horizonte de simulación [h] (default 4 h)
        dt_eval : float
            Paso de evaluación para la curva de salida [h]

        Retorna
        -------
        SimulationResult con trayectoria completa y metadatos.
        """
        # Convertir viento a m/s (la EDO usa SI)
        V_ms = wind_speed_kmh / 3.6

        # Función E(t) capturada
        def E_of_t(t: float) -> float:
            return self.emission_rate(t, E_base)

        # EDO: dM/dt = E(t) - k * V * M(t)
        def dMdt(t: float, M: np.ndarray) -> np.ndarray:
            return np.array([E_of_t(t) - self.k * V_ms * float(M[0])])

        # Evento: detección de Tipping Point (M cruza M* ascendiendo)
        def tipping_event(t: float, M: np.ndarray) -> float:
            return float(M[0]) - self.M_star

        tipping_event.terminal = False
        tipping_event.direction = 1   # solo al subir

        # Resolver con RK45 (Runge-Kutta 4-5 adaptativo)
        t_eval = np.arange(0.0, horizon + dt_eval, dt_eval)
        sol = solve_ivp(
            fun=dMdt,
            t_span=(0.0, horizon),
            y0=[M0],
            method="RK45",
            t_eval=t_eval,
            events=tipping_event,
            dense_output=True,
        )

        M_traj = sol.y[0]
        t_traj = sol.t

        # Extraer tiempo de tipping
        tipping_time: Optional[float] = None
        if sol.t_events is not None and len(sol.t_events) > 0 and sol.t_events[0] is not None:
            if len(sol.t_events[0]) > 0:
                tipping_time = float(sol.t_events[0][0])

        final_M = float(M_traj[-1])
        max_M = float(np.max(M_traj))
        safe = max_M < self.M_star

        return SimulationResult(
            t=t_traj,
            M=M_traj,
            wind_speed=wind_speed_kmh,
            E_base=E_base,
            k=self.k,
            tipping_time=tipping_time,
            final_M=final_M,
            max_M=max_M,
            safe=safe,
        )

    # ── 3. ANÁLISIS DE SENSIBILIDAD (viento) ──────────────────────

    def sensitivity_analysis(
        self,
        M0: float,
        wind_speed_kmh: float,
        E_base: float,
        variations: Optional[np.ndarray] = None,
        horizon: float = 4.0,
    ) -> SensitivityResult:
        """
        Ejecuta simulaciones paramétricas variando la velocidad del viento.

        Variaciones por defecto: ±10%, ±25%, ±50%  (según objetivo del plan)
        """
        if variations is None:
            variations = np.array([-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50])

        sims = []
        tips = []
        labels = []

        for v in variations:
            w = wind_speed_kmh * (1.0 + v)
            w = max(w, 0.5)  # mínimo físico
            sim = self.simulate(M0, w, E_base, horizon=horizon)
            sims.append(sim)
            tips.append(sim.tipping_time)
            labels.append(f"{v:+.0%}")

        return SensitivityResult(
            variations=variations,
            labels=labels,
            simulations=sims,
            tipping_times=tips,
        )

    # ── 4. ESCENARIO CON LLUVIA (wet deposition) ──────────────────

    def simulate_with_rain(
        self,
        M0: float,
        wind_speed_kmh: float,
        E_base: float,
        rain_start_h: float,
        rain_duration_h: float,
        rain_intensity: float = 0.8,
        horizon: float = 4.0,
    ) -> SimulationResult:
        """
        Simulación con deposición húmeda (wet deposition) durante lluvia.
        
        Agrega término –λ·I(t)·M(t) donde I(t) es intensidad de lluvia [0,1].
        λ = rain_intensity * k  (coef. de lavado atmosférico)
        """
        V_ms = wind_speed_kmh / 3.6
        washout = rain_intensity * self.k  # coef. deposición húmeda

        def E_of_t(t: float) -> float:
            return self.emission_rate(t, E_base)

        def raining(t: float) -> float:
            if rain_start_h <= t <= rain_start_h + rain_duration_h:
                return 1.0
            return 0.0

        def dMdt(t: float, M: np.ndarray) -> np.ndarray:
            m = float(M[0])
            return np.array([E_of_t(t) - self.k * V_ms * m - washout * raining(t) * m])

        def tipping_event(t: float, M: np.ndarray) -> float:
            return float(M[0]) - self.M_star

        tipping_event.terminal = False
        tipping_event.direction = 1

        t_eval = np.arange(0.0, horizon + 0.05, 0.05)
        sol = solve_ivp(
            fun=dMdt,
            t_span=(0.0, horizon),
            y0=[M0],
            method="RK45",
            t_eval=t_eval,
            events=tipping_event,
            dense_output=True,
        )

        M_traj = sol.y[0]
        t_traj = sol.t
        tipping_time: Optional[float] = None
        if sol.t_events is not None and len(sol.t_events) > 0 and sol.t_events[0] is not None:
            if len(sol.t_events[0]) > 0:
                tipping_time = float(sol.t_events[0][0])

        return SimulationResult(
            t=t_traj,
            M=M_traj,
            wind_speed=wind_speed_kmh,
            E_base=E_base,
            k=self.k,
            tipping_time=tipping_time,
            final_M=float(M_traj[-1]),
            max_M=float(np.max(M_traj)),
            safe=float(np.max(M_traj)) < self.M_star,
        )

    # ── 5. UTILIDADES ─────────────────────────────────────────────

    @staticmethod
    def format_tipping_time(t: Optional[float]) -> str:
        """Formatea el tiempo de tipping para display."""
        if t is None:
            return "No se proyecta cruce"
        if t < 1.0:
            return f"{t * 60:.0f} min"
        return f"{t:.2f} h"

    @staticmethod
    def equilibrium_emission(M: float, wind_speed_kmh: float, k: float) -> float:
        """
        Emisión necesaria para mantener concentración M en equilibrio
        (dM/dt = 0 → E = k·V·M).
        Útil como referencia de calibración.
        """
        V_ms = wind_speed_kmh / 3.6
        return k * V_ms * M
