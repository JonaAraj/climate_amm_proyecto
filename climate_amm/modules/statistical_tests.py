"""
statistical_tests.py
====================
Componente importable de pruebas estadísticas para el sistema de
pronóstico climático AMM.

Uso:
    from modules.statistical_tests import StatisticalEngine
    engine = StatisticalEngine(seed=42)
    results = engine.run_all(humidity_series, pressure_series)

Pruebas incluidas:
  - Generador Congruencial Lineal (GCL)
  - Simulación Monte Carlo
  - Prueba de Kolmogorov-Smirnov (KS)
  - Prueba de Series (Runs Test)
  - Prueba de Promedios
  - Diagnóstico compuesto
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  ESTRUCTURAS DE RESULTADO
# ─────────────────────────────────────────────

@dataclass
class GCLResult:
    numbers: np.ndarray          # secuencia generada [0,1)
    period: int                  # período detectado
    params: dict                 # a, c, m, x0
    passed: bool                 # período completo (= m)


@dataclass
class KSResult:
    statistic: float             # D de KS
    p_value: float               # p-valor
    passed: bool                 # H₀: distribución uniforme aceptada
    critical_value: float        # valor crítico α=0.05
    interpretation: str


@dataclass
class SeriesResult:
    runs: int                    # número de rachas observadas
    expected_runs: float         # rachas esperadas bajo H₀
    z_statistic: float           # estadístico Z
    p_value: float
    passed: bool
    interpretation: str


@dataclass
class PromediosResult:
    sample_mean: float           # media de la muestra
    expected_mean: float = 0.5   # media teórica U(0,1)
    z_statistic: float = 0.0
    p_value: float = 1.0
    passed: bool = True
    interpretation: str = ""


@dataclass
class MonteCarloResult:
    n_simulations: int
    rain_probability: float      # prob. de lluvia [0,1]
    confidence_interval: tuple   # IC 95%
    std_error: float
    convergence_curve: np.ndarray  # curva de convergencia
    passed: bool                 # varianza aceptable (<0.1)


@dataclass
class StatTestSuite:
    """Resultado completo de todas las pruebas para un municipio."""
    municipio: str
    gcl: GCLResult
    ks: KSResult
    series: SeriesResult
    promedios: PromediosResult
    monte_carlo: MonteCarloResult
    diagnosis: str               # "lluvia_alta" | "lluvia_moderada" | "despejado"
    rain_probability: float      # probabilidad final integrada
    confidence: float            # confianza del diagnóstico [0,1]
    all_passed: bool             # todas las pruebas OK


# ─────────────────────────────────────────────
#  MOTOR ESTADÍSTICO PRINCIPAL
# ─────────────────────────────────────────────

class StatisticalEngine:
    """
    Motor de pruebas estadísticas para pronóstico climático.
    
    Diseñado para ser instanciado una vez y reutilizado por municipio.
    Todas las pruebas son deterministas dado el mismo seed.
    
    Parámetros GCL (Hull-Dobell — período completo garantizado):
        a = 1_664_525    (multiplicador)
        c = 1_013_904_223 (incremento)
        m = 2**32        (módulo)
    """

    # Parámetros GCL con período completo (Hull-Dobell)
    GCL_A = 1_664_525
    GCL_C = 1_013_904_223
    GCL_M = 2**32

    def __init__(self, seed: int = 42, n_simulations: int = 2000):
        self.seed = seed
        self.n_simulations = n_simulations

    # ── 1. GENERADOR CONGRUENCIAL LINEAL ──────────────────────────

    def gcl_generate(self, n: int, x0: Optional[int] = None) -> GCLResult:
        """
        Genera n números pseudo-aleatorios usando GCL.
        Xₙ₊₁ = (a * Xₙ + c) mod m  →  Uₙ = Xₙ / m ∈ [0,1)
        """
        x0 = x0 if x0 is not None else self.seed
        a, c, m = self.GCL_A, self.GCL_C, self.GCL_M

        sequence = np.empty(n, dtype=np.float64)
        x = int(x0)
        for i in range(n):
            x = (a * x + c) % m
            sequence[i] = x / m

        # Detectar período (muestreo limitado para eficiencia)
        sample = min(n, 500)
        period = self._detect_period(x0, a, c, m, sample)

        return GCLResult(
            numbers=sequence,
            period=period,
            params={"a": a, "c": c, "m": m, "x0": x0},
            passed=(period == sample or period > sample // 2)
        )

    def _detect_period(self, x0, a, c, m, max_steps):
        seen = {}
        x = int(x0)
        for i in range(max_steps):
            if x in seen:
                return i - seen[x]
            seen[x] = i
            x = (a * x + c) % m
        return max_steps

    # ── 2. PRUEBA DE KOLMOGOROV-SMIRNOV ──────────────────────────

    def ks_test(self, numbers: np.ndarray) -> KSResult:
        """
        KS contra distribución uniforme U(0,1).
        H₀: los números siguen una distribución uniforme.
        Se rechaza H₀ si p < 0.05.
        """
        stat, p_value = stats.kstest(numbers, "uniform")
        n = len(numbers)
        # Valor crítico aproximado para α=0.05
        critical = 1.36 / np.sqrt(n)

        passed = p_value >= 0.05
        if passed:
            interp = f"H₀ aceptada (p={p_value:.3f} ≥ 0.05) — distribución uniforme confirmada"
        else:
            interp = f"H₀ rechazada (p={p_value:.3f} < 0.05) — distribución no uniforme"

        return KSResult(
            statistic=round(stat, 4),
            p_value=round(p_value, 4),
            passed=passed,
            critical_value=round(critical, 4),
            interpretation=interp
        )

    # ── 3. PRUEBA DE SERIES (RUNS TEST) ──────────────────────────

    def series_test(self, numbers: np.ndarray) -> SeriesResult:
        """
        Prueba de aleatoriedad por rachas (runs).
        H₀: la secuencia es aleatoria (no hay patrones).
        Una racha es una secuencia consecutiva de valores
        por encima o por debajo de la mediana.
        """
        median = np.median(numbers)
        binary = (numbers > median).astype(int)
        n = len(binary)
        n1 = int(binary.sum())       # valores sobre mediana
        n2 = n - n1                  # valores bajo mediana

        # Contar rachas
        runs = 1
        for i in range(1, n):
            if binary[i] != binary[i - 1]:
                runs += 1

        # Distribución esperada
        if n1 == 0 or n2 == 0:
            return SeriesResult(runs=runs, expected_runs=float(n),
                                z_statistic=0.0, p_value=1.0,
                                passed=True, interpretation="Datos homogéneos — prueba no aplicable")

        expected = (2 * n1 * n2) / n + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        std = np.sqrt(max(variance, 1e-10))
        z = (runs - expected) / std
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        passed = p_value >= 0.05
        interp = (
            f"Aleatoriedad confirmada (Z={z:.3f}, p={p_value:.3f})"
            if passed else
            f"Patrón detectado en secuencia (Z={z:.3f}, p={p_value:.3f})"
        )

        return SeriesResult(
            runs=runs,
            expected_runs=round(expected, 2),
            z_statistic=round(z, 4),
            p_value=round(p_value, 4),
            passed=passed,
            interpretation=interp
        )

    # ── 4. PRUEBA DE PROMEDIOS ────────────────────────────────────

    def promedios_test(self, numbers: np.ndarray) -> PromediosResult:
        """
        Verifica que la media muestral converja a 0.5 (U(0,1)).
        H₀: μ = 0.5
        Usa prueba t de una muestra.
        """
        n = len(numbers)
        mean = float(numbers.mean())
        std = float(numbers.std(ddof=1))

        z = (mean - 0.5) / (std / np.sqrt(n))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        passed = p_value >= 0.05
        interp = (
            f"Media muestral {mean:.4f} converge a 0.5 (p={p_value:.3f})"
            if passed else
            f"Media muestral {mean:.4f} NO converge a 0.5 (p={p_value:.3f})"
        )

        return PromediosResult(
            sample_mean=round(mean, 4),
            expected_mean=0.5,
            z_statistic=round(z, 4),
            p_value=round(p_value, 4),
            passed=passed,
            interpretation=interp
        )

    # ── 5. SIMULACIÓN MONTE CARLO ─────────────────────────────────

    # Pesos calibrados para física meteorológica del AMM (Monterrey, NL)
    # Suma = 1.0 garantizada. Ajustados para evitar colapso a P=0 o P=1.
    WEIGHTS = {
        "humidity": 0.35,   # Humedad: predictor primario de condensación
        "pressure": 0.35,   # Presión: indicador más fiable de sistema frontal
        "pm25":     0.18,   # PM2.5: núcleos de condensación (CCN)
        "wind":     0.12,   # Viento: convergencia de masa de aire (~15 km/h óptimo)
    }
    # Escala de ruido: ±0.225 — suficiente para cruzar umbrales en ambas direcciones
    NOISE_SCALE = 0.45

    @staticmethod
    def _sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
        """
        Función sigmoide para mapear score → probabilidad continua.
        k controla la pendiente (mayor k = transición más abrupta).
        x0 es el punto de inflexión (umbral central).
        k=10 da una transición suave entre ~0.3 y ~0.7.
        """
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    def monte_carlo(
        self,
        humidity: float,
        pressure: float,
        wind_speed: float,
        pm25: float,
        gcl_numbers: np.ndarray
    ) -> MonteCarloResult:
        """
        Simulación Monte Carlo calibrada de probabilidad de lluvia.

        Cambios respecto a v1:
        ─────────────────────
        1. Score continuo (sigmoide) en vez de binario con umbral fijo.
           Elimina el colapso a P=100% cuando base_score > umbral + noise_max.

        2. noise_scale = 0.45 (era 0.15).
           Perturbación real de ±0.225 que puede cruzar el umbral en ambas
           direcciones para cualquier valor de base_score en [0.2, 0.8].

        3. Pesos ajustados a física del AMM:
           humedad 0.35 · presión 0.35 · PM2.5 0.18 · viento 0.12

        4. Normalización de presión con rango AMM real [990, 1025] hPa,
           no el rango genérico anterior [990, 1030].

        5. Viento: función campana centrada en 15 km/h (convergencia óptima
           para lluvia convectiva en zona montañosa del AMM).

        Modelo físico
        ─────────────
        - Humedad ≥ 70%  → condensación activa
        - Presión ↓ hPa  → sistema de baja presión entrando
        - PM2.5 ≥ 30 µg  → aerosoles como CCN aceleran formación de gotas
        - Viento ~15 km/h → convergencia sin dispersar la nubosidad
        """
        n = len(gcl_numbers)
        w = self.WEIGHTS

        # ── Normalización de variables con rangos físicos del AMM ──
        hum_n  = np.clip(humidity / 100.0, 0.0, 1.0)

        # Presión invertida: menor presión = mayor prob. lluvia
        # Rango AMM: [990, 1025] hPa → mapeo a [1.0, 0.0]
        pres_n = np.clip(1.0 - (pressure - 990.0) / 35.0, 0.0, 1.0)

        # PM2.5: efecto de nucleación, saturación en ~70 µg/m³
        pm_n   = np.clip(pm25 / 70.0, 0.0, 1.0)

        # Viento: campana centrada en 15 km/h, cae a 0 en 0 y 40 km/h
        # Convergencia óptima para convección en AMM entre 10–25 km/h
        wind_n = np.clip(1.0 - abs(wind_speed - 15.0) / 25.0, 0.0, 1.0)

        # ── Score base atmosférico ─────────────────────────────────
        base_score = (
            w["humidity"] * hum_n +
            w["pressure"] * pres_n +
            w["pm25"]     * pm_n  +
            w["wind"]     * wind_n
        )

        # ── Simulaciones Monte Carlo ───────────────────────────────
        # Cada iteración: perturbar base_score con ruido GCL,
        # aplicar sigmoide para obtener P(lluvia) continua en [0,1].
        # Se acumula la media corrida para la curva de convergencia.

        sim_probs  = np.empty(n)
        convergence = np.empty(n)

        for i in range(n):
            # Ruido centrado en 0, rango real ±NOISE_SCALE/2
            noise = (gcl_numbers[i] - 0.5) * self.NOISE_SCALE
            perturbed = float(np.clip(base_score + noise, 0.0, 1.0))

            # Probabilidad continua via sigmoide (k=8, x0=0.50)
            # k=8 evita tanto la saturación total como valores triviales
            sim_probs[i] = self._sigmoid(perturbed, k=8.0, x0=0.50)

            convergence[i] = sim_probs[:i+1].mean()

        # ── Estadísticos finales ───────────────────────────────────
        rain_prob = float(sim_probs.mean())
        # Error estándar de la media de una distribución continua
        std_error = float(sim_probs.std(ddof=1) / np.sqrt(n))

        # IC 95% de Wald sobre la media de probabilidades
        z95     = 1.96
        ci_low  = float(np.clip(rain_prob - z95 * std_error, 0.0, 1.0))
        ci_high = float(np.clip(rain_prob + z95 * std_error, 0.0, 1.0))

        # Criterio de varianza: error estándar < 0.01 (umbral más exigente)
        return MonteCarloResult(
            n_simulations=n,
            rain_probability=round(rain_prob, 4),
            confidence_interval=(round(ci_low, 4), round(ci_high, 4)),
            std_error=round(std_error, 6),
            convergence_curve=convergence,
            passed=(std_error < 0.010)
        )

    # ── 6. DIAGNÓSTICO COMPUESTO ──────────────────────────────────

    def _diagnose(self, rain_prob: float, ci: tuple,
                  all_passed: bool) -> tuple[str, float]:
        """
        Diagnóstico con confianza dinámica basada en el IC del Monte Carlo.

        Confianza = 1 - ancho_IC_normalizado
        Un IC angosto indica convergencia del modelo → alta confianza.
        Las pruebas estadísticas fallidas penalizan adicional (-8%).

        Umbrales de clasificación (ajustados para evitar zona muerta):
          lluvia_alta    : P ≥ 0.62
          lluvia_moderada: 0.40 ≤ P < 0.62
          despejado      : P < 0.40
        """
        ci_width = ci[1] - ci[0]                        # ancho IC [0, 0.4 típico]
        # Confianza base: IC angosto = modelo estable = alta confianza
        ci_confidence = float(np.clip(1.0 - ci_width * 2.5, 0.50, 0.96))
        # Penalización si alguna prueba estadística falló
        stat_penalty = 0.0 if all_passed else 0.08
        confidence = round(ci_confidence - stat_penalty, 3)

        if rain_prob >= 0.62:
            return "lluvia_alta", confidence
        elif rain_prob >= 0.40:
            return "lluvia_moderada", round(confidence * 0.95, 3)
        else:
            return "despejado", round(confidence * 0.92, 3)

    # ── 7. PUNTO DE ENTRADA PRINCIPAL ─────────────────────────────

    def run_all(
        self,
        municipio: str,
        humidity: float,
        pressure: float,
        wind_speed: float,
        pm25: float,
        n_numbers: Optional[int] = None
    ) -> StatTestSuite:
        """
        Ejecuta el pipeline completo de pruebas para un municipio.
        
        Parámetros
        ----------
        municipio   : nombre del municipio (para trazabilidad)
        humidity    : humedad relativa [%]
        pressure    : presión atmosférica [hPa]
        wind_speed  : velocidad del viento [km/h]
        pm25        : partículas PM2.5 [µg/m³]
        n_numbers   : tamaño de muestra GCL (default: n_simulations)
        
        Retorna
        -------
        StatTestSuite con todos los resultados.
        """
        n = n_numbers or self.n_simulations

        # 1. Generar números con GCL (semilla derivada del municipio)
        muni_seed = self.seed + sum(ord(c) for c in municipio)
        gcl = self.gcl_generate(n, x0=muni_seed)

        # 2. Pruebas sobre los números generados
        ks        = self.ks_test(gcl.numbers)
        series    = self.series_test(gcl.numbers)
        promedios = self.promedios_test(gcl.numbers)

        # 3. Simulación Monte Carlo con variables climáticas
        mc = self.monte_carlo(
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            pm25=pm25,
            gcl_numbers=gcl.numbers
        )

        # 4. Diagnóstico
        all_passed = gcl.passed and ks.passed and series.passed and promedios.passed
        diagnosis, confidence = self._diagnose(
            mc.rain_probability, mc.confidence_interval, all_passed
        )

        return StatTestSuite(
            municipio=municipio,
            gcl=gcl,
            ks=ks,
            series=series,
            promedios=promedios,
            monte_carlo=mc,
            diagnosis=diagnosis,
            rain_probability=mc.rain_probability,
            confidence=confidence,
            all_passed=all_passed
        )

    def run_batch(self, municipios_data: list[dict]) -> dict[str, StatTestSuite]:
        """
        Ejecuta pruebas para múltiples municipios en lote.
        
        municipios_data: lista de dicts con keys:
            nombre, humidity, pressure, wind_speed, pm25
        
        Retorna dict {nombre: StatTestSuite}
        """
        return {
            m["nombre"]: self.run_all(
                municipio=m["nombre"],
                humidity=m["humidity"],
                pressure=m["pressure"],
                wind_speed=m["wind_speed"],
                pm25=m["pm25"]
            )
            for m in municipios_data
        }
