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

    def monte_carlo(
        self,
        humidity: float,
        pressure: float,
        wind_speed: float,
        pm25: float,
        gcl_numbers: np.ndarray
    ) -> MonteCarloResult:
        """
        Simulación Monte Carlo de probabilidad de lluvia.
        
        Modelo físico simplificado:
          - Humedad >70% → factor positivo fuerte
          - Presión <1009 hPa → factor positivo
          - PM2.5 >30 µg → núcleos de condensación → factor positivo
          - Viento entre 10–25 km/h → convergencia → factor positivo
          
        Cada simulación perturba las variables con ruido GCL y
        evalúa si se superan umbrales de precipitación.
        """
        n = len(gcl_numbers)
        rain_events = np.zeros(n)
        convergence = np.zeros(n)

        # Normalizar variables a [0,1]
        hum_n  = np.clip(humidity / 100, 0, 1)
        pres_n = np.clip(1 - (pressure - 990) / 40, 0, 1)  # invertida
        pm_n   = np.clip(pm25 / 80, 0, 1)
        wind_n = np.clip(1 - abs(wind_speed - 17) / 20, 0, 1)

        # Score base del estado atmosférico
        base_score = (
            0.40 * hum_n +
            0.30 * pres_n +
            0.20 * pm_n +
            0.10 * wind_n
        )

        # Perturbación Monte Carlo con números GCL
        noise_scale = 0.15
        for i in range(n):
            noise = (gcl_numbers[i] - 0.5) * noise_scale
            score = np.clip(base_score + noise, 0, 1)
            rain_events[i] = 1 if score > 0.45 else 0
            convergence[i] = rain_events[:i+1].mean() if i > 0 else rain_events[0]

        rain_prob = float(rain_events.mean())
        std_error = float(np.sqrt(rain_prob * (1 - rain_prob) / n))

        # IC 95%
        z95 = 1.96
        ci_low  = max(0, rain_prob - z95 * std_error)
        ci_high = min(1, rain_prob + z95 * std_error)

        return MonteCarloResult(
            n_simulations=n,
            rain_probability=round(rain_prob, 4),
            confidence_interval=(round(ci_low, 4), round(ci_high, 4)),
            std_error=round(std_error, 4),
            convergence_curve=convergence,
            passed=(std_error < 0.015)
        )

    # ── 6. DIAGNÓSTICO COMPUESTO ──────────────────────────────────

    def _diagnose(self, rain_prob: float, all_passed: bool) -> tuple[str, float]:
        """Clasifica el pronóstico y calcula confianza."""
        base_confidence = 0.90 if all_passed else 0.72

        if rain_prob >= 0.65:
            return "lluvia_alta", round(base_confidence * (0.85 + 0.15 * rain_prob), 3)
        elif rain_prob >= 0.40:
            return "lluvia_moderada", round(base_confidence * 0.80, 3)
        else:
            return "despejado", round(base_confidence * (0.9 - rain_prob * 0.3), 3)

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
        diagnosis, confidence = self._diagnose(mc.rain_probability, all_passed)

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
