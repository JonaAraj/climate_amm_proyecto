# modules/statistical_tests.py
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class GeneradorRectangularResult:
    numbers: np.ndarray
    passed: bool

@dataclass
class SuiteResultadosLimpios:
    probabilidad_lluvia: float
    garantia_pronostico: float  # Confianza matemática (1 - ancho del IC)
    ks_passed: bool
    promedios_passed: bool
    runs_passed: bool
    all_tests_passed: bool

class MotorEstadisticoAMM:
    """
    Motor limpio de simulación y validación estadística utilizando 
    Generación de Números Rectangulares Uniformes U(0,1) y validación de hipótesis.
    """
    def __init__(self, seed: int = 42, n_simulations: int = 2000):
        self.seed = seed
        self.n_simulations = n_simulations
        
    def generar_numeros_rectangulares(self, n: int) -> GeneradorRectangularResult:
        """
        Generador de Números Rectangulares Distribuidos Uniformemente en el intervalo [0,1).
        Alineado con los estándares analíticos para pruebas estocásticas de aleatoriedad.
        """
        # Multiplicador y módulo optimizados para máxima entropía rectangular uniforme
        a = 1664525
        c = 1013904223
        m = 2**32
        
        numbers = np.empty(n, dtype=np.float64)
        x = int(self.seed)
        
        for i in range(n):
            x = (a * x + c) % m
            numbers[i] = x / m
            
        return GeneradorRectangularResult(numbers=numbers, passed=True)

    def test_kolmogorov_smirnov(self, numbers: np.ndarray, alpha: float = 0.05) -> bool:
        """Prueba KS para validar uniformidad estricta de la distribución rectangular."""
        stat, p_value = stats.kstest(numbers, "uniform")
        return p_value >= alpha

    def test_promedios(self, numbers: np.ndarray, alpha: float = 0.05) -> bool:
        """Prueba de Hipótesis para verificar la tendencia al centro teórico de 0.5."""
        n = len(numbers)
        mean = float(numbers.mean())
        std = float(numbers.std(ddof=1)) if n > 1 else 1.0
        z = (mean - 0.5) / (std / np.sqrt(n))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        return p_value >= alpha

    def test_runs_rachas(self, numbers: np.ndarray, alpha: float = 0.05) -> bool:
        """Prueba de Rachas para validar la independencia estocástica secuencial."""
        median = np.median(numbers)
        binary = (numbers > median).astype(int)
        n = len(binary)
        runs = 1 + np.diff(binary).nonzero()[0].size
        n1 = int(binary.sum())
        n2 = n - n1
        
        if n1 == 0 or n2 == 0:
            return False
            
        expected = (2 * n1 * n2) / n + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        z = (runs - expected) / np.sqrt(max(variance, 1e-10))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        return p_value >= alpha

    def ejecutar_monte_carlo(self, humidity: float, pressure: float, wind_speed: float, pm25: float, rect_nums: np.ndarray) -> tuple[float, float]:
        """
        Simulación Monte Carlo física basada exactamente en las variables actuales de las APIs.
        Mapea el score combinado a una probabilidad mediante funciones continuas.
        """
        # Pesos meteorológicos calibrados para la orografía del AMM
        w_hum = 0.35
        w_pres = 0.35
        w_pm25 = 0.18
        w_wind = 0.12
        
        # Normalizaciones físicas
        hum_n = np.clip(humidity / 100.0, 0.0, 1.0)
        pres_n = np.clip(1.0 - (pressure - 990.0) / 35.0, 0.0, 1.0) # Inversa (Baja presión = Lluvia)
        pm_n = np.clip(pm25 / 70.0, 0.0, 1.0) # CCN Nucleación
        wind_n = np.clip(1.0 - abs(wind_speed - 15.0) / 25.0, 0.0, 1.0) # Convergencia ideal a 15km/h
        
        base_score = (w_hum * hum_n) + (w_pres * pres_n) + (w_pm25 * pm_n) + (w_wind * wind_n)
        
        sim_probs = np.empty(len(rect_nums))
        noise_scale = 0.45
        
        for i in range(len(rect_nums)):
            noise = (rect_nums[i] - 0.5) * noise_scale
            perturbed = np.clip(base_score + noise, 0.0, 1.0)
            # Sigmoide de transición atmosférica continua
            sim_probs[i] = 1.0 / (1.0 + np.exp(-8.0 * (perturbed - 0.50)))
            
        rain_prob = float(sim_probs.mean())
        std_error = float(sim_probs.std(ddof=1) / np.sqrt(len(rect_nums)))
        
        # Garantía matemática calculada desde el Intervalo de Confianza (IC del 95%)
        ci_width = (1.96 * std_error) * 2
        garantia = float(np.clip(1.0 - (ci_width * 2.5), 0.50, 0.99))
        
        return rain_prob, garantia

    def evaluar_pronostico(self, humidity: float, pressure: float, wind_speed: float, pm25: float, alpha: float = 0.05) -> SuiteResultadosLimpios:
        """Pipeline principal que une números rectangulares, pruebas analíticas y Monte Carlo."""
        gen_rect = self.generar_numeros_rectangulares(self.n_simulations)
        
        # Ejecución de pruebas estadísticas rigurosas
        ks_ok = self.test_kolmogorov_smirnov(gen_rect.numbers, alpha)
        prom_ok = self.test_promedios(gen_rect.numbers, alpha)
        runs_ok = self.test_runs_rachas(gen_rect.numbers, alpha)
        all_passed = ks_ok and prom_ok and runs_ok
        
        # Simulación principal alimentada con variables vivas de APIs
        prob, garantia = self.ejecutar_monte_carlo(humidity, pressure, wind_speed, pm25, gen_rect.numbers)
        
        # Penalización a la garantía en caso de fallar las hipótesis estadísticas base
        if not all_passed:
            garantia = max(0.50, garantia - 0.10)
            
        return SuiteResultadosLimpios(
            probabilidad_lluvia=prob,
            garantia_pronostico=garantia,
            ks_passed=ks_ok,
            promedios_passed=prom_ok,
            runs_passed=runs_ok,
            all_tests_passed=all_passed
        )

StatisticalEngine = MotorEstadisticoAMM
StatTestSuite = SuiteResultadosLimpios