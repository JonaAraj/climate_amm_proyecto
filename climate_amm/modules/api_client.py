"""
api_client.py
=============
Cliente optimizado para OpenWeather One Call 3.0 y Air Pollution API.
Soporta consultas individuales y batch para todos los municipios del AMM.

Uso:
    from modules.api_client import WeatherClient
    client = WeatherClient(api_key="TU_KEY")
    data = client.get_municipio("General Escobedo")
    all_data = client.get_all_municipios()
"""

import httpx
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
#  CATÁLOGO DE MUNICIPIOS AMM
# ─────────────────────────────────────────────

MUNICIPIOS_AMM = {
    "General Escobedo": {"lat": 25.7939, "lon": -100.3337, "cat": "residencial"},
    "Monterrey":        {"lat": 25.6866, "lon": -100.3161, "cat": "capital"},
    "San Nicolás":      {"lat": 25.7444, "lon": -100.2922, "cat": "industrial"},
    "Guadalupe":        {"lat": 25.6755, "lon": -100.2596, "cat": "residencial"},
    "Apodaca":          {"lat": 25.7797, "lon": -100.1878, "cat": "industrial"},
    "San Pedro G.G.":   {"lat": 25.6591, "lon": -100.4050, "cat": "residencial"},
    "Santa Catarina":   {"lat": 25.6732, "lon": -100.4620, "cat": "industrial"},
    "García":           {"lat": 25.7989, "lon": -100.5874, "cat": "residencial"},
    "Juárez":           {"lat": 25.6494, "lon": -100.1124, "cat": "industrial"},
    "Cadereyta":        {"lat": 25.5942, "lon": -99.9990,  "cat": "industrial"},
    "Santiago":         {"lat": 25.4317, "lon": -100.1450, "cat": "residencial"},
    "Salinas Victoria": {"lat": 25.9569, "lon": -100.2894, "cat": "residencial"},
    "El Carmen":        {"lat": 25.8839, "lon": -100.4836, "cat": "residencial"},
}


# ─────────────────────────────────────────────
#  ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────

@dataclass
class WeatherData:
    municipio: str
    lat: float
    lon: float
    categoria: str
    # Variables climáticas principales
    temperature: float       # °C
    feels_like: float        # °C
    humidity: float          # %
    pressure: float          # hPa
    wind_speed: float        # km/h (convertido de m/s)
    wind_deg: float          # grados
    cloud_cover: float       # %
    visibility: float        # km
    uvi: float               # índice UV
    dew_point: float         # punto de rocío °C
    # Pronóstico
    rain_1h: float           # mm última hora
    rain_prob_24h: float     # probabilidad próximas 24h [0,1]
    forecast_3d: list        # [{day, temp_max, temp_min, rain_prob, description}]
    # Contaminantes (Air Pollution API)
    aqi: int                 # 1=Bueno … 5=Muy malo
    pm25: float              # µg/m³
    pm10: float              # µg/m³
    no2: float               # µg/m³
    co: float                # µg/m³
    o3: float                # µg/m³
    so2: float               # µg/m³
    # Metadatos
    timestamp: float = field(default_factory=time.time)
    source: str = "openweather"


@dataclass
class CacheEntry:
    data: WeatherData
    expires_at: float        # timestamp de expiración


# ─────────────────────────────────────────────
#  CLIENTE PRINCIPAL
# ─────────────────────────────────────────────

class WeatherClient:
    """
    Cliente optimizado con:
    - Caché TTL configurable (evita peticiones duplicadas)
    - Consultas async en paralelo para batch de municipios
    - Fallback a datos simulados si no hay API key
    - Conversión automática de unidades
    """

    BASE_WEATHER = "https://api.openweathermap.org/data/3.0/onecall"
    BASE_AIR     = "https://api.openweathermap.org/data/2.5/air_pollution"
    BASE_METEO   = "https://api.open-meteo.com/v1/forecast"  # fallback gratuito

    AQI_LABELS = {1: "Bueno", 2: "Aceptable", 3: "Moderado", 4: "Malo", 5: "Muy malo"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 600,        # segundos (10 min = frecuencia recomendada OW)
        timeout: int = 10
    ):
        self.api_key   = api_key
        self.cache_ttl = cache_ttl
        self.timeout   = timeout
        self._cache: dict[str, CacheEntry] = {}
        self._use_real_api = bool(api_key)

    # ── CACHÉ ─────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[WeatherData]:
        entry = self._cache.get(key)
        if entry and time.time() < entry.expires_at:
            return entry.data
        return None

    def _cache_set(self, key: str, data: WeatherData):
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + self.cache_ttl
        )

    def cache_status(self) -> dict:
        now = time.time()
        return {
            k: f"expira en {int(v.expires_at - now)}s"
            for k, v in self._cache.items()
            if v.expires_at > now
        }

    # ── CONSULTA INDIVIDUAL (ASYNC) ───────────────────────────────

    async def _fetch_weather_async(
        self, client: httpx.AsyncClient, municipio: str
    ) -> WeatherData:
        """Fetch async de clima + contaminantes para un municipio."""
        cached = self._cache_get(municipio)
        if cached:
            return cached

        meta = MUNICIPIOS_AMM[municipio]
        lat, lon = meta["lat"], meta["lon"]

        if self._use_real_api:
            data = await self._fetch_real(client, municipio, lat, lon, meta["cat"])
        else:
            data = self._simulate(municipio, lat, lon, meta["cat"])

        self._cache_set(municipio, data)
        return data

    async def _fetch_real(
        self, client, municipio, lat, lon, cat
    ) -> WeatherData:
        """Peticiones reales a OpenWeather (paralelas con asyncio.gather)."""
        params_weather = {
            "lat": lat, "lon": lon,
            "exclude": "minutely,alerts",
            "units": "metric",
            "appid": self.api_key
        }
        params_air = {
            "lat": lat, "lon": lon,
            "appid": self.api_key
        }

        # Ambas peticiones en paralelo
        weather_resp, air_resp = await asyncio.gather(
            client.get(self.BASE_WEATHER, params=params_weather),
            client.get(self.BASE_AIR,     params=params_air)
        )

        w = weather_resp.json()
        a = air_resp.json()

        current    = w.get("current", {})
        hourly     = w.get("hourly", [{}])
        daily      = w.get("daily", [])
        components = a.get("list", [{}])[0].get("components", {})
        aqi_val    = a.get("list", [{}])[0].get("main", {}).get("aqi", 1)

        # Pronóstico 3 días
        forecast_3d = []
        for day in daily[:3]:
            forecast_3d.append({
                "day": day.get("dt", 0),
                "temp_max": day.get("temp", {}).get("max", 0),
                "temp_min": day.get("temp", {}).get("min", 0),
                "rain_prob": day.get("pop", 0),
                "description": day.get("weather", [{}])[0].get("description", ""),
            })

        rain_prob_24h = float(np.mean([h.get("pop", 0) for h in hourly[:24]]))

        return WeatherData(
            municipio=municipio, lat=lat, lon=lon, categoria=cat,
            temperature=current.get("temp", 0),
            feels_like=current.get("feels_like", 0),
            humidity=current.get("humidity", 0),
            pressure=current.get("pressure", 1013),
            wind_speed=round(current.get("wind_speed", 0) * 3.6, 1),  # m/s → km/h
            wind_deg=current.get("wind_deg", 0),
            cloud_cover=current.get("clouds", 0),
            visibility=round(current.get("visibility", 10000) / 1000, 1),
            uvi=current.get("uvi", 0),
            dew_point=current.get("dew_point", 0),
            rain_1h=current.get("rain", {}).get("1h", 0.0),
            rain_prob_24h=rain_prob_24h,
            forecast_3d=forecast_3d,
            aqi=aqi_val,
            pm25=components.get("pm2_5", 0),
            pm10=components.get("pm10", 0),
            no2=components.get("no2", 0),
            co=components.get("co", 0),
            o3=components.get("o3", 0),
            so2=components.get("so2", 0),
            source="openweather"
        )

    # ── DATOS SIMULADOS (sin API key) ─────────────────────────────

    def _simulate(self, municipio: str, lat: float, lon: float, cat: str) -> WeatherData:
        """
        Genera datos realistas simulados para desarrollo/demo.
        Reproducible: misma semilla = mismos datos para el mismo municipio.
        """
        rng = np.random.default_rng(seed=abs(hash(municipio)) % (2**31))
        industrial = cat == "industrial"

        hum       = float(rng.integers(55, 82))
        pres      = float(rng.integers(1003, 1018))
        wind      = round(float(rng.uniform(8, 28)), 1)
        temp      = round(float(rng.uniform(22, 36)), 1)
        pm25      = round(float(rng.uniform(28, 68) if industrial else rng.uniform(10, 35)), 1)
        pm10      = round(float(rng.uniform(50, 95) if industrial else rng.uniform(20, 50)), 1)
        rain_prob = round(float(rng.uniform(0.25, 0.85)), 2)

        forecast_3d = [
            {
                "day": int(time.time()) + i * 86400,
                "temp_max": round(temp + rng.uniform(-2, 4), 1),
                "temp_min": round(temp - rng.uniform(3, 8), 1),
                "rain_prob": round(float(rng.uniform(0.1, 0.9)), 2),
                "description": rng.choice(["lluvia ligera", "nublado", "despejado", "tormenta"]),
            }
            for i in range(3)
        ]

        return WeatherData(
            municipio=municipio, lat=lat, lon=lon, categoria=cat,
            temperature=temp,
            feels_like=round(temp - rng.uniform(1, 4), 1),
            humidity=hum,
            pressure=pres,
            wind_speed=wind,
            wind_deg=float(rng.integers(0, 360)),
            cloud_cover=float(rng.integers(20, 90)),
            visibility=round(float(rng.uniform(4, 10)), 1),
            uvi=round(float(rng.uniform(3, 10)), 1),
            dew_point=round(float(rng.uniform(12, 22)), 1),
            rain_1h=round(float(rng.uniform(0, 2.5) if rain_prob > 0.5 else 0), 1),
            rain_prob_24h=rain_prob,
            forecast_3d=forecast_3d,
            aqi=int(rng.integers(1, 4 if not industrial else 5)),
            pm25=pm25,
            pm10=pm10,
            no2=round(float(rng.uniform(18, 55) if industrial else rng.uniform(5, 20)), 1),
            co=round(float(rng.uniform(400, 900) if industrial else rng.uniform(200, 500)), 1),
            o3=round(float(rng.uniform(40, 90)), 1),
            so2=round(float(rng.uniform(5, 30) if industrial else rng.uniform(1, 8)), 1),
            source="simulado"
        )

    # ── API PÚBLICA ────────────────────────────────────────────────

    def get_municipio(self, municipio: str) -> WeatherData:
        """Obtiene datos de un municipio (síncrono, con caché)."""
        if municipio not in MUNICIPIOS_AMM:
            raise ValueError(f"Municipio '{municipio}' no encontrado. "
                             f"Disponibles: {list(MUNICIPIOS_AMM.keys())}")
        return asyncio.run(self._get_single(municipio))

    async def _get_single(self, municipio: str) -> WeatherData:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._fetch_weather_async(client, municipio)

    def get_all_municipios(self) -> dict[str, WeatherData]:
        """
        Obtiene datos de todos los municipios del AMM en paralelo.
        Optimizado: una sola sesión HTTP, todas las peticiones simultáneas.
        """
        return asyncio.run(self._get_all_async())

    async def _get_all_async(self) -> dict[str, WeatherData]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._fetch_weather_async(client, muni)
                for muni in MUNICIPIOS_AMM
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for muni, result in zip(MUNICIPIOS_AMM.keys(), results):
            if isinstance(result, Exception):
                # Fallback a simulación si falla la petición
                output[muni] = self._simulate(
                    muni,
                    MUNICIPIOS_AMM[muni]["lat"],
                    MUNICIPIOS_AMM[muni]["lon"],
                    MUNICIPIOS_AMM[muni]["cat"]
                )
            else:
                output[muni] = result

        return output

    def get_municipios_list(self) -> list[str]:
        return list(MUNICIPIOS_AMM.keys())
