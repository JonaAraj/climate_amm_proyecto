"""
api_client.py
=============
Cliente unificado: Tomorrow.io (clima) + SIMA NL (contaminación real).

Mantiene compatibilidad total con el dashboard existente.
WeatherData sigue siendo la estructura central; se agregan campos
opcionales para funcionalidades nuevas (post-lluvia, código de clima).

Uso:
    from modules.api_client import WeatherClient, MUNICIPIOS_AMM
    client = WeatherClient(tomorrow_key="KEY")
    data = client.get_municipio("General Escobedo")
    all_data = client.get_all_municipios()
"""

import httpx
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .tomorrow_client import TomorrowClient, TomorrowForecast
from .sima_client import SIMAClient


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
    temperature: float
    feels_like: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_deg: float
    cloud_cover: float
    visibility: float
    uvi: float
    dew_point: float
    # Nuevos — Tomorrow.io
    weather_code: int = 0
    rain_intensity: float = 0.0      # mm/h actual
    wind_gust: float = 0.0           # km/h
    precipitation_prob: float = 0.0  # %
    # Pronóstico
    rain_1h: float = 0.0
    rain_prob_24h: float = 0.0
    forecast_3d: list = field(default_factory=list)
    hourly_forecast: list = field(default_factory=list)  # objetos TomorrowForecast
    # Contaminantes
    aqi: int = 1
    pm25: float = 0.0
    pm10: float = 0.0
    no2: float = 0.0
    co: float = 0.0
    o3: float = 0.0
    so2: float = 0.0
    # Metadatos
    timestamp: float = field(default_factory=time.time)
    source: str = "tomorrow+sima"
    sima_estacion: str = "NORTE2"
    sima_hora: str = ""


@dataclass
class CacheEntry:
    data: WeatherData
    expires_at: float


# ─────────────────────────────────────────────
#  CLIENTE PRINCIPAL
# ─────────────────────────────────────────────

class WeatherClient:
    """
    Cliente unificado:
      • Tomorrow.io → clima, pronóstico, códigos de clima
      • SIMA NL     → contaminación real (PM2.5, PM10, etc.)
      • Fallback    → simulación si falla alguna fuente
    """

    def __init__(
        self,
        tomorrow_key: Optional[str] = None,
        sima_estacion: str = "NORTE2",
        cache_ttl: int = 600,
        timeout: int = 15,
    ):
        self.tomorrow_key = tomorrow_key
        self.sima_estacion = sima_estacion
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cache: dict[str, CacheEntry] = {}
        self._use_real = bool(tomorrow_key)

        self._tomorrow: Optional[TomorrowClient] = None
        self._sima: Optional[SIMAClient] = None
        if self._use_real:
            self._tomorrow = TomorrowClient(api_key=tomorrow_key, timeout=timeout)
            self._sima = SIMAClient(estacion=sima_estacion, timeout=timeout)

    # ── CACHÉ ─────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[WeatherData]:
        entry = self._cache.get(key)
        if entry and time.time() < entry.expires_at:
            return entry.data
        return None

    def _cache_set(self, key: str, data: WeatherData):
        self._cache[key] = CacheEntry(data=data, expires_at=time.time() + self.cache_ttl)

    # ── FETCH REAL ────────────────────────────────────────────────

    async def _fetch_real(self, municipio: str, lat: float, lon: float, cat: str) -> WeatherData:
        """Combina Tomorrow.io + SIMA NL en un solo WeatherData."""
        # Ejecutar en paralelo: clima + contaminación
        t_task = asyncio.create_task(asyncio.to_thread(
            self._tomorrow.get_municipio, municipio, lat, lon
        ))
        s_task = asyncio.create_task(asyncio.to_thread(
            self._sima.fetch
        ))

        t_data, s_data = await asyncio.gather(t_task, s_task, return_exceptions=True)

        if isinstance(t_data, Exception):
            raise t_data
        if isinstance(s_data, Exception):
            s_data = None  # SIMA puede fallar, usamos fallback de contaminantes

        # Construir forecast_3d a partir de daily de Tomorrow
        forecast_3d = []
        for day in t_data.daily[:3]:
            forecast_3d.append({
                "day": day.time_iso,
                "temp_max": day.temp,
                "temp_min": day.feels_like,  # Tomorrow daily trae min/max en temp/feels_like
                "rain_prob": day.precipitation_prob / 100.0,
                "description": self._weather_desc(day.weather_code),
            })

        # Probabilidad lluvia 24h = media de las primeras 24h horarias
        rain_prob_24h = float(np.mean([
            h.precipitation_prob / 100.0 for h in t_data.hourly[:24]
        ])) if t_data.hourly else 0.0

        # Contaminantes: SIMA tiene prioridad; si falla, usamos simulación
        pm25 = s_data.pm25 if s_data and s_data.pm25 is not None else self._sim_pm25(cat)
        pm10 = s_data.pm10 if s_data and s_data.pm10 is not None else self._sim_pm10(cat)
        no2  = s_data.no2  if s_data and s_data.no2  is not None else self._sim_no2(cat)
        so2  = s_data.so2  if s_data and s_data.so2  is not None else self._sim_so2(cat)
        co   = s_data.co   if s_data and s_data.co   is not None else self._sim_co(cat)
        o3   = s_data.o3   if s_data and s_data.o3   is not None else self._sim_o3()

        # AQI aproximado a partir de PM2.5
        aqi = self._aqi_from_pm25(pm25)

        return WeatherData(
            municipio=municipio, lat=lat, lon=lon, categoria=cat,
            temperature=t_data.temperature,
            feels_like=t_data.feels_like,
            humidity=t_data.humidity,
            pressure=t_data.pressure,
            wind_speed=t_data.wind_speed,
            wind_deg=t_data.wind_deg,
            cloud_cover=t_data.cloud_cover,
            visibility=t_data.visibility,
            uvi=t_data.uvi,
            dew_point=t_data.dew_point,
            weather_code=t_data.weather_code,
            rain_intensity=t_data.rain_intensity,
            wind_gust=t_data.wind_gust,
            precipitation_prob=t_data.precipitation_prob,
            rain_1h=t_data.rain_intensity,
            rain_prob_24h=rain_prob_24h,
            forecast_3d=forecast_3d,
            hourly_forecast=t_data.hourly,
            aqi=aqi,
            pm25=pm25,
            pm10=pm10,
            no2=no2,
            co=co,
            o3=o3,
            so2=so2,
            source="tomorrow+sima" if s_data else "tomorrow+simulado",
            sima_estacion=self.sima_estacion,
            sima_hora=s_data.hora_reporte if s_data else "",
        )

    # ── SIMULACIÓN (fallback) ─────────────────────────────────────

    def _simulate(self, municipio: str, lat: float, lon: float, cat: str) -> WeatherData:
        rng = np.random.default_rng(seed=abs(hash(municipio)) % (2**31))
        industrial = cat == "industrial"

        temp = round(float(rng.uniform(22, 36)), 1)
        hum  = float(rng.integers(55, 82))
        pres = float(rng.integers(1003, 1018))
        wind = round(float(rng.uniform(8, 28)), 1)
        pm25 = round(float(rng.uniform(28, 68) if industrial else rng.uniform(10, 35)), 1)
        pm10 = round(float(rng.uniform(50, 95) if industrial else rng.uniform(20, 50)), 1)
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
            weather_code=int(rng.choice([1000, 1001, 4000, 4200])),
            rain_intensity=round(float(rng.uniform(0, 2.5) if rain_prob > 0.5 else 0), 1),
            wind_gust=round(wind + rng.uniform(2, 10), 1),
            precipitation_prob=round(rain_prob * 100, 1),
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
            source="simulado",
        )

    # ── UTILIDADES DE CALIBRACIÓN ─────────────────────────────────

    @staticmethod
    def _weather_desc(code: int) -> str:
        desc = {
            1000: "despejado", 1001: "nublado", 1100: "parcialmente nublado",
            4000: "lluvia ligera", 4001: "lluvia", 4200: "lluvia ligera",
            4201: "lluvia fuerte", 4202: "lluvia moderada", 8001: "tormenta",
        }
        return desc.get(code, "nublado")

    @staticmethod
    def _aqi_from_pm25(pm25: float) -> int:
        if pm25 <= 12:   return 1
        if pm25 <= 35:   return 2
        if pm25 <= 55:   return 3
        if pm25 <= 150:  return 4
        return 5

    def _sim_pm25(self, cat: str) -> float:
        rng = np.random.default_rng(seed=abs(hash(cat)) % (2**31))
        return round(float(rng.uniform(28, 68) if cat == "industrial" else rng.uniform(10, 35)), 1)

    def _sim_pm10(self, cat: str) -> float:
        rng = np.random.default_rng(seed=abs(hash(cat)) % (2**31))
        return round(float(rng.uniform(50, 95) if cat == "industrial" else rng.uniform(20, 50)), 1)

    def _sim_no2(self, cat: str) -> float:
        rng = np.random.default_rng(seed=abs(hash(cat)) % (2**31))
        return round(float(rng.uniform(18, 55) if cat == "industrial" else rng.uniform(5, 20)), 1)

    def _sim_so2(self, cat: str) -> float:
        rng = np.random.default_rng(seed=abs(hash(cat)) % (2**31))
        return round(float(rng.uniform(5, 30) if cat == "industrial" else rng.uniform(1, 8)), 1)

    def _sim_co(self, cat: str) -> float:
        rng = np.random.default_rng(seed=abs(hash(cat)) % (2**31))
        return round(float(rng.uniform(400, 900) if cat == "industrial" else rng.uniform(200, 500)), 1)

    def _sim_o3(self) -> float:
        rng = np.random.default_rng(seed=42)
        return round(float(rng.uniform(40, 90)), 1)

    # ── ASYNC FETCH WRAPPER ───────────────────────────────────────

    async def _fetch_weather_async(self, client: httpx.AsyncClient, municipio: str) -> WeatherData:
        cached = self._cache_get(municipio)
        if cached:
            return cached

        meta = MUNICIPIOS_AMM[municipio]
        lat, lon = meta["lat"], meta["lon"]

        if self._use_real:
            try:
                data = await self._fetch_real(municipio, lat, lon, meta["cat"])
            except Exception:
                data = self._simulate(municipio, lat, lon, meta["cat"])
        else:
            data = self._simulate(municipio, lat, lon, meta["cat"])

        self._cache_set(municipio, data)
        return data

    # ── API PÚBLICA ───────────────────────────────────────────────

    def get_municipio(self, municipio: str) -> WeatherData:
        if municipio not in MUNICIPIOS_AMM:
            raise ValueError(f"Municipio '{municipio}' no encontrado.")
        return asyncio.run(self._get_single(municipio))

    async def _get_single(self, municipio: str) -> WeatherData:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._fetch_weather_async(client, municipio)

    def get_all_municipios(self) -> dict[str, WeatherData]:
        return asyncio.run(self._get_all_async())

    async def _get_all_async(self) -> dict[str, WeatherData]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._fetch_weather_async(client, m) for m in MUNICIPIOS_AMM]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for muni, result in zip(MUNICIPIOS_AMM.keys(), results):
            if isinstance(result, Exception):
                meta = MUNICIPIOS_AMM[muni]
                output[muni] = self._simulate(muni, meta["lat"], meta["lon"], meta["cat"])
            else:
                output[muni] = result
        return output

    def get_municipios_list(self) -> list[str]:
        return list(MUNICIPIOS_AMM.keys())

    def close(self):
        if self._tomorrow:
            self._tomorrow.close()
        if self._sima:
            self._sima.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
