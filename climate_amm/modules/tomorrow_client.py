"""
tomorrow_client.py
==================
Cliente para Tomorrow.io Weather API v4.
Reemplaza a OpenWeather como fuente meteorológica principal.

API Key proporcionada por el usuario.
"""

import httpx
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TomorrowForecast:
    """Intervalo de pronóstico de Tomorrow.io."""
    time_iso: str
    temp: float
    feels_like: float
    humidity: float
    pressure_surface: float     # hPa
    pressure_sea: float         # hPa
    wind_speed: float           # km/h
    wind_direction: float       # grados
    wind_gust: float            # km/h
    cloud_cover: float          # %
    visibility: float           # km
    uv_index: float
    precipitation_prob: float   # %
    rain_intensity: float       # mm/hr
    rain_accumulation: float    # mm
    weather_code: int
    dew_point: float


@dataclass
class TomorrowData:
    """Datos consolidados de Tomorrow.io para un municipio."""
    municipio: str
    lat: float
    lon: float
    # Actual
    temperature: float
    feels_like: float
    humidity: float
    pressure: float          # hPa (surface)
    wind_speed: float        # km/h
    wind_deg: float
    wind_gust: float
    cloud_cover: float
    visibility: float        # km
    uvi: float
    dew_point: float
    precipitation_prob: float
    rain_intensity: float
    weather_code: int
    # Pronóstico
    hourly: list[TomorrowForecast] = field(default_factory=list)
    daily: list[TomorrowForecast] = field(default_factory=list)
    # Metadatos
    timestamp: float = field(default_factory=time.time)
    source: str = "tomorrow.io"


class TomorrowClient:
    """
    Cliente para Tomorrow.io v4 Weather Forecast API.
    """

    BASE = "https://api.tomorrow.io/v4/weather/forecast"

    # Códigos de clima relevantes para lluvia
    RAIN_CODES = {4000, 4001, 4200, 4201, 4202, 4203, 4204, 4205}
    STORM_CODES = {8001, 8002, 8003}

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _fetch(self, lat: float, lon: float) -> dict:
        params = {
            "location": f"{lat},{lon}",
            "timesteps": "1h,1d",
            "units": "metric",
            "apikey": self.api_key,
        }
        resp = self._client.get(self.BASE, params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_interval(item: dict) -> TomorrowForecast:
        v = item.get("values", {})
        return TomorrowForecast(
            time_iso=item.get("time", ""),
            temp=v.get("temperature", 0),
            feels_like=v.get("temperatureApparent", 0),
            humidity=v.get("humidity", 0),
            pressure_surface=v.get("pressureSurfaceLevel", 1013),
            pressure_sea=v.get("pressureSeaLevel", 1013),
            wind_speed=round(v.get("windSpeed", 0) * 3.6, 1),   # m/s → km/h
            wind_direction=v.get("windDirection", 0),
            wind_gust=round(v.get("windGust", 0) * 3.6, 1),
            cloud_cover=v.get("cloudCover", 0),
            visibility=v.get("visibility", 10),
            uv_index=v.get("uvIndex", 0),
            precipitation_prob=v.get("precipitationProbability", 0),
            rain_intensity=v.get("rainIntensity", 0),
            rain_accumulation=v.get("rainAccumulation", 0),
            weather_code=v.get("weatherCode", 0),
            dew_point=v.get("dewPoint", 0),
        )

    def get_municipio(self, municipio: str, lat: float, lon: float) -> TomorrowData:
        """Obtiene datos actuales + pronóstico para un municipio."""
        raw = self._fetch(lat, lon)
        timelines = raw.get("timelines", {})

        hourly = [self._parse_interval(i) for i in timelines.get("hourly", [])]
        daily = [self._parse_interval(i) for i in timelines.get("daily", [])]

        # Datos "actuales" = primer intervalo horario
        now = hourly[0] if hourly else None
        if now is None:
            raise RuntimeError("Tomorrow.io no devolvió datos horarios")

        return TomorrowData(
            municipio=municipio,
            lat=lat, lon=lon,
            temperature=now.temp,
            feels_like=now.feels_like,
            humidity=now.humidity,
            pressure=now.pressure_surface,
            wind_speed=now.wind_speed,
            wind_deg=now.wind_direction,
            wind_gust=now.wind_gust,
            cloud_cover=now.cloud_cover,
            visibility=now.visibility,
            uvi=now.uv_index,
            dew_point=now.dew_point,
            precipitation_prob=now.precipitation_prob,
            rain_intensity=now.rain_intensity,
            weather_code=now.weather_code,
            hourly=hourly,
            daily=daily,
        )

    def detect_rain_events(self, hourly: list[TomorrowForecast],
                           min_prob: float = 50.0, min_intensity: float = 0.5) -> list[dict]:
        """
        Detecta eventos de lluvia significativos en el pronóstico horario.
        Retorna lista de dicts con inicio, fin, max_temp_antes, min_temp_despues.
        """
        events = []
        in_rain = False
        start_idx = 0

        for i, h in enumerate(hourly):
            is_rain = (h.precipitation_prob >= min_prob and
                       (h.rain_intensity >= min_intensity or h.weather_code in self.RAIN_CODES))
            if is_rain and not in_rain:
                in_rain = True
                start_idx = i
            elif not is_rain and in_rain:
                in_rain = False
                events.append(self._extract_event(hourly, start_idx, i - 1))

        if in_rain:
            events.append(self._extract_event(hourly, start_idx, len(hourly) - 1))

        return events

    @staticmethod
    def _extract_event(hourly: list[TomorrowForecast], start: int, end: int) -> dict:
        """Extrae métricas de un evento de lluvia."""
        # Temperatura máxima 6h antes del inicio
        pre_start = max(0, start - 6)
        temps_before = [h.temp for h in hourly[pre_start:start]]
        max_before = max(temps_before) if temps_before else hourly[start].temp

        # Temperatura mínima 6h después del fin
        post_end = min(len(hourly), end + 7)
        temps_after = [h.temp for h in hourly[end + 1:post_end]]
        min_after = min(temps_after) if temps_after else hourly[end].temp

        # Intensidad máxima durante el evento
        max_intensity = max(h.rain_intensity for h in hourly[start:end + 1])
        max_prob = max(h.precipitation_prob for h in hourly[start:end + 1])
        duration = end - start + 1

        return {
            "start_idx": start,
            "end_idx": end,
            "start_time": hourly[start].time_iso,
            "end_time": hourly[end].time_iso,
            "duration_h": duration,
            "max_intensity_mmh": max_intensity,
            "max_prob": max_prob,
            "temp_before_max": max_before,
            "temp_after_min": min_after,
            "temp_drop": round(max_before - min_after, 1),
            "codes": list(set(h.weather_code for h in hourly[start:end + 1])),
        }

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
