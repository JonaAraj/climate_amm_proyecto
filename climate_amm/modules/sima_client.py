"""
sima_client.py
==============
Cliente para consumir datos reales del SIMA Nuevo León.

Los datos se obtienen de endpoints JSON públicos (no requieren autenticación):
  • api_indice.php  → Índice de calidad del aire y contaminantes actuales
  • api_conc.php    → Concentraciones horarias (histórico del día)
  • api_meteo.php   → Datos meteorológicos horarios

Estaciones disponibles (estacion1):
  NORTE2, NORESTE, NTE2, etc. — usar 'NORTE2' como default (San Nicolás-UANL)
"""

import httpx
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SIMAData:
    """Datos consolidados de una estación SIMA."""
    estacion: str
    hora_reporte: str       # "YYYY-MM-DD HH:MM"
    # Contaminantes (µg/m³)
    pm25: Optional[float]
    pm10: Optional[float]
    o3: Optional[float]
    no2: Optional[float]
    so2: Optional[float]
    co: Optional[float]
    # Meteorología
    temperatura: Optional[float]   # °C
    humedad: Optional[float]       # %
    presion: Optional[float]       # mmHg
    viento_vel: Optional[float]    # km/h
    viento_dir: Optional[str]
    lluvia: Optional[float]        # mm/hr
    # Metadatos
    timestamp: float = 0.0
    source: str = "sima_nl"

    def presion_hpa(self) -> Optional[float]:
        """Convierte mmHg → hPa."""
        return round(self.presion * 1.33322, 1) if self.presion else None


class SIMAClient:
    """
    Cliente ligero para SIMA NL.
    No requiere API key; consume endpoints JSON públicos.
    """

    BASE = "https://aire.nl.gob.mx/SIMA2017reportes"
    DEFAULT_ESTACION = "NORTE2"

    # Mapeo de códigos SIMA a campos del dataclass
    INDICE_PARAMS = {
        "PM25_12": "pm25",
        "PM10_12": "pm10",
        "O3m": "o3",
        "NO2m": "no2",
        "SO2_1": "so2",
        "CO8m": "co",
    }

    METEO_PARAMS = {
        "TOUT": ("temperatura", float),
        "RH": ("humedad", float),
        "PRS": ("presion", float),
        "WSR": ("viento_vel", float),
        "WDR": ("viento_dir", str),
        "RAINF": ("lluvia", float),
    }

    def __init__(self, estacion: str = DEFAULT_ESTACION, timeout: int = 15):
        self.estacion = estacion
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _get(self, endpoint: str) -> list[dict]:
        url = f"{self.BASE}/{endpoint}?t={int(time.time() * 1000)}"
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    def fetch(self) -> SIMAData:
        """
        Obtiene el último registro disponible de la estación configurada.
        Combina índice, concentraciones y meteorología.
        """
        indice = self._get("api_indice.php")
        meteo = self._get("api_meteo.php")

        # Parsear índice (último valor de cada parámetro)
        vals = {
            "estacion": self.estacion,
            "hora_reporte": "",
            "pm25": None, "pm10": None, "o3": None,
            "no2": None, "so2": None, "co": None,
            "temperatura": None, "humedad": None,
            "presion": None, "viento_vel": None,
            "viento_dir": None, "lluvia": None,
            "timestamp": time.time(),
        }

        for row in indice:
            param = row.get("Parameter", "").strip()
            field = self.INDICE_PARAMS.get(param)
            if field:
                raw = row.get("HrAveData")
                if raw not in (None, "", "ND"):
                    try:
                        vals[field] = float(raw)
                    except (ValueError, TypeError):
                        pass
                if not vals["hora_reporte"] and row.get("Date"):
                    vals["hora_reporte"] = str(row["Date"])

        # Parsear meteorología (último valor de cada parámetro)
        for row in meteo:
            param = row.get("Parameter", "").strip()
            mapping = self.METEO_PARAMS.get(param)
            if mapping:
                field, cast = mapping
                raw = row.get("HrAveData")
                if raw not in (None, "", "ND"):
                    try:
                        vals[field] = cast(raw)
                    except (ValueError, TypeError):
                        pass

        return SIMAData(**vals)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
