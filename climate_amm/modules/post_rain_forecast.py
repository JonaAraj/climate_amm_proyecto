"""
post_rain_forecast.py
=====================
Módulo de pronóstico post-lluvia para el AMM.

Detecta eventos de precipitación en el forecast horario de Tomorrow.io
y proyecta la tendencia térmica posterior:
  • "Freón post-tormenta": descenso brusco de temperatura tras lluvia
  • "Calentamiento post-lluvia": inversión térmica rara en zona semiárida

Basado en datos de Tomorrow.io (hourly forecast).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PostRainForecast:
    """Resultado de un pronóstico post-lluvia."""
    has_rain: bool
    event_time: str = ""
    event_desc: str = "Sin lluvia significativa en el horizonte"
    temp_before: float = 0.0
    temp_after: float = 0.0
    temp_change: float = 0.0
    trend: str = "estable"
    icon: str = "☀️"
    hours_after: int = 6
    confidence: str = "alta"


class PostRainAnalyzer:
    """
    Analiza el pronóstico horario de Tomorrow.io para detectar
    eventos de lluvia y proyectar la tendencia térmica posterior.
    """

    def __init__(self, min_prob: float = 50.0, min_intensity: float = 0.3):
        self.min_prob = min_prob
        self.min_intensity = min_intensity

    def analyze(self, hourly: list, municipio: str = "") -> PostRainForecast:
        """
        Analiza la serie horaria y retorna el pronóstico post-lluvia
        más relevante (primer evento significativo en las próximas 48h).
        """
        if not hourly:
            return PostRainForecast(has_rain=False)

        # Detectar primer evento de lluvia
        event = self._find_first_event(hourly)
        if not event:
            return PostRainForecast(has_rain=False)

        return self._build_forecast(event, hourly, municipio)

    def _find_first_event(self, hourly: list) -> Optional[dict]:
        from .tomorrow_client import TomorrowClient
        tc = TomorrowClient("", timeout=5)
        events = tc.detect_rain_events(
            hourly, min_prob=self.min_prob, min_intensity=self.min_intensity
        )
        return events[0] if events else None

    def _build_forecast(self, event: dict, hourly: list, municipio: str) -> PostRainForecast:
        """Construye el objeto de pronóstico post-lluvia."""
        temp_before = event["temp_before_max"]
        temp_after = event["temp_after_min"]
        change = event["temp_drop"]

        if change >= 5.0:
            trend = "descenso_brusco"
            desc = (
                f"Se proyecta un evento de lluvia significativo. "
                f"Después de la precipitación, la temperatura podría caer "
                f"de {temp_before:.1f}°C a {temp_after:.1f}°C (↓{change:.1f}°C). "
                f"Efecto 'freón post-tormenta' típico del AMM."
            )
            icon = "❄️"
        elif change >= 2.0:
            trend = "descenso_moderado"
            desc = (
                f"Lluvia moderada prevista. Temperatura post-evento: "
                f"{temp_after:.1f}°C (↓{change:.1f}°C respecto al pico previo)."
            )
            icon = "🌬️"
        elif change <= -2.0:
            trend = "calentamiento"
            desc = (
                f"Inusual: después de la lluvia, la temperatura subiría "
                f"a {temp_after:.1f}°C (↑{abs(change):.1f}°C). Posible inversión térmica."
            )
            icon = "🔥"
        else:
            trend = "estable"
            desc = (
                f"Lluvia prevista, pero sin cambio térmico significativo "
                f"posterior (±{abs(change):.1f}°C)."
            )
            icon = "🌤️"

        # Confianza basada en probabilidad máxima del evento
        max_prob = event.get("max_prob", 0)
        conf = "alta" if max_prob >= 80 else "media" if max_prob >= 50 else "baja"

        return PostRainForecast(
            has_rain=True,
            event_time=event["start_time"],
            event_desc=desc,
            temp_before=temp_before,
            temp_after=temp_after,
            temp_change=change,
            trend=trend,
            icon=icon,
            hours_after=6,
            confidence=conf,
        )

    def summary_cards(self, hourly: list) -> list[dict]:
        """
        Genera tarjetas resumen de todos los eventos post-lluvia
        detectados en las próximas 48h.
        """
        from .tomorrow_client import TomorrowClient
        tc = TomorrowClient("", timeout=5)
        events = tc.detect_rain_events(hourly, min_prob=self.min_prob, min_intensity=self.min_intensity)

        cards = []
        for ev in events:
            cards.append({
                "hora_inicio": ev["start_time"][11:16] if "T" in ev["start_time"] else ev["start_time"],
                "hora_fin": ev["end_time"][11:16] if "T" in ev["end_time"] else ev["end_time"],
                "duracion_h": ev["duration_h"],
                "max_prob": ev["max_prob"],
                "temp_antes": f"{ev['temp_before_max']:.1f}°C",
                "temp_despues": f"{ev['temp_after_min']:.1f}°C",
                "cambio": f"↓{ev['temp_drop']:.1f}°C" if ev['temp_drop'] > 0 else f"↑{abs(ev['temp_drop']):.1f}°C",
            })
        return cards
