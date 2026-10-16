"""
risk.py - what the 0-100 risk score means, and what it is made of.

Score = round(100 x modelled route risk). For each leg three signals are combined with a noisy-OR,

    p_leg = 1 - (1 - security) x (1 - weather) x (1 - traffic)

and the legs are averaged weighted by the hours spent at sea (time exposed). The score is a relative comparison
between routes, not a calibrated probability, and every input is SIMULATED (see ``BASIS``).

Breakdown: each leg's p_leg is split between the three signals in proportion to their sizes, so the weather,
security and traffic parts of the route add up exactly to the total. The most severe lane is reported separately,
with its security split into the base level and what a simulated scenario added. News headlines are advisory and
never change the score (CLAUDE.md rule 2). The noisy-OR / time-exposure structure follows the Pelagic prototype's
risk_model.py; the inputs are Tideline's own.
"""
from __future__ import annotations

from dataclasses import dataclass

CAPTION = ("Risk score (0-100): modelled chance that something disrupts this voyage — weather beyond the ship's "
           "limits, a security incident, or a chokepoint slowdown. It combines weather, security and traffic "
           "signals per leg, weighted by time spent exposed. It's a relative comparison between routes, not a "
           "calibrated probability.")

# (lowest score in the band, label, colour tone used by the UI)
BANDS = ((0, "Low", "mint"), (20, "Guarded", "blue"), (40, "Elevated", "gold"), (60, "High", "red"),
         (80, "Severe", "red"))

BASIS = {
    "weather": "SIMULATED sea-state severity of each lane (synthetic weather zones); not yet checked against this "
               "ship's wave limit",
    "security": "SIMULATED: base level of each zone plus any scenario you switched on",
    "traffic": "SIMULATED slowdown from the canal-delay setting; no live traffic feed in this build",
}

TRAFFIC_FULL_RISK_DELAY_H = 168.0      # a canal delay of one week counts as traffic risk 1.0 on a canal lane
WEATHER_WEIGHT = 0.5                   # weather counts half as much as security (as in the routing objective)


def band(score: float) -> tuple[str, str]:
    """(label, colour tone) for a score; fractional scores are rounded first."""
    s = round(score)
    label, tone = BANDS[0][1], BANDS[0][2]
    for start, lab, col in BANDS:
        if s >= start:
            label, tone = lab, col
    return label, tone


def _clip(p: float) -> float:
    return min(max(p, 0.0), 1.0)


def leg_risk(p_sec: float, p_wx: float, p_traf: float) -> float:
    """Noisy-OR of the three signals: the chance that at least one of them disrupts the leg."""
    return 1.0 - (1.0 - _clip(p_sec)) * (1.0 - _clip(p_wx)) * (1.0 - _clip(p_traf))


@dataclass(frozen=True)
class RiskBreakdown:
    total: float = 0.0                 # time-weighted mean of the legs' noisy-OR (0..1)
    weather: float = 0.0               # parts: add up to ``total``
    security: float = 0.0
    traffic: float = 0.0
    top_zone: str = ""                 # the lane with the highest security risk
    top_security: float = 0.0          # = top_base + top_scenario
    top_base: float = 0.0
    top_scenario: float = 0.0

    @property
    def score(self) -> int:
        return round(100 * self.total)

    @property
    def band_label(self) -> str:
        return band(self.score)[0]


def route_risk(legs) -> RiskBreakdown:
    """Risk of a list of legs (a whole voyage, or just the part still ahead)."""
    sail = [l for l in legs if l.kind == "sail" and l.sea_h > 0]
    hours = sum(l.sea_h for l in sail)
    if hours <= 0:
        return RiskBreakdown()
    total = wx = sec = traf = 0.0
    top = None
    for l in sail:
        p_sec = l.p_sec_base + l.p_sec_scen
        p_leg = leg_risk(p_sec, l.p_wx, l.p_traf)
        raw = _clip(p_sec) + _clip(l.p_wx) + _clip(l.p_traf)
        w = l.sea_h / hours
        total += w * p_leg
        if raw > 0:
            sec += w * p_leg * _clip(p_sec) / raw
            wx += w * p_leg * _clip(l.p_wx) / raw
            traf += w * p_leg * _clip(l.p_traf) / raw
        if top is None or p_sec > top[0] + 1e-12:
            top = (p_sec, l)
    z = top[1]
    return RiskBreakdown(total=total, weather=wx, security=sec, traffic=traf, top_zone=z.zone,
                         top_security=z.p_sec_base + z.p_sec_scen, top_base=z.p_sec_base, top_scenario=z.p_sec_scen)


def explain(rb: RiskBreakdown) -> str:
    """The score in words: its parts and the most severe lane. Plain text, no HTML."""
    parts = (f"Parts of the score (they add up to {rb.total:.2f}): weather {rb.weather:.2f}, "
             f"security {rb.security:.2f}, traffic {rb.traffic:.2f}.")
    if not rb.top_zone:
        return parts
    lane = (f" Highest-security lane: {rb.top_zone}: security {rb.top_security:.2f} "
            f"(base {rb.top_base:.2f} + scenario {rb.top_scenario:.2f}).")
    return (parts + lane + " News headlines are advisory and do not change the score. "
            "It is a relative comparison between routes, not a calibrated probability.")
