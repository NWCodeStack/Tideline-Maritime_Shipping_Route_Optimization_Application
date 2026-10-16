"""
sources.py - every input behind the current plan, with where it comes from (the "Data sources" tab).

Each row says what the input is, its basis, the date it was fetched when it is live, and a link. The basis is one of
BASES, and nothing simulated is ever labelled live: if a live feed is down, the row says so and names the synthetic
fallback the simulator is using instead (CLAUDE.md rule 1).
"""
from __future__ import annotations

from economics import (CO2_PER_T_FUEL, ECA_FUEL_PREMIUM_USD_PER_T, PORT_FUEL_PRICES, effective_fuel_price)
from events import describe

BASES = {"live", "published", "estimated", "sample", "simulated", "user setting", "unavailable"}


def _row(name: str, basis: str, detail: str, date: str = "-", url: str = "") -> dict:
    return {"input": name, "basis": basis, "detail": detail, "date": date, "url": url}


def _weather_row(weather: dict | None) -> dict:
    url = "https://open-meteo.com/"
    if weather and weather.get("status") in ("ok", "partial"):
        return _row("Weather (Open-Meteo)", "live",
                    f"Waves, wind and currents matched to the ship's simulated arrival times. Status: "
                    f"{weather['status']}. {weather.get('source', '')}", weather.get("fetched_at") or "-", url)
    reason = (weather or {}).get("note") or "not fetched yet"
    return _row("Weather (Open-Meteo)", "simulated",
                f"Live forecast unavailable ({reason}). The simulator is using its synthetic sea-state model.",
                "-", url)


def _news_row(news: dict | None) -> dict:
    if news and news.get("articles"):
        if news.get("demo"):
            return _row("News and security headlines", "sample",
                        "Labelled DEMO sample headlines, not real articles. Advisory only: headlines never close a "
                        "corridor or change the risk score.")
        provider = news.get("provider", "news provider")
        url = ("https://www.gdeltproject.org/" if "GDELT" in provider else "https://news.google.com/")
        return _row("News and security headlines", "live",
                    f"{provider}. Advisory only: headlines never close a corridor or change the risk score.",
                    news.get("fetched_at") or "-", url)
    return _row("News and security headlines", "unavailable",
                "No headlines loaded. Nothing is invented; the plan does not depend on them.")


def _fuel_price_row(ops) -> dict:
    cfg = ops.cfg
    e = cfg.econ
    price = effective_fuel_price(cfg.fuel_price, cfg.origin, e)
    if e.fuel_price_mode == "origin_port":
        override = dict(e.fuel_overrides).get(cfg.origin)
        if override is not None:
            return _row("Fuel price", "user setting", f"${override:,.0f}/t at {ops.port_name(cfg.origin)}, your override.")
        info = PORT_FUEL_PRICES.get(cfg.origin)
        if info:
            return _row("Fuel price", "sample", f"${price:,.0f}/t at {ops.port_name(cfg.origin)}: {info['basis']}.")
    return _row("Fuel price", "user setting", f"${price:,.0f}/t from the fuel-price setting (not a market quote).")


def plan_sources(ops, weather: dict | None, news: dict | None, llm_label: str) -> list[dict]:
    """The rows of the Data sources tab for the current voyage and settings."""
    e = ops.cfg.econ
    G = ops.G
    metrics = getattr(ops.model, "metrics", None) or {}
    r2 = metrics.get("rf_r2")
    scenarios = describe(ops.settings)
    return [
        _row("Vessel position and clock", "simulated",
             "A simulated ship moves along the modelled lanes at the chosen playback speed. Never AIS."),
        _row("Lane network", "simulated",
             f"Synthetic network of {G.number_of_nodes()} nodes and {G.number_of_edges()} lanes with real chokepoint "
             "geography, land-checked. Not certified sea lanes."),
        _row("Fuel burn", "estimated",
             "Random forest fit to 5,000 SIMULATED voyages"
             + (f" (R² {r2:.2f}, on synthetic data only)" if r2 is not None else "")
             + "; a physics fallback is used if the model is missing."),
        _fuel_price_row(ops),
        _row("Canal tolls", "sample", "Illustrative tariffs by vessel type with user-adjustable fee multipliers."),
        _row("War-risk and security", "simulated",
             "Synthetic risk zones plus any scenario you switch on. News headlines never change it."),
        _weather_row(weather),
        _news_row(news),
        _row("ECA fuel switch", "estimated",
             f"${ECA_FUEL_PREMIUM_USD_PER_T:,.0f}/t premium on fuel burned inside simplified ECA boxes "
             "(coarse rectangles, not the legal boundaries)."),
        _row("EU ETS", "estimated",
             f"${e.eua_usd_per_t:,.0f}/t allowance x {e.ets_coverage_pct:g}% coverage for voyages calling at an EU "
             f"port; currently {'on' if e.ets_enabled else 'off'}."),
        _row("CO2 per tonne of fuel", "published",
             f"{CO2_PER_T_FUEL} t CO2 per t of heavy fuel oil (the IMO default emission factor).",
             url="https://www.imo.org/"),
        _row("Risk score", "simulated",
             "Weather, security and traffic signals combined per leg (noisy-OR) and weighted by time at sea. A "
             "relative comparison between routes, not a calibrated probability."),
        _row("Disruption scenarios", "simulated",
             (("Active: " + ", ".join(scenarios) + ". ") if scenarios else "None active. ")
             + "Scenarios are simulated, never real closures."),
        _row("Copilot", "user setting",
             f"{llm_label or 'Rules-based command mode'}. It only proposes operations from a fixed schema; it never "
             "executes code and questions never change the voyage."),
    ]
