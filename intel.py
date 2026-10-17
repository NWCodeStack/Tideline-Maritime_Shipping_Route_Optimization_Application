"""
intel.py - live maritime intelligence: Open-Meteo marine forecasts and maritime news.

Pure functions (no Streamlit) so they can be tested with mocked HTTP. app.py wraps them in ``st.cache_data``.

Weather   Open-Meteo Marine API (waves, swell, currents, sea-surface temperature) + Open-Meteo Forecast API
          (10 m wind; the marine API has no wind variable). One request per API covers every sampled waypoint.
          Each waypoint is matched to the ship's SIMULATED arrival time; times outside the forecast window are
          reported as "beyond forecast horizon", never filled in.
News      GDELT DOC 2.0 API (primary). GDELT rate-limits aggressively (1 request / 5 s, and shared IPs are often
          throttled), so Google News RSS search is used as a public fallback. Headlines are ADVISORY only: they
          never close a corridor in the simulator; the user can choose to simulate a matching scenario.

Every result carries status ("ok" | "partial" | "unavailable"), source, fetch timestamp and error text.
Set TIDELINE_OFFLINE=1 to disable all network access (tests, offline demos).
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WIND_URL = "https://api.open-meteo.com/v1/forecast"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RSS_URL = "https://news.google.com/rss/search"
TIMEOUT_S = 8
USER_AGENT = "Tideline-maritime-demo/1.0 (hackathon prototype)"
MARINE_VARS = ["wave_height", "wave_direction", "wave_period", "swell_wave_height", "ocean_current_velocity",
               "ocean_current_direction", "sea_surface_temperature"]
FORECAST_DAYS = 7


class IntelError(RuntimeError):
    pass


def offline() -> bool:
    return os.environ.get("TIDELINE_OFFLINE", "").strip() not in ("", "0", "false", "False")


def _get(url: str, params: dict, retries: int = 1, timeout: float = TIMEOUT_S) -> requests.Response:
    """GET with timeout and ONE retry on connection errors / 5xx. 429 (rate limit) is not retried."""
    if offline():
        raise IntelError("offline mode (TIDELINE_OFFLINE=1)")
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as e:
            last = IntelError(f"network error: {type(e).__name__}")
        else:
            if r.status_code == 429:
                raise IntelError("rate limited by provider (HTTP 429)")
            if r.status_code >= 500:
                last = IntelError(f"provider error HTTP {r.status_code}")
            elif r.status_code != 200:
                raise IntelError(f"HTTP {r.status_code}")
            else:
                return r
        if attempt < retries:
            time.sleep(0.8)
    raise last or IntelError("request failed")


# =================================================================================================
# Weather
# =================================================================================================
def fetch_marine(coords: tuple) -> dict:
    """Raw Open-Meteo responses for ``coords`` ((lat, lon), ...). Raises IntelError on total failure.
    Wind failure alone is tolerated (``wind`` = None)."""
    lats = ",".join(f"{la:.2f}" for la, _ in coords)
    lons = ",".join(f"{lo:.2f}" for _, lo in coords)
    marine = _get(MARINE_URL, {"latitude": lats, "longitude": lons, "hourly": ",".join(MARINE_VARS),
                               "forecast_days": FORECAST_DAYS, "timezone": "GMT"})
    try:
        m = marine.json()
    except ValueError:
        raise IntelError("marine API returned non-JSON") from None
    m = m if isinstance(m, list) else [m]
    if any("hourly" not in x for x in m):
        raise IntelError(str(m[0].get("reason", "marine API returned no hourly data"))[:120])
    wind = None
    try:
        w = _get(WIND_URL, {"latitude": lats, "longitude": lons, "hourly": "wind_speed_10m,wind_direction_10m",
                            "wind_speed_unit": "kn", "forecast_days": FORECAST_DAYS, "timezone": "GMT"}).json()
        wind = w if isinstance(w, list) else [w]
    except (IntelError, ValueError):
        wind = None
    return {"marine": m, "wind": wind, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def severity(wave_m: float | None, wind_kn: float | None) -> tuple[str, int]:
    w = wave_m or 0.0
    k = wind_kn or 0.0
    if w >= 6 or k >= 48:
        return "Severe", 3
    if w >= 4 or k >= 34:
        return "High", 2
    if w >= 2.5 or k >= 22:
        return "Moderate", 1
    return "Low", 0


ACTIONS = {
    0: "No action needed.",
    1: "Monitor; expect minor speed loss and higher fuel burn.",
    2: "Consider slowing down, securing cargo, or weather routing around the area.",
    3: "Avoid the area: simulate a storm scenario here to let the router find an alternative.",
}


def _parse_hours(times: list[str]) -> list[datetime]:
    return [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]


def assess_weather(raw: dict, points: list[dict], departure: datetime) -> dict:
    """Match each sampled waypoint to its simulated arrival time. ``points``: [{lat, lon, t (voyage h), label}]."""
    out_pts = []
    for i, p in enumerate(points):
        m = raw["marine"][i] if i < len(raw["marine"]) else None
        w = raw["wind"][i] if raw.get("wind") and i < len(raw["wind"]) else None
        eta = departure + timedelta(hours=p["t"])
        rec = {"label": p["label"], "lat": p["lat"], "lon": p["lon"], "eta": eta, "voyage_h": p["t"],
               "status": "beyond forecast horizon", "values": {}}
        if m:
            hours = _parse_hours(m["hourly"]["time"])
            if hours and hours[0] - timedelta(minutes=30) <= eta <= hours[-1] + timedelta(minutes=30):
                j = min(range(len(hours)), key=lambda k: abs((hours[k] - eta).total_seconds()))
                vals = {v: m["hourly"].get(v, [None] * len(hours))[j] for v in MARINE_VARS}
                if w and "hourly" in w:              # same GMT hourly grid as the marine API
                    ws = w["hourly"].get("wind_speed_10m") or []
                    wd = w["hourly"].get("wind_direction_10m") or []
                    vals["wind_speed_10m"] = ws[j] if j < len(ws) else None
                    vals["wind_direction_10m"] = wd[j] if j < len(wd) else None
                rec["values"] = vals
                rec["forecast_time"] = hours[j]
                if vals.get("wave_height") is None:
                    rec["status"] = "no marine data (coastal / enclosed water)"
                else:
                    rec["status"] = "forecast"
            elif hours and eta < hours[0]:
                rec["status"] = "before forecast window"
        sev, lvl = severity(rec["values"].get("wave_height"), rec["values"].get("wind_speed_10m"))
        rec["severity"], rec["level"] = (sev, lvl) if rec["status"] == "forecast" else ("n/a", -1)
        rec["action"] = ACTIONS.get(lvl, "") if rec["status"] == "forecast" else ""
        out_pts.append(rec)
    valid = [r for r in out_pts if r["status"] == "forecast"]
    worst = max(valid, key=lambda r: (r["values"].get("wave_height") or 0.0), default=None)
    units = {"wave_height": "m", "swell_wave_height": "m", "wave_period": "s", "wave_direction": "°",
             "ocean_current_velocity": "km/h", "ocean_current_direction": "°", "sea_surface_temperature": "°C",
             "wind_speed_10m": "kn", "wind_direction_10m": "°"}
    status = "ok" if valid and raw.get("wind") is not None else ("partial" if valid else "unavailable")
    note = "" if raw.get("wind") is not None else "Wind unavailable (forecast API failed); waves only."
    if not valid:
        note = "No sampled waypoint falls inside the 7-day forecast window at its simulated arrival time."
    return {"status": status, "points": out_pts, "worst": worst, "near_ship": out_pts[0] if out_pts else None,
            "fetched_at": raw["fetched_at"], "units": units, "note": note,
            "source": "Open-Meteo Marine API + Open-Meteo Forecast API (wind)"}


def operational_impact(wave_m: float | None, wind_kn: float | None = None) -> str:
    """Estimated effect on a ship using the SAME physics the routing engine uses (modelled, not measured):
    speed over ground x (1 - 0.5 w^2) with w the sea-state severity, fuel burn x (1 + 0.012 wave^2 + 0.0002 wind^2)."""
    if wave_m is None:
        return ""
    from events import wave_to_severity
    w = wave_to_severity(wave_m)
    speed_loss = 0.5 * w ** 2 * 100
    fuel_up = (0.012 * wave_m ** 2 + 0.0002 * (wind_kn or 0.0) ** 2) * 100
    return f"model estimate: ~{speed_loss:.0f}% speed loss, ~{fuel_up:.0f}% more fuel per day at sea"


def with_route(w: dict, route_sig: str, coords: tuple) -> dict:
    """Tag an assessment with the itinerary and sample points it was computed for."""
    return dict(w, route_sig=route_sig, coords=coords)


def reuse_or_unavailable(last: dict | None, route_sig: str, error: str) -> dict:
    """After a failed refresh: reuse the last good assessment ONLY if it belongs to the same itinerary (and so the
    same locations and ETAs). A forecast for a superseded route or another voyage is never shown as current."""
    if last and last.get("route_sig") == route_sig:
        return dict(last, stale=True,
                    note=f"Live refresh failed ({error}); showing the last successful forecast for this route, "
                         f"fetched {last['fetched_at']} UTC.")
    note = f"Weather unavailable: {error}."
    if last:
        note += " (A cached forecast exists for a previous route and is not shown.)"
    return weather_unavailable(note)


def current_for(w: dict | None, route_sig: str) -> dict | None:
    """The stored assessment if it matches the current itinerary, else None (used by map markers/alerts/copilot)."""
    return w if w and w.get("route_sig") == route_sig else None


def weather_unavailable(error: str) -> dict:
    return {"status": "unavailable", "points": [], "worst": None, "near_ship": None, "fetched_at": None,
            "units": {}, "note": error, "source": "Open-Meteo"}


def weather_summary(w: dict | None) -> str:
    """Plain-text summary for the copilot, built only from fetched data."""
    if not w or w["status"] == "unavailable":
        return ("Live marine weather is unavailable" + (f" ({w['note']})" if w and w.get("note") else "")
                + ". No forecast data is shown rather than guessing.")
    parts = []
    ns = w["near_ship"]
    if ns and ns["status"] == "forecast":
        v = ns["values"]
        parts.append(f"Near the ship: waves {v.get('wave_height')} m, swell {v.get('swell_wave_height')} m"
                     + (f", wind {v.get('wind_speed_10m')} kn" if v.get("wind_speed_10m") is not None else "")
                     + f" ({ns['severity']}).")
    elif ns:
        parts.append(f"Near the ship: {ns['status']}.")
    wr = w["worst"]
    if wr:
        parts.append(f"Worst ahead: {wr['values'].get('wave_height')} m waves {wr['label'].lower()} around "
                     f"{wr['eta']:%b %d %H:%M} UTC ({wr['severity']}). {wr['action']}")
    beyond = sum(1 for p in w["points"] if p["status"] != "forecast")
    if beyond:
        parts.append(f"{beyond} of {len(w['points'])} sampled waypoints are outside the forecast window or have no data.")
    parts.append(f"Source: {w['source']}, fetched {w['fetched_at']} UTC"
                 + (" (stale cached copy)" if w.get("stale") else "") + ".")
    return " ".join(parts)


# =================================================================================================
# News
# =================================================================================================
TOPIC_TERMS = ['"Red Sea"', '"Bab el-Mandeb"', '"Strait of Hormuz"', '"Suez Canal"', '"Panama Canal"',
               '"port congestion"', '"Strait of Malacca"']
REGION_RULES = [
    ("bab_el_mandeb", "Red Sea / Bab-el-Mandeb", r"red sea|bab[ -]?el[ -]?mandeb|houthi|yemen|gulf of aden|aden"),
    ("suez", "Suez Canal", r"suez"),
    ("hormuz", "Strait of Hormuz", r"hormuz|persian gulf|gulf of oman|iran"),
    ("panama", "Panama Canal", r"panama"),
    ("malacca", "Strait of Malacca", r"malacca|singapore strait"),
    ("cape", "Cape of Good Hope", r"cape of good hope|cape route"),
    ("ports", "Port congestion", r"congestion|port delay|backlog|berth"),
]
REGION_CHOKEPOINT = {"bab_el_mandeb": "Bab-el-Mandeb", "suez": "Suez Canal", "hormuz": "Strait of Hormuz",
                     "panama": "Panama Canal", "malacca": "Strait of Malacca", "cape": "Cape of Good Hope"}
ADVISORY_RE = re.compile(r"\b(clos(ed|ure|es)|shut|attack(s|ed)?|strike|struck|seiz(e|ed|ure)|blockade|suspend(s|ed)?|"
                         r"halt(s|ed)?|pause[sd]?|divert(s|ed)?|reroute[sd]?|missile|drone|threat)\b", re.I)
REGION_SCENARIO = {"bab_el_mandeb": "bab_el_mandeb_closure", "suez": "suez_closure", "hormuz": "hormuz_closure",
                   "panama": "panama_closure"}


def classify(title: str) -> list[tuple[str, str]]:
    t = title.lower()
    return [(k, label) for k, label, rx in REGION_RULES if re.search(rx, t)]


def _article(title: str, url: str, source: str, published: datetime | None, provider: str) -> dict:
    regions = classify(title)
    return {"title": title.strip(), "url": url, "source": source, "published": published, "provider": provider,
            "regions": [k for k, _ in regions], "region_labels": [lab for _, lab in regions] or ["General shipping"],
            "advisory": bool(ADVISORY_RE.search(title))}


def fetch_news_gdelt(max_records: int = 30) -> list[dict]:
    q = "(" + " OR ".join(TOPIC_TERMS) + ") (shipping OR vessel OR tanker OR \"container ship\") sourcelang:english"
    r = _get(GDELT_URL, {"query": q, "mode": "ArtList", "format": "json", "maxrecords": max_records,
                         "timespan": "7d", "sort": "DateDesc"}, retries=0, timeout=4)
    try:
        data = r.json()
    except ValueError:
        raise IntelError("GDELT returned non-JSON (often a rate-limit notice)") from None
    out = []
    for a in data.get("articles", []):
        try:
            pub = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pub = None
        if a.get("title") and a.get("url"):
            out.append(_article(a["title"], a["url"], a.get("domain", ""), pub, "GDELT DOC 2.0"))
    return out


def fetch_news_rss() -> list[dict]:
    q = "(" + " OR ".join(TOPIC_TERMS) + ") (shipping OR vessel OR tanker) when:7d"
    r = _get(RSS_URL, {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        raise IntelError("news RSS could not be parsed") from None
    out = []
    for it in root.findall("./channel/item"):
        title, link = it.findtext("title") or "", it.findtext("link") or ""
        src_el = it.find("source")
        source = (src_el.text or "") if src_el is not None else ""
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)]
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            pub = None
        if title and link:
            out.append(_article(title, link, source, pub, "Google News RSS"))
    return out


_DOWN_UNTIL: dict[str, float] = {}      # provider health (not user state): skip a failing provider for a while
PROVIDER_COOLDOWN_S = 1800


def fetch_news() -> dict:
    """Try GDELT, then Google News RSS. Raises IntelError if both fail."""
    errors = []
    for name, fn in (("GDELT", fetch_news_gdelt), ("Google News RSS", fetch_news_rss)):
        if time.time() < _DOWN_UNTIL.get(name, 0):
            errors.append(f"{name}: skipped after a recent failure")
            continue
        try:
            arts = fn()
        except IntelError as e:
            errors.append(f"{name}: {e}")
            if name == "GDELT":
                _DOWN_UNTIL[name] = time.time() + PROVIDER_COOLDOWN_S
            continue
        if arts:
            return {"articles": arts, "provider": name, "errors": errors,
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        errors.append(f"{name}: no results")
    raise IntelError("; ".join(errors))


def rank_news(articles: list[dict], route_chokepoints: list[str], limit: int = 8) -> list[dict]:
    """Dedupe, tag relevance to the selected voyage, and order: on-route first, then newest."""
    seen, out = set(), []
    for a in articles:
        key = re.sub(r"\W+", " ", a["title"].lower())[:70]
        if key in seen:
            continue
        seen.add(key)
        on_route = [REGION_CHOKEPOINT[r] for r in a["regions"] if REGION_CHOKEPOINT.get(r) in route_chokepoints]
        out.append(dict(a, on_route=on_route))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    out.sort(key=lambda a: (bool(a["on_route"]), bool(a["regions"]), a["published"] or epoch), reverse=True)
    return out[:limit]


def advisories(ranked: list[dict]) -> list[dict]:
    """Advisory alerts: on-route headlines with disruption language. Suggests (never applies) a scenario."""
    out, seen = [], set()
    for a in ranked:
        if not (a["advisory"] and a["on_route"]):
            continue
        for r in a["regions"]:
            if r in REGION_SCENARIO and r not in seen and REGION_CHOKEPOINT.get(r) in a["on_route"]:
                seen.add(r)
                out.append({"region": r, "label": REGION_CHOKEPOINT[r], "scenario": REGION_SCENARIO[r],
                            "headline": a["title"], "source": a["source"]})
    return out


DEMO_ARTICLES = [
    ("[DEMO] Sample headline: shipping lines weigh Red Sea transits after new security advisory", ["bab_el_mandeb"]),
    ("[DEMO] Sample headline: Suez Canal transit queue lengthens after grounding drill", ["suez"]),
    ("[DEMO] Sample headline: tanker traffic slows near the Strait of Hormuz", ["hormuz"]),
    ("[DEMO] Sample headline: Panama Canal draft limits eased as lake levels recover", ["panama"]),
    ("[DEMO] Sample headline: container terminal congestion builds at Shanghai", ["ports"]),
]


def demo_news() -> dict:
    """Clearly labelled sample items for offline demos. Not real articles: no URLs, no dates, no sources."""
    arts = []
    for title, regions in DEMO_ARTICLES:
        a = _article(title, "", "Demo data - not a real source", None, "Demo mode")
        a["regions"] = regions
        a["region_labels"] = [lab for k, lab, _ in REGION_RULES if k in regions]
        arts.append(a)
    return {"articles": arts, "provider": "Demo mode (sample data, not real news)", "errors": [], "fetched_at": None,
            "demo": True}


def news_summary(news: dict | None, route_chokepoints: list[str], region: str | None = None) -> str:
    if not news or not news.get("articles"):
        err = (news or {}).get("error", "")
        return "Live maritime news is unavailable" + (f" ({err})" if err else "") + ". No headlines are invented."
    ranked = rank_news(news["articles"], route_chokepoints, limit=20)
    if region:
        ranked = [a for a in ranked if region in a["regions"]] or []
    if not ranked:
        return f"No recent headlines matched that region (source: {news['provider']})."
    lines = []
    for a in ranked[:5]:
        when = f"{a['published']:%b %d}" if a["published"] else "undated"
        lines.append(f"• {a['title']} ({a['source'] or 'unknown source'}, {when})"
                     + (" [on your route]" if a["on_route"] else ""))
    head = "DEMO sample headlines (not real):" if news.get("demo") else f"Recent headlines via {news['provider']}:"
    return (head + "\n" + "\n".join(lines) + "\nHeadlines are advisory only: no corridor has been closed in the "
            "simulator because of them. Ask me to simulate a closure if you want to test the impact.")
