"""
intel_ui.py - Weather and News & Security panels (Live Maritime Intelligence).

Network calls are cached, back off on failure and never run on animation frames. Headlines are advisory: nothing here
closes a corridor or changes the risk score.
"""
from __future__ import annotations

import html
import time

import streamlit as st

import intel
import voyage as V


def render_weather(ss, ops, cached_marine, cb, flash):
    G = ops.G
    v = ops.voyage
    now = time.time()
    f1, f2 = st.columns([1.5, 1])
    fi_on = f1.toggle("Forecast-informed simulation", key="fi_mode",
                      help="Opt-in: Open-Meteo hazards ahead of the ship (waves ≥ threshold at the simulated ETA) "
                           "become labelled SIMULATED storm cells and the remaining route is re-assessed. "
                           "Not a navigation recommendation.")
    thr = f2.number_input("Hazard ≥ (m)", 2.0, 12.0, 4.0, 0.5, key="fi_threshold",
                          help="Significant wave height that counts as a hazard.")
    if v.status == "arrived" or not v.legs:
        st.caption("No remaining route to forecast.")
        return
    sig = V.route_signature(v)
    pts = V.remaining_samples(v, G)
    coords = tuple((round(p["lat"], 1), round(p["lon"], 1)) for p in pts)
    if now < ss.get("_wx_backoff", 0):
        w = intel.reuse_or_unavailable(ss.get("_wx_last"), sig, ss.get("_wx_err", "provider unavailable"))
    else:
        try:
            with st.spinner("Fetching Open-Meteo marine forecast…"):
                raw = cached_marine(coords)
            w = intel.with_route(intel.assess_weather(raw, pts, ops.cfg.departure), sig, coords)
            ss["_wx_last"] = w
        except intel.IntelError as e:
            ss["_wx_backoff"], ss["_wx_err"] = now + 120, str(e)
            w = intel.reuse_or_unavailable(ss.get("_wx_last"), sig, str(e))
    if w.get("status") != "unavailable":
        w = dict(w, route_sig=sig)
    ss["weather"] = w
    if fi_on:
        msg = ops.apply_forecast_hazards(w, thr)
        if msg:
            flash(msg)
            st.rerun()                               # re-plan visible everywhere (converges: cells only accumulate)
    status = w["status"] + (" · STALE" if w.get("stale") else "")
    st.markdown(f'<div class="intel-meta"><span class="tag-live">LIVE DATA</span> Status: <b>{html.escape(status)}</b> · '
                f'{html.escape(w["source"])}' + (f' · fetched {html.escape(w["fetched_at"])} UTC' if w.get("fetched_at")
                                                 else "") + "</div>", unsafe_allow_html=True)
    if w.get("note"):
        st.caption(w["note"])
    if not w["points"]:
        st.caption("The simulator keeps running on its synthetic sea-state model.")
        return
    rows = []
    for p in w["points"]:
        vals = p["values"]
        lvl = p["level"]
        sev = f'<span class="sev l{lvl if lvl >= 0 else "n"}">{html.escape(p["severity"])}</span>'
        if p["status"] == "forecast":
            detail = (f"waves {vals.get('wave_height')} m · swell {vals.get('swell_wave_height')} m · "
                      f"period {vals.get('wave_period')} s"
                      + (f" · wind {vals.get('wind_speed_10m')} kn" if vals.get("wind_speed_10m") is not None else "")
                      + (f" · current {vals.get('ocean_current_velocity')} km/h" if vals.get("ocean_current_velocity") is not None else "")
                      + (f" · SST {vals.get('sea_surface_temperature')} °C" if vals.get("sea_surface_temperature") is not None else ""))
            if (vals.get("wave_height") or 0) >= thr:
                detail += f" · HAZARD (≥ {thr:g} m): " + intel.operational_impact(vals.get("wave_height"),
                                                                                  vals.get("wind_speed_10m"))
        else:
            detail = p["status"]
        rows.append(f'<div class="intel-row"><b>{html.escape(p["label"])}</b> {sev}<br>{html.escape(detail)}'
                    f'<div class="intel-meta">simulated ETA {p["eta"]:%b %d %H:%M} UTC · {p["lat"]:.1f}, {p["lon"]:.1f}'
                    + (f' · {html.escape(p["action"])}' if p["level"] >= 1 else "") + "</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    worst = w.get("worst")
    if worst and worst["level"] >= 2 and not fi_on:
        st.button(f"Simulate storm at worst point ({worst['values'].get('wave_height')} m)", key="wx_storm",
                  on_click=cb(ops.add_storm, max(float(worst["values"].get("wave_height") or 5.0), 5.0),
                              worst["lat"], worst["lon"], 250.0, f"forecast {worst['label'].lower()}",
                              "manual", "open-meteo"), width="stretch")
    if any(c.get("source") == ops.FORECAST_SOURCE for c in ops.settings["storm_cells"]):
        st.button("Remove forecast-informed storm cells", key="fi_clear", on_click=cb(ops.clear_forecast_hazards),
                  width="stretch")
    st.caption("Forecasts are matched to the ship's SIMULATED arrival time at each point. Data: Open-Meteo (CC BY 4.0). "
               "Advisory unless forecast-informed simulation is on; even then the result is a modelled scenario.")


def render_news(ss, ops, cached_news, cb, route_fn):
    now = time.time()
    c1, c2 = st.columns([1, 1])
    demo = c2.toggle("Demo headlines", key="news_demo", help="Clearly labelled sample items for offline demos.")
    if c1.button("Refresh news", key="news_refresh", width="stretch",
                 disabled=now - ss.get("_news_refreshed", 0) < 15):
        ss["_news_refreshed"] = now
        ss["_news_backoff"] = 0
        cached_news.clear()
    if demo:
        news = intel.demo_news()
    elif now < ss.get("_news_backoff", 0):
        news = ss.get("_news_last") or {"articles": [], "error": ss.get("_news_err", "unavailable")}
        if ss.get("_news_last"):
            news = dict(news, stale=True)
    else:
        try:
            with st.spinner("Fetching maritime headlines…"):
                news = cached_news()
            ss["_news_last"] = news
        except intel.IntelError as e:
            ss["_news_backoff"], ss["_news_err"] = now + 180, str(e)
            news = dict(ss["_news_last"], stale=True) if ss.get("_news_last") else {"articles": [], "error": str(e)}
    ss["news"] = news
    if not news.get("articles"):
        st.warning(f"Live news unavailable: {news.get('error', 'no results')}. No headlines are shown rather than "
                   "invented. Use the demo toggle for a labelled sample.")
        return
    meta = (f"Source: {news['provider']}" + (f" · fetched {news['fetched_at']} UTC" if news.get("fetched_at") else "")
            + (" · STALE (refresh failed)" if news.get("stale") else ""))
    if news.get("errors"):
        meta += " · fallback used (" + "; ".join(news["errors"])[:120] + ")"
    st.markdown(f'<div class="intel-meta">{html.escape(meta)}</div>', unsafe_allow_html=True)
    route = route_fn()
    ranked = intel.rank_news(news["articles"], route)
    rows = []
    for a in ranked:
        when = f"{a['published']:%b %d %H:%M} UTC" if a["published"] else "no date (demo)"
        title = html.escape(a["title"])
        link = f'<a href="{html.escape(a["url"])}" target="_blank" rel="noopener">{title}</a>' if a["url"] else title
        tags = " · ".join(a["region_labels"])
        flag = ' <span class="sev l2">ON ROUTE</span>' if a["on_route"] else ""
        rows.append(f'<div class="intel-row">{link}{flag}<div class="intel-meta">{html.escape(a["source"] or "unknown")}'
                    f' · {when} · {html.escape(tags)}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    if not news.get("demo"):
        for adv in intel.advisories(ranked):
            already = ops.settings.get({"suez_closure": "suez_blockage", "hormuz_closure": "hormuz_blockade"}.get(
                adv["scenario"], adv["scenario"]))
            st.button(f"Simulate {adv['label']} closure (scenario)", key=f"adv_{adv['region']}", disabled=bool(already),
                      on_click=cb(ops.set_disruption, adv["scenario"], True), width="stretch")
    st.caption("Headlines are advisory. A headline is not evidence that a corridor is closed, so nothing is closed "
               "automatically; buttons above only run a labelled simulation scenario.")
