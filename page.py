"""
page.py - shared context for the Streamlit page.

``Page`` bundles the per-run objects (session state, the ``Ops`` controller, the cached model and graph) with the
widget callbacks and the widget <-> controller sync. The tab renderers in ``tabs.py`` take one ``Page`` instead of a
dozen closures. Every callback goes through ``Ops`` (the only code that mutates simulation state).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import copilot
import economics as ECO
import intel
import panels as P
import voyage as V
from agents import VESSELS
from data_engine import build_maritime_graph
from events import EVENT_CATALOG
from llm import get_provider
from ml_models import HULL, load_or_train_model
from ops import SPEEDS, nearest_stop_label


# ------------------------------------------------------------------------------------------------------
# Cached, read-only resources (shared across sessions; never mutated)
# ------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training fuel model on 5,000 synthetic voyages...")
def get_model():
    return load_or_train_model()


@st.cache_resource
def get_base_graph():
    return build_maritime_graph()


@st.cache_resource
def get_llm():
    return get_provider(st.secrets)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_marine(coords: tuple) -> dict:
    return intel.fetch_marine(coords)          # raises IntelError -> not cached


@st.cache_data(ttl=900, show_spinner=False)
def cached_news() -> dict:
    return intel.fetch_news()


def show_df(df: pd.DataFrame, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def md(text: str) -> str:
    """Escape characters Streamlit markdown would otherwise interpret ($ starts LaTeX)."""
    return text.replace("$", "\\$")


class Page:
    def __init__(self, ops, model, G0, ports: dict, llm, llm_label: str):
        self.ss = st.session_state
        self.ops, self.model, self.G0 = ops, model, G0
        self.ports = ports
        self.port_ids = list(ports)
        self.llm, self.llm_label = llm, llm_label

    # ---- widget <-> controller sync: widget state is rewritten from the controller on every run (before the
    # widgets are created), so a change made by the copilot or the demo shows up in the controls immediately.
    def sync_widgets(self) -> None:
        ss, ops = self.ss, self.ops
        c = ops.cfg
        ss["cfg_origin"], ss["cfg_dest"] = c.origin, c.destination
        ss["cfg_vessel"], ss["cfg_owner"] = c.vessel_key, c.owner_key
        ss["cfg_fixed_speed"] = c.cruise_speed_kn is not None
        lo, hi = HULL[VESSELS[c.vessel_key].vessel_type]["speed"]
        ss["cfg_speed"] = float(min(max(c.cruise_speed_kn or VESSELS[c.vessel_key].speed_options_kn[1], lo), hi))
        ss["cfg_fuel"] = float(c.fuel_price)
        ss["cfg_pref"] = c.preference
        e = c.econ
        ss["cfg_preset"] = ECO.preset_key(c.preference, e)
        ss["cfg_tvm"], ss["cfg_carbon"] = float(e.time_value_mult), float(e.carbon_usd_per_t)
        ss["cfg_ets_on"], ss["cfg_eua"], ss["cfg_etsc"] = e.ets_enabled, float(e.eua_usd_per_t), float(e.ets_coverage_pct)
        ss["cfg_fuelmode"] = e.fuel_price_mode
        ss["cfg_fuel_ovr"] = float(dict(e.fuel_overrides).get(c.origin, ECO.effective_fuel_price(
            c.fuel_price, c.origin, ECO.Economics(fuel_price_mode="origin_port"))))
        ss["cfg_w_time"], ss["cfg_w_cost"], ss["cfg_w_risk"] = (float(w) for w in c.weights)
        ss["cfg_avoid"] = list(c.avoid)
        ss["cfg_port_delay"], ss["cfg_canal_delay"] = float(c.port_delay_hours), float(c.canal_delay_hours)
        ss["cfg_dep_date"], ss["cfg_dep_time"] = c.departure.date(), c.departure.time().replace(tzinfo=None)
        for spec in EVENT_CATALOG:
            ss[f"inc_{spec.key}"] = ops.settings[spec.key]
        ss["inc_hurricane_track"] = ops.settings["hurricane_track"]
        ss["inc_picker"] = P.selected_incidents(ops.settings)
        ss["play_speed"] = nearest_stop_label(ops.speed)

    def show_toasts(self) -> None:
        for msg in self.ss.pop("_flash", []):
            st.toast(md(msg[:300]))

    def flash(self, msg: str) -> None:
        self.ss.setdefault("_flash", []).append(msg)

    def cb(self, fn, *a, **kw):
        """Wrap a controller call as a widget callback; its message becomes a toast."""
        def _run():
            try:
                out = fn(*a, **kw)
            except Exception as e:                  # never let a callback crash the page
                out = f"Action failed: {e}"
            if isinstance(out, str) and out:
                self.flash(out)
        return _run

    # ---- widget callbacks
    def on_voyage(self) -> None:
        ss = self.ss
        self.flash(self.ops.create_voyage(
            ss["cfg_origin"], ss["cfg_dest"], ss["cfg_vessel"],
            ss["cfg_owner"] if ss["cfg_owner"] in V.VESSEL_OWNERS[ss["cfg_vessel"]] else None))

    def on_cfg(self, field: str) -> None:
        ss = self.ss
        val = {
            "cruise_speed_kn": (float(ss["cfg_speed"]) if ss["cfg_fixed_speed"] else None),
            "fuel_price": float(ss["cfg_fuel"]),
            "preference": ss["cfg_pref"],
            "weights": (float(ss["cfg_w_time"]), float(ss["cfg_w_cost"]), float(ss["cfg_w_risk"])),
            "avoid": tuple(ss["cfg_avoid"]),
            "port_delay_hours": float(ss["cfg_port_delay"]),
            "canal_delay_hours": float(ss["cfg_canal_delay"]),
            "departure": datetime.combine(ss["cfg_dep_date"], ss["cfg_dep_time"], tzinfo=timezone.utc),
        }[field]
        self.flash(self.ops.update_config(**{field: val}))

    def on_preset(self) -> None:
        if self.ss["cfg_preset"] in ECO.PRESETS:     # "custom" is a read-out of moved sliders, not a choice
            self.flash(self.ops.apply_preset(self.ss["cfg_preset"]))

    def on_econ(self, field: str) -> None:
        ss, ops = self.ss, self.ops
        if field == "fuel_overrides":
            others = tuple((p, x) for p, x in ops.cfg.econ.fuel_overrides if p != ops.cfg.origin)
            val = others + ((ops.cfg.origin, float(ss["cfg_fuel_ovr"])),)
        else:
            val = {"time_value_mult": float(ss["cfg_tvm"]), "carbon_usd_per_t": float(ss["cfg_carbon"]),
                   "ets_enabled": bool(ss["cfg_ets_on"]), "eua_usd_per_t": float(ss["cfg_eua"]),
                   "ets_coverage_pct": float(ss["cfg_etsc"]), "fuel_price_mode": ss["cfg_fuelmode"]}[field]
        self.flash(ops.set_economics(**{field: val}))

    def on_incident(self, key: str) -> None:
        self.flash(self.ops.set_event(key, self.ss[f"inc_{key}"]))

    def on_incident_picker(self) -> None:
        for key, on in P.incident_changes(self.ss["inc_picker"], self.ops.settings):
            self.flash(self.ops.set_event(key, on))

    def on_speed(self) -> None:
        self.ops.set_speed(SPEEDS[self.ss["play_speed"]])

    def on_seek(self) -> None:
        self.ops.seek(float(self.ss["seek_h"]))

    def on_chat(self, prompt: str | None = None) -> None:
        ops = self.ops
        msg = (prompt or self.ss.get("chat_in") or "").strip()
        if not msg:
            return
        ops.chat.append({"role": "user", "content": msg})
        try:
            reply = copilot.handle(msg, ops, self.llm, self.intel_ctx())
            ops.chat.append({"role": "assistant", "content": reply.text, "mode": reply.mode,
                             "ops": [(n, a) for n, a, _ in reply.ops], "rejected": [r for _, r in reply.rejected]})
        except Exception as e:
            ops.chat.append({"role": "assistant", "content": f"Sorry, that command failed: {e}", "ops": [],
                             "mode": "command", "rejected": []})

    # ---- read-only helpers
    def current_weather(self):
        return intel.current_for(self.ss.get("weather"), V.route_signature(self.ops.voyage))

    def route_chokepoints(self) -> list[str]:
        ctx = V.build_context(self.ops.cfg, self.ops.settings, self.model)
        _, ahead, _ = V.split_at(self.ops.voyage, ctx.G)
        sail = [leg for leg in ahead if leg.kind == "sail"]
        return V.summarise(ctx, sail).chokepoints if sail else []

    def intel_ctx(self) -> dict:
        route = self.route_chokepoints()
        return {"weather_summary": intel.weather_summary(self.current_weather()),
                "news_summary": lambda region: intel.news_summary(self.ss.get("news"), route, region)}
