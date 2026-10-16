"""
ops.py - the operations controller: ONE place that changes the simulation.

Dashboard widgets, playback buttons and the AI copilot all call these methods, so a command typed in the chat
runs exactly the same code as the equivalent click. An ``Ops`` instance lives in ``st.session_state`` (one per
browser session), never in a module-level or cached object, so users never share mutable voyage state.

Every method returns a short human-readable result string; the voyage timeline records what happened.
Disruptions are SCENARIOS set by the user or the copilot, never claims about the real world.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace

import voyage as V
from data_engine import CHOKEPOINTS, NODE_DEFS
from events import EVENT_BY_KEY, STORM_AVOID_WAVE_M, default_settings, describe

PORTS = {k: v[0] for k, v in NODE_DEFS.items() if v[3] == "port"}

# Copilot / incident-panel disruption names -> (settings key, value when switched on, label)
DISRUPTIONS = {
    "suez_closure": ("suez_blockage", True, "Suez Canal closure"),
    "bab_el_mandeb_closure": ("bab_el_mandeb_closure", True, "Bab-el-Mandeb / Red Sea closure"),
    "hormuz_closure": ("hormuz_blockade", True, "Strait of Hormuz closure"),
    "panama_closure": ("panama_closure", True, "Panama Canal closure"),
    "red_sea_escalation": ("red_sea_escalation", 0.8, "Red Sea security escalation"),
    "taiwan_tension": ("taiwan_tension", 0.6, "Taiwan Strait tension"),
    "suez_fee": ("suez_toll_multiplier", 2.0, "Suez Canal fee increase"),
    "panama_fee": ("panama_toll_multiplier", 2.0, "Panama Canal fee increase"),
    "panama_drought": ("panama_drought", True, "Panama Canal drought"),
}
# Corridors the copilot can route "via" -> graph node
VIA_NODES = {"cape": "CAPE", "suez": "PSD", "panama": "COL", "horn": "CHN", "malacca": "MAL1",
             "bab_el_mandeb": "BAB", "hormuz": "HOR"}
SPEEDS = {"1 h/s": 1.0, "10 h/s": 10.0, "50 h/s": 50.0, "100 h/s": 100.0}


class Ops:
    def __init__(self, model=None):
        self.model = model
        self.cfg = V.VoyageConfig()
        self.settings = default_settings()
        self.voyage = V.plan_voyage(self.cfg, self.settings, model)
        self.playing = False
        self.speed = 10.0                 # simulated hours per real second
        self._last_tick: float | None = None
        self.chat: list[dict] = []
        self.version = 0                  # bumps whenever the route/itinerary changes (map redraw key)
        self.demo: dict | None = None     # hackathon demo results (before/after snapshot)
        self._demo_target: float | None = None   # animated demo: voyage hour at which the closure triggers
        self._demo_source = "demo"
        self.last_change: dict | None = None  # {cause, before, after} of the latest re-evaluation that moved the route
        self._pre_demo: tuple | None = None   # (cfg, settings, voyage, playing) to restore on exit

    # ---------------------------------------------------------------- helpers
    @property
    def G(self):
        return V.base_graph()

    def state(self) -> V.ShipState:
        return V.ship_state(self.voyage, self.G)

    def started(self) -> bool:
        return self.voyage.t > 0 or self.voyage.status in ("underway", "arrived")

    def _touch(self) -> None:
        self.version += 1

    def _reevaluate(self, cause: str, source: str = "manual", **kw) -> str:
        """V.reevaluate plus a before/after snapshot whenever the remaining route or voyage status changes."""
        before = self.snapshot() if self.voyage.legs else None
        sig, status = V.route_signature(self.voyage), self.voyage.status
        msg = V.reevaluate(self.voyage, self.settings, self.model, cause=cause, source=source, **kw)
        if before is not None and (V.route_signature(self.voyage) != sig or self.voyage.status != status):
            self.last_change = {"cause": cause, "before": before, "after": self.snapshot(), "message": msg}
        return msg

    def port_name(self, pid: str) -> str:
        return PORTS.get(pid, pid)

    # ---------------------------------------------------------------- voyage set-up
    def create_voyage(self, origin: str, destination: str, vessel_key: str | None = None,
                      owner_key: str | None = None, source: str = "manual") -> str:
        if origin not in PORTS or destination not in PORTS:
            return f"Unknown port. Available: {', '.join(PORTS.values())}."
        if origin == destination:
            return "Origin and destination must differ."
        changes = {"origin": origin, "destination": destination}
        if vessel_key:
            changes["vessel_key"] = vessel_key
            changes["owner_key"] = owner_key or V.VESSEL_OWNERS[vessel_key][0]
        elif owner_key:
            changes["owner_key"] = owner_key
        self.cfg = replace(self.cfg, **changes)
        if self.cfg.owner_key not in V.VESSEL_OWNERS[self.cfg.vessel_key]:
            self.cfg = replace(self.cfg, owner_key=V.VESSEL_OWNERS[self.cfg.vessel_key][0])
        self.voyage = V.plan_voyage(self.cfg, self.settings, self.model)
        self.demo = self.last_change = self._demo_target = None
        if source != "manual":
            self.voyage.timeline[-1]["source"] = source
        self.playing = False
        self._touch()
        v = self.voyage
        if v.status == "blocked":
            return f"Voyage {self.port_name(origin)} → {self.port_name(destination)} has no feasible route: {v.block_reason}"
        return v.timeline[-1]["title"] + ". " + v.timeline[-1]["detail"]

    def update_config(self, source: str = "manual", **changes) -> str:
        """Change voyage assumptions. Before departure the voyage is re-planned from the origin; underway, the
        rest of the voyage is re-optimised from the ship's current position (origin/destination/vessel are locked)."""
        locked = {"origin", "destination", "vessel_key", "owner_key", "departure"}
        if self.started() and locked & changes.keys():
            return "Origin, destination, vessel and departure are locked once the voyage is underway. Reset first."
        prev = self.cfg
        self.cfg = replace(self.cfg, **changes)
        if self.cfg.owner_key not in V.VESSEL_OWNERS[self.cfg.vessel_key]:
            self.cfg = replace(self.cfg, owner_key=V.VESSEL_OWNERS[self.cfg.vessel_key][0])
        if self.cfg == prev:
            return "No change."
        what = ", ".join(f"{k}={v}" for k, v in changes.items() if k != "weights") or "weights"
        if not self.started():
            timeline = self.voyage.timeline
            self.voyage = V.plan_voyage(self.cfg, self.settings, self.model)
            self.voyage.timeline = timeline + self.voyage.timeline
            self._touch()
            return f"Assumptions updated ({what}); voyage re-planned. " + self.voyage.timeline[-1]["detail"]
        self.voyage.config = self.cfg
        msg = self._reevaluate(cause=f"Assumption change: {what}",
                           prev_config=prev, source=source)
        self._touch()
        return msg

    # ---------------------------------------------------------------- playback
    def start(self) -> str:
        v = self.voyage
        if v.status == "blocked":
            return f"Cannot start: voyage is blocked. {v.block_reason}"
        if v.status == "arrived":
            self.reset()
            v = self.voyage
        if v.status == "planned":
            v.status = "underway"
            V.log(v, "departure", f"Departed {self.port_name(self.cfg.origin)}",
                  "Simulated live tracking started.")
        self.playing = True
        self._last_tick = None
        return f"Voyage started: {self.port_name(self.cfg.origin)} → {self.port_name(self.cfg.destination)}."

    def pause(self) -> str:
        self.playing = False
        self._last_tick = None
        return f"Paused at voyage hour {self.voyage.t:,.1f} (day {self.voyage.t / 24:.1f})."

    def resume(self) -> str:
        if self.voyage.status in ("arrived", "blocked"):
            return f"Cannot resume: voyage is {self.voyage.status}."
        return self.start()

    def reset(self) -> str:
        self.demo = self.last_change = self._demo_target = None  # before/after panels no longer describe this voyage
        self.voyage = V.plan_voyage(self.cfg, self.settings, self.model)
        self.playing = False
        self._last_tick = None
        self._touch()
        return "Voyage reset to departure with the current scenarios and assumptions."

    def set_speed(self, sim_hours_per_s: float) -> None:
        self.speed = float(sim_hours_per_s)

    def tick(self, now: float | None = None) -> bool:
        """Advance by real elapsed time x playback speed. Returns True if the ship arrived during this tick."""
        if not self.playing:
            return False
        now = time.monotonic() if now is None else now
        if self._last_tick is None:
            self._last_tick = now
            return False
        dt = min(max(now - self._last_tick, 0.0), 3.0)        # cap catch-up after a stall
        self._last_tick = now
        step = dt * self.speed
        if self._demo_target is not None:                    # animated demo: stop exactly at the approach point
            step = min(step, max(self._demo_target - self.voyage.t, 0.0))
        arrived = V.advance(self.voyage, step)
        if self._demo_target is not None and self.voyage.t >= self._demo_target - 1e-9:
            self._demo_trigger(self._demo_source)
            return False
        if arrived or self.voyage.status in ("arrived", "blocked"):
            self.playing = False
        return arrived

    def advance_hours(self, hours: float, source: str = "manual") -> str:
        v = self.voyage
        if v.status == "blocked":
            return f"Ship is holding position: voyage blocked. {v.block_reason}"
        if v.status == "arrived":
            return "Ship has already arrived."
        if v.status == "planned":
            V.log(v, "departure", f"Departed {self.port_name(self.cfg.origin)}", "Simulated clock advanced.", source)
        if self._demo_target is not None and v.t + hours >= self._demo_target:
            V.advance(v, self._demo_target - v.t)
            return self._demo_trigger(self._demo_source)
        V.advance(v, float(hours))
        s = self.state()
        return (f"Advanced {hours / 24:.1f} days → voyage day {v.t / 24:.1f}. Ship at ({s.lat:.2f}, {s.lon:.2f}), "
                f"{s.traveled_nm:,.0f} nm sailed, {s.remaining_nm:,.0f} nm to go"
                + (" - ARRIVED." if v.status == "arrived" else "."))

    def seek(self, t_hours: float) -> str:
        self._demo_target = None             # manual scrubbing cancels a pending animated-demo trigger
        V.seek(self.voyage, t_hours)
        return f"Moved to voyage hour {self.voyage.t:,.1f}."

    # ---------------------------------------------------------------- disruptions (scenarios)
    def set_event(self, key: str, value, source: str = "manual", force: bool = False, label: str | None = None) -> str:
        """Set one settings key (a scenario) and re-evaluate the voyage from the ship's current position.
        The change is atomic: if re-evaluation fails, the previous scenario settings are restored."""
        if key not in self.settings:
            return f"Unknown scenario '{key}'."
        if self.settings[key] == value and not force:
            return "Scenario unchanged."
        if label is None:
            spec = EVENT_BY_KEY.get(key)
            name = spec.label if spec else key
            if isinstance(value, bool):
                label = f"{name} {'ON' if value else 'OFF'}"
            elif isinstance(value, (int, float)):
                label = f"{name} = {value:g}"
            else:
                label = f"{name} updated"
        cause = f"SCENARIO (simulated): {label}"
        prev_settings, prev_voyage = self.settings, copy.deepcopy(self.voyage)
        self.settings = copy.deepcopy(self.settings)
        self.settings[key] = value
        try:
            msg = self._reevaluate(cause=cause, force=force, source=source)
        except Exception:
            self.settings, self.voyage = prev_settings, prev_voyage
            raise
        self._touch()
        return f"{cause}. {msg}"

    def set_disruption(self, name: str, active: bool = True, value: float | None = None, source: str = "manual",
                       force: bool = False) -> str:
        if name not in DISRUPTIONS:
            return f"Unknown disruption '{name}'."
        key, on_value, _ = DISRUPTIONS[name]
        default = EVENT_BY_KEY[key].default
        val = (value if value is not None else on_value) if active else default
        if isinstance(default, bool):
            val = bool(val)
        return self.set_event(key, val, source, force)

    def add_storm(self, wave_m: float = 6.0, lat: float | None = None, lon: float | None = None,
                  radius_nm: float = 300.0, label: str = "", source: str = "manual", origin: str = "scenario") -> str:
        """Place a storm cell. Without coordinates it is placed on the remaining route ahead of the ship."""
        if lat is None or lon is None:
            pt = self._point_ahead()
            if pt is None:
                return "No remaining route to place a storm on."
            lat, lon, where = pt
            label = label or f"ahead near {where}"
        cell = {"lat": float(lat), "lon": float(lon), "radius_nm": float(radius_nm), "wave_m": float(wave_m),
                "label": label or f"at {lat:.1f}, {lon:.1f}", "source": origin}
        cells = list(self.settings["storm_cells"]) + [cell]
        avoid = " (core avoided if an alternative exists)" if wave_m >= STORM_AVOID_WAVE_M else ""
        return self.set_event("storm_cells", cells, source,
                              label=f"storm {cell['label']}, {wave_m:g} m waves, radius {radius_nm:g} nm{avoid}")

    def clear_storms(self, source: str = "manual") -> str:
        if not self.settings["storm_cells"]:
            return "No storms to clear."
        return self.set_event("storm_cells", [], source, label="all storm cells cleared")

    def set_congestion(self, port: str, hours: float, source: str = "manual") -> str:
        if port not in PORTS:
            return "Unknown port."
        cong = dict(self.settings["port_congestion"])
        if hours > 0:
            cong[port] = float(hours)
        else:
            cong.pop(port, None)
        return self.set_event("port_congestion", cong, source,
                              label=f"port congestion at {self.port_name(port)} (+{hours:g} h berth wait)")

    def apply_settings(self, new: dict, label: str, source: str = "manual") -> str:
        """Replace all scenario settings at once (presets) with a single re-evaluation."""
        prev_settings, prev_voyage = self.settings, copy.deepcopy(self.voyage)
        self.settings = default_settings() | copy.deepcopy(new)
        try:
            msg = self._reevaluate(cause=f"SCENARIO (simulated): {label}",
                               source=source)
        except Exception:
            self.settings, self.voyage = prev_settings, prev_voyage
            raise
        self._touch()
        return f"SCENARIO (simulated): {label}. {msg}"

    def clear_all(self, source: str = "manual") -> str:
        fresh = default_settings()
        if fresh == self.settings:
            return "No active scenarios."
        old = self.settings
        self.settings = fresh
        msg = self._reevaluate(cause="All scenarios cleared", source=source)
        self._touch()
        return f"Cleared scenarios ({', '.join(describe(old))}). {msg}"

    def _point_ahead(self):
        """A point on the remaining route ~40% of the way to go (300-1500 nm ahead), avoiding canals."""
        G = self.G
        s = self.state()
        _, ahead, _ = V.split_at(self.voyage, G)
        want = min(max(0.4 * s.remaining_nm, 300.0), 1500.0)
        acc = 0.0
        for l in ahead:
            if l.kind == "hold" or l.nm <= 0:
                continue
            if acc + l.nm >= want and G.edges[l.x, l.y]["mode"] == "sea":
                p = (want - acc) / l.nm
                lat, lon = V.point_on_lane(G, l.x, l.y, l.f0 + (l.f1 - l.f0) * p)
                return lat, lon, G.nodes[l.y]["name"]
            acc += l.nm
        return None

    # ---------------------------------------------------------------- routing requests
    def reroute(self, preference: str | None = None, via: str | None = None, source: str = "manual") -> str:
        if preference and preference in V.PREFERENCES and preference != self.cfg.preference:
            self.cfg = replace(self.cfg, preference=preference)
            self.voyage.config = self.cfg
        node = VIA_NODES.get(via, via) if via else None
        msg = self._reevaluate(
                           cause=f"Reroute requested ({V.PREFERENCES[self.cfg.preference]}"
                                 + (f" via {self.G.nodes[node]['name']}" if node else "") + ")",
                           force=True, via=node, source=source)
        self._touch()
        return msg

    def options(self) -> list[V.Candidate]:
        """Feasible routes from the current position under each preference, without changing anything."""
        G = self.G
        ctx = V.build_context(self.cfg, self.settings, self.model)
        _, ahead, pos = V.split_at(self.voyage, G)
        out: list[V.Candidate] = []
        if ahead:
            hold = V.reprice(ctx, ahead)
            if hold.feasible:
                out.append(V.Candidate("Current route", hold, chosen=True))
        for pref, label in V.PREFERENCES.items():
            s = V.route_from(ctx, pos, self.cfg.destination, pref)
            if s.feasible and not any(V._same(s, c.summary) for c in out):
                out.append(V.Candidate(label, s))
        return out

    def compare_via(self, via: str) -> str:
        node = VIA_NODES.get(via, via)
        if node not in self.G:
            return "Unknown corridor."
        G = self.G
        ctx = V.build_context(self.cfg, self.settings, self.model)
        _, ahead, pos = V.split_at(self.voyage, G)
        cur = V.reprice(ctx, ahead) if ahead else None
        alt = V.route_from(ctx, pos, self.cfg.destination, self.cfg.preference, via=node)
        name = G.nodes[node]["name"]
        if not alt.feasible:
            return f"No feasible route via {name}: {alt.reason}"
        if cur is None or not cur.feasible:
            return (f"Via {name}: {alt.nm:,.0f} nm, {alt.hours / 24:.1f} d, est. ${alt.total_usd / 1e6:,.2f}M, "
                    f"risk {alt.risk_index:.0f}/100 (current route not available for comparison).")
        if V._same(alt, cur):
            return f"The current remaining route already goes via {name}: est. ${cur.total_usd / 1e6:,.2f}M."
        dc, dd, dr = alt.total_usd - cur.total_usd, (alt.hours - cur.hours) / 24, alt.risk_index - cur.risk_index
        more = "more expensive" if dc > 0 else "cheaper"
        return (f"Route via {name} ({alt.corridor}) is ${abs(dc) / 1e6:,.2f}M {more} than the current remaining route "
                f"({cur.corridor}): ${alt.total_usd / 1e6:,.2f}M vs ${cur.total_usd / 1e6:,.2f}M all-in, "
                f"{dd:+.1f} days ({alt.hours / 24:.1f} vs {cur.hours / 24:.1f}), {alt.nm - cur.nm:+,.0f} nm, "
                f"fuel {alt.fuel_t - cur.fuel_t:+,.0f} t, tolls ${(alt.toll_usd - cur.toll_usd) / 1e3:+,.0f}k, "
                f"risk {dr:+.0f} points. Modelled estimates on a synthetic network.")

    # ---------------------------------------------------------------- forecast-informed simulation (opt-in)
    FORECAST_SOURCE = "open-meteo-forecast"
    MAX_FORECAST_CELLS = 6

    def apply_forecast_hazards(self, assessment: dict, wave_threshold_m: float = 4.0,
                               radius_nm: float = 250.0) -> str | None:
        """Turn forecast hazards AHEAD of the ship (Open-Meteo waves >= threshold at the simulated ETA) into
        clearly labelled simulated storm cells and re-assess the remaining route from the ship's position.
        Cells accumulate (deduplicated, capped) so the router cannot oscillate between routes. Returns None
        when there is nothing new to apply."""
        if not assessment or assessment.get("status") == "unavailable" or assessment.get("stale"):
            return None
        hazards = [p for p in assessment.get("points", [])[1:]            # [0] is the ship itself: hazards AHEAD
                   if p["status"] == "forecast" and (p["values"].get("wave_height") or 0) >= wave_threshold_m]
        current = list(self.settings["storm_cells"])
        mine = [c for c in current if c.get("source") == self.FORECAST_SOURCE]
        seen = {(round(c["lat"], 1), round(c["lon"], 1)) for c in mine}
        new = []
        for p in hazards:
            k = (round(p["lat"], 1), round(p["lon"], 1))
            if k in seen or len(mine) + len(new) >= self.MAX_FORECAST_CELLS:
                continue
            seen.add(k)
            new.append({"lat": p["lat"], "lon": p["lon"], "radius_nm": radius_nm,
                        "wave_m": float(p["values"]["wave_height"]),
                        "label": f"forecast hazard {p['label'].lower()} (Open-Meteo, ETA {p['eta']:%b %d %H:%M} UTC)",
                        "source": self.FORECAST_SOURCE, "fetched_at": assessment.get("fetched_at")})
        if not new:
            return None
        return self.set_event(
            "storm_cells", current + new, source="forecast",
            label=f"FORECAST-INFORMED SCENARIO: {len(new)} Open-Meteo hazard(s) with waves >= {wave_threshold_m:g} m "
                  f"converted to simulated storm cells (forecast fetched {assessment.get('fetched_at')} UTC)")

    def clear_forecast_hazards(self) -> str:
        keep = [c for c in self.settings["storm_cells"] if c.get("source") != self.FORECAST_SOURCE]
        if len(keep) == len(self.settings["storm_cells"]):
            return "No forecast-informed storm cells to clear."
        return self.set_event("storm_cells", keep, label="forecast-informed storm cells removed")

    # ---------------------------------------------------------------- counterfactual previews
    def clone(self) -> "Ops":
        """An isolated copy for what-if analysis: same read-only model, deep-copied voyage/scenario state.
        Nothing done to the clone can reach this session's state."""
        c = object.__new__(Ops)
        c.__dict__.update(self.__dict__)
        c.settings = copy.deepcopy(self.settings)
        c.voyage = copy.deepcopy(self.voyage)
        c.chat, c.demo, c._pre_demo, c.playing, c._last_tick = [], None, None, False, None
        c.last_change = c._demo_target = None
        return c

    # ---------------------------------------------------------------- hackathon demo mode
    DEMO_APPROACH_FRACTION = 0.5          # halfway along the East Mediterranean -> Port Said lane

    def snapshot(self) -> dict:
        """Whole-voyage figures right now (ETA, distance, cost, risk) for before/after comparisons."""
        G = self.G
        p = V.projected(self.voyage, G)
        s = self.state()
        return {"t": self.voyage.t, "lat": s.lat, "lon": s.lon, "eta": p["eta"], "total_h": self.voyage.total_hours,
                "nm": p["nm"], "remaining_nm": s.remaining_nm, "total_usd": p["total_usd"],
                "remaining_usd": p["remaining_usd"], "fuel_t": p["fuel_t"], "toll_usd": p["toll_usd"],
                "risk": p["remaining_risk_index"], "corridor": self.route_corridor(), "status": self.voyage.status}

    DEMO_PLAYBACK_H_PER_S = 50.0          # animated demo: ~3 s time-lapse from Rotterdam to the Port Said approach

    def run_demo(self, source: str = "demo", animate: bool = False) -> str:
        """One-click scenario: Rotterdam -> Shanghai container ship, sailed to the Port Said approach, then a
        SIMULATED Suez closure re-routes it around the Cape of Good Hope from where it is. Replayable: calling it
        again restarts from scratch; ``exit_demo`` restores whatever the user had before the first run.

        ``animate=False`` jumps the simulated clock straight to the approach (labelled as a time jump).
        ``animate=True`` plays the voyage as a time-lapse and ``tick`` triggers the closure on arrival there."""
        if self._pre_demo is None:
            self._pre_demo = (self.cfg, copy.deepcopy(self.settings), copy.deepcopy(self.voyage), self.playing)
        self.cfg = replace(V.VoyageConfig(), origin="RTM", destination="SHA", vessel_key="container",
                           owner_key="high_margin", preference="balanced", departure=self.cfg.departure)
        self.settings = default_settings()
        self.voyage = V.plan_voyage(self.cfg, self.settings, self.model)
        self.playing = False
        self._last_tick = None
        v, G = self.voyage, self.G
        V.log(v, "demo", "Hackathon demo started (SIMULATED scenario)",
              "Rotterdam → Shanghai container ship; a simulated Suez Canal closure will be triggered before the canal.",
              source)
        # Sail to a point on the lane that approaches Port Said from the Mediterranean (open water, before the canal).
        acc, t_target = 0.0, None
        for l in v.legs:
            if l.kind == "sail" and l.y == "PSD" and l.f0 == 0.0:
                t_target = acc + l.wait_h + l.sea_h * self.DEMO_APPROACH_FRACTION
                break
            acc += l.hours
        if t_target is None:
            raise RuntimeError("Demo needs the planned route to approach Suez from the Mediterranean.")
        self.demo = None
        v.status = "underway"
        if animate:
            V.log(v, "departure", "Departed Rotterdam",
                  f"Demo time-lapse: {self.DEMO_PLAYBACK_H_PER_S:g} simulated hours per second to the Port Said "
                  "approach; the simulated Suez closure triggers automatically there.", source)
            self._demo_target, self._demo_source = t_target, source
            self.speed = self.DEMO_PLAYBACK_H_PER_S
            self.playing = True
            self._touch()
            return (f"HACKATHON DEMO (simulated): sailing Rotterdam → Port Said approach as a time-lapse "
                    f"(~{t_target / 24:.1f} simulated days); the Suez closure scenario will trigger there.")
        V.log(v, "departure", "Departed Rotterdam",
              f"Simulated TIME JUMP of {t_target / 24:.1f} days to the Port Said approach.", source)
        V.advance(v, t_target)
        return self._demo_trigger(source)

    def _demo_trigger(self, source: str = "demo") -> str:
        G = self.G
        self._demo_target = None
        self.playing = False
        self._last_tick = None
        before = self.snapshot()
        msg = self.set_disruption("suez_closure", True, source=source)
        after = self.snapshot()
        _, ahead, _ = V.split_at(self.voyage, G)
        via_cape = any(l.y == "CAPE" or l.x == "CAPE" for l in ahead if l.kind == "sail")
        self.demo = {"before": before, "after": after, "message": msg, "via_cape": via_cape,
                     "position": G.nodes["MED1"]["name"] + " → Port Said approach",
                     "cause": "SIMULATED Suez Canal closure while approaching Port Said (hackathon demo)"}
        self._touch()
        return (f"HACKATHON DEMO (simulated): ship at {before['lat']:.2f}, {before['lon']:.2f} approaching Port Said "
                f"on day {before['t'] / 24:.1f}. {msg}")

    def exit_demo(self) -> str:
        if self._pre_demo is None:
            return "Demo mode is not active."
        self.cfg, self.settings, self.voyage, _ = self._pre_demo
        self._pre_demo, self.demo, self.last_change, self._demo_target = None, None, None, None
        self.playing = False
        self._last_tick = None
        self._touch()
        return "Demo mode exited; your previous voyage, assumptions and scenarios were restored."

    # ---------------------------------------------------------------- explanations (read-only)
    def explain(self, topic: str = "status") -> str:
        v, G = self.voyage, self.G
        s = self.state()
        p = V.projected(v, G)
        name = self.port_name
        cfg = self.cfg
        head = (f"{V.VESSELS[cfg.vessel_key].label} {name(cfg.origin)} → {name(cfg.destination)}, "
                f"status {v.status.upper()}{' (playing)' if self.playing else ''}, voyage day {v.t / 24:.1f}.")
        if topic == "eta":
            if v.status == "blocked":
                return head + f" No ETA: voyage blocked. {v.block_reason}"
            return (head + f" ETA {p['eta']:%Y-%m-%d %H:%M} UTC ({s.remaining_h / 24:.1f} days from now, "
                           f"{s.remaining_nm:,.0f} nm to go incl. {V.berth_hours(V.build_context(cfg, self.settings, self.model), cfg.destination):.0f} h berth wait). "
                           f"Speed over ground {s.sog_kn:.1f} kn. Simulated estimate.")
        if topic == "cost":
            return (head + f" Projected all-in ${p['total_usd'] / 1e6:,.2f}M (spent ${p['spent_usd'] / 1e6:,.2f}M, "
                           f"remaining ${p['remaining_usd'] / 1e6:,.2f}M): fuel {p['fuel_t']:,.0f} t = "
                           f"${p['fuel_usd'] / 1e6:,.2f}M at ${cfg.fuel_price:,.0f}/t, tolls ${p['toll_usd'] / 1e3:,.0f}k, "
                           f"war-risk ${p['war_usd'] / 1e3:,.0f}k, time cost ${p['time_usd'] / 1e6:,.2f}M. Modelled estimates.")
        if topic == "risk":
            active = describe(self.settings)
            ctx = V.build_context(cfg, self.settings, self.model)
            _, ahead, _ = V.split_at(v, G)
            hot = []
            for l in ahead:
                if l.kind != "sail":
                    continue
                d = ctx.G.edges[l.x, l.y]
                if d["geopolitical_peak"] >= 0.25 or d["weather_peak"] >= 0.6 or d["active_events"]:
                    seg = f"{G.nodes[l.x]['name']} - {G.nodes[l.y]['name']}"
                    why = ", ".join(d["active_events"]) or (
                        f"war-risk {d['geopolitical_peak']:.2f}" if d["geopolitical_peak"] >= 0.25 else
                        f"sea state {d['weather_peak']:.2f}")
                    hot.append(f"{seg} ({why})")
            return (head + f" Remaining-route risk index {p['remaining_risk_index']:.0f}/100 "
                           f"({V.risk_label(p['remaining_risk_index'])}); whole-voyage {p['risk_index']:.0f}/100. "
                    + (f"Exposed segments ahead: {'; '.join(hot[:5])}. " if hot else "No elevated segments ahead. ")
                    + (f"Active scenarios: {', '.join(active)}. " if active else "No active scenarios. ")
                    + "Risk index = hours-weighted (geopolitical + 0.5 x weather) exposure; synthetic risk model.")
        if topic == "options":
            opts = self.options()
            if not opts:
                return head + " No feasible routes from the current position."
            rows = [f"{c.name}: {c.summary.corridor}, {c.summary.hours / 24:.1f} d, ${c.summary.total_usd / 1e6:,.2f}M, "
                    f"risk {c.summary.risk_index:.0f}" for c in opts]
            safest = min(opts, key=lambda c: (c.summary.risk_index, c.summary.hours))
            return (head + " Routes from the current position: " + " | ".join(rows)
                    + f". Safest available: {safest.name} via {safest.summary.corridor} "
                      f"(risk {safest.summary.risk_index:.0f}/100).")
        return (head + f" Position ({s.lat:.2f}, {s.lon:.2f}), course {s.course_deg:03.0f}°, {s.sog_kn:.1f} kn, "
                       f"{s.activity}. {s.traveled_nm:,.0f} nm sailed, {s.remaining_nm:,.0f} nm remaining. "
                       f"Route: {self.route_corridor()}. SIMULATED position, not AIS.")

    def route_corridor(self) -> str:
        ctx = V.build_context(self.cfg, self.settings, self.model)
        return V.summarise(ctx, [l for l in self.voyage.legs if l.kind == "sail" and ctx.G.has_edge(l.x, l.y)]).corridor
