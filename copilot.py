"""
copilot.py - AI Operations Copilot: natural language -> validated operations -> the same Ops methods the
dashboard uses.

Pipeline
--------
    user message
      -> intent extraction   (deterministic parser; optional LLM fallback that may only emit JSON operations)
      -> validation          (fixed schema: op name + typed, range-checked arguments - nothing else runs)
      -> execution           (ops.Ops methods: simulation state update + routing engine)
      -> response            (built from real application state; optional LLM rephrasing of those facts)

The LLM never executes code and never touches state directly: it can only propose operations from ``SCHEMA``,
which are validated exactly like the parser's output. All disruptions are SIMULATED SCENARIOS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ops import DISRUPTIONS, PORTS, VIA_NODES, Ops
from voyage import PREFERENCES, VESSELS

# ------------------------------------------------------------------------------------------------
# Fixed operation schema: op -> {arg: (type, required, allowed values or (lo, hi) range or None)}
# ------------------------------------------------------------------------------------------------
SCHEMA: dict[str, dict] = {
    "create_voyage": {"origin": (str, True, tuple(PORTS)), "destination": (str, True, tuple(PORTS)),
                      "vessel": (str, False, tuple(VESSELS)), "start": (bool, False, None)},
    "start": {}, "pause": {}, "resume": {}, "reset": {},
    "advance": {"hours": (float, True, (0.1, 2000.0))},
    "set_disruption": {"name": (str, True, tuple(DISRUPTIONS)), "active": (bool, False, None),
                       "value": (float, False, (0.0, 10.0)), "force": (bool, False, None)},
    "add_storm": {"wave_m": (float, False, (1.0, 20.0)), "radius_nm": (float, False, (50.0, 1200.0))},
    "clear_storms": {}, "clear_disruptions": {},
    "set_congestion": {"port": (str, True, tuple(PORTS)), "hours": (float, True, (0.0, 720.0))},
    "reroute": {"preference": (str, False, tuple(PREFERENCES)), "via": (str, False, tuple(VIA_NODES))},
    "set_preference": {"preference": (str, True, tuple(PREFERENCES))},
    "set_fuel_price": {"price": (float, False, (100.0, 3000.0)), "change_pct": (float, False, (-90.0, 300.0))},
    "set_speed": {"knots": (float, True, (6.0, 25.0))},
    "compare_route": {"via": (str, True, tuple(VIA_NODES))},
    "explain": {"topic": (str, True, ("status", "eta", "cost", "risk", "options"))},
    "weather": {},
    "news": {"region": (str, False, None)},
}
READ_ONLY = {"explain", "weather", "news", "compare_route"}


@dataclass
class Op:
    name: str
    args: dict = field(default_factory=dict)


class InvalidOp(ValueError):
    pass


def validate(op: Op) -> Op:
    """Raise InvalidOp unless ``op`` matches SCHEMA exactly; returns a copy with coerced argument types."""
    if op.name not in SCHEMA:
        raise InvalidOp(f"unsupported operation '{op.name}'")
    spec = SCHEMA[op.name]
    extra = set(op.args) - set(spec)
    if extra:
        raise InvalidOp(f"unexpected arguments for {op.name}: {sorted(extra)}")
    out = {}
    for arg, (typ, required, allowed) in spec.items():
        if arg not in op.args or op.args[arg] is None:
            if required:
                raise InvalidOp(f"{op.name} needs '{arg}'")
            continue
        val = op.args[arg]
        try:
            val = typ(val) if typ is not bool else (val if isinstance(val, bool) else str(val).lower() == "true")
        except (TypeError, ValueError):
            raise InvalidOp(f"{op.name}.{arg} must be {typ.__name__}")
        if isinstance(allowed, tuple) and len(allowed) == 2 and all(isinstance(a, float) for a in allowed):
            if not allowed[0] <= val <= allowed[1]:
                raise InvalidOp(f"{op.name}.{arg}={val} outside {allowed}")
        elif isinstance(allowed, tuple) and val not in allowed:
            raise InvalidOp(f"{op.name}.{arg}='{val}' not one of {list(allowed)}")
        out[arg] = val
    return Op(op.name, out)


# ------------------------------------------------------------------------------------------------
# Deterministic parser
# ------------------------------------------------------------------------------------------------
PORT_ALIASES = {
    "rotterdam": "RTM", "shanghai": "SHA", "singapore": "SIN", "jebel ali": "DXB", "dubai": "DXB",
    "fujairah": "FJR", "mumbai": "MUM", "bombay": "MUM", "hong kong": "HKG", "new york": "NYC",
    "los angeles": "LAX", "long beach": "LAX",
}
CORRIDOR_WORDS = {
    "suez": "suez", "bab": "bab_el_mandeb", "mandeb": "bab_el_mandeb", "red sea": "bab_el_mandeb",
    "hormuz": "hormuz", "panama": "panama", "cape of good hope": "cape", "good hope": "cape", "the cape": "cape",
    "cape horn": "horn", "malacca": "malacca",
}
CLOSE_RE = r"\b(clos(e|ed|es|ing|ure)|shut|block(ed|ade|s)?|halt(ed)?|seal(ed)?)\b"
OPEN_RE = r"\b(re-?open(ed|s)?|lift(ed)?|clear(ed)?|end(ed)?|remove[ds]?|open again|back open|cancel)\b"
NUM = r"(\d+(?:\.\d+)?)"


def _ports_in(text: str) -> list[str]:
    hits = []
    for alias, pid in PORT_ALIASES.items():
        for m in re.finditer(r"\b" + re.escape(alias.strip()) + r"\b", text):
            hits.append((m.start(), pid))
    seen, out = set(), []
    for _, pid in sorted(hits):
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _corridors_in(text: str) -> list[str]:
    out = []
    for word, key in sorted(CORRIDOR_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(word) + r"\b", text) and key not in out:
            # "cape" alone is ambiguous; "cape horn" wins over "the cape" via longest-match ordering
            if key == "cape" and "horn" in out:
                continue
            out.append(key)
    return out


def _duration_hours(text: str) -> float | None:
    m = re.search(NUM + r"\s*(day|days|d|hour|hours|hrs|h|week|weeks|wk)\b", text)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2)
    return n * (168 if unit.startswith("w") else 24 if unit.startswith("d") else 1)


def parse(text: str, ops: Ops) -> list[Op]:
    """Extract operations from a message using keyword rules. Returns [] if nothing is recognised."""
    t = " " + text.lower().replace("’", "'") + " "
    out: list[Op] = []
    cfg = ops.cfg

    # --- voyage creation
    ports = _ports_in(t)
    vessel = next((k for k, w in (("tanker", "tanker"), ("lng", "lng"), ("bulk", "bulk"), ("container", "container"))
                   if re.search(r"\b" + w, t)), None)
    wants_start = bool(re.search(r"\b(start|begin|depart|set sail|get underway|launch)\b", t))
    if len(ports) >= 2:
        m = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:[.,;!?]|$| and | then )", t)
        o = d = None
        if m:
            o = (_ports_in(" " + m.group(1) + " ") or [None])[0]
            d = (_ports_in(" " + m.group(2) + " ") or [None])[0]
        if not (o and d) or o == d:
            o, d = ports[0], ports[1]
        if (o, d) != (cfg.origin, cfg.destination) or (vessel and vessel != cfg.vessel_key) or \
                re.search(r"\b(new|create|plan|set up)\b.*\bvoyage\b", t):
            out.append(Op("create_voyage", {"origin": o, "destination": d, "vessel": vessel, "start": wants_start}))
            wants_start = False

    # --- assumptions
    verbs = r"(increase|raise|up|rise|jump|spike|decrease|drop|lower|cut|fall|down|reduce)"
    m = re.search(r"fuel[^.]*?" + verbs + r"[^.]*?" + NUM + r"\s*%", t) \
        or re.search(verbs + r"[^.]*?fuel[^.]*?" + NUM + r"\s*%", t) \
        or re.search(NUM + r"\s*%[^.]*?" + verbs + r"[^.]*fuel", t)
    if m:
        groups = [g for g in m.groups() if g]
        num = float(next(g for g in groups if re.fullmatch(r"\d+(\.\d+)?", g)))
        down = any(g in ("decrease", "drop", "lower", "cut", "fall", "down", "reduce") for g in groups)
        out.append(Op("set_fuel_price", {"change_pct": -num if down else num}))
    else:
        m = re.search(r"fuel[^.]*?(?:to|at|=)\s*\$?\s*" + NUM, t)
        if m:
            out.append(Op("set_fuel_price", {"price": float(m.group(1))}))
    m = re.search(NUM + r"\s*(?:knots|kn|kts)\b", t)
    if m and re.search(r"\b(speed|slow|steam|cruise|knots|kn)\b", t) and "storm" not in t:
        out.append(Op("set_speed", {"knots": float(m.group(1))}))
    pref = next((p for p in PREFERENCES if re.search(r"\b" + p + r"\b", t)), None)
    switch = bool(re.search(r"\b(switch|use|take|change|go with|set|prefer|reroute|optimi[sz]e for|mode)\b", t))

    # --- disruptions
    corridors = _corridors_in(t)
    closing = re.search(CLOSE_RE, t) and not re.search(r"\bhow\b.*\bexpensive\b", t)
    opening = re.search(OPEN_RE, t)
    fee = re.search(r"\b(toll|fee|fees|tolls|dues)\b", t)
    for c in corridors:
        name = {"suez": "suez_closure", "bab_el_mandeb": "bab_el_mandeb_closure", "hormuz": "hormuz_closure",
                "panama": "panama_closure"}.get(c)
        if not name:
            continue
        if fee and c in ("suez", "panama"):
            mult = re.search(r"(?:x|times|by a factor of)\s*" + NUM, t) or re.search(NUM + r"\s*x\b", t)
            pct = re.search(NUM + r"\s*%", t)
            val = float(mult.group(1)) if mult else (1 + float(pct.group(1)) / 100 if pct else 2.0)
            out.append(Op("set_disruption", {"name": f"{c}_fee", "active": not opening, "value": val}))
            continue
        if opening and not closing:
            out.append(Op("set_disruption", {"name": name, "active": False}))
        elif closing:
            out.append(Op("set_disruption", {"name": name, "active": True}))
    if re.search(r"\b(escalat|attack|missile|houthi|security threat)", t) and "bab_el_mandeb" in corridors \
            and not closing:
        out.append(Op("set_disruption", {"name": "red_sea_escalation", "active": not opening}))
    if re.search(r"\b(storm|hurricane|cyclone|typhoon|gale|heavy weather|high seas|severe weather)\b", t):
        if opening or re.search(r"\b(passed|gone|over)\b", t):
            out.append(Op("clear_storms"))
        elif not re.search(r"^\s*(what|how|show|is|are|any)\b", t.strip()) or re.search(r"\b(reroute|avoid)\b", t):
            m = re.search(NUM + r"\s*-?\s*(?:m|meter|metre|meters|metres)\b", t)
            out.append(Op("add_storm", {"wave_m": float(m.group(1)) if m else 6.0}))
    if re.search(r"\bcongest", t) and ports:
        hrs = _duration_hours(t) or 48.0
        out.append(Op("set_congestion", {"port": ports[-1], "hours": hrs}))
    if re.search(r"\b(clear|remove|reset|cancel)\b.*\b(all|every)\b.*\b(scenario|disruption|incident|event)s?\b", t):
        out.append(Op("clear_disruptions"))

    # --- clock
    if re.search(r"\b(advance|fast[- ]?forward|skip|jump|move (?:ahead|forward)|progress)\b", t):
        h = _duration_hours(t)
        if h:
            out.append(Op("advance", {"hours": h}))
    if re.search(r"\b(pause|stop|freeze|hold)\b", t) and not re.search(r"\bhold(ing)? course\b", t):
        out.append(Op("pause"))
    elif re.search(r"\b(resume|continue|unpause|carry on)\b", t):
        out.append(Op("resume"))
    if re.search(r"\b(reset|restart)\b", t) and not any(o.name == "clear_disruptions" for o in out):
        out.append(Op("reset"))
    if wants_start and not any(o.name in ("pause", "resume") for o in out):
        out.append(Op("start"))

    # --- routing requests
    asks_impact = re.search(r"\b(impact|affect|affected|matter|concern)\b", t)
    question = re.search(r"\?\s*$", t.strip()) or re.match(r"\s*(what|how|which|is|does|do|can|should|will)\b", t.strip())
    via = next((c for c in corridors if c in VIA_NODES and c in ("cape", "horn", "panama", "suez")), None)
    if re.search(r"\b(how much|cost of|compare|difference|versus|vs\.?|more expensive|cheaper)\b", t) and via \
            and not closing:
        out.append(Op("compare_route", {"via": via}))
    elif re.search(r"\b(reroute|re-route|alternative|divert|avoid|find (?:a |the )?(?:best |new )?route|replan)\b", t):
        args = {}
        if pref:
            args["preference"] = pref
        if via and re.search(r"\b(via|through|around|by way of)\b", t) and not closing:
            args["via"] = via
        out.append(Op("reroute", args))
    elif pref and switch:
        out.append(Op("set_preference", {"preference": pref}))
    elif pref and question:
        out.append(Op("explain", {"topic": "options"}))

    # --- read-only answers
    if re.search(r"\b(news|headline|headlines|reports?|security update)\b", t):
        region = next((c for c in corridors), None)
        out.append(Op("news", {"region": region}))
    if re.search(r"\b(weather|forecast|wave|waves|swell|sea state)\b", t) and not any(o.name == "add_storm" for o in out):
        out.append(Op("weather"))
    if re.search(r"\brisk\b", t) and not any(o.name == "explain" for o in out):
        out.append(Op("explain", {"topic": "risk"}))
    if re.search(r"\b(eta|arriv|when will)\w*", t) and not any(o.name == "explain" for o in out):
        out.append(Op("explain", {"topic": "eta"}))
    if re.search(r"\b(cost|expensive|spend|budget)\b", t) and not any(o.name in ("explain", "compare_route") for o in out):
        out.append(Op("explain", {"topic": "cost"}))
    if re.search(r"\b(status|where is|where's|position|summary|update me)\b", t) and not any(
            o.name == "explain" for o in out):
        out.append(Op("explain", {"topic": "status"}))
    if asks_impact and not any(o.name.startswith("explain") for o in out):
        out.append(Op("explain", {"topic": "status"}))
    return out


def _merge(ops_list: list[Op]) -> list[Op]:
    """A disruption followed by a plain 'reroute' runs as one forced re-evaluation (no duplicate timeline entry)."""
    has_dis = [o for o in ops_list if o.name in ("set_disruption", "add_storm")]
    plain_reroute = [o for o in ops_list if o.name == "reroute" and not o.args]
    if has_dis and plain_reroute:
        ops_list = [o for o in ops_list if o not in plain_reroute]
        last = has_dis[-1]
        if last.name == "set_disruption":
            last.args["force"] = True
    return ops_list


# ------------------------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------------------------
def execute(op: Op, ops: Ops, intel: dict | None = None) -> str:
    """Run ONE validated operation through the Ops controller (the same code the dashboard buttons run)."""
    a = op.args
    src = "copilot"
    if op.name == "create_voyage":
        msg = ops.create_voyage(a["origin"], a["destination"], a.get("vessel"), source=src)
        if a.get("start") and ops.voyage.status != "blocked":
            msg += " " + ops.start()
        return msg
    if op.name == "start":
        return ops.start()
    if op.name == "pause":
        return ops.pause()
    if op.name == "resume":
        return ops.resume()
    if op.name == "reset":
        return ops.reset()
    if op.name == "advance":
        return ops.advance_hours(a["hours"], source=src)
    if op.name == "set_disruption":
        return ops.set_disruption(a["name"], a.get("active", True), a.get("value"), source=src,
                                  force=a.get("force", False))
    if op.name == "add_storm":
        return ops.add_storm(a.get("wave_m", 6.0), radius_nm=a.get("radius_nm", 300.0), source=src)
    if op.name == "clear_storms":
        return ops.clear_storms(source=src)
    if op.name == "clear_disruptions":
        return ops.clear_all(source=src)
    if op.name == "set_congestion":
        return ops.set_congestion(a["port"], a["hours"], source=src)
    if op.name == "reroute":
        return ops.reroute(a.get("preference"), a.get("via"), source=src)
    if op.name == "set_preference":
        return ops.update_config(source=src, preference=a["preference"])
    if op.name == "set_fuel_price":
        price = a.get("price") or round(ops.cfg.fuel_price * (1 + a.get("change_pct", 0.0) / 100.0), 2)
        before = ops.cfg.fuel_price
        msg = ops.update_config(source=src, fuel_price=price)
        return f"Fuel price ${before:,.0f}/t → ${price:,.0f}/t (assumption). {msg}"
    if op.name == "set_speed":
        return ops.update_config(source=src, cruise_speed_kn=a["knots"])
    if op.name == "compare_route":
        return ops.compare_via(a["via"])
    if op.name == "explain":
        return ops.explain(a["topic"])
    if op.name == "weather":
        return (intel or {}).get("weather_summary") or "Live weather is unavailable right now (see the Weather tab)."
    if op.name == "news":
        fn = (intel or {}).get("news_summary")
        return fn(a.get("region")) if callable(fn) else "Live news is unavailable right now (see the News tab)."
    return "Unsupported."


HELP = ("I run in command mode (no LLM key configured, or it did not understand). Try: "
        "'Start a voyage from Rotterdam to Shanghai' · 'Advance 5 days' · 'Close Suez and reroute my ship' · "
        "'A storm with 6 m waves is on our route, reroute' · 'What is the safest route?' · "
        "'Increase fuel prices by 25%' · 'How much more expensive is the Cape of Good Hope route?' · "
        "'Pause and explain our risk' · 'Show Red Sea news' · 'Hormuz closed - does it impact my shipment?'")


@dataclass
class Reply:
    text: str
    ops: list = field(default_factory=list)          # [(op name, args, result)]
    rejected: list = field(default_factory=list)     # [(proposed op, reason)]
    mode: str = "command"                            # "command" | "llm"
    changed: bool = False


def handle(message: str, ops: Ops, llm=None, intel: dict | None = None) -> Reply:
    """Full pipeline for one chat message."""
    proposed = parse(message, ops)
    mode = "command"
    llm_reply = ""
    if not proposed and llm is not None:
        got = llm.extract_ops(message, state_brief(ops))
        if got is not None:
            mode = "llm"
            proposed = [Op(o.get("op", ""), dict(o.get("args") or {})) for o in got.get("operations", [])]
            llm_reply = got.get("reply", "")
    proposed = _merge(proposed)
    valid, rejected = [], []
    for op in proposed:
        try:
            valid.append(validate(op))
        except InvalidOp as e:
            rejected.append((op, str(e)))
    results = []
    for op in valid:
        try:
            results.append((op.name, op.args, execute(op, ops, intel)))
        except Exception as e:                       # an operation failure must never crash the app
            results.append((op.name, op.args, f"Operation failed: {e}"))
    changed = any(name not in READ_ONLY for name, *_ in results)
    if not results:
        text = llm_reply or HELP
        if rejected:
            text += " (Rejected: " + "; ".join(r for _, r in rejected) + ")"
        return Reply(text, [], rejected, mode, False)
    text = "\n\n".join(r for *_, r in results)
    if llm is not None and any(name in READ_ONLY for name, *_ in results):
        prose = llm.explain(message, text)
        if prose:
            text = prose + "\n\n---\n" + text
            mode = "llm"
    return Reply(text, results, rejected, mode, changed)


def state_brief(ops: Ops) -> str:
    """Compact, factual state for the LLM intent extractor (it must not invent positions or numbers)."""
    s = ops.state()
    cfg = ops.cfg
    return (f"voyage {cfg.origin}->{cfg.destination} vessel={cfg.vessel_key} status={ops.voyage.status} "
            f"day={ops.voyage.t / 24:.1f} pos=({s.lat:.2f},{s.lon:.2f}) preference={cfg.preference} "
            f"fuel_price={cfg.fuel_price} ports={PORTS} corridors={list(VIA_NODES)} disruptions={list(DISRUPTIONS)}")
