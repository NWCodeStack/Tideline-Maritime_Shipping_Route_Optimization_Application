"""
llm.py - optional LLM provider for the copilot.

The copilot works fully WITHOUT this (deterministic command parser). When an API key is configured
(``ANTHROPIC_API_KEY`` environment variable or ``st.secrets["ANTHROPIC_API_KEY"]``), the provider adds:

* ``extract_ops``: free-form message -> JSON ``{"operations": [{"op", "args"}], "reply"}`` constrained by a JSON
  schema. The result is only a PROPOSAL: copilot.validate() checks every operation against the fixed schema
  before anything runs. The model never executes code or changes state itself.
* ``explain``: rephrases the facts the application computed into a short answer. It is told to use only those
  facts, and the raw facts are always shown under its answer.

Keys are never hard-coded or logged. Any provider error degrades silently to command mode.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

DEFAULT_MODEL = "claude-opus-5-5"

ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "origin": {"type": "string"}, "destination": {"type": "string"}, "vessel": {"type": "string"},
        "start": {"type": "boolean"}, "hours": {"type": "number"}, "name": {"type": "string"},
        "active": {"type": "boolean"}, "value": {"type": "number"}, "wave_m": {"type": "number"},
        "radius_nm": {"type": "number"}, "port": {"type": "string"}, "preference": {"type": "string"},
        "via": {"type": "string"}, "price": {"type": "number"}, "change_pct": {"type": "number"},
        "knots": {"type": "number"}, "topic": {"type": "string"}, "region": {"type": "string"},
    },
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {"type": "array", "items": {
            "type": "object",
            "properties": {"op": {"type": "string"}, "args": ARGS_SCHEMA},
            "required": ["op", "args"],
            "additionalProperties": False,
        }},
        "reply": {"type": "string"},
    },
    "required": ["operations", "reply"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """You translate a ship-operations user's message into operations for a maritime voyage SIMULATOR.
Return only operations from this fixed list (args in parentheses; omit optional args you don't need):
create_voyage(origin, destination, vessel?, start?) start() pause() resume() reset() advance(hours)
set_disruption(name, active?, value?) add_storm(wave_m?, radius_nm?) clear_storms() clear_disruptions()
set_congestion(port, hours) reroute(preference?, via?) set_preference(preference) set_fuel_price(price? | change_pct?)
set_speed(knots) compare_route(via) explain(topic: status|eta|cost|risk|options) weather() news(region?)
run_demo() exit_demo()
Ports are the 3-letter ids listed in the state. Vessels: container, tanker, bulk, lng.
Disruption names: suez_closure, bab_el_mandeb_closure, hormuz_closure, panama_closure, red_sea_escalation,
taiwan_tension, suez_fee, panama_fee, panama_drought. Preferences: fastest, cheapest, safest, balanced.
Corridors for via: cape, suez, panama, horn, malacca, bab_el_mandeb, hormuz.
If the message is a question you cannot map to operations, return no operations and a one-sentence reply saying
which commands are supported. Never invent positions, costs or real-world facts; disruptions are simulated scenarios."""

EXPLAIN_SYSTEM = """You are the operations copilot of a maritime voyage SIMULATOR (synthetic network, simulated positions,
modelled cost/risk estimates). Answer the user's message in 2-4 sentences using ONLY the facts provided. Do not add
numbers, places or events that are not in the facts. Call disruptions scenarios, not real closures. No markdown headings."""


class LLMProvider(Protocol):
    name: str

    def extract_ops(self, message: str, state: str) -> dict | None: ...

    def explain(self, message: str, facts: str) -> str | None: ...


class AnthropicProvider:
    """Claude via the official Anthropic SDK. Uses low effort: these are short, latency-sensitive chat turns."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        import anthropic                    # optional dependency, imported only when a key is configured
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=30.0, max_retries=1)
        self.model = model
        self.name = f"Claude ({model})"
        self.last_error = ""

    def _create(self, **kw):
        # Server-side refusal fallback (opt-in beta): on a policy decline the API retries on a fallback model.
        try:
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        except (TypeError, self._anthropic.BadRequestError):   # SDK/account without the beta: plain request
            return self.client.messages.create(**kw)

    def _text(self, resp) -> str | None:
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        return next((b.text for b in resp.content if b.type == "text"), None)

    def extract_ops(self, message: str, state: str) -> dict | None:
        try:
            resp = self._create(
                model=self.model, max_tokens=2000, system=EXTRACT_SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": f"Current simulator state: {state}\n\nUser message: {message}"}],
            )
            text = self._text(resp)
            return json.loads(text) if text else None
        except (self._anthropic.APIError, json.JSONDecodeError, ValueError) as e:
            self.last_error = type(e).__name__
            return None

    def explain(self, message: str, facts: str) -> str | None:
        try:
            resp = self._create(
                model=self.model, max_tokens=1500, system=EXPLAIN_SYSTEM, output_config={"effort": "low"},
                messages=[{"role": "user", "content": f"Facts from the simulator:\n{facts}\n\nUser message: {message}"}],
            )
            return self._text(resp)
        except self._anthropic.APIError as e:
            self.last_error = type(e).__name__
            return None


def get_provider(secrets=None) -> tuple[LLMProvider | None, str]:
    """(provider or None, human-readable status). Looks in st.secrets first, then the environment."""
    key = None
    model = os.environ.get("TIDELINE_LLM_MODEL", DEFAULT_MODEL)
    try:
        if secrets is not None and "ANTHROPIC_API_KEY" in secrets:
            key = secrets["ANTHROPIC_API_KEY"]
            model = secrets.get("TIDELINE_LLM_MODEL", model)
    except Exception:                       # no secrets.toml -> Streamlit raises on access
        key = None
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, "Command mode (no LLM key configured)"
    try:
        p = AnthropicProvider(key, model)
    except ImportError:
        return None, "Command mode (anthropic package not installed)"
    return p, f"LLM assist: {p.name}"
