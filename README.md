# Tideline - Live AI Shipping Operations Simulator

Hackathon prototype (c0mpiled-14 · UC Berkeley · YC RFS: *New Operating Systems for the Physical World*).

Plan a voyage, start the simulated clock and watch a ship move along modelled sea lanes. Close a strait, raise a
storm or change fuel prices (from the incident panel or by asking the AI copilot), and the rest of the voyage is
re-optimised **from where the ship is now**. Live marine weather (Open-Meteo) and maritime headlines (GDELT /
Google News) feed the Alerts & Intelligence panel. The original multi-agent stress test (container ship vs oil tanker) is kept as a tab.

> Vessel positions are **simulated** (no AIS feed). The network, tolls, risk and costs are synthetic, modelled
> estimates. Disruptions are user-triggered **scenarios**, never verified closures. Not for navigation or safety use.

Engineering history, design decisions and house rules for contributors: see [CLAUDE.md](CLAUDE.md).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains the fuel model (~2 s) and caches it in `models/fuel_model.pkl` (git-ignored, rebuilt
automatically). No API key is needed. Set `TIDELINE_OFFLINE=1` to disable all network calls (offline demo / tests).

Other commands: `python -m pytest -q` (all 79 tests; `pytest.ini` selects them), `python app.py` (console stress-test demo),
`python ml_models.py` (train + print metrics), `python routing_engine.py` (CLI over all presets).

## Using the app

The page follows one hierarchy: **header** (status: Ready / Simulation active / paused / blocked) →
**Voyage & Simulation | Alerts & Intelligence** → **Route KPIs** → **Live route map | Operations Copilot** →
**Rerouting decision & event timeline**. Advanced configuration lives in the sidebar (collapsed by default, `»`).

* **Voyage & Simulation** - from / to / vessel (container, oil tanker, bulk carrier, LNG carrier; locked once underway,
  **Reset** to change), Start / Pause / Resume, +1 day, Reset, playback speed (1-100 simulated hours per second; at
  high speed the telemetry strip says each refresh is a time-lapse jump), the **hackathon demo** buttons, the four
  closure toggles (Suez, Bab-el-Mandeb, Hormuz, Panama) and *Clear all scenarios*. Every scenario is SIMULATED.
* **Alerts & Intelligence** - alerts (blocked voyage, reroutes, active scenarios, forecast hazards ahead, unverified news
  advisories), then **Weather** (Open-Meteo, matched to the ship's simulated arrival time; out-of-horizon, unavailable
  and stale states labelled; optional *Forecast-informed simulation*) and **News & Security** (GDELT, Google News RSS
  fallback, or a clearly labelled demo set).
* **Route KPIs** - ETA, distance remaining, projected cost (fuel, tolls, spent) and risk ahead; ETA, cost and risk carry
  a MODEL ESTIMATE label.
* **Live route map** - interactive Leaflet map (only the voyage layer is pushed each tick, so zoom/pan survive
  playback) or the **Offline map** (SVG, no tiles or folium needed) from the selector above the map. Mint = completed,
  solid = remaining, grey dashed = superseded, amber dotted = alternatives, red dashed = blocked, purple = storms.
  Drag the time slider or click a waypoint to jump.
* **Rerouting decision & event timeline** - a recommendation card (what changed, why, and ETA / distance / fuel /
  cost / risk deltas), a compact before/after table, the route options considered, and the timeline.
* **Sidebar (advanced)** - cargo owner and departure time, fuel price and fixed speed, objective
  (Fastest / Cheapest / Safest / Balanced + weights), chokepoints to avoid, canal / berth delays and port congestion,
  storm scenario, more scenarios and fees (Red Sea escalation, Taiwan, hurricane, drought, presets), methodology.

### Hackathon demo

**▶ Hackathon demo: Suez closure** plays Rotterdam → Shanghai as a time-lapse to the Port Said approach, where a
clearly labelled SIMULATED Suez closure triggers automatically; the ship is re-routed via the Cape of Good Hope from its
position and the before/after comparison appears. **Instant demo** does the same with a labelled time jump.
**Replay** restarts it; **Exit demo** restores the voyage, assumptions and scenarios you had before. It needs no live
APIs and no LLM key. The copilot understands "run the demo" / "exit demo".

### Forecast-informed simulation (opt-in)

Off by default. When on, Open-Meteo forecast points **ahead of the ship** whose significant wave height at the simulated
ETA is at or above the threshold become labelled storm cells ("FORECAST-INFORMED SCENARIO"), the remaining route is
re-assessed from the ship's position, and the before/after appears in the decision panel. Cells only accumulate
(deduplicated, capped at 6) so the route cannot oscillate; "Remove forecast-informed storm cells" undoes it. Stale or
unavailable forecasts are never applied. This is a modelled scenario, not a navigation recommendation.

Weather assessments are bound to the itinerary they were computed for: after a reroute, reset or new voyage, an older
forecast is never shown as current (a failed refresh then reports "unavailable" instead of reusing it).

### Operations Copilot

Only **explicit instructions** change the simulation; everything else is read-only:

| Message | What happens |
| --- | --- |
| `Simulate a Suez closure and reroute my ship` / `Close Suez and reroute` | applied (scenario on, voyage re-planned) |
| `What happens if the Suez Canal closes?` / `What if fuel rises 25%?` | **PREVIEW** computed on a copy - nothing changes |
| `Do not close Suez, just explain the effect of a closure` | negation → preview only |
| `Is the Strait of Hormuz closed?` | reports the simulated scenario state (and that real closures are not verified) |
| `Show news about a Suez closure` | headlines only |
| `Suez closure` / `The Suez Canal has closed.` | ambiguous → asks you to say *simulate* or *what if* |

More examples: `Advance the voyage by 5 days` · `A storm with 6-meter waves is affecting our route. Reroute to avoid
it.` · `What is the safest available route?` · `How much more expensive is the Cape of Good Hope route?` ·
`Pause the vessel and explain our current operational risk` · `Run the hackathon demo`.

Pipeline: message → intent classification (action / question / hypothetical / negated / ambiguous) → operation
extraction (deterministic parser; optional LLM fallback that can only *propose* JSON operations) → schema validation →
intent gate (the same rules for parser and LLM output) → the same `ops.Ops` methods the dashboard uses → routing engine
→ page refresh → answer built from real state. The header shows whether you are in **rules-based command mode** (no
key) or **LLM assist**; each reply is tagged `rules` or `LLM`.

## How rerouting works

1. The itinerary is cut at the ship's current simulated time; everything already sailed (distance, hours, fuel,
   tolls, cost) is kept.
2. Only the **remaining** lanes are checked. A closure the ship has already passed, or that its corridor never
   touches (e.g. Hormuz for Rotterdam → Shanghai), does not reroute it. Conditions that *improve* anywhere (a strait
   reopens) trigger a re-check too.
3. Dijkstra runs from a virtual node on the ship's current lane (it may continue or turn back along that lane) over
   the lane graph, weighted by the chosen objective:
   * Fastest = hours; Cheapest = fuel + tolls + war-risk insurance + ship operating cost;
   * Safest = hours × (geopolitical risk + 0.5 × weather risk);
   * Balanced = w_time·hours/avg + w_cost·cost/avg + w_risk·risk/avg (averages over all lanes).
   Storm cores (≥ 5 m waves) and lanes above the insurer's premium limit are avoided unless no other route exists
   ("forced through"); closures, draft limits and avoided chokepoints are never relaxed.
4. If nothing is reachable the voyage is **BLOCKED** and the ship holds position (e.g. a tanker inside the Persian
   Gulf when Hormuz closes). Ships never use overland lanes, never cross land, never teleport.

Fuel burn per lane comes from the random-forest model (`ml_models.py`); bulk carriers use the tanker hull class and
LNG carriers the container hull class (labelled as estimates).

## Layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit entrypoint: page layout, KPI / map fragments, copilot panel, intelligence panel, advanced sidebar, stress-test tab. |
| `voyage.py` | Voyage simulator: itinerary legs, clock, position interpolation, rerouting from the current position, objectives, risk index. |
| `ops.py` | Operations controller (one per browser session): the only code that changes simulation state. |
| `copilot.py` | Command parser, operation schema + validation, executor, response builder. |
| `llm.py` | Optional Anthropic provider (structured JSON operations, fact-only explanations). |
| `intel.py` | Open-Meteo marine/wind forecasts, GDELT + Google News RSS headlines, classification, advisories. |
| `folium_map.py` | Leaflet maps: stress-test map, live base map and the per-tick voyage layer. |
| `data_engine.py`, `events.py` | Synthetic lane graph (98 nodes, 111 lanes, risk zones) and scenario stressing. |
| `routing_engine.py`, `agents.py`, `ml_models.py` | Per-lane pricing, vessel/cargo profiles, fuel model. |
| `ui.py`, `map_view.py`, `geo_assets.py` | Styling, SVG fallback map, coastlines + land test. |
| `tests/` | Acceptance tests (`test_live_ops.py`), demo tests (`test_demo_mode.py`), audit-fix regressions (`test_audit_fixes.py`), original routing/data/ML tests. |

## APIs and keys

| Service | Used for | Key |
| --- | --- | --- |
| Open-Meteo Marine API + Forecast API | waves, swell, currents, SST, 10 m wind | none (free, CC BY 4.0) |
| GDELT DOC 2.0 API | headlines (primary) | none (rate-limited: 1 req / 5 s) |
| Google News RSS search | headlines (fallback) | none |
| Anthropic API (optional) | copilot natural-language assist | `ANTHROPIC_API_KEY` |

Configure the optional key as an environment variable or in `.streamlit/secrets.toml` (see
`.streamlit/secrets.toml.example`; on Streamlit Community Cloud use *App settings → Secrets*). Never commit keys.
Forecasts are cached 30 min, headlines 15 min; failing providers back off and the UI shows unavailable / stale states.

## Deploy (Streamlit Community Cloud)

Push the repo to GitHub, create an app with main file `app.py` (Python 3.11-3.13), optionally add the secret above.
`requirements.txt` is the only dependency file needed.
