# Tideline - Live AI Shipping Operations Simulator

Hackathon prototype (c0mpiled-14 · UC Berkeley · YC RFS: *New Operating Systems for the Physical World*).

Plan a voyage, start the simulated clock and watch a ship move along modelled sea lanes. Close a strait, raise a
storm or change fuel prices (from the incident panel or by asking the AI copilot), and the rest of the voyage is
re-optimised **from where the ship is now**. Live marine weather (Open-Meteo) and maritime headlines (GDELT /
Google News) feed the sidebar. The original multi-agent stress test (container ship vs oil tanker) is kept as a tab.

> Vessel positions are **simulated** (no AIS feed). The network, tolls, risk and costs are synthetic, modelled
> estimates. Disruptions are user-triggered **scenarios**, never verified closures. Not for navigation or safety use.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains the fuel model (~2 s) and caches it in `models/fuel_model.pkl` (git-ignored, rebuilt
automatically). No API key is needed. Set `TIDELINE_OFFLINE=1` to disable all network calls (offline demo / tests).

Other commands: `python -m pytest -v` (all tests), `python app.py` (console stress-test demo),
`python ml_models.py` (train + print metrics), `python routing_engine.py` (CLI over all presets).

## Using the app

**Sidebar**
* *Voyage* - departure/destination port, vessel type (container, oil tanker, bulk carrier, LNG carrier), cargo owner,
  departure date/time (UTC). Locked once the voyage is underway; **Reset** to change.
* *Vessel & economics* - fixed cruising speed (or let the optimiser pick per lane), fuel price, estimated burn.
* *Route optimisation* - objective **Fastest / Cheapest / Safest / Balanced** (balanced weights for time, cost, risk),
  chokepoints to avoid, berth wait at destination, extra canal delay.
* *Simulation* - Start / Pause / Resume, +1 day, Reset, playback speed (1, 10, 50 or 100 simulated hours per second).
* *Incident management* - Suez, Bab-el-Mandeb/Red Sea, Hormuz and Panama closures, Red Sea escalation, Suez/Panama
  fee multipliers, a storm placed on the route ahead (wave height + radius), port congestion, more scenarios and presets.
* *Alerts* - blocked voyage, reroutes, active scenarios, severe forecast weather ahead, news advisories.
* *Live Maritime Intelligence* - **Weather** (Open-Meteo forecast sampled along the remaining route and matched to the
  ship's simulated arrival time; points beyond the 7-day horizon are labelled as such) and **News & Security**
  (5-8 recent headlines with source, time, location tags and "on route" relevance; refresh button; demo toggle).

**Main area (Live operations)** - KPIs (ETA, distance remaining, projected fuel, projected all-in cost, risk index),
a telemetry strip, and the map: mint = completed track, solid = remaining plan, grey dashed = superseded route, amber
dotted = alternatives considered, red dashed = blocked plan, coloured circles = forecast conditions. Drag the voyage
time slider or click a waypoint to jump. Below: the routing decision with all candidate routes, and the event timeline.

**AI Operations Copilot** - examples:
`My ship is traveling from Rotterdam to Shanghai. Start the voyage.` · `Advance the voyage by 5 days.` ·
`The Suez Canal just closed. Find the best alternative route.` · `A storm with 6-meter waves is affecting our current
route. Reroute to avoid it.` · `What is the safest available route?` · `Increase fuel prices by 25% and recalculate
costs.` · `How much more expensive is the Cape of Good Hope route?` · `Pause the vessel and explain our current
operational risk.` · `Show me recent maritime news affecting the Red Sea.` · `The Strait of Hormuz has closed. Does
that impact my Rotterdam to Shanghai shipment?`

Pipeline: message → intent extraction (deterministic parser; optional LLM fallback) → validation against a fixed
operation schema → the same `ops.Ops` methods the sidebar uses → routing engine → map refresh → answer built from
real state. The LLM can only *propose* operations as JSON; it never runs code or touches state directly.

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
| `app.py` | Streamlit entrypoint: sidebar, live-operations tab, copilot panel, intelligence panel, stress-test tab. |
| `voyage.py` | Voyage simulator: itinerary legs, clock, position interpolation, rerouting from the current position, objectives, risk index. |
| `ops.py` | Operations controller (one per browser session): the only code that changes simulation state. |
| `copilot.py` | Command parser, operation schema + validation, executor, response builder. |
| `llm.py` | Optional Anthropic provider (structured JSON operations, fact-only explanations). |
| `intel.py` | Open-Meteo marine/wind forecasts, GDELT + Google News RSS headlines, classification, advisories. |
| `folium_map.py` | Leaflet maps: stress-test map, live base map and the per-tick voyage layer. |
| `data_engine.py`, `events.py` | Synthetic lane graph (98 nodes, 111 lanes, risk zones) and scenario stressing. |
| `routing_engine.py`, `agents.py`, `ml_models.py` | Per-lane pricing, vessel/cargo profiles, fuel model. |
| `ui.py`, `map_view.py`, `geo_assets.py` | Styling, SVG fallback map, coastlines + land test. |
| `tests/` | Acceptance tests (`test_live_ops.py`) plus the original routing/data/ML tests. |

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
