# Tideline - Live AI Shipping Operations Simulator

Hackathon prototype (c0mpiled-14 · UC Berkeley · YC RFS: *New Operating Systems for the Physical World*).

Plan a voyage, start the simulated clock and watch a ship move along modelled sea lanes. Close a strait, raise a
storm or change fuel prices (from the incident picker or by asking the AI copilot), and the rest of the voyage is
re-optimised **from where the ship is now**. Live marine weather (Open-Meteo) and maritime headlines (GDELT /
Google News) feed the Alerts & intelligence tab. The original multi-agent stress test (container ship vs oil tanker)
is kept as a second top-level tab.

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

The first launch trains the fuel model on 5,000 synthetic voyages and caches it in `models/fuel_model.pkl`
(git-ignored, rebuilt automatically). No API key is needed. Set `TIDELINE_OFFLINE=1` to disable all network calls
(offline demo / tests).

Other commands: `python -m pytest -q` (all tests; `pytest.ini` selects `test_routing.py` + `tests/`),
`python app.py` (console stress-test demo), `python ml_models.py` (train + print metrics),
`python routing_engine.py` (CLI over all presets).

## Using the app

Header, the "Latest reroute" banner, four KPI cards (ETA, distance remaining, projected cost, risk ahead) and a
telemetry strip run full width. Below them are two columns:

* **Left (never inside a tab, so switching tabs cannot reset it):** the live route map with a time slider and legend,
  and the Operations Copilot docked under it.
* **Right, five tabs** (the selected tab survives reruns):
  * **Operate** - status line (stay / change course / blocked), From / To / Vessel (container, oil tanker, bulk
    carrier, LNG carrier; locked once underway, **Reset** to change), Start / Pause / Resume, +1 day, playback speed (a slider from real time to 200 simulated hours per second; the ship glides in the browser between server syncs),
    the hackathon demo buttons, an incident picker (Suez, Bab-el-Mandeb, Hormuz, Panama closed) and a
    "Storms, fees and more scenarios" expander.
  * **Optimize** - preset (Balanced / Cheapest / Fastest / Safest / Greenest, or Custom), value of time, carbon price
    and fuel price source; **Advanced** holds objective weights, chokepoints to avoid, EU ETS, bunker price, fixed
    speed, cargo owner, departure time and delays.
  * **Route comparison** - the decision card, route options side by side (with CO2 and risk band), the cost breakdown,
    the risk panel, sea-state checkpoints ahead and a fuel-burn chart.
  * **Alerts & intelligence** - alerts, **Weather** (Open-Meteo, matched to the ship's simulated arrival time, with an
    opt-in *Forecast-informed simulation*) and **News & Security** (GDELT, Google News RSS fallback, or a labelled
    demo set), the risk legend and the event timeline.
  * **Data sources** - every input behind the plan with its basis, date and link, plus the methodology.
* The second top-level tab, **Multi-agent stress test**, runs the original container-ship vs tanker comparison.

The map is an interactive Leaflet map (only the voyage layer is pushed each tick, so zoom and pan survive playback) or
an **Offline map** (SVG, no tiles or folium needed). Mint = completed, solid = remaining, grey dashed = superseded,
amber dotted = alternatives, red dashed = blocked, purple = storms, tinted boxes = security zones by risk band, red
ring = port the ship can no longer reach, amber ring = port congestion scenario.

### Hackathon demo

**▶ Hackathon demo: Suez closure** plays Rotterdam → Shanghai as a time-lapse to the Port Said approach, where a
clearly labelled SIMULATED Suez closure triggers automatically; the ship is re-routed via the Cape of Good Hope from its
position and the before/after comparison appears. **Instant demo** does the same with a labelled time jump.
**Replay** restarts it; **Exit demo** restores the voyage, assumptions and scenarios you had before. It needs no live
APIs and no LLM key. The copilot understands "run the demo" / "exit demo".

### Panama close / reopen script

Shows that a re-plan starts from the ship's position and can turn it round:

1. Operate: From **Rotterdam**, To **Los Angeles**, then **Start** (the plan goes through Panama).
2. After about 10 simulated days (playback 10 h/s, or **+1 day** ten times) add **Panama Canal closed** in the
   incident picker. The ship is re-planned from where it is, via Cape Horn (offshore lanes along South America, no
   crossing of the mid-Atlantic).
3. Let about 4 more days pass, then remove the incident. The decision card reads **CHANGE COURSE NOW**: the ship turns
   toward Panama (a 4-hour U-turn is charged) if that beats staying on the Cape Horn route by more than the
   change-course threshold ($25k all-in).

## How the decision works

**Rerouting.** The itinerary is cut at the ship's current simulated time; everything already sailed (distance, hours,
fuel, tolls, cost) is kept. Only the **remaining** lanes are checked, and a closure the ship has passed or never
touches does not reroute it. Dijkstra runs from a virtual node on the ship's current lane (continue, or turn back along
it), then storm cores (≥ 5 m waves) and lanes above the insurer's premium limit are avoided unless nothing else exists
("forced through"); closures, draft limits and avoided chokepoints are never relaxed. If nothing is reachable the voyage
is **BLOCKED**, the ship holds position, and the nearest port it can still reach is named.

**Stay or change.** The ship leaves its committed route only if the alternative beats it by the change-course
threshold (default $25k), judged in the active objective. "Safest" also switches on a 3-point risk-score drop.

**Decision score.** Per lane: cash cost = fuel + tolls + war-risk insurance + time cost + ECA fuel switch + EU ETS.
Decision score = cash + (value of time - 1) × time cost + carbon price × CO2. Speed is the best of the vessel's options
on each lane on that score. The carbon price is a decision weight only and is never added to cash cost. Presets set the
objective, value of time and carbon price: Balanced (1×, $0), Cheapest (1×, $0), Fastest (4×, $0), Safest (1×, $0),
Greenest (1×, $200/t CO2); moving a slider reads as Custom.

**Risk score.** Each leg combines security (zone base level plus scenarios), weather (half the lane's sea-state
severity) and traffic (simulated canal delay only) as independent probabilities; legs are weighted by time spent
exposed. The score is 0-100 with bands **Low** <20, **Guarded** <40, **Elevated** <60, **High** <80, **Severe**. The
caption shown with every score: *"Risk score (0-100): modelled chance that something disrupts this voyage ... It's a
relative comparison between routes, not a calibrated probability."* Contributions of the three factors add up to the
score. Weather is sea-state severity, not yet checked against a ship wave limit.

Fuel burn per lane comes from a random-forest model (`ml_models.py`) fitted to **synthetic** voyages (its R² says
nothing about real ships); bulk carriers use the tanker hull class and LNG carriers the container hull class.

## What is simulated

| Input | Basis |
| --- | --- |
| Vessel position, ETA, fuel, cost, risk | SIMULATED / model estimates, never AIS |
| Closures, storms, congestion, escalations | SIMULATED scenarios the user switches on; never verified closures |
| Lane network (99 nodes, 113 lanes), tolls | Synthetic, land-checked lanes with real chokepoint geography |
| Weather forecast | LIVE (Open-Meteo), with source and fetch time; advisory unless forecast-informed mode is on |
| News headlines | LIVE (GDELT / Google News) or a labelled demo set; advisory, never closes a corridor or changes risk |
| ECA zones, EU ETS | ESTIMATED |
| Port bunker prices | SAMPLE table |
| Carbon price, value of time, fuel price | USER SETTINGS |

There is no port-draft data, so the app never marks a port "too shallow".

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
| a pasted news report | labelled PREVIEW of a proposed scenario; applied only by the exact reply `Simulate the proposed scenario` |

More examples: `Advance the voyage by 5 days` · `A storm with 6-meter waves is affecting our route. Reroute to avoid
it.` · `What is the safest available route?` · `Pause the vessel and explain our current operational risk` ·
`Run the hackathon demo`.

Pipeline: message → intent classification (action / question / hypothetical / negated / ambiguous / pasted report) →
operation extraction (deterministic parser; optional LLM fallback that can only *propose* JSON operations) → schema
validation → intent gate (the same rules for parser and LLM output) → the same `ops.Ops` methods the dashboard uses →
routing engine → page refresh → answer built from real state. Pasted text is treated as data, never as instructions.

## Layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit entrypoint: `render_ui` wiring, console demo. |
| `page.py` | `Page` context: cached resources, widget callbacks, widget <-> controller sync. |
| `tabs.py` | Header, copilot and the five right-hand tab renderers. |
| `live.py` | KPI and map fragments that re-run every second while playing. |
| `intel_ui.py`, `stress_ui.py` | Weather / News panels; the multi-agent stress-test tab. |
| `voyage.py`, `geo.py` | Voyage simulator (legs, clock, rerouting, objectives); lane geometry. |
| `ops.py` | Operations controller (one per browser session): the only code that changes simulation state. |
| `economics.py`, `risk.py` | Presets, ECA / ETS / carbon / bunker lines; the 0-100 risk score. |
| `panels.py`, `sources.py`, `newsparse.py` | Pure helpers behind the tabs and map; Data sources rows; pasted-report analysis. |
| `copilot.py`, `llm.py` | Command parser, validation, executor; optional Anthropic provider. |
| `intel.py` | Open-Meteo forecasts, GDELT + Google News RSS headlines, advisories. |
| `folium_map.py`, `map_view.py` | Leaflet maps (with hazard overlay); offline SVG map. |
| `data_engine.py`, `events.py` | Synthetic lane graph, risk zones, scenario stressing. |
| `routing_engine.py`, `agents.py`, `ml_models.py` | Per-lane pricing, vessel/cargo profiles, fuel model. |
| `ui.py`, `geo_assets.py` | Styling and HTML builders; coastlines + land test. |
| `tests/` | One file per phase plus coverage-gap, live-ops, demo, audit and original routing/data/ML tests. |

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

Push the repo to GitHub, create an app with main file `app.py` (Python 3.12), optionally add the secret above.
`requirements.txt` is the only dependency file needed. Not deployed yet.
