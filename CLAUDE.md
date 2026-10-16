# CLAUDE.md - Tideline project memory

Context for anyone (human or AI) picking up this repo. It condenses the full build thread: what the project is,
what was asked for, what was decided and why, how the code fits together, and what is still open.
User-facing docs live in `README.md`; this file is the engineering history and the house rules.

## What this is

**Tideline** (previously called *Pelagic* / *Multi-Agent Maritime Routing & Supply-Chain Stress-Tester*) is a UC Berkeley
hackathon project (c0mpiled-14, YC RFS *Operating Systems for the Physical World*). It is a Python + Streamlit
"live AI shipping operations simulator":

* a synthetic global lane network (98 nodes, 111 lanes, real chokepoint geography, land-checked lanes);
* a random-forest fuel model trained on 5,000 **synthetic** voyages;
* a simulated vessel that moves along the lanes, with mid-voyage dynamic rerouting from its current position;
* an Operations Copilot (rules-based command parser, optional Anthropic LLM assist);
* live Open-Meteo marine weather and GDELT / Google News headlines (advisory only);
* a one-click hackathon demo (simulated Suez closure -> Cape of Good Hope diversion);
* the original multi-agent stress test (container ship vs oil tanker) kept as a second tab.

Repo: `maritime-stress-tester/` (git, branch `main`, no remote yet - **not pushed or deployed**).

## Commands

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py                                    # app (trains models/fuel_model.pkl on first launch, ~2 s)
python -m pytest -q                                     # 79 tests (pytest.ini selects test_routing.py + tests/)
python app.py                                           # console stress-test demo
```
`TIDELINE_OFFLINE=1` disables all network calls (tests, offline demos). No API key is required for anything.
Optional `ANTHROPIC_API_KEY` (env or `.streamlit/secrets.toml`, see `.streamlit/secrets.toml.example`) enables LLM assist.

## Architecture (where things live)

| File | Role |
| --- | --- |
| `app.py` | Streamlit entrypoint. Page layout, the two playback fragments (`kpi_fragment`, `map_fragment`), copilot panel, Weather/News panel (`render_weather`, `render_news`), advanced sidebar, stress-test tab (`stress_tab`), `console_demo()`. |
| `voyage.py` | Simulator core: `VoyageConfig`, `Leg` (a fraction `f0->f1` of one lane `x->y`), `Voyage`, `plan_voyage`, `advance`, `seek`, `ship_state`, `split_at`, `reevaluate` (reroute from current position), `route_from` (Dijkstra from a virtual node on the current lane), objectives, risk index, `route_signature`. |
| `ops.py` | `Ops` controller - the ONLY code that mutates simulation state. One instance per browser session in `st.session_state["ops"]`. Disruptions, storms, congestion, config changes, playback (`tick`), demo (`run_demo`, `_demo_trigger`, `exit_demo`), forecast-informed hazards, `clone()` for previews, `snapshot()`, `last_change`. |
| `copilot.py` | Intent classification (`classify`), deterministic parser (`parse`), fixed op `SCHEMA` + `validate`, intent `gate`, `execute`, `preview`, `handle` pipeline. |
| `llm.py` | Optional `AnthropicProvider` (official SDK, `claude-opus-5-5`, JSON-schema structured output, refusal fallback beta); `get_provider`. Proposes ops only. |
| `intel.py` | Open-Meteo marine + wind fetch, `assess_weather` (match forecasts to simulated ETAs), route-bound staleness helpers, `operational_impact`; GDELT -> Google News RSS fallback, classification, `rank_news`, `advisories`, labelled demo headlines. Pure functions (testable with mocked `requests`). |
| `folium_map.py` | Leaflet maps: stress-test map, live base map (`build_ops_base_map`), per-tick voyage layer (`build_ops_layer`). Esri dark-gray tiles (keyless; CartoDB now needs a key). |
| `map_view.py` | Self-contained SVG maps: stress-test fallback and the offline live-ops map (`build_ops_map_html`). |
| `ui.py` | CSS + small HTML builders (topbar/status pill, KPI cards, live strip, timeline, recommendation card, comparison table, banner, methodology). |
| `data_engine.py`, `events.py` | Lane graph + risk zones; scenario stressing (`apply_events`), incl. Bab-el-Mandeb/Panama closures, fees, storm cells, port congestion. |
| `routing_engine.py`, `agents.py`, `ml_models.py` | Per-lane pricing (fuel/tolls/war-risk/time), vessel & cargo-owner profiles (container, tanker, bulk, LNG), fuel model. |
| `tests/` | `test_live_ops.py` (8 acceptance tests + more), `test_demo_mode.py`, `test_audit_fixes.py`, original data/ML/routing tests. |

## Non-negotiable rules (from the user's requirements)

1. **Honesty labels.** Vessel positions are SIMULATED (never AIS). ETA/fuel/cost/risk are MODEL ESTIMATES. Disruptions
   are SIMULATED SCENARIOS, never real closures. Open-Meteo / news data is LIVE DATA shown with source + fetch time.
   Demo headlines are labelled demo. No fabricated data, headlines, dates or sources; no unsupported accuracy claims
   (the fuel model's R² is fit to synthetic data only). Not for navigation or safety decisions.
2. **Headlines never close corridors.** News can only raise advisories / offer a button to *simulate* a scenario.
3. **One mutation path.** Dashboard widgets, the demo and the copilot all call `Ops` methods - no duplicated routing logic.
4. **Copilot safety.** Only explicit instructions change state. Questions -> read-only answers; hypotheticals and
   negations ("what if", "don't close…, just explain") -> PREVIEW on `ops.clone()`; ambiguous statements ("Suez closure")
   -> ask for clarification. The LLM may only propose ops from `SCHEMA`; they pass `validate` + `gate` like parser output.
   It never executes code.
5. **Geographic validity.** Ships stay on drawn lanes (no land crossings, no teleporting, no overland lanes for ships).
   Reroutes start at the ship's current position and preserve distance/time/cost already sailed. Only the *remaining*
   route is checked (Hormuz closure does not reroute Rotterdam -> Shanghai; a passed chokepoint does not matter). No
   feasible passage -> status BLOCKED, ship holds position (e.g. tanker inside the Gulf when Hormuz closes).
6. **Works offline and without keys**, including the demo. Network calls are cached (forecast 30 min, news 15 min),
   time out, back off on failure, and never run on animation frames.
7. **Session isolation.** Mutable state only in `st.session_state`; cached resources (model, base graph, priced graphs
   in `voyage._priced_graph` LRU) are read-only.
8. **Keep it lightweight and Streamlit-compatible** (local + free Streamlit Community Cloud). Don't swap frameworks.
9. **Don't push or deploy** until the user reviews. Commit only when asked (the user asked for commits of this work).

## How key mechanisms work

* **Rerouting** (`voyage.reevaluate`): split itinerary at `t` (`split_at` cuts the current leg exactly; canal queue
  time and tolls are tracked with `Leg.entered` / `queued_h`), check remaining lanes for closures/changed costs, also
  re-check when any lane *improves* (reopened strait), build candidates for Fastest/Cheapest/Safest/Balanced from a
  virtual node on the current lane (forward, or turn back along the lane), pick by the configured objective with a
  0.5% hold-course tolerance and a 2x penalty for "forced through" (storm/over-premium) routes. Two passes: strict,
  then relaxed soft blocks; hard closures, draft limits and avoided chokepoints never relax.
* **Objectives**: Fastest = hours; Cheapest = fuel + tolls + war-risk + ship operating cost; Safest = sea hours x
  (geo + 0.5 x weather risk); Balanced = weighted sum of the three, each normalised by the network-wide lane average.
* **Playback**: `Ops.tick()` advances by real elapsed seconds x speed (capped 3 s). Two fragments (`kpi_fragment`,
  `map_fragment`) run every 1 s while playing and both call `tick` (the second sees ~0 s). Arrival / demo trigger /
  block -> `st.rerun()` of the whole app.
* **Map without flicker**: base folium map cached per scenario settings in session; only the voyage `FeatureGroup` is
  pushed via `st_folium(feature_group_to_add=...)`. `st_folium` attaches that group to the cached map, so it is
  **detached after every call** (`del base._children["feature_group_*"]`) - otherwise the map script changes each tick
  and the map remounts. Zoom/pan survive playback.
* **Widget <-> state sync**: `sync_widgets()` rewrites widget keys from `ops` before widgets render each run; all
  widgets use `on_change` callbacks. Chat uses `st.chat_input(on_submit=...)` so copilot changes show immediately.
* **Weather staleness**: assessments carry `route_sig = voyage.route_signature(v)`; after a reroute an older forecast
  is never shown as current (`intel.reuse_or_unavailable`, `intel.current_for`).
* **Forecast-informed simulation** (opt-in toggle in the Weather tab): Open-Meteo points *ahead of the ship* with
  waves >= threshold become storm cells with source `open-meteo-forecast`, deduplicated and capped at 6 (accumulate,
  never replace -> no route oscillation); stale/unavailable data is never applied; removable in one click.
* **Demo** (`Ops.run_demo(animate=...)`): saves the user's session once (`_pre_demo`), plans RTM -> SHA container,
  sails to 50% of the East-Med -> Port Said lane (time-lapse at 50 h/s, or an instant labelled time jump), triggers the
  simulated Suez closure, records before/after in `ops.demo`. Replay is deterministic; `exit_demo` restores the session;
  scrubbing cancels a pending trigger; manual advance stops exactly at the trigger point.

## Build history (this thread)

1. **Handoff** from a claude.ai chat ("Hackathon 10/16" / "Hackathon project ideas"); claude.ai would not stay signed
   in to the in-app browser and Claude in Chrome was not connected, so the user pasted the handoff notes. Project found
   on disk, venv created, model trained, 24 original tests passed, Streamlit verified. Fixed the map: CartoDB tiles now
   require an API key -> switched to keyless Esri World Dark Gray. Git initialised; baseline commit `7e5d866`.
2. **Live ops upgrade** (commit `5799b73`): vessel simulator, mid-voyage rerouting, incident panel, copilot + optional
   LLM, Open-Meteo + news feeds, bulk/LNG vessels, fixed cruising speed, delays, objectives, 24 acceptance tests,
   README, requirements, `.gitignore`, secrets example. GDELT rate-limits this machine (429/timeouts) -> Google News
   RSS fallback added. Bugs found and fixed while verifying: storm-label formatting crash (made scenario changes
   atomic), clearing scenarios did not revert to Suez (added "improved lanes" re-check), map remount every tick (see
   detach trick above).
3. **Hackathon demo mode** + review zip (`dist/tideline-review.zip`, also copied to Downloads) for an external
   (ChatGPT) review.
4. **Pre-deployment audit** from that review, implemented in priority order:
   * Fix 1 (P0) copilot intent bug - all five reported failures ("What happens if Suez closes?", "Do not close Suez…",
     "Show news about a Suez closure", "What if fuel prices increase by 25%?", "Is Hormuz closed?") now leave state
     unchanged; previews run on a clone. Note: "The Strait of Hormuz has closed. Does that impact my shipment?" is now
     answered with a PREVIEW (impact question) instead of applying the scenario.
   * Fix 2/3 (P0) UI reorganised to the user's sketch: header with status pill -> **Voyage & Simulation | Alerts &
     Intelligence** (Weather / News tabs directly under alerts) -> 4 container-aware KPI cards -> **Live route map |
     Operations Copilot** -> **Rerouting decision & event timeline**. Sidebar = advanced settings only, collapsed by
     default. Text 12-15 px; no overflow verified at 1440 / 1280 / 768 / 375 px.
   * Fix 4 (P1) forecast-informed mode + route-bound weather staleness.
   * Fix 5 (P1) offline SVG live map (routes, ship, closures, storms, completed/remaining/superseded/alternatives);
     time-lapse indicator in the telemetry strip.
   * Fix 6 (P1) SIMULATED / MODEL ESTIMATE / LIVE DATA labels, rules vs LLM mode chip, methodology in an expander.
   * Fix 7 (P1) animated demo + "Instant demo", recommendation card + compact before/after, replay/exit.
   * Fix 8 (P0) clean-venv install from `requirements.txt` (version-bounded), full suite (79 passed), real
     `streamlit run` startup with no keys, Esri tiles confirmed loading (15/15), secret scan clean, `.gitignore`
     covers `.env*`, `secrets.toml`, `*.pkl`, `models/`, `*.log`, caches, `dist/`, `.venv`, `.claude/`. Fuel model now
     runs with `n_jobs=1` (removed ~76k joblib warnings; test run 72 s -> 34 s).

## Known limitations / possible next steps

* Offline SVG map re-renders each tick during playback (loses its own zoom); the interactive map keeps zoom.
* The Weather/News slot is briefly empty while a new forecast loads after a reroute.
* The router cannot "wait out" a storm; a severe Mediterranean storm diverts around the Cape.
* LLM mode is implemented but untested end-to-end (no key was available).
* GDELT is usually rate-limited from this machine; news typically comes from the Google News RSS fallback.
* Deployment: push to GitHub, create a Streamlit Community Cloud app with main file `app.py`, Python 3.12;
  optional secret `ANTHROPIC_API_KEY`. Waiting on the user's go-ahead.

## Conventions

* Match surrounding style: dataclasses, small pure functions, f-strings, comments only where the *why* is non-obvious.
* HTML builders in `ui.py` must return single-line HTML (Streamlit markdown turns indented HTML into code blocks).
* Escape `$` in text rendered with `st.markdown` (it starts LaTeX) - see `md()` in `app.py`.
* Every new state-changing feature goes through `Ops`, gets a timeline entry with a `source`, and a test.
* Run `python -m pytest -q` and check the app in a browser before claiming something works.
