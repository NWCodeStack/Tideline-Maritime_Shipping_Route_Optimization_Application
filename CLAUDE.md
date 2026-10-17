# CLAUDE.md - Tideline project memory

Context for anyone (human or AI) picking up this repo. It condenses the full build thread: what the project is, what
was asked for, what was decided and why, how the code fits together, and what is still open.
User-facing docs live in `README.md` (refreshed in Phase 6); this file is
the engineering history and the house rules. Last updated 2026-10-16, after Phase 6.

## What this is

**Tideline** (previously called *Pelagic* / *Multi-Agent Maritime Routing & Supply-Chain Stress-Tester*) is a UC Berkeley
hackathon project (c0mpiled-14, YC RFS *Operating Systems for the Physical World*). It is a Python + Streamlit
"live AI shipping operations simulator":

* a synthetic global lane network (99 nodes, 113 lanes, real chokepoint geography, land-checked lanes);
* a random-forest fuel model trained on 5,000 **synthetic** voyages;
* a simulated vessel that moves along the lanes, with mid-voyage rerouting from its current position, a U-turn cost and
  a stay-or-change decision threshold;
* a decision score beyond fuel/tolls: ECA fuel switch, EU ETS, a carbon shadow price, value of time, per-port bunker
  prices, five objective presets, speed chosen per lane;
* a 0-100 risk score with bands, a caption and a breakdown that adds up;
* an Operations Copilot (rules-based command parser, optional Anthropic LLM assist, pasted news treated as data);
* live Open-Meteo marine weather and GDELT / Google News headlines (advisory only);
* a hazard overlay on the live map (security zones by risk band, port warnings);
* a one-click hackathon demo (simulated Suez closure -> Cape of Good Hope diversion);
* the original multi-agent stress test (container ship vs oil tanker) kept as a second top-level tab.

## State of the repo (read this first)

* Branch `main`, **nothing pushed, nothing deployed, no remote configured.** The user said "don't push yet" at every step.
* Tag `v1-demo` = the last commit before the upgrade (`d242ec6`), the safe fallback demo.
* Commits since then (all local):

| Commit | What |
| --- | --- |
| `6a30fd0` | Phase 1: re-plan from the ship's position, U-turn cost, stay-or-change threshold |
| `654895b` | Phase 2: presets, ECA / EU ETS / carbon lines, speed per leg, per-port fuel, cost breakdown |
| `b4688d3` | Phase 3: risk score bands, caption, breakdown that adds up |
| `7c7fbde` | Phase 4: persistent map beside five tabs, sidebar retired, Data sources tab |
| `228a819` | Phase 5A: honest infeasibility (nearest reachable port) |
| `8d15ec5` | Phase 5B: hazard overlay (zones by risk band, port warnings) |
| `b1a21be` | Phase 5C: pasted news reports are data, never instructions |
| `f21a833` | Phase 5 review fix: map legend explains the overlay |
| `78e1236`.. | Phase 6: render_ui split (page.py, tabs.py, live.py, ...), route_from helpers, geo.py, lint, README + CLAUDE.md refresh |

* Hidden checkpoints (not on any branch): `git for-each-ref refs/checkpoints` (one per phase); log in
  `.claude/checkpoints.log` (git-ignored). Phase tags `phase1`..`phase5` also exist.
* **279 tests pass, 95% coverage.** The review/checkpoint pattern after every phase: tests first, implement, full suite,
  lint (`ruff --select F,B`), `bandit -ll`, browser check, code review, commit locally.

## Commands

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (Python 3.14 was used; 3.12 untested locally)
pip install -r requirements.txt
streamlit run app.py                                    # app (trains models/fuel_model.pkl on first launch)
python -m pytest -q                                     # 279 tests (pytest.ini selects test_routing.py + tests/)
python app.py                                           # console stress-test demo
```
`TIDELINE_OFFLINE=1` disables all network calls (tests, offline demos). **No API key is required for anything**, and
the app is meant to stay free to run and deploy. Optional `ANTHROPIC_API_KEY` (env or `.streamlit/secrets.toml`, see
`.streamlit/secrets.toml.example`) only enables LLM assist; leave it unset on Streamlit Cloud to stay free.
Dev-only tools used in the venv (not in `requirements.txt`): `pytest-cov`, `ruff`, `bandit`, `pip-audit`, `openpyxl`.

## Page layout (Phase 4)

Header, "Latest reroute" banner, 4 KPI cards and the telemetry strip are full width. Below them two columns:

* **Left (never inside a tab, so switching tabs cannot reset it):** live route map (+ time slider + legend) and the
  Operations Copilot (chat, example buttons, "Simulate this scenario" button while a pasted-report proposal is pending).
* **Right, `st.tabs(key="right_tabs")`** (selection survives reruns): **Operate** (status line, From/To/Vessel,
  playback, demo buttons, compact incident picker `inc_picker`, "Storms, fees and more scenarios" expander),
  **Optimize** (preset, value of time, carbon price, fuel source; everything else under "Advanced"),
  **Route comparison** (decision card, options side by side with CO2 and risk band, cost breakdown, risk panel,
  checkpoint table, fuel-burn chart), **Alerts & intelligence** (alerts, Weather and News sub-tabs, risk legend, event
  timeline), **Data sources** (every input with basis/date/link, methodology).
* Top-level tabs stay `Live operations` and `Multi-agent stress test` (the stress test is NOT a right-hand tab).
* The sidebar is gone (empty). The first right-hand tab is named "Operate" to avoid two tabs called "Live operations".

## Architecture (where things live)

| File | Role |
| --- | --- |
| `app.py` | Streamlit entrypoint: `render_ui` (~55 lines of wiring), `console_demo()`. |
| `page.py` | `Page` context (session state, `Ops`, model, graph), widget callbacks, `sync_widgets`, cached resources. |
| `tabs.py` | `render_header`, `render_copilot`, `render_operate/optimize/compare/alerts/sources`, `render_intel_panel`; `RIGHT_TABS`. |
| `live.py` | `kpi_fragment`, `map_fragment` (re-run every `TICK_S` while playing), `HAS_FOLIUM`. |
| `intel_ui.py`, `stress_ui.py` | `render_weather`/`render_news`; `stress_tab` (the original stress test). |
| `geo.py` | Lane geometry (`node_ll`, `point_on_lane`, `bearing_deg`, `legs_polyline`), re-exported by `voyage`. |
| `voyage.py` | Simulator core: `VoyageConfig` (incl. `econ`, `change_course_threshold_usd`), `Leg` (fraction `f0->f1` of a lane; carries speed, ECA/ETS/CO2/carbon, risk signals, `turn` flag), `Voyage` (`recommendation`), `plan_voyage`, `split_at`, `reevaluate`, `route_from`, `arrival_lane`, `reprice`, `nearest_reachable_port`, `reachable_ports`, `_worth_changing`, `_change_headline`, objectives, `route_signature`. |
| `ops.py` | `Ops` controller, the ONLY code that mutates simulation state. Disruptions, storms, congestion, config, `set_economics`, `apply_preset`, playback, demo, `clone()`, `snapshot()`, `proposal` (pending pasted-report scenario). |
| `economics.py` | `Economics` (frozen dataclass), `PRESETS`, `ECA_BOXES`, `PORT_FUEL_PRICES`, `eca_fraction`, `ets_applies`, `effective_fuel_price`, `breakdown` (cost lines with a data basis). |
| `risk.py` | Risk score: `leg_risk` (noisy-OR), `route_risk` (time-weighted, parts add up), `band`, `explain`, `CAPTION`, `BASIS`. |
| `panels.py` | Pure helpers behind the tabs and map: `INCIDENTS`, `incident_changes`, `options_rows`, `fuel_burn_series`, `checkpoint_rows`, `status_line`, `hazard_zones`, `hazard_tooltip`, `port_warnings`. |
| `sources.py` | `plan_sources`: rows of the Data sources tab; `BASES` = live / published / estimated / sample / simulated / user setting / unavailable. |
| `newsparse.py` | Pasted-report analysis: `looks_like_report`, `is_confirmation`, `analyse` -> `Finding` (regions, wording, on-route, at most one proposal). Rules only, no key. |
| `copilot.py` | `classify`, `parse`, `SCHEMA`/`validate`, `gate`, `execute`, `preview`, `handle`; `_report_reply` / `_apply_proposal` for pasted reports. |
| `llm.py` | Optional `AnthropicProvider` (proposes ops only). |
| `intel.py` | Open-Meteo fetch/assess, GDELT -> Google News RSS fallback, region rules, advisories. |
| `folium_map.py` | Leaflet maps: `build_ops_base_map(hazards=...)` (cached, rebuilt only when scenarios change), `build_ops_layer(port_warnings=...)` (per tick). Esri dark-gray tiles (keyless). |
| `map_view.py` | Offline SVG maps (no hazard overlay). |
| `ui.py` | CSS + single-line HTML builders (cards, banner, decision, status line, risk chip/panel, cost breakdown). |
| `data_engine.py`, `events.py` | Lane graph, risk zones, scenario stressing. |
| `routing_engine.py`, `agents.py`, `ml_models.py` | Per-lane pricing (`price_edge_batch` takes `econ`), vessel/owner profiles, fuel model. |
| `tests/` | One file per phase (`test_phase1_reroute`, `..2_economics`, `..3_risk`, `..4_layout` (AppTest), `..4_panels`, `..5_infeasible`, `..5_hazards`, `..5_pasted`) plus `test_coverage_gaps`, `test_live_ops`, `test_demo_mode`, `test_audit_fixes`, originals. |

## Non-negotiable rules (from the user's requirements)

1. **Honesty labels.** Vessel positions are SIMULATED (never AIS). ETA/fuel/cost/risk are MODEL ESTIMATES. Disruptions
   are SIMULATED SCENARIOS, never real closures. Open-Meteo / news data is LIVE DATA with source + fetch time. ECA/ETS
   are ESTIMATED, port bunker prices SAMPLE, the carbon price a USER SETTING. No fabricated data, headlines, dates or
   sources; no unsupported accuracy claims (the fuel model's R² is fit to synthetic data only). No port-draft data
   exists, so there are **no "too shallow" markers**. Not for navigation or safety decisions.
2. **Headlines never close corridors or change the risk score.** News raises advisories or proposes a scenario to simulate.
3. **One mutation path.** Widgets, the demo and the copilot all call `Ops` methods.
4. **Copilot safety.** Only explicit instructions change state. Questions are read-only; hypotheticals and negations
   are PREVIEWs on `ops.clone()`; ambiguous statements ask for clarification. The LLM may only propose ops from `SCHEMA`.
5. **Pasted text is data, never instructions.** A paste (long multi-sentence text with news cues, or a `News:` prefix)
   gets a labelled PREVIEW and changes nothing; only a short, exact "Simulate the proposed scenario" applies the stored
   proposal, and that phrase inside a long text confirms nothing. A short paste with no prefix is indistinguishable from
   a typed command (documented limitation).
6. **Geographic validity.** Ships stay on drawn lanes; reroutes start at the ship's position and keep what was already
   sailed and spent; only the remaining route is checked; no feasible passage -> BLOCKED with the nearest reachable port named.
7. **Works offline and without keys**, including the demo. Network calls are cached, time out, back off, never run on
   animation frames.
8. **Session isolation.** Mutable state only in `st.session_state`; cached resources are read-only.
9. **Lightweight and Streamlit-compatible** (local + free Streamlit Community Cloud). Don't swap frameworks.
10. **Don't push or deploy** until the user says so. Commit only when asked. No local paths or personal emails in
    tracked files (a public repo is planned).

## How key mechanisms work

* **Rerouting** (`voyage.reevaluate`): split the itinerary at `t`; candidates for Fastest/Cheapest/Safest/Balanced from a
  virtual node on the current lane (forward, or turn back). **Root cause of the Panama bug** (ship turned back toward
  Europe after a reopen): the lane graph had no South American Atlantic coast, so the Cape Horn detour crossed the
  Atlantic via West Africa. Fixed with offshore Guianas lanes (`ANN-GUI1-SAT4`).
* **Turns**: reversing costs `TURN_PENALTY_H` (4 h) whether the ship is mid-lane, sitting at a node (`arrival_lane`), or
  goes forward to the next node and straight back. No turning inside a canal; a canal transit in progress completes
  even if the canal closes. A mid-lane turn wait keeps the ship's lane position (`split_at`, `ship_state`).
* **Stay or change** (`_worth_changing`): leave the committed route only if the alternative beats it by
  `change_course_threshold_usd` (default $25k), judged in the active objective (fastest = hours saved x delay cost;
  cheapest = expenses; balanced/safest = all-in cost; carbon shadow price counted). "Safest" also switches on a 3-point
  risk-score drop. Bypassed for explicit `force`/`via` requests and when the objective itself changed (which now
  re-plans). Output: `Voyage.recommendation` = stay / change_course / blocked and a one-line `CHANGE COURSE NOW: turning
  north-west from off the Guianas, saves 13.6 d and $5.5M vs continuing via Cape Horn` headline.
* **Decision score** (`economics.py`, `routing_engine.price_edge_batch`): cash = fuel + tolls + war-risk + time + ECA + EU
  ETS; score = cash + (value-of-time - 1) x time cost + carbon price x CO2. Speed is the best of the vessel's options
  per lane on that score; cash totals exclude the carbon shadow price. Presets (Balanced/Cheapest/Fastest/Safest/
  Greenest) set objective + value of time + carbon price; moving a slider reads as "Custom". Default RTM->SHA cost rose
  from $10.08M to $10.62M because ECA and ETS are on by default.
* **Risk score** (`risk.py`): per leg noisy-OR of security (base + scenario), weather (half the lane sea-state severity)
  and traffic (simulated canal delay only); time-weighted over the route; score = round(100 x risk); bands Low <20,
  Guarded <40, Elevated <60, High <80, Severe. Parts are attributed proportionally so they sum exactly. The routing
  objective for "Safest" still minimises risk-hours (`Leg.risk`); the score shows average intensity.
* **Hazard overlay** (Phase 5B): `panels.hazard_zones` = current level per security zone (stressed vs base lane samples),
  drawn in the cached base map; `panels.port_warnings` (unreachable via `voyage.reachable_ports`, or congested) in the
  per-tick layer because it depends on the ship's position (1.3 ms per tick; zones 21 ms, only on scenario change).
* **Pasted news** (Phase 5C): `copilot.handle` checks confirmation, then `looks_like_report`, before normal parsing;
  `_report_reply` stores `ops.proposal`; `_apply_proposal` runs exactly that op. `Ops.proposal` is cleared on reset, new
  voyage and clone. Closure only when the text says closed/blocked/blockade, otherwise Red Sea escalation 0.5 or Taiwan 0.4.
* **Playback**: `Ops.tick()` by real elapsed seconds x speed; `kpi_fragment` and `map_fragment` run every 1 s while playing.
* **Map without flicker**: base map cached per scenario settings; only the voyage `FeatureGroup` is pushed; it is
  detached after every `st_folium` call. Tab switches never rerun. Known quirk: the first one or two reruns after a
  page load recreate the map once (same on the pre-Phase-4 commit).
* **Widget <-> state sync**: `sync_widgets()` rewrites widget keys from `ops` each run; widgets use `on_change` callbacks.
* **Demo**, **weather staleness**, **forecast-informed simulation**: unchanged from the original build (see git history).

## Build history

1. **Handoff** from a claude.ai chat ("Hackathon 10/16"); project found on disk, venv created, model trained, 24
   original tests passed. Map tiles switched to keyless Esri. Baseline commit `7e5d866`.
2. **Live ops upgrade** (`5799b73`): vessel simulator, rerouting, incident panel, copilot + optional LLM, Open-Meteo + news,
   bulk/LNG vessels, README, requirements, secrets example. Google News RSS fallback because GDELT rate-limits.
3. **Hackathon demo mode** + review zip for an external review (`f3e70e3`), then the **pre-deployment audit** fixes
   (copilot intent bug, UI reorganisation, forecast-informed mode, offline SVG map, honesty labels, animated demo,
   clean-venv install, secret scan).
4. **Coverage + release hygiene** (`d242ec6`, 2026-10-16): 35 coverage-gap tests (87% -> 92%), MIT LICENSE, local path
   scrubbed from this file, `.coverage` ignored; AgentShield scan (secrets/permissions/hooks/MCP 100; B only because this
   file lacks prompt-defence wording), pip-audit clean, bandit clean, clean-venv startup verified with no key.
   `v1-demo` tag.
5. **Upgrade from `tideline-upgrade-prompt.md`** (a teammate's Pelagic prototype was the reference; its newer copy has
   the Python modules; it is kept outside the repo). Phase 0 plan approved with these user decisions: stress test stays
   a top-level tab; Phase 2 first cut (per-port fuel, speed per leg, ECA/ETS/carbon lines, presets), bunkering stops
   deferred; tag `v1-demo` locally; no Playwright; fuel-burn chart instead of fuel-on-board; copilot docked under the
   map; `render_ui` split deferred to Phase 6.
   * Phase 1 (23 tests): see "Rerouting", "Turns", "Stay or change".
   * Phase 2 (26 tests): see "Decision score". Per-port prices follow Pelagic's normal-times table where it has the port.
   * Phase 3 (28 tests): see "Risk score". The caption is the user's wording verbatim ("weather beyond the ship's
     limits" is not strictly true yet: weather is sea-state severity, not checked against a wave limit; the basis column says so).
   * Phase 4 (36 tests): see "Page layout". Verified in the built-in browser: same map element and zoom through every tab
     switch, tab memory across reruns, no horizontal overflow at 375 and 768 px.
   * Phase 5 (48 tests): A honest infeasibility, B hazard overlay, C pasted news. Deferred with reasons: "Today vs Normal"
     (needs verified, dated, sourced prices; Pelagic's snapshot values could not be verified offline) and the fleet
     picker (4 vessels x 5 owners already exist; named ships need fuel curves the model lacks).
6. **Code reviews** found and fixed: Phase 1 (queue time double-counted in the U-turn cost; the dollar threshold ignored
   non-balanced objectives; changing the objective never re-planned), Phase 2 (threshold ignored the carbon price),
   Phase 3 (arrived voyages reported a 0 risk; untested panel/tooltip), Phase 5 (overlay was colour-only: legend added).

## Known limitations / next steps

**Phase 6 (done locally, commits `Phase 6a`..): what was and was not done**
1. `render_ui` is now ~55 lines in `app.py` (131 lines total). The page lives in `page.py` (`Page` context with the
   callbacks and widget sync), `tabs.py` (header, copilot, five tab renderers), `live.py` (KPI + map fragments),
   `intel_ui.py`, `stress_ui.py`. `tests/test_phase6_structure.py` guards function length (< 120) and nesting (<= 4).
   `route_from` was split into `_routing_graph` / `_legs_from_path` / `_no_passage`, and lane geometry moved to `geo.py`
   (re-exported from `voyage`). **Not done:** `voyage.py` is still ~1,050 lines and `reevaluate` ~125 lines; splitting
   them risked circular imports (ship-state code needs `Voyage`/`Leg`), so it is left for a later pass.
2. `ruff --select F,B` is clean (unused imports/variables, loop variables, `raise ... from None`, `zip(strict=False)`,
   which keeps the old behaviour). No dead-code tool beyond ruff was run. `bandit -ll` is NOT clean: 3 medium notes that
   pre-date Phase 6 (`ET.fromstring` on the Google News RSS in `intel.py`; `pickle.load` of the locally trained model in
   `ml_models.py` and `routing_engine.py`). Not fixed.
3. `README.md` rewritten for the current layout, with the decision score, risk caption and bands, a "What is simulated"
   table and the Panama close/reopen script. The Panama script's timings (close ~day 10, reopen 4 days later) come from
   `tests/test_phase1_reroute.py`; the script was not replayed by hand in the browser.
4. Acceptance check: Panama close/reopen turns toward Panama (test `TestReopenTurnsTowardPanama`); in the built-in
   browser the same map iframe survived all five right-hand tab switches (it is recreated after a scenario change, as
   before); risk scores show a band and the caption; every cost line shows its basis; no API key needed; tests pass.

**Open items and known limits**
* Not deployed. When approved: push to GitHub (`gh` is not installed here: create an empty repo on github.com, add the
  remote), then Streamlit Community Cloud with main file `app.py`, Python 3.12 (untested locally; only 3.14 here),
  no secrets. First launch trains the fuel model in ~20-30 s on the cloud.
* Deferred: bunkering stops and tank/reserve hard rules, deadlines (so "what if we arrive a day later?" is unsupported),
  Pareto set, "Today vs Normal" snapshot, fleet picker, port-draft markers.
* Weather risk is not checked against a ship wave limit; traffic risk is only the simulated canal delay.
* Offline SVG map has no hazard overlay and re-renders each tick (loses its own zoom).
* The router cannot "wait out" a storm; the weather/news slot is briefly empty after a reroute; GDELT is usually
  rate-limited here (Google News RSS fallback); LLM mode is untested end to end (no key).
* Every full rerun executes all five tab bodies (Streamlit has no lazy tabs), including the options solve.
* The right-hand tab strip overflows below ~800 px (last label scrolls); the 1280/1440 px widths were not re-tested.
* A scratch folder named `tl_prev` at the root of the dev machine's C: drive (a copy of an older commit used for a
  before/after test) can be deleted by hand; the built-in safety check blocked removing it from the session.

## Working notes for AI sessions in this repo

* Hooks in this environment: a "fact-forcing gate" refuses the first Write/Edit of a file until you state the callers,
  what it supersedes and the user's instruction (and a real Grep/Glob for new files); it also asks for facts before
  deletes. State them in text, then retry the same call. Do not batch several first-edits in one parallel message: some
  get blocked while others apply. A built-in safety check refuses `rm -rf` of non-project folders and must not be
  worked around.
* `AppTest.from_file` needs an absolute path (relative paths resolve against the test file). Headless layout tests live
  in `tests/test_phase4_layout.py`; what they cannot see (the map element surviving tab switches) was checked in the
  built-in browser with a marker on the Leaflet container.
* Browser checks: Streamlit element refs go stale after every rerun; re-`find` before clicking.
* Windows: no bash-only tricks in scripts; pip installs can hit the long-path limit under the scratchpad (use a short path).

## Conventions

* Match surrounding style: dataclasses, small pure functions, f-strings, comments only where the *why* is non-obvious.
* HTML builders in `ui.py` / `panels.py` must return single-line HTML and escape anything variable.
* Escape `$` in text rendered with `st.markdown` (LaTeX) - see `md()` in `app.py`.
* Every new state-changing feature goes through `Ops`, gets a timeline entry with a `source`, and a test; write the
  test first (one `tests/test_phaseN_*.py` per feature, unittest style).
* Run `python -m pytest -q` and check the app in a browser before claiming something works.
