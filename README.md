# Multi-Agent Maritime Routing & Supply-Chain Stress-Tester

Hackathon prototype (c0mpiled-14 · YC RFS: *New Operating Systems for the Physical World*).
A synthetic global shipping network, an ML fuel model, and agents with different economics. Trigger a
shock (Hormuz blockade, Suez toll x5, hurricane...) and watch a container ship carrying Apple goods and an
oil tanker carrying crude choose different routes, or fail to find one.

## Run

```bash
pip install networkx pandas scikit-learn streamlit folium streamlit-folium pytest
streamlit run app.py
```
The first launch trains the fuel model (~2 s) and caches it in `models/fuel_model.pkl`.
Without folium installed the app falls back to its built-in SVG map. Press **Simulate and Optimize Route** in the sidebar to apply changes.
Other commands: `pytest -v test_routing.py` (Hormuz open/blockaded tanker tests), `python ml_models.py` (train + print metrics), `python routing_engine.py` (CLI demo over
all presets), `python -m unittest discover -s tests -v`.

## Layout

| File | Role |
| --- | --- |
| `data_engine.py` | Synthetic NetworkX graph: 98 nodes, 111 lanes, 9 ports (Shanghai, Rotterdam, LA, Singapore, Dubai...). Edge attrs `base_distance_nm`, `weather_risk`, `geopolitical_risk`. Hard-coded bounding boxes (Hormuz, Red Sea, Bab-el-Mandeb, Malacca, Taiwan Strait, Cape of Good Hope...) give lanes inside them high baseline risk. |
| `ml_models.py` | 5,000 synthetic voyages -> RandomForest fuel model (R² ~0.98 vs ~0.90 linear), saved with `pickle`. |
| `agents.py` | Vessel profiles (Container, Tanker) x cargo owners (High-margin, Low-margin, Energy trader). |
| `events.py` | Event catalog, presets and `apply_events` (returns a stressed copy of the graph). |
| `routing_engine.py` | Loads `models/fuel_model.pkl` at import (physics fallback if missing); `calculate_edge_cost` per lane; `find_optimal_route` = Dijkstra on it; also batched engine, Dijkstra / A*, strict-then-forced two-pass feasibility, calm-vs-stressed comparison. |
| `folium_map.py` | Leaflet map: blue container routes, amber tanker routes, risk-coloured lanes, event zones. |
| `app.py`, `ui.py`, `map_view.py` | Streamlit UI (flat dark theme based on the Pelagic demo, no gradients), SVG map with pan/zoom. |
| `geo_assets.py`, `assets/` | Natural Earth coastlines + hand-built outlines for the Americas, used for the map and a land-crossing test. |

## Cost model

```
edge cost = ML_fuel_tonnes * fuel_price
          + toll_weight * toll
          + geopolitical_risk * insurance_sensitivity * days_exposed
          + travel_hours * delay_penalty_per_hour
```
Each agent also picks the best of three speeds per lane, so the fuel model's cubic speed curve matters.
`toll_weight` = operator tolerance x cargo-owner attitude (Apple 0.21 sponsors tolls; IKEA 1.5 resists).
Reported "All-in cost" uses plain dollars (no toll weighting). Insurance premium multiple is
`1 + 11 * peak_risk²`; containers refuse lanes above 3x, tankers above 4x, unless no route remains
("forced through"). Closures and draft limits (Panama 15.2 m, Suez 20.1 m) are never relaxed.

"Fuel saved" and "tolls avoided" compare the new route with **holding course**: the calm-sea route repriced
under the active events.

## Things to know

* All data and dollar figures are synthetic/illustrative. Lanes are simplified waypoint chains, not certified sea lanes.
* The hurricane only affects routes that cross its track (pick Cape Verde / Main development / Caribbean in the sidebar).
* Hormuz blockade: the tanker from Jebel Ali moves cargo overland to Fujairah ($1.5M, flagged in the UI).
