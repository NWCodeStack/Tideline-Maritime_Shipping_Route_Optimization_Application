"""
map_view.py - renders the maritime network and chosen routes as one self-contained HTML/SVG page.

Streamlit embeds the string returned by ``build_map_html`` with ``components.html``. No external
libraries or tiles: land is the Natural Earth polygons plus hand-built outlines from ``geo_assets``,
reprojected to Mercator. Pan with drag, zoom with the wheel or the +/- buttons.
Visual language is deliberately flat: solid fills, 1-2 px strokes, no gradients, glows or blurs.
"""
from __future__ import annotations

import html
import json

from data_engine import GEO_ZONE, RED_SEA_ZONES
from events import HURRICANE_TRACKS, normalise
from geo_assets import WORLD_H, WORLD_W, land_svg_markup, project

SEA = "#0a1a28"
LAND = "#1c3144"
GRID = "#10253a"
LANE = "#2b4459"
AMBER = "#f5bf75"
RED = "#fe7d81"
STORM = "#a78bfa"


def _split_dateline(coords):
    """Split a [(lat, lon)] polyline wherever it jumps across the antimeridian."""
    parts, cur, prev = [], [], None
    for lat, lon in coords:
        if prev is not None and abs(lon - prev) > 180:
            parts.append(cur)
            cur = []
        cur.append((lat, lon))
        prev = lon
    if cur:
        parts.append(cur)
    return parts


def _poly_d(coords) -> str:
    d = []
    for part in _split_dateline(coords):
        pts = [project(la, lo) for la, lo in part]
        if len(pts) >= 2:
            d.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts))
    return " ".join(d)


def _zone_rect(key: str, cls: str) -> str:
    z = GEO_ZONE[key]
    x0, y0 = project(z.lat_max, z.lon_min)
    x1, y1 = project(z.lat_min, z.lon_max)
    return (f'<rect class="zone {cls}" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}">'
            f"<title>{html.escape(z.label)}</title></rect>")


def build_ops_map_html(G, settings: dict, done_pts, ahead_pts, superseded=(), alternatives=(), blocked=(),
                       ship=None, ship_status: str = "", route_color: str = "#58aaff", height: int = 500,
                       contract: dict | None = None) -> str:
    """Offline live-operations map (no tiles, no folium): network, scenario zones, completed / remaining /
    superseded / alternative / blocked routes, storm cells, and the SIMULATED ship position. With ``contract``
    (built with ``project=geo_assets.project``) the browser animates the ship between server syncs."""
    import math
    s = normalise(settings)
    over = []
    for name, pts in alternatives:
        if len(pts) > 1:
            over.append(f'<path class="alt" d="{_poly_d(pts)}"><title>Alternative considered: {html.escape(name)}</title></path>')
    for pts in superseded:
        if len(pts) > 1:
            over.append(f'<path class="sup" d="{_poly_d(pts)}"><title>Previous route (superseded)</title></path>')
    if len(blocked) > 1:
        over.append(f'<path class="blk" d="{_poly_d(blocked)}"><title>Planned route - BLOCKED</title></path>')
    if len(ahead_pts) > 1:
        over.append(f'<path class="route" stroke="{route_color}" d="{_poly_d(ahead_pts)}"><title>Remaining route</title></path>')
    if len(done_pts) > 1:
        over.append(f'<path class="done" d="{_poly_d(done_pts)}"><title>Completed track</title></path>')
    for cell in s.get("storm_cells", []):
        cx, cy = project(cell["lat"], cell["lon"])
        r = cell["radius_nm"] / 60.0 * (WORLD_W / 360.0) / max(math.cos(math.radians(cell["lat"])), 0.2)
        over.append(f'<circle class="zone storm" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"><title>Storm '
                    f'{html.escape(cell.get("label", ""))} ({cell["wave_m"]:g} m) - simulated scenario</title></circle>')
    for key in ("suez", "panama", "hormuz", "bab_el_mandeb"):
        on = {"suez": s.get("suez_blockage"), "panama": s.get("panama_closure"), "hormuz": s.get("hormuz_blockade"),
              "bab_el_mandeb": s.get("bab_el_mandeb_closure")}[key]
        if on:
            lat, lon = {"suez": (30.5, 32.4), "panama": (9.1, -79.7), "hormuz": (26.5, 56.5),
                        "bab_el_mandeb": (12.6, 43.4)}[key]
            x, y = project(lat, lon)
            over.append(f'<g class="xmark"><circle cx="{x:.1f}" cy="{y:.1f}" r="9"/><path d="M{x - 5:.1f},{y - 5:.1f} '
                        f'L{x + 5:.1f},{y + 5:.1f} M{x + 5:.1f},{y - 5:.1f} L{x - 5:.1f},{y + 5:.1f}"/>'
                        f'<title>{key.replace("_", "-").title()} CLOSED (simulated scenario)</title></g>')
    extra = ""
    if contract is not None:                  # animated: the browser script drives the ship, tip and completed track
        import anim
        over.append('<path id="tl-done" class="done"><title>Completed track</title></path>'
                    '<path id="tl-tip" class="done"/>'
                    '<g id="tl-ship"><g id="tl-rot"><path class="ship" fill="#ffffff" d="M0,-11 L7,9 L0,5 L-7,9 Z"/></g>'
                    '<text class="shiplabel" x="12" y="-10">SIMULATED POSITION</text></g>')
        extra = anim.svg_script(anim.contract_json({**contract, "world_w": WORLD_W}))
    elif ship is not None:
        x, y = project(ship.lat, ship.lon)
        fill = {"blocked": RED, "arrived": "#4de0c8"}.get(ship_status, "#ffffff")
        over.append(f'<g transform="translate({x:.1f},{y:.1f}) rotate({ship.course_deg:.0f})"><path class="ship" fill="{fill}" '
                    f'd="M0,-11 L7,9 L0,5 L-7,9 Z"/></g><text class="shiplabel" x="{x + 12:.1f}" y="{y - 10:.1f}">'
                    f'SIMULATED POSITION</text>')
    legend = ('<span><i style="background:#4de0c8"></i>completed</span>'
              f'<span><i style="background:{route_color}"></i>remaining</span>'
              '<span class="k"><b style="color:#829aac"></b>superseded</span>'
              f'<span class="k"><b style="color:{AMBER}"></b>alternative</span>'
              f'<span class="k"><b style="color:{RED}"></b>blocked / closed</span>'
              f'<span class="k"><b style="color:{STORM}"></b>storm</span>')
    return build_map_html(G, [], settings, height=height, overlay="".join(over), legend_html=legend, extra_script=extra)


def build_map_html(G, routes: list[dict], settings: dict | None = None, height: int = 520, overlay: str = "",
                   legend_html: str | None = None, extra_script: str = "") -> str:
    """``routes`` items: {label, color, result (RouteResult), hold (RouteResult | None)}."""
    s = normalise(settings)

    lanes = []
    for u, v, d in G.edges(data=True):
        if d["mode"] == "overland":
            cls = "lane overland"
        elif d["blocked"]:
            cls = "lane blocked"
        elif d["weather_peak"] >= 0.6:
            cls = "lane stormy"
        elif d["geopolitical_peak"] >= 0.3:
            cls = "lane risky"
        else:
            cls = "lane"
        coords = list(zip(d["sample_lat"].tolist(), d["sample_lon"].tolist(), strict=False))
        tip = (f"{G.nodes[u]['name']} - {G.nodes[v]['name']} | geopolitical {d['geopolitical_risk']:.2f}, "
               f"weather {d['weather_risk']:.2f}" + (" | CLOSED" if d["blocked"] else ""))
        lanes.append(f'<path class="{cls}" d="{_poly_d(coords)}"><title>{html.escape(tip)}</title></path>')

    zones = []
    if s["hormuz_blockade"]:
        zones += [_zone_rect("hormuz", "red"), _zone_rect("persian_gulf", "red")]
    if s["red_sea_escalation"] > 0:
        zones += [_zone_rect(k, "amber") for k in RED_SEA_ZONES]
    if s["taiwan_tension"] > 0:
        zones += [_zone_rect("taiwan_strait", "amber"), _zone_rect("south_china_sea", "amber")]
    if s["atlantic_hurricane"] > 0:
        import math
        label, clat, clon, radius_nm = HURRICANE_TRACKS[s["hurricane_track"]]
        cx, cy = project(clat, clon)
        r = radius_nm / 60.0 * (WORLD_W / 360.0) / math.cos(math.radians(clat))
        zones.append(f'<circle class="zone storm" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"><title>Hurricane: '
                     f"{html.escape(label)}</title></circle>")

    route_svg = []
    for r in routes:
        res, hold = r["result"], r.get("hold")
        if hold is not None and hold.feasible and res.feasible and hold.path != res.path:
            route_svg.append(f'<path class="hold" stroke="{r["color"]}" d="{_poly_d(hold.coords)}"/>')
        if res.feasible:
            route_svg.append(f'<path class="route" stroke="{r["color"]}" d="{_poly_d(res.coords)}">'
                             f'<title>{html.escape(r["label"])}</title></path>')

    ports = []
    for _n, d in G.nodes(data=True):
        if d["kind"] != "port":
            continue
        x, y = project(d["lat"], d["lon"])
        ports.append(f'<circle class="port" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'
                     f'<text class="plabel" x="{x + 9:.1f}" y="{y + 4:.1f}">{html.escape(d["name"])}</text>')

    labels = []
    for _key, (lat, lon, text) in {
        "hormuz": (27.9, 56.3, "Hormuz"), "suez": (31.4, 33.2, "Suez"), "bab": (11.0, 43.4, "Bab-el-Mandeb"),
        "malacca": (4.7, 99.5, "Malacca"), "panama": (8.2, -79.2, "Panama"),
        "cape": (-36.5, 20.0, "Cape of Good Hope"), "gib": (36.6, -5.4, "Gibraltar"),
    }.items():
        x, y = project(lat, lon)
        labels.append(f'<text class="clabel" x="{x:.1f}" y="{y:.1f}">{text}</text>')

    legend = "".join(
        f'<span><i style="background:{r["color"]}"></i>{html.escape(r["label"])}</span>' for r in routes)
    legend_block = (f'<div class="legend">{legend_html}</div>' if legend_html is not None else
                    f'<div class="legend">{legend}<span class="k"><b style="color:{AMBER}"></b>high war-risk</span>'
                    f'<span class="k"><b style="color:{STORM}"></b>storm</span><span class="k"><b style="color:{RED}">'
                    f'</b>closed</span><span class="k"><b style="color:#829aac"></b>previous route</span></div>')
    grid = "".join(
        f'<line x1="{project(0, lon)[0]:.1f}" y1="0" x2="{project(0, lon)[0]:.1f}" y2="{WORLD_H:.0f}"/>'
        for lon in range(-150, 181, 30)
    ) + "".join(
        f'<line x1="0" y1="{project(lat, 0)[1]:.1f}" x2="{WORLD_W:.0f}" y2="{project(lat, 0)[1]:.1f}"/>'
        for lat in range(-45, 66, 15))

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:{SEA};font-family:'DM Sans',system-ui,Arial,sans-serif;overflow:hidden}}
#wrap{{position:relative;height:{height}px;border:1px solid #21374b;border-radius:10px;overflow:hidden;background:{SEA}}}
svg{{width:100%;height:100%;display:block;cursor:grab;touch-action:none;user-select:none}}svg:active{{cursor:grabbing}}
.grid line{{stroke:{GRID};stroke-width:1}}
.land{{fill:{LAND}}}.land .sea{{fill:{SEA}}}
.lane{{fill:none;stroke:{LANE};stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}}
.lane.risky{{stroke:{AMBER};opacity:.55}}.lane.stormy{{stroke:{STORM};opacity:.6}}
.lane.blocked{{stroke:{RED};stroke-dasharray:6 6;stroke-width:2.2}}.lane.overland{{stroke:#8ba5b8;stroke-dasharray:2 5}}
.zone{{stroke-width:1.5;stroke-dasharray:6 5}}
.zone.red{{fill:#fe7d8126;stroke:{RED}}}.zone.amber{{fill:#f5bf7524;stroke:{AMBER}}}.zone.storm{{fill:#a78bfa1f;stroke:{STORM}}}
.route{{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}}
.hold{{fill:none;stroke-width:2;stroke-dasharray:2 7;stroke-linecap:round;opacity:.8}}
.port{{fill:#e8f3fa;stroke:{SEA};stroke-width:2}}
.plabel{{font:700 13px 'DM Sans',sans-serif;fill:#e8f3fa;paint-order:stroke;stroke:{SEA};stroke-width:4}}
.clabel{{font:600 11px 'DM Sans',sans-serif;fill:#6f8da1;paint-order:stroke;stroke:{SEA};stroke-width:3;letter-spacing:.04em}}
.legend{{position:absolute;left:12px;bottom:12px;background:#0d1c2c;border:1px solid #21374b;border-radius:8px;padding:8px 10px;
 font-size:11px;color:#a9c3d0;display:flex;gap:14px;flex-wrap:wrap;max-width:75%}}
.legend i{{display:inline-block;width:18px;height:4px;margin-right:6px;vertical-align:middle;border-radius:2px}}
.legend .k{{color:#829aac}}.legend .k b{{display:inline-block;width:14px;height:0;border-top:2px dashed;margin-right:5px;vertical-align:middle}}
.done{{fill:none;stroke:#4de0c8;stroke-width:4.5;stroke-linecap:round;stroke-linejoin:round}}
.sup{{fill:none;stroke:#829aac;stroke-width:2.5;stroke-dasharray:6 8;stroke-linecap:round}}
.alt{{fill:none;stroke:{AMBER};stroke-width:2;stroke-dasharray:1 7;stroke-linecap:round;opacity:.7}}
.blk{{fill:none;stroke:{RED};stroke-width:3;stroke-dasharray:4 6;stroke-linecap:round}}
.xmark circle{{fill:#39232f;stroke:{RED};stroke-width:2}}.xmark path{{stroke:{RED};stroke-width:2.5}}
.ship{{stroke:{SEA};stroke-width:1.5}}
.shiplabel{{font:800 11px 'DM Sans',sans-serif;fill:#e8f3fa;letter-spacing:.08em;paint-order:stroke;stroke:{SEA};stroke-width:4}}
.tools{{position:absolute;right:12px;top:12px;display:flex;gap:6px}}
.tools button{{width:30px;height:30px;border:1px solid #335068;background:#132738;color:#a8c7d4;border-radius:8px;font-size:15px;cursor:pointer}}
.tools button:hover{{color:#fff;background:#1a3348}}
</style></head><body><div id="wrap">
<svg id="m" viewBox="0 0 {WORLD_W:.0f} {WORLD_H:.0f}" preserveAspectRatio="xMidYMid slice">
<rect width="{WORLD_W:.0f}" height="{WORLD_H:.0f}" fill="{SEA}"/>
<g class="grid">{grid}</g>{land_svg_markup()}
<g>{''.join(lanes)}</g><g>{''.join(zones)}</g><g>{''.join(labels)}</g><g>{''.join(route_svg)}</g><g>{''.join(ports)}</g>
<g>{overlay}</g>
</svg>
<div class="tools"><button id="zi" title="Zoom in">+</button><button id="zo" title="Zoom out">&minus;</button><button id="zr" title="Reset">&#x2922;</button></div>
{legend_block}</div>
<script>
const svg=document.getElementById('m'),W={WORLD_W:.0f},H={WORLD_H:.0f},KEY={json.dumps('tl-svg-view' if extra_script else '')};
const state={json.dumps({"x": 0, "y": 0, "w": WORLD_W, "h": WORLD_H})};
function fit(){{const r=svg.getBoundingClientRect(),ar=r.width/r.height;
 let w=W,h=W/ar; if(h>H){{h=H;w=H*ar}} state.w=w;state.h=h;state.x=(W-w)/2;state.y=(H-h)/2;apply()}}
function apply(){{state.w=Math.min(Math.max(state.w,140),W);state.h=state.w*(svg.getBoundingClientRect().height/svg.getBoundingClientRect().width);
 state.x=Math.min(Math.max(state.x,0),W-state.w);state.y=Math.min(Math.max(state.y,0),Math.max(H-state.h,0));
 svg.setAttribute('viewBox',state.x+' '+state.y+' '+state.w+' '+state.h);
 window.__tlScale=svg.getBoundingClientRect().width/state.w;
 if(KEY){{try{{sessionStorage.setItem(KEY,JSON.stringify([state.x,state.y,state.w]))}}catch(e){{}}}}}}
function zoom(f,cx,cy){{const r=svg.getBoundingClientRect();cx=cx??r.width/2;cy=cy??r.height/2;
 const px=state.x+cx/r.width*state.w,py=state.y+cy/r.height*state.h;const nw=state.w/f;
 const nh=nw*(r.height/r.width);state.x=px-cx/r.width*nw;state.y=py-cy/r.height*nh;state.w=nw;apply()}}
svg.addEventListener('wheel',e=>{{e.preventDefault();const r=svg.getBoundingClientRect();zoom(e.deltaY<0?1.2:1/1.2,e.clientX-r.left,e.clientY-r.top)}},{{passive:false}});
let drag=null;svg.addEventListener('pointerdown',e=>{{drag={{x:e.clientX,y:e.clientY,sx:state.x,sy:state.y}};svg.setPointerCapture(e.pointerId)}});
svg.addEventListener('pointermove',e=>{{if(!drag)return;const r=svg.getBoundingClientRect();
 state.x=drag.sx-(e.clientX-drag.x)/r.width*state.w;state.y=drag.sy-(e.clientY-drag.y)/r.height*state.h;apply()}});
svg.addEventListener('pointerup',()=>drag=null);
document.getElementById('zi').onclick=()=>zoom(1.4);document.getElementById('zo').onclick=()=>zoom(1/1.4);document.getElementById('zr').onclick=fit;
window.addEventListener('resize',fit);
let saved=null;if(KEY){{try{{saved=JSON.parse(sessionStorage.getItem(KEY))}}catch(e){{}}}}
if(saved&&saved.length===3&&saved.every(Number.isFinite)){{state.x=saved[0];state.y=saved[1];state.w=saved[2];apply()}}else{{fit()}}
</script>{f'<script>{extra_script}</script>' if extra_script else ''}</body></html>"""
