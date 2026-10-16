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


def build_map_html(G, routes: list[dict], settings: dict | None = None, height: int = 520) -> str:
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
        coords = list(zip(d["sample_lat"].tolist(), d["sample_lon"].tolist()))
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
    for n, d in G.nodes(data=True):
        if d["kind"] != "port":
            continue
        x, y = project(d["lat"], d["lon"])
        ports.append(f'<circle class="port" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'
                     f'<text class="plabel" x="{x + 9:.1f}" y="{y + 4:.1f}">{html.escape(d["name"])}</text>')

    labels = []
    for key, (lat, lon, text) in {
        "hormuz": (27.9, 56.3, "Hormuz"), "suez": (31.4, 33.2, "Suez"), "bab": (11.0, 43.4, "Bab-el-Mandeb"),
        "malacca": (4.7, 99.5, "Malacca"), "panama": (8.2, -79.2, "Panama"),
        "cape": (-36.5, 20.0, "Cape of Good Hope"), "gib": (36.6, -5.4, "Gibraltar"),
    }.items():
        x, y = project(lat, lon)
        labels.append(f'<text class="clabel" x="{x:.1f}" y="{y:.1f}">{text}</text>')

    legend = "".join(
        f'<span><i style="background:{r["color"]}"></i>{html.escape(r["label"])}</span>' for r in routes)
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
.tools{{position:absolute;right:12px;top:12px;display:flex;gap:6px}}
.tools button{{width:30px;height:30px;border:1px solid #335068;background:#132738;color:#a8c7d4;border-radius:8px;font-size:15px;cursor:pointer}}
.tools button:hover{{color:#fff;background:#1a3348}}
</style></head><body><div id="wrap">
<svg id="m" viewBox="0 0 {WORLD_W:.0f} {WORLD_H:.0f}" preserveAspectRatio="xMidYMid slice">
<rect width="{WORLD_W:.0f}" height="{WORLD_H:.0f}" fill="{SEA}"/>
<g class="grid">{grid}</g>{land_svg_markup()}
<g>{''.join(lanes)}</g><g>{''.join(zones)}</g><g>{''.join(labels)}</g><g>{''.join(route_svg)}</g><g>{''.join(ports)}</g>
</svg>
<div class="tools"><button id="zi" title="Zoom in">+</button><button id="zo" title="Zoom out">&minus;</button><button id="zr" title="Reset">&#x2922;</button></div>
<div class="legend">{legend}<span class="k"><b style="color:{AMBER}"></b>high war-risk</span><span class="k"><b style="color:{STORM}"></b>storm</span>
<span class="k"><b style="color:{RED}"></b>closed</span><span class="k"><b style="color:#829aac"></b>previous route</span></div></div>
<script>
const svg=document.getElementById('m'),W={WORLD_W:.0f},H={WORLD_H:.0f};
const state={json.dumps({"x": 0, "y": 0, "w": WORLD_W, "h": WORLD_H})};
function fit(){{const r=svg.getBoundingClientRect(),ar=r.width/r.height;
 let w=W,h=W/ar; if(h>H){{h=H;w=H*ar}} state.w=w;state.h=h;state.x=(W-w)/2;state.y=(H-h)/2;apply()}}
function apply(){{state.w=Math.min(Math.max(state.w,140),W);state.h=state.w*(svg.getBoundingClientRect().height/svg.getBoundingClientRect().width);
 state.x=Math.min(Math.max(state.x,0),W-state.w);state.y=Math.min(Math.max(state.y,0),Math.max(H-state.h,0));
 svg.setAttribute('viewBox',state.x+' '+state.y+' '+state.w+' '+state.h)}}
function zoom(f,cx,cy){{const r=svg.getBoundingClientRect();cx=cx??r.width/2;cy=cy??r.height/2;
 const px=state.x+cx/r.width*state.w,py=state.y+cy/r.height*state.h;const nw=state.w/f;
 const nh=nw*(r.height/r.width);state.x=px-cx/r.width*nw;state.y=py-cy/r.height*nh;state.w=nw;apply()}}
svg.addEventListener('wheel',e=>{{e.preventDefault();const r=svg.getBoundingClientRect();zoom(e.deltaY<0?1.2:1/1.2,e.clientX-r.left,e.clientY-r.top)}},{{passive:false}});
let drag=null;svg.addEventListener('pointerdown',e=>{{drag={{x:e.clientX,y:e.clientY,sx:state.x,sy:state.y}};svg.setPointerCapture(e.pointerId)}});
svg.addEventListener('pointermove',e=>{{if(!drag)return;const r=svg.getBoundingClientRect();
 state.x=drag.sx-(e.clientX-drag.x)/r.width*state.w;state.y=drag.sy-(e.clientY-drag.y)/r.height*state.h;apply()}});
svg.addEventListener('pointerup',()=>drag=null);
document.getElementById('zi').onclick=()=>zoom(1.4);document.getElementById('zo').onclick=()=>zoom(1/1.4);document.getElementById('zr').onclick=fit;
window.addEventListener('resize',fit);fit();
</script></body></html>"""
