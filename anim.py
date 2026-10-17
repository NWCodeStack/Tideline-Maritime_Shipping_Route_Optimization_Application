"""
anim.py - smooth, browser-side playback of the SIMULATED ship.

The server stays the authoritative clock (``Ops.tick``). It describes the motion to the browser once, as a
*contract*: the whole itinerary as time-stamped samples plus (anchor hour, speed, playing, stop hour). The browser
stamps its own clock when the contract arrives and interpolates between server syncs with requestAnimationFrame, so
the ship glides at the display rate without any per-second map reload. The contract is re-sent only when it changes
(play / pause, speed, seek, reroute, stall resync), so while playing it is byte-identical from tick to tick.

``build_track`` mirrors ``voyage.ship_state`` sample for sample (``interpolate`` is the same maths the script runs),
so the animated ship is exactly where the server says it is. Positions are SIMULATED, never AIS.
"""
from __future__ import annotations

import json
import math

from geo import bearing_deg, node_ll, point_on_lane, unwrap

SAMPLE_NM = 300.0          # moving legs are sampled at least this densely so straight-line steps stay on the lane


# ------------------------------------------------------------------------------------------------ track
def _leg_samples(G, leg, t0: float, out: list) -> None:
    """Samples for one leg. Sample k describes the segment from sample k to k+1 (lat/lon move linearly in time)."""
    if leg.kind == "hold":
        if leg.x != leg.y and leg.f0 > 0.0:             # held part-way along a lane (e.g. turning around)
            lat, lon = point_on_lane(G, leg.x, leg.y, leg.f0)
        else:
            lat, lon = node_ll(G, leg.x)
        out.append([t0, lat, lon, 0.0, 0.0, "berth wait" if leg.note == "berth wait" else "holding"])
        return
    course = bearing_deg(*node_ll(G, leg.x), *node_ll(G, leg.y))
    if leg.wait_h > 0.0:                                 # canal queue or U-turn: the ship waits at the start
        lat, lon = point_on_lane(G, leg.x, leg.y, leg.f0)
        out.append([t0, lat, lon, course, 0.0, "turning around" if leg.turn else "canal queue"])
    sog = leg.nm / max(leg.sea_h, 1e-9)
    steps = max(1, math.ceil(leg.nm / SAMPLE_NM))
    for j in range(steps):
        frac = j / steps
        lat, lon = point_on_lane(G, leg.x, leg.y, leg.f0 + (leg.f1 - leg.f0) * frac)
        out.append([t0 + leg.wait_h + leg.sea_h * frac, lat, lon, course, sog, "sailing"])


def _end_sample(G, leg, t_end: float, prev: list) -> list:
    if leg.kind == "hold":
        return [t_end, prev[1], prev[2], 0.0, 0.0, prev[5]]
    lat, lon = point_on_lane(G, leg.x, leg.y, leg.f1)
    return [t_end, lat, lon, prev[3], 0.0, prev[5]]


def _unwrap_lon(samples: list) -> None:
    """Make sample longitudes continuous across the antimeridian (175 -> 185), like the map's polylines."""
    for s, (_, lon) in zip(samples, unwrap([(s[1], s[2]) for s in samples]), strict=True):
        s[2] = lon


def build_track(v, G) -> list[list]:
    """The whole itinerary as ``[t_h, lat, lon, course_deg, sog_kn, activity]`` samples, covering waits and holds."""
    if not v.legs:
        lat, lon = node_ll(G, v.config.origin)
        return [[0.0, lat, lon, 0.0, 0.0, "holding"]]
    out: list[list] = []
    acc = 0.0
    for leg in v.legs:
        _leg_samples(G, leg, acc, out)
        acc += leg.hours
    out.append(_end_sample(G, v.legs[-1], acc, out[-1]))
    _unwrap_lon(out)
    return out


def interpolate(track: list, t: float) -> dict:
    """Position at voyage hour ``t``: the maths the browser script runs, kept here so it can be tested."""
    n = len(track)
    if n == 1 or t <= track[0][0]:
        a = b = track[0]
        f = 0.0
        k = 0
    elif t >= track[-1][0]:
        a = b = track[-1]
        f = 0.0
        k = n - 1
    else:
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if track[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        k = lo
        a, b = track[k], track[k + 1]
        dt = b[0] - a[0]
        f = (t - a[0]) / dt if dt > 1e-12 else 0.0
    return {"lat": a[1] + (b[1] - a[1]) * f, "lon": a[2] + (b[2] - a[2]) * f, "course": a[3], "sog": a[4],
            "activity": a[5], "index": k}


# ------------------------------------------------------------------------------------------------ contract
def _status(ops) -> str:
    v = ops.voyage
    if v.status == "blocked":
        return "blocked"
    if v.status == "arrived":
        return "arrived"
    return "ready" if (not ops.playing and v.t <= 0.0) else "active"


def build_contract(ops, G=None, project=None, perf: bool = False, track: list | None = None) -> dict:
    """Everything the browser needs to animate the ship. Contains no live clock reading: while nothing changes
    (no play / speed / seek / reroute / resync) two calls return identical content.

    ``project(lat, lon) -> (x, y)`` replaces lat/lon by map coordinates (the offline SVG map)."""
    G = G if G is not None else ops.G
    v = ops.voyage
    raw = track if track is not None else build_track(v, G)
    acts = sorted({s[5] for s in raw})
    index = {a: i for i, a in enumerate(acts)}
    digits = 2 if project else 5
    packed = []
    for t, lat, lon, course, sog, act in raw:
        a, b = project(lat, lon) if project else (lat, lon)
        packed.append([round(t, 4), round(a, digits), round(b, digits), round(course, 1), round(sog, 2), index[act]])
    stop = v.total_hours if ops._demo_target is None else min(v.total_hours, ops._demo_target)
    contract = {
        "v": 1, "route_version": ops.version, "epoch": ops.epoch, "resync": ops.resync, "snap": ops.epoch_snap,
        "t_anchor_h": round(ops.anchor_h if ops.playing else v.t, 4), "speed_h_per_s": ops.speed,
        "playing": bool(ops.playing), "stop_h": round(stop, 4), "status": _status(ops),
        "acts": acts, "track": packed,
    }
    if perf:
        contract["perf"] = True
    return contract


def contract_json(contract: dict) -> str:
    """Compact, key-sorted JSON that is safe inside a <script> element (no <, > or &)."""
    text = json.dumps(contract, separators=(",", ":"), sort_keys=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def clamp_time(contract: dict, elapsed_s: float) -> float:
    """The voyage hour the browser shows ``elapsed_s`` real seconds after receiving the contract."""
    t = contract["t_anchor_h"]
    if contract["playing"]:
        t += max(elapsed_s, 0.0) * contract["speed_h_per_s"]
    return min(max(t, 0.0), contract["stop_h"])


def plan_points(track: list) -> list[tuple[float, float]]:
    """The planned route as (a, b) points (unwrapped), consecutive duplicates removed."""
    pts: list[tuple[float, float]] = []
    for s in track:
        p = (s[1], s[2])
        if not pts or pts[-1] != p:
            pts.append(p)
    return pts


# ------------------------------------------------------------------------------------------------ browser script
# One animation loop per window (``window.__tl``): a new contract stops the old loop before starting its own.
# Frame pacing is adaptive: requestAnimationFrame follows the display but draws at most MAX_FPS, skips frames in
# which the ship moved less than MIN_PX on screen, halves the rate (down to MIN_FPS) while drawing costs more than
# SLOW_MS, restores it below FAST_MS, draws nothing while the tab is hidden, and stops entirely at rest.
# The loop makes no network calls and allocates nothing per frame.
JS_CORE = """
function tlStart(C, A) {
  var W = window, TL = W.__tl = W.__tl || {loops: 0, stop: null, stats: {}};
  if (TL.stop) { TL.stop(); }
  var prev = TL.prev, st = TL.stats;
  var MAX_FPS = 60, MIN_FPS = 20, MIN_PX = 0.25, SLOW_MS = 6, FAST_MS = 3, EASE_MS = 300;
  var track = C.track, last = track.length - 1;
  var recv = performance.now();
  var seg = 0, doneSeg = -1, alive = true, running = false, raf = 0, finalDrawn = false, shown = C.t_anchor_h;
  var interval = 1000 / MAX_FPS, costEma = 0, ease = null;
  var lastDraw = -1e9, lastA = NaN, lastB = NaN, lastCourse = NaN, lastAct = -1;
  var statT = recv, statDrawn = 0;
  var cur = {a: 0, b: 0, course: 0, sog: 0, act: 0, t: 0, now: 0};
  st.drawn = 0; st.skipped = 0; st.frames = 0; st.fps = 0; st.costMs = 0; st.intervalMs = interval;

  function smooth(x) { return x * x * (3 - 2 * x); }
  function target(now) {
    var t = C.playing ? C.t_anchor_h + (now - recv) / 1000 * C.speed_h_per_s : C.t_anchor_h;
    return t > C.stop_h ? C.stop_h : (t < 0 ? 0 : t);
  }
  function timeAt(now) {
    var t = target(now);
    if (ease) {
      var p = (now - ease.t0) / EASE_MS;
      if (p >= 1) { ease = null; }
      else {
        t += ease.off * (1 - smooth(p));
        t = t > C.stop_h ? C.stop_h : (t < 0 ? 0 : t);
      }
    }
    return t;
  }
  function findSeg(t) {
    var lo = 0, hi = last;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (track[mid][0] <= t) { lo = mid; } else { hi = mid; } }
    return lo;
  }
  function place(t) {
    if (t < track[seg][0]) { seg = findSeg(t); }
    while (seg < last - 1 && track[seg + 1][0] <= t) { seg++; }
    var a = track[seg], b = track[seg < last ? seg + 1 : last];
    var dt = b[0] - a[0], f = dt > 1e-9 ? (t - a[0]) / dt : (t >= b[0] ? 1 : 0);
    f = f < 0 ? 0 : (f > 1 ? 1 : f);
    cur.a = a[1] + (b[1] - a[1]) * f; cur.b = a[2] + (b[2] - a[2]) * f;
    cur.course = a[3]; cur.sog = a[4]; cur.act = a[5]; cur.t = t;
  }
  function frame(now) {
    raf = 0;
    if (!alive) { return; }
    st.frames++;
    var t = timeAt(now);
    var rest = !ease && (!C.playing || t >= C.stop_h - 1e-9);
    if (!document.hidden && (now - lastDraw >= interval - 1 || (rest && !finalDrawn))) {
      var prevSeg = seg;
      place(t);
      var changed = seg !== prevSeg || cur.course !== lastCourse || cur.act !== lastAct;
      var px = (Math.abs(cur.a - lastA) + Math.abs(cur.b - lastB)) * A.pxPerUnit();
      if (changed || !(px < MIN_PX) || (rest && !finalDrawn)) {
        var c0 = performance.now();
        if (seg !== doneSeg) { A.setDone(track, seg); doneSeg = seg; }
        cur.now = now;
        A.draw(cur, changed);
        var cost = performance.now() - c0;
        costEma = costEma * 0.9 + cost * 0.1;
        if (costEma > SLOW_MS && interval < 1000 / MIN_FPS) { interval = Math.min(interval * 2, 1000 / MIN_FPS); }
        else if (costEma < FAST_MS && interval > 1000 / MAX_FPS) { interval = Math.max(interval / 2, 1000 / MAX_FPS); }
        lastA = cur.a; lastB = cur.b; lastCourse = cur.course; lastAct = cur.act; lastDraw = now;
        st.drawn++;
        if (rest) { finalDrawn = true; }
      } else { st.skipped++; }
    }
    shown = t;
    if (now - statT >= 1000) {
      st.fps = (st.drawn - statDrawn) * 1000 / (now - statT);
      statDrawn = st.drawn; statT = now; st.costMs = costEma; st.intervalMs = interval;
      if (C.perf) {
        console.log('[tl-perf] fps=' + st.fps.toFixed(1) + ' cost_ms=' + costEma.toFixed(2) +
          ' interval_ms=' + interval.toFixed(1) + ' skipped=' + st.skipped + ' loops=' + TL.loops);
      }
    }
    if (rest && finalDrawn) { running = false; return; }
    raf = requestAnimationFrame(frame);
  }
  function kick() { if (alive && !running) { running = true; raf = requestAnimationFrame(frame); } }
  function onVis() {
    if (document.hidden) { return; }
    ease = null; finalDrawn = false; lastDraw = -1e9;     // back on screen: jump to the true time, no catch-up sweep
    kick();
  }
  document.addEventListener('visibilitychange', onVis);

  if (prev && prev.rv === C.route_version && !C.snap) {   // same route, clock event (pause / speed / resync): ease, don't jump
    var off = prev.t - target(recv);
    if (Math.abs(off) > 1e-6) { ease = {off: off, t0: recv}; }
  }
  A.init(C, cur);
  TL.loops++;
  TL.running = function () { return running; };
  TL.now = function () { return timeAt(performance.now()); };
  TL.stop = function () {
    shown = timeAt(performance.now());
    TL.prev = {t: shown, rv: C.route_version};
    alive = false;
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    document.removeEventListener('visibilitychange', onVis);
    A.destroy();
    TL.loops--; TL.stop = null; running = false;
  };
  kick();
}
"""

_LEAFLET_ADAPTER = """
(function () {
  var W = window;
  if (!W.L || typeof map_div === 'undefined') { return; }
  var L = W.L, map = map_div, done = null, tip = null, marker = null, rot = null, tipLL = null;
  var doneTo = -1, fronted = false, lastTip = 0, lastCourse = NaN, status = C.status;
  var FILL = {blocked: '#e5534b', arrived: '#4de0c8'};
  var ACT = {arrived: 'arrived', blocked: 'holding', ready: 'not departed'};
  var LINE = {color: '#4de0c8', weight: 4.5, opacity: 0.95, lineCap: 'round', lineJoin: 'round'};
  var curRef = null;
  function label() {
    var c = curRef, act = ACT[status] || C.acts[c.act];
    var sog = ACT[status] ? 0 : c.sog;
    return 'SIMULATED position ' + c.a.toFixed(2) + ', ' + c.b.toFixed(2) + ' \\u00b7 ' + sog.toFixed(1) +
      ' kn \\u00b7 course ' + ('00' + Math.round(c.course)).slice(-3) + '\\u00b0 \\u00b7 ' + act;
  }
  var adapter = {
    init: function (contract, cur) {
      curRef = cur;
      var s0 = contract.track[0], f = [s0[1], s0[2]];
      done = L.polyline([], LINE).addTo(map).bindTooltip('Completed track');
      tip = L.polyline([f, f], LINE).addTo(map);
      tipLL = tip.getLatLngs();
      var svg = '<div class="tl-rot" style="width:26px;height:26px"><svg viewBox="0 0 24 24" width="26" height="26">' +
        '<path d="M12 2 L18 20 L12 16 L6 20 Z" fill="' + (FILL[status] || '#ffffff') +
        '" stroke="#07121f" stroke-width="1.5"/></svg></div>';
      marker = L.marker(f, {icon: L.divIcon({className: 'tl-ship', html: svg, iconSize: [26, 26], iconAnchor: [13, 13]}),
        keyboard: false, zIndexOffset: 1000}).addTo(map);
      marker.bindTooltip(function () { return label(); }, {direction: 'top', offset: [0, -10]});
      var el = marker.getElement();
      rot = el ? el.firstChild : null;
    },
    pxPerUnit: function () { return 256 * Math.pow(2, map.getZoom()) / 360; },
    setDone: function (track, seg) {
      var i, p;
      if (seg < doneTo || seg - doneTo > 8) {
        var pts = [];
        for (i = 0; i <= seg; i++) {
          p = track[i];
          if (!pts.length || pts[pts.length - 1][0] !== p[1] || pts[pts.length - 1][1] !== p[2]) { pts.push([p[1], p[2]]); }
        }
        done.setLatLngs(pts);
      } else {
        for (i = doneTo + 1; i <= seg; i++) { p = track[i]; done.addLatLng([p[1], p[2]]); }
      }
      doneTo = seg;
      var s = track[seg];
      tip.setLatLngs([[s[1], s[2]], [s[1], s[2]]]);
      tipLL = tip.getLatLngs();
    },
    draw: function (cur, changed) {
      curRef = cur;
      var ll = marker._latlng;
      if (ll) { ll.lat = cur.a; ll.lng = cur.b; marker.update(); } else { marker.setLatLng([cur.a, cur.b]); }
      tipLL[1].lat = cur.a; tipLL[1].lng = cur.b; tip.redraw();
      if (!fronted) { done.bringToFront(); tip.bringToFront(); fronted = true; }
      if (changed && rot && cur.course !== lastCourse) { rot.style.transform = 'rotate(' + cur.course + 'deg)'; lastCourse = cur.course; }
      if (cur.now - lastTip > 250 && marker.isTooltipOpen()) { marker.setTooltipContent(label()); lastTip = cur.now; }
    },
    destroy: function () {
      [done, tip, marker].forEach(function (l) { if (l) { map.removeLayer(l); } });
    }
  };
  tlStart(C, adapter);
})();
"""

_SVG_ADAPTER = """
(function () {
  var W = window, svg = document.getElementById('m');
  if (!svg) { return; }
  var WORLD = C.world_w || 2400;
  var done = document.getElementById('tl-done'), tip = document.getElementById('tl-tip');
  var ship = document.getElementById('tl-ship'), rot = document.getElementById('tl-rot');
  var lastCourse = NaN, x0 = 0, y0 = 0;
  function wrapx(x) { return ((x % WORLD) + WORLD) % WORLD; }
  var adapter = {
    init: function (contract, cur) { },
    pxPerUnit: function () { return W.__tlScale || 1; },
    setDone: function (track, seg) {
      var d = '', px = NaN, i, p, x;
      for (i = 0; i <= seg; i++) {
        p = track[i]; x = wrapx(p[1]);
        d += (isNaN(px) || Math.abs(x - px) > WORLD / 2 ? 'M' : 'L') + x.toFixed(1) + ',' + p[2].toFixed(1) + ' ';
        px = x;
      }
      done.setAttribute('d', d);
      var s = track[seg]; x0 = wrapx(s[1]); y0 = s[2];
    },
    draw: function (cur, changed) {
      var x = wrapx(cur.a);
      ship.setAttribute('transform', 'translate(' + x.toFixed(1) + ',' + cur.b.toFixed(1) + ')');
      tip.setAttribute('d', Math.abs(x - x0) > WORLD / 2 ? '' :
        'M' + x0.toFixed(1) + ',' + y0.toFixed(1) + ' L' + x.toFixed(1) + ',' + cur.b.toFixed(1));
      if (changed && cur.course !== lastCourse) { rot.setAttribute('transform', 'rotate(' + cur.course.toFixed(0) + ')'); lastCourse = cur.course; }
    },
    destroy: function () { }
  };
  tlStart(C, adapter);
})();
"""


def leaflet_script(contract_json_text: str) -> str:
    """Script run inside the st_folium iframe (``map_div`` is the map): the ship, completed track and loop."""
    return "(function(){var C=" + contract_json_text + ";\n" + JS_CORE + "\n" + _LEAFLET_ADAPTER + "\n})();"


def svg_script(contract_json_text: str) -> str:
    """Script for the offline SVG map page: same contract and loop, drawn into the page's own SVG."""
    return "(function(){var C=" + contract_json_text + ";\n" + JS_CORE + "\n" + _SVG_ADAPTER + "\n})();"
