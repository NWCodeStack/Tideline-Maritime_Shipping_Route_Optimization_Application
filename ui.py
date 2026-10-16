"""
ui.py - styling and small HTML builders for the Tideline Streamlit app.

Dark navy panels, mint / amber accents, Space Grotesk headings. Strictly flat: solid colours only, no gradients,
glows, blurs or decorative shadows. Text sizes: 12-13 px for labels/meta, 14-15 px for body, so nothing important
is set below 12 px. KPI grids are container-aware (``repeat(auto-fit, minmax(...))``) rather than viewport-based.

Note: every builder returns single-level HTML with no blank lines or indentation, because Streamlit's
markdown parser turns indented or blank-line-separated HTML into code blocks.
"""
from __future__ import annotations

import html

from risk import BANDS as RISK_BANDS, BASIS as RISK_BASIS, CAPTION as RISK_CAPTION, band
from routing_engine import Comparison, RouteResult, _usd, route_label

MINT, GOLD, RED, BLUE = "#4de0c8", "#f5bf75", "#fe7d81", "#58aaff"
CONTAINER_COLOR, TANKER_COLOR = "#58aaff", GOLD     # blue container ships, amber tankers

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07121f;--panel:#0d1c2c;--panel2:#102236;--line:#21374b;--muted:#8fa7b8;--ink:#e8f3fa;--mint:#4de0c8;--gold:#f5bf75;--red:#fe7d81;--blue:#58aaff}
html,body,.stApp{background:var(--bg);color:var(--ink);font-family:'DM Sans',system-ui,Arial,sans-serif;font-size:15px}
header[data-testid="stHeader"]{background:transparent;height:0}
.block-container{max-width:1720px;padding:0 clamp(12px,2.4vw,40px) 40px}
section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding:18px 16px 28px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.02em}
.topbar{min-height:58px;border-bottom:1px solid #193044;background:#081724;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between;margin:0 calc(-1 * clamp(12px,2.4vw,40px)) 10px;padding:8px clamp(14px,2.4vw,40px)}
.brand{display:flex;align-items:center;gap:12px;letter-spacing:.05em;font-family:'Space Grotesk',sans-serif;font-size:19px;font-weight:700}
.brand-mark{width:34px;height:34px;border:1px solid #3b9a98;background:#164149;border-radius:10px;display:grid;place-items:center}
.brand small{color:#6f98a6;font:600 11px 'DM Sans';letter-spacing:.18em;display:block;margin-top:2px;line-height:1.2}
.topbar-right{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;font-size:12.5px;color:#9bb5c4}
.pill{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:4px 12px;font-weight:700;font-size:13px;border:1px solid}
.pill .dot{width:8px;height:8px;border-radius:50%;background:currentColor}
.pill.active{color:#72f0d5;border-color:#2c8b7b;background:#0f3b37}.pill.paused{color:#f6bc87;border-color:#66513d;background:#2e2522}
.pill.blocked{color:#ffc0c3;border-color:#894351;background:#39232f}.pill.idle{color:#a9c9d6;border-color:#2a5668;background:#112b38}
.pill.active .dot{animation:pulse 1.6s ease-in-out infinite}
.subline{font-size:13.5px;color:#9bb5c4;margin:2px 0 12px}.subline b{color:var(--ink)}
.section-name{font-size:12px;text-transform:uppercase;letter-spacing:.11em;font-weight:800;color:#b9d2dc;margin:4px 0 8px}
.panel-title{font:600 17px 'Space Grotesk',sans-serif;color:var(--ink);margin:0 0 2px}
.panel-sub{font-size:12.5px;color:var(--muted);margin:0 0 10px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(210px,100%),1fr));gap:12px;margin:4px 0 10px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 16px;min-width:0}
.metric-label{font-weight:700;color:#9ab3c4;text-transform:uppercase;letter-spacing:.08em;font-size:12px}
.metric-value{font:600 clamp(20px,1.9vw,27px)/1.2 'Space Grotesk',sans-serif;margin-top:6px;letter-spacing:-.02em;overflow-wrap:anywhere}
.metric-sub{font-size:12.5px;color:#9bb5c4;margin-top:4px;overflow-wrap:anywhere}
.est{display:inline-block;font-size:10.5px;font-weight:800;letter-spacing:.08em;color:#c4b5fd;border:1px solid #4b3f78;border-radius:4px;padding:0 5px;margin-left:6px;vertical-align:middle}
.mint{color:var(--mint)}.blue{color:var(--blue)}.gold{color:var(--gold)}.red{color:var(--red)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:16px}
.pane-top{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;justify-content:space-between;padding:11px 16px;border-bottom:1px solid #1b3245}
.pane-top h2{font:600 16px 'Space Grotesk',sans-serif;margin:0;padding:0}.small-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.agent{border:1px solid #284255;border-radius:10px;background:#0c2030;padding:14px;margin-bottom:10px}
.agent.infeasible{border-color:#894351;background:#2a1b24}
.agent-top{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}
.agent-name{font:600 14px 'Space Grotesk'}.chip{font-size:11px;border-radius:4px;padding:1px 6px;text-transform:uppercase;letter-spacing:.06em;border:1px solid}
.chip.ok{background:#164c45;color:#72f0d5;border-color:#2c8b7b}.chip.warn{background:#3a2d2c;color:#f6bc87;border-color:#66513d}.chip.bad{background:#39232f;color:#ffc0c3;border-color:#894351}
.agent-route{font-size:12.5px;color:#9bb5c4;margin-top:6px}
.stats{display:flex;gap:16px;margin-top:10px;font-size:12.5px;color:#c9dfeb;flex-wrap:wrap}
.stats small{color:#86a7b8;font-size:11px;text-transform:uppercase;letter-spacing:.06em;display:block}.stats b{font-size:14px;font-weight:700}
.alert-box{border:1px solid #7c5640;background:#352a29;border-radius:9px;padding:9px 11px;color:#f9d2a3;font-size:13px;line-height:1.5;margin-bottom:7px}
.alert-box.critical{border-color:#894351;background:#39232f;color:#ffc0c3}.alert-box b{display:block;font-size:13px;color:#ffe0c2;margin-bottom:1px}
.alert-box.critical b{color:#ffbec5}.alert-box.info{border-color:#2a5668;background:#112b38;color:#a9c9d6}.alert-box.info b{color:#d3f4ee}
.methodology{padding:14px 16px;border:1px solid #253e50;background:#0a1b2b;border-radius:12px;color:#9ab3c4;font-size:13px;line-height:1.6}
.methodology strong{color:#cde7ed}
.footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;padding:18px 0 6px;color:#6f8ea0;font-size:12px}.footer b{color:#89b1bf}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{background:transparent;color:#9ab3c4}
.stTabs [aria-selected="true"]{color:var(--mint)}
.live-strip{display:flex;flex-wrap:wrap;align-items:center;gap:6px 16px;background:#081724;border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin:0 0 10px;font-size:12.5px;color:#9bb5c4}
.live-strip b{color:var(--ink);font-weight:700;font-variant-numeric:tabular-nums}
.live-badge{display:inline-flex;align-items:center;gap:7px;font-weight:800;letter-spacing:.1em;font-size:12px;color:var(--mint);text-transform:uppercase}
.live-badge.paused{color:var(--gold)}.live-badge.blocked{color:var(--red)}.live-badge.idle{color:var(--muted)}
.timejump{color:#c4b5fd;font-weight:700}
.pulse{width:8px;height:8px;border-radius:50%;background:currentColor;display:inline-block}
.live-badge.playing .pulse{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@media (prefers-reduced-motion:reduce){.live-badge.playing .pulse,.pill.active .dot{animation:none}}
.tl{border-left:2px solid #21374b;margin-left:6px;padding-left:14px}
.tl-item{position:relative;margin-bottom:12px;font-size:13px;color:#a9c1cf;line-height:1.5}
.tl-item:before{content:"";position:absolute;left:-20px;top:4px;width:10px;height:10px;border-radius:50%;background:#21374b;border:2px solid var(--bg)}
.tl-item.reroute:before{background:var(--blue)}.tl-item.blocked:before{background:var(--red)}.tl-item.assessment:before{background:var(--gold)}
.tl-item.arrival:before,.tl-item.departure:before,.tl-item.plan:before{background:var(--mint)}
.tl-item.demo:before{background:#a78bfa}
.tl-item b{color:var(--ink);font-size:13.5px}.tl-meta{font-size:12px;color:#7f9db0;letter-spacing:.02em}
.src{font-size:11px;border:1px solid #2a5668;border-radius:4px;padding:0 5px;margin-left:6px;color:#8fd3e6;text-transform:uppercase;letter-spacing:.05em}
.decision{border:1px solid #2a5668;background:#0c2030;border-radius:10px;padding:11px 14px;font-size:13.5px;color:#c9dfeb;line-height:1.55;margin-bottom:10px}
.decision.blocked{border-color:#894351;background:#2a1b24;color:#ffc0c3}
.statusline{border:1px solid #2a5668;background:#0c2030;border-radius:10px;padding:8px 12px;font-size:13.5px;color:#c9dfeb;margin-bottom:8px}
.statusline.stay{border-color:#2a6a58;background:#0c2422}
.statusline.change{border-color:#8a7030;background:#271f12;color:#f0dfb8}
.statusline.blocked{border-color:#894351;background:#2a1b24;color:#ffc0c3}
.decision.stay{border-color:#2a6a58;background:#0c2422}
.decision.change{border-color:#8a7030;background:#271f12;color:#f0dfb8}
.reco{border:1px solid #2c8b7b;background:#0c2a2a;border-radius:12px;padding:12px 16px;margin-bottom:10px}
.reco.blocked{border-color:#894351;background:#2a1b24}
.reco h3{font:600 17px 'Space Grotesk',sans-serif;margin:0 0 4px;color:var(--ink)}
.reco p{margin:0;font-size:13.5px;color:#b8d0dc;line-height:1.5}
.deltas{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.delta{font-size:13px;border:1px solid #284255;background:#0c2030;border-radius:6px;padding:3px 9px;color:#c9dfeb}
.delta b{font-weight:700}.delta.bad b{color:var(--red)}.delta.good b{color:var(--mint)}
.banner{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;border:1px solid #4b3f78;background:#14142a;border-radius:10px;padding:8px 14px;margin:0 0 12px;font-size:13.5px;color:#d9d3ff}
.banner b{color:#fff}
.copilot-head{display:flex;flex-wrap:wrap;gap:6px;justify-content:space-between;align-items:center;margin-bottom:6px}
.mode-chip{font-size:11.5px;border-radius:4px;padding:2px 8px;border:1px solid #2c8b7b;color:#72f0d5;background:#164c45;letter-spacing:.04em}
.mode-chip.cmd{border-color:#2a5668;color:#a9c9d6;background:#112b38}
.op-chip{display:inline-block;font-size:11.5px;border:1px solid #284255;background:#0c2030;border-radius:4px;padding:1px 6px;margin:0 4px 4px 0;color:#9bd0e0;font-family:ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.intel-row{font-size:13px;color:#b3c9d6;line-height:1.5;border-bottom:1px solid #1b3245;padding:7px 0}
.intel-row a{color:#8fd3e6;text-decoration:none;font-weight:600}.intel-row a:hover{text-decoration:underline}
.intel-meta{font-size:12px;color:#86a3b5}
.sev{font-size:11px;font-weight:800;border-radius:4px;padding:0 5px;letter-spacing:.04em}
.sev.l0{background:#164c45;color:#72f0d5}.sev.l1{background:#3a2d2c;color:#f6bc87}.sev.l2{background:#47301f;color:#ffb27a}.sev.l3{background:#39232f;color:#ffc0c3}.sev.ln{background:#112b38;color:#9ab3c4}
.demo-table{width:100%;border-collapse:collapse;font-size:13px;color:#c9dfeb}
.demo-table th{text-align:left;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:#9ab3c4;padding:5px 8px;border-bottom:1px solid #21374b}
.demo-table td{padding:5px 8px;border-bottom:1px solid #1b3245;font-variant-numeric:tabular-nums}
.demo-table td.basis{font-size:11.5px;color:#8ba5b8;font-variant-numeric:normal}
.demo-table tr.tot td{border-top:1px solid #2a5668;color:#e8f3fa}
.demo-table tr.weight td{color:#b7a7e6}
.rchip{display:inline-block;font-size:12px;font-weight:700;border:1px solid currentColor;border-radius:999px;padding:1px 9px;cursor:help}
.rchip.mint{color:var(--mint)}.rchip.blue{color:var(--blue)}.rchip.gold{color:var(--gold)}.rchip.red{color:var(--red)}
.tag-sim{font-size:11px;font-weight:800;letter-spacing:.06em;color:#ffd8a8;border:1px solid #7c5640;border-radius:4px;padding:0 5px}
.tag-live{font-size:11px;font-weight:800;letter-spacing:.06em;color:#72f0d5;border:1px solid #2c8b7b;border-radius:4px;padding:0 5px}
@media(max-width:700px){.topbar{margin-bottom:6px}.brand{font-size:17px}.metric-value{font-size:22px}}
"""


def _flat(s: str) -> str:
    return " ".join(line.strip() for line in s.splitlines()).replace("> <", "><")


PILL = {"active": "Simulation active", "paused": "Simulation paused", "blocked": "Voyage blocked",
        "idle": "Ready", "arrived": "Arrived"}


def topbar(state: str = "idle", llm_label: str = "") -> str:
    tone = "idle" if state == "arrived" else state
    meta = (f'<span class="tag-sim">SIMULATED VESSEL</span><span class="tag-live">LIVE WEATHER &amp; NEWS</span>'
            + (f'<span>{html.escape(llm_label)}</span>' if llm_label else ""))
    return _flat(f"""
<div class="topbar"><div class="brand"><div class="brand-mark">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 15.7C5.4 9.4 8 17.8 12 12s7.5-5 10-3.5" stroke="#71efdb" stroke-width="2.2" stroke-linecap="round"/><path d="M4 20c3-5 6 0 9-4s5-3 7-2" stroke="#9ad0d4" stroke-width="1.7" stroke-linecap="round"/></svg>
</div><div>TIDELINE<small>MARITIME OPERATIONS SIMULATOR</small></div></div>
<div class="topbar-right">{meta}<span class="pill {tone}"><span class="dot"></span>{PILL.get(state, state)}</span></div></div>""")


def metric_cards(cards: list[tuple]) -> str:
    """cards: (label, value, sub, tone[, estimate[, tooltip]]) with tone in {"", "mint", "gold", "red", "blue"}."""
    out = []
    for c in cards:
        l, v, s, t = c[:4]
        est = '<span class="est">MODEL ESTIMATE</span>' if len(c) > 4 and c[4] else ""
        tip = f' title="{html.escape(c[5], quote=True)}"' if len(c) > 5 and c[5] else ""
        out.append(f'<div class="metric"{tip}><div class="metric-label">{html.escape(l)}{est}</div>'
                   f'<div class="metric-value {t}">{html.escape(v)}</div><div class="metric-sub">{html.escape(s)}</div></div>')
    return f'<div class="metric-grid">{"".join(out)}</div>'


def live_strip(state: str, items: list[tuple], jump_h: float = 0.0) -> str:
    """Telemetry row. state in playing | paused | blocked | idle; items: (label, value)."""
    badge = {"playing": "Simulated tracking", "paused": "Simulated tracking · paused",
             "blocked": "Simulated tracking · voyage blocked", "idle": "Simulated tracking · not started"}[state]
    cells = "".join(f"<span>{html.escape(l)} <b>{html.escape(v)}</b></span>" for l, v in items)
    jump = (f'<span class="timejump">TIME-LAPSE: each refresh jumps ~{jump_h:g} simulated hours</span>'
            if state == "playing" and jump_h >= 6 else "")
    return f'<div class="live-strip"><span class="live-badge {state}"><span class="pulse"></span>{badge}</span>{cells}{jump}</div>'


def timeline(entries: list[dict], limit: int = 14) -> str:
    if not entries:
        return '<div class="intel-meta">No events yet.</div>'
    rows = []
    for e in list(reversed(entries))[:limit]:
        src = f'<span class="src">{html.escape(e["source"])}</span>' if e.get("source") else ""
        rows.append(f'<div class="tl-item {html.escape(e["kind"])}"><div class="tl-meta">Voyage day {e["t"] / 24:.1f} · '
                    f'{html.escape(e["at"])} (simulated clock){src}</div><b>{html.escape(e["title"])}</b>'
                    f'<div>{html.escape(e["detail"])}</div></div>')
    return '<div class="tl">' + "".join(rows) + "</div>"


def _deltas(before: dict, after: dict) -> list[tuple[str, str, bool]]:
    d_days = (after["eta"] - before["eta"]).total_seconds() / 86400
    dc = after["total_usd"] - before["total_usd"]
    return [("ETA", f"{d_days:+.1f} d", d_days > 0.05),
            ("Distance", f"{after['nm'] - before['nm']:+,.0f} nm", after["nm"] > before["nm"] + 1),
            ("Fuel", f"{after['fuel_t'] - before['fuel_t']:+,.0f} t", after["fuel_t"] > before["fuel_t"] + 1),
            ("Cost", ("+" if dc >= 0 else "−") + _usd(abs(dc)), dc > 1000),
            ("Risk ahead", f"{before['risk']:.0f} {band(before['risk'])[0]} → {after['risk']:.0f} "
                           f"{band(after['risk'])[0]}", after["risk"] > before["risk"] + 0.5)]


def change_banner(change: dict, demo: bool = False) -> str:
    """One-line summary at the top of the page (the full comparison lives in the decision section)."""
    b, a = change["before"], change["after"]
    chips = " · ".join(f"{k} <b>{html.escape(v)}</b>" for k, v, _ in _deltas(b, a))
    head = "Hackathon demo · SIMULATED Suez closure" if demo else "Latest reroute"
    return (f'<div class="banner"><b>{head}</b><span>{html.escape(b["corridor"])} → <b>{html.escape(a["corridor"])}</b></span>'
            f'<span>{chips}</span><span class="intel-meta">Modelled estimates · comparison below</span></div>')


def recommendation(change: dict, status: str, preference: str) -> str:
    """Compact decision card: what is recommended, why, and the before/after deltas."""
    b, a = change["before"], change["after"]
    blocked = status == "blocked"
    title = ("Voyage blocked: hold position" if blocked else
             f"Recommended: continue via {a['corridor']}" if b["corridor"] != a["corridor"] else
             f"Recommended: hold course via {a['corridor']}")
    why = html.escape(change["cause"])
    chips = "".join(f'<span class="delta {"bad" if bad else "good"}">{k} <b>{html.escape(v)}</b></span>'
                    for k, v, bad in _deltas(b, a))
    return _flat(f"""<div class="reco{' blocked' if blocked else ''}"><h3>{html.escape(title)}</h3>
<p>Trigger: {why}. Re-planned from the ship's position ({b['lat']:.2f}, {b['lon']:.2f}) on voyage day {b['t'] / 24:.1f}
under the {html.escape(preference)} objective. Figures are model estimates for the whole voyage.</p>
<div class="deltas">{chips}</div></div>""")


def comparison_table(before: dict, after: dict, before_label: str = "Before", after_label: str = "After") -> str:
    """Compact before/after table (whole-voyage figures, modelled estimates)."""
    def row(label, b, a, delta, bad):
        tone = "red" if bad else "mint"
        return (f'<tr><td>{html.escape(label)}</td><td>{html.escape(b)}</td><td>{html.escape(a)}</td>'
                f'<td class="{tone}">{html.escape(delta)}</td></tr>')
    d = {k: (v, bad) for k, v, bad in _deltas(before, after)}
    rows = "".join([
        row("ETA (simulated clock)", f"{before['eta']:%b %d %H:%M}", f"{after['eta']:%b %d %H:%M}", *d["ETA"]),
        row("Voyage distance", f"{before['nm']:,.0f} nm", f"{after['nm']:,.0f} nm", *d["Distance"]),
        row("Fuel (model)", f"{before['fuel_t']:,.0f} t", f"{after['fuel_t']:,.0f} t", *d["Fuel"]),
        row("All-in cost (model)", _usd(before["total_usd"]), _usd(after["total_usd"]), *d["Cost"]),
        row("Risk index ahead (model)", f"{before['risk']:.0f}/100", f"{after['risk']:.0f}/100", *d["Risk ahead"]),
        row("Corridor", before["corridor"], after["corridor"],
            "diverted" if before["corridor"] != after["corridor"] else "same", False),
    ])
    return (f'<table class="demo-table"><tr><th></th><th>{html.escape(before_label)}</th>'
            f'<th>{html.escape(after_label)}</th><th>Change</th></tr>{rows}</table>')


def demo_panel(before: dict, after: dict) -> str:
    """Kept for compatibility: the demo's before/after table with its labels."""
    return comparison_table(before, after, "Before (via Suez)", "After closure")


def decision(text: str, blocked: bool = False, rec: str = "") -> str:
    """Decision card; ``rec`` ("stay" | "change_course") tints it so the recommendation reads at a glance."""
    kind = " blocked" if blocked else {"stay": " stay", "change_course": " change"}.get(rec, "")
    return f'<div class="decision{kind}">{html.escape(text)}</div>'


def risk_chip(score: float) -> str:
    """Colour chip "27 · Guarded"; hovering shows what the score means."""
    label, tone = band(score)
    return (f'<span class="rchip {tone}" title="{html.escape(RISK_CAPTION, quote=True)}">'
            f'{round(score)} · {html.escape(label)}</span>')


def risk_panel(rb) -> str:
    """The risk score explained: caption, bands, and the parts that add up to the total."""
    esc = html.escape
    rows = "".join(f'<tr><td>{n}</td><td>{100 * v:.1f} pts</td><td class="basis">{esc(RISK_BASIS[k])}</td></tr>'
                   for n, k, v in (("Weather", "weather", rb.weather), ("Security", "security", rb.security),
                                   ("Traffic", "traffic", rb.traffic)))
    rows += (f'<tr class="tot"><td><b>Risk score</b></td><td><b>{rb.score}/100 · {esc(rb.band_label)}</b></td>'
             f'<td class="basis">parts above add up to the score</td></tr>')
    bands = " · ".join(f"{start}+ {esc(lab)}" for start, lab, _ in RISK_BANDS)
    top = (f'<div class="intel-meta" style="margin-top:6px">Highest-security lane: <b>{esc(rb.top_zone)}</b>: '
           f'security {rb.top_security:.2f} (base {rb.top_base:.2f} + scenario {rb.top_scenario:.2f}). '
           f'News headlines are advisory and do not change the score.</div>') if rb.top_zone else ""
    return (f'<div class="intel-meta" style="margin-bottom:6px">{esc(RISK_CAPTION)}</div>'
            f'<table class="demo-table"><tr><th>Signal</th><th>Share of score</th><th>Basis</th></tr>{rows}</table>'
            f'{top}<div class="intel-meta" style="margin-top:6px">Bands: {bands}.</div>')


def cost_breakdown(lines: list[dict], co2_t: float) -> str:
    """Every cost factor on its own line with the basis of its data. The cash lines add up to the projected cost;
    the carbon shadow price is a decision weight, so it sits below the total and is not added to it."""
    esc = html.escape
    cash = [l for l in lines if l["cash"]]
    rows = "".join(f'<tr><td>{esc(l["label"])}</td><td>{_usd(l["usd"])}</td><td class="basis">{esc(l["basis"])}</td></tr>'
                   for l in cash)
    rows += (f'<tr class="tot"><td><b>Cash total (projected cost)</b></td><td><b>{_usd(sum(l["usd"] for l in cash))}</b>'
             f'</td><td class="basis">sum of the lines above</td></tr>')
    rows += "".join(f'<tr class="weight"><td>{esc(l["label"])}</td><td>{_usd(l["usd"])}</td>'
                    f'<td class="basis">{esc(l["basis"])}</td></tr>' for l in lines if not l["cash"])
    return (f'<table class="demo-table"><tr><th>Factor</th><th>Whole voyage</th><th>Basis</th></tr>{rows}</table>'
            f'<div class="intel-meta" style="margin-top:6px">CO2 emitted: {co2_t:,.0f} t (fuel x 3.114). '
            f'All values are SIMULATED or ESTIMATED.</div>')


def panel_head(title: str, label: str = "") -> str:
    return (f'<div class="panel" style="margin-bottom:0;border-bottom-left-radius:0;border-bottom-right-radius:0;'
            f'border-bottom:0"><div class="pane-top"><h2>{html.escape(title)}</h2>'
            f'<span class="small-label">{html.escape(label)}</span></div></div>')


def agent_card(c: Comparison, color: str, origin: str, destination: str) -> str:
    s: RouteResult = c.stressed
    if not s.feasible:
        return _flat(f"""
<div class="agent infeasible"><div class="agent-top"><span class="swatch" style="background:{color}"></span>
<span class="agent-name">{html.escape(c.agent.name)}</span><span class="chip bad">No feasible route</span></div>
<div class="agent-route">{html.escape(origin)} → {html.escape(destination)}: {html.escape(s.reason)}</div></div>""")
    chip = ('<span class="chip warn">Forced through</span>' if s.forced_through
            else '<span class="chip ok">Rerouted</span>' if c.rerouted else '<span class="chip ok">On course</span>')
    saved = "n/a" if c.cost_vs_hold_usd is None else ("same" if abs(c.cost_vs_hold_usd) < 500 else
            (f"{_usd(c.cost_vs_hold_usd)} saved" if c.cost_vs_hold_usd > 0 else f"{_usd(-c.cost_vs_hold_usd)} extra"))
    return _flat(f"""
<div class="agent"><div class="agent-top"><span class="swatch" style="background:{color}"></span>
<span class="agent-name">{html.escape(c.agent.name)}</span>{chip}</div>
<div class="agent-route">{html.escape(origin)} → {html.escape(destination)} · {html.escape(route_label(s))}</div>
<div class="stats"><div><small>Transit</small><b>{s.days:.1f} d</b></div><div><small>All-in</small><b>{_usd(s.total_usd)}</b></div>
<div><small>Tolls</small><b>{_usd(s.toll_usd)}</b></div><div><small>Fuel</small><b>{s.fuel_t:,.0f} t</b></div>
<div><small>vs holding course</small><b>{saved}</b></div></div></div>""")


def alert(title: str, body: str, kind: str = "") -> str:
    return f'<div class="alert-box {kind}"><b>{html.escape(title)}</b>{html.escape(body)}</div>'


METHODOLOGY = _flat("""
<div class="methodology"><strong>What is real and what is modelled.</strong><br>
<b>SIMULATED:</b> vessel positions (no AIS feed - the ship moves along modelled lanes at modelled speeds) and every disruption
(user-triggered scenarios, never verified closures). <b>MODEL ESTIMATES:</b> ETA, fuel, cost and risk index come from a synthetic
lane network, illustrative tolls/insurance and a random-forest fuel model trained and scored only on 5,000 synthetic voyages
(its R² measures fit to that synthetic data, not real-world accuracy). <b>LIVE DATA:</b> Open-Meteo forecasts and GDELT / Google
News headlines, always shown with source and fetch time; headlines are advisory and never close a corridor. Lanes are simplified
waypoint chains, not certified sea lanes. Not for navigation, safety, underwriting or insurance decisions.</div>""")

FOOTER = ('<div class="footer"><span><b>TIDELINE</b> / c0mpiled-14 · UC Berkeley · YC RFS: New Operating Systems for '
          'the Physical World</span><span>Simulated vessels · model estimates · demo only</span></div>')
