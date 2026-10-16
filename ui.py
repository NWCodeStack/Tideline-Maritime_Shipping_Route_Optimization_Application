"""
ui.py - styling and small HTML builders for the Streamlit app.

The look follows the Pelagic dashboard (dark navy panels, mint / amber accents, Space Grotesk headings)
but is strictly flat: solid colours only, no gradients, glows, blurs or decorative shadows.

Note: every builder returns single-level HTML with no blank lines or indentation, because Streamlit's
markdown parser turns indented or blank-line-separated HTML into code blocks.
"""
from __future__ import annotations

import html

from routing_engine import Comparison, RouteResult, _usd, route_label

MINT, GOLD, RED, BLUE = "#4de0c8", "#f5bf75", "#fe7d81", "#58aaff"
CONTAINER_COLOR, TANKER_COLOR = "#58aaff", GOLD     # blue container ships, amber tankers

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07121f;--panel:#0d1c2c;--panel2:#102236;--line:#21374b;--muted:#829aac;--ink:#e8f3fa;--mint:#4de0c8;--gold:#f5bf75;--red:#fe7d81;--blue:#58aaff}
html,body,.stApp{background:var(--bg);color:var(--ink);font-family:'DM Sans',system-ui,Arial,sans-serif}
header[data-testid="stHeader"]{background:transparent;height:0}
.block-container{max-width:1780px;padding:0 clamp(18px,3vw,54px) 44px}
section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding:18px 18px 28px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.02em}
.topbar{height:70px;border-bottom:1px solid #193044;background:#081724;display:flex;align-items:center;justify-content:space-between;margin:0 calc(-1 * clamp(18px,3vw,54px));padding:0 clamp(20px,3vw,54px)}
.brand{display:flex;align-items:center;gap:13px;letter-spacing:.05em;font-family:'Space Grotesk',sans-serif;font-size:19px;font-weight:700}
.brand-mark{width:36px;height:36px;border:1px solid #3b9a98;background:#164149;border-radius:11px;display:grid;place-items:center}
.brand small{color:#5d8290;font:600 10px 'DM Sans';letter-spacing:.21em;display:block;margin-top:3px;line-height:1.2}
.nav-meta{color:#8ba5b4;font-size:12px}.status-dot{width:7px;height:7px;border-radius:50%;background:var(--mint);display:inline-block;margin-right:8px}
.intro{margin:30px 0 22px}.eyebrow{font-size:10px;color:var(--mint);letter-spacing:.22em;font-weight:800;text-transform:uppercase;margin-bottom:8px}
.intro h1{font:600 clamp(27px,3.1vw,40px)/1.13 'Space Grotesk',sans-serif;letter-spacing:-.04em;margin:0 0 9px;padding:0}
.intro p{margin:0;color:#829bab;font-size:13px;max-width:760px}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:17px 20px;min-height:118px}
.metric-label{font-weight:700;color:#8ba5b8;text-transform:uppercase;letter-spacing:.12em;font-size:10px}
.metric-value{font:600 clamp(21px,2.1vw,29px)/1.25 'Space Grotesk',sans-serif;margin-top:12px;letter-spacing:-.025em}
.metric-sub{font-size:11px;color:#92adba;margin-top:6px}.mint{color:var(--mint)}.blue{color:var(--blue)}.gold{color:var(--gold)}.red{color:var(--red)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:16px}
.pane-top{display:flex;align-items:center;justify-content:space-between;padding:15px 20px;border-bottom:1px solid #1b3245}
.pane-top h2{font:600 15px 'Space Grotesk',sans-serif;margin:0;padding:0}.small-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em}
.pane-body{padding:14px 16px}
.agent{border:1px solid #284255;border-radius:10px;background:#0c2030;padding:15px;margin-bottom:10px}
.agent.infeasible{border-color:#894351;background:#2a1b24}
.agent-top{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}
.agent-name{font:600 13px 'Space Grotesk'}.chip{font-size:9px;border-radius:4px;padding:1px 6px;text-transform:uppercase;letter-spacing:.07em;border:1px solid}
.chip.ok{background:#164c45;color:#72f0d5;border-color:#2c8b7b}.chip.warn{background:#3a2d2c;color:#f6bc87;border-color:#66513d}.chip.bad{background:#39232f;color:#ffc0c3;border-color:#894351}
.agent-route{font-size:11px;color:#9bb5c4;margin-top:6px}
.stats{display:flex;gap:16px;margin-top:11px;font-size:11px;color:#c9dfeb;flex-wrap:wrap}
.stats small{color:#7193a4;font-size:9px;text-transform:uppercase;letter-spacing:.07em;display:block}.stats b{font-size:13px;font-weight:700}
.alert-box{border:1px solid #7c5640;background:#352a29;border-radius:9px;padding:11px 12px;color:#f9d2a3;font-size:11px;line-height:1.55;margin-bottom:9px}
.alert-box.critical{border-color:#894351;background:#39232f;color:#ffc0c3}.alert-box b{display:block;font-size:12px;color:#ffe0c2;margin-bottom:2px}
.alert-box.critical b{color:#ffbec5}.alert-box.info{border-color:#2a5668;background:#112b38;color:#a9c9d6}.alert-box.info b{color:#d3f4ee}
.section-name{font-size:11px;text-transform:uppercase;letter-spacing:.12em;font-weight:800;color:#b9d2dc;margin:6px 0 10px}
.methodology{margin-top:18px;padding:17px 20px;border:1px solid #253e50;background:#0a1b2b;border-radius:12px;color:#8eaab9;font-size:11px}
.methodology strong{color:#cde7ed}
.footer{display:flex;justify-content:space-between;gap:16px;padding:22px 0 8px;color:#577589;font-size:10px}.footer b{color:#89b1bf}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{background:transparent;color:#8ba5b8}
.stTabs [aria-selected="true"]{color:var(--mint)}
@media(max-width:800px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def _flat(s: str) -> str:
    return "".join(line.strip() for line in s.splitlines())


def topbar() -> str:
    return _flat("""
<div class="topbar"><div class="brand"><div class="brand-mark">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 15.7C5.4 9.4 8 17.8 12 12s7.5-5 10-3.5" stroke="#71efdb" stroke-width="2.2" stroke-linecap="round"/><path d="M4 20c3-5 6 0 9-4s5-3 7-2" stroke="#9ad0d4" stroke-width="1.7" stroke-linecap="round"/></svg>
</div><div>STRESS-TESTER<small>MARITIME ROUTING AGENTS</small></div></div>
<div class="nav-meta"><span class="status-dot"></span>SIMULATION <span style="color:#527486;margin:0 8px">/</span> SYNTHETIC DATA</div></div>""")


def intro(events: list[str]) -> str:
    sub = ("Active shocks: " + " · ".join(events)) if events else "No shocks active. Switch on an event in the sidebar to stress the network."
    return _flat(f"""
<div class="intro"><div class="eyebrow">◎ Multi-agent routing / Supply-chain stress test</div>
<h1>Same ocean, different decisions.</h1>
<p>Each ship and cargo owner prices fuel, tolls, war-risk and delay differently, so one shock sends them down different routes. {html.escape(sub)}</p></div>""")


def metric_cards(cards: list[tuple]) -> str:
    """cards: (label, value, sub, tone) with tone in {"", "mint", "gold", "red"}."""
    inner = "".join(
        f'<div class="metric"><div class="metric-label">{html.escape(l)}</div>'
        f'<div class="metric-value {t}">{html.escape(v)}</div><div class="metric-sub">{html.escape(s)}</div></div>'
        for l, v, s, t in cards)
    return f'<div class="metric-grid">{inner}</div>'


def panel_head(title: str, label: str = "") -> str:
    return f'<div class="panel" style="margin-bottom:0;border-bottom-left-radius:0;border-bottom-right-radius:0;border-bottom:0"><div class="pane-top"><h2>{html.escape(title)}</h2><span class="small-label">{html.escape(label)}</span></div></div>'


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
<div class="methodology"><strong>Prototype boundaries.</strong> The network, risk zones, tolls and voyages are synthetic.
Geometries follow real coastlines and chokepoint bounding boxes, but lanes are simplified waypoint chains, not certified sea lanes.
Fuel burn comes from a random-forest model trained on 5,000 simulated voyages. Dollar figures are illustrative assumptions.
Not for navigation, underwriting or insurance decisions.</div>""")

FOOTER = '<div class="footer"><span><b>STRESS-TESTER</b> / c0mpiled-14 · UC Berkeley · YC RFS: New Operating Systems for the Physical World</span><span>Synthetic data · demo only</span></div>'
