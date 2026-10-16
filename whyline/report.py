"""Human-readable output: a one-line headline, a terminal report, and a
self-contained HTML report with a waterfall chart (no JS, no CDN)."""

from __future__ import annotations

import html as _html
import math
from typing import TYPE_CHECKING

from .metrics import Ratio

if TYPE_CHECKING:
    from .explain import Cause, Explanation


def _pct(x: float) -> str:
    if not math.isfinite(x):
        return "—"
    return ("−" if x < 0 else "") + f"{abs(x) * 100:.0f}%"


def _rel(e: "Explanation") -> str:
    if not e.base_value:
        return ""
    r = (e.current_value / e.base_value - 1) * 100
    return f", {'+' if r >= 0 else '−'}{abs(r):.1f}%"


def _verb(e: "Explanation") -> str:
    return "rose" if e.delta > 0 else "fell" if e.delta < 0 else "was flat"


def headline(e: "Explanation") -> str:
    m = e.metric
    s = (f"{m.label.capitalize()} {_verb(e)} from {m.fmt(e.base_value)} to "
         f"{m.fmt(e.current_value)} ({m.fmt_delta(e.delta)}{_rel(e)}).")
    if e.total_z is not None and math.isfinite(e.total_z) and abs(e.total_z) < 2:
        s += f" That is within normal period-to-period variation (z = {e.total_z:+.1f})."
    if not e.causes:
        return s + " No segment stands out from noise."
    top = e.causes[0]
    n = len(e.causes)
    return (s + f" {n} cause{'s' if n > 1 else ''} account for {_pct(e.explained_share)} of it; "
            f"the largest is {_mech_phrase(top)} {top.name}.")


def _mech_phrase(c: "Cause") -> str:
    up = c.impact > 0
    return {"rate": "a rate increase in" if c.components.get("rate", 0) > 0 else "a rate drop in",
            "mix": "a mix shift toward" if c.values.get("current_share", 0) > c.values.get("base_share", 0)
            else "a mix shift away from",
            "change": "growth in" if c.values.get("current", 0) > c.values.get("base", 0) else "a decline in",
            }.get(c.mechanism, "") or ("a lift in" if up else "a drop in")


def describe(c: "Cause", e: "Explanation") -> str:
    """One sentence of evidence for a cause."""
    m, v = e.metric, c.values
    if isinstance(m, Ratio):
        unit = m.denominator
        if c.mechanism == "rate":
            return (f"{m.label} {m.fmt(v['base'])} → {m.fmt(v['current'])} within this segment, "
                    f"which is {v['current_share'] * 100:.1f}% of {unit}.")
        overall = m.fmt(v["overall_base"])
        better = "better" if v["base"] > v["overall_base"] else "worse"
        return (f"share of {unit} {v['base_share'] * 100:.1f}% → {v['current_share'] * 100:.1f}%; "
                f"it runs at {m.fmt((v['base'] + v['current']) / 2)} vs {overall} overall "
                f"({better} than average). Its own {m.label} barely moved "
                f"({m.fmt(v['base'])} → {m.fmt(v['current'])}): this is who arrived, not how they behaved."
                if abs(c.components.get("rate", 0)) < 0.25 * abs(c.components.get("mix", 0)) else
                f"share of {unit} {v['base_share'] * 100:.1f}% → {v['current_share'] * 100:.1f}%; "
                f"it runs at {m.fmt(v['current'])} vs {overall} overall.")
    return f"{m.label} {m.fmt(v['base'])} → {m.fmt(v['current'])} in this segment."


def _evidence(c: "Cause") -> str:
    if c.z is None or not math.isfinite(c.z):
        return "no noise estimate"
    z = ("+" if c.z >= 0 else "−") + f"{abs(c.z):.1f}"
    return f"z = {z} {c.z_method} → {c.verdict}"


# ----------------------------------------------------------------------------
# terminal
# ----------------------------------------------------------------------------

def render_text(e: "Explanation") -> str:
    m = e.metric
    arrow = "▲" if e.delta > 0 else "▼" if e.delta < 0 else "■"
    out = [
        f"whyline · {m.label} · {e.baseline} → {e.current}",
        "",
        f"  {m.fmt(e.base_value)} → {m.fmt(e.current_value)}   {arrow} {m.fmt_delta(e.delta)}{_rel(e)}",
        "",
    ]
    if not e.causes:
        out.append("  " + headline(e))
        return "\n".join(out)

    out.append(f"  {len(e.causes)} cause{'s' if len(e.causes) > 1 else ''} explain "
               f"{_pct(e.explained_share)} of the change.\n")
    w = max(12, *(len(m.fmt_delta(c.impact)) for c in e.causes))
    out.append(f"  {'#':>2}  {'impact':>{w}}  {'share':>6}  {'mechanism':<9}  segment")
    for i, c in enumerate(e.causes, 1):
        out.append(f"  {i:>2}  {m.fmt_delta(c.impact):>{w}}  {_pct(c.share):>6}  {c.mechanism:<9}  {c.name}")
        out.append(f"  {'':>2}  {'':>{w}}  {'':>6}  {'':<9}  {describe(c, e)}")
        out.append(f"  {'':>2}  {'':>{w}}  {'':>6}  {'':<9}  {_evidence(c)}")
    out.append(f"  {'':>2}  {m.fmt_delta(e.residual):>{w}}  "
               f"{_pct(e.residual / e.delta if e.delta else float('nan')):>6}  {'':<9}  unexplained")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#fff;
--down:#c2410c;--up:#15803d;--base:#57534e;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#9c9a92;
--line:#2c2b28;--card:#1c1b19;--down:#fb923c;--up:#4ade80;--base:#a8a29e;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 64px}
.eyebrow{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted)}
h1{font-size:30px;line-height:1.2;margin:8px 0 6px;font-weight:650;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 28px;max-width:62ch}
.kpis{display:flex;gap:28px;flex-wrap:wrap;margin:0 0 28px}
.kpi b{display:block;font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.kpi span{font-size:12px;color:var(--muted)}
.down{color:var(--down)}.up{color:var(--up)}
figure{margin:0 0 32px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
figcaption{font-size:12px;color:var(--muted);margin-top:6px}
svg text{font:11px ui-sans-serif,system-ui,sans-serif;fill:var(--muted)}
svg .v{fill:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
ol{list-style:none;padding:0;margin:0;display:grid;gap:12px}
li{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
display:grid;grid-template-columns:auto 1fr auto;gap:4px 16px;align-items:baseline}
.n{font:600 13px ui-monospace,Menlo,monospace;color:var(--muted)}
.seg{font-weight:600}.seg code{font:inherit;background:none}
.imp{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.detail{grid-column:2/4;color:var(--muted);font-size:14px}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:99px;
border:1px solid var(--line);margin-right:6px;color:var(--muted)}
.tag.strong{border-color:var(--accent);color:var(--accent)}
.resid{color:var(--muted);font-size:14px;margin-top:14px}
footer{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
"""


def _waterfall(e: "Explanation") -> str:
    m = e.metric
    steps = [(e.baseline, e.base_value, "base")]
    for i, c in enumerate(e.causes, 1):
        steps.append((f"#{i}", c.impact, "delta"))
    steps.append(("other", e.residual, "delta"))
    steps.append((e.current, e.current_value, "base"))

    level, spans = 0.0, []
    for label, val, kind in steps:
        if kind == "base":
            spans.append((label, None, val, kind, val))
            level = val
        else:
            spans.append((label, level, level + val, kind, val))
            level += val
    tops = [s[2] for s in spans] + [s[1] for s in spans if s[1] is not None]
    hi, lo = max(tops), min(tops)
    pad = (hi - lo) * 0.25 or abs(hi) * 0.1 or 1
    y_lo, y_hi = lo - pad, hi + pad * 0.6

    W, H, L, B, T = 760, 260, 8, 34, 22
    bw = (W - L * 2) / len(spans)
    ys = lambda v: T + (y_hi - v) / (y_hi - y_lo) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Waterfall from {e.baseline} to {e.current}">']
    parts.append(f'<line x1="{L}" x2="{W - L}" y1="{H - B}" y2="{H - B}" stroke="var(--line)"/>')
    prev_top = None
    for i, (label, a, b, kind, val) in enumerate(spans):
        x = L + i * bw + bw * 0.18
        w = bw * 0.64
        if kind == "base":
            y0, y1 = ys(b), H - B
            color = "var(--base)"
        else:
            y0, y1 = sorted((ys(a), ys(b)))
            color = "var(--up)" if val > 0 else "var(--down)"
            if label == "other":
                color = "var(--muted)"
        h = max(y1 - y0, 1.5)
        parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="2" fill="{color}"'
                     f'{' opacity=".55"' if label == "other" else ""}/>')
        if prev_top is not None:
            parts.append(f'<line x1="{x - bw * 0.36:.1f}" x2="{x:.1f}" y1="{prev_top:.1f}" y2="{prev_top:.1f}" '
                         f'stroke="var(--muted)" stroke-dasharray="2 3"/>')
        prev_top = ys(b)
        txt = m.fmt(b) if kind == "base" else m.fmt_delta(val)
        parts.append(f'<text class="v" x="{x + w / 2:.1f}" y="{y0 - 6:.1f}" text-anchor="middle">{txt}</text>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{H - B + 18}" text-anchor="middle">{_html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(e: "Explanation") -> str:
    m = e.metric
    esc = _html.escape
    cls = "up" if e.delta > 0 else "down"
    items = []
    for i, c in enumerate(e.causes, 1):
        icls = "up" if c.impact > 0 else "down"
        verdict_cls = "strong" if c.verdict in ("strong", "likely real") else ""
        items.append(
            f'<li><span class="n">#{i}</span>'
            f'<span class="seg">{esc(_mech_phrase(c).capitalize())} <code>{esc(c.name)}</code></span>'
            f'<span class="imp {icls}">{esc(m.fmt_delta(c.impact))} · {_pct(c.share)}</span>'
            f'<span class="detail"><span class="tag">{c.mechanism}</span>'
            f'<span class="tag {verdict_cls}">{esc(_evidence(c))}</span><br>{esc(describe(c, e))}</span></li>')
    resid_share = _pct(e.residual / e.delta) if e.delta else "—"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>whyline · {esc(m.label)}</title><style>{_CSS}</style></head><body><main>
<div class="eyebrow">whyline · {esc(e.baseline)} → {esc(e.current)}</div>
<h1>Why did {esc(m.label)} {'rise' if e.delta > 0 else 'fall'}?</h1>
<p class="lede">{esc(headline(e))}</p>
<div class="kpis">
 <div class="kpi"><b>{esc(m.fmt(e.base_value))}</b><span>{esc(e.baseline)}</span></div>
 <div class="kpi"><b>{esc(m.fmt(e.current_value))}</b><span>{esc(e.current)}</span></div>
 <div class="kpi"><b class="{cls}">{esc(m.fmt_delta(e.delta))}</b><span>change</span></div>
 <div class="kpi"><b>{_pct(e.explained_share)}</b><span>explained by {len(e.causes)} causes</span></div>
</div>
<figure>{_waterfall(e)}<figcaption>Each bar is a cause's Shapley-attributed impact; the bars sum exactly
to the observed change. Axis does not start at zero.</figcaption></figure>
<ol>{''.join(items)}</ol>
<p class="resid">Unexplained remainder: {esc(m.fmt_delta(e.residual))} ({resid_share}) — the sum of many small movements.</p>
<footer>Sliced by {esc(', '.join(e.dimensions))}. <b>rate</b> = the segment's own {esc(m.label)} changed;
<b>mix</b> = the segment's share of {esc(getattr(m, 'denominator', 'the total'))} changed.
Noise scores compare each effect with the same segment's movement in
{e.history_periods or 'no'} earlier period-pairs. Generated by whyline.</footer>
</main></body></html>"""
