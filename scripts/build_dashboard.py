"""Build the static results page (docs/index.html) from data/dashboard/*.json.

One self-contained HTML file: charts are plain HTML/CSS rows and inline SVG,
with a little vanilla JavaScript for tooltips, the metric toggle, the session
drill-down and the theme switch. Every number comes from the pipeline's
exported aggregates; nothing here touches raw data.

Usage: python scripts/build_dashboard.py
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dashboard"
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/HalcyonVector/LogLens"


def load(name: str) -> dict:
    return json.loads((DATA / f"hdfs_{name}.json").read_text())


EV = load("evaluation")
BASE = load("baselines")
LSTM = load("lstm")
ABL = load("ablation")
PIPE = load("pipeline")
DRILL = load("drilldown")
PARSING = load("parsing")
SWEEP = load("parse_sweep")

BEST = EV["best_lstm"]

# Display name, family. Family picks the colour everywhere on the page.
META = {
    "pca": ("PCA, Xu et al. 2009", "count"),
    "iforest": ("Isolation Forest", "count"),
    "lightgbm": ("LightGBM (sees labels)", "supervised"),
    "seq_lookup": ("Sequence lookup + novelty", "naive"),
    "warn_rule": ("Any WARN line", "naive"),
    "deeplog_topg": ("DeepLog LSTM, top-g", "sequence"),
    "deeplog_end_topg": ("LSTM + END, top-g", "sequence"),
    "deeplog_end_logprob": ("LSTM + END, log-prob", "sequence"),
    "deeplog_end_tuned_logprob": ("LSTM + END, tuned, log-prob", "sequence"),
}
FAMILY_LABEL = {"count": "Count model", "sequence": "Sequence model", "supervised": "Supervised reference", "naive": "Rule or lookup"}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def n(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def tip(title: str, rows: list[tuple[str, str]]) -> str:
    """Tooltip payload, read by the page script with textContent only."""
    return esc(json.dumps({"t": title, "r": rows}))


def dur(seconds: float) -> str:
    s = float(seconds)
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ------------------------------------------------------------------ numbers --

M = EV["models"]
pca, best = M["pca"], M[BEST]
ov = EV["overlap"]["pca_vs_best_lstm"]
EV_TPL = {t["event_id"]: t for t in EV["templates"]}
gap_best = LSTM["deeplog_end_tuned"]["leakage_gap"]["nll"]["adjusted_gap_points"]
gap_pca = BASE["leakage_gap"]["pca"]["adjusted_gap_points"]
dup = BASE["duplication"]
tout = EV["timeout"]
ops = EV["operations"]
lat = EV["latency"]


# ------------------------------------------------------------------ charts --


def leaderboard() -> str:
    order = sorted(M, key=lambda k: -M[k]["f1"])
    rows = []
    for metric, lo, hi in (("f1", 0.5, 1.0), ("pr_auc", 0.4, 1.0)):
        body = []
        for k in order:
            m = M[k]
            name, fam = META[k]
            v, (c0, c1) = m[metric], m["ci"][metric]
            x = lambda t: 100 * (max(lo, min(hi, t)) - lo) / (hi - lo)
            cut = v < lo
            t = tip(name, [("F1", f"{m['f1']:.3f}  [{m['ci']['f1'][0]:.3f} to {m['ci']['f1'][1]:.3f}]"),
                           ("PR-AUC", f"{m['pr_auc']:.3f}"), ("Precision", f"{m['precision']:.3f}"),
                           ("Recall", f"{m['recall']:.3f}"), ("False alarms", n(m["fp"])),
                           ("Missed anomalies", n(m["fn"]))])
            body.append(
                f'<div class="lb-row{" is-best" if k in ("pca", BEST) else ""}" tabindex="0" data-tip="{t}">'
                f'<div class="lb-name"><span class="key fam-{fam}"></span>{esc(name)}</div>'
                f'<div class="lb-track">'
                f'<span class="lb-ci fam-{fam}" style="left:{x(c0):.2f}%;width:{max(0.6, x(c1) - x(c0)):.2f}%"></span>'
                f'<span class="lb-dot fam-{fam}{" is-cut" if cut else ""}" style="left:{x(v):.2f}%"></span></div>'
                f'<div class="lb-val num">{v:.3f}</div></div>'
            )
        ticks = "".join(
            f'<span style="left:{100 * (t - lo) / (hi - lo):.2f}%">{t:.1f}</span>'
            for t in [lo + i * 0.1 for i in range(int(round((hi - lo) / 0.1)) + 1)]
        )
        rows.append(
            f'<div class="lb" data-metric="{metric}"{" hidden" if metric != "f1" else ""}>'
            f'{"".join(body)}<div class="lb-axis"><div></div><div class="lb-ticks">{ticks}</div><div></div></div></div>'
        )
    table = "".join(
        f"<tr><td>{esc(META[k][0])}</td><td class=num>{M[k]['f1']:.3f}</td>"
        f"<td class=num>{M[k]['ci']['f1'][0]:.3f} to {M[k]['ci']['f1'][1]:.3f}</td>"
        f"<td class=num>{M[k]['pr_auc']:.3f}</td><td class=num>{M[k]['precision']:.3f}</td>"
        f"<td class=num>{M[k]['recall']:.3f}</td><td class=num>{n(M[k]['fp'])}</td></tr>"
        for k in order
    )
    return f"""
<div class="toggle" role="tablist" aria-label="Metric">
  <button role="tab" aria-selected="true" data-show="f1">F1 at the validation threshold</button>
  <button role="tab" aria-selected="false" data-show="pr_auc">PR-AUC</button>
</div>
{"".join(rows)}
<p class="note">Bars are 95% bootstrap intervals (1,000 resamples of the 172,519 test sessions). LSTM rows are seed 0, whose per-event scores are saved; the README gives 3-seed means. Scores below the axis start are pinned to its left edge.</p>
<details><summary>Table</summary><div class="tbl"><table><thead><tr><th>Model</th><th>F1</th><th>95% interval</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>False alarms</th></tr></thead><tbody>{table}</tbody></table></div></details>"""


def waffle(key: str, other: str) -> str:
    ov = EV["overlap"][key]
    groups = [
        ("both", ov["both"], "Both catch it", "w-both"),
        ("only_a", ov["only_a"], "Only PCA", "w-pca"),
        ("only_b", ov["only_b"], f"Only {other}", "w-lstm"),
        ("neither", ov["neither"], "Neither", "w-none"),
    ]
    cells = []
    for key, count, label, cls in groups:
        t = tip(label, [("Test anomalies", n(count)), ("Share", pct(count / ov["anomalies"]))])
        cells.append(f'<span class="w {cls}" style="--n:{count}" data-tip="{t}" tabindex="0" aria-label="{esc(label)}: {count}"></span>')
    squares = []
    for key, count, label, cls in groups:
        squares.extend([cls] * count)
    svg_cells = []
    cols = 62
    for i, cls in enumerate(squares):
        r, c = divmod(i, cols)
        svg_cells.append(f'<rect class="{cls}" x="{c * 8}" y="{r * 8}" width="6" height="6" rx="1"/>')
    rows_n = math.ceil(len(squares) / cols)
    legend = "".join(
        f'<li><span class="sw {cls}"></span><b class="num">{n(count)}</b> {esc(label)}</li>'
        for _, count, label, cls in groups
    )
    return f"""
<figure class="waffle">
  <svg viewBox="0 0 {cols * 8} {rows_n * 8}" role="img" aria-label="{ov['anomalies']} test anomalies, one square each: {ov['both']} caught by both models, {ov['only_a']} only by PCA, {ov['only_b']} only by {esc(other)}, {ov['neither']} by neither.">{"".join(svg_cells)}</svg>
  <ul class="legend">{legend}</ul>
</figure>
<div class="hit-strip">{"".join(cells)}</div>"""


def gap_bars() -> str:
    rows = [
        ("pca", BASE["leakage_gap"]["pca"]["adjusted_gap_points"]),
        ("iforest", BASE["leakage_gap"]["iforest"]["adjusted_gap_points"]),
        ("lightgbm", BASE["leakage_gap"]["lightgbm"]["adjusted_gap_points"]),
        ("deeplog_topg", LSTM["deeplog"]["leakage_gap"]["adjusted_gap_points"]),
        ("deeplog_end_topg", LSTM["deeplog_end"]["leakage_gap"]["adjusted_gap_points"]),
        ("deeplog_end_logprob", LSTM["deeplog_end"]["leakage_gap"]["nll"]["adjusted_gap_points"]),
        ("deeplog_end_tuned_logprob", LSTM["deeplog_end_tuned"]["leakage_gap"]["nll"]["adjusted_gap_points"]),
    ]
    lo, hi = -2.0, 10.0
    zero = 100 * (0 - lo) / (hi - lo)
    out = []
    for k, g in rows:
        name, fam = META[k]
        left = 100 * (min(g, 0) - lo) / (hi - lo)
        width = 100 * abs(g) / (hi - lo)
        t = tip(name, [("Random-split gain", f"{g:+.1f} F1 points"), ("At the chronological anomaly rate", "yes")])
        out.append(
            f'<div class="gb-row" tabindex="0" data-tip="{t}"><div class="lb-name"><span class="key fam-{fam}"></span>{esc(name)}</div>'
            f'<div class="gb-track"><span class="gb-zero" style="left:{zero:.2f}%"></span>'
            f'<span class="gb-bar fam-{fam}{" neg" if g < 0 else ""}" style="left:{left:.2f}%;width:{max(width, 0.5):.2f}%"></span></div>'
            f'<div class="lb-val num">{g:+.1f}</div></div>'
        )
    ticks = "".join(f'<span style="left:{100 * (t - lo) / (hi - lo):.2f}%">{t:+d}</span>' for t in range(-2, 11, 2))
    return f'<div class="gb">{"".join(out)}<div class="lb-axis"><div></div><div class="lb-ticks">{ticks}</div><div></div></div></div>'


def unseen_dumbbell() -> str:
    b, l = BASE["results"]["chrono"], LSTM
    rows = [
        ("pca", b["pca"]["f1"], b["pca"]["unseen_only"]["f1"]),
        ("lightgbm", b["lightgbm"]["f1"], b["lightgbm"]["unseen_only"]["f1"]),
        ("iforest", b["iforest"]["f1"], b["iforest"]["unseen_only"]["f1"]),
        ("seq_lookup", b["seq_lookup"]["f1"], b["seq_lookup"]["unseen_only"]["f1"]),
        ("deeplog_topg", l["deeplog"]["results"]["chrono"]["summary"]["f1"]["mean"], l["deeplog"]["results"]["chrono"]["summary"]["unseen_f1"]["mean"]),
        ("deeplog_end_tuned_logprob", l["deeplog_end_tuned"]["results"]["chrono"]["summary"]["nll_f1"]["mean"], l["deeplog_end_tuned"]["results"]["chrono"]["summary"]["nll_unseen_f1"]["mean"]),
    ]
    lo, hi = 0.5, 1.0
    x = lambda v: 100 * (max(lo, min(hi, v)) - lo) / (hi - lo)
    out = []
    for k, all_f1, un in rows:
        name, fam = META[k]
        a, u = x(all_f1), x(un)
        t = tip(name, [("F1, all test sessions", f"{all_f1:.3f}"), ("F1, never-seen sequences", f"{un:.3f}"), ("Change", f"{un - all_f1:+.3f}")])
        out.append(
            f'<div class="lb-row" tabindex="0" data-tip="{t}"><div class="lb-name"><span class="key fam-{fam}"></span>{esc(name)}</div>'
            f'<div class="lb-track"><span class="db-line" style="left:{min(a, u):.2f}%;width:{abs(a - u):.2f}%"></span>'
            f'<span class="db-all" style="left:{a:.2f}%"></span><span class="lb-dot fam-{fam}" style="left:{u:.2f}%"></span></div>'
            f'<div class="lb-val num">{un:.3f}</div></div>'
        )
    ticks = "".join(f'<span style="left:{100 * (t - lo) / (hi - lo):.2f}%">{t:.1f}</span>' for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    return (f'<div class="lb">{"".join(out)}<div class="lb-axis"><div></div><div class="lb-ticks">{ticks}</div><div></div></div></div>'
            '<ul class="legend inline"><li><span class="sw ring"></span>F1 on every test session</li><li><span class="sw dot"></span>F1 on the 1,699 never-seen sequences</li></ul>')


def alerts_bars() -> str:
    keys = ["pca", "lightgbm", "iforest", "seq_lookup", "deeplog_topg", BEST]
    per = ops["per_model"]
    hi = max(per[k]["alerts_per_hour"] for k in keys) * 1.08
    out = []
    for k in keys:
        name, fam = META[k]
        a, f = per[k]["alerts_per_hour"], per[k]["false_alarms_per_hour"]
        tpw, fpw = 100 * (a - f) / hi, 100 * f / hi
        t = tip(name, [("Alerts per hour", n(a, 1)), ("of which false", n(f, 1)), ("False alarms per 1,000 normal sessions", n(per[k]["false_alarms_per_1000_normal"], 2))])
        out.append(
            f'<div class="lb-row" tabindex="0" data-tip="{t}"><div class="lb-name"><span class="key fam-{fam}"></span>{esc(name)}</div>'
            f'<div class="st-track"><span class="st-true fam-{fam}" style="width:{tpw:.2f}%"></span>'
            f'<span class="st-false fam-{fam}" style="width:{fpw:.2f}%"></span></div>'
            f'<div class="lb-val num">{a:,.0f}</div></div>'
        )
    return (f'<div class="lb">{"".join(out)}</div>'
            '<ul class="legend inline"><li><span class="sw solid"></span>Alerts on real anomalies</li><li><span class="sw hatch"></span>False alarms</li></ul>')


def timeline(W: int = 816, H: int = 290, R: int = 140, cls: str = "desk", every: int = 1) -> str:
    ph = ops["per_hour"]
    hours = sorted(ph["_sessions"])
    # The log ends a few minutes into its last hour; that sliver would read as a collapse.
    typical = sorted(ph["_sessions"].values())[len(hours) // 2]
    hours = [h for h in hours if ph["_sessions"][h] >= 0.25 * typical]
    lab = [ph["_labelled_anomalies"].get(h, 0) for h in hours]
    series = {
        "pca": [ph["pca"]["tp"].get(h, 0) + ph["pca"]["fp"].get(h, 0) for h in hours],
        BEST: [ph[BEST]["tp"].get(h, 0) + ph[BEST]["fp"].get(h, 0) for h in hours],
    }
    L, T, B = 48, 14, 30
    ymax = max(max(lab), *(max(v) for v in series.values())) * 1.1
    step = 10 ** math.floor(math.log10(ymax / 4))
    tick = max(step, round(ymax / 4 / step) * step)
    X = lambda i: L + (W - L - R) * (i / max(1, len(hours) - 1))
    Y = lambda v: T + (H - T - B) * (1 - v / ymax)
    grid = "".join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="tick" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{int(v):,}</text>'
                   for v in [tick * i for i in range(0, int(ymax // tick) + 1)])
    xl = "".join(f'<text class="tick" x="{X(i):.1f}" y="{H - 8}" text-anchor="middle">{datetime.fromisoformat(h).strftime("%H:00")}</text>' for i, h in enumerate(hours) if i % every == 0)

    def path(vals):
        return "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))

    lines = [f'<path class="ln ln-label" d="{path(lab)}"/>']
    ends = [("Labelled anomalies", lab[-1], "ln-label")]
    for k, vals in series.items():
        fam = META[k][1]
        lines.append(f'<path class="ln fam-{fam}" d="{path(vals)}"/>')
        lines.append("".join(f'<circle class="pt fam-{fam}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4"/>' for i, v in enumerate(vals)))
        ends.append(("PCA alerts" if k == "pca" else "LSTM alerts", vals[-1], f"fam-{fam}"))
    # Direct end labels, spaced so they never overlap.
    ends.sort(key=lambda e: Y(e[1]))
    placed, last = [], -99
    for label, v, _cls in ends:
        y = max(Y(v) + 4, last + 15)
        placed.append(f'<text class="endlab" x="{W - R + 8}" y="{y:.1f}">{esc(label)}</text>')
        last = y
    hover = "".join(
        f'<rect class="hov" x="{X(i) - (W - L - R) / max(1, len(hours) - 1) / 2:.1f}" y="{T}" width="{(W - L - R) / max(1, len(hours) - 1):.1f}" height="{H - T - B}" '
        f'data-tip="{tip(datetime.fromisoformat(h).strftime("11 Nov 2008, %H:00"), [("Labelled anomalies", n(lab[i])), ("PCA alerts", n(series["pca"][i])), ("LSTM alerts", n(series[BEST][i])), ("Sessions starting", n(ph["_sessions"][h]))])}" tabindex="0"/>'
        for i, h in enumerate(hours)
    )
    return f'<svg class="chart {cls}" viewBox="0 0 {W} {H}" role="img" aria-label="Alerts per hour in the test period against labelled anomalies">{grid}{xl}{"".join(lines)}{"".join(placed)}{hover}</svg>'


def latency_chart() -> str:
    keys = ["pca", "lightgbm", "deeplog_topg", BEST]
    lo, hi = math.log10(1), math.log10(48 * 3600)
    x = lambda s: 100 * (math.log10(max(1, s)) - lo) / (hi - lo)
    out = []
    for k in keys:
        if k not in lat:
            continue
        name, fam = META[k]
        s = lat[k]["seconds"]
        t = tip(name, [("Median time to alert", dur(s["p50"])), ("10th to 90th percentile", f"{dur(s['p10'])} to {dur(s['p90'])}"),
                       ("Alert within a minute", pct(lat[k]["within_1_minute"], 0)), ("Median events seen", n(lat[k]["events"]["p50"]))])
        out.append(
            f'<div class="lb-row" tabindex="0" data-tip="{t}"><div class="lb-name"><span class="key fam-{fam}"></span>{esc(name)}</div>'
            f'<div class="lb-track"><span class="lb-ci fam-{fam}" style="left:{x(s["p10"]):.2f}%;width:{max(0.8, x(s["p90"]) - x(s["p10"])):.2f}%"></span>'
            f'<span class="lb-dot fam-{fam}" style="left:{x(s["p50"]):.2f}%"></span></div>'
            f'<div class="lb-val num">{dur(s["p50"])}</div></div>'
        )
    marks = [(1, "1 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"), (6 * 3600, "6 h"), (24 * 3600, "24 h")]
    ticks = "".join(f'<span style="left:{x(s):.2f}%">{lab}</span>' for s, lab in marks)
    tx = x(tout["timeout_s"])
    return (f'<div class="lb lat"><div class="lat-timeout{" flip" if tx > 55 else ""}" style="--x:{tx:.2f}"><span>idle timeout {dur(tout["timeout_s"])}</span></div>'
            f'{"".join(out)}<div class="lb-axis"><div></div><div class="lb-ticks">{ticks}</div><div></div></div></div>'
            '<ul class="legend inline"><li><span class="sw dot"></span>Median time from a session\'s first event to the alert</li><li><span class="sw bar"></span>10th to 90th percentile</li></ul>')


def ablation_chart(W: int = 816, H: int = 300, R: int = 170, cls: str = "desk") -> str:
    rows = ABL["rows"]
    fr = ABL["fractions"]
    L, T, B = 48, 14, 34
    X = lambda f: L + (W - L - R) * (math.log10(f) - math.log10(fr[0])) / (math.log10(fr[-1]) - math.log10(fr[0]))
    lo = 0.5
    Y = lambda v: T + (H - T - B) * (1 - (max(lo, v) - lo) / (1 - lo))
    grid = "".join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="tick" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>' for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    xl = "".join(f'<text class="tick" x="{X(f):.1f}" y="{H - 10}" text-anchor="middle">{f:.0%}</text>' for f in fr)
    names = {"pca": "PCA", "lightgbm": "LightGBM", "iforest": "Isolation Forest", "deeplog_end_logprob": "LSTM + END"}
    fams = {"pca": "count", "lightgbm": "supervised", "iforest": "muted", "deeplog_end_logprob": "sequence"}
    lines, ends = [], []
    for k in ("iforest", "lightgbm", "pca", "deeplog_end_logprob"):
        pts = sorted((r["fraction"], r["f1"], r["train_sessions"]) for r in rows if r["model"] == k)
        band = sorted((r["fraction"], r.get("f1_min", r["f1"]), r.get("f1_max", r["f1"])) for r in rows if r["model"] == k)
        if any(lo_ != hi_ for _, lo_, hi_ in band):
            up = " L".join(f"{X(f):.1f},{Y(h):.1f}" for f, _, h in band)
            dn = " L".join(f"{X(f):.1f},{Y(l_):.1f}" for f, l_, _ in reversed(band))
            lines.append(f'<path class="band fam-{fams[k]}" d="M{up} L{dn} Z"/>')
        d = "M" + " L".join(f"{X(f):.1f},{Y(v):.1f}" for f, v, _ in pts)
        lines.append(f'<path class="ln fam-{fams[k]}" d="{d}"/>')
        for f, v, ns in pts:
            rr = next(r for r in rows if r["model"] == k and r["fraction"] == f)
            extra = [("Range over draws", f"{rr['f1_min']:.3f} to {rr['f1_max']:.3f}")] if "f1_min" in rr and rr["f1_min"] != rr["f1_max"] else []
            t = tip(names[k], [("Training sessions", n(ns)), ("Share of training period", f"{f:.0%}"), ("Test F1", f"{v:.3f}"), *extra])
            lines.append(f'<circle class="pt fam-{fams[k]}" cx="{X(f):.1f}" cy="{Y(v):.1f}" r="4"/><circle class="hit" cx="{X(f):.1f}" cy="{Y(v):.1f}" r="12" data-tip="{t}" tabindex="0"/>')
        ends.append((names[k], pts[-1][1], f"fam-{fams[k]}"))
    ends.sort(key=lambda e: Y(e[1]))
    placed, last = [], -99
    for label, v, _cls in ends:
        y = max(Y(v) + 4, last + 15)
        placed.append(f'<text class="endlab" x="{W - R + 10}" y="{y:.1f}">{esc(label)} <tspan class="num">{v:.3f}</tspan></text>')
        last = y
    return f'<svg class="chart {cls}" viewBox="0 0 {W} {H}" role="img" aria-label="Test F1 against the share of training sessions used">{grid}{xl}{"".join(lines)}{"".join(placed)}</svg>'


def template_table() -> str:
    lines = {t["event_id"]: t["all_lines"] for t in PARSING["templates"]}
    rows = [t for t in EV["templates"] if t["event_id"] not in ("END",)]
    top_s = max(t["sessions"] for t in rows)
    body = []
    for t in rows:
        rate = t["anomaly_rate"]
        body.append(
            f'<tr><td class="mono id">{esc(t["event_id"])}</td><td class="mono tpl">{esc(t["template"])}</td>'
            f'<td class="num">{n(lines.get(t["event_id"], 0))}</td>'
            f'<td><div class="mini"><span style="width:{100 * t["sessions"] / top_s:.1f}%"></span></div><span class="num">{n(t["sessions"])}</span></td>'
            f'<td><div class="mini rate"><span style="width:{100 * rate:.1f}%"></span></div><span class="num">{pct(rate)}</span></td></tr>'
        )
    return ('<div class="tbl tall"><table class="tpl-table"><thead><tr><th>Event</th><th>Template (learned by Drain3)</th><th>Lines</th>'
            '<th>Sessions containing it</th><th>Of those, anomalous</th></tr></thead><tbody>' + "".join(body) + "</tbody></table></div>")


def error_list(model: str, kind: str, label: str) -> str:
    rows = EV["errors"][model][kind][:5]
    items = "".join(
        f'<li><span class="mono">{esc(r["template"])}</span><span class="num lift">{">100" if r["lift"] > 100 else f"{r['lift']:.1f}"}x</span>'
        f'<span class="sub">in {pct(r["share"], 0)} of these sessions vs {pct(r["reference_share"], 1)} of {label}</span></li>'
        for r in rows if r["lift"]
    )
    return f'<ol class="errs">{items}</ol>'


def pipeline_ribbon() -> str:
    p = PIPE
    stages = [
        ("Raw log", n(p["raw_lines"]), "lines, 38.7 hours of a 200+ node Hadoop cluster"),
        ("Bronze", n(p["quarantine_rows"]), "lines quarantined; raw = bronze + quarantine"),
        ("Silver", str(p["templates"]), f"templates, {pct(p['grouping_accuracy'], 2)} grouping accuracy vs LogHub's ground truth"),
        ("Gold", n(p["sessions"]), f"block sessions, only {n(p['distinct_sequences'])} distinct sequences"),
        ("Checks", f"{p['checks_passed']}/{p['checks_total']}", "pipeline checks passed on the full data"),
    ]
    return '<ol class="ribbon">' + "".join(
        f'<li><span class="stage mono">{esc(a)}</span><b class="num">{b}</b><span class="sub">{esc(c)}</span></li>' for a, b, c in stages
    ) + "</ol>"


def censoring() -> str:
    """How the end of the log cuts the test period short."""
    tm = {t["event_id"]: t for t in EV["templates"]}
    ntr, nte = PIPE["per_split"]["train"]["sessions"], PIPE["per_split"]["test"]["sessions"]
    picks = [("T05", "Deleting block"), ("T07", "Served block"), ("T08", "Got exception while serving"), ("T10", "Verification succeeded")]
    out = []
    for e, label in picks:
        if e not in tm:
            continue
        a, b = tm[e]["train_sessions"] / ntr, tm[e]["test_sessions"] / nte
        t = tip(label, [("Training sessions containing it", pct(a)), ("Test sessions containing it", pct(b)), ("Template", tm[e]["template"])])
        out.append(
            f'<div class="cz-row" tabindex="0" data-tip="{t}"><div class="lb-name mono">{esc(label)}</div>'
            f'<div class="cz-bars"><span class="cz-a" style="width:{100 * a:.1f}%"></span><span class="cz-b" style="width:{100 * b:.1f}%"></span></div>'
            f'<div class="cz-val num">{pct(a, 0)} <span>→</span> {pct(b, 0)}</div></div>'
        )
    return ('<div class="cz">' + "".join(out) + '</div><ul class="legend inline"><li><span class="sw cz-a"></span>Training sessions</li>'
            '<li><span class="sw cz-b"></span>Test sessions</li></ul>')


def bench_bars() -> str:
    b = PIPE["matching_benchmark"]
    labels = {"regex": "Spark SQL regex", "udf": "Drain3 in a pandas UDF", "dict": "Lookup join"}
    hi = max(b[k]["seconds"] for k in labels) * 1.1
    return '<div class="lb small">' + "".join(
        f'<div class="lb-row" tabindex="0" data-tip="{tip(labels[k], [("Seconds for 11.2M lines", str(b[k]["seconds"])), ("Lines per second", n(b[k]["lines_per_s"]))])}">'
        f'<div class="lb-name">{labels[k]}</div><div class="st-track"><span class="st-true fam-neutral" style="width:{100 * b[k]["seconds"] / hi:.1f}%"></span></div>'
        f'<div class="lb-val num">{b[k]["seconds"]:.0f} s</div></div>'
        for k in ("regex", "udf", "dict")
    ) + "</div>"


def hero_tape() -> str:
    """A real test session the LSTM caught, drawn as its event sequence."""
    cands = [s for s in DRILL["sessions"] if s["group"] == "pca_missed_lstm_caught" and s["trigger"] is not None]
    cands = cands or [s for s in DRILL["sessions"] if s["group"] == "caught_by_lstm" and s["trigger"] is not None]
    s = min(cands, key=lambda c: c["n_events"])
    names = DRILL["templates"]
    chips = []
    for i, (e, sc) in enumerate(zip(s["events"], s["surprise"])):
        cls = "chip trig" if i == s["trigger"] else "chip"
        t = tip(f"{e}  {'(fires here)' if i == s['trigger'] else ''}".strip(), [("Template", names.get(e, e)), ("Surprise (-log p)", f"{sc:.2f}")])
        chips.append(f'<span class="{cls}" style="--s:{min(1, sc / 8):.3f}" data-tip="{t}" tabindex="0">{esc(e)}</span>')
    return (f'<div class="tape" aria-label="Event sequence of test session {esc(s["block_id"])}">{"".join(chips)}</div>'
            f'<p class="tape-cap mono">{esc(s["block_id"])} · labelled anomalous · PCA missed it · the LSTM fired at event {s["trigger"] + 1} of {len(s["events"])}</p>')


# --------------------------------------------------------------------- page --


def page() -> str:
    pca_ci, best_ci = pca["ci"]["f1"], best["ci"]["f1"]
    diff = EV["paired_differences"][f"pca_minus_{BEST}_f1"]
    lat_pca, lat_lstm = lat["pca"]["seconds"]["p50"], lat["deeplog_topg"]["seconds"]["p50"]
    abl1 = {r["model"]: r for r in ABL["rows"] if r["fraction"] == ABL["fractions"][0]}
    drill_json = json.dumps({"templates": DRILL["templates"], "sessions": DRILL["sessions"]}).replace("</", "<\\/")
    built = datetime.now().strftime("%d %b %Y")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LogLens</title>
<meta name="description" content="Do sequence models really catch anomalous logs? 11.2M HDFS log lines, a chronological split, and a 2009 PCA baseline that still wins.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230b0b0b'/%3E%3Crect x='6' y='9' width='14' height='3' rx='1.5' fill='%23fcfcfb'/%3E%3Crect x='6' y='15' width='20' height='3' rx='1.5' fill='%23eb6834'/%3E%3Crect x='6' y='21' width='10' height='3' rx='1.5' fill='%23fcfcfb'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<header class="top">
  <a class="brand mono" href="#top">LogLens<span>/hdfs</span></a>
  <nav class="mono"><a href="#verdict">Verdict</a><a href="#split">Split</a><a href="#misses">Misses</a><a href="#operate">Operate</a><a href="#data">Data</a><a href="#sessions">Sessions</a></nav>
  <button class="theme" type="button" aria-label="Switch light or dark theme"><span aria-hidden="true"></span></button>
</header>

<main id="top">
<section class="hero">
  <p class="kicker mono">Log anomaly detection · HDFS_v1 · 11,175,629 lines · chronological evaluation</p>
  <h1>A 2009 PCA baseline still beats a <em>DeepLog-style LSTM</em> once the test set comes from the future.</h1>
  <p class="lede">Sessions are split by time: train on the first 60%, choose every threshold on the next 10%, test on the last 30%. Under that rule, principal component analysis on event counts flags anomalous HDFS blocks with an F1 of <b class="num">{pca['f1']:.3f}</b>. The best LSTM reaches <b class="num">{best['f1']:.3f}</b>: it actually catches more anomalies ({pct(best['recall'])} against {pct(pca['recall'])}), but raises {best['fp']:,} false alarms to PCA's {pca['fp']}.</p>
  <div class="tiles">
    <div class="tile"><span class="lab"><span class="key fam-count"></span>PCA on event counts</span><b class="num">{pca['f1']:.3f}</b><span class="sub">F1, 95% interval {pca_ci[0]:.3f} to {pca_ci[1]:.3f}</span></div>
    <div class="tile"><span class="lab"><span class="key fam-sequence"></span>Best LSTM</span><b class="num">{best['f1']:.3f}</b><span class="sub">F1, 95% interval {best_ci[0]:.3f} to {best_ci[1]:.3f}</span></div>
    <div class="tile"><span class="lab">PCA minus LSTM</span><b class="num">{diff['mean'] * 100:+.1f}</b><span class="sub">F1 points, paired interval {diff['ci'][0] * 100:+.1f} to {diff['ci'][1] * 100:+.1f}</span></div>
    <div class="tile"><span class="lab">Random-split inflation</span><b class="num">{gap_best:+.1f}</b><span class="sub">F1 points for the LSTM, {gap_pca:+.1f} for PCA</span></div>
  </div>
  {hero_tape()}
</section>

<section id="verdict" class="sec">
  <div class="sec-head"><span class="idx mono">01</span><h2>Which detector wins on future data?</h2>
  <p>Nine detectors, one chronological test set of 172,519 sessions (3,723 labelled anomalous). Every threshold was chosen on the validation period and applied unchanged.</p></div>
  <div class="sec-body">{leaderboard()}
  <ul class="legend inline fams"><li><span class="key fam-count"></span>Count model</li><li><span class="key fam-sequence"></span>Sequence model</li><li><span class="key fam-supervised"></span>Supervised reference</li><li><span class="key fam-naive"></span>Rule or lookup</li></ul></div>
</section>

<section id="split" class="sec">
  <div class="sec-head"><span class="idx mono">02</span><h2>Does a random split flatter the models?</h2>
  <p>Only the sequence models. The count models barely notice, because HDFS repeats itself: <b class="num">{pct(dup['chrono']['test_seq_seen_in_train'])}</b> of chronological test sessions replay an event sequence already seen in training ({pct(dup['random']['test_seq_seen_in_train'])} under a random split).</p></div>
  <div class="sec-body">
    <h3>Extra F1 from a random split, at the same anomaly rate</h3>
    {gap_bars()}
    <p class="note">The random test set has more anomalies (2.94% against 2.16%), which lifts precision for any detector, so each random-split score is rescaled to the chronological rate before subtracting. LSTM rows are 3-seed means. The WARN rule and the lookup are left out: they get worse under a random split, because the noisiest warnings only occur early in the log.</p>
    <h3>The honest test: sequences never seen in training</h3>
    {unseen_dumbbell()}
    <h3>A time-ordered test set is also the end of the log</h3>
    <p class="prose">The test period is the last 5 hours of a 38.7-hour log, and HDFS blocks are often deleted hours after they are written. So {pct(1 - EV_TPL['T05']['test_sessions'] / PIPE['per_split']['test']['sessions'], 0)} of test sessions end before their delete events, and the read traffic of the early hours has stopped. Every model is judged on these cut-off sessions; a normal session that was merely cut off looks just like one that stopped early.</p>
    {censoring()}
  </div>
</section>

<section id="misses" class="sec">
  <div class="sec-head"><span class="idx mono">03</span><h2>What does each model miss?</h2>
  <p>Each square is one of the {n(ov['anomalies'])} anomalous test sessions. PCA judges how often events happen; DeepLog judges what comes next, and the two miss almost completely different anomalies. Adding an END event lets the LSTM see sessions that stop early, and then it catches nearly everything PCA does.</p></div>
  <div class="sec-body">
    <div class="two waffles">
      <div><h3>Published DeepLog and PCA</h3>{waffle("pca_vs_deeplog", "DeepLog")}</div>
      <div><h3>LSTM with an END event and PCA</h3>{waffle("pca_vs_best_lstm", "the LSTM")}</div>
    </div>
    <div class="two">
      <div><h3>What the LSTM's {n(EV["errors"][BEST]["false_alarms"])} false alarms contain</h3>{error_list(BEST, "in_false_alarms", "normal test sessions")}</div>
      <div><h3>What PCA's {n(EV["errors"]["pca"]["false_alarms"])} false alarms contain</h3>{error_list("pca", "in_false_alarms", "normal test sessions")}</div>
    </div>
    <p class="note">Ranked by how much more often a template appears in these sessions than in normal test sessions. PCA's few false alarms all carry read traffic ("Served block"), which almost vanished from the test period. The LSTM's false alarms lean on replication events. Both models' misses are replication sessions too: the replication templates appear in every missed anomaly against {pct(EV["errors"]["pca"]["in_misses"][0]["reference_share"], 0)} of all anomalies. One anomaly in five simply stops early, which is why the LSTM needs the END event at all.</p>
  </div>
</section>

<section id="operate" class="sec">
  <div class="sec-head"><span class="idx mono">04</span><h2>What would it be like to run?</h2>
  <p>Over the {ops['test_hours']:.1f} test hours, PCA raises about {ops['per_model']['pca']['alerts_per_hour']:,.0f} alerts an hour and almost all are real. Speed is a different story: a count model has to wait for a session to finish, and HDFS sessions can stay open for hours.</p></div>
  <div class="sec-body">
    <h3>Alerts per hour</h3>{alerts_bars()}
    <h3>Alerts and labelled anomalies, hour by hour</h3>{timeline()}{timeline(360, 260, 112, "mob", 2)}<p class="note">By the hour each session started. The log ends 16 minutes into its last hour, so that hour is left out.</p>
    <h3>Time from a session's first event to its alert</h3>{latency_chart()}
    <p class="note">A live system never knows a session has ended; it can only assume so after a stretch of silence. Normal HDFS blocks routinely go quiet for hours before they are deleted, so that stretch has to be {dur(tout['timeout_s'])} (the 99th percentile of the longest pause inside normal training sessions). Count models, and the LSTM whenever it fires on the END event, can only alert after that wait. Published DeepLog fires on the event itself: median {dur(lat_lstm)} against {dur(lat_pca)} for PCA.</p>
  </div>
</section>

<section id="data" class="sec">
  <div class="sec-head"><span class="idx mono">05</span><h2>How much training data, and how clean?</h2>
  <p>Less than you would think, but PCA gets fragile. With 1% of the training period ({n(abl1['pca']['train_sessions'])} sessions, about what DeepLog was trained on), PCA's F1 ranges from {abl1['pca']['f1_min']:.3f} to {abl1['pca']['f1_max']:.3f} depending on which sessions it happens to see; LightGBM barely moves, and the LSTM scores {abl1['deeplog_end_logprob']['f1']:.3f}.</p></div>
  <div class="sec-body">
    <h3>Test F1 against the share of training sessions used</h3>{ablation_chart()}{ablation_chart(360, 280, 128, "mob")}<p class="note">Shaded bands are the range over 5 random draws of the training sessions for the baselines; the LSTM is one draw per size.</p>
    <h3>Pipeline</h3>{pipeline_ribbon()}
    <div class="two">
      <div><h3>Tagging 11.2M lines with templates</h3>{bench_bars()}<p class="note">Same templates, three matchers; all three agree on every line.</p></div>
      <div><h3>Why the parser threshold matters</h3><p class="prose">At Drain3's default similarity threshold (0.4), "PacketResponder N for block X <span class="mono">terminating</span>" (1.7M normal lines) and "... <span class="mono">Interrupted</span>" (49 failure lines) became one template, hiding a failure signal inside the most common event. Grouping accuracy was {pct(SWEEP['results'][1]['test']['grouping_accuracy'])}; at the validation-chosen 0.9 it is {pct(PIPE['grouping_accuracy_test'], 2)} on test.</p></div>
    </div>
    <h3>Every learned template</h3>{template_table()}
  </div>
</section>

<section id="sessions" class="sec">
  <div class="sec-head"><span class="idx mono">06</span><h2>Read the sessions yourself</h2>
  <p>Real test sessions, as event sequences. Each event is shaded by how surprised the LSTM was to see it; the outlined event is where it fired. Hover an event for its log template.</p></div>
  <div class="sec-body">
    <div class="toggle groups" role="tablist" aria-label="Session group">
      <button role="tab" aria-selected="true" data-group="pca_missed_lstm_caught">PCA missed, LSTM caught</button>
      <button role="tab" aria-selected="false" data-group="lstm_missed_pca_caught">LSTM missed, PCA caught</button>
      <button role="tab" aria-selected="false" data-group="caught_by_lstm">LSTM caught</button>
      <button role="tab" aria-selected="false" data-group="lstm_false_alarm">LSTM false alarm</button>
    </div>
    <div class="drill"><ul class="drill-list" aria-label="Sessions"></ul><div class="drill-view" aria-live="polite"></div></div>
  </div>
</section>
</main>

<footer class="foot">
  <div class="foot-grid">
    <div><h4>About</h4><p>LogLens tests whether log anomaly detectors keep their published scores when evaluated on later data than they were trained on. It is a research benchmark on public 2008 data, not a production detector.</p><p><a href="{REPO}" target="_blank" rel="noopener">Source code on GitHub</a></p></div>
    <div><h4>Data</h4><ul>
      <li><a href="https://github.com/logpai/loghub" target="_blank" rel="noopener">LogHub</a>: the HDFS_v1 log (11.2M lines), block labels, ground-truth templates and event traces</li>
      <li><a href="https://zenodo.org/records/8196385" target="_blank" rel="noopener">LogHub on Zenodo</a>: the archive the pipeline downloads and checksums</li>
      <li><a href="https://github.com/logpai/Drain3" target="_blank" rel="noopener">Drain3</a>: the template miner fitted on the training period</li></ul></div>
    <div><h4>References</h4><ul>
      <li><a href="https://doi.org/10.1145/1629575.1629587" target="_blank" rel="noopener">Xu et al., Detecting Large-Scale System Problems by Mining Console Logs (SOSP 2009)</a></li>
      <li><a href="https://doi.org/10.1145/3133956.3134015" target="_blank" rel="noopener">Du et al., DeepLog (CCS 2017)</a></li>
      <li><a href="https://doi.org/10.1109/ICWS.2017.13" target="_blank" rel="noopener">He et al., Drain (ICWS 2017)</a></li>
      <li><a href="https://doi.org/10.1109/ICDM.2008.17" target="_blank" rel="noopener">Liu et al., Isolation Forest (ICDM 2008)</a></li>
      <li><a href="https://arxiv.org/abs/2202.04301" target="_blank" rel="noopener">Le and Zhang, Log-based Anomaly Detection with Deep Learning: How Far Are We? (ICSE 2022)</a></li>
      <li><a href="https://arxiv.org/abs/2008.06448" target="_blank" rel="noopener">Zhu et al., Loghub (ISSRE 2023)</a></li></ul></div>
  </div>
  <div class="bar"><span>Built by <b>Sagnik</b></span><a href="https://github.com/HalcyonVector" target="_blank" rel="noopener">GitHub @HalcyonVector</a><a href="{REPO}" target="_blank" rel="noopener">Repository</a><a href="{REPO}/blob/main/LICENSE" target="_blank" rel="noopener">MIT License</a><span class="muted">Built {built}</span></div>
</footer>

<div class="tt" role="tooltip" hidden></div>
<script type="application/json" id="drill-data">{drill_json}</script>
<script>{JS}</script>
</body>
</html>
"""


CSS = r"""
:root{
  color-scheme: light;
  --page:#f6f5f1; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --count:#2a78d6; --sequence:#eb6834; --supervised:#1baf7a; --naive:#a8a69f; --neutral:#52514e;
  --count-wash:rgba(42,120,214,.18); --sequence-wash:rgba(235,104,52,.20); --supervised-wash:rgba(27,175,122,.20); --naive-wash:rgba(137,135,129,.20);
  --both:#d6d4cc; --none:#0b0b0b;
  --heat-0:#fbe6dc; --heat-1:#d95926;
  --shadow:0 1px 0 rgba(11,11,11,.04), 0 8px 30px rgba(11,11,11,.08);
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    --count:#3987e5; --sequence:#d95926; --supervised:#199e70; --naive:#6f6e69; --neutral:#c3c2b7;
    --count-wash:rgba(57,135,229,.28); --sequence-wash:rgba(217,89,38,.30); --supervised-wash:rgba(25,158,112,.30); --naive-wash:rgba(137,135,129,.28);
    --both:#3a3a37; --none:#f0efec;
    --heat-0:#3a2219; --heat-1:#ff7a45;
    --shadow:0 1px 0 rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --count:#3987e5; --sequence:#d95926; --supervised:#199e70; --naive:#6f6e69; --neutral:#c3c2b7;
  --count-wash:rgba(57,135,229,.28); --sequence-wash:rgba(217,89,38,.30); --supervised-wash:rgba(25,158,112,.30); --naive-wash:rgba(137,135,129,.28);
  --both:#3a3a37; --none:#f0efec;
  --heat-0:#3a2219; --heat-1:#ff7a45;
  --shadow:0 1px 0 rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
[hidden]{display:none!important}
.sec,.hero{scroll-margin-top:64px}
html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;overflow-x:hidden}
.mono{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace}
.num{font-variant-numeric:tabular-nums}
a{color:inherit;text-decoration-color:var(--axis);text-underline-offset:3px}
a:hover{text-decoration-color:var(--ink)}
b{font-weight:600}
.top{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:24px;padding:12px max(16px,calc((100vw - 1180px)/2));background:color-mix(in srgb,var(--page) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--grid)}
.brand{font-weight:500;text-decoration:none;font-size:15px}
.brand span{color:var(--muted)}
.top nav{display:flex;gap:18px;font-size:13px;flex:1;overflow-x:auto;scrollbar-width:none}
.top nav a{text-decoration:none;color:var(--ink-2);white-space:nowrap}
.top nav a:hover{color:var(--ink)}
.theme{width:34px;height:34px;border-radius:50%;border:1px solid var(--grid);background:var(--surface);cursor:pointer;display:grid;place-items:center;flex:none}
.theme span{width:14px;height:14px;border-radius:50%;background:linear-gradient(90deg,var(--ink) 50%,transparent 50%);border:1.5px solid var(--ink)}
main{max-width:1180px;margin:0 auto;padding:0 16px}
.hero{padding:64px 0 40px}
.kicker{font-size:12.5px;color:var(--ink-2);letter-spacing:.02em;margin:0 0 18px}
h1{font-size:clamp(32px,5.2vw,60px);line-height:1.04;letter-spacing:-.025em;font-weight:600;margin:0;max-width:20ch}
h1 em{font-style:normal;background:linear-gradient(transparent 62%,var(--sequence-wash) 62%)}
.lede{font-size:clamp(17px,1.6vw,19.5px);color:var(--ink-2);max-width:62ch;margin:26px 0 34px}
.lede b{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--grid);border:1px solid var(--grid);border-radius:14px;overflow:hidden}
.tile{background:var(--surface);padding:18px 20px 20px;display:flex;flex-direction:column;gap:4px}
.tile .lab{font-size:13px;color:var(--ink-2);display:flex;align-items:center;gap:8px}
.tile b{font-size:clamp(30px,3.4vw,42px);letter-spacing:-.02em;line-height:1.1;font-weight:600}
.tile .sub{font-size:12.5px;color:var(--muted)}
.key{display:inline-block;width:10px;height:10px;border-radius:3px;flex:none}
.key.fam-count{background:var(--count)} .key.fam-sequence{background:var(--sequence)} .key.fam-supervised{background:var(--supervised)} .key.fam-naive{background:var(--naive)}
.tape{display:flex;flex-wrap:wrap;gap:22px 5px;margin:34px 0 8px;padding:28px 16px 16px;background:var(--surface);border:1px solid var(--grid);border-radius:12px}
.chip{font:500 12px/1 "IBM Plex Mono",monospace;padding:7px 8px;border-radius:6px;background:color-mix(in srgb,var(--heat-1) calc(var(--s) * 100%),var(--heat-0));color:var(--ink);cursor:default;outline:none}
.chip{color:color-mix(in srgb,#fff calc(var(--s) * 100%),var(--ink))}
.chip.trig{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--ink);position:relative}
.chip.trig::after{content:"fires";position:absolute;left:50%;top:-20px;transform:translateX(-50%);font-size:10.5px;color:var(--ink);letter-spacing:.04em}
.chip:focus-visible{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--count)}
.tape-cap{font-size:12px;color:var(--muted);margin:6px 0 0}
.sec{display:grid;grid-template-columns:300px minmax(0,1fr);gap:48px;padding:56px 0;border-top:1px solid var(--grid)}
.sec-head{position:sticky;top:76px;align-self:start}
.idx{font-size:12px;color:var(--muted);display:block;margin-bottom:10px}
h2{font-size:clamp(24px,2.4vw,30px);line-height:1.15;letter-spacing:-.015em;margin:0 0 14px;font-weight:600}
.sec-head p{color:var(--ink-2);margin:0;font-size:15.5px}
h3{font-size:14px;font-weight:600;margin:34px 0 14px;color:var(--ink)}
.sec-body>h3:first-child,.sec-body>.toggle:first-child{margin-top:0}
.note{font-size:13px;color:var(--muted);margin:14px 0 0;max-width:72ch}
.prose{font-size:14.5px;color:var(--ink-2);margin:0}
.toggle{display:inline-flex;gap:4px;padding:4px;background:var(--surface);border:1px solid var(--grid);border-radius:999px;margin-bottom:18px;flex-wrap:wrap}
.toggle button{font:500 13px "IBM Plex Sans",sans-serif;border:0;background:transparent;color:var(--ink-2);padding:7px 14px;border-radius:999px;cursor:pointer}
.toggle button[aria-selected="true"]{background:var(--ink);color:var(--page)}
.toggle.groups{border-radius:14px}
.toggle.groups button{border-radius:10px}
.lb{display:flex;flex-direction:column;gap:2px;position:relative}
.lb-row,.gb-row{display:grid;grid-template-columns:230px 1fr 64px;align-items:center;gap:14px;padding:7px 8px;border-radius:8px;outline:none}
.lb-row:hover,.lb-row:focus-visible,.gb-row:hover,.gb-row:focus-visible{background:var(--surface)}
.lb-row.is-best .lb-name{font-weight:600}
.lb-name{font-size:14px;display:flex;align-items:center;gap:9px;min-width:0}
.lb-val{font-size:14px;text-align:right;color:var(--ink)}
.lb-track,.gb-track,.st-track{position:relative;height:22px}
.lb-track::before,.gb-track::before{content:"";position:absolute;left:0;right:0;top:50%;height:1px;background:var(--grid)}
.lb-ci{position:absolute;top:50%;height:8px;transform:translateY(-50%);border-radius:4px}
.lb-ci.fam-count{background:var(--count-wash)} .lb-ci.fam-sequence{background:var(--sequence-wash)} .lb-ci.fam-supervised{background:var(--supervised-wash)} .lb-ci.fam-naive{background:var(--naive-wash)}
.lb-dot{position:absolute;top:50%;width:12px;height:12px;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px var(--page)}
.lb-row:hover .lb-dot,.lb-row:focus-visible .lb-dot{box-shadow:0 0 0 2px var(--surface)}
.lb-dot.fam-count{background:var(--count)} .lb-dot.fam-sequence{background:var(--sequence)} .lb-dot.fam-supervised{background:var(--supervised)} .lb-dot.fam-naive{background:var(--naive)}
.lb-dot.is-cut{border-radius:3px}
.db-line{position:absolute;top:50%;height:2px;transform:translateY(-50%);background:var(--axis)}
.db-all{position:absolute;top:50%;width:12px;height:12px;border-radius:50%;transform:translate(-50%,-50%);border:2px solid var(--ink-2);background:var(--page)}
.lb-axis{display:grid;grid-template-columns:230px 1fr 64px;gap:14px;padding:4px 8px 0}
.lb-ticks{position:relative;height:18px;font-size:11.5px;color:var(--muted)}
.lb-ticks span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.gb-zero{position:absolute;top:0;bottom:0;width:1px;background:var(--axis)}
.gb-bar{position:absolute;top:50%;height:12px;transform:translateY(-50%);border-radius:0 4px 4px 0}
.gb-bar.neg{border-radius:4px 0 0 4px}
.gb-bar.fam-count{background:var(--count)} .gb-bar.fam-sequence{background:var(--sequence)} .gb-bar.fam-supervised{background:var(--supervised)}
.st-track{display:flex;align-items:center;gap:2px}
.st-track span{height:12px}
.st-true{border-radius:0}
.st-true.fam-count{background:var(--count)} .st-true.fam-sequence{background:var(--sequence)} .st-true.fam-supervised{background:var(--supervised)} .st-true.fam-naive{background:var(--naive)} .st-true.fam-neutral{background:var(--neutral);border-radius:0 4px 4px 0}
.st-false{border-radius:0 4px 4px 0;background-image:repeating-linear-gradient(45deg,currentColor 0 2px,transparent 2px 5px)}
.st-false.fam-count{color:var(--count);box-shadow:inset 0 0 0 1px var(--count)} .st-false.fam-sequence{color:var(--sequence);box-shadow:inset 0 0 0 1px var(--sequence)} .st-false.fam-supervised{color:var(--supervised);box-shadow:inset 0 0 0 1px var(--supervised)} .st-false.fam-naive{color:var(--naive);box-shadow:inset 0 0 0 1px var(--naive)}
.lb.small .lb-row{grid-template-columns:180px 1fr 56px}
.legend{list-style:none;padding:0;margin:14px 0 0;display:flex;flex-wrap:wrap;gap:8px 22px;font-size:13px;color:var(--ink-2)}
.legend li{display:flex;align-items:center;gap:8px}
.legend b{color:var(--ink)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px}
.sw.w-both{background:var(--both)} .sw.w-pca{background:var(--count)} .sw.w-lstm{background:var(--sequence)} .sw.w-none{background:var(--none)}
.sw.ring{border-radius:50%;border:2px solid var(--ink-2)} .sw.dot{border-radius:50%;background:var(--ink-2)}
.sw.bar{width:18px;height:8px;border-radius:4px;background:var(--axis)}
.sw.solid{background:var(--ink-2)} .sw.hatch{color:var(--ink-2);box-shadow:inset 0 0 0 1px var(--ink-2);background-image:repeating-linear-gradient(45deg,currentColor 0 2px,transparent 2px 5px)}
.waffle{margin:0}
.waffle svg{width:100%;height:auto;display:block}
.waffle rect.w-both{fill:var(--both)} .waffle rect.w-pca{fill:var(--count)} .waffle rect.w-lstm{fill:var(--sequence)} .waffle rect.w-none{fill:var(--none)}
.hit-strip{display:flex;height:8px;margin-top:10px;border-radius:4px;overflow:hidden;gap:2px}
.hit-strip .w{flex:var(--n) 1 0;outline:none}
.hit-strip .w-both{background:var(--both)} .hit-strip .w-pca{background:var(--count)} .hit-strip .w-lstm{background:var(--sequence)} .hit-strip .w-none{background:var(--none)}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:32px}
.sec-body,.sec-head{min-width:0}
.errs{list-style:none;counter-reset:e;padding:0;margin:0;display:flex;flex-direction:column;gap:10px}
.errs li{display:grid;grid-template-columns:1fr auto;gap:2px 12px;font-size:13px;padding:10px 12px;background:var(--surface);border:1px solid var(--grid);border-radius:10px}
.errs .mono{font-size:12px;overflow-wrap:anywhere}
.errs .lift{font-weight:600;font-size:14px}
.errs .sub{grid-column:1/-1;color:var(--muted);font-size:12px}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart.mob{display:none}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .tick{fill:var(--muted);font:12px "IBM Plex Sans",sans-serif}
.chart .endlab{fill:var(--ink-2);font:12.5px "IBM Plex Sans",sans-serif}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .ln.fam-count{stroke:var(--count)} .chart .ln.fam-sequence{stroke:var(--sequence)} .chart .ln.fam-supervised{stroke:var(--supervised)} .chart .ln.fam-muted{stroke:var(--naive)}
.chart .ln.ln-label{stroke:var(--ink);stroke-width:1.5}
.chart .pt{stroke:var(--surface);stroke-width:2}
.chart .pt.fam-count{fill:var(--count)} .chart .pt.fam-sequence{fill:var(--sequence)} .chart .pt.fam-supervised{fill:var(--supervised)} .chart .pt.fam-muted{fill:var(--naive)}
.chart .hov,.chart .hit{fill:transparent;outline:none;cursor:default}
.chart .hov:hover,.chart .hov:focus-visible{fill:var(--ring)}
.chart .band{stroke:none;opacity:.16}
.chart .band.fam-count{fill:var(--count)} .chart .band.fam-supervised{fill:var(--supervised)} .chart .band.fam-muted{fill:var(--naive)}
.cz{display:flex;flex-direction:column;gap:4px;margin-top:16px}
.cz-row{display:grid;grid-template-columns:230px 1fr 110px;gap:14px;align-items:center;padding:6px 8px;border-radius:8px;outline:none}
.cz-row:hover,.cz-row:focus-visible{background:var(--surface)}
.cz-row .lb-name{font-size:12.5px}
.cz-bars{display:flex;flex-direction:column;gap:3px}
.cz-bars span{display:block;height:7px;border-radius:0 4px 4px 0;min-width:2px}
.cz-a,.sw.cz-a{background:var(--axis)} .cz-b,.sw.cz-b{background:var(--ink)}
.cz-val{font-size:13px;text-align:right;white-space:nowrap} .cz-val span{color:var(--muted)}
.waffles h3{margin-top:0}
.lat{padding-top:26px}
.lat-timeout{position:absolute;top:0;bottom:22px;left:calc(252px + (100% - 338px) * var(--x) / 100);border-left:1px solid var(--ink-2);pointer-events:none}
.lat-timeout span{position:absolute;top:0;left:6px;font-size:11.5px;color:var(--ink-2);white-space:nowrap}
.lat-timeout.flip span{left:auto;right:6px}
.ribbon{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--grid);border:1px solid var(--grid);border-radius:12px;overflow:hidden}
.ribbon li{background:var(--surface);padding:14px 16px;display:flex;flex-direction:column;gap:2px;position:relative}
.ribbon li:not(:last-child)::after{content:"";position:absolute;right:-6px;top:50%;width:10px;height:10px;background:var(--surface);border-top:1px solid var(--grid);border-right:1px solid var(--grid);transform:translateY(-50%) rotate(45deg);z-index:1}
.ribbon .stage{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.ribbon b{font-size:22px;letter-spacing:-.01em}
.ribbon .sub{font-size:12px;color:var(--ink-2)}
.tbl{overflow-x:auto;border:1px solid var(--grid);border-radius:10px;background:var(--surface)}
.tbl.tall{max-height:520px;overflow-y:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th{position:sticky;top:0;background:var(--surface);text-align:left;font-weight:600;color:var(--ink-2);padding:10px 12px;border-bottom:1px solid var(--grid);white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--grid);vertical-align:middle}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right}
td.mono{font-family:"IBM Plex Mono",monospace;font-size:12px}
td.tpl{min-width:320px;overflow-wrap:anywhere}
td.id{color:var(--ink-2)}
.mini{display:inline-block;width:70px;height:6px;background:var(--grid);border-radius:3px;margin-right:8px;vertical-align:middle;overflow:hidden}
.mini span{display:block;height:100%;background:var(--neutral);border-radius:3px}
.mini.rate span{background:var(--sequence)}
details{margin-top:16px}
summary{cursor:pointer;font-size:13px;color:var(--ink-2)}
details .tbl{margin-top:10px}
.drill{display:grid;grid-template-columns:260px minmax(0,1fr);gap:20px;min-height:320px}
.drill-list{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:4px;max-height:480px;overflow-y:auto}
.drill-list button{width:100%;text-align:left;font:13px "IBM Plex Mono",monospace;padding:9px 11px;border:1px solid transparent;border-radius:9px;background:transparent;color:var(--ink-2);cursor:pointer;display:flex;justify-content:space-between;gap:8px}
.drill-list button span{color:var(--muted)}
.drill-list button[aria-pressed="true"]{background:var(--surface);border-color:var(--grid);color:var(--ink)}
.drill-view{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:18px}
.drill-view .meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin-bottom:14px}
.drill-view .meta b{color:var(--ink)}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px}
.badge{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--grid);color:var(--ink-2)}
.badge.on{color:var(--ink);border-color:var(--ink-2)}
.drill-view .tape{margin:0;padding:22px 0 0;border:0;background:none}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);margin-top:14px}
.scale i{display:block;width:120px;height:8px;border-radius:4px;background:linear-gradient(90deg,var(--heat-0),var(--heat-1))}
.tt{position:fixed;z-index:50;pointer-events:none;max-width:320px;background:var(--surface);color:var(--ink);border:1px solid var(--grid);border-radius:10px;padding:10px 12px;box-shadow:var(--shadow);font-size:12.5px}
.tt .t{font-weight:600;margin-bottom:6px;color:var(--ink-2);font-size:12px}
.tt .r{display:flex;justify-content:space-between;gap:16px}
.tt .r b{font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600;text-align:right;overflow-wrap:anywhere}
.tt .r span{color:var(--ink-2)}
.foot{border-top:1px solid var(--grid);margin-top:40px;background:var(--surface)}
.foot-grid{max-width:1180px;margin:0 auto;padding:40px 16px 24px;display:grid;grid-template-columns:1fr 1.2fr 1.6fr;gap:40px;font-size:13.5px;color:var(--ink-2)}
.foot h4{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px;font-weight:500}
.foot ul{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:9px}
.foot p{margin:0 0 10px}
.foot .bar{max-width:1180px;margin:0 auto;padding:16px;display:flex;flex-wrap:wrap;gap:8px 22px;font-size:13px;border-top:1px solid var(--grid);color:var(--ink-2)}
.foot .bar b{color:var(--ink)}
.muted{color:var(--muted)}
@media (max-width:960px){
  .sec{grid-template-columns:minmax(0,1fr);gap:22px;padding:44px 0}
  .sec-head{position:static}
  .tiles{grid-template-columns:repeat(2,1fr)}
  .ribbon{grid-template-columns:1fr}
  .ribbon li:not(:last-child)::after{display:none}
  .foot-grid{grid-template-columns:1fr;gap:26px}
}
@media (max-width:640px){
  .hero{padding:40px 0 28px}
  .top nav{display:none}
  .top{justify-content:space-between}
  .lb-row,.gb-row{grid-template-columns:1fr 56px;grid-template-areas:"n v" "t t";row-gap:4px}
  .lb-row .lb-name,.gb-row .lb-name{grid-area:n} .lb-row .lb-val,.gb-row .lb-val{grid-area:v} .lb-track,.gb-track,.st-track{grid-area:t}
  .lb-axis{grid-template-columns:1fr} .lb-axis>div:first-child,.lb-axis>div:last-child{display:none}
  .lb.small .lb-row{grid-template-columns:1fr 56px}
  .lat-timeout{left:calc(8px + (100% - 16px) * var(--x) / 100)}
  .two{grid-template-columns:minmax(0,1fr)}
  .chart.desk{display:none} .chart.mob{display:block}
  .cz-row{grid-template-columns:1fr 96px;grid-template-areas:"n v" "b b";row-gap:4px} .cz-row .lb-name{grid-area:n} .cz-row .cz-val{grid-area:v} .cz-row .cz-bars{grid-area:b}
  .drill{grid-template-columns:minmax(0,1fr)}
  .drill-list{flex-direction:row;overflow-x:auto;max-height:none}
  .drill-list button{white-space:nowrap}
  .tiles{grid-template-columns:1fr 1fr}
  .tile{padding:14px}
}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media (forced-colors:active){.lb-dot,.key,.sw,.gb-bar,.st-true{forced-color-adjust:none}}
"""

JS = r"""
(() => {
  const root = document.documentElement;
  try { const t = localStorage.getItem('loglens-theme'); if (t) root.dataset.theme = t; } catch (e) {}
  document.querySelector('.theme').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('loglens-theme', root.dataset.theme); } catch (e) {}
  });

  // Tooltips: one element, filled with textContent only.
  const tt = document.querySelector('.tt');
  function show(el, x, y) {
    let d; try { d = JSON.parse(el.dataset.tip); } catch (e) { return; }
    tt.replaceChildren();
    const t = document.createElement('div'); t.className = 't'; t.textContent = d.t; tt.append(t);
    for (const [k, v] of d.r) {
      const r = document.createElement('div'); r.className = 'r';
      const a = document.createElement('span'); a.textContent = k;
      const b = document.createElement('b'); b.textContent = v;
      r.append(a, b); tt.append(r);
    }
    tt.hidden = false;
    const w = tt.offsetWidth, h = tt.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > innerWidth - 8) left = x - w - 14;
    if (top + h > innerHeight - 8) top = y - h - 14;
    tt.style.left = Math.max(8, left) + 'px'; tt.style.top = Math.max(8, top) + 'px';
  }
  const hide = () => { tt.hidden = true; };
  document.addEventListener('pointermove', e => {
    const el = e.target.closest('[data-tip]');
    if (el) show(el, e.clientX, e.clientY); else hide();
  });
  document.addEventListener('focusin', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) return hide();
    const r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.bottom);
  });
  document.addEventListener('scroll', hide, { passive: true });

  // Metric toggle on the leaderboard.
  document.querySelectorAll('.toggle:not(.groups) button').forEach(btn => btn.addEventListener('click', () => {
    btn.parentElement.querySelectorAll('button').forEach(b => b.setAttribute('aria-selected', b === btn));
    document.querySelectorAll('.lb[data-metric]').forEach(l => l.hidden = l.dataset.metric !== btn.dataset.show);
  }));

  // Session drill-down.
  const data = JSON.parse(document.getElementById('drill-data').textContent);
  const list = document.querySelector('.drill-list');
  const view = document.querySelector('.drill-view');
  const MODEL = { pca: 'PCA', lightgbm: 'LightGBM', iforest: 'Isolation Forest', deeplog_end_tuned_logprob: 'LSTM' };
  const fmt = s => s < 60 ? s + ' s' : s < 3600 ? Math.round(s / 60) + ' min' : (s / 3600).toFixed(1) + ' h';

  function render(s) {
    view.replaceChildren();
    const meta = document.createElement('div'); meta.className = 'meta';
    const add = (label, value) => { const sp = document.createElement('span'); sp.append(label + ' '); const b = document.createElement('b'); b.textContent = value; sp.append(b); meta.append(sp); };
    add('Block', s.block_id); add('Label', s.label ? 'anomalous' : 'normal');
    add('Events', String(s.n_events)); add('Open for', fmt(s.duration_s));
    add('Sessions with this exact sequence', s.sessions_like_this.toLocaleString());
    const badges = document.createElement('div'); badges.className = 'badges';
    for (const [k, v] of Object.entries(MODEL)) {
      const b = document.createElement('span'); const on = s.flagged_by.includes(k);
      b.className = 'badge' + (on ? ' on' : ''); b.textContent = v + (on ? ' flags it' : ' passes it'); badges.append(b);
    }
    const tape = document.createElement('div'); tape.className = 'tape';
    s.events.forEach((e, i) => {
      const c = document.createElement('span');
      c.className = 'chip' + (i === s.trigger ? ' trig' : '');
      c.style.setProperty('--s', Math.min(1, s.surprise[i] / 8).toFixed(3));
      c.textContent = e; c.tabIndex = 0;
      c.dataset.tip = JSON.stringify({ t: e + (i === s.trigger ? '  (fires here)' : ''), r: [['Template', data.templates[e] || e], ['Surprise (-log p)', s.surprise[i].toFixed(2)], ['Position', (i + 1) + ' of ' + s.events.length]] });
      tape.append(c);
    });
    const scale = document.createElement('div'); scale.className = 'scale';
    scale.append('Expected'); scale.append(document.createElement('i')); scale.append('Surprising');
    view.append(meta, badges, tape, scale);
  }
  function group(g) {
    document.querySelectorAll('.toggle.groups button').forEach(b => b.setAttribute('aria-selected', b.dataset.group === g));
    list.replaceChildren();
    const items = data.sessions.filter(s => s.group === g);
    items.forEach((s, i) => {
      const li = document.createElement('li'); const b = document.createElement('button');
      b.type = 'button'; b.setAttribute('aria-pressed', i === 0);
      b.append(s.block_id.slice(0, 16) + '…');
      const n = document.createElement('span'); n.textContent = s.n_events + ' ev'; b.append(n);
      b.addEventListener('click', () => { list.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b)); render(s); });
      li.append(b); list.append(li);
    });
    if (items.length) render(items[0]); else view.textContent = 'No sessions in this group.';
  }
  document.querySelectorAll('.toggle.groups button').forEach(b => b.addEventListener('click', () => group(b.dataset.group)));
  group('pca_missed_lstm_caught');
})();
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
