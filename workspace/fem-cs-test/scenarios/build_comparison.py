"""Build the FEM 4-way KPI comparison deliverable (baseline vs S1 / S2 / S3).

Project : fem-cs-test
Scenarios:
  baseline-fem-hpfd-network-0386   (reference)
  s1-fem-hpfd-network-0386         (review cycle: lsk 30->45 / 60->90)
  s2-fem-hpfd-network-0386         (line capacity: 24->16 h/day)
  s3-fem-hpfd-network-0386         (AO standardized: 0/5/15d = 75/5/20%)

This script does NOT re-query the database. It consumes the per-KPI extract
CSVs already written by the per-scenario analysis builders, so the comparison
is guaranteed to be methodologically identical to the four individual reports.
For each scenario it auto-discovers the latest `analysis/<tag>/extracts` folder.

Run from repo root with the venv interpreter:
  .\\.venv\\Scripts\\python.exe .\\workspace\\fem-cs-test\\scenarios\\build_comparison.py
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- identity -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
PROJECT = "fem-cs-test"
DBNAME = "fem_test"
LINE = "HPFD"

# (key, scenario folder, run_id, lever one-liner)
SCENARIOS = [
    ("Baseline", "baseline-fem-hpfd-network-0386",
     "db_baseline-fem-hpfd-network-0386_20260602_223559", "reference (no lever)"),
    ("S1", "s1-fem-hpfd-network-0386",
     "db_s1-fem-hpfd-network-0386_20260630_142419",
     "review cycle lengthened: M4_MaterialLocationLineCfg.lsk 30\u219245 (19 rows), 60\u219290 (7 rows)"),
    ("S2", "s2-fem-hpfd-network-0386",
     "db_s2-fem-hpfd-network-0386_20260630_142529",
     "line capacity cut: M4_LineCapacity.capacity 24\u219216 h/day (all 91 days)"),
    ("S3", "s3-fem-hpfd-network-0386",
     "db_s3-fem-hpfd-network-0386_20260630_142620",
     "AO profile standardized: M1_AOConfig advance-days 0d=75% / 5d=5% / 15d=20%"),
]
KEYS = [s[0] for s in SCENARIOS]
BASE = "Baseline"

MONTHS = ["2026-02", "2026-03"]
MONTH_LABEL = {"2026-02": "Feb 2026", "2026-03": "Mar 2026"}
SCOPE_DCS = ["A668", "A672", "A673", "A680", "A715", "A716", "C810", "C816"]

COLOR = {"Baseline": "#4e79a7", "S1": "#f28e2b", "S2": "#59a14f", "S3": "#e15759"}

RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M") + "-fem-compare"
OUT_DIR = ROOT / "workspace" / PROJECT / "comparisons" / RUN_TAG
EXTRACTS_OUT = OUT_DIR / "extracts"


# --- locate + load extracts -----------------------------------------------------
def latest_extracts(scen_folder: str) -> Path:
    adir = ROOT / "workspace" / PROJECT / "scenarios" / scen_folder / "analysis"
    cands = [p for p in adir.glob("*/extracts") if (p / "kpi1_tof_by_month.csv").exists()]
    if not cands:
        raise SystemExit(f"no analysis extracts found for {scen_folder} under {adir}")
    cands.sort(key=lambda p: p.parent.stat().st_mtime)
    return cands[-1]


def load_all() -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}
    src: dict[str, str] = {}
    for key, folder, _rid, _lever in SCENARIOS:
        ex = latest_extracts(folder)
        src[key] = str(ex.parent.relative_to(ROOT)).replace("\\", "/")
        data[key] = {
            "tof_m": pd.read_csv(ex / "kpi1_tof_by_month.csv"),
            "tof_loc": pd.read_csv(ex / "kpi1_tof_by_location_month.csv"),
            "co_m": pd.read_csv(ex / "kpi2_changeover_by_month.csv"),
            "dfc_m": pd.read_csv(ex / "kpi4_dfc_by_month.csv"),
            "ptt": pd.read_csv(ex / "kpi5_production_time_by_month.csv"),
            "apq": pd.read_csv(ex / "kpi6_apq_by_line.csv"),
            "deep": pd.read_csv(ex / "kpi7_deepdive_by_line.csv"),
        }
    return data, src


# --- wide builders (index = category, columns = scenario) -----------------------
def month_wide(data, kpi, col):
    out = pd.DataFrame(index=MONTHS)
    for key in KEYS:
        s = data[key][kpi].set_index("month")[col]
        out[key] = [s.get(m, np.nan) for m in MONTHS]
    out.index = [MONTH_LABEL[m] for m in MONTHS]
    return out


def scalar_row(data, kpi, col):
    return {key: float(data[key][kpi][col].iloc[0]) for key in KEYS}


# --- charts ---------------------------------------------------------------------
def fig_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def grouped_bar(wide, ylabel, title, pct=False, fmt="{:.0f}"):
    cats = list(wide.index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    n = len(KEYS)
    x = np.arange(len(cats))
    w = 0.8 / n
    for i, key in enumerate(KEYS):
        vals = wide[key].values.astype(float)
        if pct:
            vals = vals * 100.0
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=key, color=COLOR[key])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    if pct:
        ax.set_ylim(0, 108)
    return fig_uri(fig)


def scenario_bar(values: dict, ylabel, title, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = KEYS
    vals = [values[k] for k in KEYS]
    bars = ax.bar(xs, vals, 0.55, color=[COLOR[k] for k in KEYS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


def deep_bar(pkg: dict, conv: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    groups = ["PKG CO / MSU", "CONV CO / MSU"]
    x = np.arange(len(groups))
    n = len(KEYS)
    w = 0.8 / n
    for i, key in enumerate(KEYS):
        vals = [pkg[key], conv[key]]
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=key, color=COLOR[key])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("CO events per MSU")
    ax.set_title("Change-over per MSU (lower is better)")
    ax.legend(ncol=4, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


# --- HTML helpers ---------------------------------------------------------------
CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}
.wrap{max-width:1140px;margin:0 auto;padding:28px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:34px 0 10px;border-bottom:2px solid #4e79a7;padding-bottom:5px}
h3{font-size:15px;margin:18px 0 6px;color:#34495e}
.sub{color:#666;font-size:13px;margin-bottom:18px}
.card{background:#fff;border-radius:8px;padding:18px 22px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
table.kpi{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%}
table.kpi th{background:#34495e;color:#fff;padding:6px 9px;text-align:right}
table.kpi th:first-child{text-align:left}
table.kpi td{padding:5px 9px;text-align:right;border-bottom:1px solid #eee}
table.kpi td:first-child{text-align:left;font-weight:600}
img{max-width:100%;height:auto;margin:8px 0}
.kv{font-size:13px;color:#444} .kv b{color:#222}
.note{background:#fff8e1;border-left:4px solid #f0ad4e;padding:8px 12px;font-size:13px;margin:10px 0}
.pos{color:#2e7d32;font-weight:600} .neg{color:#c62828;font-weight:600} .flat{color:#888}
.legend span{display:inline-block;margin-right:14px;font-size:12px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:middle}
"""


def sect(title, body):
    return f'<div class="card"><h2>{title}</h2>{body}</div>'


def delta_span(delta, unit, good_up=True, decimals=1):
    if not np.isfinite(delta):
        return '<span class="flat">n/a</span>'
    if abs(delta) < (0.5 * 10 ** (-decimals)):
        return f'<span class="flat">\u00b10{unit}</span>'
    good = (delta > 0) == good_up
    cls = "pos" if good else "neg"
    sign = "+" if delta > 0 else "\u2212"
    return f'<span class="{cls}">{sign}{abs(delta):.{decimals}f}{unit}</span>'


def matrix_html(rows):
    """rows: list of (label, {key->value}, fmt, unit, good_up, decimals)."""
    head = "".join(f"<th>{k}</th>" for k in KEYS)
    head_d = "".join(f"<th>{k} \u0394</th>" for k in KEYS if k != BASE)
    h = [f'<table class="kpi"><thead><tr><th>KPI</th>{head}{head_d}</tr></thead><tbody>']
    for label, vals, fmt, unit, good_up, dec in rows:
        cells = "".join(f"<td>{fmt.format(vals[k])}</td>" for k in KEYS)
        dcells = ""
        for k in KEYS:
            if k == BASE:
                continue
            dcells += f"<td>{delta_span(vals[k] - vals[BASE], unit, good_up, dec)}</td>"
        h.append(f"<tr><td>{label}</td>{cells}{dcells}</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def wide_html(wide, fmt="{:.1f}", pct=False, scale=1.0):
    d = wide.copy().astype(float)
    if pct:
        d = d * 100.0
    else:
        d = d * scale
    disp = d.copy()
    for c in disp.columns:
        disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    disp.insert(0, "month", disp.index)
    return disp.to_html(index=False, border=0, classes="kpi", na_rep="")


def main():
    EXTRACTS_OUT.mkdir(parents=True, exist_ok=True)
    print("[1/4] loading per-scenario extracts ...")
    data, src = load_all()

    # wide tables
    tof = month_wide(data, "tof_m", "tof")
    dfc = month_wide(data, "dfc_m", "dfc_days")
    dfc_inv = month_wide(data, "dfc_m", "dc_inv")
    co_cnt = month_wide(data, "co_m", "co_count")
    co_time = month_wide(data, "co_m", "co_time")
    ptt_total = month_wide(data, "ptt", "total_hr")
    ptt_prod = month_wide(data, "ptt", "prod_hr")
    ptt_co = month_wide(data, "ptt", "co_hr")
    apq = scalar_row(data, "apq", "apq")
    runs = scalar_row(data, "apq", "runs")
    msu = scalar_row(data, "apq", "msu")
    pkg = scalar_row(data, "deep", "PKG_CO_per_MSU")
    conv = scalar_row(data, "deep", "CONV_CO_per_MSU")

    print("[2/4] rendering charts ...")
    charts = {
        "tof": grouped_bar(tof, "ToF (%)", "ToF by month \u2014 all scenarios", pct=True, fmt="{:.1f}"),
        "dfc": grouped_bar(dfc, "DFC (days)", "Month-end DFC by month", fmt="{:.0f}"),
        "co": grouped_bar(co_cnt, "Change-over count", "Change-over count by month", fmt="{:.0f}"),
        "ptt": grouped_bar(ptt_total, "hours", "Production total time by month", fmt="{:.0f}"),
        "apq": scenario_bar(apq, "APQ (MSU / run)", "APQ \u2014 MSU per production run (HPFD)", fmt="{:.3f}"),
        "deep": deep_bar(pkg, conv),
    }

    print("[3/4] writing extracts + workbook ...")
    # comparison extracts
    tof.to_csv(EXTRACTS_OUT / "cmp_tof_by_month.csv")
    dfc.to_csv(EXTRACTS_OUT / "cmp_dfc_by_month.csv")
    co_cnt.to_csv(EXTRACTS_OUT / "cmp_changeover_count_by_month.csv")
    co_time.to_csv(EXTRACTS_OUT / "cmp_changeover_time_by_month.csv")
    ptt_total.to_csv(EXTRACTS_OUT / "cmp_production_total_hr_by_month.csv")
    pd.DataFrame({"scenario": KEYS, "apq": [apq[k] for k in KEYS],
                  "runs": [runs[k] for k in KEYS], "msu": [msu[k] for k in KEYS],
                  "pkg_co_per_msu": [pkg[k] for k in KEYS],
                  "conv_co_per_msu": [conv[k] for k in KEYS]}).to_csv(
        EXTRACTS_OUT / "cmp_apq_deepdive_by_scenario.csv", index=False)

    xlsx = OUT_DIR / f"{PROJECT}_comparison_{RUN_TAG}.xlsx"
    meta = pd.DataFrame({"key": KEYS,
                         "scenario": [s[1] for s in SCENARIOS],
                         "run_id": [s[2] for s in SCENARIOS],
                         "lever": [s[3] for s in SCENARIOS],
                         "source_analysis": [src[k] for k in KEYS]})
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        meta.to_excel(xw, sheet_name="0_scenarios", index=False)
        tof.to_excel(xw, sheet_name="1_ToF_by_month")
        dfc.to_excel(xw, sheet_name="2_DFC_by_month")
        dfc_inv.to_excel(xw, sheet_name="2_DFC_inv_by_month")
        co_cnt.to_excel(xw, sheet_name="3_CO_count_by_month")
        co_time.to_excel(xw, sheet_name="3_CO_time_by_month")
        ptt_total.to_excel(xw, sheet_name="4_ProdTotalHr_by_month")
        ptt_prod.to_excel(xw, sheet_name="4_ProdHr_by_month")
        ptt_co.to_excel(xw, sheet_name="4_COHr_by_month")
        pd.DataFrame({"scenario": KEYS, "apq": [apq[k] for k in KEYS],
                      "runs": [runs[k] for k in KEYS], "msu": [msu[k] for k in KEYS]}
                     ).to_excel(xw, sheet_name="5_APQ_by_scenario", index=False)
        pd.DataFrame({"scenario": KEYS, "pkg_co_per_msu": [pkg[k] for k in KEYS],
                      "conv_co_per_msu": [conv[k] for k in KEYS]}
                     ).to_excel(xw, sheet_name="6_DeepDive_by_scenario", index=False)

    print("[4/4] writing comparison.html + run.md ...")
    html = build_html(data, src, tof, dfc, dfc_inv, co_cnt, co_time, ptt_total,
                      ptt_prod, ptt_co, apq, runs, msu, pkg, conv, charts)
    (OUT_DIR / "comparison.html").write_text(html, encoding="utf-8")
    write_run_md(src, tof, dfc, co_cnt, ptt_total, apq, pkg, conv, xlsx)
    print(f"\nDONE -> {OUT_DIR}")
    print("  report   : comparison.html")
    print(f"  workbook : {xlsx.name}")


def build_html(data, src, tof, dfc, dfc_inv, co_cnt, co_time, ptt_total,
               ptt_prod, ptt_co, apq, runs, msu, pkg, conv, charts):
    legend = '<div class="legend">' + "".join(
        f'<span><i style="background:{COLOR[k]}"></i>{k}</span>' for k in KEYS) + "</div>"

    # scenario / lever table
    lever_rows = "".join(
        f"<tr><td>{k}</td><td>{s[1]}</td><td style='text-align:left'>{s[3]}</td>"
        f"<td style='text-align:left'><code>{s[2]}</code></td></tr>"
        for k, s in zip(KEYS, SCENARIOS))
    lever_tbl = (f'<table class="kpi"><thead><tr><th>Scenario</th><th>config</th>'
                 f'<th>Lever vs baseline</th><th>run_id (DB {DBNAME})</th></tr></thead>'
                 f'<tbody>{lever_rows}</tbody></table>')

    # headline matrix
    headline = matrix_html([
        ("ToF Feb 2026 (%)", {k: tof.loc["Feb 2026", k] * 100 for k in KEYS}, "{:.1f}", "pp", True, 1),
        ("ToF Mar 2026 (%)", {k: tof.loc["Mar 2026", k] * 100 for k in KEYS}, "{:.1f}", "pp", True, 1),
        ("DFC Feb 2026 (days)", {k: dfc.loc["Feb 2026", k] for k in KEYS}, "{:.1f}", "d", True, 1),
        ("DFC Mar 2026 (days)", {k: dfc.loc["Mar 2026", k] for k in KEYS}, "{:.1f}", "d", True, 1),
        ("CO count Feb 2026", {k: co_cnt.loc["Feb 2026", k] for k in KEYS}, "{:.0f}", "", False, 0),
        ("CO count Mar 2026", {k: co_cnt.loc["Mar 2026", k] for k in KEYS}, "{:.0f}", "", False, 0),
        ("Prod total time Feb (hr)", {k: ptt_total.loc["Feb 2026", k] for k in KEYS}, "{:.1f}", "h", False, 1),
        ("Prod total time Mar (hr)", {k: ptt_total.loc["Mar 2026", k] for k in KEYS}, "{:.1f}", "h", False, 1),
        ("APQ (MSU/run)", apq, "{:.3f}", "", True, 3),
        ("PKG CO / MSU", pkg, "{:.4f}", "", False, 4),
        ("CONV CO / MSU", conv, "{:.4f}", "", False, 4),
    ])

    h = [f"""<!doctype html><html><head><meta charset="utf-8">
    <title>FEM HPFD \u2014 Baseline vs S1/S2/S3 KPI Comparison</title><style>{CSS}</style></head>
    <body><div class="wrap">
    <h1>FEM &times; HPFD &mdash; Baseline vs S1 / S2 / S3 KPI Comparison</h1>
    <div class="sub">Project <b>{PROJECT}</b> &nbsp;|&nbsp; DB <b>{DBNAME}</b> &nbsp;|&nbsp; line <b>{LINE}</b> @ plant 0386
    &nbsp;|&nbsp; months Feb &amp; Mar 2026 &nbsp;|&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}</div>"""]

    h.append(sect("Scenarios &amp; levers", f"""
    {lever_tbl}
    <div class="note">All four runs share the same scope (8 customer DCs: {', '.join(SCOPE_DCS)};
    HPFD line at plant 0386; Feb&ndash;Mar 2026) and the same KPI methodology. Each scenario changes
    exactly one lever vs the baseline. Deltas below are <b>scenario &minus; baseline</b>; green = favourable,
    red = unfavourable. KPI&nbsp;3 (change-over cost) is skipped per the baseline study.</div>"""))

    h.append(sect("Headline KPI matrix", f"""
    {legend}
    {headline}
    <div class="kv"><p>Delta colour convention: ToF &uarr; good; DFC &uarr; good (more coverage); change-over count &darr; good;
    production total time &darr; good; APQ &uarr; good; CO/MSU &darr; good. DFC deltas are directional only
    &mdash; very high coverage can also signal excess inventory.</p></div>"""))

    h.append(sect("1 &middot; ToF (service) by month", f"""
    {legend}<img src="{charts['tof']}">
    <h3>ToF % by month</h3>{wide_html(tof, fmt="{:.1f}", pct=True)}
    <div class="kv"><p>Feb is essentially flat across all scenarios (98.7&ndash;98.9%). The differentiator is March:
    S1 (longer review cycle) drops to {tof.loc['Mar 2026','S1']*100:.1f}% while S3 holds highest at
    {tof.loc['Mar 2026','S3']*100:.1f}%; S2 (lower capacity) stays close to baseline.</p></div>"""))

    h.append(sect("2 &middot; Month-end DFC (days forward coverage)", f"""
    {legend}<img src="{charts['dfc']}">
    <h3>DFC (days) by month</h3>{wide_html(dfc, fmt="{:.1f}")}
    <h3>Month-end DC inventory by month</h3>{wide_html(dfc_inv, fmt="{:,.0f}")}
    <div class="kv"><p>S1 front-loads inventory (Feb DFC {dfc.loc['Feb 2026','S1']:.1f}d vs baseline
    {dfc.loc['Feb 2026','Baseline']:.1f}d) then runs it down hard by end of March
    ({dfc.loc['Mar 2026','S1']:.1f}d) &mdash; consistent with the larger, less frequent replenishment batches.
    S2 and S3 track the baseline closely.</p></div>"""))

    h.append(sect("3 &middot; Change-over (count &amp; time, HPFD)", f"""
    {legend}<img src="{charts['co']}">
    <h3>Change-over count by month</h3>{wide_html(co_cnt, fmt="{:.0f}")}
    <h3>Change-over time (hr) by month</h3>{wide_html(co_time, fmt="{:.1f}")}
    <div class="kv"><p>Change-over activity is nearly identical across scenarios (Feb 19&ndash;20 events, Mar 14).
    None of the three levers materially changes the change-over burden on the line.</p></div>"""))

    h.append(sect("4 &middot; Production total time (production + change-over)", f"""
    {legend}<img src="{charts['ptt']}">
    <h3>Total line time (hr) by month</h3>{wide_html(ptt_total, fmt="{:.1f}")}
    <h3>Production hours (hr) by month</h3>{wide_html(ptt_prod, fmt="{:.1f}")}
    <div class="kv"><p>S1 shifts volume forward: Feb total line time rises to {ptt_total.loc['Feb 2026','S1']:.1f} hr
    (baseline {ptt_total.loc['Feb 2026','Baseline']:.1f} hr) and March falls to {ptt_total.loc['Mar 2026','S1']:.1f} hr.
    S2 and S3 are within ~1 hr of baseline both months.</p></div>"""))

    h.append(sect("5 &middot; APQ (avg MSU produced per run)", f"""
    <div class="grid2" style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start">
    <div><img src="{charts['apq']}"></div>
    <div><h3>APQ &amp; production runs</h3>
    <table class="kpi"><thead><tr><th>Scenario</th><th>APQ (MSU/run)</th><th>total MSU</th><th>runs</th></tr></thead>
    <tbody>{''.join(f"<tr><td>{k}</td><td>{apq[k]:.3f}</td><td>{msu[k]:,.1f}</td><td>{int(runs[k])}</td></tr>" for k in KEYS)}</tbody></table>
    </div></div>
    <div class="kv"><p>APQ is stable (5.77&ndash;5.84 MSU/run). S1 is marginally highest from its larger campaigns;
    S2 is marginally lowest. Differences are small relative to the base-product campaign effect noted in the
    per-scenario reports.</p></div>"""))

    h.append(sect("6 &middot; Change-over deep dive (PKG / CONV per MSU)", f"""
    {legend}<img src="{charts['deep']}">
    <table class="kpi"><thead><tr><th>Scenario</th><th>PKG CO / MSU</th><th>CONV CO / MSU</th></tr></thead>
    <tbody>{''.join(f"<tr><td>{k}</td><td>{pkg[k]:.4f}</td><td>{conv[k]:.4f}</td></tr>" for k in KEYS)}</tbody></table>
    <div class="kv"><p>Change-over intensity per unit volume is flat across scenarios (PKG ~0.152, CONV ~0.044 CO/MSU),
    confirming none of the levers trades off change-over efficiency. Lower is better.</p></div>"""))

    h.append(f"""<div class="card"><h2>Source</h2><div class="kv">
    <p>Built from each scenario's latest per-KPI extracts (no re-query); methodology identical to the four
    individual reports:</p>
    <table class="kpi"><thead><tr><th>Scenario</th><th>source analysis run</th></tr></thead><tbody>
    {''.join(f"<tr><td>{k}</td><td style='text-align:left'>{src[k]}</td></tr>" for k in KEYS)}
    </tbody></table></div></div>""")

    h.append("</div></body></html>")
    return "".join(h)


def write_run_md(src, tof, dfc, co_cnt, ptt_total, apq, pkg, conv, xlsx):
    lines = [
        f"# FEM KPI comparison - baseline vs S1/S2/S3 - run {RUN_TAG}",
        "",
        "- level: cross-scenario comparison",
        f"- project: {PROJECT}",
        f"- database: {DBNAME}",
        f"- line: {LINE} @ plant 0386 | months: Feb 2026, Mar 2026",
        f"- generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Scenarios",
    ]
    for k, s in zip(KEYS, SCENARIOS):
        lines.append(f"- {k}: `{s[1]}` | run_id `{s[2]}` | lever: {s[3]}")
    lines += ["", "## Headline (Baseline / S1 / S2 / S3)"]
    fmt4 = lambda d, key: " / ".join(f"{d.loc[key, k]:.1f}" for k in KEYS)
    lines += [
        f"- ToF Feb %: " + " / ".join(f"{tof.loc['Feb 2026',k]*100:.1f}" for k in KEYS),
        f"- ToF Mar %: " + " / ".join(f"{tof.loc['Mar 2026',k]*100:.1f}" for k in KEYS),
        f"- DFC Feb (days): " + fmt4(dfc, "Feb 2026"),
        f"- DFC Mar (days): " + fmt4(dfc, "Mar 2026"),
        f"- CO count Feb: " + " / ".join(f"{int(co_cnt.loc['Feb 2026',k])}" for k in KEYS),
        f"- CO count Mar: " + " / ".join(f"{int(co_cnt.loc['Mar 2026',k])}" for k in KEYS),
        f"- Prod total time Feb (hr): " + fmt4(ptt_total, "Feb 2026"),
        f"- Prod total time Mar (hr): " + fmt4(ptt_total, "Mar 2026"),
        f"- APQ (MSU/run): " + " / ".join(f"{apq[k]:.3f}" for k in KEYS),
        f"- PKG CO/MSU: " + " / ".join(f"{pkg[k]:.4f}" for k in KEYS),
        f"- CONV CO/MSU: " + " / ".join(f"{conv[k]:.4f}" for k in KEYS),
        "",
        "## Reading",
        "- S1 (longer review cycle): front-loads inventory in Feb, runs it down by end-Mar; March service "
        "is the lowest of the four; Feb production time rises.",
        "- S2 (capacity 24->16 h/day): tracks the baseline on every KPI -> capacity is not the binding "
        "constraint in this network/period.",
        "- S3 (AO standardized): best March service, all other KPIs close to baseline.",
        "- Change-over count/time, APQ and CO/MSU are effectively unchanged across all scenarios.",
        "",
        "## Source",
    ]
    for k in KEYS:
        lines.append(f"- {k}: {src[k]}")
    lines += [
        "",
        "## Outputs",
        f"- report: comparison.html",
        f"- workbook: {xlsx.name}",
        "- extracts/: per-KPI comparison CSVs",
    ]
    (OUT_DIR / "run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
