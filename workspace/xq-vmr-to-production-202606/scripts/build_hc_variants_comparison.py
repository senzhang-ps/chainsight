#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark comparison: baseline-hc-s1 and baseline-hc-s2 vs baseline-hc.

The three scenarios of project ``xq-vmr-to-production-202606`` are identical in
every input except the M1 advance-order (AO) configuration:

  * baseline-hc     : empirical per-key AO distribution (the benchmark).
  * baseline-hc-s1  : one standardized profile for all keys, moderate advancing
                      (advance_days 0=0.45, 3=0.05, 8=0.25, 12=0.25).
  * baseline-hc-s2  : one standardized profile, aggressive advancing
                      (advance_days 0=0.15, 3=0.05, 8=0.40, 12=0.40).

This script reuses the vetted single-scenario analysis engine
``build_baseline_hc_analysis`` (imported as ``bh``) to recompute every KPI for
each run_id, then renders a side-by-side comparison (baseline-hc as the
benchmark column, s1 and s2 each with a delta column) as an Excel workbook, a
self-contained HTML report, CSV extracts and a run.md, written at the
*project* level (workspace/<project>/analysis/<run-id>/).

The category and su_factor maps are resolved ONCE over the union of materials
across the three runs (material attributes are scenario-independent), so
Databricks is hit only once.

Read-only against PostgreSQL; safe to re-run.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import build_baseline_hc_analysis as bh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- scenarios ----------------------------------------------------------------
# (label, run_id, config_name)   baseline-hc MUST be first (it is the benchmark)
SCENARIOS = [
    ("baseline-hc",    "db_baseline-hc_20260617_203548",    "baseline-hc"),
    ("baseline-hc-s1", "db_baseline-hc-s1_20260706_103801", "baseline-hc-s1"),
    ("baseline-hc-s2", "db_baseline-hc-s2_20260706_110742", "baseline-hc-s2"),
]
BENCH = "baseline-hc"
ORDER = [s[0] for s in SCENARIOS]
VARIANTS = [s for s in ORDER if s != BENCH]

SCEN_COLOR = {"baseline-hc": "#4b5563", "baseline-hc-s1": "#2563eb",
              "baseline-hc-s2": "#ea580c"}

# one-line description of the AO lever for each scenario (shown in the report)
SCEN_DESC = {
    "baseline-hc":    "Empirical per-key AO distribution (benchmark)",
    "baseline-hc-s1": "Standardized AO, moderate advancing: 0d=45%, 3d=5%, 8d=25%, 12d=25%",
    "baseline-hc-s2": "Standardized AO, aggressive advancing: 0d=15%, 3d=5%, 8d=40%, 12d=40%",
}


# =============================================================================
# per-scenario KPI computation (reuses bh compute_* on a shared cat / su map)
# =============================================================================
def compute_scenario_from_raw(t, cat, su_factor, su_source):
    """Recompute every KPI for one scenario's raw pull tuple.

    ``t`` is the 14-tuple returned by ``bh.pull_postgres()``. The month-keyed
    frames are capped to ``bh.MONTH_MAX`` exactly as ``bh.main`` does before any
    KPI is computed, so the reporting horizon matches the single-scenario runs.
    """
    (orders, ships, line_map, changeover, co_def, stock_me, fcst_weekly, dc_info,
     prod_plan, prd_rate, prod_runs, global_network, fcst_all, moq) = t

    mm = bh.MONTH_MAX
    orders = orders[orders["month"] <= mm].reset_index(drop=True)
    ships = ships[ships["month"] <= mm].reset_index(drop=True)
    changeover = changeover[changeover["month"] <= mm].reset_index(drop=True)
    stock_me = stock_me[stock_me["month"] <= mm].reset_index(drop=True)
    prod_plan = prod_plan[prod_plan["month"] <= mm].reset_index(drop=True)

    base, cfr_main, cfr_by_line, cfr_by_cat, cfr_by_month, cfr_line_total = \
        bh.compute_cfr(orders, ships, line_map, cat)
    dfc_base, dfc_main, dfc_by_cat, dfc_by_line, dfc_by_month, dfc_info = \
        bh.compute_dfc(stock_me, fcst_weekly, base)
    (ptt_base, ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month,
     ptt_line_total, ptt_cat_total, ptt_info) = \
        bh.compute_ptt(prod_plan, prd_rate, changeover, base)
    apq_by_line, apq_by_material, apq_info = \
        bh.compute_apq(prod_runs, base, su_factor, su_source)
    msu_by_line, msu_material, msu_info = \
        bh.compute_msu(prod_runs, changeover, su_factor, su_source)
    cov_by_sku, cov_by_line, cov_by_cat, cov_info = \
        bh.compute_coverage(global_network, fcst_all, moq, su_factor, su_source,
                            apq_by_material, base, line_map)

    # ---- headline scalars ----------------------------------------------------
    tot_order = float(base["order_qty"].sum())
    tot_ship = float(base["shipment_qty"].sum())
    tof_overall = tot_ship / tot_order if tot_order else float("nan")

    full = dfc_by_month[~dfc_by_month["partial_window"]]
    dfc_full = (float(full["dc_inv"].sum()) /
                float(full["avg_daily_demand"].sum())
                if full["avg_daily_demand"].sum() else float("nan"))

    co_count = int(changeover["changeover_count"].sum())
    co_cost = float(changeover["changeover_cost"].sum())

    headline = {
        "tof_overall": tof_overall,
        "dfc_full_month": dfc_full,
        "co_count": co_count,
        "co_cost": co_cost,
        "prod_hr_total": ptt_info["prod_hr_total"],
        "co_hr_total": ptt_info["co_hr_total"],
        "total_hr_total": ptt_info["total_hr_total"],
        "co_share_total": ptt_info["co_share_total"],
        "apq_overall": apq_info["apq_overall"],
        "total_runs": apq_info["total_runs"],
        "total_msu": apq_info["total_msu"],
        "wash_count": msu_info["total_wash"],
        "wash_per_msu": msu_info["wash_per_msu_overall"],
        "moq_cov_overall": cov_info["moq_coverage_overall"],
        "apq_cov_overall": cov_info["apq_coverage_overall"],
    }

    # ---- by-dimension frames (already the right grain) -----------------------
    tof_month = base.groupby("month", as_index=False).agg(
        order_qty=("order_qty", "sum"), shipment_qty=("shipment_qty", "sum"))
    tof_month["tof"] = tof_month["shipment_qty"] / tof_month["order_qty"].replace(0, np.nan)

    tof_line = base.groupby("line", as_index=False).agg(
        order_qty=("order_qty", "sum"), shipment_qty=("shipment_qty", "sum"))
    tof_line["tof"] = tof_line["shipment_qty"] / tof_line["order_qty"].replace(0, np.nan)

    tof_cat = base.groupby("category", as_index=False).agg(
        order_qty=("order_qty", "sum"), shipment_qty=("shipment_qty", "sum"))
    tof_cat["tof"] = tof_cat["shipment_qty"] / tof_cat["order_qty"].replace(0, np.nan)

    co_line = changeover.groupby("line", as_index=False).agg(
        co_count=("changeover_count", "sum"), co_cost=("changeover_cost", "sum"))

    return {
        "headline": headline,
        "tof_month": tof_month[["month", "tof"]],
        "tof_line": tof_line[["line", "tof"]],
        "tof_cat": tof_cat[["category", "tof"]],
        "dfc_month": dfc_by_month[["month", "dfc_days", "partial_window"]].copy(),
        "co_line": co_line,
        "prod_line": ptt_line_total[["line", "total_hr", "prod_hr", "co_hr"]].copy(),
        "apq_line": apq_by_line[["line", "apq", "runs", "msu"]].copy(),
        "cov_line": cov_by_line[["line", "moq_coverage", "apq_coverage", "n_sku"]].copy(),
        "msu_line": msu_by_line[["line", "wash_count", "wash_per_msu", "msu"]].copy(),
    }


# =============================================================================
# comparison-table builder
# =============================================================================
def compare(results, key, val, order=ORDER, bench=BENCH):
    """Merge one per-scenario metric into a wide table + delta-vs-bench columns.

    ``val`` is a ``(frame_name, column)`` pair identifying the metric inside each
    scenario's result dict.
    """
    frame_name, col = val
    out = None
    for scen in order:
        d = results[scen][frame_name][[key, col]].rename(columns={col: scen})
        out = d if out is None else out.merge(d, on=key, how="outer")
    for scen in order:
        if scen != bench:
            out[f"Δ {scen}"] = out[scen] - out[bench]
    return out


# =============================================================================
# rendering helpers
# =============================================================================
def fmt_pct(v):
    return "—" if pd.isna(v) else f"{v * 100:.1f}%"


def fmt_num(v, d=1):
    return "—" if pd.isna(v) else f"{v:,.{d}f}"


def fmt_int(v):
    return "—" if pd.isna(v) else f"{int(round(v)):,}"


def grouped_bar(piv, title, ylabel, ylim=None, rot=0, pct=False):
    cols = [c for c in ORDER if c in piv.columns]
    piv = piv[cols]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    piv.plot(kind="bar", ax=ax, width=0.8,
             color=[SCEN_COLOR[c] for c in piv.columns])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(title="scenario", fontsize=8)
    ax.tick_params(axis="x", rotation=rot)
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    uri = bh.fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def render_compare(df, key, valfmt, kind, better, key_label=None):
    """Render a comparison df to an HTML table with coloured delta cells.

    ``kind`` controls delta formatting: 'pp' (percentage points), 'abs'
    (absolute) or 'abspct' (absolute + % change). ``better`` in {'high','low',
    None} colours a delta green when it moves the KPI in the good direction.
    """
    key_label = key_label or key
    cols = ORDER + [f"Δ {v}" for v in VARIANTS]
    th = "".join(f"<th>{c}</th>" for c in cols)
    head = f"<tr><th>{key_label}</th>{th}</tr>"
    rows = []
    for _, r in df.iterrows():
        cells = [f"<td class='k'>{r[key]}</td>"]
        for scen in ORDER:
            cells.append(f"<td>{valfmt(r[scen])}</td>")
        for v in VARIANTS:
            d = r[f"Δ {v}"]
            if pd.isna(d):
                cells.append("<td>—</td>")
                continue
            if kind == "pp":
                txt = f"{d * 100:+.1f} pp"
            elif kind == "abspct":
                bv = r[BENCH]
                pct = (d / bv * 100) if (bv not in (0, None) and not pd.isna(bv) and bv != 0) else float("nan")
                txt = f"{d:+,.1f}" + ("" if pd.isna(pct) else f" ({pct:+.1f}%)")
            else:  # abs
                txt = f"{d:+,.2f}"
            cls = ""
            if better == "high":
                cls = "up" if d > 0 else ("down" if d < 0 else "")
            elif better == "low":
                cls = "up" if d < 0 else ("down" if d > 0 else "")
            cells.append(f"<td class='{cls}'>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='cmp'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def render_summary(results, orderlog_rows):
    """Headline KPI table: metrics as rows, scenarios as columns + deltas."""
    METRICS = [
        # (label, key, fmt, kind, better)
        ("Order fill rate (TOF), overall", "tof_overall", fmt_pct, "pp", "high"),
        ("DFC, full-month avg (days)", "dfc_full_month", lambda v: fmt_num(v, 1), "abs", None),
        ("Changeover count (total)", "co_count", fmt_int, "abspct", "low"),
        ("Changeover cost (total)", "co_cost", fmt_int, "abspct", "low"),
        ("Production time (hr)", "prod_hr_total", lambda v: fmt_num(v, 0), "abspct", None),
        ("Changeover time (hr)", "co_hr_total", lambda v: fmt_num(v, 0), "abspct", "low"),
        ("Total time (hr)", "total_hr_total", lambda v: fmt_num(v, 0), "abspct", "low"),
        ("Changeover time share (%)", "co_share_total", lambda v: fmt_num(v, 1) + "%", "abs", "low"),
        ("APQ (MSU / run), overall", "apq_overall", lambda v: fmt_num(v, 3), "abs", "high"),
        ("Production runs (total)", "total_runs", fmt_int, "abspct", "low"),
        ("Production volume (MSU)", "total_msu", lambda v: fmt_num(v, 0), "abspct", None),
        ("Wash changeovers (count)", "wash_count", fmt_int, "abspct", "low"),
        ("Wash per MSU", "wash_per_msu", lambda v: fmt_num(v, 3), "abs", "low"),
        ("MOQ coverage, overall (days)", "moq_cov_overall", lambda v: fmt_num(v, 1), "abs", None),
        ("APQ coverage, overall (days)", "apq_cov_overall", lambda v: fmt_num(v, 1), "abs", None),
        ("Order-log rows (AO fingerprint)", "_orderlog", fmt_int, "abspct", None),
    ]
    th = "".join(f"<th>{c}</th>" for c in ORDER + [f"Δ {v}" for v in VARIANTS])
    head = f"<tr><th>KPI</th>{th}</tr>"
    rows = []
    for label, key, ff, kind, better in METRICS:
        vals = {}
        for scen in ORDER:
            vals[scen] = (orderlog_rows[scen] if key == "_orderlog"
                          else results[scen]["headline"][key])
        cells = [f"<td class='k'>{label}</td>"]
        for scen in ORDER:
            cells.append(f"<td>{ff(vals[scen])}</td>")
        for v in VARIANTS:
            d = vals[v] - vals[BENCH]
            if pd.isna(d):
                cells.append("<td>—</td>"); continue
            if kind == "pp":
                txt = f"{d * 100:+.1f} pp"
            elif kind == "abspct":
                bv = vals[BENCH]
                pct = (d / bv * 100) if bv else float("nan")
                txt = f"{d:+,.1f}" + ("" if pd.isna(pct) else f" ({pct:+.1f}%)")
            else:
                txt = f"{d:+,.3f}" if abs(d) < 10 else f"{d:+,.1f}"
            cls = ""
            if better == "high":
                cls = "up" if d > 0 else ("down" if d < 0 else "")
            elif better == "low":
                cls = "up" if d < 0 else ("down" if d > 0 else "")
            cells.append(f"<td class='{cls}'>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='cmp summary'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


# =============================================================================
# orderlog row counts (structural fingerprint of the AO lever)
# =============================================================================
def orderlog_rowcounts():
    counts = {}
    with bh.pg_connect() as conn:
        for label, rid, _ in SCENARIOS:
            d = bh.q(conn, "SELECT count(*) AS n FROM module1_output_orderlog "
                           "WHERE run_id = %s", (rid,))
            counts[label] = int(d["n"].iloc[0])
    return counts


# =============================================================================
# main
# =============================================================================
def main():
    print("=" * 74)
    print("HC AO-variant comparison  |  s1, s2  vs  baseline-hc")
    print("=" * 74)

    # ---- 1. pull raw for all three scenarios --------------------------------
    raw = {}
    for label, rid, cfg in SCENARIOS:
        print(f"[pull] {label:<16} run_id={rid}")
        bh.RUN_ID = rid
        bh.CONFIG_NAME = cfg
        raw[label] = bh.pull_postgres()

    # ---- 2. resolve category + su_factor ONCE over the material union -------
    all_mats, all_prod = set(), set()
    for label in ORDER:
        t = raw[label]
        orders, ships = t[0], t[1]
        prod_runs, moq = t[10], t[13]
        all_mats |= set(orders["material"].astype(str))
        all_mats |= set(ships["material"].astype(str))
        all_prod |= set(prod_runs["material"].astype(str))
        all_prod |= set(moq["material"].astype(str))
    print(f"[map ] category over {len(all_mats)} materials, "
          f"su_factor over {len(all_prod)} materials (Databricks once)")
    cat, cat_source, cat_note = bh.get_category_map(sorted(all_mats))
    su_factor, su_source, su_info = bh.get_su_factor_map(sorted(all_prod))

    # ---- 3. compute every KPI per scenario ----------------------------------
    results = {}
    for label in ORDER:
        print(f"[calc] {label}")
        results[label] = compute_scenario_from_raw(raw[label], cat, su_factor, su_source)

    orderlog_rows = orderlog_rowcounts()

    # ---- 4. build comparison tables -----------------------------------------
    cmp = {}
    cmp["tof_month"] = compare(results, "month", ("tof_month", "tof"))
    cmp["tof_line"] = compare(results, "line", ("tof_line", "tof"))
    cmp["tof_cat"] = compare(results, "category", ("tof_cat", "tof"))
    cmp["dfc_month"] = compare(results, "month", ("dfc_month", "dfc_days"))
    cmp["co_line"] = compare(results, "line", ("co_line", "co_count"))
    cmp["cocost_line"] = compare(results, "line", ("co_line", "co_cost"))
    cmp["prod_line"] = compare(results, "line", ("prod_line", "total_hr"))
    cmp["apq_line"] = compare(results, "line", ("apq_line", "apq"))
    cmp["cov_line"] = compare(results, "line", ("cov_line", "moq_coverage"))
    cmp["wash_line"] = compare(results, "line", ("msu_line", "wash_count"))

    # headline scalar frame (for workbook Summary sheet)
    hkeys = list(results[BENCH]["headline"].keys())
    summ = pd.DataFrame({"metric": hkeys})
    for scen in ORDER:
        summ[scen] = [results[scen]["headline"][k] for k in hkeys]
    summ.loc[len(summ)] = ["orderlog_rows"] + [orderlog_rows[s] for s in ORDER]

    # ---- 5. output location (project level) ---------------------------------
    run_tag = datetime.now().strftime("%Y%m%d-%H%M") + "-hc-s1-s2-vs-baseline-hc"
    out = bh.ROOT / "workspace" / bh.PROJECT / "analysis" / run_tag
    (out / "extracts").mkdir(parents=True, exist_ok=True)

    meta = {
        "project": bh.PROJECT,
        "scope": "HC only, plant XQ (1864)",
        "benchmark": BENCH,
        "variants": VARIANTS,
        "run_tag": run_tag,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "month_max": bh.MONTH_MAX,
        "sim_start": str(bh.SIM_START.date()),
        "runs": {lbl: rid for lbl, rid, _ in SCENARIOS},
        "cat_note": cat_note,
        "su_info": su_info,
    }

    # ---- 6. extracts --------------------------------------------------------
    summ.to_csv(out / "extracts" / "headline_summary.csv", index=False,
                encoding="utf-8-sig")
    for name, d in cmp.items():
        d.to_csv(out / "extracts" / f"cmp_{name}.csv", index=False,
                 encoding="utf-8-sig")

    # ---- 7. charts ----------------------------------------------------------
    charts = {}

    def piv_of(cmpname, key):
        return cmp[cmpname].set_index(key)[ORDER]

    charts["tof_month"] = grouped_bar(piv_of("tof_month", "month"),
                                      "TOF by month", "TOF", ylim=(0, 1.02), pct=True)
    charts["tof_line"] = grouped_bar(piv_of("tof_line", "line"),
                                     "TOF by production line (full period)", "TOF",
                                     ylim=(0, 1.02), pct=True, rot=30)
    charts["dfc_month"] = grouped_bar(piv_of("dfc_month", "month"),
                                      "Month-end DFC by month", "days")
    charts["co_line"] = grouped_bar(piv_of("co_line", "line"),
                                    "Changeover count by line", "count", rot=30)
    charts["prod_line"] = grouped_bar(piv_of("prod_line", "line"),
                                      "Production total time by line", "hours", rot=30)
    charts["apq_line"] = grouped_bar(piv_of("apq_line", "line"),
                                     "APQ (MSU per run) by line", "MSU/run", rot=30)
    charts["cov_line"] = grouped_bar(piv_of("cov_line", "line"),
                                     "MOQ coverage by line", "days", rot=30)

    # headline TOF / DFC / APQ single-value bars
    head_tof = pd.DataFrame({s: [results[s]["headline"]["tof_overall"]] for s in ORDER},
                            index=["overall"])
    charts["tof_overall"] = grouped_bar(head_tof, "Overall TOF", "TOF",
                                        ylim=(0, 1.02), pct=True)

    # ---- 8. workbook --------------------------------------------------------
    xlsx = out / f"{bh.PROJECT}_comparison_{run_tag}.xlsx"
    write_workbook(xlsx, meta, summ, cmp)

    # ---- 9. html ------------------------------------------------------------
    html = out / "analysis.html"
    write_html(html, meta, charts, results, cmp, orderlog_rows)

    # ---- 10. run.md + LATEST ------------------------------------------------
    write_run_md(out, meta, results, orderlog_rows)
    (bh.ROOT / "workspace" / bh.PROJECT / "analysis" / "LATEST.md").write_text(
        f"# Latest project-level analysis\n\n"
        f"- **{run_tag}** — HC AO-variant comparison (s1, s2 vs baseline-hc)\n"
        f"  - [analysis.html](./{run_tag}/analysis.html)\n"
        f"  - [workbook]({xlsx.name})\n"
        f"  - generated {meta['generated']}\n", encoding="utf-8")

    print("-" * 74)
    print(f"[done] {out}")
    print(f"       workbook : {xlsx.name}")
    print(f"       html     : analysis.html")
    return out


# =============================================================================
# workbook writer (self-contained; does NOT touch report_helpers)
# =============================================================================
def write_workbook(xlsx, meta, summ, cmp):
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        # Notes
        notes = pd.DataFrame({
            "field": ["project", "scope", "benchmark", "variants", "month_max",
                      "sim_start", "generated", "baseline-hc run_id",
                      "baseline-hc-s1 run_id", "baseline-hc-s2 run_id",
                      "category source", "su_factor source"],
            "value": [meta["project"], meta["scope"], meta["benchmark"],
                      ", ".join(meta["variants"]), meta["month_max"],
                      meta["sim_start"], meta["generated"],
                      meta["runs"]["baseline-hc"], meta["runs"]["baseline-hc-s1"],
                      meta["runs"]["baseline-hc-s2"], str(meta["cat_note"]),
                      str(meta["su_info"])],
        })
        notes.to_excel(xw, sheet_name="Notes", index=False)
        summ.to_excel(xw, sheet_name="Summary", index=False)
        sheet_names = {
            "tof_month": "TOF by month", "tof_line": "TOF by line",
            "tof_cat": "TOF by category", "dfc_month": "DFC by month",
            "co_line": "Changeover count by line",
            "cocost_line": "Changeover cost by line",
            "prod_line": "Production time by line", "apq_line": "APQ by line",
            "cov_line": "MOQ coverage by line", "wash_line": "Wash count by line",
        }
        for name, d in cmp.items():
            d.to_excel(xw, sheet_name=sheet_names.get(name, name)[:31], index=False)


# =============================================================================
# HTML writer (self-contained comparison report)
# =============================================================================
def write_html(html_path, meta, charts, results, cmp, orderlog_rows):
    def img(key):
        return f"<img src='{charts[key]}' alt='{key}'/>" if key in charts else ""

    scen_rows = "".join(
        f"<tr><td class='k'>{s}</td><td>{meta['runs'][s]}</td>"
        f"<td>{SCEN_DESC[s]}</td></tr>" for s in ORDER)

    summary_tbl = render_summary(results, orderlog_rows)

    tof_month_tbl = render_compare(cmp["tof_month"], "month", fmt_pct, "pp", "high")
    tof_line_tbl = render_compare(cmp["tof_line"], "line", fmt_pct, "pp", "high")
    tof_cat_tbl = render_compare(cmp["tof_cat"], "category", fmt_pct, "pp", "high")
    dfc_tbl = render_compare(cmp["dfc_month"], "month",
                             lambda v: fmt_num(v, 1), "abs", None)
    co_tbl = render_compare(cmp["co_line"], "line", fmt_int, "abspct", "low")
    prod_tbl = render_compare(cmp["prod_line"], "line",
                              lambda v: fmt_num(v, 0), "abspct", "low")
    apq_tbl = render_compare(cmp["apq_line"], "line",
                             lambda v: fmt_num(v, 3), "abs", "high")
    cov_tbl = render_compare(cmp["cov_line"], "line",
                             lambda v: fmt_num(v, 1), "abs", None)

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      color:#1f2937;max-width:1180px;margin:0 auto;padding:24px 28px;line-height:1.5;}
    h1{font-size:24px;margin:0 0 4px;} h2{font-size:19px;margin:34px 0 8px;
      border-bottom:2px solid #e5e7eb;padding-bottom:4px;}
    h3{font-size:15px;margin:20px 0 6px;color:#374151;}
    .sub{color:#6b7280;font-size:13px;margin:0 0 18px;}
    table{border-collapse:collapse;font-size:13px;margin:10px 0;width:100%;}
    th,td{border:1px solid #e5e7eb;padding:5px 9px;text-align:right;}
    th{background:#f9fafb;font-weight:600;text-align:right;}
    td.k,th:first-child{text-align:left;}
    table.cmp td.up{color:#15803d;font-weight:600;}
    table.cmp td.down{color:#b91c1c;font-weight:600;}
    table.summary td.k{font-weight:600;}
    img{max-width:100%;border:1px solid #eee;border-radius:6px;margin:8px 0;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    .card{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;}
    .note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
      padding:10px 14px;font-size:13px;margin:12px 0;}
    ul{margin:6px 0 6px 4px;} li{margin:3px 0;}
    code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12px;}
    """

    # ---- narrative: derive a couple of headline deltas for the exec summary --
    def hd(scen, k):
        return results[scen]["headline"][k]

    def dpp(scen, k):
        return (hd(scen, k) - hd(BENCH, k)) * 100

    exec_pts = []
    for v in VARIANTS:
        tof_d = dpp(v, "tof_overall")
        co_d = hd(v, "co_count") - hd(BENCH, "co_count")
        co_pct = co_d / hd(BENCH, "co_count") * 100 if hd(BENCH, "co_count") else 0
        apq_d = hd(v, "apq_overall") - hd(BENCH, "apq_overall")
        runs_d = hd(v, "total_runs") - hd(BENCH, "total_runs")
        exec_pts.append(
            f"<li><b>{v}</b> — overall TOF {tof_d:+.1f} pp, changeovers "
            f"{co_d:+,.0f} ({co_pct:+.1f}%), APQ {apq_d:+.3f} MSU/run, "
            f"production runs {runs_d:+,.0f} vs baseline-hc.</li>")
    exec_html = "<ul>" + "".join(exec_pts) + "</ul>"

    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HC AO-variant comparison — {meta['project']}</title><style>{css}</style></head>
<body>
<h1>HC advance-order variant comparison</h1>
<p class='sub'>Project <b>{meta['project']}</b> &middot; {meta['scope']} &middot;
benchmark <b>{BENCH}</b> &middot; horizon &le; {meta['month_max']} &middot;
generated {meta['generated']}</p>

<div class='note'><b>What differs between the runs:</b> only the M1 advance-order
(AO) configuration. Every other input (demand, network, capacity, MOQ, su_factor,
category map) is identical, so all KPI differences below are attributable to how
much customer demand is <i>advanced</i> (ordered earlier than its due date).</div>

<h2>1&nbsp;&middot;&nbsp;Scenarios</h2>
<table><thead><tr><th>scenario</th><th>run_id</th><th>AO configuration</th></tr></thead>
<tbody>{scen_rows}</tbody></table>

<h2>2&nbsp;&middot;&nbsp;Executive summary</h2>
{exec_html}

<h2>3&nbsp;&middot;&nbsp;Headline KPI comparison</h2>
<p class='sub'>Green = moves the KPI in the favourable direction vs baseline-hc;
red = unfavourable. Δ columns are variant − baseline-hc.</p>
{summary_tbl}

<h2>4&nbsp;&middot;&nbsp;Order fill rate (TOF)</h2>
<div class='grid'><div>{img('tof_overall')}</div><div>{img('tof_month')}</div></div>
<h3>TOF by month</h3>{tof_month_tbl}
<h3>TOF by production line (full period)</h3>{img('tof_line')}{tof_line_tbl}
<h3>TOF by category</h3>{tof_cat_tbl}

<h2>5&nbsp;&middot;&nbsp;Month-end DFC (days forward coverage)</h2>
{img('dfc_month')}
{dfc_tbl}

<h2>6&nbsp;&middot;&nbsp;Changeovers</h2>
{img('co_line')}
<h3>Changeover count by line</h3>{co_tbl}

<h2>7&nbsp;&middot;&nbsp;Production total time</h2>
{img('prod_line')}
<h3>Production total time by line (hours)</h3>{prod_tbl}

<h2>8&nbsp;&middot;&nbsp;APQ — average MSU per production run</h2>
{img('apq_line')}
<h3>APQ by line</h3>{apq_tbl}

<h2>9&nbsp;&middot;&nbsp;MOQ coverage</h2>
{img('cov_line')}
<h3>MOQ coverage by line (days)</h3>{cov_tbl}

<h2>10&nbsp;&middot;&nbsp;Notes</h2>
<ul>
<li>All KPIs recomputed with the vetted single-scenario engine
(<code>build_baseline_hc_analysis</code>); category and su_factor resolved once
over the material union across the three runs.</li>
<li>Order-log row counts (in the headline table) are the structural fingerprint
of the AO lever: standardizing the AO profile spreads every key across the four
advance-day buckets, so s1/s2 carry more advance-order rows than the sparse
empirical baseline.</li>
<li>DFC is a month-end snapshot; the two partial-window months at the horizon
edge are shown for completeness but excluded from the full-month average.</li>
</ul>
</body></html>"""
    html_path.write_text(doc, encoding="utf-8")


# =============================================================================
# run.md
# =============================================================================
def write_run_md(out, meta, results, orderlog_rows):
    def hd(s, k):
        return results[s]["headline"][k]

    lines = [
        f"# HC AO-variant comparison — {meta['project']}",
        "",
        f"- generated: {meta['generated']}",
        f"- benchmark: `{BENCH}`",
        f"- variants: {', '.join('`'+v+'`' for v in VARIANTS)}",
        f"- scope: {meta['scope']}, horizon ≤ {meta['month_max']}",
        "",
        "## Runs",
        "",
        "| scenario | run_id | AO configuration |",
        "|---|---|---|",
    ]
    for s in ORDER:
        lines.append(f"| {s} | `{meta['runs'][s]}` | {SCEN_DESC[s]} |")
    lines += [
        "",
        "## Headline deltas vs baseline-hc",
        "",
        "| KPI | baseline-hc | s1 | s2 |",
        "|---|---|---|---|",
        f"| Overall TOF | {hd(BENCH,'tof_overall')*100:.1f}% | "
        f"{hd('baseline-hc-s1','tof_overall')*100:.1f}% | "
        f"{hd('baseline-hc-s2','tof_overall')*100:.1f}% |",
        f"| DFC full-month avg (days) | {hd(BENCH,'dfc_full_month'):.1f} | "
        f"{hd('baseline-hc-s1','dfc_full_month'):.1f} | "
        f"{hd('baseline-hc-s2','dfc_full_month'):.1f} |",
        f"| Changeover count | {hd(BENCH,'co_count'):,} | "
        f"{hd('baseline-hc-s1','co_count'):,} | {hd('baseline-hc-s2','co_count'):,} |",
        f"| Production time (hr) | {hd(BENCH,'prod_hr_total'):,.0f} | "
        f"{hd('baseline-hc-s1','prod_hr_total'):,.0f} | "
        f"{hd('baseline-hc-s2','prod_hr_total'):,.0f} |",
        f"| APQ (MSU/run) | {hd(BENCH,'apq_overall'):.3f} | "
        f"{hd('baseline-hc-s1','apq_overall'):.3f} | "
        f"{hd('baseline-hc-s2','apq_overall'):.3f} |",
        f"| Production runs | {hd(BENCH,'total_runs'):,} | "
        f"{hd('baseline-hc-s1','total_runs'):,} | {hd('baseline-hc-s2','total_runs'):,} |",
        f"| Order-log rows | {orderlog_rows[BENCH]:,} | "
        f"{orderlog_rows['baseline-hc-s1']:,} | {orderlog_rows['baseline-hc-s2']:,} |",
        "",
        "See `analysis.html` for the full comparison with charts and the workbook "
        "for all extract tables.",
        "",
    ]
    (out / "run.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
