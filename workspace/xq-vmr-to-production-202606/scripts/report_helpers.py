"""Report rendering helpers for the XQ-VMR baseline analysis.

Builds the consolidated Excel workbook, the self-contained HTML report,
run.md and the analysis/LATEST.md pointer+log.
"""
from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# --- formatting helpers ---------------------------------------------------------
def pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x * 100:.1f}%"


def num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:,.0f}"


def _months(df) -> list[str]:
    return sorted(df["month"].unique().tolist())


def cfr_color(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#f3f4f6"
    if v >= 0.99:
        return "#bbf7d0"
    if v >= 0.95:
        return "#dcfce7"
    if v >= 0.90:
        return "#fef9c3"
    if v >= 0.80:
        return "#fed7aa"
    return "#fecaca"


def days(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:,.1f}"


def hrs(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:,.0f}"


def qty(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:,.0f}"


def msu_fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:,.1f}"


def ratio3(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:,.3f}"


def dfc_color(v) -> str:
    """Inventory-coverage shading: very low (stockout risk) = red, very high (excess) = blue."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#f3f4f6"
    if v < 7:
        return "#fecaca"        # < 1 week: shortage risk
    if v < 14:
        return "#fed7aa"        # 1-2 weeks
    if v <= 45:
        return "#dcfce7"        # ~2-6 weeks: healthy
    if v <= 75:
        return "#fef9c3"        # 6-10 weeks: building
    return "#bfdbfe"            # > 10 weeks: excess



# --- HTML table builders --------------------------------------------------------
def cfr_pivot_html(cfr_main: pd.DataFrame, cfr_line_total: pd.DataFrame) -> str:
    months = _months(cfr_main)
    piv = cfr_main.pivot_table(index=["category", "line"], columns="month", values="cfr")
    full = cfr_line_total.set_index(["category", "line"])["cfr"]
    head = "".join(f"<th>{m}</th>" for m in months)
    rows = []
    for (cat, line), r in piv.iterrows():
        cells = []
        for m in months:
            v = r.get(m, np.nan)
            cells.append(f'<td style="background:{cfr_color(v)}">{pct(v)}</td>')
        fv = full.get((cat, line), np.nan)
        cells.append(f'<td style="background:{cfr_color(fv)};font-weight:600">{pct(fv)}</td>')
        rows.append(f"<tr><td>{_html.escape(str(cat))}</td><td>{_html.escape(str(line))}</td>{''.join(cells)}</tr>")
    return (
        f'<table class="tbl"><thead><tr><th>category</th><th>line</th>{head}'
        f'<th>Full&nbsp;period</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def count_pivot_html(changeover: pd.DataFrame, value: str, money: bool) -> str:
    months = _months(changeover)
    piv = changeover.pivot_table(index=["line", "changeover_id"], columns="month", values=value, aggfunc="sum")
    head = "".join(f"<th>{m}</th>" for m in months)
    rows = []
    for (line, cid), r in piv.iterrows():
        cells = "".join(f"<td>{num(r.get(m, np.nan))}</td>" for m in months)
        total = np.nansum([r.get(m, np.nan) for m in months])
        rows.append(
            f"<tr><td>{_html.escape(str(line))}</td><td>{_html.escape(str(cid))}</td>"
            f"{cells}<td style='font-weight:600'>{num(total)}</td></tr>"
        )
    # column totals
    foot_cells = "".join(f"<td>{num(piv[m].sum() if m in piv else np.nan)}</td>" for m in months)
    grand = num(np.nansum(piv.values))
    return (
        f'<table class="tbl"><thead><tr><th>line</th><th>changeover&nbsp;id</th>{head}'
        f'<th>Total</th></tr></thead><tbody>{"".join(rows)}</tbody>'
        f'<tfoot><tr><td colspan="2">Total</td>{foot_cells}<td>{grand}</td></tr></tfoot></table>'
    )


def simple_table_html(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                tds.append(f"<td>{fmt[c](v)}</td>")
            else:
                tds.append(f"<td>{_html.escape(str(v))}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def dfc_pivot_html(dfc_main: pd.DataFrame, partial_months: list[str]) -> str:
    months = _months(dfc_main)
    # "Unmapped" is a residual category/line mapping bucket, not a real production line;
    # its tiny forward demand produces meaningless DFC, so keep it out of the displayed pivot.
    src = dfc_main[(dfc_main["category"] != "Unmapped") & (dfc_main["line"] != "Unmapped")]
    piv = src.pivot_table(index=["category", "line"], columns="month", values="dfc_days")
    # drop rows that are entirely undefined (no forward demand anywhere -> no coverage signal)
    piv = piv.dropna(how="all")
    head = "".join(
        f"<th>{m}{'<sup>*</sup>' if m in partial_months else ''}</th>" for m in months
    )
    rows = []
    for (cat, line), r in piv.iterrows():
        cells = "".join(
            f'<td style="background:{dfc_color(r.get(m, np.nan))}">{days(r.get(m, np.nan))}</td>'
            for m in months
        )
        rows.append(
            f"<tr><td>{_html.escape(str(cat))}</td><td>{_html.escape(str(line))}</td>{cells}</tr>"
        )
    return (
        f'<table class="tbl"><thead><tr><th>category</th><th>line</th>{head}'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def ptt_pivot_html(ptt_main: pd.DataFrame) -> str:
    """Total production time (hr) pivot: category x line rows, month columns, + full-period total."""
    months = _months(ptt_main)
    piv = ptt_main.pivot_table(index=["category", "line"], columns="month",
                               values="total_hr", aggfunc="sum")
    head = "".join(f"<th>{m}</th>" for m in months)
    rows = []
    for (cat, line), r in piv.iterrows():
        cells = "".join(f"<td>{hrs(r.get(m, np.nan))}</td>" for m in months)
        total = np.nansum([r.get(m, np.nan) for m in months])
        rows.append(
            f"<tr><td>{_html.escape(str(cat))}</td><td>{_html.escape(str(line))}</td>"
            f"{cells}<td style='font-weight:600'>{hrs(total)}</td></tr>"
        )
    return (
        f'<table class="tbl"><thead><tr><th>category</th><th>line</th>{head}'
        f'<th>Full&nbsp;period</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def ptt_line_split_html(ptt_line_total: pd.DataFrame) -> str:
    """Per-line production vs changeover vs total hours + changeover share."""
    t = ptt_line_total.sort_values("total_hr", ascending=False)
    rows = []
    for _, r in t.iterrows():
        rows.append(
            f"<tr><td>{_html.escape(str(r['line']))}</td>"
            f"<td>{hrs(r['prod_hr'])}</td><td>{hrs(r['co_hr'])}</td>"
            f"<td style='font-weight:600'>{hrs(r['total_hr'])}</td>"
            f"<td>{r['co_share']:.1f}%</td></tr>"
        )
    tot_p = t["prod_hr"].sum(); tot_c = t["co_hr"].sum(); tot_t = t["total_hr"].sum()
    tot_s = tot_c / tot_t * 100 if tot_t else 0.0
    rows.append(
        f"<tr style='font-weight:700;border-top:2px solid #cbd5e1'><td>Total</td>"
        f"<td>{hrs(tot_p)}</td><td>{hrs(tot_c)}</td><td>{hrs(tot_t)}</td><td>{tot_s:.1f}%</td></tr>"
    )
    return (
        '<table class="tbl"><thead><tr><th>line</th><th>production&nbsp;hr</th>'
        '<th>changeover&nbsp;hr</th><th>total&nbsp;hr</th><th>changeover&nbsp;%</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def apq_line_html(apq_by_line: pd.DataFrame) -> str:
    """Per-line APQ table. MSU-aware: when an 'msu' column is present, report MSU
    produced and APQ in MSU/run; otherwise fall back to constrained qty / qty-per-run."""
    msu_mode = "msu" in apq_by_line.columns
    vol_col = "msu" if msu_mode else "con_qty"
    vol_fmt = msu_fmt if msu_mode else qty
    apq_fmt = msu_fmt if msu_mode else qty
    vol_hdr = "MSU&nbsp;produced" if msu_mode else "constrained&nbsp;qty"
    apq_hdr = "APQ&nbsp;(MSU/run)" if msu_mode else "APQ&nbsp;(qty/run)"
    t = apq_by_line.sort_values("apq", ascending=False)
    rows = []
    for _, r in t.iterrows():
        rows.append(
            f"<tr><td>{_html.escape(str(r['line']))}</td>"
            f"<td>{vol_fmt(r[vol_col])}</td><td>{qty(r['runs'])}</td>"
            f"<td style='font-weight:600'>{apq_fmt(r['apq'])}</td></tr>"
        )
    tq = t[vol_col].sum(); tr = t["runs"].sum()
    rows.append(
        f"<tr style='font-weight:700;border-top:2px solid #cbd5e1'><td>Total</td>"
        f"<td>{vol_fmt(tq)}</td><td>{qty(tr)}</td><td>{apq_fmt(tq / tr if tr else np.nan)}</td></tr>"
    )
    return (
        f'<table class="tbl"><thead><tr><th>line</th><th>{vol_hdr}</th>'
        f'<th>production&nbsp;runs</th><th>{apq_hdr}</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def apq_material_html(apq_by_material: pd.DataFrame, top: int = 25) -> str:
    """Top-N materials by production volume with their APQ. MSU-aware (see apq_line_html)."""
    msu_mode = "msu" in apq_by_material.columns
    vol_col = "msu" if msu_mode else "con_qty"
    vol_fmt = msu_fmt if msu_mode else qty
    apq_fmt = msu_fmt if msu_mode else qty
    vol_hdr = "MSU&nbsp;produced" if msu_mode else "constrained&nbsp;qty"
    apq_hdr = "APQ&nbsp;(MSU/run)" if msu_mode else "APQ"
    t = apq_by_material.sort_values(vol_col, ascending=False).head(top)
    rows = []
    for _, r in t.iterrows():
        rows.append(
            f"<tr><td>{_html.escape(str(r['material']))}</td>"
            f"<td>{_html.escape(str(r['line']))}</td>"
            f"<td>{_html.escape(str(r['category']))}</td>"
            f"<td>{vol_fmt(r[vol_col])}</td><td>{qty(r['runs'])}</td>"
            f"<td style='font-weight:600'>{apq_fmt(r['apq'])}</td></tr>"
        )
    return (
        f'<table class="tbl"><thead><tr><th>material</th><th>line</th><th>category</th>'
        f'<th>{vol_hdr}</th><th>runs</th><th>{apq_hdr}</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def msu_line_html(msu_by_line: pd.DataFrame) -> str:
    """Per-line wash count, MSU, and washes per MSU."""
    t = msu_by_line.sort_values("wash_count", ascending=False)
    rows = []
    for _, r in t.iterrows():
        rows.append(
            f"<tr><td>{_html.escape(str(r['line']))}</td>"
            f"<td>{qty(r['wash_count'])}</td><td>{msu_fmt(r['msu'])}</td>"
            f"<td style='font-weight:600'>{ratio3(r['wash_per_msu'])}</td></tr>"
        )
    tw = t["wash_count"].sum(); tm = t["msu"].sum()
    rows.append(
        f"<tr style='font-weight:700;border-top:2px solid #cbd5e1'><td>Total</td>"
        f"<td>{qty(tw)}</td><td>{msu_fmt(tm)}</td><td>{ratio3(tw / tm if tm else np.nan)}</td></tr>"
    )
    return (
        '<table class="tbl"><thead><tr><th>line</th><th>wash&nbsp;count&nbsp;(id&nbsp;2/3)</th>'
        '<th>MSU&nbsp;produced</th><th>washes&nbsp;/&nbsp;MSU</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def coverage_group_html(df: pd.DataFrame, level_col: str) -> str:
    """MOQ / APQ coverage (days) rollup by production line or category.

    Coverage columns are demand-weighted ratios of sums (already computed
    upstream); this just renders them, sorted by MOQ coverage descending.
    """
    t = df.sort_values("moq_coverage", ascending=False, na_position="last")
    rows = []
    for _, r in t.iterrows():
        rows.append(
            f"<tr><td>{_html.escape(str(r[level_col]))}</td>"
            f"<td>{qty(r['n_sku'])}</td>"
            f"<td>{msu_fmt(r['fcst_msu'])}</td>"
            f"<td>{msu_fmt(r['moq_msu'])}</td>"
            f"<td>{msu_fmt(r['apq_msu'])}</td>"
            f"<td style='font-weight:600'>{days(r['moq_coverage'])}</td>"
            f"<td style='font-weight:600'>{days(r['apq_coverage'])}</td></tr>"
        )
    return (
        f'<table class="tbl"><thead><tr><th>{_html.escape(level_col)}</th><th>#&nbsp;SKU</th>'
        f'<th>wk1-18&nbsp;fcst&nbsp;(MSU)</th><th>MOQ&nbsp;(MSU)</th><th>APQ&nbsp;(MSU)</th>'
        f'<th>MOQ&nbsp;coverage&nbsp;(days)</th><th>APQ&nbsp;coverage&nbsp;(days)</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def coverage_sku_html(cov_by_sku: pd.DataFrame) -> str:
    """Per-SKU MOQ / APQ coverage detail (all in-scope SKUs), sorted by forecast."""
    t = cov_by_sku.sort_values("fcst_msu", ascending=False)
    rows = []
    for _, r in t.iterrows():
        rows.append(
            f"<tr><td>{_html.escape(str(r['material']))}</td>"
            f"<td>{_html.escape(str(r['line']))}</td>"
            f"<td>{_html.escape(str(r['category']))}</td>"
            f"<td>{msu_fmt(r['fcst_msu'])}</td>"
            f"<td>{ratio3(r['daily_fcst_msu'])}</td>"
            f"<td>{msu_fmt(r['moq_msu'])}</td>"
            f"<td>{msu_fmt(r['apq_msu'])}</td>"
            f"<td style='font-weight:600'>{days(r['moq_coverage'])}</td>"
            f"<td style='font-weight:600'>{days(r['apq_coverage'])}</td></tr>"
        )
    return (
        '<table class="tbl"><thead><tr><th>material</th><th>line</th><th>category</th>'
        '<th>wk1-18&nbsp;fcst&nbsp;(MSU)</th><th>daily&nbsp;fcst&nbsp;(MSU)</th>'
        '<th>MOQ&nbsp;(MSU)</th><th>APQ&nbsp;(MSU)</th>'
        '<th>MOQ&nbsp;cov&nbsp;(d)</th><th>APQ&nbsp;cov&nbsp;(d)</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def write_workbook(xlsx: Path, meta: dict, cfr_main, cfr_by_line, cfr_by_cat,
                   cfr_by_month, cfr_line_total, co_count, co_cost, co_def,
                   dfc_main, dfc_by_line, dfc_by_cat, dfc_by_month,
                   ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month, ptt_line_total,
                   apq_by_line, apq_by_material, msu_by_line, msu_material,
                   cov_by_sku=None, cov_by_line=None, cov_by_cat=None):
    if bool(meta.get("apq_is_msu", False)):
        apq_def = (f"APQ = total MSU produced / production-run count = "
                   f"{meta['apq_total_msu']:,.1f} / {meta['apq_total_runs']} = "
                   f"{meta['apq_overall']:,.3f} MSU/run overall; reported by line and by material")
    else:
        apq_def = (f"APQ = total con_planned_qty / production-run count = "
                   f"{meta['apq_total_con_qty']:,.0f} / {meta['apq_total_runs']} = "
                   f"{meta['apq_overall']:,.1f} units/run overall; reported by line and by material")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        notes = pd.DataFrame({
            "field": [
                "project", "scenario", "db run_id", "config_name", "analysis run_id",
                "generated", "category source", "category note",
                "category coverage (databricks)", "category coverage (enriched)",
                "category resolution (by order qty)", "category enrichment rule",
                "TOF definition", "line attribution", "changeover id meaning",
                "changeover cost", "DFC definition", "DFC inventory scope",
                "DFC demand basis", "DFC forward window", "DFC partial-window months",
                "production-time definition", "production-time qty basis",
                "production-time rate", "production-time changeover basis",
                "production-time changeover->category allocation", "production-time months",
                "APQ definition", "APQ run basis", "APQ qty basis",
                "wash count definition", "MSU definition", "SU factor source",
                "unmapped-line materials", "unmapped-category materials",
            ],
            "value": [
                meta["project"], meta["scenario"], meta["run_id_db"], meta["config_name"],
                meta["analysis_run_id"], meta["generated"], meta["category_source"],
                meta["category_note"] or "(fresh Databricks pull)",
                meta["category_coverage"], meta["category_coverage_enriched"],
                (f"databricks {meta['cat_databricks_share']} ({meta['cat_databricks_n']} mat), "
                 f"line-inferred {meta['cat_inferred_share']} ({meta['cat_inferred_n']} mat), "
                 f"unmapped {meta['cat_unmapped_share']} ({meta['cat_unmapped_n']} mat)"),
                "new VMR materials absent from SKU master inherit category from their delegate_line "
                "when that line is single-category among mapped materials; mixed lines stay Unmapped",
                "TOF = sum(shipment_qty) / sum(order_qty); ratio of sums; order log deduped on 7-field key",
                "material -> delegate_line via cfg_m4_materiallocationlinecfg (material level, all demand locations)",
                "changeover_type column in module4_output_changeoverlog == changeover_id in cfg_m4_changeoverdefinition",
                "cost column from module4_output_changeoverlog (already per-event cost x count)",
                (f"DFC = month-end DC ending_soh / avg daily forward demand over next "
                 f"{meta['dfc_forward_days']} days; ratio of sums per rollup"),
                f"DC inventory only ({meta['dfc_dc_locations_n']} DC locations; plant 1864 excluded)",
                (f"demand forecast (cfg_m1_demandforecast, weekly->daily) at DC locations; "
                 f"plant demand {meta['dfc_plant_demand_share']} excluded"),
                (f"sim start {meta['dfc_sim_start']}; forecast horizon ends {meta['dfc_horizon_end']}; "
                 f"month-end snapshots"),
                ", ".join(meta["dfc_partial_months"]) or "none",
                ("production total time (hr) = production time + changeover time; "
                 "production time = con_planned_qty / prd_rate (unit/hr) per material, summed"),
                meta["ptt_qty_basis"],
                "cfg_m4_materiallocationlinecfg.prd_rate (unit/hr), one rate per material",
                ("module4_output_changeoverlog.time (already = changeover_id x count x "
                 "per-event changeover time, hr), summed per line"),
                (f"changeover hours allocated to category pro-rata by production hours within "
                 f"line x month; mixed line(s): {', '.join(meta['ptt_mixed_lines']) or 'none'}"),
                ", ".join(meta["ptt_months"]) or "none",
                apq_def,
                meta["apq_run_basis"],
                meta["apq_qty_basis"],
                (f"wash count = sum of changeover count where changeover id in ({meta['msu_wash_ids']}) "
                 f"from module4_output_changeoverlog, per line"),
                (f"MSU = sum(con_planned_qty x su_factor / 1000) over a line's materials; "
                 f"wash/MSU = wash count / MSU (overall {meta['msu_wash_per_msu_overall']:.3f})"),
                (f"su_factor from local workbook '{meta['msu_suf_path']}' (primary, {meta['msu_n_suf']} mat) "
                 f"+ Databricks ps_psc_sku_master.su_factor_for_buom (fallback, {meta['msu_n_dbx']} mat) "
                 f"+ manual input (fallback, {meta.get('msu_n_manual', 0)} mat); "
                 f"missing {meta['msu_n_missing']}"),
                ", ".join(meta["unmapped_line_materials"]) or "none",
                ", ".join(meta["unmapped_category_materials"]) or "none",
            ],
        })
        if cov_by_line is not None:
            notes = pd.concat([notes, pd.DataFrame({
                "field": ["8 MOQ/APQ coverage definition", "8 MOQ/APQ coverage scope",
                          "8 MOQ/APQ coverage rollup"],
                "value": [
                    (f"coverage (days) = quantity (MSU) / (wk1-18 forecast MSU / {meta.get('cov_days', 126)}); "
                     f"MOQ = current min_batch x su_factor / 1000; APQ = simulated avg MSU per production run"),
                    (f"per SKU produced at 1864; wk1-18 = forecast weeks 1..18 ({meta.get('cov_window', '')}) "
                     f"summed over the SKU's 1864 sourcing sub-network (incl. 1864 when a node); "
                     f"{meta.get('cov_n_sku', '?')} SKUs in scope, APQ on {meta.get('cov_n_sku_apq', '?')}"),
                    ("by line / by category = demand-weighted ratio of sums "
                     "(sum MOQ or sum APQ / sum daily forecast); APQ denominator restricted to produced SKUs"),
                ],
            })], ignore_index=True)
        notes.to_excel(xw, sheet_name="Notes", index=False)

        cfr_main.to_excel(xw, sheet_name="1_TOF_cat_line_month", index=False)
        cfr_main.pivot_table(index=["category", "line"], columns="month", values="cfr").to_excel(
            xw, sheet_name="1_TOF_pivot")
        cfr_by_line.to_excel(xw, sheet_name="1_TOF_by_line_month", index=False)
        cfr_by_cat.to_excel(xw, sheet_name="1_TOF_by_category_month", index=False)

        co_count.to_excel(xw, sheet_name="2_Changeover_count", index=False)
        co_count.pivot_table(index=["line", "changeover_id"], columns="month",
                             values="changeover_count", aggfunc="sum").to_excel(
            xw, sheet_name="2_CO_count_pivot")

        co_cost.to_excel(xw, sheet_name="3_Changeover_cost", index=False)
        co_cost.pivot_table(index=["line", "changeover_id"], columns="month",
                            values="changeover_cost", aggfunc="sum").to_excel(
            xw, sheet_name="3_CO_cost_pivot")

        dfc_main.to_excel(xw, sheet_name="4_DFC_cat_line_month", index=False)
        dfc_main.pivot_table(index=["category", "line"], columns="month",
                             values="dfc_days").to_excel(xw, sheet_name="4_DFC_pivot")
        dfc_by_cat.to_excel(xw, sheet_name="4_DFC_by_category_month", index=False)
        dfc_by_line.to_excel(xw, sheet_name="4_DFC_by_line_month", index=False)

        ptt_main.to_excel(xw, sheet_name="5_ProdTime_cat_line_month", index=False)
        ptt_main.pivot_table(index=["category", "line"], columns="month",
                             values="total_hr", aggfunc="sum").to_excel(
            xw, sheet_name="5_ProdTime_pivot")
        ptt_line_total.to_excel(xw, sheet_name="5_ProdTime_by_line_total", index=False)
        ptt_by_line.to_excel(xw, sheet_name="5_ProdTime_by_line_month", index=False)
        ptt_by_cat.to_excel(xw, sheet_name="5_ProdTime_by_category_month", index=False)
        ptt_by_month.to_excel(xw, sheet_name="5_ProdTime_by_month", index=False)

        apq_by_line.to_excel(xw, sheet_name="6_APQ_by_line", index=False)
        apq_by_material.to_excel(xw, sheet_name="6_APQ_by_material", index=False)

        msu_by_line.to_excel(xw, sheet_name="7_Wash_MSU_by_line", index=False)
        msu_material.to_excel(xw, sheet_name="7_MSU_by_material", index=False)

        if cov_by_line is not None:
            cov_by_line.to_excel(xw, sheet_name="8_Coverage_by_line", index=False)
            cov_by_cat.to_excel(xw, sheet_name="8_Coverage_by_category", index=False)
            cov_by_sku.to_excel(xw, sheet_name="8_Coverage_by_SKU", index=False)

        co_def.to_excel(xw, sheet_name="ChangeoverDefs", index=False)

        # column widths
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 48)


# --- HTML report ----------------------------------------------------------------
def _findings(cfr_line_total, cfr_by_cat, cfr_by_month, changeover,
              dfc_by_cat, dfc_by_line, dfc_by_month, full_months,
              ptt_line_total, ptt_by_cat, ptt_by_month,
              apq_by_line, apq_by_material, msu_by_line):
    tot_o = cfr_line_total["order_qty"].sum()
    tot_s = cfr_line_total["shipment_qty"].sum()
    overall = tot_s / tot_o if tot_o else np.nan

    cat = cfr_line_total.groupby("category").agg(o=("order_qty", "sum"), s=("shipment_qty", "sum"))
    cat["cfr"] = cat["s"] / cat["o"]
    line = cfr_line_total.groupby("line").agg(o=("order_qty", "sum"), s=("shipment_qty", "sum"))
    line["cfr"] = line["s"] / line["o"]
    worst_line = line["cfr"].idxmin()
    worst_line_v = line["cfr"].min()
    mon = cfr_by_month.set_index("month")["cfr"]
    worst_mon = mon.idxmin()
    worst_mon_v = mon.min()

    co_cnt = int(changeover["changeover_count"].sum())
    co_cost = changeover["changeover_cost"].sum()
    cost_line = changeover.groupby("line")["changeover_cost"].sum().sort_values(ascending=False)
    cost_id = changeover.groupby("changeover_id")["changeover_cost"].sum()
    cnt_id = changeover.groupby("changeover_id")["changeover_count"].sum()

    # DFC findings (restrict headline numbers to full-window months for reliability)
    def _wavg_dfc(g):
        d = g["avg_daily_demand"].replace(0, np.nan).sum()
        return g["dc_inv"].sum() / d if d and d >= 1.0 else np.nan

    dcm = dfc_by_month[dfc_by_month["month"].isin(full_months)] if full_months else dfc_by_month
    dfc_overall = _wavg_dfc(dcm) if not dcm.empty else np.nan
    dfc_mon = dfc_by_month.set_index("month")["dfc_days"]
    dfc_cat_full = (dfc_by_cat[dfc_by_cat["month"].isin(full_months)]
                    if full_months else dfc_by_cat)
    dfc_cat = (dfc_cat_full.groupby("category").apply(_wavg_dfc, include_groups=False)
               .rename("dfc").dropna())
    dfc_line_full = (dfc_by_line[dfc_by_line["month"].isin(full_months)]
                     if full_months else dfc_by_line)
    # exclude the residual "Unmapped" line bucket (not a real line; tiny demand -> noise)
    dfc_line_full = dfc_line_full[dfc_line_full["line"] != "Unmapped"]
    dfc_line = (dfc_line_full.groupby("line").apply(_wavg_dfc, include_groups=False)
                .rename("dfc").dropna().sort_values())

    # production-time findings
    ptt_prod = float(ptt_line_total["prod_hr"].sum())
    ptt_co = float(ptt_line_total["co_hr"].sum())
    ptt_tot = float(ptt_line_total["total_hr"].sum())
    ptt_co_share = ptt_co / ptt_tot * 100 if ptt_tot else 0.0
    ptt_line_sorted = ptt_line_total.sort_values("total_hr", ascending=False)
    ptt_busy_line = ptt_line_sorted.iloc[0]["line"] if not ptt_line_sorted.empty else "—"
    ptt_busy_line_v = float(ptt_line_sorted.iloc[0]["total_hr"]) if not ptt_line_sorted.empty else np.nan
    pm = ptt_by_month.set_index("month")["total_hr"]
    ptt_busy_mon = pm.idxmax() if not pm.empty else "—"
    ptt_busy_mon_v = float(pm.max()) if not pm.empty else np.nan
    ptt_cat = ptt_by_cat.groupby("category")["total_hr"].sum().sort_values(ascending=False)
    ptt_co_share_line = ptt_line_total.set_index("line")["co_share"].sort_values(ascending=False)

    # APQ findings (MSU-aware: prefer MSU volume when present)
    apq_vol_col = "msu" if "msu" in apq_by_line.columns else "con_qty"
    apq_q = float(apq_by_line[apq_vol_col].sum())
    apq_r = int(apq_by_line["runs"].sum())
    apq_overall = apq_q / apq_r if apq_r else np.nan
    apq_line = apq_by_line.set_index("line")["apq"].sort_values(ascending=False)
    apq_hi_mat = apq_by_material.sort_values("apq", ascending=False).head(5)
    apq_lo_mat = apq_by_material.sort_values("apq", ascending=True).head(5)

    # wash / MSU findings
    msu_wash = int(msu_by_line["wash_count"].sum())
    msu_msu = float(msu_by_line["msu"].sum())
    msu_overall = msu_wash / msu_msu if msu_msu else np.nan
    msu_wpm = msu_by_line.set_index("line")["wash_per_msu"].sort_values(ascending=False)
    msu_msu_line = msu_by_line.set_index("line")["msu"].sort_values(ascending=False)
    msu_wash_line = msu_by_line.set_index("line")["wash_count"].sort_values(ascending=False)

    return {
        "overall": overall, "cat": cat, "line": line, "worst_line": worst_line,
        "worst_line_v": worst_line_v, "worst_mon": worst_mon, "worst_mon_v": worst_mon_v,
        "co_cnt": co_cnt, "co_cost": co_cost, "cost_line": cost_line,
        "cost_id": cost_id, "cnt_id": cnt_id,
        "dfc_overall": dfc_overall, "dfc_mon": dfc_mon, "dfc_cat": dfc_cat, "dfc_line": dfc_line,
        "ptt_prod": ptt_prod, "ptt_co": ptt_co, "ptt_tot": ptt_tot, "ptt_co_share": ptt_co_share,
        "ptt_busy_line": ptt_busy_line, "ptt_busy_line_v": ptt_busy_line_v,
        "ptt_busy_mon": ptt_busy_mon, "ptt_busy_mon_v": ptt_busy_mon_v,
        "ptt_cat": ptt_cat, "ptt_co_share_line": ptt_co_share_line,
        "apq_overall": apq_overall, "apq_line": apq_line,
        "apq_hi_mat": apq_hi_mat, "apq_lo_mat": apq_lo_mat,
        "msu_wash": msu_wash, "msu_msu": msu_msu, "msu_overall": msu_overall,
        "msu_wpm": msu_wpm, "msu_msu_line": msu_msu_line, "msu_wash_line": msu_wash_line,
    }


def write_html(html_path: Path, meta, charts, cfr_main, cfr_by_line, cfr_by_cat,
               cfr_by_month, cfr_line_total, co_count, co_cost, co_def, changeover,
               dfc_main, dfc_by_cat, dfc_by_line, dfc_by_month,
               ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month, ptt_line_total,
               apq_by_line, apq_by_material, msu_by_line,
               cov_by_sku=None, cov_by_line=None, cov_by_cat=None):
    full_months = meta["dfc_full_months"]
    partial_months = meta["dfc_partial_months"]
    f = _findings(cfr_line_total, cfr_by_cat, cfr_by_month, changeover,
                  dfc_by_cat, dfc_by_line, dfc_by_month, full_months,
                  ptt_line_total, ptt_by_cat, ptt_by_month,
                  apq_by_line, apq_by_material, msu_by_line)

    def img(key, alt):
        uri = charts.get(key)
        return f'<img src="{uri}" alt="{alt}" style="max-width:100%;height:auto;">' if uri else ""

    # APQ unit handling (MSU-aware; defaults keep the legacy units/run wording)
    apq_is_msu = bool(meta.get("apq_is_msu", False))
    apqf = msu_fmt if apq_is_msu else qty
    if apq_is_msu:
        apq_unit = "MSU/run"
        apq_metric = "MSU"
        apq_title_metric = "MSU"
        apq_vol_value = meta["apq_total_msu"]
        apq_vol_fmt = msu_fmt
        apq_vol_label = "MSU produced"
        apq_legend_metric = "total MSU produced"
        apq_span = (f"{msu_fmt(meta['apq_total_msu'])} MSU "
                    f"({qty(meta['apq_total_con_qty'])} constrained units)")
        apq_numerator = "MSU (con_planned_qty x SU factor / 1000)"
        apq_metric_long = "MSU produced"
        apq_def_numer = "Σ (con_planned_qty × SU factor ÷ 1000) = MSU produced"
    else:
        apq_unit = "units/run"
        apq_metric = "constrained qty"
        apq_title_metric = "constrained planned qty"
        apq_vol_value = meta["apq_total_con_qty"]
        apq_vol_fmt = qty
        apq_vol_label = "constrained qty"
        apq_legend_metric = "total constrained planned qty"
        apq_span = f"{qty(meta['apq_total_con_qty'])} constrained units"
        apq_numerator = "con_planned_qty (the constrained plan, equal to produced qty here)"
        apq_metric_long = "constrained planned quantity"
        apq_def_numer = "Σ con_planned_qty"

    cat_rows = "".join(
        f"<li><b>{_html.escape(str(c))}</b>: TOF {pct(r['cfr'])} "
        f"(shipped {num(r['s'])} / ordered {num(r['o'])})</li>"
        for c, r in f["cat"].iterrows()
    )
    cost_line_rows = "".join(
        f"<li><b>{_html.escape(str(l))}</b>: {num(v)}</li>" for l, v in f["cost_line"].head(3).items()
    )
    id_rows = "".join(
        f"<li>id <b>{_html.escape(str(i))}</b>: {num(int(f['cnt_id'][i]))} changeovers, cost {num(f['cost_id'][i])}</li>"
        for i in sorted(f["cost_id"].index)
    )
    dfc_cat_rows = "".join(
        f"<li><b>{_html.escape(str(c))}</b>: {days(v)} days</li>" for c, v in f["dfc_cat"].items()
    )
    dfc_line_rows = "".join(
        f"<li><b>{_html.escape(str(l))}</b>: {days(v)} days</li>" for l, v in f["dfc_line"].items()
    )
    ptt_cat_rows = "".join(
        f"<li><b>{_html.escape(str(c))}</b>: {hrs(v)} hr</li>" for c, v in f["ptt_cat"].items()
    )
    apq_line_rows = "".join(
        f"<li><b>{_html.escape(str(l))}</b>: {apqf(v)} {apq_unit}</li>" for l, v in f["apq_line"].items()
    )
    msu_wpm_rows = "".join(
        f"<li><b>{_html.escape(str(l))}</b>: {ratio3(v)} washes/MSU "
        f"({qty(f['msu_wash_line'].get(l, np.nan))} washes / {msu_fmt(f['msu_msu_line'].get(l, np.nan))} MSU)</li>"
        for l, v in f["msu_wpm"].items()
    )

    cat_note = meta["category_note"] or "Fresh pull from Databricks ps_psc_sku_master (category_en)."
    unmapped_line = ", ".join(meta["unmapped_line_materials"]) or "none"
    unmapped_cat = ", ".join(meta["unmapped_category_materials"]) or "none"

    # KPI 8 (MOQ / APQ coverage) fragments — optional; empty strings keep the
    # non-HC sibling report (which passes no coverage) byte-for-byte unchanged.
    has_cov = cov_by_line is not None and bool(meta.get("cov_has"))
    if has_cov:
        cov_window_disp = str(meta["cov_window"]).replace("..", " \u2192 ")
        cov_summary_card = (
            f'<div class="card"><div class="v">{days(meta["cov_moq_overall"])}</div>'
            f'<div class="l">MOQ coverage (days a min-batch covers) &mdash; JASO 6.29-11.1</div></div>'
            f'<div class="card"><div class="v">{days(meta["cov_apq_overall"])}</div>'
            f'<div class="l">APQ coverage (days an avg run covers) &mdash; JASO 6.29-11.1</div></div>'
        )
        cov_bg_item = ("<li><b>MOQ / APQ coverage</b> (days of demand a minimum batch / an average "
                       "production run covers) by SKU, production line and category</li>")
        cov_scope_item = (
            f"<li><b>MOQ / APQ coverage:</b> for every SKU produced at 1864, MOQ (current min_batch) and "
            f"APQ (simulated avg MSU per run) in MSU \u00f7 daily forecast, where daily forecast = wk1-18 "
            f"forecast (MSU) \u00f7 {meta['cov_days']} days. wk1-18 forecast is summed over the SKU's 1864 "
            f"sourcing sub-network (the plant node plus every DC whose sourcing chain reaches 1864). "
            f"{meta['cov_n_sku']} SKUs in scope (APQ on {meta['cov_n_sku_apq']}).</li>")
        cov_finding = (
            f"<li>A minimum production batch covers <b>{days(meta['cov_moq_overall'])} days</b> of demand "
            f"overall, while an average production run covers <b>{days(meta['cov_apq_overall'])} days</b> "
            f"(JASO 6.29-11.1, demand-weighted). The by-line / by-category tables highlight where batch sizes "
            f"run long relative to demand &mdash; candidates for smaller, more frequent runs.</li>")
        cov_appendix = (
            f"<li><b>MOQ / APQ coverage</b> (days) = quantity (MSU) \u00f7 daily forecast (MSU/day), where "
            f"daily forecast = \u03a3 demand-forecast qty over weeks 1..18 (== {meta['cov_window']}, "
            f"{meta['cov_days']} days) at the SKU's 1864 sourcing sub-network \u00d7 SU factor \u00f7 1000, "
            f"then \u00f7 {meta['cov_days']}. <i>MOQ</i> = current production <code>min_batch</code> "
            f"(<code>cfg_m4_materiallocationlinecfg</code>) \u00d7 SU factor \u00f7 1000. <i>APQ</i> = simulated "
            f"avg MSU produced per production run (KPI 6). The 1864 sub-network is the set of locations whose "
            f"<code>sourcing</code> chain in <code>cfg_global_network</code> reaches plant 1864 (multi-echelon; "
            f"the plant itself is included when it carries forecast). Line / category rollups are "
            f"demand-weighted ratios of sums (\u03a3 MOQ or \u03a3 APQ \u00f7 \u03a3 daily forecast); the APQ "
            f"rollup denominator is restricted to produced SKUs. {meta['cov_n_sku']} SKUs in scope; "
            f"{meta['cov_n_no_fcst']} produced SKUs excluded (no forecast in their 1864 network), "
            f"{meta['cov_n_missing_su']} excluded (missing SU factor). Extracts: "
            f"<code>coverage_by_sku.csv</code>, <code>coverage_by_line.csv</code>, "
            f"<code>coverage_by_category.csv</code>.</li>")

        def _covimg(key, alt):
            uri = charts.get(key)
            return (f'<div class="chart"><img src="{uri}" alt="{alt}" '
                    f'style="max-width:100%;height:auto;"></div>') if uri else ""

        cov_section = f"""
<h2>KPI 8 &mdash; MOQ / APQ coverage (days) by SKU, production line &amp; category</h2>
<div class="legend">Coverage (days) = quantity (MSU) \u00f7 daily forecast (MSU/day); daily forecast = wk1-18
  forecast (MSU) \u00f7 {meta['cov_days']}. <b>MOQ</b> = current production min_batch (\u00d7 SU factor \u00f7 1000);
  <b>APQ</b> = simulated avg MSU produced per production run. Window: JASO {cov_window_disp} ({meta['cov_days']} days).
  Line / category rollups are demand-weighted ratios of sums.</div>
<div class="kpi-cards">
  <div class="card"><div class="v">{days(meta['cov_moq_overall'])}</div><div class="l">Overall MOQ coverage (days)</div></div>
  <div class="card"><div class="v">{days(meta['cov_apq_overall'])}</div><div class="l">Overall APQ coverage (days)</div></div>
  <div class="card"><div class="v">{qty(meta['cov_n_sku'])}</div><div class="l">SKUs in scope (APQ on {qty(meta['cov_n_sku_apq'])})</div></div>
  <div class="card"><div class="v">{msu_fmt(meta['cov_total_fcst_msu'])}</div><div class="l">Total wk1-18 forecast (MSU)</div></div>
</div>
<h3>Coverage by production line</h3>
{coverage_group_html(cov_by_line, 'line')}
{_covimg('cov_line', 'MOQ vs APQ coverage by line')}
<h3>Coverage by category</h3>
{coverage_group_html(cov_by_cat, 'category')}
{_covimg('cov_cat', 'MOQ vs APQ coverage by category')}
<h3>By-SKU detail &mdash; APQ / MOQ coverage with current APQ, MOQ &amp; forecast (MSU)</h3>
<p class="small">All {meta['cov_n_sku']} in-scope SKUs, sorted by wk1-18 forecast. Full copy in
<code>extracts/coverage_by_sku.csv</code> and workbook sheet <code>8_Coverage_by_SKU</code>.
APQ coverage shows &mdash; for SKUs with no production run in the window.</p>
{coverage_sku_html(cov_by_sku)}
"""
    else:
        cov_summary_card = cov_bg_item = cov_scope_item = ""
        cov_finding = cov_appendix = cov_section = ""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>XQ VMR Baseline — TOF, Changeover, DFC, Production Time, APQ &amp; MSU Analysis</title>
<style>
  :root {{ --ink:#0f172a; --mut:#64748b; --line:#e2e8f0; --accent:#2563eb; --bg:#f8fafc; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
         margin:0; background:var(--bg); line-height:1.55; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:32px 28px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:20px; margin:38px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--accent); }}
  h3 {{ font-size:15px; margin:22px 0 8px; color:var(--mut); text-transform:uppercase; letter-spacing:.04em; }}
  .band {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 18px; margin:14px 0; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:8px 22px; font-size:13px; color:var(--mut); }}
  .meta b {{ color:var(--ink); }}
  table.tbl {{ border-collapse:collapse; width:100%; font-size:12.5px; margin:10px 0 4px; background:#fff; }}
  table.tbl th, table.tbl td {{ border:1px solid var(--line); padding:5px 8px; text-align:right; }}
  table.tbl th {{ background:#f1f5f9; text-align:center; position:sticky; top:0; }}
  table.tbl td:first-child, table.tbl td:nth-child(2) {{ text-align:left; }}
  table.tbl tfoot td {{ font-weight:600; background:#f8fafc; }}
  .chart {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px; margin:14px 0; }}
  .kpi-cards {{ display:flex; flex-wrap:wrap; gap:14px; margin:12px 0; }}
  .card {{ flex:1 1 180px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .card .v {{ font-size:24px; font-weight:700; color:var(--accent); }}
  .card .l {{ font-size:12px; color:var(--mut); }}
  ul {{ margin:6px 0 6px 18px; }} li {{ margin:3px 0; }}
  .legend {{ font-size:12px; color:var(--mut); margin:6px 0; }}
  .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:middle; margin:0 4px 0 10px; }}
  .small {{ font-size:12px; color:var(--mut); }}
  code {{ background:#eef2ff; padding:1px 5px; border-radius:4px; font-size:12px; }}
</style></head><body><div class="wrap">

<h1>XQ VMR Baseline — TOF, Changeover, DFC, Production Time, APQ &amp; MSU Analysis</h1>
<div class="meta band">
  <span><b>Project:</b> {meta['project']}</span>
  <span><b>Scenario:</b> {meta['scenario']}</span>
  <span><b>DB run_id:</b> <code>{meta['run_id_db']}</code></span>
  <span><b>Period:</b> 2026-06-29 → 2026-10-31 (reported through end of October)</span>
  <span><b>Plant:</b> XQ (1864)</span>
  <span><b>Generated:</b> {meta['generated']}</span>
</div>

<h2>Executive Summary</h2>
<div class="band">
<div class="kpi-cards">
  <div class="card"><div class="v">{pct(f['overall'])}</div><div class="l">Overall TOF (full period, all lines &amp; categories)</div></div>
  <div class="card"><div class="v">{days(f['dfc_overall'])}</div><div class="l">Avg month-end DFC (DC inventory, full-window months)</div></div>
  <div class="card"><div class="v">{num(f['co_cnt'])}</div><div class="l">Total changeovers</div></div>
  <div class="card"><div class="v">{num(f['co_cost'])}</div><div class="l">Total changeover cost</div></div>
  <div class="card"><div class="v">{hrs(f['ptt_tot'])}</div><div class="l">Production total time (hr): production + changeover</div></div>
  <div class="card"><div class="v">{apqf(f['apq_overall'])}</div><div class="l">APQ — avg {apq_metric} per production run</div></div>
  <div class="card"><div class="v">{ratio3(f['msu_overall'])}</div><div class="l">Washes per MSU (id 2/3 changeovers ÷ MSU produced)</div></div>
  {cov_summary_card}
</div>
<ul>
  <li>Baseline case fill rate across the XQ VMR-to-production flow is <b>{pct(f['overall'])}</b> for the full window.
      By category: {''.join(f"{_html.escape(str(c))} {pct(r['cfr'])}; " for c, r in f['cat'].iterrows())}</li>
  <li>Service is weakest on line <b>{_html.escape(str(f['worst_line']))}</b> ({pct(f['worst_line_v'])}) and in month
      <b>{f['worst_mon']}</b> ({pct(f['worst_mon_v'])}).</li>
  <li>Month-end DC inventory coverage (DFC) averages <b>{days(f['dfc_overall'])} days</b> over the full-window months;
      thinnest on line <b>{_html.escape(str(f['dfc_line'].index[0])) if len(f['dfc_line']) else '&mdash;'}</b>
      ({days(f['dfc_line'].iloc[0]) if len(f['dfc_line']) else '&mdash;'} days).</li>
  <li>Changeover effort totals <b>{num(f['co_cnt'])}</b> events / <b>{num(f['co_cost'])}</b> cost; the bulk concentrates
      on line <b>{_html.escape(str(f['cost_line'].index[0]))}</b> and on changeover id with the highest unit cost.</li>
  <li>Production total time is <b>{hrs(f['ptt_tot'])} hr</b> (production {hrs(f['ptt_prod'])} hr + changeover
      {hrs(f['ptt_co'])} hr; changeover is <b>{f['ptt_co_share']:.1f}%</b> of line time), busiest on line
      <b>{_html.escape(str(f['ptt_busy_line']))}</b> ({hrs(f['ptt_busy_line_v'])} hr).</li>
  <li>Average production-run size (APQ) is <b>{apqf(f['apq_overall'])} {apq_unit}</b> overall
      ({apq_span} / {num(meta['apq_total_runs'])} runs); largest batches on line
      <b>{_html.escape(str(f['apq_line'].index[0])) if len(f['apq_line']) else '&mdash;'}</b>
      ({apqf(f['apq_line'].iloc[0]) if len(f['apq_line']) else '&mdash;'} {apq_unit}).</li>
  <li>Wash changeovers (id {meta['msu_wash_ids']}) total <b>{num(f['msu_wash'])}</b> against <b>{msu_fmt(f['msu_msu'])} MSU</b>
      produced = <b>{ratio3(f['msu_overall'])}</b> washes/MSU; most wash-intensive line
      <b>{_html.escape(str(f['msu_wpm'].index[0])) if len(f['msu_wpm']) else '&mdash;'}</b>
      ({ratio3(f['msu_wpm'].iloc[0]) if len(f['msu_wpm']) else '&mdash;'} washes/MSU).</li>
</ul>
<p class="small">Technical detail, definitions and data lineage are in the Appendix.</p>
</div>

<h2>Background &amp; Context</h2>
<div class="band">
<p>This is the <b>baseline</b> reference run for project <code>{meta['project']}</code>: the current-state simulation of the
HC/PCC flow at the XQ plant (1864) from VMR to production. The simulation runs 2026-06-29 to 2026-11-01; this report covers
results <b>through end of October 2026</b> (the partial November tail is excluded). It establishes the comparison
point for later what-if scenarios on AO ratio and advance lead time. The project's focus metrics are MOE, national inventory
and service; this report drills into three operational views requested on top of that baseline:</p>
<ol>
  <li><b>TOF</b> (case fill rate / service) by category × production line × month</li>
  <li><b>Changeover count</b> by production line × month × changeover id</li>
  <li><b>Changeover cost</b> by production line × month × changeover id</li>
  <li><b>Month-end DFC</b> (days forward coverage) by category × production line × month</li>
  <li><b>Production total time</b> (production + changeover, hr) by category × production line × month</li>
  <li><b>APQ</b> (avg {apq_title_metric} per production run) by production line and by material</li>
  <li><b>Wash count / MSU</b> by production line (wash = changeover id 2/3; MSU = qty × SU factor ÷ 1000)</li>
  {cov_bg_item}
</ol>
</div>

<h2>Analysis Scope</h2>
<div class="band">
<ul>
  <li><b>Simulation window:</b> 2026-06-29 → 2026-11-01; <b>reporting horizon capped at 2026-10-31</b> (the partial
      November tail is excluded from every KPI). June is a partial month.</li>
  <li><b>Network:</b> XQ plant 1864; customer demand is booked at downstream DC locations and attributed back to the
      producing line.</li>
  <li><b>Category:</b> HC (category_en = <i>Hair</i>) and PCC. Source: {cat_note} Databricks coverage
      {meta['category_coverage']}; after line-inference for new VMR materials, {meta['category_coverage_enriched']}.
      By order qty: databricks {meta['cat_databricks_share']}, line-inferred {meta['cat_inferred_share']},
      unmapped {meta['cat_unmapped_share']}.</li>
  <li><b>Production line:</b> each material mapped to its <code>delegate_line</code> from
      <code>cfg_m4_materiallocationlinecfg</code> (material level). Unmapped materials: {unmapped_line}.</li>
  <li><b>Month-end DFC:</b> DC ending inventory ({meta['dfc_dc_locations_n']} DC locations, plant 1864 excluded) divided by
      average daily demand-forecast over the next {meta['dfc_forward_days']} days. Forecast horizon ends
      {meta['dfc_horizon_end']}, so months {', '.join(partial_months) or 'none'} have a shortened forward window
      (flagged with <sup>*</sup>) and are read with caution.</li>
  <li><b>Production total time:</b> production time (constrained planned qty ÷ production rate, unit/hr) plus changeover
      time, in hours, by month ({', '.join(meta['ptt_months'])}, from <code>production_plan_date</code>). All production is
      at plant 1864; no production is booked in June/November.</li>
  <li><b>APQ:</b> total {apq_legend_metric} ÷ number of production runs, where a run is a
      <code>module4_output_productionplan</code> row that begins with a changeover (<code>changeover_id</code> not null);
      reported by production line and by material.</li>
  <li><b>Wash count / MSU:</b> wash count = changeover-id-{meta['msu_wash_ids']} event count per line; MSU = Σ (constrained
      qty × SU factor ÷ 1000) over the line's materials. SU factor from the local SUF workbook (primary;
      {meta['msu_n_suf']} materials) with Databricks <code>su_factor_for_buom</code> fallback ({meta['msu_n_dbx']})
      and manual input ({meta.get('msu_n_manual', 0)}).</li>
  {cov_scope_item}
  <li><b>Scenario identity:</b> output filtered by <code>run_id = {meta['run_id_db']}</code>; config by
      <code>config_name = {meta['config_name']}</code>.</li>
</ul>
</div>

<h2>KPI 1 — TOF by category × production line × month</h2>
<div class="legend">TOF shading:
  <span class="swatch" style="background:#bbf7d0"></span>≥99%
  <span class="swatch" style="background:#dcfce7"></span>≥95%
  <span class="swatch" style="background:#fef9c3"></span>≥90%
  <span class="swatch" style="background:#fed7aa"></span>≥80%
  <span class="swatch" style="background:#fecaca"></span>&lt;80%
</div>
{cfr_pivot_html(cfr_main, cfr_line_total)}
<div class="chart">{img('cfr_month_cat', 'TOF by month by category')}</div>
<div class="chart">{img('cfr_line_cat', 'TOF by line by category')}</div>
<h3>TOF by category (full period)</h3>
<ul>{cat_rows}</ul>
<p class="small"><b>Data note.</b> {meta['cat_inferred_share']} of order quantity comes from new VMR materials
("21&hellip;" codes) not yet in the HANA SKU master; their category is inferred from their single-category
delegate line ({meta['cat_inferred_n']} materials, all Hair on lines XQHG/XQHK). A residual
{meta['cat_unmapped_share']} ({meta['cat_unmapped_n']} materials on the mixed line XQHD) stays
<i>Unmapped</i> to avoid a wrong split. See <code>extracts/category_resolution.csv</code> for the per-material source.</p>

<h2>KPI 2 — Changeover count by production line × month × changeover id</h2>
{count_pivot_html(changeover, 'changeover_count', money=False)}
<div class="chart">{img('co_count_month', 'Changeover count by month')}</div>

<h2>KPI 3 — Changeover cost by production line × month × changeover id</h2>
{count_pivot_html(changeover, 'changeover_cost', money=True)}
<div class="chart">{img('co_cost_line', 'Changeover cost by line')}</div>
<div class="chart">{img('co_cost_month', 'Changeover cost by month')}</div>
<h3>Changeover by id (full period)</h3>
<ul>{id_rows}</ul>
<p class="small">Changeover ids with zero cost (typically id 0 and 1) are no-cost / same-family transitions; cost concentrates
on the higher ids. Per-id definitions are in <code>ChangeoverDefs</code> of the workbook.</p>

<h2>KPI 4 — Month-end DFC by category × production line × month</h2>
<div class="legend">DFC = DC ending inventory ÷ avg daily forward {meta['dfc_forward_days']}-day demand forecast. Shading:
  <span class="swatch" style="background:#fecaca"></span>&lt;7d (shortage risk)
  <span class="swatch" style="background:#fed7aa"></span>7–14d
  <span class="swatch" style="background:#dcfce7"></span>14–45d (healthy)
  <span class="swatch" style="background:#fef9c3"></span>45–75d
  <span class="swatch" style="background:#bfdbfe"></span>&gt;75d (excess)
  &nbsp;&nbsp;<sup>*</sup> shortened forward window.
</div>
{dfc_pivot_html(dfc_main, partial_months)}
<div class="chart">{img('dfc_month_cat', 'Month-end DFC by category')}</div>
<div class="chart">{img('dfc_line_month', 'Month-end DFC by line')}</div>
<h3>DFC by category (full-window months avg)</h3>
<ul>{dfc_cat_rows}</ul>
<h3>DFC by production line (full-window months avg)</h3>
<ul>{dfc_line_rows}</ul>
<p class="small"><b>Scope note.</b> DFC counts only downstream DC inventory ({meta['dfc_dc_locations_n']} locations); plant 1864
stock ({meta['dfc_plant_demand_share']} of forecast demand sits at the plant) is excluded so coverage reflects
customer-facing positions. Months {', '.join(partial_months) or 'none'} are marked <sup>*</sup> because the forward
demand window is truncated by the forecast horizon ({meta['dfc_horizon_end']}); headline averages above use full-window
months only. Cells shown as &mdash; have no forward demand in the window (coverage undefined). The few materials carrying DC stock with no DC
forecast (an <i>Unmapped</i> category, ~59k units) are kept in <code>extracts/dfc_by_category_line_month.csv</code> but
omitted from the pivot/charts as they carry no coverage signal. Per-material detail is in
<code>extracts/dfc_material_month_base.csv</code>.</p>

<h2>KPI 5 — Production total time (production + changeover) by category × production line × month</h2>
<div class="legend">Production total time (hr) = <b>production time</b> (constrained planned qty ÷ production rate, unit/hr)
  + <b>changeover time</b> (changeover id × count × per-event time, hr). Plant 1864; months from
  <code>production_plan_date</code> ({', '.join(meta['ptt_months'])}).</div>
<div class="kpi-cards">
  <div class="card"><div class="v">{hrs(f['ptt_prod'])}</div><div class="l">Production time (hr)</div></div>
  <div class="card"><div class="v">{hrs(f['ptt_co'])}</div><div class="l">Changeover time (hr)</div></div>
  <div class="card"><div class="v">{hrs(f['ptt_tot'])}</div><div class="l">Total time (hr)</div></div>
  <div class="card"><div class="v">{f['ptt_co_share']:.1f}%</div><div class="l">Changeover share of total</div></div>
</div>
<h3>Total time (hr) by category × line × month</h3>
{ptt_pivot_html(ptt_main)}
<div class="chart">{img('ptt_month_split', 'Production total time by month, production vs changeover')}</div>
<div class="chart">{img('ptt_line_cat', 'Production total time by line, stacked by category')}</div>
<h3>Production vs changeover by line (full period)</h3>
{ptt_line_split_html(ptt_line_total)}
<h3>Total time by category (full period)</h3>
<ul>{ptt_cat_rows}</ul>
<p class="small"><b>Scope note.</b> Production time uses the <b>constrained</b> planned quantity
(<code>con_planned_qty</code>, which equals produced qty in this run) divided by the per-material production rate
(<code>prd_rate</code>, unit/hr); rate coverage is 288/288 production materials. Changeover hours come straight from
the simulation log (<code>module4_output_changeoverlog.time</code>, already equal to id × count × per-event time).
Changeover is recorded per line only, so it is allocated to categories <b>pro-rata by production hours</b> within each
line × month; the only mixed line is <b>{', '.join(meta['ptt_mixed_lines']) or 'none'}</b> (Hair+PCC), so every other line's
changeover maps to a single category unambiguously. All production sits at plant 1864 and is booked on
<code>production_plan_date</code> (Jul–Oct); June and November carry no production rows. Full detail per
material/grain is in <code>extracts/prodtime_*.csv</code>.</p>

<h2>KPI 6 — APQ (avg {apq_title_metric} per production run) by line &amp; by material</h2>
<div class="legend">APQ = {apq_legend_metric} ÷ production-run count. A <b>production run</b> = a plan row that
  starts with a changeover (<code>changeover_id</code> not null); the {num(meta['apq_total_runs'])} runs span
  {apq_span}.</div>
<div class="kpi-cards">
  <div class="card"><div class="v">{apqf(meta['apq_overall'])}</div><div class="l">Overall APQ ({apq_unit})</div></div>
  <div class="card"><div class="v">{num(meta['apq_total_runs'])}</div><div class="l">Production runs</div></div>
  <div class="card"><div class="v">{apq_vol_fmt(apq_vol_value)}</div><div class="l">Total {apq_vol_label}</div></div>
  <div class="card"><div class="v">{num(meta['apq_n_materials'])}</div><div class="l">Materials produced</div></div>
</div>
<h3>APQ by production line</h3>
{apq_line_html(apq_by_line)}
<div class="chart">{img('apq_line', 'APQ by production line')}</div>
<h3>Top materials by constrained qty (with APQ)</h3>
{apq_material_html(apq_by_material, 25)}
<p class="small"><b>Scope note.</b> APQ averages the {apq_metric_long} over the number of production runs.
The denominator counts plan rows whose <code>changeover_id</code> is non-null (each such row is a run that begins with a
changeover); the continuation rows with a null changeover id are excluded so a single run is
not double-counted. The numerator is {apq_numerator}. By
material, APQ uses that material's own run count across the period. A higher APQ means larger, less-frequent batches; a
lower APQ means more, smaller runs (more setups per unit). Full per-material detail is in
<code>extracts/apq_by_material.csv</code>.</p>

<h2>KPI 7 — Wash count / MSU by production line</h2>
<div class="legend">Wash count = changeover events with <b>changeover id {meta['msu_wash_ids']}</b> (the wash transitions in this run).
  MSU = Σ (constrained qty × SU factor ÷ 1000) over the line's materials. <b>washes / MSU</b> = wash count ÷ MSU
  (lower is better — fewer washes per unit of volume produced).</div>
<div class="kpi-cards">
  <div class="card"><div class="v">{num(meta['msu_total_wash'])}</div><div class="l">Total wash changeovers (id {meta['msu_wash_ids']})</div></div>
  <div class="card"><div class="v">{msu_fmt(meta['msu_total_msu'])}</div><div class="l">Total MSU produced</div></div>
  <div class="card"><div class="v">{ratio3(meta['msu_wash_per_msu_overall'])}</div><div class="l">Overall washes / MSU</div></div>
  <div class="card"><div class="v">{num(meta['msu_n_suf'])}+{num(meta['msu_n_dbx'])}+{num(meta.get('msu_n_manual', 0))}</div><div class="l">SU factors: SUF workbook + Databricks + manual</div></div>
</div>
{msu_line_html(msu_by_line)}
<div class="chart">{img('msu_line', 'Wash count vs MSU by line')}</div>
<h3>Washes per MSU by line (most to least intensive)</h3>
<ul>{msu_wpm_rows}</ul>
<p class="small"><b>Scope note.</b> Wash count sums the simulated changeover events with id in {meta['msu_wash_ids']} from
<code>module4_output_changeoverlog</code> per line. MSU converts production volume to MSU units via the per-material SU
factor: the local SUF workbook (<code>{meta['msu_suf_path']}</code>) is the primary source ({meta['msu_n_suf']} materials,
including the new VMR “21…” codes that are absent from the SKU master), Databricks
<code>su_factor_for_buom</code> is the fallback ({meta['msu_n_dbx']} materials), and a manual-input workbook fills the
remainder ({meta.get('msu_n_manual', 0)} materials); coverage is {meta['msu_n_suf'] + meta['msu_n_dbx'] + meta.get('msu_n_manual', 0)}/{meta['apq_n_materials']}
with {meta['msu_n_missing']} missing. Where both sources overlap they agree to within ~0.05%. Per-material MSU detail is in
<code>extracts/msu_by_material.csv</code>.</p>
{cov_section}
<h2>Key Findings</h2>
<div class="band"><ul>
  <li>Overall baseline TOF is <b>{pct(f['overall'])}</b>. Category split — {''.join(f"{_html.escape(str(c))}: {pct(r['cfr'])}; " for c, r in f['cat'].iterrows())}</li>
  <li>Line <b>{_html.escape(str(f['worst_line']))}</b> has the lowest service ({pct(f['worst_line_v'])}); month
      <b>{f['worst_mon']}</b> is the weakest period ({pct(f['worst_mon_v'])}).</li>
  <li>Month-end DC coverage averages <b>{days(f['dfc_overall'])} days</b>; lowest-coverage lines are
      {''.join(f"{_html.escape(str(l))} ({days(v)}d); " for l, v in f['dfc_line'].head(3).items())}
      Read service (TOF) and coverage (DFC) together: thin DFC on a low-TOF line points to a true supply gap.</li>
  <li>Changeover cost concentrates on: {''.join(f"{_html.escape(str(l))} ({num(v)}); " for l, v in f['cost_line'].head(3).items())}</li>
  <li>The dominant cost driver is the high-cost changeover id(s); review the count×unit-cost mix per line for reduction levers.</li>
  <li>Production total time is <b>{hrs(f['ptt_tot'])} hr</b> across {len(meta['ptt_months'])} producing months, of which
      <b>{f['ptt_co_share']:.1f}%</b> ({hrs(f['ptt_co'])} hr) is changeover. Busiest line
      <b>{_html.escape(str(f['ptt_busy_line']))}</b> ({hrs(f['ptt_busy_line_v'])} hr); busiest month
      <b>{f['ptt_busy_mon']}</b> ({hrs(f['ptt_busy_mon_v'])} hr). Lines with the highest changeover share of line time:
      {''.join(f"{_html.escape(str(l))} ({v:.0f}%); " for l, v in f['ptt_co_share_line'].head(3).items())}</li>
  <li>APQ averages <b>{apqf(f['apq_overall'])} {apq_unit}</b>. Largest batches:
      {''.join(f"{_html.escape(str(l))} ({apqf(v)}); " for l, v in f['apq_line'].head(3).items())}
      smallest: {''.join(f"{_html.escape(str(l))} ({apqf(v)}); " for l, v in f['apq_line'].tail(2).items())}
      Lower APQ lines run more, smaller batches (more setups per unit).</li>
  <li>Wash intensity is <b>{ratio3(f['msu_overall'])}</b> washes/MSU overall ({num(f['msu_wash'])} wash changeovers /
      {msu_fmt(f['msu_msu'])} MSU). Most wash-intensive:
      {''.join(f"{_html.escape(str(l))} ({ratio3(v)}); " for l, v in f['msu_wpm'].head(3).items())}
      these lines change formula most often relative to volume — prime targets for sequencing/family-batching.</li>
  {cov_finding}
</ul></div>

<h2>Recommendations</h2>
<div class="band"><ul>
  <li>Use this baseline strictly as the reference for the upcoming AO-ratio and advance-lead-time what-ifs; compare the same
      four views to isolate service vs. inventory vs. changeover trade-offs.</li>
  <li>Investigate line <b>{_html.escape(str(f['worst_line']))}</b> and month <b>{f['worst_mon']}</b> for the service gap
      (capacity, changeover load, or supply timing); cross-check its month-end DFC to tell a demand miss from a stock-out.</li>
  <li>Where DFC runs high while TOF is healthy, there is inventory-reduction headroom; where DFC is thin
      (&lt;1–2 weeks), protect service before pushing AO ratio up.</li>
  <li>Target changeover-cost reduction on the top cost lines via sequencing/family-batching to cut high-cost changeover ids.</li>
</ul></div>

<h2>Appendix — Definitions &amp; Calculation Notes</h2>
<div class="band small">
<ul>
  <li><b>TOF</b> = Σ shipment_qty / Σ order_qty (ratio of sums; numerator and denominator aggregated separately to each
      month × category × line, then divided). Order log (<code>module1_output_orderlog</code>) is deduplicated on the
      7-field natural key (simulation_date, date, material, location, demand_type, advance_days, quantity) before summing;
      shipment log (<code>module1_output_shipmentlog</code>) needs no dedup. Month = business <code>date</code> as YYYY-MM.</li>
  <li><b>Production line</b> = <code>delegate_line</code> from <code>cfg_m4_materiallocationlinecfg</code>, joined on material
      (each material has exactly one delegate line). Demand at all DC locations is attributed to the producing line.
      Unmapped materials: {unmapped_line}.</li>
  <li><b>Changeover count &amp; cost</b> = Σ count / Σ cost from <code>module4_output_changeoverlog</code>, grouped by
      month × line × changeover_type. The <code>changeover_type</code> column equals the <code>changeover_id</code> in
      <code>cfg_m4_changeoverdefinition</code>.</li>
  <li><b>Month-end DFC</b> (days forward coverage) = month-end DC <code>ending_soh</code> (from
      <code>module5_output_stockonhandlog</code>, summed over the {meta['dfc_dc_locations_n']} DC locations, plant 1864
      excluded) ÷ average daily forward demand. Forward demand uses the weekly demand forecast
      (<code>cfg_m1_demandforecast</code>) at DC locations, spread to a daily rate and summed over the next
      {meta['dfc_forward_days']} days (clamped to the forecast horizon {meta['dfc_horizon_end']}). DFC is a ratio of sums
      at each rollup (Σ inventory ÷ Σ daily-demand-rate), so groups are demand-weighted. Month-end snapshot dates:
      {'; '.join(f"{m} = {d}" for m, d in meta['dfc_month_end'].items())}. Months with a shortened window:
      {', '.join(partial_months) or 'none'}. Sim day 1 = {meta['dfc_sim_start']} (forecast week 1).</li>
  <li><b>Category</b> = <code>category_en</code> from Databricks <code>cdl_ps_hana_prd.sl.ps_psc_sku_master</code>
      (material_num &rarr; category_en). {cat_note} New VMR materials absent from the master inherit the category of
      their <code>delegate_line</code> when that line is single-category among mapped materials (flagged
      <code>category_source = line-inferred</code> in <code>extracts/category_resolution.csv</code>); mixed lines stay
      Unmapped. Resolution by order qty: databricks {meta['cat_databricks_share']}, line-inferred
      {meta['cat_inferred_share']}, unmapped {meta['cat_unmapped_share']}. Materials still without a category: {unmapped_cat}.</li>
  <li><b>Production total time</b> (hr) = production time + changeover time. <i>Production time</i> =
      <code>con_planned_qty</code> (constrained planned production, from <code>module4_output_productionplan</code>) ÷
      <code>prd_rate</code> (production rate, unit/hr, from <code>cfg_m4_materiallocationlinecfg</code>), computed per
      material then summed. <i>Changeover time</i> = <code>module4_output_changeoverlog.time</code> (already = changeover
      id × count × per-event changeover time, hr), summed per line. Month = <code>production_plan_date</code> (the day the
      line runs); production spans {', '.join(meta['ptt_months'])} at plant 1864. Changeover is line-level and is
      allocated to category pro-rata by production hours within line × month (only the mixed line
      {', '.join(meta['ptt_mixed_lines']) or 'none'} is split). Extracts: <code>prodtime_*.csv</code>.</li>
  <li><b>APQ</b> (avg {apq_title_metric} per production run) = {apq_def_numer} ÷ production-run count.
      A <i>production run</i> = a <code>module4_output_productionplan</code> row whose <code>changeover_id</code> is non-null
      (the row begins with a changeover); continuation rows with a null changeover id are not counted as separate runs.
      Reported overall, by production line, and by material (each material over its own run count). Run basis:
      {meta['apq_run_basis']}; qty basis: {meta['apq_qty_basis']}. Extracts: <code>apq_by_line.csv</code>,
      <code>apq_by_material.csv</code>.</li>
  <li><b>Wash count / MSU</b> = wash count ÷ MSU per line. <i>Wash count</i> = changeover events with id in
      {meta['msu_wash_ids']} from <code>module4_output_changeoverlog</code>. <i>MSU</i> = Σ (<code>con_planned_qty</code> ×
      SU factor ÷ 1000) over the line's materials. SU factor source: SUF workbook
      <code>{meta['msu_suf_path']}</code> (primary, {meta['msu_n_suf']} materials) with Databricks
      <code>su_factor_for_buom</code> fallback ({meta['msu_n_dbx']} materials) and manual input
      ({meta.get('msu_n_manual', 0)} materials); {meta['msu_n_missing']} missing. Overlapping
      sources agree to ~0.05%. Extracts: <code>wash_msu_by_line.csv</code>, <code>msu_by_material.csv</code>.</li>
  {cov_appendix}
  <li><b>Scope filters:</b> run_id = <code>{meta['run_id_db']}</code>; config_name = <code>{meta['config_name']}</code>.</li>
</ul>
<p>Extract CSVs backing every table are in <code>extracts/</code>; the consolidated workbook is the companion <code>.xlsx</code>.</p>
</div>

</div></body></html>"""
    html_path.write_text(doc, encoding="utf-8")


# --- run.md + LATEST.md ---------------------------------------------------------
def write_run_md(run_dir: Path, meta: dict):
    if bool(meta.get("apq_is_msu", False)):
        apq_kpi6 = "APQ (avg MSU produced per production run) by production line and by material"
        apq_md = (f"- APQ = sum(con_planned_qty x SU factor / 1000) = MSU / production-run count. "
                  f"A run = a module4_output_productionplan row with non-null changeover_id (begins with a "
                  f"changeover); null-changeover continuation rows are not counted. "
                  f"Overall {meta['apq_overall']:.3f} MSU/run = {meta['apq_total_msu']:.1f} MSU / "
                  f"{meta['apq_total_runs']}. By line and by material.")
    else:
        apq_kpi6 = "APQ (avg constrained planned qty per production run) by production line and by material"
        apq_md = (f"- APQ = sum(con_planned_qty) / production-run count. A run = a module4_output_productionplan "
                  f"row with non-null changeover_id (begins with a changeover); null-changeover continuation rows "
                  f"are not counted. Overall {meta['apq_overall']:.1f} units/run = "
                  f"{meta['apq_total_con_qty']:.0f} / {meta['apq_total_runs']}. By line and by material.")
    cov_kpi = ("\n8. MOQ / APQ coverage (days) by SKU / production line / category (JASO 6.29-11.1)"
               if meta.get("cov_has") else "")
    cov_method = ("" if not meta.get("cov_has") else
        f"\n- MOQ/APQ coverage (days) = quantity(MSU) / (wk1-18 forecast MSU / {meta.get('cov_days', 126)}). "
        f"MOQ = current min_batch x SU/1000; APQ = simulated avg MSU per run. wk1-18 forecast summed over each "
        f"SKU's 1864 sourcing sub-network (cfg_global_network). By line/category = demand-weighted ratio of sums. "
        f"Overall MOQ cov {meta.get('cov_moq_overall')} d, APQ cov {meta.get('cov_apq_overall')} d; "
        f"{meta.get('cov_n_sku')} SKUs (APQ on {meta.get('cov_n_sku_apq')}).")
    txt = f"""# Analysis run {meta['analysis_run_id']}

- **Level:** scenario ({meta['scenario']})
- **Project:** {meta['project']}
- **DB run_id:** {meta['run_id_db']}
- **config_name:** {meta['config_name']}
- **Generated:** {meta['generated']}

## KPIs produced
1. TOF by category x production line x month
2. Changeover count by production line x month x changeover id
3. Changeover cost by production line x month x changeover id
4. Month-end DFC (days forward coverage) by category x production line x month
5. Production total time (production + changeover, hr) by category x production line x month
6. {apq_kpi6}
7. Wash count / MSU by production line (wash = changeover id 2/3; MSU = qty x SU factor / 1000){cov_kpi}

## Scope & method
- Period 2026-06-29 → 2026-10-31 (reported through end of October; Nov tail excluded), XQ plant 1864, categories HC (Hair) / PCC.
- TOF = Σ shipment / Σ order (ratio of sums); order log deduped on 7-field key; month = business date.
- Production line = delegate_line from cfg_m4_materiallocationlinecfg (material level; demand at all DCs attributed to producing line).
- Changeover count/cost from module4_output_changeoverlog grouped by month × line × changeover_type (= changeover id).
- DFC = month-end DC ending_soh (module5_output_stockonhandlog, {meta['dfc_dc_locations_n']} DC locations, plant 1864 excluded) / avg daily forward {meta['dfc_forward_days']}-day demand forecast (cfg_m1_demandforecast at DCs). Ratio of sums per rollup.
- Production total time = production time + changeover time (hr). Production time = con_planned_qty (module4_output_productionplan) / prd_rate (cfg_m4_materiallocationlinecfg, unit/hr) per material then summed; changeover time = module4_output_changeoverlog.time summed; month = production_plan_date ({', '.join(meta['ptt_months'])}). Changeover allocated to category pro-rata by production hours within line x month (mixed line: {', '.join(meta['ptt_mixed_lines']) or 'none'}).
{apq_md}
- Wash count / MSU = wash count / MSU per line. Wash count = changeover events with id in {meta['msu_wash_ids']} (module4_output_changeoverlog). MSU = sum(con_planned_qty x SU factor / 1000) over the line's materials. Overall {meta['msu_wash_per_msu_overall']:.3f} washes/MSU = {meta['msu_total_wash']} / {meta['msu_total_msu']:.1f} MSU.{cov_method}

## Data lineage
- Category source: {meta['category_source']} — {meta['category_note'] or 'fresh Databricks pull (category_en)'}.
- Category coverage: databricks {meta['category_coverage']}; enriched {meta['category_coverage_enriched']}.
- Category resolution by order qty: databricks {meta['cat_databricks_share']}, line-inferred {meta['cat_inferred_share']}, unmapped {meta['cat_unmapped_share']}.
- Line-inferred materials ({meta['cat_inferred_n']}): new VMR codes on single-category lines XQHG/XQHK → Hair.
- DFC forward window {meta['dfc_forward_days']}d; forecast horizon ends {meta['dfc_horizon_end']}; partial-window months: {', '.join(meta['dfc_partial_months']) or 'none'}.
- SU factor source: SUF workbook {meta['msu_suf_path']} (primary, {meta['msu_n_suf']} materials) + Databricks su_factor_for_buom fallback ({meta['msu_n_dbx']} materials); {meta['msu_n_missing']} missing. Overlapping sources agree to ~0.05%.
- Unmapped-line materials: {', '.join(meta['unmapped_line_materials']) or 'none'}.
- Unmapped-category materials: {', '.join(meta['unmapped_category_materials']) or 'none'}.

## Outputs
- `{meta['project']}_result_summary_{meta['analysis_run_id']}.xlsx`
- `analysis.html`
- `extracts/` (per-KPI CSVs + base tables)
"""
    (run_dir / "run.md").write_text(txt, encoding="utf-8")


def update_latest(analysis_root: Path, run_id: str, meta: dict):
    latest = analysis_root / "LATEST.md"
    cov_kpi = ", MOQ/APQ coverage by SKU/line/category" if meta.get("cov_has") else ""
    entry = (
        f"- **{run_id}** - scenario `{meta['scenario']}`, run_id `{meta['run_id_db']}` "
        f"| KPIs: TOF by category x line x month, changeover count & cost by line x month x id, "
        f"month-end DFC by category x line x month, production total time by category x line x month, "
        f"APQ by line & material, wash count / MSU by line{cov_kpi} "
        f"| category: databricks {meta['cat_databricks_share']} + line-inferred {meta['cat_inferred_share']} "
        f"+ unmapped {meta['cat_unmapped_share']} "
        f"| generated {meta['generated']}\n"
    )
    if latest.exists():
        old = latest.read_text(encoding="utf-8")
        # replace current pointer line, keep the log below
        lines = old.splitlines()
        log_idx = next((i for i, l in enumerate(lines) if l.strip().lower().startswith("## run log")), None)
        log_tail = "\n".join(lines[log_idx + 1:]) if log_idx is not None else ""
        new = f"current: {run_id}\n\n## Run log\n{entry}{log_tail}\n"
    else:
        new = f"current: {run_id}\n\n## Run log\n{entry}"
    latest.write_text(new, encoding="utf-8")
