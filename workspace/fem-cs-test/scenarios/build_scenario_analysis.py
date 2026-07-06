"""Build a FEM scenario KPI analysis deliverable (parameterized).

Project : fem-cs-test
Scenario: s1 | s2 | s3 -fem-hpfd-network-0386 (selected via CLI arg; default s1)
run_id  : the confirmed run_id for that scenario in DB fem_test

Scope (user-defined):
  locations (DC): C816, A673, A672, C810, A715, A716, A668, A680
  months        : 2026-02 (Feb), 2026-03 (Mar)

KPIs (changeover COST / KPI 3 intentionally skipped per user request):
  1. ToF  (= shipment_qty / order_qty)  by FC (location) x month, and overall by month
  2. Change-over count + time           by HPFD line x month x changeover id
  4. Month-end DFC (days fwd coverage)  by DC x month, and overall by month
  5. Production total time (prod + CO)  by HPFD line x month
  6. APQ (avg MSU produced / run)       overall, by line (HPFD), by material
  7. Change-over deep dive              PKG CO/MSU and CONV CO/MSU by production line

Terminology
  FGC line12 = the HPFD production line product set (all sim materials are HPFD-line)
  FC         = the customer DC location
  MSU        = con_planned_qty * su_factor / 1000   (su_factor from Databricks ps_psc_sku_master)

Run from repo root with the venv interpreter (arg = s1 | s2 | s3):
  .\\.venv\\Scripts\\python.exe .\\workspace\\fem-cs-test\\scenarios\\build_scenario_analysis.py s1
"""
from __future__ import annotations

import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- identity / scope -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
PROJECT = "fem-cs-test"

# Scenario identity is selected by CLI arg: s1 | s2 | s3 (default s1).
# Each maps to its confirmed simulation run_id in DB fem_test (see each
# scenario's results/run_id.md). config_name == scenario folder name.
_SCEN = (sys.argv[1].lower() if len(sys.argv) > 1 else "s1")
_RUN_IDS = {
    "s1": "db_s1-fem-hpfd-network-0386_20260630_142419",
    "s2": "db_s2-fem-hpfd-network-0386_20260630_142529",
    "s3": "db_s3-fem-hpfd-network-0386_20260630_142620",
}
if _SCEN not in _RUN_IDS:
    raise SystemExit(f"unknown scenario '{_SCEN}'; expected one of {sorted(_RUN_IDS)}")
LABEL = _SCEN.upper()                                   # S1 / S2 / S3
SCENARIO = f"{_SCEN}-fem-hpfd-network-0386"
RUN_ID = _RUN_IDS[_SCEN]
CONFIG_NAME = SCENARIO
DBNAME = "fem_test"
LINE = "HPFD"

SCOPE_DCS = ["A668", "A672", "A673", "A680", "A715", "A716", "C810", "C816"]
MONTHS = ["2026-02", "2026-03"]
MONTH_LABEL = {"2026-02": "Feb 2026", "2026-03": "Mar 2026"}

# DFC parameters
SIM_START = pd.Timestamp("2026-02-02")   # sim day 1 == demand-forecast week 1 start
DFC_FORWARD_DAYS = 30

# KPI7 changeover-id classification (literal per user spec; "2-1" has no events in
# this run, "2-11" does -> applied as given, 2-1 contributes 0)
PKG_IDS = {"1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3"}
CONV_IDS = {"2-1", "2-11", "2-2", "2-3", "3"}

RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M") + f"-fem-{_SCEN}-kpi"
ANALYSIS_ROOT = ROOT / "workspace" / PROJECT / "scenarios" / SCENARIO / "analysis"
OUT_DIR = ANALYSIS_ROOT / RUN_TAG
EXTRACTS = OUT_DIR / "extracts"


# --- DB helpers -----------------------------------------------------------------
def pg_connect():
    cfg = yaml.safe_load((ROOT / "config/defaults.yaml").read_text(encoding="utf-8"))["database"]
    return psycopg.connect(
        host=cfg["host"], port=int(cfg["port"]), dbname=DBNAME,
        user=cfg["user"], password=str(cfg["password"]),
        client_encoding="UTF8", connect_timeout=10,
    )


def q(conn, sql: str, params: tuple) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# --- 1. pull raw aggregates -----------------------------------------------------
def pull_postgres():
    with pg_connect() as conn:
        # ToF: order log deduped on the 7-field natural key (same scope both sides)
        orders = q(conn, """
            select to_char(date::date,'YYYY-MM') as month, location,
                   sum(quantity)::float8 as order_qty
            from (select distinct simulation_date, date, material, location,
                         demand_type, advance_days, quantity
                  from module1_output_orderlog
                  where run_id=%s and location = any(%s)
                    and to_char(date::date,'YYYY-MM') = any(%s)) d
            group by 1,2
        """, (RUN_ID, SCOPE_DCS, MONTHS))

        ships = q(conn, """
            select to_char(date::date,'YYYY-MM') as month, location,
                   sum(quantity)::float8 as shipment_qty
            from module1_output_shipmentlog
            where run_id=%s and location = any(%s)
              and to_char(date::date,'YYYY-MM') = any(%s)
            group by 1,2
        """, (RUN_ID, SCOPE_DCS, MONTHS))

        # Changeover: count + time + cost by month x changeover id (line HPFD)
        changeover = q(conn, """
            select to_char(date::date,'YYYY-MM') as month,
                   changeover_type as changeover_id,
                   sum(count)::bigint  as co_count,
                   sum(time)::float8   as co_time,
                   sum(cost)::float8   as co_cost
            from module4_output_changeoverlog
            where run_id=%s and line=%s and to_char(date::date,'YYYY-MM') = any(%s)
            group by 1,2
        """, (RUN_ID, LINE, MONTHS))

        # Production plan: constrained planned qty per material x month (line runs)
        prod_plan = q(conn, """
            select to_char(production_plan_date::date,'YYYY-MM') as month, material,
                   sum(con_planned_qty)::float8 as con_qty
            from module4_output_productionplan
            where run_id=%s and to_char(production_plan_date::date,'YYYY-MM') = any(%s)
            group by 1,2
        """, (RUN_ID, MONTHS))

        prd_rate = q(conn, """
            select material, max(prd_rate)::float8 as prd_rate
            from cfg_m4_materiallocationlinecfg where config_name=%s group by material
        """, (CONFIG_NAME,))

        # APQ / deep-dive: per material constrained qty + run count. A "run" is a
        # production campaign (PO) = a contiguous block of production days; a >1-day
        # gap starts a new run. This equals the change-over count for every material
        # that begins each campaign with a change-over, and additionally captures the
        # line's base product (mounted at sim start, produced without a change-over).
        prod_runs = q(conn, """
            with daily as (
                select material, production_plan_date::date as d,
                       sum(con_planned_qty)::float8 as qty,
                       count(*) filter (where changeover_id is not null)::int as co_cnt
                from module4_output_productionplan
                where run_id=%s and to_char(production_plan_date::date,'YYYY-MM') = any(%s)
                group by material, production_plan_date::date
            ),
            marked as (
                select material, d, qty, co_cnt,
                       case when (d - lag(d) over (partition by material order by d)) <= 1
                            then 0 else 1 end as is_new
                from daily
            )
            select material,
                   sum(qty)::float8    as con_qty,
                   sum(is_new)::bigint as runs,
                   sum(co_cnt)::bigint as co_runs
            from marked
            group by material
        """, (RUN_ID, MONTHS))

        # DFC: month-end ending_soh at scope DCs, per location x material x month
        stock_me = q(conn, """
            with me as (
              select to_char(date::date,'YYYY-MM') ym, max(date::date) med
              from module5_output_stockonhandlog
              where run_id=%s and location = any(%s)
                and to_char(date::date,'YYYY-MM') = any(%s)
              group by 1
            )
            select me.ym as month, s.location, s.material,
                   sum(s.ending_soh)::float8 as ending_inv, me.med as month_end_date
            from module5_output_stockonhandlog s
            join me on me.med = s.date::date
            where s.run_id=%s and s.location = any(%s)
            group by me.ym, s.location, s.material, me.med
        """, (RUN_ID, SCOPE_DCS, MONTHS, RUN_ID, SCOPE_DCS))

        # Weekly DC demand forecast per location x material
        fcst_weekly = q(conn, """
            select week, location, material, sum(quantity)::float8 as fcst_qty
            from cfg_m1_demandforecast
            where config_name=%s and location = any(%s)
            group by week, location, material
        """, (CONFIG_NAME, SCOPE_DCS))

        # the HPFD material universe (for SU factor lookup)
        hpfd_mats_df = q(conn, """
            select distinct material from cfg_m4_materiallocationlinecfg
            where config_name=%s and delegate_line=%s
        """, (CONFIG_NAME, LINE))

        # distinct materials actually ordered at the scope DCs in the scope months
        ordered_mats_df = q(conn, """
            select count(distinct material)::int as n
            from module1_output_orderlog
            where run_id=%s and location = any(%s)
              and to_char(date::date,'YYYY-MM') = any(%s)
        """, (RUN_ID, SCOPE_DCS, MONTHS))

    for df in (orders, ships, changeover, prod_plan, prd_rate, prod_runs,
               stock_me, fcst_weekly, hpfd_mats_df):
        if "material" in df.columns:
            df["material"] = df["material"].astype(str)
    n_ordered = int(ordered_mats_df["n"].iloc[0]) if len(ordered_mats_df) else 0
    return (orders, ships, changeover, prod_plan, prd_rate, prod_runs,
            stock_me, fcst_weekly, sorted(hpfd_mats_df["material"].tolist()), n_ordered)


# --- 2. SU factor (Databricks ps_psc_sku_master.su_factor_for_buom) -------------
def get_su_factor(materials: list[str]):
    sys.path.insert(0, str(ROOT / "workspace" / "xq-vmr-to-production-202606" / "scripts"))
    factor: dict[str, float] = {}
    err = ""
    try:
        from category_mapping import fetch_su_factor
        dbx = fetch_su_factor(materials)
        for _, r in dbx.iterrows():
            if pd.notna(r["su_factor"]):
                factor[str(r["material"])] = float(r["su_factor"])
    except Exception as exc:  # noqa: BLE001 - disclose, leave missing
        err = str(exc)
    missing = sorted(m for m in materials if m not in factor)
    info = {"n_dbx": len(factor), "n_missing": len(missing), "missing": missing,
            "error": err, "source": "databricks ps_psc_sku_master.su_factor_for_buom"}
    return factor, info


# --- 3. KPI 1: ToF --------------------------------------------------------------
def compute_tof(orders, ships):
    base = orders.merge(ships, on=["month", "location"], how="outer")
    base["order_qty"] = base["order_qty"].fillna(0.0)
    base["shipment_qty"] = base["shipment_qty"].fillna(0.0)

    def rollup(keys):
        g = base.groupby(keys, as_index=False).agg(
            order_qty=("order_qty", "sum"), shipment_qty=("shipment_qty", "sum"))
        g["tof"] = (g["shipment_qty"] / g["order_qty"].replace(0, np.nan)).round(4)
        g["order_qty"] = g["order_qty"].round(1)
        g["shipment_qty"] = g["shipment_qty"].round(1)
        return g

    tof_loc = rollup(["month", "location"]).sort_values(["month", "location"])
    tof_month = rollup(["month"]).sort_values("month")
    # location-pivot for display (rows = location, cols = month)
    piv = tof_loc.pivot(index="location", columns="month", values="tof")
    piv = piv.reindex(SCOPE_DCS)
    return base, tof_loc, tof_month, piv


# --- 4. KPI 2: Changeover count + time ------------------------------------------
def compute_changeover(changeover):
    co = changeover.copy()
    co["changeover_id"] = co["changeover_id"].astype(str)
    by_id = co.groupby(["month", "changeover_id"], as_index=False).agg(
        co_count=("co_count", "sum"), co_time=("co_time", "sum"))
    by_id["co_time"] = by_id["co_time"].round(2)
    by_month = co.groupby("month", as_index=False).agg(
        co_count=("co_count", "sum"), co_time=("co_time", "sum"))
    by_month["co_time"] = by_month["co_time"].round(2)
    cnt_piv = by_id.pivot(index="changeover_id", columns="month", values="co_count").fillna(0).astype(int)
    cnt_piv = cnt_piv.sort_index()
    return by_id, by_month, cnt_piv


# --- 5. KPI 4: Month-end DFC ----------------------------------------------------
def _week_calendar(weeks):
    ws = SIM_START + pd.to_timedelta([(int(w) - 1) * 7 for w in weeks], unit="D")
    return pd.DataFrame({"week": list(weeks), "wk_start": ws,
                         "wk_end": ws + pd.Timedelta(days=6)})


def compute_dfc(stock_me, fcst_weekly):
    wk = _week_calendar(sorted(fcst_weekly["week"].unique()))
    horizon_end = wk["wk_end"].max()
    wk_start, wk_end = wk["wk_start"].values, wk["wk_end"].values

    me = (stock_me[["month", "month_end_date"]].drop_duplicates()
          .assign(month_end_date=lambda d: pd.to_datetime(d["month_end_date"])))
    month_end_map = dict(zip(me["month"], me["month_end_date"]))

    win, wd = {}, {}
    for m, d in month_end_map.items():
        ws = d + pd.Timedelta(days=1)
        we = min(d + pd.Timedelta(days=DFC_FORWARD_DAYS), horizon_end)
        win[m], wd[m] = (ws, we), int((we - ws).days + 1) if ws <= we else 0

    # forward demand per (month, location, material) via week/window overlap
    fwd_rows = []
    for m, (ws, we) in win.items():
        if ws > we:
            continue
        end = np.minimum(wk_end, we.to_datetime64())
        start = np.maximum(wk_start, ws.to_datetime64())
        ov = np.clip((end - start) / np.timedelta64(1, "D") + 1.0, 0.0, None)
        wko = wk.assign(overlap=ov)
        wko = wko[wko["overlap"] > 0]
        if wko.empty:
            continue
        fm = fcst_weekly.merge(wko[["week", "overlap"]], on="week", how="inner")
        fm["contrib"] = fm["fcst_qty"] * fm["overlap"] / 7.0
        g = fm.groupby(["location", "material"], as_index=False)["contrib"].sum() \
              .rename(columns={"contrib": "forward_demand"})
        g["month"] = m
        fwd_rows.append(g)
    fwd = (pd.concat(fwd_rows, ignore_index=True) if fwd_rows
           else pd.DataFrame(columns=["location", "material", "forward_demand", "month"]))

    dfc_base = stock_me[["month", "location", "material", "ending_inv"]].merge(
        fwd, on=["month", "location", "material"], how="outer")
    dfc_base["ending_inv"] = dfc_base["ending_inv"].fillna(0.0)
    dfc_base["forward_demand"] = dfc_base["forward_demand"].fillna(0.0)

    def droll(keys):
        g = dfc_base.groupby(keys, as_index=False).agg(
            dc_inv=("ending_inv", "sum"), fwd_demand=("forward_demand", "sum"))
        g["window_days"] = g["month"].map(wd)
        g["avg_daily_demand"] = g["fwd_demand"] / g["window_days"].replace(0, np.nan)
        denom = g["avg_daily_demand"].where(g["avg_daily_demand"] >= 1.0, np.nan)
        g["dfc_days"] = (g["dc_inv"] / denom).round(1)
        g["avg_daily_demand"] = g["avg_daily_demand"].round(1)
        g["dc_inv"] = g["dc_inv"].round(1)
        g["fwd_demand"] = g["fwd_demand"].round(1)
        return g

    dfc_loc = droll(["month", "location"]).sort_values(["month", "location"])
    dfc_month = droll(["month"]).sort_values("month")
    piv = dfc_loc.pivot(index="location", columns="month", values="dfc_days").reindex(SCOPE_DCS)
    info = {"sim_start": str(SIM_START.date()), "forward_days": DFC_FORWARD_DAYS,
            "horizon_end": str(horizon_end.date()),
            "month_end": {m: str(d.date()) for m, d in month_end_map.items()},
            "window_days": {m: int(wd[m]) for m in wd}}
    return dfc_base, dfc_loc, dfc_month, piv, info


# --- 6. KPI 5: Production total time --------------------------------------------
def compute_ptt(prod_plan, prd_rate, changeover):
    rate = dict(zip(prd_rate["material"], prd_rate["prd_rate"]))
    pp = prod_plan.copy()
    pp["prd_rate"] = pp["material"].map(rate)
    pp["prod_hr"] = (pp["con_qty"] / pp["prd_rate"].replace(0, np.nan)).fillna(0.0)
    prod_hr = pp.groupby("month", as_index=False).agg(
        prod_hr=("prod_hr", "sum"), con_qty=("con_qty", "sum"))
    co_hr = changeover.groupby("month", as_index=False)["co_time"].sum() \
        .rename(columns={"co_time": "co_hr"})
    ptt = prod_hr.merge(co_hr, on="month", how="outer")
    for c in ("prod_hr", "con_qty", "co_hr"):
        ptt[c] = ptt[c].fillna(0.0)
    ptt["total_hr"] = ptt["prod_hr"] + ptt["co_hr"]
    ptt["co_share_pct"] = (ptt["co_hr"] / ptt["total_hr"].replace(0, np.nan) * 100).round(1)
    for c in ("prod_hr", "co_hr", "total_hr", "con_qty"):
        ptt[c] = ptt[c].round(1)
    ptt = ptt.sort_values("month")
    return ptt


# --- 7. KPI 6: APQ --------------------------------------------------------------
def compute_apq(prod_runs, su_factor):
    pr = prod_runs.copy()
    pr["con_qty"] = pr["con_qty"].fillna(0.0)
    pr["runs"] = pr["runs"].fillna(0).astype("int64")        # production campaigns (POs)
    pr["co_runs"] = pr["co_runs"].fillna(0).astype("int64")  # of which change-over-started
    pr["su_factor"] = pr["material"].map(su_factor)
    pr["su_source"] = np.where(pr["material"].map(su_factor).notna(), "databricks", "missing")
    pr["msu"] = pr["con_qty"] * pr["su_factor"].fillna(0.0) / 1000.0

    by_material = pr[["material", "su_factor", "su_source", "con_qty", "msu", "runs"]].copy()
    by_material["apq"] = (by_material["msu"] / by_material["runs"].replace(0, np.nan)).round(3)
    by_material["msu"] = by_material["msu"].round(3)
    by_material["con_qty"] = by_material["con_qty"].round(1)
    by_material = by_material.sort_values("msu", ascending=False)

    tot_msu, tot_qty, tot_runs = float(pr["msu"].sum()), float(pr["con_qty"].sum()), int(pr["runs"].sum())
    by_line = pd.DataFrame([{
        "line": LINE, "msu": round(tot_msu, 2), "con_qty": round(tot_qty, 1),
        "runs": tot_runs, "apq": round(tot_msu / tot_runs, 3) if tot_runs else np.nan}])
    # The line's base product is produced in campaigns that carry NO logged
    # change-over (it is already mounted at sim start). Those campaigns are still
    # real runs (counted via the contiguous-day rule) but have a high MSU/run, so
    # disclose them and report the change-over-started runs separately.
    base = pr[(pr["runs"] > 0) & (pr["co_runs"] == 0)]
    base_msu = float(base["msu"].sum())
    co_runs_total = int(pr["co_runs"].sum())
    base_runs = int(tot_runs - co_runs_total)
    info = {"total_msu": round(tot_msu, 2), "total_con_qty": round(tot_qty, 1),
            "total_runs": tot_runs, "co_runs": co_runs_total, "base_runs": base_runs,
            "apq_overall": round(tot_msu / tot_runs, 3) if tot_runs else float("nan"),
            "n_materials": int(pr["material"].nunique()),
            "n_missing_su": int(pr["su_source"].eq("missing").sum()),
            "base_materials": base["material"].tolist(),
            "base_msu": round(base_msu, 2),
            "base_msu_pct": round(base_msu / tot_msu * 100, 1) if tot_msu else 0.0,
            "apq_ex_base": round((tot_msu - base_msu) / co_runs_total, 3) if co_runs_total else float("nan"),
            "run_basis": "contiguous production campaign (PO); a >1-day gap starts a new run",
            "qty_basis": "MSU = con_planned_qty * su_factor / 1000 (Databricks su_factor)"}
    return by_line, by_material, info


# --- 8. KPI 7: Changeover deep dive (PKG / CONV CO per MSU) ----------------------
def compute_deepdive(changeover, prod_runs, su_factor):
    pr = prod_runs.copy()
    pr["su_factor"] = pr["material"].map(su_factor)
    pr["msu"] = pr["con_qty"].fillna(0.0) * pr["su_factor"].fillna(0.0) / 1000.0
    total_msu = float(pr["msu"].sum())

    co = changeover.copy()
    co["changeover_id"] = co["changeover_id"].astype(str)
    pkg_count = int(co.loc[co["changeover_id"].isin(PKG_IDS), "co_count"].sum())
    conv_count = int(co.loc[co["changeover_id"].isin(CONV_IDS), "co_count"].sum())

    out = pd.DataFrame([{
        "line": LINE,
        "PKG_CO_count": pkg_count,
        "CONV_CO_count": conv_count,
        "MSU": round(total_msu, 2),
        "PKG_CO_per_MSU": round(pkg_count / total_msu, 4) if total_msu else np.nan,
        "CONV_CO_per_MSU": round(conv_count / total_msu, 4) if total_msu else np.nan,
    }])
    present_ids = set(co["changeover_id"].unique())
    union_ids = PKG_IDS | CONV_IDS
    absent = sorted(union_ids - present_ids)
    extra = sorted(present_ids - union_ids)

    # by month breakdown (transparency)
    rows = []
    for m in MONTHS:
        cm = co[co["month"] == m]
        pkg = int(cm.loc[cm["changeover_id"].isin(PKG_IDS), "co_count"].sum())
        conv = int(cm.loc[cm["changeover_id"].isin(CONV_IDS), "co_count"].sum())
        rows.append({"month": m, "PKG_CO_count": pkg, "CONV_CO_count": conv})
    by_month = pd.DataFrame(rows)

    note_parts = []
    if absent:
        note_parts.append("classified change-over id(s) "
                          + ", ".join(f"'{i}'" for i in absent) + " absent (0 events)")
    if extra:
        note_parts.append("change-over id(s) " + ", ".join(f"'{i}'" for i in extra)
                          + " present but outside PKG/CONV classification")
    note = "; ".join(note_parts) if note_parts else "all classified change-over ids present"
    info = {"pkg_ids": ", ".join(sorted(PKG_IDS)), "conv_ids": ", ".join(sorted(CONV_IDS)),
            "total_msu": round(total_msu, 2), "note": note}
    return out, by_month, info


# --- charts ---------------------------------------------------------------------
COL = {"feb": "#4e79a7", "mar": "#f28e2b", "prod": "#59a14f", "co": "#e15759",
       "pkg": "#4e79a7", "conv": "#f28e2b"}


def fig_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def chart_tof(tof_loc):
    fig, ax = plt.subplots(figsize=(8, 4))
    locs = SCOPE_DCS
    x = np.arange(len(locs))
    w = 0.38
    for i, m in enumerate(MONTHS):
        d = tof_loc[tof_loc["month"] == m].set_index("location")["tof"].reindex(locs)
        ax.bar(x + (i - 0.5) * w, (d * 100).values, w, label=MONTH_LABEL[m],
               color=COL["feb" if i == 0 else "mar"])
    ax.set_xticks(x); ax.set_xticklabels(locs)
    ax.set_ylabel("ToF (%)"); ax.set_ylim(0, 105)
    ax.axhline(100, color="#888", lw=0.8, ls="--")
    ax.set_title("ToF by DC and month"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


def chart_changeover(cnt_piv):
    fig, ax = plt.subplots(figsize=(8, 4))
    ids = list(cnt_piv.index)
    x = np.arange(len(ids)); w = 0.38
    for i, m in enumerate(MONTHS):
        vals = cnt_piv[m].values if m in cnt_piv.columns else np.zeros(len(ids))
        ax.bar(x + (i - 0.5) * w, vals, w, label=MONTH_LABEL[m],
               color=COL["feb" if i == 0 else "mar"])
    ax.set_xticks(x); ax.set_xticklabels(ids)
    ax.set_ylabel("Change-over count"); ax.set_xlabel("changeover id")
    ax.set_title("Change-over count by id and month (HPFD)"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


def chart_dfc(dfc_loc):
    fig, ax = plt.subplots(figsize=(8, 4))
    locs = SCOPE_DCS
    x = np.arange(len(locs)); w = 0.38
    for i, m in enumerate(MONTHS):
        d = dfc_loc[dfc_loc["month"] == m].set_index("location")["dfc_days"].reindex(locs)
        ax.bar(x + (i - 0.5) * w, d.values, w, label=MONTH_LABEL[m],
               color=COL["feb" if i == 0 else "mar"])
    ax.set_xticks(x); ax.set_xticklabels(locs)
    ax.set_ylabel("DFC (days)")
    ax.set_title("Month-end DFC by DC"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


def chart_ptt(ptt):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    months = ptt["month"].tolist()
    x = np.arange(len(months))
    ax.bar(x, ptt["prod_hr"], 0.5, label="production hr", color=COL["prod"])
    ax.bar(x, ptt["co_hr"], 0.5, bottom=ptt["prod_hr"], label="changeover hr", color=COL["co"])
    ax.set_xticks(x); ax.set_xticklabels([MONTH_LABEL[m] for m in months])
    ax.set_ylabel("hours"); ax.set_title("Production total time (HPFD)"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


def chart_apq(apq_material, exclude=()):
    d = apq_material.dropna(subset=["apq"]).copy()
    exc = [str(m) for m in exclude]
    note = ""
    if exc:
        ex = d[d["material"].astype(str).isin(exc)]
        d = d[~d["material"].astype(str).isin(exc)]
        if len(ex):
            note = "  (excl. " + ", ".join(f"{m}={v:.1f}" for m, v in
                                            zip(ex["material"].astype(str), ex["apq"])) + ")"
    d = d.sort_values("apq", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(d))))
    ax.barh(d["material"].astype(str), d["apq"], color=COL["feb"])
    ax.set_xlabel("APQ (MSU / run)"); ax.set_title("APQ by material (HPFD, Feb+Mar)" + note)
    ax.grid(axis="x", alpha=0.3)
    return fig_uri(fig)


def chart_deepdive(deep):
    r = deep.iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["PKG CO / MSU", "CONV CO / MSU"]
    vals = [r["PKG_CO_per_MSU"], r["CONV_CO_per_MSU"]]
    ax.bar(labels, vals, 0.5, color=[COL["pkg"], COL["conv"]])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("CO events per MSU"); ax.set_title("Change-over per MSU (HPFD, lower is better)")
    ax.grid(axis="y", alpha=0.3)
    return fig_uri(fig)


# --- output: Excel + HTML + extracts --------------------------------------------
def df_html(df, fmt=None, pct_cols=()):
    d = df.copy()
    for c in pct_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v*100:.2f}%")
    return d.to_html(index=False, na_rep="", float_format=lambda v: f"{v:,.3f}",
                     border=0, classes="kpi")


def main():
    EXTRACTS.mkdir(parents=True, exist_ok=True)
    print(f"[1/6] pulling raw aggregates from {DBNAME} ...")
    (orders, ships, changeover, prod_plan, prd_rate, prod_runs,
     stock_me, fcst_weekly, hpfd_mats, n_ordered) = pull_postgres()

    print(f"[2/6] fetching SU factor (Databricks) for {len(hpfd_mats)} materials ...")
    su_factor, su_info = get_su_factor(hpfd_mats)
    print(f"      su covered={su_info['n_dbx']} missing={su_info['n_missing']}")

    print("[3/6] computing KPIs ...")
    tof_base, tof_loc, tof_month, tof_piv = compute_tof(orders, ships)
    co_by_id, co_by_month, co_cnt_piv = compute_changeover(changeover)
    dfc_base, dfc_loc, dfc_month, dfc_piv, dfc_info = compute_dfc(stock_me, fcst_weekly)
    ptt = compute_ptt(prod_plan, prd_rate, changeover)
    apq_line, apq_material, apq_info = compute_apq(prod_runs, su_factor)
    deep, deep_month, deep_info = compute_deepdive(changeover, prod_runs, su_factor)

    # --- extracts ---
    print("[4/6] writing extracts + workbook ...")
    tof_loc.to_csv(EXTRACTS / "kpi1_tof_by_location_month.csv", index=False)
    tof_month.to_csv(EXTRACTS / "kpi1_tof_by_month.csv", index=False)
    co_by_id.to_csv(EXTRACTS / "kpi2_changeover_by_month_id.csv", index=False)
    co_by_month.to_csv(EXTRACTS / "kpi2_changeover_by_month.csv", index=False)
    dfc_loc.to_csv(EXTRACTS / "kpi4_dfc_by_location_month.csv", index=False)
    dfc_month.to_csv(EXTRACTS / "kpi4_dfc_by_month.csv", index=False)
    dfc_base.to_csv(EXTRACTS / "kpi4_dfc_material_detail.csv", index=False)
    ptt.to_csv(EXTRACTS / "kpi5_production_time_by_month.csv", index=False)
    apq_line.to_csv(EXTRACTS / "kpi6_apq_by_line.csv", index=False)
    apq_material.to_csv(EXTRACTS / "kpi6_apq_by_material.csv", index=False)
    deep.to_csv(EXTRACTS / "kpi7_deepdive_by_line.csv", index=False)
    deep_month.to_csv(EXTRACTS / "kpi7_deepdive_by_month.csv", index=False)

    # --- Excel workbook ---
    xlsx = OUT_DIR / f"{PROJECT}_result_summary_{RUN_TAG}.xlsx"
    readme = pd.DataFrame({
        "item": ["project", "scenario", "run_id", "database", "line", "scope_locations",
                 "scope_months", "ToF", "Changeover", "DFC", "Production time", "APQ",
                 "Deep dive PKG ids", "Deep dive CONV ids", "MSU", "SU factor source",
                 "skipped"],
        "value": [PROJECT, SCENARIO, RUN_ID, DBNAME, LINE, ", ".join(SCOPE_DCS),
                  ", ".join(MONTH_LABEL[m] for m in MONTHS),
                  "shipment_qty / order_qty (order log deduped on 7-key; shipment log not deduped)",
                  "module4_output_changeoverlog count & time by month x changeover id",
                  "month-end DC ending_soh / avg daily forward demand over next 30 days",
                  "production hr (con_qty/prd_rate) + changeover hr (changeover log time)",
                  "total MSU / production-run count (run = changeover event)",
                  ", ".join(sorted(PKG_IDS)) + "  (2-1 has 0 events this run)",
                  ", ".join(sorted(CONV_IDS)),
                  "con_planned_qty * su_factor / 1000",
                  su_info["source"],
                  "KPI 3 change-over cost (per user request)"],
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="0_README", index=False)
        tof_loc.to_excel(xw, sheet_name="1_ToF_by_loc_month", index=False)
        tof_month.to_excel(xw, sheet_name="1_ToF_by_month", index=False)
        tof_piv.round(4).to_excel(xw, sheet_name="1_ToF_pivot")
        co_by_id.to_excel(xw, sheet_name="2_Changeover_by_id", index=False)
        co_by_month.to_excel(xw, sheet_name="2_Changeover_by_month", index=False)
        co_cnt_piv.to_excel(xw, sheet_name="2_Changeover_count_pivot")
        dfc_loc.to_excel(xw, sheet_name="4_DFC_by_loc_month", index=False)
        dfc_month.to_excel(xw, sheet_name="4_DFC_by_month", index=False)
        ptt.to_excel(xw, sheet_name="5_ProductionTime", index=False)
        apq_line.to_excel(xw, sheet_name="6_APQ_by_line", index=False)
        apq_material.to_excel(xw, sheet_name="6_APQ_by_material", index=False)
        deep.to_excel(xw, sheet_name="7_DeepDive_by_line", index=False)
        deep_month.to_excel(xw, sheet_name="7_DeepDive_by_month", index=False)

    # --- charts ---
    print("[5/6] rendering charts + HTML ...")
    charts = {
        "tof": chart_tof(tof_loc),
        "changeover": chart_changeover(co_cnt_piv),
        "dfc": chart_dfc(dfc_loc),
        "ptt": chart_ptt(ptt),
        "apq": chart_apq(apq_material, apq_info["base_materials"]),
        "deep": chart_deepdive(deep),
    }

    html = build_html(tof_loc, tof_month, tof_piv, co_by_id, co_cnt_piv, co_by_month,
                      dfc_loc, dfc_month, dfc_info, ptt, apq_line, apq_material, apq_info,
                      deep, deep_month, deep_info, su_info, charts, n_ordered)
    (OUT_DIR / "analysis.html").write_text(html, encoding="utf-8")

    print("[6/6] writing run.md + LATEST.md ...")
    write_run_md(tof_month, co_by_month, dfc_month, ptt, apq_info, deep, deep_info,
                 su_info, dfc_info, n_ordered)
    print(f"\nDONE -> {OUT_DIR}")
    print(f"  workbook : {xlsx.name}")
    print(f"  report   : analysis.html")


def build_html(tof_loc, tof_month, tof_piv, co_by_id, co_cnt_piv, co_by_month,
               dfc_loc, dfc_month, dfc_info, ptt, apq_line, apq_material, apq_info,
               deep, deep_month, deep_info, su_info, charts, n_ordered):
    r = deep.iloc[0]
    tof_overall = tof_month.copy()
    css = """
    body{font-family:'Segoe UI',Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}
    .wrap{max-width:1080px;margin:0 auto;padding:28px}
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
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
    .big{font-size:30px;font-weight:700;color:#4e79a7}
    """
    def sect(title, body):
        return f'<div class="card"><h2>{title}</h2>{body}</div>'

    tof_avg = tof_overall.assign(m=tof_overall["month"].map(MONTH_LABEL))
    tof_kv = " &nbsp;|&nbsp; ".join(
        f"<b>{row['m']}</b>: {row['tof']*100:.1f}% ({row['shipment_qty']:,.0f}/{row['order_qty']:,.0f})"
        for _, row in tof_avg.iterrows())

    ptt_disp = ptt.assign(month=ptt["month"].map(MONTH_LABEL))
    dfc_m = dfc_month.assign(month=dfc_month["month"].map(MONTH_LABEL))
    co_m = co_by_month.assign(month=co_by_month["month"].map(MONTH_LABEL))

    h = [f"""<!doctype html><html><head><meta charset="utf-8">
    <title>FEM {LABEL} KPI Analysis</title><style>{css}</style></head><body><div class="wrap">
    <h1>FEM &times; HPFD {LABEL} &mdash; KPI Analysis</h1>
    <div class="sub">Scenario <b>{SCENARIO}</b> &nbsp;|&nbsp; run_id <code>{RUN_ID}</code>
    &nbsp;|&nbsp; DB <b>{DBNAME}</b> &nbsp;|&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}</div>"""]

    h.append(sect("Analysis scope &amp; definitions", f"""
    <div class="kv">
    <p><b>Locations (FC / DC):</b> {', '.join(SCOPE_DCS)} &nbsp; (8 customer DCs)<br>
    <b>Months:</b> {', '.join(MONTH_LABEL[m] for m in MONTHS)} &nbsp;|&nbsp;
    <b>Production line (FGC line12):</b> {LINE} at plant 0386<br>
    <b>ToF</b> = &Sigma; shipment_qty / &Sigma; order_qty (order log deduped on the 7-field key; shipment log not deduped).<br>
    <b>DFC</b> = month-end DC ending inventory &divide; average daily forward demand (weekly forecast spread to daily, next {DFC_FORWARD_DAYS} days).<br>
    <b>MSU</b> = constrained planned qty &times; SU factor &divide; 1000 (SU factor from {su_info['source']}; coverage {su_info['n_dbx']}/{su_info['n_dbx']+su_info['n_missing']}).<br>
    <b>Production run</b> = a production campaign / PO ({apq_info['run_basis']}).</p>
    <div class="note">All {n_ordered} materials ordered at the scope DCs are HPFD-line materials, so the network is fully HPFD-scoped.
    KPI&nbsp;3 (change-over cost) is intentionally skipped per request; the change-over log carries 0 cost in this run.</div>
    </div>"""))

    h.append(sect("1 &middot; ToF (Time on Floor / service)", f"""
    <h3>Overall by month</h3><p class="kv">{tof_kv}</p>
    <img src="{charts['tof']}">
    <h3>By DC and month (ToF %)</h3>
    {df_html((tof_piv*100).round(1).reset_index().rename(columns={MONTHS[0]:MONTH_LABEL[MONTHS[0]],MONTHS[1]:MONTH_LABEL[MONTHS[1]]}))}
    """))

    h.append(sect("2 &middot; Change-over (count &amp; time, HPFD)", f"""
    <img src="{charts['changeover']}">
    <h3>Count by change-over id and month</h3>
    {df_html(co_cnt_piv.reset_index().rename(columns={MONTHS[0]:MONTH_LABEL[MONTHS[0]],MONTHS[1]:MONTH_LABEL[MONTHS[1]]}))}
    <h3>Totals by month</h3>{df_html(co_m.rename(columns={'co_count':'CO count','co_time':'CO time (hr)'}))}
    """))

    h.append(sect("4 &middot; Month-end DFC (days forward coverage)", f"""
    <img src="{charts['dfc']}">
    <h3>By DC and month (DFC days)</h3>
    {df_html(dfc_loc.assign(month=dfc_loc['month'].map(MONTH_LABEL))[['month','location','dc_inv','avg_daily_demand','dfc_days']].rename(columns={'dc_inv':'month-end inv','avg_daily_demand':'avg daily demand','dfc_days':'DFC (days)'}))}
    <h3>Overall (all scope DCs) by month</h3>
    {df_html(dfc_m[['month','dc_inv','avg_daily_demand','dfc_days']].rename(columns={'dc_inv':'month-end inv','avg_daily_demand':'avg daily demand','dfc_days':'DFC (days)'}))}
    <p class="kv">Month-end dates: {', '.join(f"{MONTH_LABEL[k]} = {v}" for k,v in dfc_info['month_end'].items())};
    forward window = {DFC_FORWARD_DAYS} days (both months full, horizon {dfc_info['horizon_end']}).</p>
    """))

    h.append(sect("5 &middot; Production total time (production + change-over, HPFD)", f"""
    <img src="{charts['ptt']}">
    {df_html(ptt_disp[['month','con_qty','prod_hr','co_hr','total_hr','co_share_pct']].rename(columns={'con_qty':'constrained qty','prod_hr':'production hr','co_hr':'changeover hr','total_hr':'total hr','co_share_pct':'CO share %'}))}
    """))

    h.append(sect("6 &middot; APQ (avg MSU produced per run)", f"""
    <div class="grid2">
    <div><h3>Overall &amp; by line (HPFD)</h3>
    <p class="big">{apq_info['apq_overall']:.3f}</p>
    <p class="kv">MSU/run overall &nbsp;|&nbsp; total MSU {apq_info['total_msu']:,.1f}, runs {apq_info['total_runs']}, materials {apq_info['n_materials']}</p>
    {df_html(apq_line.rename(columns={'msu':'MSU','con_qty':'constrained qty','runs':'runs','apq':'APQ (MSU/run)'}))}
    </div><div><img src="{charts['apq']}"></div></div>
    <div class="note">Material(s) {', '.join(apq_info['base_materials'])} are the line's <b>base product</b>, already
    mounted at sim start, so their production campaigns carry <b>no logged change-over</b>. They are still counted as real
    runs via the campaign rule ({apq_info['base_runs']} base-product campaign(s) in scope) and carry {apq_info['base_msu']:,.1f} MSU
    ({apq_info['base_msu_pct']:.0f}% of volume), giving a high MSU/run. The headline APQ uses {apq_info['total_runs']} runs
    = {apq_info['co_runs']} change-over-started + {apq_info['base_runs']} base-product campaign(s); the change-over-started
    runs on their own average ~{apq_info['apq_ex_base']:.3f} MSU/run.</div>
    <h3>By material</h3>
    {df_html(apq_material.rename(columns={'su_factor':'SU factor','su_source':'SU src','con_qty':'constrained qty','msu':'MSU','runs':'runs','apq':'APQ (MSU/run)'}))}
    """))

    h.append(sect("7 &middot; Change-over deep dive (PKG / CONV CO per MSU)", f"""
    <img src="{charts['deep']}">
    <div class="kv"><p>
    <b>PKG</b> = change-overs on the <b>packaging</b> section of the line; <b>CONV</b> = change-overs on the
    <b>converting (making)</b> section. The two id sets overlap (2-2, 2-3, 3 count for both), so PKG and CONV are
    two complementary views of the line's change-over burden, not a mutually exclusive split.<br>
    <b>PKG CO count</b> (ids {deep_info['pkg_ids']}) = {int(r['PKG_CO_count'])} &nbsp;&rarr;&nbsp; <b>{r['PKG_CO_per_MSU']:.4f}</b> per MSU<br>
    <b>CONV CO count</b> (ids {deep_info['conv_ids']}) = {int(r['CONV_CO_count'])} &nbsp;&rarr;&nbsp; <b>{r['CONV_CO_per_MSU']:.4f}</b> per MSU<br>
    <b>MSU</b> (Feb+Mar) = {deep_info['total_msu']:,.1f}</p>
    <div class="note">{deep_info['note']}. Lower CO/MSU is better (fewer change-overs per unit volume).</div>
    </div>
    <h3>CO count by month</h3>{df_html(deep_month.assign(month=deep_month['month'].map(MONTH_LABEL)).rename(columns={'PKG_CO_count':'PKG CO count','CONV_CO_count':'CONV CO count'}))}
    """))

    h.append("</div></body></html>")
    return "".join(h)


def write_run_md(tof_month, co_by_month, dfc_month, ptt, apq_info, deep, deep_info,
                 su_info, dfc_info, n_ordered):
    r = deep.iloc[0]
    lines = [
        f"# FEM {LABEL} KPI analysis - run {RUN_TAG}",
        "",
        f"- level: scenario",
        f"- project: {PROJECT}",
        f"- scenario: {SCENARIO}",
        f"- run_id: {RUN_ID}",
        f"- database: {DBNAME}",
        f"- generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Scope",
        f"- locations (DC): {', '.join(SCOPE_DCS)}",
        f"- months: {', '.join(MONTH_LABEL[m] for m in MONTHS)}",
        f"- production line (FGC line12): {LINE} at plant 0386",
        "- KPI 3 (change-over cost) skipped per user request.",
        "",
        "## Definitions",
        "- ToF = shipment_qty / order_qty (order log deduped on 7-key; shipment log not deduped).",
        f"- DFC = month-end DC ending_soh / avg daily forward demand over next {DFC_FORWARD_DAYS} days.",
        f"- MSU = con_planned_qty * su_factor / 1000 (su_factor: {su_info['source']}; coverage {su_info['n_dbx']}/{su_info['n_dbx']+su_info['n_missing']}).",
        "- Production run = a contiguous production campaign (PO); a >1-day gap starts a new run. Equals the change-over count for change-over-started materials, plus the base product's change-over-free campaigns.",
        f"- PKG CO ids = {', '.join(sorted(PKG_IDS))}; CONV CO ids = {', '.join(sorted(CONV_IDS))}.",
        "",
        "## Headline results",
    ]
    for _, row in tof_month.iterrows():
        lines.append(f"- ToF {MONTH_LABEL[row['month']]}: {row['tof']*100:.1f}% "
                     f"({row['shipment_qty']:,.0f}/{row['order_qty']:,.0f})")
    for _, row in dfc_month.iterrows():
        lines.append(f"- DFC {MONTH_LABEL[row['month']]} (all DCs): {row['dfc_days']:.1f} days "
                     f"(inv {row['dc_inv']:,.0f} / {row['avg_daily_demand']:,.1f} per day)")
    for _, row in co_by_month.iterrows():
        lines.append(f"- Change-over {MONTH_LABEL[row['month']]}: {int(row['co_count'])} events, "
                     f"{row['co_time']:.1f} hr")
    for _, row in ptt.iterrows():
        lines.append(f"- Production total time {MONTH_LABEL[row['month']]}: {row['total_hr']:.1f} hr "
                     f"(prod {row['prod_hr']:.1f} + CO {row['co_hr']:.1f}; CO share {row['co_share_pct']:.1f}%)")
    lines.append(f"- APQ overall: {apq_info['apq_overall']:.3f} MSU/run "
                 f"(total MSU {apq_info['total_msu']:,.1f} / {apq_info['total_runs']} runs)")
    lines.append(f"- PKG CO/MSU: {r['PKG_CO_per_MSU']:.4f} ({int(r['PKG_CO_count'])} CO / {deep.iloc[0]['MSU']:,.1f} MSU)")
    lines.append(f"- CONV CO/MSU: {r['CONV_CO_per_MSU']:.4f} ({int(r['CONV_CO_count'])} CO / {deep.iloc[0]['MSU']:,.1f} MSU)")
    lines += [
        "",
        "## Notes",
        f"- All {n_ordered} materials ordered at the scope DCs are HPFD-line materials (network fully HPFD-scoped).",
        f"- Change-over note: {deep_info['note']}.",
        f"- APQ note: material(s) {', '.join(apq_info['base_materials'])} are the line's base product (already mounted at "
        f"sim start), so their campaigns carry no logged change-over but ARE counted as runs via the contiguous-campaign "
        f"rule ({apq_info['base_runs']} base-product campaign(s) in scope). They carry "
        f"{apq_info['base_msu']:,.1f} MSU ({apq_info['base_msu_pct']:.0f}% of volume). Headline APQ uses "
        f"{apq_info['total_runs']} runs = {apq_info['co_runs']} change-over-started + {apq_info['base_runs']} "
        f"base-product campaign(s); change-over-started runs alone average ~{apq_info['apq_ex_base']:.3f} MSU/run.",
        f"- DFC month-end dates: {', '.join(f'{MONTH_LABEL[k]}={v}' for k,v in dfc_info['month_end'].items())}; both months have full {DFC_FORWARD_DAYS}-day forward windows.",
        "",
        "## Outputs",
        f"- workbook: {PROJECT}_result_summary_{RUN_TAG}.xlsx",
        "- report: analysis.html",
        "- extracts/: per-KPI CSVs",
    ]
    (OUT_DIR / "run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # update LATEST.md (prepend run entry, set current pointer)
    latest = ANALYSIS_ROOT / "LATEST.md"
    entry = [
        f"current: {RUN_TAG}",
        "",
        "## Run log (newest first)",
        "",
        f"### {RUN_TAG}",
        f"- level: scenario | run_id: {RUN_ID} | DB: {DBNAME}",
        f"- scope: DCs {', '.join(SCOPE_DCS)}; months {', '.join(MONTH_LABEL[m] for m in MONTHS)}",
        "- KPIs: ToF, change-over count/time, month-end DFC, production time, APQ, PKG/CONV CO per MSU.",
        "- KPI 3 (change-over cost) skipped per user request.",
        "",
    ]
    prev = ""
    if latest.exists():
        txt = latest.read_text(encoding="utf-8")
        idx = txt.find("## Run log")
        prev = txt[idx + len("## Run log (newest first)\n"):] if idx >= 0 else ""
    (ANALYSIS_ROOT / "LATEST.md").write_text("\n".join(entry) + prev, encoding="utf-8")


if __name__ == "__main__":
    main()
