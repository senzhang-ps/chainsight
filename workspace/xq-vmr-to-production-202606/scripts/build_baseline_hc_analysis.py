"""Build the XQ-VMR baseline analysis deliverable.

Scenario: baseline-hc | run_id: db_baseline-hc_20260617_203548

KPIs:
  1. TOF by category x production line x month
  2. Changeover count by production line x month x changeover id
  3. Changeover cost by production line x month x changeover id
  4. Month-end DFC (days forward coverage) by category x production line x month
  5. Production total time (production + changeover, hr) by category x production line x month
  6. APQ (avg MSU produced per production run) by production line and by material
  7. Wash count / MSU by production line (wash = changeover id 2/3; MSU = qty x su_factor / 1000)
  8. MOQ / APQ coverage (days) by SKU / production line / category (JASO window 6.29-11.1)

Outputs a self-contained, timestamped run folder under
  workspace/xq-vmr-to-production-202606/scenarios/baseline/analysis/<run-id>/
containing extracts/, an Excel workbook, an HTML report, and run.md, and
updates analysis/LATEST.md.

Run from the repo root with the venv interpreter:
  .\\.venv\\Scripts\\python.exe .\\workspace\\xq-vmr-to-production-202606\\scripts\\build_baseline_analysis.py
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

# --- repo-root-relative imports -------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
PROJECT = "xq-vmr-to-production-202606"
SCENARIO = "baseline-hc"
RUN_ID = "db_baseline-hc_20260617_203548"
CONFIG_NAME = "baseline-hc"

# month-end DFC parameters
SIM_START = pd.Timestamp("2026-06-29")   # simulation day 1 == demand-forecast week 1 start
DFC_FORWARD_DAYS = 30                     # forward window for the coverage rate
PLANT_LOC = "1864"                        # XQ plant; excluded from DC-only inventory & demand

# reporting horizon: cap all month-keyed KPIs at end of October (exclude 2026-11)
MONTH_MAX = "2026-10"

# KPI8 MOQ/APQ coverage: the window 2026-06-29..2026-11-01 == 126 days == demand-
# forecast weeks 1..18; coverage (days) = quantity(MSU) / (wk1-18 fcst MSU / 126).
COVERAGE_DAYS = 126
COVERAGE_WEEK_MAX = 18

# KPI7 MSU: wash changeover ids + the per-material SU-factor source workbook
WASH_CHANGEOVER_IDS = ("2", "3")          # "wash" changeovers in this run
SUF_XLSX = (ROOT / "workspace" / PROJECT / "scenarios" / SCENARIO / "config"
            / "SUF for XQ HC ChainSight.xlsx")  # SU-factor primary source (overrides DBx)
_CONFIG_DIR = ROOT / "workspace" / PROJECT / "scenarios" / SCENARIO / "config"
MISSING_SUF_XLSX = _CONFIG_DIR / "missing SUF manual input.xlsx"      # manual fallback for SU factor
MISSING_CAT_XLSX = _CONFIG_DIR / "missing category manual input.xlsx"  # manual fallback for category

PROJECT_DIR = ROOT / "workspace" / PROJECT
SCEN_DIR = PROJECT_DIR / "scenarios" / SCENARIO
ANALYSIS_ROOT = SCEN_DIR / "analysis"

sys.path.insert(0, str(PROJECT_DIR / "scripts"))


# --- DB helpers -----------------------------------------------------------------
def pg_connect():
    cfg = yaml.safe_load((ROOT / "config/defaults.yaml").read_text(encoding="utf-8"))["database"]
    return psycopg.connect(
        host=cfg["host"], port=int(cfg["port"]), dbname=cfg["database"],
        user=cfg["user"], password=str(cfg["password"]),
        client_encoding="UTF8", connect_timeout=10,
    )


def q(conn, sql: str, params: tuple) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# --- 1. pull raw aggregates from Postgres --------------------------------------
def pull_postgres():
    with pg_connect() as conn:
        orders = q(conn, """
            select to_char(date::date,'YYYY-MM') as month, material,
                   sum(quantity)::float8 as order_qty
            from (select distinct simulation_date, date, material, location,
                         demand_type, advance_days, quantity
                  from module1_output_orderlog where run_id=%s) d
            group by 1,2
        """, (RUN_ID,))

        ships = q(conn, """
            select to_char(date::date,'YYYY-MM') as month, material,
                   sum(quantity)::float8 as shipment_qty
            from module1_output_shipmentlog where run_id=%s
            group by 1,2
        """, (RUN_ID,))

        line_map = q(conn, """
            select distinct material, delegate_line as line
            from cfg_m4_materiallocationlinecfg where config_name=%s
        """, (CONFIG_NAME,))

        changeover = q(conn, """
            select to_char(date::date,'YYYY-MM') as month, line,
                   changeover_type as changeover_id,
                   sum(count)::bigint as changeover_count,
                   sum(cost)::float8 as changeover_cost,
                   sum(time)::float8 as changeover_time,
                   sum(mu_loss)::float8 as mu_loss
            from module4_output_changeoverlog where run_id=%s
            group by 1,2,3
        """, (RUN_ID,))

        co_def = q(conn, """
            select line, changeover_id, time, cost, mu_loss
            from cfg_m4_changeoverdefinition where config_name=%s
            order by line, changeover_id
        """, (CONFIG_NAME,))

        # KPI4 inputs -------------------------------------------------------------
        # DC set = demand-forecast locations excluding the plant (1864).
        # Month-end ending_soh summed over the DC set, per material per month.
        stock_me = q(conn, """
            with dcs as (
              select distinct location from cfg_m1_demandforecast
              where config_name=%s and location <> %s
            ),
            me as (
              select to_char(date::date,'YYYY-MM') ym, max(date::date) med
              from module5_output_stockonhandlog where run_id=%s group by 1
            )
            select me.ym as month, s.material,
                   sum(s.ending_soh)::float8 as dc_ending_inv,
                   me.med as month_end_date
            from module5_output_stockonhandlog s
            join me on me.med = s.date::date
            join dcs on dcs.location = s.location
            where s.run_id=%s
            group by me.ym, s.material, me.med
        """, (CONFIG_NAME, PLANT_LOC, RUN_ID, RUN_ID))

        # Weekly demand forecast at DC locations, per material per week.
        fcst_weekly = q(conn, """
            select week, material, sum(quantity)::float8 as fcst_qty
            from cfg_m1_demandforecast
            where config_name=%s and location <> %s
            group by week, material
        """, (CONFIG_NAME, PLANT_LOC))

        dc_loc_df = q(conn, """
            select distinct location from cfg_m1_demandforecast
            where config_name=%s and location <> %s order by location
        """, (CONFIG_NAME, PLANT_LOC))

        fcst_split = q(conn, """
            select case when location=%s then 'plant' else 'dc' end cls,
                   sum(quantity)::float8 q
            from cfg_m1_demandforecast where config_name=%s group by 1
        """, (PLANT_LOC, CONFIG_NAME))

        # KPI5 inputs ------------------------------------------------------------
        # Constrained planned production qty per material x line x month
        # (month from production_plan_date = the day the line actually runs).
        prod_plan = q(conn, """
            select to_char(production_plan_date::date,'YYYY-MM') as month,
                   material, line,
                   sum(con_planned_qty)::float8 as con_qty
            from module4_output_productionplan where run_id=%s
            group by 1,2,3
        """, (RUN_ID,))

        # Production rate (unit/hr) per material (one rate per material at plant).
        prd_rate = q(conn, """
            select material, max(prd_rate)::float8 as prd_rate
            from cfg_m4_materiallocationlinecfg where config_name=%s
            group by material
        """, (CONFIG_NAME,))

        # KPI6 inputs ------------------------------------------------------------
        # APQ = total MSU produced / number of production runs (MSU = con_planned_qty
        # x su_factor / 1000). A run begins with a changeover, so each plan row with a
        # non-null changeover_id is one run; the null-changeover_id rows are
        # continuations of the current run (no preceding changeover) and are not
        # counted as separate runs. Grain: material x line.
        prod_runs = q(conn, """
            select material, line,
                   sum(con_planned_qty)::float8 as con_qty,
                   count(*) filter (where changeover_id is not null)::bigint as runs
            from module4_output_productionplan where run_id=%s
            group by material, line
        """, (RUN_ID,))

        # KPI8 inputs (MOQ / APQ coverage) --------------------------------------
        # Global_Network sourcing tree (to resolve each SKU's 1864 sub-network),
        # the demand forecast for weeks 1..18 (the 6.29-11.1 window) across every
        # location, and the current production MOQ (min_batch) per material.
        global_network = q(conn, """
            select material, location, sourcing
            from cfg_global_network where config_name=%s
        """, (CONFIG_NAME,))

        fcst_all = q(conn, """
            select material, location, sum(quantity)::float8 as fcst_cs
            from cfg_m1_demandforecast
            where config_name=%s and week <= %s
            group by material, location
        """, (CONFIG_NAME, COVERAGE_WEEK_MAX))

        moq = q(conn, """
            select material, max(min_batch)::float8 as min_batch
            from cfg_m4_materiallocationlinecfg where config_name=%s
            group by material
        """, (CONFIG_NAME,))

    for df in (orders, ships, line_map, changeover, co_def, stock_me, fcst_weekly,
               prod_plan, prd_rate, prod_runs, global_network, fcst_all, moq):
        if "material" in df.columns:
            df["material"] = df["material"].astype(str)
    # collapse any accidental dupes to one line per material
    line_map = line_map.dropna(subset=["line"]).drop_duplicates(subset=["material"])

    split = dict(zip(fcst_split["cls"], fcst_split["q"]))
    dc_info = {
        "dc_locations": dc_loc_df["location"].tolist(),
        "dc_demand_total": float(split.get("dc", 0.0)),
        "plant_demand_total": float(split.get("plant", 0.0)),
    }
    return (orders, ships, line_map, changeover, co_def, stock_me, fcst_weekly,
            dc_info, prod_plan, prd_rate, prod_runs, global_network, fcst_all, moq)


# --- 2. category mapping (Databricks primary, snapshot fallback) ----------------
def get_category_map(materials: list[str]):
    source = "databricks"
    note = ""
    try:
        from category_mapping import fetch_category_mapping
        cat = fetch_category_mapping(materials)
        if cat.empty or cat["category_en"].notna().sum() == 0:
            raise RuntimeError("Databricks returned no category rows")
    except Exception as exc:  # noqa: BLE001 - fall back to snapshot, disclose it
        source = "snapshot_fallback"
        note = f"Databricks pull failed ({exc!s}); used stale snapshot from sdc project."
        snap = ROOT / "workspace/sdc-space-rccp-simulation-202605/data/ps_psc_sku_master_category_en.csv"
        cat = pd.read_csv(snap, dtype=str)
    cat["material"] = cat["material"].astype(str)
    cat = cat.drop_duplicates(subset=["material"])

    # manual-input fallback: fill categories the source still misses
    have = set(cat.loc[cat["category_en"].notna(), "material"])
    need = [m for m in materials if m not in have]
    n_manual = 0
    if need and MISSING_CAT_XLSX.exists():
        man = pd.read_excel(MISSING_CAT_XLSX, dtype=str)
        man["material"] = man["Material"].astype(str).str.strip()
        man["category_en"] = man["category_en"].astype(str).str.strip()
        man = man[man["category_en"].notna() & (man["category_en"] != "")
                   & (man["category_en"].str.lower() != "nan")]
        man = man[man["material"].isin(need)].drop_duplicates("material")
        if not man.empty:
            cat = pd.concat([cat, man[["material", "category_en"]]], ignore_index=True)
            cat = cat.drop_duplicates(subset=["material"])
            n_manual = len(man)
    if n_manual:
        note = (note + " | " if note else "") + f"manual-input filled {n_manual} categories"
    return cat, source, note


def get_su_factor_map(materials: list[str]):
    """Per-material SU factor for MSU = qty * su_factor / 1000.

    Primary source is the local SUF workbook (purpose-built for XQ HC, higher
    precision, and the only source covering the new VMR "21..." codes). Databricks
    ps_psc_sku_master.su_factor_for_buom is the fallback for any material the
    workbook misses. Returns (factor_map, source_map, info) where source_map flags
    each material's origin (suf_file / databricks) for full disclosure.
    """
    factor: dict[str, float] = {}
    source: dict[str, str] = {}

    # 1) local SUF workbook (primary)
    n_suf = 0
    suf_err = ""
    try:
        suf = pd.read_excel(SUF_XLSX, sheet_name="Sheet1", dtype=str)
        suf["material"] = suf["Material"].astype(str).str.strip()
        suf["su"] = pd.to_numeric(suf["SUF"], errors="coerce")
        suf = suf.dropna(subset=["su"]).drop_duplicates("material")
        sm = dict(zip(suf["material"], suf["su"]))
        for m in materials:
            if m in sm:
                factor[m] = float(sm[m])
                source[m] = "suf_file"
                n_suf += 1
    except Exception as exc:  # noqa: BLE001 - workbook optional, fall back to DBx
        suf_err = str(exc)

    # 2) Databricks fallback for anything still missing
    n_dbx = 0
    dbx_err = ""
    need = [m for m in materials if m not in factor]
    if need:
        try:
            from category_mapping import fetch_su_factor
            dbx = fetch_su_factor(need)
            dm = dict(zip(dbx["material"], dbx["su_factor"]))
            for m in need:
                v = dm.get(m)
                if v is not None and pd.notna(v):
                    factor[m] = float(v)
                    source[m] = "databricks"
                    n_dbx += 1
        except Exception as exc:  # noqa: BLE001 - disclose, leave missing
            dbx_err = str(exc)

    # 3) manual-input workbook fallback for anything still missing
    n_manual = 0
    man_err = ""
    need2 = [m for m in materials if m not in factor]
    if need2 and MISSING_SUF_XLSX.exists():
        try:
            man = pd.read_excel(MISSING_SUF_XLSX, dtype=str)
            man["material"] = man["Material"].astype(str).str.strip()
            man["su"] = pd.to_numeric(man["SUF"], errors="coerce")
            man = man.dropna(subset=["su"]).drop_duplicates("material")
            mm = dict(zip(man["material"], man["su"]))
            for m in need2:
                v = mm.get(m)
                if v is not None and pd.notna(v):
                    factor[m] = float(v)
                    source[m] = "manual_input"
                    n_manual += 1
        except Exception as exc:  # noqa: BLE001 - disclose, leave missing
            man_err = str(exc)

    missing = sorted(m for m in materials if m not in factor)
    info = {
        "n_suf": n_suf, "n_dbx": n_dbx, "n_manual": n_manual, "n_missing": len(missing),
        "missing": missing, "suf_error": suf_err, "dbx_error": dbx_err, "manual_error": man_err,
        "suf_path": str(SUF_XLSX.relative_to(ROOT)),
    }
    return factor, source, info


# --- 3. compute TOF -------------------------------------------------------------
def compute_cfr(orders, ships, line_map, cat):
    base = orders.merge(ships, on=["month", "material"], how="outer")
    base["order_qty"] = base["order_qty"].fillna(0.0)
    base["shipment_qty"] = base["shipment_qty"].fillna(0.0)

    line_lookup = dict(zip(line_map["material"], line_map["line"]))
    cat_lookup = dict(zip(cat["material"], cat["category_en"]))
    base["line"] = base["material"].map(line_lookup).fillna("Unmapped")
    base["category"] = base["material"].map(cat_lookup)
    base["category_source"] = np.where(base["category"].notna(), "databricks", "unmapped")

    # --- line-inference enrichment ------------------------------------------------
    # New VMR materials ("21..." codes) are not yet in the HANA SKU master, so they
    # carry no category_en. A line is "category-deterministic" when every
    # Databricks-mapped material on it shares exactly one category; an unmapped
    # material on such a line inherits that category. Mixed lines (e.g. XQHD) are
    # left Unmapped to avoid a wrong assignment. Every inferred row is flagged via
    # category_source = "line-inferred" for full transparency.
    mapped = base.loc[base["category"].notna(), ["material", "line", "category"]].drop_duplicates()
    mapped = mapped[mapped["line"] != "Unmapped"]
    line_ncat = mapped.groupby("line")["category"].nunique()
    det_lines = line_ncat[line_ncat == 1].index
    line_to_cat = (mapped[mapped["line"].isin(det_lines)]
                   .drop_duplicates("line").set_index("line")["category"].to_dict())
    need = base["category"].isna() & base["line"].isin(line_to_cat.keys())
    base.loc[need, "category"] = base.loc[need, "line"].map(line_to_cat)
    base.loc[need, "category_source"] = "line-inferred"

    base["category"] = base["category"].where(base["category"].notna(), "Unmapped")

    def rollup(df, keys):
        g = df.groupby(keys, as_index=False).agg(
            order_qty=("order_qty", "sum"),
            shipment_qty=("shipment_qty", "sum"),
        )
        g["cfr"] = g["shipment_qty"] / g["order_qty"].replace(0, np.nan)
        g["cfr"] = g["cfr"].round(4)
        g["order_qty"] = g["order_qty"].round(1)
        g["shipment_qty"] = g["shipment_qty"].round(1)
        return g

    cfr_main = rollup(base, ["month", "category", "line"]).sort_values(["month", "category", "line"])
    cfr_by_line = rollup(base, ["month", "line"]).sort_values(["month", "line"])
    cfr_by_cat = rollup(base, ["month", "category"]).sort_values(["month", "category"])
    cfr_by_month = rollup(base, ["month"]).sort_values(["month"])
    cfr_line_total = rollup(base, ["category", "line"]).sort_values(["category", "line"])
    return base, cfr_main, cfr_by_line, cfr_by_cat, cfr_by_month, cfr_line_total


# --- 4. compute month-end DFC (Days Forward Coverage) ---------------------------
def _week_calendar(weeks: list[int]) -> pd.DataFrame:
    """Map forecast week index -> [start, end] calendar dates (week 1 == SIM_START)."""
    ws = SIM_START + pd.to_timedelta([(int(w) - 1) * 7 for w in weeks], unit="D")
    return pd.DataFrame({"week": list(weeks), "wk_start": ws,
                         "wk_end": ws + pd.Timedelta(days=6)})


def compute_dfc(stock_me, fcst_weekly, base):
    """Month-end DFC = DC ending inventory / average daily forward demand.

    Forward demand = weekly DC demand forecast spread to a daily rate (qty/7) and
    summed over the next DFC_FORWARD_DAYS, clamped to the forecast horizon. DFC is
    a ratio of sums at each rollup level (aggregate inventory / aggregate daily
    demand rate), so groups are demand-weighted. Category/line come from the same
    per-material resolution used for TOF.
    """
    res = base.drop_duplicates("material")[["material", "line", "category", "category_source"]]

    wk = _week_calendar(sorted(fcst_weekly["week"].unique()))
    horizon_end = wk["wk_end"].max()
    wk_start = wk["wk_start"].values
    wk_end = wk["wk_end"].values

    me = (stock_me[["month", "month_end_date"]].drop_duplicates()
          .assign(month_end_date=lambda d: pd.to_datetime(d["month_end_date"]))
          .sort_values("month"))
    month_end_map = dict(zip(me["month"], me["month_end_date"]))

    win, wd = {}, {}
    for m, d in month_end_map.items():
        ws = d + pd.Timedelta(days=1)
        we = min(d + pd.Timedelta(days=DFC_FORWARD_DAYS), horizon_end)
        win[m] = (ws, we)
        wd[m] = int((we - ws).days + 1) if ws <= we else 0

    # forward demand per (month, material) via week/window overlap
    fwd_rows = []
    for m, (ws, we) in win.items():
        if ws > we:
            continue
        end = np.minimum(wk_end, we.to_datetime64())
        start = np.maximum(wk_start, ws.to_datetime64())
        ov = (end - start) / np.timedelta64(1, "D") + 1.0
        ov = np.clip(ov, 0.0, None)
        wk_ov = wk.assign(overlap=ov)
        wk_ov = wk_ov[wk_ov["overlap"] > 0]
        if wk_ov.empty:
            continue
        fm = fcst_weekly.merge(wk_ov[["week", "overlap"]], on="week", how="inner")
        fm["contrib"] = fm["fcst_qty"] * fm["overlap"] / 7.0
        g = fm.groupby("material", as_index=False)["contrib"].sum().rename(
            columns={"contrib": "forward_demand"})
        g["month"] = m
        fwd_rows.append(g)
    fwd = (pd.concat(fwd_rows, ignore_index=True) if fwd_rows
           else pd.DataFrame(columns=["material", "forward_demand", "month"]))

    dfc_base = stock_me[["month", "material", "dc_ending_inv"]].merge(
        fwd, on=["month", "material"], how="outer")
    dfc_base["dc_ending_inv"] = dfc_base["dc_ending_inv"].fillna(0.0)
    dfc_base["forward_demand"] = dfc_base["forward_demand"].fillna(0.0)
    dfc_base = dfc_base.merge(res, on="material", how="left")
    dfc_base["line"] = dfc_base["line"].fillna("Unmapped")
    dfc_base["category"] = dfc_base["category"].fillna("Unmapped")
    dfc_base["category_source"] = dfc_base["category_source"].fillna("unmapped")
    dfc_base["window_days"] = dfc_base["month"].map(wd)

    def droll(keys):
        g = dfc_base.groupby(keys, as_index=False).agg(
            dc_inv=("dc_ending_inv", "sum"),
            fwd_demand=("forward_demand", "sum"),
        )
        g["window_days"] = g["month"].map(wd)
        g["avg_daily_demand"] = g["fwd_demand"] / g["window_days"].replace(0, np.nan)
        # DFC is undefined where there is effectively no forward demand (denominator ~0):
        # inventory with no demand to consume it is not a meaningful "days of coverage".
        denom = g["avg_daily_demand"].where(g["avg_daily_demand"] >= 1.0, np.nan)
        g["dfc_days"] = (g["dc_inv"] / denom).round(1)
        g["avg_daily_demand"] = g["avg_daily_demand"].round(1)
        g["dc_inv"] = g["dc_inv"].round(1)
        g["fwd_demand"] = g["fwd_demand"].round(1)
        g["partial_window"] = g["window_days"] < DFC_FORWARD_DAYS
        return g

    dfc_main = droll(["month", "category", "line"]).sort_values(["month", "category", "line"])
    dfc_by_cat = droll(["month", "category"]).sort_values(["month", "category"])
    dfc_by_line = droll(["month", "line"]).sort_values(["month", "line"])
    dfc_by_month = droll(["month"]).sort_values(["month"])

    info = {
        "sim_start": str(SIM_START.date()),
        "forward_days": DFC_FORWARD_DAYS,
        "horizon_end": str(horizon_end.date()),
        "window_days": {m: int(wd[m]) for m in wd},
        "partial_months": [m for m in wd if wd[m] < DFC_FORWARD_DAYS],
        "month_end": {m: str(d.date()) for m, d in month_end_map.items()},
        "full_months": [m for m in wd if wd[m] >= DFC_FORWARD_DAYS],
    }
    return dfc_base, dfc_main, dfc_by_cat, dfc_by_line, dfc_by_month, info


# --- 5. compute production total time (production + changeover, hours) ----------
def compute_ptt(prod_plan, prd_rate, changeover, base):
    """Production total time (hr) = production time + changeover time.

    production time = constrained planned qty (con_planned_qty) / production rate
                      (prd_rate, unit/hr), computed per material then summed.
    changeover time = simulated changeover hours from the log (already = changeover
                      id x count x per-event changeover time, hr) rolled up per line.
    Changeover is recorded per line only; it is allocated to categories in
    proportion to production hours within each (line, month). The only mixed line
    is XQHD (Hair+PCC); every other line is single-category so its changeover maps
    unambiguously. Category resolution matches the TOF/DFC KPIs.
    """
    res = base.drop_duplicates("material")[["material", "line", "category"]]
    mat_cat = dict(zip(res["material"], res["category"]))
    # deterministic line -> category (single non-Unmapped category on the line) for
    # any production material absent from the TOF order universe.
    nonu = res[res["category"] != "Unmapped"]
    line_ncat = nonu.groupby("line")["category"].nunique()
    det_lines = line_ncat[line_ncat == 1].index
    line_cat = (nonu[nonu["line"].isin(det_lines)].drop_duplicates("line")
                .set_index("line")["category"].to_dict())
    rate = dict(zip(prd_rate["material"], prd_rate["prd_rate"]))

    pp = prod_plan.copy()
    pp["prd_rate"] = pp["material"].map(rate)
    pp["category"] = pp["material"].map(mat_cat)
    nmask = pp["category"].isna()
    pp.loc[nmask, "category"] = pp.loc[nmask, "line"].map(line_cat)
    pp["category"] = pp["category"].fillna("Unmapped")
    pp["prod_hr"] = (pp["con_qty"] / pp["prd_rate"].replace(0, np.nan)).fillna(0.0)

    # production hours + constrained qty at (month, line, category)
    ph = pp.groupby(["month", "line", "category"], as_index=False).agg(
        prod_hr=("prod_hr", "sum"), con_qty=("con_qty", "sum"))

    # changeover hours at (month, line) from the simulation log
    ch = (changeover.groupby(["month", "line"], as_index=False)["changeover_time"]
          .sum().rename(columns={"changeover_time": "co_hr"}))

    # allocate changeover to category by production-hour share within (month, line)
    p_ml = ph.groupby(["month", "line"], as_index=False)["prod_hr"].sum().rename(
        columns={"prod_hr": "p_ml"})
    a = ph.merge(p_ml, on=["month", "line"]).merge(ch, on=["month", "line"], how="left")
    a["co_hr"] = a["co_hr"].fillna(0.0)
    a["co_alloc"] = np.where(a["p_ml"] > 0, a["co_hr"] * a["prod_hr"] / a["p_ml"], 0.0)
    a = a[["month", "line", "category", "prod_hr", "con_qty", "co_alloc"]]

    # fallback: (month, line) with changeover but no production that month ->
    # split by the line's all-month production-hour share (else assign to Unmapped).
    prod_ml = set(p_ml[["month", "line"]].itertuples(index=False, name=None))
    miss = ch[[(m, ln) not in prod_ml for m, ln in
               zip(ch["month"], ch["line"])]]
    if not miss.empty:
        p_l = ph.groupby(["line", "category"], as_index=False)["prod_hr"].sum()
        p_lt = p_l.groupby("line", as_index=False)["prod_hr"].sum().rename(
            columns={"prod_hr": "p_l"})
        extra = []
        for _, r in miss.iterrows():
            wl = p_l[p_l["line"] == r["line"]].merge(p_lt, on="line")
            if not wl.empty and wl["p_l"].iloc[0] > 0:
                for _, c in wl.iterrows():
                    extra.append({"month": r["month"], "line": r["line"],
                                  "category": c["category"], "prod_hr": 0.0,
                                  "con_qty": 0.0,
                                  "co_alloc": r["co_hr"] * c["prod_hr"] / c["p_l"]})
            else:
                extra.append({"month": r["month"], "line": r["line"],
                              "category": "Unmapped", "prod_hr": 0.0,
                              "con_qty": 0.0, "co_alloc": r["co_hr"]})
        a = pd.concat([a, pd.DataFrame(extra)], ignore_index=True)

    a = a.groupby(["month", "line", "category"], as_index=False).agg(
        prod_hr=("prod_hr", "sum"), con_qty=("con_qty", "sum"), co_hr=("co_alloc", "sum"))
    a["total_hr"] = a["prod_hr"] + a["co_hr"]
    ptt_base = a.copy()

    def roll(keys):
        g = a.groupby(keys, as_index=False).agg(
            prod_hr=("prod_hr", "sum"), co_hr=("co_hr", "sum"),
            total_hr=("total_hr", "sum"), con_qty=("con_qty", "sum"))
        g["co_share"] = (g["co_hr"] / g["total_hr"].replace(0, np.nan) * 100).round(1)
        for c in ("prod_hr", "co_hr", "total_hr", "con_qty"):
            g[c] = g[c].round(1)
        return g

    ptt_main = roll(["month", "category", "line"]).sort_values(["month", "category", "line"])
    ptt_by_line = roll(["month", "line"]).sort_values(["month", "line"])
    ptt_by_cat = roll(["month", "category"]).sort_values(["month", "category"])
    ptt_by_month = roll(["month"]).sort_values("month")
    ptt_line_total = roll(["line"]).sort_values("total_hr", ascending=False)
    ptt_cat_total = roll(["category"]).sort_values("total_hr", ascending=False)

    info = {
        "prod_hr_total": round(float(ph["prod_hr"].sum()), 1),
        "co_hr_total": round(float(ch["co_hr"].sum()), 1),
        "total_hr_total": round(float(a["total_hr"].sum()), 1),
        "co_share_total": round(float(ch["co_hr"].sum()) /
                                float(a["total_hr"].sum()) * 100, 1)
        if a["total_hr"].sum() else 0.0,
        "months": sorted(a["month"].unique().tolist()),
        "mixed_lines": sorted(line_ncat[line_ncat > 1].index.tolist()),
        "n_lines": int(a["line"].nunique()),
        "qty_basis": "con_planned_qty (constrained planned = produced here)",
        "month_basis": "production_plan_date",
    }
    return (ptt_base, ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month,
            ptt_line_total, ptt_cat_total, info)


def compute_apq(prod_runs, base, su_factor, su_source):
    """APQ = average MSU produced per production run.

    A production run begins with a changeover, so the run count is the number of
    plan rows with a non-null changeover_id (continuation rows are excluded).
    Production volume is expressed in MSU = con_planned_qty * su_factor / 1000
    (su_factor from the SUF workbook, Databricks fallback) so APQ = total MSU /
    run count, reported by production line and by material. Category is attached
    for context using the same resolution as TOF.
    """
    pr = prod_runs.copy()
    pr["con_qty"] = pr["con_qty"].fillna(0.0)
    pr["runs"] = pr["runs"].fillna(0).astype("int64")
    pr["su_factor"] = pr["material"].map(su_factor)
    pr["su_source"] = pr["material"].map(su_source).fillna("missing")
    pr["msu"] = pr["con_qty"] * pr["su_factor"].fillna(0.0) / 1000.0

    res = base.drop_duplicates("material")[["material", "line", "category"]]
    mat_cat = dict(zip(res["material"], res["category"]))
    nonu = res[res["category"] != "Unmapped"]
    line_ncat = nonu.groupby("line")["category"].nunique()
    det_lines = line_ncat[line_ncat == 1].index
    line_cat = (nonu[nonu["line"].isin(det_lines)].drop_duplicates("line")
                .set_index("line")["category"].to_dict())

    def apq_frame(g, keys):
        out = g.groupby(keys, as_index=False).agg(
            msu=("msu", "sum"), con_qty=("con_qty", "sum"), runs=("runs", "sum"))
        out["apq"] = (out["msu"] / out["runs"].replace(0, np.nan)).round(3)
        out["msu"] = out["msu"].round(2)
        out["con_qty"] = out["con_qty"].round(1)
        return out

    apq_by_line = apq_frame(pr, ["line"]).sort_values("apq", ascending=False)

    pm = pr.groupby("material", as_index=False).agg(
        msu=("msu", "sum"), con_qty=("con_qty", "sum"), runs=("runs", "sum"),
        su_factor=("su_factor", "first"), su_source=("su_source", "first"))
    # one representative line per material (its delegate line; each has exactly one)
    mat_line = pr.sort_values("con_qty", ascending=False).drop_duplicates("material")
    mat_line = dict(zip(mat_line["material"], mat_line["line"]))
    pm["line"] = pm["material"].map(mat_line)
    pm["category"] = pm["material"].map(mat_cat)
    nmask = pm["category"].isna()
    pm.loc[nmask, "category"] = pm.loc[nmask, "line"].map(line_cat)
    pm["category"] = pm["category"].fillna("Unmapped")
    pm["apq"] = (pm["msu"] / pm["runs"].replace(0, np.nan)).round(3)
    pm["msu"] = pm["msu"].round(3)
    pm["con_qty"] = pm["con_qty"].round(1)
    apq_by_material = pm[["material", "line", "category", "su_factor", "su_source",
                          "con_qty", "msu", "runs", "apq"]] \
        .sort_values("msu", ascending=False)

    tot_msu = float(pr["msu"].sum())
    tot_qty = float(pr["con_qty"].sum())
    tot_runs = int(pr["runs"].sum())
    n_missing = int(pr.drop_duplicates("material")["su_source"].eq("missing").sum())
    info = {
        "total_msu": round(tot_msu, 1),
        "total_con_qty": round(tot_qty, 1),
        "total_runs": tot_runs,
        "apq_overall": round(tot_msu / tot_runs, 3) if tot_runs else float("nan"),
        "n_materials": int(pm["material"].nunique()),
        "n_lines": int(apq_by_line["line"].nunique()),
        "n_missing_su": n_missing,
        "run_basis": "non-null changeover_id plan row (= a changeover starts the run)",
        "qty_basis": "MSU = con_planned_qty x su_factor / 1000 (SUF workbook primary, Databricks fallback)",
    }
    return apq_by_line, apq_by_material, info


def compute_msu(prod_runs, changeover, su_factor, su_source):
    """Wash count and MSU by production line, plus washes per MSU.

    wash count = sum of changeover counts where changeover id is a "wash" id
                 (2 or 3) from the simulation changeover log, per line.
    MSU        = sum over the line's materials of con_planned_qty * su_factor / 1000
                 (su_factor from the SUF workbook, Databricks fallback).
    wash / MSU = wash count / MSU (washes per MSU of production volume).
    """
    pr = prod_runs.copy()
    pr["con_qty"] = pr["con_qty"].fillna(0.0)
    pr["su_factor"] = pr["material"].map(su_factor)
    pr["su_source"] = pr["material"].map(su_source).fillna("missing")
    pr["msu"] = pr["con_qty"] * pr["su_factor"] / 1000.0

    msu_line = pr.groupby("line", as_index=False).agg(
        con_qty=("con_qty", "sum"), msu=("msu", "sum"))

    wash = (changeover[changeover["changeover_id"].astype(str).isin(WASH_CHANGEOVER_IDS)]
            .groupby("line", as_index=False)["changeover_count"].sum()
            .rename(columns={"changeover_count": "wash_count"}))

    out = msu_line.merge(wash, on="line", how="left")
    out["wash_count"] = out["wash_count"].fillna(0).astype("int64")
    out["wash_per_msu"] = (out["wash_count"] / out["msu"].replace(0, np.nan)).round(3)
    out["msu_per_wash"] = (out["msu"] / out["wash_count"].replace(0, np.nan)).round(2)
    out["con_qty"] = out["con_qty"].round(1)
    out["msu"] = out["msu"].round(2)
    msu_by_line = out.sort_values("wash_count", ascending=False)

    # per-material MSU detail (for the extract / traceability)
    msu_material = pr.groupby(["material", "line"], as_index=False).agg(
        con_qty=("con_qty", "sum"),
        su_factor=("su_factor", "first"),
        su_source=("su_source", "first"),
        msu=("msu", "sum"))
    msu_material["con_qty"] = msu_material["con_qty"].round(1)
    msu_material["msu"] = msu_material["msu"].round(3)
    msu_material = msu_material.sort_values("msu", ascending=False)

    tot_wash = int(out["wash_count"].sum())
    tot_msu = float(out["msu"].sum())
    src_counts = pr.drop_duplicates("material")["su_source"].value_counts().to_dict()
    info = {
        "total_wash": tot_wash,
        "total_msu": round(tot_msu, 1),
        "wash_per_msu_overall": round(tot_wash / tot_msu, 3) if tot_msu else float("nan"),
        "n_suf": int(src_counts.get("suf_file", 0)),
        "n_dbx": int(src_counts.get("databricks", 0)),
        "n_manual": int(src_counts.get("manual_input", 0)),
        "n_missing": int(src_counts.get("missing", 0)),
        "wash_ids": ", ".join(WASH_CHANGEOVER_IDS),
    }
    return msu_by_line, msu_material, info


# --- 8. compute MOQ / APQ coverage (days) --------------------------------------
def _net_1864(sub, plant=PLANT_LOC):
    """Locations in the sourcing sub-tree rooted at plant 1864 for one material.

    A location is in-network when its sourcing chain (location -> sourcing ->
    ...) reaches 1864; the plant node itself is included when it appears as a
    location row (e.g. a plant that also carries forecast). Multi-echelon: a DC
    that sources from another DC which in turn sources from 1864 is in-network.
    """
    parent = dict(zip(sub["location"], sub["sourcing"]))
    res: set[str] = set()
    if plant in parent:                        # 1864 is itself a location node
        res.add(plant)
    for loc in parent:
        cur, seen = loc, set()
        while cur and cur not in seen and cur != "nan":
            seen.add(cur)
            if cur == plant:
                res.add(loc)
                break
            cur = parent.get(cur)
    return res


def compute_coverage(global_network, fcst_all, moq, su_factor, su_source,
                     apq_by_material, base, line_map):
    """MOQ / APQ coverage in days, by SKU / production line / category.

    For each SKU produced at plant 1864 (a min_batch row in M4):
      * 1864-network = the Global_Network sub-tree rooted at 1864 (see _net_1864).
        Forecast is summed over this network only (incl. 1864 if it is a node).
      * wk1-18 forecast (MSU) = Sigma demand qty over the 1864-network for weeks
        1..18 (== 2026-06-29..2026-11-01) x su_factor / 1000.
      * daily forecast (MSU/day) = wk1-18 forecast / 126.
      * MOQ (MSU) = the current production min_batch x su_factor / 1000.
      * APQ (MSU) = simulated avg MSU produced per production run (KPI6).
      * MOQ coverage (days) = MOQ / daily forecast.
      * APQ coverage (days) = APQ / daily forecast.
    Line/category rollups are ratios of sums (Sigma MOQ or Sigma APQ over Sigma
    daily forecast), i.e. demand-weighted, matching the DFC convention. The APQ
    rollup denominator is restricted to SKUs that were actually produced (have an
    APQ) so its numerator and denominator share the same SKU set.
    """
    gn = global_network.copy()
    gn["material"] = gn["material"].astype(str)
    gn["location"] = gn["location"].astype(str).str.strip()
    gn["sourcing"] = gn["sourcing"].astype(str).str.strip()

    net_map = {m: _net_1864(sub) for m, sub in gn.groupby("material")}
    in_net = {(m, loc) for m, net in net_map.items() for loc in net}

    fa = fcst_all.copy()
    fa["material"] = fa["material"].astype(str)
    fa["location"] = fa["location"].astype(str).str.strip()
    fa = fa[[(m, l) in in_net for m, l in zip(fa["material"], fa["location"])]]
    fsum = fa.groupby("material", as_index=False)["fcst_cs"].sum()

    df = moq.copy()
    df["material"] = df["material"].astype(str)
    df = df.merge(fsum, on="material", how="left")
    df["fcst_cs"] = df["fcst_cs"].fillna(0.0)
    df["su_factor"] = df["material"].map(su_factor)
    df["su_source"] = df["material"].map(su_source).fillna("missing")
    df["fcst_msu"] = df["fcst_cs"] * df["su_factor"] / 1000.0
    df["moq_msu"] = df["min_batch"] * df["su_factor"] / 1000.0
    df["daily_fcst_msu"] = df["fcst_msu"] / COVERAGE_DAYS
    df["n_network"] = df["material"].map(
        {m: len(n) for m, n in net_map.items()}).fillna(0).astype("int64")

    apq_map = dict(zip(apq_by_material["material"].astype(str), apq_by_material["apq"]))
    df["apq_msu"] = df["material"].map(apq_map)

    # line = M4 delegate line; category via the shared TOF resolution + line-inference
    line_lookup = dict(zip(line_map["material"].astype(str), line_map["line"]))
    df["line"] = df["material"].map(line_lookup).fillna("Unmapped")
    res = base.drop_duplicates("material")[["material", "line", "category"]].copy()
    res["material"] = res["material"].astype(str)
    mat_cat = dict(zip(res["material"], res["category"]))
    nonu = res[res["category"] != "Unmapped"]
    line_ncat = nonu.groupby("line")["category"].nunique()
    det_lines = line_ncat[line_ncat == 1].index
    line_cat = (nonu[nonu["line"].isin(det_lines)].drop_duplicates("line")
                .set_index("line")["category"].to_dict())
    df["category"] = df["material"].map(mat_cat)
    nmask = df["category"].isna()
    df.loc[nmask, "category"] = df.loc[nmask, "line"].map(line_cat)
    df["category"] = df["category"].fillna("Unmapped")

    # per-SKU coverage (defined only where the SKU has demand in its 1864 network)
    denom = df["daily_fcst_msu"].where(df["daily_fcst_msu"] > 0, np.nan)
    df["moq_coverage"] = (df["moq_msu"] / denom).round(1)
    df["apq_coverage"] = (df["apq_msu"] / denom).round(1)

    n_no_fcst = int((df["fcst_cs"] <= 0).sum())
    n_missing_su_scope = int(((df["fcst_cs"] > 0) & df["su_factor"].isna()).sum())

    cov = df[df["daily_fcst_msu"] > 0].copy()
    cov["apq_msu_f"] = cov["apq_msu"].fillna(0.0)
    cov["daily_fcst_msu_apq"] = np.where(cov["apq_msu"].notna(),
                                         cov["daily_fcst_msu"], 0.0)

    def roll(key):
        g = cov.groupby(key, as_index=False).agg(
            n_sku=("material", "nunique"),
            n_sku_apq=("apq_msu", lambda s: int(s.notna().sum())),
            fcst_msu=("fcst_msu", "sum"),
            daily_fcst_msu=("daily_fcst_msu", "sum"),
            daily_fcst_msu_apq=("daily_fcst_msu_apq", "sum"),
            moq_msu=("moq_msu", "sum"),
            apq_msu=("apq_msu_f", "sum"),
        )
        g["moq_coverage"] = (g["moq_msu"] / g["daily_fcst_msu"].replace(0, np.nan)).round(1)
        g["apq_coverage"] = (g["apq_msu"] / g["daily_fcst_msu_apq"].replace(0, np.nan)).round(1)
        for c in ("fcst_msu", "moq_msu", "apq_msu"):
            g[c] = g[c].round(1)
        g["daily_fcst_msu"] = g["daily_fcst_msu"].round(3)
        return g.drop(columns=["daily_fcst_msu_apq"])

    cov_by_line = roll("line").sort_values("line").reset_index(drop=True)
    cov_by_cat = roll("category").sort_values("category").reset_index(drop=True)

    cov_by_sku = (cov[["material", "line", "category", "su_factor", "su_source",
                       "n_network", "fcst_cs", "fcst_msu", "daily_fcst_msu",
                       "min_batch", "moq_msu", "apq_msu", "moq_coverage",
                       "apq_coverage"]]
                  .rename(columns={"min_batch": "moq_cs"})
                  .sort_values(["line", "material"]).reset_index(drop=True))
    cov_by_sku["fcst_cs"] = cov_by_sku["fcst_cs"].round(1)
    cov_by_sku["fcst_msu"] = cov_by_sku["fcst_msu"].round(2)
    cov_by_sku["daily_fcst_msu"] = cov_by_sku["daily_fcst_msu"].round(4)
    cov_by_sku["moq_msu"] = cov_by_sku["moq_msu"].round(2)
    cov_by_sku["apq_msu"] = cov_by_sku["apq_msu"].round(2)

    tot_moq = float(cov["moq_msu"].sum())
    tot_daily = float(cov["daily_fcst_msu"].sum())
    tot_apq = float(cov["apq_msu_f"].sum())
    tot_daily_apq = float(cov["daily_fcst_msu_apq"].sum())
    info = {
        "days": COVERAGE_DAYS,
        "week_max": COVERAGE_WEEK_MAX,
        "window": "2026-06-29..2026-11-01",
        "n_sku": int(cov["material"].nunique()),
        "n_sku_apq": int(cov["apq_msu"].notna().sum()),
        "moq_coverage_overall": round(tot_moq / tot_daily, 1) if tot_daily else float("nan"),
        "apq_coverage_overall": round(tot_apq / tot_daily_apq, 1) if tot_daily_apq else float("nan"),
        "total_moq_msu": round(tot_moq, 1),
        "total_apq_msu": round(tot_apq, 1),
        "total_fcst_msu": round(float(cov["fcst_msu"].sum()), 1),
        "n_no_fcst": n_no_fcst,
        "n_missing_su_scope": n_missing_su_scope,
    }
    return cov_by_sku, cov_by_line, cov_by_cat, info


# --- chart helper ---------------------------------------------------------------
def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def build_charts(cfr_by_cat, cfr_line_total, changeover, dfc_by_cat, dfc_by_line,
                 ptt_by_month, ptt_main, apq_by_line, msu_by_line,
                 cov_by_line=None, cov_by_cat=None):
    charts = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                             "axes.spines.top": False, "axes.spines.right": False})
        PAL = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#dc2626",
               "#0891b2", "#ca8a04", "#4b5563"]

        # Chart 1: TOF by month split by category
        piv = cfr_by_cat.pivot(index="month", columns="category", values="cfr")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        piv.plot(kind="bar", ax=ax, color=PAL[: piv.shape[1]], width=0.8)
        ax.set_ylabel("TOF"); ax.set_xlabel(""); ax.set_ylim(0, 1.02)
        ax.set_title("TOF by month, by category"); ax.legend(title="category", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["cfr_month_cat"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 2: TOF full-period by line split by category
        piv2 = cfr_line_total.pivot(index="line", columns="category", values="cfr")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        piv2.plot(kind="bar", ax=ax, color=PAL[: piv2.shape[1]], width=0.8)
        ax.set_ylabel("TOF"); ax.set_xlabel(""); ax.set_ylim(0, 1.02)
        ax.set_title("TOF full period, by production line & category"); ax.legend(title="category", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["cfr_line_cat"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 3: changeover count by month stacked by id
        cc = changeover.groupby(["month", "changeover_id"], as_index=False)["changeover_count"].sum()
        pcc = cc.pivot(index="month", columns="changeover_id", values="changeover_count").fillna(0)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        pcc.plot(kind="bar", stacked=True, ax=ax, color=PAL[: pcc.shape[1]], width=0.8)
        ax.set_ylabel("changeover count"); ax.set_xlabel("")
        ax.set_title("Changeover count by month, stacked by changeover id"); ax.legend(title="id", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["co_count_month"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 4: changeover cost by line stacked by id (full period)
        cl = changeover.groupby(["line", "changeover_id"], as_index=False)["changeover_cost"].sum()
        pcl = cl.pivot(index="line", columns="changeover_id", values="changeover_cost").fillna(0)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        pcl.plot(kind="bar", stacked=True, ax=ax, color=PAL[: pcl.shape[1]], width=0.8)
        ax.set_ylabel("changeover cost"); ax.set_xlabel("")
        ax.set_title("Changeover cost by line (full period), stacked by changeover id"); ax.legend(title="id", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["co_cost_line"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 5: changeover cost by month stacked by id
        cm = changeover.groupby(["month", "changeover_id"], as_index=False)["changeover_cost"].sum()
        pcm = cm.pivot(index="month", columns="changeover_id", values="changeover_cost").fillna(0)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        pcm.plot(kind="bar", stacked=True, ax=ax, color=PAL[: pcm.shape[1]], width=0.8)
        ax.set_ylabel("changeover cost"); ax.set_xlabel("")
        ax.set_title("Changeover cost by month, stacked by changeover id"); ax.legend(title="id", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["co_cost_month"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 6: month-end DFC by category
        pivd = dfc_by_cat.pivot(index="month", columns="category", values="dfc_days")
        pivd = pivd.dropna(axis=1, how="all")  # drop categories with no coverage signal
        fig, ax = plt.subplots(figsize=(7, 3.4))
        pivd.plot(kind="bar", ax=ax, color=PAL[: pivd.shape[1]], width=0.8)
        ax.set_ylabel("DFC (days)"); ax.set_xlabel("")
        ax.set_title("Month-end DFC by category (DC inv \u00f7 forward 30-day forecast)")
        ax.legend(title="category", fontsize=8); ax.tick_params(axis="x", rotation=0)
        charts["dfc_month_cat"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 7: month-end DFC by production line (trend)
        pivl = dfc_by_line.pivot(index="month", columns="line", values="dfc_days")
        pivl = pivl.dropna(axis=1, how="all").drop(columns=["Unmapped"], errors="ignore")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        pivl.plot(kind="line", marker="o", ax=ax, color=PAL[: pivl.shape[1]])
        ax.set_ylabel("DFC (days)"); ax.set_xlabel("")
        ax.set_title("Month-end DFC by production line")
        ax.legend(title="line", fontsize=8, ncol=2); ax.tick_params(axis="x", rotation=0)
        charts["dfc_line_month"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 8: production total time by month, stacked production vs changeover
        pm = ptt_by_month.set_index("month")[["prod_hr", "co_hr"]]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        pm.plot(kind="bar", stacked=True, ax=ax, color=["#2563eb", "#ea580c"], width=0.7)
        ax.set_ylabel("hours"); ax.set_xlabel("")
        ax.set_title("Production total time by month (production + changeover)")
        ax.legend(["production", "changeover"], fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["ptt_month_split"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 9: production total time by line, stacked by category (full period)
        lc = ptt_main.groupby(["line", "category"], as_index=False)["total_hr"].sum()
        pivp = lc.pivot(index="line", columns="category", values="total_hr").fillna(0)
        pivp = pivp.loc[pivp.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        pivp.plot(kind="bar", stacked=True, ax=ax, color=PAL[: pivp.shape[1]], width=0.8)
        ax.set_ylabel("total time (hr)"); ax.set_xlabel("")
        ax.set_title("Production total time by line, stacked by category")
        ax.legend(title="category", fontsize=8); ax.tick_params(axis="x", rotation=0)
        charts["ptt_line_cat"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 10: APQ (avg MSU per production run) by line
        ab = apq_by_line.sort_values("apq", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(ab["line"], ab["apq"], color="#2563eb", width=0.7)
        ax.set_ylabel("APQ (MSU / run)"); ax.set_xlabel("")
        ax.set_title("APQ \u2014 avg MSU produced per production run, by line")
        for x, v in zip(range(len(ab)), ab["apq"]):
            ax.text(x, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        charts["apq_line"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 11: wash count vs MSU by line (dual axis) + wash/MSU annotation
        mb = msu_by_line.sort_values("wash_count", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = range(len(mb))
        ax.bar([i - 0.2 for i in x], mb["wash_count"], width=0.4,
               color="#dc2626", label="wash count")
        ax.set_ylabel("wash count (id 2/3)", color="#dc2626")
        ax.tick_params(axis="y", labelcolor="#dc2626")
        ax2 = ax.twinx()
        ax2.bar([i + 0.2 for i in x], mb["msu"], width=0.4, color="#2563eb", label="MSU")
        ax2.set_ylabel("MSU produced", color="#2563eb")
        ax2.tick_params(axis="y", labelcolor="#2563eb"); ax2.grid(False)
        ax.set_xticks(list(x)); ax.set_xticklabels(mb["line"])
        ax.set_title("Wash count vs MSU by line (label = washes / MSU)")
        for i, (w, m, r) in enumerate(zip(mb["wash_count"], mb["msu"], mb["wash_per_msu"])):
            ax2.text(i + 0.2, m, f"{r:.2f}", ha="center", va="bottom", fontsize=8, color="#1e3a8a")
        charts["msu_line"] = fig_to_data_uri(fig); plt.close(fig)

        # Chart 12 & 13: MOQ vs APQ coverage (days) by line and by category
        def _cov_grouped(df, key, title):
            d = df[df[key] != "Unmapped"].sort_values("moq_coverage", ascending=False)
            x = list(range(len(d)))
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.bar([i - 0.2 for i in x], d["moq_coverage"], width=0.4,
                   color="#2563eb", label="MOQ coverage")
            ax.bar([i + 0.2 for i in x], d["apq_coverage"], width=0.4,
                   color="#ea580c", label="APQ coverage")
            ax.set_ylabel("coverage (days)"); ax.set_xlabel("")
            ax.set_xticks(x); ax.set_xticklabels(d[key], rotation=0)
            ax.set_title(title); ax.legend(fontsize=8)
            uri = fig_to_data_uri(fig); plt.close(fig)
            return uri

        if cov_by_line is not None and not cov_by_line.empty:
            charts["cov_line"] = _cov_grouped(
                cov_by_line, "line", "MOQ vs APQ coverage (days) by production line")
        if cov_by_cat is not None and not cov_by_cat.empty:
            charts["cov_cat"] = _cov_grouped(
                cov_by_cat, "category", "MOQ vs APQ coverage (days) by category")

    except Exception as exc:  # noqa: BLE001
        charts["_error"] = str(exc)
    return charts


def main():
    from report_helpers import write_workbook, write_html, write_run_md, update_latest

    run_id = datetime.now().strftime("%Y%m%d-%H%M") + f"-{SCENARIO}-tof-changeover-dfc-prodtime-apq-msu"
    run_dir = ANALYSIS_ROOT / run_id
    extracts = run_dir / "extracts"
    extracts.mkdir(parents=True, exist_ok=True)

    print("Pulling Postgres aggregates ...")
    (orders, ships, line_map, changeover, co_def, stock_me, fcst_weekly, dc_info,
     prod_plan, prd_rate, prod_runs, global_network, fcst_all, moq) = pull_postgres()

    # cap reporting horizon at end of October (drop 2026-11 from every month-keyed table)
    orders = orders[orders["month"] <= MONTH_MAX].reset_index(drop=True)
    ships = ships[ships["month"] <= MONTH_MAX].reset_index(drop=True)
    changeover = changeover[changeover["month"] <= MONTH_MAX].reset_index(drop=True)
    stock_me = stock_me[stock_me["month"] <= MONTH_MAX].reset_index(drop=True)
    prod_plan = prod_plan[prod_plan["month"] <= MONTH_MAX].reset_index(drop=True)
    print(f"Reporting horizon capped at {MONTH_MAX} (2026-11 excluded)")

    materials = sorted(set(orders["material"]) | set(ships["material"]))
    print(f"Materials in TOF universe: {len(materials)}")

    print("Pulling category mapping ...")
    cat, cat_source, cat_note = get_category_map(materials)
    cat_cov = len(set(materials) & set(cat.loc[cat['category_en'].notna(), 'material']))
    print(f"Category source: {cat_source} | coverage {cat_cov}/{len(materials)}")

    base, cfr_main, cfr_by_line, cfr_by_cat, cfr_by_month, cfr_line_total = compute_cfr(
        orders, ships, line_map, cat
    )

    # KPI4: month-end DFC (Days Forward Coverage)
    dfc_base, dfc_main, dfc_by_cat, dfc_by_line, dfc_by_month, dfc_info = compute_dfc(
        stock_me, fcst_weekly, base
    )

    # KPI5: production total time (production + changeover, hours)
    (ptt_base, ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month, ptt_line_total,
     ptt_cat_total, ptt_info) = compute_ptt(prod_plan, prd_rate, changeover, base)

    # SU factors (shared by APQ-in-MSU and the wash/MSU KPI): SUF workbook primary, Databricks fallback
    print("Resolving SU factors (SUF workbook primary, Databricks fallback) ...")
    # union of produced materials and every line-config material: the coverage KPI
    # is defined over each SKU produced at 1864, including any not run this window
    prod_materials = sorted(set(prod_runs["material"].astype(str))
                            | set(moq["material"].astype(str)))
    su_factor, su_source, su_info = get_su_factor_map(prod_materials)
    print(f"SU factor coverage: suf_file {su_info['n_suf']} + databricks {su_info['n_dbx']} "
          f"+ manual {su_info['n_manual']} + missing {su_info['n_missing']} of {len(prod_materials)}")

    # KPI6: APQ (avg MSU per production run)
    apq_by_line, apq_by_material, apq_info = compute_apq(prod_runs, base, su_factor, su_source)

    # KPI7: wash count / MSU by line (SU factor: SUF workbook primary, Databricks fallback)
    msu_by_line, msu_material, msu_info = compute_msu(prod_runs, changeover, su_factor, su_source)

    # KPI8: MOQ / APQ coverage (days) by SKU / production line / category
    cov_by_sku, cov_by_line, cov_by_cat, cov_info = compute_coverage(
        global_network, fcst_all, moq, su_factor, su_source, apq_by_material,
        base, line_map)

    # changeover KPI tables
    co_count = changeover[["month", "line", "changeover_id", "changeover_count"]].sort_values(
        ["month", "line", "changeover_id"]
    )
    co_cost = changeover[["month", "line", "changeover_id", "changeover_cost"]].copy()
    co_cost["changeover_cost"] = co_cost["changeover_cost"].round(1)
    co_cost = co_cost.sort_values(["month", "line", "changeover_id"])

    # --- write extracts ---
    cfr_main.to_csv(extracts / "cfr_by_category_line_month.csv", index=False)
    co_count.to_csv(extracts / "changeover_count_by_line_month_id.csv", index=False)
    co_cost.to_csv(extracts / "changeover_cost_by_line_month_id.csv", index=False)
    cat.to_csv(extracts / "material_category_databricks.csv", index=False)
    line_map.to_csv(extracts / "material_line_map.csv", index=False)
    base.to_csv(extracts / "cfr_material_month_base.csv", index=False)
    cfr_by_line.to_csv(extracts / "cfr_by_line_month.csv", index=False)
    co_def.to_csv(extracts / "changeover_definitions.csv", index=False)

    # per-material category resolution (source of truth for the by-category KPI)
    resolution = (base.groupby(["material", "line", "category", "category_source"], as_index=False)
                  .agg(order_qty=("order_qty", "sum"), shipment_qty=("shipment_qty", "sum"))
                  .sort_values(["category_source", "line", "material"]))
    resolution.to_csv(extracts / "category_resolution.csv", index=False)

    # DFC extracts (KPI4)
    dfc_main.to_csv(extracts / "dfc_by_category_line_month.csv", index=False)
    dfc_by_line.to_csv(extracts / "dfc_by_line_month.csv", index=False)
    dfc_by_cat.to_csv(extracts / "dfc_by_category_month.csv", index=False)
    dfc_by_month.to_csv(extracts / "dfc_by_month.csv", index=False)
    dfc_base.to_csv(extracts / "dfc_material_month_base.csv", index=False)

    # Production-time extracts (KPI5)
    ptt_main.to_csv(extracts / "prodtime_by_category_line_month.csv", index=False)
    ptt_by_line.to_csv(extracts / "prodtime_by_line_month.csv", index=False)
    ptt_by_cat.to_csv(extracts / "prodtime_by_category_month.csv", index=False)
    ptt_by_month.to_csv(extracts / "prodtime_by_month.csv", index=False)
    ptt_line_total.to_csv(extracts / "prodtime_by_line_total.csv", index=False)
    ptt_cat_total.to_csv(extracts / "prodtime_by_category_total.csv", index=False)
    ptt_base.to_csv(extracts / "prodtime_base.csv", index=False)

    # APQ extracts (KPI6) + wash/MSU extracts (KPI7)
    apq_by_line.to_csv(extracts / "apq_by_line.csv", index=False)
    apq_by_material.to_csv(extracts / "apq_by_material.csv", index=False)
    msu_by_line.to_csv(extracts / "wash_msu_by_line.csv", index=False)
    msu_material.to_csv(extracts / "msu_by_material.csv", index=False)

    # Coverage extracts (KPI8)
    cov_by_sku.to_csv(extracts / "coverage_by_sku.csv", index=False)
    cov_by_line.to_csv(extracts / "coverage_by_line.csv", index=False)
    cov_by_cat.to_csv(extracts / "coverage_by_category.csv", index=False)

    # enrichment / coverage stats (material counts + order-qty shares)
    tot_o = float(base["order_qty"].sum())
    by_src = base.groupby("category_source")["order_qty"].sum()

    def share(src):
        return f"{by_src.get(src, 0.0) / tot_o * 100:.1f}%" if tot_o else "0.0%"

    mat_src = base.drop_duplicates("material").groupby("category_source")["material"].count()
    inferred_materials = sorted(base.loc[base["category_source"] == "line-inferred", "material"].unique())
    unmapped_line = sorted(base.loc[base["line"] == "Unmapped", "material"].unique())
    unmapped_cat = sorted(base.loc[base["category"] == "Unmapped", "material"].unique())
    enriched_cov = int(mat_src.get("databricks", 0) + mat_src.get("line-inferred", 0))
    tot_fc = dc_info["dc_demand_total"] + dc_info["plant_demand_total"]
    dfc_plant_share = f"{dc_info['plant_demand_total'] / tot_fc * 100:.1f}%" if tot_fc else "0.0%"
    meta = {
        "run_id_db": RUN_ID, "config_name": CONFIG_NAME, "project": PROJECT, "scenario": SCENARIO,
        "analysis_run_id": run_id, "category_source": cat_source, "category_note": cat_note,
        "category_coverage": f"{cat_cov}/{len(materials)}",
        "category_coverage_enriched": f"{enriched_cov}/{len(materials)}",
        "materials": len(materials),
        "cat_databricks_n": int(mat_src.get("databricks", 0)),
        "cat_inferred_n": int(mat_src.get("line-inferred", 0)),
        "cat_unmapped_n": int(mat_src.get("unmapped", 0)),
        "cat_databricks_share": share("databricks"),
        "cat_inferred_share": share("line-inferred"),
        "cat_unmapped_share": share("unmapped"),
        "inferred_materials": inferred_materials,
        "unmapped_line_materials": unmapped_line,
        "unmapped_category_materials": unmapped_cat,
        # DFC (KPI4) lineage
        "dfc_sim_start": dfc_info["sim_start"],
        "dfc_forward_days": dfc_info["forward_days"],
        "dfc_horizon_end": dfc_info["horizon_end"],
        "dfc_window_days": dfc_info["window_days"],
        "dfc_partial_months": dfc_info["partial_months"],
        "dfc_full_months": dfc_info["full_months"],
        "dfc_month_end": dfc_info["month_end"],
        "dfc_dc_locations_n": len(dc_info["dc_locations"]),
        "dfc_plant_demand_share": dfc_plant_share,
        # Production time (KPI5) lineage
        "ptt_prod_hr_total": ptt_info["prod_hr_total"],
        "ptt_co_hr_total": ptt_info["co_hr_total"],
        "ptt_total_hr": ptt_info["total_hr_total"],
        "ptt_co_share_total": ptt_info["co_share_total"],
        "ptt_months": ptt_info["months"],
        "ptt_mixed_lines": ptt_info["mixed_lines"],
        "ptt_n_lines": ptt_info["n_lines"],
        "ptt_qty_basis": ptt_info["qty_basis"],
        "ptt_month_basis": ptt_info["month_basis"],
        # APQ (KPI6) lineage
        "apq_overall": apq_info["apq_overall"],
        "apq_total_con_qty": apq_info["total_con_qty"],
        "apq_total_msu": apq_info["total_msu"],
        "apq_is_msu": True,
        "apq_total_runs": apq_info["total_runs"],
        "apq_n_materials": apq_info["n_materials"],
        "apq_run_basis": apq_info["run_basis"],
        "apq_qty_basis": apq_info["qty_basis"],
        # wash/MSU (KPI7) lineage
        "msu_total_wash": msu_info["total_wash"],
        "msu_total_msu": msu_info["total_msu"],
        "msu_wash_per_msu_overall": msu_info["wash_per_msu_overall"],
        "msu_wash_ids": msu_info["wash_ids"],
        "msu_n_suf": msu_info["n_suf"],
        "msu_n_dbx": msu_info["n_dbx"],
        "msu_n_manual": msu_info["n_manual"],
        "msu_n_missing": msu_info["n_missing"],
        "msu_suf_path": su_info["suf_path"],
        # MOQ/APQ coverage (KPI8) lineage
        "cov_has": True,
        "cov_days": cov_info["days"],
        "cov_week_max": cov_info["week_max"],
        "cov_window": cov_info["window"],
        "cov_n_sku": cov_info["n_sku"],
        "cov_n_sku_apq": cov_info["n_sku_apq"],
        "cov_moq_overall": cov_info["moq_coverage_overall"],
        "cov_apq_overall": cov_info["apq_coverage_overall"],
        "cov_total_moq_msu": cov_info["total_moq_msu"],
        "cov_total_apq_msu": cov_info["total_apq_msu"],
        "cov_total_fcst_msu": cov_info["total_fcst_msu"],
        "cov_n_no_fcst": cov_info["n_no_fcst"],
        "cov_n_missing_su": cov_info["n_missing_su_scope"],
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    print(f"Category resolution by order qty: databricks {share('databricks')}, "
          f"line-inferred {share('line-inferred')}, unmapped {share('unmapped')}")
    print(f"DFC: DC set {len(dc_info['dc_locations'])} locs | plant demand excluded {dfc_plant_share} "
          f"| full-window months {dfc_info['full_months']} | partial {dfc_info['partial_months']}")
    print(f"ProdTime: production {ptt_info['prod_hr_total']:,.0f} hr + changeover "
          f"{ptt_info['co_hr_total']:,.0f} hr = {ptt_info['total_hr_total']:,.0f} hr "
          f"(changeover {ptt_info['co_share_total']}%) | months {ptt_info['months']} "
          f"| mixed lines {ptt_info['mixed_lines']}")
    print(f"APQ: total MSU {apq_info['total_msu']:,.1f} / {apq_info['total_runs']} runs "
          f"= {apq_info['apq_overall']:,.3f} MSU/run | {apq_info['n_materials']} materials "
          f"| SU missing {apq_info['n_missing_su']}")
    print(f"Wash/MSU: wash {msu_info['total_wash']} (id {msu_info['wash_ids']}) | "
          f"MSU {msu_info['total_msu']:,.1f} | overall wash/MSU {msu_info['wash_per_msu_overall']:.3f} "
          f"| SU factor: suf {su_info['n_suf']} + dbx {su_info['n_dbx']} + manual {su_info['n_manual']} + missing {su_info['n_missing']}")
    print(f"Coverage(KPI8): {cov_info['n_sku']} SKUs (APQ on {cov_info['n_sku_apq']}) | overall "
          f"MOQ cov {cov_info['moq_coverage_overall']} d, APQ cov {cov_info['apq_coverage_overall']} d "
          f"| dropped {cov_info['n_no_fcst']} no-fcst, {cov_info['n_missing_su_scope']} missing-su")
    for _m in ("80859250", "90450569"):
        _r = cov_by_sku[cov_by_sku["material"] == _m]
        if not _r.empty:
            _r = _r.iloc[0]
            print(f"  validate {_m}: fcst {_r['fcst_msu']:.1f} MSU, MOQ {_r['moq_msu']:.1f} MSU, "
                  f"MOQ cov {_r['moq_coverage']} d | APQ {_r['apq_msu']} MSU, APQ cov {_r['apq_coverage']} d")

    charts = build_charts(cfr_by_cat, cfr_line_total, changeover, dfc_by_cat, dfc_by_line,
                          ptt_by_month, ptt_main, apq_by_line, msu_by_line,
                          cov_by_line, cov_by_cat)

    # --- Excel workbook ---
    xlsx = run_dir / f"{PROJECT}_result_summary_{run_id}.xlsx"
    write_workbook(xlsx, meta, cfr_main, cfr_by_line, cfr_by_cat, cfr_by_month,
                   cfr_line_total, co_count, co_cost, co_def,
                   dfc_main, dfc_by_line, dfc_by_cat, dfc_by_month,
                   ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month, ptt_line_total,
                   apq_by_line, apq_by_material, msu_by_line, msu_material,
                   cov_by_sku=cov_by_sku, cov_by_line=cov_by_line, cov_by_cat=cov_by_cat)

    # --- HTML report ---
    html = run_dir / "analysis.html"
    write_html(html, meta, charts, cfr_main, cfr_by_line, cfr_by_cat, cfr_by_month,
               cfr_line_total, co_count, co_cost, co_def, changeover,
               dfc_main, dfc_by_cat, dfc_by_line, dfc_by_month,
               ptt_main, ptt_by_line, ptt_by_cat, ptt_by_month, ptt_line_total,
               apq_by_line, apq_by_material, msu_by_line,
               cov_by_sku=cov_by_sku, cov_by_line=cov_by_line, cov_by_cat=cov_by_cat)

    # --- run.md + LATEST.md ---
    write_run_md(run_dir, meta)
    update_latest(ANALYSIS_ROOT, run_id, meta)

    print("\nDONE")
    print("Run folder:", run_dir)
    print("Workbook  :", xlsx.name)
    print("Report    :", html.name)


if __name__ == "__main__":
    main()

