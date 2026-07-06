"""ChainSight config validator for the xq-vmr baseline-hc scenario.

Implements the chainsight-configure-validation skill (Mode B benchmark:
M4_MaterialLocationLineCfg origins expanded downstream through Global_Network).
DFC soft check (Soft-001) is skipped per user instruction.

Run from repo root with the venv interpreter:
  .\\.venv\\Scripts\\python.exe .\\workspace\\xq-vmr-to-production-202606\\scenarios\\baseline-hc\\config\\config_validation\\validate_config.py
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
# Optional arg: validate a different config dir (used to diff against baseline).
if len(sys.argv) > 1:
    CFG = Path(sys.argv[1]).resolve()
    OUTDIR = CFG / "config_validation"
else:
    CFG = _HERE.parent
    OUTDIR = _HERE
SCENARIO = CFG.parent.name
OUTDIR.mkdir(parents=True, exist_ok=True)
ANCHOR_PLANT = "1864"
SIM_START = date(2026, 6, 29)
SIM_END = date(2026, 11, 1)
SIM_WEEKS = ((SIM_END - SIM_START).days // 7) + 1  # weeks spanned by the period

# ---- tab registry (skill availability families) -------------------------------
MANDATORY = [
    "Global_Network", "Global_LeadTime", "Global_DemandPriority",
    "M1_DemandForecast", "M1_OrderCalendar", "M1_InitialInventory",
    "M1_ForecastError", "M3_SafetyStock", "M5_PushPullModel",
    "M5_DeployConfig", "M6_MaterialMD", "M6_TruckReleaseCon",
    "M6_TruckTypeSpecs", "M6_DeliveryDelayDistribution",
]
CONDITIONAL = [
    "M1_AOConfig", "M1_DPSConfig", "M1_SupplyChoiceConfig",
    "M4_MaterialLocationLineCfg", "M4_LineCapacity", "M4_ProductionReliability",
    "M4_ChangeoverDefinition", "M4_ChangeoverMatrix",
]
OPTIONAL = ["Global_Seed", "Global_SpaceCapacity", "M6_MDQBypassRules", "M6_TruckCapacityPlan"]
ALL_TABS = MANDATORY + CONDITIONAL + OPTIONAL

issues: list[dict] = []
_seq = 0


def add(rule, sev, tab, scope, key, msg, expected, fix, rows=None):
    global _seq
    _seq += 1
    issues.append({
        "id": f"ISSUE-{_seq:03d}", "rule": rule, "sev": sev, "tab": tab,
        "scope": scope, "key": str(key), "msg": msg, "exp": expected,
        "fix": fix, "rows": rows or [],
    })


def load(name):
    p = CFG / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def is_int(s):
    try:
        return float(s) == int(float(s))
    except Exception:
        return False


def is_num(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def chk_date(s):
    try:
        pd.Timestamp(s)
        return True
    except Exception:
        return False


tabs = {n: load(n) for n in ALL_TABS}
maintained = {n for n, df in tabs.items() if df is not None}
na_rules: dict[str, list[str]] = {"Availability": [], "Schema": [],
                                  "Consistency": [], "Hard": [], "Soft": []}

# =============================================================== Availability ===
for t in MANDATORY:
    if t not in maintained:
        add("Availability-001", "ERROR", t, "tab", "-",
            "Mandatory tab missing", "Tab present", f"Provide {t}.csv")
# Conditional: M4 module is enabled (production scenario) -> M4 tabs expected.
M4_ENABLED = "M4_MaterialLocationLineCfg" in maintained
for t in ["M4_MaterialLocationLineCfg", "M4_LineCapacity",
          "M4_ProductionReliability", "M4_ChangeoverDefinition", "M4_ChangeoverMatrix"]:
    if M4_ENABLED and t not in maintained:
        add("Availability-002", "ERROR", t, "tab", "-",
            "Conditional tab missing (M4 enabled)", "Tab present", f"Provide {t}.csv")
for t in ["M1_DPSConfig", "M1_SupplyChoiceConfig"]:
    na_rules["Availability"].append(f"Availability-002 ({t} module disabled)")

# =================================================================== Schema ====
# (required cols, not fully empty, date format, numeric, key not null)
SCHEMA = {
    "Global_Network": dict(keys=["material", "location", "sourcing", "location_type"],
                           dates=["eff_from", "eff_to"], nums=[]),
    "Global_LeadTime": dict(keys=["sending", "receiving"], dates=[],
                            nums=["PDT", "GR", "MCT", "OTD"]),
    "Global_DemandPriority": dict(keys=["demand_element"], dates=[], nums=["priority"]),
    "M1_DemandForecast": dict(keys=["week", "material", "location"], dates=[],
                              nums=["quantity"]),
    "M1_InitialInventory": dict(keys=["material", "location"], dates=[], nums=["quantity"]),
    "M1_ForecastError": dict(keys=["material", "location", "order_type"], dates=[],
                             nums=["error_std_percent"]),
    "M1_OrderCalendar": dict(keys=["date"], dates=["date"], nums=[]),
    "M3_SafetyStock": dict(keys=["material", "location", "date"], dates=["date"],
                           nums=["safety_stock_qty"]),
    "M1_AOConfig": dict(keys=["material", "location"], dates=[],
                        nums=["advance_days", "ao_percent"]),
    "M4_MaterialLocationLineCfg": dict(keys=["material", "location", "delegate_line"],
                                       dates=[],
                                       nums=["prd_rate", "min_batch", "rv", "ptf", "lsk", "day", "MCT"]),
    "M4_LineCapacity": dict(keys=["location", "line", "date"], dates=["date"],
                            nums=["capacity"]),
    "M4_ChangeoverDefinition": dict(keys=["changeover_id", "line"], dates=[],
                                    nums=["time", "cost", "mu_loss"]),
    "M4_ChangeoverMatrix": dict(keys=["from_material", "to_material", "changeover_id"],
                                dates=[], nums=[]),
    "M4_ProductionReliability": dict(keys=["location", "line"], dates=[], nums=[]),
    "M5_PushPullModel": dict(keys=["material", "sending", "model"], dates=[], nums=[]),
    "M5_DeployConfig": dict(keys=["material", "sending", "receiving"], dates=[],
                            nums=["moq", "rv"]),
    "M6_MaterialMD": dict(keys=["material"], dates=[],
                          nums=["demand_unit_to_weight", "demand_unit_to_volume"]),
    "M6_TruckReleaseCon": dict(keys=["sending", "receiving", "truck_type"], dates=[],
                               nums=["WFR", "VFR"]),
    "M6_TruckTypeSpecs": dict(keys=["truck_type"], dates=[],
                              nums=["capacity_qty_in_weight", "capacity_qty_in_volume"]),
    "M6_DeliveryDelayDistribution": dict(keys=["sending", "receiving"], dates=[],
                                         nums=["delay_days", "probability"]),
}
for t, spec in SCHEMA.items():
    df = tabs.get(t)
    if df is None:
        continue
    for k in spec["keys"]:
        if k not in df.columns:
            add("Schema-001", "ERROR", t, k, "-", "Required column missing",
                "Column present", f"Add column {k}")
            continue
        col = df[k].astype(str).str.strip()
        # Schema-002: column not fully empty
        if (col == "").all():
            add("Schema-002", "ERROR", t, k, f"{len(df)} rows", "Required column fully empty",
                "Column has values", f"Populate {k}")
        # Schema-005: key not null
        blank = col == ""
        # Documented exception: Plant nodes legitimately have empty sourcing
        # (top-of-network self-WIP); owned by Hard-Global_Network-002.
        if t == "Global_Network" and k == "sourcing":
            blank = blank & (df["location_type"].astype(str).str.strip().str.lower() != "plant")
        if blank.any():
            bad_rows = (df.index[blank] + 2).tolist()[:50]
            add("Schema-005", "ERROR", t, k, f"{int(blank.sum())} rows",
                "Key field null", "Key not null", f"Fill {k}",
                rows=[{"Source Excel Row": r, "Field": k} for r in bad_rows])
    for d in spec["dates"]:
        if d in df.columns:
            m = ~df[d].map(chk_date) & (df[d].astype(str).str.strip() != "")
            # M6_DeliveryDelayDistribution date may be 'ALL'
            if t == "M6_DeliveryDelayDistribution":
                m &= df[d].astype(str).str.strip().str.upper() != "ALL"
            if m.any():
                add("Schema-003", "ERROR", t, d, ", ".join(df.loc[m, d].unique()[:5]),
                    "Date not parseable", "YYYY-MM-DD", "Fix date values")
    for c in spec["nums"]:
        if c in df.columns:
            m = ~df[c].map(is_num) & (df[c].astype(str).str.strip() != "")
            if m.any():
                add("Schema-004", "ERROR", t, c, ", ".join(df.loc[m, c].unique()[:5]),
                    "Numeric not parseable", "Valid number", f"Fix {c}")

# =========================================== Mode B benchmark construction =====
gn = tabs["Global_Network"]
mlc = tabs.get("M4_MaterialLocationLineCfg")
bench_nodes: set[tuple[str, str]] = set()   # (material, location)
bench_mats: set[str] = set()
bench_lanes: set[tuple[str, str]] = set()   # (sending, receiving) along reachable chains

if gn is not None and mlc is not None:
    # per-material downstream edges: sourcing -> location
    edges_by_mat: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in gn.itertuples():
        edges_by_mat[r.material][r.sourcing].add(r.location)
    origins_by_mat: dict[str, set] = defaultdict(set)
    for r in mlc.itertuples():
        origins_by_mat[r.material].add(r.location)
    bench_mats = set(origins_by_mat)
    for mat, origins in origins_by_mat.items():
        edges = edges_by_mat.get(mat, {})
        seen = set(origins)
        q = deque(origins)
        for o in origins:
            bench_nodes.add((mat, o))
        while q:
            cur = q.popleft()
            for nx in edges.get(cur, ()):
                bench_nodes.add((mat, nx))
                if cur != nx:
                    bench_lanes.add((cur, nx))
                if nx not in seen:
                    seen.add(nx)
                    q.append(nx)

bench_loc_by_mat = defaultdict(set)
for m, l in bench_nodes:
    bench_loc_by_mat[m].add(l)
# DC nodes = benchmark nodes that are NOT the anchor plant (need demand-side master data)
dc_nodes = {(m, l) for (m, l) in bench_nodes if l != ANCHOR_PLANT}

# =============================================================== Consistency ===
def node_set(tab, mcol, lcol):
    df = tabs.get(tab)
    if df is None:
        return set()
    return {(r[mcol], r[lcol]) for _, r in df[[mcol, lcol]].iterrows()}


# Consistency-001: every benchmark (material, location) present where demand-side
# tabs are required. Sourcing-only/plant nodes are NOT required to carry demand,
# initial inventory, or safety stock -> compare against DC nodes only for those.
demand_required = {
    "M1_DemandForecast": ("material", "location"),
    "M1_InitialInventory": ("material", "location"),
    "M1_ForecastError": ("material", "location"),
    "M3_SafetyStock": ("material", "location"),
}
for tab, (mc, lc) in demand_required.items():
    have = node_set(tab, mc, lc)
    miss = dc_nodes - have
    if miss:
        sample = sorted(miss)[:5]
        add("Consistency-001", "ERROR", tab, "material-location (DC nodes)",
            f"{len(miss)} of {len(dc_nodes)} missing e.g. {sample}",
            "Benchmark DC node absent", "Every benchmark DC node present",
            "Add missing material-location rows",
            rows=[{"material": m, "location": l} for m, l in sorted(miss)])
# M5_DeployConfig may contain MORE than benchmark but not less (DC lanes)
dc = tabs.get("M5_DeployConfig")
if dc is not None:
    have = {(r["material"], r["receiving"]) for _, r in dc.iterrows()}
    miss = dc_nodes - have
    if miss:
        add("Consistency-001", "ERROR", "M5_DeployConfig", "material-receiving (DC nodes)",
            f"{len(miss)} of {len(dc_nodes)} missing e.g. {sorted(miss)[:5]}",
            "Benchmark DC node absent from deploy config",
            "Every benchmark DC node present (extra allowed)",
            "Add deploy rows",
            rows=[{"material": m, "receiving": l} for m, l in sorted(miss)])

# Consistency-002: material coverage. Master-data tabs may have MORE, not less.
mat_required = {
    "M1_DemandForecast": "material", "M1_InitialInventory": "material",
    "M1_ForecastError": "material", "M3_SafetyStock": "material",
    "M5_DeployConfig": "material", "M6_MaterialMD": "material",
}
for tab, mc in mat_required.items():
    df = tabs.get(tab)
    if df is None:
        continue
    have = set(df[mc])
    miss = bench_mats - have
    if miss:
        add("Consistency-002", "ERROR", tab, "material",
            f"{len(miss)} of {len(bench_mats)} missing e.g. {sorted(miss)[:5]}",
            "Benchmark material absent", "Every benchmark material present (extra allowed)",
            "Add material rows",
            rows=[{"material": m} for m in sorted(miss)])

# Consistency-003: sending-receiving lanes. LeadTime/Deploy/TruckRelease >= benchmark.
lane_required = {
    "Global_LeadTime": ("sending", "receiving"),
    "M5_DeployConfig": ("sending", "receiving"),
    "M6_TruckReleaseCon": ("sending", "receiving"),
}
for tab, (sc, rc) in lane_required.items():
    df = tabs.get(tab)
    if df is None:
        continue
    have = {(r[sc], r[rc]) for _, r in df.iterrows()}
    miss = bench_lanes - have
    if miss:
        add("Consistency-003", "ERROR", tab, "sending-receiving",
            f"{len(miss)} of {len(bench_lanes)} missing e.g. {sorted(miss)[:5]}",
            "Benchmark lane absent", "Every benchmark lane present (extra allowed)",
            "Add lane rows",
            rows=[{"sending": s, "receiving": r} for s, r in sorted(miss)])

# Consistency-004: location in M3/M4/M5PushPull must exist in Global_Network loc/sourcing
gn_locs = set()
if gn is not None:
    gn_locs = set(gn["location"]) | set(gn["sourcing"])
loc_tabs = {
    "M3_SafetyStock": "location", "M4_MaterialLocationLineCfg": "location",
    "M4_LineCapacity": "location", "M4_ProductionReliability": "location",
    "M5_PushPullModel": "sending",
}
for tab, lc in loc_tabs.items():
    df = tabs.get(tab)
    if df is None:
        continue
    bad = sorted(set(df[lc]) - gn_locs)
    if bad:
        add("Consistency-004", "ERROR", tab, lc, f"{len(bad)} e.g. {bad[:5]}",
            "Location not in Global_Network", "location in Global_Network loc/sourcing",
            "Add network node or fix location")

# Consistency-005: M4 delegate_line-location must match LineCapacity & ProductionReliability
if mlc is not None:
    m4_ll = {(r["location"], r["delegate_line"]) for _, r in mlc.iterrows()}
    for tab in ["M4_LineCapacity", "M4_ProductionReliability"]:
        df = tabs.get(tab)
        if df is None:
            continue
        ll = {(r["location"], r["line"]) for _, r in df.iterrows()}
        miss = m4_ll - ll
        if miss:
            add("Consistency-005", "ERROR", tab, "location-line",
                f"{len(miss)} e.g. {sorted(miss)[:5]}",
                "M4 delegate line-location missing", "Each M4 line-location present",
                "Add line-location rows")

# Consistency-006: M4_ChangeoverDefinition.changeover_id vs M4_ChangeoverMatrix.changeover_id
cd = tabs.get("M4_ChangeoverDefinition")
cm = tabs.get("M4_ChangeoverMatrix")
if cd is not None and cm is not None:
    def_ids = set(cd["changeover_id"])
    mtx_ids = set(cm["changeover_id"])
    only_mtx = sorted(mtx_ids - def_ids)
    if only_mtx:
        add("Consistency-006", "ERROR", "M4_ChangeoverMatrix", "changeover_id",
            f"{only_mtx[:10]}", "Matrix uses changeover_id absent from Definition",
            "Every matrix changeover_id defined", "Add definition rows")

# Consistency-007: truck_type in M6_TruckReleaseCon must exist in M6_TruckTypeSpecs
tr = tabs.get("M6_TruckReleaseCon")
ts = tabs.get("M6_TruckTypeSpecs")
if tr is not None and ts is not None:
    bad = sorted(set(tr["truck_type"]) - set(ts["truck_type"]))
    if bad:
        add("Consistency-007", "ERROR", "M6_TruckReleaseCon", "truck_type",
            f"{bad[:5]}", "truck_type not in TruckTypeSpecs",
            "Each truck_type specified", "Add truck spec rows")

# ==================================================================== Hard =====
# Hard-Global_Network-001: sourcing uniqueness over overlapping periods
if gn is not None:
    g = gn.copy()
    g["ef"] = pd.to_datetime(g["eff_from"], errors="coerce")
    g["et"] = pd.to_datetime(g["eff_to"], errors="coerce")
    for (m, loc), grp in g.groupby(["material", "location"]):
        rows = grp[["sourcing", "ef", "et"]].dropna().values.tolist()
        flagged = False
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                s1, f1, t1 = rows[i]
                s2, f2, t2 = rows[j]
                if s1 != s2 and f1 <= t2 and f2 <= t1:
                    add("Hard-Global_Network-001", "ERROR", "Global_Network", "sourcing",
                        f"{m} @ {loc}", f"Overlapping sourcing {s1} vs {s2}",
                        "<=1 sourcing per material-location-period", "Resolve duplicate sourcing")
                    flagged = True
                    break
            if flagged:
                break

# Hard-Global_Network-002: top-of-network demand node needs self Plant WIP entry
fc = tabs.get("M1_DemandForecast")
if gn is not None and fc is not None:
    # topmost = location that never appears as a 'location' with a different sourcing
    replenished = {r.location for r in gn.itertuples() if r.sourcing != r.location}
    fc_locs = set(fc["location"])
    self_plant = {(r.material, r.location) for r in gn.itertuples()
                  if r.sourcing == r.location and str(r.location_type).strip().lower() == "plant"}
    miss = []
    for m, l in dc_nodes | bench_nodes:
        if l in fc_locs and l not in replenished and (m, l) not in self_plant:
            # only flag plant-level top nodes that carry forecast
            if l == ANCHOR_PLANT:
                miss.append((m, l))
    miss = sorted(set(miss))
    if miss:
        add("Hard-Global_Network-002", "ERROR", "Global_Network", "plant WIP node",
            f"{len(miss)} e.g. {miss[:5]}",
            "Top-of-network demand node lacks Plant self-sourcing WIP entry",
            "Every top node with forecast has Plant single-point entry",
            "Add self-sourcing Plant rows",
            rows=[{"material": m, "location": l} for m, l in miss])

# Hard-Global_LeadTime-001/002/003
lt = tabs.get("Global_LeadTime")
bypass = tabs.get("M6_MDQBypassRules")
if lt is not None:
    for _, r in lt.iterrows():
        lane = f"{r['sending']}->{r['receiving']}"
        for c in ["PDT", "OTD", "GR", "MCT"]:
            if r[c].strip() == "" or not is_int(r[c]):
                add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", c, lane,
                    f"{c}={r[c]} not integer/empty", "PDT/OTD/GR/MCT integer", f"Fix {c}")
        if is_num(r["OTD"]) and float(r["OTD"]) < 0:
            add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", "OTD", lane,
                f"OTD={r['OTD']} < 0", "OTD >= 0", "Fix OTD")
        if is_num(r["GR"]) and float(r["GR"]) < 0:
            add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", "GR", lane,
                f"GR={r['GR']} < 0", "GR >= 0", "Fix GR")
        if is_num(r["MCT"]) and float(r["MCT"]) <= 0:
            add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", "MCT", lane,
                f"MCT={r['MCT']} <= 0", "MCT > 0", "Fix MCT")
        if is_num(r["OTD"]) and is_num(r["PDT"]) and float(r["OTD"]) > float(r["PDT"]):
            add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", "OTD/PDT", lane,
                f"OTD {r['OTD']} > PDT {r['PDT']}", "OTD <= PDT", "Fix OTD/PDT")
    dup = lt.groupby(["sending", "receiving"]).size()
    for (s, rcv), n in dup[dup > 1].items():
        add("Hard-Global_LeadTime-002", "ERROR", "Global_LeadTime", "sending-receiving",
            f"{s}->{rcv}", f"{n} duplicate rows", "Exactly one row per lane", "Deduplicate")
    # Hard-Global_LeadTime-003: zero-slack lane requires positive-wait bypass
    bypass_lanes = set()
    if bypass is not None and "condition_logic" in bypass.columns:
        for _, br in bypass.iterrows():
            cl = str(br.get("condition_logic", ""))
            if "waiting_days" in cl and ">" in cl:
                bypass_lanes.add((br["sending"], br["receiving"]))
    for _, r in lt.iterrows():
        if is_num(r["OTD"]) and is_num(r["PDT"]) and float(r["OTD"]) == float(r["PDT"]):
            if (r["sending"], r["receiving"]) not in bypass_lanes:
                add("Hard-Global_LeadTime-003", "ERROR", "Global_LeadTime", "OTD=PDT lane",
                    f"{r['sending']}->{r['receiving']}",
                    "Zero-slack lane without waiting_days bypass",
                    "OTD=PDT lane has waiting_days>x bypass", "Add bypass rule")

# Hard-Global_DemandPriority-001/002/003/004
dp = tabs.get("Global_DemandPriority")
if dp is not None:
    for _, r in dp.iterrows():
        if not (is_int(r["priority"]) and int(float(r["priority"])) >= 1):
            add("Hard-Global_DemandPriority-002", "ERROR", "Global_DemandPriority",
                "priority", r["demand_element"], f"priority={r['priority']}",
                "integer >= 1", "Fix priority")
    elems = set(dp["demand_element"])
    for need in ["push replenishment", "soft push replenishment"]:
        if need not in elems:
            add("Hard-Global_DemandPriority-003", "ERROR", "Global_DemandPriority",
                "demand_element", need, f"'{need}' missing",
                "push & soft push replenishment present", f"Add '{need}'")
    for e in elems:
        if "ao" in e.split():
            add("Hard-Global_DemandPriority-004", "ERROR", "Global_DemandPriority",
                "demand_element", e, "lowercase 'ao' token", "uppercase AO", "Use 'AO'")
    # Hard-001 layered family coverage: network depth = longest sourcing chain
    if gn is not None:
        depth = 1
        edges = defaultdict(set)
        for r in gn.itertuples():
            if r.sourcing != r.location:
                edges[r.location].add(r.sourcing)

        def chain_len(node, seen):
            best = 1
            for up in edges.get(node, ()):
                if up not in seen:
                    best = max(best, 1 + chain_len(up, seen | {up}))
            return best
        for n in list(edges.keys()):
            depth = max(depth, chain_len(n, {n}))
        prefix = "net demand for " * (depth - 1)
        for fam in ["normal", "AO", "customer", "forecast", "safety"]:
            need = (prefix + fam).strip()
            if depth >= 1 and need not in elems:
                add("Hard-Global_DemandPriority-001", "ERROR", "Global_DemandPriority",
                    "demand_element", need,
                    f"layered family missing at network depth {depth}",
                    "families cover max network depth", f"Add '{need}'")

# Hard quantity / range checks
def qty_check(tab, col, rule, keycols, lo=0.0, lo_strict=False, hi=None):
    df = tabs.get(tab)
    if df is None or col not in df.columns:
        return
    bad_rows = []
    for idx, r in df.iterrows():
        if not is_num(r[col]):
            continue
        v = float(r[col])
        ok = (v > lo) if lo_strict else (v >= lo)
        if hi is not None:
            ok = ok and v <= hi
        if not ok:
            bad_rows.append((idx, r))
    if bad_rows:
        key = " / ".join(keycols)
        sample = bad_rows[0][1]
        add(rule, "ERROR", tab, col,
            " / ".join(str(sample[k]) for k in keycols if k in df.columns),
            f"{len(bad_rows)} rows out of range (e.g. {col}={sample[col]})",
            f"{col} bound ({'>' if lo_strict else '>='}{lo}"
            + (f", <= {hi}" if hi is not None else "") + ")",
            f"Fix {col}",
            rows=[{"Source Excel Row": i + 2, **{k: r[k] for k in keycols if k in df.columns},
                   col: r[col]} for i, r in bad_rows[:50]])


qty_check("M1_InitialInventory", "quantity", "Hard-M1_InitialInventory-001",
          ["material", "location"])
qty_check("M1_DemandForecast", "quantity", "Hard-M1_DemandForecast-002",
          ["material", "location", "week"])
qty_check("M1_ForecastError", "error_std_percent", "Hard-M1_ForecastError-001",
          ["material", "location"])
qty_check("M3_SafetyStock", "safety_stock_qty", "Hard-M3_SafetyStock-002",
          ["material", "location", "date"])
qty_check("M6_MaterialMD", "demand_unit_to_weight", "Hard-M6_MaterialMD-001",
          ["material"], lo=0.0, lo_strict=True)
qty_check("M6_MaterialMD", "demand_unit_to_volume", "Hard-M6_MaterialMD-001",
          ["material"], lo=0.0, lo_strict=True)
qty_check("M6_TruckTypeSpecs", "capacity_qty_in_weight", "Hard-M6_TruckTypeSpecs-001",
          ["truck_type"], lo=0.0, lo_strict=True)
qty_check("M6_TruckTypeSpecs", "capacity_qty_in_volume", "Hard-M6_TruckTypeSpecs-001",
          ["truck_type"], lo=0.0, lo_strict=True)
qty_check("M4_LineCapacity", "capacity", "Hard-M4_LineCapacity-002",
          ["location", "line", "date"], lo=0.0, lo_strict=True, hi=24)

# Hard-M1_AOConfig-001/002
ao = tabs.get("M1_AOConfig")
if ao is not None:
    for _, r in ao.iterrows():
        if is_num(r["ao_percent"]) and not (0 <= float(r["ao_percent"]) <= 1):
            add("Hard-M1_AOConfig-001", "ERROR", "M1_AOConfig", "ao_percent",
                f"{r['material']} @ {r['location']}", f"ao_percent={r['ao_percent']}",
                "0 <= ao_percent <= 1", "Fix ao_percent")
    ao["aop"] = pd.to_numeric(ao["ao_percent"], errors="coerce").fillna(0)
    tot = ao.groupby(["material", "location"])["aop"].sum()
    for (m, l), v in tot[tot > 1.0000001].items():
        add("Hard-M1_AOConfig-002", "ERROR", "M1_AOConfig", "ao_percent sum",
            f"{m} @ {l}", f"sum(ao_percent)={v:.4f} > 1", "sum <= 1 per material-location",
            "Reduce ao_percent")

# Hard-M4_MaterialLocationLineCfg-001/002/003
if mlc is not None:
    for _, r in mlc.iterrows():
        ml = f"{r['material']} @ {r['location']}"
        for c, strict in [("prd_rate", True), ("min_batch", True), ("rv", True),
                          ("lsk", True), ("day", True), ("MCT", True), ("ptf", False)]:
            if not is_num(r[c]):
                add("Hard-M4_MaterialLocationLineCfg-002", "ERROR",
                    "M4_MaterialLocationLineCfg", c, ml, f"{c}={r[c]} not numeric",
                    "numeric", f"Fix {c}")
                continue
            v = float(r[c])
            if strict and v <= 0:
                add("Hard-M4_MaterialLocationLineCfg-002", "ERROR",
                    "M4_MaterialLocationLineCfg", c, ml, f"{c}={r[c]} <= 0",
                    f"{c} > 0", f"Fix {c}")
            if not strict and v < 0:
                add("Hard-M4_MaterialLocationLineCfg-002", "ERROR",
                    "M4_MaterialLocationLineCfg", c, ml, f"{c}={r[c]} < 0",
                    f"{c} >= 0", f"Fix {c}")
        if is_num(r["day"]) and is_num(r["lsk"]) and float(r["day"]) > float(r["lsk"]):
            add("Hard-M4_MaterialLocationLineCfg-001", "ERROR",
                "M4_MaterialLocationLineCfg", "day/lsk", ml,
                f"day {r['day']} > lsk {r['lsk']}", "day <= lsk", "Fix day/lsk")
    dele = mlc.groupby(["material", "location"])["delegate_line"].nunique()
    for (m, l), n in dele[dele > 1].items():
        add("Hard-M4_MaterialLocationLineCfg-003", "ERROR",
            "M4_MaterialLocationLineCfg", "delegate_line", f"{m} @ {l}",
            f"{n} different delegate_line", "unique delegate_line per material-location",
            "Resolve duplicate delegation")

# Hard-M4_ChangeoverMatrix-001/002: same-line pairs + completeness n^2-n
if mlc is not None and cm is not None:
    line_of = {}  # material -> set(lines)
    for r in mlc.itertuples():
        line_of.setdefault(r.material, set()).add(r.delegate_line)
    line_mats = defaultdict(set)
    for r in mlc.itertuples():
        line_mats[r.delegate_line].add(r.material)
    # 001: from/to must share a line
    bad_pairs = 0
    for r in cm.itertuples():
        lf = line_of.get(r.from_material, set())
        lt_ = line_of.get(r.to_material, set())
        if not (lf & lt_):
            bad_pairs += 1
    if bad_pairs:
        add("Hard-M4_ChangeoverMatrix-001", "ERROR", "M4_ChangeoverMatrix",
            "from/to material", f"{bad_pairs} pairs",
            "changeover pair not delegated to same line",
            "each pair shares a line", "Remove cross-line pairs")
    # 002: completeness per line = n^2 - n
    cm_pairs_by_line = defaultdict(set)
    pairset = {(r.from_material, r.to_material) for r in cm.itertuples()}
    for line, mats in line_mats.items():
        n = len(mats)
        expected = n * n - n
        present = sum(1 for a in mats for b in mats if a != b and (a, b) in pairset)
        if present != expected:
            add("Hard-M4_ChangeoverMatrix-002", "ERROR", "M4_ChangeoverMatrix",
                f"line {line}", f"{present}/{expected} pairs",
                f"incomplete changeover matrix (n={n})",
                "n^2-n pairs per line", "Add missing changeover pairs")

# Hard-M5_PushPullModel-001
pp = tabs.get("M5_PushPullModel")
if pp is not None:
    bad = sorted(set(pp["model"]) - {"push", "soft push"})
    if bad:
        add("Hard-M5_PushPullModel-001", "ERROR", "M5_PushPullModel", "model",
            f"{bad[:5]}", "invalid model value", "push or soft push", "Fix model")

# Hard-M5_DeployConfig-001 numeric
if dc is not None:
    for c in ["moq", "rv"]:
        m = ~dc[c].map(is_num) & (dc[c].astype(str).str.strip() != "")
        if m.any():
            add("Hard-M5_DeployConfig-001", "ERROR", "M5_DeployConfig", c,
                ", ".join(dc.loc[m, c].unique()[:5]), "non-numeric", "numeric", f"Fix {c}")

# Hard-M6_TruckReleaseCon-001 fill-rate
if tr is not None:
    for _, r in tr.iterrows():
        for c in ["WFR", "VFR"]:
            if is_num(r[c]) and not (0 < float(r[c]) < 1):
                add("Hard-M6_TruckReleaseCon-001", "ERROR", "M6_TruckReleaseCon", c,
                    f"{r['sending']}->{r['receiving']}", f"{c}={r[c]}", "0 < rate < 1", f"Fix {c}")

# Hard date coverage: M1_OrderCalendar, M3_SafetyStock, M4_LineCapacity
def missing_dates(have_dates):
    need = []
    d = SIM_START
    while d <= SIM_END:
        if str(d) not in have_dates:
            need.append(str(d))
        d += timedelta(days=1)
    return need


oc = tabs.get("M1_OrderCalendar")
if oc is not None:
    have = {str(pd.Timestamp(d).date()) for d in oc["date"] if chk_date(d)}
    need = missing_dates(have)
    if need:
        add("Hard-M1_OrderCalendar-001", "ERROR", "M1_OrderCalendar", "date",
            f"{len(need)} missing e.g. {need[:3]}", "calendar gaps in sim period",
            "cover full sim period", "Add missing dates")

ss = tabs.get("M3_SafetyStock")
if ss is not None:
    ss_dates = {str(pd.Timestamp(d).date()) for d in ss["date"].unique() if chk_date(d)}
    glob_need = missing_dates(ss_dates)
    if glob_need:
        add("Hard-M3_SafetyStock-001", "ERROR", "M3_SafetyStock", "date",
            f"{len(glob_need)} dates absent from whole tab e.g. {glob_need[:3]}",
            "safety stock date gaps in sim period",
            "every material-location daily coverage", "Extend safety stock dates")

lc = tabs.get("M4_LineCapacity")
if lc is not None:
    cap_dates = {str(pd.Timestamp(d).date()) for d in lc["date"].unique() if chk_date(d)}
    need = missing_dates(cap_dates)
    if need:
        add("Hard-M4_LineCapacity-001", "ERROR", "M4_LineCapacity", "date",
            f"{len(need)} missing e.g. {need[:3]}", "line capacity date gaps",
            "cover full sim period", "Add capacity dates")

# Hard-M1_DemandForecast-001 week coverage (>= sim weeks + 2)
if fc is not None:
    weeks = pd.to_numeric(fc["week"], errors="coerce").dropna().astype(int)
    if len(weeks):
        wmin, wmax = weeks.min(), weeks.max()
        need_max = SIM_WEEKS + 2
        if wmin > 1 or wmax < need_max:
            add("Hard-M1_DemandForecast-001", "ERROR", "M1_DemandForecast", "week",
                f"weeks {wmin}..{wmax}", f"coverage < 1..{need_max}",
                f"week 1..{need_max} (sim weeks {SIM_WEEKS}+2)", "Extend forecast weeks")

# ==================================================================== Soft =====
na_rules["Soft"].append("Soft-001 (DFC init-inventory check skipped per user)")
# Soft-002: min_batch too large vs week1-4 daily avg demand
if mlc is not None and fc is not None:
    fcw = fc.copy()
    fcw["w"] = pd.to_numeric(fcw["week"], errors="coerce")
    fcw["q"] = pd.to_numeric(fcw["quantity"], errors="coerce").fillna(0)
    early = fcw[(fcw["w"] >= 1) & (fcw["w"] <= 4)]
    daily_avg = early.groupby("material")["q"].sum() / (4 * 7)
    big = []
    for r in mlc.itertuples():
        da = daily_avg.get(r.material, 0)
        if is_num(r.min_batch) and da > 0 and float(r.min_batch) > 5 * da * 7:
            big.append((r.material, r.location, float(r.min_batch), round(da, 1)))
    if big:
        add("Soft-002", "WARNING", "M4_MaterialLocationLineCfg", "min_batch",
            f"{len(big)} materials e.g. {[b[0] for b in big[:3]]}",
            "min_batch large vs wk1-4 daily avg demand (>5 weeks of demand)",
            "min_batch reasonable vs demand", "Review batch sizing",
            rows=[{"material": m, "location": l, "min_batch": mb, "daily_avg_demand": da}
                  for m, l, mb, da in big[:100]])
# Soft-003: lead-time slack vs bypass wait
if lt is not None and bypass is not None and "condition_logic" in bypass.columns:
    wait_x = {}
    for _, br in bypass.iterrows():
        cl = str(br.get("condition_logic", ""))
        if "waiting_days" in cl and ">" in cl:
            try:
                x = float(cl.split(">")[1].strip().split()[0])
                wait_x[(br["sending"], br["receiving"])] = x
            except Exception:
                pass
    risky = []
    for _, r in lt.iterrows():
        key = (r["sending"], r["receiving"])
        if key in wait_x and is_num(r["OTD"]) and is_num(r["PDT"]):
            slack = float(r["PDT"]) - float(r["OTD"])
            if wait_x[key] > slack:
                risky.append((key, slack, wait_x[key]))
    if risky:
        add("Soft-003", "WARNING", "Global_LeadTime", "lead-time slack vs bypass",
            f"{len(risky)} lanes e.g. {[r[0] for r in risky[:3]]}",
            "bypass waiting_days exceeds lead-time slack",
            "waiting_days x <= PDT-OTD", "Review bypass/lead-time")
# Soft-004: safety stock excessive vs demand (>5x weekly forecast)
if ss is not None and fc is not None:
    fcw = fc.copy()
    fcw["w"] = pd.to_numeric(fcw["week"], errors="coerce")
    fcw["q"] = pd.to_numeric(fcw["quantity"], errors="coerce").fillna(0)
    wk_qty = fcw.groupby(["material", "location", "w"])["q"].sum().to_dict()
    sdf = ss.copy()
    sdf["d"] = pd.to_datetime(sdf["date"], errors="coerce")
    sdf["q"] = pd.to_numeric(sdf["safety_stock_qty"], errors="coerce").fillna(0)
    sdf["wk"] = ((sdf["d"] - pd.Timestamp(SIM_START)).dt.days // 7) + 1
    over = 0
    for r in sdf.itertuples():
        wq = wk_qty.get((r.material, r.location, float(r.wk)))
        if wq and wq > 0 and r.q > 5 * wq:
            over += 1
    if over:
        add("Soft-004", "WARNING", "M3_SafetyStock", "safety_stock_qty",
            f"{over} rows", "safety stock > 5x weekly demand forecast",
            "safety stock <= 5x weekly demand", "Review safety stock provisioning")

# ================================================================== Report =====
errs = [i for i in issues if i["sev"] == "ERROR"]
warns = [i for i in issues if i["sev"] == "WARNING"]
status = "FAILED" if errs else ("PASS WITH WARNINGS" if warns else "PASS")
ready = "Not ready" if errs else ("Ready with risk" if warns else "Ready")
n_files = len([t for t in tabs.values() if t is not None])

# diff mode: dump issue keys to JSON (for baseline-vs-baseline-hc comparison) and
# optionally skip writing reports so an existing baseline report is not clobbered.
import json
import os
if os.environ.get("CHAINSIGHT_ISSUES_JSON"):
    keyed = [{"rule": i["rule"], "tab": i["tab"], "scope": i["scope"],
              "key": i["key"], "sev": i["sev"], "msg": i["msg"]} for i in issues]
    Path(os.environ["CHAINSIGHT_ISSUES_JSON"]).write_text(
        json.dumps(keyed, ensure_ascii=False, indent=2), encoding="utf-8")
if os.environ.get("CHAINSIGHT_NO_REPORT"):
    print(f"[diff mode] {SCENARIO}: ERROR {len(errs)} WARNING {len(warns)} "
          f"(reports not written)")
    sys.exit(0)

# tab-level rollup
tab_err = defaultdict(int)
tab_warn = defaultdict(int)
for i in issues:
    if i["sev"] == "ERROR":
        tab_err[i["tab"]] += 1
    else:
        tab_warn[i["tab"]] += 1


def tab_status(t):
    if t not in maintained:
        return "NOT MAINTAINED"
    if tab_err[t]:
        return "FAILED"
    if tab_warn[t]:
        return "WARNING"
    return "PASS"


fam_counts = {f: 0 for f in na_rules}
for i in issues:
    fam = i["rule"].split("-")[0]
    if fam in fam_counts:
        fam_counts[fam] += 1

L = []
L.append(f"# {SCENARIO} config validation report\n")
L.append("## 1. Validation summary\n")
L.append("| Field | Value |")
L.append("| --- | --- |")
L.append(f"| Validation status | {status} |")
L.append(f"| Simulation readiness | {ready} |")
L.append(f"| Configuration folder | {CFG} ({n_files} config files) |")
L.append(f"| Simulation period | {SIM_START} to {SIM_END} |")
L.append(f"| Scope benchmark | Mode B — M4 origin expansion ({len(bench_mats)} materials, "
         f"{len(bench_nodes)} nodes, {len(bench_lanes)} lanes) |")
L.append(f"| Total ERROR count | {len(errs)} |")
L.append(f"| Total WARNING count | {len(warns)} |")
L.append("\n## 2. Issue detail\n")
if issues:
    L.append("| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i in issues:
        L.append(f"| {i['id']} | {i['rule']} | {i['sev']} | {i['tab']} | {i['scope']} | "
                 f"{i['key']} | {i['msg']} | {i['exp']} | {i['fix']} |")
else:
    L.append("_No issues found._")
L.append("\n## 3. Tab-level summary\n")
L.append("| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |")
L.append("| --- | --- | --- | --- | --- | --- |")
for t in ALL_TABS:
    st = tab_status(t)
    remark = "" if t in maintained else "Tab not provided"
    L.append(f"| {t} | {st} | {tab_err[t]} | {tab_warn[t]} |  | {remark} |")
L.append("\n## 4. Readiness conclusion\n")
L.append("| Decision | Condition | Conclusion text |")
L.append("| --- | --- | --- |")
L.append("| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. |")
L.append("| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. |")
L.append("| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |")
L.append(f"\n**Result: {status} — {ready}.**\n")
L.append("\n## 5. Rule coverage\n")
L.append("| Rule family | Evaluated | Not applicable | Not-applicable IDs + reason |")
L.append("| --- | --- | --- | --- |")
for fam in ["Availability", "Schema", "Consistency", "Hard", "Soft"]:
    na = na_rules[fam]
    L.append(f"| {fam} | {fam_counts[fam]} | {len(na)} | {'; '.join(na) if na else '—'} |")

report_md = OUTDIR / f"{SCENARIO}-config-validation-report.md"
report_md.write_text("\n".join(L), encoding="utf-8")
print(f"Wrote {report_md.relative_to(CFG.parents[3])}")

# ---- Excel report -------------------------------------------------------------
xlsx = OUTDIR / f"{SCENARIO}-config-validation-report.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    pd.DataFrame([
        ("Validation status", status), ("Simulation readiness", ready),
        ("Configuration folder", f"{CFG} ({n_files} files)"),
        ("Simulation period", f"{SIM_START} to {SIM_END}"),
        ("Scope benchmark", f"Mode B ({len(bench_mats)} mats, {len(bench_nodes)} nodes, {len(bench_lanes)} lanes)"),
        ("Total ERROR count", len(errs)), ("Total WARNING count", len(warns)),
    ], columns=["Field", "Value"]).to_excel(xw, "Validation Summary", index=False)

    idf = pd.DataFrame([{
        "Issue ID": i["id"], "Rule ID": i["rule"], "Severity": i["sev"], "Tab": i["tab"],
        "Field / Scope": i["scope"], "Key value": i["key"], "Issue": i["msg"],
        "Expected": i["exp"], "Suggested fix": i["fix"],
        "Detail sheet": i["id"] if i["rows"] else "",
        "Detail row count": len(i["rows"]),
    } for i in issues]) if issues else pd.DataFrame(
        columns=["Issue ID", "Rule ID", "Severity", "Tab", "Field / Scope", "Key value",
                 "Issue", "Expected", "Suggested fix", "Detail sheet", "Detail row count"])
    idf.to_excel(xw, "Issue Detail", index=False)

    pd.DataFrame([{
        "Tab": t, "Status": tab_status(t), "ERROR count": tab_err[t],
        "WARNING count": tab_warn[t], "Main issue type": "",
        "Remark": "" if t in maintained else "Tab not provided",
    } for t in ALL_TABS]).to_excel(xw, "Tab Summary", index=False)

    pd.DataFrame([
        ("FAILED", "Any ERROR exists", "Not ready for simulation."),
        ("PASS WITH WARNINGS", "No ERROR, >=1 WARNING", "Usable; review warnings."),
        ("PASS", "No ERROR and no WARNING", "Ready for simulation."),
    ], columns=["Decision", "Condition", "Conclusion text"]).to_excel(
        xw, "Readiness", index=False)

    pd.DataFrame([{
        "Rule family": fam, "Evaluated": fam_counts[fam],
        "Not applicable": len(na_rules[fam]),
        "Not-applicable IDs + reason": "; ".join(na_rules[fam]) or "—",
    } for fam in ["Availability", "Schema", "Consistency", "Hard", "Soft"]]).to_excel(
        xw, "Rule Coverage", index=False)

    for i in issues:
        if not i["rows"]:
            continue
        det = pd.DataFrame(i["rows"])
        det.insert(0, "Issue ID", i["id"])
        det.insert(1, "Rule ID", i["rule"])
        det.insert(2, "Severity", i["sev"])
        det.insert(3, "Source Tab", i["tab"])
        det.to_excel(xw, i["id"][:31], index=False)
print(f"Wrote {xlsx.relative_to(CFG.parents[3])}")

# console summary
print(f"\n=== {SCENARIO}: {status} | ERROR {len(errs)} | WARNING {len(warns)} ===")
for i in issues:
    print(f"  [{i['sev']}] {i['id']} {i['rule']} :: {i['tab']} :: {i['msg']}")
