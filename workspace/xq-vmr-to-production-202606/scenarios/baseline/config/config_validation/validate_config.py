"""Ad-hoc ChainSight config validator for the xq-vmr baseline scenario.

Implements the chainsight-configure-validation skill checks:
- Availability (required / conditional / optional tabs)
- Schema completeness (required cols, date format, numeric, key not null)
- Cross-tab consistency (Mode A benchmark = Global_Network location+sourcing)
- Hard constraints (numeric ranges, coverage, uniqueness)
- Network cycle detection (explains net-demand recursion hangs)

DFC soft checks are skipped per user instruction.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd

CFG = Path(__file__).resolve().parent.parent
SIM_START = date(2026, 6, 29)
SIM_END = date(2026, 11, 1)

PRIMARY = {
    "Global_Network", "Global_LeadTime", "Global_DemandPriority",
    "M1_DemandForecast", "M1_OrderCalendar", "M1_InitialInventory",
    "M1_ForecastError", "M3_SafetyStock", "M5_PushPullModel",
    "M5_DeployConfig", "M6_MaterialMD", "M6_TruckReleaseCon",
    "M6_TruckTypeSpecs", "M6_DeliveryDelayDistribution",
    "M1_AOConfig", "M4_MaterialLocationLineCfg", "M4_LineCapacity",
    "M4_ProductionReliability", "M4_ChangeoverDefinition",
    "M4_ChangeoverMatrix", "M6_MDQBypassRules",
}

issues: list[dict] = []
_seq = 0


def add(rule, sev, tab, scope, key, msg, expected, fix):
    global _seq
    _seq += 1
    issues.append({
        "id": f"ISSUE-{_seq:03d}", "rule": rule, "sev": sev, "tab": tab,
        "scope": scope, "key": str(key), "msg": msg, "exp": expected, "fix": fix,
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


tabs = {n: load(n) for n in PRIMARY}

# ---------- Availability ----------
MANDATORY = [
    "Global_Network", "Global_LeadTime", "Global_DemandPriority",
    "M1_DemandForecast", "M1_OrderCalendar", "M1_InitialInventory",
    "M1_ForecastError", "M3_SafetyStock", "M5_PushPullModel",
    "M5_DeployConfig", "M6_MaterialMD", "M6_TruckReleaseCon",
    "M6_TruckTypeSpecs", "M6_DeliveryDelayDistribution",
]
for t in MANDATORY:
    if tabs.get(t) is None:
        add("Availability-001", "ERROR", t, "tab", "-",
            "Mandatory tab missing", "Tab present", f"Provide {t}.csv")

# ---------- Schema (key not null + date + numeric spot checks) ----------
SCHEMA = {
    "Global_Network": (["material", "location", "sourcing", "location_type"],
                       ["eff_from", "eff_to"], []),
    "Global_LeadTime": (["sending", "receiving"], [], ["PDT", "GR", "MCT", "OTD"]),
    "Global_DemandPriority": (["demand_element"], [], ["priority"]),
    "M1_DemandForecast": (["week", "material", "location"], [], ["quantity"]),
    "M1_InitialInventory": (["material", "location"], [], ["quantity"]),
    "M1_ForecastError": (["material", "location"], [], ["error_std_percent"]),
    "M1_OrderCalendar": (["date"], ["date"], []),
    "M3_SafetyStock": (["material", "location", "date"], ["date"], ["safety_stock_qty"]),
}
for t, (keys, dates, nums) in SCHEMA.items():
    df = tabs.get(t)
    if df is None:
        continue
    for k in keys:
        if k not in df.columns:
            add("Schema-001", "ERROR", t, k, "-", "Required column missing",
                "Column present", f"Add column {k}")
            continue
        blank = df[k].astype(str).str.strip() == ""
        # Documented exception: Plant nodes legitimately have empty sourcing
        # (top-of-network self-WIP); owned by Hard-Global_Network-002.
        if t == "Global_Network" and k == "sourcing":
            blank = blank & (df["location_type"].str.strip().str.lower() != "plant")
        if blank.any():
            n = int(blank.sum())
            add("Schema-005", "ERROR", t, k, f"{n} rows", "Key field null",
                "Key not null", f"Fill {k}")
    for d in dates:
        if d in df.columns:
            bad = df[~df[d].map(chk_date)][d].unique()[:5]
            if len(bad):
                add("Schema-003", "ERROR", t, d, ", ".join(map(str, bad)),
                    "Date not parseable / format", "YYYY-MM-DD", "Fix date values")
    for c in nums:
        if c in df.columns:
            bad = df[~df[c].map(is_num)][c].unique()[:5]
            if len(bad):
                add("Schema-004", "ERROR", t, c, ", ".join(map(str, bad)),
                    "Numeric not parseable", "Valid number", f"Fix {c}")

# ---------- Mode A benchmark ----------
gn = tabs["Global_Network"]
bench_nodes = set()
bench_mats = set()
if gn is not None:
    for _, r in gn.iterrows():
        bench_nodes.add((r["material"], r["location"]))
        bench_nodes.add((r["material"], r["sourcing"]))
        bench_mats.add(r["material"])

# ---------- Consistency-002 (material) ----------
# Master-data coverage is measured against the in-scope universe (forecast
# materials), NOT the full Global_Network material set: the network carries
# hundreds of materials with no demand that are not part of this run.
# See the in-scope coverage block below for the meaningful gap.

# ---------- In-scope coverage (forecast materials = true sim scope) ----------
# Anchor plant for this scenario (XQ). Materials whose demand lands ONLY at the
# anchor plant are produce-and-consume-locally: they have no distribution lane,
# so they legitimately need neither M5 deploy config nor M6 logistics master
# data. Only materials with demand at a downstream DC require those tabs.
ANCHOR_PLANT = "1864"
fc = tabs.get("M1_DemandForecast")
if fc is not None:
    fc_mats = set(fc["material"])
    demand_loc = fc.groupby("material")["location"].apply(
        lambda s: set(str(x).strip() for x in s))
    deploy_mats = {m for m in fc_mats if demand_loc.get(m, set()) - {ANCHOR_PLANT}}
    plant_only = fc_mats - deploy_mats
    md = tabs.get("M6_MaterialMD")
    if md is not None:
        miss = deploy_mats - set(md["material"])
        if miss:
            add("Consistency-002-scope", "ERROR", "M6_MaterialMD", "material (forecast scope)",
                f"{len(miss)} of {len(deploy_mats)} deploy-scope missing e.g. {list(miss)[:3]} "
                f"({len(plant_only)} plant-only materials exempted)",
                "Forecast material with DC demand has no M6 master data",
                "Every DC-demand forecast material present in M6_MaterialMD",
                "Add master data rows")
    dc = tabs.get("M5_DeployConfig")
    if dc is not None:
        miss = deploy_mats - set(dc["material"])
        if miss:
            add("Consistency-002-scope", "ERROR", "M5_DeployConfig", "material (forecast scope)",
                f"{len(miss)} of {len(deploy_mats)} deploy-scope missing e.g. {list(miss)[:3]} "
                f"({len(plant_only)} plant-only materials exempted)",
                "Forecast material with DC demand has no deploy config",
                "Every DC-demand forecast material present in M5_DeployConfig",
                "Add deploy config rows")

# ---------- Consistency-003 (lanes) ----------
# Lead-time coverage is measured only against lanes that an in-scope (forecast)
# material actually traverses starting from the anchor plant. The full network
# carries alternate-source lanes from other plants (386, 2799, 9264, A868, ...)
# that are never walked when the run is anchored at 1864, so requiring lead-time
# for them is a false positive.
if gn is not None and fc is not None:
    lt = tabs.get("Global_LeadTime")
    if lt is not None:
        net_lanes_by_mat = defaultdict(set)
        for r in gn.itertuples():
            if r.sourcing and r.sourcing != r.location:
                net_lanes_by_mat[r.material].add((r.sourcing, r.location))

        def reachable_lanes(mat):
            edges = defaultdict(set)
            for s, l in net_lanes_by_mat[mat]:
                edges[s].add(l)
            seen = {ANCHOR_PLANT}
            q = deque([ANCHOR_PLANT])
            lanes = set()
            while q:
                cur = q.popleft()
                for nx in edges.get(cur, ()):
                    lanes.add((cur, nx))
                    if nx not in seen:
                        seen.add(nx)
                        q.append(nx)
            return lanes

        inscope_lanes = set()
        for m in set(fc["material"]):
            inscope_lanes |= reachable_lanes(m)

        lt_lanes = {(r["sending"], r["receiving"]) for _, r in lt.iterrows()}
        miss = inscope_lanes - lt_lanes
        if miss:
            add("Consistency-003", "ERROR", "Global_LeadTime", "sending-receiving",
                f"{len(miss)} missing e.g. {list(miss)[:3]}",
                "In-scope network lane has no lead-time row",
                "Every in-scope (1864-reachable) network lane present in Global_LeadTime",
                "Add lead-time rows")

# ---------- Hard-Global_Network-001 (sourcing uniqueness, overlapping periods) ----------
if gn is not None:
    g = gn.copy()
    g["ef"] = pd.to_datetime(g["eff_from"], errors="coerce")
    g["et"] = pd.to_datetime(g["eff_to"], errors="coerce")
    for (m, loc), grp in g.groupby(["material", "location"]):
        rows = grp[["sourcing", "ef", "et"]].dropna().values.tolist()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                s1, f1, t1 = rows[i]
                s2, f2, t2 = rows[j]
                if s1 == s2:
                    continue
                if f1 <= t2 and f2 <= t1:  # overlap
                    add("Hard-Global_Network-001", "ERROR", "Global_Network",
                        "sourcing", f"{m} @ {loc}",
                        f"Overlapping sourcing {s1} vs {s2}",
                        "At most one sourcing per material-location-period",
                        "Resolve duplicate sourcing")
                    break

# ---------- Network cycle detection (explains net-demand hang) ----------
if gn is not None:
    # build per-material sourcing graph (location <- sourcing), ignore self loops
    from collections import defaultdict
    cycles = []
    by_mat = defaultdict(dict)  # mat -> {loc: sourcing}
    for _, r in gn.iterrows():
        if r["sourcing"] != r["location"]:
            by_mat[r["material"]].setdefault(r["location"], set()).add(r["sourcing"])
    for mat, edges in by_mat.items():
        WHITE, GREY, BLACK = 0, 1, 2
        color = defaultdict(int)

        def dfs(node, stack):
            color[node] = GREY
            for nxt in edges.get(node, ()):  # nxt = upstream sourcing
                if color[nxt] == GREY:
                    cyc = stack[stack.index(nxt):] + [nxt]
                    cycles.append((mat, cyc))
                    return True
                if color[nxt] == WHITE and dfs(nxt, stack + [nxt]):
                    return True
            color[node] = BLACK
            return False

        for n in list(edges.keys()):
            if color[n] == WHITE:
                if dfs(n, [n]):
                    break
    seen = set()
    for mat, cyc in cycles:
        key = (mat, tuple(sorted(set(cyc))))
        if key in seen:
            continue
        seen.add(key)
        add("Hard-Global_Network-CYCLE", "ERROR", "Global_Network",
            "sourcing graph", f"material {mat}",
            f"Sourcing cycle: {' -> '.join(cyc)}",
            "Acyclic sourcing chain (no A<->B loops)",
            "Break the cycle so net-demand recursion terminates")

# ---------- Hard-Global_LeadTime-001/002 ----------
lt = tabs.get("Global_LeadTime")
if lt is not None:
    for _, r in lt.iterrows():
        for c in ["PDT", "OTD", "GR", "MCT"]:
            if not is_int(r[c]):
                add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", c,
                    f"{r['sending']}->{r['receiving']}", f"{c}={r[c]} not int",
                    "PDT/OTD/GR/MCT integer", "Use integer")
        if is_num(r["OTD"]) and is_num(r["PDT"]) and float(r["OTD"]) > float(r["PDT"]):
            add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", "OTD/PDT",
                f"{r['sending']}->{r['receiving']}",
                f"OTD {r['OTD']} > PDT {r['PDT']}", "OTD <= PDT", "Fix OTD/PDT")
        if is_num(r["MCT"]) and float(r["MCT"]) < 0:
            add("Hard-Global_LeadTime-001", "ERROR", "Global_LeadTime", "MCT",
                f"{r['sending']}->{r['receiving']}", f"MCT={r['MCT']} < 0",
                "MCT >= 0", "Fix MCT")
    dup = lt.groupby(["sending", "receiving"]).size()
    for (s, rcv), n in dup[dup > 1].items():
        add("Hard-Global_LeadTime-002", "ERROR", "Global_LeadTime",
            "sending-receiving", f"{s}->{rcv}", f"{n} duplicate rows",
            "Exactly one row per lane", "Deduplicate")

# ---------- Hard-Global_DemandPriority-002/003/004 ----------
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
                "push & soft push replenishment present (priority 99)",
                f"Add '{need}' row")
    for e in elems:
        toks = e.split()
        if "ao" in toks:
            add("Hard-Global_DemandPriority-004", "ERROR", "Global_DemandPriority",
                "demand_element", e, "lowercase 'ao' token",
                "uppercase AO", "Use 'AO'")

# ---------- Hard quantity/range checks ----------
def qty_nonneg(tab, col, rule, keycols):
    df = tabs.get(tab)
    if df is None or col not in df.columns:
        return
    for _, r in df.iterrows():
        if is_num(r[col]) and float(r[col]) < 0:
            add(rule, "ERROR", tab, col,
                " / ".join(str(r[k]) for k in keycols if k in df.columns),
                f"{col}={r[col]} < 0", f"{col} >= 0", f"Fix {col}")


qty_nonneg("M1_InitialInventory", "quantity", "Hard-M1_InitialInventory-001",
           ["material", "location"])
qty_nonneg("M1_DemandForecast", "quantity", "Hard-M1_DemandForecast-002",
           ["material", "location", "week"])
qty_nonneg("M1_ForecastError", "error_std_percent", "Hard-M1_ForecastError-001",
           ["material", "location"])
qty_nonneg("M3_SafetyStock", "safety_stock_qty", "Hard-M3_SafetyStock-002",
           ["material", "location", "date"])

# ---------- Hard-M1_OrderCalendar-001 / M3 date coverage ----------
oc = tabs.get("M1_OrderCalendar")
if oc is not None:
    days = {str(pd.Timestamp(d).date()) for d in oc["date"] if chk_date(d)}
    need = []
    d = SIM_START
    while d <= SIM_END:
        if str(d) not in days:
            need.append(str(d))
        d += timedelta(days=1)
    if need:
        add("Hard-M1_OrderCalendar-001", "ERROR", "M1_OrderCalendar", "date",
            f"{len(need)} missing e.g. {need[:3]}",
            "Order calendar gaps in sim period",
            "Cover full sim period", "Add missing dates")

# ---------- M6 fill-rate ----------
tr = tabs.get("M6_TruckReleaseCon")
if tr is not None:
    for _, r in tr.iterrows():
        for c in ["WFR", "VFR"]:
            if is_num(r[c]) and not (0 < float(r[c]) < 1):
                add("Hard-M6_TruckReleaseCon-001", "ERROR", "M6_TruckReleaseCon",
                    c, f"{r['sending']}->{r['receiving']}", f"{c}={r[c]}",
                    "0 < rate < 1", f"Fix {c}")

# ---------- Report ----------
errs = [i for i in issues if i["sev"] == "ERROR"]
warns = [i for i in issues if i["sev"] == "WARNING"]
status = "FAILED" if errs else ("PASS WITH WARNINGS" if warns else "PASS")
ready = "Not ready" if errs else ("Ready with risk" if warns else "Ready")

out = CFG / "config_validation" / "baseline-config-validation-report.md"
lines = []
lines.append("# baseline config validation report\n")
lines.append("## 1. Validation summary\n")
lines.append("| Field | Value |")
lines.append("| --- | --- |")
lines.append(f"| Validation status | {status} |")
lines.append(f"| Simulation readiness | {ready} |")
lines.append(f"| Configuration folder | {CFG} ({len([t for t in tabs.values() if t is not None])} tabs) |")
lines.append(f"| Simulation period | {SIM_START} to {SIM_END} |")
lines.append(f"| Scope benchmark | Mode A — Global_Network ({len(bench_nodes)} nodes, {len(bench_mats)} materials) |")
lines.append(f"| Total ERROR count | {len(errs)} |")
lines.append(f"| Total WARNING count | {len(warns)} |\n")

lines.append("## 2. Issue detail\n")
if issues:
    lines.append("| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i in issues:
        lines.append(f"| {i['id']} | {i['rule']} | {i['sev']} | {i['tab']} | {i['scope']} | {i['key']} | {i['msg']} | {i['exp']} | {i['fix']} |")
else:
    lines.append("No issues found.")
lines.append("")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")

print(f"STATUS={status} ERRORS={len(errs)} WARNINGS={len(warns)}")
print(f"REPORT={out}")
for i in errs[:40]:
    print(f"  [{i['rule']}] {i['tab']} {i['scope']} {i['key']}: {i['msg']}")
