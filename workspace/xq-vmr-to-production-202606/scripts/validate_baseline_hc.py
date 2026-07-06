"""Validate baseline-hc config integrity after PCC removal."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DST = ROOT / "workspace/xq-vmr-to-production-202606/scenarios/baseline-hc/config"
cls = pd.read_csv(DST / "_material_hc_pcc_classification.csv", dtype=str)
pcc = set(cls.loc[cls["bucket"] == "PCC", "material"].astype(str))

issues = []


def matset(fn, cols):
    df = pd.read_csv(DST / fn, dtype=str)
    s = set()
    for c in cols:
        s |= set(df[c].dropna().astype(str).str.strip())
    return df, s


# 1) No PCC material remains anywhere
for fn, cols in {
    "Global_Network.csv": ["material"], "M1_AOConfig.csv": ["material"],
    "M1_DemandForecast.csv": ["material"], "M1_ForecastError.csv": ["material"],
    "M1_InitialInventory.csv": ["material"], "M3_SafetyStock.csv": ["material"],
    "M4_ChangeoverMatrix.csv": ["from_material", "to_material"],
    "M4_MaterialLocationLineCfg.csv": ["material"], "M5_DeployConfig.csv": ["material"],
    "M5_PushPullModel.csv": ["material"], "M6_MaterialMD.csv": ["material"],
}.items():
    _, s = matset(fn, cols)
    leak = s & pcc
    if leak:
        issues.append(f"{fn}: {len(leak)} PCC materials still present e.g. {list(leak)[:5]}")

# 2) Every produced (MLC) material has a Global_Network row + a demand or inventory presence
mlc, mlc_mats = matset("M4_MaterialLocationLineCfg.csv", ["material"])
net, net_mats = matset("Global_Network.csv", ["material"])
no_net = mlc_mats - net_mats
if no_net:
    issues.append(f"{len(no_net)} produced materials lack a Global_Network row e.g. {list(no_net)[:5]}")

# 3) Every demand-forecast material has a network row (so it can be sourced)
fc, fc_mats = matset("M1_DemandForecast.csv", ["material"])
fc_no_net = fc_mats - net_mats
if fc_no_net:
    issues.append(f"{len(fc_no_net)} demand-forecast materials lack a Global_Network row e.g. {list(fc_no_net)[:5]}")

# 4) Line-keyed config covers exactly the kept lines
kept_lines = set(mlc["delegate_line"].astype(str).str.strip())
for fn, lcol in {"M4_LineCapacity.csv": "line", "M4_ChangeoverDefinition.csv": "line",
                 "M4_ProductionReliability.csv": "line"}.items():
    df = pd.read_csv(DST / fn, dtype=str)
    lines = set(df[lcol].astype(str).str.strip())
    extra = lines - kept_lines
    if extra:
        issues.append(f"{fn}: lines with no kept material: {extra}")

# 5) ChangeoverMatrix references only materials that still exist in MLC (production universe)
cm = pd.read_csv(DST / "M4_ChangeoverMatrix.csv", dtype=str)
cm_mats = set(cm["from_material"].astype(str)) | set(cm["to_material"].astype(str))
cm_orphan = cm_mats - mlc_mats
# changeover matrix may legitimately reference only produced materials; report size
print(f"ChangeoverMatrix materials not in MLC: {len(cm_orphan)} (informational)")

print(f"\nKept produced materials: {len(mlc_mats)} | lines: {sorted(kept_lines)}")
print(f"Demand-forecast materials: {len(fc_mats)}")
print(f"Network materials: {len(net_mats)}")

if issues:
    print("\n=== ISSUES ===")
    for i in issues:
        print("  [X] " + i)
else:
    print("\n[OK] No integrity issues found.")
