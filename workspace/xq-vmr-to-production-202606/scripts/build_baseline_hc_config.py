"""Build baseline-hc config from baseline by removing all confirmed-PCC config.

Reads the HC/PCC classification (tools/_hc_classification.csv) and the baseline
config, writes a filtered copy under scenarios/baseline-hc/config/.

Filtering rules:
  - Material-keyed tables: drop rows referencing a confirmed-PCC material.
      M4_ChangeoverMatrix: drop if EITHER from_material or to_material is PCC.
  - Line-keyed tables (LineCapacity / ChangeoverDefinition / ProductionReliability):
      drop rows for lines that retain no HC/kept material after filtering.
  - All other (non-material, non-line) tables: copied verbatim.
Only the primary config CSVs are copied (audit/scratch CSVs and xlsx are skipped,
except the reference xlsx the analysis depends on).
"""
from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCEN = ROOT / "workspace/xq-vmr-to-production-202606/scenarios"
SRC = SCEN / "baseline/config"
DST = SCEN / "baseline-hc/config"
DST.mkdir(parents=True, exist_ok=True)

cls = pd.read_csv(DST / "_material_hc_pcc_classification.csv", dtype=str)
pcc = set(cls.loc[cls["bucket"] == "PCC", "material"].astype(str))
kept_materials = set(cls.loc[cls["bucket"] != "PCC", "material"].astype(str))
print(f"PCC materials to remove: {len(pcc)} | kept materials: {len(kept_materials)}")

# material-keyed tables -> columns to test
MAT_COLS = {
    "Global_Network.csv": ["material"],
    "M1_AOConfig.csv": ["material"],
    "M1_DemandForecast.csv": ["material"],
    "M1_ForecastError.csv": ["material"],
    "M1_InitialInventory.csv": ["material"],
    "M3_SafetyStock.csv": ["material"],
    "M4_ChangeoverMatrix.csv": ["from_material", "to_material"],
    "M4_MaterialLocationLineCfg.csv": ["material"],
    "M5_DeployConfig.csv": ["material"],
    "M5_PushPullModel.csv": ["material"],
    "M6_MaterialMD.csv": ["material"],
}
# line-keyed tables -> line column
LINE_COLS = {
    "M4_LineCapacity.csv": "line",
    "M4_ChangeoverDefinition.csv": "line",
    "M4_ProductionReliability.csv": "line",
}
# verbatim copies (no material / line key, but still primary config)
VERBATIM = [
    "Global_DemandPriority.csv",
    "Global_LeadTime.csv",
    "M1_OrderCalendar.csv",
    "M6_DeliveryDelayDistribution.csv",
    "M6_MDQBypassRules.csv",
    "M6_TruckReleaseCon.csv",
    "M6_TruckTypeSpecs.csv",
]
# reference workbooks the analysis depends on
XLSX_COPY = [
    "SUF for XQ HC ChainSight.xlsx",
    "missing SUF manual input.xlsx",
    "missing category manual input.xlsx",
]

report = []

# 1) material-keyed
kept_mlc_lines: set[str] = set()
for fn, cols in MAT_COLS.items():
    df = pd.read_csv(SRC / fn, dtype=str)
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= ~df[c].astype(str).str.strip().isin(pcc)
    out = df[mask].copy()
    out.to_csv(DST / fn, index=False)
    report.append((fn, before, len(out), before - len(out)))
    if fn == "M4_MaterialLocationLineCfg.csv":
        kept_mlc_lines = set(out["delegate_line"].astype(str).str.strip())

print(f"\nLines still used by kept produced materials: {sorted(kept_mlc_lines)}")

# 2) line-keyed: keep only lines still used
for fn, lcol in LINE_COLS.items():
    df = pd.read_csv(SRC / fn, dtype=str)
    before = len(df)
    out = df[df[lcol].astype(str).str.strip().isin(kept_mlc_lines)].copy()
    out.to_csv(DST / fn, index=False)
    report.append((fn, before, len(out), before - len(out)))

# 3) verbatim
for fn in VERBATIM:
    shutil.copy2(SRC / fn, DST / fn)
    n = sum(1 for _ in open(SRC / fn, encoding="utf-8")) - 1
    report.append((fn, n, n, 0))

# 4) xlsx references
for fn in XLSX_COPY:
    if (SRC / fn).exists():
        shutil.copy2(SRC / fn, DST / fn)

print("\n=== filter report (file: before -> after, removed) ===")
for fn, b, a, r in report:
    print(f"  {fn:<35} {b:>8} -> {a:>8}  (-{r})")
print(f"\nWrote {len(report)} config files to {DST.relative_to(ROOT)}")
