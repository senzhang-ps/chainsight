"""Classify every baseline-config material as HC vs PCC using LOCAL files only.

HC rule (any one):
  - material in 'SUF for XQ HC ChainSight.xlsx'
  - cached category_en == 'Hair'  (analysis extract material_category_databricks.csv)
  - manual 'missing category manual input.xlsx' category_en == 'Hair'
  - delegate_line == 'XQHK'  (user: XQHK is all HC)
PCC rule (any one, and not HC):
  - cached category_en == 'PCC'
  - manual category_en == 'PCC'
  - delegate_line in pure-PCC lines
Else: Unclassified.
Read-only; writes the per-material classification into the baseline-hc config folder.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROJ = ROOT / "workspace/xq-vmr-to-production-202606"
CFG = PROJ / "scenarios/baseline/config"
EXTR = (PROJ / "scenarios/baseline/analysis"
        / "20260617-1016-baseline-cfr-changeover-dfc-prodtime-apq-msu/extracts")
OUT = PROJ / "scenarios/baseline-hc/config/_material_hc_pcc_classification.csv"
PURE_PCC_LINES = {"XQ Line 10", "XQHA", "XQHH", "XQHJ", "XQHV"}
# user-confirmed PCC (produced on Hair-capable lines but confirmed PCC by user)
PCC_OVERRIDE = {"80853434", "80856600", "80882631", "80882635",
                "80882636", "80895949", "80895950", "80895951"}

# --- union of all config materials + which files each appears in
MATERIAL_FILES = {
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
allmats: set[str] = set()
for fn, cols in MATERIAL_FILES.items():
    df = pd.read_csv(CFG / fn, dtype=str)
    for c in cols:
        allmats |= set(df[c].dropna().astype(str).str.strip())
allmats.discard("")

# --- signals
mlc = pd.read_csv(CFG / "M4_MaterialLocationLineCfg.csv", dtype=str)
linemap = dict(zip(mlc["material"].astype(str), mlc["delegate_line"].astype(str)))
produced = set(mlc["material"].astype(str))

suf = pd.read_excel(CFG / "SUF for XQ HC ChainSight.xlsx", dtype=str)
suf_hc = set(suf["Material"].astype(str).str.strip())

dbx = pd.read_csv(EXTR / "material_category_databricks.csv", dtype=str)
dbx_cat = {str(m): str(c) for m, c in zip(dbx["material"], dbx["category_en"]) if pd.notna(c)}

man = pd.read_excel(CFG / "missing category manual input.xlsx", dtype=str)
man_cat = {str(r["Material"]).strip(): str(r["category_en"]).strip()
           for _, r in man.iterrows() if str(r.get("category_en", "")).strip().lower() not in ("", "nan")}


def classify(m: str):
    line = linemap.get(m)
    if m in PCC_OVERRIDE:
        return "PCC"
    hc = (m in suf_hc) or (dbx_cat.get(m) == "Hair") or (man_cat.get(m) == "Hair") or (line == "XQHK")
    if hc:
        return "HC"
    pcc = (dbx_cat.get(m) == "PCC") or (man_cat.get(m) == "PCC") or (line in PURE_PCC_LINES)
    if pcc:
        return "PCC"
    return "Unclassified"


rows = []
for m in sorted(allmats):
    rows.append({
        "material": m,
        "bucket": classify(m),
        "produced_at_xq": m in produced,
        "delegate_line": linemap.get(m, ""),
        "in_suf_hc": m in suf_hc,
        "dbx_category_en": dbx_cat.get(m, ""),
        "manual_category_en": man_cat.get(m, ""),
    })
res = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT, index=False)

print("=== ALL config materials ===")
print(res["bucket"].value_counts().to_dict())
print("\n=== PRODUCED at XQ (in M4_MaterialLocationLineCfg) ===")
print(res[res.produced_at_xq]["bucket"].value_counts().to_dict())
print("\n=== NOT produced at XQ ===")
np_ = res[~res.produced_at_xq]
print(np_["bucket"].value_counts().to_dict())
print(f"\nUnclassified total: {(res.bucket=='Unclassified').sum()} "
      f"(produced={((res.bucket=='Unclassified')&res.produced_at_xq).sum()}, "
      f"not-produced={((res.bucket=='Unclassified')&~res.produced_at_xq).sum()})")
print(f"\nwrote {OUT.relative_to(ROOT)}")
