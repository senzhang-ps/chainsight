## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | FAILED |
| Simulation readiness | Not ready |
| Configuration file | /home/zhangs37/ai_chainsight/workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/baseline-fem-hpfd-network-0386.xlsx |
| Simulation period | 2026-02-02 to 2026-05-03 |
| Scope benchmark | Re-check on latest user-updated workbook |
| Total ERROR count | 5 |
| Total WARNING count | 2 |

## 2. Issue detail

| Issue ID | Severity | Tab | Issue | Sample / Note |
| --- | --- | --- | --- | --- |
| ISSUE-032 | ERROR | M1_DemandForecast | week coverage incomplete: min=1, max=13 | expected min=1 and max>=15 |
| ISSUE-015 | ERROR | M3_SafetyStock | missing 25 material-location combos | 80799573-E568, 80799574-C563, 80799575-E556, 80799576-E556, 80799577-E556, 80817373-E230, 80817374-E230, 80817375-E230, 80817378-E230, 80858047-C867 |
| ISSUE-016 | ERROR | M3_SafetyStock | extra 2733 material-location combos | 80684724-A668, 80684724-A672, 80684724-A673, 80684724-A680, 80684724-A715, 80684724-A716, 80684724-C810, 80684724-C816, 80684724-C819, 80684724-C867 |
| ISSUE-018 | ERROR | M5_DeployConfig | extra 24 material-receiving combos | 80728861-E560, 80799571-C819, 80799571-E556, 80799573-C819, 80799574-E556, 80799575-E467, 80856029-E560, 80856029-E564, 80858047-C819, 80858047-D767 |
| ISSUE-036 | ERROR | M4_LineCapacity | date coverage incomplete for 1 location-line combos | 0386-HPFD |
| WARN-001 | WARNING | M1_InitialInventory | week1 demand zero/missing for 57 inventory combos | 80728859-0386, 80728861-0386, 80799571-0386, 80799571-C819, 80799573-0386, 80799573-E556, 80799573-E568, 80799573-E569, 80799574-0386, 80799575-0386 |
| WARN-SCOPE | WARNING | Validation scope | This re-check follows the current workbook state only. Some historical business exceptions such as sourcing-only 0386 still need human interpretation. | Review manually if needed |

## 3. Readiness conclusion

Configuration is **not ready** for simulation. Please resolve the remaining errors before running the ChainSight engine.