## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | FAILED |
| Simulation readiness | Not ready |
| Configuration file | /home/zhangs37/ai_chainsight/workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/baseline-fem-hpfd-network-0386.xlsx |
| Simulation period | 2026-02-02 to 2026-05-03 |
| Scope benchmark | Working-session re-check benchmark = active M1_DemandForecast scope, with user-approved business exceptions recorded in manual-fix-log |
| Total ERROR count | 8 |
| Total WARNING count | 2 |

## 2. Issue detail

| Issue ID | Severity | Tab | Issue | Sample / Note |
| --- | --- | --- | --- | --- |
| ISSUE-032 | ERROR | M1_DemandForecast | week coverage incomplete: min=1, max=13 | expected min=1 and max>=15 |
| ISSUE-015 | ERROR | M3_SafetyStock | missing 31 material-location combos | 80799573-E568, 80799574-C563, 80799575-E556, 80799576-E556, 80799577-E556, 80817373-D594, 80817373-E230, 80817374-D594, 80817374-E230, 80817375-D594 |
| ISSUE-016 | ERROR | M3_SafetyStock | extra 37 material-location combos | 80728859-0386, 80728861-0386, 80799571-0386, 80799573-0386, 80799574-0386, 80799575-0386, 80799576-0386, 80799577-0386, 80804391-0386, 80817373-0386 |
| ISSUE-018 | ERROR | M5_DeployConfig | extra 24 material-receiving combos | 80728861-E560, 80799571-C819, 80799571-E556, 80799573-C819, 80799574-E556, 80799575-E467, 80856029-E560, 80856029-E564, 80858047-C819, 80858047-D767 |
| ISSUE-034 | ERROR | M3_SafetyStock | daily coverage incomplete for 394 combos | 80728859-0386, 80728859-A668, 80728859-A672, 80728859-A673, 80728859-A680, 80728859-A715, 80728859-A716, 80728859-C810, 80728859-C816, 80728859-C819 |
| ISSUE-035 | ERROR | M4_MaterialLocationLineCfg | numeric fields invalid for 29 rows | 80878361-0386:rv, 80878362-0386:rv, 80878369-0386:rv, 80799574-0386:rv, 80799573-0386:rv, 80856029-0386:rv, 80799571-0386:rv, 80858047-0386:rv, 80799576-0386:rv, 80817373-0386:rv |
| ISSUE-036 | ERROR | M4_LineCapacity | date coverage incomplete for 1 location-line combos | 0386-HPFD |
| ISSUE-023 | ERROR | M6_MaterialMD | missing 7 materials | 80817373, 80817374, 80817375, 80817378, 80892811, 82278637, 83900927 |
| WARN-001 | WARNING | M1_InitialInventory | week1 demand zero/missing for 57 inventory combos | 80728859-0386, 80728861-0386, 80799571-0386, 80799571-C819, 80799573-0386, 80799573-E556, 80799573-E568, 80799573-E569, 80799574-0386, 80799575-0386 |
| WARN-SCOPE | WARNING | Validation scope | This re-check follows the working-session business decisions: 0386 treated as sourcing-only for some checks; several M3 / M4 / M6 gaps intentionally left for manual follow-up. | See manual-fix-log-2026-05-22.md |

## 3. Readiness conclusion

Configuration is **not ready** for simulation. The remaining blockers are the incomplete forecast horizon, unresolved safety-stock scope and daily coverage gaps, unresolved M4 `rv` numeric blanks, incomplete M4 line-capacity calendar, and missing M6 material master rows.