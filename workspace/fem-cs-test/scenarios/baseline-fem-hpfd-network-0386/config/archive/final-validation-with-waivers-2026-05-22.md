# Final Validation With Business Waivers

## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | PASS WITH BUSINESS WAIVERS |
| Simulation readiness | Ready for simulation with approved scope exceptions |
| Configuration file | /home/zhangs37/ai_chainsight/workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/baseline-fem-hpfd-network-0386.xlsx |
| Simulation period | 2026-02-02 to 2026-05-03 |
| Validation date | 2026-05-22 |
| Remaining technical issues | 3 |
| Approved business waivers | 3 |

## 2. Final conclusion

The configuration is accepted as **ready for simulation** under the working-session business decisions made on 2026-05-22.

Three technical validation findings remain in a strict rule-based sense, but all three were explicitly accepted by the user as business waivers for this run. All other previously identified blockers were either fixed, normalized, or reinterpreted as rule-scope issues.

## 3. Approved business waivers

### Waiver 1 — Forecast horizon shorter than full validation expectation
**Original issue**
- `M1_DemandForecast` week coverage ends at week `13` instead of extending to at least week `15`.

**Current state**
- `week` has been normalized to numeric values.
- Current maintained range is `1 ~ 13`.

**Business decision**
- User confirmed this is acceptable for the current run.
- The simulation result review will intentionally ignore the last two weeks.

**Waiver statement**
- Accepted for this run.
- No further action required before simulation.

---

### Waiver 2 — M3_SafetyStock missing combinations
**Original issue**
- `M3_SafetyStock` is still missing `25` material-location combinations relative to `M1_DemandForecast` scope.

**Sample missing combinations**
- `80799573-E568`
- `80799574-C563`
- `80799575-E556`
- `80799576-E556`
- `80799577-E556`
- `80817373-E230`
- `80817374-E230`
- `80817375-E230`
- `80817378-E230`
- `80858047-C867`

**Business decision**
- User confirmed these do not need to be backfilled for the current run.

**Waiver statement**
- Accepted for this run.
- No further action required before simulation.

---

### Waiver 3 — M5_DeployConfig extra combinations
**Original issue**
- `M5_DeployConfig` still contains `24` extra `material-receiving` combinations outside the active `M1_DemandForecast` scope.

**Sample extra combinations**
- `80728861-E560`
- `80799571-C819`
- `80799571-E556`
- `80799573-C819`
- `80799574-E556`
- `80799575-E467`
- `80856029-E560`
- `80856029-E564`
- `80858047-C819`
- `80858047-D767`

**Business decision**
- User confirmed these extra deploy rows do not need to be removed for the current run.

**Waiver statement**
- Accepted for this run.
- No further action required before simulation.

## 4. Issues confirmed as fixed

The following previously reported blockers were fixed during the working session:

- `M1_DemandForecast.week` converted from text labels such as `WEEK06` to numeric week values.
- `M1_OrderCalendar` rebuilt to full simulation-period daily coverage.
- `M1_InitialInventory` location normalization (`386 -> 0386`) completed.
- `M1_InitialInventory` missing combinations were backfilled.
- `M1_ForecastError` missing combinations were backfilled with agreed default `0.75` standard error.
- `M4_MaterialLocationLineCfg` structure was normalized and non-RV numeric fields were aligned with agreed values.
- `M4_ChangeoverMatrix` reverse-direction pairs were added to complete the symmetric half.
- `M5_PushPullModel` invalid `pull` rows were removed.
- `M3_SafetyStock` out-of-scope extra combinations were trimmed down to zero remaining extras.
- `M6_MaterialMD` extra materials were trimmed to the active maintained material scope.
- `Global_DemandPriority` missing multi-layer `net demand for ...` patterns were appended.

## 5. Validation findings that are no longer considered active blockers

### M4_LineCapacity
The earlier finding that `0386-HPFD` had incomplete date coverage was rechecked against the current workbook and found to be **not active**. The line-capacity date coverage is present for the expected period in the latest file.

### 0386 location interpretation
Several historical findings involving `0386` were treated during this session as validation-rule scope issues rather than true configuration defects, because `0386` is used as a **sourcing-only location** in this case.

## 6. Follow-up recommendations

These are not blockers for the current run, but should be considered for future cleanup:

1. Extend `M1_DemandForecast` from week `13` to week `15` or later for full-horizon validation compliance.
2. Revisit whether the waived `M3_SafetyStock` missing combinations should eventually be backfilled.
3. Revisit whether the waived `M5_DeployConfig` extra combinations should eventually be trimmed.
4. Update the validation skill logic to handle sourcing-only locations such as `0386` correctly.

## 7. Final acceptance statement

Based on the user-approved waivers and the latest workbook re-check, the configuration is accepted as:

**PASS WITH BUSINESS WAIVERS**

and may proceed to the next simulation step with the documented interpretation limits above.
