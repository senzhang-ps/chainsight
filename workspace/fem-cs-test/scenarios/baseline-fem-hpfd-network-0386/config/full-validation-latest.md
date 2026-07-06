# Full Validation Report

## 1. Context

- Config file: `workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/baseline-fem-hpfd-network-0386.xlsx`
- Simulation period: `2026-02-02` to `2026-05-03`
- Scope benchmark for this re-check: `M1_DemandForecast material-location combinations`
- Low DFC threshold: not applied in this custom re-check
- High DFC threshold: not applied in this custom re-check

## 2. Summary

- Validation status: **FAILED**
- Error count: **6**
- Warning count: **3**

## 3. Errors

1. **M1_DemandForecast** — week horizon shorter than expected, max=13
   - sample: expected >= 15
2. **M1_InitialInventory** — extra 37 material-location combos vs M1_DemandForecast
   - sample: 80728859-0386, 80728861-0386, 80799571-0386, 80799571-C819, 80799573-0386, 80799574-0386, 80799575-0386, 80799575-E467, 80799576-0386, 80799577-0386
3. **M1_ForecastError** — extra 29 material-location combos vs M1_DemandForecast
   - sample: 80728859-0386, 80728861-0386, 80799571-0386, 80799573-0386, 80799574-0386, 80799575-0386, 80799576-0386, 80799577-0386, 80804391-0386, 80817373-0386
4. **M3_SafetyStock** — missing 25 material-location combos vs M1_DemandForecast
   - sample: 80799573-E568, 80799574-C563, 80799575-E556, 80799576-E556, 80799577-E556, 80817373-E230, 80817374-E230, 80817375-E230, 80817378-E230, 80858047-C867
5. **M5_DeployConfig** — extra 24 material-receiving combos vs M1_DemandForecast
   - sample: 80728861-E560, 80799571-C819, 80799571-E556, 80799573-C819, 80799574-E556, 80799575-E467, 80856029-E560, 80856029-E564, 80858047-C819, 80858047-D767
6. **M4_LineCapacity** — date coverage incomplete for 1 line combos
   - sample: 0386-HPFD

## 4. Warnings

1. **Global_Network** — missing 43 material-location combos vs M1_DemandForecast
   - sample: 80728859-C867, 80728859-D608, 80728861-C867, 80728861-D608, 80799571-C867, 80799571-D608, 80799573-C867, 80799574-C563, 80799574-C867, 80799574-D608
2. **Global_Network** — extra 3263 material-location combos vs M1_DemandForecast
   - sample: 80684724-A668, 80684724-A680, 80684724-A715, 80684724-A716, 80684724-C819, 80684724-C937, 80684724-D767, 80684724-D874, 80684724-D876, 80684724-E556
3. **Global_LeadTime** — 50 rows violate 0 <= OTD < PDT
   - sample: --:2/2, --:11/11, --:14/14, --:13/13, --:11/11, --:6/6, --:8/8, --:7/7, --:5/5, --:3/3

## 5. Conclusion

Configuration is **not ready** for simulation under strict validation rules. Resolve the remaining errors or explicitly waive them before running.