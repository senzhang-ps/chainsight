# baseline-hc config validation report

## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | FAILED |
| Simulation readiness | Not ready |
| Configuration folder | C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\workspace\xq-vmr-to-production-202606\scenarios\baseline-hc\config (21 config files) |
| Simulation period | 2026-06-29 to 2026-11-01 |
| Scope benchmark | Mode B — M4 origin expansion (195 materials, 1336 nodes, 31 lanes) |
| Total ERROR count | 464 |
| Total WARNING count | 3 |

## 2. Issue detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-001 | Consistency-001 | ERROR | M1_DemandForecast | material-location (DC nodes) | 283 of 1141 missing e.g. [('80703933', 'E560'), ('80717309', 'D352'), ('80717309', 'D608'), ('80717309', 'D874'), ('80717311', 'D352')] | Benchmark DC node absent | Every benchmark DC node present | Add missing material-location rows |
| ISSUE-002 | Consistency-001 | ERROR | M1_InitialInventory | material-location (DC nodes) | 283 of 1141 missing e.g. [('80703933', 'E560'), ('80717309', 'D352'), ('80717309', 'D608'), ('80717309', 'D874'), ('80717311', 'D352')] | Benchmark DC node absent | Every benchmark DC node present | Add missing material-location rows |
| ISSUE-003 | Consistency-001 | ERROR | M1_ForecastError | material-location (DC nodes) | 222 of 1141 missing e.g. [('80703933', 'E560'), ('80717309', 'D352'), ('80717311', 'D352'), ('80717311', 'E546'), ('80717311', 'E560')] | Benchmark DC node absent | Every benchmark DC node present | Add missing material-location rows |
| ISSUE-004 | Consistency-001 | ERROR | M3_SafetyStock | material-location (DC nodes) | 222 of 1141 missing e.g. [('80703933', 'E560'), ('80717309', 'D352'), ('80717311', 'D352'), ('80717311', 'E546'), ('80717311', 'E560')] | Benchmark DC node absent | Every benchmark DC node present | Add missing material-location rows |
| ISSUE-005 | Consistency-001 | ERROR | M5_DeployConfig | material-receiving (DC nodes) | 9 of 1141 missing e.g. [('80875074', 'A672'), ('80875074', 'A715'), ('80875074', 'A716'), ('80875074', 'C819'), ('80875074', 'D352')] | Benchmark DC node absent from deploy config | Every benchmark DC node present (extra allowed) | Add deploy rows |
| ISSUE-006 | Consistency-002 | ERROR | M1_DemandForecast | material | 4 of 195 missing e.g. ['21293322', '21310635', '21387762', '80875074'] | Benchmark material absent | Every benchmark material present (extra allowed) | Add material rows |
| ISSUE-007 | Consistency-002 | ERROR | M1_InitialInventory | material | 63 of 195 missing e.g. ['21156898', '21158410', '21158411', '21162178', '21167440'] | Benchmark material absent | Every benchmark material present (extra allowed) | Add material rows |
| ISSUE-008 | Consistency-002 | ERROR | M1_ForecastError | material | 63 of 195 missing e.g. ['21156898', '21158410', '21158411', '21162178', '21167440'] | Benchmark material absent | Every benchmark material present (extra allowed) | Add material rows |
| ISSUE-009 | Consistency-002 | ERROR | M3_SafetyStock | material | 63 of 195 missing e.g. ['21156898', '21158410', '21158411', '21162178', '21167440'] | Benchmark material absent | Every benchmark material present (extra allowed) | Add material rows |
| ISSUE-010 | Consistency-002 | ERROR | M5_DeployConfig | material | 63 of 195 missing e.g. ['21156898', '21158410', '21158411', '21162178', '21167440'] | Benchmark material absent | Every benchmark material present (extra allowed) | Add material rows |
| ISSUE-011 | Consistency-002 | ERROR | M6_MaterialMD | material | 63 of 195 missing e.g. ['21156898', '21158410', '21158411', '21162178', '21167440'] | Benchmark material absent | Every benchmark material present (extra allowed) | Add material rows |
| ISSUE-012 | Hard-Global_Network-001 | ERROR | Global_Network | sourcing | 80865258 @ A673 | Overlapping sourcing A868 vs C810 | <=1 sourcing per material-location-period | Resolve duplicate sourcing |
| ISSUE-013 | Hard-Global_Network-001 | ERROR | Global_Network | sourcing | 80865258 @ A680 | Overlapping sourcing 386 vs C810 | <=1 sourcing per material-location-period | Resolve duplicate sourcing |
| ISSUE-014 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->E564 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-015 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->D608 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-016 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->E569 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-017 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->C819 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-018 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->E568 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-019 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->E560 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-020 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->C819 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-021 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->E556 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-022 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->D874 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-023 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A680->D876 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-024 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C810->C867 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-025 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->A716 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-026 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->A672 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-027 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->A715 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-028 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->A680 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-029 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->A668 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-030 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->C816 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-031 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->E564 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-032 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->A716 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-033 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->A715 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-034 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->A680 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-035 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->A668 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-036 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->D352 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-037 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->A673 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-038 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->E546 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-039 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->C937 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-040 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->D767 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-041 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->C810 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-042 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A672->D352 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-043 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->E295 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-044 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->C816 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-045 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->A668 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-046 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | 1864->A680 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-047 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->A672 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-048 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->A715 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-049 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | A673->A716 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-050 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->A668 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-051 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->C866 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-052 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | C816->E327 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-053 | Hard-Global_LeadTime-001 | ERROR | Global_LeadTime | MCT | D352->A672 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-054 | Hard-Global_DemandPriority-001 | ERROR | Global_DemandPriority | demand_element | net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for normal | layered family missing at network depth 9 | families cover max network depth | Add 'net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for normal' |
| ISSUE-055 | Hard-Global_DemandPriority-001 | ERROR | Global_DemandPriority | demand_element | net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for AO | layered family missing at network depth 9 | families cover max network depth | Add 'net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for AO' |
| ISSUE-056 | Hard-Global_DemandPriority-001 | ERROR | Global_DemandPriority | demand_element | net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for customer | layered family missing at network depth 9 | families cover max network depth | Add 'net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for customer' |
| ISSUE-057 | Hard-Global_DemandPriority-001 | ERROR | Global_DemandPriority | demand_element | net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for forecast | layered family missing at network depth 9 | families cover max network depth | Add 'net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for forecast' |
| ISSUE-058 | Hard-Global_DemandPriority-001 | ERROR | Global_DemandPriority | demand_element | net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for safety | layered family missing at network depth 9 | families cover max network depth | Add 'net demand for net demand for net demand for net demand for net demand for net demand for net demand for net demand for safety' |
| ISSUE-059 | Hard-M6_TruckTypeSpecs-001 | ERROR | M6_TruckTypeSpecs | capacity_qty_in_weight | 1864-D352 | 1 rows out of range (e.g. capacity_qty_in_weight=0) | capacity_qty_in_weight bound (>0.0) | Fix capacity_qty_in_weight |
| ISSUE-060 | Hard-M6_TruckTypeSpecs-001 | ERROR | M6_TruckTypeSpecs | capacity_qty_in_volume | 1864-D352 | 1 rows out of range (e.g. capacity_qty_in_volume=0) | capacity_qty_in_volume bound (>0.0) | Fix capacity_qty_in_volume |
| ISSUE-061 | Hard-M4_LineCapacity-002 | ERROR | M4_LineCapacity | capacity | 1864 / XQHD / 2026-09-25 | 12 rows out of range (e.g. capacity=0) | capacity bound (>0.0, <= 24) | Fix capacity |
| ISSUE-062 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80703933 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-063 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80703933 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-064 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80703933 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-065 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717309 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-066 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717309 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-067 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717311 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-068 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717311 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-069 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717311 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-070 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717315 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-071 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80717315 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-072 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80737957 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-073 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80737957 @ C937 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-074 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80737957 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-075 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754087 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-076 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754087 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-077 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754087 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-078 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754087 @ D191 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-079 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754088 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-080 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754088 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-081 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754088 @ C731 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-082 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80754088 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-083 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812157 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-084 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812157 @ C819 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-085 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812157 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-086 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812158 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-087 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812158 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-088 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812162 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-089 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812163 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-090 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80812165 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-091 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813607 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-092 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813610 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-093 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813610 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-094 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813610 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-095 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813610 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-096 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813614 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-097 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813614 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-098 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813614 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-099 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813614 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-100 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813614 @ C937 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-101 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813614 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-102 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813615 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-103 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813615 @ C819 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-104 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813619 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-105 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813619 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-106 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813619 @ C819 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-107 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813619 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-108 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813620 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-109 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813620 @ C819 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-110 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813621 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-111 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813624 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-112 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813624 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-113 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813624 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-114 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813625 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-115 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813625 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-116 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813625 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-117 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813629 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-118 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813629 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-119 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813631 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-120 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813631 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-121 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80813637 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-122 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814061 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-123 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814075 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-124 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814077 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-125 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814081 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-126 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814081 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-127 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814081 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-128 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814081 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-129 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814086 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-130 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814086 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-131 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814086 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-132 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814086 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-133 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814089 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-134 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814089 @ C563 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-135 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814090 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-136 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814091 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-137 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814091 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-138 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814092 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-139 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814093 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-140 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814093 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-141 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814093 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-142 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814093 @ D191 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-143 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814094 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-144 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814094 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-145 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814095 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-146 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814095 @ C937 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-147 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814096 @ E564 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-148 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814097 @ C937 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-149 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814097 @ E556 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-150 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814098 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-151 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814099 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-152 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80814099 @ C563 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-153 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815642 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-154 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815642 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-155 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815643 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-156 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815644 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-157 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815644 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-158 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815644 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-159 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815644 @ E568 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-160 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815645 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-161 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815645 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-162 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815645 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-163 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815646 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-164 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815646 @ E564 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-165 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815647 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-166 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815647 @ C731 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-167 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815647 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-168 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815648 @ C819 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-169 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815648 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-170 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815649 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-171 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80815649 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-172 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820463 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-173 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820463 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-174 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820464 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-175 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820464 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-176 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820464 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-177 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820470 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-178 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80820470 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-179 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841209 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-180 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841211 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-181 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841211 @ E564 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-182 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841212 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-183 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841212 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-184 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841212 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-185 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841219 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-186 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841219 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-187 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841220 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-188 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841220 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-189 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841221 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-190 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841221 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-191 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841221 @ E564 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-192 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841222 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-193 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80841222 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-194 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80849561 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-195 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80849561 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-196 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80849561 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-197 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80853505 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-198 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80853520 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-199 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80853520 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-200 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80853520 @ E568 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-201 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856843 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-202 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856843 @ C819 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-203 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856846 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-204 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856846 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-205 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856849 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-206 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856849 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-207 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856849 @ D191 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-208 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856849 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-209 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856852 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-210 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856852 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-211 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856852 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-212 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80856852 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-213 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859226 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-214 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859226 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-215 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859230 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-216 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859230 @ D352 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-217 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859230 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-218 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859233 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-219 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859235 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-220 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859235 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-221 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859235 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-222 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859235 @ C937 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-223 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859237 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-224 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859237 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-225 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859239 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-226 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859239 @ C731 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-227 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859239 @ D901 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-228 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859240 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-229 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859244 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-230 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859244 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-231 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859246 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-232 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859246 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-233 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859246 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-234 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859247 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-235 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859247 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-236 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80859250 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-237 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865246 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-238 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865246 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-239 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865248 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-240 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865248 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-241 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865252 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-242 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865258 @ A716 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-243 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865258 @ C810 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-244 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865259 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-245 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80865259 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-246 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80875074 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-247 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80875074 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-248 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80875074 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-249 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80883600 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-250 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80883600 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-251 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889182 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-252 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889182 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-253 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889183 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-254 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889183 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-255 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889184 @ A672 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-256 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889185 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-257 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 80889185 @ D767 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-258 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 82326949 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-259 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 82326949 @ A680 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-260 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 82326949 @ C816 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-261 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 82326949 @ E569 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-262 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 83900170 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-263 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 83900170 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-264 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 83900170 @ A715 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-265 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 83900171 @ A673 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-266 | Hard-M1_AOConfig-002 | ERROR | M1_AOConfig | ao_percent sum | 83906914 @ A668 | sum(ao_percent)=1.0001 > 1 | sum <= 1 per material-location | Reduce ao_percent |
| ISSUE-267 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21158410 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-268 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21158411 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-269 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21162178 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-270 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21181021 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-271 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21181022 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-272 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21181256 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-273 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21168321 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-274 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21168322 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-275 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21168326 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-276 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21167440 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-277 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21178248 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-278 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21172345 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-279 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21172346 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-280 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21286727 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-281 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21293322 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-282 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21286728 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-283 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21296430 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-284 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21387762 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-285 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21461976 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-286 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21344150 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-287 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21319825 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-288 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21329726 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-289 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21350032 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-290 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21310636 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-291 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21309829 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-292 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21309833 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-293 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21310635 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-294 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21298079 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-295 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21301337 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-296 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80856843 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-297 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80856849 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-298 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21469016 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-299 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21469015 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-300 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21447907 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-301 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21156898 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-302 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21178234 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-303 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21178241 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-304 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21171678 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-305 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21344138 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-306 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21334685 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-307 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21334484 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-308 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21319824 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-309 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 90450569 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-310 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21387644 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-311 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21200191 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-312 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21184776 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-313 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21205101 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-314 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21403972 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-315 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21425608 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-316 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21420779 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-317 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80728531 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-318 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813617 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-319 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813619 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-320 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814097 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-321 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815644 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-322 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841211 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-323 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841222 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-324 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80853520 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-325 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859231 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-326 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859241 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-327 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80880191 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-328 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859238 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-329 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859249 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-330 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883600 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-331 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883602 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-332 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 82326949 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-333 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80703933 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-334 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80717309 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-335 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80717311 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-336 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80717315 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-337 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80720217 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-338 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80737957 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-339 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80754087 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-340 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80754088 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-341 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80812157 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-342 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80812158 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-343 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80812162 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-344 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80812163 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-345 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80812165 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-346 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813607 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-347 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813610 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-348 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813614 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-349 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813615 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-350 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813620 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-351 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813621 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-352 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813624 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-353 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813625 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-354 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813629 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-355 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813631 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-356 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80813637 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-357 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814061 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-358 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814072 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-359 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814073 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-360 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814075 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-361 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814077 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-362 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814081 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-363 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814086 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-364 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814088 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-365 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814089 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-366 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814090 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-367 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814091 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-368 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814092 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-369 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814093 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-370 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814094 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-371 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814095 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-372 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814096 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-373 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814098 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-374 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80814099 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-375 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815642 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-376 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815643 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-377 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815645 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-378 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815646 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-379 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815647 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-380 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815648 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-381 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80815649 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-382 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80820463 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-383 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80820464 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-384 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80820470 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-385 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841208 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-386 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841209 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-387 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841212 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-388 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841220 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-389 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841221 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-390 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80849561 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-391 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80853505 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-392 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80856846 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-393 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859226 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-394 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859227 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-395 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859229 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-396 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859230 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-397 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859232 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-398 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859233 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-399 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859234 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-400 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859235 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-401 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859236 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-402 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859237 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-403 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859239 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-404 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859240 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-405 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859242 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-406 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859243 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-407 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859244 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-408 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859245 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-409 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859246 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-410 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859247 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-411 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859248 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-412 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80859250 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-413 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80864647 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-414 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80865246 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-415 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80865248 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-416 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80865252 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-417 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80865258 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-418 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80865259 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-419 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80875074 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-420 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80880192 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-421 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80880193 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-422 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80880197 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-423 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80880199 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-424 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80880201 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-425 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883591 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-426 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883592 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-427 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883610 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-428 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883611 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-429 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883614 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-430 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883615 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-431 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883621 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-432 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80889182 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-433 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80889183 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-434 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80889184 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-435 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80889185 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-436 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83900170 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-437 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83900171 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-438 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83906903 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-439 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83906914 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-440 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83911500 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-441 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80891038 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-442 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80891042 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-443 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80856852 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-444 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21184777 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-445 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80883613 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-446 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80891039 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-447 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918364 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-448 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 80841219 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-449 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21461977 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-450 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21457337 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-451 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918523 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-452 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918515 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-453 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918522 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-454 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918516 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-455 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918519 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-456 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918521 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-457 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918517 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-458 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21357433 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-459 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 83918520 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-460 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21548669 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-461 | Hard-M4_MaterialLocationLineCfg-002 | ERROR | M4_MaterialLocationLineCfg | MCT | 21553433 @ 1864 | MCT=0 <= 0 | MCT > 0 | Fix MCT |
| ISSUE-462 | Hard-M4_ChangeoverMatrix-001 | ERROR | M4_ChangeoverMatrix | from/to material | 31027 pairs | changeover pair not delegated to same line | each pair shares a line | Remove cross-line pairs |
| ISSUE-463 | Hard-M5_PushPullModel-001 | ERROR | M5_PushPullModel | model | ['Pull', 'Push', 'pull'] | invalid model value | push or soft push | Fix model |
| ISSUE-464 | Hard-M4_LineCapacity-001 | ERROR | M4_LineCapacity | date | 3 missing e.g. ['2026-06-29', '2026-06-30', '2026-11-01'] | line capacity date gaps | cover full sim period | Add capacity dates |
| ISSUE-465 | Soft-002 | WARNING | M4_MaterialLocationLineCfg | min_batch | 25 materials e.g. ['21181256', '21469015', '21156898'] | min_batch large vs wk1-4 daily avg demand (>5 weeks of demand) | min_batch reasonable vs demand | Review batch sizing |
| ISSUE-466 | Soft-003 | WARNING | Global_LeadTime | lead-time slack vs bypass | 21 lanes e.g. [('C816', 'E569'), ('A673', 'C819'), ('A672', 'E568')] | bypass waiting_days exceeds lead-time slack | waiting_days x <= PDT-OTD | Review bypass/lead-time |
| ISSUE-467 | Soft-004 | WARNING | M3_SafetyStock | safety_stock_qty | 490 rows | safety stock > 5x weekly demand forecast | safety stock <= 5x weekly demand | Review safety stock provisioning |

## 3. Tab-level summary

| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |
| --- | --- | --- | --- | --- | --- |
| Global_Network | FAILED | 2 | 0 |  |  |
| Global_LeadTime | FAILED | 40 | 1 |  |  |
| Global_DemandPriority | FAILED | 5 | 0 |  |  |
| M1_DemandForecast | FAILED | 2 | 0 |  |  |
| M1_OrderCalendar | PASS | 0 | 0 |  |  |
| M1_InitialInventory | FAILED | 2 | 0 |  |  |
| M1_ForecastError | FAILED | 2 | 0 |  |  |
| M3_SafetyStock | FAILED | 2 | 1 |  |  |
| M5_PushPullModel | FAILED | 1 | 0 |  |  |
| M5_DeployConfig | FAILED | 2 | 0 |  |  |
| M6_MaterialMD | FAILED | 1 | 0 |  |  |
| M6_TruckReleaseCon | PASS | 0 | 0 |  |  |
| M6_TruckTypeSpecs | FAILED | 2 | 0 |  |  |
| M6_DeliveryDelayDistribution | PASS | 0 | 0 |  |  |
| M1_AOConfig | FAILED | 205 | 0 |  |  |
| M1_DPSConfig | NOT MAINTAINED | 0 | 0 |  | Tab not provided |
| M1_SupplyChoiceConfig | NOT MAINTAINED | 0 | 0 |  | Tab not provided |
| M4_MaterialLocationLineCfg | FAILED | 195 | 1 |  |  |
| M4_LineCapacity | FAILED | 2 | 0 |  |  |
| M4_ProductionReliability | PASS | 0 | 0 |  |  |
| M4_ChangeoverDefinition | PASS | 0 | 0 |  |  |
| M4_ChangeoverMatrix | FAILED | 1 | 0 |  |  |
| Global_Seed | NOT MAINTAINED | 0 | 0 |  | Tab not provided |
| Global_SpaceCapacity | NOT MAINTAINED | 0 | 0 |  | Tab not provided |
| M6_MDQBypassRules | PASS | 0 | 0 |  |  |
| M6_TruckCapacityPlan | NOT MAINTAINED | 0 | 0 |  | Tab not provided |

## 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. |
| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. |
| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |

**Result: FAILED — Not ready.**


## 5. Rule coverage

| Rule family | Evaluated | Not applicable | Not-applicable IDs + reason |
| --- | --- | --- | --- |
| Availability | 0 | 2 | Availability-002 (M1_DPSConfig module disabled); Availability-002 (M1_SupplyChoiceConfig module disabled) |
| Schema | 0 | 0 | — |
| Consistency | 11 | 0 | — |
| Hard | 453 | 0 | — |
| Soft | 3 | 1 | Soft-001 (DFC init-inventory check skipped per user) |