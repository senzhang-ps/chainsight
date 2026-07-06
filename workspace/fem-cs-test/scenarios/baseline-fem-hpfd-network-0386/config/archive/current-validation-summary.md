# Current Validation Summary

Configuration file:
- `workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/baseline-fem-hpfd-network-0386.xlsx`

Validation re-check date:
- `2026-05-22`

## Overall status
- Configuration is **not fully ready** yet.
- After today's fixes, the remaining issues are concentrated in:
  - `M1_DemandForecast`
  - `M3_SafetyStock`
  - `M4_MaterialLocationLineCfg`
  - `M4_LineCapacity`
  - `M6_MaterialMD`

## Remaining issues

### 1. M1_DemandForecast
**Issue**
- Week field has been successfully recoded to numeric.
- However, week coverage is still incomplete.

**Current status**
- `min week = 1`
- `max week = 13`

**Remaining gap**
- `week 14`
- `week 15`

**Action owner**
- Manual follow-up

---

### 2. M3_SafetyStock
#### 2.1 Missing material-location combinations
**Issue**
- `31` material-location combinations are still missing.

**Sample**
- `80799573-E568`
- `80799574-C563`
- `80799575-E556`
- `80799576-E556`
- `80799577-E556`
- `80817373-D594`
- `80817373-E230`
- `80817374-D594`
- `80817374-E230`
- `80817375-D594`

**Decision taken**
- This batch is currently treated as acceptable and was not filled in this round.

#### 2.2 Extra material-location combinations
**Issue**
- `37` extra material-location combinations still remain.

**Sample**
- `80728859-0386`
- `80728861-0386`
- `80799571-0386`
- `80799573-0386`
- `80799574-0386`
- `80799575-0386`
- `80799576-0386`
- `80799577-0386`
- `80804391-0386`
- `80817373-0386`

#### 2.3 Daily coverage incomplete
**Issue**
- `394` material-location combinations do not yet have full daily coverage across the simulation period.

**Sample**
- `80728859-A668`
- `80728859-A672`
- `80728859-A673`
- `80728859-A680`
- `80728859-A715`
- `80728859-A716`
- `80728859-C810`
- `80728859-C816`
- `80728859-C819`

**Action owner**
- Manual follow-up

---

### 3. M4_MaterialLocationLineCfg
**Issue**
- `29` rows still fail numeric validation.

**Root cause**
- Remaining invalid field is `rv`, which is still blank.

**Sample rows**
- `80878361-0386`
- `80878362-0386`
- `80878369-0386`
- `80799574-0386`
- `80799573-0386`
- `80856029-0386`
- `80799571-0386`
- `80858047-0386`
- `80799576-0386`
- `80817373-0386`

**Action owner**
- Manual follow-up

---

### 4. M4_LineCapacity
**Issue**
- Date coverage is still incomplete for one maintained line.

**Affected combination**
- `0386 - HPFD`

**Action owner**
- Manual follow-up

---

### 5. M6_MaterialMD
**Issue**
- `7` materials are still missing from `M6_MaterialMD`.

**Missing materials**
- `80817373`
- `80817374`
- `80817375`
- `80817378`
- `80892811`
- `82278637`
- `83900927`

**Action owner**
- Manual follow-up

---

## Notes
- `0386` is treated in this working session as a **sourcing-only location**.
- Some historical validation findings related to `0386` are considered rule issues rather than configuration defects.
- Validation skill logic should be updated accordingly in the future.
