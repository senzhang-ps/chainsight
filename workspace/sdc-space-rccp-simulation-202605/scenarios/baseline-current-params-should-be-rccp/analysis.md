# Analysis

## Evidence Level

Partial run readout from PostgreSQL `test_db`, filtered by run id
`db_SDC_V1_20260510_152317`.

- Inventory-oriented outputs currently available through `2026-07-06`:
	`orchestrator_unrestricted_inventory`, `module5_output_stockonhandlog`
- Order outputs available through `2026-08-10`, but shipment outputs only
	through `2026-07-06`; CFR below is therefore reported only for the
	overlapping window `2026-06-29` to `2026-07-06`
- `module4_output_productionplan` currently has `0` rows for this run, so
	no production-capacity interpretation is included yet

## Current Space RCCP Snapshot

Method used for this interim readout:

- Source table: `orchestrator_unrestricted_inventory`
- Scope filter: 12 `DC` locations from `cfg_global_network` where
	`config_name = 'SDC_V1'`
- Unit conversion: `quantity * cfg_m6_materialmd.demand_unit_to_volume`
- Aggregation: daily DC inventory in CBM, then monthly max by location

Current overall DC peak already written by the run:

- `2026-07-03`: `15,353,773.21` CBM
- `2026-07-04`: `15,246,642.18` CBM
- `2026-06-30`: `15,196,613.73` CBM

Current monthly peak by DC location:

### 2026-06

| location | monthly peak inventory cbm |
|---|---:|
| E556-ZZ | 2,576,536.74 |
| D767 | 2,116,555.69 |
| E569-CQ | 1,653,297.79 |
| E569-KM | 1,565,315.52 |
| C937 | 1,406,369.14 |
| E564-FZ | 1,369,908.05 |
| E564-NC | 1,353,936.98 |
| E568 | 1,189,058.55 |
| E569-GY | 982,693.20 |
| C819 | 741,336.03 |
| E560 | 498,888.61 |
| E556-QD | 289,563.48 |

### 2026-07

| location | monthly peak inventory cbm |
|---|---:|
| E556-ZZ | 2,670,599.82 |
| D767 | 1,958,932.61 |
| E569-CQ | 1,810,707.60 |
| E569-KM | 1,552,003.48 |
| E564-FZ | 1,509,912.76 |
| C937 | 1,374,557.92 |
| E564-NC | 1,322,014.43 |
| E568 | 1,126,213.51 |
| E569-GY | 1,102,487.63 |
| C819 | 684,868.53 |
| E560 | 475,587.69 |
| E556-QD | 254,586.08 |

Interim readout:

- The current peak DC space burden is concentrated in `E556-ZZ`, `D767`,
	and `E569-CQ`
- As of the currently written dates, `E556-ZZ` is the largest node in both
	June and July
- July data is still partial because the run has only written through
	`2026-07-06`

## Current CFR Snapshot

Using the overlapping output window `2026-06-29` to `2026-07-06`:

- Total shipment quantity: `294,707`
- Total deduplicated order quantity: `295,026`
- Interim CFR: `0.9989`

Daily CFR in the currently available window:

| date | shipment qty | order qty | cfr |
|---|---:|---:|---:|
| 2026-06-29 | 31,343 | 31,343 | 1.0000 |
| 2026-06-30 | 33,020 | 33,030 | 0.9997 |
| 2026-07-01 | 34,057 | 34,083 | 0.9992 |
| 2026-07-02 | 35,233 | 35,269 | 0.9990 |
| 2026-07-03 | 36,877 | 36,930 | 0.9986 |
| 2026-07-04 | 39,352 | 39,402 | 0.9987 |
| 2026-07-05 | 41,388 | 41,451 | 0.9985 |
| 2026-07-06 | 43,437 | 43,518 | 0.9981 |

## Cautions

- This is not a final scenario result because the run is still in progress
- Space-capacity comparison is not included yet in this readout
- Production-related KPIs are not yet available because module 4 outputs
	have not been written for this run

## By DC By Month Summary

Scope and timing notes:

- RCCP below is the monthly peak daily inventory in CBM for the 12 `DC`
	nodes already written to PostgreSQL
- CFR below is calculated only on the currently overlapping output window
	`2026-06-29` to `2026-07-06`, so both June and July are partial-month
	snapshots rather than final month-end values

### RCCP by DC by month

| month | DC | RCCP (CBM) |
|---|---|---:|
| 2026-06 | C819 | 741,336.03 |
| 2026-06 | C937 | 1,406,369.14 |
| 2026-06 | D767 | 2,116,555.69 |
| 2026-06 | E556-QD | 289,563.48 |
| 2026-06 | E556-ZZ | 2,576,536.74 |
| 2026-06 | E560 | 498,888.61 |
| 2026-06 | E564-FZ | 1,369,908.05 |
| 2026-06 | E564-NC | 1,353,936.98 |
| 2026-06 | E568 | 1,189,058.55 |
| 2026-06 | E569-CQ | 1,653,297.79 |
| 2026-06 | E569-GY | 982,693.20 |
| 2026-06 | E569-KM | 1,565,315.52 |
| 2026-07 | C819 | 684,868.53 |
| 2026-07 | C937 | 1,374,557.92 |
| 2026-07 | D767 | 1,958,932.61 |
| 2026-07 | E556-QD | 254,586.08 |
| 2026-07 | E556-ZZ | 2,670,599.82 |
| 2026-07 | E560 | 475,587.69 |
| 2026-07 | E564-FZ | 1,509,912.76 |
| 2026-07 | E564-NC | 1,322,014.43 |
| 2026-07 | E568 | 1,126,213.51 |
| 2026-07 | E569-CQ | 1,810,707.60 |
| 2026-07 | E569-GY | 1,102,487.63 |
| 2026-07 | E569-KM | 1,552,003.48 |

## Final Space RCCP By Month By DC

This section is the completed June-August RCCP readout for the full run,
using the same method as above but on the final written inventory window
through `2026-08-31`.

Method:

- Source table: `orchestrator_unrestricted_inventory`
- Scope: 12 `DC` locations from `cfg_global_network` where
	`config_name = 'SDC_V1'`
- Space conversion: `quantity * cfg_m6_materialmd.demand_unit_to_volume`
- RCCP definition: daily DC inventory in CBM, then monthly peak by DC

### 2026-06

| DC | RCCP (CBM) | peak date |
|---|---:|---|
| E556-ZZ | 2,576,536.74 | 2026-06-30 |
| D767 | 2,116,555.69 | 2026-06-29 |
| E569-CQ | 1,653,297.79 | 2026-06-29 |
| E569-KM | 1,565,315.52 | 2026-06-29 |
| C937 | 1,406,369.14 | 2026-06-29 |
| E564-FZ | 1,369,908.05 | 2026-06-29 |
| E564-NC | 1,353,936.98 | 2026-06-30 |
| E568 | 1,189,058.55 | 2026-06-29 |
| E569-GY | 982,693.20 | 2026-06-29 |
| C819 | 741,336.03 | 2026-06-29 |
| E560 | 498,888.61 | 2026-06-29 |
| E556-QD | 289,563.48 | 2026-06-29 |

### 2026-07

| DC | RCCP (CBM) | peak date |
|---|---:|---|
| E556-ZZ | 2,670,599.82 | 2026-07-01 |
| D767 | 1,958,932.61 | 2026-07-03 |
| E569-CQ | 1,810,707.60 | 2026-07-04 |
| E569-KM | 1,552,003.48 | 2026-07-04 |
| E564-FZ | 1,509,912.76 | 2026-07-01 |
| C937 | 1,374,557.92 | 2026-07-01 |
| E564-NC | 1,322,014.43 | 2026-07-01 |
| E568 | 1,126,213.51 | 2026-07-01 |
| E569-GY | 1,102,487.63 | 2026-07-04 |
| C819 | 776,674.72 | 2026-07-08 |
| E556-QD | 542,771.01 | 2026-07-29 |
| E560 | 475,587.69 | 2026-07-01 |

### 2026-08

| DC | RCCP (CBM) | peak date |
|---|---:|---|
| E556-ZZ | 1,466,636.98 | 2026-08-01 |
| E569-CQ | 1,172,147.38 | 2026-08-01 |
| D767 | 1,033,399.84 | 2026-08-01 |
| E569-KM | 902,120.20 | 2026-08-07 |
| E564-FZ | 824,340.36 | 2026-08-02 |
| E569-GY | 781,348.23 | 2026-08-01 |
| C819 | 759,196.25 | 2026-08-04 |
| C937 | 755,396.22 | 2026-08-31 |
| E564-NC | 541,239.60 | 2026-08-03 |
| E556-QD | 518,619.35 | 2026-08-05 |
| E568 | 348,492.93 | 2026-08-28 |
| E560 | 195,849.58 | 2026-08-13 |

Reading:

- July is the peak month for the network space burden; `E556-ZZ`, `D767`,
	`E569-CQ`, and `E569-KM` remain the largest RCCP nodes.
- August space drops broadly across the network, especially at `E556-ZZ`,
	`D767`, `E568`, `E564-NC`, and `E560`.
- `E556-QD` is the notable exception versus the earlier partial readout:
	its July and August monthly peaks are both materially above the early-July
	partial snapshot, reaching `542,771.01` CBM in July and `518,619.35` CBM in
	August.

### CFR by DC by month

| month | DC | shipment qty | order qty | CFR |
|---|---|---:|---:|---:|
| 2026-06 | C819 | 4,427.00 | 4,427.00 | 1.0000 |
| 2026-06 | C937 | 6,056.00 | 6,057.00 | 0.9998 |
| 2026-06 | D767 | 10,662.00 | 10,667.00 | 0.9995 |
| 2026-06 | E556-QD | 1,912.00 | 1,912.00 | 1.0000 |
| 2026-06 | E556-ZZ | 5,102.00 | 5,103.00 | 0.9998 |
| 2026-06 | E560 | 1,214.00 | 1,214.00 | 1.0000 |
| 2026-06 | E564-FZ | 5,578.00 | 5,579.00 | 0.9998 |
| 2026-06 | E564-NC | 3,776.00 | 3,776.00 | 1.0000 |
| 2026-06 | E568 | 3,160.00 | 3,160.00 | 1.0000 |
| 2026-06 | E569-CQ | 8,019.00 | 8,021.00 | 0.9998 |
| 2026-06 | E569-GY | 6,229.00 | 6,229.00 | 1.0000 |
| 2026-06 | E569-KM | 8,228.00 | 8,228.00 | 1.0000 |
| 2026-07 | C819 | 16,275.00 | 16,284.00 | 0.9994 |
| 2026-07 | C937 | 20,774.00 | 20,801.00 | 0.9987 |
| 2026-07 | D767 | 42,093.00 | 42,212.00 | 0.9972 |
| 2026-07 | E556-QD | 6,846.00 | 6,858.00 | 0.9983 |
| 2026-07 | E556-ZZ | 17,729.00 | 17,751.00 | 0.9988 |
| 2026-07 | E560 | 4,106.00 | 4,106.00 | 1.0000 |
| 2026-07 | E564-FZ | 19,278.00 | 19,299.00 | 0.9989 |
| 2026-07 | E564-NC | 13,033.00 | 13,048.00 | 0.9989 |
| 2026-07 | E568 | 11,510.00 | 11,517.00 | 0.9994 |
| 2026-07 | E569-CQ | 28,936.00 | 28,964.00 | 0.9990 |
| 2026-07 | E569-GY | 21,534.00 | 21,556.00 | 0.9990 |
| 2026-07 | E569-KM | 28,230.00 | 28,257.00 | 0.9990 |

## July CFR Drop Diagnosis

The July service decline is not explained by a total stop in transfer or a
missing transportation plan. The current database points to a different
failure mode:

- Direct mechanism: July shipments did not keep up with July order volume at
	several DCs, and the shipment shortfall matched the cut quantity in
	`module1_output_cutlog`
- Root-cause label from the engine: July unfulfilled quantities in
	`module5_output_unfulfilledlog` are consistently tagged as
	`supply shortage`
- Transport evidence: `module6_output_unsatisfiedmdqlog` is empty for this
	run, so there is no logged MDQ block / transport-threshold failure in the
	current outputs
- Transfer evidence: the low-CFR DCs still received inbound deliveries during
	July, so this is not a simple no-transfer situation
- Inventory nuance: the worst-affected DCs did not spend full days at zero
	total on-hand inventory, which suggests the shortage is at material mix /
	SKU level rather than a complete DC-level stockout

Key July shortfall examples by DC:

| DC | shipment qty | order qty | shortfall qty | CFR |
|---|---:|---:|---:|---:|
| E556-QD | 27,643 | 35,916 | 8,273 | 0.7697 |
| E569-GY | 91,818 | 107,738 | 15,920 | 0.8522 |
| E569-KM | 119,085 | 135,254 | 16,169 | 0.8805 |
| C937 | 91,939 | 102,722 | 10,783 | 0.8950 |
| C819 | 73,927 | 81,780 | 7,853 | 0.9040 |
| D767 | 222,238 | 244,418 | 22,180 | 0.9093 |

Unfulfilled-demand mix across all DCs in July:

| demand element | unfulfilled qty | share |
|---|---:|---:|
| safety | 607,413 | 85.69% |
| AO | 74,941 | 10.57% |
| normal | 26,506 | 3.74% |

Interpretation:

- July CFR fell primarily because supply shortage prevented enough quantity
	from being shipped against demand
- The shortage is dominated by `safety` replenishment pressure rather than
	`normal` demand alone
- There were inbound transfers in July, but they were not sufficient to close
	the demand gap at the affected DCs

## July By Receiving CFR vs PDT Through 2026-07-26

Method note:

- CFR below is aggregated by receiving DC over `2026-07-01` to `2026-07-26`
- Actual timing below is calculated from inbound shipments with
	`actual_ship_date` in the same window
- Actual `PDT = waiting MOQ + actual OTD`
- Receiving-level configured PDT is the average of inbound lane config values
	observed in shipped records during the window

| receiving | shipment qty | order qty | shortfall qty | CFR | actual waiting MOQ days | actual OTD days | actual PDT days | cfg PDT days | PDT gap days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E556-QD | 27,643.00 | 35,916.00 | 8,273.00 | 0.7697 | 2.7992 | 9.7595 | 12.5587 | 10.7132 | 1.8455 |
| E569-GY | 91,818.00 | 107,738.00 | 15,920.00 | 0.8522 | 0.9761 | 3.8824 | 4.8585 | 4.6037 | 0.2548 |
| E569-KM | 119,085.00 | 135,254.00 | 16,169.00 | 0.8805 | 0.8926 | 4.0579 | 4.9506 | 5.5704 | -0.6199 |
| C937 | 91,939.00 | 102,722.00 | 10,783.00 | 0.8950 | 1.5035 | 3.0170 | 4.5204 | 3.9775 | 0.5430 |
| C819 | 73,927.00 | 81,780.00 | 7,853.00 | 0.9040 | 1.6082 | 5.5080 | 7.1163 | 6.3726 | 0.7436 |
| D767 | 222,238.00 | 244,418.00 | 22,180.00 | 0.9093 | 0.4320 | 2.3914 | 2.8234 | 3.5151 | -0.6917 |
| E564-FZ | 87,756.00 | 95,999.00 | 8,243.00 | 0.9141 | 1.7457 | 3.8071 | 5.5528 | 4.8825 | 0.6703 |
| E569-CQ | 142,933.00 | 154,043.00 | 11,110.00 | 0.9279 | 0.4454 | 3.8091 | 4.2545 | 4.4604 | -0.2059 |
| E568 | 56,178.00 | 59,164.00 | 2,986.00 | 0.9495 | 2.6303 | 7.0772 | 9.7076 | 7.1873 | 2.5203 |
| E564-NC | 62,288.00 | 63,315.00 | 1,027.00 | 0.9838 | 2.8485 | 4.6746 | 7.5231 | 6.0469 | 1.4762 |
| E560 | 19,855.00 | 20,045.00 | 190.00 | 0.9905 | 4.3063 | 6.7456 | 11.0519 | 7.8529 | 3.1990 |
| E556-ZZ | 93,370.00 | 93,589.00 | 219.00 | 0.9977 | 2.6035 | 1.9924 | 4.5959 | 3.1304 | 1.4656 |

Interpretation:

- PDT mismatch contributes to slow replenishment for some DCs, but it is not
	a complete explanation of July CFR loss across all receiving nodes
- `E556-QD` is the clearest case where low CFR and delayed replenishment move
	together: CFR is `0.7697` and actual PDT is `1.8455` days slower than config
- `C819`, `C937`, and `E564-FZ` also show both below-target CFR and positive
	PDT gap, which is consistent with replenishment arriving later than plan
- But `E569-KM` and `D767` still have low CFR even though actual PDT is
	better than configured PDT, so those nodes are not explained by PDT delay
	alone
- `E556-ZZ` and `E560` show that a large PDT gap does not automatically force
	poor CFR; service can still remain high when supply is sufficient
- The strongest current reading is: July service loss is primarily a
	supply-availability issue, while PDT mismatch is a secondary amplifier on
	selected receiving nodes, especially `E556-QD`

## RCA Report — Ordered Jul-Aug Classification

This section supersedes the earlier partial-run RCA. It uses completed
July-August service data for run id `db_SDC_V1_20260510_152317` and assigns
each shortfall material-location in low-CFR receiving DCs by the user-requested
order:

Important interpretation note:

- The classification is applied only to the residual `shortfall_qty` after the
	DC has already shipped whatever it could from available stock.
- So a DC can still fulfill part of the orders from its own on-hand inventory;
	that fulfilled part is already reflected in `shipment_qty` and is not what
	these buckets are explaining.
- `non-self` means `sending <> receiving`. Self-loop rows are excluded because
	they do not create new inbound supply into the receiving DC.

1. `1_no_safety_stock`: there are orders but no positive safety stock setting.
2. `2_no_deployment_plan`: safety stock exists, but no non-self deployment plan
	was produced to replenish the remaining shortage.
3. `3_deployed_never_shipped`: deployment exists, but no non-self shipment was
	written against that planned replenishment.
4. `4_pdt_gr_mismatch`: shipment exists, but weighted actual total lead time
	`PDT + GR` is more than `0.5` day slower than configured `pdt + gr`.

Low-CFR receiving scope is July-August CFR below `0.95`:

- `E556-QD`, `E568`, `E569-GY`, `C937`, `E569-KM`, `E564-FZ`, `E564-NC`,
	`E560`, `C819`, `D767`, `E569-CQ`, `E556-ZZ`

### By Receiving DC

| receiving | CFR | dominant cause | dominant shortfall qty | second signal | second qty |
|---|---:|---|---:|---|---:|
| E556-QD | 0.7386 | 1_no_safety_stock | 15,053.00 | 4_pdt_gr_mismatch | 8,618.00 |
| E568 | 0.7848 | 1_no_safety_stock | 22,694.00 | 4_pdt_gr_mismatch | 5,697.00 |
| E569-GY | 0.7913 | 1_no_safety_stock | 50,192.00 | residual_other | 4,694.00 |
| C937 | 0.7954 | 1_no_safety_stock | 46,334.00 | residual_other | 5,195.00 |
| E569-KM | 0.8031 | 1_no_safety_stock | 58,669.00 | residual_other | 5,403.00 |
| E564-FZ | 0.8162 | 1_no_safety_stock | 37,290.00 | 4_pdt_gr_mismatch | 3,427.00 |
| E564-NC | 0.8216 | 1_no_safety_stock | 23,282.00 | 4_pdt_gr_mismatch | 3,056.00 |
| E560 | 0.8438 | 4_pdt_gr_mismatch | 4,691.00 | residual_other | 1,855.00 |
| C819 | 0.8710 | 1_no_safety_stock | 18,310.00 | residual_other | 5,946.00 |
| D767 | 0.8807 | 1_no_safety_stock | 68,771.00 | residual_other | 3,450.00 |
| E569-CQ | 0.8888 | 1_no_safety_stock | 37,970.00 | residual_other | 4,713.00 |
| E556-ZZ | 0.9027 | 1_no_safety_stock | 21,690.00 | residual_other | 628.00 |

### Ordered Findings

#### 1. Inventory target setting issue is the primary root cause

This is the dominant bucket in 11 of 12 low-CFR receiving DCs. The biggest
shortfall SKUs all fall into `1_no_safety_stock` even though deployment and
shipment records do exist for those SKUs.

Representative examples:

| receiving | material | shortfall qty | safety stock | deploy qty | shipped qty | actual total LT | cfg total LT |
|---|---|---:|---:|---:|---:|---:|---:|
| E569-KM | 80736570 | 12,772.00 | 0.00 | 6,363.00 | 6,355.00 | 2.0453 | 4.3000 |
| E569-GY | 80736570 | 11,862.00 | 0.00 | 5,259.00 | 5,259.00 | 2.6098 | 3.6000 |
| C937 | 80856618 | 9,635.00 | 0.00 | 3,509.00 | 3,291.00 | 4.8857 | 6.2000 |
| D767 | 80893106 | 7,574.00 | 0.00 | 2,399.00 | 2,399.00 | 3.3918 | 4.8000 |

Reading:

- The first failure is usually not transport and not slow lead time.
- The main problem is that the shortfall material-location entered the run
	without a positive safety-stock target, so inventory target setting was too
	weak relative to realized demand.

#### 2. Deployment-plan miss exists, but it is not material

Only one classified case landed in `2_no_deployment_plan`, and the shortfall is
trivial:

| receiving | material | shortfall qty | note |
|---|---|---:|---|
| E556-QD | one SKU | 5.00 | safety stock exists, but no non-self deployment plan was written |

Reading:

- This bucket exists in the data, but it is not the explanation for the July-
	August CFR loss.
- It does not mean the DC could not ship anything locally; it means the unmet
	part of demand was not backed by any external replenishment plan.

#### 3. Deployed but never shipped was not observed in the ordered RCA set

No low-CFR material-location was classified into `3_deployed_never_shipped`
under the July-August ordered logic.

Reading:

- There are open deployments in the broader run history, but for the final
	July-August low-CFR classification, this is not a primary root cause bucket.
- The service problem is therefore not explained by a widespread pattern of
	planned replenishment that never got a shipment record.
- If a DC shipped some orders from local stock, that is still compatible with
	this definition; the bucket is only about whether an external replenishment
	plan later turned into an actual shipment.

#### 4. PDT + GR mismatch is a secondary but real cause on selected DCs

This bucket is meaningful at `E560`, and secondary at `E556-QD`, `E568`,
`E564-FZ`, `E564-NC`, `C819`, and `E556-ZZ`.

Receiving-level totals:

| receiving | 4_pdt_gr_mismatch shortfall qty |
|---|---:|
| E556-QD | 8,618.00 |
| E568 | 5,697.00 |
| E560 | 4,691.00 |
| E564-FZ | 3,427.00 |
| E564-NC | 3,056.00 |
| C819 | 1,431.00 |
| E556-ZZ | 342.00 |

Reading:

- `E560` is the only low-CFR DC where lead-time mismatch is the dominant
	bucket instead of missing safety stock.
- `E556-QD` is still mainly a missing-safety-stock case, but late actual
	`PDT + GR` adds a large secondary penalty.
- For most other low-CFR DCs, lead-time mismatch matters less than the missing
	safety-stock issue.

Lane-level decomposition for the mismatch lanes:

- Scope below is restricted to receiving DCs already flagged in
	`4_pdt_gr_mismatch`, then further filtered to lanes where simulated total
	lead time is more than `0.5` day slower than configured total lead time.
- Simulated `waiting MOQ = actual_ship_date - planned_deployment_date`.
- Simulated `OTD = actual_delivery_date - actual_ship_date`.
- Simulated `GR = goods receipt date - actual_delivery_date`.
- Configured `waiting MOQ` is derived as `cfg pdt - cfg otd`, because
	`cfg_global_leadtime` stores `pdt`, `otd`, and `gr` separately.

| sending | receiving | delivered qty | sim waiting MOQ | cfg waiting MOQ | waiting gap | sim OTD | cfg OTD | OTD gap | sim GR | cfg GR | GR gap | sim total LT | cfg total LT | total gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D235 | E560 | 1,733.00 | 3.2314 | 0.5000 | 2.7314 | 13.1927 | 12.0000 | 1.1927 | 0.0000 | 1.0000 | -1.0000 | 16.4241 | 13.5000 | 2.9241 |
| C816 | E564-NC | 13,394.00 | 3.7669 | 1.0000 | 2.7669 | 13.2005 | 13.0000 | 0.2005 | 0.0000 | 0.4000 | -0.4000 | 16.9674 | 14.4000 | 2.5674 |
| A673 | E560 | 8,777.00 | 3.8338 | 1.0000 | 2.8338 | 12.2328 | 11.5000 | 0.7328 | 0.0000 | 1.0000 | -1.0000 | 16.0665 | 13.5000 | 2.5665 |
| C816 | E564-FZ | 24,810.00 | 3.2437 | 1.0000 | 2.2437 | 10.2123 | 10.0000 | 0.2123 | 0.0000 | 0.3000 | -0.3000 | 13.4560 | 11.3000 | 2.1560 |
| A672 | E556-QD | 37,828.00 | 3.3558 | 1.0000 | 2.3558 | 8.2017 | 8.3000 | -0.0983 | -0.0205 | 0.5000 | -0.5205 | 11.5370 | 9.8000 | 1.7370 |
| A673 | E556-QD | 18,016.00 | 3.0624 | 1.0000 | 2.0624 | 16.2141 | 15.8000 | 0.4141 | -0.0391 | 1.0000 | -1.0391 | 19.2375 | 17.8000 | 1.4375 |
| A672 | E560 | 19,889.00 | 3.3711 | 1.0000 | 2.3711 | 4.2020 | 4.7000 | -0.4980 | 0.0000 | 0.7000 | -0.7000 | 7.5730 | 6.4000 | 1.1730 |
| A673 | E556-ZZ | 43,731.00 | 3.0696 | 1.0000 | 2.0696 | 1.1951 | 0.8000 | 0.3951 | 0.0000 | 1.4000 | -1.4000 | 4.2647 | 3.2000 | 1.0647 |
| A673 | C819 | 47,631.00 | 2.7640 | 1.0000 | 1.7640 | 8.1839 | 8.2000 | -0.0161 | -0.0042 | 0.7000 | -0.7042 | 10.9436 | 9.9000 | 1.0436 |
| A672 | E568 | 57,598.00 | 2.2661 | 1.0000 | 1.2661 | 7.2102 | 6.2000 | 1.0102 | -0.0322 | 1.3000 | -1.3322 | 9.4440 | 8.5000 | 0.9440 |

What this means:

- The dominant gap is `waiting MOQ`, not `GR`. All listed lanes have waiting
	time above config by `1.27` to `2.83` days.
- Simulated `GR` is near zero on these lanes and is consistently below the
	configured `gr`, so the mismatch is not coming from downstream receipt delay.
- `OTD` is a secondary contributor on a few lanes, especially `D235 -> E560`,
	`A673 -> E560`, and `A672 -> E568`, but it is smaller than the waiting-MOQ
	gap on most lanes.
- So the current `PDT + GR mismatch` is best read as a release / waiting issue
	first, with transport transit only contributing on selected lanes.

### Final RCA

The ordered July-August RCA is clear:

1. The primary root cause is inventory target setting: most shortfall SKUs in
	low-CFR DCs had orders but no positive safety stock setting.
2. Missing deployment plan exists only as a tiny edge case and does not explain
	the service loss.
3. Deployed-but-never-shipped is not present as a classified root cause in the
	final July-August low-CFR set.
4. Parameter centerline mismatch in `PDT + GR` is real, but it is a secondary
	cause except for `E560` and part of `E556-QD`.

Practical conclusion:

- Fix safety-stock / target-setting first for `D767`, `E569-KM`, `E569-GY`,
	`C937`, `E564-FZ`, `E569-CQ`, `E568`, `E564-NC`, `C819`, and `E556-ZZ`.
- Treat `E556-QD` as a mixed case: missing safety stock first, then actual
	`PDT + GR` slower than config.
- Treat `E560` as the clearest lead-time-centerline issue in the low-CFR set.