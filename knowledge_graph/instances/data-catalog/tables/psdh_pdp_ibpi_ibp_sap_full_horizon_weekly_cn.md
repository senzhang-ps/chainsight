---
fqn: cdl_ps_hana_prd.ods.psdh_pdp_ibpi_ibp_sap_full_horizon_weekly_cn
description: IBPI safety stock proposal with full horizon — weekly safety stock recommendations, demand stats, and inventory parameters
synced_at: '2026-04-10T10:20:39Z'
tags:
- safety_stock
- ibpi
- scope_source
related_ontology:
- class: cs:SafetyStock
  role: primary
related_config:
- M3_SafetyStock
- M1_ForecastError
scope_dimensions:
- locid
- prdid
- keyfiguredate
- pgsubsector
columns:
- name: skey
  type: BIGINT
  desc: Skey
- name: locid
  type: STRING
  desc: location id
- name: prdid
  type: STRING
  desc: product id
- name: keyfiguredate
  type: INT
  desc: week timing
- name: extractiondate
  type: STRING
  desc: extraction date
- name: prddescr
  type: STRING
  desc: product description
- name: planner
  type: STRING
  desc: planner
- name: pgtdcval
  type: STRING
  desc: tdc val
- name: brand
  type: STRING
  desc: brand id
- name: category
  type: STRING
  desc: category
- name: pgcustomizationtype
  type: STRING
  desc: pg customization type
- name: pgaggcov
  type: STRING
  desc: pg agg cov
- name: pgevdayavai
  type: STRING
  desc: pgevdayavai
- name: pgchangedate
  type: INT
  desc: in forecast change date
- name: pgspaced
  type: STRING
  desc: pg spaced
- name: locdescr
  type: STRING
  desc: location description
- name: pgmrpcontroller
  type: STRING
  desc: pg mrp controller
- name: pgownership
  type: STRING
  desc: ownership
- name: pgconversion
  type: STRING
  desc: pg conversion
- name: pgstockingprofile
  type: STRING
  desc: pg stocking profile
- name: pgproducingplant
  type: STRING
  desc: producing plant
- name: pgreviewneed
  type: STRING
  desc: pg review need
- name: pgsector
  type: STRING
  desc: pg sector
- name: locfr
  type: STRING
  desc: location from
- name: stockingnodetype
  type: STRING
  desc: stocking node type
- name: pgsubsector
  type: STRING
  desc: subsector
- name: pgtransitionperiod
  type: INT
  desc: in transition period weeks
- name: pgtransitiontype
  type: STRING
  desc: in transition type
- name: uomid
  type: STRING
  desc: uom id
- name: plunitid
  type: STRING
  desc: plunitid
- name: zioaggsales
  type: DECIMAL
  desc: 'DD_3: Aggregated Customer Shipments [QTY]'
- name: zioaggforecast
  type: DECIMAL
  desc: 'DD_3: Aggregated Demand Forecast [QTY]'
- name: zioaggfcstbiasused
  type: DECIMAL
  desc: 'DD_3: Aggregated Demand Forecast Bias Used [%]'
- name: zpublishindicator
  type: DECIMAL
  desc: 'PC_1i: Published Indicator [flag]'
- name: finaliosafetystock
  type: DECIMAL
  desc: 'IP_1: Final Safety Stock [QTY]'
- name: zioforecast
  type: DECIMAL
  desc: 'DD_1: Demand Forecast [QTY]'
- name: zrecomsafetyfinalqty
  type: DECIMAL
  desc: 'IP_1: Recommended Safety Stock Adjusted [QTY]'
- name: adjustediosafetystock
  type: DECIMAL
  desc: 'IP_1s: Safety Override by System Owner [QTY]'
- name: propagateddemandmean
  type: DECIMAL
  desc: 'DD_1: Propagated (=filtered) Demand Mean (Node) [QTY]'
- name: ziosalesqty
  type: DECIMAL
  desc: 'DD_2: Customer Shipments [QTY]'
- name: zioforecastqty
  type: DECIMAL
  desc: 'DD_2: Demand Forecast original-for reference [QTY]'
- name: zincrhppptotalsafetystockqty
  type: DECIMAL
  desc: 'HP_2: HPPP Total Incremental Safety [QTY]'
- name: zfinalsafetydaysofsupply
  type: DECIMAL
  desc: 'IP_1: Final Safety Stock [days]'
- name: finaliosafetystockval
  type: DECIMAL
  desc: 'IP_1: Final Safety Stock Value [USD]'
- name: zopsafetydays
  type: DECIMAL
  desc: 'IP_1: Operational Safety Days [days]'
- name: zopsafetystockqty
  type: DECIMAL
  desc: 'IP_1: Operational Safety Stock [QTY]'
- name: iosafetystockdaysofsupply
  type: DECIMAL
  desc: 'IP_1: Recommended Safety Stock Original [days]'
- name: zrecommendedsafetystock
  type: DECIMAL
  desc: 'IP_1: Recommended Safety Stock Original [QTY]'
- name: recommendedsafetystockval
  type: DECIMAL
  desc: 'IP_1: Recommended Safety Stock Value - original [USD]'
- name: zminimumsafetyinputqty
  type: DECIMAL
  desc: 'MM_1i: Minimum Safety [QTY]'
- name: zminimumsafetymoqmult
  type: DECIMAL
  desc: 'MM_1i: Minimum Safety MOQ Multiplier [#]'
- name: zreviewindicator
  type: DECIMAL
  desc: 'IP_1i: Review Indicator [flag]'
- name: zincrdrpsafetystockqty
  type: DECIMAL
  desc: 'IP_1s: Planner Adjustment as per BI [QTY]'
- name: internalavailableinfull
  type: DECIMAL
  desc: 'IP_2: Internal Non-Stockout Probability [%]'
- name: averageservicelevel
  type: DECIMAL
  desc: 'IP_3: Average Service Level [%]'
- name: ioavgcyclestock
  type: DECIMAL
  desc: 'IP_3: Cycle Stock (Average) [QTY]'
- name: iotargetcyclestock
  type: DECIMAL
  desc: 'IP_3: Cycle Stock (Target) [QTY]'
- name: ioavgcyclestockval
  type: DECIMAL
  desc: 'IP_3: Cycle Stock Value (Average) [USD]'
- name: iotargetcyclestockval
  type: DECIMAL
  desc: 'IP_3: Cycle Stock Value (Target) [QTY][BuoM]'
- name: safetystockdemandvar
  type: DECIMAL
  desc: 'IP_3: Demand Variability Safety Stock [QTY]'
- name: outgoingbacklogmean
  type: DECIMAL
  desc: 'IP_3: Internal Backorder Mean [QTY][BuoM]'
- name: outgoingbacklogstddev
  type: DECIMAL
  desc: 'IP_3: Internal Backorder Std Dev. [QTY][BuoM]'
- name: outgoingsrctolocbacklogmean
  type: DECIMAL
  desc: 'IP_3: Production Backorder Mean [QTY]'
- name: outgoingsrctolocbacklogstddev
  type: DECIMAL
  desc: 'IP_3: Production Backorder Std Dev. [QTY]'
- name: safetystockservicevar
  type: DECIMAL
  desc: 'IP_3: Service Variability Safety Stock [QTY]'
- name: safetystocksupplyvar
  type: DECIMAL
  desc: 'IP_3: Supply Variability Safety Stock [QTY]'
- name: safetystocklotsize
  type: DECIMAL
  desc: 'IP_3: Zero Lot Size Additional Safety Stock [QTY]'
- name: pbr
  type: DECIMAL
  desc: 'MD_1: Periods Between Reviews [weeks]'
- name: pinclotsize
  type: DECIMAL
  desc: 'MD_1: Production Incremental Lot Size [QTY][BuoM]'
- name: pleadtime
  type: DECIMAL
  desc: 'MD_1: Production Lead Time [weeks]'
- name: pleadtimevariability
  type: DECIMAL
  desc: 'MD_1: Production Lead Time Error CV [%]'
- name: pminlotsize
  type: DECIMAL
  desc: 'MD_1: Production Minimum Lot Size [QTY][BuoM]'
- name: targetservicelevel
  type: DECIMAL
  desc: 'MD_1: Target Service Level [%]'
- name: ztleadtimeincr
  type: DECIMAL
  desc: 'MD_1: Transportation Incremental Lead Time [weeks]'
- name: tinclotsize
  type: DECIMAL
  desc: 'MD_1: Transportation Incremental Lot Size [QTY][BuoM]'
- name: tleadtime
  type: DECIMAL
  desc: 'MD_1: Transportation Lead Time [weeks]'
- name: tminlotsize
  type: DECIMAL
  desc: 'MD_1: Transportation Incremental Lot Size [QTY][BuoM]'
- name: zdrpreactiontime
  type: DECIMAL
  desc: 'MD_1i: DRP Reaction Time [DAYS]'
- name: zpbror
  type: DECIMAL
  desc: 'MD_1i: Periods Between Review Over-ride [weeks]'
- name: zpinclotsizeor
  type: DECIMAL
  desc: 'MD_1i: Production Incremental Lot Size Over-ride [QTY]'
- name: zpleadtimeor
  type: DECIMAL
  desc: 'MD_1i: Production Lead Time Over-ride [weeks]'
- name: zpminlotsizeor
  type: DECIMAL
  desc: 'MD_1i: Production Minimum Lot Size Over-ride [QTY]'
- name: zstockingprofile
  type: DECIMAL
  desc: 'MD_1i: Stocking Profile'
- name: ztargetservicelevelor
  type: DECIMAL
  desc: 'MD_1i: Target Service Level Over-ride [%]'
- name: ztinclotsizeor
  type: DECIMAL
  desc: 'MD_1i: Transportation Incremental Lot Size Over-ride [QTY]'
- name: ztleadtimeor
  type: DECIMAL
  desc: 'MD_1i: Transportation Lead Time Over-ride [weeks]'
- name: ztminlotsizeor
  type: DECIMAL
  desc: 'MD_1i: Transportation Minimum Lot Size Over-ride [QTY]'
- name: dependentlocationdemandstddev
  type: DECIMAL
  desc: 'MD_2: Dependent Demand Std Dev. [QTY][BuoM]'
- name: maxinternalservicelevel
  type: DECIMAL
  desc: 'MD_2: Maximum Internal Service Level [%]'
- name: mininternalservicelevel
  type: DECIMAL
  desc: 'MD_2: Minimum Internal Service Level [%]'
- name: tleadtimevariability
  type: DECIMAL
  desc: 'MD_1: Transportation Lead Time Error CV [%]'
- name: zfrozenperiod
  type: DECIMAL
  desc: 'MD_2i: Frozen Period for writeback [weeks]'
- name: zmaxleadtime
  type: DECIMAL
  desc: 'MD_3: Maximum Lead-time (use for LAG)'
- name: zmaxsafetydays
  type: DECIMAL
  desc: 'MX_1: Maximum Stock [days]'
- name: zmaxsafetystockqty
  type: DECIMAL
  desc: 'MX_1: Maximum Stock [QTY]'
- name: zmaxstockhorizon
  type: DECIMAL
  desc: 'MX_1i: Horizon for Maximum Buildup [weeks]'
- name: zsafetychangesummaryalert
  type: DECIMAL
  desc: 'RN_1: ALERT Safety Change Summary'
- name: zsafetydayspctchangealert
  type: DECIMAL
  desc: 'RN_1: ALERT Safety Days [%]'
- name: zsafetydayschangealert
  type: DECIMAL
  desc: 'RN_1: ALERT Safety Days [days]'
- name: zrnsafetynotzeroalert
  type: DECIMAL
  desc: 'RN_1: ALERT Safety Not Zero'
- name: zsafetyqtypctchangealert
  type: DECIMAL
  desc: 'RN_1: ALERT Safety QTY [%]'
- name: zsafetyqtychangealert
  type: DECIMAL
  desc: 'RN_1: ALERT Safety QTY [QTY]'
- name: zfcstshpmntcheck
  type: DECIMAL
  desc: 'RN_1: Code has forecast or shipments'
- name: zdevfinalsafetyvslcdayspct
  type: DECIMAL
  desc: 'RN_1: Safety Days Change [%]'
- name: zdevfinalsafetyvslcdays
  type: DECIMAL
  desc: 'RN_1: Safety Days Change [days]'
- name: zdevfinalsafetyvslcqtypct
  type: DECIMAL
  desc: 'RN_1: Safety QTY Change [%]'
- name: zdevfinalsafetyvslcqty
  type: DECIMAL
  desc: 'RN_1: Safety QTY Change [QTY]'
- name: zfcstqtypctchangealert
  type: DECIMAL
  desc: 'RN_2: ALERT Forecast [%]'
- name: zfcstqtychangealert
  type: DECIMAL
  desc: 'RN_2: ALERT Forecast [QTY]'
- name: zopsafetyoverridecheck
  type: DECIMAL
  desc: zop safety override check
- name: zsafetydaysover
  type: DECIMAL
  desc: 'RN_2: ALERT Safety Days beyond limit'
- name: zrnswitchdatealert
  type: DECIMAL
  desc: 'RN_2: ALERT Switch Date Mismatch'
- name: zdevioforecastvslcqtypct
  type: DECIMAL
  desc: 'RN_2: Forecast Change [%]'
- name: zdevioforecastvslcqty
  type: DECIMAL
  desc: 'RN_2: Forecast Change [QTY]'
- name: ziolincrcheck
  type: DECIMAL
  desc: ziolincrcheck
- name: zrnmissedvolumeqty
  type: DECIMAL
  desc: 'RN_2: Missed Volume [QTY]'
- name: zshipvsfcstcheck
  type: DECIMAL
  desc: 'RN_2: ShipVsFcst [%]'
- name: zfinalreviewcheck
  type: DECIMAL
  desc: 'RN_3: Final RN Status'
- name: zopsafetystockqtyval
  type: DECIMAL
  desc: 'IP_1: Operational Safety Stock [USD]'
- name: ztotalopsafetystockqtyval
  type: DECIMAL
  desc: 'IP_1: Total Operational Safety Stock [USD]'
- name: ztotalopsafetystockqty
  type: DECIMAL
  desc: 'IP_1: Total Operational Safety Stock [QTY]'
- name: ziosalesqtyval
  type: DECIMAL
  desc: 'DD_2: Customer Shipments [USD]'
- name: zioforecastqtyval
  type: DECIMAL
  desc: 'DD_2: Demand Forecast original-for reference [USD]'
- name: zminimumsafetydaysqty
  type: DECIMAL
  desc: 'MM_1: Minimum Safety Qty from Days [QTY]'
- name: zmaxsafetyinputqty
  type: DECIMAL
  desc: 'MM_1i: Maximum Safety [QTY]'
- name: zmaximumsafetydaysinqty
  type: DECIMAL
  desc: 'MM_1: Maximum Safety Qty from Days [QTY]'
- name: zrecomsafetyfinalqtydos
  type: DECIMAL
  desc: 'IP_1: Recommended Safety Stock Adjusted [days]'
- name: pglocprodregion
  type: STRING
  desc: pg loc prod region
- name: pgloccountry
  type: STRING
  desc: pg loc country
- name: zrollmaxandrecsafetyorropqty
  type: DECIMAL
  desc: 'MM_1: Minimum Safety for Rolling max with Rec SS or ROP[QTY]'
- name: zsafetyadjreasoncode
  type: STRING
  desc: 'IP_1i : Safety Adjustment Reason Code'
- name: znrltexport
  type: DECIMAL
  desc: 'MD_3: Calculated Total Leadtime [days]'
- name: zmaxstockcycleqty
  type: DECIMAL
  desc: 'MX_1: Cycle Stock [QTY]'
- name: pgcoverrule1ovr
  type: STRING
  desc: pgcoverrule1ovr
- name: zhsafetyadjreasoncode
  type: STRING
  desc: 'IP_1i : Safety Adjustment Reason Code ( Helper)'
- name: zdexclusionindicatoror
  type: INT
  desc: 'DX: Demand Exclusion Indicator Override'
- name: zpropagateddemandmeanori
  type: DECIMAL
  desc: 'DD_1: Original Propagated (=total) Demand Mean (Node) [QTY]'
- name: zpropagateddemandmeanexp
  type: DECIMAL
  desc: 'DD_1: MEA Propagated Demand with Exclusions [QTY]'
- name: zmaximumsafetydays
  type: DECIMAL
  desc: 'MM_1i: Maximum Safety [days]'
- name: zadvancedperiodqty
  type: DECIMAL
  desc: 'IN_1: Anticipation Stock-calc [QTY]'
- name: pgminiolss
  type: DECIMAL
  desc: 'MM_1: Anticipation Safety from IOPT [QTY]'
- name: distributiontype
  type: STRING
  desc: distribution type
- name: ziolagforecasterrorcvcustface
  type: DECIMAL
  desc: ziolagforecasterrorcvcustface
- name: zdtdistributiontype
  type: STRING
  desc: 'MD_1: Distribution Type over-ride [1(N), 2(G)]'
- name: zdexclusionindicator
  type: STRING
  desc: 'DX: Demand Exclusion Indicator'
- name: zhpipoandfznperiod
  type: INT
  desc: 'IP_1: Safety Override by PiPo or Frozen Horizon [QTY]'
- name: zhpppsafetyoverrideqty
  type: DECIMAL
  desc: 'HP_2: HPPP Safety Override Qty'
- name: zfinalwdsafetydaysofsupply
  type: INT
  desc: 'IP_1: Final Safety Days using work days [days]'
- name: pgadvanceperiods
  type: INT
  desc: in advance vs changedate weeks
- name: pgafterperiod
  type: INT
  desc: in after vs changedate weeks
- name: pgdaysofsafety
  type: INT
  desc: in anticipation stock days
- name: pganticipationcalc
  type: DECIMAL
  desc: in anticipation method
- name: pgcalname
  type: STRING
  desc: pgcalname
- name: pginitiativename
  type: STRING
  desc: pg initiative name
- name: zaveraginghorizon
  type: DECIMAL
  desc: 'MD_2: Averaging Horizon [weeks]'
- name: zoutliermethod
  type: DECIMAL
  desc: 'MD_1i: Outliers [1(TOLERANCE), 2(MEAN) 3(MEDIAN)]'
- name: zminimumsafetydays
  type: DECIMAL
  desc: 'MM_1i: Minimum Safety  [Days]'
- name: zalertkeyuserhitcount
  type: DECIMAL
  desc: 'DA_1: Key user Alert HIT count'
- name: zalerthppphitcount
  type: DECIMAL
  desc: 'DA_1: HPPP Alert HIT count'
- name: zalertplanneradjhitcount
  type: DECIMAL
  desc: 'DA_1: Planner Adjustment Alert HIT count'
- name: zalertmaxhitcount
  type: DECIMAL
  desc: 'DA_1: Maximum Alert HIT count:'
- name: zalertminhitcount
  type: DECIMAL
  desc: 'DA_1: Minimum Alert HIT count'
- name: zhalertcalbaseqty
  type: DECIMAL
  desc: 'DA_1: Calculation Base KF [QTY]'
- name: zminzerolotsafetysharepct
  type: DECIMAL
  desc: 'MM_1i: Minimum Zero Lot Size %age share [%]'
- name: pgsourcesystem
  type: STRING
  desc: PG source system
- name: ziolagforecasterrorcvall
  type: DECIMAL
  desc: 'DD_1: IO Lagged Demand Forecast Error CV [%]'
- name: pgconsumerunitsize
  type: STRING
  desc: pg consumer unit size
- name: extractiontime
  type: STRING
  desc: extraction time
- name: dw_batch_num
  type: STRING
  desc: dw batch number
- name: dw_create_time
  type: TIMESTAMP
  desc: dw create time
- name: dw_last_update_time
  type: TIMESTAMP
  desc: dw last update time
- name: dw_source_sys
  type: STRING
  desc: Dw Source System
- name: dw_source_table
  type: STRING
  desc: Dw Source Table
---

# Safety Stock & CoV

## 描述

IBPI safety proposal-full

## 使用注意

- 表列数非常多 (173+)，只需关注: `finaliosafetystock` (安全库存量), `zfinalsafetydaysofsupply` (安全天数), `ziolagforecasterrorcvall` (CoV%)
- 始终使用最新 `extractiontime` 的数据
- `keyfiguredate` 是每周一，需与仿真期对齐
- ChainSight 安全库存需按 daily × material × location 粒度配置，周数据需展开为每日（每天相同值）
- `M1_ForecastError` 的 `error_std_percent` 适用于 normal 和 AO 两种 order_type（除非用户指定不同 CoV）

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 别名
- `IBPI` → Integrated Business Planning - Inventory (safety stock target setting)（字段 `None`, confidence=1.0）

### 模式
- **extraction_version** (观察次数 3): Always use MAX(extractiondate) to get latest data snapshot; extractiondate is STRING format YYYYMMDD
- **keyfigure_weekly** (观察次数 2): keyfiguredate is INT YYYYMMDD representing Monday of each week; use CROSS JOIN day_offset(0-6) to expand to daily for ChainSight M3_SafetyStock config
- **field_mapping_m3** (观察次数 3): prdid→material, locid→location, finaliosafetystock→safety_stock_qty, zfinalsafetydaysofsupply→safety days

### 约束
- **wide_table**: 173+ columns; only query needed fields to avoid timeout
