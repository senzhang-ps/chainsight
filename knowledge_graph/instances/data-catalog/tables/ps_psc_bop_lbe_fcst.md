---
fqn: cdl_ps_hana_prd.sl.ps_psc_bop_lbe_fcst
description: BOP/LBE demand forecast weekly by material and site
synced_at: '2026-04-10T10:20:39Z'
tags:
- demand_forecast
- scope_source
- transactional
related_ontology:
- class: cs:DemandForecast
  role: primary
related_config:
- M1_DemandForecast
scope_dimensions:
- material_num
- site_id
- frcst_vers_date
columns:
- name: frcst_vers_date
  type: INT
  desc: Forecast version date (YYYYMMDD). Each Monday a new version is published.
  maps_to: null
- name: material_num
  type: STRING
  desc: Material number (18-digit LPAD with leading zeros)
  maps_to: material
- name: site_id
  type: STRING
  desc: Plant/DC location code
  maps_to: location
- name: fcst_qty_in_su
  type: DECIMAL
  desc: Forecast quantity in Statistical Units (SU)
  maps_to: quantity
- name: fcst_qty_in_cs
  type: DECIMAL
  desc: Forecast quantity in Case Count (converted from SU via master data)
- name: fcst_type
  type: STRING
  desc: "Forecast type: 'BOP' (Business Operation Plan) or 'LBE' (Latest Best Estimate)"
- name: tp_start_date
  type: DATE
  desc: Time period start date (needs calendar table to convert to week number)
  maps_to: week
- name: tp_week_num
  type: INT
  desc: Time period week number within forecast horizon
- name: month_num
  type: INT
  desc: Month number
- name: year_num
  type: INT
  desc: Year number
- name: dw_last_update_time
  type: TIMESTAMP
  desc: Data warehouse last update timestamp
- name: data_refresh_time
  type: TIMESTAMP
  desc: Data refresh timestamp
---

# Demand Forecast

## 描述

BOP/LBE demand forecast weekly by material and site

## 使用注意

- **必须询问用户**: BOP 还是 LBE？通过 `fcst_type` 字段过滤（'BOP' 或 'LBE'）
- **版本选择**: 每周一出一版 forecast (`frcst_vers_date`)。每月只有一版是 BOP，其余是 LBE。默认取最新版本，但用户可指定特定版本日期
- **默认时间范围**: 13 周（如无特殊要求）
- **过滤负数**: forecast 中可能存在负值（调整项），默认过滤掉 `fcst_qty_in_su < 0` 的行
- `tp_start_date` 需通过日历表（`psdh_md_time_fdim`）转换为 `wk_start_date`
- `week` 在 ChainSight 中从 1 开始顺序编号，不是日历周数。例如 `tp_start_date` 20260501 → 先转为该周一 20260427 → 在 config 中为 week 1
- `fcst_qty_in_su` 对应 ChainSight `quantity` 字段
- `fcst_qty_in_cs` 是用 Case Count 主数据换算的，不要与 APO 实时预测比较
- 对于采用 IDF 的 BU，实时预测会被 IDF 修改

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 别名
- `LBE` → Latest Best Estimate（字段 `fcst_type`, confidence=1.0）
- `BOP` → Business Operation Plan（字段 `fcst_type`, confidence=1.0）

### 模式
- **version_selection** (观察次数 5): MAX(frcst_vers_date) selects latest forecast version; each Monday a new LBE version is published
- **negative_filter** (观察次数 3): Filter fcst_qty_in_su < 0 rows (adjustment items); default behavior unless user requests otherwise
- **bop_version_date** (观察次数 2): March BOP=20260302, April BOP=20260330; BOP is published once per month

### 约束
- **must_ask_fcst_type**: Always ask user: BOP or LBE? Do not assume
