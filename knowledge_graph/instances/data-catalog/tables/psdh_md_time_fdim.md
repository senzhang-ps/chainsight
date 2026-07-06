---
fqn: cdl_ps_hana_prd.ods.psdh_md_time_fdim
description: Time dimension table with day/week/month/quarter/fiscal year hierarchies and technical period mappings
synced_at: '2026-04-10T10:20:39Z'
tags:
- calendar
- reference
- time_dimension
related_ontology:
- class: cs:Calendar
  role: primary
- class: cs:DemandForecast
  role: reference
related_config:
- M1_DemandForecast
scope_dimensions:
- day_date
- wk_start_date
- mth_start_date
columns:
- name: skey
  type: BIGINT
- name: day_num
  type: INT
- name: day_abbr_name
  type: STRING
- name: day_name
  type: STRING
- name: day_date
  type: DATE
- name: day_of_wk_name
  type: STRING
- name: tp_num
  type: INT
- name: tp_abbr_name
  type: STRING
- name: tp_name
  type: STRING
- name: tp_start_date
  type: DATE
- name: tp_end_date
  type: DATE
- name: wk_num
  type: INT
- name: wk_abbr_name
  type: STRING
- name: wk_name
  type: STRING
- name: wk_label
  type: STRING
- name: wk_start_date
  type: DATE
- name: wk_end_date
  type: DATE
- name: mth_num
  type: INT
- name: mth_abbr_name
  type: STRING
- name: mth_name
  type: STRING
- name: mth_start_date
  type: DATE
- name: mth_end_date
  type: DATE
- name: mth_of_yr_name
  type: STRING
- name: mth_day_in_mth_num
  type: INT
- name: mth_fisc_perd_num
  type: INT
- name: mth_fisc_perd_name
  type: STRING
- name: fisc_yr_perd
  type: INT
- name: qtr_num
  type: INT
- name: qtr_abbr_name
  type: STRING
- name: qtr_name
  type: STRING
- name: qtr_start_date
  type: DATE
- name: qtr_end_date
  type: DATE
- name: qtr_fisc_yr_num
  type: INT
- name: qtr_fisc_yr_name
  type: STRING
- name: hy_num
  type: INT
- name: hy_fisc_yr_num
  type: INT
- name: cal_yr_num
  type: INT
- name: cal_yr_abbr_name
  type: STRING
- name: cal_yr_name
  type: STRING
- name: cal_yr_start_date
  type: DATE
- name: cal_yr_end_date
  type: DATE
- name: fisc_yr_num
  type: INT
- name: fisc_yr_abbr_name
  type: STRING
- name: fisc_yr_name
  type: STRING
- name: fisc_yr_start_date
  type: DATE
- name: fisc_yr_end_date
  type: DATE
- name: sop_flag
  type: STRING
- name: time_skid
  type: BIGINT
- name: process_run_key
  type: BIGINT
- name: secure_group_key
  type: BIGINT
- name: data_provider_code
  type: STRING
- name: load_date
  type: STRING
- name: dw_batch_num
  type: STRING
- name: dw_create_time
  type: TIMESTAMP
- name: dw_last_update_time
  type: TIMESTAMP
- name: dw_source_sys
  type: STRING
- name: dw_source_table
  type: STRING
---

# Time Dimension

## 描述

Master time dimension table with full calendar hierarchies — day, technical period (tp), ISO week, month, quarter, half-year, calendar year, and fiscal year. The primary table for date-based conversions throughout ChainSight.

## 使用注意

- 主日历表，包含 technical period、周、月、季度、财年等维度
- `tp_start_date` / `tp_end_date` 用于 technical period → calendar date 映射
- `wk_start_date` 是关键字段，用于将 `tp_start_date` 转为 ChainSight 周编号

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 别名

### 模式

### 约束
