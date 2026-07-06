---
fqn: cdl_ps_hana_prd.sl.ps_psc_factory_calendar
description: Factory calendar view identifying workdays and non-workdays per factory
synced_at: '2026-04-10T10:20:39Z'
tags:
- calendar
- reference
related_ontology:
- class: cs:Calendar
  role: primary
related_config:
- M1_OrderCalendar
scope_dimensions:
- factory_calendar
columns:
- name: factory_calendar
  type: STRING
  desc: Factory calendar identifier
- name: calendar_date
  type: DATE
  desc: Calendar date
  maps_to: calendar_date
- name: snapshot_time
  type: TIMESTAMP
  desc: Data snapshot timestamp
- name: work_day
  type: STRING
  desc: Whether the date is a workday (Y/N)
  maps_to: is_workday
- name: work_day_id
  type: INT
  desc: Sequential workday counter
- name: next_work_day_id
  type: INT
  desc: Sequential ID of the next workday
- name: dw_last_update_time
  type: TIMESTAMP
  desc: Data warehouse last update timestamp
- name: data_refresh_time
  type: TIMESTAMP
  desc: Data refresh timestamp
---

# Factory Calendar

## 描述

Factory calendar view identifying whether each calendar date is a workday for a given factory. Used to generate M1 order calendar and to determine working day sequences for production and logistics scheduling.

## 使用注意

- 当前 ChainSight `M1_OrderCalendar` 配置为从仿真起始日（周一）开始的完整仿真期
- `work_day` 字段是字符串类型 (Y/N)，不是 boolean

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 别名

### 模式

### 约束
