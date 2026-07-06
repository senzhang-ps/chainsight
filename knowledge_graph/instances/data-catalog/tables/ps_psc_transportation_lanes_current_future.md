---
fqn: cdl_ps_hana_prd.sl.ps_psc_transportation_lanes_current_future
description: Latest version transportation lanes by material with current and future effective dates
synced_at: '2026-04-10T10:20:39Z'
tags:
- network_topology
- scope_source
- snapshot
related_ontology:
- class: cs:TransportationLane
  role: primary
- class: cs:Location
  role: supplementary
related_config:
- Global_Network
- Global_LeadTime
- M5_PushPullModel
scope_dimensions:
- material_num
- location
- source_location
columns:
- name: material_num
  type: STRING
  desc: Material number
  maps_to: material
- name: location
  type: STRING
  desc: Receiving location (Location To)
  maps_to: receiving
- name: source_location
  type: STRING
  desc: Sending location (Location From)
  maps_to: sending
- name: effective_from
  type: TIMESTAMP
  desc: Lane validity start date
  maps_to: eff_from
- name: effective_to
  type: TIMESTAMP
  desc: Lane validity end date
  maps_to: eff_to
- name: pdt
  type: STRING
  desc: Planning Delivery Time (hhh:mm:ss format)
  maps_to: pdt
- name: gr
  type: STRING
  desc: Goods Receipt processing time (hhh:mm:ss format)
  maps_to: gr_days
- name: version
  type: DATE
  desc: CDL refresh version date (take latest)
- name: means_of_transport
  type: STRING
  desc: Transport mode
- name: qr_push
  type: STRING
  desc: Ignore QM release time flag
- name: qr_relevant
  type: STRING
  desc: QR relevance flag
- name: push_pull
  type: STRING
  desc: Lane-dependent push/pull model setting
- name: pav_data_refresh_timestamp
  type: TIMESTAMP
  desc: PAV data refresh timestamp in UTC
- name: dw_last_update_time
  type: TIMESTAMP
  desc: Data warehouse last update timestamp
- name: data_refresh_time
  type: TIMESTAMP
  desc: Data refresh timestamp
---

# Transportation Lanes

## 描述

Latest version transportation lanes by material, including both current and future effective periods. Core table for network topology — defines which materials flow between which locations with what lead times.

## 使用注意

- `effective_from` / `effective_to` 表示线路有效期，支持动态网络建模
- `pdt` = 计划配送时间, `gr` = GR 处理时间，注意格式是 `hhh:mm:ss` 字符串，需转换为天数
- `push_pull` 字段直接对应 `M5_PushPullModel` config
- `version` 字段是 CDL 刷新日期，取最新版本
- 一条 material 可以有多条 lane（多个 source_location → location 路径）

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 别名

### 模式

### 约束
