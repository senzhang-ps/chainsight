---
fqn: cdl_ps_hana_prd.dwd.vw_sc_inv_material_storage_location_batch_fact
description: Current inventory by material, plant, storage location, and batch with stock type breakdown
synced_at: '2026-04-10T10:20:39Z'
tags:
- inventory
- snapshot
- scope_source
related_ontology:
- class: cs:InitialInventory
  role: primary
- class: cs:Batch
  role: supplementary
related_config:
- M1_InitialInventory
scope_dimensions:
- material_num
- plant_code
- storage_location_code
columns:
- name: material_storage_location_batch_skey
  type: BIGINT
- name: client_sys_code
  type: STRING
- name: material_num
  type: STRING
- name: plant_code
  type: STRING
- name: storage_location_code
  type: STRING
- name: batch_num
  type: STRING
- name: is_delete_flag
  type: STRING
- name: create_date
  type: STRING
- name: gc_create_date
  type: STRING
- name: create_user_name
  type: STRING
- name: last_change_date
  type: STRING
- name: gc_last_change_date
  type: STRING
- name: update_user_name
  type: STRING
- name: current_period_fiscal_year
  type: STRING
- name: current_period
  type: STRING
- name: physical_inventory_block_ind
  type: STRING
- name: stock_qty
  type: DECIMAL
- name: transfer_stock
  type: DECIMAL
- name: quality_insp_stock
  type: DECIMAL
- name: all_restrict_batch_total_stock
  type: DECIMAL
- name: block_stock
  type: DECIMAL
- name: block_stock_return
  type: DECIMAL
- name: valuate_unrestricted_use_stock_in_previous_period
  type: INT
- name: transfer_previous_period_stock
  type: INT
- name: quality_insp_previous_period_stock
  type: INT
- name: previous_period_restrict_use_stock
  type: INT
- name: block_stock_previous_period
  type: INT
- name: block_stock_return_previous_period
  type: INT
- name: whse_stock_current_year_physical_inventory_ind
  type: STRING
- name: stock_in_quality_insp_current_year_physical_inventory_ind
  type: STRING
- name: restrict_use_stock_physical_inventory_ind
  type: STRING
- name: block_stock_physical_inventory_ind
  type: STRING
- name: stock_prior_year_physical_inventory_ind
  type: STRING
- name: stock_in_quality_insp_prior_period_physical_inventory_ind
  type: STRING
- name: restrict_use_stock_prior_period_physical_inventory_ind
  type: STRING
- name: block_stock_in_prior_period_physical_inventory_ind
  type: STRING
- name: material_original_country
  type: STRING
- name: unrestricted_use_stock_last_post_cnt_date
  type: STRING
- name: gc_unrestricted_use_stock_last_post_cnt_date
  type: STRING
- name: current_physical_inventory_ind_fiscal_year
  type: STRING
- name: storage_location_exist_ind
  type: STRING
- name: stock_segment
  type: STRING
- name: sys_change_operation
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
- name: bd_mod_utc_time_stamp
  type: TIMESTAMP
- name: stg_etl_date_time
  type: TIMESTAMP
- name: simp_chng_utc_time_stamp
  type: TIMESTAMP
---

# Current Inventory (Batch Level)

## 描述

Current inventory snapshot at material × plant × storage location × batch granularity. Contains unrestricted, transfer, quality inspection, and blocked stock quantities. Primary source for ChainSight M1 initial inventory configuration.

## 使用注意

- 包含 unrestricted (`stock_qty`)、transfer、quality inspection、blocked 四种库存类型
- ChainSight 初始库存可能不直接使用最新库存数据，需与业务确认
- 数据粒度: material × plant × storage_location × batch

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 模式
- **stock_type_filter** (观察次数 2): stock_qty = unrestricted inventory; 还有 transfer/quality/blocked 四种类型，ChainSight 通常只用 unrestricted

### 约束
- **material_num_18digit**: dwd层 material_num 18位前导零(000000000083927819)，JOIN sl层时必须 LPAD(sl.material_num, 18, '0')
- **snapshot_table**: 此表为当下快照，不需要时间范围筛选
