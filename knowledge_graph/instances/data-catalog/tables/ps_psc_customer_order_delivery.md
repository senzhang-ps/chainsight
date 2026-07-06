---
fqn: cdl_ps_hana_prd.sl.ps_psc_customer_order_delivery
description: Customer order and delivery data with order/confirm/delivery/GI/cut quantities across CS/SU/IT units
synced_at: '2026-04-10T10:20:39Z'
tags:
- customer_order
- delivery
- scope_source
related_ontology:
- class: cs:CustomerOrder
  role: primary
related_config:
- M1_AOConfig
scope_dimensions:
- material_num
- plant_code
- order_date
- category_en
columns:
- name: sales_order_num
  type: BIGINT
- name: cust_po_num
  type: STRING
- name: sales_doc_type_code
  type: STRING
- name: soldto_code
  type: STRING
- name: order_date
  type: DATE
- name: order_time
  type: STRING
- name: sales_doc_date
  type: DATE
- name: material_availability_date
  type: DATE
- name: actual_ship_date
  type: DATE
- name: reqstd_delivery_date
  type: DATE
- name: order_reason_code
  type: STRING
- name: sales_order_item_num
  type: BIGINT
- name: material_num
  type: STRING
- name: shipper_barcode
  type: STRING
- name: item_barcode
  type: STRING
- name: product_name_en
  type: STRING
- name: product_name_cn
  type: STRING
- name: sales_unit
  type: STRING
- name: plant_code
  type: STRING
- name: shipto_code
  type: STRING
- name: shipto_name_en
  type: STRING
- name: shipto_name_cn
  type: STRING
- name: category_en
  type: STRING
- name: delivery_num
  type: BIGINT
- name: delivery_material_num
  type: STRING
- name: sales_order_status
  type: STRING
- name: it_per_cs
  type: INT
- name: sw_per_cs
  type: INT
- name: su_factor_for_buom
  type: DECIMAL
- name: brand_en
  type: STRING
- name: product_line_en
  type: STRING
- name: banner_name
  type: STRING
- name: ecom_banner_name
  type: STRING
- name: channel_name
  type: STRING
- name: order_original_qty_in_cs
  type: DECIMAL
- name: order_original_qty_in_su
  type: DECIMAL
- name: order_original_qty_in_it
  type: DECIMAL
- name: order_confirm_qty_in_cs
  type: DECIMAL
- name: order_confirm_qty_in_su
  type: DECIMAL
- name: order_confirm_qty_in_it
  type: DECIMAL
- name: delivery_qty_in_cs
  type: DECIMAL
- name: delivery_qty_in_su
  type: DECIMAL
- name: delivery_qty_in_it
  type: DECIMAL
- name: open_qty_in_cs
  type: DECIMAL
- name: open_qty_in_su
  type: DECIMAL
- name: open_qty_in_it
  type: DECIMAL
- name: gi_qty_in_cs
  type: DECIMAL
- name: gi_qty_in_su
  type: DECIMAL
- name: gi_qty_in_it
  type: DECIMAL
- name: cut_qty_in_cs
  type: DECIMAL
- name: cut_qty_in_su
  type: DECIMAL
- name: cut_qty_in_it
  type: DECIMAL
- name: cut_reason_code
  type: STRING
- name: cut_reason_desc_cn
  type: STRING
- name: division_name
  type: STRING
- name: is_order_sum_flag
  type: STRING
- name: apo_final_sourcing_plant_code
  type: STRING
- name: dw_last_update_time
  type: TIMESTAMP
- name: data_refresh_time
  type: TIMESTAMP
---

# Customer Order & Delivery

## 描述

Customer order and delivery transactional data. Contains the full order lifecycle: original order → confirmed → delivered → GI (goods issue) → cut quantities. Key source for CFR (Customer Fill Rate) calculation and AO (Advance Order) configuration analysis.

## 使用注意

- **⚠️ `is_order_sum_flag`**: 计算 CFR denominator (`confirm_qty`) 时**必须**加 `is_order_sum_flag = 'Y'` 过滤。同一 SO item 因 material conversion 会拆成多行 (`ordered_material` ≠ `delivery_material`)，不加 filter 会重复计算 `confirm_qty`。`delivery_qty` 不需要过滤（每行是独立物理发货）
- **⚠️ `sales_order_status='99'`**: 客户取消订单，confirm 有量但 delivery=0，**CFR 计算必须排除**，否则会严重拉低 CFR（实测偏差可达 11pp）。正确字段名是 `sales_order_status`（不是 `order_status`）
- 数量字段有 CS/SU/IT 三种单位，需确保与 ChainSight 配置的单位一致
- **AO% 和 `advance_days` 计算逻辑**:
  1. 按 material × location 粒度，在用户指定的时间范围内计算（建议至少 3 个月数据）
  2. `advance_days = sales_doc_date - order_date`（实测: `sales_doc_date > order_date`），等于 0 则为 normal order，不配置到 `M1_AOConfig`
  3. 按 `material_num × plant_code × advance_days` 汇总 `delivery_qty_in_cs`
  4. `ao_percent = sum(delivery_qty where advance_days > 0) / sum(total delivery_qty)`

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 模式
- **ao_advance_days** (观察次数 3): advance_days = DATEDIFF(sales_doc_date, order_date); sales_doc_date > order_date → advance_days > 0 = AO; = 0 = normal order
- **ao_bucket_definition** (观察次数 2): AO5: advance 1-5天, AO3: 6-11天, AO1: >=12天, Normal: <=0天

### 约束
- **qty_unit_variants**: CS/SU/IT 三种单位，必须确认与 ChainSight 配置单位一致
- **category_en_short_name** (观察次数 2): sl 层所有表的 category_en 使用缩写形式（如 'Skin' 而非 'Skin Care'），适用于 ps_psc_sku_master、ps_psc_customer_order_delivery 等
- **cfr_correct_formula** (观察次数 3): CFR = SUM(delivery_qty_in_su) / SUM(confirm_qty_in_su WHERE is_order_sum_flag='Y' AND sales_order_status!='99')。flag=N 的 conversion split 行 delivery 极小，对分子无实际影响
