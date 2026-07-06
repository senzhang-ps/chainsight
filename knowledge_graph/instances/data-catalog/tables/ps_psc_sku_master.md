---
fqn: cdl_ps_hana_prd.sl.ps_psc_sku_master
description: Product/SKU master data with full product hierarchy, segmentation, and physical attributes
synced_at: '2026-04-11T08:00:00Z'
tags:
- material_master
- product_hierarchy
- scope_filter
related_ontology:
- class: cs:Material
  role: primary
related_config:
- M6_MaterialMD
scope_dimensions:
- sector
- sub_sector
- category_en
- brand_en
- spaced_sku_segmentation
- tier
- form
- segmentation_1st_level
- segmentation_2nd_level
columns:
- name: material_num
  type: STRING
- name: product_name_en
  type: STRING
- name: product_name_cn
  type: STRING
- name: category_en
  type: STRING
- name: sfu_code
  type: STRING
- name: apo_final_sourcing_plant_code
  type: STRING
- name: sell_barcode
  type: STRING
- name: shipper_barcode
  type: STRING
- name: new_product_name_en
  type: STRING
- name: buom
  type: STRING
- name: su_factor_for_buom
  type: DECIMAL
- name: form
  type: STRING
- name: brand_en
  type: STRING
- name: product_line_en
  type: STRING
- name: lineup
  type: STRING
- name: variant_en
  type: STRING
- name: type
  type: STRING
- name: promotion_pack_type
  type: STRING
- name: segmentation_1st_level
  type: STRING
- name: segmentation_2nd_level
  type: STRING
- name: spo
  type: STRING
- name: ni_name_manual_upload
  type: STRING
- name: iopt_initiative_name
  type: STRING
- name: dsbp_initiative_name
  type: STRING
- name: selling_mkt
  type: STRING
- name: tier
  type: STRING
- name: size_segment_cn
  type: STRING
- name: size
  type: STRING
- name: spaced_sku_segmentation
  type: STRING
- name: level_1
  type: STRING
- name: level_2
  type: STRING
- name: level_3
  type: STRING
- name: level_4
  type: STRING
- name: level_5
  type: STRING
- name: level_6
  type: STRING
- name: status
  type: STRING
- name: tdc_val
  type: STRING
- name: tdc_val_desc
  type: STRING
- name: sub_group
  type: STRING
- name: attribute_1
  type: STRING
- name: attribute_2
  type: STRING
- name: attribute_3
  type: STRING
- name: attribute_4
  type: STRING
- name: attribute_5
  type: STRING
- name: attribute_6
  type: STRING
- name: attribute_7
  type: STRING
- name: promotion_pack_detail
  type: STRING
- name: product_nature_1
  type: STRING
- name: life_cycle_stage
  type: STRING
- name: life_cycle_stage_status
  type: STRING
- name: item_status
  type: STRING
- name: net_weight_buom
  type: DECIMAL
- name: gross_weight_buom
  type: DECIMAL
- name: ni_flag
  type: STRING
- name: sos_date
  type: DATE
- name: internal_project_name
  type: STRING
- name: conversion_type
  type: STRING
- name: app_reason
  type: STRING
- name: old_item_code
  type: STRING
- name: old_barcode
  type: STRING
- name: local_import
  type: STRING
- name: volume_cs
  type: DECIMAL
- name: volume_unit
  type: STRING
- name: gross_weight_cs
  type: DECIMAL
- name: weight_unit
  type: STRING
- name: cs_per_pallet
  type: DECIMAL
- name: volume_per_pallet
  type: DECIMAL
- name: gross_weight_per_pallet
  type: DECIMAL
- name: launch_area
  type: STRING
- name: it_per_cs
  type: INT
- name: sw_per_cs
  type: INT
- name: demand_plan_level_3_name_cn
  type: STRING
- name: sold_in_cn
  type: STRING
- name: sold_in_hk
  type: STRING
- name: sold_in_x_border
  type: STRING
- name: quality_guaranteed_day
  type: STRING
- name: csu_ind
  type: STRING
- name: product_length
  type: STRING
- name: quality_guaranteed_month
  type: STRING
- name: sub_brand_en
  type: STRING
- name: sub_brand_cn
  type: STRING
- name: product_source
  type: STRING
- name: brand_code
  type: STRING
- name: category_cn
  type: STRING
- name: brand_cn
  type: STRING
- name: full_brand_en
  type: STRING
- name: brand_product_form_cn
  type: STRING
- name: variant_cn
  type: STRING
- name: full_variant_en
  type: STRING
- name: product_form_en
  type: STRING
- name: brand_product_form_en
  type: STRING
- name: product_form_cn
  type: STRING
- name: inner_barcode
  type: STRING
- name: item_barcode
  type: STRING
- name: item_nature
  type: STRING
- name: sector
  type: STRING
- name: brand_element
  type: STRING
- name: brand_form
  type: STRING
- name: category_code
  type: STRING
- name: full_category_en
  type: STRING
- name: category
  type: STRING
- name: sub_sector
  type: STRING
- name: brand
  type: STRING
- name: brand_segment
  type: STRING
- name: sub_brand
  type: STRING
- name: full_variant_cn
  type: STRING
- name: components
  type: STRING
- name: height_it_in_cm
  type: DECIMAL
- name: width_it_in_cm
  type: DECIMAL
- name: length_it_in_cm
  type: DECIMAL
- name: case_cnt
  type: INT
- name: size_segment
  type: STRING
- name: length_cs_in_cm
  type: DECIMAL
- name: height_cs_in_cm
  type: DECIMAL
- name: width_cs_in_cm
  type: DECIMAL
- name: bu_attr
  type: STRING
- name: inactive_date
  type: DATE
- name: last_shipment_date
  type: DATE
- name: import_item_type_name
  type: STRING
- name: dim_unit
  type: STRING
- name: local_hierarchy_flag
  type: STRING
- name: ni_sos_date
  type: DATE
- name: ni_project_name
  type: STRING
- name: sap_size_main_product
  type: STRING
- name: sap_size_combined
  type: STRING
- name: price_tier
  type: STRING
- name: sap_size_total
  type: STRING
- name: new_form
  type: STRING
- name: cn_size_total
  type: STRING
- name: spp_normal_dm_price
  type: DECIMAL
- name: pack_cnt
  type: INT
- name: item_bundle_pack
  type: INT
- name: spp_top_dm_price
  type: DECIMAL
- name: srp
  type: DECIMAL
- name: launch_area_channel_name
  type: STRING
- name: launch_area_market_name
  type: STRING
- name: launch_area_banner_name
  type: STRING
- name: full_category_cn
  type: STRING
- name: full_brand_cn
  type: STRING
- name: official_sos_date
  type: DATE
- name: price_effective_date
  type: DATE
- name: manufacture_city_name
  type: STRING
- name: quality_guaranteed
  type: STRING
- name: quality_guaranteed_date
  type: INT
- name: sale_org_code
  type: STRING
- name: case_200_for_sale_unit_include_tax
  type: DECIMAL
- name: case_200_include_tax
  type: DECIMAL
- name: case_200_exclude_tax
  type: DECIMAL
- name: case_800_for_sale_unit_include_tax
  type: DECIMAL
- name: case_800_for_sale_unit_exclude_tax
  type: DECIMAL
- name: case_800_include_tax
  type: DECIMAL
- name: case_800_exclude_tax
  type: DECIMAL
- name: case_2000_include_tax
  type: DECIMAL
- name: case_2000_exclude_tax
  type: DECIMAL
- name: case_3500_include_tax
  type: DECIMAL
- name: case_3500_exclude_tax
  type: DECIMAL
- name: case_200_for_sale_unit_base_price_include_tax
  type: DECIMAL
- name: case_800_for_sale_unit_base_price_include_tax
  type: DECIMAL
- name: case_2000_for_sale_unit_base_price_include_tax
  type: DECIMAL
- name: case_3500_for_sale_unit_base_price_include_tax
  type: DECIMAL
- name: sale_unit_to_cust_type
  type: STRING
- name: sale_unit_to_consumer_type
  type: STRING
- name: sellable_effective_to_date
  type: DATE
- name: is_sioc_flag
  type: STRING
- name: lsr_end_date
  type: DATE
- name: lsr_io_num
  type: STRING
- name: lsr_start_date
  type: DATE
- name: tpr_start_date
  type: DATE
- name: tpr_end_date
  type: DATE
- name: tpr
  type: DECIMAL
- name: tpr_io_num
  type: STRING
- name: lsr
  type: DECIMAL
- name: dw_last_update_time
  type: TIMESTAMP
- name: data_refresh_time
  type: TIMESTAMP
---

# SKU Master

## 描述

Product/SKU master data containing full product hierarchy (sector → sub_sector → category → brand → variant), segmentation levels, physical attributes (weight, volume, barcode), and lifecycle status. The primary table for scope resolution — filter by category/brand/tier to get material lists.

## 使用注意

- 产品主数据，含产品层级、细分、物理属性
- 用于 scope 解析（按 sector / category / brand / tier 过滤 material）
- `material_num` 使用前需 LPAD 到 18 位做 JOIN

## 已学习知识

> 由 dreaming cycle 从 episodic memory 自动沉淀；人工可直接编辑本节。

### 别名
- `HC` → Hair Care（字段 `sub_sector`, confidence=1.0）
- `OC` → Oral Care（字段 `sub_sector`, confidence=0.9）

### 模式
- **scope_filter_entry** (观察次数 4): Hair Care scope 通常先查 sub_sector → 再用 tlane 展开 location

### 约束
- **material_num_8digit**: sl层 material_num 为 8 位纯数字(83927819)，与 dwd 层 18 位前导零格式不同，JOIN 时必须 LPAD 或 CAST
