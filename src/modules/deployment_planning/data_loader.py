# -*- coding: utf-8 -*-
"""
数据加载模块

提供从各种数据源（文件、Orchestrator、Module1等）加载数据的功能。
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import pandas as pd

from .normalizer import normalize_identifiers
from .constants import (
    DATE_FIELDS_MAP,
    REQUIRED_SHEETS,
    SDL_REQUIRED_COLUMNS
)


def load_module1_daily_shipment(
    module1_output_dir: str,
    current_date: pd.Timestamp
) -> pd.DataFrame:
    """
    加载 Module1 当日发货数据（ShipmentLog）。

    Args:
        module1_output_dir: Module1输出目录
        current_date: 当前日期

    Returns:
        pd.DataFrame: 包含date, material, location, quantity的发货数据
    """
    required_cols = ['date', 'material', 'location', 'quantity']

    try:
        date_str = current_date.strftime('%Y%m%d')
        module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"

        if not os.path.exists(module1_file):
            print(f"⚠️  Module1输出文件不存在: {module1_file}")
            return pd.DataFrame(columns=required_cols)

        xl = pd.ExcelFile(module1_file)
        if 'ShipmentLog' not in xl.sheet_names:
            print(f"⚠️  Module1输出文件中无ShipmentLog表: {module1_file}")
            return pd.DataFrame(columns=required_cols)

        shipment_df = xl.parse('ShipmentLog')
        if not all(col in shipment_df.columns for col in required_cols):
            print(f"⚠️  Module1输出文件缺少必要字段: {module1_file}")
            return pd.DataFrame(columns=required_cols)

        result_df = shipment_df[required_cols].copy()
        return normalize_identifiers(result_df)

    except Exception as e:
        print(f"⚠️  加载Module1发货数据失败: {e}")
        return pd.DataFrame(columns=required_cols)


def load_module1_daily_orders(
    module1_output_dir: str,
    current_date: pd.Timestamp
) -> pd.DataFrame:
    """
    加载 Module1 当日订单池（OrderLog）。

    选择requirement_date >= current_date的订单。

    Args:
        module1_output_dir: Module1输出目录
        current_date: 当前日期

    Returns:
        pd.DataFrame: 订单数据，包含date, material, location等字段
    """
    cols = [
        'date', 'material', 'location',
        'demand_type', 'quantity', 'simulation_date'
    ]

    try:
        date_str = current_date.strftime('%Y%m%d')
        module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"

        if not os.path.exists(module1_file):
            print(f"⚠️  Module1输出文件不存在: {module1_file}")
            return pd.DataFrame(columns=cols)

        xl = pd.ExcelFile(module1_file)
        if 'OrderLog' not in xl.sheet_names:
            print(f"⚠️  Module1输出文件中无OrderLog表: {module1_file}")
            return pd.DataFrame(columns=cols)

        df = xl.parse('OrderLog')

        # 日期转换
        for c in ['date', 'simulation_date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

        # 筛选有效订单
        if 'date' in df.columns:
            df = df[df['date'] >= current_date]

        # 确保列存在
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NaT if c in ['date', 'simulation_date'] else None

        result_df = df[cols].copy()
        return normalize_identifiers(result_df)

    except Exception as e:
        print(f"⚠️  加载Module1订单数据失败: {e}")
        return pd.DataFrame(columns=cols)


def load_orchestrator_delivery_gr(
    orchestrator: object,
    current_date: pd.Timestamp
) -> pd.DataFrame:
    """
    从 Orchestrator 加载当日收货（GR）视图。

    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期

    Returns:
        pd.DataFrame: 收货数据，包含date, material, receiving, quantity
    """
    required_cols = ['date', 'material', 'receiving', 'quantity']
    col_mapping = {
        'location': 'receiving',
        'gr_qty': 'quantity',
        'received_qty': 'quantity'
    }

    try:
        date_str = current_date.strftime('%Y-%m-%d')
        delivery_gr_view = orchestrator.get_delivery_gr_view(date_str)

        if not isinstance(delivery_gr_view, pd.DataFrame):
            print("⚠️Orchestrator返回空的delivery_gr_view", flush=True)
            return pd.DataFrame(columns=required_cols)

        if delivery_gr_view.empty:
            return pd.DataFrame(columns=required_cols)

        renamed_df = delivery_gr_view.copy()
        for old_col, new_col in col_mapping.items():
            if old_col in renamed_df.columns:
                renamed_df = renamed_df.rename(columns={old_col: new_col})

        missing_cols = [
            col for col in required_cols
            if col not in renamed_df.columns
        ]
        if missing_cols:
            print(
                f"⚠️Orchestrator delivery_gr_view缺少字段: {missing_cols}",
                flush=True
            )
            return pd.DataFrame(columns=required_cols)

        result_df = renamed_df[required_cols].copy()
        return normalize_identifiers(result_df)

    except Exception as e:
        print(f"⚠️从Orchestrator加载收货数据失败: {e}", flush=True)
        return pd.DataFrame(columns=required_cols)


def load_orchestrator_open_deployment(
    orchestrator: object,
    current_date: pd.Timestamp
) -> pd.DataFrame:
    """
    从 Orchestrator 加载开放调拨（Open Deployment）视图。

    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期

    Returns:
        pd.DataFrame: 开放调拨数据
    """
    required_cols = ['material', 'sending', 'receiving', 'quantity']
    col_mapping = {
        'location': 'sending',
        'deployed_qty': 'quantity',
        'planned_qty': 'quantity'
    }

    try:
        date_str = current_date.strftime('%Y-%m-%d')
        open_deployment_view = orchestrator.get_open_deployment_view(date_str)

        if not isinstance(open_deployment_view, pd.DataFrame):
            print("⚠️Orchestrator返回空的open_deployment_view", flush=True)
            return pd.DataFrame(columns=required_cols)

        if open_deployment_view.empty:
            return pd.DataFrame(columns=required_cols)

        renamed_df = open_deployment_view.copy()
        for old_col, new_col in col_mapping.items():
            if old_col in renamed_df.columns:
                renamed_df = renamed_df.rename(columns={old_col: new_col})

        missing_cols = [
            col for col in required_cols
            if col not in renamed_df.columns
        ]
        if missing_cols:
            print(
                f"⚠️Orchestrator open_deployment_view缺少字段: {missing_cols}",
                flush=True
            )
            return pd.DataFrame(columns=required_cols)

        result_df = renamed_df[required_cols].copy()
        return normalize_identifiers(result_df)

    except Exception as e:
        print(f"⚠️从Orchestrator加载开放调拨数据失败: {e}", flush=True)
        return pd.DataFrame(columns=required_cols)


def _load_static_config(config_dict: dict, config: dict) -> None:
    """
    从配置字典加载静态配置表。

    Args:
        config_dict: 原始配置字典
        config: 目标配置字典（会被修改）
    """
    static_tables = {
        'SafetyStock': 'M3_SafetyStock',
        'Network': 'Global_Network',
        'LeadTime': 'Global_LeadTime',
        'DemandPriority': 'Global_DemandPriority',
        'PushPullModel': 'M5_PushPullModel',
        'DeployConfig': 'M5_DeployConfig',
    }

    for sheet_name, config_key in static_tables.items():
        config[sheet_name] = config_dict.get(config_key, pd.DataFrame())

    # 应用字符串格式化
    for sheet_name in static_tables.keys():
        if not config[sheet_name].empty:
            config[sheet_name] = normalize_identifiers(config[sheet_name])


def _load_module1_data_from_memory(
    module1_result: dict,
    config: dict
) -> None:
    """
    从内存加载Module1数据（数据库模式）。

    Args:
        module1_result: Module1运行结果
        config: 目标配置字典（会被修改）
    """
    orders_df = module1_result.get('orders_df', pd.DataFrame())
    supply_demand_df = module1_result.get('supply_demand_df', pd.DataFrame())
    shipment_df = module1_result.get('shipment_df', pd.DataFrame())

    config['OrderLog'] = (
        normalize_identifiers(orders_df)
        if not orders_df.empty else pd.DataFrame()
    )

    if not supply_demand_df.empty:
        config['SupplyDemandLog'] = normalize_identifiers(supply_demand_df)

    config['TodayShipment'] = (
        normalize_identifiers(shipment_df)
        if not shipment_df.empty else pd.DataFrame()
    )


def _load_module1_data_from_file(
    module1_output_dir: str,
    current_date: pd.Timestamp,
    config: dict
) -> None:
    """
    从文件并行加载Module1数据。

    Args:
        module1_output_dir: Module1输出目录
        current_date: 当前日期
        config: 目标配置字典（会被修改）
    """
    date_str = current_date.strftime('%Y%m%d')
    module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"

    def _load_orderlog():
        try:
            return load_module1_daily_orders(module1_output_dir, current_date)
        except Exception as e:
            print(f"  ⚠️  加载OrderLog失败: {e}")
            return pd.DataFrame()

    def _load_supplydemand():
        try:
            if os.path.exists(module1_file):
                xl = pd.ExcelFile(module1_file)
                if 'SupplyDemandLog' in xl.sheet_names:
                    df = xl.parse('SupplyDemandLog')
                    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            return pd.DataFrame()
        except Exception as e:
            print(f"  ⚠️  加载SupplyDemandLog失败: {e}")
            return pd.DataFrame()

    def _load_todayshipment():
        try:
            return load_module1_daily_shipment(module1_output_dir, current_date)
        except Exception as e:
            print(f"  ⚠️  加载TodayShipment失败: {e}")
            return pd.DataFrame()

    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_order = ex.submit(_load_orderlog)
            f_sdl = ex.submit(_load_supplydemand)
            f_ship = ex.submit(_load_todayshipment)

            order_df = f_order.result()
            config['OrderLog'] = (
                order_df if isinstance(order_df, pd.DataFrame)
                else pd.DataFrame()
            )

            sdl_df = f_sdl.result()
            if isinstance(sdl_df, pd.DataFrame) and not sdl_df.empty:
                config['SupplyDemandLog'] = sdl_df

            ship_df = f_ship.result()
            config['TodayShipment'] = (
                ship_df if isinstance(ship_df, pd.DataFrame)
                else pd.DataFrame()
            )
    except Exception as e:
        print(f"⚠️并行加载 Module1 数据失败，回退串行: {e}")
        config['OrderLog'] = load_module1_daily_orders(
            module1_output_dir, current_date
        )
        config['TodayShipment'] = load_module1_daily_shipment(
            module1_output_dir, current_date
        )


def _load_production_from_orchestrator(
    orchestrator: object,
    current_date: pd.Timestamp,
    config: dict
) -> None:
    """
    从Orchestrator加载生产计划数据。

    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期
        config: 目标配置字典（会被修改）
    """
    date_str = current_date.strftime('%Y-%m-%d')
    try:
        prod_gr = orchestrator.get_production_gr_view(date_str)
        if isinstance(prod_gr, pd.DataFrame) and not prod_gr.empty:
            prod_gr = prod_gr.rename(columns={'date': 'available_date'})
            prod_gr = prod_gr[
                ['material', 'location', 'available_date', 'quantity']
            ]
            if 'available_date' in prod_gr.columns:
                prod_gr['available_date'] = pd.to_datetime(
                    prod_gr['available_date']
                )
            for col in ['quantity']:
                if col in prod_gr.columns:
                    prod_gr[col] = pd.to_numeric(
                        prod_gr[col], errors='coerce'
                    ).fillna(0)
            config['ProductionPlan'] = prod_gr
        else:
            print("⚠️Orchestrator当日无历史生产GR数据", flush=True)
    except Exception as e:
        print(f"⚠️从 Orchestrator 加载生产计划失败: {e}", flush=True)


def _load_production_from_module4(
    module4_output_path: str,
    config: dict
) -> None:
    """
    从Module4文件加载生产计划数据。

    Args:
        module4_output_path: Module4输出文件路径
        config: 目标配置字典（会被修改）
    """
    if not module4_output_path or not os.path.exists(module4_output_path):
        return

    try:
        xl = pd.ExcelFile(module4_output_path)
        if 'ProductionPlan' not in xl.sheet_names:
            return

        m4_production = xl.parse('ProductionPlan')
        if m4_production.empty:
            return

        if 'available_date' not in m4_production.columns:
            if 'date' in m4_production.columns:
                m4_production = m4_production.rename(
                    columns={'date': 'available_date'}
                )

        if 'available_date' in m4_production.columns:
            m4_production['available_date'] = pd.to_datetime(
                m4_production['available_date'], errors='coerce'
            )

        for col in ['produced_qty', 'uncon_planned_qty', 'planned_qty', 'quantity']:
            if col in m4_production.columns:
                m4_production[col] = pd.to_numeric(
                    m4_production[col], errors='coerce'
                ).fillna(0)

        config['ProductionPlan'] = m4_production
    except Exception as e:
        print(f"  ⚠️  无法从 Module4 加载 ProductionPlan: {e}")


def _load_orchestrator_dynamic_data(
    orchestrator: object,
    current_date: pd.Timestamp,
    config: dict
) -> None:
    """
    从Orchestrator并行加载动态数据。

    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期
        config: 目标配置字典（会被修改）
    """
    date_str = current_date.strftime('%Y-%m-%d')

    def _get_beginning():
        try:
            return orchestrator.get_beginning_inventory_view(date_str)
        except Exception as e:
            print(f"  ⚠️  加载BeginningInventory失败: {e}")
            return pd.DataFrame()

    def _get_intransit():
        try:
            return orchestrator.get_planning_intransit_view(date_str)
        except Exception as e:
            print(f"  ⚠️  加载InTransit失败: {e}")
            return pd.DataFrame()

    def _get_delivery_gr():
        try:
            return load_orchestrator_delivery_gr(orchestrator, current_date)
        except Exception as e:
            print(f"  ⚠️  加载DeliveryGR失败: {e}")
            return pd.DataFrame()

    def _get_open_deployment():
        try:
            return load_orchestrator_open_deployment(orchestrator, current_date)
        except Exception as e:
            print(f"  ⚠️  加载OpenDeployment失败: {e}")
            return pd.DataFrame()

    def _get_space_quota():
        try:
            return orchestrator.get_space_quota_view(date_str)
        except Exception as e:
            print(f"  ⚠️  加载ReceivingSpace失败: {e}")
            return pd.DataFrame()

    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_inv = ex.submit(_get_beginning)
            f_it = ex.submit(_get_intransit)
            f_gr = ex.submit(_get_delivery_gr)
            f_open = ex.submit(_get_open_deployment)
            f_space = ex.submit(_get_space_quota)

            results = {
                'InventoryLog': f_inv.result(),
                'InTransit': f_it.result(),
                'DeliveryGR': f_gr.result(),
                'OpenDeployment': f_open.result(),
                'ReceivingSpace': f_space.result()
            }

            for key, df in results.items():
                if isinstance(df, pd.DataFrame):
                    config[key] = df
                else:
                    print(f"  ⚠️  并行加载 {key} 返回非DataFrame，回退为空表")
                    config[key] = pd.DataFrame()

    except Exception as e:
        print(f"  ⚠️  并行加载 Orchestrator 数据失败，回退串行: {e}")
        try:
            config['InventoryLog'] = orchestrator.get_beginning_inventory_view(
                date_str
            )
            config['InTransit'] = orchestrator.get_planning_intransit_view(
                date_str
            )
            config['DeliveryGR'] = load_orchestrator_delivery_gr(
                orchestrator, current_date
            )
            config['OpenDeployment'] = load_orchestrator_open_deployment(
                orchestrator, current_date
            )
            config['ReceivingSpace'] = orchestrator.get_space_quota_view(
                date_str
            )
        except Exception as e2:
            print(f"  ⚠️  从 Orchestrator 加载动态数据失败: {e2}")


def _process_date_fields(config: dict) -> None:
    """
    处理配置中的日期字段。

    Args:
        config: 配置字典（会被修改）
    """
    for sheet, fields in DATE_FIELDS_MAP.items():
        if sheet not in config or config[sheet].empty:
            continue
        for f in fields:
            if f in config[sheet].columns:
                config[sheet][f] = pd.to_datetime(config[sheet][f])


def load_integrated_config(
    config_dict: dict,
    module1_output_dir: str,
    module4_output_path: str,
    orchestrator: object,
    current_date: pd.Timestamp,
    module1_result: Optional[dict] = None
) -> dict:
    """
    加载集成配置数据（替代load_config）。

    Args:
        config_dict: 配置字典
        module1_output_dir: Module1输出目录
        module4_output_path: Module4输出文件路径
        orchestrator: Orchestrator实例
        current_date: 当前日期
        module1_result: Module1运行结果（可选，用于数据库模式）

    Returns:
        dict: 完整的配置字典
    """
    t0 = time.perf_counter()
    config: Dict = {}
    validation_log = []

    # 1. 加载静态配置
    _load_static_config(config_dict, config)

    # 2. 加载Module1数据
    config['SupplyDemandLog'] = config_dict.get(
        'M5_SupplyDemandLog', pd.DataFrame()
    )

    if module1_result is not None:
        _load_module1_data_from_memory(module1_result, config)
    elif module1_output_dir and current_date:
        _load_module1_data_from_file(module1_output_dir, current_date, config)
    else:
        config['OrderLog'] = pd.DataFrame()
        config['TodayShipment'] = pd.DataFrame()

    # 3. 加载生产计划
    config['ProductionPlan'] = pd.DataFrame()
    if orchestrator and current_date:
        _load_production_from_orchestrator(orchestrator, current_date, config)

    if config['ProductionPlan'].empty:
        _load_production_from_module4(module4_output_path, config)

    # 4. 加载M4_MaterialLocationLineCfg
    config['M4_MaterialLocationLineCfg'] = config_dict.get(
        'M4_MaterialLocationLineCfg', pd.DataFrame()
    )

    # 5. 加载Orchestrator动态数据
    if orchestrator and current_date:
        _load_orchestrator_dynamic_data(orchestrator, current_date, config)
    else:
        for key in ['InventoryLog', 'InTransit', 'DeliveryGR',
                    'OpenDeployment', 'ReceivingSpace']:
            config[key] = pd.DataFrame()

    # 6. 规范化ProductionPlan
    if 'ProductionPlan' in config and isinstance(
        config['ProductionPlan'], pd.DataFrame
    ):
        pp = config['ProductionPlan']
        if not pp.empty and 'available_date' not in pp.columns:
            if 'date' in pp.columns:
                pp = pp.rename(columns={'date': 'available_date'})
        if 'available_date' in pp.columns:
            pp['available_date'] = pd.to_datetime(
                pp['available_date'], errors='coerce'
            )
        config['ProductionPlan'] = pp

    # 7. 确保必要的键存在
    for key in ['SupplyDemandLog', 'ProductionPlan', 'InventoryLog',
                'InTransit', 'ReceivingSpace']:
        if key not in config:
            config[key] = pd.DataFrame()

    # 8. 处理日期字段
    _process_date_fields(config)

    # 9. 最终格式化
    for sheet_name, df in sorted(config.items()):
        if isinstance(df, pd.DataFrame) and not df.empty:
            config[sheet_name] = normalize_identifiers(df)

    config['ValidationLog'] = validation_log
    print(f"[M5] load_integrated_config 用时: {time.perf_counter()-t0:.3f}s")
    return config


def load_config(input_path: str) -> dict:
    """
    独立模式读取Excel配置。

    Args:
        input_path: 输入Excel文件路径

    Returns:
        dict: 配置字典
    """
    config = {}
    validation_log = []
    xl = pd.ExcelFile(input_path)

    for sheet in REQUIRED_SHEETS:
        if sheet not in xl.sheet_names:
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': f'Missing required sheet: {sheet}'
            })
            config[sheet] = pd.DataFrame()
        else:
            config[sheet] = xl.parse(sheet)

    # 字段校验
    if not config['SupplyDemandLog'].empty:
        missing_cols = [
            c for c in SDL_REQUIRED_COLUMNS
            if c not in config['SupplyDemandLog'].columns
        ]
        if missing_cols:
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': (
                    f'SupplyDemandLog missing columns: '
                    f'{",".join(missing_cols)}'
                )
            })

    # 日期类型处理
    _process_date_fields(config)

    # 格式化标识符字段
    for sheet_name, df in config.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            config[sheet_name] = normalize_identifiers(df)

    config['ValidationLog'] = validation_log
    return config
