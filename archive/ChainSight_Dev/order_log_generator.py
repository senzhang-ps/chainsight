#!/usr/bin/env python3
"""
独立订单日志生成器

基于Module1的订单生成逻辑，从配置文件中读取M1_DemandForecast, M1_ForecastError, 
M1_OrderCalendar, M1_AOConfig, M1_DPSConfig, M1_SupplyChoiceConfig等配置表，
生成指定日期范围内的订单日志。

用法:
    python order_log_generator.py \
        --config /path/to/config.xlsx \
        --start-date 2024-01-01 \
        --end-date 2024-01-31 \
        --output-dir /path/to/output

功能:
- 读取配置文件中的Module1相关配置表
- 将周度需求预测转换为日度预测
- 应用DPS和Supply Choice配置
- 基于订单日历和AO配置生成订单
- 应用预测误差生成最终订单数量
- 输出指定日期范围内的订单日志
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import os
import re
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import warnings

# 导入Module1的核心函数
from module1 import (
    _normalize_identifiers,
    apply_dps,
    apply_supply_choice,
    expand_forecast_to_days_integer_split,
    generate_daily_orders,
    generate_quantity_with_percent_error,
    consume_forecast_ao_logic,
    consume_forecast_normal_logic
)

def load_order_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """
    加载订单生成所需的配置表
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置数据字典
    """
    print(f"📋 加载配置文件: {config_path}")
    
    try:
        xl = pd.ExcelFile(config_path)
        config_dict = {}
        
        # 定义Module1相关的配置表映射
        module1_sheets = {
            'M1_DemandForecast': 'demand_forecast',
            'M1_ForecastError': 'forecast_error', 
            'M1_OrderCalendar': 'order_calendar',
            'M1_AOConfig': 'ao_config',
            'M1_DPSConfig': 'dps_config',
            'M1_SupplyChoiceConfig': 'supply_choice'
        }
        
        # 加载配置表
        for sheet_name, key in module1_sheets.items():
            if sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                config_dict[key] = _normalize_identifiers(df)
                print(f"  ✅ 加载配置表: {sheet_name} ({len(df)} 行)")
            else:
                config_dict[key] = pd.DataFrame()
                print(f"  ⚠️  配置表不存在: {sheet_name}")
        
        # 验证必需配置
        required_configs = ['demand_forecast', 'order_calendar', 'ao_config', 'forecast_error']
        missing_configs = [k for k in required_configs if config_dict[k].empty]
        
        if missing_configs:
            raise ValueError(f"缺少必需的配置表: {missing_configs}")
        
        return config_dict
        
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")

def generate_order_log_for_date_range(
    config_dict: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    output_dir: str,
    random_seed: Optional[int] = None,
    config_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    为指定日期范围生成订单日志
    
    Args:
        config_dict: 配置数据字典
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        output_dir: 输出目录
        random_seed: 随机种子
        
    Returns:
        dict: 生成结果统计

    Notes:
        2025-10-10: 新增 config_name 参数，用于把配置文件名称附加到输出文件名，
        便于区分不同配置来源的订单日志。保留向后兼容（未提供则保持旧命名）。
    """
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"🎲 设置随机种子: {random_seed}")
    
    # 解析日期
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    
    print(f"📅 生成订单日志: {start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}")
    
    # 获取配置数据
    demand_forecast = config_dict['demand_forecast']
    forecast_error = config_dict['forecast_error']
    order_calendar = config_dict['order_calendar']
    ao_config = config_dict['ao_config']
    dps_config = config_dict['dps_config']
    supply_choice_config = config_dict['supply_choice']
    
    # 处理订单日历
    order_calendar['date'] = pd.to_datetime(order_calendar['date'])
    
    # 检查订单日历的日期范围
    cal_start = order_calendar['date'].min()
    cal_end = order_calendar['date'].max()
    print(f"📅 订单日历日期范围: {cal_start.strftime('%Y-%m-%d')} 到 {cal_end.strftime('%Y-%m-%d')}")
    
    # 检查请求的日期范围是否在订单日历范围内
    if start_dt < cal_start or end_dt > cal_end:
        print(f"⚠️  警告: 请求的日期范围超出订单日历范围")
        print(f"   请求范围: {start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}")
        print(f"   日历范围: {cal_start.strftime('%Y-%m-%d')} 到 {cal_end.strftime('%Y-%m-%d')}")
        
        # 调整日期范围到订单日历范围内
        adjusted_start = max(start_dt, cal_start)
        adjusted_end = min(end_dt, cal_end)
        
        if adjusted_start > adjusted_end:
            print(f"❌ 错误: 调整后的日期范围无效")
            return {
                'total_orders': 0,
                'total_quantity': 0,
                'ao_orders': 0,
                'normal_orders': 0,
                'materials': 0,
                'locations': 0,
                'output_file': None
            }
        
        print(f"🔧 调整日期范围到: {adjusted_start.strftime('%Y-%m-%d')} 到 {adjusted_end.strftime('%Y-%m-%d')}")
        start_dt = adjusted_start
        end_dt = adjusted_end
    
    # 将周度预测转换为日度预测
    if 'week' in demand_forecast.columns:
        print("🔄 将周度预测转换为日度预测...")
        
        # 应用DPS配置
        if not dps_config.empty:
            print("  📊 应用DPS配置...")
            demand_forecast = apply_dps(demand_forecast, dps_config)
        
        # 应用Supply Choice配置
        if not supply_choice_config.empty:
            print("  📊 应用Supply Choice配置...")
            demand_forecast = apply_supply_choice(demand_forecast, supply_choice_config)
        
        # 转换为日度预测
        max_week = int(demand_forecast['week'].max()) if not demand_forecast.empty else 1
        daily_demand_forecast = expand_forecast_to_days_integer_split(
            demand_forecast, start_dt, max_week, end_dt
        )
        print(f"  ✅ 转换完成: {len(daily_demand_forecast)} 个日度记录")
    else:
        # 已经是日度数据
        daily_demand_forecast = demand_forecast.copy()
        print(f"📊 使用现有日度预测: {len(daily_demand_forecast)} 个记录")
    
    # 生成订单日志
    all_orders = []
    current_forecast = daily_demand_forecast.copy()
    
    # 按日期生成订单
    current_date = start_dt
    while current_date <= end_dt:
        print(f"  📅 处理日期: {current_date.strftime('%Y-%m-%d')}")
        
        # 生成当日订单
        daily_orders, consumed_forecast = generate_daily_orders(
            current_date, daily_demand_forecast, current_forecast,
            ao_config, order_calendar, forecast_error
        )
        
        if not daily_orders.empty:
            all_orders.append(daily_orders)
            print(f"    📋 生成 {len(daily_orders)} 个订单")
        
        # 更新预测状态
        current_forecast = consumed_forecast
        
        current_date += pd.Timedelta(days=1)
    
    # 合并所有订单
    if all_orders:
        orders_df = pd.concat(all_orders, ignore_index=True)
        orders_df['quantity'] = orders_df['quantity'].astype(int)
        orders_df = _normalize_identifiers(orders_df)
    else:
        orders_df = pd.DataFrame()
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    # 生成输出文件名（新增: 附加配置文件名 stem）
    if config_name:
        # 仅取文件名（含扩展）再取 stem，并做简单清洗（空格等非安全字符替换为下划线）
        cfg_stem = Path(config_name).stem
        cfg_stem_clean = re.sub(r'[^A-Za-z0-9._-]+', '_', cfg_stem)
        output_file = os.path.join(
            output_dir,
            f"order_log_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}_{cfg_stem_clean}.xlsx"
        )
    else:
        output_file = os.path.join(output_dir, f"order_log_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.xlsx")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        orders_df.to_excel(writer, sheet_name='OrderLog', index=False)
        
        # 创建汇总信息
        summary_data = pd.DataFrame([{
            'Start_Date': start_dt.strftime('%Y-%m-%d'),
            'End_Date': end_dt.strftime('%Y-%m-%d'),
            'Total_Orders': len(orders_df),
            'Total_Quantity': int(orders_df['quantity'].sum()) if not orders_df.empty else 0,
            'AO_Orders': len(orders_df[orders_df['demand_type'] == 'AO']) if not orders_df.empty else 0,
            'Normal_Orders': len(orders_df[orders_df['demand_type'] == 'normal']) if not orders_df.empty else 0,
            'Materials': orders_df['material'].nunique() if not orders_df.empty else 0,
            'Locations': orders_df['location'].nunique() if not orders_df.empty else 0,
            'Generated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"💾 订单日志已保存: {output_file}")
    
    # 返回统计信息
    stats = {
        'total_orders': len(orders_df),
        'total_quantity': int(orders_df['quantity'].sum()) if not orders_df.empty else 0,
        'ao_orders': len(orders_df[orders_df['demand_type'] == 'AO']) if not orders_df.empty else 0,
        'normal_orders': len(orders_df[orders_df['demand_type'] == 'normal']) if not orders_df.empty else 0,
        'materials': orders_df['material'].nunique() if not orders_df.empty else 0,
        'locations': orders_df['location'].nunique() if not orders_df.empty else 0,
        'output_file': output_file
    }
    
    return stats

def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="order_log_generator",
        add_help=True,
        description="基于配置文件生成指定日期范围的订单日志"
    )
    
    parser.add_argument(
        "--config",
        required=True,
        help="配置文件路径 (.xlsx)"
    )
    
    parser.add_argument(
        "--start-date",
        required=True,
        help="开始日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date", 
        required=True,
        help="结束日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        help="随机种子 (可选)"
    )
    
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    """主函数"""
    try:
        # 解析参数
        args = _parse_args(argv or sys.argv[1:])
        
        # 验证配置文件
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
            raise ValueError("配置文件必须是Excel文件 (.xlsx/.xlsm/.xls)")
        
        # 验证日期格式
        try:
            pd.to_datetime(args.start_date)
            pd.to_datetime(args.end_date)
        except Exception as e:
            raise ValueError(f"日期格式错误: {e}")
        
        # 验证日期范围
        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date)
        if start_dt > end_dt:
            raise ValueError("开始日期不能晚于结束日期")
        
        print("🚀 订单日志生成器启动")
        print(f"📂 配置文件: {config_path}")
        print(f"📅 日期范围: {args.start_date} 到 {args.end_date}")
        print(f"📁 输出目录: {args.output_dir}")
        
        # 加载配置
        config_dict = load_order_config(str(config_path))
        
        # 生成订单日志
        stats = generate_order_log_for_date_range(
            config_dict=config_dict,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            config_name=str(config_path)
        )
        
        # 输出统计信息
        print("\n📊 生成完成统计:")
        print(f"  📋 总订单数: {stats['total_orders']}")
        print(f"  📦 总数量: {stats['total_quantity']}")
        print(f"  🔄 AO订单: {stats['ao_orders']}")
        print(f"  📝 普通订单: {stats['normal_orders']}")
        print(f"  🏷️  物料种类: {stats['materials']}")
        print(f"  📍 地点数量: {stats['locations']}")
        print(f"  💾 输出文件: {stats['output_file']}")
        
        print("\n✅ 订单日志生成完成!")
        return 0
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        raise SystemExit(1) from exc

# 使用示例:
# python order_log_generator.py \
#   --config /path/to/config.xlsx \
#   --start-date 2024-01-01 \
#   --end-date 2024-01-31 \
#   --output-dir /path/to/output \
#   --random-seed 42
