# module 6 - Physical Flow Management Module
# module_6_2_4log.py
#
# Integration Mode Support:
# - Standalone Mode: Excel file input/output (legacy)
# - Integrated Mode: Config dict + Orchestrator integration
#
# Data Sources (Integrated):
# - OpenDeployment: orchestrator.get_open_deployment(current_date)
# - M6_ Configs: M6_TruckReleaseCon, M6_TruckCapacityPlan, etc.
# - Global_ Configs: Global_DemandPriority, Global_LeadTime
#
# Orchestrator Integration:
# - orchestrator.process_delivery_plan(delivery_plan_df, simulation_date)
#   - Generates in-transit inventory records
#   - Generates delivery_gr records  
#   - Offsets open_deployment records
#
# Execution Pattern: Daily processing following Module4/5 pattern
# Module Execution Order: Module1 → Module4 → Module5 → Module6 → Module3

import pandas as pd
import numpy as np
import ast
import os
from typing import Tuple, List, Dict
from math import floor
from datetime import datetime

# ---------------------- Safe rule evaluator ----------------------
class SafeExpressionEvaluator:
    def __init__(self, allowed_names):
        self.allowed_names = set(allowed_names)

    def eval(self, expr: str, context: dict) -> bool:
        expr = (expr or '').strip()
        expr = expr.replace('AND','and').replace('OR','or').replace('NOT','not')
        if not expr:
            return False
        node = ast.parse(expr, mode='eval')
        return self._eval_node(node.body, context)

    def _eval_node(self, node, context):
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, context) for v in node.values]
            if isinstance(node.op, ast.And):  return all(values)
            if isinstance(node.op, ast.Or):   return any(values)
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            results = []
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                if   isinstance(op, ast.Eq):   results.append(left == right)
                elif isinstance(op, ast.NotEq):results.append(left != right)
                elif isinstance(op, ast.Lt):   results.append(left <  right)
                elif isinstance(op, ast.LtE):  results.append(left <= right)
                elif isinstance(op, ast.Gt):   results.append(left >  right)
                elif isinstance(op, ast.GtE):  results.append(left >= right)
                else: raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
                left = right
            return all(results)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._eval_node(node.operand, context)
        elif isinstance(node, ast.Name):
            if node.id not in self.allowed_names:
                raise ValueError(f"Variable '{node.id}' is not allowed")
            if node.id not in context:
                raise ValueError(f"Variable '{node.id}' not found in context")
            return context[node.id]
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Unsupported syntax: {ast.dump(node)}")

def load_standalone_config(input_excel: str) -> dict:
    """
    加载独立模式的配置数据（从 Excel 文件）
    """
    print(f"📝 正在从 '{input_excel}' 读取输入数据...")
    try:
        config = {
            'DeploymentPlan': pd.read_excel(input_excel, sheet_name='DeploymentPlan'),
            'TruckReleaseCon': pd.read_excel(input_excel, sheet_name='TruckReleaseCon'),
            'TruckCapacityPlan': pd.read_excel(input_excel, sheet_name='TruckCapacityPlan'),
            'TruckTypeSpecs': pd.read_excel(input_excel, sheet_name='TruckTypeSpecs'),
            'MaterialMD': pd.read_excel(input_excel, sheet_name='MaterialMD'),
            'DemandPriority': pd.read_excel(input_excel, sheet_name='DemandPriority'),
            'LeadTime': pd.read_excel(input_excel, sheet_name='LeadTime'),
            'DeliveryDelayDistribution': pd.read_excel(input_excel, sheet_name='DeliveryDelayDistribution'),
            'MDQBypassRules': pd.read_excel(input_excel, sheet_name='MDQBypassRules')
        }
        
        # Optional seed in file
        xl = pd.ExcelFile(input_excel)
        if 'RandomSeed' in xl.sheet_names:
            rs = pd.read_excel(input_excel, sheet_name='RandomSeed')
            if 'random_seed' in rs.columns and not rs.empty and pd.notna(rs.iloc[0]['random_seed']):
                file_seed = int(rs.iloc[0]['random_seed'])
                np.random.seed(file_seed)
                print(f"🌱 已从文件设置随机种子: {file_seed}")
        
        return config
        
    except Exception as e:
        print(f"❌ 读取输入失败: {e}")
        raise

def load_integrated_config(
    config_dict: dict,
    orchestrator: object,
    current_date: pd.Timestamp
) -> dict:
    """
    加载集成配置数据，替代原来的Excel文件输入
    
    Args:
        config_dict: 配置数据字典
        orchestrator: Orchestrator实例
        current_date: 当前日期
        
    Returns:
        dict: 集成配置数据
    """
    config = {}
    validation_log = []
    
    try:
        # 1. 从Orchestrator加载OpenDeployment（替代DeploymentPlan）
        open_deployment = orchestrator.get_open_deployment(current_date)
        print(f"  🔍 从Orchestrator获取open_deployment: {len(open_deployment) if open_deployment is not None else 0} 条")
        
        if open_deployment is None or open_deployment.empty:
            print(f"[WARN] No open deployment found for {current_date.strftime('%Y-%m-%d')}")
            open_deployment = pd.DataFrame(columns=[
                'material', 'sending', 'receiving', 'planned_deployment_date', 
                'deployed_qty', 'demand_element', 'ori_deployment_uid'
            ])
        else:
            # 按路线类型统计
            open_deployment['route_type'] = open_deployment.apply(
                lambda row: 'self_loop' if row['sending'] == row['receiving'] else 'cross_node', axis=1
            )
            route_stats = open_deployment['route_type'].value_counts()
            print(f"  📊 路线统计: {route_stats.to_dict()}")
            
            # 显示跨节点路线详情
            cross_node = open_deployment[open_deployment['route_type'] == 'cross_node']
            if len(cross_node) > 0:
                cross_routes = cross_node.groupby(['sending', 'receiving']).size().reset_index(name='count')
                print(f"  🚚 跨节点路线详情:")
                for _, row in cross_routes.iterrows():
                    print(f"    {row['sending']} -> {row['receiving']}: {row['count']} 项")
            else:
                print(f"  ⚠️  无跨节点路线数据")
        
        # 确保日期字段正确格式化
        if not open_deployment.empty and 'planned_deployment_date' in open_deployment.columns:
            open_deployment['planned_deployment_date'] = pd.to_datetime(open_deployment['planned_deployment_date'])
        
        config['DeploymentPlan'] = open_deployment
        
        # 2. 从配置表加载M6_开头的配置数据
        m6_configs = {
            'TruckReleaseCon': 'M6_TruckReleaseCon',
            'TruckCapacityPlan': 'M6_TruckCapacityPlan', 
            'TruckTypeSpecs': 'M6_TruckTypeSpecs',
            'MaterialMD': 'M6_MaterialMD',
            'DeliveryDelayDistribution': 'M6_DeliveryDelayDistribution',
            'MDQBypassRules': 'M6_MDQBypassRules'
        }
        
        for config_key, sheet_name in m6_configs.items():
            if sheet_name in config_dict:
                config[config_key] = config_dict[sheet_name].copy()
            else:
                validation_log.append({
                    'sheet': sheet_name, 'row': '', 
                    'issue': f'Missing required configuration sheet: {sheet_name}'
                })
                config[config_key] = pd.DataFrame()
        
        # 3. 从配置表加载Global_开头的共享配置数据
        global_configs = {
            'DemandPriority': 'Global_DemandPriority',
            'LeadTime': 'Global_LeadTime'
        }
        
        for config_key, sheet_name in global_configs.items():
            if sheet_name in config_dict:
                config[config_key] = config_dict[sheet_name].copy()
            else:
                validation_log.append({
                    'sheet': sheet_name, 'row': '',
                    'issue': f'Missing required global configuration sheet: {sheet_name}'
                })
                config[config_key] = pd.DataFrame()
        
        # 4. 日期字段处理
        date_fields = {
            'DeploymentPlan': ['planned_deployment_date'],
            'TruckCapacityPlan': ['date', 'eff_from', 'eff_to'],
            'DeliveryDelayDistribution': ['date'] if 'date' in config.get('DeliveryDelayDistribution', pd.DataFrame()).columns else []
        }
        
        for sheet, fields in date_fields.items():
            if sheet in config and not config[sheet].empty:
                for field in fields:
                    if field in config[sheet].columns:
                        config[sheet][field] = pd.to_datetime(config[sheet][field], errors='coerce')
        
        config['ValidationLog'] = validation_log
        print(f"✅ Integrated config loaded: {len(config['DeploymentPlan'])} deployment plans, {len(validation_log)} validation issues")
        
    except Exception as e:
        print(f"❌ Error loading integrated config: {str(e)}")
        validation_log.append({'sheet': 'General', 'row': '', 'issue': f'Config loading error: {str(e)}'})
        config['ValidationLog'] = validation_log
        # 提供默认空配置以防止程序崩溃
        for key in ['DeploymentPlan', 'TruckReleaseCon', 'TruckCapacityPlan', 'TruckTypeSpecs', 
                   'MaterialMD', 'DeliveryDelayDistribution', 'MDQBypassRules', 'DemandPriority', 'LeadTime']:
            if key not in config:
                config[key] = pd.DataFrame()
    
    return config

# ---------------------- Helpers ----------------------
def _generate_validation_report(validation_log: List[Dict], output_file: str):
    """
    生成validation.txt报告
    
    Args:
        validation_log: 验证日志列表
        output_file: 输出文件路径
    """
    # 生成validation.txt文件路径
    output_dir = os.path.dirname(output_file)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    validation_file = os.path.join(output_dir, f"{base_name}_validation.txt")
    
    from datetime import datetime
    
    # 统计validation问题
    errors = [log for log in validation_log if log.get('severity') == 'ERROR']
    warnings = [log for log in validation_log if log.get('severity') != 'ERROR']
    
    with open(validation_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODULE6 VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Issues: {len(validation_log)}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Warnings: {len(warnings)}\n")
        f.write("\n")
        
        if errors:
            f.write("❌ CRITICAL ERRORS FOUND\n")
            f.write("Following issues may cause data loss or incorrect processing:\n")
            f.write("-" * 60 + "\n")
            for i, error in enumerate(errors, 1):
                f.write(f"{i}. {error.get('issue', 'Unknown error')}\n")
                if 'impact' in error:
                    f.write(f"   Impact: {error['impact']}\n")
                if 'missing_element' in error:
                    f.write(f"   Missing Element: {error['missing_element']}\n")
                if 'affected_records' in error:
                    f.write(f"   Affected Records: {error['affected_records']}\n")
                if 'route_breakdown' in error:
                    f.write(f"   Route Breakdown: {error['route_breakdown']}\n")
                f.write("\n")
        
        if warnings:
            f.write("⚠️  WARNINGS\n")
            f.write("Following issues should be reviewed but may not block processing:\n")
            f.write("-" * 60 + "\n")
            for i, warning in enumerate(warnings, 1):
                f.write(f"{i}. {warning.get('issue', 'Unknown warning')}\n")
                f.write(f"   Sheet: {warning.get('sheet', 'Unknown')}\n")
                f.write("\n")
        
        if not errors and not warnings:
            f.write("✅ ALL VALIDATIONS PASSED\n")
            f.write("No configuration issues detected.\n")
            f.write("All demand_element types are properly configured.\n")
            f.write("All material metadata is available.\n")
            f.write("All truck configurations are valid.\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("=" * 80 + "\n")
        
        if errors:
            f.write("1. Fix all ERROR-level issues before proceeding\n")
            f.write("2. Add missing demand_element configurations to Global_DemandPriority\n")
            f.write("3. Ensure all material metadata is defined in M6_MaterialMD\n")
        elif warnings:
            f.write("1. Review WARNING-level issues for optimization opportunities\n")
            f.write("2. Check truck capacity and configuration alignment\n")
        else:
            f.write("1. Configuration is optimal for current deployment plans\n")
            f.write("2. Monitor validation reports in future runs\n")
            f.write("3. Consider adding bypass rules if delivery performance is suboptimal\n")
        
        f.write("\n")
        f.write("For detailed information, check the ValidationLog sheet in the Excel output.\n")
        f.write("=" * 80 + "\n")
    
    if validation_log:
        status = "with issues" if (errors or warnings) else "clean"
        print(f"📄 Validation报告已生成 ({status}): {validation_file}")
    else:
        print(f"📄 Validation报告已生成 (no issues): {validation_file}")


def should_bypass_mdq(
    context: dict,
    rules: pd.DataFrame,
    evaluator: SafeExpressionEvaluator
) -> Tuple[bool, str]:
    for _, rule in rules.iterrows():
        matched = True
        for col in ['sending','receiving','truck_type','demand_element']:
            rule_val = str(rule.get(col, 'ALL'))
            if rule_val != 'ALL' and str(context.get(col)) != rule_val:
                matched = False
                break
        if not matched:
            continue
        try:
            expr = rule['condition_logic']
            if evaluator.eval(expr, context):
                print(f"  🔹 规则命中: RuleID={rule.get('rule_id')}, Condition='{expr}'")
                return True, rule.get('rule_id')
        except Exception as e:
            print(f"  ⚠️  规则表达式评估失败 (RuleID={rule.get('rule_id')}): {e}")
            continue
    return False, None

def _normalize_capacity_plan(truck_cap_df: pd.DataFrame,
                             sim_start: pd.Timestamp,
                             sim_end: pd.Timestamp) -> pd.DataFrame:
    """
    兼容两种输入：逐日(date) 或 区间(eff_from, eff_to)；展开并聚合为日粒度（重叠求和）
    """
    df = truck_cap_df.copy()
    has_range = {'eff_from','eff_to'}.issubset(set(df.columns))
    has_daily = 'date' in df.columns
    parts = []

    if has_daily:
        d = df[['date','sending','receiving','truck_type','truck_number']].copy()
        d['date'] = pd.to_datetime(d['date'])
        d = d[(d['date'] >= sim_start) & (d['date'] <= sim_end)]
        parts.append(d)

    if has_range:
        r = df[['eff_from','eff_to','sending','receiving','truck_type','truck_number']].copy()
        r['eff_from'] = pd.to_datetime(r['eff_from'])
        r['eff_to']   = pd.to_datetime(r['eff_to'])
        r['from_clip'] = r['eff_from'].clip(lower=sim_start)
        r['to_clip']   = r['eff_to'].clip(upper=sim_end)
        r = r[r['from_clip'] <= r['to_clip']]
        exploded = []
        for _, row in r.iterrows():
            dates = pd.date_range(row['from_clip'], row['to_clip'], freq='D')
            exploded.append(pd.DataFrame({
                'date': dates,
                'sending': row['sending'],
                'receiving': row['receiving'],
                'truck_type': row['truck_type'],
                'truck_number': row['truck_number'],
            }))
        if exploded:
            parts.append(pd.concat(exploded, ignore_index=True))

    if not parts:
        return pd.DataFrame(columns=['date','sending','receiving','truck_type','truck_number'])

    cap_daily = pd.concat(parts, ignore_index=True)
    cap_daily = cap_daily.groupby(['date','sending','receiving','truck_type'], as_index=False)['truck_number'].sum()
    return cap_daily

def sample_delivery_delay(sending, receiving, dist_df: pd.DataFrame) -> int:
    """
    采样延迟：
    1) 精确路线；2) 兜底行 sending=ALL & receiving=ALL；3) 默认 0 天
    """
    if dist_df is None or dist_df.empty:
        return 0
    exact = dist_df[(dist_df['sending']==sending) & (dist_df['receiving']==receiving)]
    if not exact.empty:
        delays = exact['delay_days'].to_numpy()
        probs  = exact['probability'].to_numpy()
    else:
        global_mask = (dist_df['sending'].astype(str).str.upper()=='ALL') & \
                      (dist_df['receiving'].astype(str).str.upper()=='ALL')
        global_rows = dist_df[global_mask]
        if not global_rows.empty:
            delays = global_rows['delay_days'].to_numpy()
            probs  = global_rows['probability'].to_numpy()
        else:
            return 0
    probs = np.array(probs, dtype=float)
    if probs.sum() <= 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(delays, p=probs))

def run_daily_physical_flow(
    config_dict: dict,
    orchestrator: object,
    current_date: pd.Timestamp,
    output_dir: str,
    max_wait_days: int = 7,
    random_seed: int = None
) -> dict:
    """
    每日物流执行函数，处理当日的部署计划
    
    Args:
        config_dict: 配置数据字典
        orchestrator: Orchestrator实例
        current_date: 当前仿真日期
        output_dir: 输出目录
        max_wait_days: 最大等待天数
        random_seed: 随机种子
        
    Returns:
        dict: 包含输出结果的字典
    """
    print(f"📊 Module6 Daily Physical Flow - {current_date.strftime('%Y-%m-%d')}")
    
    # 加载集成配置
    config = load_integrated_config(config_dict, orchestrator, current_date)
    
    # 生成输出文件名
    daily_output_file = f"{output_dir}/Module6Output_{current_date.strftime('%Y%m%d')}.xlsx"
    
    # 调用主逻辑（集成模式）
    result = main(
        # 集成模式参数
        config_dict=config_dict,
        orchestrator=orchestrator,
        current_date=current_date.strftime('%Y-%m-%d'),
        output_path=daily_output_file,
        # 公共参数
        max_wait_days=max_wait_days,
        random_seed=random_seed
    )
    
    return {
        'daily_output_file': daily_output_file,
        'delivery_plan': result.get('delivery_plan', pd.DataFrame()),
        'vehicle_log': result.get('vehicle_log', pd.DataFrame()),
        'validation_log': result.get('validation_log', []),
        'statistics': result.get('statistics', {})
    }

# ---------------------- Core: packing & shipping ----------------------
def calculate_physical_inventory(
    orchestrator: object,
    current_date: pd.Timestamp
) -> Dict[Tuple[str, str], float]:
    """
    获取指定日期的实物库存（创建副本，避免直接修改Orchestrator库存）
    
    由于Orchestrator在Module6执行时，其unrestricted_inventory已经按照以下公式更新：
    实物库存 = 期初unrestricted inventory + 当日production GR + 当日delivery GR - 当日shipment
    
    注意：创建库存副本，避免Module6直接修改Orchestrator的实际库存。
    
    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期
        
    Returns:
        Dict: 实物库存字典副本 {(material, location): physical_quantity}
    """
    try:
        # 创建Orchestrator库存的副本，避免直接修改原始库存
        physical_inventory = {}
        for key, qty in orchestrator.unrestricted_inventory.items():
            physical_inventory[key] = float(qty)
        
        print(f"  📊 实物库存统计: {len(physical_inventory)} 个SKU-地点组合")
        if physical_inventory:
            total_items = sum(1 for qty in physical_inventory.values() if qty > 0)
            positive_qty = sum(qty for qty in physical_inventory.values() if qty > 0)
            print(f"  ✅ 有库存SKU: {total_items}/{len(physical_inventory)}, 总量: {positive_qty:.1f}")
            
            # 调试：显示前5个实物库存明细
            for i, (key, qty) in enumerate(list(physical_inventory.items())[:5]):
                material, location = key
                print(f"    实物库存: {material}@{location}: {qty:.1f}")
        
        return physical_inventory
        
    except Exception as e:
        print(f"  ⚠️  获取实物库存失败: {e}")
        return {}

def run_physical_flow_module(
    # Standalone mode parameters
    input_excel: str = None,
    simulation_start: str = None,
    simulation_end: str = None,
    output_excel: str = None,
    # Integrated mode parameters
    config_dict: dict = None,
    orchestrator: object = None,
    current_date: str = None,
    output_path: str = None,
    # Common parameters
    max_wait_days: int = 7,
    random_seed: int = None
):
    # 判断运行模式
    if config_dict is not None:
        # 集成模式
        print("🔄 Module6 运行于集成模式")
        sim_date = pd.to_datetime(current_date)
        sim_dates = pd.DatetimeIndex([sim_date])  # 单日处理，使用DatetimeIndex
        output_file = output_path
        # 加载集成配置
        config = load_integrated_config(config_dict, orchestrator, sim_date)
    else:
        # 独立模式 - 保持向后兼容
        print("📜 Module6 运行于独立模式")
        config = load_standalone_config(input_excel)
        sim_start = pd.to_datetime(simulation_start)
        sim_end = pd.to_datetime(simulation_end)
        output_file = output_excel
        
        # 计算仿真日期范围
        dp = config['DeploymentPlan']
        if not dp.empty:
            sim_dates = pd.date_range(
                max(sim_start, dp['planned_deployment_date'].min()),
                min(sim_end, dp['planned_deployment_date'].max() + pd.Timedelta(days=max_wait_days))
            )
        else:
            sim_dates = pd.date_range(sim_start, sim_end)

    # Seed
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"🌱 随机种子已设置为: {random_seed}")

    print(f"📅 仿真时间范围: {sim_dates.min().date()} 到 {sim_dates.max().date()}")

    # 数据验证
    validation_log = list(config.get('ValidationLog', []))
    print("📊 开始数据校验...")

    # 获取数据
    dp = config['DeploymentPlan']
    truck_con = config['TruckReleaseCon']
    truck_cap = config['TruckCapacityPlan']
    truck_specs = config['TruckTypeSpecs']
    material_md = config['MaterialMD']
    demand_prio = config['DemandPriority']
    lead_time = config['LeadTime']
    delay_dist = config['DeliveryDelayDistribution']
    bypass_rules = config['MDQBypassRules']

    prio_map = demand_prio.set_index('demand_element')['priority'].to_dict()
    missing_prio = dp[~dp['demand_element'].isin(prio_map.keys())]
    if not missing_prio.empty:
        # 记录缺失的demand_element详细信息
        missing_elements = missing_prio['demand_element'].unique()
        for val in missing_elements:
            missing_records = missing_prio[missing_prio['demand_element'] == val]
            # 统计路线类型
            route_stats = missing_records.apply(
                lambda row: 'self_loop' if row['sending'] == row['receiving'] else 'cross_node', axis=1
            ).value_counts()
            route_info = ', '.join([f"{k}: {v}" for k, v in route_stats.items()])
            
            validation_log.append({
                'sheet': 'Global_DemandPriority',
                'row': '',
                'issue': f'Missing priority configuration for demand_element "{val}" '
                        f'(affects {len(missing_records)} records: {route_info}). '
                        f'Records will be filtered out and not processed.',
                'severity': 'ERROR',
                'impact': f'Data Loss - {len(missing_records)} deployment plans excluded',
                'missing_element': val,
                'affected_records': len(missing_records),
                'route_breakdown': route_info
            })
        
        print(f"  ⚠️  发现 {len(missing_elements)} 个缺失的demand_element配置，将过滤 {len(missing_prio)} 条记录")
        for val in missing_elements:
            missing_count = len(missing_prio[missing_prio['demand_element'] == val])
            print(f"    - '{val}': {missing_count} 条记录")
        
        # 过滤掉缺失priority的记录
        dp = dp[dp['demand_element'].isin(prio_map.keys())]
        print(f"  📊 过滤后保留: {len(dp)} 条记录")

    mat_map = material_md.set_index('material')[['demand_unit_to_weight','demand_unit_to_volume']].to_dict('index')
    missing_mat = dp[~dp['material'].isin(mat_map.keys())]
    if not missing_mat.empty:
        missing_materials = missing_mat['material'].unique()
        for val in missing_materials:
            missing_records = missing_mat[missing_mat['material'] == val]
            validation_log.append({
                'sheet': 'M6_MaterialMD',
                'row': '',
                'issue': f'Missing material metadata for "{val}" '
                        f'(affects {len(missing_records)} records). '
                        f'Default unit conversion factors (1.0) will be used.',
                'severity': 'WARNING',
                'impact': f'Default Values Used - {len(missing_records)} records use defaults',
                'missing_material': val,
                'affected_records': len(missing_records)
            })
        
        print(f"  ⚠️  发现 {len(missing_materials)} 个缺失的material配置，将使用默认值")
        # 使用默认值处理
        dp['demand_unit_to_weight'] = dp['material'].map(lambda x: mat_map.get(x, {}).get('demand_unit_to_weight', 1.0))
        dp['demand_unit_to_volume'] = dp['material'].map(lambda x: mat_map.get(x, {}).get('demand_unit_to_volume', 1.0))
    else:
        dp = dp.merge(material_md, on='material', how='left')
        dp['demand_unit_to_weight'] = dp['demand_unit_to_weight'].fillna(1.0)
        dp['demand_unit_to_volume'] = dp['demand_unit_to_volume'].fillna(1.0)

    spec_map = truck_specs.set_index('truck_type').to_dict('index')

    # --- 阈值>1.0 的配置告警（仍允许，但不会靠阈值触发） ---
    bad_th = truck_con[(truck_con['WFR'] > 1.0) | (truck_con['VFR'] > 1.0)]
    for _, r in bad_th.iterrows():
        validation_log.append({
            'sheet': 'M6_TruckReleaseCon', 
            'row': '',
            'issue': f"Threshold > 1.0 for route {r['sending']}->{r['receiving']} type {r['truck_type']} "
                     f"(WFR={r['WFR']}, VFR={r['VFR']}). Will never trigger by threshold; bypass only.",
            'severity': 'WARNING',
            'impact': 'Configuration Issue - Route can only be triggered by bypass rules',
            'route': f"{r['sending']}->{r['receiving']}",
            'truck_type': r['truck_type'],
            'wfr_threshold': r['WFR'],
            'vfr_threshold': r['VFR']
        })

    # 车型规格缺失
    missing_specs = set(truck_con['truck_type'].unique()) - set(spec_map.keys())
    for val in missing_specs:
        affected_routes = truck_con[truck_con['truck_type'] == val]
        validation_log.append({
            'sheet': 'M6_TruckTypeSpecs',
            'row': '',
            'issue': f'Missing truck type specification for "{val}" '
                    f'(affects {len(affected_routes)} route configurations). '
                    f'Routes using this truck type will be skipped.',
            'severity': 'ERROR',
            'impact': f'Data Loss - {len(affected_routes)} route configurations unavailable',
            'missing_truck_type': val,
            'affected_routes': len(affected_routes)
        })

    # Normalize fields
    dp['planned_deployment_date'] = pd.to_datetime(dp['planned_deployment_date'])
    lead_time[['PDT','GR']] = lead_time[['PDT','GR']].astype(int)
    delay_dist['delay_days'] = delay_dist['delay_days'].astype(int)

    # UIDs & priority - 使用Orchestrator提供的原始UID
    dp = dp.reset_index(drop=True)
    # 如果没有ori_deployment_uid列，才重新生成
    if 'ori_deployment_uid' not in dp.columns or dp['ori_deployment_uid'].isnull().any():
        print(f"  ⚠️  检测到缺失UID，重新生成")
        dp['ori_deployment_uid'] = [f'UID{i:06d}' for i in dp.index]
    else:
        print(f"  ✅ 使用Orchestrator提供的原UID")
    
    dp['priority'] = dp['demand_element'].map(prio_map)
    dp['waiting_days'] = 0
    dp['simulation_date'] = dp['planned_deployment_date']
    
    print(f"  🔍 处理后的部署计划数量: {len(dp)}")
    # 按路线类型统计
    dp['route_type_debug'] = dp.apply(lambda row: 'self_loop' if row['sending'] == row['receiving'] else 'cross_node', axis=1)
    route_debug_stats = dp['route_type_debug'].value_counts()
    print(f"  📊 部署计划路线统计: {route_debug_stats.to_dict()}")

    # Dict index
    dp_dict = dp.set_index('ori_deployment_uid').to_dict('index')
    # Pending state
    agg_status = {
        uid: {'qty': row['deployed_qty'], 'waiting': 1, 'planned': row['planned_deployment_date']}
        for uid, row in dp_dict.items()
    }

    # Dates & capacity map
    if config_dict is not None:
        # 集成模式: 使用单日范围
        sim_start = sim_dates.min()
        sim_end = sim_dates.max()
    else:
        # 独立模式: 使用原始日期范围
        sim_start = pd.to_datetime(simulation_start)
        sim_end   = pd.to_datetime(simulation_end)
        sim_dates = pd.date_range(
            max(sim_start, dp['planned_deployment_date'].min()),
            min(sim_end,   dp['planned_deployment_date'].max() + pd.Timedelta(days=max_wait_days))
        )
    print(f"📅 模拟时间范围: {sim_dates.min().date()} 到 {sim_dates.max().date()}")

    cap_daily = _normalize_capacity_plan(truck_cap.copy(), sim_start, sim_end)
    cap_map   = cap_daily.set_index(['date','sending','receiving','truck_type'])['truck_number'].to_dict()
    print(f"[Capacity] normalized rows: {len(cap_daily)}")

    # Outputs
    delivery_plan, unsat_log, bypass_log = [], [], []
    vehicle_log = []          # 逐车一行
    evaluator = SafeExpressionEvaluator(
        ['waiting_days','deployed_qty_ratio','exception_MDQ','sending','receiving','truck_type','demand_element']
    )

    # ---------------------- Main simulation loop ----------------------
    # 集成模式：获取当日可用库存用于库存检查
    available_inventory = {}
    inventory_check_enabled = config_dict is not None and orchestrator is not None
    
    for sim_date in sim_dates:
        print(f"\n📆 模拟日期: {sim_date.date()}")
        
        # 集成模式：获取当日实物库存
        if inventory_check_enabled:
            available_inventory = calculate_physical_inventory(orchestrator, sim_date)
            print(f"    💰 库存检查已启用（使用实物库存）")
        else:
            print(f"    ⚠️  独立模式：跳过库存检查")
        
        print(f"    🗒 部署计划状态检查:")
        active_plans = {uid: st for uid, st in agg_status.items() if st['qty'] > 0}
        print(f"    📈 有效计划数: {len(active_plans)}/{len(agg_status)}")
        
        if active_plans:
            print(f"    🔍 前5个有效计划:")
            for i, (uid, st) in enumerate(list(active_plans.items())[:5]):
                print(f"      {i+1}. {uid}: qty={st['qty']}, planned={st['planned']}, waiting={st['waiting']}")
        
        pending_rows = []
        # collect todays pendings
        for uid, st in agg_status.items():
            if st['qty'] <= 0:
                continue
            planned_date = pd.to_datetime(st['planned'])
            waiting_days = (sim_date - planned_date).days + 1
            if waiting_days > max_wait_days:
                continue
            
            # 集成模式：直接处理所有open deployment数据，无需时间过滤
            # 因为Orchestrator已经管理了部署计划的生命周期，确保数据的合理性
            if config_dict is not None:
                # 集成模式：信任Orchestrator的数据，直接处理
                pass  # 无需额外时间过滤
            else:
                # 独立模式：保持原逻辑，只处理当天及过去的计划
                if planned_date > sim_date:
                    continue
                    
            full = dp_dict[uid]
            
            # 调试：记录跨节点计划的详细信息
            route_type = "自循环" if full['sending'] == full['receiving'] else "跨节点"
            if route_type == "跨节点":
                print(f"    🔍 跨节点计划 {uid}: {full['sending']}->{full['receiving']}, planned={planned_date.date()}, waiting={waiting_days}天, qty={st['qty']}")
            
            pending_rows.append({
                'ori_deployment_uid': uid,
                'material': full['material'],
                'sending': full['sending'],
                'receiving': full['receiving'],
                'planned_deployment_date': planned_date,
                'deployed_qty': st['qty'],
                'demand_element': full['demand_element'],
                'demand_unit_to_weight': full['demand_unit_to_weight'],
                'demand_unit_to_volume': full['demand_unit_to_volume'],
                'priority': full['priority'],
                'waiting_days': waiting_days,
            })

        if not pending_rows:
            print("  ✅ 无待处理需求")
            continue

        pendf = pd.DataFrame(pending_rows)
        print(f"  📦 发现 {len(pending_rows)} 个待处理需求")
        
        # 调试：按路线类型统计
        route_stats = pendf.groupby(['sending', 'receiving']).size().reset_index(name='count')
        print(f"  📊 路线统计:")
        for _, row in route_stats.iterrows():
            route_type = "自循环" if row['sending'] == row['receiving'] else "跨节点"
            print(f"    {row['sending']} -> {row['receiving']}: {row['count']} 项 ({route_type})")
        
        # 过滤出跨节点路线（忽略自循环）
        cross_node_df = pendf[pendf['sending'] != pendf['receiving']].copy()
        if cross_node_df.empty:
            print("  ✅ 无跨节点需求需要处理")
            continue
            
        # 全局优先级排序：按优先级(asc) + 计划日期(asc) + 路线排序，确保高优先级需求优先获得库存
        cross_node_df_sorted = cross_node_df.sort_values([
            'priority',  # 1=最高优先级
            'planned_deployment_date',  # 越早越优先
            'sending',  # 稳定排序
            'receiving'  # 稳定排序
        ]).reset_index(drop=True)
        
        print(f"  🎯 跨节点需求已按全局优先级排序: {len(cross_node_df_sorted)} 项")
        if len(cross_node_df_sorted) > 0:
            print(f"  📋 前5个高优先级需求:")
            for i, (_, row) in enumerate(cross_node_df_sorted.head().iterrows()):
                print(f"    {i+1}. {row['material']}@{row['sending']}->{row['receiving']}: qty={row['deployed_qty']}, priority={row['priority']}, date={row['planned_deployment_date'].date()}")
        
        # 新的处理逻辑：按全局优先级逐条处理需求，在路线级别进行车辆分配
        processed_routes = set()  # 记录已处理的路线
        
        for _, row in cross_node_df_sorted.iterrows():
            sending = row['sending']
            receiving = row['receiving']
            route_key = (sending, receiving)
            
            # 如果该路线已经被处理过，跳过
            if route_key in processed_routes:
                continue
                
            processed_routes.add(route_key)
            
            # 获取该路线的所有需求（按全局优先级排序）
            route_demands = cross_node_df_sorted[
                (cross_node_df_sorted['sending'] == sending) & 
                (cross_node_df_sorted['receiving'] == receiving)
            ].copy()
            
            print(f"    🚚 处理跨节点路线: {sending} -> {receiving} ({len(route_demands)} 项需求)")

            
            truck_cfgs = truck_con[(truck_con['sending']==sending) & (truck_con['receiving']==receiving)]
            if truck_cfgs.empty:
                print("      ⚠️  该路线无可用卡车配置")
                continue

            # truck_type order by optimal first
            optimal_types = truck_cfgs[truck_cfgs['optimal_type']=='Y']['truck_type'].tolist()
            all_types = truck_cfgs['truck_type'].tolist()
            type_seq = optimal_types + [x for x in all_types if x not in optimal_types]
            print(f"      🚛 车型序列: {type_seq}")

            # 使用全局排序后的需求列表
            remaining_demands = route_demands.copy()
            # 仅用于 UnsatisfiedMDQLog 展示
            route_mdq = truck_cfgs['MDQ'].min() if not truck_cfgs.empty else np.nan
            
            # try each truck type
            for truck_type in type_seq:
                if remaining_demands.empty:
                    break

                n_truck_total = int(cap_map.get((sim_date, sending, receiving, truck_type), 0))
                if n_truck_total == 0:
                    print(f"      🚫 {truck_type}: 今日无可用车辆")
                    continue

                conf = truck_cfgs[truck_cfgs['truck_type']==truck_type].iloc[0]
                spec = spec_map.get(truck_type)
                if not spec:
                    print(f"      ⚠️  {truck_type}: 车型规格缺失，跳过")
                    continue

                wfr_th, vfr_th = float(conf['WFR']), float(conf['VFR'])
                mdq = float(conf['MDQ']) if pd.notna(conf['MDQ']) else 0.0
                cap_w = float(spec['capacity_qty_in_weight'])
                cap_v = float(spec['capacity_qty_in_volume'])
                print(f"      🚛 尝试车型: {truck_type} (可用: {n_truck_total} 辆) 阈值 WFR={wfr_th}, VFR={vfr_th}")

                used = 0  # 已用车辆数（该车型）
                while used < n_truck_total and not remaining_demands.empty:
                    # 一辆车的“打包器”：严格不超容量；允许对最后一个需求“部分装载”
                    load_records = []   # {idx, load_qty}
                    q_units = 0.0
                    w_sum = 0.0
                    v_sum = 0.0
                    # 库存检查：跟踪按物料的累计装载量
                    material_loaded = {}  # {material: total_loaded_qty}

                    # —— 初次扫描：尽量装入，但不超过 cap（无 ∞）——
                    for idx, demand_row in remaining_demands.iterrows():
                        qty_pending = float(demand_row['deployed_qty'])
                        if qty_pending <= 0:
                            continue
                        uw = float(demand_row['demand_unit_to_weight'])
                        uv = float(demand_row['demand_unit_to_volume'])
                        material = demand_row['material']

                        cap_w_rem = max(0.0, cap_w - w_sum)
                        cap_v_rem = max(0.0, cap_v - v_sum)

                        limits = [qty_pending]            # 订单剩余量硬上限
                        if uw > 0: limits.append(floor(cap_w_rem / uw))
                        if uv > 0: limits.append(floor(cap_v_rem / uv))
                        
                        # 库存检查：限制发货量不超过可用库存
                        if inventory_check_enabled:
                            inv_key = (material, sending)
                            available_qty = available_inventory.get(inv_key, 0)
                            already_loaded = material_loaded.get(material, 0)
                            inventory_limit = max(0, available_qty - already_loaded)
                            limits.append(inventory_limit)
                            
                            if inventory_limit <= 0:
                                # 库存不足，跳过该物料
                                print(f"        🚫 库存不足: {material}@{sending}, 可用={available_qty}, 已装={already_loaded}")
                                continue

                        addable = int(max(0, min(limits)))
                        if addable <= 0:
                            continue

                        load_records.append({'idx': idx, 'load_qty': addable, 'demand_row': demand_row})
                        q_units += addable
                        w_sum  += addable * uw
                        v_sum  += addable * uv
                        
                        # 更新物料的累计装载量
                        material_loaded[material] = material_loaded.get(material, 0) + addable

                        if w_sum >= cap_w or v_sum >= cap_v:
                            break

                    # 当前车装载比例
                    wfr = (w_sum / cap_w) if cap_w > 0 else 0.0
                    vfr = (v_sum / cap_v) if cap_v > 0 else 0.0

                    # 代表性上下文
                    if load_records:
                        # 使用已装载记录中优先级最高的需求
                        highest_record = min(
                            load_records,
                            key=lambda r: (r['demand_row']['priority'], r['demand_row']['planned_deployment_date'])
                        )
                        repr_type = highest_record['demand_row']['demand_element']
                        repr_wait = highest_record['demand_row']['waiting_days']
                    else:
                        repr_type, repr_wait = None, 0

                    context = {
                        'sending': sending, 'receiving': receiving, 'truck_type': truck_type,
                        'demand_element': repr_type, 'waiting_days': repr_wait,
                        'deployed_qty_ratio': (q_units/mdq) if mdq > 0 else 0.0,
                        'exception_MDQ': 1 if mdq == 0 else 0
                    }
                    bypass, rule_id = should_bypass_mdq(context, bypass_rules, evaluator)

                    trigger_cause = None
                    if load_records and (wfr >= wfr_th or vfr >= vfr_th):
                        trigger_cause = 'threshold'
                    elif load_records and bypass:
                        trigger_cause = 'bypass'

                    if trigger_cause:
                        # —— 触发后再尽量贴近 1.0（仍不超）——
                        taken = {r['idx'] for r in load_records}
                        for idx, demand_row in remaining_demands.iterrows():
                            if idx in taken:
                                continue
                            qty_pending = float(demand_row['deployed_qty'])
                            if qty_pending <= 0:
                                continue
                            uw = float(demand_row['demand_unit_to_weight'])
                            uv = float(demand_row['demand_unit_to_volume'])
                            material = demand_row['material']

                            cap_w_rem = max(0.0, cap_w - w_sum)
                            cap_v_rem = max(0.0, cap_v - v_sum)

                            limits = [qty_pending]
                            if uw > 0: limits.append(floor(cap_w_rem / uw))
                            if uv > 0: limits.append(floor(cap_v_rem / uv))
                            
                            # 库存检查：限制发货量不超过可用库存
                            if inventory_check_enabled:
                                inv_key = (material, sending)
                                available_qty = available_inventory.get(inv_key, 0)
                                already_loaded = material_loaded.get(material, 0)
                                inventory_limit = max(0, available_qty - already_loaded)
                                limits.append(inventory_limit)
                                
                                if inventory_limit <= 0:
                                    continue

                            addable = int(max(0, min(limits)))
                            if addable <= 0:
                                continue

                            load_records.append({'idx': idx, 'load_qty': addable, 'demand_row': demand_row})
                            q_units += addable
                            w_sum  += addable * uw
                            v_sum  += addable * uv
                            
                            # 更新物料的累计装载量
                            material_loaded[material] = material_loaded.get(material, 0) + addable
                            
                            if w_sum >= cap_w or v_sum >= cap_v:
                                break

                        # 更新比例（最终值）
                        wfr = (w_sum / cap_w) if cap_w > 0 else 0.0
                        vfr = (v_sum / cap_v) if cap_v > 0 else 0.0

                        # === 车辆级日志（逐车一条） ===
                        vehicle_no = used + 1
                        vehicle_uid = f"{sim_date:%Y%m%d}-{sending}-{receiving}-{truck_type}-#{vehicle_no}"
                        vehicle_log.append({
                            'date': sim_date,
                            'sending': sending, 'receiving': receiving, 'truck_type': truck_type,
                            'vehicle_no': vehicle_no,
                            'vehicle_uid': vehicle_uid,
                            'total_units': int(q_units),
                            'total_weight': w_sum,
                            'total_volume': v_sum,
                            'WFR': min(wfr, 1.0),
                            'VFR': min(vfr, 1.0),
                            'trigger': trigger_cause
                        })

                        # === 明细发运记录生成 ===
                        for rec in load_records:
                            # 使用demand_row而不是group.loc
                            sub = rec['demand_row']
                            uid = sub['ori_deployment_uid']
                            lt  = lead_time[(lead_time['sending']==sending) & (lead_time['receiving']==receiving)]
                            PDT = int(lt['PDT'].iloc[0]) if not lt.empty else 0
                            GR  = int(lt['GR'].iloc[0])  if not lt.empty else 0
                            delay = sample_delivery_delay(sending, receiving, delay_dist)
                            #actual_delivery_date 的新定义是 actual_ship_date + OTD + GR + delay；
                            #PDT 的角色从“定义”降级为“计划用估计值”（仅用于 M5 倒排、排程预估，而非实际物流入库时效）。
                            lt = lead_time[(lead_time['sending']==sending) & (lead_time['receiving']==receiving)]
                            if lt.empty:
                                raise ValueError(f"缺少路线 {sending}->{receiving} 的 LeadTime 行")

                            OTD = int(pd.to_numeric(lt['OTD'].iloc[0])) if 'OTD' in lt.columns else 0
                            GR  = int(pd.to_numeric(lt['GR'].iloc[0]))  if 'GR'  in lt.columns else 0

                            OTD = max(0, OTD)
                            GR  = max(0, GR)

                            ship_date = sim_date  # 发运日定义
                            eta = ship_date + pd.Timedelta(days=OTD + GR + delay)

                            delivery_plan.append({
                                'vehicle_uid': vehicle_uid,
                                'ori_deployment_uid': uid,
                                'material': sub['material'],
                                'sending': sending, 'receiving': receiving,
                                'planned_deployment_date': sub['planned_deployment_date'],
                                'actual_ship_date': ship_date,
                                'actual_delivery_date': eta,
                                'delivery_qty': rec['load_qty'], 'truck_type': truck_type,
                                'truck_load_pct': min(max(wfr, vfr), 1.0),
                                'WFR': min(wfr, 1.0), 'VFR': min(vfr, 1.0)
                            })
                            
                            # 注意: 库存扣减由Orchestrator统一处理，Module6只负责生成delivery plan
                            # 更新可用库存（仅用于Module6内部的后续规划计算）
                            if inventory_check_enabled:
                                inv_key = (sub['material'], sending)
                                if inv_key in available_inventory:
                                    available_inventory[inv_key] -= rec['load_qty']
                                    available_inventory[inv_key] = max(0, available_inventory[inv_key])  # 避免负数
                            
                            # 更新agg_status和原始数据
                            agg_status[uid]['qty'] = max(0, agg_status[uid]['qty'] - rec['load_qty'])
                            # 更新remaining_demands中的数量
                            remaining_demands.at[rec['idx'], 'deployed_qty'] = max(0, sub['deployed_qty'] - rec['load_qty'])

                        # 只在“bypass 触发”时记录命中（阈值触发不记）
                        if trigger_cause == 'bypass':
                            for rec in load_records:
                                sub = rec['demand_row']
                                bypass_log.append({
                                    'ori_deployment_uid': sub['ori_deployment_uid'],
                                    'rule_id': rule_id, 'simulation_date': sim_date,
                                    'context_snapshot': str(context),
                                    'vehicle_uid': vehicle_uid
                                })

                        # 更新remaining_demands，移除已处理完毕的需求
                        remaining_demands = remaining_demands[remaining_demands['deployed_qty'] > 0].copy()
                        used += 1
                    else:
                        print(f"        🚫 {truck_type}: 聚合未触发发运，剩余 {len(remaining_demands)} 项")
                        break  # 换车型

            # 路线级收尾：处理该路线剩余的未发出需求
            route_remaining = route_demands[route_demands['deployed_qty'] > 0]
            for _, row in route_remaining.iterrows():
                uid = row['ori_deployment_uid']
                if agg_status[uid]['qty'] <= 0:
                    continue
                agg_status[uid]['waiting'] += 1
                if agg_status[uid]['waiting'] > max_wait_days:
                    unsat_log.append({
                        'ori_deployment_uid': uid, 'material': row['material'],
                        'sending': sending, 'receiving': receiving, 'demand_element': row['demand_element'],
                        'planned_deployment_date': row['planned_deployment_date'],
                        'simulation_date': sim_date, 'waiting_days': agg_status[uid]['waiting'],
                        'accumulated_qty': agg_status[uid]['qty'], 'min_MDQ': route_mdq,
                        'reason': 'waited_too_long'
                    })
                    agg_status[uid]['qty'] = 0

    # ---------------------- Usage summary ----------------------
    if vehicle_log:
        vehicle_df = pd.DataFrame(vehicle_log)
        usage = (vehicle_df
                 .groupby(['date','sending','receiving','truck_type'], as_index=False)
                 .agg(truck_used=('vehicle_uid','nunique')))
    else:
        vehicle_df = pd.DataFrame(columns=['date','sending','receiving','truck_type','vehicle_no','vehicle_uid',
                                           'total_units','total_weight','total_volume','WFR','VFR','trigger'])
        usage = pd.DataFrame(columns=['date','sending','receiving','truck_type','truck_used'])

    # ---------------------- Write outputs ----------------------
    delivery_plan_df = pd.DataFrame(delivery_plan)
    vehicle_df_final = vehicle_df if vehicle_log else pd.DataFrame(columns=[
        'date','sending','receiving','truck_type','vehicle_no','vehicle_uid',
        'total_units','total_weight','total_volume','WFR','VFR','trigger'
    ])
    usage_df = usage if vehicle_log else pd.DataFrame(columns=['date','sending','receiving','truck_type','truck_used'])
    unsat_df = pd.DataFrame(unsat_log)
    validation_df = pd.DataFrame(validation_log)
    bypass_df = pd.DataFrame(bypass_log)
    
    # Excel 输出
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        delivery_plan_df.to_excel(writer, sheet_name='DeliveryPlan', index=False)
        vehicle_df_final.to_excel(writer, sheet_name='VehicleLog', index=False)
        usage_df.to_excel(writer, sheet_name='TruckUsageLog', index=False)
        unsat_df.to_excel(writer, sheet_name='UnsatisfiedMDQLog', index=False)
        validation_df.to_excel(writer, sheet_name='ValidationLog', index=False)
        bypass_df.to_excel(writer, sheet_name='BypassRuleHitLog', index=False)
    
    # 生成validation.txt报告
    _generate_validation_report(validation_log, output_file)

    # 注意：Orchestrator状态更新由main_integration.py统一处理
    # 避免重复调用导致双重库存扣减
    if config_dict is not None and orchestrator is not None and not delivery_plan_df.empty:
        print(f"✅ Processed {len(delivery_plan_df)} M6 delivery plans for {current_date}")
        print(f"✅ Orchestrator 状态将由main_integration统一更新")
        # 移除直接调用，避免与main_integration.py中的process_module6_delivery重复

    statistics = {
        'delivery_count': len(delivery_plan),
        'vehicle_count': usage_df['truck_used'].sum() if not usage_df.empty else 0,
        'unsatisfied_count': len(unsat_log),
        'bypass_count': len(bypass_log)
    }
    
    print(f"\n🎉 仿真完成! 输出已保存至: {output_file}")
    try:
        print(f"📊 统计: 发运 {statistics['delivery_count']} 明细，车辆 {statistics['vehicle_count']} 车，未满足 {statistics['unsatisfied_count']} 项，bypass 命中 {statistics['bypass_count']} 次")
    except Exception as e:
        print(f"[WARN] printing statistics failed: {e}")
    # 返回结果用于集成模式
    return {
        'delivery_plan': delivery_plan_df,
        'vehicle_log': vehicle_df_final,
        'truck_usage': usage_df,
        'unsatisfied_log': unsat_df,
        'validation_log': validation_df,
        'bypass_log': bypass_df,
        'statistics': statistics
    }

# 主函数别名（保持与Module4/5一致）
main = run_physical_flow_module


# ======================== Example ========================
if __name__ == "__main__":
    # Standalone mode example
    run_physical_flow_module(
        input_excel='Module_6_1_1/config_SC.xlsx',   
        simulation_start='2025-08-01',
        simulation_end='2025-08-03',
        output_excel='Module_6_1_1/output_SC.xlsx',
        random_seed=42
    )
