# -*- coding: utf-8 -*-
"""
DB版本 vs ChainSight_Dev 数据对比验证工具

比较两个版本的仿真输出数据，确保数据一致性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from datetime import datetime
import sys

# 配置
CHAINSIGHT_DEV_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\test_files\BC_S5\run_20260122_171104")
DB_CONNECTION = "postgresql://postgres:123456@localhost:5432/test_db"


def connect_db():
    """连接到PostgreSQL数据库"""
    return psycopg2.connect(DB_CONNECTION)


def load_db_table(table_name: str) -> pd.DataFrame:
    """从数据库加载表"""
    conn = connect_db()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return df
    except Exception as e:
        print(f"  ⚠️ 加载表 {table_name} 失败: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_excel_sheet(file_path: Path) -> pd.DataFrame:
    """加载Excel文件"""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"  ⚠️ 加载文件 {file_path} 失败: {e}")
        return pd.DataFrame()


def load_csv_file(file_path: Path) -> pd.DataFrame:
    """加载CSV文件"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"  ⚠️ 加载文件 {file_path} 失败: {e}")
        return pd.DataFrame()


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name: str, 
                       key_cols: list = None, value_cols: list = None,
                       tolerance: float = 0.01) -> dict:
    """
    比较两个DataFrame
    
    Args:
        df1: ChainSight_Dev输出
        df2: DB版本输出  
        name: 比较名称
        key_cols: 用于匹配的关键列
        value_cols: 需要比较的值列
        tolerance: 数值比较的容差
    
    Returns:
        比较结果字典
    """
    result = {
        'name': name,
        'dev_rows': len(df1),
        'db_rows': len(df2),
        'row_match': len(df1) == len(df2),
        'passed': False,
        'details': []
    }
    
    if df1.empty and df2.empty:
        result['passed'] = True
        result['details'].append("两个输出均为空")
        return result
    
    if df1.empty or df2.empty:
        result['details'].append(f"数据不完整: Dev={len(df1)}行, DB={len(df2)}行")
        return result
    
    # 标准化列名
    df1.columns = [c.lower().strip() for c in df1.columns]
    df2.columns = [c.lower().strip() for c in df2.columns]
    
    # 找出共同列
    common_cols = set(df1.columns) & set(df2.columns)
    
    if not common_cols:
        result['details'].append("没有共同的列名")
        return result
    
    # 如果指定了key_cols，按key排序后比较
    if key_cols:
        key_cols_lower = [k.lower() for k in key_cols if k.lower() in common_cols]
        if key_cols_lower:
            df1 = df1.sort_values(key_cols_lower).reset_index(drop=True)
            df2 = df2.sort_values(key_cols_lower).reset_index(drop=True)
    
    # 确定要比较的列
    if value_cols:
        compare_cols = [c.lower() for c in value_cols if c.lower() in common_cols]
    else:
        compare_cols = list(common_cols)
    
    # 逐列比较
    mismatches = []
    for col in compare_cols:
        if col not in df1.columns or col not in df2.columns:
            continue
            
        col1 = df1[col]
        col2 = df2[col]
        
        # 截断到相同长度
        min_len = min(len(col1), len(col2))
        col1 = col1.iloc[:min_len]
        col2 = col2.iloc[:min_len]
        
        # 比较
        try:
            if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
                # 数值比较
                diff = np.abs(col1.fillna(0) - col2.fillna(0))
                max_diff = diff.max()
                mismatch_count = (diff > tolerance).sum()
                if mismatch_count > 0:
                    mismatches.append(f"{col}: {mismatch_count}个值差异超过{tolerance} (最大差异: {max_diff:.4f})")
            else:
                # 字符串比较
                col1_str = col1.astype(str).fillna('')
                col2_str = col2.astype(str).fillna('')
                mismatch_count = (col1_str != col2_str).sum()
                if mismatch_count > 0:
                    mismatches.append(f"{col}: {mismatch_count}个值不匹配")
        except Exception as e:
            mismatches.append(f"{col}: 比较失败 - {e}")
    
    if not mismatches:
        result['passed'] = True
        result['details'].append(f"✅ 所有 {len(compare_cols)} 列数据一致")
    else:
        result['details'].extend(mismatches)
    
    return result


def compare_module1():
    """比较Module1输出"""
    print("\n" + "=" * 60)
    print("📊 Module1 对比")
    print("=" * 60)
    
    results = []
    
    # ChainSight_Dev使用xlsx格式，包含多个sheet
    dev_file = CHAINSIGHT_DEV_OUTPUT / "module1" / "module1_output_20251010.xlsx"
    
    # OrderLog - 使用聚合比较而不是逐行比较
    try:
        dev_orderlog = pd.read_excel(dev_file, sheet_name="OrderLog")
    except Exception as e:
        print(f"  ⚠️ 加载Dev OrderLog失败: {e}")
        dev_orderlog = pd.DataFrame()
    
    db_orderlog = load_db_table("module1_output_orderlog")
    # 只取最后一天的数据进行比较 - sim_date格式为20251010
    if not db_orderlog.empty and 'sim_date' in db_orderlog.columns:
        db_orderlog = db_orderlog[db_orderlog['sim_date'].astype(str) == '20251010']
    
    # 使用聚合方式比较OrderLog
    orderlog_result = {
        'name': 'OrderLog',
        'dev_rows': len(dev_orderlog),
        'db_rows': len(db_orderlog),
        'row_match': len(dev_orderlog) == len(db_orderlog),
        'passed': False,
        'details': []
    }
    
    if not dev_orderlog.empty and not db_orderlog.empty:
        # 标准化列名
        dev_orderlog.columns = [c.lower().strip() for c in dev_orderlog.columns]
        db_orderlog.columns = [c.lower().strip() for c in db_orderlog.columns]
        
        # 按material+location聚合比较总数量
        dev_qty_col = 'quantity' if 'quantity' in dev_orderlog.columns else 'order_qty'
        db_qty_col = 'quantity' if 'quantity' in db_orderlog.columns else 'order_qty'
        
        if dev_qty_col in dev_orderlog.columns and db_qty_col in db_orderlog.columns:
            dev_total = dev_orderlog[dev_qty_col].sum()
            db_total = db_orderlog[db_qty_col].sum()
            
            # 按material聚合
            dev_by_mat = dev_orderlog.groupby('material')[dev_qty_col].sum().sort_index()
            db_by_mat = db_orderlog.groupby('material')[db_qty_col].sum().sort_index()
            
            total_match = abs(dev_total - db_total) < 1
            mat_match = len(dev_by_mat) == len(db_by_mat)
            
            if total_match and mat_match:
                # 检查每个material的总量
                common_mats = set(dev_by_mat.index) & set(db_by_mat.index)
                mat_diff_count = 0
                for mat in common_mats:
                    if abs(dev_by_mat[mat] - db_by_mat[mat]) > 1:
                        mat_diff_count += 1
                
                if mat_diff_count == 0:
                    orderlog_result['passed'] = True
                    orderlog_result['details'].append(f"✅ 聚合验证通过: 总量={dev_total:.0f}, 物料数={len(common_mats)}")
                else:
                    orderlog_result['details'].append(f"部分物料数量不一致: {mat_diff_count}个")
            else:
                orderlog_result['details'].append(f"总量: Dev={dev_total:.0f}, DB={db_total:.0f}")
                orderlog_result['details'].append(f"物料数: Dev={len(dev_by_mat)}, DB={len(db_by_mat)}")
        else:
            orderlog_result['details'].append("找不到quantity列")
    elif dev_orderlog.empty and db_orderlog.empty:
        orderlog_result['passed'] = True
        orderlog_result['details'].append("两个输出均为空")
    else:
        orderlog_result['details'].append(f"数据不完整: Dev={len(dev_orderlog)}行, DB={len(db_orderlog)}行")
    
    results.append(orderlog_result)
    print(f"  OrderLog: Dev={orderlog_result['dev_rows']}行, DB={orderlog_result['db_rows']}行 - {'✅ 通过' if orderlog_result['passed'] else '❌ 失败'}")
    for detail in orderlog_result['details'][:3]:
        print(f"    {detail}")
    
    # ShipmentLog
    try:
        dev_shipment = pd.read_excel(dev_file, sheet_name="ShipmentLog")
    except Exception as e:
        print(f"  ⚠️ 加载Dev ShipmentLog失败: {e}")
        dev_shipment = pd.DataFrame()
    
    db_shipment = load_db_table("module1_output_shipmentlog")
    if not db_shipment.empty and 'sim_date' in db_shipment.columns:
        db_shipment = db_shipment[db_shipment['sim_date'].astype(str) == '20251010']
    
    result = compare_dataframes(
        dev_shipment, db_shipment, "ShipmentLog",
        key_cols=['material', 'location'],
        value_cols=['shipped_qty']
    )
    results.append(result)
    print(f"  ShipmentLog: Dev={result['dev_rows']}行, DB={result['db_rows']}行 - {'✅ 通过' if result['passed'] else '❌ 失败'}")
    
    return results


def compare_module3():
    """比较Module3输出"""
    print("\n" + "=" * 60)
    print("📊 Module3 对比")
    print("=" * 60)
    
    results = []
    
    # NetDemand - ChainSight_Dev使用xlsx格式
    dev_file = CHAINSIGHT_DEV_OUTPUT / "module3" / "Module3Output_20251010.xlsx"
    try:
        dev_netdemand = pd.read_excel(dev_file, sheet_name="NetDemand")
    except Exception as e:
        print(f"  ⚠️ 加载Dev NetDemand失败: {e}")
        dev_netdemand = pd.DataFrame()
    
    db_netdemand = load_db_table("module3_output_netdemand")
    if not db_netdemand.empty and 'sim_date' in db_netdemand.columns:
        db_netdemand = db_netdemand[db_netdemand['sim_date'].astype(str) == '20251010']
    
    result = compare_dataframes(
        dev_netdemand, db_netdemand, "NetDemand",
        key_cols=['material', 'location'],
        value_cols=['net_demand', 'gross_demand', 'available_inventory']
    )
    results.append(result)
    print(f"  NetDemand: Dev={result['dev_rows']}行, DB={result['db_rows']}行 - {'✅ 通过' if result['passed'] else '❌ 失败'}")
    for detail in result['details'][:3]:
        print(f"    {detail}")
    
    return results


def compare_module5():
    """比较Module5输出"""
    print("\n" + "=" * 60)
    print("📊 Module5 对比")
    print("=" * 60)
    
    results = []
    
    # DeploymentPlan - 查找xlsx文件
    dev_files = list((CHAINSIGHT_DEV_OUTPUT / "module5").glob("Module5Output_20251010*.xlsx"))
    if dev_files:
        try:
            dev_deploy = pd.read_excel(dev_files[0], sheet_name="DeploymentPlan")
        except Exception as e:
            print(f"  ⚠️ 加载Dev DeploymentPlan失败: {e}")
            dev_deploy = pd.DataFrame()
    else:
        dev_deploy = pd.DataFrame()
    
    db_deploy = load_db_table("module5_output_deploymentplan")
    if not db_deploy.empty and 'sim_date' in db_deploy.columns:
        db_deploy = db_deploy[db_deploy['sim_date'].astype(str) == '20251010']
    
    result = compare_dataframes(
        dev_deploy, db_deploy, "DeploymentPlan",
        key_cols=['material', 'from_location', 'to_location'],
        value_cols=['deploy_qty', 'demand_type']
    )
    results.append(result)
    print(f"  DeploymentPlan: Dev={result['dev_rows']}行, DB={result['db_rows']}行 - {'✅ 通过' if result['passed'] else '❌ 失败'}")
    for detail in result['details'][:3]:
        print(f"    {detail}")
    
    return results


def compare_summary():
    """比较汇总报告"""
    print("\n" + "=" * 60)
    print("📊 Summary 对比")
    print("=" * 60)
    
    results = []
    
    # Full Deployment Plan
    dev_full_deploy = load_excel_sheet(CHAINSIGHT_DEV_OUTPUT / "summary" / "full_deployment_plan_report.xlsx")
    db_full_deploy = load_db_table("summary_output_fulldeploymentplan")
    
    result = compare_dataframes(
        dev_full_deploy, db_full_deploy, "FullDeploymentPlan",
        key_cols=['material', 'from_location', 'to_location', 'sim_date'],
        value_cols=['deploy_qty']
    )
    results.append(result)
    print(f"  FullDeploymentPlan: Dev={result['dev_rows']}行, DB={result['db_rows']}行 - {'✅ 通过' if result['passed'] else '❌ 失败'}")
    
    # Historical Inventory
    dev_inv = load_csv_file(CHAINSIGHT_DEV_OUTPUT / "summary" / "historical_inventory_record.csv")
    db_inv = load_db_table("summary_historical_inventory_record")
    
    result = compare_dataframes(
        dev_inv, db_inv, "HistoricalInventory",
        key_cols=['material', 'location', 'sim_date'],
        value_cols=['opening_qty', 'closing_qty']
    )
    results.append(result)
    print(f"  HistoricalInventory: Dev={result['dev_rows']}行, DB={result['db_rows']}行 - {'✅ 通过' if result['passed'] else '❌ 失败'}")
    for detail in result['details'][:3]:
        print(f"    {detail}")
    
    return results


def compare_orchestrator_state():
    """比较Orchestrator最终状态"""
    print("\n" + "=" * 60)
    print("📊 Orchestrator状态对比")
    print("=" * 60)
    
    # ChainSight_Dev最终状态（从日志解析或最后一天的orchestrator输出）
    dev_state = {
        'date': '2025-10-10',
        'total_inventory_items': 519,
        'total_inventory_quantity': 319539,
        'open_deployment_count': 861,
        'in_transit_count': 543,
        'production_gr_count': 3,
        'delivery_gr_count': 94,
        'shipment_count': 282
    }
    
    # DB版本最终状态（从数据库查询）
    # 使用SQL直接查询，避免Python筛选问题
    conn = connect_db()
    cur = conn.cursor()
    
    # 查询库存
    cur.execute("SELECT COUNT(*), COALESCE(SUM(quantity), 0) FROM orchestrator_unrestricted_inventory WHERE sim_date='20251010'")
    inv_row = cur.fetchone()
    db_inv_items = inv_row[0] if inv_row else 0
    db_inv_qty = int(inv_row[1]) if inv_row and inv_row[1] else 0
    
    # 查询open deployment
    cur.execute("SELECT COUNT(*) FROM orchestrator_open_deployment WHERE sim_date='20251010'")
    deploy_row = cur.fetchone()
    db_deploy_count = deploy_row[0] if deploy_row else 0
    
    # 查询in-transit
    cur.execute("SELECT COUNT(*) FROM orchestrator_planning_intransit WHERE sim_date='20251010'")
    intransit_row = cur.fetchone()
    db_intransit_count = intransit_row[0] if intransit_row else 0
    
    conn.close()
    
    db_state = {
        'date': '2025-10-10',
        'total_inventory_items': db_inv_items,
        'total_inventory_quantity': db_inv_qty,
        'open_deployment_count': db_deploy_count,
        'in_transit_count': db_intransit_count
    }
    
    print(f"  指标                    | ChainSight_Dev | DB版本    | 一致")
    print(f"  " + "-" * 56)
    
    all_match = True
    for key in ['total_inventory_items', 'total_inventory_quantity', 'open_deployment_count', 'in_transit_count']:
        dev_val = dev_state.get(key, 'N/A')
        db_val = db_state.get(key, 'N/A')
        match = dev_val == db_val
        all_match = all_match and match
        status = "✅" if match else "❌"
        print(f"  {key:24} | {str(dev_val):14} | {str(db_val):9} | {status}")
    
    return all_match


def main():
    """主函数"""
    print("=" * 60)
    print("🔍 ChainSight DB版本 vs ChainSight_Dev 数据对比验证")
    print("=" * 60)
    print(f"📅 对比时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 ChainSight_Dev输出: {CHAINSIGHT_DEV_OUTPUT}")
    print(f"🗄️  DB连接: {DB_CONNECTION}")
    
    all_results = []
    
    # Module对比
    all_results.extend(compare_module1())
    all_results.extend(compare_module3())
    all_results.extend(compare_module5())
    all_results.extend(compare_summary())
    
    # Orchestrator状态对比
    orch_match = compare_orchestrator_state()
    
    # 汇总
    print("\n" + "=" * 60)
    print("📊 对比汇总")
    print("=" * 60)
    
    passed = sum(1 for r in all_results if r['passed'])
    total = len(all_results)
    
    print(f"  模块对比: {passed}/{total} 通过")
    print(f"  Orchestrator状态: {'✅ 一致' if orch_match else '❌ 不一致'}")
    
    if passed == total and orch_match:
        print("\n🎉 验证通过！DB版本与ChainSight_Dev输出数据一致")
        return 0
    else:
        print("\n⚠️ 发现差异，请检查上述详细信息")
        for r in all_results:
            if not r['passed']:
                print(f"  ❌ {r['name']}: {r['details'][0] if r['details'] else '未知原因'}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
