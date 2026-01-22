# -*- coding: utf-8 -*-
"""
对比数据库版本和本地 ChainSight_Dev 版本的输出数据
确保两者数据一致性
"""

import pandas as pd
import numpy as np
import psycopg
from pathlib import Path
from datetime import datetime

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'test_db',
    'user': 'postgres',
    'password': '123456'
}

# 本地输出目录
LOCAL_OUTPUT_DIR = Path(r"C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev\BC_S5\run_20260121_195229")

# 日期范围
SIM_DATES = ['2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10']


def connect_db():
    """连接数据库"""
    conn_str = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"
    return psycopg.connect(conn_str)


def load_db_data(table_name: str, sim_dates: list = None) -> pd.DataFrame:
    """从数据库加载数据"""
    with connect_db() as conn:
        if sim_dates:
            # 尝试不同的日期列名
            date_cols = ['sim_date', 'date', 'plan_deploy_date', 'available_date']
            for date_col in date_cols:
                try:
                    query = f"SELECT * FROM {table_name} WHERE {date_col} IN ({','.join(['%s']*len(sim_dates))})"
                    df = pd.read_sql(query, conn, params=sim_dates)
                    if not df.empty:
                        print(f"  ✅ {table_name}: {len(df)} rows (filtered by {date_col})")
                        return df
                except Exception:
                    continue
            
            # 如果没有日期列，加载全部数据
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                print(f"  ✅ {table_name}: {len(df)} rows (no date filter)")
                return df
            except Exception as e:
                print(f"  ❌ {table_name}: {e}")
                return pd.DataFrame()
        else:
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                print(f"  ✅ {table_name}: {len(df)} rows")
                return df
            except Exception as e:
                print(f"  ❌ {table_name}: {e}")
                return pd.DataFrame()


def load_local_module5_data() -> dict:
    """加载本地 Module5 输出数据"""
    module5_dir = LOCAL_OUTPUT_DIR / "module5"
    result = {
        'deployment_plan': [],
        'unfulfilled_log': [],
        'stock_on_hand_log': []
    }
    
    for date_str in SIM_DATES:
        date_fmt = date_str.replace('-', '')
        xlsx_path = module5_dir / f"Module5Output_{date_fmt}.xlsx"
        
        if xlsx_path.exists():
            try:
                xl = pd.ExcelFile(xlsx_path)
                
                # DeploymentPlan
                if 'DeploymentPlan' in xl.sheet_names:
                    df = pd.read_excel(xl, 'DeploymentPlan')
                    df['sim_date'] = date_str
                    result['deployment_plan'].append(df)
                
                # UnfulfilledLog
                if 'UnfulfilledLog' in xl.sheet_names:
                    df = pd.read_excel(xl, 'UnfulfilledLog')
                    df['sim_date'] = date_str
                    result['unfulfilled_log'].append(df)
                
                # StockOnHandLog
                if 'StockOnHandLog' in xl.sheet_names:
                    df = pd.read_excel(xl, 'StockOnHandLog')
                    df['sim_date'] = date_str
                    result['stock_on_hand_log'].append(df)
                    
            except Exception as e:
                print(f"  ❌ {xlsx_path.name}: {e}")
    
    # 合并所有日期数据
    for key in result:
        if result[key]:
            result[key] = pd.concat(result[key], ignore_index=True)
            print(f"  ✅ Local {key}: {len(result[key])} rows")
        else:
            result[key] = pd.DataFrame()
    
    return result


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """标准化 DataFrame 用于比较"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # 标准化列名（全小写）
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # 移除数据库特有的列
    drop_cols = ['id', 'run_id', 'created_at', 'updated_at']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 标准化字符串列
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 标准化数值列
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # 安全转换为整数（处理inf值）
        df[col] = df[col].replace([np.inf, -np.inf], 0).astype(int)
    
    # 标准化日期列
    date_cols = ['date', 'sim_date', 'plan_deploy_date', 'requirement_date', 'available_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
    
    return df


def compare_dataframes(df_db: pd.DataFrame, df_local: pd.DataFrame, name: str, key_cols: list) -> dict:
    """比较两个 DataFrame"""
    result = {
        'name': name,
        'db_rows': len(df_db),
        'local_rows': len(df_local),
        'match': False,
        'differences': []
    }
    
    if df_db.empty and df_local.empty:
        result['match'] = True
        return result
    
    if df_db.empty or df_local.empty:
        result['differences'].append(f"One is empty: DB={len(df_db)}, Local={len(df_local)}")
        return result
    
    # 标准化
    df_db = normalize_df(df_db)
    df_local = normalize_df(df_local)
    
    # 找出共同列
    common_cols = sorted(set(df_db.columns) & set(df_local.columns))
    if not common_cols:
        result['differences'].append("No common columns")
        return result
    
    # 只保留共同列
    df_db = df_db[common_cols].copy()
    df_local = df_local[common_cols].copy()
    
    # 行数比较
    if len(df_db) != len(df_local):
        result['differences'].append(f"Row count mismatch: DB={len(df_db)}, Local={len(df_local)}")
    
    # 按key列排序后比较
    key_cols_present = [c for c in key_cols if c in common_cols]
    if key_cols_present:
        df_db = df_db.sort_values(key_cols_present).reset_index(drop=True)
        df_local = df_local.sort_values(key_cols_present).reset_index(drop=True)
    
    # 逐列比较
    mismatched_cols = []
    for col in common_cols:
        try:
            if not df_db[col].equals(df_local[col]):
                # 计算差异数量
                min_len = min(len(df_db), len(df_local))
                diff_count = sum(df_db[col].iloc[:min_len] != df_local[col].iloc[:min_len])
                mismatched_cols.append(f"{col}: {diff_count} diffs")
        except Exception as e:
            mismatched_cols.append(f"{col}: compare error - {e}")
    
    if mismatched_cols:
        result['differences'].extend(mismatched_cols)
    else:
        result['match'] = True
    
    return result


def main():
    print("=" * 80)
    print("🔍 数据库版本 vs 本地 ChainSight_Dev 版本 数据对比")
    print("=" * 80)
    print(f"📅 仿真日期范围: {SIM_DATES[0]} 到 {SIM_DATES[-1]}")
    print(f"📂 本地输出目录: {LOCAL_OUTPUT_DIR}")
    print()
    
    # 1. 加载本地数据
    print("📥 加载本地数据...")
    local_data = load_local_module5_data()
    print()
    
    # 2. 加载数据库数据
    print("📥 加载数据库数据...")
    db_tables = {
        'deployment_plan': 'module5_output_deploymentplan',
        'unfulfilled_log': 'module5_output_unfulfilledlog',
        'stock_on_hand_log': 'module5_output_stockonhandlog'
    }
    
    db_data = {}
    for key, table in db_tables.items():
        db_data[key] = load_db_data(table, SIM_DATES)
    print()
    
    # 3. 比较数据
    print("📊 数据比较结果:")
    print("-" * 80)
    
    comparisons = [
        ('deployment_plan', ['sim_date', 'material', 'sending', 'receiving', 'demand_element']),
        ('unfulfilled_log', ['sim_date', 'date', 'material', 'sending', 'receiving', 'demand_element']),
        ('stock_on_hand_log', ['sim_date', 'material', 'location'])
    ]
    
    all_match = True
    for data_key, key_cols in comparisons:
        result = compare_dataframes(
            db_data.get(data_key, pd.DataFrame()),
            local_data.get(data_key, pd.DataFrame()),
            data_key,
            key_cols
        )
        
        status = "✅ 匹配" if result['match'] else "❌ 不匹配"
        print(f"\n{data_key}:")
        print(f"  状态: {status}")
        print(f"  DB行数: {result['db_rows']}, Local行数: {result['local_rows']}")
        
        if result['differences']:
            print(f"  差异:")
            for diff in result['differences'][:10]:  # 只显示前10个差异
                print(f"    - {diff}")
            if len(result['differences']) > 10:
                print(f"    ... 还有 {len(result['differences']) - 10} 个差异")
        
        if not result['match']:
            all_match = False
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ 所有数据匹配！数据库版本与本地版本一致。")
    else:
        print("❌ 存在数据差异，需要进一步检查。")
    print("=" * 80)
    
    # 4. 详细分析 UnfulfilledLog 差异
    if not db_data.get('unfulfilled_log', pd.DataFrame()).empty and \
       not local_data.get('unfulfilled_log', pd.DataFrame()).empty:
        print("\n📊 UnfulfilledLog 详细分析:")
        print("-" * 80)
        
        df_db = normalize_df(db_data['unfulfilled_log'])
        df_local = normalize_df(local_data['unfulfilled_log'])
        
        # 按日期统计
        if 'date' in df_db.columns and 'date' in df_local.columns:
            print("\n按日期统计行数:")
            for date in SIM_DATES:
                db_count = len(df_db[df_db['date'] == date]) if 'date' in df_db.columns else 0
                local_count = len(df_local[df_local['date'] == date]) if 'date' in df_local.columns else 0
                match_icon = "✅" if db_count == local_count else "❌"
                print(f"  {date}: DB={db_count}, Local={local_count} {match_icon}")
        
        # 按 material 统计
        if 'material' in df_db.columns and 'material' in df_local.columns:
            print("\n按物料统计总 unfulfilled_qty:")
            db_by_mat = df_db.groupby('material')['unfulfilled_qty'].sum() if 'unfulfilled_qty' in df_db.columns else pd.Series()
            local_by_mat = df_local.groupby('material')['unfulfilled_qty'].sum() if 'unfulfilled_qty' in df_local.columns else pd.Series()
            
            all_mats = sorted(set(db_by_mat.index) | set(local_by_mat.index))
            for mat in all_mats[:10]:  # 只显示前10个
                db_qty = db_by_mat.get(mat, 0)
                local_qty = local_by_mat.get(mat, 0)
                if db_qty != local_qty:
                    print(f"  {mat}: DB={db_qty}, Local={local_qty} ❌ (差异: {db_qty - local_qty})")


if __name__ == '__main__':
    main()
