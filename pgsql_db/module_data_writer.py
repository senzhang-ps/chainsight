"""
模块数据写入器
将各Module的输入/输出数据写入PostgreSQL数据库
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import os
from datetime import datetime

from .db_connection import DatabaseConnection


class ModuleDataWriter:
    """模块数据写入器"""
    
    def __init__(self, db: DatabaseConnection):
        """
        初始化模块数据写入器
        
        Args:
            db: 数据库连接实例
        """
        self.db = db
        self.written_tables: Dict[str, Dict] = {}
    
    def write_module_output(
        self,
        module_name: str,
        output_dir: str,
        run_id: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """
        写入单个模块的输出数据
        
        Args:
            module_name: 模块名称 (module1, module3, module4, module5, module6)
            output_dir: 模块输出目录
            run_id: 运行ID（用于区分不同运行）
            if_exists: 如果表存在的处理方式
        
        Returns:
            dict: 每个输出文件的写入行数
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"⚠️模块输出目录不存在: {output_dir}")
            return {}
        
        results = {}
        print(f"\n📁 写入 {module_name} 输出数据...")
        
        # 查找所有Excel文件
        excel_files = list(output_path.glob("*.xlsx"))
        
        for excel_file in excel_files:
            try:
                file_results = self._write_excel_file(
                    excel_file, 
                    module_name, 
                    run_id,
                    if_exists
                )
                results[excel_file.name] = file_results
            except Exception as e:
                print(f"  ❌ 文件写入失败 [{excel_file.name}]: {e}")
                results[excel_file.name] = {"error": str(e)}
        
        # 查找所有CSV文件
        csv_files = list(output_path.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                import re
                # 从文件名中提取日期（格式：xxx_YYYYMMDD.csv）
                date_match = re.search(r'_(\d{8})(?:\.csv)?$', csv_file.stem)
                file_date = date_match.group(1) if date_match else None
                
                # 构建表名（去除日期部分）
                stem_without_date = re.sub(r'_\d{8}$', '', csv_file.stem)
                table_name = f"{module_name}_{self._clean_name(stem_without_date)}"
                
                # 添加日期列
                if file_date:
                    if df.empty:
                        df['file_date'] = pd.Series(dtype='string')
                    else:
                        df['file_date'] = file_date
                
                if run_id:
                    if df.empty:
                        df['run_id'] = pd.Series(dtype='string')
                    else:
                        df['run_id'] = run_id
                        
                self.db.create_table_from_df(df, table_name, if_exists)
                results[csv_file.name] = len(df)
                self.written_tables[table_name] = {
                    "source": str(csv_file),
                    "module": module_name,
                    "rows": len(df)
                }
            except Exception as e:
                print(f"  ❌ CSV文件写入失败 [{csv_file.name}]: {e}")
                results[csv_file.name] = {"error": str(e)}
        
        return results
    
    def _write_excel_file(
        self,
        excel_path: Path,
        module_name: str,
        run_id: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """写入单个Excel文件的所有sheet
        
        文件名中的日期（如 Module1Output_20251006.xlsx）会被提取为 file_date 列，
        而不是作为表名的一部分，这样相同结构的数据会追加到同一张表中。
        """
        import re
        results = {}
        
        xl = pd.ExcelFile(excel_path)
        
        # 从文件名中提取日期（格式：xxx_YYYYMMDD.xlsx）
        date_match = re.search(r'_(\d{8})(?:\.xlsx)?$', excel_path.stem)
        file_date = date_match.group(1) if date_match else None
        
        # 构建表名前缀（去除日期部分）
        stem_without_date = re.sub(r'_\d{8}$', '', excel_path.stem)
        file_prefix = f"{module_name}_{self._clean_name(stem_without_date)}"
        
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name)
                
                # 构建表名（不包含日期）
                table_name = f"{file_prefix}_{self._clean_name(sheet_name)}"
                
                # 添加日期列（从文件名提取）
                if file_date:
                    df['file_date'] = file_date
                
                # 添加run_id列
                if run_id:
                    df['run_id'] = run_id
                
                # 写入数据库 (即使 df.empty 也会创建表结构)
                self.db.create_table_from_df(df, table_name, if_exists)
                results[sheet_name] = len(df)
                
                self.written_tables[table_name] = {
                    "source": str(excel_path),
                    "sheet": sheet_name,
                    "module": module_name,
                    "rows": len(df)
                }
                
            except Exception as e:
                print(f"    ⚠️ Sheet [{sheet_name}] 写入失败: {e}")
                results[sheet_name] = -1
        
        return results
    
    def write_all_modules(
        self,
        run_output_dir: str,
        run_id: str = None,
        if_exists: str = "append"
    ) -> Dict[str, Dict]:
        """
        写入所有模块的输出数据
        
        Args:
            run_output_dir: 运行输出根目录
            run_id: 运行ID
            if_exists: 如果表存在的处理方式
        
        Returns:
            dict: 所有模块的写入结果
        """
        base_path = Path(run_output_dir)
        
        if not base_path.exists():
            raise FileNotFoundError(f"运行输出目录不存在: {run_output_dir}")
        
        # 自动生成run_id
        if run_id is None:
            run_id = base_path.name
        
        print("\n" + "=" * 60)
        print(f"写入模块输出数据到数据库")
        print(f"运行目录: {run_output_dir}")
        print(f"运行ID: {run_id}")
        print("=" * 60)
        
        all_results = {}
        modules = ['module1', 'module3', 'module4', 'module5', 'module6', 'orchestrator', 'summary']
        
        start_time = time.time()
        
        for module in modules:
            module_dir = base_path / module
            if module_dir.exists():
                results = self.write_module_output(
                    module, 
                    str(module_dir), 
                    run_id,
                    if_exists
                )
                all_results[module] = results
            else:
                print(f"⚠️模块目录不存在: {module}")
        
        elapsed = time.time() - start_time
        
        # 统计
        total_tables = len(self.written_tables)
        total_rows = sum(info.get('rows', 0) for info in self.written_tables.values())
        
        print("\n" + "=" * 60)
        print(f"📊 写入汇总:")
        print(f"   表数量: {total_tables}")
        print(f"   总行数: {total_rows}")
        print(f"   耗时: {elapsed:.2f}s")
        print("=" * 60)
        
        return all_results
    
    def write_orchestrator_data(
        self,
        orchestrator_dir: str,
        run_id: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """
        写入Orchestrator输出的CSV数据
        
        Args:
            orchestrator_dir: Orchestrator输出目录
            run_id: 运行ID
            if_exists: 如果表存在的处理方式
        
        Returns:
            dict: 每个文件的写入行数
        """
        orch_path = Path(orchestrator_dir)
        if not orch_path.exists():
            print(f"⚠️Orchestrator目录不存在: {orchestrator_dir}")
            return {}
        
        results = {}
        print(f"\n📁 写入Orchestrator数据...")
        
        # 定义Orchestrator输出文件类型
        file_patterns = [
            "unrestricted_inventory_*.csv",
            "open_deployment_*.csv",
            "planning_intransit_*.csv",
            "space_quota_*.csv",
            "delivery_gr_*.csv",
            "production_gr_*.csv",
            "shipment_log_*.csv",
            "delivery_shipment_log_*.csv",
            "inventory_change_log_*.csv",
            "daily_logs_*.csv"
        ]
        
        for pattern in file_patterns:
            csv_files = list(orch_path.glob(pattern))
            
            # 提取基础表名
            base_table_name = pattern.replace("_*.csv", "").replace("*.csv", "")
            table_name = f"orchestrator_{base_table_name}"
            
            if csv_files:
                # 合并所有同类型文件
                dfs = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        # 从文件名提取日期
                        date_part = csv_file.stem.split("_")[-1]
                        
                        if df.empty:
                            df['file_date'] = pd.Series(dtype='string')
                            if run_id:
                                df['run_id'] = pd.Series(dtype='string')
                        else:
                            df['file_date'] = date_part
                            if run_id:
                                df['run_id'] = run_id
                        dfs.append(df)
                    except Exception as e:
                        print(f"    ⚠️ 读取失败 [{csv_file.name}]: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    self.db.create_table_from_df(combined_df, table_name, if_exists)
                    results[base_table_name] = len(combined_df)
                    self.written_tables[table_name] = {
                        "source": str(orch_path),
                        "file_count": len(csv_files),
                        "rows": len(combined_df)
                    }
        
        return results
    
    def write_module_results_from_dict(
        self,
        all_results: Dict[str, Any],
        run_id: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """
        从内存中的模块结果字典直接写入数据库
        
        这个方法用于数据库模式，绕过文件系统直接将DataFrame写入数据库。
        
        Args:
            all_results: 模块运行结果字典，结构为:
                {
                    'module1': [{'orders_df': df, 'shipment_df': df, 'cut_df': df, ...}, ...],
                    'module3': [{'net_demand_df': df}, ...],
                    'module4': [{'production_df': df}, ...],
                    'module5': [{'deployment_plan': df, 'stock_log': df, ...}, ...],
                    'module6': [{'delivery_plan': df, 'truck_usage': df, ...}, ...]
                }
            run_id: 运行ID
            if_exists: 如果表存在的处理方式
        
        Returns:
            dict: 每个表的写入行数
        """
        results = {}
        
        print("\n" + "=" * 60)
        print("📤 从内存写入模块输出到数据库")
        print(f"运行ID: {run_id}")
        print("=" * 60)
        
        # 定义模块输出的DataFrame到表名的映射
        # 键名必须与模块返回的字典键名一致
        module_df_mapping = {
            'module1': {
                'orders_df': 'module1_output_orderlog',
                'shipment_df': 'module1_output_shipmentlog',
                'cut_df': 'module1_output_cutlog',
                'supply_demand_df': 'module1_output_supplydemandlog',
            },
            'module3': {
                'net_demand_df': 'module3_output_netdemand',
            },
            'module4': {
                'production_df': 'module4_output_productionplan',
                'exceed_log': 'module4_output_capacityexceed',
                'issues_df': 'module4_output_validation',
                'changeover_log': 'module4_output_changeover',
            },
            'module5': {
                'deployment_plan': 'module5_output_deploymentplan',
                'stock_on_hand_log': 'module5_output_stockonhandlog',
                'unfulfilled_log': 'module5_output_unfulfilledlog',
                'validation_log': 'module5_output_validation',
            },
            'module6': {
                'delivery_plan': 'module6_output_deliveryplan',
                'truck_usage': 'module6_output_truckusagelog',
                'vehicle_log': 'module6_output_vehiclelog',
                'validation_log': 'module6_output_validationlog',
                'unsatisfied_log': 'module6_output_unsatisfiedlog',
                'bypass_log': 'module6_output_bypasslog',
            },
        }
        
        for module_name, df_mapping in module_df_mapping.items():
            module_results = all_results.get(module_name, [])
            
            if not module_results:
                continue
            
            print(f"\n📁 写入 {module_name} 输出...")
            
            # 收集同一个表的所有数据
            table_data = {table_name: [] for table_name in df_mapping.values()}
            
            for day_result in module_results:
                if not isinstance(day_result, dict):
                    continue
                
                for df_key, table_name in df_mapping.items():
                    df = day_result.get(df_key)
                    if df is not None and isinstance(df, pd.DataFrame):
                        table_data[table_name].append(df)
            
            # 写入每个表
            for table_name, dfs in table_data.items():
                if not dfs:
                    continue
                
                # 合并所有天的数据
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # 添加run_id列
                if run_id:
                    if combined_df.empty:
                        combined_df['run_id'] = pd.Series(dtype='string')
                    else:
                        combined_df['run_id'] = run_id
                
                # 写入数据库 (即使 combined_df.empty 也会创建表结构)
                try:
                    self.db.create_table_from_df(combined_df, table_name, if_exists)
                    results[table_name] = len(combined_df)
                    self.written_tables[table_name] = {
                        "module": module_name,
                        "rows": len(combined_df)
                    }
                except Exception as e:
                    print(f"  ❌ 写入表 {table_name} 失败: {e}")
                    results[table_name] = -1
        
        # 统计
        total_tables = sum(1 for v in results.values() if v > 0)
        total_rows = sum(v for v in results.values() if v > 0)
        
        print(f"\n✅ 模块输出写入完成:")
        print(f"   表数量: {total_tables}")
        print(f"   总行数: {total_rows}")
        
        return results
    
    def _clean_name(self, name: str) -> str:
        """清理名称"""
        clean = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        if clean and clean[0].isdigit():
            clean = "_" + clean
        return clean.lower()
    
    def get_written_tables(self) -> Dict[str, Dict]:
        """获取已写入的表信息"""
        return self.written_tables
    
    def print_summary(self):
        """打印写入汇总"""
        if not self.written_tables:
            print("暂无写入记录")
            return
        
        print("\n📋 已写入表汇总:")
        print("-" * 80)
        print(f"{'表名':<50} {'行数':>10} {'来源':<20}")
        print("-" * 80)
        
        for table_name, info in self.written_tables.items():
            source = info.get('module', info.get('source', 'unknown'))[:20]
            rows = info.get('rows', 0)
            print(f"{table_name:<50} {rows:>10} {source:<20}")
        
        print("-" * 80)
        print(f"共 {len(self.written_tables)} 个表")


def write_run_data_to_db(
    run_output_dir: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456"
) -> bool:
    """
    将运行输出数据写入数据库
    
    Args:
        run_output_dir: 运行输出目录
        db_host: 数据库主机
        db_port: 数据库端口
        db_name: 数据库名称
        db_user: 用户名
        db_password: 密码
    
    Returns:
        bool: 是否成功
    """
    # 创建数据库连接
    db = DatabaseConnection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    # 测试连接
    conn_result = db.test_connection()
    if not conn_result["success"]:
        print(f"❌ 数据库连接失败: {conn_result['message']}")
        return False
    
    # 创建写入器并执行
    writer = ModuleDataWriter(db)
    
    try:
        writer.write_all_modules(run_output_dir)
        writer.print_summary()
        return True
    except Exception as e:
        print(f"❌ 写入失败: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        write_run_data_to_db(sys.argv[1])
    else:
        print("用法: python module_data_writer.py <run_output_dir>")
