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

from psycopg import sql
from .db_connection import DatabaseConnection
from .table_schemas import MODULE6_OUTPUT_SCHEMAS, get_columns


class ModuleDataWriter:
    """模块数据写入器"""
    
    def __init__(self, db: DatabaseConnection, config_name: str = None):
        """
        初始化模块数据写入器
        
        参数：
            db: 数据库连接实例
            config_name: 配置名称（如 BC_S5, BC_S9），用于区分不同配置的数据
        """
        self.db = db
        self.config_name = config_name
        self.written_tables: Dict[str, Dict] = {}
    
    def truncate_output_tables(self, run_id: str = None) -> int:
        """
        删除指定 run_id 的所有模块输出表数据。

        使用 DELETE WHERE run_id = %s 而非 TRUNCATE，确保不同场景（BC/OC）
        的历史数据互不干扰。若 run_id 为 None，则跳过删除并发出警告。

        参数：
            run_id: 本次运行的唯一标识符（必须提供）

        返回：
            int: 成功删除数据的表数量
        """
        if not run_id:
            print("\n[WARN] truncate_output_tables: run_id 为空，跳过删除操作")
            return 0

        print("\n" + "=" * 60)
        print(f"[DEL]  删除 run_id={run_id} 的数据库输出表数据")
        print("=" * 60)

        # 定义所有需要按 run_id 删除的输出表
        output_tables = [
            # Module1输出表
            'module1_output_orderlog',
            'module1_output_shipmentlog',
            'module1_output_cutlog',
            'module1_output_supplydemandlog',
            'module1_output_summary',
            # Module3输出表
            'module3_output_netdemand',
            # Module4输出表
            'module4_output_productionplan',
            'module4_output_capacityexceed',
            'module4_output_validation',
            'module4_output_changeoverlog',
            # Module5输出表
            'module5_output_deploymentplan',
            'module5_output_unfulfilledlog',
            'module5_output_stockonhandlog',
            'module5_output_validation',
            # Module6输出表
            'module6_output_deliveryplan',
            'module6_output_vehiclelog',
            'module6_output_truckusagelog',
            'module6_output_unsatisfiedmdqlog',
            'module6_output_validationlog',
            'module6_output_bypassrulehitlog',
            # 汇总输出表
            'summary_output_ordershipmentcutsummary',
            'summary_output_fullchangeoverlog',
            'summary_output_fullcapacityexceed',
            'summary_output_fullproductionplan',
            'summary_output_fulldeploymentplan',
            'summary_output_fulldeliveryplan',
            'summary_output_fulltruckusage',
            # 编排器状态表
            'orchestrator_unrestricted_inventory',
            'orchestrator_open_deployment',
            'orchestrator_open_deployment_pastdue_cleanup',
            'orchestrator_planning_intransit',
            'orchestrator_space_quota',
            'orchestrator_delivery_gr',
            'orchestrator_production_gr',
            'orchestrator_production_plan_backlog',
            'orchestrator_shipment_log',
            'orchestrator_delivery_shipment_log',
            'orchestrator_inventory_change_log',
            'orchestrator_daily_logs',
        ]

        deleted_count = 0

    # 所有输出表的完整列表（类级别常量，供预建表和清理复用）
    ALL_OUTPUT_TABLES = [
        # Module1输出表
        'module1_output_orderlog',
        'module1_output_shipmentlog',
        'module1_output_cutlog',
        'module1_output_supplydemandlog',
        'module1_output_summary',
        # Module3输出表
        'module3_output_netdemand',
        # Module4输出表
        'module4_output_productionplan',
        'module4_output_capacityexceed',
        'module4_output_validation',
        'module4_output_changeoverlog',
        # Module5输出表
        'module5_output_deploymentplan',
        'module5_output_unfulfilledlog',
        'module5_output_stockonhandlog',
        'module5_output_validation',
        # Module6输出表
        'module6_output_deliveryplan',
        'module6_output_vehiclelog',
        'module6_output_truckusagelog',
        'module6_output_unsatisfiedmdqlog',
        'module6_output_validationlog',
        'module6_output_bypassrulehitlog',
        # 汇总输出表
        'summary_output_ordershipmentcutsummary',
        'summary_output_fullchangeoverlog',
        'summary_output_fullcapacityexceed',
        'summary_output_fullproductionplan',
        'summary_output_fulldeploymentplan',
        'summary_output_fulldeliveryplan',
        'summary_output_fulltruckusage',
        # 编排器状态表
        'orchestrator_unrestricted_inventory',
        'orchestrator_open_deployment',
        'orchestrator_open_deployment_pastdue_cleanup',
        'orchestrator_planning_intransit',
        'orchestrator_space_quota',
        'orchestrator_delivery_gr',
        'orchestrator_production_gr',
        'orchestrator_production_plan_backlog',
        'orchestrator_shipment_log',
        'orchestrator_delivery_shipment_log',
        'orchestrator_inventory_change_log',
        'orchestrator_daily_logs',
    ]

    def ensure_output_tables_exist(self) -> int:
        """在仿真开始前预建所有输出表的空结构。

        每张表至少包含公共列 run_id, sim_date, config_name, db_write_time，
        后续 COPY 写入时若 DataFrame 带有更多列，会自动 ALTER TABLE ADD COLUMN。

        返回：
            int: 新创建的表数量
        """
        common_columns = [
            '"run_id" TEXT',
            '"sim_date" TEXT',
            '"config_name" TEXT',
            '"db_write_time" TIMESTAMP',
        ]
        cols_sql = ", ".join(common_columns)
        created = 0
        with self.db.get_cursor() as cur:
            for table_name in self.ALL_OUTPUT_TABLES:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = %s)",
                    (table_name,),
                )
                if not cur.fetchone()[0]:
                    cur.execute(f'CREATE TABLE "{table_name}" ({cols_sql})')
                    created += 1
        if created > 0:
            print(f"  📋 预建输出表: 新建 {created} 张（共 {len(self.ALL_OUTPUT_TABLES)} 张）")
        else:
            print(f"  ✅ 所有 {len(self.ALL_OUTPUT_TABLES)} 张输出表已存在")
        return created

        try:
            with self.db.get_cursor() as cursor:
                for table_name in output_tables:
                    try:
                        # 检查表是否存在
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_schema = 'public'
                                AND table_name = %s
                            )
                        """, (table_name,))
                        exists = cursor.fetchone()[0]

                        if exists:
                            cursor.execute(
                                f'DELETE FROM "{table_name}" WHERE run_id = %s',
                                (run_id,)
                            )
                            deleted_count += 1
                            print(f"  [v] 已删除 run_id={run_id} 的数据: {table_name}")
                    except Exception as e:
                        print(f"  [x] 删除失败 {table_name}: {e}")

            print(f"\n[OK] 共处理 {deleted_count} 个表（按 run_id={run_id} 删除）")

        except Exception as e:
            print(f"\n[ERROR] 删除表数据时出错: {e}")

        return deleted_count

    def prepare_orchestrator_day_dataframes(
        self,
        orchestrator_dir: str,
        run_id: str | None,
        sim_date: str,
    ) -> Dict[str, tuple[pd.DataFrame, List[str]]]:
        """预处理单个仿真日的 Orchestrator CSV 数据，供数据库 COPY 使用。

        这样可以让 Orchestrator 的当日状态写入与模块批量写入在同一事务内提交，
        保证批次结果的一致性与可回滚性。
        """
        orch_path = Path(orchestrator_dir)
        if not orch_path.exists():
            return {}

        date_key = pd.to_datetime(sim_date).strftime("%Y%m%d")
        file_patterns = [
            "unrestricted_inventory",
            "open_deployment",
            "open_deployment_pastdue_cleanup",
            "planning_intransit",
            "space_quota",
            "delivery_gr",
            "production_gr",
            "production_plan_backlog",
            "shipment_log",
            "delivery_shipment_log",
            "inventory_change_log",
            "daily_logs",
        ]

        prepared: Dict[str, tuple[pd.DataFrame, List[str]]] = {}
        for base_name in file_patterns:
            csv_file = orch_path / f"{base_name}_{date_key}.csv"
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            df = df.copy()
            df["file_date"] = date_key
            df["sim_date"] = pd.to_datetime(sim_date).strftime("%Y-%m-%d")
            if run_id:
                df["run_id"] = run_id
            if self.config_name:
                df["config_name"] = self.config_name
            df["db_write_time"] = datetime.now()

            clean_columns = [self._clean_name(str(col)) for col in df.columns]
            df.columns = clean_columns
            prepared[f"orchestrator_{base_name}"] = (df, clean_columns)

        return prepared
    
    def write_summary_only(
        self,
        output_dir: str,
        run_id: str = None,
        if_exists: str = "replace"
    ) -> Dict[str, int]:
        """
        【优化版】只写入Summary文件和Orchestrator数据
        
        这个方法专门用于仿真结束后的批量写入，避免每天重复写入模块输出。
        Summary文件已包含所有模块的完整汇总数据。
        
        优化效果：
        - 减少写入次数：从5天×6模块×多Sheet → 8个Summary文件 + 10个Orchestrator文件
        - 避免重复数据：不再写入每天的模块输出（summary已包含汇总）
        - 预计提升：写入时间从~54秒降至~15秒
        
        参数：
            output_dir: 运行输出根目录
            run_id: 运行ID
            if_exists: 如果表存在的处理方式 ('replace'推荐，确保干净数据)
        
        返回：
            dict: 每个表的写入行数
        """
        import time
        start_time = time.time()
        base_path = Path(output_dir)
        
        if not base_path.exists():
            raise FileNotFoundError(f"输出目录不存在: {output_dir}")
        
        if run_id is None:
            run_id = base_path.name
        
        print("\n" + "=" * 60)
        print("[OUT] 【优化模式】批量写入Summary和Orchestrator数据")
        print(f"运行目录: {output_dir}")
        print(f"运行ID: {run_id}")
        print("=" * 60)
        
        results = {}
        
        # 1. 写入Summary数据（最重要）
        summary_dir = base_path / "summary"
        if summary_dir.exists():
            print("\n[DATA] 写入Summary汇总数据...")
            summary_results = self._write_summary_files_fast(str(summary_dir), run_id, if_exists)
            results.update(summary_results)
        
        # 2. 写入Orchestrator状态数据
        orch_dir = base_path / "orchestrator"
        if orch_dir.exists():
            print("\n[DIR] 写入Orchestrator状态数据...")
            orch_results = self.write_orchestrator_data(str(orch_dir), run_id=run_id, if_exists=if_exists)
            results.update(orch_results)
        
        elapsed = time.time() - start_time
        total_tables = len([v for v in results.values() if isinstance(v, int) and v >= 0])
        total_rows = sum(v for v in results.values() if isinstance(v, int) and v > 0)
        
        print("\n" + "=" * 60)
        print(f"[OK] 优化写入完成!")
        print(f"   表数量: {total_tables}")
        print(f"   总行数: {total_rows:,}")
        print(f"   耗时: {elapsed:.2f}秒")
        print("=" * 60)
        
        return results
    
    def _write_summary_files_fast(
        self,
        summary_dir: str,
        run_id: str = None,
        if_exists: str = "replace"
    ) -> Dict[str, int]:
        """快速写入Summary文件"""
        summary_path = Path(summary_dir)
        results = {}
        
        # 汇总文件到表名的映射
        file_table_mapping = {
            "full_changeover_report.xlsx": "summary_output_fullchangeoverlog",
            "full_delivery_plan_report.xlsx": "summary_output_fulldeliveryplan",
            "full_deployment_plan_report.xlsx": "summary_output_fulldeploymentplan",
            "full_exceed_capacity_report.xlsx": "summary_output_fullcapacityexceed",
            "full_order_shipment_cut_report.xlsx": "summary_output_ordershipmentcutsummary",
            "full_production_plan_report.xlsx": "summary_output_fullproductionplan",
            "full_truck_usage_report.xlsx": "summary_output_fulltruckusage",
            "historical_inventory_record.csv": "summary_historical_inventory_record",
        }
        
        for file_name, table_name in file_table_mapping.items():
            file_path = summary_path / file_name
            if file_path.exists():
                try:
                    # 读取数据
                    if file_name.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    
                    # 添加run_id列
                    if run_id and not df.empty:
                        df['run_id'] = run_id
                    
                    # 写入数据库（传入config_name）
                    self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
                    results[table_name] = len(df)
                    self.written_tables[table_name] = {
                        "source": str(file_path),
                        "module": "summary",
                        "rows": len(df)
                    }
                    print(f"  [OK] {table_name}: {len(df):,} 行")
                except Exception as e:
                    print(f"  [ERROR] {file_name}: {e}")
                    results[table_name] = -1
        
        return results
    
    def write_module_output(
        self,
        module_name: str,
        output_dir: str,
        run_id: str = None,
        sim_date: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """
        写入单个模块的输出数据
        
        参数：
            module_name: 模块名称 (module1, module3, module4, module5, module6)
            output_dir: 模块输出目录
            run_id: 运行ID（用于区分不同运行）
            sim_date: 仿真日期（格式：YYYYMMDD，如 20251006）
            if_exists: 如果表存在的处理方式
                - 'replace': 第一个文件替换表，后续文件追加
                - 'append': 所有文件追加
        
        返回：
            dict: 每个输出文件的写入行数
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"[WARN]模块输出目录不存在: {output_dir}")
            return {}
        
        results = {}
        print(f"\n[DIR] 写入 {module_name} 输出数据...")
        
        # 跟踪每个表是否已经被写入过（用于 replace 模式）
        # 第一次写入用 replace，后续用 append
        tables_written: Dict[str, bool] = {}
        
        # 查找所有Excel文件并按日期排序
        excel_files = sorted(output_path.glob("*.xlsx"))
        
        for excel_file in excel_files:
            try:
                file_results = self._write_excel_file(
                    excel_file, 
                    module_name, 
                    run_id,
                    sim_date,
                    if_exists,
                    tables_written  # 传递已写入表的跟踪字典
                )
                results[excel_file.name] = file_results
            except Exception as e:
                print(f"  [ERROR] 文件写入失败 [{excel_file.name}]: {e}")
                results[excel_file.name] = {"error": str(e)}
        
        # 查找所有CSV文件并按日期排序
        csv_files = sorted(output_path.glob("*.csv"))
        
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
                
                # 添加仿真日期列（从文件名提取）
                actual_sim_date = sim_date if sim_date else file_date
                if actual_sim_date:
                    if df.empty:
                        df['sim_date'] = pd.Series(dtype='string')
                    else:
                        df['sim_date'] = actual_sim_date
                
                if run_id:
                    if df.empty:
                        df['run_id'] = pd.Series(dtype='string')
                    else:
                        df['run_id'] = run_id
                
                # 确定实际的 if_exists 模式
                # 如果是 replace 模式，第一次写入用 replace，后续用 append
                actual_if_exists = if_exists
                if if_exists == 'replace':
                    if table_name in tables_written:
                        actual_if_exists = 'append'
                    else:
                        tables_written[table_name] = True
                        
                self.db.create_table_from_df(df, table_name, actual_if_exists, config_name=self.config_name)
                results[csv_file.name] = len(df)
                
                # 更新或累加行数
                if table_name in self.written_tables:
                    self.written_tables[table_name]['rows'] += len(df)
                else:
                    self.written_tables[table_name] = {
                        "source": str(csv_file),
                        "module": module_name,
                        "rows": len(df)
                    }
            except Exception as e:
                print(f"  [ERROR] CSV文件写入失败 [{csv_file.name}]: {e}")
                results[csv_file.name] = {"error": str(e)}
        
        return results
    
    def _write_excel_file(
        self,
        excel_path: Path,
        module_name: str,
        run_id: str = None,
        sim_date: str = None,
        if_exists: str = "append",
        tables_written: Dict[str, bool] = None
    ) -> Dict[str, int]:
        """写入单个Excel文件的所有sheet
        
        文件名中的日期（如 Module1Output_20251006.xlsx）会被提取为 file_date 列，
        sim_date 参数用于标识当前仿真日期。
        
        对于空表，如果有预定义的列名结构则使用，否则保持空表。
        
        参数：
            tables_written: 已写入表的字典，用于跟踪 replace 模式下哪些表已经被写入
        """
        import re
        results = {}
        
        if tables_written is None:
            tables_written = {}
        
        xl = pd.ExcelFile(excel_path)
        
        # 从文件名中提取日期（格式：xxx_YYYYMMDD.xlsx）
        date_match = re.search(r'_(\d{8})(?:\.xlsx)?$', excel_path.stem)
        file_date = date_match.group(1) if date_match else None
        
        # 使用简化的表名格式: {module_name}_output_{sheet_name}
        # 这样与内存模式的表名保持一致
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name)
                
                # 构建表名（简化格式，与内存模式一致）
                table_name = f"{module_name}_output_{self._clean_name(sheet_name)}"
                
                # 如果DataFrame为空，尝试使用预定义的列名（如果有的话）
                if df.empty:
                    schema_columns = get_columns(module_name, sheet_name)
                    if schema_columns:  # 只有定义了列名时才使用
                        df = pd.DataFrame(columns=schema_columns)
                    # 否则保持空DataFrame（与基线保持一致）
                
                # 添加日期列（从文件名提取）
                if file_date:
                    if df.empty:
                        df['file_date'] = pd.Series(dtype='string')
                    else:
                        df['file_date'] = file_date
                
                # 添加仿真日期列（从参数传入，如果没有则使用文件日期）
                actual_sim_date = sim_date if sim_date else file_date
                if actual_sim_date:
                    if df.empty:
                        df['sim_date'] = pd.Series(dtype='string')
                    else:
                        df['sim_date'] = actual_sim_date
                
                # 添加run_id列
                if run_id:
                    df['run_id'] = run_id
                
                # 确定实际的 if_exists 模式
                # 如果是 replace 模式，第一次写入用 replace，后续用 append
                actual_if_exists = if_exists
                if if_exists == 'replace':
                    if table_name in tables_written:
                        actual_if_exists = 'append'
                    else:
                        tables_written[table_name] = True
                
                # 写入数据库 (即使 df.empty 也会创建表结构，传入config_name)
                self.db.create_table_from_df(df, table_name, actual_if_exists, config_name=self.config_name)
                results[sheet_name] = len(df)
                
                # 更新或累加行数
                if table_name in self.written_tables:
                    self.written_tables[table_name]['rows'] += len(df)
                else:
                    self.written_tables[table_name] = {
                        "source": str(excel_path),
                        "sheet": sheet_name,
                        "module": module_name,
                        "rows": len(df)
                    }
                
            except Exception as e:
                print(f"    [WARN] Sheet [{sheet_name}] 写入失败: {e}")
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
        
        参数：
            run_output_dir: 运行输出根目录
            run_id: 运行ID
            if_exists: 如果表存在的处理方式
        
        返回：
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
                    module_name=module, 
                    output_dir=str(module_dir), 
                    run_id=run_id,
                    sim_date=None,  # sim_date 从文件名中提取
                    if_exists=if_exists
                )
                all_results[module] = results
            else:
                print(f"[WARN]模块目录不存在: {module}")
        
        elapsed = time.time() - start_time
        
        # 统计
        total_tables = len(self.written_tables)
        total_rows = sum(info.get('rows', 0) for info in self.written_tables.values())
        
        print("\n" + "=" * 60)
        print(f"[DATA] 写入汇总:")
        print(f"   表数量: {total_tables}")
        print(f"   总行数: {total_rows}")
        print(f"   耗时: {elapsed:.2f}s")
        print("=" * 60)
        
        return all_results
    
    def write_orchestrator_data(
        self,
        orchestrator_dir: str,
        run_id: str = None,
        sim_date: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """
        写入Orchestrator输出的CSV数据
        
        参数：
            orchestrator_dir: Orchestrator输出目录
            run_id: 运行ID
            sim_date: 仿真日期（格式：YYYYMMDD）
            if_exists: 如果表存在的处理方式
        
        返回：
            dict: 每个文件的写入行数
        """
        orch_path = Path(orchestrator_dir)
        if not orch_path.exists():
            print(f"[WARN] Orchestrator目录不存在: {orchestrator_dir}")
            return {}
        
        results = {}
        print(f"\n[INFO] 写入Orchestrator数据...")
        
        # 定义Orchestrator输出文件类型
        file_patterns = [
            "unrestricted_inventory_*.csv",
            "open_deployment_*.csv",
            "open_deployment_pastdue_cleanup_*.csv",
            "planning_intransit_*.csv",
            "space_quota_*.csv",
            "delivery_gr_*.csv",
            "production_gr_*.csv",
            "production_plan_backlog_*.csv",
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
                            df['sim_date'] = pd.Series(dtype='string')
                            if run_id:
                                df['run_id'] = pd.Series(dtype='string')
                        else:
                            df['file_date'] = date_part
                            # [FIX-风险6] 统一 sim_date 格式为 YYYY-MM-DD
                            # 文件名日期可能为 YYYYMMDD，需要转换
                            if sim_date:
                                df['sim_date'] = sim_date
                            elif len(date_part) == 8 and date_part.isdigit():
                                # YYYYMMDD -> YYYY-MM-DD
                                df['sim_date'] = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            else:
                                df['sim_date'] = date_part
                            if run_id:
                                df['run_id'] = run_id
                        dfs.append(df)
                    except Exception as e:
                        print(f"    [WARN] 读取失败 [{csv_file.name}]: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    self.db.create_table_from_df(combined_df, table_name, if_exists, config_name=self.config_name)
                    results[base_table_name] = len(combined_df)
                    self.written_tables[table_name] = {
                        "source": str(orch_path),
                        "file_count": len(csv_files),
                        "rows": len(combined_df)
                    }
        
        return results
    
    def delete_batch_data(self, run_id: str, batch_start_date: str) -> None:
        """
        删除指定批次起始日期及之后的模块输出数据（幂等保障）。
        用于续跑时清理上次中断批次的残留数据，再重写。

        [FIX-风险3] 区分可跳过异常与真正错误
        [FIX-风险4] 覆盖 汇总与编排器 表
        [FIX-风险5] 添加 run_id 过滤

        参数：
            run_id: 运行ID，用于按 run_id 过滤删除
            batch_start_date: 批次第一天 (YYYY-MM-DD)，删除 sim_date >= 此值的行
        """
        # [FIX-风险4] 完整表列表，与 truncate_output_tables 保持一致
        output_tables = [
            # 模块输出表
            'module1_output_orderlog',
            'module1_output_shipmentlog',
            'module1_output_cutlog',
            'module1_output_supplydemandlog',
            'module1_output_summary',
            'module3_output_netdemand',
            'module4_output_productionplan',
            'module4_output_capacityexceed',
            'module4_output_validation',
            'module4_output_changeoverlog',
            'module5_output_deploymentplan',
            'module5_output_unfulfilledlog',
            'module5_output_stockonhandlog',
            'module5_output_validation',
            'module6_output_deliveryplan',
            'module6_output_vehiclelog',
            'module6_output_truckusagelog',
            'module6_output_unsatisfiedmdqlog',
            'module6_output_validationlog',
            'module6_output_bypassrulehitlog',
            # 汇总输出表
            'summary_output_ordershipmentcutsummary',
            'summary_output_fullchangeoverlog',
            'summary_output_fullcapacityexceed',
            'summary_output_fullproductionplan',
            'summary_output_fulldeploymentplan',
            'summary_output_fulldeliveryplan',
            'summary_output_fulltruckusage',
            # 编排器状态表
            'orchestrator_unrestricted_inventory',
            'orchestrator_open_deployment',
            'orchestrator_open_deployment_pastdue_cleanup',
            'orchestrator_planning_intransit',
            'orchestrator_space_quota',
            'orchestrator_delivery_gr',
            'orchestrator_production_gr',
            'orchestrator_production_plan_backlog',
            'orchestrator_shipment_log',
            'orchestrator_delivery_shipment_log',
            'orchestrator_inventory_change_log',
            'orchestrator_daily_logs',
        ]
        deleted_total = 0
        for table_name in output_tables:
            try:
                # 检查表是否存在
                rows = self.db.execute_query(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = %s)",
                    (table_name,)
                )
                if not rows or not rows[0][0]:
                    continue
                # 检查是否有 sim_date 列
                sim_date_check = self.db.execute_query(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = 'sim_date')",
                    (table_name,)
                )
                if not sim_date_check or not sim_date_check[0][0]:
                    continue
                # [FIX-风险5] 检查是否有 run_id 列，按 sim_date + run_id 删除
                run_id_check = self.db.execute_query(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = 'run_id')",
                    (table_name,)
                )
                has_run_id = run_id_check and run_id_check[0][0]
                if has_run_id and run_id:
                    self.db.execute_non_query(
                        sql.SQL('DELETE FROM {} WHERE sim_date >= %s AND run_id = %s').format(
                            sql.Identifier(table_name)
                        ),
                        (batch_start_date, run_id)
                    )
                else:
                    self.db.execute_non_query(
                        sql.SQL('DELETE FROM {} WHERE sim_date >= %s').format(
                            sql.Identifier(table_name)
                        ),
                        (batch_start_date,)
                    )
                deleted_total += 1
            except Exception as e:
                # [FIX-风险3] 区分可跳过异常与真正错误
                err_str = str(e).lower()
                if 'does not exist' in err_str or 'column' in err_str:
                    continue  # 表或列不存在，合理跳过
                else:
                    import logging
                    logging.error(f"delete_batch_data: 表 {table_name} 删除失败: {e}")
                    raise  # 真正的错误（连接中断、锁超时等），上报
        if deleted_total > 0:
            print(f"  🗑️  已清理批次 {batch_start_date} 起的旧数据（{deleted_total} 张表）")

    def _filter_module1_orders_for_day(
        self,
        df: pd.DataFrame,
        day_sim_date: str | None,
    ) -> pd.DataFrame:
        """仅保留当前仿真日新生成的 Module1 订单。

        集成模式下的 `orders_df` 为累计口径，后续日期会包含前序日期已生成的订单。
        但数据库对账与本地 Excel 比对都按行级 `simulation_date` 视为“当日 `OrderLog`”，
        因此在写库前必须先将累计结果切回当前仿真日口径，再补充 `sim_date`。
        """
        if df.empty or not day_sim_date or 'simulation_date' not in df.columns:
            return df

        config_name = (self.config_name or '').upper()
        if not config_name.startswith('OC'):
            return df

        filtered = df.copy()
        sim_dates = pd.to_datetime(filtered['simulation_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        return filtered[sim_dates == day_sim_date].copy()

    def prepare_batch_dataframes(
        self,
        all_results: Dict[str, Any],
        run_id: str = None,
    ) -> Dict[str, tuple]:
        """
        预处理批次数据为可写入的 DataFrames（不涉及 DB 操作）。

        用于 _flush_batch_to_db 在事务外预处理数据，
        然后在事务内通过 _atomic_copy_batch 写入。

        参数：
            all_results: 模块运行结果字典
            run_id: 运行ID

        返回：
            dict: {table_name: (combined_df, clean_columns_list)}
        """
        prepared = {}

        module_df_mapping = {
            'module1': {
                'orders_df': 'module1_output_orderlog',
                'shipment_df': 'module1_output_shipmentlog',
                'cut_df': 'module1_output_cutlog',
                'supply_demand_df': 'module1_output_supplydemandlog',
                'summary_df': 'module1_output_summary',
            },
            'module3': {
                'net_demand_df': 'module3_output_netdemand',
            },
            'module4': {
                'production_df': 'module4_output_productionplan',
                'exceed_log': 'module4_output_capacityexceed',
                'issues_df': 'module4_output_validation',
                'changeover_log': 'module4_output_changeoverlog',
            },
            'module5': {
                'deployment_plan': 'module5_output_deploymentplan',
                'unfulfilled_log': 'module5_output_unfulfilledlog',
                'stock_on_hand_log': 'module5_output_stockonhandlog',
                'validation_log': 'module5_output_validation',
            },
            'module6': {
                'delivery_plan': 'module6_output_deliveryplan',
                'vehicle_log': 'module6_output_vehiclelog',
                'truck_usage': 'module6_output_truckusagelog',
                'unsatisfied_log': 'module6_output_unsatisfiedmdqlog',
                'validation_log': 'module6_output_validationlog',
                'bypass_log': 'module6_output_bypassrulehitlog',
            },
        }

        for module_name, df_mapping in module_df_mapping.items():
            module_results = all_results.get(module_name, [])
            if not module_results:
                continue

            table_data = {tn: [] for tn in df_mapping.values()}

            for day_result in module_results:
                if not isinstance(day_result, dict):
                    continue
                # 提取仿真日期
                day_sim_date = None
                if 'simulation_date' in day_result:
                    sim_date_obj = day_result['simulation_date']
                    if hasattr(sim_date_obj, 'strftime'):
                        day_sim_date = sim_date_obj.strftime('%Y-%m-%d')

                for df_key, table_name in df_mapping.items():
                    df = day_result.get(df_key)
                    if df is not None and isinstance(df, pd.DataFrame):
                        df = df.copy()
                        if module_name == 'module1' and df_key == 'orders_df':
                            df = self._filter_module1_orders_for_day(df, day_sim_date)
                        if day_sim_date:
                            if df.empty:
                                df['sim_date'] = pd.Series(dtype='string')
                            else:
                                df['sim_date'] = day_sim_date
                        else:
                            df['sim_date'] = pd.Series(dtype='string') if df.empty else None
                        table_data[table_name].append(df)

            for table_name, dfs in table_data.items():
                if not dfs:
                    continue
                combined_df = pd.concat(dfs, ignore_index=True)
                if 'sim_date' not in combined_df.columns:
                    combined_df['sim_date'] = pd.Series(dtype='string')
                if run_id:
                    if combined_df.empty:
                        combined_df['run_id'] = pd.Series(dtype='string')
                    else:
                        combined_df['run_id'] = run_id
                # 添加 config_name
                if self.config_name and not combined_df.empty:
                    combined_df['config_name'] = self.config_name
                # 添加写入时间
                from datetime import datetime
                if not combined_df.empty:
                    combined_df['db_write_time'] = datetime.now()
                else:
                    combined_df['db_write_time'] = pd.Series(dtype='datetime64[ns]')
                # 清理列名
                clean_columns = [self._clean_name(str(col)) for col in combined_df.columns]
                combined_df.columns = clean_columns
                prepared[table_name] = (combined_df, clean_columns)

        return prepared

    def write_module_results_from_dict(
        self,
        all_results: Dict[str, Any],
        run_id: str = None,
        sim_date: str = None,
        if_exists: str = "append",
        truncate_first: bool = True
    ) -> Dict[str, int]:
        """
        从内存中的模块结果字典直接写入数据库
        
        这个方法用于数据库模式，绕过文件系统直接将DataFrame写入数据库。
        
        参数：
            all_results: 模块运行结果字典，结构为:
                {
                    'module1': [{'orders_df': df, 'shipment_df': df, 'cut_df': df, ...}, ...],
                    'module3': [{'net_demand_df': df}, ...],
                    'module4': [{'production_df': df}, ...],
                    'module5': [{'deployment_plan': df, 'stock_log': df, ...}, ...],
                    'module6': [{'delivery_plan': df, 'truck_usage': df, ...}, ...]
                }
            run_id: 运行ID
            sim_date: 仿真日期（格式：YYYYMMDD）
            if_exists: 如果表存在的处理方式
            truncate_first: 是否在写入前先清空所有输出表（默认True）
        
        返回：
            dict: 每个表的写入行数
        """
        results = {}
        
        # [DEL] 在写入前按 run_id 删除当前运行的旧数据（不影响其他 run_id 数据）
        if truncate_first:
            self.truncate_output_tables(run_id=run_id)
        
        print("\n" + "=" * 60)
        print("[OUT] 从内存写入模块输出到数据库")
        print(f"运行ID: {run_id}")
        print("=" * 60)
        
        # 定义模块输出的DataFrame到表名的映射
        # 键名必须与模块返回的字典键名一致
        module_df_mapping = {
            'module1': {
                # `orders_df` 是当天新增订单（`today_orders_df`），与比对脚本从 xlsx 过滤后的口径一致
                # `simulation_date` 与当天口径一致
                'orders_df': 'module1_output_orderlog',
                'shipment_df': 'module1_output_shipmentlog',
                'cut_df': 'module1_output_cutlog',
                'supply_demand_df': 'module1_output_supplydemandlog',
                'summary_df': 'module1_output_summary',  # 添加Summary映射
            },
            'module3': {
                'net_demand_df': 'module3_output_netdemand',
            },
            'module4': {
                'production_df': 'module4_output_productionplan',
                'exceed_log': 'module4_output_capacityexceed',
                'issues_df': 'module4_output_validation',
                'changeover_log': 'module4_output_changeoverlog',
            },
            'module5': {
                'deployment_plan': 'module5_output_deploymentplan',
                'unfulfilled_log': 'module5_output_unfulfilledlog',
                'stock_on_hand_log': 'module5_output_stockonhandlog',
                'validation_log': 'module5_output_validation',
            },
            'module6': {
                'delivery_plan': 'module6_output_deliveryplan',
                'vehicle_log': 'module6_output_vehiclelog',
                'truck_usage': 'module6_output_truckusagelog',
                'unsatisfied_log': 'module6_output_unsatisfiedmdqlog',
                'validation_log': 'module6_output_validationlog',
                'bypass_log': 'module6_output_bypassrulehitlog',
            },
        }
        
        for module_name, df_mapping in module_df_mapping.items():
            module_results = all_results.get(module_name, [])
            
            # [DEBUG] DEBUG: 打印每个模块的结果详情
            print(f"\n[DEBUG] DEBUG: {module_name} 结果检查:")
            print(f"   module_results 类型: {type(module_results)}")
            print(f"   module_results 长度: {len(module_results) if module_results else 0}")
            if module_results and len(module_results) > 0:
                first_result = module_results[0]
                if isinstance(first_result, dict):
                    print(f"   第一天结果的键: {list(first_result.keys())}")
                    print(f"   期望的键 (df_mapping): {list(df_mapping.keys())}")
                    # 检查键是否匹配
                    expected_keys = set(df_mapping.keys())
                    actual_keys = set(first_result.keys())
                    matched = expected_keys & actual_keys
                    missing = expected_keys - actual_keys
                    extra = actual_keys - expected_keys
                    print(f"   匹配的键: {matched}")
                    if missing:
                        print(f"   [WARN] 缺失的键: {missing}")
                    if extra:
                        print(f"   额外的键: {extra}")
            
            if not module_results:
                print(f"   [WARN] {module_name} 没有结果，跳过")
                continue
            
            print(f"\n[DIR] 写入 {module_name} 输出...")
            
            # 收集同一个表的所有数据
            table_data = {table_name: [] for table_name in df_mapping.values()}
            
            for day_result in module_results:
                if not isinstance(day_result, dict):
                    continue
                
                # 提取当天的仿真日期
                day_sim_date = None
                if 'simulation_date' in day_result:
                    sim_date_obj = day_result['simulation_date']
                    if hasattr(sim_date_obj, 'strftime'):
                        day_sim_date = sim_date_obj.strftime('%Y-%m-%d')
                
                for df_key, table_name in df_mapping.items():
                    df = day_result.get(df_key)
                    # [DEBUG] DEBUG: 打印每个 DataFrame 键的检查结果
                    if df is not None:
                        if isinstance(df, pd.DataFrame):
                            print(f"      [v] {df_key} -> {table_name}: DataFrame shape={df.shape}")
                        else:
                            print(f"      [x] {df_key} -> {table_name}: 不是DataFrame, type={type(df).__name__}")
                    else:
                        print(f"      [x] {df_key} -> {table_name}: None")
                    if df is not None and isinstance(df, pd.DataFrame):
                        # 为每一天的数据添加 sim_date（包括空 DataFrame）
                        df = df.copy()  # 避免修改原始数据
                        if module_name == 'module1' and df_key == 'orders_df':
                            df = self._filter_module1_orders_for_day(df, day_sim_date)
                        if day_sim_date:
                            if df.empty:
                                df['sim_date'] = pd.Series(dtype='string')
                            else:
                                df['sim_date'] = day_sim_date
                        else:
                            # 即使没有日期，也要添加空的 sim_date 列
                            df['sim_date'] = pd.Series(dtype='string') if df.empty else None
                        table_data[table_name].append(df)
            
            # 写入每个表
            for table_name, dfs in table_data.items():
                if not dfs:
                    # [FIX] FIX: 即使没有数据，也创建空表结构
                    # 创建带有 sim_date 和 run_id 列的空 DataFrame
                    combined_df = pd.DataFrame()
                    combined_df['sim_date'] = pd.Series(dtype='string')
                    if run_id:
                        combined_df['run_id'] = pd.Series(dtype='string')
                    # 写入空表到数据库
                    try:
                        self.db.create_table_from_df(combined_df, table_name, if_exists, config_name=self.config_name)
                        results[table_name] = 0
                        self.written_tables[table_name] = {
                            "module": module_name,
                            "rows": 0
                        }
                        print(f"  [OK] 创建空表 {table_name}")
                    except Exception as e:
                        print(f"  [ERROR] 创建空表 {table_name} 失败: {e}")
                        results[table_name] = -1
                    continue
                
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # 确保 sim_date 列存在（即使是空表）
                if 'sim_date' not in combined_df.columns:
                    combined_df['sim_date'] = pd.Series(dtype='string')
                
                # 添加run_id列
                if run_id:
                    if combined_df.empty:
                        combined_df['run_id'] = pd.Series(dtype='string')
                    else:
                        combined_df['run_id'] = run_id
                
                # 写入数据库 (即使 combined_df.empty 也会创建表结构，传入config_name)
                try:
                    self.db.create_table_from_df(combined_df, table_name, if_exists, config_name=self.config_name)
                    results[table_name] = len(combined_df)
                    self.written_tables[table_name] = {
                        "module": module_name,
                        "rows": len(combined_df)
                    }
                except Exception as e:
                    print(f"  [ERROR] 写入表 {table_name} 失败: {e}")
                    results[table_name] = -1
        
        # 统计
        total_tables = sum(1 for v in results.values() if v > 0)
        total_rows = sum(v for v in results.values() if v > 0)
        
        print(f"\n[OK] 模块输出写入完成:")
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
        
        print("\n[INFO] 已写入表汇总:")
        print("-" * 80)
        print(f"{'表名':<50} {'行数':>10} {'来源':<20}")
        print("-" * 80)
        
        for table_name, info in self.written_tables.items():
            source = info.get('module', info.get('source', 'unknown'))[:20]
            rows = info.get('rows', 0)
            print(f"{table_name:<50} {rows:>10} {source:<20}")
        
        print("-" * 80)
        print(f"共 {len(self.written_tables)} 个表")
    
    def generate_summary_reports_from_db(
        self,
        run_id: str = None,
        start_date: str = None,
        end_date: str = None,
        if_exists: str = "replace"
    ) -> Dict[str, int]:
        """
        从数据库中的模块输出表直接生成Summary汇总报告
        
        当使用 --use-db 模式时，模块不会输出xlsx文件，因此无法使用
        SummaryReportGenerator从文件生成汇总报告。此方法直接从已写入
        数据库的模块输出表中聚合数据，生成7个Summary报告表：
        
        1. summary_output_ordershipmentcutsummary - 订单/发货/缺货汇总
        2. summary_output_fullchangeoverlog - 换产日志
        3. summary_output_fullcapacityexceed - 产能超限
        4. summary_output_fullproductionplan - 生产计划
        5. summary_output_fulldeploymentplan - 部署计划
        6. summary_output_fulldeliveryplan - 交付计划
        7. summary_output_fulltruckusage - 卡车使用
        
        参数：
            run_id: 运行ID，用于筛选数据
            start_date: 开始日期 (YYYY-MM-DD)，用于过滤数据
            end_date: 结束日期 (YYYY-MM-DD)，用于过滤数据
            if_exists: 如果表存在的处理方式 ('replace'推荐)
        
        返回：
            dict: 每个表的写入行数
        """
        import time
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("[DATA] 从数据库生成Summary汇总报告")
        print(f"运行ID: {run_id}")
        if start_date and end_date:
            print(f"日期范围: {start_date} 到 {end_date}")
        print("=" * 60)
        
        results = {}
        
        # 转换日期为datetime用于过滤
        end_date_dt = pd.to_datetime(end_date) if end_date else None
        
        # 1. 生成 order_shipment_cut 汇总报告
        try:
            results['summary_output_ordershipmentcutsummary'] = self._generate_order_shipment_cut_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] order_shipment_cut 生成失败: {e}")
            results['summary_output_ordershipmentcutsummary'] = -1
        
        # 2. 生成 changeover 汇总报告
        try:
            results['summary_output_fullchangeoverlog'] = self._generate_changeover_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] changeover 生成失败: {e}")
            results['summary_output_fullchangeoverlog'] = -1
        
        # 3. 生成 capacity_exceed 汇总报告
        try:
            results['summary_output_fullcapacityexceed'] = self._generate_capacity_exceed_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] capacity_exceed 生成失败: {e}")
            results['summary_output_fullcapacityexceed'] = -1
        
        # 4. 生成 production_plan 汇总报告
        try:
            results['summary_output_fullproductionplan'] = self._generate_production_plan_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] production_plan 生成失败: {e}")
            results['summary_output_fullproductionplan'] = -1
        
        # 5. 生成 deployment_plan 汇总报告
        try:
            results['summary_output_fulldeploymentplan'] = self._generate_deployment_plan_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] deployment_plan 生成失败: {e}")
            results['summary_output_fulldeploymentplan'] = -1
        
        # 6. 生成 delivery_plan 汇总报告
        try:
            results['summary_output_fulldeliveryplan'] = self._generate_delivery_plan_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] delivery_plan 生成失败: {e}")
            results['summary_output_fulldeliveryplan'] = -1
        
        # 7. 生成 truck_usage 汇总报告
        try:
            results['summary_output_fulltruckusage'] = self._generate_truck_usage_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            print(f"  [ERROR] truck_usage 生成失败: {e}")
            results['summary_output_fulltruckusage'] = -1
        
        elapsed = time.time() - start_time
        total_tables = sum(1 for v in results.values() if isinstance(v, int) and v >= 0)
        total_rows = sum(v for v in results.values() if isinstance(v, int) and v > 0)
        
        print("\n" + "=" * 60)
        print(f"[OK] Summary汇总报告生成完成!")
        print(f"   表数量: {total_tables}")
        print(f"   总行数: {total_rows:,}")
        print(f"   耗时: {elapsed:.2f}秒")
        print("=" * 60)
        
        return results
    
    def _generate_order_shipment_cut_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成订单/发货/缺货汇总报告"""
        table_name = "summary_output_ordershipmentcutsummary"
        
        try:
            orders_df = self.db.read_table("module1_output_orderlog")
        except:
            orders_df = pd.DataFrame()
        
        try:
            shipments_df = self.db.read_table("module1_output_shipmentlog")
        except:
            shipments_df = pd.DataFrame()
        
        try:
            cuts_df = self.db.read_table("module1_output_cutlog")
        except:
            cuts_df = pd.DataFrame()
        
        if run_id:
            if not orders_df.empty and 'run_id' in orders_df.columns:
                orders_df = orders_df[orders_df['run_id'] == run_id].reset_index(drop=True)
            if not shipments_df.empty and 'run_id' in shipments_df.columns:
                shipments_df = shipments_df[shipments_df['run_id'] == run_id].reset_index(drop=True)
            if not cuts_df.empty and 'run_id' in cuts_df.columns:
                cuts_df = cuts_df[cuts_df['run_id'] == run_id].reset_index(drop=True)
        
        # Dev 从文件路径提取 simulation_date（取路径中第一个8位数字日期，
        # 实际来自 config_name 如 OC_Paste_S1_20251224 → '2025-12-24'）。
        # DB 需要复制此行为：从 config_name 列提取日期，统一覆盖 simulation_date。
        import re
        config_sim_date = None
        for df_ref in [orders_df, shipments_df, cuts_df]:
            if not df_ref.empty and 'config_name' in df_ref.columns:
                sample_config = str(df_ref['config_name'].iloc[0])
                m = re.search(r'(\d{8})', sample_config)
                if m:
                    config_sim_date = pd.to_datetime(m.group(1), format='%Y%m%d').strftime('%Y-%m-%d')
                break
        
        for df_ref in [orders_df, shipments_df, cuts_df]:
            if not df_ref.empty:
                if config_sim_date:
                    # 与 Dev 一致：所有行使用从 config_name 提取的同一 simulation_date
                    df_ref['simulation_date'] = config_sim_date
                elif 'simulation_date' not in df_ref.columns and 'sim_date' in df_ref.columns:
                    df_ref['simulation_date'] = df_ref['sim_date']
        
        # 去重（与 Dev 一致：按 date, material, location, quantity, simulation_date [+ demand_type]）
        if not orders_df.empty:
            dedup_cols = ['date', 'material', 'location', 'quantity', 'simulation_date']
            if 'demand_type' in orders_df.columns:
                dedup_cols.append('demand_type')
            existing_cols = [c for c in dedup_cols if c in orders_df.columns]
            if existing_cols:
                orders_df = orders_df.drop_duplicates(subset=existing_cols, keep='first')
        
        group_cols = ['date', 'material', 'location', 'simulation_date']
        merge_cols = ['date', 'material', 'location', 'simulation_date']
        
        order_agg = pd.DataFrame(columns=merge_cols + ['order_qty'])
        if not orders_df.empty:
            orders_df['date'] = pd.to_datetime(orders_df['date'], errors='coerce')
            existing_group = [c for c in group_cols if c in orders_df.columns]
            order_agg = orders_df.groupby(existing_group, dropna=False).agg(
                {'quantity': 'sum'}
            ).reset_index()
            order_agg.rename(columns={'quantity': 'order_qty'}, inplace=True)
        
        shipment_agg = pd.DataFrame(columns=merge_cols + ['shipment_qty'])
        if not shipments_df.empty:
            shipments_df['date'] = pd.to_datetime(shipments_df['date'], errors='coerce')
            existing_group = [c for c in group_cols if c in shipments_df.columns]
            shipment_agg = shipments_df.groupby(existing_group, dropna=False).agg(
                {'quantity': 'sum'}
            ).reset_index()
            shipment_agg.rename(columns={'quantity': 'shipment_qty'}, inplace=True)
        
        cut_agg = pd.DataFrame(columns=merge_cols + ['cut_qty'])
        if not cuts_df.empty:
            cuts_df['date'] = pd.to_datetime(cuts_df['date'], errors='coerce')
            existing_group = [c for c in group_cols if c in cuts_df.columns]
            cut_agg = cuts_df.groupby(existing_group, dropna=False).agg(
                {'quantity': 'sum'}
            ).reset_index()
            cut_agg.rename(columns={'quantity': 'cut_qty'}, inplace=True)
        
        existing_merge = [c for c in merge_cols if c in order_agg.columns or c in shipment_agg.columns or c in cut_agg.columns]
        if not existing_merge:
            existing_merge = merge_cols
        
        summary = order_agg
        if not shipment_agg.empty:
            summary = summary.merge(shipment_agg, on=existing_merge, how='outer')
        if not cut_agg.empty:
            summary = summary.merge(cut_agg, on=existing_merge, how='outer')
        
        if summary.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        for col in ['order_qty', 'shipment_qty', 'cut_qty']:
            if col in summary.columns:
                summary[col] = summary[col].fillna(0).astype(int)
            else:
                summary[col] = 0
        
        # Dev 保留 CutLog 原始 cut_qty（不覆盖），仅做一致性校验
        # 不再用 clip(lower=0) 覆盖
        
        if end_date_dt is not None and 'date' in summary.columns:
            summary = summary[summary['date'] <= end_date_dt]
        
        summary = summary.sort_values(
            [c for c in ['simulation_date', 'date', 'material', 'location'] if c in summary.columns]
        )
        
        cols = ['simulation_date', 'date', 'material', 'location', 'order_qty', 'shipment_qty', 'cut_qty']
        cols = [c for c in cols if c in summary.columns]
        summary = summary[cols]
        
        if run_id:
            summary['run_id'] = run_id
        
        self.db.create_table_from_df(summary, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(summary)}
        print(f"  [OK] {table_name}: {len(summary):,} 行")
        
        return len(summary)
    
    def _generate_changeover_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成换产汇总报告"""
        table_name = "summary_output_fullchangeoverlog"
        
        try:
            df = self.db.read_table("module4_output_changeoverlog")
        except:
            print(f"  [WARN] {table_name}: 源表不存在")
            return 0
        
        if df.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        # 按run_id过滤
        if run_id and 'run_id' in df.columns:
            df = df[df['run_id'] == run_id].reset_index(drop=True)
        
        # 按日期过滤（与 Dev 版本一致：只在 changeover_end_date 上过滤，不在 date 上过滤）
        # Dev 数据中 changeover_end_date 不存在，所以过滤是 no-op，保留全部行
        # DB 数据中 date 列值可能 > end_date，但不应在 date 上过滤
        if end_date_dt is not None:
            for col in ['changeover_start_date', 'changeover_end_date', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            if 'changeover_end_date' in df.columns:
                df = df[(df['changeover_end_date'].isna()) | (df['changeover_end_date'] <= end_date_dt)]
        
        if df.empty:
            print(f"  [WARN] {table_name}: 过滤后无数据")
            return 0
        
        # 添加run_id
        if run_id and 'run_id' not in df.columns:
            df['run_id'] = run_id
        
        # 写入数据库
        self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(df)}
        print(f"  [OK] {table_name}: {len(df):,} 行")
        
        return len(df)
    
    def _generate_capacity_exceed_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成产能超限汇总报告"""
        table_name = "summary_output_fullcapacityexceed"
        
        try:
            df = self.db.read_table("module4_output_capacityexceed")
        except:
            print(f"  [WARN] {table_name}: 源表不存在")
            return 0
        
        if df.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        # 按run_id过滤
        if run_id and 'run_id' in df.columns:
            df = df[df['run_id'] == run_id]
        
        # 按日期过滤
        if end_date_dt is not None and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[(df['date'].isna()) | (df['date'] <= end_date_dt)]
        
        if df.empty:
            print(f"  [WARN] {table_name}: 过滤后无数据")
            return 0
        
        # 添加run_id
        if run_id and 'run_id' not in df.columns:
            df['run_id'] = run_id
        
        # 写入数据库
        self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(df)}
        print(f"  [OK] {table_name}: {len(df):,} 行")
        
        return len(df)
    
    def _generate_production_plan_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成生产计划汇总报告"""
        table_name = "summary_output_fullproductionplan"
        
        try:
            df = self.db.read_table("module4_output_productionplan")
        except:
            print(f"  [WARN] {table_name}: 源表不存在")
            return 0
        
        if df.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        # 按run_id过滤
        if run_id and 'run_id' in df.columns:
            df = df[df['run_id'] == run_id].reset_index(drop=True)
        
        if end_date_dt is not None and 'available_date' in df.columns:
            df['available_date'] = pd.to_datetime(df['available_date'], errors='coerce')
            df = df[(df['available_date'].isna()) | (df['available_date'] <= end_date_dt)]
        
        if df.empty:
            print(f"  [WARN] {table_name}: 过滤后无数据")
            return 0
        
        # 添加run_id
        if run_id and 'run_id' not in df.columns:
            df['run_id'] = run_id
        
        # 写入数据库
        self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(df)}
        print(f"  [OK] {table_name}: {len(df):,} 行")
        
        return len(df)
    
    def _generate_deployment_plan_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成部署计划汇总报告"""
        table_name = "summary_output_fulldeploymentplan"
        
        try:
            df = self.db.read_table("module5_output_deploymentplan")
        except:
            print(f"  [WARN] {table_name}: 源表不存在")
            return 0
        
        if df.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        # 按run_id过滤
        if run_id and 'run_id' in df.columns:
            df = df[df['run_id'] == run_id].reset_index(drop=True)
        
        # 按日期过滤（与 Dev 版本 _generate_deployment_report 保持一致）
        # Dev 版本过滤列: ['deployment_date', 'arrival_date', 'ship_date', 'date']
        # 只有 'date' 列存在于实际数据中，且值在仿真范围内，所以过滤是 no-op
        if end_date_dt is not None:
            date_cols = ['deployment_date', 'arrival_date', 'ship_date', 'date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # 创建过滤mask（使用 df.index 确保对齐）
            mask = pd.Series([True] * len(df), index=df.index)
            for col in date_cols:
                if col in df.columns:
                    mask = mask & ((df[col].isna()) | (df[col] <= end_date_dt))
            df = df[mask]
        
        if df.empty:
            print(f"  [WARN] {table_name}: 过滤后无数据")
            return 0
        
        # 添加run_id
        if run_id and 'run_id' not in df.columns:
            df['run_id'] = run_id
        
        # 写入数据库
        self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(df)}
        print(f"  [OK] {table_name}: {len(df):,} 行")
        
        return len(df)
    
    def _generate_delivery_plan_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成交付计划汇总报告"""
        table_name = "summary_output_fulldeliveryplan"
        
        try:
            df = self.db.read_table("module6_output_deliveryplan")
        except:
            print(f"  [WARN] {table_name}: 源表不存在")
            return 0
        
        if df.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        # 按run_id过滤
        if run_id and 'run_id' in df.columns:
            df = df[df['run_id'] == run_id].reset_index(drop=True)
        
        # 按日期过滤（与 Dev 版本 _generate_delivery_report 保持一致）
        # Dev 版本过滤列: ['planned_deploy_date', 'actual_ship_date']
        # DB 表列名是 planned_deployment_date（不匹配 planned_deploy_date），所以只在 actual_ship_date 上实际过滤
        if end_date_dt is not None:
            date_cols = ['planned_deploy_date', 'actual_ship_date', 'date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # 创建过滤mask（使用 df.index 确保对齐）
            mask = pd.Series([True] * len(df), index=df.index)
            for col in ['planned_deploy_date', 'actual_ship_date']:
                if col in df.columns:
                    mask = mask & ((df[col].isna()) | (df[col] <= end_date_dt))
            df = df[mask]
        
        if df.empty:
            print(f"  [WARN] {table_name}: 过滤后无数据")
            return 0
        
        # 添加run_id
        if run_id and 'run_id' not in df.columns:
            df['run_id'] = run_id
        
        # 写入数据库
        self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(df)}
        print(f"  [OK] {table_name}: {len(df):,} 行")
        
        return len(df)
    
    def _generate_truck_usage_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成卡车使用汇总报告"""
        table_name = "summary_output_fulltruckusage"
        
        try:
            df = self.db.read_table("module6_output_truckusagelog")
        except:
            print(f"  [WARN] {table_name}: 源表不存在")
            return 0
        
        if df.empty:
            print(f"  [WARN] {table_name}: 无数据")
            return 0
        
        # 按run_id过滤
        if run_id and 'run_id' in df.columns:
            df = df[df['run_id'] == run_id]
        
        # 按日期过滤
        if end_date_dt is not None and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[(df['date'].isna()) | (df['date'] <= end_date_dt)]
        
        if df.empty:
            print(f"  [WARN] {table_name}: 过滤后无数据")
            return 0
        
        # 添加run_id
        if run_id and 'run_id' not in df.columns:
            df['run_id'] = run_id
        
        # 写入数据库
        self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
        self.written_tables[table_name] = {"module": "summary", "rows": len(df)}
        print(f"  [OK] {table_name}: {len(df):,} 行")
        
        return len(df)


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
    
    优化流程：
    1. 批量写入所有表数据（不创建索引）
    2. 完成所有写入后，批量创建所有索引
    3. 显著提升总体性能（避免写入期间的I/O竞争）
    
    参数：
        run_output_dir: 运行输出目录
        db_host: 数据库主机
        db_port: 数据库端口
        db_name: 数据库名称
        db_user: 用户名
        db_password: 密码
    
    返回：
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
        print(f"[ERROR] 数据库连接失败: {conn_result['message']}")
        return False
    
    # 创建写入器并执行
    writer = ModuleDataWriter(db)
    
    try:
        # 阶段1：批量写入所有表数据（不创建索引）
        writer.write_all_modules(run_output_dir)
        writer.print_summary()
        
        # 阶段2：所有数据写入完成后，批量创建所有索引
        db.build_all_pending_indexes()
        
        return True
    except Exception as e:
        print(f"[ERROR] 写入失败: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        write_run_data_to_db(sys.argv[1])
    else:
        print("用法: python module_data_writer.py <run_output_dir>")
