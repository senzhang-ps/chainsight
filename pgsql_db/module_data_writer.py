"""
模块数据写入器
将各模块的输入/输出数据写入 PostgreSQL 数据库
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import os
from datetime import datetime

from psycopg import errors, sql
from .db_connection import DatabaseConnection
from .table_schemas import MODULE6_OUTPUT_SCHEMAS, get_columns
from src.utils.numeric_safe import safe_int_series

_summary_logger = logging.getLogger("SupplyChainSimulation")


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

    def _read_table_for_run(self, table_name: str, run_id: str | None = None) -> pd.DataFrame:
        """当表包含 run_id 元数据时，只读取当前运行的数据。"""
        if not run_id:
            return self.db.read_table(table_name)

        try:
            return self.db.read_table(table_name, filters={"run_id": run_id})
        except errors.UndefinedColumn:
            # 旧表可能没有 run_id 列，此时回退到历史读取逻辑。
            df = self.db.read_table(table_name)
            if "run_id" in df.columns:
                return df[df["run_id"] == run_id].reset_index(drop=True)
            return df
        except errors.UndefinedTable:
            return pd.DataFrame()
    
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
            return 0


        # 定义所有需要按 run_id 删除的输出表
        output_tables = [
            # 模块1输出表
            'module1_output_orderlog',
            'module1_output_shipmentlog',
            'module1_output_cutlog',
            'module1_output_supplydemandlog',
            'module1_output_summary',
            # 模块3输出表
            'module3_output_netdemand',
            # 模块4输出表
            'module4_output_productionplan',
            'module4_output_capacityexceed',
            'module4_output_validation',
            'module4_output_changeoverlog',
            # 模块5输出表
            'module5_output_deploymentplan',
            'module5_output_unfulfilledlog',
            'module5_output_stockonhandlog',
            'module5_output_validation',
            # 模块6输出表
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
            'summary_historical_inventory_record',
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

        for table_name in output_tables:
            try:
                with self.db.get_cursor() as cursor:
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
                            sql.SQL('DELETE FROM {} WHERE run_id = %s').format(
                                sql.Identifier(table_name)
                            ),
                            (run_id,)
                        )
                        deleted_count += 1
            except Exception:
                pass

        return deleted_count

    # 所有输出表的完整列表（类级别常量，供预建表和清理复用）
    ALL_OUTPUT_TABLES = [
        # 模块1输出表
        'module1_output_orderlog',
        'module1_output_shipmentlog',
        'module1_output_cutlog',
        'module1_output_supplydemandlog',
        'module1_output_summary',
        # 模块3输出表
        'module3_output_netdemand',
        # 模块4输出表
        'module4_output_productionplan',
        'module4_output_capacityexceed',
        'module4_output_validation',
        'module4_output_changeoverlog',
        # 模块5输出表
        'module5_output_deploymentplan',
        'module5_output_unfulfilledlog',
        'module5_output_stockonhandlog',
        'module5_output_validation',
        # 模块6输出表
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
        'summary_historical_inventory_record',
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

        每张表至少包含公共列 run_id、sim_date、config_name、db_write_time，
        后续批量写入时若数据帧带有更多列，会自动补齐缺失字段。

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
            pass
        else:
            pass
        return created

    def prepare_orchestrator_day_dataframes(
        self,
        orchestrator_dir: str,
        run_id: str | None,
        sim_date: str,
    ) -> Dict[str, tuple[pd.DataFrame, List[str]]]:
        """预处理单个仿真日的编排器 CSV 数据，供数据库批量写入使用。

        这样可以让编排器的当日状态写入与模块批量写入在同一事务内提交，
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
        【优化版】只写入汇总文件和编排器数据
        
        这个方法专门用于仿真结束后的批量写入，避免每天重复写入模块输出。
        汇总文件已包含所有模块的完整汇总数据。
        
        优化效果：
        - 减少写入次数：从5天×6模块×多工作表 → 8个汇总文件 + 10个编排器文件
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
        
        
        results = {}
        
        # 1. 写入汇总数据（最重要）
        summary_dir = base_path / "summary"
        if summary_dir.exists():
            summary_results = self._write_summary_files_fast(str(summary_dir), run_id, if_exists)
            results.update(summary_results)
        
        # 2. 写入编排器状态数据
        orch_dir = base_path / "orchestrator"
        if orch_dir.exists():
            orch_results = self.write_orchestrator_data(str(orch_dir), run_id=run_id, if_exists=if_exists)
            results.update(orch_results)
        
        elapsed = time.time() - start_time
        total_tables = len([v for v in results.values() if isinstance(v, int) and v >= 0])
        total_rows = sum(v for v in results.values() if isinstance(v, int) and v > 0)
        
        
        return results
    
    def _write_summary_files_fast(
        self,
        summary_dir: str,
        run_id: str = None,
        if_exists: str = "replace"
    ) -> Dict[str, int]:
        """快速写入汇总文件"""
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
                    
                    # 添加 run_id 列
                    if run_id and not df.empty:
                        df['run_id'] = run_id
                    
                    # 写入数据库（传入 config_name）
                    self.db.create_table_from_df(df, table_name, if_exists, config_name=self.config_name)
                    results[table_name] = len(df)
                    self.written_tables[table_name] = {
                        "source": str(file_path),
                        "module": "summary",
                        "rows": len(df)
                    }
                except Exception as e:
                    raise
        
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
            return {}
        
        results = {}
        
        # 跟踪每个表是否已经被写入过（用于 replace 模式）
        # 第一次写入用 replace，后续用 append
        tables_written: Dict[str, bool] = {}
        
        # 查找所有 Excel 文件并按日期排序
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
                raise
        
        # 查找所有 CSV 文件并按日期排序
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
                raise
        
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
        """写入单个 Excel 文件的所有工作表
        
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
        
        # 使用简化的表名格式：{module_name}_output_{sheet_name}
        # 这样与内存模式的表名保持一致
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name)
                
                # 构建表名（简化格式，与内存模式一致）
                table_name = f"{module_name}_output_{self._clean_name(sheet_name)}"
                
                # 如果数据帧为空，尝试使用预定义的列名（如果有的话）
                if df.empty:
                    schema_columns = get_columns(module_name, sheet_name)
                    if schema_columns:  # 只有定义了列名时才使用
                        df = pd.DataFrame(columns=schema_columns)
                    # 否则保持空数据帧（与基线保持一致）
                
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
                
                # 添加 run_id 列
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
                
                # 写入数据库（即使 df.empty 也会创建表结构，并传入 config_name）
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
                raise
        
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
        
        # 自动生成 run_id
        if run_id is None:
            run_id = base_path.name
        
        
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
                pass
        
        elapsed = time.time() - start_time
        
        # 统计
        total_tables = len(self.written_tables)
        total_rows = sum(info.get('rows', 0) for info in self.written_tables.values())
        
        
        return all_results
    
    def write_orchestrator_data(
        self,
        orchestrator_dir: str,
        run_id: str = None,
        sim_date: str = None,
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """
        写入编排器输出的 CSV 数据
        
        参数：
            orchestrator_dir: 编排器输出目录
            run_id: 运行ID
            sim_date: 仿真日期（格式：YYYYMMDD）
            if_exists: 如果表存在的处理方式
        
        返回：
            dict: 每个文件的写入行数
        """
        orch_path = Path(orchestrator_dir)
        if not orch_path.exists():
            return {}
        
        results = {}
        
        # 定义编排器输出文件类型
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
                             # 统一 sim_date 格式为 YYYY-MM-DD
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
                        raise
                
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

         区分可跳过异常与真正错误
         覆盖 汇总与编排器 表
         添加 run_id 过滤

         参数：
            run_id: 运行ID，用于按 run_id 过滤删除
            batch_start_date: 批次第一天 (YYYY-MM-DD)，删除 sim_date >= 此值的行
        """
        # 完整表列表，与 truncate_output_tables 保持一致
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
            'summary_historical_inventory_record',
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
                # 检查是否有 run_id 列，按 sim_date + run_id 删除
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
                # 区分可跳过异常与真正错误
                err_str = str(e).lower()
                if 'does not exist' in err_str or 'column' in err_str:
                    continue  # 表或列不存在，合理跳过
                else:
                    import logging
                    logging.error(f"delete_batch_data: 表 {table_name} 删除失败: {e}")
                    raise  # 真正的错误（连接中断、锁超时等），上报
        if deleted_total > 0:
            pass

    def _filter_module1_orders_for_day(
        self,
        df: pd.DataFrame,
        day_sim_date: str | None,
    ) -> pd.DataFrame:
        """仅保留当前仿真日新生成的模块1订单。

        集成模式下的 `orders_df` 为累计口径，后续日期会包含前序日期已生成的订单。
        但数据库对账与本地 Excel 比对都按行级 `simulation_date` 视为“当日订单日志”，
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

    def _zero_fill_quantity_for_db(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """在数据库写入边界把异常订单数量归零。"""
        if df.empty or table_name != 'module1_output_orderlog' or 'quantity' not in df.columns:
            return df

        cleaned = df.copy()
        cleaned['quantity'] = safe_int_series(
            cleaned['quantity'],
            context=f'db.{table_name}.quantity',
        )
        return cleaned

    def prepare_batch_dataframes(
        self,
        all_results: Dict[str, Any],
        run_id: str = None,
    ) -> Dict[str, tuple]:
        """
        预处理批次数据为可写入的数据帧（不涉及数据库操作）。

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
                        if df_key == 'unfulfilled_log':
                            try:
                                _sl = int((df['sending'] == df['receiving']).sum()) \
                                    if not df.empty and 'sending' in df.columns else 0
                            except Exception:
                                _sl = -1
                            _summary_logger.info(
                                f"[PROBE-A] batch_results day unfulfilled_log rows={len(df)} "
                                f"(self_loop={_sl}) sim_date={day_sim_date}"
                            )
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
                if table_name == 'module5_output_unfulfilledlog':
                    _summary_logger.info(
                        f"[PROBE-A] concat module5_output_unfulfilledlog rows={len(combined_df)}"
                    )
                # 方案B：module4 两张表与本地文件输出（output_writer.write_output）套同一列 schema，
                # 否则 DB 表会缺 changeover 三列、且 capacityexceed 串成
                # production_plan_date/unmet_uncon_planned_qty（仅 DB 写入层对齐，不影响 module5 共享结果）。
                if table_name in ('module4_output_productionplan', 'module4_output_capacityexceed'):
                    from src.modules.production_planning.constants import (
                        PLAN_COLUMNS,
                        EXCEED_COLUMNS,
                    )
                    _schema = list(
                        PLAN_COLUMNS
                        if table_name == 'module4_output_productionplan'
                        else EXCEED_COLUMNS
                    )
                    # 补齐 schema 业务列（缺则空），丢弃不在 schema 的多余业务列，
                    # 但务必保留已存在的元数据列（尤其 sim_date，否则按 sim_date 过滤会查不到行）。
                    for _col in _schema:
                        if _col not in combined_df.columns:
                            combined_df[_col] = pd.Series(dtype='object', index=combined_df.index)
                    _meta_keep = [
                        c for c in combined_df.columns
                        if c not in _schema
                        and c in ('sim_date', 'file_date', 'run_id', 'config_name', 'db_write_time')
                    ]
                    combined_df = combined_df[_schema + _meta_keep]
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
                combined_df = self._zero_fill_quantity_for_db(combined_df, table_name)
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
        
        这个方法用于数据库模式，绕过文件系统直接将数据帧写入数据库。
        
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
        
        # [删除] 在写入前按 run_id 删除当前运行的旧数据（不影响其他 run_id 数据）
        if truncate_first:
            self.truncate_output_tables(run_id=run_id)
        
        
        # 定义模块输出数据帧到表名的映射
        # 键名必须与模块返回的字典键名一致
        module_df_mapping = {
            'module1': {
                # `orders_df` 是当天新增订单（`today_orders_df`），与比对脚本从 xlsx 过滤后的口径一致。
                # `simulation_date` 与当天口径一致
                'orders_df': 'module1_output_orderlog',
                'shipment_df': 'module1_output_shipmentlog',
                'cut_df': 'module1_output_cutlog',
                'supply_demand_df': 'module1_output_supplydemandlog',
                'summary_df': 'module1_output_summary',  # 添加汇总映射
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
            
            # [调试] 打印每个模块的结果详情
            if module_results and len(module_results) > 0:
                first_result = module_results[0]
                if isinstance(first_result, dict):
                    # 检查键是否匹配
                    expected_keys = set(df_mapping.keys())
                    actual_keys = set(first_result.keys())
                    matched = expected_keys & actual_keys
                    missing = expected_keys - actual_keys
                    extra = actual_keys - expected_keys
                    if missing:
                        pass
                    if extra:
                        pass
            
            if not module_results:
                continue
            
            
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
                    # [调试] 打印每个数据帧键的检查结果
                    if df is not None:
                        if isinstance(df, pd.DataFrame):
                            pass
                        else:
                            pass
                    else:
                        pass
                    if df is not None and isinstance(df, pd.DataFrame):
                        # 为每一天的数据添加 sim_date（包括空数据帧）
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
                    # 即使没有数据，也创建空表结构
                    # 创建带有 sim_date 和 run_id 列的空数据帧
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
                    except Exception as e:
                        raise
                    continue

                combined_df = pd.concat(dfs, ignore_index=True)

                # 确保 sim_date 列存在（即使是空表）
                if 'sim_date' not in combined_df.columns:
                    combined_df['sim_date'] = pd.Series(dtype='string')

                # 添加 run_id 列
                if run_id:
                    if combined_df.empty:
                        combined_df['run_id'] = pd.Series(dtype='string')
                    else:
                        combined_df['run_id'] = run_id

                # 写入数据库（即使 combined_df.empty 也会创建表结构，并传入 config_name）
                combined_df = self._zero_fill_quantity_for_db(combined_df, table_name)
                try:
                    self.db.create_table_from_df(combined_df, table_name, if_exists, config_name=self.config_name)
                    results[table_name] = len(combined_df)
                    self.written_tables[table_name] = {
                        "module": module_name,
                        "rows": len(combined_df)
                    }
                except Exception as e:
                    raise
        
        # 统计
        total_tables = sum(1 for v in results.values() if v > 0)
        total_rows = sum(v for v in results.values() if v > 0)
        
        
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
            return
        
        
        for table_name, info in self.written_tables.items():
            source = info.get('module', info.get('source', 'unknown'))[:20]
            rows = info.get('rows', 0)
        
    
    def generate_summary_reports_from_db(
        self,
        run_id: str = None,
        start_date: str = None,
        end_date: str = None,
        if_exists: str = "replace"
    ) -> Dict[str, int]:
        """
        从数据库中的模块输出表直接生成汇总报告
        
        当使用 --use-db 模式时，模块不会输出xlsx文件，因此无法使用
        文件汇总生成器从文件生成汇总报告。此方法直接从已写入
        数据库的模块输出表中聚合数据，生成7个汇总报告表：
        
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
        
        if start_date and end_date:
            pass
        
        results = {}

        # 转换日期为 datetime，用于过滤
        end_date_dt = pd.to_datetime(end_date) if end_date else None

        # 为汇总聚合的高频过滤/分组列建立复合索引（幂等，IF NOT EXISTS）
        self._ensure_summary_source_indexes()

        # 1. 生成订单/发货/缺货汇总报告
        try:
            results['summary_output_ordershipmentcutsummary'] = self._generate_order_shipment_cut_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_ordershipmentcutsummary: {e}"
            )
            raise

        # 2. 生成换产汇总报告
        try:
            results['summary_output_fullchangeoverlog'] = self._generate_changeover_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_fullchangeoverlog: {e}"
            )
            raise

        # 3. 生成产能超限汇总报告
        try:
            results['summary_output_fullcapacityexceed'] = self._generate_capacity_exceed_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_fullcapacityexceed: {e}"
            )
            raise

        # 4. 生成生产计划汇总报告
        try:
            results['summary_output_fullproductionplan'] = self._generate_production_plan_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_fullproductionplan: {e}"
            )
            raise

        # 5. 生成部署计划汇总报告
        try:
            results['summary_output_fulldeploymentplan'] = self._generate_deployment_plan_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_fulldeploymentplan: {e}"
            )
            raise

        # 6. 生成交付计划汇总报告
        try:
            results['summary_output_fulldeliveryplan'] = self._generate_delivery_plan_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_fulldeliveryplan: {e}"
            )
            raise

        # 7. 生成卡车使用汇总报告
        try:
            results['summary_output_fulltruckusage'] = self._generate_truck_usage_summary(
                run_id, end_date_dt, if_exists
            )
        except Exception as e:
            _summary_logger.exception(
                f"[ERROR] Summary 子表生成失败 summary_output_fulltruckusage: {e}"
            )
            raise
        
        elapsed = time.time() - start_time
        total_tables = sum(1 for v in results.values() if isinstance(v, int) and v >= 0)
        total_rows = sum(v for v in results.values() if isinstance(v, int) and v > 0)
        
        
        return results
    
    def _get_table_columns(self, table_name: str) -> set:
        """返回表的列名集合；表不存在时返回空集合。"""
        try:
            with self.db.get_cursor(commit=False) as cursor:
                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = %s
                    """,
                    (table_name,),
                )
                return {row[0] for row in cursor.fetchall()}
        except Exception:
            return set()

    def _get_table_column_info(self, table_name: str) -> List[tuple[str, str]]:
        """按表内字段顺序返回字段名和 PostgreSQL 类型。"""
        try:
            with self.db.get_cursor(commit=False) as cursor:
                cursor.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table_name,),
                )
                return [(row[0], str(row[1])) for row in cursor.fetchall()]
        except Exception:
            return []

    # 汇总聚合 / 过滤的高频列组合，用于建立复合 BTREE 索引。
    # 前导列 run_id 服务 WHERE 等值过滤，其余列服务 GROUP BY / ORDER BY，
    # 让 PostgreSQL 尽量走索引扫描并省去额外排序。
    def _get_table_column_types(self, table_name: str) -> Dict[str, str]:
        """返回每个字段在 information_schema 中登记的小写数据类型。"""
        try:
            with self.db.get_cursor(commit=False) as cursor:
                cursor.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                    """,
                    (table_name,),
                )
                return {row[0]: str(row[1]).lower() for row in cursor.fetchall()}
        except Exception:
            return {}

    def _ensure_table_columns(self, table_name: str, column_defs: Dict[str, str]) -> bool:
        """必要时创建表并补齐缺失字段，不删除既有数据。"""
        existed = self.db.table_exists(table_name)
        cols_sql = sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(pg_type))
            for col, pg_type in column_defs.items()
        )
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                    sql.Identifier(table_name),
                    cols_sql,
                )
            )

        existing_cols = self._get_table_columns(table_name)
        for col, pg_type in column_defs.items():
            if col in existing_cols:
                continue
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                        sql.Identifier(table_name),
                        sql.Identifier(col),
                        sql.SQL(pg_type),
                    )
                )

        # 类型对齐：仅处理"声明 BIGINT 但既有列是浮点/numeric"的安全收窄。
        # 旧版本曾把 qty 列建成 DOUBLE PRECISION；此处用 TRUNC 向零截断改回 BIGINT，
        # 与 safe_int_series 截断语义一致，且对已是整数值的历史行无损。
        # 其余类型差异一律不动，避免误伤 TEXT/TIMESTAMP 等列。
        _FLOATY = {"double precision", "real", "numeric"}
        existing_types = self._get_table_column_types(table_name)
        for col, pg_type in column_defs.items():
            if pg_type.strip().upper() != "BIGINT":
                continue
            if existing_types.get(col) in _FLOATY:
                try:
                    with self.db.get_cursor() as cursor:
                        cursor.execute(
                            sql.SQL(
                                "ALTER TABLE {tbl} ALTER COLUMN {col} TYPE BIGINT "
                                "USING TRUNC({col})::bigint"
                            ).format(
                                tbl=sql.Identifier(table_name),
                                col=sql.Identifier(col),
                            )
                        )
                except Exception as e:
                    _summary_logger.warning(
                        "[SCHEMA] 列类型对齐跳过 %s.%s: %s", table_name, col, e
                    )
        return existed

    def _prepare_summary_target(
        self,
        table_name: str,
        column_defs: Dict[str, str],
        if_exists: str,
        run_id: Optional[str],
    ) -> None:
        """确保目标表结构正确，并按当前运行应用汇总表的 replace/fail 语义。"""
        existed = self._ensure_table_columns(table_name, column_defs)
        if if_exists == "fail" and existed:
            raise ValueError(f"Table {table_name} already exists")
        if if_exists != "replace" or not existed:
            return

        cols = self._get_table_columns(table_name)
        with self.db.get_cursor() as cursor:
            if run_id and "run_id" in cols:
                cursor.execute(
                    sql.SQL("DELETE FROM {} WHERE run_id = %s").format(
                        sql.Identifier(table_name)
                    ),
                    (run_id,),
                )
            elif not run_id:
                cursor.execute(
                    sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name))
                )

    def _timestamp_expr(self, column: str, col_type: Optional[str]) -> Any:
        """构造时间戳表达式，避免随意转换无法解析的文本。"""
        ident = sql.Identifier(column)
        if col_type in {"date", "timestamp without time zone", "timestamp with time zone"}:
            return sql.SQL("{}::timestamp").format(ident)

        return sql.SQL(
            """
            CASE
                WHEN NULLIF({col}::text, '') IS NULL THEN NULL::timestamp
                WHEN {col}::text ~ '^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
                    THEN ({col}::text)::timestamp
                ELSE NULL::timestamp
            END
            """
        ).format(col=ident)

    def _date_filter_condition(
        self,
        column: str,
        col_type: Optional[str],
        keep_invalid_text: bool = True,
    ) -> Any:
        """构造 date <= end_date 的过滤条件，并兼容文本类型日期列。"""
        ident = sql.Identifier(column)
        if col_type in {"date", "timestamp without time zone", "timestamp with time zone"}:
            return sql.SQL("({col} IS NULL OR {col} <= %s)").format(col=ident)

        invalid_result = sql.SQL("TRUE") if keep_invalid_text else sql.SQL("FALSE")
        return sql.SQL(
            """
            (
                {col} IS NULL
                OR NULLIF({col}::text, '') IS NULL
                OR CASE
                    WHEN {col}::text ~ '^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
                        THEN ({col}::text)::timestamp <= %s
                    ELSE {invalid_result}
                END
            )
            """
        ).format(col=ident, invalid_result=invalid_result)

    @staticmethod
    def _summary_passthrough_type(col_name: str, pg_type: str) -> str:
        """返回透传 Summary 目标列类型，元数据列使用固定类型。"""
        col_lower = col_name.lower()
        if col_lower in {"run_id", "config_name"}:
            return "TEXT"
        if col_lower == "db_write_time":
            return "TIMESTAMP"
        return pg_type or "TEXT"

    @staticmethod
    def _is_text_pg_type(pg_type: str) -> bool:
        return (pg_type or "").lower() in {
            "text",
            "character varying",
            "character",
            "varchar",
            "char",
        }

    @staticmethod
    def _is_date_pg_type(pg_type: str) -> bool:
        return (pg_type or "").lower() in {
            "date",
            "timestamp without time zone",
            "timestamp with time zone",
            "timestamp",
        }

    @staticmethod
    def _is_integer_pg_type(pg_type: str) -> bool:
        return (pg_type or "").lower() in {
            "smallint",
            "integer",
            "bigint",
        }

    @staticmethod
    def _is_float_pg_type(pg_type: str) -> bool:
        return (pg_type or "").lower() in {
            "real",
            "double precision",
            "numeric",
        }

    def _build_passthrough_column_defs(
        self,
        source_column_info: List[tuple[str, str]],
        run_id: Optional[str],
    ) -> Dict[str, str]:
        """基于源表字段构造透传 Summary 目标表字段定义。"""
        column_defs: Dict[str, str] = {}
        for col_name, pg_type in source_column_info:
            column_defs[col_name] = self._summary_passthrough_type(col_name, pg_type)

        if run_id and "run_id" not in column_defs:
            column_defs["run_id"] = "TEXT"
        if self.config_name and "config_name" not in column_defs:
            column_defs["config_name"] = "TEXT"
        if "db_write_time" not in column_defs:
            column_defs["db_write_time"] = "TIMESTAMP"
        return column_defs

    def _ensure_passthrough_summary_target(
        self,
        target_table: str,
        column_defs: Dict[str, str],
    ) -> bool:
        """准备 SQL-only 透传 Summary 目标表，只做安全建表、补列和宽化。"""
        existed = self.db.table_exists(target_table)
        cols_sql = sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(pg_type))
            for col, pg_type in column_defs.items()
        )
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                    sql.Identifier(target_table),
                    cols_sql,
                )
            )

        existing_cols = self._get_table_columns(target_table)
        for col, pg_type in column_defs.items():
            if col in existing_cols:
                continue
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                        sql.Identifier(target_table),
                        sql.Identifier(col),
                        sql.SQL(pg_type),
                    )
                )
            existing_cols.add(col)

        existing_types = self._get_table_column_types(target_table)
        for col, desired_type in column_defs.items():
            existing_type = existing_types.get(col)
            if not existing_type:
                continue
            desired_lower = desired_type.lower()
            try:
                if self._is_text_pg_type(desired_lower) and not self._is_text_pg_type(existing_type):
                    with self.db.get_cursor() as cursor:
                        cursor.execute(
                            sql.SQL(
                                "ALTER TABLE {} ALTER COLUMN {} TYPE TEXT USING {}::TEXT"
                            ).format(
                                sql.Identifier(target_table),
                                sql.Identifier(col),
                                sql.Identifier(col),
                            )
                        )
                elif self._is_float_pg_type(desired_lower) and self._is_integer_pg_type(existing_type):
                    with self.db.get_cursor() as cursor:
                        cursor.execute(
                            sql.SQL(
                                "ALTER TABLE {} ALTER COLUMN {} TYPE DOUBLE PRECISION "
                                "USING {}::DOUBLE PRECISION"
                            ).format(
                                sql.Identifier(target_table),
                                sql.Identifier(col),
                                sql.Identifier(col),
                            )
                        )
            except Exception as e:
                _summary_logger.warning(
                    "[SCHEMA] 透传 Summary 列类型对齐跳过 %s.%s: %s",
                    target_table, col, e,
                )
        return existed

    def _build_summary_where_clause(
        self,
        source_cols: set,
        col_types: Dict[str, str],
        run_id: Optional[str],
        end_date_dt: Optional[pd.Timestamp],
        date_filter_cols: Optional[List[str]],
    ) -> tuple[Any, List[Any]]:
        """构造 Summary 源表过滤条件，复用当前 run_id 与日期口径。"""
        where_parts = []
        params: List[Any] = []
        if run_id and "run_id" in source_cols:
            where_parts.append(sql.SQL("run_id = %s"))
            params.append(run_id)
        if end_date_dt is not None:
            for col in date_filter_cols or []:
                if col in source_cols:
                    where_parts.append(
                        self._date_filter_condition(col, col_types.get(col))
                    )
                    params.append(end_date_dt)

        if not where_parts:
            return sql.SQL(""), params
        return sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts), params

    def _build_summary_order_clause(
        self,
        source_cols: set,
        sort_cols: Optional[List[str]],
        stable_tiebreaker: bool,
    ) -> Any:
        """构造 Summary 透传输出排序条件。"""
        order_terms: List[Any] = []
        if sort_cols:
            order_terms = [sql.Identifier(c) for c in sort_cols if c in source_cols]
        if stable_tiebreaker:
            order_terms.append(sql.SQL("ctid"))
        if not order_terms:
            return sql.SQL("")
        return sql.SQL("ORDER BY ") + sql.SQL(", ").join(order_terms)

    def _summary_source_has_rows(
        self,
        source_table: str,
        where_sql: Any,
        where_params: List[Any],
    ) -> bool:
        """判断源表过滤后是否存在数据，避免无数据时创建空 Summary 表。"""
        query = sql.SQL("SELECT 1 FROM {tbl} {where} LIMIT 1").format(
            tbl=sql.Identifier(source_table),
            where=where_sql,
        )
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(query, tuple(where_params) if where_params else None)
            return cursor.fetchone() is not None

    def _build_passthrough_select_items(
        self,
        source_columns: List[str],
        output_columns: List[str],
        run_id: Optional[str],
        target_col_types: Optional[Dict[str, str]] = None,
    ) -> tuple[List[Any], List[Any]]:
        """构造 INSERT SELECT 的 SELECT 字段表达式，并返回占位符参数。"""
        # 将源表字段列表转为集合，便于判断某个目标字段是否真实存在于源表。
        source_col_set = set(source_columns)
        # 目标表字段类型可能为空；为空时使用空字典，避免后续 get 访问失败。
        target_col_types = target_col_types or {}
        # select_items 保存 SELECT 后面的每个输出表达式，例如 col AS col、NOW() AS db_write_time。
        select_items: List[Any] = []
        # params 保存 select_items 中使用到的 %s 占位符参数，例如 config_name 或补写的 run_id。
        params: List[Any] = []
        # 按目标表字段顺序逐列生成 SELECT 表达式，保证 INSERT 字段与 SELECT 字段一一对应。
        for col in output_columns:
            # 读取目标字段当前类型；用于判断是否需要显式转成文本以兼容历史表结构。
            target_type = target_col_types.get(col, "")
            # config_name 是运行配置标签：无论源表是否有旧值，Summary 统一写当前 writer.config_name。
            if col == "config_name" and self.config_name:
                # 使用 %s::text 参数化写入，避免把配置名直接拼进 SQL。
                select_items.append(sql.SQL("%s::text AS {}").format(sql.Identifier(col)))
                # 记录与上方 %s 对应的真实参数值。
                params.append(self.config_name)
            # db_write_time 是本次 Summary 写入时间，由数据库当前时间生成。
            elif col == "db_write_time":
                # 如果历史目标表把 db_write_time 建成了文本列，则把 NOW() 显式转文本，避免插入类型冲突。
                if self._is_text_pg_type(target_type):
                    select_items.append(
                        sql.SQL("NOW()::text AS {}").format(sql.Identifier(col))
                    )
                # 正常情况下 db_write_time 是时间戳列，直接写 NOW()。
                else:
                    select_items.append(sql.SQL("NOW() AS {}").format(sql.Identifier(col)))
            # run_id 若源表不存在但调用方传入了 run_id，则在 SELECT 中补一列常量。
            elif col == "run_id" and col not in source_col_set and run_id:
                # 使用参数化常量补 run_id，保证目标 Summary 表仍可按 run_id 隔离。
                select_items.append(sql.SQL("%s::text AS {}").format(sql.Identifier(col)))
                # 记录与上方 %s 对应的当前运行标识。
                params.append(run_id)
            # 其他字段都是透传字段：默认从源表读取同名字段并写入目标表同名字段。
            else:
                # 默认源字段表达式就是源表同名字段。
                source_expr = sql.Identifier(col)
                # 如果目标字段是文本类型，则显式 source_col::text，兼容历史表被宽化为 TEXT 的场景。
                if self._is_text_pg_type(target_type):
                    source_expr = sql.SQL("{}::text").format(sql.Identifier(col))
                # 生成形如 source_col AS target_col 的表达式，字段名用 sql.Identifier 安全转义。
                select_items.append(
                    sql.SQL("{} AS {}").format(source_expr, sql.Identifier(col))
                )
        # 返回 SELECT 表达式列表和对应的参数列表，由 _insert_select_to_summary 统一组装执行。
        return select_items, params

    def _insert_select_to_summary(
        self,
        source_table: str,
        target_table: str,
        run_id: Optional[str],
        end_date_dt: Optional[pd.Timestamp],
        date_filter_cols: List[str],
        sort_cols: Optional[List[str]] = None,
        stable_tiebreaker: bool = False,
        if_exists: str = "replace",
    ) -> int:
        """用 INSERT SELECT 在数据库内生成透传型 Summary 表，避免 pandas 搬运。

        参数：
            source_table: 来源模块输出表名，例如 module6_output_truckusagelog。
            target_table: 目标 Summary 表名，例如 summary_output_fulltruckusage。
            run_id: 当前运行标识；源表有 run_id 列时用于过滤，目标表 replace 时用于隔离删除。
            end_date_dt: Summary 截止日期；不为空时按 date_filter_cols 下推日期过滤。
            date_filter_cols: 需要应用 end_date_dt 的候选日期列；只对源表实际存在的列生效。
            sort_cols: 输出排序字段；只对源表实际存在的字段生效。
            stable_tiebreaker: 是否追加 ctid 作为稳定排序补充键。
            if_exists: 目标表存在时的处理方式，支持 replace / append / fail。

        返回：
            int: 实际插入目标 Summary 表的行数。
        """
        # 读取源表字段名和字段类型，后续建目标表、拼 SELECT 字段都依赖这份信息。
        source_column_info = self._get_table_column_info(source_table)
        # 源表不存在或没有字段时，没有可透传的数据，直接返回 0。
        if not source_column_info:
            return 0

        # source_cols 是源表字段名集合，用于判断 run_id、日期列、排序列是否存在。
        source_cols = {col for col, _ in source_column_info}
        # col_types 是源表字段类型字典，用于构造兼容日期/文本日期的过滤条件。
        col_types = {col: str(pg_type).lower() for col, pg_type in source_column_info}

        # 检查目标 Summary 表是否已经存在，以决定 fail/replace 的处理方式。
        target_exists = self.db.table_exists(target_table)
        if target_exists:
            # fail 模式下目标表已存在即报错，避免误覆盖既有结果。
            if if_exists == "fail":
                raise ValueError(f"Table {target_table} already exists")
            # replace 模式不重建整表，优先按 run_id 删除当前运行旧数据。
            if if_exists == "replace":
                # 读取目标表字段，用于确认是否可以按 run_id 精准删除。
                target_cols = self._get_table_columns(target_table)
                # 删除旧数据属于写操作，使用默认提交游标。
                with self.db.get_cursor() as cursor:
                    # 有 run_id 且目标表存在 run_id 列时，只删除当前运行对应的 Summary 数据。
                    if run_id and "run_id" in target_cols:
                        cursor.execute(
                            # 目标表名使用 sql.Identifier 安全拼接，run_id 使用参数绑定。
                            sql.SQL("DELETE FROM {} WHERE run_id = %s").format(
                                sql.Identifier(target_table)
                            ),
                            (run_id,),
                        )
                    # 没有 run_id 时无法做运行级隔离，只能按历史兼容逻辑清空目标表。
                    elif not run_id:
                        cursor.execute(
                            # TRUNCATE 仅在缺少 run_id 的历史兼容场景使用。
                            sql.SQL("TRUNCATE TABLE {}").format(
                                sql.Identifier(target_table)
                            )
                        )

        # 构造 WHERE 过滤片段和对应参数，包括 run_id 过滤与 end_date 日期过滤。
        where_sql, where_params = self._build_summary_where_clause(
            source_cols, col_types, run_id, end_date_dt, date_filter_cols
        )
        try:
            # 先用 SELECT 1 判断过滤后是否有数据；无数据时不创建空 Summary 表。
            if not self._summary_source_has_rows(source_table, where_sql, where_params):
                return 0
        # 源表或字段不存在时按无数据处理，不中断整个 Summary 生成流程。
        except (errors.UndefinedTable, errors.UndefinedColumn):
            return 0

        # 基于源表字段构造目标 Summary 表字段定义，并补齐 run_id/config_name/db_write_time 元数据列。
        column_defs = self._build_passthrough_column_defs(source_column_info, run_id)
        # 确保目标表存在、字段齐全，并对历史表做必要的安全类型兼容。
        self._ensure_passthrough_summary_target(target_table, column_defs)

        # output_columns 是目标表写入字段顺序，必须与 SELECT 输出表达式顺序一致。
        output_columns = list(column_defs.keys())
        # source_columns 保留源表字段顺序，用于判断哪些目标字段来自源表、哪些字段需要补常量。
        source_columns = [col for col, _ in source_column_info]
        # 读取目标表当前字段类型；历史表可能已有不同类型，需要 SELECT 侧做兼容转换。
        target_col_types = self._get_table_column_types(target_table)
        # 构造 SELECT 输出表达式和 SELECT 侧常量参数，例如 config_name、补写 run_id。
        select_items, select_params = self._build_passthrough_select_items(
            source_columns, output_columns, run_id, target_col_types
        )
        # 构造 ORDER BY 片段；无排序字段时返回空 SQL 片段。
        order_sql = self._build_summary_order_clause(
            source_cols, sort_cols, stable_tiebreaker
        )
        # 后 6 张 summary_output_full... 表的核心写入 SQL：
        # target      = 目标 Summary 表，例如 summary_output_fulltruckusage。
        # target_cols = 目标表写入字段列表，顺序来自 output_columns。
        # select_items= SELECT 输出表达式列表，顺序必须与 target_cols 完全一致。
        # source      = 来源模块输出表，例如 module6_output_truckusagelog。
        # where       = run_id 与 end_date 过滤条件；无过滤条件时为空片段。
        # order       = 稳定排序条件；调用方未传排序字段时为空片段。
        query = sql.SQL(
            """
            INSERT INTO {target} ({target_cols})
            SELECT {select_items}
            FROM {source}
            {where}
            {order}
            """
        ).format(
            # target：目标 Summary 表名，用 sql.Identifier 防止表名拼接风险。
            target=sql.Identifier(target_table),
            # target_cols：INSERT INTO (...) 中的目标字段列表。
            target_cols=sql.SQL(", ").join(sql.Identifier(c) for c in output_columns),
            # select_items：SELECT 后面的字段表达式，包含源字段透传和元数据字段覆盖。
            select_items=sql.SQL(", ").join(select_items),
            # source：FROM 后面的来源模块输出表名。
            source=sql.Identifier(source_table),
            # where：WHERE 过滤片段，包含当前 run_id 和日期上限过滤。
            where=where_sql,
            # order：ORDER BY 排序片段，用于保持输出顺序稳定。
            order=order_sql,
        )

        try:
            # 执行 INSERT SELECT；参数顺序为 SELECT 常量参数在前，WHERE 条件参数在后。
            with self.db.get_cursor() as cursor:
                cursor.execute(query, tuple(select_params + where_params))
                # rowcount 即本次插入到 Summary 目标表的行数；数据库返回 None 时按 0 处理。
                rows = max(cursor.rowcount or 0, 0)
        except (errors.UndefinedTable, errors.UndefinedColumn):
            return 0

        if rows > 0:
            self.written_tables[target_table] = {"module": "summary", "rows": rows}
        return rows

    _SUMMARY_SOURCE_INDEX_SPECS = [
        ("module1_output_orderlog", ["run_id", "date", "material", "location"]),
        ("module1_output_shipmentlog", ["run_id", "date", "material", "location"]),
        ("module1_output_cutlog", ["run_id", "date", "material", "location"]),
        ("module4_output_changeoverlog", ["run_id", "changeover_end_date"]),
        ("module4_output_capacityexceed", ["run_id", "date"]),
        ("module4_output_productionplan", ["run_id", "available_date"]),
        ("module5_output_deploymentplan", ["run_id", "date"]),
        ("module6_output_deliveryplan", ["run_id", "actual_ship_date"]),
        ("module6_output_truckusagelog", ["run_id", "date"]),
    ]

    def _ensure_summary_source_indexes(self) -> int:
        """为 summary 聚合的源表建立复合索引（幂等）。

        - CREATE INDEX IF NOT EXISTS：已存在则跳过，无副作用。
        - 仅对存在的表、且复合列全部存在时建立；缺列时退化为已存在列的前缀。
        - 单个索引建立失败不影响其余（大表首次建索引耗时，但仅一次性成本）。

        返回成功执行 CREATE 的索引数量（已存在的也计入，因为 IF NOT EXISTS 不报错）。
        """
        created = 0
        for table_name, cols in self._SUMMARY_SOURCE_INDEX_SPECS:
            existing = self._get_table_columns(table_name)
            if not existing:
                continue
            idx_cols = [c for c in cols if c in existing]
            # 至少需要 run_id + 1 个分组/过滤列才有意义
            if len(idx_cols) < 2:
                continue
            index_name = f"idx_{table_name}_summary"
            col_idents = sql.SQL(", ").join(sql.Identifier(c) for c in idx_cols)
            query = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} ({cols})"
            ).format(
                idx=sql.Identifier(index_name),
                tbl=sql.Identifier(table_name),
                cols=col_idents,
            )
            try:
                with self.db.get_cursor() as cursor:
                    cursor.execute(query)
                created += 1
            except Exception as e:
                _summary_logger.warning(
                    "[INDEX] 复合索引建立跳过 %s(%s): %s",
                    table_name, ", ".join(idx_cols), e,
                )
        return created

    def _sample_config_sim_date(self, run_id: str, candidate_tables: list) -> str:
        """从候选表的 config_name 列采样一行，提取 8 位日期 → 'YYYY-MM-DD'。"""
        import re
        for tbl in candidate_tables:
            cols = self._get_table_columns(tbl)
            if 'config_name' not in cols:
                continue
            try:
                where = sql.SQL("WHERE run_id = %s") if (run_id and 'run_id' in cols) else sql.SQL("")
                params = (run_id,) if (run_id and 'run_id' in cols) else None
                query = sql.SQL(
                    "SELECT config_name FROM {tbl} {where} LIMIT 1"
                ).format(tbl=sql.Identifier(tbl), where=where)
                with self.db.get_cursor(commit=False) as cursor:
                    cursor.execute(query, params)
                    row = cursor.fetchone()
            except (errors.UndefinedTable, errors.UndefinedColumn):
                continue
            if not row or row[0] is None:
                continue
            m = re.search(r'(\d{8})', str(row[0]))
            if m:
                return pd.to_datetime(m.group(1), format='%Y%m%d').strftime('%Y-%m-%d')
        return None

    def _build_log_agg_cte(
        self,
        table_name: str,
        qty_alias: str,
        dedup: bool,
        run_id: Optional[str],
        sim_date_constant: Optional[str],
        params: List[Any],
    ) -> Any:
        """构造源表聚合公共表表达式；源表缺失时返回空结果。"""
        cols = self._get_table_columns(table_name)
        col_types = self._get_table_column_types(table_name)
        required = {"date", "material", "location", "quantity"}
        if not cols or not required.issubset(cols):
            return sql.SQL(
                """
                SELECT
                    NULL::timestamp AS date,
                    NULL::text AS material,
                    NULL::text AS location,
                    NULL::timestamp AS simulation_date,
                    0::double precision AS {qty_alias}
                WHERE FALSE
                """
            ).format(qty_alias=sql.Identifier(qty_alias))

        date_expr = self._timestamp_expr("date", col_types.get("date"))
        if sim_date_constant is not None:
            sim_expr = sql.SQL("%s::timestamp")
            params.append(sim_date_constant)
        elif "simulation_date" in cols:
            sim_expr = self._timestamp_expr(
                "simulation_date", col_types.get("simulation_date")
            )
        elif "sim_date" in cols:
            sim_expr = self._timestamp_expr("sim_date", col_types.get("sim_date"))
        else:
            sim_expr = sql.SQL("NULL::timestamp")

        select_items = [
            sql.SQL("{} AS date").format(date_expr),
            sql.SQL("material::text AS material"),
            sql.SQL("location::text AS location"),
            sql.SQL("{} AS simulation_date").format(sim_expr),
            sql.SQL("quantity::double precision AS quantity"),
        ]
        if dedup and "demand_type" in cols:
            select_items.append(sql.SQL("demand_type::text AS demand_type"))

        where_sql = sql.SQL("")
        if run_id and "run_id" in cols:
            where_sql = sql.SQL("WHERE run_id = %s")
            params.append(run_id)

        base_select = sql.SQL(
            """
            SELECT {select_items}
            FROM {table}
            {where}
            """
        ).format(
            select_items=sql.SQL(", ").join(select_items),
            table=sql.Identifier(table_name),
            where=where_sql,
        )
        if dedup:
            # 订单全局去重（跨 simulation_date）：每个唯一订单
            # (date, material, location, quantity[, demand_type]) 仅保留“首次出现”，
            # 即最早的 simulation_date。与本地 _generate_order_shipment_report.drop_seen_orders
            # 的全局 seen-set 口径一致（本地按日期升序处理文件，首次=最早 sim_date）。
            # 此前用 `DISTINCT *`（去重键含 simulation_date）会把同一订单在每个 sim_date 各留一份，
            # 导致 OSC 行数偏多（DB 58908 vs 本地 56128）。
            _partition_items = [
                sql.SQL("date"),
                sql.SQL("material"),
                sql.SQL("location"),
                sql.SQL("quantity"),
            ]
            if "demand_type" in cols:
                _partition_items.append(sql.SQL("demand_type"))
            base_select = sql.SQL(
                """
                SELECT date, material, location, simulation_date, quantity
                FROM (
                    SELECT raw.*, ROW_NUMBER() OVER (
                        PARTITION BY {partition}
                        ORDER BY simulation_date ASC NULLS LAST
                    ) AS _osc_rn
                    FROM ({base}) raw
                ) ranked
                WHERE _osc_rn = 1
                """
            ).format(
                partition=sql.SQL(", ").join(_partition_items),
                base=base_select,
            )

        return sql.SQL(
            """
            SELECT
                date,
                material,
                location,
                simulation_date,
                SUM(quantity) AS {qty_alias}
            FROM ({base}) src
            GROUP BY date, material, location, simulation_date
            """
        ).format(
            qty_alias=sql.Identifier(qty_alias),
            base=base_select,
        )

    def _stream_filter_to_summary(
        self,
        source_table: str,
        target_table: str,
        run_id: Optional[str],
        end_date_dt: Optional[pd.Timestamp],
        date_filter_cols: List[str],
        sort_cols: Optional[List[str]] = None,
        stable_tiebreaker: bool = False,
        if_exists: str = 'replace',
        chunksize: int = 100_000,
    ) -> int:
        """数据库侧下推过滤与排序，服务端游标按 chunksize 流式写入汇总表。

        - date_filter_cols 每列生成 ``(col IS NULL OR col <= end_date)``，并用 AND 串联，
          仅对源表实际存在的列下推（与旧实现的 ``if col in df.columns`` 防御性一致）。
        - sort_cols 仅取源表实际存在的列；stable_tiebreaker=True 时追加 ``ctid`` 末位
          作为稳定排序的补充键，等价旧实现的 ``kind='mergesort'`` 稳定排序。
        - 服务端游标 + ``ORDER BY`` 在 PostgreSQL 端全局排序后分发，每个分块已是全表顺序。
        - 首批写入用调用方 if_exists，其余批次强制 'append'。
        - 表不存在 / 无符合行 → 返回 0。
        """
        # 如果目标 Summary 表已经存在，需要先根据 if_exists 策略处理旧数据。
        if self.db.table_exists(target_table):
            # fail 模式表示目标表已存在时直接报错，不允许覆盖或追加。
            if if_exists == "fail":
                raise ValueError(f"Table {target_table} already exists")
            # replace 模式在这里不整表重建，而是优先清理当前 run_id 的旧数据。
            if if_exists == "replace":
                # 读取目标表字段，用于判断是否能按 run_id 精准删除。
                target_cols = self._get_table_columns(target_table)
                # 删除或清空目标表属于写操作，因此使用默认提交游标。
                with self.db.get_cursor() as cursor:
                    # 如果调用方传入 run_id 且目标表有 run_id 列，只删除当前运行旧数据。
                    if run_id and "run_id" in target_cols:
                        cursor.execute(
                            # 安全拼接目标表名，避免表名字符串直接进入 SQL。
                            sql.SQL("DELETE FROM {} WHERE run_id = %s").format(
                                sql.Identifier(target_table)
                            ),
                            # run_id 作为参数绑定，避免 SQL 注入和转义问题。
                            (run_id,),
                        )
                    # 如果没有 run_id，无法做运行级隔离，只能回退为清空整张目标表。
                    elif not run_id:
                        cursor.execute(
                            # TRUNCATE 只在没有 run_id 的历史兼容场景使用。
                            sql.SQL("TRUNCATE TABLE {}").format(
                                sql.Identifier(target_table)
                            )
                        )

        # 读取源表字段集合；源表不存在或无法读取时返回空集合。
        cols = self._get_table_columns(source_table)
        # 如果源表不存在或没有字段，说明没有可生成的 Summary 数据，直接返回 0。
        if not cols:
            return 0
        # 读取源表字段类型，用于构造日期过滤条件时区分时间类型和文本类型。
        col_types = self._get_table_column_types(source_table)

        # where_parts 保存 WHERE 子句中的每个条件片段。
        where_parts = []
        # params 保存 WHERE 条件中 %s 占位符对应的参数值。
        params: list = []
        # 如果传入 run_id 且源表有 run_id 列，则只读取当前运行的数据。
        if run_id and 'run_id' in cols:
            # 添加 run_id 等值过滤条件。
            where_parts.append(sql.SQL("run_id = %s"))
            # 记录 run_id 参数，顺序必须与 where_parts 中的 %s 一致。
            params.append(run_id)
        # 如果传入结束日期，则对调用方指定的日期字段逐个下推过滤。
        if end_date_dt is not None:
            # date_filter_cols 可能为空；为空时用 [] 避免循环报错。
            for col in date_filter_cols or []:
                # 只对源表真实存在的日期列加过滤，兼容不同版本表结构。
                if col in cols:
                    # 构造兼容 DATE/TIMESTAMP/TEXT 的日期过滤条件。
                    where_parts.append(
                        self._date_filter_condition(col, col_types.get(col))
                    )
                    # 每个日期过滤条件都需要一个 end_date_dt 参数。
                    params.append(end_date_dt)
        # 默认没有 WHERE 条件时使用空 SQL 片段。
        where_sql = sql.SQL("")
        # 如果存在任意过滤条件，则用 AND 串联成完整 WHERE 子句。
        if where_parts:
            where_sql = sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts)

        # 默认没有 ORDER BY 条件时使用空 SQL 片段。
        order_sql = sql.SQL("")
        # order_terms 保存 ORDER BY 后面的字段或表达式。
        order_terms: list = []
        # 如果调用方指定排序字段，只保留源表实际存在的字段。
        if sort_cols:
            order_terms = [sql.Identifier(c) for c in sort_cols if c in cols]
        # stable_tiebreaker=True 时追加 ctid，模拟 pandas mergesort 的稳定排序效果。
        if stable_tiebreaker:
            order_terms.append(sql.SQL("ctid"))
        # 如果存在排序项，则拼出完整 ORDER BY 子句。
        if order_terms:
            order_sql = sql.SQL("ORDER BY ") + sql.SQL(", ").join(order_terms)

        # 构造源表查询：读取源表全部字段，并下推 WHERE / ORDER BY 到 PostgreSQL。
        query = sql.SQL("SELECT * FROM {tbl} {where} {order}").format(
            # tbl 是来源模块输出表名，用 sql.Identifier 安全转义。
            tbl=sql.Identifier(source_table),
            # where 是前面构造的过滤片段；无过滤时为空。
            where=where_sql,
            # order 是前面构造的排序片段；无排序时为空。
            order=order_sql,
        )

        # total 统计最终写入目标 Summary 表的总行数。
        total = 0
        # is_first 标记当前是否为第一批 chunk；第一批使用调用方 if_exists。
        is_first = True
        try:
            # 使用服务端游标按 chunksize 分块读取，避免一次性把全量结果拉入内存。
            for chunk_df in self.db.iter_query_chunks(
                # query 是已经拼好的 SELECT * FROM source WHERE ... ORDER BY ...。
                query, tuple(params) if params else None, chunksize=chunksize
            ):
                # 空分块没有可写入数据，直接跳过。
                if chunk_df.empty:
                    continue
                # 与旧实现一致：若源表缺 run_id 列，则按 run_id 补一列
                # 这样目标 Summary 表仍然可以按当前运行进行隔离和清理。
                if run_id and 'run_id' not in chunk_df.columns:
                    chunk_df['run_id'] = run_id
                # 第一批使用调用方传入的 if_exists，后续批次统一 append，避免覆盖前面批次。
                chunk_if_exists = if_exists if is_first else 'append'
                # 复用 DataFrame 写库逻辑：自动建表、补列、写入 config_name / db_write_time。
                self.db.create_table_from_df(
                    # 当前分块数据。
                    chunk_df,
                    # 目标 Summary 表名。
                    target_table,
                    # 第一批 replace/append/fail，后续固定 append。
                    if_exists=chunk_if_exists,
                    # 写入当前配置名，用于区分不同配置输出。
                    config_name=self.config_name,
                )
                # 累加本批写入行数。
                total += len(chunk_df)
                # 第一批写入完成后，后续批次都必须追加。
                is_first = False
        # 源表或字段在查询过程中不存在时，按无数据处理，不中断整体 Summary 生成。
        except (errors.UndefinedTable, errors.UndefinedColumn):
            return 0

        # 如果成功写入了数据，记录到 written_tables，供上层日志汇总使用。
        if total > 0:
            self.written_tables[target_table] = {"module": "summary", "rows": total}
        # 返回本 Summary 子表实际写入行数。
        return total

    def _generate_order_shipment_cut_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成订单/发货/缺货汇总报告（数据库侧去重 + 聚合，避免全表加载）。"""
        # 汇总目标表 summary_output_ordershipmentcutsummary：用于保存订单、发货、缺货三类数量的日粒度汇总结果。
        table_name = "summary_output_ordershipmentcutsummary"

        # 三张源表：分别来自模块1输出的订单日志、发货日志、缺货日志。
        orders_tbl = "module1_output_orderlog"
        shipments_tbl = "module1_output_shipmentlog"
        cuts_tbl = "module1_output_cutlog"

        # 目标表结构定义：如果目标表不存在则按该结构创建；如果已存在则校验/对齐字段。
        column_defs = {
            # 仿真批次日期：与本地汇总对齐，通常从 config_name 中提取。
            "simulation_date": "TIMESTAMP",
            # 业务日期：订单/发货/缺货发生日期。
            "date": "TIMESTAMP",
            # 物料编码。
            "material": "TEXT",
            # 地点/节点编码。
            "location": "TEXT",
            # 订单汇总数量，使用大整型对齐安全整数语义。
            "order_qty": "BIGINT",
            # 发货汇总数量，使用大整型对齐安全整数语义。
            "shipment_qty": "BIGINT",
            # 缺货汇总数量，必须来自缺货日志聚合值。
            "cut_qty": "BIGINT",
            # 本次运行唯一标识，用于隔离不同运行的数据。
            "run_id": "TEXT",
            # 配置名称标签，只用于输出标识，不作为本段查询的聚合维度。
            "config_name": "TEXT",
            # 数据库写入时间。
            "db_write_time": "TIMESTAMP",
        }
        # 准备目标表：replace 时只清理当前 run_id 的旧汇总，避免影响其他运行。
        self._prepare_summary_target(table_name, column_defs, if_exists, run_id)

        # 从源表 config_name 中抽样提取 simulation_date，确保三张日志使用同一仿真日期口径。
        config_sim_date = self._sample_config_sim_date(
            run_id, [orders_tbl, shipments_tbl, cuts_tbl]
        )

        # 查询参数列表：_build_log_agg_cte 和主查询会按顺序追加占位符参数。
        params: List[Any] = []

        # 构造订单聚合公共表表达式：dedup=True，表示订单日志需要在数据库侧去重后再聚合。
        order_agg_sql = self._build_log_agg_cte(
            orders_tbl, "order_qty", True, run_id, config_sim_date, params
        )

        # 构造发货聚合公共表表达式：发货日志不做订单式去重，只按键汇总数量。
        shipment_agg_sql = self._build_log_agg_cte(
            shipments_tbl, "shipment_qty", False, run_id, config_sim_date, params
        )

        # 构造缺货聚合公共表表达式：缺货日志的数量聚合后作为最终 cut_qty 来源。
        cut_agg_sql = self._build_log_agg_cte(
            cuts_tbl, "cut_qty", False, run_id, config_sim_date, params
        )

        # 结束日期过滤条件默认为空；只有传入 end_date_dt 时才限制输出日期。
        end_filter_sql = sql.SQL("")
        if end_date_dt is not None:
            # 过滤合并集合的业务日期 date，保证三类日志统一按同一输出日期过滤。
            end_filter_sql = sql.SQL("WHERE date <= %s")
            params.append(end_date_dt)

        # 主插入查询最后两个参数：写入汇总行上的 run_id 与 config_name 标签。
        params.extend([run_id, self.config_name])

        # 构造完整查询：所有聚合、连接、插入都在 PostgreSQL 内完成，避免应用侧持有大数据帧。
        query = sql.SQL(
            """
            WITH
            -- 1) 在数据库内分别聚合三类源日志。
            --    _build_log_agg_cte 负责按 run_id 过滤、归一化 simulation_date、
            --    转换 quantity 类型，以及订单日志在需要时去重。
            --    order_agg 输出列：业务日期/物料/地点/仿真日期/订单数量。
            order_agg AS ({order_agg}),
            --    shipment_agg 输出列：业务日期/物料/地点/仿真日期/发货数量。
            shipment_agg AS ({shipment_agg}),
            --    cut_agg 输出列：业务日期/物料/地点/仿真日期/缺货数量。
            cut_agg AS ({cut_agg}),
            -- 2) 把三类聚合结果纵向合并为一张带三列数量的流，缺失的数量列填 0。
            --    用 UNION ALL（不去重）保留每个聚合的原值，随后由 GROUP BY 求和。
            --    每个键在 order_agg/shipment_agg/cut_agg 中各至多一行，因此
            --    GROUP BY 后 SUM(order_qty) 即等于该键的订单聚合值（或 0），
            --    与原先 COALESCE(单值,0) 完全等价；但避免了 IS NOT DISTINCT FROM
            --    联接退化成 O(n^2) 嵌套循环的性能问题。
            combined AS (
                SELECT date, material, location, simulation_date,
                       order_qty AS order_qty,
                       0::double precision AS shipment_qty,
                       0::double precision AS cut_qty
                FROM order_agg
                UNION ALL
                SELECT date, material, location, simulation_date,
                       0::double precision, shipment_qty, 0::double precision
                FROM shipment_agg
                UNION ALL
                SELECT date, material, location, simulation_date,
                       0::double precision, 0::double precision, cut_qty
                FROM cut_agg
            ),
            summary_rows AS (
                -- 3) 按键 hash 聚合；GROUP BY 天然把 NULL 键归为同一组，
                --    等价原先 keys UNION + IS NOT DISTINCT FROM 的空值安全语义。
                --    数量列向零截断为大整型，对齐 safe_int / int 的本地语义。
                --    业务口径：cut_qty 仍取缺货日志聚合值，不由 order-shipment 重算。
                SELECT
                    simulation_date,
                    date,
                    material,
                    location,
                    -- 订单数量：求和后小数向零截断，再转大整型。
                    TRUNC(SUM(order_qty))::bigint AS order_qty,
                    -- 发货数量：求和后小数向零截断，再转大整型。
                    TRUNC(SUM(shipment_qty))::bigint AS shipment_qty,
                    -- 缺货数量：缺货日志聚合值求和后截断为大整型。
                    TRUNC(SUM(cut_qty))::bigint AS cut_qty
                FROM combined
                -- 动态日期过滤条件；无结束日期时这里为空查询片段。
                {end_filter}
                GROUP BY simulation_date, date, material, location
            )
            -- 4) 聚合结果直接在 PostgreSQL 内写入目标表，避免应用侧物化大数据帧。
            INSERT INTO {target} (
                -- 输出列：仿真日期。
                simulation_date,
                -- 输出列：业务日期。
                date,
                -- 输出列：物料。
                material,
                -- 输出列：地点。
                location,
                -- 输出列：订单数量。
                order_qty,
                -- 输出列：发货数量。
                shipment_qty,
                -- 输出列：缺货数量。
                cut_qty,
                -- 输出列：当前运行标识。
                run_id,
                -- 输出列：当前配置名称。
                config_name,
                -- 输出列：写入时间。
                db_write_time
            )
            SELECT
                -- 从 summary_rows 取仿真日期。
                simulation_date,
                -- 从 summary_rows 取业务日期。
                date,
                -- 从 summary_rows 取物料。
                material,
                -- 从 summary_rows 取地点。
                location,
                -- 从 summary_rows 取订单数量。
                order_qty,
                -- 从 summary_rows 取发货数量。
                shipment_qty,
                -- 从 summary_rows 取缺货数量。
                cut_qty,
                -- 写入本次运行标识；注意 run_id 是隔离维度，但不是连接键。
                %s AS run_id,
                -- 写入配置名称标签；仅用于标识输出来源。
                %s AS config_name,
                -- 使用数据库当前时间作为写入时间。
                NOW() AS db_write_time
            FROM summary_rows
            -- 稳定输出顺序，便于对比、排查和回归测试。
            ORDER BY simulation_date, date, material, location
            """
        ).format(
            # 注入订单聚合公共表表达式。
            order_agg=order_agg_sql,
            # 注入发货聚合公共表表达式。
            shipment_agg=shipment_agg_sql,
            # 注入缺货聚合公共表表达式。
            cut_agg=cut_agg_sql,
            # 注入结束日期动态过滤片段。
            end_filter=end_filter_sql,
            # 注入目标表名，并通过 sql.Identifier 防止查询标识符拼接风险。
            target=sql.Identifier(table_name),
        )

        try:
            # 执行整条查询：PostgreSQL 内完成聚合并直接插入。
            with self.db.get_cursor() as cursor:
                cursor.execute(query, tuple(params))
                # cursor.rowcount 表示插入影响行数；若数据库返回 None，则按 0 处理。
                rows = max(cursor.rowcount or 0, 0)
        except (errors.UndefinedTable, errors.UndefinedColumn):
            # 兼容源表或字段不存在的场景：该汇总子表返回 0 行，不中断整个流程。
            return 0

        if rows > 0:
            # 记录本次汇总写入结果，供上层汇总日志使用。
            self.written_tables[table_name] = {"module": "summary", "rows": rows}
        return rows

    def _generate_changeover_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成换产汇总报告（数据库内 INSERT SELECT 透传）"""
        return self._insert_select_to_summary(
            source_table="module4_output_changeoverlog",
            target_table="summary_output_fullchangeoverlog",
            run_id=run_id,
            end_date_dt=end_date_dt,
            # 与本地版本一致：只在 changeover_end_date 上过滤
            date_filter_cols=['changeover_end_date'],
            sort_cols=[
                'date', 'material', 'sending', 'receiving',
                'planned_delivery_date', 'demand_element',
                'demand_qty', 'planned_qty', 'deployed_qty_invcon',
                'deploy_qty_with_plan_order', 'deploy_from_in_transit',
                'deploy_from_open_deployment_inbound',
                'deploy_from_future_production', 'deployed_qty',
                'leadtime', 'orig_location', 'is_cross_node', 'quota',
            ],
            stable_tiebreaker=True,  # 等价旧实现的 kind='mergesort'
            if_exists=if_exists,
        )
    
    def _generate_capacity_exceed_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成产能超限汇总报告（数据库内 INSERT SELECT 透传）"""
        return self._insert_select_to_summary(
            source_table="module4_output_capacityexceed",
            target_table="summary_output_fullcapacityexceed",
            run_id=run_id,
            end_date_dt=end_date_dt,
            date_filter_cols=['date'],
            if_exists=if_exists,
        )
    
    def _generate_production_plan_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成生产计划汇总报告（数据库内 INSERT SELECT 透传）"""
        return self._insert_select_to_summary(
            source_table="module4_output_productionplan",
            target_table="summary_output_fullproductionplan",
            run_id=run_id,
            end_date_dt=end_date_dt,
            date_filter_cols=['available_date'],
            if_exists=if_exists,
        )
    
    def _generate_deployment_plan_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成部署计划汇总报告（数据库内 INSERT SELECT 透传）"""
        return self._insert_select_to_summary(
            source_table="module5_output_deploymentplan",
            target_table="summary_output_fulldeploymentplan",
            run_id=run_id,
            end_date_dt=end_date_dt,
            # 与本地版本 _generate_deployment_report 保持一致
            date_filter_cols=['deployment_date', 'arrival_date', 'ship_date', 'date'],
            if_exists=if_exists,
        )
    
    def _generate_delivery_plan_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成交付计划汇总报告（数据库内 INSERT SELECT 透传）"""
        # 与本地版本 _generate_delivery_report 保持一致：仅在
        # planned_deploy_date / actual_ship_date 上过滤（DB 表列名是
        # planned_deployment_date，与 planned_deploy_date 不匹配，过滤实际仅在
        # actual_ship_date 生效——保留两列以兼容历史结构）
        return self._insert_select_to_summary(
            source_table="module6_output_deliveryplan",
            target_table="summary_output_fulldeliveryplan",
            run_id=run_id,
            end_date_dt=end_date_dt,
            date_filter_cols=['planned_deploy_date', 'actual_ship_date'],
            if_exists=if_exists,
        )
    
    def _generate_truck_usage_summary(
        self,
        run_id: str,
        end_date_dt: pd.Timestamp,
        if_exists: str
    ) -> int:
        """生成卡车使用汇总报告（数据库内 INSERT SELECT 透传）"""
        return self._insert_select_to_summary(
            source_table="module6_output_truckusagelog",
            target_table="summary_output_fulltruckusage",
            run_id=run_id,
            end_date_dt=end_date_dt,
            date_filter_cols=['date'],
            if_exists=if_exists,
        )


def write_run_data_to_db(
    run_output_dir: str,
    db_host: Optional[str] = None,
    db_port: Optional[int] = None,
    db_name: Optional[str] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None
) -> bool:
    """
    将运行输出数据写入数据库

    未显式传入的字段将从 ``config/defaults.yaml`` 的 ``database:`` 节点读取。

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
        raise RuntimeError(f"数据库连接失败: {conn_result.get('message')}")
    
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
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        write_run_data_to_db(sys.argv[1])
    else:
        print("用法: python module_data_writer.py <run_output_dir>")
