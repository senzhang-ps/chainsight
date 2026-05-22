"""汇总报告生成器

在全周期仿真结束后，按照约定的目录结构输出 7 类完整报告：
订单发运切分、产能超限、换型、调拨计划、生产计划、交付计划、车辆使用与历史库存。
所有报告基于 Orchestrator 内存状态或既有 CSV/XLSX 落盘文件生成。
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import glob

from src.utils.normalization import normalize_material, normalize_location

logger = logging.getLogger("SupplyChainSimulation." + __name__)

class SummaryReportGenerator:
    """汇总报告生成器"""
    
    LOCAL_SUMMARY_EXCEL_ROW_LIMIT = 1_000_000

    def __init__(self, output_base_dir: str, config_dict: dict = None):
        """
        初始化汇总报告生成器
        
        Args:
            output_base_dir: 输出基础目录
            config_dict: 配置数据字典（可选，用于获取safety stock等配置信息）
        """
        self.output_base_dir = Path(output_base_dir)
        self.summary_dir = self.output_base_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.config_dict = config_dict or {}
        
        # 各模块输出目录
        self.module_dirs = {
            'module1': self.output_base_dir / "module1",
            'module3': self.output_base_dir / "module3",
            'module4': self.output_base_dir / "module4",
            'module5': self.output_base_dir / "module5",
            'module6': self.output_base_dir / "module6",
            'orchestrator': self.output_base_dir / "orchestrator"
        }
    
    @staticmethod
    def _normalize_material_value(material_str: str) -> str:
        """标准化 material 值：移除.0后缀 (薄封装,统一委托 src.utils.normalization)。"""
        return normalize_material(material_str, treat_missing_tokens=True)

    @staticmethod
    def _normalize_location_value(location_str: str) -> str:
        """标准化 location 值：纯数字补齐为4位，字母数字保持原样 (薄封装,统一委托)。"""
        return normalize_location(
            location_str, mode="numeric_only", treat_missing_tokens=True
        )
    
    def generate_all_reports(self, start_date: str, end_date: str) -> Dict[str, str]:
        """
        生成所有汇总报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, str]: 报告名称到文件路径的映射
        """
        logger.info(f"开始生成汇总报告 ({start_date} 到 {end_date})")
        
        # 保存日期范围，用于后续过滤
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        report_files = {}
        
        # 收集所有天的数据文件
        daily_files = self._collect_daily_files(start_date, end_date)
        
        # 生成各个汇总报告
        report_files['order_shipment_cut'] = self._generate_order_shipment_report(daily_files)
        report_files['exceed_capacity'] = self._generate_capacity_exceed_report(daily_files)
        report_files['changeover'] = self._generate_changeover_report(daily_files)
        report_files['deployment_plan'] = self._generate_deployment_report(daily_files)
        report_files['production_plan'] = self._generate_production_report(daily_files)
        report_files['delivery_plan'] = self._generate_delivery_report(daily_files)
        report_files['truck_usage'] = self._generate_truck_usage_report(daily_files)
        report_files['historical_inventory'] = self._generate_historical_inventory_report(start_date, end_date)
        
        logger.info(f"汇总报告生成完成，输出目录: {self.summary_dir}")
        return report_files
    
    def _collect_daily_files(self, start_date: str, end_date: str) -> Dict[str, List[str]]:
        """收集指定日期范围内的所有模块输出文件"""
        date_range = pd.date_range(start_date, end_date, freq='D')
        daily_files = {module: [] for module in self.module_dirs.keys()}
        
        for date in date_range:
            date_str = date.strftime('%Y%m%d')
            
            # Module1 输出文件
            m1_file = self.module_dirs['module1'] / f"module1_output_{date_str}.xlsx"
            if m1_file.exists():
                daily_files['module1'].append(str(m1_file))
            
            # Module3 输出文件
            m3_file = self.module_dirs['module3'] / f"Module3Output_{date_str}.xlsx"
            if m3_file.exists():
                daily_files['module3'].append(str(m3_file))
            
            # Module4 输出文件
            m4_file = self.module_dirs['module4'] / f"Module4Output_{date_str}.xlsx"
            if m4_file.exists():
                daily_files['module4'].append(str(m4_file))
            
            # Module5 输出文件
            m5_file = self.module_dirs['module5'] / f"Module5Output_{date_str}.xlsx"
            if m5_file.exists():
                daily_files['module5'].append(str(m5_file))
            
            # Module6 输出文件
            m6_file = self.module_dirs['module6'] / f"Module6Output_{date_str}.xlsx"
            if m6_file.exists():
                daily_files['module6'].append(str(m6_file))
        
        return daily_files

    def _count_excel_sheet_rows(self, file_paths: List[str], sheet_names: List[str]) -> Optional[int]:
        """Count worksheet data rows without loading sheet values into pandas."""
        total = 0
        try:
            from openpyxl import load_workbook
        except Exception as e:
            logger.warning(f"Unable to import openpyxl for summary row counting: {e}")
            return None

        for file_path in file_paths:
            try:
                workbook = load_workbook(file_path, read_only=True, data_only=True)
                try:
                    for sheet_name in sheet_names:
                        if sheet_name not in workbook.sheetnames:
                            continue
                        worksheet = workbook[sheet_name]
                        total += max((worksheet.max_row or 0) - 1, 0)
                finally:
                    workbook.close()
            except Exception as e:
                logger.warning(f"Unable to count rows in {file_path}: {e}")
                return None
        return total

    def _should_write_summary_csv(self, input_rows: Optional[int], output_rows: int) -> bool:
        limit = self.LOCAL_SUMMARY_EXCEL_ROW_LIMIT
        return input_rows is None or input_rows > limit or output_rows > limit

    def _summary_output_paths(self, stem: str) -> tuple[Path, Path]:
        return (
            self.summary_dir / f"{stem}.xlsx",
            self.summary_dir / f"{stem}.csv",
        )

    def _remove_stale_summary_file(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            logger.warning(f"Unable to remove stale summary file {path}: {e}")

    def _write_summary_output(
        self,
        df: pd.DataFrame,
        stem: str,
        sheet_name: str,
        input_rows: Optional[int],
        columns: Optional[List[str]] = None,
    ) -> str:
        """Write xlsx for small local summaries and csv for large or unknown data."""
        if df is None:
            df = pd.DataFrame(columns=columns or [])
        if columns:
            for column in columns:
                if column not in df.columns:
                    df[column] = pd.Series(dtype='object')
            df = df[columns]

        xlsx_path, csv_path = self._summary_output_paths(stem)
        if self._should_write_summary_csv(input_rows, len(df)):
            self._remove_stale_summary_file(xlsx_path)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(
                f"Summary {stem}: input_rows={input_rows}, output_rows={len(df)}, format=csv"
            )
            return str(csv_path)

        self._remove_stale_summary_file(csv_path)
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(
            f"Summary {stem}: input_rows={input_rows}, output_rows={len(df)}, format=xlsx"
        )
        return str(xlsx_path)

    def _read_excel_sheet(
        self,
        file_path: str,
        sheet_name: str,
        usecols: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        with pd.ExcelFile(file_path) as xl:
            if sheet_name not in xl.sheet_names:
                return None
            if usecols:
                wanted = set(usecols)
                return xl.parse(sheet_name, usecols=lambda col: col in wanted)
            return xl.parse(sheet_name)

    def _apply_end_date_filters(
        self,
        df: pd.DataFrame,
        parse_cols: List[str],
        filter_cols: List[str],
    ) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for col in parse_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if not filter_cols:
            return df

        mask = pd.Series(True, index=df.index)
        for col in filter_cols:
            if col in df.columns:
                mask = mask & (df[col].isna() | (df[col] <= self.end_date))
        return df[mask]

    def _prepare_tabular_summary_frame(
        self,
        df: pd.DataFrame,
        file_path: str,
        parse_cols: List[str],
        filter_cols: List[str],
        file_date_col: Optional[str] = None,
        reorder_date_first: bool = False,
        sort_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if file_date_col:
            df[file_date_col] = self._extract_date_from_filename(file_path)
        df = self._apply_end_date_filters(df, parse_cols, filter_cols)
        if reorder_date_first and 'date' in df.columns:
            ordered_columns = ['date'] + [col for col in df.columns if col != 'date']
            df = df[ordered_columns]
        if sort_cols:
            actual_sort_cols = [col for col in sort_cols if col in df.columns]
            if actual_sort_cols:
                df = df.sort_values(by=actual_sort_cols, kind='mergesort').reset_index(drop=True)
        return df

    def _append_summary_csv(self, df: pd.DataFrame, csv_path: Path, write_header: bool) -> None:
        encoding = 'utf-8-sig' if write_header else 'utf-8'
        df.to_csv(
            csv_path,
            index=False,
            mode='w' if write_header else 'a',
            header=write_header,
            encoding=encoding,
        )

    def _stream_tabular_summary_to_csv(
        self,
        files: List[str],
        source_sheet: str,
        stem: str,
        empty_columns: List[str],
        parse_cols: List[str],
        filter_cols: List[str],
        file_date_col: Optional[str] = None,
        reorder_date_first: bool = False,
        sort_cols: Optional[List[str]] = None,
        input_rows: Optional[int] = None,
    ) -> str:
        xlsx_path, csv_path = self._summary_output_paths(stem)
        self._remove_stale_summary_file(xlsx_path)
        self._remove_stale_summary_file(csv_path)

        wrote_header = False
        output_rows = 0
        for file_path in files:
            try:
                df = self._read_excel_sheet(file_path, source_sheet)
                if df is None or df.empty:
                    continue
                df = self._prepare_tabular_summary_frame(
                    df=df,
                    file_path=file_path,
                    parse_cols=parse_cols,
                    filter_cols=filter_cols,
                    file_date_col=file_date_col,
                    reorder_date_first=reorder_date_first,
                    sort_cols=sort_cols,
                )
                if df.empty:
                    continue
                self._append_summary_csv(df, csv_path, write_header=not wrote_header)
                wrote_header = True
                output_rows += len(df)
            except Exception as e:
                logger.warning(f"Failed to stream {source_sheet} from {file_path}: {e}")

        if not wrote_header:
            pd.DataFrame(columns=empty_columns).to_csv(
                csv_path, index=False, encoding='utf-8-sig'
            )
        logger.info(
            f"Summary {stem}: input_rows={input_rows}, output_rows={output_rows}, format=csv"
        )
        return str(csv_path)

    def _generate_tabular_summary_report(
        self,
        daily_files: Dict,
        module_name: str,
        source_sheet: str,
        stem: str,
        output_sheet: str,
        empty_columns: List[str],
        parse_cols: Optional[List[str]] = None,
        filter_cols: Optional[List[str]] = None,
        file_date_col: Optional[str] = None,
        reorder_date_first: bool = False,
        sort_cols: Optional[List[str]] = None,
    ) -> str:
        files = daily_files.get(module_name, [])
        parse_cols = parse_cols or []
        filter_cols = filter_cols or []
        input_rows = self._count_excel_sheet_rows(files, [source_sheet])
        if input_rows is None or input_rows > self.LOCAL_SUMMARY_EXCEL_ROW_LIMIT:
            return self._stream_tabular_summary_to_csv(
                files=files,
                source_sheet=source_sheet,
                stem=stem,
                empty_columns=empty_columns,
                parse_cols=parse_cols,
                filter_cols=filter_cols,
                file_date_col=file_date_col,
                reorder_date_first=reorder_date_first,
                sort_cols=sort_cols,
                input_rows=input_rows,
            )

        frames = []
        for file_path in files:
            try:
                df = self._read_excel_sheet(file_path, source_sheet)
                if df is None or df.empty:
                    continue
                df = self._prepare_tabular_summary_frame(
                    df=df,
                    file_path=file_path,
                    parse_cols=parse_cols,
                    filter_cols=filter_cols,
                    file_date_col=file_date_col,
                    reorder_date_first=reorder_date_first,
                    sort_cols=None,
                )
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if sort_cols:
                actual_sort_cols = [col for col in sort_cols if col in combined.columns]
                if actual_sort_cols:
                    combined = combined.sort_values(
                        by=actual_sort_cols,
                        kind='mergesort',
                    ).reset_index(drop=True)
        else:
            combined = pd.DataFrame(columns=empty_columns)

        return self._write_summary_output(
            combined,
            stem=stem,
            sheet_name=output_sheet,
            input_rows=input_rows,
            columns=None if frames else empty_columns,
        )

    def _generate_order_shipment_report(self, daily_files: Dict) -> str:
        """Generate order/shipment/cut summary with bounded memory."""
        column_order = [
            'simulation_date', 'date', 'material', 'location',
            'order_qty', 'shipment_qty', 'cut_qty',
        ]
        files = daily_files.get('module1', [])
        input_rows = self._count_excel_sheet_rows(files, ['OrderLog', 'ShipmentLog', 'CutLog'])

        order_qty: Dict[tuple, float] = {}
        shipment_qty: Dict[tuple, float] = {}
        cut_keys = set()

        def accumulate_quantity(df: pd.DataFrame, simulation_date: str, target: Dict[tuple, float]) -> None:
            required = {'date', 'material', 'location', 'quantity'}
            if df is None or df.empty or not required.issubset(df.columns):
                return
            work = df[['date', 'material', 'location', 'quantity']].copy()
            work['date'] = pd.to_datetime(work['date'], errors='coerce')
            work = work[work['date'].notna() & (work['date'] <= self.end_date)]
            if work.empty:
                return
            work['quantity'] = pd.to_numeric(work['quantity'], errors='coerce').fillna(0)
            grouped = work.groupby(
                ['date', 'material', 'location'],
                dropna=False,
            )['quantity'].sum()
            for (date_value, material, location), qty in grouped.items():
                key = (simulation_date, date_value, material, location)
                target[key] = target.get(key, 0) + float(qty)

        for file_path in files:
            try:
                simulation_date = self._extract_date_from_filename(file_path)
                with pd.ExcelFile(file_path) as xl:
                    if 'OrderLog' in xl.sheet_names:
                        orders_df = xl.parse('OrderLog')
                        if not orders_df.empty:
                            orders_df['simulation_date'] = simulation_date
                            dedup_cols = [
                                'date', 'material', 'location', 'quantity', 'simulation_date'
                            ]
                            if 'demand_type' in orders_df.columns:
                                dedup_cols.append('demand_type')
                            dedup_cols = [col for col in dedup_cols if col in orders_df.columns]
                            orders_df = orders_df.drop_duplicates(subset=dedup_cols, keep='first')
                            accumulate_quantity(orders_df, simulation_date, order_qty)

                    if 'ShipmentLog' in xl.sheet_names:
                        shipments_df = xl.parse('ShipmentLog')
                        accumulate_quantity(shipments_df, simulation_date, shipment_qty)

                    if 'CutLog' in xl.sheet_names:
                        cuts_df = xl.parse('CutLog')
                        cut_qty: Dict[tuple, float] = {}
                        accumulate_quantity(cuts_df, simulation_date, cut_qty)
                        cut_keys.update(cut_qty.keys())
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        all_keys = set(order_qty.keys()) | set(shipment_qty.keys()) | cut_keys
        records = []
        for key in sorted(all_keys, key=lambda item: tuple(str(part) for part in item)):
            order_value = int(order_qty.get(key, 0))
            shipment_value = int(shipment_qty.get(key, 0))
            records.append({
                'simulation_date': key[0],
                'date': key[1],
                'material': key[2],
                'location': key[3],
                'order_qty': order_value,
                'shipment_qty': shipment_value,
                'cut_qty': max(order_value - shipment_value, 0),
            })

        summary = pd.DataFrame.from_records(records, columns=column_order)
        if not summary.empty:
            summary = summary.sort_values(
                ['simulation_date', 'date', 'material', 'location']
            ).reset_index(drop=True)

        return self._write_summary_output(
            summary,
            stem='full_order_shipment_cut_report',
            sheet_name='OrderShipmentCutSummary',
            input_rows=input_rows,
            columns=column_order,
        )

    def _generate_delivery_report(self, daily_files: Dict) -> str:
        return self._generate_tabular_summary_report(
            daily_files=daily_files,
            module_name='module6',
            source_sheet='DeliveryPlan',
            stem='full_delivery_plan_report',
            output_sheet='FullDeliveryPlan',
            empty_columns=[
                'date', 'material', 'sending', 'receiving', 'planned_qty',
                'delivered_qty', 'planned_deploy_date', 'actual_ship_date',
            ],
            parse_cols=['planned_deploy_date', 'actual_ship_date', 'date'],
            filter_cols=['planned_deploy_date', 'actual_ship_date'],
            file_date_col='date',
            reorder_date_first=True,
        )

    def _generate_truck_usage_report(self, daily_files: Dict) -> str:
        return self._generate_tabular_summary_report(
            daily_files=daily_files,
            module_name='module6',
            source_sheet='TruckUsageLog',
            stem='full_truck_usage_report',
            output_sheet='FullTruckUsage',
            empty_columns=[
                'date', 'sending', 'receiving', 'truck_type',
                'available_trucks', 'used_trucks', 'wfr', 'vfr',
            ],
            parse_cols=['date'],
            filter_cols=['date'],
        )

    def _generate_capacity_exceed_report(self, daily_files: Dict) -> str:
        return self._generate_tabular_summary_report(
            daily_files=daily_files,
            module_name='module4',
            source_sheet='CapacityExceed',
            stem='full_exceed_capacity_report',
            output_sheet='FullCapacityExceed',
            empty_columns=['date', 'location', 'material', 'capacity', 'demand', 'exceed_qty'],
            parse_cols=['date'],
            filter_cols=['date'],
        )

    def _generate_changeover_report(self, daily_files: Dict) -> str:
        return self._generate_tabular_summary_report(
            daily_files=daily_files,
            module_name='module4',
            source_sheet='ChangeoverLog',
            stem='full_changeover_report',
            output_sheet='FullChangeoverLog',
            empty_columns=[
                'changeover_start_date', 'changeover_end_date',
                'location', 'from_material', 'to_material',
            ],
            parse_cols=['changeover_start_date', 'changeover_end_date', 'date'],
            filter_cols=['changeover_end_date'],
        )

    def _generate_deployment_report(self, daily_files: Dict) -> str:
        sort_cols = [
            'date', 'material', 'sending', 'receiving',
            'planned_delivery_date', 'demand_element',
            'demand_qty', 'planned_qty', 'deployed_qty_invCon',
            'deploy_qty_with_plan_order', 'deploy_from_in_transit',
            'deploy_from_open_deployment_inbound',
            'deploy_from_future_production', 'deployed_qty',
            'leadtime', 'orig_location', 'is_cross_node', 'quota',
        ]
        return self._generate_tabular_summary_report(
            daily_files=daily_files,
            module_name='module5',
            source_sheet='DeploymentPlan',
            stem='full_deployment_plan_report',
            output_sheet='FullDeploymentPlan',
            empty_columns=['deployment_date', 'material', 'sending', 'receiving', 'quantity'],
            parse_cols=['deployment_date', 'arrival_date', 'ship_date', 'date'],
            filter_cols=['deployment_date', 'arrival_date', 'ship_date', 'date'],
            sort_cols=sort_cols,
        )

    def _generate_production_report(self, daily_files: Dict) -> str:
        return self._generate_tabular_summary_report(
            daily_files=daily_files,
            module_name='module4',
            source_sheet='ProductionPlan',
            stem='full_production_plan_report',
            output_sheet='FullProductionPlan',
            empty_columns=['available_date', 'material', 'location', 'quantity'],
            parse_cols=['available_date', 'production_plan_date'],
            filter_cols=['available_date'],
        )

    def _extract_date_from_filename(self, file_path: str) -> str:
        """从文件名中提取日期"""
        import re
        match = re.search(r'(\d{8})', file_path)
        if match:
            date_str = match.group(1)
            return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')
        return None
    
    def _generate_historical_inventory_report(self, start_date: str, end_date: str) -> str:
        """生成历史库存记录CSV报告"""
        output_file = self.summary_dir / "historical_inventory_record.csv"
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        orchestrator_dir = self.module_dirs['orchestrator']
        module1_dir = self.module_dirs['module1']
        
        # 读取safety stock配置
        safety_stock_dict = {}
        if 'M3_SafetyStock' in self.config_dict and not self.config_dict['M3_SafetyStock'].empty:
            ss_df = self.config_dict['M3_SafetyStock'].copy()
            # 确保日期格式
            if 'date' in ss_df.columns:
                ss_df['date'] = pd.to_datetime(ss_df['date'])
            for _, row in ss_df.iterrows():
                material = self._normalize_material_value(str(row['material']))
                location = self._normalize_location_value(str(row['location']))
                key = (material, location)
                # 使用最新的safety stock值（如果有多个日期）
                raw_ss_qty = row.get('safety_stock_qty', 0)
                if pd.isna(raw_ss_qty):
                    raw_ss_qty = 0
                safety_stock_dict[key] = int(float(raw_ss_qty))
        
        all_records = []
        
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            date_str_file = date.strftime('%Y%m%d')
            
            # 初始化当日数据容器
            ending_inv_dict = {}
            in_transit_dict = {}
            production_gr_dict = {}
            delivery_gr_dict = {}
            order_dict = {}
            shipment_dict = {}
            delivery_ship_dict = {}
            supply_demand_dict = {}
            
            # 1. 读取期末库存 (unrestricted_inventory)
            inv_file = orchestrator_dir / f"unrestricted_inventory_{date_str_file}.csv"
            if inv_file.exists():
                try:
                    inv_df = pd.read_csv(inv_file)
                    for _, row in inv_df.iterrows():
                        # 🔧 确保使用标准化的 material 和 location
                        material = self._normalize_material_value(str(row['material']))
                        location = self._normalize_location_value(str(row['location']))
                        key = (material, location)
                        ending_inv_dict[key] = int(row['quantity'])
                except Exception as e:
                    logger.warning(f"Failed to read {inv_file}: {e}")
            
            # 2. 读取在途库存 (planning_intransit)
            intransit_file = orchestrator_dir / f"planning_intransit_{date_str_file}.csv"
            if intransit_file.exists():
                try:
                    intransit_df = pd.read_csv(intransit_file)
                    if not intransit_df.empty:
                        # 按接收地点汇总在途数量
                        for _, row in intransit_df.iterrows():
                            material = self._normalize_material_value(str(row['material']))
                            location = self._normalize_location_value(str(row['receiving']))
                            key = (material, location)
                            in_transit_dict[key] = in_transit_dict.get(key, 0) + int(row['quantity'])
                except Exception as e:
                    logger.warning(f"Failed to read {intransit_file}: {e}")
            
            # 3. 读取生产入库 (production_gr)
            prod_gr_file = orchestrator_dir / f"production_gr_{date_str_file}.csv"
            if prod_gr_file.exists():
                try:
                    prod_gr_df = pd.read_csv(prod_gr_file)
                    if not prod_gr_df.empty:
                        for _, row in prod_gr_df.iterrows():
                            material = self._normalize_material_value(str(row['material']))
                            location = self._normalize_location_value(str(row['location']))
                            key = (material, location)
                            production_gr_dict[key] = production_gr_dict.get(key, 0) + int(row['quantity'])
                except Exception as e:
                    logger.warning(f"Failed to read {prod_gr_file}: {e}")
            
            # 4. 读取配送入库 (delivery_gr)
            del_gr_file = orchestrator_dir / f"delivery_gr_{date_str_file}.csv"
            if del_gr_file.exists():
                try:
                    del_gr_df = pd.read_csv(del_gr_file)
                    if not del_gr_df.empty:
                        for _, row in del_gr_df.iterrows():
                            material = self._normalize_material_value(str(row['material']))
                            location = self._normalize_location_value(str(row['receiving']))
                            key = (material, location)
                            delivery_gr_dict[key] = delivery_gr_dict.get(key, 0) + int(row['quantity'])
                except Exception as e:
                    logger.warning(f"Failed to read {del_gr_file}: {e}")
            
            # 5. 读取订单 (order from module1)
            order_file = module1_dir / f"module1_output_{date_str_file}.xlsx"
            if order_file.exists():
                try:
                    xl = pd.ExcelFile(order_file)
                    if 'OrderLog' in xl.sheet_names:
                        order_df = xl.parse('OrderLog')
                        # 只统计当日到期的订单
                        today_orders = order_df[pd.to_datetime(order_df['date']) == date]
                        if not today_orders.empty:
                            for _, row in today_orders.iterrows():
                                material = self._normalize_material_value(str(row['material']))
                                location = self._normalize_location_value(str(row['location']))
                                key = (material, location)
                                order_dict[key] = order_dict.get(key, 0) + int(row['quantity'])
                    
                    # 读取供需日志 (supply demand log)
                    if 'SupplyDemandLog' in xl.sheet_names:
                        sd_df = xl.parse('SupplyDemandLog')
                        if not sd_df.empty and 'date' in sd_df.columns:
                            # 只统计当日的供需数据
                            today_sd = sd_df[pd.to_datetime(sd_df['date']) == date]
                            if not today_sd.empty:
                                for _, row in today_sd.iterrows():
                                    material = self._normalize_material_value(str(row['material']))
                                    location = self._normalize_location_value(str(row['location']))
                                    key = (material, location)
                                    # 汇总所有demand_element的quantity
                                    supply_demand_dict[key] = supply_demand_dict.get(key, 0) + int(row.get('quantity', 0))
                except Exception as e:
                    logger.warning(f"Failed to read orders/supply-demand from {order_file}: {e}")
            
            # 6. 读取发货 (shipment from orchestrator)
            shipment_file = orchestrator_dir / f"shipment_log_{date_str_file}.csv"
            if shipment_file.exists():
                try:
                    shipment_df = pd.read_csv(shipment_file)
                    if not shipment_df.empty:
                        for _, row in shipment_df.iterrows():
                            material = self._normalize_material_value(str(row['material']))
                            location = self._normalize_location_value(str(row['location']))
                            key = (material, location)
                            shipment_dict[key] = shipment_dict.get(key, 0) + int(row['quantity'])
                except Exception as e:
                    logger.warning(f"Failed to read {shipment_file}: {e}")
            
            # 7. 读取配送发货 (delivery_shipment from orchestrator)
            del_ship_file = orchestrator_dir / f"delivery_shipment_log_{date_str_file}.csv"
            if del_ship_file.exists():
                try:
                    del_ship_df = pd.read_csv(del_ship_file)
                    if not del_ship_df.empty:
                        for _, row in del_ship_df.iterrows():
                            material = self._normalize_material_value(str(row['material']))
                            location = self._normalize_location_value(str(row['sending']))
                            key = (material, location)
                            delivery_ship_dict[key] = delivery_ship_dict.get(key, 0) + int(row['quantity'])
                except Exception as e:
                    logger.warning(f"Failed to read {del_ship_file}: {e}")
            
            # 整合所有数据 - 以期末库存为基准，包含所有出现过的 (material, location)
            all_keys = set()
            all_keys.update(ending_inv_dict.keys())
            all_keys.update(in_transit_dict.keys())
            all_keys.update(production_gr_dict.keys())
            all_keys.update(delivery_gr_dict.keys())
            all_keys.update(order_dict.keys())
            all_keys.update(shipment_dict.keys())
            all_keys.update(delivery_ship_dict.keys())
            all_keys.update(supply_demand_dict.keys())
            all_keys.update(safety_stock_dict.keys())
            
            for material, location in sorted(all_keys):
                record = {
                    'date': date_str,
                    'material': material,
                    'location': location,
                    'ending_inventory': ending_inv_dict.get((material, location), 0),
                    'in_transit': in_transit_dict.get((material, location), 0),
                    'production_gr': production_gr_dict.get((material, location), 0),
                    'delivery_gr': delivery_gr_dict.get((material, location), 0),
                    'order': order_dict.get((material, location), 0),
                    'shipment': shipment_dict.get((material, location), 0),
                    'delivery_ship': delivery_ship_dict.get((material, location), 0),
                    'supply_demand': supply_demand_dict.get((material, location), 0),
                    'safety_stock': safety_stock_dict.get((material, location), 0)
                }
                all_records.append(record)
        
        # 创建DataFrame并输出为CSV
        if all_records:
            historical_df = pd.DataFrame(all_records)
            historical_df.to_csv(output_file, index=False)
            logger.info(f"历史库存记录已生成: {output_file} ({len(historical_df)} 条记录)")
        else:
            # 创建空文件，包含列头
            empty_df = pd.DataFrame(columns=['date', 'material', 'location', 'ending_inventory', 
                                            'in_transit', 'production_gr', 'delivery_gr', 
                                            'order', 'shipment', 'delivery_ship', 'supply_demand', 'safety_stock'])
            empty_df.to_csv(output_file, index=False)
            logger.warning(f"历史库存记录为空，已创建空文件: {output_file}")
        
        return str(output_file)
