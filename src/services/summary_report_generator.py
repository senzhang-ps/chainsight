# summary_report_generator.py
# 汇总报告生成器 - 全周期模拟结束后输出7类full报告

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from pathlib import Path
import glob

class SummaryReportGenerator:
    """汇总报告生成器"""
    
    def __init__(self, output_base_dir: str, config_dict: Optional[dict] = None):
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
        """标准化 material 值：作为文本处理，仅移除误读的.0后缀"""
        if not material_str or material_str in ['nan', 'None', '']:
            return ""
        
        material_str = str(material_str).strip()
        
        try:
            # 仅当是以 .0 结尾的纯数字字符串时，才移除 .0
            # 这样可以保留 '00123' 这种带前导零的物料编码
            if material_str.endswith('.0') and material_str[:-2].replace('.', '').replace('-', '').isdigit():
                return str(int(float(material_str)))
            return material_str
        except:
            return material_str
    
    @staticmethod
    def _normalize_location_value(location_str: str) -> str:
        """标准化 location 值：纯数字补齐为4位，字母数字保持原样"""
        if not location_str or location_str in ['nan', 'None', '']:
            return ""
        location_str = str(location_str).strip()
        try:
            # 如果是纯数字，补齐为4位
            if location_str.isdigit():
                return str(int(location_str)).zfill(4)
            # 字母数字混合，保持原样
            return location_str
        except:
            return location_str
    
    def generate_all_reports(self, start_date: str, end_date: str) -> Dict[str, str]:
        """
        生成所有汇总报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, str]: 报告名称到文件路径的映射
        """
        print(f"📊 开始生成汇总报告 ({start_date} 到 {end_date})")
        
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
        
        # 检测是否是临时目录（Windows: Temp, Linux: /tmp）
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        is_temp_dir = str(self.summary_dir).startswith(str(temp_dir)) or '/tmp/' in str(self.summary_dir)
        
        if is_temp_dir:
            print(f"✅ 汇总报告生成完成（将写入数据库）")
        else:
            print(f"✅ 汇总报告生成完成，输出目录: {self.summary_dir}")
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
    
    def _generate_order_shipment_report(self, daily_files: Dict) -> str:
        """生成订单发货切割汇总报告"""
        output_file = self.summary_dir / "full_order_shipment_cut_report.xlsx"
        
        all_orders = []
        all_shipments = []
        all_cuts = []
        
        # 收集所有天的数据，并添加simulation_date
        for file_path in daily_files['module1']:
            try:
                # 从文件名提取simulation_date
                simulation_date = self._extract_date_from_filename(file_path)
                
                with pd.ExcelFile(file_path) as xl:
                    if 'OrderLog' in xl.sheet_names:
                        orders_df = xl.parse('OrderLog')
                        if not orders_df.empty and simulation_date:
                            orders_df['simulation_date'] = simulation_date
                        all_orders.append(orders_df)
                    
                    if 'ShipmentLog' in xl.sheet_names:
                        shipments_df = xl.parse('ShipmentLog')
                        if not shipments_df.empty and simulation_date:
                            shipments_df['simulation_date'] = simulation_date
                        all_shipments.append(shipments_df)
                    
                    if 'CutLog' in xl.sheet_names:
                        cuts_df = xl.parse('CutLog')
                        if not cuts_df.empty and simulation_date:
                            cuts_df['simulation_date'] = simulation_date
                        all_cuts.append(cuts_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        # 合并并汇总数据
        summary_data = []
        
        # 合并所有数据
        combined_orders = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
        combined_shipments = pd.concat(all_shipments, ignore_index=True) if all_shipments else pd.DataFrame()
        combined_cuts = pd.concat(all_cuts, ignore_index=True) if all_cuts else pd.DataFrame()
        
        # 🔧 关键修复：去除重复的 AO 订单
        # AO 订单可能在多个 simulation_date 被重复记录，需要按唯一键去重
        if not combined_orders.empty:
            original_count = len(combined_orders)
            # 按 date, material, location, quantity,（若有）demand_type, simulation_date 去重（保留第一条）
            dedup_cols = ['date', 'material', 'location', 'quantity', 'simulation_date']
            if 'demand_type' in combined_orders.columns:
                dedup_cols.append('demand_type')
            combined_orders = combined_orders.drop_duplicates(subset=dedup_cols, keep='first')
            dedup_count = len(combined_orders)
            if original_count != dedup_count:
                print(f"📊 OrderLog 去重：原始 {original_count} 条 → 去重后 {dedup_count} 条（移除了 {original_count - dedup_count} 条重复的 AO 订单）")
        
        # 🔍 调试：显示 CutLog 数据
        # print(f"\n🔍 Order Shipment Cut Report 数据统计:")
        # print(f"  OrderLog 记录数: {len(combined_orders)}")
        # print(f"  ShipmentLog 记录数: {len(combined_shipments)}")
        # print(f"  CutLog 记录数: {len(combined_cuts)}")
        # if not combined_cuts.empty:
        #     print(f"  CutLog 前5条记录:")
        #     print(combined_cuts.head())
        
        # 构建汇总表
        if not combined_orders.empty or not combined_shipments.empty or not combined_cuts.empty:
            # 准备订单数据（去掉 demand_type，合并所有类型的订单）
            if not combined_orders.empty:
                order_agg = combined_orders.groupby(
                    ['date', 'material', 'location', 'simulation_date'], 
                    dropna=False
                ).agg({'quantity': 'sum'}).reset_index()
                order_agg.rename(columns={'quantity': 'order_qty'}, inplace=True)
            else:
                order_agg = pd.DataFrame(columns=['date', 'material', 'location', 'simulation_date', 'order_qty'])
            
            # 准备发货数据（去掉 demand_type，合并所有类型的发货）
            if not combined_shipments.empty:
                shipment_agg = combined_shipments.groupby(
                    ['date', 'material', 'location', 'simulation_date'], 
                    dropna=False
                ).agg({'quantity': 'sum'}).reset_index()
                shipment_agg.rename(columns={'quantity': 'shipment_qty'}, inplace=True)
            else:
                shipment_agg = pd.DataFrame(columns=['date', 'material', 'location', 'simulation_date', 'shipment_qty'])
            
            # 准备缺货数据（去掉 demand_type，合并所有类型的缺货）
            if not combined_cuts.empty:
                cut_agg = combined_cuts.groupby(
                    ['date', 'material', 'location', 'simulation_date'], 
                    dropna=False
                ).agg({'quantity': 'sum'}).reset_index()
                cut_agg.rename(columns={'quantity': 'cut_qty'}, inplace=True)
            else:
                cut_agg = pd.DataFrame(columns=['date', 'material', 'location', 'simulation_date', 'cut_qty'])
            
            # 合并三个汇总表
            summary = order_agg.merge(
                shipment_agg, 
                on=['date', 'material', 'location', 'simulation_date'], 
                how='outer'
            )
            summary = summary.merge(
                cut_agg, 
                on=['date', 'material', 'location', 'simulation_date'], 
                how='outer'
            )
            
            # 填充缺失值为0
            summary['order_qty'] = summary['order_qty'].fillna(0).infer_objects(copy=False).astype(int)
            summary['shipment_qty'] = summary['shipment_qty'].fillna(0).infer_objects(copy=False).astype(int)
            summary['cut_qty'] = summary['cut_qty'].fillna(0).infer_objects(copy=False).astype(int)
            
            # 🔧 过滤掉超出模拟日期范围的数据
            if 'date' in summary.columns:
                original_count = len(summary)
                summary['date'] = pd.to_datetime(summary['date'])
                summary = summary[summary['date'] <= self.end_date]
                filtered_count = len(summary)
                if original_count != filtered_count:
                    print(f"📊 过滤超出日期范围的记录：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出 {self.end_date.date()} 的记录）")
            
            # 🔧 验证：计算 cut_qty 是否等于 order - shipment
            summary['calculated_cut'] = (summary['order_qty'] - summary['shipment_qty']).clip(lower=0)
            
            # 检查不一致的记录
            inconsistent = summary[summary['cut_qty'] != summary['calculated_cut']]
            if not inconsistent.empty:
                print(f"\n⚠️  发现 {len(inconsistent)} 条 cut_qty 不一致的记录:")
                print(f"  (cut_qty 应该 = max(0, order_qty - shipment_qty))")
                print(inconsistent[['date', 'material', 'location', 'order_qty', 'shipment_qty', 'cut_qty', 'calculated_cut']].head(10))
            
            # 移除临时列
            summary = summary.drop(columns=['calculated_cut'])
            
            # 按日期和物料排序
            summary = summary.sort_values(['simulation_date', 'date', 'material', 'location'])
            
            # 调整列顺序（去掉 demand_type）
            column_order = ['simulation_date', 'date', 'material', 'location', 'order_qty', 'shipment_qty', 'cut_qty']
            summary = summary[column_order]
            
            # 输出到Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name='OrderShipmentCutSummary', index=False)
        
        return str(output_file)
    
    def _generate_delivery_report(self, daily_files: Dict) -> str:
        """生成交付计划汇总报告"""
        output_file = self.summary_dir / "full_delivery_plan_report.xlsx"
        
        all_deliveries = []
        
        for file_path in daily_files['module6']:
            try:
                with pd.ExcelFile(file_path) as xl:
                    if 'DeliveryPlan' in xl.sheet_names:
                        delivery_df = xl.parse('DeliveryPlan')
                        # 添加date, planned_deploy_date, actual_ship_date字段
                        file_date = self._extract_date_from_filename(file_path)
                        delivery_df['date'] = file_date
                        all_deliveries.append(delivery_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_deliveries:
            combined_deliveries = pd.concat(all_deliveries, ignore_index=True)
            
            # 🔧 过滤掉超出模拟日期范围的数据
            original_count = len(combined_deliveries)
            date_cols_to_filter = ['planned_deploy_date', 'actual_ship_date', 'date']
            for col in date_cols_to_filter:
                if col in combined_deliveries.columns:
                    combined_deliveries[col] = pd.to_datetime(combined_deliveries[col], errors='coerce')
            
            # 只保留 planned_deploy_date 或 actual_ship_date 在范围内的记录
            if 'planned_deploy_date' in combined_deliveries.columns or 'actual_ship_date' in combined_deliveries.columns:
                mask = pd.Series([True] * len(combined_deliveries))
                if 'planned_deploy_date' in combined_deliveries.columns:
                    mask = mask & ((combined_deliveries['planned_deploy_date'].isna()) | (combined_deliveries['planned_deploy_date'] <= self.end_date))
                if 'actual_ship_date' in combined_deliveries.columns:
                    mask = mask & ((combined_deliveries['actual_ship_date'].isna()) | (combined_deliveries['actual_ship_date'] <= self.end_date))
                combined_deliveries = combined_deliveries[mask]
                
                filtered_count = len(combined_deliveries)
                if original_count != filtered_count:
                    print(f"📊 Delivery Plan 过滤：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出日期范围的记录）")
            
            # 按需求添加字段: date, material, sending, receiving, planned_qty, delivered_qty, planned_deploy_date, actual_ship_date
            required_columns = ['date', 'material', 'sending', 'receiving', 'planned_qty', 'delivered_qty', 'planned_deploy_date', 'actual_ship_date']
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_deliveries.to_excel(writer, sheet_name='FullDeliveryPlan', index=False)
        
        return str(output_file)
    
    def _generate_truck_usage_report(self, daily_files: Dict) -> str:
        """生成卡车使用汇总报告"""
        output_file = self.summary_dir / "full_truck_usage_report.xlsx"
        
        all_usage = []
        
        for file_path in daily_files['module6']:
            try:
                with pd.ExcelFile(file_path) as xl:
                    if 'TruckUsageLog' in xl.sheet_names:
                        usage_df = xl.parse('TruckUsageLog')
                        all_usage.append(usage_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_usage:
            combined_usage = pd.concat(all_usage, ignore_index=True)
            
            # 🔧 过滤掉超出模拟日期范围的数据
            if 'date' in combined_usage.columns:
                original_count = len(combined_usage)
                combined_usage['date'] = pd.to_datetime(combined_usage['date'], errors='coerce')
                combined_usage = combined_usage[
                    (combined_usage['date'].isna()) | 
                    (combined_usage['date'] <= self.end_date)
                ]
                filtered_count = len(combined_usage)
                if original_count != filtered_count:
                    print(f"📊 Truck Usage 过滤：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出日期范围的记录）")
            
            # 按需求字段: date, sending, receiving, truck_type, available_trucks, used_trucks, wfr, vfr
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_usage.to_excel(writer, sheet_name='FullTruckUsage', index=False)
        
        return str(output_file)
    
    def _generate_capacity_exceed_report(self, daily_files: Dict) -> str:
        """生成产能超限汇总报告"""
        output_file = self.summary_dir / "full_exceed_capacity_report.xlsx"
        
        all_exceeds = []
        
        for file_path in daily_files['module4']:
            try:
                with pd.ExcelFile(file_path) as xl:
                    if 'CapacityExceed' in xl.sheet_names:
                        exceed_df = xl.parse('CapacityExceed')
                        all_exceeds.append(exceed_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_exceeds:
            combined_exceeds = pd.concat(all_exceeds, ignore_index=True)
            
            # 🔧 过滤掉超出模拟日期范围的数据
            if 'date' in combined_exceeds.columns:
                original_count = len(combined_exceeds)
                combined_exceeds['date'] = pd.to_datetime(combined_exceeds['date'], errors='coerce')
                combined_exceeds = combined_exceeds[
                    (combined_exceeds['date'].isna()) | 
                    (combined_exceeds['date'] <= self.end_date)
                ]
                filtered_count = len(combined_exceeds)
                if original_count != filtered_count:
                    print(f"📊 Capacity Exceed 过滤：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出日期范围的记录）")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_exceeds.to_excel(writer, sheet_name='FullCapacityExceed', index=False)
        
        return str(output_file)
    
    def _generate_changeover_report(self, daily_files: Dict) -> str:
        """生成换线汇总报告"""
        output_file = self.summary_dir / "full_changeover_report.xlsx"
        
        all_changeovers = []
        
        for file_path in daily_files['module4']:
            try:
                with pd.ExcelFile(file_path) as xl:
                    if 'ChangeoverLog' in xl.sheet_names:
                        changeover_df = xl.parse('ChangeoverLog')
                        all_changeovers.append(changeover_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_changeovers:
            # 过滤掉空的 DataFrame 以避免 FutureWarning
            non_empty_changeovers = [df for df in all_changeovers if not df.empty]
            if non_empty_changeovers:
                combined_changeovers = pd.concat(non_empty_changeovers, ignore_index=True)
                
                # 🔧 过滤掉超出模拟日期范围的数据
                original_count = len(combined_changeovers)
                date_cols_to_filter = ['changeover_start_date', 'changeover_end_date', 'date']
                for col in date_cols_to_filter:
                    if col in combined_changeovers.columns:
                        combined_changeovers[col] = pd.to_datetime(combined_changeovers[col], errors='coerce')
                
                # 只保留 changeover_end_date 在范围内的记录
                if 'changeover_end_date' in combined_changeovers.columns:
                    combined_changeovers = combined_changeovers[
                        (combined_changeovers['changeover_end_date'].isna()) | 
                        (combined_changeovers['changeover_end_date'] <= self.end_date)
                    ]
                    filtered_count = len(combined_changeovers)
                    if original_count != filtered_count:
                        print(f"📊 Changeover Log 过滤：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出日期范围的记录）")
                
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    combined_changeovers.to_excel(writer, sheet_name='FullChangeoverLog', index=False)
        
        return str(output_file)
    
    def _generate_deployment_report(self, daily_files: Dict) -> str:
        """生成部署计划汇总报告"""
        output_file = self.summary_dir / "full_deployment_plan_report.xlsx"
        
        all_deployments = []
        
        for file_path in daily_files['module5']:
            try:
                with pd.ExcelFile(file_path) as xl:
                    if 'DeploymentPlan' in xl.sheet_names:
                        deployment_df = xl.parse('DeploymentPlan')
                        all_deployments.append(deployment_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_deployments:
            combined_deployments = pd.concat(all_deployments, ignore_index=True)
            
            # 🔧 过滤掉超出模拟日期范围的数据
            original_count = len(combined_deployments)
            # Deployment 可能的日期字段：deployment_date, arrival_date, ship_date 等
            date_cols_to_filter = ['deployment_date', 'arrival_date', 'ship_date', 'date']
            for col in date_cols_to_filter:
                if col in combined_deployments.columns:
                    combined_deployments[col] = pd.to_datetime(combined_deployments[col], errors='coerce')
            
            # 过滤：如果有任何日期列，只保留日期在范围内的记录
            mask = pd.Series([True] * len(combined_deployments))
            for col in ['deployment_date', 'arrival_date', 'ship_date', 'date']:
                if col in combined_deployments.columns:
                    mask = mask & ((combined_deployments[col].isna()) | (combined_deployments[col] <= self.end_date))
            
            combined_deployments = combined_deployments[mask]
            filtered_count = len(combined_deployments)
            if original_count != filtered_count:
                print(f"📊 Deployment Plan 过滤：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出日期范围的记录）")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_deployments.to_excel(writer, sheet_name='FullDeploymentPlan', index=False)
        
        return str(output_file)
    
    def _generate_production_report(self, daily_files: Dict) -> str:
        """生成生产计划汇总报告"""
        output_file = self.summary_dir / "full_production_plan_report.xlsx"
        
        all_productions = []
        
        for file_path in daily_files['module4']:
            try:
                with pd.ExcelFile(file_path) as xl:
                    if 'ProductionPlan' in xl.sheet_names:
                        production_df = xl.parse('ProductionPlan')
                        all_productions.append(production_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_productions:
            combined_productions = pd.concat(all_productions, ignore_index=True)
            
            # 🔧 过滤掉超出模拟日期范围的数据
            original_count = len(combined_productions)
            date_cols_to_filter = ['available_date', 'production_plan_date']
            for col in date_cols_to_filter:
                if col in combined_productions.columns:
                    combined_productions[col] = pd.to_datetime(combined_productions[col], errors='coerce')
            
            # 只保留 available_date 在范围内的记录（这是实际产品可用的日期）
            if 'available_date' in combined_productions.columns:
                combined_productions = combined_productions[
                    (combined_productions['available_date'].isna()) | 
                    (combined_productions['available_date'] <= self.end_date)
                ]
                filtered_count = len(combined_productions)
                if original_count != filtered_count:
                    print(f"📊 Production Plan 过滤：{original_count} 条 → {filtered_count} 条（移除了 {original_count - filtered_count} 条超出日期范围的记录）")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_productions.to_excel(writer, sheet_name='FullProductionPlan', index=False)
        
        return str(output_file)
    
    def _extract_date_from_filename(self, file_path: str) -> Optional[str]:
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
            # 向量化构建safety_stock_dict
            ss_df['material'] = ss_df['material'].astype(str).map(self._normalize_material_value)
            ss_df['location'] = ss_df['location'].astype(str).map(self._normalize_location_value)
            ss_df['ss_qty'] = pd.to_numeric(ss_df.get('safety_stock_qty', 0), errors='coerce').fillna(0).astype(int)
            # 如果有多个日期,按日期排序后取最后一条(最新值)
            if 'date' in ss_df.columns:
                ss_df = ss_df.sort_values('date')
            safety_stock_dict = (
                ss_df.groupby(['material', 'location'])['ss_qty']
                .last().to_dict()
            )
        
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
                    if not inv_df.empty:
                        inv_df = inv_df.assign(
                            material=inv_df['material'].astype(str).map(self._normalize_material_value),
                            location=inv_df['location'].astype(str).map(self._normalize_location_value),
                            quantity=pd.to_numeric(inv_df.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                        )
                        ending_inv_dict = (
                            inv_df.groupby(['material', 'location'], dropna=False)['quantity']
                            .sum()
                            .astype(int)
                            .to_dict()
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {inv_file}: {e}")
            
            # 2. 读取在途库存 (planning_intransit)
            intransit_file = orchestrator_dir / f"planning_intransit_{date_str_file}.csv"
            if intransit_file.exists():
                try:
                    intransit_df = pd.read_csv(intransit_file)
                    if not intransit_df.empty:
                        tmp = intransit_df.assign(
                            material=intransit_df['material'].astype(str).map(self._normalize_material_value),
                            receiving=intransit_df['receiving'].astype(str).map(self._normalize_location_value),
                            quantity=pd.to_numeric(intransit_df.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                        )
                        in_transit_dict = (
                            tmp.groupby(['material', 'receiving'], dropna=False)['quantity']
                            .sum()
                            .astype(int)
                            .rename_axis(['material', 'location'])
                            .to_dict()
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {intransit_file}: {e}")
            
            # 3. 读取生产入库 (production_gr)
            prod_gr_file = orchestrator_dir / f"production_gr_{date_str_file}.csv"
            if prod_gr_file.exists():
                try:
                    prod_gr_df = pd.read_csv(prod_gr_file)
                    if not prod_gr_df.empty:
                        tmp = prod_gr_df.assign(
                            material=prod_gr_df['material'].astype(str).map(self._normalize_material_value),
                            location=prod_gr_df['location'].astype(str).map(self._normalize_location_value),
                            quantity=pd.to_numeric(prod_gr_df.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                        )
                        production_gr_dict = (
                            tmp.groupby(['material', 'location'], dropna=False)['quantity']
                            .sum()
                            .astype(int)
                            .to_dict()
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {prod_gr_file}: {e}")
            
            # 4. 读取配送入库 (delivery_gr)
            del_gr_file = orchestrator_dir / f"delivery_gr_{date_str_file}.csv"
            if del_gr_file.exists():
                try:
                    del_gr_df = pd.read_csv(del_gr_file)
                    if not del_gr_df.empty:
                        tmp = del_gr_df.assign(
                            material=del_gr_df['material'].astype(str).map(self._normalize_material_value),
                            receiving=del_gr_df['receiving'].astype(str).map(self._normalize_location_value),
                            quantity=pd.to_numeric(del_gr_df.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                        )
                        delivery_gr_dict = (
                            tmp.groupby(['material', 'receiving'], dropna=False)['quantity']
                            .sum()
                            .astype(int)
                            .rename_axis(['material', 'location'])
                            .to_dict()
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {del_gr_file}: {e}")
            
            # 5. 读取订单 (order from module1)
            order_file = module1_dir / f"module1_output_{date_str_file}.xlsx"
            if order_file.exists():
                try:
                    with pd.ExcelFile(order_file) as xl:
                        if 'OrderLog' in xl.sheet_names:
                            order_df = xl.parse('OrderLog')
                            if not order_df.empty and 'date' in order_df.columns:
                                today_orders = order_df[pd.to_datetime(order_df['date']) == date]
                                if not today_orders.empty:
                                    tmp = today_orders.assign(
                                        material=today_orders['material'].astype(str).map(self._normalize_material_value),
                                        location=today_orders['location'].astype(str).map(self._normalize_location_value),
                                        quantity=pd.to_numeric(today_orders.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                                    )
                                    order_dict = (
                                        tmp.groupby(['material', 'location'], dropna=False)['quantity']
                                        .sum()
                                        .astype(int)
                                        .to_dict()
                                    )
                        # 读取供需日志 (supply demand log)
                        if 'SupplyDemandLog' in xl.sheet_names:
                            sd_df = xl.parse('SupplyDemandLog')
                            if not sd_df.empty and 'date' in sd_df.columns:
                                today_sd = sd_df[pd.to_datetime(sd_df['date']) == date]
                                if not today_sd.empty:
                                    tmp = today_sd.assign(
                                        material=today_sd['material'].astype(str).map(self._normalize_material_value),
                                        location=today_sd['location'].astype(str).map(self._normalize_location_value),
                                        quantity=pd.to_numeric(today_sd.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                                    )
                                    supply_demand_dict = (
                                        tmp.groupby(['material', 'location'], dropna=False)['quantity']
                                        .sum()
                                        .astype(int)
                                        .to_dict()
                                    )
                except Exception as e:
                    print(f"Warning: Failed to read orders/supply-demand from {order_file}: {e}")
            
            # 6. 读取发货 (shipment from orchestrator)
            shipment_file = orchestrator_dir / f"shipment_log_{date_str_file}.csv"
            if shipment_file.exists():
                try:
                    shipment_df = pd.read_csv(shipment_file)
                    if not shipment_df.empty:
                        tmp = shipment_df.assign(
                            material=shipment_df['material'].astype(str).map(self._normalize_material_value),
                            location=shipment_df['location'].astype(str).map(self._normalize_location_value),
                            quantity=pd.to_numeric(shipment_df.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                        )
                        shipment_dict = (
                            tmp.groupby(['material', 'location'], dropna=False)['quantity']
                            .sum()
                            .astype(int)
                            .to_dict()
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {shipment_file}: {e}")
            
            # 7. 读取配送发货 (delivery_shipment from orchestrator)
            del_ship_file = orchestrator_dir / f"delivery_shipment_log_{date_str_file}.csv"
            if del_ship_file.exists():
                try:
                    del_ship_df = pd.read_csv(del_ship_file)
                    if not del_ship_df.empty:
                        tmp = del_ship_df.assign(
                            material=del_ship_df['material'].astype(str).map(self._normalize_material_value),
                            sending=del_ship_df['sending'].astype(str).map(self._normalize_location_value),
                            quantity=pd.to_numeric(del_ship_df.get('quantity', 0), errors='coerce').fillna(0).astype(int)
                        )
                        delivery_ship_dict = (
                            tmp.groupby(['material', 'sending'], dropna=False)['quantity']
                            .sum()
                            .astype(int)
                            .rename_axis(['material', 'location'])
                            .to_dict()
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {del_ship_file}: {e}")
            
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
            print(f"📊 历史库存记录已生成: {output_file} ({len(historical_df)} 条记录)")
        else:
            # 创建空文件，包含列头
            empty_df = pd.DataFrame(columns=['date', 'material', 'location', 'ending_inventory', 
                                            'in_transit', 'production_gr', 'delivery_gr', 
                                            'order', 'shipment', 'delivery_ship', 'supply_demand', 'safety_stock'])
            empty_df.to_csv(output_file, index=False)
            print(f"⚠️  历史库存记录为空，已创建空文件: {output_file}")
        
        return str(output_file)