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
    
    def __init__(self, output_base_dir: str):
        """
        初始化汇总报告生成器
        
        Args:
            output_base_dir: 输出基础目录
        """
        self.output_base_dir = Path(output_base_dir)
        self.summary_dir = self.output_base_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        # 各模块输出目录
        self.module_dirs = {
            'module1': self.output_base_dir / "module1",
            'module3': self.output_base_dir / "module3",
            'module4': self.output_base_dir / "module4",
            'module5': self.output_base_dir / "module5",
            'module6': self.output_base_dir / "module6",
            'orchestrator': self.output_base_dir / "orchestrator"
        }
    
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
        
        # 收集所有天的数据
        for file_path in daily_files['module1']:
            try:
                xl = pd.ExcelFile(file_path)
                
                if 'OrderLog' in xl.sheet_names:
                    orders_df = xl.parse('OrderLog')
                    all_orders.append(orders_df)
                
                if 'ShipmentLog' in xl.sheet_names:
                    shipments_df = xl.parse('ShipmentLog')
                    all_shipments.append(shipments_df)
                
                if 'CutLog' in xl.sheet_names:
                    cuts_df = xl.parse('CutLog')
                    all_cuts.append(cuts_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        # 合并数据并输出
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if all_orders:
                combined_orders = pd.concat(all_orders, ignore_index=True)
                combined_orders.to_excel(writer, sheet_name='FullOrderLog', index=False)
            
            if all_shipments:
                combined_shipments = pd.concat(all_shipments, ignore_index=True)
                combined_shipments.to_excel(writer, sheet_name='FullShipmentLog', index=False)
            
            if all_cuts:
                combined_cuts = pd.concat(all_cuts, ignore_index=True)
                combined_cuts.to_excel(writer, sheet_name='FullCutLog', index=False)
        
        return str(output_file)
    
    def _generate_delivery_report(self, daily_files: Dict) -> str:
        """生成交付计划汇总报告"""
        output_file = self.summary_dir / "full_delivery_plan_report.xlsx"
        
        all_deliveries = []
        
        for file_path in daily_files['module6']:
            try:
                xl = pd.ExcelFile(file_path)
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
                xl = pd.ExcelFile(file_path)
                if 'TruckUsageLog' in xl.sheet_names:
                    usage_df = xl.parse('TruckUsageLog')
                    all_usage.append(usage_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_usage:
            combined_usage = pd.concat(all_usage, ignore_index=True)
            
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
                xl = pd.ExcelFile(file_path)
                if 'CapacityExceed' in xl.sheet_names:
                    exceed_df = xl.parse('CapacityExceed')
                    all_exceeds.append(exceed_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_exceeds:
            combined_exceeds = pd.concat(all_exceeds, ignore_index=True)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_exceeds.to_excel(writer, sheet_name='FullCapacityExceed', index=False)
        
        return str(output_file)
    
    def _generate_changeover_report(self, daily_files: Dict) -> str:
        """生成换线汇总报告"""
        output_file = self.summary_dir / "full_changeover_report.xlsx"
        
        all_changeovers = []
        
        for file_path in daily_files['module4']:
            try:
                xl = pd.ExcelFile(file_path)
                if 'ChangeoverLog' in xl.sheet_names:
                    changeover_df = xl.parse('ChangeoverLog')
                    all_changeovers.append(changeover_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_changeovers:
            combined_changeovers = pd.concat(all_changeovers, ignore_index=True)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_changeovers.to_excel(writer, sheet_name='FullChangeoverLog', index=False)
        
        return str(output_file)
    
    def _generate_deployment_report(self, daily_files: Dict) -> str:
        """生成部署计划汇总报告"""
        output_file = self.summary_dir / "full_deployment_plan_report.xlsx"
        
        all_deployments = []
        
        for file_path in daily_files['module5']:
            try:
                xl = pd.ExcelFile(file_path)
                if 'DeploymentPlan' in xl.sheet_names:
                    deployment_df = xl.parse('DeploymentPlan')
                    all_deployments.append(deployment_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_deployments:
            combined_deployments = pd.concat(all_deployments, ignore_index=True)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_deployments.to_excel(writer, sheet_name='FullDeploymentPlan', index=False)
        
        return str(output_file)
    
    def _generate_production_report(self, daily_files: Dict) -> str:
        """生成生产计划汇总报告"""
        output_file = self.summary_dir / "full_production_plan_report.xlsx"
        
        all_productions = []
        
        for file_path in daily_files['module4']:
            try:
                xl = pd.ExcelFile(file_path)
                if 'ProductionPlan' in xl.sheet_names:
                    production_df = xl.parse('ProductionPlan')
                    all_productions.append(production_df)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
        
        if all_productions:
            combined_productions = pd.concat(all_productions, ignore_index=True)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_productions.to_excel(writer, sheet_name='FullProductionPlan', index=False)
        
        return str(output_file)
    
    def _extract_date_from_filename(self, file_path: str) -> str:
        """从文件名中提取日期"""
        import re
        match = re.search(r'(\d{8})', file_path)
        if match:
            date_str = match.group(1)
            return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')
        return None