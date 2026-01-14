"""
config_validator.py

整体目的：
- 在仿真运行前，对配置文件（Excel 多表）进行系统化校验，确保关键数据完备、类型正确、数值范围合理、并在跨模块之间保持一致性。
- 将所有校验信息（INFO/WARN/ERROR）统一写入验证报告 `validation.txt`，供用户快速定位并修复问题。

功能点：
- 全局配置校验：网络、运输时效（LeadTime）、需求优先级、仓储容量等核心表的必填列、日期与数值有效性校验。
- 模块配置校验：对 Module1、3、4、5、6 的关键输入表执行针对性规则检查（如数量非负、参数正数、日期可解析、枚举取值合法等）。
- 跨模块一致性检查：例如库存地点是否存在于网络，卡车放行路线是否在 LeadTime 中有对应 OTD 等。
- 统一输出：通过 `ValidationManager` 收集消息，最终生成验证报告文件，便于追踪。

使用方法（入口）：
- 调用 `run_pre_simulation_validation(config_path, output_dir)` 加载 Excel 配置并执行所有校验，返回是否通过和报告路径。
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from pathlib import Path
from .validation_manager import ValidationManager

class ConfigValidator:
    """配置验证器

    目的：
    - 聚合并执行全量配置校验，封装各模块与跨模块的验证逻辑。

    依赖：
    - `ValidationManager`：用于记录 INFO/WARN/ERROR 并输出报告。
    """
    
    def __init__(self, validation_manager: ValidationManager):
        """初始化配置验证器

        Args:
            validation_manager: 验证管理器实例，用于收集与输出验证报告。

        逻辑概述：
        - 保存 `ValidationManager` 引用，供后续各校验函数调用统一记录接口。
        """
        self.vm = validation_manager
    
    def validate_all_configurations(self, config_path: str, config_dict: Dict) -> bool:
        """验证所有配置（主控流程）

        目的：
        - 调度执行全局、各模块及跨模块校验，汇总整体结果。

        Args:
            config_path: 配置文件路径（用于存在性检查与报错信息指引）。
            config_dict: 由 Excel 各 `sheet_name -> pandas.DataFrame` 组成的配置字典。

        Returns:
            bool: 所有校验均通过返回 True；任一校验失败返回 False。

        输入数据：
        - `config_dict` 需包含预期的工作表，如 `Global_Network`, `Global_LeadTime`, `M1_InitialInventory` 等。

        输出/副作用：
        - 通过 `ValidationManager` 记录校验信息，并在最终写入验证报告。

        逻辑：
        - 先检查配置文件存在性；再依次调用各校验子函数；最后以 `all()` 汇总布尔结果并记录完成/失败。
        """
        self.vm.add_info("ConfigValidator", "Start", "Starting configuration validation")
        
        # 验证配置文件是否存在
        if not os.path.exists(config_path):
            self.vm.add_error("ConfigValidator", "FileNotFound", f"Configuration file not found: {config_path}")
            return False
        
        # 验证各模块配置
        validation_results = []
        
        validation_results.append(self._validate_global_configs(config_dict))
        validation_results.append(self._validate_module1_configs(config_dict))
        validation_results.append(self._validate_module3_configs(config_dict))
        validation_results.append(self._validate_module4_configs(config_dict))
        validation_results.append(self._validate_module5_configs(config_dict))
        validation_results.append(self._validate_module6_configs(config_dict))
        validation_results.append(self._validate_cross_module_consistency(config_dict))
        
        # 汇总验证结果
        overall_result = all(validation_results)
        
        if overall_result:
            self.vm.add_info("ConfigValidator", "Complete", "All configuration validations passed")
        else:
            self.vm.add_error("ConfigValidator", "Failed", "Configuration validation failed")
        
        return overall_result
    
    def _validate_global_configs(self, config_dict: Dict) -> bool:
        """验证全局配置

        目的：
        - 对 `Global_Network`, `Global_LeadTime`, `Global_DemandPriority`, `Global_SpaceCapacity` 等全局表进行完备性与合理性校验。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 全局配置是否通过校验。

        输入数据：
            - `Global_Network`: 必填列校验、日期列转换与范围合理性检查。
            - `Global_LeadTime`: 必填列校验、数值列类型与范围（OTD/GR 非负；PDT/MCT 为正）、同一路线 OTD 唯一性检查。
            - `Global_DemandPriority`: 必填列与优先级正数校验。
            - `Global_SpaceCapacity`: 必填列、日期转换与容量正数校验。

        输出/副作用：
        - 通过 `ValidationManager` 输出细粒度信息与错误；无直接数据变更。

        逻辑：
        - 针对每张表执行列存在性→数据类型转换→数值/日期范围校验→特定业务规则校验，累计结果并返回。
        """
        self.vm.add_info("Global", "Start", "Validating global configurations")
        
        results = []
        
        # Global_Network 验证
        if 'Global_Network' in config_dict:
            network_df = config_dict['Global_Network']
            if not network_df.empty:
                required_cols = ['material', 'location', 'sourcing', 'location_type', 'eff_from', 'eff_to']
                results.append(self.vm.validate_required_columns(
                    network_df, required_cols, 'Global_Network', 'Global'
                ))
                
                # 日期转换
                network_df = self.vm.safe_date_conversion(network_df, 'eff_from', 'Global')
                network_df = self.vm.safe_date_conversion(network_df, 'eff_to', 'Global')
                
                # 验证日期范围
                results.append(self.vm.validate_date_ranges(
                    network_df, 'eff_from', 'eff_to', 'Global_Network', 'Global'
                ))
            else:
                self.vm.add_warning("Global", "EmptyConfig", "Global_Network configuration is empty")
        else:
            self.vm.add_error("Global", "MissingConfig", "Global_Network configuration is missing")
            results.append(False)
        
        # Global_LeadTime 验证

        if 'Global_LeadTime' in config_dict:
            leadtime_df = config_dict['Global_LeadTime']
            if not leadtime_df.empty:
                # ✅ OTD 加入必需列
                required_cols = ['sending', 'receiving', 'PDT', 'OTD', 'GR', 'MCT']
                results.append(self.vm.validate_required_columns(
                    leadtime_df, required_cols, 'Global_LeadTime', 'Global'
                ))

                # 数值列转型
                for col in ['PDT', 'OTD', 'GR', 'MCT']:
                    if col in leadtime_df.columns:
                        leadtime_df[col] = pd.to_numeric(leadtime_df[col], errors='coerce')

                # ✅ OTD/GR 非负（允许 0）；PDT/MCT 可按你原策略（如下：正数）
                # 若你希望 PDT/MCT 也允许 0，可改成类似的非负校验
                non_negative_cols = ['OTD', 'GR']
                for col in non_negative_cols:
                    if col in leadtime_df.columns:
                        neg_cnt = leadtime_df[leadtime_df[col] < 0].shape[0]
                        nan_cnt = leadtime_df[leadtime_df[col].isna()].shape[0]
                        if neg_cnt > 0:
                            self.vm.add_error("Global", "InvalidValues",
                                            f"Global_LeadTime column '{col}' has {neg_cnt} negative values")
                            results.append(False)
                        if nan_cnt > 0:
                            self.vm.add_error("Global", "InvalidValues",
                                            f"Global_LeadTime column '{col}' has {nan_cnt} NaN/invalid values")
                            results.append(False)

                results.append(self.vm.validate_positive_numbers(
                    leadtime_df, ['PDT', 'MCT'], 'Global_LeadTime', 'Global'
                ))

                # ✅ 同一路线 OTD 必须唯一
                if 'OTD' in leadtime_df.columns:
                    nunique_df = (leadtime_df
                                .groupby(['sending','receiving'])['OTD']
                                .nunique(dropna=False)
                                .reset_index(name='n'))
                    bad = nunique_df[nunique_df['n'] > 1]
                    if not bad.empty:
                        sample = bad.head(10).apply(lambda r: f"{r['sending']}→{r['receiving']}", axis=1).tolist()
                        self.vm.add_error("Global", "InconsistentOTD",
                                        f"Global_LeadTime has routes with multiple OTD values: {sample}"
                                        + (" ..." if len(bad) > 10 else ""))
                        results.append(False)
            else:
                self.vm.add_warning("Global", "EmptyConfig", "Global_LeadTime configuration is empty")
        else:
            self.vm.add_error("Global", "MissingConfig", "Global_LeadTime configuration is missing")
            results.append(False)
        
        # Global_DemandPriority 验证
        if 'Global_DemandPriority' in config_dict:
            priority_df = config_dict['Global_DemandPriority']
            if not priority_df.empty:
                required_cols = ['demand_element', 'priority']
                results.append(self.vm.validate_required_columns(
                    priority_df, required_cols, 'Global_DemandPriority', 'Global'
                ))
                
                # 验证优先级为正数
                results.append(self.vm.validate_positive_numbers(
                    priority_df, ['priority'], 'Global_DemandPriority', 'Global'
                ))
            else:
                self.vm.add_warning("Global", "EmptyConfig", "Global_DemandPriority configuration is empty")
        else:
            self.vm.add_error("Global", "MissingConfig", "Global_DemandPriority configuration is missing")
            results.append(False)
        
        # Global_SpaceCapacity 验证
        if 'Global_SpaceCapacity' in config_dict:
            space_df = config_dict['Global_SpaceCapacity']
            if not space_df.empty:
                required_cols = ['location', 'eff_from', 'eff_to', 'capacity']
                results.append(self.vm.validate_required_columns(
                    space_df, required_cols, 'Global_SpaceCapacity', 'Global'
                ))
                
                # 日期转换
                space_df = self.vm.safe_date_conversion(space_df, 'eff_from', 'Global')
                space_df = self.vm.safe_date_conversion(space_df, 'eff_to', 'Global')
                
                # 验证容量为正数
                results.append(self.vm.validate_positive_numbers(
                    space_df, ['capacity'], 'Global_SpaceCapacity', 'Global'
                ))
            else:
                self.vm.add_warning("Global", "EmptyConfig", "Global_SpaceCapacity configuration is empty")
        else:
            self.vm.add_warning("Global", "MissingConfig", "Global_SpaceCapacity configuration is missing")
        
        return all(results) if results else False
    
    def _validate_module1_configs(self, config_dict: Dict) -> bool:
        """验证 Module1 配置

        目的：
        - 校验初始库存、需求预测、下单日历的必填列与数值/日期有效性。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 模块1配置是否通过校验（若相关表缺失/为空，多为警告并总体返回 True）。

        输入数据：
            - `M1_InitialInventory`: 必填列与数量非负校验。
            - `M1_DemandForecast`: 必填列与数量为正校验。
            - `M1_OrderCalendar`: 必填列与日期可解析校验。

        输出/副作用：
        - 记录校验信息，不修改数据。

        逻辑：
        - 针对存在的相关表执行规则校验；缺失时记录警告/错误并累积结果。
        """
        self.vm.add_info("Module1", "Start", "Validating Module1 configurations")
        
        results = []
        
        # M1_InitialInventory 验证
        if 'M1_InitialInventory' in config_dict:
            inventory_df = config_dict['M1_InitialInventory']
            if not inventory_df.empty:
                required_cols = ['material', 'location', 'quantity']
                results.append(self.vm.validate_required_columns(
                    inventory_df, required_cols, 'M1_InitialInventory', 'Module1'
                ))
                
                # 验证库存数量为非负数
                non_negative_qty = inventory_df[inventory_df['quantity'] >= 0]
                if len(non_negative_qty) < len(inventory_df):
                    negative_count = len(inventory_df) - len(non_negative_qty)
                    self.vm.add_error("Module1", "InvalidValues", 
                                    f"M1_InitialInventory has {negative_count} negative quantities")
                    results.append(False)
            else:
                self.vm.add_warning("Module1", "EmptyConfig", "M1_InitialInventory configuration is empty")
        else:
            self.vm.add_error("Module1", "MissingConfig", "M1_InitialInventory configuration is missing")
            results.append(False)
        
        # M1_DemandForecast 验证
        if 'M1_DemandForecast' in config_dict:
            forecast_df = config_dict['M1_DemandForecast']
            if not forecast_df.empty:
                required_cols = ['material', 'location', 'week', 'quantity']
                results.append(self.vm.validate_required_columns(
                    forecast_df, required_cols, 'M1_DemandForecast', 'Module1'
                ))
                
                # 验证数量为正数
                results.append(self.vm.validate_positive_numbers(
                    forecast_df, ['quantity'], 'M1_DemandForecast', 'Module1'
                ))
            else:
                self.vm.add_warning("Module1", "EmptyConfig", "M1_DemandForecast configuration is empty")
        else:
            self.vm.add_warning("Module1", "MissingConfig", "M1_DemandForecast configuration is missing")
        
        # M1_OrderCalendar 验证
        if 'M1_OrderCalendar' in config_dict:
            calendar_df = config_dict['M1_OrderCalendar']
            if not calendar_df.empty:
                required_cols = ['date', 'order_day_flag']
                results.append(self.vm.validate_required_columns(
                    calendar_df, required_cols, 'M1_OrderCalendar', 'Module1'
                ))
                
                # 日期转换
                calendar_df = self.vm.safe_date_conversion(calendar_df, 'date', 'Module1')
            else:
                self.vm.add_warning("Module1", "EmptyConfig", "M1_OrderCalendar configuration is empty")
        else:
            self.vm.add_warning("Module1", "MissingConfig", "M1_OrderCalendar configuration is missing")
        
        return all(results) if results else True
    
    def _validate_module3_configs(self, config_dict: Dict) -> bool:
        """验证 Module3 配置

        目的：
        - 校验安全库存表的必填列、日期解析与数量非负性。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 模块3配置是否通过校验（为空/缺失多为警告并总体返回 True）。

        输入数据：
            - `M3_SafetyStock`: 检查列完整性、日期转换与 `safety_stock_qty` 非负。

        输出/副作用：
        - 记录校验信息，不修改数据。

        逻辑：
        - 逐项验证并累积结果，返回总体布尔值。
        """
        self.vm.add_info("Module3", "Start", "Validating Module3 configurations")
        
        results = []
        
        # M3_SafetyStock 验证
        if 'M3_SafetyStock' in config_dict:
            safety_df = config_dict['M3_SafetyStock']
            if not safety_df.empty:
                required_cols = ['material', 'location', 'date', 'safety_stock_qty']
                results.append(self.vm.validate_required_columns(
                    safety_df, required_cols, 'M3_SafetyStock', 'Module3'
                ))
                
                # 日期转换
                safety_df = self.vm.safe_date_conversion(safety_df, 'date', 'Module3')
                
                # 验证安全库存为非负数
                non_negative_stock = safety_df[safety_df['safety_stock_qty'] >= 0]
                if len(non_negative_stock) < len(safety_df):
                    negative_count = len(safety_df) - len(non_negative_stock)
                    self.vm.add_error("Module3", "InvalidValues", 
                                    f"M3_SafetyStock has {negative_count} negative safety stock values")
                    results.append(False)
            else:
                self.vm.add_warning("Module3", "EmptyConfig", "M3_SafetyStock configuration is empty")
        else:
            self.vm.add_warning("Module3", "MissingConfig", "M3_SafetyStock configuration is missing")
        
        return all(results) if results else True
    
    def _validate_module4_configs(self, config_dict: Dict) -> bool:
        """验证 Module4 配置

        目的：
        - 校验生产相关参数配置与产线产能表的完备性与数值/日期有效性。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 模块4配置是否通过校验（严格要求关键表存在并有效）。

        输入数据：
            - `M4_MaterialLocationLineCfg`: 检查必填列；除 `ptf` 允许为 0 外，其余关键数值需为正；`ptf` 需非负。
            - `M4_LineCapacity`: 检查必填列、日期解析、`capacity` 为正。

        输出/副作用：
        - 记录校验信息；不变更数据。

        逻辑：
        - 若关键配置缺失/为空，直接判定为失败；否则逐项校验后汇总布尔值。
        """
        self.vm.add_info("Module4", "Start", "Validating Module4 configurations")
        
        results = []
        
        # M4_MaterialLocationLineCfg 验证
        if 'M4_MaterialLocationLineCfg' in config_dict:
            mlcfg_df = config_dict['M4_MaterialLocationLineCfg']
            if not mlcfg_df.empty:
                required_cols = ['material', 'location', 'delegate_line', 'prd_rate', 'min_batch', 'rv', 'ptf', 'lsk', 'day', 'MCT']
                results.append(self.vm.validate_required_columns(
                    mlcfg_df, required_cols, 'M4_MaterialLocationLineCfg', 'Module4'
                ))
                
                # 验证数值为正数（除了ptf，ptf允许为0）
                positive_cols = ['prd_rate', 'min_batch', 'rv', 'lsk', 'day', 'MCT']
                results.append(self.vm.validate_positive_numbers(
                    mlcfg_df, positive_cols, 'M4_MaterialLocationLineCfg', 'Module4'
                ))
                
                # 单独验证ptf为非负数（允许为0）
                if 'ptf' in mlcfg_df.columns:
                    negative_ptf = mlcfg_df[mlcfg_df['ptf'] < 0]
                    if not negative_ptf.empty:
                        self.vm.add_error("Module4", "InvalidValues", 
                                        f"M4_MaterialLocationLineCfg column 'ptf' has {len(negative_ptf)} negative values (ptf must be >= 0)")
                        results.append(False)
            else:
                self.vm.add_error("Module4", "EmptyConfig", "M4_MaterialLocationLineCfg configuration is empty")
                results.append(False)
        else:
            self.vm.add_error("Module4", "MissingConfig", "M4_MaterialLocationLineCfg configuration is missing")
            results.append(False)
        
        # M4_LineCapacity 验证
        if 'M4_LineCapacity' in config_dict:
            capacity_df = config_dict['M4_LineCapacity']
            if not capacity_df.empty:
                required_cols = ['location', 'line', 'date', 'capacity']
                results.append(self.vm.validate_required_columns(
                    capacity_df, required_cols, 'M4_LineCapacity', 'Module4'
                ))
                
                # 日期转换
                capacity_df = self.vm.safe_date_conversion(capacity_df, 'date', 'Module4')
                
                # 验证产能为正数
                results.append(self.vm.validate_positive_numbers(
                    capacity_df, ['capacity'], 'M4_LineCapacity', 'Module4'
                ))
            else:
                self.vm.add_error("Module4", "EmptyConfig", "M4_LineCapacity configuration is empty")
                results.append(False)
        else:
            self.vm.add_error("Module4", "MissingConfig", "M4_LineCapacity configuration is missing")
            results.append(False)
        
        # 其他 Module4 配置验证...
        return all(results) if results else False
    
    def _validate_module5_configs(self, config_dict: Dict) -> bool:
        """验证 Module5 配置

        目的：
        - 校验推拉模型的必填列与枚举取值合法性。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 模块5配置是否通过校验（缺失/为空返回警告但总体 True）。

        输入数据：
            - `M5_PushPullModel`: 检查列完整性与 `model` 取值是否在允许集合（push/pull/soft push 等）。

        输出/副作用：
        - 记录校验信息；不更改数据。

        逻辑：
        - 校验失败时记录错误并累积结果；否则返回通过。
        """
        self.vm.add_info("Module5", "Start", "Validating Module5 configurations")
        
        results = []
        
        # M5_PushPullModel 验证
        if 'M5_PushPullModel' in config_dict:
            pushpull_df = config_dict['M5_PushPullModel']
            if not pushpull_df.empty:
                required_cols = ['material', 'sending', 'model']
                results.append(self.vm.validate_required_columns(
                    pushpull_df, required_cols, 'M5_PushPullModel', 'Module5'
                ))
                
                # 验证推拉模式值
                valid_modes = ['push', 'pull', 'PUSH', 'PULL','soft push','SOFT PUSH']
                invalid_modes = pushpull_df[~pushpull_df['model'].isin(valid_modes)]
                if not invalid_modes.empty:
                    self.vm.add_error("Module5", "InvalidValues", 
                                    f"M5_PushPullModel has {len(invalid_modes)} invalid model values")
                    results.append(False)
            else:
                self.vm.add_warning("Module5", "EmptyConfig", "M5_PushPullModel configuration is empty")
        else:
            self.vm.add_warning("Module5", "MissingConfig", "M5_PushPullModel configuration is missing")
        
        return all(results) if results else True
    
    def _validate_module6_configs(self, config_dict: Dict) -> bool:
        """验证 Module6 配置

        目的：
        - 校验卡车放行参数的必填列与比例范围有效性。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 模块6配置是否通过校验（缺失/为空返回警告但总体 True）。

        输入数据：
            - `M6_TruckReleaseCon`: 检查列完整性；`WFR` 与 `VFR` 需在 (0, 1] 区间。

        输出/副作用：
            - 记录校验信息；不修改数据。

        逻辑：
            - 针对比例列执行范围过滤，发现异常即记录错误并返回失败。
        """
        self.vm.add_info("Module6", "Start", "Validating Module6 configurations")
        
        results = []
        
        # M6_TruckReleaseCon 验证
        if 'M6_TruckReleaseCon' in config_dict:
            truck_con_df = config_dict['M6_TruckReleaseCon']
            if not truck_con_df.empty:
                required_cols = ['sending', 'receiving', 'truck_type', 'WFR', 'VFR']
                results.append(self.vm.validate_required_columns(
                    truck_con_df, required_cols, 'M6_TruckReleaseCon', 'Module6'
                ))
                
                # 验证 WFR 和 VFR 为正数且 <= 1.0
                for col in ['WFR', 'VFR']:
                    invalid_values = truck_con_df[(truck_con_df[col] <= 0) | (truck_con_df[col] > 1.0)]
                    if not invalid_values.empty:
                        self.vm.add_error("Module6", "InvalidValues", 
                                        f"M6_TruckReleaseCon column '{col}' has {len(invalid_values)} values outside (0, 1] range")
                        results.append(False)
            else:
                self.vm.add_warning("Module6", "EmptyConfig", "M6_TruckReleaseCon configuration is empty")
        else:
            self.vm.add_warning("Module6", "MissingConfig", "M6_TruckReleaseCon configuration is missing")
        
        return all(results) if results else True
    
    def _validate_cross_module_consistency(self, config_dict: Dict) -> bool:
        """验证跨模块配置一致性

        目的：
        - 保证多个模块间的引用关系与业务主数据保持一致，如地点在网络存在、跨节点运输在 LeadTime 有对应 OTD 等。

        Args:
            config_dict: Excel 配置字典。

        Returns:
            bool: 一致性校验是否通过（如无检查项则返回 True）。

        输入数据：
            - `Global_Network`: 提取 `location` 与 `sourcing` 并构建地点全集用于对照。
            - `M1_InitialInventory`: 校验 `location` 是否出现在网络地点/来源集合中。
            - `Global_DemandPriority`: 检查是否包含基础需求类型（如 normal, AO）。
            - `M6_TruckReleaseCon` 与 `Global_LeadTime`: 验证跨节点路线在 LeadTime 中具备唯一且存在的 OTD。

        输出/副作用：
            - 通过 `ValidationManager` 记录详细的匹配情况与缺失项，便于定位问题。

        逻辑：
            - 分场景执行集合对比与连接检查，发现缺失或不一致时输出错误/警告并累积结果。
        """
        self.vm.add_info("CrossModule", "Start", "Validating cross-module consistency")
        
        results = []
        
        # 验证网络配置与其他模块的一致性
        if 'Global_Network' in config_dict and not config_dict['Global_Network'].empty:
            network_df = config_dict['Global_Network']
            
            # 获取网络中的所有地点（包括location和sourcing）
            network_locations = set(network_df['location'].unique())
            network_sourcings = set(network_df['sourcing'].dropna().unique())  # 排除空值
            all_network_places = network_locations.union(network_sourcings)
            
            # 检查 M1_InitialInventory 中的 location 是否在网络中
            if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
                inventory_df = config_dict['M1_InitialInventory']
                inventory_locations = set(inventory_df['location'].unique())
                
                # 修改验证逻辑：检查initial inventory的location是否在network的location或sourcing中
                missing_locations = inventory_locations - all_network_places
                
                if missing_locations:
                    self.vm.add_error("CrossModule", "Consistency", 
                                    f"M1_InitialInventory contains locations not in Global_Network (location or sourcing): {missing_locations}")
                    results.append(False)
                
                # 额外检查：确保所有initial inventory的location都在network的location中
                # 因为initial inventory应该只包含实际存在的地点
                missing_in_network_locations = inventory_locations - network_locations
                if missing_in_network_locations:
                    self.vm.add_warning("CrossModule", "Consistency", 
                                      f"M1_InitialInventory contains locations that are not defined as 'location' in Global_Network (they might only be in 'sourcing'): {missing_in_network_locations}")
                
                # 输出详细的验证信息（安全排序，处理混合数据类型）
                try:
                    network_locations_sorted = sorted([str(x) for x in network_locations])
                    network_sourcings_sorted = sorted([str(x) for x in network_sourcings])
                    all_network_places_sorted = sorted([str(x) for x in all_network_places])
                    inventory_locations_sorted = sorted([str(x) for x in inventory_locations])
                    
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"Network locations: {network_locations_sorted}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"Network sourcings: {network_sourcings_sorted}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"All network places: {all_network_places_sorted}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"Initial inventory locations: {inventory_locations_sorted}")
                except Exception as e:
                    self.vm.add_warning("CrossModule", "ValidationDetails", 
                                      f"Could not sort location data for display due to mixed data types: {e}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"Network locations count: {len(network_locations)}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"Network sourcings count: {len(network_sourcings)}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"All network places count: {len(all_network_places)}")
                    self.vm.add_info("CrossModule", "ValidationDetails", 
                                   f"Initial inventory locations count: {len(inventory_locations)}")
    
        # 验证需求优先级配置的一致性
        if 'Global_DemandPriority' in config_dict and not config_dict['Global_DemandPriority'].empty:
            priority_df = config_dict['Global_DemandPriority']
            defined_demand_types = set(priority_df['demand_element'].unique())
            
            # 检查是否包含基本的需求类型
            required_demand_types = {'normal', 'AO'}
            missing_types = required_demand_types - defined_demand_types
            
            if missing_types:
                self.vm.add_warning("CrossModule", "Consistency", 
                                  f"Global_DemandPriority missing basic demand types: {missing_types}")
        
        return all(results) if results else True
        # === 验证：M6_TruckReleaseCon 里的所有跨节点路线，必须在 Global_LeadTime 里有 OTD ===
        if 'M6_TruckReleaseCon' in config_dict and not config_dict['M6_TruckReleaseCon'].empty \
        and 'Global_LeadTime' in config_dict and not config_dict['Global_LeadTime'].empty:
            truck_con = config_dict['M6_TruckReleaseCon'][['sending','receiving']].drop_duplicates()
            truck_con = truck_con[truck_con['sending'] != truck_con['receiving']]

            lt = config_dict['Global_LeadTime'][['sending','receiving','OTD']].copy()
            # 确保数值化
            lt['OTD'] = pd.to_numeric(lt['OTD'], errors='coerce')

            merged = truck_con.merge(lt, on=['sending','receiving'], how='left', indicator=True)
            missing = merged[merged['OTD'].isna()]

            if not missing.empty:
                routes = (missing[['sending','receiving']]
                        .astype(str)
                        .agg('→'.join, axis=1)
                        .tolist())
                sample = routes[:10]
                self.vm.add_error("CrossModule", "MissingOTD",
                                f"Routes in M6_TruckReleaseCon missing OTD in Global_LeadTime: {sample}"
                                + (" ..." if len(routes) > 10 else ""))
                results.append(False)

def run_pre_simulation_validation(config_path: str, output_dir: str) -> tuple:
    """运行仿真前配置验证（外部入口）

    目的：
    - 从 Excel 文件加载所有配置表，执行完整的配置校验流程，并输出验证报告。

    Args:
        config_path: 配置文件路径（Excel）。
        output_dir: 验证报告输出目录。

    Returns:
        tuple: `(passed: bool, report_path: str)`，分别表示是否通过校验与验证报告的文件路径。

    输入数据：
        - Excel 工作簿，包含若干工作表；函数将它们解析为 `Dict[str, DataFrame]`。

    输出/副作用：
        - 在 `output_dir` 下生成验证报告文件；并返回其路径。

    逻辑：
        - 先创建 `ValidationManager`；读取 Excel；若读取失败则写报告并返回失败；否则实例化 `ConfigValidator` 执行校验，最终写入报告并返回结果与路径。
    """
    # 创建验证管理器
    validation_manager = ValidationManager(output_dir)
    
    # 加载配置
    try:
        # 使用更高效的方式读取所有工作表，且只读取数据值以加速
        config_dict = pd.read_excel(config_path, sheet_name=None, engine='openpyxl', engine_kwargs={'data_only': True})
    except Exception as e:
        validation_manager.add_error("ConfigLoader", "LoadError", f"Failed to load config file: {str(e)}")
        report_path = validation_manager.write_report()
        return False, report_path
    
    # 创建配置验证器并运行验证
    validator = ConfigValidator(validation_manager)
    validation_passed = validator.validate_all_configurations(config_path, config_dict)
    
    # 写入验证报告
    report_path = validation_manager.write_report()
    
    return validation_passed, report_path