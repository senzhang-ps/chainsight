# validation_manager.py
# 统一验证管理器 - 集中处理配置验证和错误报告
# 在仿真开始前运行，输出 validation.txt 供用户查看

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

class ValidationManager:
    """统一验证管理器"""
    
    def __init__(self, output_dir: str):
        """
        初始化验证管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_file = self.output_dir / "validation_report.txt"
        
        self.errors = []
        self.warnings = []
        self.infos = []
        self.start_time = datetime.now()
    
    def add_error(self, module: str, category: str, message: str):
        """添加错误信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.errors.append(f"[{timestamp}] [ERROR] [{module}] [{category}] {message}")
    
    def add_warning(self, module: str, category: str, message: str):
        """添加警告信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.warnings.append(f"[{timestamp}] [WARNING] [{module}] [{category}] {message}")
    
    def add_info(self, module: str, category: str, message: str):
        """添加信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.infos.append(f"[{timestamp}] [INFO] [{module}] [{category}] {message}")
    
    def has_errors(self) -> bool:
        """检查是否有错误"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """检查是否有警告"""
        return len(self.warnings) > 0
    
    def get_error_count(self) -> int:
        """获取错误数量"""
        return len(self.errors)
    
    def get_warning_count(self) -> int:
        """获取警告数量"""
        return len(self.warnings)
    
    def write_report(self) -> str:
        """
        写入验证报告到文件
        
        Returns:
            str: 报告文件路径
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        with open(self.validation_file, 'w', encoding='utf-8') as f:
            # 写入报告头部
            f.write("=" * 80 + "\n")
            f.write("SUPPLY CHAIN INTEGRATION VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Validation Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Validation End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Total Errors: {len(self.errors)}\n")
            f.write(f"Total Warnings: {len(self.warnings)}\n")
            f.write(f"Total Infos: {len(self.infos)}\n")
            f.write("\n")
            
            # 写入汇总
            if self.errors:
                f.write("❌ VALIDATION FAILED - CRITICAL ERRORS FOUND\n")
                f.write("Please fix all errors before running simulation.\n")
            elif self.warnings:
                f.write("⚠️  VALIDATION PASSED WITH WARNINGS\n")
                f.write("Simulation can proceed but please review warnings.\n")
            else:
                f.write("✅ VALIDATION PASSED\n")
                f.write("All configuration checks passed successfully.\n")
            
            f.write("\n")
            
            # 写入错误
            if self.errors:
                f.write("ERRORS:\n")
                f.write("-" * 40 + "\n")
                for error in self.errors:
                    f.write(f"{error}\n")
                f.write("\n")
            
            # 写入警告
            if self.warnings:
                f.write("WARNINGS:\n")
                f.write("-" * 40 + "\n")
                for warning in self.warnings:
                    f.write(f"{warning}\n")
                f.write("\n")
            
            # 写入信息
            if self.infos:
                f.write("INFORMATION:\n")
                f.write("-" * 40 + "\n")
                for info in self.infos:
                    f.write(f"{info}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
        
        return str(self.validation_file)

    def safe_date_conversion(self, df: pd.DataFrame, column: str, module: str) -> pd.DataFrame:
        """
        安全的日期转换，记录转换失败的行
        
        Args:
            df: 数据框
            column: 日期列名
            module: 模块名
            
        Returns:
            pd.DataFrame: 转换后的数据框
        """
        if column not in df.columns:
            self.add_warning(module, "DataConversion", f"Column '{column}' not found in dataframe")
            return df
        
        original_count = len(df)
        original_null_count = df[column].isnull().sum()
        
        # 执行日期转换
        df[column] = pd.to_datetime(df[column], errors='coerce')
        
        # 检查转换后的空值
        new_null_count = df[column].isnull().sum()
        failed_conversions = new_null_count - original_null_count
        
        if failed_conversions > 0:
            self.add_warning(
                module, 
                "DataConversion", 
                f"Column '{column}': {failed_conversions}/{original_count} rows failed date conversion"
            )
        
        if original_count > 0:
            success_rate = ((original_count - failed_conversions) / original_count) * 100
            self.add_info(
                module,
                "DataConversion", 
                f"Column '{column}': {success_rate:.1f}% conversion success rate"
            )
        
        return df

    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str], 
                                sheet_name: str, module: str) -> bool:
        """
        验证必需列是否存在
        
        Args:
            df: 数据框
            required_columns: 必需列列表
            sheet_name: 表单名
            module: 模块名
            
        Returns:
            bool: 验证是否通过
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.add_error(
                module, 
                "MissingColumns", 
                f"Sheet '{sheet_name}' missing required columns: {missing_columns}"
            )
            return False
        
        # 检查必需列是否有空值
        for col in required_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.add_warning(
                    module,
                    "NullValues",
                    f"Sheet '{sheet_name}' column '{col}' has {null_count} null values"
                )
        
        return True

    def validate_positive_numbers(self, df: pd.DataFrame, columns: List[str], 
                                sheet_name: str, module: str) -> bool:
        """
        验证数值列是否为正数
        
        Args:
            df: 数据框
            columns: 数值列列表
            sheet_name: 表单名
            module: 模块名
            
        Returns:
            bool: 验证是否通过
        """
        has_issues = False
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # 检查负数
            invalid_values = df[df[col] < 0]
            if not invalid_values.empty:
                self.add_error(
                    module,
                    "InvalidValues",
                    f"Sheet '{sheet_name}' column '{col}' has {len(invalid_values)} non-positive values"
                )
                has_issues = True
        
        return not has_issues

    def validate_date_ranges(self, df: pd.DataFrame, start_col: str, end_col: str,
                           sheet_name: str, module: str) -> bool:
        """
        验证日期范围的有效性
        
        Args:
            df: 数据框
            start_col: 开始日期列
            end_col: 结束日期列
            sheet_name: 表单名
            module: 模块名
            
        Returns:
            bool: 验证是否通过
        """
        if start_col not in df.columns or end_col not in df.columns:
            return True
        
        # 检查结束日期是否早于开始日期
        invalid_ranges = df[df[end_col] < df[start_col]]
        
        if not invalid_ranges.empty:
            self.add_error(
                module,
                "InvalidDateRange",
                f"Sheet '{sheet_name}' has {len(invalid_ranges)} rows where {end_col} < {start_col}"
            )
            return False
        
        return True

def execute_with_validation(func, validation_manager: ValidationManager, 
                          module_name: str, *args, **kwargs):
    """
    带验证的函数执行装饰器
    
    Args:
        func: 要执行的函数
        validation_manager: 验证管理器
        module_name: 模块名
        *args, **kwargs: 函数参数
        
    Returns:
        函数执行结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        validation_manager.add_error(
            module_name, 
            "ExecutionError", 
            f"Function {func.__name__} failed: {str(e)}"
        )
        # 根据错误类型决定是否继续执行
        if is_critical_error(e):
            raise
        return get_default_result(func.__name__)

def is_critical_error(error: Exception) -> bool:
    """
    判断是否为关键错误
    
    Args:
        error: 异常对象
        
    Returns:
        bool: 是否为关键错误
    """
    critical_error_types = (
        FileNotFoundError,
        PermissionError,
        KeyError,
        ValueError,
        TypeError
    )
    
    return isinstance(error, critical_error_types)

def get_default_result(func_name: str) -> Any:
    """
    获取函数的默认返回值
    
    Args:
        func_name: 函数名
        
    Returns:
        默认返回值
    """
    # 根据函数名返回适当的默认值
    if 'dataframe' in func_name.lower() or 'load' in func_name.lower():
        return pd.DataFrame()
    elif 'dict' in func_name.lower():
        return {}
    elif 'list' in func_name.lower():
        return []
    else:
        return None