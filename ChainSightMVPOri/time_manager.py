# time_manager.py
# 统一时间管理器 - 解决各模块对当前日期处理方式不统一的问题
# 提供标准化的日期处理接口

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class SimulationTimeManager:
    """
    仿真时间管理器
    
    统一管理仿真过程中的时间处理，确保各模块使用相同的时间标准
    """
    
    def __init__(self, start_date: str):
        """
        初始化时间管理器
        
        Args:
            start_date: 仿真开始日期 (YYYY-MM-DD)
        """
        self.start_date = pd.to_datetime(start_date).normalize()
        self.current_date = self.start_date
        self.date_format = '%Y-%m-%d'
        self.file_date_format = '%Y%m%d'
    
    def get_current_date(self) -> pd.Timestamp:
        """
        获取标准化的当前日期 (pandas Timestamp, normalized)
        
        Returns:
            pd.Timestamp: 当前仿真日期
        """
        return self.current_date
    
    def get_date_string(self, fmt: Optional[str] = None) -> str:
        """
        获取格式化的日期字符串
        
        Args:
            fmt: 日期格式，默认为 '%Y-%m-%d'
            
        Returns:
            str: 格式化的日期字符串
        """
        if fmt is None:
            fmt = self.date_format
        return self.current_date.strftime(fmt)
    
    def get_file_date_string(self) -> str:
        """
        获取文件命名用的日期字符串 (YYYYMMDD)
        
        Returns:
            str: 文件命名用的日期字符串
        """
        return self.current_date.strftime(self.file_date_format)
    
    def get_previous_date(self, days: int = 1) -> pd.Timestamp:
        """
        获取前几天的日期
        
        Args:
            days: 前推天数，默认为1天
            
        Returns:
            pd.Timestamp: 前几天的日期
        """
        return self.current_date - pd.Timedelta(days=days)
    
    def get_previous_date_string(self, days: int = 1, fmt: Optional[str] = None) -> str:
        """
        获取前几天的日期字符串
        
        Args:
            days: 前推天数，默认为1天
            fmt: 日期格式，默认为 '%Y-%m-%d'
            
        Returns:
            str: 前几天的日期字符串
        """
        if fmt is None:
            fmt = self.date_format
        prev_date = self.get_previous_date(days)
        return prev_date.strftime(fmt)
    
    def get_previous_file_date_string(self, days: int = 1) -> str:
        """
        获取前几天的文件日期字符串
        
        Args:
            days: 前推天数，默认为1天
            
        Returns:
            str: 前几天的文件日期字符串
        """
        prev_date = self.get_previous_date(days)
        return prev_date.strftime(self.file_date_format)
    
    def get_future_date(self, days: int = 1) -> pd.Timestamp:
        """
        获取未来几天的日期
        
        Args:
            days: 未来天数，默认为1天
            
        Returns:
            pd.Timestamp: 未来几天的日期
        """
        return self.current_date + pd.Timedelta(days=days)
    
    def advance_day(self) -> pd.Timestamp:
        """
        推进到下一天
        
        Returns:
            pd.Timestamp: 推进后的当前日期
        """
        self.current_date += pd.Timedelta(days=1)
        return self.current_date
    
    def set_current_date(self, date: str) -> pd.Timestamp:
        """
        设置当前日期
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            pd.Timestamp: 设置后的当前日期
        """
        self.current_date = pd.to_datetime(date).normalize()
        return self.current_date
    
    def is_simulation_day(self, date: pd.Timestamp) -> bool:
        """
        检查是否为仿真日期范围内
        
        Args:
            date: 要检查的日期
            
        Returns:
            bool: 是否在仿真范围内
        """
        return date >= self.start_date
    
    def get_simulation_days_count(self) -> int:
        """
        获取仿真已进行的天数
        
        Returns:
            int: 仿真天数
        """
        return (self.current_date - self.start_date).days + 1
    
    def generate_date_range(self, end_date: str) -> pd.DatetimeIndex:
        """
        生成仿真日期范围
        
        Args:
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            pd.DatetimeIndex: 日期范围
        """
        end_dt = pd.to_datetime(end_date).normalize()
        return pd.date_range(self.start_date, end_dt, freq='D')
    
    def format_date_for_module(self, date: pd.Timestamp, module: str) -> str:
        """
        为特定模块格式化日期
        
        Args:
            date: 日期
            module: 模块名
            
        Returns:
            str: 格式化的日期字符串
        """
        # 可以根据不同模块的需求返回不同格式
        module_formats = {
            'module1': '%Y-%m-%d',
            'module3': '%Y-%m-%d', 
            'module4': '%Y-%m-%d',
            'module5': '%Y-%m-%d',
            'module6': '%Y-%m-%d',
            'orchestrator': '%Y-%m-%d'
        }
        
        fmt = module_formats.get(module.lower(), self.date_format)
        return date.strftime(fmt)
    
    def parse_file_date(self, file_date_str: str) -> pd.Timestamp:
        """
        解析文件日期字符串
        
        Args:
            file_date_str: 文件日期字符串 (YYYYMMDD)
            
        Returns:
            pd.Timestamp: 解析后的日期
        """
        return pd.to_datetime(file_date_str, format=self.file_date_format).normalize()
    
    def validate_date_sequence(self, dates: list) -> bool:
        """
        验证日期序列的连续性
        
        Args:
            dates: 日期列表
            
        Returns:
            bool: 是否连续
        """
        if len(dates) <= 1:
            return True
        
        sorted_dates = sorted([pd.to_datetime(d).normalize() for d in dates])
        
        for i in range(1, len(sorted_dates)):
            diff = (sorted_dates[i] - sorted_dates[i-1]).days
            if diff != 1:
                return False
        
        return True
    
    def get_summary(self) -> dict:
        """
        获取时间管理器状态摘要
        
        Returns:
            dict: 状态摘要
        """
        return {
            'start_date': self.start_date.strftime(self.date_format),
            'current_date': self.current_date.strftime(self.date_format),
            'simulation_days': self.get_simulation_days_count(),
            'date_format': self.date_format,
            'file_date_format': self.file_date_format
        }

# 全局时间管理器实例
_global_time_manager: Optional[SimulationTimeManager] = None

def initialize_time_manager(start_date: str) -> SimulationTimeManager:
    """
    初始化全局时间管理器
    
    Args:
        start_date: 仿真开始日期
        
    Returns:
        SimulationTimeManager: 时间管理器实例
    """
    global _global_time_manager
    _global_time_manager = SimulationTimeManager(start_date)
    return _global_time_manager

def get_time_manager() -> SimulationTimeManager:
    """
    获取全局时间管理器
    
    Returns:
        SimulationTimeManager: 时间管理器实例
        
    Raises:
        RuntimeError: 如果时间管理器未初始化
    """
    global _global_time_manager
    if _global_time_manager is None:
        raise RuntimeError("Time manager not initialized. Call initialize_time_manager() first.")
    return _global_time_manager

def reset_time_manager():
    """重置全局时间管理器"""
    global _global_time_manager
    _global_time_manager = None