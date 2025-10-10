#!/usr/bin/env python3
"""
日志配置模块 - 统一管理所有模块的日志输出
支持同时输出到terminal和txt文件
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class DualLogger:
    """双输出日志器 - 同时输出到控制台和文件"""
    
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        """
        初始化双输出日志器
        
        Args:
            output_dir: 日志文件输出目录
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"simulation_log_{timestamp}.txt"
        
        # 配置根日志记录器
        self.logger = logging.getLogger("SupplyChainSimulation")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除已有的处理器（避免重复）
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的信息
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 记录日志器初始化
        self.logger.info(f"日志系统已初始化，日志文件: {self.log_file}")
    
    def get_logger(self):
        """获取日志记录器"""
        return self.logger
    
    def info(self, msg: str):
        """记录INFO级别日志"""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """记录DEBUG级别日志"""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """记录WARNING级别日志"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """记录ERROR级别日志"""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(msg)


class PrintRedirector:
    """重定向print输出到日志系统"""
    
    def __init__(self, logger):
        self.logger = logger
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def write(self, message):
        """重定向write方法"""
        if message.strip():  # 忽略空行
            self.logger.info(message.strip())
    
    def flush(self):
        """实现flush方法（必需）"""
        pass
    
    def start_redirect(self):
        """开始重定向print"""
        sys.stdout = self
        sys.stderr = self
    
    def stop_redirect(self):
        """停止重定向print"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


def setup_logging(output_dir: str, log_level: str = "INFO", redirect_print: bool = False):
    """
    设置日志系统
    
    Args:
        output_dir: 日志输出目录
        log_level: 日志级别
        redirect_print: 是否重定向所有print输出
    
    Returns:
        tuple: (logger, redirector) 如果redirect_print=True，否则只返回logger
    """
    dual_logger = DualLogger(output_dir, log_level)
    logger = dual_logger.get_logger()
    
    if redirect_print:
        redirector = PrintRedirector(logger)
        redirector.start_redirect()
        return logger, redirector
    
    return logger, None


# 便捷函数：创建简易文件日志
def create_simple_file_logger(output_dir: str, filename: str = "simulation.log"):
    """
    创建简单的文件日志记录器
    
    Args:
        output_dir: 输出目录
        filename: 日志文件名
    
    Returns:
        logging.Logger对象
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / filename
    
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# 使用示例
if __name__ == "__main__":
    # 示例1: 使用DualLogger（推荐）
    logger, _ = setup_logging("./test_logs", log_level="DEBUG")
    logger.info("这是一条INFO日志")
    logger.debug("这是一条DEBUG日志")
    logger.warning("这是一条WARNING日志")
    
    # 示例2: 重定向所有print
    logger2, redirector = setup_logging("./test_logs", log_level="INFO", redirect_print=True)
    print("这条print会被重定向到日志文件")
    print("🚀 支持emoji和中文")
    
    # 恢复原始输出
    if redirector:
        redirector.stop_redirect()
    print("这条print回到了原始terminal")
