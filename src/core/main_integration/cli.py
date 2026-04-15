"""
cli.py

命令行接口入口点模块。
"""

import sys
import os
import argparse
import io

# Windows UTF-8 编码设置 - 解决emoji和中文输出问题
if sys.platform == 'win32':
    # 设置stdout/stderr为UTF-8编码
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from .simulation_file import run_integrated_simulation
from .resume import check_resume_capability


def main():
    """主函数 - 命令行入口执行集成仿真

    目的：
    - 解析命令行参数，进行存在性检查与默认值处理，支持仅检查断点续跑状态或执行完整仿真。

    输入/输出/逻辑：
    - 解析参数→检查配置文件→构造默认输出目录→可选断点续跑检查→调用 `run_integrated_simulation` 并打印结果或错误。
    """
    # 配置文件路径（可以通过命令行参数或环境变量指定）
    import argparse
    
    parser = argparse.ArgumentParser(description="运行供应链集成仿真")
    parser.add_argument("--config", "-c", 
                       default="./config/integration_config.json",
                       help="配置文件路径 (默认: ./config/integration_config.json)")
    parser.add_argument("--start-date", "-s", 
                       default="2024-01-01",
                       help="仿真开始日期 (默认: 2024-01-01)")
    parser.add_argument("--end-date", "-e", 
                       default="2024-01-05",
                       help="仿真结束日期 (默认: 2024-01-05)")
    parser.add_argument("--output", "-o", 
                       default=None,
                       help="输出目录 (默认: 根据配置文件名生成)")
    parser.add_argument("--force-restart", 
                       action="store_true",
                       help="强制从头开始，忽略断点续跑状态 (默认: False)")
    parser.add_argument("--check-resume", 
                       action="store_true",
                       help="仅检查断点续跑状态，不执行仿真 (默认: False)")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        sys.exit(1)
    
    # 如果没有指定输出目录，根据配置文件名生成
    if args.output is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f"./{config_name}_output"
    
    # 处理断点续跑检查选项
    if args.check_resume:
        resume_info = check_resume_capability(args.output, args.start_date, args.end_date)
        
        
        if resume_info.get('already_completed', False):
            pass
            pass
            pass
        elif resume_info['can_resume']:
            pass
            pass
            pass
        else:
            pass
            pass
        
        return  # 仅检查，不执行
    
    # 处理强制重启选项
    if args.force_restart:
        pass
        # 通过参数透传给仿真入口，显式要求忽略已有断点续跑状态
    
    try:
        # 向仿真入口透传 force_restart 参数
        result = run_integrated_simulation(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base_dir=args.output,
            force_restart=args.force_restart
        )
        
        if result.get('is_resuming', False):
            pass
            pass
            pass
        else:
            pass
            pass
        
        if result.get('already_completed', False):
            pass
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
