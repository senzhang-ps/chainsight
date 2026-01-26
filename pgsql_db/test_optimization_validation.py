# -*- coding: utf-8 -*-
"""
优化验证测试脚本

测试目的：
1. 运行优化后的 DB 版本仿真
2. 与 ChainSight_Dev 的输出进行对比
3. 确保数据一致性

使用方法：
    python -m pgsql_db.test_optimization_validation --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_chainsight_dev_output(output_dir: Path, module: str, date: str) -> Dict[str, pd.DataFrame]:
    """
    加载 ChainSight_Dev 的输出文件。
    
    Args:
        output_dir: 输出目录
        date: 日期字符串 YYYYMMDD
        module: 模块名（如 module3）
        
    Returns:
        输出数据字典
    """
    results = {}
    module_dir = output_dir / module
    
    if not module_dir.exists():
        print(f"⚠️ 目录不存在: {module_dir}")
        return results
    
    # 查找该日期的文件
    date_pattern = date.replace('-', '')
    for file_path in module_dir.glob(f"*{date_pattern}*"):
        if file_path.suffix in ['.csv', '.xlsx']:
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                results[file_path.name] = df
            except Exception as e:
                print(f"⚠️ 加载失败 {file_path}: {e}")
    
    return results


def compare_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name: str,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    比较两个 DataFrame。
    
    Args:
        df1: 第一个DataFrame (ChainSight_Dev)
        df2: 第二个DataFrame (优化版本)
        name: 名称
        tolerance: 数值容差
        
    Returns:
        (是否一致, 差异列表)
    """
    differences = []
    
    # 检查行数
    if len(df1) != len(df2):
        differences.append(f"行数不同: {len(df1)} vs {len(df2)}")
    
    # 检查列
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 != cols2:
        extra_in_1 = cols1 - cols2
        extra_in_2 = cols2 - cols1
        if extra_in_1:
            differences.append(f"原版多出列: {extra_in_1}")
        if extra_in_2:
            differences.append(f"优化版多出列: {extra_in_2}")
    
    # 比较共同列的数据
    common_cols = cols1 & cols2
    
    for col in common_cols:
        try:
            # 尝试数值比较
            v1 = pd.to_numeric(df1[col], errors='coerce')
            v2 = pd.to_numeric(df2[col], errors='coerce')
            
            if v1.notna().any() and v2.notna().any():
                # 数值列比较
                diff = (v1 - v2).abs()
                max_diff = diff.max()
                if max_diff > tolerance:
                    differences.append(
                        f"列 '{col}' 数值差异: 最大差值={max_diff:.6f}"
                    )
            else:
                # 字符串比较
                if not df1[col].astype(str).equals(df2[col].astype(str)):
                    mismatch_count = (
                        df1[col].astype(str) != df2[col].astype(str)
                    ).sum()
                    differences.append(
                        f"列 '{col}' 字符串不匹配: {mismatch_count} 行"
                    )
        except Exception as e:
            differences.append(f"列 '{col}' 比较失败: {e}")
    
    is_consistent = len(differences) == 0
    return is_consistent, differences


def run_chainsight_dev_simulation(
    config_name: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    运行 ChainSight_Dev 版本作为基准。
    
    Args:
        config_name: 配置名称
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        
    Returns:
        运行结果
    """
    print("\n" + "=" * 60)
    print("🔧 运行 ChainSight_Dev 基准版本")
    print("=" * 60)
    
    chainsight_dev_dir = PROJECT_ROOT / "ChainSight_Dev"
    
    if not chainsight_dev_dir.exists():
        print(f"❌ ChainSight_Dev 目录不存在: {chainsight_dev_dir}")
        return {'success': False, 'error': 'ChainSight_Dev not found'}
    
    # 切换到 ChainSight_Dev 目录并运行
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(chainsight_dev_dir))
        
        # 运行 ChainSight_Dev
        import subprocess
        cmd = [
            sys.executable, 'run.py',
            '--config', config_name,
            '--start-date', start_date,
            '--end-date', end_date
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        t0 = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            print(f"❌ ChainSight_Dev 运行失败:")
            print(result.stderr)
            return {
                'success': False,
                'error': result.stderr,
                'elapsed': elapsed
            }
        
        print(f"✅ ChainSight_Dev 完成, 耗时: {elapsed:.2f}秒")
        
        return {
            'success': True,
            'elapsed': elapsed,
            'stdout': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print("❌ ChainSight_Dev 运行超时")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ ChainSight_Dev 运行异常: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        os.chdir(original_cwd)


def run_optimized_simulation(
    config_name: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    use_db: bool = True
) -> Dict[str, Any]:
    """
    运行优化版本的仿真。
    
    Args:
        config_name: 配置名称
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        use_db: 是否使用数据库模式
        
    Returns:
        运行结果
    """
    print("\n" + "=" * 60)
    print("🚀 运行优化版本")
    print("=" * 60)
    
    try:
        # 运行优化版本
        cmd = [
            sys.executable, 'run.py',
            '--config', config_name,
            '--start-date', start_date,
            '--end-date', end_date
        ]
        
        if use_db:
            cmd.append('--use-db')
        
        print(f"执行命令: {' '.join(cmd)}")
        t0 = time.time()
        
        import subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT)
        )
        
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            print(f"❌ 优化版本运行失败:")
            print(result.stderr)
            return {
                'success': False,
                'error': result.stderr,
                'elapsed': elapsed
            }
        
        print(f"✅ 优化版本完成, 耗时: {elapsed:.2f}秒")
        
        return {
            'success': True,
            'elapsed': elapsed,
            'stdout': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print("❌ 优化版本运行超时")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ 优化版本运行异常: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def find_latest_output_dir(base_dir: Path, prefix: str) -> Optional[Path]:
    """
    查找最新的输出目录。
    
    Args:
        base_dir: 基础目录
        prefix: 目录前缀
        
    Returns:
        最新的输出目录
    """
    matching_dirs = list(base_dir.glob(f"{prefix}*"))
    
    if not matching_dirs:
        return None
    
    # 按修改时间排序
    matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matching_dirs[0]


def validate_outputs(
    chainsight_dev_output: Path,
    optimized_output: Path,
    modules: List[str] = None,
    dates: List[str] = None
) -> Dict[str, Any]:
    """
    验证两个版本的输出一致性。
    
    Args:
        chainsight_dev_output: ChainSight_Dev 输出目录
        optimized_output: 优化版本输出目录
        modules: 要验证的模块列表
        dates: 要验证的日期列表
        
    Returns:
        验证结果
    """
    print("\n" + "=" * 60)
    print("🔍 验证输出一致性")
    print("=" * 60)
    
    if modules is None:
        modules = ['module1', 'module3', 'module4', 'module5', 'module6']
    
    validation_results = {
        'total_files': 0,
        'consistent_files': 0,
        'inconsistent_files': 0,
        'details': []
    }
    
    for module in modules:
        dev_module_dir = chainsight_dev_output / module
        opt_module_dir = optimized_output / module
        
        if not dev_module_dir.exists():
            print(f"⚠️ ChainSight_Dev 没有 {module} 输出")
            continue
        
        if not opt_module_dir.exists():
            print(f"⚠️ 优化版本没有 {module} 输出")
            continue
        
        print(f"\n📂 比较 {module}:")
        
        # 获取所有输出文件
        dev_files = set(f.name for f in dev_module_dir.glob('*.xlsx'))
        opt_files = set(f.name for f in opt_module_dir.glob('*.xlsx'))
        
        common_files = dev_files & opt_files
        
        for filename in sorted(common_files):
            # 如果指定了日期过滤
            if dates:
                if not any(d.replace('-', '') in filename for d in dates):
                    continue
            
            validation_results['total_files'] += 1
            
            try:
                dev_path = dev_module_dir / filename
                opt_path = opt_module_dir / filename
                
                # 对于 Excel 文件，比较所有 sheet
                dev_xlsx = pd.ExcelFile(dev_path)
                opt_xlsx = pd.ExcelFile(opt_path)
                
                file_consistent = True
                file_differences = []
                
                for sheet in dev_xlsx.sheet_names:
                    if sheet not in opt_xlsx.sheet_names:
                        file_differences.append(f"优化版缺少sheet: {sheet}")
                        file_consistent = False
                        continue
                    
                    df_dev = pd.read_excel(dev_xlsx, sheet_name=sheet)
                    df_opt = pd.read_excel(opt_xlsx, sheet_name=sheet)
                    
                    consistent, diffs = compare_dataframes(
                        df_dev, df_opt, f"{filename}/{sheet}"
                    )
                    
                    if not consistent:
                        file_consistent = False
                        file_differences.extend(
                            [f"[{sheet}] {d}" for d in diffs]
                        )
                
                if file_consistent:
                    validation_results['consistent_files'] += 1
                    print(f"  ✅ {filename}")
                else:
                    validation_results['inconsistent_files'] += 1
                    print(f"  ❌ {filename}")
                    for diff in file_differences[:3]:  # 只显示前3个差异
                        print(f"     - {diff}")
                    if len(file_differences) > 3:
                        print(f"     ... 还有 {len(file_differences)-3} 个差异")
                    
                    validation_results['details'].append({
                        'file': filename,
                        'differences': file_differences
                    })
                    
            except Exception as e:
                validation_results['inconsistent_files'] += 1
                print(f"  ❌ {filename}: 比较失败 - {e}")
                validation_results['details'].append({
                    'file': filename,
                    'error': str(e)
                })
    
    # 打印汇总
    print("\n" + "-" * 40)
    print("📊 验证汇总:")
    print(f"   总文件数: {validation_results['total_files']}")
    print(f"   一致: {validation_results['consistent_files']}")
    print(f"   不一致: {validation_results['inconsistent_files']}")
    
    if validation_results['inconsistent_files'] == 0:
        print("✅ 所有输出一致!")
    else:
        print("⚠️ 存在不一致，请检查详细日志")
    
    return validation_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='优化验证测试'
    )
    parser.add_argument(
        '--config', '-c',
        default='BC_S5',
        help='配置名称'
    )
    parser.add_argument(
        '--start-date', '-s',
        default='2025-10-06',
        help='开始日期'
    )
    parser.add_argument(
        '--end-date', '-e',
        default='2025-10-10',
        help='结束日期'
    )
    parser.add_argument(
        '--skip-dev',
        action='store_true',
        help='跳过 ChainSight_Dev 运行'
    )
    parser.add_argument(
        '--skip-optimized',
        action='store_true',
        help='跳过优化版本运行'
    )
    parser.add_argument(
        '--dev-output',
        help='ChainSight_Dev 输出目录（如果跳过运行）'
    )
    parser.add_argument(
        '--opt-output',
        help='优化版本输出目录（如果跳过运行）'
    )
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='不使用数据库模式'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🧪 优化验证测试")
    print("=" * 70)
    print(f"配置: {args.config}")
    print(f"日期范围: {args.start_date} 到 {args.end_date}")
    print("=" * 70)
    
    output_base = PROJECT_ROOT / "outputs"
    
    # 运行 ChainSight_Dev
    if not args.skip_dev:
        dev_result = run_chainsight_dev_simulation(
            args.config,
            args.start_date,
            args.end_date,
            output_base
        )
        if not dev_result['success']:
            print("❌ ChainSight_Dev 运行失败，无法继续验证")
            return 1
        dev_elapsed = dev_result['elapsed']
    else:
        dev_elapsed = 0
        print("⏭️ 跳过 ChainSight_Dev 运行")
    
    # 运行优化版本
    if not args.skip_optimized:
        opt_result = run_optimized_simulation(
            args.config,
            args.start_date,
            args.end_date,
            output_base,
            use_db=not args.no_db
        )
        if not opt_result['success']:
            print("❌ 优化版本运行失败")
            return 1
        opt_elapsed = opt_result['elapsed']
    else:
        opt_elapsed = 0
        print("⏭️ 跳过优化版本运行")
    
    # 查找输出目录
    if args.dev_output:
        dev_output_dir = Path(args.dev_output)
    else:
        # 在 ChainSight_Dev/outputs 下查找
        chainsight_dev_outputs = PROJECT_ROOT / "ChainSight_Dev" / "outputs"
        dev_output_dir = find_latest_output_dir(
            chainsight_dev_outputs, args.config
        )
    
    if args.opt_output:
        opt_output_dir = Path(args.opt_output)
    else:
        opt_output_dir = find_latest_output_dir(output_base, args.config)
    
    if dev_output_dir and opt_output_dir:
        print(f"\nChainSight_Dev 输出: {dev_output_dir}")
        print(f"优化版本输出: {opt_output_dir}")
        
        # 验证输出一致性
        validation = validate_outputs(
            dev_output_dir,
            opt_output_dir
        )
        
        # 打印性能对比
        print("\n" + "=" * 60)
        print("⏱️ 性能对比")
        print("=" * 60)
        if dev_elapsed > 0:
            print(f"ChainSight_Dev 耗时: {dev_elapsed:.2f}秒")
        if opt_elapsed > 0:
            print(f"优化版本耗时: {opt_elapsed:.2f}秒")
        if dev_elapsed > 0 and opt_elapsed > 0:
            speedup = dev_elapsed / opt_elapsed
            print(f"加速比: {speedup:.2f}x")
        print("=" * 60)
        
        if validation['inconsistent_files'] > 0:
            return 1
    else:
        print("⚠️ 找不到输出目录，跳过验证")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
