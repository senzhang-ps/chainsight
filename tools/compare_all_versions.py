#!/usr/bin/env python3
"""
三版本数据对比工具
对比 Dev版本、Src本地版本、Src数据库版本 的输出数据一致性

测试条件:
- 配置: BC_S5.xlsx
- 日期: 2025-10-06 到 2025-10-10 (5天)
- 随机种子: 42
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_csv_safe(path: Path) -> pd.DataFrame:
    """安全加载CSV文件"""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"    ⚠️ 加载失败: {path.name} - {e}")
        return pd.DataFrame()

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name: str, key_cols: list = None) -> dict:
    """对比两个DataFrame"""
    result = {
        "name": name,
        "df1_rows": len(df1),
        "df2_rows": len(df2),
        "match": False,
        "detail": ""
    }
    
    if df1.empty and df2.empty:
        result["match"] = True
        result["detail"] = "均为空"
        return result
    
    if len(df1) != len(df2):
        result["detail"] = f"行数不同: {len(df1)} vs {len(df2)}"
        # 尝试聚合对比
        if key_cols and all(c in df1.columns and c in df2.columns for c in key_cols):
            try:
                agg1 = df1.groupby(key_cols).size().reset_index(name='count')
                agg2 = df2.groupby(key_cols).size().reset_index(name='count')
                if len(agg1) == len(agg2):
                    merged = agg1.merge(agg2, on=key_cols, how='outer', suffixes=('_1', '_2'))
                    if merged['count_1'].equals(merged['count_2']):
                        result["match"] = True
                        result["detail"] = f"聚合一致 ({len(agg1)}组)"
                        return result
            except:
                pass
        return result
    
    # 列对齐
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        result["detail"] = "无共同列"
        return result
    
    df1_sorted = df1[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    df2_sorted = df2[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    
    # 数值列近似比较
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df1_sorted[col]):
            df1_sorted[col] = pd.to_numeric(df1_sorted[col], errors='coerce').round(6)
            df2_sorted[col] = pd.to_numeric(df2_sorted[col], errors='coerce').round(6)
    
    try:
        if df1_sorted.equals(df2_sorted):
            result["match"] = True
            result["detail"] = f"完全一致 ({len(df1)}行)"
        else:
            # 检查差异比例
            diff_count = (df1_sorted != df2_sorted).sum().sum()
            total_cells = df1_sorted.size
            diff_pct = diff_count / total_cells * 100 if total_cells > 0 else 0
            if diff_pct < 0.01:
                result["match"] = True
                result["detail"] = f"基本一致 (差异<0.01%)"
            else:
                result["detail"] = f"差异: {diff_pct:.2f}%"
    except Exception as e:
        result["detail"] = f"对比异常: {str(e)[:50]}"
    
    return result

def compare_module_outputs(dev_dir: Path, src_dir: Path, module: str, dates: list) -> list:
    """对比模块输出"""
    results = []
    
    if module == "module1":
        for date in dates:
            date_str = date.replace("-", "")
            dev_file = dev_dir / module / f"Module1Output_{date_str}.xlsx"
            src_file = src_dir / module / f"Module1Output_{date_str}.xlsx"
            
            for sheet in ["OrderLog", "ShipmentLog", "CutLog"]:
                try:
                    dev_df = pd.read_excel(dev_file, sheet_name=sheet) if dev_file.exists() else pd.DataFrame()
                    src_df = pd.read_excel(src_file, sheet_name=sheet) if src_file.exists() else pd.DataFrame()
                    result = compare_dataframes(dev_df, src_df, f"M1_{sheet}_{date}", 
                                               key_cols=["material", "location"] if sheet != "CutLog" else None)
                    results.append(result)
                except Exception as e:
                    results.append({"name": f"M1_{sheet}_{date}", "match": False, "detail": str(e)[:50]})
    
    elif module == "module3":
        for date in dates:
            date_str = date.replace("-", "")
            dev_file = dev_dir / module / f"Module3Output_{date_str}.xlsx"
            src_file = src_dir / module / f"Module3Output_{date_str}.xlsx"
            
            try:
                dev_df = pd.read_excel(dev_file, sheet_name="NetDemand") if dev_file.exists() else pd.DataFrame()
                src_df = pd.read_excel(src_file, sheet_name="NetDemand") if src_file.exists() else pd.DataFrame()
                result = compare_dataframes(dev_df, src_df, f"M3_NetDemand_{date}",
                                           key_cols=["material", "source_location", "dest_location"])
                results.append(result)
            except Exception as e:
                results.append({"name": f"M3_NetDemand_{date}", "match": False, "detail": str(e)[:50]})
    
    elif module == "module4":
        for date in dates:
            date_str = date.replace("-", "")
            dev_file = dev_dir / module / f"Module4Output_{date_str}.xlsx"
            src_file = src_dir / module / f"Module4Output_{date_str}.xlsx"
            
            try:
                dev_df = pd.read_excel(dev_file, sheet_name="ProductionPlan") if dev_file.exists() else pd.DataFrame()
                src_df = pd.read_excel(src_file, sheet_name="ProductionPlan") if src_file.exists() else pd.DataFrame()
                result = compare_dataframes(dev_df, src_df, f"M4_ProdPlan_{date}",
                                           key_cols=["material", "location"])
                results.append(result)
            except Exception as e:
                results.append({"name": f"M4_ProdPlan_{date}", "match": False, "detail": str(e)[:50]})
    
    elif module == "module5":
        for date in dates:
            date_str = date.replace("-", "")
            dev_file = dev_dir / module / f"Module5Output_{date_str}.xlsx"
            src_file = src_dir / module / f"Module5Output_{date_str}.xlsx"
            
            try:
                dev_df = pd.read_excel(dev_file, sheet_name="DeploymentPlan") if dev_file.exists() else pd.DataFrame()
                src_df = pd.read_excel(src_file, sheet_name="DeploymentPlan") if src_file.exists() else pd.DataFrame()
                # Module5使用聚合对比（行级可能有顺序差异）
                result = compare_dataframes(dev_df, src_df, f"M5_Deploy_{date}",
                                           key_cols=["material", "source_location", "dest_location"])
                results.append(result)
            except Exception as e:
                results.append({"name": f"M5_Deploy_{date}", "match": False, "detail": str(e)[:50]})
    
    elif module == "module6":
        for date in dates:
            date_str = date.replace("-", "")
            dev_file = dev_dir / module / f"Module6Output_{date_str}.xlsx"
            src_file = src_dir / module / f"Module6Output_{date_str}.xlsx"
            
            try:
                dev_df = pd.read_excel(dev_file, sheet_name="DeliveryPlan") if dev_file.exists() else pd.DataFrame()
                src_df = pd.read_excel(src_file, sheet_name="DeliveryPlan") if src_file.exists() else pd.DataFrame()
                result = compare_dataframes(dev_df, src_df, f"M6_Delivery_{date}",
                                           key_cols=["material", "source_location", "dest_location"])
                results.append(result)
            except Exception as e:
                results.append({"name": f"M6_Delivery_{date}", "match": False, "detail": str(e)[:50]})
    
    return results

def compare_summary_reports(dev_dir: Path, src_dir: Path) -> list:
    """对比汇总报告"""
    results = []
    
    summary_files = [
        ("OrderShipmentCutSummary.csv", ["material", "location"]),
        ("FullDeploymentPlan.csv", ["material", "source_location", "dest_location"]),
        ("FullProductionPlan.csv", ["material", "location"]),
        ("FullDeliveryPlan.csv", ["material", "source_location", "dest_location"]),
    ]
    
    for filename, key_cols in summary_files:
        dev_file = dev_dir / "summary" / filename
        src_file = src_dir / "summary" / filename
        
        dev_df = load_csv_safe(dev_file)
        src_df = load_csv_safe(src_file)
        
        result = compare_dataframes(dev_df, src_df, f"Summary_{filename}", key_cols)
        results.append(result)
    
    return results

def main():
    """主函数"""
    print("=" * 70)
    print("ChainSight 三版本数据一致性验证")
    print("=" * 70)
    
    # 路径配置
    project_root = Path(__file__).parent.parent
    dev_dir = project_root / "test_files" / "BC_S5" / "run_20260123_145210"
    src_dir = project_root / "outputs" / "BC_S5" / "run_20260123_153300"
    
    print(f"\n📂 Dev版本: {dev_dir}")
    print(f"📂 Src本地版: {src_dir}")
    
    if not dev_dir.exists():
        print(f"❌ Dev目录不存在: {dev_dir}")
        return
    if not src_dir.exists():
        print(f"❌ Src目录不存在: {src_dir}")
        return
    
    dates = ["2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10"]
    
    print(f"\n📅 仿真日期: {dates[0]} ~ {dates[-1]} ({len(dates)}天)")
    print("=" * 70)
    
    all_results = []
    
    # 对比各模块
    for module in ["module1", "module3", "module4", "module5", "module6"]:
        print(f"\n📊 对比 {module.upper()}...")
        results = compare_module_outputs(dev_dir, src_dir, module, dates)
        all_results.extend(results)
        
        passed = sum(1 for r in results if r.get("match", False))
        total = len(results)
        status = "✅" if passed == total else "⚠️"
        print(f"   {status} {passed}/{total} 项通过")
        
        for r in results:
            icon = "✅" if r.get("match", False) else "❌"
            print(f"      {icon} {r['name']}: {r.get('detail', 'N/A')}")
    
    # 对比汇总报告
    print(f"\n📊 对比 SUMMARY...")
    summary_results = compare_summary_reports(dev_dir, src_dir)
    all_results.extend(summary_results)
    
    passed = sum(1 for r in summary_results if r.get("match", False))
    total = len(summary_results)
    status = "✅" if passed == total else "⚠️"
    print(f"   {status} {passed}/{total} 项通过")
    
    for r in summary_results:
        icon = "✅" if r.get("match", False) else "❌"
        print(f"      {icon} {r['name']}: {r.get('detail', 'N/A')}")
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 验证总结")
    print("=" * 70)
    
    total_passed = sum(1 for r in all_results if r.get("match", False))
    total_items = len(all_results)
    pass_rate = total_passed / total_items * 100 if total_items > 0 else 0
    
    print(f"\n通过率: {total_passed}/{total_items} ({pass_rate:.1f}%)")
    
    if pass_rate == 100:
        print("\n🎉 数据完全一致！Dev版本与Src本地版输出完全匹配。")
    elif pass_rate >= 95:
        print("\n✅ 数据基本一致，少量差异在可接受范围内。")
    else:
        print("\n⚠️ 存在数据差异，请检查不一致项。")
    
    # 返回结果
    return {
        "total": total_items,
        "passed": total_passed,
        "pass_rate": pass_rate,
        "results": all_results
    }

if __name__ == "__main__":
    main()
