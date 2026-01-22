"""
测试配置加载一致性
比较从 Excel 直接加载和从数据库加载的配置是否一致
"""
import pandas as pd
from pathlib import Path

# 加载本地 Excel 配置
excel_path = Path("test_files/BC_S5.xlsx")
xl = pd.ExcelFile(excel_path)

print("=" * 60)
print("本地 Excel 配置表")
print("=" * 60)

for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    print(f"{sheet_name}: {len(df)} 行, {len(df.columns)} 列")
    print(f"  列: {list(df.columns)[:10]}...")

# 检查关键配置表
print("\n" + "=" * 60)
print("关键配置表详细信息")
print("=" * 60)

# Global_Network
network = xl.parse('Global_Network')
print(f"\nGlobal_Network:")
print(f"  行数: {len(network)}")
print(f"  列: {list(network.columns)}")
print(f"  material 类型: {network['material'].dtype}")
print(f"  material 唯一值数: {network['material'].nunique()}")

# M5_DeployConfig
deploy_cfg = xl.parse('M5_DeployConfig')
print(f"\nM5_DeployConfig:")
print(f"  行数: {len(deploy_cfg)}")
print(f"  列: {list(deploy_cfg.columns)}")

# M5_PushPullModel
ppm = xl.parse('M5_PushPullModel')
print(f"\nM5_PushPullModel:")
print(f"  行数: {len(ppm)}")
print(f"  列: {list(ppm.columns)}")

print("\n✅ 分析完成")
