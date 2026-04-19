"""比较两次模拟运行的结果"""
import pandas as pd
import os

base_path = r'c:\Users\25936\Desktop\test\chainsight-fix-future-production-in-net-demand\chainsight-fix-future-production-in-net-demand\BC_S5'
old_run = 'run_20260119_145719'
new_run = 'run_20260121_104059'

print('='*80)
print('BC_S5 两次运行结果对比分析')
print('='*80)
print(f'旧运行: {old_run}')
print(f'新运行: {new_run}')
print()

# 1. 比较历史库存记录
print('📦 1. 历史库存记录 (historical_inventory_record.csv) 对比:')
print('-'*60)
old_inv = pd.read_csv(os.path.join(base_path, old_run, 'summary', 'historical_inventory_record.csv'))
new_inv = pd.read_csv(os.path.join(base_path, new_run, 'summary', 'historical_inventory_record.csv'))
print(f'旧运行记录数: {len(old_inv)}')
print(f'新运行记录数: {len(new_inv)}')

if 'date' in old_inv.columns:
    print(f'旧运行日期范围: {old_inv["date"].min()} 到 {old_inv["date"].max()}')
    print(f'新运行日期范围: {new_inv["date"].min()} 到 {new_inv["date"].max()}')

# 限制到相同日期范围进行比较
common_dates = set(old_inv['date'].unique()) & set(new_inv['date'].unique())
print(f'共同日期数: {len(common_dates)}')

if common_dates:
    old_filtered = old_inv[old_inv['date'].isin(common_dates)]
    new_filtered = new_inv[new_inv['date'].isin(common_dates)]
    
    if 'location' in old_inv.columns and 'material' in old_inv.columns:
        qty_col = 'ending_inventory' if 'ending_inventory' in old_inv.columns else 'quantity'
        old_grouped = old_filtered.groupby(['date', 'location', 'material'])[qty_col].sum().reset_index()
        new_grouped = new_filtered.groupby(['date', 'location', 'material'])[qty_col].sum().reset_index()
        merged = old_grouped.merge(new_grouped, on=['date', 'location', 'material'], how='outer', suffixes=('_old', '_new'))
        merged['diff'] = merged[f'{qty_col}_new'].fillna(0) - merged[f'{qty_col}_old'].fillna(0)
        diff_count = (merged['diff'] != 0).sum()
        print(f'库存数量差异记录数: {diff_count}')
        if diff_count > 0:
            print('库存差异示例 (最多5条):')
            print(merged[merged['diff'] != 0].head())
        else:
            print('✅ 所有共同日期的库存数据完全一致!')

print()

# 2. 比较订单发货cut报告
print('📋 2. 订单发货Cut报告对比:')
print('-'*60)
try:
    old_order = pd.read_excel(os.path.join(base_path, old_run, 'summary', 'full_order_shipment_cut_report.xlsx'))
    new_order = pd.read_excel(os.path.join(base_path, new_run, 'summary', 'full_order_shipment_cut_report.xlsx'))
    print(f'旧运行记录数: {len(old_order)}')
    print(f'新运行记录数: {len(new_order)}')
    
    # 比较总发货量
    if 'shipment_quantity' in old_order.columns:
        old_ship = old_order['shipment_quantity'].sum()
        new_ship = new_order['shipment_quantity'].sum()
        print(f'旧运行总发货量: {old_ship:,.0f}')
        print(f'新运行总发货量: {new_ship:,.0f}')
        if old_ship == new_ship:
            print('✅ 总发货量一致!')
        else:
            print(f'⚠️ 发货量差异: {new_ship - old_ship:,.0f}')
    
    if 'cut_quantity' in old_order.columns:
        old_cut = old_order['cut_quantity'].sum()
        new_cut = new_order['cut_quantity'].sum()
        print(f'旧运行总Cut量: {old_cut:,.0f}')
        print(f'新运行总Cut量: {new_cut:,.0f}')
        if old_cut == new_cut:
            print('✅ 总Cut量一致!')
        else:
            print(f'⚠️ Cut量差异: {new_cut - old_cut:,.0f}')
except Exception as e:
    print(f'读取订单报告出错: {e}')

print()

# 3. 比较部署计划报告
print('🚚 3. 部署计划报告对比:')
print('-'*60)
try:
    old_deploy = pd.read_excel(os.path.join(base_path, old_run, 'summary', 'full_deployment_plan_report.xlsx'))
    new_deploy = pd.read_excel(os.path.join(base_path, new_run, 'summary', 'full_deployment_plan_report.xlsx'))
    print(f'旧运行记录数: {len(old_deploy)}')
    print(f'新运行记录数: {len(new_deploy)}')
    
    if 'qty' in old_deploy.columns:
        old_qty = old_deploy['qty'].sum()
        new_qty = new_deploy['qty'].sum()
        print(f'旧运行总部署量: {old_qty:,.0f}')
        print(f'新运行总部署量: {new_qty:,.0f}')
        if old_qty == new_qty:
            print('✅ 总部署量一致!')
        else:
            print(f'⚠️ 部署量差异: {new_qty - old_qty:,.0f}')
except Exception as e:
    print(f'读取部署报告出错: {e}')

print()

# 4. 比较生产计划报告
print('🏭 4. 生产计划报告对比:')
print('-'*60)
try:
    old_prod = pd.read_excel(os.path.join(base_path, old_run, 'summary', 'full_production_plan_report.xlsx'))
    new_prod = pd.read_excel(os.path.join(base_path, new_run, 'summary', 'full_production_plan_report.xlsx'))
    print(f'旧运行记录数: {len(old_prod)}')
    print(f'新运行记录数: {len(new_prod)}')
    
    qty_col = None
    for col in ['production_qty', 'qty', 'quantity']:
        if col in old_prod.columns:
            qty_col = col
            break
    
    if qty_col:
        old_qty = old_prod[qty_col].sum()
        new_qty = new_prod[qty_col].sum()
        print(f'旧运行总生产量: {old_qty:,.0f}')
        print(f'新运行总生产量: {new_qty:,.0f}')
        if old_qty == new_qty:
            print('✅ 总生产量一致!')
        else:
            print(f'⚠️ 生产量差异: {new_qty - old_qty:,.0f}')
except Exception as e:
    print(f'读取生产报告出错: {e}')

print()

# 5. 比较交付计划报告
print('📦 5. 交付计划报告对比:')
print('-'*60)
try:
    old_delivery = pd.read_excel(os.path.join(base_path, old_run, 'summary', 'full_delivery_plan_report.xlsx'))
    new_delivery = pd.read_excel(os.path.join(base_path, new_run, 'summary', 'full_delivery_plan_report.xlsx'))
    print(f'旧运行记录数: {len(old_delivery)}')
    print(f'新运行记录数: {len(new_delivery)}')
    
    qty_col = None
    for col in ['delivery_qty', 'qty', 'quantity']:
        if col in old_delivery.columns:
            qty_col = col
            break
    
    if qty_col:
        old_qty = old_delivery[qty_col].sum()
        new_qty = new_delivery[qty_col].sum()
        print(f'旧运行总交付量: {old_qty:,.0f}')
        print(f'新运行总交付量: {new_qty:,.0f}')
        if old_qty == new_qty:
            print('✅ 总交付量一致!')
        else:
            print(f'⚠️ 交付量差异: {new_qty - old_qty:,.0f}')
except Exception as e:
    print(f'读取交付报告出错: {e}')

print()
print('='*80)
print('对比分析完成!')
print('='*80)
