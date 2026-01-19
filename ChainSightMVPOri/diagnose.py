import pandas as pd
import cProfile
import pstats
import io

# 模拟你真实的数据规模
print("创建接近真实规模的测试数据...")
large_df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=50000). astype(str). tolist() * 10,  # 50万行
    'material': ['MAT' + str(i % 1000) for i in range(500000)],
    'location': ['LOC' + str(i % 100) for i in range(500000)],
    'value': range(500000)
})

print(f"数据规模: {len(large_df):,} 行")
print(f"date列类型: {large_df['date']. dtype}")

# 模拟你的calculate_daily_net_demand操作
def simulate_module3_operation():
    """模拟Module3的8736次循环"""
    dates = pd.date_range('2020-01-01', periods=100). astype(str).tolist()
    
    results = []
    for date in dates:  # 只跑100次，真实是8736次
        mask = large_df['date'] == date
        filtered = large_df[mask]
        results.append(len(filtered))
    
    return sum(results)

# 性能分析
print("\n【场景1：当前方式 - object类型日期】")
print("运行中...")

profiler = cProfile.Profile()
profiler.enable()

result1 = simulate_module3_operation()

profiler.disable()
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps. print_stats(10)
print(s.getvalue())

# 优化后
print("\n【场景2：优化方式 - datetime类型日期】")
large_df['date_dt'] = pd.to_datetime(large_df['date'])
dates_dt = pd.to_datetime(pd.date_range('2020-01-01', periods=100))

def simulate_optimized():
    results = []
    for date in dates_dt:
        mask = large_df['date_dt'] == date
        filtered = large_df[mask]
        results. append(len(filtered))
    return sum(results)

print("运行中...")

profiler2 = cProfile. Profile()
profiler2.enable()

result2 = simulate_optimized()

profiler2. disable()
s2 = io.StringIO()
ps2 = pstats.Stats(profiler2, stream=s2). sort_stats('cumulative')
ps2.print_stats(10)
print(s2.getvalue())

print("\n" + "="*60)
print("对比结果")
print("="*60)
print(f"结果验证: {result1} vs {result2}")