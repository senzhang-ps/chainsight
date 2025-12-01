# Module1 性能优化总结 / Module1 Performance Optimization Summary

## 问题 / Problem

module1.py在长时间仿真时运行时间很长，性能随仿真天数线性下降。

Module1.py runs very slowly during long simulations, with performance degrading linearly as simulation days increase.

## 根本原因 / Root Cause

每次运行时读取所有历史订单文件，导致：
- 第1天：读取0个文件
- 第30天：读取29个文件
- 第365天：读取364个文件

Every run reads all historical order files, causing:
- Day 1: Read 0 files
- Day 30: Read 29 files
- Day 365: Read 364 files

## 解决方案 / Solution

### 核心优化 / Core Optimization

**只读取最近(max_advance_days + 1)天的历史文件**

**Only read the most recent (max_advance_days + 1) days of historical files**

```python
# Before / 改动前
for fname in os.listdir(m1_output_dir):
    if fdate < current_date:
        # Read ALL past files / 读取所有历史文件

# After / 改动后
earliest_relevant_date = current_date - pd.Timedelta(days=max_advance_days + 1)
for fname in os.listdir(m1_output_dir):
    if fdate < earliest_relevant_date:
        continue  # Skip old files / 跳过旧文件
```

### 关键原则 / Key Principles

1. ✅ **max_advance_days从配置动态获取** / **max_advance_days dynamically retrieved from config**
   ```python
   max_advance_days = int(ao_config['advance_days'].max(skipna=True))
   ```

2. ✅ **保持30天预测窗口** / **Keep 30-day forecast window**
   - 这是业务逻辑要求 / Required by business logic
   - 只优化查询方式，不改变窗口大小 / Only optimize query method, not window size

3. ✅ **历史文件读取 = max_advance_days + 1** / **Historical file reading = max_advance_days + 1**
   - 因为AO订单最多提前max_advance_days天生成 / Because AO orders are generated at most max_advance_days in advance
   - +1确保覆盖边界情况 / +1 ensures boundary cases are covered

## 性能提升 / Performance Improvement

以max_advance_days=10为例 / Example with max_advance_days=10:

| 仿真周期<br/>Simulation Period | 文件读取<br/>Files Read | 性能提升<br/>Speedup |
|---|---|---|
| 30天 / 30 days | 11 vs 30个 | **2.7x faster** |
| 90天 / 90 days | 11 vs 90个 | **8.2x faster** |
| 365天 / 365 days | 11 vs 365个 | **33.2x faster** |

## 测试验证 / Test Verification

✅ 所有测试通过 / All tests passed:
- 优化功能测试 / Optimization functional tests: 4/4
- NaN处理测试 / NaN handling tests: 2/2
- 安全检查 / Security check: 0 issues
- 代码审查 / Code review: No blocking issues

## 配置要求 / Configuration Requirements

- **必须**: M1_AOConfig 表包含 'advance_days' 列
- **默认**: DEFAULT_MAX_ADVANCE_DAYS = 10（仅作为后备值）
- **自适应**: 支持配置表中任意advance_days值

- **Required**: M1_AOConfig table contains 'advance_days' column
- **Default**: DEFAULT_MAX_ADVANCE_DAYS = 10 (fallback value only)
- **Adaptive**: Supports any advance_days value in config

## 维护建议 / Maintenance Recommendations

1. ⚠️ **不要改变30天预测窗口** / **Don't change 30-day forecast window**
   - 这是业务逻辑要求 / This is a business logic requirement

2. ✅ **max_advance_days自动适应配置** / **max_advance_days automatically adapts to config**
   - 无需手动调整 / No manual adjustment needed

3. ✅ **监控配置变化** / **Monitor config changes**
   - 如果advance_days大幅增加（如>30天），考虑进一步优化
   - If advance_days increases significantly (e.g., >30 days), consider further optimization

## 文件清单 / File List

- `module1.py`: 优化后的核心代码 / Optimized core code
- `MODULE1_OPTIMIZATION_NOTES.md`: 详细技术文档（中文）/ Detailed technical doc (Chinese)
- `OPTIMIZATION_SUMMARY.md`: 本文件 - 双语总结 / This file - bilingual summary

## 作者信息 / Author Info

- 优化日期 / Optimization Date: 2025-12-01
- 基于配置 / Based on Config: M1_AOConfig max(advance_days) = 10
- 测试状态 / Test Status: ✅ All Passed
