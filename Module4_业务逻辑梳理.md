# Module4 生产计划排程业务逻辑梳理

## 1. 模块概述

Module4是供应链系统中的**高级生产计划排程(APS - Advanced Planning and Scheduling)**模块，主要负责：
- 基于净需求生成无约束生产计划
- 考虑产能约束进行生产排程
- 优化换型序列以减少换型损失
- 模拟生产可靠性并生成最终生产计划

## 2. 核心业务流程

### 2.1 数据输入流程
```
Module3输出 → 净需求数据 → 配置参数 → 产能数据
    ↓
NetDemand → MaterialLocationLineCfg → LineCapacity → ChangeoverMatrix
```

### 2.2 主要执行步骤
1. **数据加载与验证**
2. **无约束计划生成**
3. **产能分配与约束处理**
4. **换型优化**
5. **生产可靠性模拟**
6. **输出生成**

## 3. 详细业务逻辑

### 3.1 数据加载与预处理

#### 3.1.1 净需求数据加载
- **数据源**: Module3输出的NetDemand数据
- **过滤条件**: 
  - `layer = 0` (最下游需求)
  - 数量取绝对值(处理负数需求)
- **时间范围**: 基于`requirement_date`字段

#### 3.1.2 配置参数加载
- **MaterialLocationLineCfg**: 物料-位置-产线配置
- **LineCapacity**: 产线产能配置
- **ChangeoverMatrix**: 换型矩阵
- **ChangeoverDefinition**: 换型定义(时间、成本、MU损失)
- **ProductionReliability**: 生产可靠性参数

### 3.2 无约束生产计划生成

#### 3.2.1 计划窗口计算
```python
def compute_planning_window(simulation_date, ptf, lsk):
    window_start = simulation_date + timedelta(days=ptf)          # 冻结期后
    window_end = simulation_date + timedelta(days=ptf + lsk - 1)  # 计划期结束
    return window_start, window_end
```

**关键参数**:
- **PTF (Planning Time Fence)**: 计划冻结期，冻结期内不允许调整
- **LSK (Lot Size Key)**: 计划周期长度

#### 3.2.2 审查日判断
```python
def is_review_day(simulation_date, simulation_start, lsk, day):
    days_since_start = (simulation_date - simulation_start).days
    first_review_day = int(day)  # 首次审查偏移天数
    return (days_since_start - first_review_day) % int(lsk) == 0 and days_since_start >= first_review_day
```

**逻辑说明**:
- 每个物料有独立的审查周期(LSK)
- 审查日 = 首次审查日 + n×LSK (n为整数)
- 只有在审查日才生成生产计划

#### 3.2.3 批量计算与舍入
```python
def round_up(q, mb, rv):
    base = max(q, mb)           # 不小于最小批量
    return base if base % rv == 0 else int(np.ceil(base / rv) * rv)  # 向上舍入到舍入体积
```

**参数说明**:
- **MB (Min Batch)**: 最小生产批量
- **RV (Rounding Volume)**: 舍入体积

### 3.3 产能分配与约束处理

#### 3.3.1 换型序列优化
```python
def optimal_changeover_sequence(batches, co_mat, co_def, line):
    # 1. 从最大批量开始
    # 2. 选择换型时间最短的下一个物料
    # 3. 重复直到所有批次排完
```

**优化策略**:
- **贪心算法**: 每次选择换型成本最低的相邻物料
- **目标**: 最小化总换型时间

#### 3.3.2 产能分配算法
```python
def centralized_capacity_allocation_with_changeover():
    for (line, sim_date), uncon_grp in uncon.groupby(['line', 'simulation_date']):
        # 1. 应用换型优化
        # 2. 分配换型时间
        # 3. 分配生产产能
        # 4. 记录未满足需求
```

**分配逻辑**:
1. **换型时间优先**: 先分配必要的换型时间
2. **产能分配**: 按生产速率分配剩余产能
3. **时间窗口**: 在计划窗口内分配生产

### 3.4 生产可靠性模拟

#### 3.4.1 可靠性计算
```python
def simulate_production(plan, pr_cfg, seed=None):
    rng = np.random.RandomState(seed)
    pr_map = pr_cfg.set_index(['location', 'line'])['pr'].to_dict()
    plan['produced_qty'] = plan.apply(
        lambda r: rng.binomial(int(r['con_planned_qty']), pr_map.get((r['location'], 'line'), 1)),
        axis=1
    )
```

**模拟逻辑**:
- 使用二项分布模拟生产成功概率
- 考虑产线位置的生产可靠性参数
- 生成实际生产数量

### 3.5 换型指标计算

#### 3.5.1 换型统计
```python
def calculate_changeover_metrics(production_plan, changeover_def):
    # 按日期、位置、产线、换型类型分组统计
    # 计算总时间、成本、MU损失
```

**输出指标**:
- **换型次数**: 每种换型类型的发生次数
- **换型时间**: 总换型时间
- **换型成本**: 总换型成本
- **MU损失**: 总产能损失

## 4. 关键业务规则

### 4.1 时间约束规则
- **冻结期**: PTF天内不允许调整生产计划
- **计划期**: LSK天内的需求进行计划排程
- **审查周期**: 每个物料按LSK周期进行计划审查

### 4.2 产能约束规则
- **产线产能**: 考虑每日可用产能限制
- **换型时间**: 物料切换需要消耗换型时间
- **生产速率**: 不同物料在不同产线的生产速率不同

### 4.3 批量约束规则
- **最小批量**: 生产数量不能小于最小批量
- **舍入体积**: 生产数量必须为舍入体积的整数倍
- **需求聚合**: 同一计划窗口内的需求进行聚合

## 5. 输出数据结构

### 5.1 ProductionPlan (生产计划)
- `material`: 物料编码
- `location`: 生产位置
- `line`: 产线
- `production_plan_date`: 计划生产日期
- `available_date`: 可用日期(MCT后)
- `con_planned_qty`: 约束后计划数量
- `produced_qty`: 模拟实际生产数量
- `changeover_id`: 换型标识

### 5.2 CapacityExceed (产能超限)
- `unmet_uncon_planned_qty`: 未满足的无约束计划数量
- 记录因产能不足无法满足的需求

### 5.3 ChangeoverLog (换型日志)
- `changeover_type`: 换型类型
- `count`: 换型次数
- `time`: 总换型时间
- `cost`: 总换型成本
- `mu_loss`: 总产能损失

### 5.4 Validation (验证报告)
- 配置验证问题
- 数据一致性检查结果

## 6. 业务价值

### 6.1 生产计划优化
- **换型优化**: 减少换型损失，提高产能利用率
- **批量优化**: 满足最小批量和舍入要求
- **时间窗口**: 在合理的时间范围内进行计划

### 6.2 约束管理
- **产能约束**: 确保计划在产能范围内可行
- **时间约束**: 考虑冻结期和计划期限制
- **批量约束**: 满足生产批量要求

### 6.3 决策支持
- **产能分析**: 识别产能瓶颈和超限情况
- **换型分析**: 量化换型成本和时间影响
- **可靠性评估**: 考虑生产不确定性

## 7. 与其他模块的集成

### 7.1 上游依赖
- **Module3**: 提供净需求数据
- **配置系统**: 提供业务参数配置

### 7.2 下游输出
- **Module5**: 提供生产计划数据
- **Module6**: 提供生产执行数据
- **Orchestrator**: 协调整体执行流程

## 8. 配置参数说明

### 8.1 核心业务参数
- **PTF**: 计划冻结期(天)
- **LSK**: 计划周期长度(天)
- **MB**: 最小生产批量
- **RV**: 舍入体积
- **MCT**: 最小周期时间

### 8.2 技术参数
- **生产速率**: 单位时间内的生产数量
- **换型时间**: 物料切换所需时间
- **生产可靠性**: 生产成功的概率

## 9. 异常处理机制

### 9.1 数据异常
- 配置文件缺失或格式错误
- 净需求数据为空或格式不正确
- 产能数据缺失或异常

### 9.2 业务异常
- 物料-位置-产线配置缺失
- 产能不足以满足需求
- 换型定义不完整

### 9.3 处理策略
- 记录详细错误信息到验证报告
- 生成空的生产计划(如果可能)
- 继续执行流程，确保系统稳定性

## 10. 性能优化考虑

### 10.1 算法优化
- 换型序列优化使用贪心算法
- 批量处理减少循环次数
- 索引优化提高数据查询效率

### 10.2 内存管理
- 分批处理大量数据
- 及时释放临时变量
- 使用高效的数据结构

### 10.3 并行处理
- 支持多产线并行计算
- 独立处理不同物料组
- 可扩展的架构设计


