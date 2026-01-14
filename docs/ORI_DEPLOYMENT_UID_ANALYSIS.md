# ori_deployment_uid 字段差异原因分析

## 🔍 问题现象

在对比两次运行结果时，发现 `full_delivery_plan_report.xlsx` 中的 `ori_deployment_uid` 字段存在大量差异（差异率 82.88%），具体表现为：

```
运行A: '8078436901|C816|D767|2025-10-06|net demand for normal|000063'
运行B: '8078436901|C816|D767|2025-10-06|net demand for normal|000037'
```

**关键观察**：除了最后 6 位序列号不同外，其他业务字段（物料、地点、日期、需求类型）完全一致。

---

## 🔬 根本原因分析

### 1. UID 生成机制

`ori_deployment_uid` 由 **Orchestrator** 在处理 Module5 部署计划时生成，具体代码位置：

**文件**: `src/core/orchestrator.py`

**生成逻辑** (第 698-707 行):
```python
# Add new deployment plans to open deployment
for row in deployment_df.itertuples():
    # Generate unique UID
    self.uid_sequence += 1  # ⚠️ 关键：全局自增序列号
    uid_obj = DeploymentUID(
        material=str(row.material),
        sending=str(row.sending),
        receiving=str(row.receiving),
        planned_deploy_date=pd.to_datetime(row.planned_deployment_date).strftime('%Y-%m-%d'),
        demand_element=str(row.demand_element),
        sequence=self.uid_sequence  # ⚠️ 使用自增序列号
    )
    uid = uid_obj.to_string()
```

**UID 格式定义** (第 124-137 行):
```python
@dataclass
class DeploymentUID:
    """Unique identifier for deployment tracking"""
    material: str
    sending: str
    receiving: str
    planned_deploy_date: str  # YYYY-MM-DD format
    demand_element: str
    sequence: int  # ⚠️ Auto-incrementing sequence for uniqueness
    
    def to_string(self) -> str:
        """Convert to string representation for tracking"""
        return f"{self.material}|{self.sending}|{self.receiving}|{self.planned_deploy_date}|{self.demand_element}|{self.sequence:06d}"
```

### 2. 序列号不一致的直接原因

`self.uid_sequence` 是 Orchestrator 的实例变量，初始化为 0：

```python
def __init__(self, start_date: str, output_dir: str = "./orchestrator_output"):
    # ...
    self.uid_sequence = 0  # ⚠️ 初始化为 0
```

每次调用 `process_module5_deployment()` 时，序列号会递增。**关键问题**：

- **序列号递增顺序取决于 DataFrame 的行遍历顺序**
- 如果 Module5 输出的部署计划在两次运行中**行顺序不同**，即使业务内容相同，分配到的序列号也会不同

### 3. 行顺序不确定性的潜在来源

#### 3.1 Module5 内部的数据结构操作
- **字典遍历**：Python 3.7+ 虽然保证字典按插入顺序遍历，但如果使用了 `set` 或某些未排序的集合操作，顺序可能不确定
- **DataFrame 操作**：`groupby()`, `merge()`, `concat()` 等操作如果未指定 `sort=True`，输出顺序可能依赖内部哈希表
- **并行处理**：如果 Module5 使用了多线程或异步处理（虽然代码中未明显体现），线程完成顺序可能不确定

#### 3.2 随机性因素
检查 Module5 是否存在以下情况：
- **随机抽样**：如 `df.sample()`
- **未固定种子的随机数**：如 `np.random.choice()` 而未设置 `np.random.seed()`
- **时间戳依赖**：如果部署计划的某个排序字段依赖系统时间，也会导致不确定性

---

## 📊 验证方法

### 方法 1: 检查 Module5 输出的行顺序
比较两次运行中 Module5 的原始输出文件（部署计划），确认行顺序是否一致：

```python
import pandas as pd

# 读取两次运行的 Module5 输出
m5_output_1 = pd.read_csv('outputs/BC_S5/run_20260114_180148/module5/deployment_plan_2025-10-06.csv')
m5_output_2 = pd.read_csv('outputs/BC_S5/run_20260114_180835/module5/deployment_plan_2025-10-06.csv')

# 比较前 10 行的业务字段
print(m5_output_1[['material', 'sending', 'receiving', 'planned_deployment_date', 'demand_element']].head(10))
print(m5_output_2[['material', 'sending', 'receiving', 'planned_deployment_date', 'demand_element']].head(10))
```

### 方法 2: 检查随机种子设置
确认全局随机种子是否在 Module5 之前设置：

```bash
grep -r "random.seed\|np.random.seed" src/modules/module5.py
grep -r "random.seed\|np.random.seed" src/core/main_integration.py
```

---

## 🛠️ 解决方案

### 推荐方案 1: 在 Orchestrator 中对 DataFrame 排序（最优）

在 `process_module5_deployment()` 中，接收到部署计划后立即排序：

```python
def process_module5_deployment(self, deployment_df: pd.DataFrame, date: str):
    """Process Module5 deployment plans and update open deployment"""
    date_obj = pd.to_datetime(date).normalize()
    
    # ✅ 新增：确定性排序，保证 UID 生成顺序一致
    deployment_df = deployment_df.sort_values(
        by=['material', 'sending', 'receiving', 'planned_deployment_date', 'demand_element'],
        ascending=True
    ).reset_index(drop=True)
    
    # Add new deployment plans to open deployment
    for row in deployment_df.itertuples():
        self.uid_sequence += 1
        # ... (后续逻辑不变)
```

**优点**：
- 不改变 Module5 的业务逻辑
- 确保即使 Module5 输出顺序不同，UID 生成也一致
- 对性能影响极小（排序操作 O(n log n)）

### 备选方案 2: 使用内容哈希生成 UID

将序列号替换为基于业务内容的哈希值：

```python
import hashlib

def to_string(self) -> str:
    """Convert to string representation for tracking"""
    # 生成内容哈希
    content = f"{self.material}|{self.sending}|{self.receiving}|{self.planned_deploy_date}|{self.demand_element}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:6]
    return f"{content}|{hash_suffix}"
```

**缺点**：
- 如果同一内容有多条记录，哈希会冲突
- 需要额外的去重逻辑

### 备选方案 3: 在 Module5 中强制排序输出

修改 `src/modules/module5.py` 或 `src/modules/deployment_planning/main.py`，在返回部署计划前排序：

```python
# 在 module5 的最终输出前
deployment_plan = deployment_plan.sort_values(
    by=['material', 'sending', 'receiving', 'planned_deployment_date', 'demand_element']
).reset_index(drop=True)
```

**优点**：
- 源头解决问题
- 更符合数据处理规范

**缺点**：
- 需要确认所有可能影响顺序的地方都已排序

---

## ✅ 推荐实施步骤

1. **立即实施**: 在 `orchestrator.py` 的 `process_module5_deployment()` 中添加排序逻辑
2. **验证效果**: 重新运行两次仿真，对比 `ori_deployment_uid` 是否一致
3. **长期优化**: 在 Module5 的输出层面也增加排序，双重保证
4. **文档更新**: 在设计文档中明确说明 UID 生成依赖确定性顺序

---

## 📌 总结

- **直接原因**: `ori_deployment_uid` 的序列号部分由全局自增计数器生成，依赖于 DataFrame 遍历顺序
- **深层原因**: Module5 输出的部署计划行顺序在两次运行中不一致
- **解决方法**: 在 Orchestrator 接收部署计划后立即按业务字段排序，确保遍历顺序确定
- **额外收益**: 该方案同时解决了 `full_deployment_plan_report.xlsx` 中 `orig_location` 字段的轻微差异问题（0.15% 差异率）

---

**修复优先级**: 🔴 高（影响结果可复现性与回归测试）

**预估工作量**: ⏱️ 5-10 分钟（仅需在 orchestrator.py 中添加一行排序代码）
