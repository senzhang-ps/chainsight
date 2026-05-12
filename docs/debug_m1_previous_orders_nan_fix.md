# Module1 断点续跑历史订单 NaN 问题说明

## 现象

数据库模式断点续跑时，Module1 在合并历史订单与当天订单后报错：

```text
pandas.errors.IntCastingNaNError:
Cannot convert non-finite values (NA or inf) to integer
```

诊断日志显示异常来源不是当天新生成订单，而是历史订单：

```text
_quantity_debug_source = previous_orders_df
date = NaT
material = NaN
location = NaN
quantity = NaN
```

## 排查结论

SQL 查询 `module1_output_orderlog` 后没有发现 `quantity`、`date`、`material`、`location` 等字段存在 NaN 或 NULL 异常数据。

因此问题不是数据库表中存了坏数据，也不是订单生成计算过程直接产生 NaN，而是断点续跑时从数据库回退恢复历史订单的代码把列名丢失了。

## 根因

断点续跑逻辑位于：

```text
src/core/main_integration/simulation_db.py
```

旧逻辑：

```python
_fallback_rows = db.execute_query(_fallback_sql, (run_id_override, _prev_date))
m1_previous_orders = pd.DataFrame(_fallback_rows)
```

`execute_query()` 返回的是 `list[tuple]`，只有值，没有列名。例如：

```python
[
    (
        datetime.date(2026, 5, 10),   # date
        'MAT-001',                    # material
        'LOC-001',                    # location
        'normal',                     # demand_type
        120.0,                        # quantity
        datetime.date(2026, 5, 10),   # simulation_date
        0,                            # advance_days
        '2026-05-10',                 # sim_date
        'run-id',                     # run_id
        'config-name',                # config_name
        datetime.datetime(...)        # db_write_time
    )
]
```

直接执行 `pd.DataFrame(_fallback_rows)` 后，列名会变成数字：

```text
0, 1, 2, 3, 4, ...
```

而不是：

```text
date, material, location, demand_type, quantity, simulation_date, advance_days
```

后续与当天订单 `concat` 时，历史订单的真实值仍在数字列中，但业务列名不匹配，导致命名列 `date/material/location/quantity` 全部变成 NaN。

## 修复内容

### 1. 新增保留列名的查询方法

文件：

```text
pgsql_db/db_connection.py
```

新增：

```python
def execute_query_df(self, query: str, params: tuple = None) -> pd.DataFrame:
    """Execute a query and return a DataFrame preserving column names."""
    with self.get_cursor(commit=False) as cursor:
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
```

### 2. 新增 Module1 历史订单恢复函数

文件：

```text
src/core/main_integration/simulation_db.py
```

新增 `_load_m1_previous_orders_from_orderlog()`：

- 使用 `execute_query_df()` 保留数据库列名。
- 只恢复 Module1 所需的业务列：
  - `date`
  - `material`
  - `location`
  - `demand_type`
  - `quantity`
  - `simulation_date`
  - `advance_days`
- 丢弃 `run_id`、`sim_date`、`config_name`、`db_write_time` 等写库元数据列。
- 恢复 `date`、`simulation_date` 为 datetime。

### 3. 替换断点续跑回退恢复逻辑

旧逻辑用 tuple 列表构建 DataFrame，列名丢失。

新逻辑改为：

```python
m1_previous_orders = _load_m1_previous_orders_from_orderlog(
    db,
    run_id=run_id_override,
    previous_batch_end=_prev_date,
)
```

## 相关诊断增强

文件：

```text
src/modules/demand_planning/integration.py
```

在 `_normalize_orders()` 的 `quantity.astype(int)` 前增加诊断：

- 检查 `quantity` 是否为 NaN、inf、-inf 或不可转数字。
- 打印异常行。
- 标记异常来源：
  - `previous_orders_df`
  - `today_orders_df`
  - `history_orderlog_files`

这次日志中的异常来源是：

```text
previous_orders_df
```

## 另一处已修复的问题

`M6_DeliveryDelayDistribution.date` 业务上允许填 `ALL`，但之前数据库建表将 `date` 推断为 PostgreSQL `DATE`，导致 COPY 报：

```text
InvalidDatetimeFormat: invalid input syntax for type date: "ALL"
```

修复位于：

```text
pgsql_db/db_connection.py
```

现在：

- datetime/date dtype 的 `date` 列仍建为 `DATE`。
- object/string dtype 的 `date` 列建为 `TEXT`，可保存 `ALL`。
- 已存在的 DATE/TIMESTAMP 列在需要时会自动升级为 TEXT。

## 验证

已新增并通过测试：

```text
tests/test_simulation_db_resume.py
tests/test_db_connection_types.py
tests/test_demand_planning_integration.py
```

验证命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m py_compile src\core\main_integration\simulation_db.py pgsql_db\db_connection.py
```

## 最终判断

本次 Module1 NaN 问题属于断点续跑历史订单恢复路径的列名丢失问题。

不是数据库表中已有 NaN。

不是订单计算过程直接生成 NaN。

