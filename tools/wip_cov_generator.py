"""
WIP 需求变异系数（CoV）计算脚本
================================

目的
----
计算半成品（WIP）的需求变异系数（Coefficient of Variation, CoV），
基于成品（Pack）的平均需求（Forecast）、成品的预测误差 CoV，以及成品与半成品之间的 BOM 关系，
通过方差传播（Variance Propagation）合成得到 WIP 的 CoV。

核心逻辑与公式
---------------
- 合并键：(`pack_code`, `location`)；`bom` 通过 `pack_code` 关联到 WIP。
- 统一列名与类型，规范化键值（字符串化并去空格）。
- 识别 `cov` 单位（百分比或小数），统一转为小数进行计算。
- 解析 `bom.wip_percentage`（支持 "50%"、`50` 或 `0.5`），统一为小数。
- 若某成品在某地点缺失 `avg_fcst`，将其视为 0（`avg_qty=0`）。
- 计算：
    - `sigma_i = CoV_i * μ_i`
    - `SUM_num = Σ((a_i^2) * (sigma_i^2))`
    - `SUM_den = Σ(a_i * μ_i)`
    - `CoV_C = sqrt(SUM_num) / SUM_den`（当 `SUM_den==0` 定义为 0）

输入与输出
---------
- 输入：同一个 Excel 文件中的三个 Sheet：`avg_fcst`、`bom`、`cov`。
- 输出：Excel 文件 Sheet `wip_cov`，包含列：`material`（WIP code）、`location`、`error_std_percent`。

边界与健壮性处理
-----------------
- 缺失 `avg_fcst`：按 0 处理，避免因缺失而丢行。
- `bom.wip_percentage`：兼容百分数字符串与数值，统一为小数。
- `cov` 单位：自动识别百分比或小数；统一按小数计算，最终按原单位输出。
- `SUM_den==0`：保留输出行并将 CoV 设为 0。
- 数据清洗：仅在 `error_std_percent` 或 `wip_percentage` 缺失时丢弃行。
"""
import math
import pandas as pd


INPUT_FILE = "C:\\Users\\zhang.s.37\\OneDrive - Procter and Gamble\\9-unattended planning\\ChainSight\\Oral Care Case\\CoV Input.xlsx"
OUTPUT_FILE = "C:\\Users\\zhang.s.37\\OneDrive - Procter and Gamble\\9-unattended planning\\ChainSight\\Oral Care Case\\wip_cov_result.xlsx"

SHEET_FCST = "avg_fcst"
SHEET_BOM = "bom"
SHEET_COV = "cov"


def load_data():
    """加载 Excel 数据。

    读取 `INPUT_FILE` 中的三个 Sheet：`avg_fcst`、`bom`、`cov`，
    返回对应的原始 DataFrame（三者未做列名或类型转换）。

    返回
    ----
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        依次为 (fcst_raw, bom_raw, cov_raw)。

    可能异常
    -------
    - `PermissionError`：文件被占用或无权限（OneDrive 同步/占用）。
    - `FileNotFoundError`：路径不存在。
    - `ValueError`：指定的 Sheet 不存在。
    """
    fcst = pd.read_excel(INPUT_FILE, sheet_name=SHEET_FCST)
    bom = pd.read_excel(INPUT_FILE, sheet_name=SHEET_BOM)
    cov = pd.read_excel(INPUT_FILE, sheet_name=SHEET_COV)
    return fcst, bom, cov


def prepare_data(fcst, bom, cov):
    """标准化并预处理输入数据。

    - 重命名列：将 `material`→`pack_code`（用于键合并），统一 `location`、数值列名。
    - 键规范化：`pack_code`、`location` 转为字符串并去除首尾空格。
    - 数值转换：`avg_qty`、`error_std_percent` 转为数值，无法解析的设为缺失。
    - BOM 百分比解析：支持 "50%"、`50` 或 `0.5`，统一转换为小数（0.5）。

    参数
    ----
    fcst, bom, cov : pd.DataFrame
        `load_data()` 返回的原始表。

    返回
    ----
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        预处理后的 (fcst, bom, cov)。
    """
    # ---------- avg_fcst ----------
    fcst = fcst.rename(columns={
        "material": "pack_code",
        "location": "location",
        "avg_qty": "avg_qty"
    })
    # join key -> string & strip
    fcst["pack_code"] = fcst["pack_code"].astype(str).str.strip()
    fcst["location"] = fcst["location"].astype(str).str.strip()
    fcst["avg_qty"] = pd.to_numeric(fcst["avg_qty"], errors="coerce")

    # ---------- cov ----------
    cov = cov.rename(columns={
        "material": "pack_code",
        "location": "location",
        "error_std_percent": "error_std_percent"
    })
    cov["pack_code"] = cov["pack_code"].astype(str).str.strip()
    cov["location"] = cov["location"].astype(str).str.strip()
    cov["error_std_percent"] = pd.to_numeric(cov["error_std_percent"],
                                             errors="coerce")

    # ---------- bom ----------
    bom = bom.rename(columns={
        "pack code": "pack_code",
        "wip code": "wip_code",
        "wip percentage": "wip_percentage"
    })
    bom["pack_code"] = bom["pack_code"].astype(str).str.strip()
    bom["wip_code"] = bom["wip_code"].astype(str).str.strip()
    # 处理百分数字符串，如 "50%" -> 0.5；也兼容纯数字 50 -> 0.5 或 0.5 保持为 0.5
    wip_raw = bom["wip_percentage"].astype(str).str.strip()
    has_pct = wip_raw.str.contains("%", na=False)
    wip_raw = wip_raw.str.replace("%", "", regex=False)
    bom["wip_percentage"] = pd.to_numeric(wip_raw, errors="coerce")
    # 如果原始包含百分号或数值中位数 > 1，按百分数处理（除以100）
    if has_pct.any() or bom["wip_percentage"].dropna().median() > 1:
        bom["wip_percentage"] = bom["wip_percentage"] / 100.0

    return fcst, bom, cov


def detect_cov_is_percent(df_cov):
    """判断 `error_std_percent` 是否为百分比单位。

    根据该列的非缺失值的绝对值中位数是否大于 1 来判断：
    - 大于 1：视为百分比（例如 25 表示 25%）
    - 小于等于 1：视为小数（例如 0.25 表示 25%）
    - 若列为空：默认视为百分比（返回 True）。

    参数
    ----
    df_cov : pd.DataFrame
        含 `error_std_percent` 列的 DataFrame。

    返回
    ----
    bool
        True 表示百分比；False 表示小数。
    """
    s = df_cov["error_std_percent"].dropna().abs()
    if s.empty:
        return True
    return s.median() > 1.0

def compute_wip_cov():
    """主流程：计算并输出 WIP 的 CoV。

    步骤概览
    --------
    1. 加载并预处理数据（列名统一、键规范化、类型转换、BOM 百分比解析）。
    2. 以 `cov` 左连接 `fcst`（键：`pack_code`,`location`），缺失的 `avg_qty` 填 0。
    3. 与 `bom` 按 `pack_code` 合并，得到 WIP 关联明细。
    4. 数据清洗：仅在 `error_std_percent` 或 `wip_percentage` 缺失时丢弃。
    5. 单位识别：将成品 CoV 统一为小数计算（必要时除以 100）。
    6. 中间量计算：`sigma_i`、`num_term_i`、`den_term_i`。
    7. 按 `(wip_code, location)` 聚合得到 `SUM_num`、`SUM_den`。
    8. 计算 WIP CoV：`sqrt(SUM_num)/SUM_den`；当 `SUM_den==0` 时设为 0。
    9. 写出结果至 `OUTPUT_FILE` 的 `wip_cov` Sheet。

    输出
    ----
    Excel 文件：列为 `material`（即 WIP code）、`location`、`error_std_percent`。

    边界与注意事项
    --------------
    - `avg_fcst` 缺失视为 0，确保行不被丢弃。
    - `cov` 单位自动识别（百分比/小数），统一到小数再根据单位输出。
    - `SUM_den==0` 的组会保留且 CoV 设为 0。
    - 输入路径在 OneDrive 下，若遇到权限或占用，请关闭占用程序或复制到本地路径后再运行。
    """
    fcst_raw, bom_raw, cov_raw = load_data()
    fcst, bom, cov = prepare_data(fcst_raw, bom_raw, cov_raw)

    print(f"fcst rows: {len(fcst)}")
    print(f"cov rows:  {len(cov)}")
    print(f"bom rows:  {len(bom)}")

    # 1. 合并 cov + fcst（pack_code + location）
    # 要求：如果某 pack_code 在该 location 缺失 avg_fcst，则将 avg_qty 视为 0
    pack_fcst_cov = pd.merge(
        cov,
        fcst,
        on=["pack_code", "location"],
        how="left"
    )
    pack_fcst_cov["avg_qty"] = pack_fcst_cov["avg_qty"].fillna(0)
    print(f"after merge cov+fcst (left): {len(pack_fcst_cov)} rows")

    # 2. 再与 bom 合并（pack_code）
    detail = pd.merge(
        pack_fcst_cov,
        bom,
        on="pack_code",
        how="inner"
    )
    print(f"after merge with bom: {len(detail)} rows")

    # 3. 过滤明显坏数据
    # 不因 avg_qty 缺失而丢弃，avg_qty 已在上一步填充为 0
    detail = detail.dropna(subset=["error_std_percent", "wip_percentage"])
    print(f"after dropna: {len(detail)} rows")

    if detail.empty:
        print("No rows left after dropna, please check missing values.")
        result = detail[['pack_code', 'location']].head(0).copy()
        result["error_std_percent"] = []
        result = result.rename(columns={"pack_code": "material"})
        result.to_excel(OUTPUT_FILE, sheet_name="wip_cov", index=False)
        return

    # 4. CoV 单位判断：百分数还是小数
    is_percent = detect_cov_is_percent(detail)
    print(f"is_percent = {is_percent}")

    if is_percent:
        detail["COV_dec"] = detail["error_std_percent"] / 100.0
    else:
        detail["COV_dec"] = detail["error_std_percent"]

    # sigma_i = COV * μ
    detail["sigma_i"] = detail["COV_dec"] * detail["avg_qty"]

    # num_term_i = a_i^2 * sigma_i^2
    detail["num_term_i"] = (detail["wip_percentage"] ** 2) * (
        detail["sigma_i"] ** 2
    )

    # den_term_i = a_i * μ_i
    detail["den_term_i"] = detail["wip_percentage"] * detail["avg_qty"]

    # 5. 按 (wip_code, location) 聚合
    grouped = detail.groupby(["wip_code", "location"], as_index=False).agg(
        SUM_num=("num_term_i", "sum"),
        SUM_den=("den_term_i", "sum")
    )
    print(f"grouped rows (wip_code, location): {len(grouped)}")

    # 不再过滤 SUM_den<=0；对于 SUM_den==0 的情况，CoV 定义为 0 以保证有输出
    zero_den_count = (grouped["SUM_den"] == 0).sum()
    print(f"rows with SUM_den==0: {zero_den_count}")

    # 6. 计算 WIP CoV
    grouped["COV_C"] = 0.0
    mask_den_pos = grouped["SUM_den"] > 0
    grouped.loc[mask_den_pos, "COV_C"] = (
        grouped.loc[mask_den_pos, "SUM_num"] ** 0.5
    ) / grouped.loc[mask_den_pos, "SUM_den"]

    if is_percent:
        grouped["error_std_percent"] = grouped["COV_C"] * 100.0
    else:
        grouped["error_std_percent"] = grouped["COV_C"]

    result = grouped[["wip_code", "location", "error_std_percent"]].copy()
    result = result.rename(columns={"wip_code": "material"})
    result["error_std_percent"] = result["error_std_percent"].round(4)

    result.to_excel(OUTPUT_FILE, sheet_name="wip_cov", index=False)
    print(f"Done. Result saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_wip_cov()