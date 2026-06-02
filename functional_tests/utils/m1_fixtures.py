from __future__ import annotations

from pathlib import Path
import pandas as pd


ORDER_LOG_COLUMNS = [
    "date",
    "material",
    "location",
    "demand_type",
    "simulation_date",
    "advance_days",
    "quantity",
]

SHIPMENT_LOG_COLUMNS = [
    "date",
    "material",
    "location",
    "quantity",
    "demand_type",
    "order_id",
]

CUT_LOG_COLUMNS = ["date", "material", "location", "quantity"]
SUPPLY_DEMAND_LOG_COLUMNS = ["date", "material", "location", "quantity", "demand_element"]
FUNCTIONAL_TEST_TMP = Path(__file__).resolve().parents[1] / "_tmp"


class M1TestOrchestrator:
    def __init__(self, start_date: str = "2026-05-04") -> None:
        self.start_date = pd.Timestamp(start_date)

    def get_beginning_inventory_view(self, date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(date),
                    "material": "1001",
                    "location": "0001",
                    "quantity": 1000,
                }
            ]
        )

    def get_production_gr_view(self, date: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "material", "location", "quantity"])

    def get_delivery_gr_view(self, date: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date", "material", "receiving", "quantity", "ori_deployment_uid", "vehicle_uid"]
        )


def build_weekly_m1_config() -> dict[str, pd.DataFrame]:
    return {
        "M1_DemandForecast": pd.DataFrame(
            [
                {"material": "1001", "location": "0001", "week": 1, "quantity": 70},
            ]
        ),
        "M1_ForecastError": pd.DataFrame(
            [
                {
                    "material": "1001",
                    "location": "0001",
                    "order_type": "AO",
                    "error_std_percent": 0.0,
                },
                {
                    "material": "1001",
                    "location": "0001",
                    "order_type": "normal",
                    "error_std_percent": 0.0,
                },
            ]
        ),
        "M1_OrderCalendar": pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-05-04")},
                {"date": pd.Timestamp("2026-05-06")},
            ]
        ),
        "M1_AOConfig": pd.DataFrame(
            [
                {
                    "material": "1001",
                    "location": "0001",
                    "advance_days": 2,
                    "ao_percent": 0.2,
                }
            ]
        ),
        "M1_DPSConfig": pd.DataFrame(),
        "M1_SupplyChoiceConfig": pd.DataFrame(),
    }


def build_daily_forecast(start_date: str = "2026-05-04") -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    rows = []
    for offset in range(7):
        rows.append(
            {
                "date": start + pd.Timedelta(days=offset),
                "material": "1001",
                "location": "0001",
                "week": 1,
                "demand_type": "normal",
                "quantity": 10,
                "original_quantity": 10,
            }
        )
    return pd.DataFrame(rows)


def normalize_for_compare(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    out = out[columns]
    for col in ["date", "simulation_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    if "quantity" in out.columns:
        out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(columns).reset_index(drop=True) if not out.empty else out


def load_real_m1_config(config_file: str | Path) -> dict[str, pd.DataFrame]:
    """从 xlsx 加载真实 M1 配置。"""
    if isinstance(config_file, str):
        config_file = Path(config_file)

    return {
        "M1_DemandForecast": pd.read_excel(config_file, sheet_name="M1_DemandForecast"),
        "M1_ForecastError": pd.read_excel(config_file, sheet_name="M1_ForecastError"),
        "M1_OrderCalendar": pd.read_excel(config_file, sheet_name="M1_OrderCalendar"),
        "M1_AOConfig": pd.read_excel(config_file, sheet_name="M1_AOConfig"),
        "M1_DPSConfig": pd.read_excel(config_file, sheet_name="M1_DPSConfig") if "M1_DPSConfig" in pd.ExcelFile(config_file).sheet_names else pd.DataFrame(),
        "M1_SupplyChoiceConfig": pd.read_excel(config_file, sheet_name="M1_SupplyChoiceConfig") if "M1_SupplyChoiceConfig" in pd.ExcelFile(config_file).sheet_names else pd.DataFrame(),
    }


def filter_sku_from_config(config: dict[str, pd.DataFrame], material, location: str) -> dict[str, pd.DataFrame]:
    """从配置中筛选单个 SKU 的数据。"""
    sku_config = config.copy()

    # 筛选 DemandForecast
    df = sku_config["M1_DemandForecast"]
    sku_config["M1_DemandForecast"] = df[
        (df["material"] == material) & (df["location"] == location)
    ].copy()

    # 筛选 AOConfig（可选）
    if not sku_config["M1_AOConfig"].empty:
        ao = sku_config["M1_AOConfig"]
        sku_config["M1_AOConfig"] = ao[
            (ao["material"] == material) & (ao["location"] == location)
        ].copy()

    # 筛选 ForecastError（可选）
    if not sku_config["M1_ForecastError"].empty:
        fe = sku_config["M1_ForecastError"]
        # ForecastError 通常只有 material/location，不区分 SKU，所以全保留或按 material/location 筛选
        if {"material", "location"}.issubset(fe.columns):
            sku_config["M1_ForecastError"] = fe[
                (fe["material"] == material) & (fe["location"] == location)
            ].copy()

    return sku_config
