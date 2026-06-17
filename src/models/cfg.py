"""cfg 模型组 —— 输入配置表的 SA declarative 声明（Django 风格）。

每张表直接声明为一个 SA declarative 类（像 orch.py 那样），
富元数据挂在 Column.info（comment_source / range / enumerate / date_flag / default / local_name）
和 __table_args__["info"]（sheet_name / required_import / quality_check_enabled / import_enabled）。

直接 import 模型类使用：

    from src.models.cfg import GlobalNetwork, CONFIG_TABLE_REGISTRY
    table_name = GlobalNetwork.__tablename__          # "cfg_global_network"
    for col in GlobalNetwork.__table__.columns:       # 取列 + 富元数据
        col.info.get("range")

本文件由 analyze/gen_cfg_models_v2.py 从原 CONFIG_TABLE_SCHEMAS + defaults.yaml 生成。
原 config_table_schema.py 保持不动（DQ checker / excel_importer 继续用它）。
"""
from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, Numeric, Text
from .base import Base, CfgBase


# ── Global_seed ─────────────────────────────────────────
class GlobalSeed(Base, CfgBase):
    """配置表 Global_seed。"""
    __tablename__ = "cfg_global_seed"
    __table_args__ = {'info': { "sheet_name": 'Global_seed', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['seed'] }}
    seed = Column(BigInteger, primary_key=True, comment='全局随机种子', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── Global_Network ─────────────────────────────────────────
class GlobalNetwork(Base, CfgBase):
    """配置表 Global_Network。"""
    __tablename__ = "cfg_global_network"
    __table_args__ = {'info': { "sheet_name": 'Global_Network', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['sourcing', 'location', 'material', 'eff_from', 'eff_to'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'M' })
    location = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'S' })
    sourcing = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'L' })
    location_type = Column(Text, comment='收货点类型', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'DC' })
    eff_from = Column(DateTime, primary_key=True, comment='生效时间', info={ "comment_source": 'description', "db_field_exists": '是', "default": '1970-01-01', "date_flag": 'start' })
    eff_to = Column(DateTime, primary_key=True, comment='失效时间', info={ "comment_source": 'description', "db_field_exists": '是', "default": '9999-12-31', "date_flag": 'end' })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── Global_SpaceCapacity ─────────────────────────────────────────
class GlobalSpacecapacity(Base, CfgBase):
    """配置表 Global_SpaceCapacity。"""
    __tablename__ = "cfg_global_spacecapacity"
    __table_args__ = {'info': { "sheet_name": 'Global_SpaceCapacity', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['location', 'eff_from', 'eff_to'] }}
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'L' })
    eff_from = Column(DateTime, primary_key=True, comment='生效时间', info={ "comment_source": 'description', "db_field_exists": '是', "default": '1970-01-01T00:00:00', "date_flag": 'start' })
    eff_to = Column(DateTime, primary_key=True, comment='失效时间', info={ "comment_source": 'description', "db_field_exists": '是', "default": '9999-12-31T00:00:00', "date_flag": 'end' })
    capacity = Column(Float, comment='仓容', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── Global_LeadTime ─────────────────────────────────────────
class GlobalLeadtime(Base, CfgBase):
    """配置表 Global_LeadTime。"""
    __tablename__ = "cfg_global_leadtime"
    __table_args__ = {'info': { "sheet_name": 'Global_LeadTime', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['sending', 'receiving'] }}
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'description', "db_field_exists": '是' })
    receiving = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'description', "db_field_exists": '是' })
    pdt = Column(BigInteger, comment='计划交付时长', info={ "local_name": 'PDT', "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    gr = Column(BigInteger, comment='收货时长', info={ "local_name": 'GR', "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    mct = Column(BigInteger, comment='微检时间', info={ "local_name": 'MCT', "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    otd = Column(BigInteger, comment='运输时间', info={ "local_name": 'OTD', "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── Global_DemandPriority ─────────────────────────────────────────
class GlobalDemandpriority(Base, CfgBase):
    """配置表 Global_DemandPriority。"""
    __tablename__ = "cfg_global_demandpriority"
    __table_args__ = {'info': { "sheet_name": 'Global_DemandPriority', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['demand_element'] }}
    demand_element = Column(Text, primary_key=True, comment='需求类型优先级', info={ "comment_source": 'description', "db_field_exists": '是' })
    priority = Column(BigInteger, comment='需求类型优先级', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_InitialInventory ─────────────────────────────────────────
class M1Initialinventory(Base, CfgBase):
    """配置表 M1_InitialInventory。"""
    __tablename__ = "cfg_m1_initialinventory"
    __table_args__ = {'info': { "sheet_name": 'M1_InitialInventory', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['location', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    quantity = Column(Float, comment='初始库存数量', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_DemandForecast ─────────────────────────────────────────
class M1Demandforecast(Base, CfgBase):
    """配置表 M1_DemandForecast。"""
    __tablename__ = "cfg_m1_demandforecast"
    __table_args__ = {'info': { "sheet_name": 'M1_DemandForecast', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['location', 'week', 'material'] }}
    week = Column(BigInteger, primary_key=True, comment='需求周', info={ "comment_source": 'description', "db_field_exists": '是', "default": 0 })
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'M' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'L' })
    quantity = Column(Float, comment='需求数量', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_ForecastError ─────────────────────────────────────────
class M1Forecasterror(Base, CfgBase):
    """配置表 M1_ForecastError。"""
    __tablename__ = "cfg_m1_forecasterror"
    __table_args__ = {'info': { "sheet_name": 'M1_ForecastError', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['location', 'order_type', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料', info={ "comment_source": 'description', "db_field_exists": '是' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    order_type = Column(Text, primary_key=True, comment='订单类型', info={ "comment_source": 'description', "db_field_exists": '是' })
    error_std_percent = Column(Float, comment='误差比例', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_OrderCalendar ─────────────────────────────────────────
class M1Ordercalendar(Base, CfgBase):
    """配置表 M1_OrderCalendar。"""
    __tablename__ = "cfg_m1_ordercalendar"
    __table_args__ = {'info': { "sheet_name": 'M1_OrderCalendar', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['date'] }}
    date = Column(DateTime, primary_key=True, comment='订单日', info={ "comment_source": 'description', "db_field_exists": '是' })
    order_day_flag = Column(BigInteger, comment='是否', info={ "comment_source": 'description', "db_field_exists": '是' })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_AOConfig ─────────────────────────────────────────
class M1Aoconfig(Base, CfgBase):
    """配置表 M1_AOConfig。"""
    __tablename__ = "cfg_m1_aoconfig"
    __table_args__ = {'info': { "sheet_name": 'M1_AOConfig', "import_enabled": True, "required_import": False, "optional_import": True, "quality_check_enabled": True, "primary_key": ['location', 'advance_days', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'M' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'L' })
    advance_days = Column(BigInteger, primary_key=True, comment='提前期', info={ "comment_source": 'description', "db_field_exists": '是', "default": 99, "range": (0, None) })
    ao_percent = Column(Float, comment='AO比例', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, 1) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_DPSConfig ─────────────────────────────────────────
class M1Dpsconfig(Base, CfgBase):
    """配置表 M1_DPSConfig。"""
    __tablename__ = "cfg_m1_dpsconfig"
    __table_args__ = {'info': { "sheet_name": 'M1_DPSConfig', "import_enabled": True, "required_import": False, "optional_import": True, "quality_check_enabled": True, "primary_key": ['location', 'dps_location', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料', info={ "comment_source": 'description', "db_field_exists": '是' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    dps_location = Column(Text, primary_key=True, comment='工厂', info={ "comment_source": 'description', "db_field_exists": '是' })
    dps_percent = Column(Float, comment='DPS地点分配比例', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, 1) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M1_SupplyChoiceConfig ─────────────────────────────────────────
class M1Supplychoiceconfig(Base, CfgBase):
    """配置表 M1_SupplyChoiceConfig。"""
    __tablename__ = "cfg_m1_supplychoiceconfig"
    __table_args__ = {'info': { "sheet_name": 'M1_SupplyChoiceConfig', "import_enabled": True, "required_import": False, "optional_import": True, "quality_check_enabled": True, "primary_key": ['location', 'week', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'M' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是', "default": 'L' })
    week = Column(BigInteger, primary_key=True, comment='仿真周', info={ "comment_source": 'description', "db_field_exists": '是', "default": 99 })
    adjust_quantity = Column(Float, comment='调整数量', info={ "comment_source": 'description', "db_field_exists": '是' })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M3_SafetyStock ─────────────────────────────────────────
class M3Safetystock(Base, CfgBase):
    """配置表 M3_SafetyStock。"""
    __tablename__ = "cfg_m3_safetystock"
    __table_args__ = {'info': { "sheet_name": 'M3_SafetyStock', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['date', 'location', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    date = Column(DateTime, primary_key=True, comment='时间', info={ "comment_source": 'description', "db_field_exists": '是' })
    safety_stock_qty = Column(Float, comment='安全库存数量', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M4_MaterialLocationLineCfg ─────────────────────────────────────────
class M4Materiallocationlinecfg(Base, CfgBase):
    """配置表 M4_MaterialLocationLineCfg。"""
    __tablename__ = "cfg_m4_materiallocationlinecfg"
    __table_args__ = {'info': { "sheet_name": 'M4_MaterialLocationLineCfg', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['location', 'delegate_line', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'description', "db_field_exists": '是' })
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    delegate_line = Column(Text, primary_key=True, comment='产线', info={ "comment_source": 'description', "db_field_exists": '是' })
    prd_rate = Column(Float, comment='线速', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    min_batch = Column(Float, comment='最少生产批量', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    rv = Column(Float, comment='rounding value', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    ptf = Column(Integer, comment='planning time fence', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    lsk = Column(Integer, comment='lot size key', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    day = Column(BigInteger, comment='day', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    mct = Column(Integer, comment='微检时间', info={ "local_name": 'MCT', "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M4_LineCapacity ─────────────────────────────────────────
class M4Linecapacity(Base, CfgBase):
    """配置表 M4_LineCapacity。"""
    __tablename__ = "cfg_m4_linecapacity"
    __table_args__ = {'info': { "sheet_name": 'M4_LineCapacity', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['line', 'location', 'date'] }}
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    line = Column(Text, primary_key=True, comment='生产线', info={ "comment_source": 'description', "db_field_exists": '是' })
    date = Column(DateTime, primary_key=True, comment='日期', info={ "comment_source": 'description', "db_field_exists": '是' })
    capacity = Column(Float, comment='可用产能小时', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M4_ChangeoverMatrix ─────────────────────────────────────────
class M4Changeovermatrix(Base, CfgBase):
    """配置表 M4_ChangeoverMatrix。"""
    __tablename__ = "cfg_m4_changeovermatrix"
    __table_args__ = {'info': { "sheet_name": 'M4_ChangeoverMatrix', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['from_material', 'to_material'] }}
    from_material = Column(Text, primary_key=True, comment='起始物料', info={ "comment_source": 'description', "db_field_exists": '是' })
    to_material = Column(Text, primary_key=True, comment='切换物料', info={ "comment_source": 'description', "db_field_exists": '是' })
    changeover_id = Column(Text, comment='转产类型', info={ "comment_source": 'description', "db_field_exists": '是' })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M4_ChangeoverDefinition ─────────────────────────────────────────
class M4Changeoverdefinition(Base, CfgBase):
    """配置表 M4_ChangeoverDefinition。"""
    __tablename__ = "cfg_m4_changeoverdefinition"
    __table_args__ = {'info': { "sheet_name": 'M4_ChangeoverDefinition', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['changeover_id', 'line'] }}
    changeover_id = Column(Text, primary_key=True, comment='转产类型', info={ "comment_source": 'description', "db_field_exists": '是' })
    line = Column(Text, primary_key=True, comment='生产线', info={ "comment_source": 'description', "db_field_exists": '是' })
    time = Column(Float, comment='转产时间', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    cost = Column(Float, comment='转产成本', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    mu_loss = Column(Float, comment='转产材料损失', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M4_ProductionReliability ─────────────────────────────────────────
class M4Productionreliability(Base, CfgBase):
    """配置表 M4_ProductionReliability。"""
    __tablename__ = "cfg_m4_productionreliability"
    __table_args__ = {'info': { "sheet_name": 'M4_ProductionReliability', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['line', 'location'] }}
    location = Column(Text, primary_key=True, comment='地点', info={ "comment_source": 'description', "db_field_exists": '是' })
    line = Column(Text, primary_key=True, comment='产线', info={ "comment_source": 'description', "db_field_exists": '是' })
    pr = Column(Float, comment='PR', info={ "comment_source": 'description', "db_field_exists": '是', "range": (0, 1) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M5_PushPullModel ─────────────────────────────────────────
class M5Pushpullmodel(Base, CfgBase):
    """配置表 M5_PushPullModel。"""
    __tablename__ = "cfg_m5_pushpullmodel"
    __table_args__ = {'info': { "sheet_name": 'M5_PushPullModel', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['sending', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    model = Column(Text, comment='push/pull/soft push策略模式', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "enumerate": ['push', 'soft push'] })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M5_DeployConfig ─────────────────────────────────────────
class M5Deployconfig(Base, CfgBase):
    """配置表 M5_DeployConfig。"""
    __tablename__ = "cfg_m5_deployconfig"
    __table_args__ = {'info': { "sheet_name": 'M5_DeployConfig', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['sending', 'receiving', 'material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    receiving = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    moq = Column(Float, comment='最小起订量', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    rv = Column(Float, comment='四舍五入值', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    lsk = Column(Integer, comment='批量规模键', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    day = Column(BigInteger, comment='部署配置天数参数', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M6_TruckReleaseCon ─────────────────────────────────────────
class M6Truckreleasecon(Base, CfgBase):
    """配置表 M6_TruckReleaseCon。"""
    __tablename__ = "cfg_m6_truckreleasecon"
    __table_args__ = {'info': { "sheet_name": 'M6_TruckReleaseCon', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['sending', 'receiving', 'truck_type'] }}
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    receiving = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    truck_type = Column(Text, primary_key=True, comment='车型', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    optimal_type = Column(Text, comment='装车优化类型', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    wfr = Column(Float, comment='重量装载率阈值', info={ "local_name": 'WFR', "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    vfr = Column(Float, comment='体积装载率阈值', info={ "local_name": 'VFR', "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    # mdp需要废弃
    mdq = Column(Float, comment='最小发运数量阈值', info={ "local_name": 'MDQ', "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M6_MaterialMD ─────────────────────────────────────────
class M6Materialmd(Base, CfgBase):
    """配置表 M6_MaterialMD。"""
    __tablename__ = "cfg_m6_materialmd"
    __table_args__ = {'info': { "sheet_name": 'M6_MaterialMD', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['material'] }}
    material = Column(Text, primary_key=True, comment='物料号', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    demand_unit_to_weight = Column(Float, comment='需求单位到重量的换算系数', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    demand_unit_to_volume = Column(Float, comment='需求单位到体积的换算系数', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M6_DeliveryDelayDistribution ─────────────────────────────────────────
class M6Deliverydelaydistribution(Base, CfgBase):
    """配置表 M6_DeliveryDelayDistribution。"""
    __tablename__ = "cfg_m6_deliverydelaydistribution"
    __table_args__ = {'info': { "sheet_name": 'M6_DeliveryDelayDistribution', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['date', 'receiving', 'sending', 'delay_days'] }}
    date = Column(Text, primary_key=True, comment='日期', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    receiving = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    delay_days = Column(BigInteger, primary_key=True, comment='运输延迟天数', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    probability = Column(Float, comment='运输延迟概率', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, 1) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M6_MDQBypassRules ─────────────────────────────────────────
class M6Mdqbypassrules(Base, CfgBase):
    """配置表 M6_MDQBypassRules。"""
    __tablename__ = "cfg_m6_mdqbypassrules"
    __table_args__ = {'info': { "sheet_name": 'M6_MDQBypassRules', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['demand_element', 'sending', 'receiving', 'truck_type'] }}
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    receiving = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    truck_type = Column(Text, primary_key=True, comment='车型', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    demand_element = Column(Text, primary_key=True, comment='需求类型', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    condition_logic = Column(Text, comment='MDQ绕过规则条件逻辑', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    rule_id = Column(Text, comment='规则ID', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M6_TruckTypeSpecs ─────────────────────────────────────────
class M6Trucktypespecs(Base, CfgBase):
    """配置表 M6_TruckTypeSpecs。"""
    __tablename__ = "cfg_m6_trucktypespecs"
    __table_args__ = {'info': { "sheet_name": 'M6_TruckTypeSpecs', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['truck_type'] }}
    truck_type = Column(Text, primary_key=True, comment='车型', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是' })
    capacity_qty_in_weight = Column(Float, comment='车型重量容量', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    capacity_qty_in_volume = Column(Float, comment='车型体积容量', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── M6_TruckCapacityPlan ─────────────────────────────────────────
class M6Truckcapacityplan(Base, CfgBase):
    """配置表 M6_TruckCapacityPlan。"""
    __tablename__ = "cfg_m6_truckcapacityplan"
    __table_args__ = {'info': { "sheet_name": 'M6_TruckCapacityPlan', "import_enabled": True, "required_import": True, "optional_import": False, "quality_check_enabled": True, "primary_key": ['date', 'receiving', 'sending', 'truck_type'] }}
    date = Column(Text, primary_key=True, comment='日期', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "default": '1942-01-01T00:00:00' })
    sending = Column(Text, primary_key=True, comment='发货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "default": 'S' })
    receiving = Column(Text, primary_key=True, comment='收货点', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "default": 'R' })
    truck_type = Column(Text, primary_key=True, comment='车型', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "default": 'T' })
    truck_number = Column(Integer, comment='可用车辆数量', info={ "comment_source": 'manual_mapping_completion', "db_field_exists": '是', "default": 999, "range": (0, None) })
    config_name = Column(Text, comment='配置名', info={ "comment_source": 'remark', "db_field_exists": '是' })
    db_write_time = Column(DateTime, comment='写库时间', info={ "comment_source": 'remark', "db_field_exists": '是' })

# ── 注册表（sheet_name → SA 类） ────────────────────────
CONFIG_TABLE_REGISTRY: dict[str, type] = {
    'Global_seed': GlobalSeed,
    'Global_Network': GlobalNetwork,
    'Global_SpaceCapacity': GlobalSpacecapacity,
    'Global_LeadTime': GlobalLeadtime,
    'Global_DemandPriority': GlobalDemandpriority,
    'M1_InitialInventory': M1Initialinventory,
    'M1_DemandForecast': M1Demandforecast,
    'M1_ForecastError': M1Forecasterror,
    'M1_OrderCalendar': M1Ordercalendar,
    'M1_AOConfig': M1Aoconfig,
    'M1_DPSConfig': M1Dpsconfig,
    'M1_SupplyChoiceConfig': M1Supplychoiceconfig,
    'M3_SafetyStock': M3Safetystock,
    'M4_MaterialLocationLineCfg': M4Materiallocationlinecfg,
    'M4_LineCapacity': M4Linecapacity,
    'M4_ChangeoverMatrix': M4Changeovermatrix,
    'M4_ChangeoverDefinition': M4Changeoverdefinition,
    'M4_ProductionReliability': M4Productionreliability,
    'M5_PushPullModel': M5Pushpullmodel,
    'M5_DeployConfig': M5Deployconfig,
    'M6_TruckReleaseCon': M6Truckreleasecon,
    'M6_MaterialMD': M6Materialmd,
    'M6_DeliveryDelayDistribution': M6Deliverydelaydistribution,
    'M6_MDQBypassRules': M6Mdqbypassrules,
    'M6_TruckTypeSpecs': M6Trucktypespecs,
    'M6_TruckCapacityPlan': M6Truckcapacityplan,
}

def sheet_to_cfg_table() -> dict[str, str]:
    """sheet_name → cfg_* 完整表名（Model.__tablename__ 派生）。"""
    return {sheet: model.__tablename__ for sheet, model in CONFIG_TABLE_REGISTRY.items()}

__all__ = [
    "CONFIG_TABLE_REGISTRY",
    "sheet_to_cfg_table",
    "GlobalSeed",
    "GlobalNetwork",
    "GlobalSpacecapacity",
    "GlobalLeadtime",
    "GlobalDemandpriority",
    "M1Initialinventory",
    "M1Demandforecast",
    "M1Forecasterror",
    "M1Ordercalendar",
    "M1Aoconfig",
    "M1Dpsconfig",
    "M1Supplychoiceconfig",
    "M3Safetystock",
    "M4Materiallocationlinecfg",
    "M4Linecapacity",
    "M4Changeovermatrix",
    "M4Changeoverdefinition",
    "M4Productionreliability",
    "M5Pushpullmodel",
    "M5Deployconfig",
    "M6Truckreleasecon",
    "M6Materialmd",
    "M6Deliverydelaydistribution",
    "M6Mdqbypassrules",
    "M6Trucktypespecs",
    "M6Truckcapacityplan",
    "build_schema_compat_dict",
]


# ── SA 类型 → db_type 字符串反向映射 ──────────────────────────────────────────

_SA_TYPE_TO_DB_TYPE: dict[type, str] = {
    Text: "str",
    BigInteger: "bigint",
    Integer: "integer",
    Float: "double precision",
    Numeric: "numeric",
    DateTime: "timestamp without time zone",
}


def _col_db_type(col: "Column") -> str:
    """从 SA Column 的类型推导 db_type 字符串。"""
    sa_type = type(col.type)
    return _SA_TYPE_TO_DB_TYPE.get(sa_type, "str")


def build_schema_compat_dict() -> dict[str, dict]:
    """从 SA 模型注册表构建与 DQ checker 兼容的 schema dict。

    产出形状与 ``CONFIG_TABLE_SCHEMAS`` 一致，让 ``ConfigTableQualityRules``
    可以直接消费——但数据来源是 cfg 模型的 Column.info / __table_args__["info"]，
    不再依赖 ``config_table_schema.py``。

    Returns:
        dict[sheet_name, table_cfg] — table_cfg 含 local_sheet / db_table /
        import_enabled / primary_key / fields / system_fields，与 checker 期望
        的格式完全一致。
    """
    from sqlalchemy import Column as SAColumn

    result: dict[str, dict] = {}

    for sheet, model_cls in CONFIG_TABLE_REGISTRY.items():
        table_info = model_cls.__table__.info or {}
        pk_list = list(table_info.get("primary_key") or [])

        # 判断哪些列是 system_fields（config_name / db_write_time）
        system_col_names = {"config_name", "db_write_time"}

        fields: list[dict] = []
        system_fields: list[dict] = []
        for col in model_cls.__table__.columns:
            col_name = col.name
            info = col.info or {}
            db_type_str = _col_db_type(col)

            # local_name：优先取 info["local_name"]，否则等于 db_name
            local_name = info.get("local_name", col_name)

            # notnull: Column nullable 的反值
            notnull = not col.nullable if col.nullable is not None else True

            # primary_key: 优先取 pk_list，否则从 col.primary_key 推导
            is_pk = col_name in pk_list or col.primary_key

            field_dict: dict = {
                "local_name": local_name,
                "db_name": col_name,
                "db_type": db_type_str,
                "primary_key": is_pk,
                "notnull": notnull,
                "comment": col.comment or "",
            }

            # 可选元数据（只在 info 中存在时添加）
            for key in ("comment_source", "db_field_exists", "default"):
                if key in info:
                    field_dict[key] = info[key]
            for key in ("range", "enumerate", "date_flag"):
                if key in info:
                    field_dict[key] = info[key]

            if col_name in system_col_names:
                system_fields.append(field_dict)
            else:
                fields.append(field_dict)

        result[sheet] = {
            "local_sheet": table_info.get("sheet_name", sheet),
            "db_table": model_cls.__tablename__,
            "import_enabled": table_info.get("import_enabled", True),
            "primary_key": pk_list,
            "fields": fields,
            "system_fields": system_fields,
        }

    return result