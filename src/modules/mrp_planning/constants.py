"""
Module3 常量定义。

存放所有Module3使用的常量值，便于统一管理和修改。
"""

# MOQ/RV默认值
DEFAULT_MOQ = 1
DEFAULT_RV = 1

# 时间窗口默认值
DEFAULT_HORIZON = 1
DEFAULT_PTF = 0
DEFAULT_LSK = 1

# DataFrame列名常量
COL_MATERIAL = 'material'
COL_LOCATION = 'location'
COL_SENDING = 'sending'
COL_RECEIVING = 'receiving'
COL_SOURCING = 'sourcing'
COL_DATE = 'date'
COL_QUANTITY = 'quantity'
COL_LAYER = 'layer'

# 地点类型常量
LOCATION_TYPE_PLANT = 'Plant'
LOCATION_TYPE_DC = 'DC'

# 需求类型常量
DEMAND_TYPE_AO = 'AO'
DEMAND_ELEMENT_AO = 'net demand for AO'
DEMAND_ELEMENT_FORECAST = 'net demand for forecast'
DEMAND_ELEMENT_SAFETY = 'net demand for safety'

# 标识符列名列表
IDENTIFIER_COLUMNS = [
    COL_MATERIAL,
    COL_LOCATION,
    COL_SENDING,
    COL_RECEIVING,
    COL_SOURCING,
]

# 地点类型列名列表(需要4位前导零格式化)
LOCATION_TYPE_COLUMNS = [
    COL_LOCATION,
    COL_SENDING,
    COL_RECEIVING,
]

# Sheet名称映射
SHEET_MAPPING = {
    'M3_SafetyStock': ('safety_stock', None),
    'Global_Network': ('network_config', None),
    'Global_LeadTime': ('lead_time_config', None),
}
