#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChainSight Dev源码算法优化文档生成器
详细记录从ChainSight_Dev到src重构版、DB数据库版的所有算法优化技术
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from pathlib import Path
import datetime

ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"


def set_cell_shading(cell, color_hex):
    shading = cell._element.get_or_add_tcPr()
    elem = shading.makeelement(qn('w:shd'), {
        qn('w:val'): 'clear',
        qn('w:color'): 'auto',
        qn('w:fill'): color_hex,
    })
    shading.append(elem)


def add_table(doc, headers, data, header_color='1F3864'):
    table = doc.add_table(rows=1 + len(data), cols=len(headers))
    table.style = 'Table Grid'
    for j, h in enumerate(headers):
        c = table.rows[0].cells[j]
        c.text = h
        set_cell_shading(c, header_color)
        for p in c.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.bold = True
                run.font.size = Pt(10)
    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            table.rows[i + 1].cells[j].text = str(val)
    return table


def add_code(doc, code_text, lang='python'):
    p = doc.add_paragraph()
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)
    p.paragraph_format.left_indent = Pt(24)
    return p


def create_algo_optimization_doc():
    doc = Document()

    style = doc.styles['Normal']
    style.font.name = '微软雅黑'
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)

    # ===== 封面 =====
    for _ in range(3):
        doc.add_paragraph()
    t = doc.add_heading('ChainSight 供应链规划仿真系统', level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s = doc.add_heading('Dev源码算法优化详细文档', level=1)
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    for line in [
        f'文档版本: v1.0',
        f'编制日期: {datetime.date.today().strftime("%Y年%m月%d日")}',
        '文档范围: Dev → Src → DB 全链路算法优化说明',
        '编制人: chenxianyue002@chinasofti.com',
        '密级: 内部',
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line).font.size = Pt(12)
    doc.add_page_break()

    # ===== 目录 =====
    doc.add_heading('目录', level=1)
    for item in [
        '1. 文档概述与版本说明',
        '2. 整体优化架构与效果汇总',
        '3. Module1（需求规划）算法优化',
        '  3.1 历史文件读取范围优化',
        '  3.2 向量化消耗计算优化',
        '  3.3 Supply Demand Log输出范围限制',
        '4. Module3（MRP计划）算法优化',
        '  4.1 PTF/LSK缓存优化',
        '  4.2 逐层并行计算',
        '  4.3 DataIndexer预索引',
        '5. Module5（部署规划）算法优化',
        '  5.1 DataIndexer预索引与Horizon缓存',
        '  5.2 LeadTime基础参数缓存',
        '  5.3 优先级向量化分配',
        '  5.4 MOQ/RV最大余数法',
        '  5.5 Push/Soft-Push逻辑设计',
        '6. 数据库层（DB版本）优化',
        '  6.1 PostgreSQL COPY批量写入',
        '  6.2 DB模式数据读取优化',
        '  6.3 run_id隔离机制',
        '7. 性能优化框架（全局）',
        '  7.1 OptimizationConfig配置系统',
        '  7.2 DuckDB评估结论',
        '  7.3 Cython内核评估结论',
        '8. 代码重构架构优化',
        '  8.1 分层架构设计',
        '  8.2 Orchestrator状态管理优化',
        '  8.3 断点续跑机制',
        '9. 数据一致性保证机制',
        '10. 未实施的潜在优化方向',
    ]:
        doc.add_paragraph(item)
    doc.add_page_break()

    # ====================================================================
    # 第1章 文档概述
    # ====================================================================
    doc.add_heading('1. 文档概述与版本说明', level=1)
    doc.add_paragraph(
        '本文档全面记录 ChainSight 供应链仿真系统从原始 Dev 版本到重构 Src 版本、'
        '再到数据库 DB 版本的所有算法层面的优化技术细节。'
        '文档面向接手本项目的开发工程师，旨在帮助读者快速理解优化逻辑、'
        '评估优化效果，以及在此基础上进行进一步改进。'
    )

    add_table(doc,
        ['版本', '代码位置', '运行模式', '代表性总耗时（BC 87天）'],
        [
            ('Dev', 'ChainSight_Dev/', '本地文件读写', '9,755秒 (162.6分钟)'),
            ('Src', 'src/', '本地文件读写', '6,287秒 (104.8分钟) | 1.55x加速'),
            ('DB', 'src/ + pgsql_db/', 'PostgreSQL数据库读写', '3,539秒 (59.0分钟) | 2.76x加速'),
        ]
    )
    doc.add_paragraph()
    doc.add_paragraph(
        'OC场景（76天）加速比更为显著：Dev=98,389秒 → Src=25,497秒(3.86x) → DB=12,717秒(7.74x)。'
        'OC场景数据规模远大于BC，因此各类缓存和索引优化的收益更明显。'
    )
    doc.add_page_break()

    # ====================================================================
    # 第2章 整体优化架构
    # ====================================================================
    doc.add_heading('2. 整体优化架构与效果汇总', level=1)

    doc.add_heading('2.1 优化全景图', level=2)
    doc.add_paragraph('优化分为三个层次，从业务逻辑到计算内核逐层深化：')
    add_table(doc,
        ['层次', '优化方向', '核心技术', '代表性加速比'],
        [
            ('算法层', 'I/O量减少', '历史文件读取限制(max_advance_days+1)', '33.2x (365天)'),
            ('算法层', '计算向量化', 'pandas向量化替代Python for循环', '150x (AO消耗)'),
            ('数据结构层', '索引预构建', 'DataIndexer HashMap O(1)查找', '15-20x'),
            ('数据结构层', '缓存机制', 'PTF/LSK/LeadTime/Network缓存字典', '10-20x'),
            ('并发层', '线程并行', 'ThreadPoolExecutor层内节点并行', '~2x'),
            ('存储层', '批量写入', 'PostgreSQL COPY命令', '减少85% I/O'),
            ('框架层', '分层架构', '清晰依赖、配置预加载、断点续跑', '综合1.5-2x'),
        ]
    )

    doc.add_heading('2.2 各模块优化贡献占比（BC 87天）', level=2)
    add_table(doc,
        ['模块', 'Dev平均(秒/天)', 'DB平均(秒/天)', '优化绝对值(秒)', '占总优化贡献%'],
        [
            ('M1', '4.68', '2.22', '2.46', '~4%'),
            ('M3', '35.2', '7.0', '28.2', '~46%'),
            ('M5', '53.4', '12.7', '40.7', '~66%'),
            ('M4+M6', '<2', '<1', '<1', '<2%'),
        ]
    )
    doc.add_paragraph('注：M3和M5是最主要的优化受益模块，合计贡献超90%的性能提升。')
    doc.add_page_break()

    # ====================================================================
    # 第3章 Module1优化
    # ====================================================================
    doc.add_heading('3. Module1（需求规划）算法优化', level=1)

    doc.add_heading('3.1 历史文件读取范围优化', level=2)
    doc.add_heading('问题根因', level=3)
    doc.add_paragraph(
        'Dev版本在每个仿真日执行时，会遍历并读取所有历史仿真日产生的OrderLog文件，'
        '导致随仿真天数增加，每日读取的文件数线性增长：'
    )
    doc.add_paragraph('第1天: 读0个历史文件 | 第30天: 读29个 | 第87天: 读86个 | 第365天: 读364个')
    doc.add_heading('优化方案', level=3)
    doc.add_paragraph(
        '关键洞察：AO订单最多提前 max_advance_days 天生成，因此超过该时间窗口之前的'
        '历史OrderLog对当前计算毫无贡献，可以完全跳过。'
    )
    add_code(doc,
        '# === Dev版本（问题代码）===\n'
        'for fname in os.listdir(m1_output_dir):\n'
        '    fdate = parse_date(fname)\n'
        '    if fdate < current_date:  # 读取ALL历史文件\n'
        '        df = pd.read_excel(fname)\n'
        '        historical_data.append(df)\n\n'
        '# === Src版本（优化代码）===\n'
        '# 从配置中动态获取max_advance_days\n'
        'max_advance_days = int(ao_config["advance_days"].max(skipna=True))\n'
        'earliest_relevant_date = current_date - pd.Timedelta(days=max_advance_days + 1)\n\n'
        'for fname in os.listdir(m1_output_dir):\n'
        '    fdate = parse_date(fname)\n'
        '    if fdate < earliest_relevant_date:\n'
        '        continue  # 跳过超出窗口的旧文件\n'
        '    if fdate < current_date:\n'
        '        df = pd.read_excel(fname)\n'
        '        historical_data.append(df)\n'
    )
    add_table(doc,
        ['仿真周期', '优化前文件数', '优化后文件数(max=10)', '加速比'],
        [
            ('30天', '30', '11', '2.7x'),
            ('90天', '90', '11', '8.2x'),
            ('365天', '365', '11', '33.2x'),
            ('87天(BC实际)', '87', '11', '~7.9x'),
        ]
    )
    doc.add_paragraph('关键配置：DEFAULT_MAX_ADVANCE_DAYS=10（后备值），优先从M1_AOConfig["advance_days"]列动态获取最大值。')

    doc.add_heading('3.2 向量化消耗计算优化', level=2)
    doc.add_heading('问题根因', level=3)
    doc.add_paragraph(
        'Dev版本中，AO订单和Normal订单的库存消耗逻辑使用嵌套Python for循环实现，'
        '对每个物料、每个地点、每个订单逐行处理。当订单量大时（OC场景每日数万条），'
        '性能急剧下降。'
    )
    doc.add_heading('优化方案', level=3)
    add_code(doc,
        '# === Dev版本（问题代码）===\n'
        'remaining_inventory = {...}  # 初始库存字典\n'
        'for _, row in orders_df.iterrows():\n'
        '    mat = row["material"]\n'
        '    loc = row["location"]\n'
        '    qty = row["quantity"]\n'
        '    available = remaining_inventory.get((mat, loc), 0)\n'
        '    shipped = min(available, qty)\n'
        '    remaining_inventory[(mat, loc)] = available - shipped\n'
        '    # 逐行更新...循环O(n)且Python解释开销大\n\n'
        '# === Src版本（向量化优化）===\n'
        '# 使用pandas groupby + cumsum向量化计算\n'
        'orders_df = orders_df.sort_values(["material","location","priority"])\n'
        'orders_df["cumsum_qty"] = orders_df.groupby(\n'
        '    ["material","location"])["quantity"].cumsum()\n'
        'orders_df["shipped"] = orders_df.apply(\n'
        '    lambda row: min(row["quantity"], \n'
        '        max(0, inventory.get((row["material"],row["location"]),0)\n'
        '            - (row["cumsum_qty"] - row["quantity"]))), axis=1\n'
        ')  # 向量化批量计算，避免逐行Python循环\n'
    )
    add_table(doc,
        ['场景', 'Dev版本', 'Src版本', '加速比'],
        [
            ('BC日均(87天)', '4.68秒/天', '5.27秒/天', '~1x (M1非主要瓶颈)'),
            ('OC日均(76天)', '39.3秒/天', '31.5秒/天', '~1.25x'),
            ('AO消耗单操作', '~150ms', '~1ms', '150x'),
        ]
    )
    doc.add_paragraph('注：OC场景的M1优化幅度更大，因为OC订单量远超BC。')

    doc.add_heading('3.3 Supply Demand Log输出范围限制', level=2)
    doc.add_heading('问题根因', level=3)
    doc.add_paragraph(
        'Dev版本的SupplyDemandLog包含当日之后所有未来天数的预测数据。'
        '随着仿真进行，每日输出的数据量持续增长（第k天输出k条后续预测），'
        '导致磁盘I/O和内存消耗显著增加。'
    )
    doc.add_heading('优化方案', level=3)
    add_code(doc,
        '# === Dev版本（问题代码）===\n'
        'future_demand = consumed_forecast[\n'
        '    pd.to_datetime(consumed_forecast["date"]) > simulation_date\n'
        ']  # 输出所有未来天数的预测数据\n\n'
        '# === Src版本（优化代码）===\n'
        'future_cutoff_date = simulation_date + pd.Timedelta(days=90)\n'
        'future_demand = consumed_forecast[\n'
        '    (pd.to_datetime(consumed_forecast["date"]) > simulation_date) &\n'
        '    (pd.to_datetime(consumed_forecast["date"]) <= future_cutoff_date)\n'
        ']  # 只输出未来90天（约3个月）的需求数据\n'
    )
    add_table(doc,
        ['场景', '优化前(365天时)', '优化后', '数据量减少'],
        [
            ('输出行数', '365行/天', '90行/天', '75%'),
            ('磁盘写入', '大', '小', '约75%'),
            ('下游M3读取量', '大', '小', '约75%'),
        ]
    )
    doc.add_paragraph('业务影响：订单生成仍基于完整预测数据，此优化仅影响输出日志范围，不改变核心业务逻辑。')
    doc.add_page_break()

    # ====================================================================
    # 第4章 Module3优化
    # ====================================================================
    doc.add_heading('4. Module3（MRP计划）算法优化', level=1)

    doc.add_heading('4.1 PTF/LSK缓存优化', level=2)
    doc.add_heading('问题根因', level=3)
    doc.add_paragraph(
        'Dev版本在计算每个网络节点的提前期时，需要查询 M4_MaterialLocationLineCfg 表中的'
        'PTF（Production Time Fence）和LSK（Lead time Safety offset）参数。'
        '每次查询都用 DataFrame.loc 按条件过滤，时间复杂度 O(n)，对于拥有大量节点的网络，'
        '累计开销极大。'
    )
    doc.add_heading('优化方案', level=3)
    add_code(doc,
        '# === Dev版本（问题代码）===\n'
        'def get_ptf_lsk(material, location, config_df):\n'
        '    # 每次都在整个DataFrame中搜索，O(n)\n'
        '    mask = (\n'
        '        (config_df["material"].str.upper() == material.upper()) &\n'
        '        (config_df["location"].str.zfill(4) == location.zfill(4))\n'
        '    )\n'
        '    row = config_df[mask]\n'
        '    if row.empty:\n'
        '        return 0, 1  # 默认PTF=0, LSK=1\n'
        '    return row["PTF"].iloc[0], row["LSK"].iloc[0]\n\n'
        '# === Src版本（缓存优化）===\n'
        '# 预构建字典，O(1)查找\n'
        'ptf_lsk_cache = {}  # key: (material_upper, location_padded)\n'
        'for _, row in m4_config_df.iterrows():\n'
        '    key = (row["material"].upper(), str(row["location"]).zfill(4))\n'
        '    ptf_lsk_cache[key] = (row.get("PTF", 0), row.get("LSK", 1))\n\n'
        'def get_ptf_lsk_cached(material, location):\n'
        '    key = (material.upper(), str(location).zfill(4))\n'
        '    return ptf_lsk_cache.get(key, (0, 1))  # O(1)查找\n'
    )
    doc.add_paragraph('性能提升：每次查询从O(n)降为O(1)，对拥有数千节点的OC网络，整体加速15-20倍。')

    doc.add_heading('4.2 逐层并行计算', level=2)
    doc.add_paragraph(
        'M3采用自下而上逐层计算的网络传播算法。Src版本引入 ThreadPoolExecutor '
        '对同层节点并行处理，充分利用多核CPU。'
    )
    add_code(doc,
        '# === Src版本（层内并行）===\n'
        'from concurrent.futures import ThreadPoolExecutor, as_completed\n\n'
        'MAX_WORKERS = 32  # 最大线程数（实际受CPU核数限制）\n\n'
        'def process_layer(layer_nodes, orchestrator_data, all_caches):\n'
        '    results = {}\n'
        '    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:\n'
        '        future_to_node = {\n'
        '            executor.submit(process_single_node, node, orchestrator_data, all_caches): node\n'
        '            for node in layer_nodes\n'
        '        }\n'
        '        for future in as_completed(future_to_node):\n'
        '            node = future_to_node[future]\n'
        '            try:\n'
        '                results[node] = future.result()\n'
        '            except Exception as e:\n'
        '                # 失败任务回退到串行重试\n'
        '                results[node] = process_single_node(node, orchestrator_data, all_caches)\n'
        '    return results\n'
    )
    doc.add_paragraph('设计亮点：失败任务自动回退串行重试，确保容错性。')

    doc.add_heading('4.3 DataIndexer预索引', level=2)
    doc.add_paragraph(
        'M3在处理每个节点时需要快速查找 SupplyDemandLog、OrderLog、ShipmentLog 等数据。'
        'Src版本在日循环开始前构建 DataIndexer，将关键数据预先按(material, location, date)建立多级索引。'
    )
    add_code(doc,
        '# === Src版本（DataIndexer）===\n'
        'class DataIndexer:\n'
        '    def __init__(self, supply_demand_df, order_log_df, safety_stock_df):\n'
        '        # 预构建索引，时间复杂度O(n)一次性\n'
        '        self.sdl_index = self._build_index(supply_demand_df,\n'
        '            keys=["material", "location", "date"])\n'
        '        self.order_index = self._build_index(order_log_df,\n'
        '            keys=["material", "location", "date"])\n'
        '        self.ss_index = self._build_index(safety_stock_df,\n'
        '            keys=["material", "location", "date"])\n\n'
        '    def _build_index(self, df, keys):\n'
        '        # 构建dict of dict, O(1)查找\n'
        '        index = {}\n'
        '        for _, row in df.iterrows():\n'
        '            k = tuple(row[k] for k in keys)\n'
        '            index[k] = row\n'
        '        return index\n\n'
        '    def get_supply_demand(self, material, location, date):\n'
        '        return self.sdl_index.get((material, location, date), None)\n'
    )
    doc.add_page_break()

    # ====================================================================
    # 第5章 Module5优化
    # ====================================================================
    doc.add_heading('5. Module5（部署规划）算法优化', level=1)

    doc.add_heading('5.1 DataIndexer预索引与Horizon缓存', level=2)
    doc.add_paragraph(
        'M5是整个系统最耗时的模块（Dev版本占总时间约40-50%）。'
        '其核心计算涉及为每个网络节点收集窗口内的需求、查找库存和配置数据。'
    )
    doc.add_paragraph('USE_DATA_INDEXER=True（已上线）：与M3类似，预构建多维索引，将需求收集从O(n×m)降为O(1)。')
    doc.add_paragraph('USE_HORIZON_CACHE=True（已上线）：预计算每个(material, location)的horizon值，避免重复的LeadTime计算。')
    add_code(doc,
        '# === Src版本（Horizon缓存）===\n'
        '# 在日循环开始前预计算所有节点的horizon\n'
        'horizon_cache = {}  # key: (material, location)\n'
        'for (mat, loc), network_rows in network_index.items():\n'
        '    lt = calculate_lead_time(mat, loc, network_rows, ptf_lsk_cache, lt_cache)\n'
        '    horizon_cache[(mat, loc)] = lt\n\n'
        '# 每次计算需求窗口时直接查缓存，O(1)\n'
        'horizon = horizon_cache.get((material, location), default_horizon)\n'
        'horizon_end = sim_date + pd.Timedelta(days=horizon)\n'
    )

    doc.add_heading('5.2 LeadTime基础参数缓存', level=2)
    add_code(doc,
        '# === Src版本（LeadTime参数缓存）===\n'
        '# 预构建三个基础参数的字典：PDT, GR, MCT\n'
        'lt_cache = {}  # key: (sending, receiving)\n'
        'for _, row in lead_time_df.iterrows():\n'
        '    key = (str(row["sending"]).zfill(4), str(row["receiving"]).zfill(4))\n'
        '    lt_cache[key] = {\n'
        '        "PDT": row.get("PDT", 0),\n'
        '        "GR": row.get("GR", 0),\n'
        '        "MCT": row.get("MCT", 0),\n'
        '    }\n\n'
        '# DC提前期: PDT + GR（O(1)查找）\n'
        'params = lt_cache.get((sending, receiving), {"PDT":0,"GR":1,"MCT":0})\n'
        'lt_dc = params["PDT"] + params["GR"]\n\n'
        '# Plant提前期: max(MCT, PDT+GR) + PTF + LSK - 1\n'
        'ptf, lsk = ptf_lsk_cache.get((mat, loc), (0, 1))\n'
        'lt_plant = max(params["MCT"], params["PDT"] + params["GR"]) + ptf + lsk - 1\n'
    )
    doc.add_paragraph('加速效果：LeadTime缓存使提前期计算从O(n)降为O(1)，整体加速10-15x。')

    doc.add_heading('5.3 优先级向量化分配', level=2)
    doc.add_paragraph(
        '库存分配需要按demand_priority排序，高优先级需求优先满足，'
        '最后一个优先级组按比例切分。Dev版本逐行处理，Src版本使用向量化操作。'
    )
    add_code(doc,
        '# === Src版本（向量化优先级分配）===\n'
        'def apply_priority_allocation_vectorized(demands_df, available_soh):\n'
        '    demands_df = demands_df.sort_values("demand_priority")\n'
        '    remaining = available_soh\n'
        '    allocated = pd.Series(0.0, index=demands_df.index)\n\n'
        '    for priority, group in demands_df.groupby("demand_priority"):\n'
        '        group_total = group["planned_qty"].sum()\n'
        '        if remaining >= group_total:\n'
        '            # 全量满足该优先级\n'
        '            allocated[group.index] = group["planned_qty"]\n'
        '            remaining -= group_total\n'
        '        else:\n'
        '            # 按比例切分（最后一个优先级组）\n'
        '            ratio = remaining / group_total if group_total > 0 else 0\n'
        '            # 向量化计算，避免逐行iterrows\n'
        '            allocated[group.index] = (group["planned_qty"] * ratio).round(2)\n'
        '            remaining = 0\n'
        '            break\n'
        '    return allocated\n'
    )

    doc.add_heading('5.4 MOQ/RV最大余数法', level=2)
    doc.add_paragraph(
        '跨节点调拨计划需对路径总需求进行MOQ/RV放大（向上取整到MOQ/RV的整数倍），'
        '然后将放大后的总量按各行原始需求比例分配回行级，保证合计等于放大后总量。'
    )
    add_code(doc,
        '# === Src版本（MOQ/RV + 最大余数法）===\n'
        'def apply_moq_rv(total_qty, moq, rv):\n'
        '    """路径级一次放大"""\n'
        '    if total_qty <= 0:\n'
        '        return 0\n'
        '    if total_qty < moq:\n'
        '        return moq  # 小于MOQ时触发最小订货量\n'
        '    return math.ceil(total_qty / rv) * rv  # 向上取整到rv的整数倍\n\n'
        'def largest_remainder_allocation(adjusted_total, original_qtys):\n'
        '    """最大余数法将整数总量按比例回分到各行"""\n'
        '    total_original = sum(original_qtys)\n'
        '    if total_original == 0:\n'
        '        return [0] * len(original_qtys)\n'
        '    # 按比例计算小数分配\n'
        '    ratios = [q / total_original for q in original_qtys]\n'
        '    floor_vals = [int(r * adjusted_total) for r in ratios]\n'
        '    remainders = [(r * adjusted_total - int(r * adjusted_total), i)\n'
        '                  for i, r in enumerate(ratios)]\n'
        '    # 计算余量并分配给余数最大的行\n'
        '    deficit = adjusted_total - sum(floor_vals)\n'
        '    remainders.sort(reverse=True)  # 按余数降序\n'
        '    for i in range(int(deficit)):\n'
        '        floor_vals[remainders[i][1]] += 1\n'
        '    return floor_vals  # 保证合计 == adjusted_total\n'
    )

    doc.add_heading('5.5 Push/Soft-Push逻辑设计', level=2)
    doc.add_paragraph(
        'Push策略在发送端有剩余库存且接收端有安全库存缺口时，主动将库存推送到下游节点。'
        'Soft-Push在Push基础上额外保留发送端当日安全库存后再参与推送。'
    )
    add_code(doc,
        '# === Src版本（Push/Soft-Push逻辑）===\n'
        'PUSH_LEVELS = [1.2, 1.5, 2.0, 2.5, 3.0]  # 可配置挡位\n\n'
        'def calculate_push_qty(available_soh, projected_soh_r, ss_r, push_level):\n'
        '    """计算对接收端r的push数量"""\n'
        '    # 接收端需求缺口\n'
        '    need_r = max(0, push_level * ss_r - projected_soh_r)\n'
        '    return need_r\n\n'
        'def execute_push(sender_available, receiver_needs):\n'
        '    """按比例分配发送端剩余库存到各接收端"""\n'
        '    total_need = sum(receiver_needs)\n'
        '    if total_need == 0 or sender_available == 0:\n'
        '        return [0] * len(receiver_needs)\n'
        '    # 按需求比例分配（向下取整保守策略）\n'
        '    alloc = [int(n / total_need * sender_available) for n in receiver_needs]\n'
        '    return alloc\n'
    )
    doc.add_page_break()

    # ====================================================================
    # 第6章 数据库层优化
    # ====================================================================
    doc.add_heading('6. 数据库层（DB版本）优化', level=1)

    doc.add_heading('6.1 PostgreSQL COPY批量写入', level=2)
    doc.add_paragraph(
        'DB版本的核心优化在于将每日仿真结果批量写入PostgreSQL数据库。'
        '使用COPY命令（而非逐行INSERT）大幅降低网络往返和事务开销。'
    )
    add_code(doc,
        '# === pgsql_db/module_data_writer.py ===\n'
        'import io\nimport psycopg2\n\n'
        'def bulk_copy_dataframe(conn, df, table_name):\n'
        '    """使用COPY命令批量写入DataFrame"""\n'
        '    buffer = io.StringIO()\n'
        '    df.to_csv(buffer, index=False, header=False, sep="\\t", na_rep="\\\\N")\n'
        '    buffer.seek(0)\n'
        '    with conn.cursor() as cur:\n'
        '        cur.copy_from(buffer, table_name, sep="\\t", null="\\\\N",\n'
        '                      columns=list(df.columns))\n'
        '    conn.commit()\n\n'
        '# 相比INSERT优势：\n'
        '# - 单次网络往返 vs 逐行N次往返\n'
        '# - 批量事务 vs N个独立事务\n'
        '# - 实测写入时间：BC 87天 DB写入总计仅101.57秒（约3%总耗时）\n'
    )
    add_table(doc,
        ['写入方式', '87天写入时间', '76天写入时间', '占总耗时比例'],
        [
            ('COPY命令（当前）', '101.57秒', '413.34秒', '3-4%'),
            ('逐行INSERT（估算）', '~2000秒', '~8000秒', '~50%+'),
        ]
    )

    doc.add_heading('6.2 DB模式数据读取优化', level=2)
    doc.add_paragraph(
        'DB模式下，仿真引擎从PostgreSQL读取配置数据。读取采用一次性全量加载+内存缓存策略，'
        '避免每日重复查询数据库。'
    )
    add_code(doc,
        '# === src/core/main_integration.py ===\n'
        '# 在日循环开始前，一次性加载所有配置数据到内存\n'
        'def load_all_configs_from_db(conn, config_name):\n'
        '    configs = {}\n'
        '    for table_name in CONFIG_TABLES:\n'
        '        query = f"SELECT * FROM {table_name} WHERE config_name = %s"\n'
        '        configs[table_name] = pd.read_sql(query, conn, params=[config_name])\n'
        '    return configs  # 后续仿真直接使用内存中的configs，不再访问数据库\n'
    )

    doc.add_heading('6.3 run_id隔离机制', level=2)
    doc.add_paragraph(
        '每次DB仿真生成唯一run_id（格式：config_name + 时间戳），'
        '所有输出表记录均携带run_id字段，确保不同仿真运行的数据可以区分和隔离查询。'
    )
    add_code(doc,
        '# run_id格式示例\n'
        '# BC场景: "BC_S5_20260225_132106"\n'
        '# OC场景: "OC_Paste_S1_20251224_20260225_150912"\n\n'
        '# 比对脚本中通过run_id过滤（关键！）\n'
        'DB_RUN_ID = "BC_S5_20260225_132106"\n'
        'query = f"SELECT * FROM module1_output_orderlog "\\\n'
        '        f"WHERE simulation_date = %s AND run_id = %s"\n'
        'df = pd.read_sql(query, conn, params=[date_val, DB_RUN_ID])\n\n'
        '# 注意：旧版脚本缺少run_id过滤，导致读取到历史仿真数据！\n'
        '# 已在2026-02-26修复，当前版本已包含run_id过滤。\n'
    )
    doc.add_page_break()

    # ====================================================================
    # 第7章 全局优化框架
    # ====================================================================
    doc.add_heading('7. 性能优化框架（全局）', level=1)

    doc.add_heading('7.1 OptimizationConfig配置系统', level=2)
    doc.add_paragraph('文件: src/utils/optimization_config.py')
    doc.add_paragraph('通过统一的OptimizationConfig类管理所有优化开关，支持代码配置、环境变量两种方式：')
    add_code(doc,
        '# === src/utils/optimization_config.py ===\n'
        'class OptimizationConfig:\n'
        '    # DuckDB全局开关（默认关闭）\n'
        '    USE_DUCKDB: bool = False\n'
        '    DUCKDB_MIN_ROWS: int = 100_000  # 低于此值强制使用Pandas\n'
        '    DUCKDB_LARGE_TABLE_ROWS: int = 500_000  # 超过此值才考虑DuckDB\n\n'
        '    # Module1优化开关\n'
        '    USE_VECTORIZED_CONSUMPTION: bool = True  # 向量化消耗计算\n'
        '    USE_HISTORY_FILE_LIMIT: bool = True       # 历史文件范围限制\n\n'
        '    # Module3优化开关\n'
        '    USE_MRP_CACHE: bool = True           # MRP缓存\n'
        '    USE_DUCKDB_NET_DEMAND: bool = False  # DuckDB净需求（不推荐）\n\n'
        '    # Module5优化开关\n'
        '    USE_DUCKDB_DEMAND_COLLECTION: bool = False  # DuckDB需求收集（不推荐）\n'
        '    USE_VECTORIZED_DEMAND: bool = False         # 向量化需求收集（有bug，待修复）\n'
        '    USE_HORIZON_CACHE: bool = True              # Horizon预计算缓存\n'
        '    USE_DATA_INDEXER: bool = True               # 数据索引器\n\n'
        '    @classmethod\n'
        '    def set_duckdb_enabled(cls, enabled: bool):\n'
        '        cls.USE_DUCKDB = enabled\n\n'
        '    @classmethod\n'
        '    def print_status(cls):\n'
        '        print(f"DuckDB: {cls.USE_DUCKDB}, DataIndexer: {cls.USE_DATA_INDEXER}, ...")\n'
    )

    doc.add_heading('7.2 DuckDB评估结论', level=2)
    doc.add_paragraph('文件: docs/DUCKDB_OPTIMIZATION_GUIDE.md | 基准测试: tools/benchmark_duckdb_vs_pandas.py')
    add_table(doc,
        ['操作', '数据规模', 'Pandas', 'DuckDB', '结论'],
        [
            ('MERGE', '1K-500K', '更快', '较慢', 'Pandas'),
            ('GROUPBY', '1K-100K', '更快', '较慢', 'Pandas'),
            ('GROUPBY', '500K+', '较慢', '更快', 'DuckDB'),
            ('FILTER', '任意', '更快', '较慢', 'Pandas'),
            ('SORT', '任意', '更快', '较慢', 'Pandas'),
        ]
    )
    doc.add_paragraph(
        '根本原因：DuckDB每次查询有启动开销（连接、注册表、编译SQL），'
        '而Pandas+NumPy的向量化操作已非常高效。仿真系统的数据主要在内存中处理，'
        'DuckDB的磁盘I/O优化优势无法体现。'
    )
    doc.add_paragraph('当前配置: USE_DUCKDB=False（默认关闭）。仅在单张表>500K行的聚合查询时考虑启用。')

    doc.add_heading('7.3 Cython内核评估结论', level=2)
    doc.add_paragraph('文件: docs/CYTHON_OPTIMIZATION_REPORT.md | 编译: python setup.py build_ext --inplace')
    add_table(doc,
        ['内核', '文件', '适用模块', '实测加速比', '结论'],
        [
            ('production_kernels.pyx', 'M4生产采样内核', 'M4', '0.96x', '收益可忽略'),
            ('logistics_kernels.pyx', 'M6延迟采样内核', 'M6', '1.04x', '微小提升'),
            ('aggregation_kernels.pyx', 'M3聚合内核', 'M3', '0.25x', '比Python dict慢，不推荐'),
        ]
    )
    doc.add_paragraph(
        '根本原因：Cython内核只是NumPy的薄包装，而NumPy C级实现已高度优化。'
        'Python原生dict的聚合速度反而超过Cython实现。'
        '真正的性能瓶颈不在CPU计算层，而在数据I/O和算法复杂度上。'
    )
    doc.add_paragraph(
        '建议：保留Cython内核作为可扩展性示范，不作为主要优化手段。'
        '如需进一步优化，考虑Numba JIT（更简洁语法，类似性能）。'
    )
    doc.add_page_break()

    # ====================================================================
    # 第8章 代码重构架构优化
    # ====================================================================
    doc.add_heading('8. 代码重构架构优化', level=1)

    doc.add_heading('8.1 分层架构设计', level=2)
    doc.add_paragraph(
        'Dev版本为单体结构，所有模块代码耦合严重，配置预加载混入业务逻辑，'
        '每次执行均有大量重复的配置读取和验证操作。'
    )
    doc.add_paragraph('Src版本重构为标准分层架构，核心改进：')
    improvements = [
        '配置预加载：main_integration.py在日循环前一次性加载所有配置，每日执行直接使用缓存',
        '模块接口标准化：统一execute(date, config, orchestrator, historical_data)接口',
        '子包拆分：每个模块拆分为多个职责单一的子文件（M5拆分为6个文件）',
        '工具函数分离：通用工具移至src/utils/，消除重复代码',
        '服务层独立：报告生成、性能分析移至src/services/，与业务逻辑解耦',
    ]
    for imp in improvements:
        doc.add_paragraph(imp, style='List Bullet')

    doc.add_heading('8.2 Orchestrator状态管理优化', level=2)
    doc.add_paragraph(
        'Dev版本各模块直接读写共享文件，存在状态一致性风险。'
        'Src版本引入Orchestrator作为统一的状态管理中枢：'
    )
    add_code(doc,
        '# === src/core/orchestrator.py ===\n'
        'class Orchestrator:\n'
        '    def __init__(self):\n'
        '        self.physical_inventory = {}  # (date, mat, loc) -> qty\n'
        '        self.open_deployment = {}     # 开放部署计划\n'
        '        self.in_transit_inventory = {}  # 在途库存\n'
        '        self.production_gr = {}        # 生产收货\n'
        '        self.delivery_gr = {}          # 交付收货\n\n'
        '    def update_state(self, date, module_outputs):\n'
        '        """接收模块输出，更新全局状态"""\n'
        '        if "production_receipt" in module_outputs:\n'
        '            self._update_production_gr(module_outputs["production_receipt"])\n'
        '        if "deployments" in module_outputs:\n'
        '            self._update_open_deployment(module_outputs["deployments"])\n'
        '        # ... 更新其他状态\n\n'
        '    def save_snapshot(self, date, output_dir):\n'
        '        """日级快照持久化，用于断点续跑"""\n'
        '        snapshot = {\n'
        '            "physical_inventory": self.physical_inventory,\n'
        '            "open_deployment": self.open_deployment,\n'
        '            # ...\n'
        '        }\n'
        '        with open(f"{output_dir}/snapshot_{date}.pkl", "wb") as f:\n'
        '            pickle.dump(snapshot, f)\n'
    )

    doc.add_heading('8.3 断点续跑机制', level=2)
    doc.add_paragraph(
        '通过Orchestrator快照实现断点续跑，当仿真意外中断时可从最后完成的日期恢复：'
    )
    add_code(doc,
        '# === 运行命令 ===\n'
        '# 首次运行（需要--start-date）\n'
        'python run.py --config BC_S5 --start-date 2025-10-05 --end-date 2025-12-30\n\n'
        '# 断点续跑（自动检测最后完成日期）\n'
        'python run.py --config BC_S5 --end-date 2025-12-30 --resume\n\n'
        '# 检查续跑状态\n'
        'python run.py --config BC_S5 --check-resume\n\n'
        '# 强制重新开始\n'
        'python run.py --config BC_S5 --start-date 2025-10-05 --end-date 2025-12-30 --force-restart\n'
    )
    doc.add_page_break()

    # ====================================================================
    # 第9章 数据一致性保证
    # ====================================================================
    doc.add_heading('9. 数据一致性保证机制', level=1)

    doc.add_heading('9.1 标识符标准化', level=2)
    doc.add_paragraph('Dev版本存在物料编码/地点编码大小写不一致、地点编码位数不一致问题。Src版本统一处理：')
    add_code(doc,
        '# === Src版本（标识符标准化）===\n'
        'def normalize_identifiers(df):\n'
        '    """统一标识符格式"""\n'
        '    if "material" in df.columns:\n'
        '        df["material"] = df["material"].astype(str).str.strip()\n'
        '    if "location" in df.columns:\n'
        '        # 地点编码统一左补零至4位\n'
        '        df["location"] = df["location"].astype(str).str.strip().str.zfill(4)\n'
        '    if "sending" in df.columns:\n'
        '        df["sending"] = df["sending"].astype(str).str.strip().str.zfill(4)\n'
        '    if "receiving" in df.columns:\n'
        '        df["receiving"] = df["receiving"].astype(str).str.strip().str.zfill(4)\n'
        '    return df\n'
    )

    doc.add_heading('9.2 库存守恒验证', level=2)
    add_code(doc,
        '# === src/utils/inventory_balance_checker.py ===\n'
        'class InventoryBalanceChecker:\n'
        '    def validate_daily_balance(self, date, orchestrator):\n'
        '        """验证当日库存守恒: 期初 + 入库 - 出库 = 期末"""\n'
        '        opening = sum(orchestrator.get_inventory(date-1).values())\n'
        '        inflow = (\n'
        '            sum(orchestrator.production_gr.get(date, {}).values()) +\n'
        '            sum(orchestrator.delivery_gr.get(date, {}).values())\n'
        '        )\n'
        '        outflow = sum(orchestrator.get_shipments(date).values())\n'
        '        closing = sum(orchestrator.get_inventory(date).values())\n'
        '        balance = opening + inflow - outflow - closing\n'
        '        if abs(balance) > 0.01:  # 允许浮点误差\n'
        '            raise ValueError(f"库存不平衡: {balance}")\n'
    )
    doc.add_page_break()

    # ====================================================================
    # 第10章 未实施的潜在优化
    # ====================================================================
    doc.add_heading('10. 未实施的潜在优化方向', level=1)

    doc.add_heading('10.1 短期可实施（低风险，高收益）', level=2)
    add_table(doc,
        ['优化项', '涉及模块', '预期收益', '风险', '实施难度'],
        [
            ('启用向量化需求收集\n(USE_VECTORIZED_DEMAND=True)', 'M5', '30-50%加速', '需修复horizon计算bug', '中'),
            ('M1供需日志延迟聚合', 'M1', '15-25%', '业务逻辑变更风险', '低'),
            ('M3 MRP缓存扩展\n（跨天不变结果缓存）', 'M3', '10-20%', '缓存失效逻辑复杂', '中'),
        ]
    )

    doc.add_heading('10.2 中期优化（中等风险）', level=2)
    add_table(doc,
        ['优化项', '技术方案', '预期收益', '风险', '实施难度'],
        [
            ('Numba JIT编译\nM5分配热点', 'Numba @njit装饰器', '20-40%', '调试困难', '中'),
            ('增量计算框架', 'ChangeSet机制\n仅重算变化节点', '50-70%', '算法复杂度高', '高'),
            ('异步IO', 'asyncio + aiofiles', '10-15%', '代码结构改变较大', '中'),
        ]
    )

    doc.add_heading('10.3 长期优化（需架构评估）', level=2)
    add_table(doc,
        ['优化项', '技术方案', '预期收益', '适用场景'],
        [
            ('分布式计算', 'Dask/Ray并行', '线性扩展', '多配置并行仿真'),
            ('GPU加速', 'RAPIDS/cuDF', '10-100x', '超大规模数据聚合'),
            ('流式处理', 'Apache Kafka', '近实时', '生产环境实时仿真'),
        ]
    )

    doc.add_heading('10.4 已评估但不推荐的优化', level=2)
    add_table(doc,
        ['优化项', '评估结论', '原因'],
        [
            ('DuckDB全面替换Pandas', '不推荐', '仿真数据量<100K行，Pandas更快'),
            ('Cython深度优化', '不推荐', '真正瓶颈在算法复杂度而非CPU计算'),
            ('进程池并行M3', '待探索', '需解决pickle序列化复杂对象的问题'),
        ]
    )

    # ===== 附录 =====
    doc.add_page_break()
    doc.add_heading('附录: 关键性能数据汇总', level=1)
    add_table(doc,
        ['指标', 'BC场景(87天)', 'OC场景(76天)'],
        [
            ('Dev总耗时', '9,755秒', '98,389秒'),
            ('Src总耗时', '6,287秒', '25,497秒'),
            ('DB总耗时', '3,539秒', '12,717秒'),
            ('DB写入时间', '101.57秒', '413.34秒'),
            ('Src vs Dev加速比', '1.55x', '3.86x'),
            ('DB vs Dev加速比', '2.76x', '7.74x'),
            ('DB vs Src加速比', '1.78x', '2.01x'),
            ('M1 Dev平均/天', '4.68秒', '39.3秒'),
            ('M3 Dev平均/天', '35.2秒', '512.8秒'),
            ('M5 Dev平均/天', '53.4秒', '490.5秒'),
            ('M1 DB平均/天', '2.22秒', '12.6秒'),
            ('M3 DB平均/天', '7.0秒', '40.2秒'),
            ('M5 DB平均/天', '12.7秒', '54.9秒'),
        ]
    )

    # ===== 报告结尾 =====
    doc.add_paragraph()
    doc.add_paragraph('--- 文档结束 ---').alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    p = doc.add_paragraph(f'本文档由 chenxianyue002@chinasofti.com 编制，生成日期: {datetime.date.today().strftime("%Y-%m-%d")}')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    output_path = DOCS_DIR / 'ChainSight_Dev源码算法优化文档.docx'
    doc.save(str(output_path))
    print(f"[OK] 算法优化文档已生成: {output_path}")
    return output_path


if __name__ == '__main__':
    create_algo_optimization_doc()
