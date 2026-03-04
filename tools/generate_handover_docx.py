#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChainSight 项目交接文档生成器
生成非常详细的Word交接文档，涵盖项目全貌、架构设计、模块说明、
数据一致性验证结果、性能数据、运维指南、常见问题等。
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from pathlib import Path
import datetime

ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"


def set_cell_shading(cell, color_hex):
    """设置单元格底色"""
    shading = cell._element.get_or_add_tcPr()
    shading_elem = shading.makeelement(qn('w:shd'), {
        qn('w:val'): 'clear',
        qn('w:color'): 'auto',
        qn('w:fill'): color_hex,
    })
    shading.append(shading_elem)


def set_table_header_style(table, header_color='2F5496'):
    """设置表格头部样式"""
    for cell in table.rows[0].cells:
        set_cell_shading(cell, header_color)
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.bold = True
                run.font.size = Pt(10)


def add_styled_table(doc, headers, data, col_widths=None, header_color='2F5496'):
    """添加带样式的表格"""
    table = doc.add_table(rows=1 + len(data), cols=len(headers))
    table.style = 'Table Grid'
    # 表头
    for j, h in enumerate(headers):
        table.rows[0].cells[j].text = h
    set_table_header_style(table, header_color)
    # 数据行
    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            table.rows[i + 1].cells[j].text = str(val)
    return table


def create_handover_document():
    doc = Document()

    # ===== 全局样式 =====
    style = doc.styles['Normal']
    style.font.name = '微软雅黑'
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    # ===== 封面 =====
    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_heading('ChainSight 供应链规划仿真系统', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_heading('项目交接文档', level=1)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    doc.add_paragraph()

    info_lines = [
        f'文档版本: v1.0',
        f'编制日期: {datetime.date.today().strftime("%Y年%m月%d日")}',
        '项目名称: ChainSight Supply Chain Simulation',
        '编制人: chenxianyue002@chinasofti.com',
        '审核人: chenxianyue002@chinasofti.com',
        '密级: 内部',
    ]
    for line in info_lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line).font.size = Pt(12)

    doc.add_page_break()

    # ===== 修订记录 =====
    doc.add_heading('修订记录', level=1)
    add_styled_table(doc,
        ['版本', '日期', '修订人', '修订内容'],
        [
            ('v1.0', datetime.date.today().strftime('%Y-%m-%d'), 'chenxianyue002@chinasofti.com', '初始版本，包含完整项目交接信息'),
        ]
    )
    doc.add_page_break()

    # ===== 目录 =====
    doc.add_heading('目录', level=1)
    toc = [
        '1. 项目概述',
        '  1.1 项目背景与目标',
        '  1.2 项目范围',
        '  1.3 技术栈',
        '  1.4 项目时间线',
        '2. 系统架构',
        '  2.1 整体分层架构',
        '  2.2 模块执行流程',
        '  2.3 目录结构',
        '  2.4 依赖关系',
        '3. 核心业务模块详解',
        '  3.1 Module1 - 需求规划',
        '  3.2 Module3 - MRP计划',
        '  3.3 Module4 - 生产计划',
        '  3.4 Module5 - 部署规划',
        '  3.5 Module6 - 物流执行',
        '  3.6 Orchestrator - 状态管理',
        '4. 数据库层设计',
        '  4.1 数据库架构',
        '  4.2 表命名规则',
        '  4.3 关键表结构',
        '5. 三版本数据一致性验证',
        '  5.1 验证方法论',
        '  5.2 BC场景验证结果',
        '  5.3 OC场景验证结果',
        '  5.4 综合验证结论',
        '6. 性能优化成果',
        '  6.1 BC场景性能数据',
        '  6.2 OC场景性能数据',
        '  6.3 优化技术汇总',
        '  6.4 模块级瓶颈分析',
        '7. 运维与部署指南',
        '  7.1 环境搭建',
        '  7.2 运行方式',
        '  7.3 断点续跑',
        '  7.4 数据库管理',
        '8. 关键文件清单',
        '9. 已知问题与注意事项',
        '10. 后续优化建议',
        '附录A: 17份测试报告索引',
        '附录B: 关键脚本参考',
        '附录C: 数据库连接信息',
    ]
    for item in toc:
        doc.add_paragraph(item)
    doc.add_page_break()

    # ====================================================================
    # 第1章 项目概述
    # ====================================================================
    doc.add_heading('1. 项目概述', level=1)

    doc.add_heading('1.1 项目背景与目标', level=2)
    doc.add_paragraph(
        'ChainSight 是一套生产级的供应链规划仿真系统，用于模拟从需求预测、'
        '生产排程、库存部署到物流执行的完整供应链业务流程。系统采用日度滚动仿真模式，'
        '在每个仿真日依次执行五大业务模块（M1→M4→M5→M6→M3），并由 Orchestrator 维护全局库存状态。'
    )
    doc.add_paragraph(
        '本项目的核心目标是对原始开发版本（ChainSight_Dev）进行代码重构和性能优化，'
        '生成两个新版本（Src 本地重构版 和 DB 数据库版），并通过严格的三版本数据一致性验证，'
        '确保重构后业务逻辑100%等价，同时实现显著的性能提升。'
    )

    doc.add_heading('1.2 项目范围', level=2)
    add_styled_table(doc,
        ['维度', '内容'],
        [
            ('代码重构', '将单体代码拆分为分层架构（CLI→Core→Modules→Utils→Services）'),
            ('性能优化', '向量化计算、缓存机制、索引预构建、并行处理、DB批量写入'),
            ('数据库集成', 'PostgreSQL数据持久化，支持--use-db模式'),
            ('数据验证', 'BC场景（87天）和OC场景（76天）的三版本完整数据比对'),
            ('报告输出', '17份10天间隔测试报告（BC 9份 + OC 8份）'),
            ('技术探索', 'Cython内核优化、DuckDB数据处理（均已评估但收益有限）'),
        ]
    )

    doc.add_heading('1.3 技术栈', level=2)
    add_styled_table(doc,
        ['组件', '技术', '版本', '用途'],
        [
            ('编程语言', 'Python', '3.12.9', '主要开发语言'),
            ('数据处理', 'pandas', '2.2.3', '核心数据分析和转换'),
            ('数值计算', 'numpy', '2.0+', '数组和矩阵运算'),
            ('数据持久化', 'PostgreSQL', '14+', '生产环境数据存储'),
            ('高性能数据', 'DuckDB', '1.1.3', '可选的分析型数据处理（默认关闭）'),
            ('文件读写', 'openpyxl', '3.1.5', 'Excel配置和输出'),
            ('性能优化', 'Cython', '3.0+', '可选的计算内核（收益有限）'),
        ]
    )

    doc.add_heading('1.4 项目时间线', level=2)
    add_styled_table(doc,
        ['阶段', '时间', '里程碑'],
        [
            ('架构重构', '2025-12 ~ 2026-01', '完成分层架构重构，Src版本5倍加速'),
            ('Module1优化', '2025-12-01', '历史文件读取限制，33.2x加速（365天场景）'),
            ('Supply Demand Log优化', '2025-12-01', '输出范围限制90天，数据量减少75%'),
            ('Cython探索', '2026-01-28', '3个Cython内核编译成功，但加速效果有限（0.96x-1.04x）'),
            ('DuckDB评估', '2026-01-21', '基准测试显示Pandas在<100K行场景更快，DuckDB默认关闭'),
            ('DB版本开发', '2026-01-21', '完成PostgreSQL集成，DB版2.76x加速（BC），7.74x加速（OC）'),
            ('BC全量验证', '2026-02-25', 'BC 87天三版本比对16/16表全PASS'),
            ('OC全量验证', '2026-02-26', 'OC 76天三版本比对15/15表全PASS（含src_vs_db补充）'),
            ('报告生成', '2026-02-26', '17份10天间隔测试报告生成完毕'),
            ('交接文档', datetime.date.today().strftime('%Y-%m-%d'), '项目交接文档编制'),
        ]
    )
    doc.add_page_break()

    # ====================================================================
    # 第2章 系统架构
    # ====================================================================
    doc.add_heading('2. 系统架构', level=1)

    doc.add_heading('2.1 整体分层架构', level=2)
    doc.add_paragraph(
        '系统采用标准分层架构设计，共分为以下层次：'
    )
    layers = [
        ('CLI层', 'run.py（项目根目录）', '解析命令行参数，调度Core层，无业务逻辑'),
        ('Core层', 'src/core/', 'Orchestrator状态管理、main_integration日循环编排、run.py参数校验'),
        ('Modules层', 'src/modules/', '5个独立业务模块（M1/M3/M4/M5/M6），各含子包'),
        ('Utils层', 'src/utils/', '配置校验、日志、数据验证、库存平衡检查、时间管理等通用工具'),
        ('Services层', 'src/services/', '汇总报告生成、性能分析器'),
        ('Database层', 'pgsql_db/', 'PostgreSQL连接、初始化、表定义、批量写入、DuckDB集成'),
        ('Tools层', 'tools/', '独立的辅助脚本（对比、报告生成、基准测试等）'),
    ]
    add_styled_table(doc,
        ['层次', '位置', '职责'],
        layers
    )

    doc.add_heading('2.2 模块执行流程', level=2)
    doc.add_paragraph('每日仿真按以下固定顺序执行5个业务模块，由Orchestrator在模块间同步全局状态：')
    flow_text = (
        'FOR each_day in [start_date, end_date]:\n'
        '  1. M1 (Demand Planning) - 需求预测、订单生成、发货日志\n'
        '     -> Orchestrator.update_state()\n'
        '  2. M4 (Production Planning) - 生产计划、产能分配、生产收货\n'
        '     -> Orchestrator.update_state()\n'
        '  3. M5 (Deployment Planning) - 部署计划、库存推送、库存预测\n'
        '     -> Orchestrator.update_state()\n'
        '  4. M6 (Logistics Execution) - 交付计划、车辆装载、运输日志\n'
        '     -> Orchestrator.update_state()\n'
        '  5. M3 (MRP Planning) - 净需求计算、MRP仿真、建议采购单\n'
        '     -> Orchestrator.update_state()\n'
        '  6. Generate Daily Summary & Snapshot'
    )
    p = doc.add_paragraph()
    run = p.add_run(flow_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)

    doc.add_heading('2.3 目录结构', level=2)
    dir_text = (
        'chainsight/\n'
        '├── run.py                    # 主入口CLI\n'
        '├── run.ps1                   # PowerShell启动脚本\n'
        '├── setup.py                  # Cython编译配置\n'
        '├── requirements.txt          # 依赖声明\n'
        '├── src/                      # 核心源代码包\n'
        '│   ├── core/                 # 编排引擎\n'
        '│   │   ├── main_integration.py  # 日循环调度\n'
        '│   │   ├── orchestrator.py      # 全局状态管理\n'
        '│   │   ├── parallel_executor.py # 并行执行框架\n'
        '│   │   └── run.py               # CLI解析\n'
        '│   ├── modules/              # 业务模块\n'
        '│   │   ├── module1.py ~ module6.py\n'
        '│   │   ├── demand_planning/      # M1子包\n'
        '│   │   ├── mrp_planning/         # M3子包\n'
        '│   │   ├── production_planning/  # M4子包\n'
        '│   │   ├── deployment_planning/  # M5子包\n'
        '│   │   └── logistics_execution/  # M6子包\n'
        '│   ├── utils/                # 通用工具\n'
        '│   ├── services/             # 业务服务\n'
        '│   └── cython_kernels/       # Cython优化内核\n'
        '├── pgsql_db/                 # 数据库层（18个文件）\n'
        '├── tools/                    # 辅助工具脚本\n'
        '├── tests/                    # 测试\n'
        '├── test_files/               # 测试数据与对比工具\n'
        '├── config/                   # 配置文件\n'
        '├── docs/                     # 设计文档\n'
        '├── ChainSight_Dev/           # 原始Dev版本（基线对比用）\n'
        '└── outputs/                  # 运行输出（自动生成）\n'
    )
    p = doc.add_paragraph()
    run = p.add_run(dir_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)

    doc.add_heading('2.4 依赖关系', level=2)
    doc.add_paragraph('推荐的导入方向：CLI → Core → Modules/Utils/Services → Cython Kernels')
    doc.add_paragraph('Core → Database Layer（pgsql_db）')
    doc.add_paragraph('禁止的依赖：Modules之间不直接依赖（通过Orchestrator交互），不允许循环依赖。')

    doc.add_page_break()

    # ====================================================================
    # 第3章 核心业务模块
    # ====================================================================
    doc.add_heading('3. 核心业务模块详解', level=1)

    doc.add_heading('3.1 Module1 - 需求规划 (Demand Planning)', level=2)
    doc.add_paragraph('入口文件: src/modules/module1.py')
    doc.add_paragraph('子包: src/modules/demand_planning/ (forecast_processor, order_generator, shipment_tracker)')
    doc.add_paragraph('核心功能：')
    m1_funcs = [
        'DPS拆分：将周度需求预测拆分为日度粒度',
        'AO订单消耗：按advance_days提前生成AO订单，消耗未来预测量',
        'Normal订单消耗：当日到货的常规订单消耗',
        '供需日志(SupplyDemandLog)：生成90天窗口的供需平衡日志',
        '发货日志(ShipmentLog)：记录客户发货',
        '削减日志(CutLog)：记录因库存不足被削减的订单',
    ]
    for f in m1_funcs:
        doc.add_paragraph(f, style='List Bullet')
    doc.add_paragraph('输出表（每日）：OrderLog, ShipmentLog, CutLog, SupplyDemandLog, Summary')

    doc.add_heading('3.2 Module3 - MRP计划 (MRP Planning)', level=2)
    doc.add_paragraph('入口文件: src/modules/module3.py')
    doc.add_paragraph('子包: src/modules/mrp_planning/ (net_demand_calculator, mrp_simulator)')
    doc.add_paragraph('核心功能：')
    m3_funcs = [
        '网络层级识别：BFS自顶向下分配层级（assign_location_layers）',
        '提前期计算：DC用PDT+GR，Plant用max(MCT,PDT+GR)+PTF+LSK-1',
        '可用量口径：期初库存+在途+当日GR+未来生产-当日发货-开放调拨出库',
        '需求缺口消耗：AO→Forecast→SafetyStock依次消耗',
        '缺口上传与MOQ/RV放大：自下而上逐层传递，最大余数法回分',
        '逐层并行：线程池并行计算同层节点，失败回退串行',
    ]
    for f in m3_funcs:
        doc.add_paragraph(f, style='List Bullet')
    doc.add_paragraph('输出表（每日）：NetDemand')

    doc.add_heading('3.3 Module4 - 生产计划 (Production Planning)', level=2)
    doc.add_paragraph('入口文件: src/modules/module4.py')
    doc.add_paragraph('子包: src/modules/production_planning/ (plan_builder, capacity_allocator)')
    doc.add_paragraph('核心功能：APS调度、产能约束分配、换产矩阵、生产收货GR')
    doc.add_paragraph('输出表（每日）：ProductionPlan, CapacityExceed, ChangeoverLog')

    doc.add_heading('3.4 Module5 - 部署规划 (Deployment Planning)', level=2)
    doc.add_paragraph('入口文件: src/modules/module5.py')
    doc.add_paragraph('子包: src/modules/deployment_planning/ (allocation_optimizer, inventory_manager, push_planner)')
    doc.add_paragraph('核心功能：')
    m5_funcs = [
        '库存口径：projected_soh（含在途/未来）和dynamic_soh（当日可分配）',
        '需求收集：窗口[sim_date, horizon_end]内的SupplyDemandLog/SafetyStock/OrderLog',
        '优先级分配：按demand_priority升序，最后优先级组按比例切分',
        'MOQ/RV放大：跨节点一次放大+最大余数法回分（自循环不应用）',
        '接收仓容配额(ReceivingSpace)：跨节点到货限额，超额记入UnfulfilledLog',
        'Push/Soft-Push：发送端剩余库存按挡位下推到接收端',
    ]
    for f in m5_funcs:
        doc.add_paragraph(f, style='List Bullet')
    doc.add_paragraph('输出表（每日）：DeploymentPlan, UnfulfilledLog, StockOnHandLog, Validation')

    doc.add_heading('3.5 Module6 - 物流执行 (Logistics Execution)', level=2)
    doc.add_paragraph('入口文件: src/modules/module6.py')
    doc.add_paragraph('子包: src/modules/logistics_execution/ (vehicle_packer, delivery_executor)')
    doc.add_paragraph('核心功能：发车规则判断、车辆装载优化、在途跟踪、交付确认')
    doc.add_paragraph('输出表（每日）：DeliveryPlan, VehicleLog, TruckUsageLog')

    doc.add_heading('3.6 Orchestrator - 全局状态管理', level=2)
    doc.add_paragraph('文件: src/core/orchestrator.py')
    doc.add_paragraph('维护的全局状态：')
    orch_states = [
        'physical_inventory：物理库存（按日期/物料/地点）',
        'open_deployment：开放部署计划',
        'in_transit_inventory：在途库存',
        'production_gr：生产收货',
        'delivery_gr：交付收货',
        'space_capacity：仓容限额',
    ]
    for s in orch_states:
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph('关键方法：update_state()（接收模块更新）、get_inventory()（查询库存）、validate_consistency()（库存守恒验证）')

    doc.add_page_break()

    # ====================================================================
    # 第4章 数据库层
    # ====================================================================
    doc.add_heading('4. 数据库层设计', level=1)

    doc.add_heading('4.1 数据库架构', level=2)
    doc.add_paragraph('数据库层位于 pgsql_db/ 目录，包含18个Python文件。核心组件：')
    add_styled_table(doc,
        ['组件', '文件', '功能'],
        [
            ('连接管理', 'db_connection.py', '连接池管理，自动重连'),
            ('数据库初始化', 'db_initializer.py', '自动建库建表，检测缺失表并创建'),
            ('表结构定义', 'table_schemas.py', '统一的表DDL定义'),
            ('表名映射', 'table_mapping.py', 'Excel Sheet名→数据库表名的映射关系'),
            ('数据写入', 'module_data_writer.py', '批量COPY写入，单事务提交'),
            ('Excel导入', 'excel_importer.py', '从Excel配置文件导入到数据库'),
            ('DuckDB处理', 'duckdb_processor.py', '可选的DuckDB分析查询'),
        ]
    )

    doc.add_heading('4.2 表命名规则', level=2)
    doc.add_paragraph('配置表采用"同结构同表"规则，通过config_name字段区分不同配置：')
    add_styled_table(doc,
        ['表类型', '命名格式', '示例', '说明'],
        [
            ('配置表', 'cfg_*', 'cfg_m1_demandforecast', '所有配置共用，config_name区分'),
            ('M1输出', 'module1_output_*', 'module1_output_orderlog', '模块输出表'),
            ('M3输出', 'module3_output_*', 'module3_output_netdemand', '模块输出表'),
            ('M4输出', 'module4_output_*', 'module4_output_productionplan', '模块输出表'),
            ('M5输出', 'module5_output_*', 'module5_output_deploymentplan', '模块输出表'),
            ('M6输出', 'module6_output_*', 'module6_output_deliveryplan', '模块输出表'),
            ('Orchestrator', 'orchestrator_*', 'orchestrator_daily_logs', '编排器日志'),
            ('汇总报告', 'summary_*', 'summary_historical_inventory_record', '汇总数据'),
        ]
    )

    doc.add_heading('4.3 关键注意事项', level=2)
    doc.add_paragraph(
        '重要警告：pgsql_db/module_data_writer.py 中有 truncate_output_tables() 逻辑，'
        '每次新仿真会清空所有输出表。BC和OC共享同一组表，启动新仿真前必须确认已有数据已备份或比对完毕。',
    )

    doc.add_page_break()

    # ====================================================================
    # 第5章 三版本数据一致性验证
    # ====================================================================
    doc.add_heading('5. 三版本数据一致性验证', level=1)

    doc.add_heading('5.1 验证方法论', level=2)
    doc.add_paragraph(
        '对每个场景（BC/OC），在三个版本（Dev/Src/DB）之间进行逐天、逐表、逐行的数据一致性比对。'
        '比对维度包括：行数一致、列名一致、数值精确匹配。'
    )
    add_styled_table(doc,
        ['版本', '说明', '代码位置', '运行方式'],
        [
            ('Dev', '原始开发版本（基准）', 'ChainSight_Dev/', '直接运行module*.py'),
            ('Src', '重构本地版本', 'src/', 'python run.py --config ...'),
            ('DB', '重构数据库版本', 'src/ + pgsql_db/', 'python run.py --config ... --use-db'),
        ]
    )

    doc.add_heading('5.2 BC场景验证结果', level=2)
    doc.add_paragraph('场景: BC_S5, 87天 (2025-10-05 ~ 2025-12-30)')
    doc.add_paragraph('比对表数: 16个')
    doc.add_paragraph()

    bc_tables = [
        ('Module1', 'OrderLog (订单日志)', '424,848', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'ShipmentLog (发货日志)', '24,252', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'CutLog (削减日志)', '24,252', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'SupplyDemandLog (供需日志)', '2,084,295', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'Summary (每日汇总)', '87', 'PASS', 'PASS', 'PASS'),
        ('Module3', 'NetDemand (净需求)', '30,702', 'PASS', 'PASS', 'PASS'),
        ('Module4', 'ProductionPlan (生产计划)', '379', 'PASS', 'PASS', 'PASS'),
        ('Module4', 'CapacityExceed (超容量)', '15', 'PASS', 'PASS', 'PASS'),
        ('Module4', 'ChangeoverLog (换型日志)', '146', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'DeploymentPlan (部署计划)', '1,149,959', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'UnfulfilledLog (未满足)', '513,212', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'StockOnHandLog (库存)', '72,471', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'Validation (验证)', '40,368', 'PASS', 'PASS', 'PASS'),
        ('Module6', 'DeliveryPlan (交付计划)', '43,055', 'PASS', 'PASS', 'PASS'),
        ('Module6', 'VehicleLog (车辆日志)', '258', 'PASS', 'PASS', 'PASS'),
        ('Module6', 'TruckUsageLog (卡车使用)', '192', 'PASS', 'PASS', 'PASS'),
    ]
    add_styled_table(doc,
        ['模块', '表名', '总行数(87天)', 'Dev vs Src', 'Dev vs DB', 'Src vs DB'],
        bc_tables
    )
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run('BC验证结论: 16/16表全部PASS，三版本数据100%一致')
    run.bold = True

    doc.add_heading('5.3 OC场景验证结果', level=2)
    doc.add_paragraph('场景: OC_Paste_S1_20251224, 76天 (2025-12-15 ~ 2026-02-28)')
    doc.add_paragraph('比对表数: 15个')
    doc.add_paragraph()

    oc_tables = [
        ('Module1', 'OrderLog', '1,300,506', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'ShipmentLog', '157,433', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'CutLog', '157,433', 'PASS', 'PASS', 'PASS'),
        ('Module1', 'Summary', '76', 'PASS', 'PASS', 'PASS'),
        ('Module3', 'NetDemand', '93,693', 'PASS', 'PASS', 'PASS'),
        ('Module4', 'ProductionPlan', '685', 'PASS', 'PASS', 'PASS'),
        ('Module4', 'CapacityExceed', '3,018', 'PASS', 'PASS', 'PASS'),
        ('Module4', 'ChangeoverLog', '515', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'DeploymentPlan', '3,336,330', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'UnfulfilledLog', '527,164', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'StockOnHandLog', '387,730', 'PASS', 'PASS', 'PASS'),
        ('Module5', 'Validation', '623,200', 'PASS', 'PASS', 'PASS'),
        ('Module6', 'DeliveryPlan', '71,145', 'PASS', 'PASS', 'PASS'),
        ('Module6', 'VehicleLog', '823', 'PASS', 'PASS', 'PASS'),
        ('Module6', 'TruckUsageLog', '803', 'PASS', 'PASS', 'PASS'),
    ]
    add_styled_table(doc,
        ['模块', '表名', '总行数(76天)', 'Dev vs Src', 'Dev vs DB', 'Src vs DB'],
        oc_tables
    )
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run('OC验证结论: 15/15表全部PASS，三版本数据100%一致')
    run.bold = True

    doc.add_heading('5.4 综合验证结论', level=2)
    add_styled_table(doc,
        ['验证维度', 'BC场景', 'OC场景'],
        [
            ('仿真天数', '87天', '76天'),
            ('比对表数', '16个', '15个'),
            ('Dev vs Src', '16/16 PASS', '15/15 PASS'),
            ('Dev vs DB', '16/16 PASS', '15/15 PASS'),
            ('Src vs DB', '16/16 PASS', '15/15 PASS'),
            ('数据一致性', '100%', '100%'),
        ]
    )
    p = doc.add_paragraph()
    run = p.add_run('总结：两个场景共31个表、3对比较组（共93次比对），全部PASS，零差异。')
    run.bold = True

    doc.add_page_break()

    # ====================================================================
    # 第6章 性能优化成果
    # ====================================================================
    doc.add_heading('6. 性能优化成果', level=1)

    doc.add_heading('6.1 BC场景性能数据 (87天)', level=2)
    add_styled_table(doc,
        ['版本', '总耗时', '平均每天', '加速比(vs Dev)'],
        [
            ('Dev', '9,755秒 (162.6分钟)', '112.1秒/天', '1.00x (基线)'),
            ('Src', '6,287秒 (104.8分钟)', '72.3秒/天', '1.55x'),
            ('DB', '3,539秒 (59.0分钟)', '40.7秒/天', '2.76x'),
        ]
    )
    doc.add_paragraph()
    doc.add_paragraph('BC模块级耗时（秒/天均值）：')
    add_styled_table(doc,
        ['模块', 'Dev', 'Src', 'DB', 'DB加速(vs Dev)'],
        [
            ('M1 订单生成', '4.68', '5.27', '2.22', '2.11x'),
            ('M5 部署规划', '53.4', '28.5', '12.7', '4.20x'),
            ('M3 净需求计算', '35.2', '15.5', '7.0', '5.03x'),
        ]
    )

    doc.add_heading('6.2 OC场景性能数据 (76天)', level=2)
    add_styled_table(doc,
        ['版本', '总耗时', '平均每天', '加速比(vs Dev)'],
        [
            ('Dev', '98,389秒 (27.3小时)', '1,294.6秒/天', '1.00x (基线)'),
            ('Src', '25,497秒 (7.1小时)', '335.5秒/天', '3.86x'),
            ('DB', '12,717秒 (3.5小时)', '167.3秒/天', '7.74x'),
        ]
    )
    doc.add_paragraph()
    doc.add_paragraph('OC模块级耗时（秒/天均值）：')
    add_styled_table(doc,
        ['模块', 'Dev', 'Src', 'DB', 'DB加速(vs Dev)'],
        [
            ('M1 订单生成', '39.3', '31.5', '12.6', '3.12x'),
            ('M5 部署规划', '490.5', '138.3', '54.9', '8.93x'),
            ('M3 净需求计算', '512.8', '89.7', '40.2', '12.76x'),
        ]
    )

    doc.add_heading('6.3 优化技术汇总', level=2)
    add_styled_table(doc,
        ['优化项', '技术', '适用模块', '加速效果', '状态'],
        [
            ('历史文件读取限制', '时间窗口过滤(max_advance_days+1)', 'M1', '33.2x(365天)', '已上线'),
            ('Supply Demand Log限制', '90天输出窗口', 'M1', '数据量减75%', '已上线'),
            ('向量化消耗计算', 'pandas向量化替代循环', 'M1', '150x', '已上线'),
            ('PTF/LSK缓存', '预构建字典缓存', 'M3/M5', '15-20x', '已上线'),
            ('DataIndexer索引', 'HashMap预索引', 'M5', 'O(n*m)→O(1)', '已上线'),
            ('LeadTime缓存', '基础参数预计算', 'M5', '10-15x', '已上线'),
            ('ThreadPoolExecutor并行', '层内节点并行', 'M3', '2x', '已上线'),
            ('DB COPY批量写入', 'PostgreSQL COPY命令', 'DB层', '写入减85%', '已上线'),
            ('Cython内核', '编译型加速', 'M4/M6', '0.96x-1.04x', '已部署但收益有限'),
            ('DuckDB', '列式存储引擎', '全局', 'Pandas更快(<100K行)', '默认关闭'),
        ]
    )

    doc.add_heading('6.4 模块级瓶颈分析', level=2)
    doc.add_paragraph('当前性能瓶颈集中在M5和M3：')
    add_styled_table(doc,
        ['模块', '瓶颈点', '耗时占比', '优化潜力'],
        [
            ('M5 需求收集', '窗口内多数据源汇聚和索引查找', '~15%', '50-70%（启用向量化需求收集）'),
            ('M5 分配计算', '优先级分配和MOQ/RV放大', '~25%', '30-50%'),
            ('M3 MRP仿真', '逐层逐节点串行计算', '~35%', '20-30%（进程池/增量计算）'),
            ('M1 供需日志', '每天生成大量日志记录', '~5%', '15-25%（批量/延迟聚合）'),
        ]
    )

    doc.add_page_break()

    # ====================================================================
    # 第7章 运维与部署指南
    # ====================================================================
    doc.add_heading('7. 运维与部署指南', level=1)

    doc.add_heading('7.1 环境搭建', level=2)
    doc.add_paragraph('步骤1: 创建虚拟环境')
    p = doc.add_paragraph()
    run = p.add_run('python -m venv .venv312\n.venv312\\Scripts\\activate\npip install -r requirements.txt')
    run.font.name = 'Consolas'
    run.font.size = Pt(9)

    doc.add_paragraph('步骤2: 可选 - 编译Cython扩展')
    p = doc.add_paragraph()
    run = p.add_run('python setup.py build_ext --inplace')
    run.font.name = 'Consolas'
    run.font.size = Pt(9)

    doc.add_paragraph('步骤3: 可选 - 配置PostgreSQL')
    doc.add_paragraph('安装PostgreSQL 14+，创建数据库test_db，用户postgres/密码123456。DB模式会自动初始化表结构。')

    doc.add_heading('7.2 运行方式', level=2)
    doc.add_paragraph('本地文件模式：')
    p = doc.add_paragraph()
    run = p.add_run('python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-06 --end-date 2025-10-10')
    run.font.name = 'Consolas'
    run.font.size = Pt(9)

    doc.add_paragraph()
    doc.add_paragraph('数据库模式：')
    p = doc.add_paragraph()
    run = p.add_run('python run.py --config BC_S5 --start-date 2025-10-06 --end-date 2025-10-10 --use-db')
    run.font.name = 'Consolas'
    run.font.size = Pt(9)

    doc.add_heading('7.3 断点续跑', level=2)
    doc.add_paragraph('系统支持断点续跑，使用--resume参数从上次中断处继续：')
    p = doc.add_paragraph()
    run = p.add_run('python run.py --config BC_S5 --end-date 2025-12-30 --resume')
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    doc.add_paragraph('使用--check-resume检查续跑状态，使用--force-restart强制重新开始。')

    doc.add_heading('7.4 数据库管理', level=2)
    doc.add_paragraph('数据库连接参数：host=localhost, port=5432, dbname=test_db, user=postgres, password=123456')
    doc.add_paragraph(
        '重要：每次新仿真会truncate所有输出表（BC和OC共享表）。'
        '如需保留已有数据，请在启动新仿真前备份。'
    )

    doc.add_page_break()

    # ====================================================================
    # 第8章 关键文件清单
    # ====================================================================
    doc.add_heading('8. 关键文件清单', level=1)

    add_styled_table(doc,
        ['文件', '用途', '重要程度'],
        [
            ('run.py', '主入口CLI', '高'),
            ('src/core/main_integration.py', '日循环编排器', '高'),
            ('src/core/orchestrator.py', '全局状态管理', '高'),
            ('src/modules/module1.py', 'M1需求规划', '高'),
            ('src/modules/module3.py', 'M3 MRP计划', '高'),
            ('src/modules/module5.py', 'M5部署规划', '高'),
            ('pgsql_db/module_data_writer.py', 'DB批量写入（含truncate风险）', '高'),
            ('pgsql_db/table_mapping.py', 'Excel-DB表名映射', '中'),
            ('pgsql_db/table_schemas.py', '表结构DDL定义', '中'),
            ('tools/bc_3way_content_compare.py', 'BC三版本比对脚本', '中'),
            ('tools/oc_3way_content_compare.py', 'OC三版本比对脚本', '中'),
            ('tools/generate_bc_10day_reports.py', 'BC报告生成脚本', '中'),
            ('tools/generate_oc_10day_reports.py', 'OC报告生成脚本', '中'),
            ('.claude/MEMORY.md', 'Claude任务记忆模块', '中'),
        ]
    )

    doc.add_page_break()

    # ====================================================================
    # 第9章 已知问题
    # ====================================================================
    doc.add_heading('9. 已知问题与注意事项', level=1)
    issues = [
        ('TRUNCATE风险', '每次新仿真会清空所有输出表，BC和OC共享同一组表。',
         '启动新仿真前确认已有数据已备份或比对完毕。'),
        ('数据库当前状态', '数据库中只保留OC数据（run_id: OC_Paste_S1_20251224_20260225_150912），'
         'BC数据已被OC仿真覆盖。', 'BC比对结果已保存在JSON文件中，不影响已生成报告。'),
        ('DuckDB默认关闭', '基准测试显示Pandas在<100K行场景更快。',
         '仅在数据量>500K行的聚合查询场景启用DuckDB。'),
        ('Cython收益有限', '三个Cython内核加速比仅0.96x-1.04x。',
         '保留作为可扩展性示范，不作为主要优化手段。'),
        ('向量化需求收集', 'M5的USE_VECTORIZED_DEMAND_COLLECTION当前被禁用。',
         '需修复horizon计算问题后启用，预期30-50%加速。'),
    ]
    for title, desc, action in issues:
        doc.add_heading(title, level=3)
        doc.add_paragraph(f'描述: {desc}')
        doc.add_paragraph(f'处理: {action}')

    doc.add_page_break()

    # ====================================================================
    # 第10章 后续优化建议
    # ====================================================================
    doc.add_heading('10. 后续优化建议', level=1)

    doc.add_heading('10.1 短期优化（低风险）', level=2)
    short = [
        '启用向量化需求收集(USE_VECTORIZED_DEMAND_COLLECTION=True)：修复horizon计算问题，预期M5需求收集30-50%加速',
        'M1供需日志批量处理：延迟聚合替代逐日生成，减少I/O',
        'M3 MRP并行度提升：探索进程池处理（需解决pickle问题）',
    ]
    for s in short:
        doc.add_paragraph(s, style='List Bullet')

    doc.add_heading('10.2 中期优化（中等风险）', level=2)
    mid = [
        'Numba JIT编译：对M5分配循环热点进行JIT加速',
        '增量计算：缓存跨天不变的计算结果，仅重算变化部分',
        '异步IO：数据加载和写入异步化，实现流水线处理',
    ]
    for m in mid:
        doc.add_paragraph(m, style='List Bullet')

    doc.add_heading('10.3 长期优化（需评估）', level=2)
    longterm = [
        'ChangeSet增量计算框架：跟踪每日变化集，仅重算受影响的节点',
        'DuckDB SQL替代：在数据量>500K行时将pandas操作迁移到DuckDB SQL',
        '分布式计算（Dask/Ray）：用于大规模仿真场景的水平扩展',
    ]
    for l in longterm:
        doc.add_paragraph(l, style='List Bullet')

    doc.add_page_break()

    # ====================================================================
    # 附录A: 17份测试报告索引
    # ====================================================================
    doc.add_heading('附录A: 17份测试报告索引', level=1)

    doc.add_heading('BC报告（9份）', level=2)
    bc_reports = [
        ('Day 01-10', '2025-10-05 ~ 2025-10-14', 'BC算法优化测试报告_Day01-10_20251005-20251014.md'),
        ('Day 11-20', '2025-10-15 ~ 2025-10-24', 'BC算法优化测试报告_Day11-20_20251015-20251024.md'),
        ('Day 21-30', '2025-10-25 ~ 2025-11-03', 'BC算法优化测试报告_Day21-30_20251025-20251103.md'),
        ('Day 31-40', '2025-11-04 ~ 2025-11-13', 'BC算法优化测试报告_Day31-40_20251104-20251113.md'),
        ('Day 41-50', '2025-11-14 ~ 2025-11-23', 'BC算法优化测试报告_Day41-50_20251114-20251123.md'),
        ('Day 51-60', '2025-11-24 ~ 2025-12-03', 'BC算法优化测试报告_Day51-60_20251124-20251203.md'),
        ('Day 61-70', '2025-12-04 ~ 2025-12-13', 'BC算法优化测试报告_Day61-70_20251204-20251213.md'),
        ('Day 71-80', '2025-12-14 ~ 2025-12-23', 'BC算法优化测试报告_Day71-80_20251214-20251223.md'),
        ('Day 81-87', '2025-12-24 ~ 2025-12-30', 'BC算法优化测试报告_Day81-87_20251224-20251230.md'),
    ]
    add_styled_table(doc,
        ['区间', '日期范围', '文件名'],
        bc_reports
    )

    doc.add_heading('OC报告（8份，含三列对比）', level=2)
    oc_reports = [
        ('Day 01-10', '2025-12-15 ~ 2025-12-24', 'OC算法优化测试报告_Day01-10_20251215-20251224.md'),
        ('Day 11-20', '2025-12-25 ~ 2026-01-03', 'OC算法优化测试报告_Day11-20_20251225-20260103.md'),
        ('Day 21-30', '2026-01-04 ~ 2026-01-13', 'OC算法优化测试报告_Day21-30_20260104-20260113.md'),
        ('Day 31-40', '2026-01-14 ~ 2026-01-23', 'OC算法优化测试报告_Day31-40_20260114-20260123.md'),
        ('Day 41-50', '2026-01-24 ~ 2026-02-02', 'OC算法优化测试报告_Day41-50_20260124-20260202.md'),
        ('Day 51-60', '2026-02-03 ~ 2026-02-12', 'OC算法优化测试报告_Day51-60_20260203-20260212.md'),
        ('Day 61-70', '2026-02-13 ~ 2026-02-22', 'OC算法优化测试报告_Day61-70_20260213-20260222.md'),
        ('Day 71-76', '2026-02-23 ~ 2026-02-28', 'OC算法优化测试报告_Day71-76_20260223-20260228.md'),
    ]
    add_styled_table(doc,
        ['区间', '日期范围', '文件名'],
        oc_reports
    )

    doc.add_page_break()

    # ====================================================================
    # 附录B: 关键脚本参考
    # ====================================================================
    doc.add_heading('附录B: 关键脚本参考', level=1)

    commands = [
        ('BC本地仿真', 'python run.py --config test_files/BC_S5.xlsx --start-date 2025-10-05 --end-date 2025-12-30'),
        ('BC DB仿真', 'python run.py --config BC_S5 --use-db --start-date 2025-10-05 --end-date 2025-12-30'),
        ('OC本地仿真', 'python run.py --config config/OC_Paste_S1_20251224 --start-date 2025-12-15 --end-date 2026-02-28'),
        ('OC DB仿真', 'python run.py --config OC_Paste_S1_20251224 --use-db --start-date 2025-12-15 --end-date 2026-02-28'),
        ('BC三版本比对', 'python tools/bc_3way_content_compare.py'),
        ('OC三版本比对', 'python tools/oc_3way_content_compare.py'),
        ('BC报告生成', 'python tools/generate_bc_10day_reports.py'),
        ('OC报告生成', 'python tools/generate_oc_10day_reports.py'),
    ]
    add_styled_table(doc,
        ['操作', '命令'],
        commands
    )

    doc.add_page_break()

    # ====================================================================
    # 附录C: 数据库连接信息
    # ====================================================================
    doc.add_heading('附录C: 数据库连接信息', level=1)
    add_styled_table(doc,
        ['参数', '值'],
        [
            ('Host', 'localhost'),
            ('Port', '5432'),
            ('Database', 'test_db'),
            ('User', 'postgres'),
            ('Password', '123456'),
            ('Client Encoding', 'utf8'),
        ]
    )
    doc.add_paragraph()
    doc.add_paragraph('数据路径参考：')
    add_styled_table(doc,
        ['场景', '版本', '路径/Run ID'],
        [
            ('BC', 'Dev', 'ChainSight_Dev/BC_S5/run_20260211_181635/'),
            ('BC', 'Src', 'outputs/BC_S5/run_20260211_145215/'),
            ('BC', 'DB', 'outputs/db_BC_S5_20260225_132106/ (run_id: BC_S5_20260225_132106)'),
            ('OC', 'Dev', 'outputs/run_20260127_142402/'),
            ('OC', 'Src', 'outputs/OC_Paste_S1_20251224/run_20260209_222302/'),
            ('OC', 'DB', 'outputs/db_OC_Paste_S1_20251224_20260225_150912/ (run_id: OC_Paste_S1_20251224_20260225_150912)'),
        ]
    )

    # ===== 报告结尾 =====
    doc.add_paragraph()
    doc.add_paragraph('--- 文档结束 ---').alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    p = doc.add_paragraph(f'本文档由 chenxianyue002@chinasofti.com 编制，生成日期: {datetime.date.today().strftime("%Y-%m-%d")}')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 保存
    output_path = DOCS_DIR / 'ChainSight项目交接文档.docx'
    doc.save(str(output_path))
    print(f"[OK] Word交接文档已生成: {output_path}")
    return output_path


if __name__ == '__main__':
    create_handover_document()
