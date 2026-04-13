#!/usr/bin/env python3
"""Generate comprehensive ChainSight architecture diagram (draw.io format).
Page 1 redesigned for maximum clarity."""
import html, os

E = lambda t: html.escape(str(t), quote=True)

def C(id, val, sty, par="1", x=0, y=0, w=100, h=50):
    return f'<mxCell id="{id}" value="{E(val)}" style="{sty}" vertex="1" parent="{par}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'

def EG(id, lbl, sty, src, tgt, par="1"):
    l = f' value="{E(lbl)}"' if lbl else ''
    return f'<mxCell id="{id}"{l} style="{sty}" edge="1" source="{src}" target="{tgt}" parent="{par}"><mxGeometry relative="1" as="geometry"/></mxCell>'

def PG(name, pid, cells, pw=2800, ph=1800):
    return f'<diagram name="{E(name)}" id="{pid}"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{pw}" pageHeight="{ph}" math="0" shadow="0"><root>\n<mxCell id="0"/>\n<mxCell id="1" parent="0"/>\n' + '\n'.join(cells) + '\n</root></mxGraphModel></diagram>'

# ── Styles ──
SW = lambda bg, bc: f"shape=swimlane;startSize=40;fillColor={bg};strokeColor={bc};fontStyle=1;fontSize=16;horizontal=1;collapsible=0;rounded=1;arcSize=8;swimlaneLine=1;fontColor=#333;shadow=1;"
CT = lambda bg, bc, fs=13: f"rounded=1;whiteSpace=wrap;html=1;container=1;collapsible=0;fillColor={bg};strokeColor={bc};strokeWidth=2;fontStyle=1;fontSize={fs};verticalAlign=top;spacingTop=8;arcSize=10;shadow=1;"
BX = lambda bg, bc, fs=11: f"rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize={fs};arcSize=10;shadow=0;fontStyle=0;"
BB = lambda bg, bc, fs=11: f"rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize={fs};arcSize=10;shadow=0;fontStyle=1;"
FI = lambda bg, bc: f"rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize=9;align=left;verticalAlign=top;spacingLeft=6;spacingTop=4;overflow=hidden;arcSize=4;"
AR = lambda c, w=2: f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth={w};strokeColor={c};fontSize=10;fontColor={c};"
DA = lambda c: f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor={c};dashed=1;dashPattern=8 8;fontSize=9;fontColor={c};"
TX = lambda fs=11, fc="#333", s=0: f"text;html=1;fontSize={fs};fontStyle={s};align=center;fillColor=none;strokeColor=none;fontColor={fc};"
TL = lambda fs=11, fc="#333", s=0: f"text;html=1;fontSize={fs};fontStyle={s};align=left;fillColor=none;strokeColor=none;fontColor={fc};"
HB = lambda bg, bc, fs=12: f"rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize={fs};arcSize=10;shadow=1;fontStyle=1;strokeWidth=3;"
DM = lambda bg, bc: f"rhombus;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize=11;shadow=0;"
EL = lambda bg, bc, fs=14: f"ellipse;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize={fs};fontStyle=1;shadow=1;"
NT = lambda bg, bc: f"shape=note;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};fontSize=9;align=left;verticalAlign=top;spacingLeft=6;spacingTop=4;shadow=0;size=14;"

# Module arrow style (thick, prominent)
MA = lambda: "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=4;strokeColor=#1B5E20;exitX=1;exitY=0.5;entryX=0;entryY=0.5;endArrow=blockThin;endFill=1;endSize=14;fontSize=14;fontStyle=1;fontColor=#1B5E20;labelBackgroundColor=#FFFFFF;"

# ════════════════════════════════════════════════════════
#  PAGE 1: 系统架构总览 (REDESIGNED for clarity)
# ════════════════════════════════════════════════════════
def page1():
    c = []

    # ── Title ──
    c.append(C("p1t", "<b>ChainSight v2.0 — 重构后系统架构总览</b>", TX(26,"#1a237e",1), x=500, y=8, w=1800, h=45))

    # ══════════════════════════════════════════════════
    #  Layer 1: 入口与执行模式
    # ══════════════════════════════════════════════════
    c.append(C("L1", "入口与执行模式  Entry Layer", SW("#BBDEFB","#1565C0"), x=30, y=65, w=2740, h=115))
    entries = [
        ("e1", "File 模式", "python run.py --config BC_S5.xlsx\n→ simulation_file.py", 80),
        ("e2", "DB 模式", "python run.py --db-config BC_S5\n→ simulation_db.py", 700),
        ("e3", "CLI 模式", "python -m src.core.main_integration\n→ cli.py", 1320),
        ("e4", "Parallel 模式", "CHAINSIGHT_PARALLEL=true\nM1 ∥ M4 ∥ M5 并行 (ThreadPool)", 1940),
    ]
    for eid, title, desc, x in entries:
        c.append(C(eid,
            f"<b style='font-size:13px'>{title}</b><br><font style='font-size:9px;color:#555'>{desc.replace(chr(10),'<br>')}</font>",
            BB("#E3F2FD","#1565C0",12), "L1", x, 42, 560, 58))

    # ══════════════════════════════════════════════════
    #  Layer 2: 核心编排层
    # ══════════════════════════════════════════════════
    c.append(C("L2", "核心编排层  Core Orchestration Layer", SW("#FFE0B2","#E65100"), x=30, y=215, w=2740, h=320))

    # ── Orchestrator (SSOT) — the visual centerpiece ──
    c.append(C("orch", "<b style='font-size:17px'>Orchestrator</b><br><font color='#D84315' style='font-size:12px'><b>中央状态枢纽 (Single Source of Truth)</b></font>",
        HB("#FBE9E7","#BF360C",14), "L2", 30, 44, 720, 255))
    # State grid inside Orchestrator
    state_items = [
        ("os1", "unrestricted_inventory", "非限制库存 Dict[(mat,loc)→qty]"),
        ("os2", "open_deployment", "未完成调拨 Dict[uid→record]"),
        ("os3", "in_transit", "在途库存 Dict[uid→record]"),
        ("os4", "production_gr / delivery_gr", "生产入库·交付入库 (按日)"),
        ("os5", "production_plan_backlog", "已确认生产计划 List"),
        ("os6", "daily_beginning/ending_inventory", "日首·日末库存快照"),
    ]
    for i, (sid, name, desc) in enumerate(state_items):
        yy = 55 + i * 27
        c.append(C(sid, f"<font style='font-size:9px'><b>{name}</b> — {desc}</font>", TL(9,"#333"), "orch", 15, yy, 690, 22))

    c.append(C("os_sep", "", "line;strokeColor=#E65100;strokeWidth=1;fillColor=none;", "orch", 15, 220, 690, 1))
    c.append(C("os_mx", "<font style='font-size:10px'><b>Mixin × 5:</b>  Views · Processors · DailyOps · Persistence · InventoryLog</font>",
        TL(10,"#BF360C",1), "orch", 15, 225, 690, 22))

    # ── main_integration/ ──
    c.append(C("mi", "<b>main_integration/</b><font color='#888' style='font-size:10px'>  (11 文件 · 集成执行框架)</font>",
        CT("#FFF8E1","#F57F17",12), "L2", 780, 44, 680, 255))
    mi_items = [
        ("mi1", "simulation_file.py", "File 模式主循环"),
        ("mi2", "simulation_db.py", "DB 模式主循环"),
        ("mi3", "config_loader.py", "Excel → config_dict"),
        ("mi4", "production_planning_runner.py", "M4 集成包装"),
        ("mi5", "runtime_state.py", "DbRuntimeState 跨日状态"),
        ("mi6", "memory_store.py", "DuckDB 零磁盘IO 路由"),
        ("mi7", "resume.py / seed.py / normalize.py", "断点续跑·随机种子·规范化"),
        ("mi8", "db_helpers.py", "DB 写入工具"),
    ]
    for i, (mid, fn, desc) in enumerate(mi_items):
        yy = 32 + i * 27
        c.append(C(mid, f"<font style='font-size:9px'><b>{fn}</b> — {desc}</font>", TL(9,"#333"), "mi", 12, yy, 650, 22))

    # ── parallel_executor/ ──
    c.append(C("pe", "<b>parallel_executor/</b>",
        CT("#E8EAF6","#3949AB",12), "L2", 1490, 44, 380, 120))
    c.append(C("pe1", "<font style='font-size:10px'><b>ParallelExecutor</b><br>ThreadPoolExecutor<br>M1 ∥ M4 ∥ M5 并行执行<br>ParallelTaskResult @dataclass</font>",
        TL(10,"#333"), "pe", 12, 30, 350, 75))

    # ── run/ ──
    c.append(C("ru", "<b>run/</b><font color='#888' style='font-size:10px'>  (6 文件)</font>",
        CT("#E8EAF6","#3949AB",12), "L2", 1490, 178, 380, 120))
    c.append(C("ru1", "<font style='font-size:10px'>run_main.py — argparse 调度<br>db_runner.py — DB 调度<br>db_config.py · output_dir.py<br>local_writer.py · utils.py</font>",
        TL(10,"#333"), "ru", 12, 30, 350, 70))

    # ── Orchestrator ↔ main_integration arrow ──
    c.append(EG("a_om", "调用", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#E65100;exitX=1;exitY=0.5;entryX=0;entryY=0.5;fontSize=10;fontColor=#E65100;dashed=1;", "orch", "mi", "L2"))

    # ══════════════════════════════════════════════════
    #  Layer 3: 业务模块层 — 每日仿真执行顺序
    # ══════════════════════════════════════════════════
    c.append(C("L3", "业务模块层  Modules Layer", SW("#C8E6C9","#2E7D32"), x=30, y=570, w=2740, h=430))

    # Execution order subtitle
    c.append(C("exo", "<b>每日仿真执行顺序</b>", TX(14,"#1B5E20",1), "L3", x=1000, y=2, w=700, h=28))

    # Module definitions
    modules = [
        ("m1", "M1", "需求规划", "demand_planning/", "#C8E6C9", "#2E7D32",
         "run_daily_order_generation()",
         ["需求预测 → 日订单", "订单消耗 (AO/Normal)", "发货与缺料计算", "DPS 地点拆分"], "11"),
        ("m3", "M3", "MRP 计划", "mrp_planning/", "#A5D6A7", "#1B5E20",
         "run_integrated_mode()",
         ["多层 BOM 分配", "净需求计算", "提前期推断", "DataIndexer O(1) 查询"], "11"),
        ("m4", "M4", "生产计划", "production_planning/", "#81C784", "#33691E",
         "run_daily_production_planning()",
         ["DailyProductionPlanner", "无约束计划 + 换产优化", "产能约束分配", "产线状态持久化"], "10"),
        ("m5", "M5", "调拨规划", "deployment_planning/", "#DCEDC8", "#558B2F",
         "main()",
         ["多层部署分配", "MOQ/RV 取整", "优先级向量化分配", "Push/Soft-push 策略"], "17"),
        ("m6", "M6", "物流执行", "logistics_execution/", "#F0F4C3", "#827717",
         "run_daily_physical_flow()",
         ["VehiclePacker 装载优化", "交付延迟随机抽样", "MDQ 旁路规则", "物理库存跟踪"], "16"),
    ]

    mw = 450   # module width
    mh = 300   # module height
    gap = 55   # gap between modules
    total = 5 * mw + 4 * gap  # = 2470
    ml = (2700 - total) // 2  # left margin = 115

    for i, (mid, code, cname, pkg, bg, bc, entry, feats, fcount) in enumerate(modules):
        x = ml + i * (mw + gap)
        # Module container box
        feat_html = '<br>'.join(f'• {f}' for f in feats)
        label = (
            f"<b style='font-size:22px;color:{bc}'>{code}</b><br>"
            f"<b style='font-size:14px'>{cname}</b><br>"
            f"<i style='font-size:10px;color:#666'>{pkg}</i>"
            f"<hr size='1' color='{bc}'>"
            f"<font style='font-size:10px;color:#333'><b>{entry}</b></font><br><br>"
            f"<font style='font-size:10px;color:#555'>{feat_html}</font><br><br>"
            f"<font style='font-size:9px;color:#999'>— {fcount} 个文件</font>"
        )
        style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={bg};strokeColor={bc};strokeWidth=2;fontSize=11;arcSize=12;shadow=1;verticalAlign=top;spacingTop=12;spacingLeft=10;spacingRight=10;align=center;"
        c.append(C(mid, label, style, "L3", x, 40, mw, mh))

    # Execution order arrows between modules (thick, prominent)
    arrow_labels = ["①", "②", "③", "④"]
    mod_ids = ["m1", "m3", "m4", "m5", "m6"]
    for i in range(4):
        c.append(EG(f"ma{i}", arrow_labels[i], MA(), mod_ids[i], mod_ids[i+1], "L3"))

    # Orchestrator interaction bar below modules
    bar_y = mh + 52
    c.append(C("oi_bar",
        "<b style='font-size:12px;color:#D84315'>↕  Orchestrator 中央状态交互  ↕</b><br>"
        "<font style='font-size:9px;color:#888'>各模块通过 Orchestrator 读取状态视图 / 写入处理结果　(详见 Page 5)</font>",
        f"rounded=1;whiteSpace=wrap;html=1;fillColor=#FBE9E7;strokeColor=#D84315;strokeWidth=2;fontSize=11;arcSize=8;dashed=1;dashPattern=8 4;shadow=0;",
        "L3", ml, bar_y, total, 45))

    # ══════════════════════════════════════════════════
    #  Layer 4: 支撑层
    # ══════════════════════════════════════════════════
    c.append(C("L4", "支撑层  Support Layer", SW("#E0E0E0","#616161"), x=30, y=1040, w=2740, h=250))

    # ── Config System ──
    c.append(C("cfg", "<b style='font-size:13px'>配置系统 Config</b>",
        CT("#F3E5F5","#7B1FA2",12), "L4", 25, 42, 830, 185))
    cfg_items = [
        ("cg1","config/loader.py","YAML 配置加载 + 深度合并 + 缓存\nget_config() / get_module_config() / get_shared_config()"),
        ("cg2","default_config.yaml","全局默认参数 · 各模块默认值"),
        ("cg3","Excel 配置 (BC_S5.xlsx)","20+ Sheet: SIT_Design, M1_DemandForecast,\nM4_LineCapacity, M6_TruckSpecs ..."),
        ("cg4","环境变量覆盖","CHAINSIGHT_CONFIG=/path.yaml"),
    ]
    for i,(cid,title,desc) in enumerate(cfg_items):
        yy = 32 + i * 37
        c.append(C(cid, f"<font style='font-size:9px'><b>{title}</b> — {desc.replace(chr(10),' ')}</font>", TL(9), "cfg", 12, yy, 800, 30))

    # ── Shared Utils ──
    c.append(C("utl", "<b style='font-size:13px'>共享工具 Utilities</b>",
        CT("#B2EBF2","#00838F",12), "L4", 885, 42, 940, 185))
    utl_items = [
        ("ut1","normalization.py","标识符规范化 (location/material)"),
        ("ut2","time_manager.py","SimulationTimeManager 日期统一管理"),
        ("ut3","validation_manager.py","ValidationManager 错误/警告/信息收集"),
        ("ut4","config_validator.py","run_pre_simulation_validation() 预校验"),
        ("ut5","inventory_balance_checker.py","库存收支平衡验证"),
        ("ut6","duckdb_sql_wrapper.py","DuckDB merge/groupby/filter 加速"),
        ("ut7","memory_data_store.py","MemoryDataStore DuckDB 内存路由"),
        ("ut8","resource_config.py / cpu_config.py","CPU/内存自动检测 · get_optimal_workers()"),
    ]
    for i,(uid,title,desc) in enumerate(utl_items):
        col = i // 4
        row = i % 4
        xx = 12 + col * 460
        yy = 32 + row * 36
        c.append(C(uid, f"<font style='font-size:9px'><b>{title}</b><br>{desc}</font>", TL(9), "utl", xx, yy, 445, 30))

    # ── Services ──
    c.append(C("svc", "<b style='font-size:13px'>服务 Services</b>",
        CT("#FCE4EC","#AD1457",12), "L4", 1855, 42, 560, 185))
    svc_items = [
        ("sv1","SummaryReportGenerator","generate_all_reports()\n8 份汇总报告：订单/产能/换产/\n调拨/生产/交付/卡车/库存"),
        ("sv2","PerformanceProfiler","cProfile 上下文管理器\n各模块性能采集与报告输出"),
    ]
    for i,(sid,title,desc) in enumerate(svc_items):
        yy = 32 + i * 80
        c.append(C(sid, f"<font style='font-size:10px'><b>{title}</b></font><hr size='1'><font style='font-size:9px'>{desc.replace(chr(10),'<br>')}</font>",
            FI("#FCE4EC","#AD1457"), "svc", 15, yy, 525, 68))

    # ══════════════════════════════════════════════════
    #  Inter-layer arrows
    # ══════════════════════════════════════════════════
    c.append(EG("la1", "调用", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#1565C0;fontSize=11;fontColor=#1565C0;fontStyle=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;", "L1", "L2"))
    c.append(EG("la2", "编排调度", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#E65100;fontSize=11;fontColor=#E65100;fontStyle=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;", "L2", "L3"))
    c.append(EG("la3", "配置 · 工具 · 服务", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=1.5;strokeColor=#616161;fontSize=10;fontColor=#616161;dashed=1;dashPattern=6 4;exitX=0.5;exitY=0;entryX=0.5;entryY=1;", "L4", "L3"))

    # ══════════════════════════════════════════════════
    #  Legend
    # ══════════════════════════════════════════════════
    c.append(C("leg", "<b>图例</b>", TX(12,"#333",1), x=30, y=1300, w=80, h=25))
    legend = [
        ("#BBDEFB","#1565C0","入口层"),
        ("#FFE0B2","#E65100","核心编排层"),
        ("#C8E6C9","#2E7D32","业务模块层"),
        ("#E0E0E0","#616161","支撑层"),
        ("#FBE9E7","#BF360C","Orchestrator SSOT"),
    ]
    for i,(bg,bc,lb) in enumerate(legend):
        xx = 30 + i * 280
        c.append(C(f"lg{i}", "", f"rounded=1;fillColor={bg};strokeColor={bc};strokeWidth=2;", x=xx, y=1330, w=22, h=14))
        c.append(C(f"lt{i}", lb, TL(10), x=xx+28, y=1327, w=200, h=20))
    # Arrow legend
    c.append(C("lga", "", f"shape=singleArrow;fillColor=#1B5E20;strokeColor=#1B5E20;arrowWidth=0.4;arrowSize=0.3;", x=1430, y=1330, w=35, h=14))
    c.append(C("lgt", "<font style='font-size:10px'>模块执行顺序 (M1→M3→M4→M5→M6)</font>", TL(10), x=1470, y=1327, w=350, h=20))

    # Page note
    c.append(C("pn", "<font style='font-size:9px;color:#999'>详细内容请参阅: Page 2 (Core 层详解) · Page 3 (模块内部结构) · Page 4 (执行流程) · Page 5 (数据流) · Page 6 (基础设施)</font>",
        TX(9,"#999"), x=500, y=1355, w=1800, h=20))

    return PG("1. 系统架构总览", "p1", c, 2800, 1400)


# ════════════════════════════════════════════════════════
#  PAGE 2: Core 层详解
# ════════════════════════════════════════════════════════
def page2():
    c = []
    c.append(C("p2t", "<b>Core 层详解 — Orchestrator · main_integration · parallel_executor</b>", TX(20,"#E65100",1), x=400, y=10, w=2000, h=35))
    c.append(C("oc", "<b>Orchestrator（中央状态枢纽 · Single Source of Truth）</b>", CT("#FBE9E7","#D84315",14), x=20, y=60, w=1340, h=820))
    c.append(C("ocm", "<b>Mixin 多继承架构</b>", BB("#FFCCBC","#BF360C",12), "oc", 20, 38, 1290, 30))
    mixins = [
        ("mx1","OrchestratorViewsMixin","views.py","只读查询方法\nget_unrestricted_inventory_view()\nget_open_deployment_view()\nget_in_transit_view()"),
        ("mx2","OrchestratorProcessorsMixin","processors.py","模块输出处理\nprocess_module1_shipments()\nprocess_module4_production()\nprocess_module5_deployment()\nprocess_module6_delivery()"),
        ("mx3","OrchestratorDailyOpsMixin","daily_ops.py","每日处理管线\nrun_daily_processing()\ncleanup_past_due_open_deployments()\n按序处理 M1→M4→M5→M6"),
        ("mx4","OrchestratorPersistenceMixin","persistence.py","状态持久化\nsave_daily_state() → CSV\nload_state_from_csv()\n日快照: inventory/deployment/transit"),
        ("mx5","OrchestratorInventoryLogMixin","inventory_log.py","库存日志\nlog_daily_beginning_inventory()\nlog_daily_ending_inventory()\n收支明细记录"),
    ]
    for i,(mid,name,fn,desc) in enumerate(mixins):
        xx = 20 + (i % 3) * 425
        yy = 78 + (i // 3) * 145
        c.append(C(mid, f"<b>{name}</b><br><font color='#888' style='font-size:8px'>{fn}</font><hr size='1'><font style='font-size:9px'>{desc.replace(chr(10),'<br>')}</font>", FI("#FFF3E0","#E65100"), "oc", xx, yy, 405, 130))
    c.append(C("ocs", "<b>核心状态变量 (State)</b>", BB("#FFCCBC","#BF360C",12), "oc", 20, 375, 1290, 30))
    states = [
        ("st1","unrestricted_inventory","Dict[(material, location) → qty]","非限制库存（主库存）"),
        ("st2","open_deployment","Dict[uid → record]","未完成调拨单"),
        ("st3","in_transit","Dict[uid → record]","在途库存"),
        ("st4","production_gr","List[dict]","生产入库记录（按日）"),
        ("st5","delivery_gr","List[dict]","交付入库记录（按日）"),
        ("st6","shipment_log","List[dict]","客户发货日志"),
        ("st7","production_plan_backlog","List[dict]","已确认生产计划"),
        ("st8","space_capacity","pd.DataFrame","库位空间约束"),
        ("st9","daily_beginning/ending_inventory","Dict[date → snapshot]","日首/日末库存快照"),
        ("st10","uid_sequence","int (auto-increment)","调拨UID自增序列"),
    ]
    for i,(sid,name,typ,desc) in enumerate(states):
        xx = 20 + (i % 2) * 645
        yy = 415 + (i // 2) * 38
        c.append(C(sid, f"<b>{name}</b>: <font color='#6a1b9a'>{typ}</font> — <font color='#555'>{desc}</font>", TL(9,"#333"), "oc", xx, yy, 630, 32))
    c.append(C("uid", "<b>DeploymentUID @dataclass (models.py)</b><hr size='1'><font style='font-size:9px'>material: str | sending: str | receiving: str<br>planned_deploy_date: str (YYYY-MM-DD)<br>demand_element: str | sequence: int (自增)<br>to_string() → UID字符串 | from_string() → 解析</font>", FI("#FFF3E0","#E65100"), "oc", 20, 620, 400, 100))
    c.append(C("nrm", "<b>normalize.py</b><hr size='1'><font style='font-size:9px'>normalize_identifier()<br>标识符清洗：去空格/统一大小写<br>供 Orchestrator 内部使用</font>", FI("#FFF3E0","#E65100"), "oc", 440, 620, 300, 100))
    c.append(C("orc_main", "<b>orchestrator_main.py</b><hr size='1'><font style='font-size:9px'>class Orchestrator(Views, Processors,<br>  DailyOps, Persistence, InventoryLog):<br>  __init__(start_date, output_dir)<br>  ← 五大 Mixin 合成的完整类</font>", FI("#FFF3E0","#E65100"), "oc", 760, 620, 380, 100))
    c.append(C("mic", "<b>main_integration/（集成执行框架 · 11 个文件）</b>", CT("#FFF8E1","#F57F17",13), x=1400, y=60, w=1380, h=820))
    mi_files = [
        ("mf1","simulation_file.py","File 模式主入口","run_integrated_simulation(config_path, start_date, end_date, output_base_dir)\n→ 预校验 → 创建Orchestrator → 日循环 → 汇总报告"),
        ("mf2","simulation_db.py","DB 模式主入口","run_integrated_simulation_from_dict(config_dict, ...)\n→ DbRuntimeState → MemoryDataStore → 日循环 → DB写入"),
        ("mf3","cli.py","CLI 入口","main() → argparse → 路由到 file/db 模式\npython -m src.core.main_integration"),
        ("mf4","config_loader.py","配置加载器","load_configuration(config_path) → config_dict\n读取 Excel → 标准化 Sheet 名 → 统一 DataFrame 格式"),
        ("mf5","production_planning\n_runner.py","M4 集成包装","run_module4_integrated(config_dict, date, ...)\nload_current_date_production_gr()"),
        ("mf6","runtime_state.py","跨日运行时状态","class DbRuntimeState:\n  m4_line_states / m4_allocated_capacity\n  previous_m3_result / cleanup_audit_df"),
        ("mf7","memory_store.py","内存数据路由","配置 MemoryDataStore (DuckDB in-memory)\nwrite_module_output() / read_module_output()\n零磁盘IO数据传递 (50-100× faster)"),
        ("mf8","resume.py","断点续跑","check_resume_capability(output_dir, start, end)\n检测已完成日期 → 返回 actual_start_date"),
        ("mf9","seed.py","随机种子","set_module_seeds(config_dict)\n初始化各模块随机种子 → 可复现性"),
        ("mf10","normalize.py","配置规范化","normalize_config_dict(config_dict)\n统一列名/数据类型/缺失值处理"),
        ("mf11","db_helpers.py","DB 写入工具","write_daily_output_to_db()\nbatch_upsert() / ensure_table_exists()"),
    ]
    for i,(fid,fn,role,desc) in enumerate(mi_files):
        xx = 15 + (i % 2) * 680
        yy = 38 + (i // 2) * 130
        c.append(C(fid, f"<b>{fn}</b> <font color='#888' style='font-size:9px'>— {role}</font><hr size='1'><font style='font-size:9px'>{desc.replace(chr(10),'<br>')}</font>", FI("#FFFDE7","#F57F17"), "mic", xx, yy, 660, 118))
    c.append(C("pec", "<b>parallel_executor/（并行执行器 · 4 个文件）</b>", CT("#E8EAF6","#3949AB",13), x=20, y=900, w=1340, h=200))
    c.append(C("pf1", "<b>parallel_executor_main.py</b><hr size='1'><font style='font-size:9px'>class ParallelExecutor:\n  max_workers: int | enable_parallel: bool\n  executor: ThreadPoolExecutor\n  run_parallel_stage(tasks) → (results_dict, all_success)\n  M1 ∥ M4 ∥ M5 三模块并行执行</font>", FI("#E8EAF6","#3949AB"), "pec", 20, 38, 420, 140))
    c.append(C("pf2", "<b>models.py</b><hr size='1'><font style='font-size:9px'>@dataclass ParallelTaskResult:\n  task_name / status ('success'|'error'|'timeout')\n  result / error / start_time / end_time\n  @property elapsed_time → float</font>", FI("#E8EAF6","#3949AB"), "pec", 460, 38, 400, 90))
    c.append(C("pf3", "<b>convenience.py</b><br><font style='font-size:9px'>便捷函数封装</font>", FI("#E8EAF6","#3949AB"), "pec", 460, 138, 200, 38))
    c.append(C("ruc", "<b>run/（生产入口 · 6 个文件）</b>", CT("#E8EAF6","#3949AB",13), x=1400, y=900, w=1380, h=200))
    run_files = [
        ("rf1","run_main.py","argparse CLI 调度器\n--config / --db-config / --start-date / --end-date"),
        ("rf2","db_runner.py","数据库模式调度\n加载 DB 配置 → 调用 simulation_db"),
        ("rf3","db_config.py","DB 连接配置加载\nPostgreSQL host/port/dbname"),
        ("rf4","output_dir.py","输出目录管理\ncreate_output_structure()"),
        ("rf5","local_writer.py","本地文件写入器\nDB模式的本地副本输出"),
        ("rf6","utils.py","工具函数\n日期解析/路径处理"),
    ]
    for i,(rid,fn,desc) in enumerate(run_files):
        xx = 15 + (i % 3) * 450
        yy = 38 + (i // 3) * 80
        c.append(C(rid, f"<b>{fn}</b><hr size='1'><font style='font-size:9px'>{desc.replace(chr(10),'<br>')}</font>", FI("#E8EAF6","#3949AB"), "ruc", xx, yy, 430, 68))
    return PG("2. Core 层详解", "p2", c, 2800, 1120)


# ════════════════════════════════════════════════════════
#  PAGE 3: 业务模块内部结构
# ════════════════════════════════════════════════════════
def page3():
    c = []
    c.append(C("p3t", "<b>业务模块内部结构详解 — M1 · M3 · M4 · M5 · M6</b>", TX(20,"#2E7D32",1), x=500, y=5, w=1800, h=35))
    c.append(C("M1", "<b>M1 需求规划  demand_planning/ (11 文件)</b>", CT("#C8E6C9","#2E7D32",13), x=20, y=50, w=880, h=700))
    m1f = [
        ("m1f1","__init__.py","模块导出\nrun_daily_order_generation() 别名"),
        ("m1f2","integration.py","集成入口\nrun_daily_order_generation(config_dict, sim_date, output_dir,\n  orchestrator, skip_file_output, previous_orders_df)\n→ {'orders_df','shipment_df','cut_df','supply_demand_log_df'}"),
        ("m1f3","config.py","load_config(xlsx_path) → config dict\napply_dps() / apply_supply_choice()"),
        ("m1f4","constants.py","模块常量定义\n列名/默认值/阈值"),
        ("m1f5","dps.py","DPS 地点拆分逻辑\napply_dps(df, dps_cfg) → 拆分后 DataFrame"),
        ("m1f6","forecast.py","预测展开\nexpand_forecast_to_days_integer_split(weekly_df)\n周预测 → 日预测（整数拆分）"),
        ("m1f7","order.py","订单生成\ngenerate_daily_orders(date, daily_forecast, ao_config)\n含误差建模 (正态/均匀分布)"),
        ("m1f8","consume.py","订单消耗\nAO 订单 vs 普通订单消耗逻辑\n库存分配优先级处理"),
        ("m1f9","consume_optimized.py","消耗优化版\n向量化消耗计算 (大数据场景)"),
        ("m1f10","shipment.py","发货计算\nsimulate_shipment_for_single_day(orders_df, inventory)\n→ shipment_df + shortage_df"),
        ("m1f11","normalization.py\nio_utils.py","规范化 (re-export → utils)\nIO 工具函数"),
    ]
    for i,(fid,fn,desc) in enumerate(m1f):
        yy = 35 + i * 58
        c.append(C(fid, f"<b>{fn}</b><hr size='1'><font style='font-size:8px'>{desc.replace(chr(10),'<br>')}</font>", FI("#E8F5E9","#2E7D32"), "M1", 15, yy, 845, 52))
    c.append(C("M3", "<b>M3 MRP 计划  mrp_planning/ (11 文件)</b>", CT("#A5D6A7","#1B5E20",13), x=920, y=50, w=880, h=700))
    m3f = [
        ("m3f1","__init__.py","模块导出\nrun_integrated_mode() 别名"),
        ("m3f2","integration.py","集成入口\nrun_integrated_mode(m1_output_dir, orchestrator, config_dict,\n  start_date, end_date, output_dir, skip_file_output, m1_result)\n→ {'net_demand_df','consolidated_demand_df',...}"),
        ("m3f3","mrp_simulation.py","MRP 仿真核心\nrun_mrp_layered_simulation_daily(sim_date, daily_supply_demand,\n  daily_orders, ...) → 按层计算净需求"),
        ("m3f4","net_demand.py","净需求计算\ncalculate_daily_net_demand(...)\n= 毛需求 - 在途 - 库存 + 安全库存"),
        ("m3f5","lead_time.py","提前期推断\ndetermine_lead_time(material, from_loc, to_loc, lead_time_df)\n多级匹配回退策略"),
        ("m3f6","layer_assignment.py","多层 BOM 分配\nassign_location_layers(network_df)\n根据网络拓扑分配层级"),
        ("m3f7","node_processor.py","节点处理器\nclass NodeProcessor\n单节点 MRP 计算逻辑"),
        ("m3f8","data_indexer.py","数据索引器\nclass DataIndexer\n预建索引 → O(1) 查询 (替代 O(n×m) 过滤)"),
        ("m3f9","config_loader.py\nconstants.py","配置加载 + 常量\n模块专用配置读取"),
        ("m3f10","utils.py","工具函数\nMOQ/RV 取整·规范化·largest_remainder"),
        ("m3f11","duckdb_batch\n_calculator.py","DuckDB 批量计算\n大规模层级计算加速"),
    ]
    for i,(fid,fn,desc) in enumerate(m3f):
        yy = 35 + i * 58
        c.append(C(fid, f"<b>{fn}</b><hr size='1'><font style='font-size:8px'>{desc.replace(chr(10),'<br>')}</font>", FI("#C8E6C9","#1B5E20"), "M3", 15, yy, 845, 52))
    c.append(C("M4", "<b>M4 生产计划  production_planning/ (10 文件)</b>", CT("#81C784","#33691E",13), x=1820, y=50, w=880, h=700))
    m4f = [
        ("m4f1","__init__.py","模块导出\nrun_daily_production_planning() 别名"),
        ("m4f2","main.py","主执行类\nclass DailyProductionPlanner:\n  load_config() → load_daily_net_demand()\n  → build_unconstrained_plan() → allocate() → simulate()"),
        ("m4f3","config_loader.py","配置加载\n产线配置/产能/换产矩阵/可靠性参数"),
        ("m4f4","constants.py\ntypes.py","常量 + 数据类型\n@dataclass: LineState, ChangeoverInfo,\nPlanRecord, ExceedRecord"),
        ("m4f5","demand_loader.py","需求加载\nload_daily_net_demand(m3_output_dir, sim_date)\n从 M3 结果读取净需求"),
        ("m4f6","plan_builder.py","计划构建\nbuild_unconstrained_plan() 无约束计划\noptimal_changeover_sequence() 换产优化"),
        ("m4f7","capacity_allocator.py","产能分配\ncentralized_capacity_allocation_with_changeover()\n约束分配 + 换产时间扣减 + 模拟生产"),
        ("m4f8","state_manager.py","状态管理\nsave_line_state() / load_line_state()\nsave_allocated_capacity() / load_all_previous()"),
        ("m4f9","output_writer.py","输出写入\n日产计划 Excel / 超产日志 / 换产日志"),
        ("m4f10","utils.py\nduckdb_batch\n_calculator.py","工具 + DuckDB 加速\n产能计算优化"),
    ]
    for i,(fid,fn,desc) in enumerate(m4f):
        yy = 35 + i * 64
        c.append(C(fid, f"<b>{fn}</b><hr size='1'><font style='font-size:8px'>{desc.replace(chr(10),'<br>')}</font>", FI("#AED581","#33691E"), "M4", 15, yy, 845, 58))
    c.append(C("M5", "<b>M5 调拨规划  deployment_planning/ (17 文件)</b>", CT("#DCEDC8","#558B2F",13), x=20, y=770, w=1340, h=800))
    m5f = [
        ("m5f1","__init__.py","模块导出: main() 别名"),
        ("m5f2","main.py","集成入口\nmain(config_file, m1_dir, m4_dir, orchestrator, sim_date, ...)\n→ {'deployment_df','validation_log',...}"),
        ("m5f3","data_loader.py","数据加载\nload_integrated_config(config_dict, date)\n加载配置 + M1/M4/Orchestrator 数据"),
        ("m5f4","allocation.py","分配核心\napply_moq_rv(demands, ptf_lsk_cache)\napply_priority_allocation_vectorized()\napply_receiving_space_quota()"),
        ("m5f5","push_allocation.py","Push/Soft-push\npush_softpush_allocation(...)\nPush 模式专用分配逻辑"),
        ("m5f6","demand_collector.py","需求收集\ncollect_node_demands(...)\n逐节点收集需求"),
        ("m5f7","demand_collector\n_vectorized.py","向量化需求收集\ncollect_demands_batch_vectorized()\nPandas 向量化操作 (性能优化)"),
        ("m5f8","inventory.py","库存计算\ncalculate_projected_inventory()\ncalculate_available_inventory()\ncheck_stock()"),
        ("m5f9","cache_utils.py","缓存工具\n构建 PTF/LSK 缓存·提前期缓存\n网络缓存·安全库存索引"),
        ("m5f10","horizon_batch\n_calculator.py","批量计算\n多时间窗口批量分配计算"),
        ("m5f11","multiprocess\n_optimizer.py","多进程优化\nThreadPoolExecutor 层级并行处理\nprocess_layer_multiprocess()"),
        ("m5f12","batch_optimizer.py","批量优化器\nBatchOptimizer 类"),
        ("m5f13","validation.py","校验逻辑\n分配结果合理性验证"),
        ("m5f14","normalizer.py","规范化 (re-export)\n→ src.utils.normalization"),
        ("m5f15","config.py\nconfig_loader.py","配置定义 + 加载\n模块专用配置参数"),
        ("m5f16","constants.py","常量定义"),
        ("m5f17","duckdb_batch\n_calculator.py","DuckDB 加速\n大规模分配计算"),
    ]
    for i,(fid,fn,desc) in enumerate(m5f):
        xx = 15 + (i % 2) * 660
        yy = 35 + (i // 2) * 82
        c.append(C(fid, f"<b>{fn}</b><hr size='1'><font style='font-size:8px'>{desc.replace(chr(10),'<br>')}</font>", FI("#F1F8E9","#558B2F"), "M5", xx, yy, 640, 75))
    c.append(C("M6", "<b>M6 物流执行  logistics_execution/ (16 文件)</b>", CT("#F0F4C3","#827717",13), x=1380, y=770, w=1320, h=800))
    m6f = [
        ("m6f1","__init__.py","模块导出\nrun_daily_physical_flow() 别名"),
        ("m6f2","main.py","集成入口\nrun_daily_physical_flow(config_dict, orchestrator, date, ...)\n→ {'truck_load_df','delivery_df','physical_inventory_df',...}"),
        ("m6f3","initializer.py","初始化\ninitialize_run_params()\n运行参数设置"),
        ("m6f4","data_preparer.py","数据准备\nprepare_data(config, orchestrator)\n加载 M5 调拨 + 卡车配置"),
        ("m6f5","config_loader.py","配置加载\n加载 M5 调拨结果 + 卡车规格\n+ 交付延迟分布 + MDQ 规则"),
        ("m6f6","simulation_engine.py","仿真引擎\nrun_simulation_loop()\n每日物流仿真主循环"),
        ("m6f7","route_processor.py","路线处理\nprocess_routes()\n按路线分组处理调拨"),
        ("m6f8","vehicle_packer.py","装车优化\nclass VehiclePacker\n多目标装载 (重量+体积约束)\npack_vehicles() → truck_loads"),
        ("m6f9","delivery_processor.py","交付处理\nsample_delivery_delay(distribution_params)\n随机交付延迟抽样"),
        ("m6f10","constraint_enforcer.py","约束执行\nenforce_shipment_constraint()\n发货约束强制执行"),
        ("m6f11","expression\n_evaluator.py","表达式评估器\nclass SafeExpressionEvaluator\n安全 MDQ 旁路规则评估\n防止 ReDoS 攻击"),
        ("m6f12","inventory_manager.py","库存管理\ncalculate_physical_inventory()\n物理库存跟踪与更新"),
        ("m6f13","capacity_manager.py","运力管理\n卡车运力规范化\n容量/重量/体积计算"),
        ("m6f14","validators.py","校验器\n输入/输出数据验证"),
        ("m6f15","output_builder.py","输出构建\ngenerate_outputs()\n组装最终输出 DataFrames"),
        ("m6f16","constants.py","常量定义\n模块专用常量"),
    ]
    for i,(fid,fn,desc) in enumerate(m6f):
        xx = 15 + (i % 2) * 650
        yy = 35 + (i // 2) * 92
        c.append(C(fid, f"<b>{fn}</b><hr size='1'><font style='font-size:8px'>{desc.replace(chr(10),'<br>')}</font>", FI("#FFFDE7","#827717"), "M6", xx, yy, 630, 84))
    return PG("3. 业务模块内部结构", "p3", c, 2720, 1590)


# ════════════════════════════════════════════════════════
#  PAGE 4: 每日仿真执行流程
# ════════════════════════════════════════════════════════
def page4():
    c = []
    c.append(C("p4t", "<b>每日仿真执行流程 (File 模式)</b>", TX(20,"#1565C0",1), x=300, y=5, w=1000, h=35))
    xm, xo, xn = 550, 1250, 60
    c.append(C("f0", "<b>开始</b>", EL("#1565C0","#0D47A1"), x=xm, y=50, w=160, h=50))
    c.append(C("f1", "<b>1. 预校验配置</b><br><font style='font-size:9px'>run_pre_simulation_validation(config_path)<br>检查 Excel Sheet/列/数据类型/M1-M6规则</font>", BB("#E3F2FD","#1565C0",11), x=xm-60, y=120, w=280, h=70))
    c.append(C("f1o", "<font style='font-size:9px'>→ validation_report.txt</font>", BX("#E8F5E9","#2E7D32",9), x=xo, y=130, w=200, h=30))
    c.append(C("f2", "<b>2. 断点续跑检查</b><br><font style='font-size:9px'>check_resume_capability(output_dir, start, end)<br>检测已完成日期 → actual_start_date</font>", BB("#E3F2FD","#1565C0",11), x=xm-60, y=210, w=280, h=70))
    c.append(C("f3", "<b>3. 创建 Orchestrator</b><br><font style='font-size:9px'>create_orchestrator(start_date, output_dir)<br>初始化中央状态枢纽</font>", BB("#FFF3E0","#E65100",11), x=xm-60, y=300, w=280, h=65))
    c.append(C("f4", "<b>4. 初始化时间管理器</b><br><font style='font-size:9px'>SimulationTimeManager(start_date)</font>", BB("#E3F2FD","#1565C0",11), x=xm-60, y=385, w=280, h=50))
    c.append(C("f5", "<b>5. 加载配置</b><br><font style='font-size:9px'>load_configuration(config_path)<br>Excel → config_dict (20+ DataFrames)</font>", BB("#F3E5F5","#7B1FA2",11), x=xm-60, y=455, w=280, h=65))
    c.append(C("f6", "<b>6. 设置随机种子</b><br><font style='font-size:9px'>set_module_seeds(config_dict) → 可复现性</font>", BB("#E3F2FD","#1565C0",11), x=xm-60, y=540, w=280, h=50))
    c.append(C("fls", "<b>每日循环开始</b><br><font style='font-size:10px'>for date in [actual_start, end_date]</font>", HB("#BBDEFB","#0D47A1",12), x=xm-80, y=620, w=320, h=50))
    c.append(C("fm1", "<b>7a. M1 需求规划</b><br><font style='font-size:9px'>run_daily_order_generation(config_dict, date,<br>  output_dir, orchestrator)</font>", BB("#C8E6C9","#2E7D32",11), x=xm-80, y=700, w=320, h=65))
    c.append(C("fm1o", "<font style='font-size:8px'>→ orders_df<br>→ shipment_df<br>→ cut_df (缺料)<br>→ supply_demand_log_df</font>", BX("#E8F5E9","#2E7D32",8), x=xo, y=695, w=200, h=65))
    c.append(C("fm1n", "<font style='font-size:8px'><b>输入：</b><br>• config_dict[M1_*]<br>• orchestrator.unrestricted_inventory<br>• previous_orders_df</font>", NT("#FFFDE7","#F57F17"), x=xn, y=695, w=200, h=70))
    c.append(C("fm3", "<b>7b. M3 MRP 计划</b><br><font style='font-size:9px'>run_integrated_mode(m1_dir, orchestrator,<br>  config_dict, start, end, output_dir, m1_result)</font>", BB("#A5D6A7","#1B5E20",11), x=xm-80, y=795, w=320, h=65))
    c.append(C("fm3o", "<font style='font-size:8px'>→ net_demand_df<br>→ consolidated_demand_df<br>→ safety_demand_df</font>", BX("#E8F5E9","#1B5E20",8), x=xo, y=795, w=200, h=55))
    c.append(C("fm3n", "<font style='font-size:8px'><b>输入：</b><br>• M1 orders/shipment<br>• orchestrator: open_deployment,<br>  in_transit, delivery_gr,<br>  production_plan_backlog</font>", NT("#FFFDE7","#F57F17"), x=xn, y=790, w=200, h=80))
    c.append(C("fm4", "<b>7c. M4 生产计划</b><br><font style='font-size:9px'>run_module4_integrated(config_dict, date,<br>  output_dir, m3_result, runtime_state)</font>", BB("#81C784","#33691E",11), x=xm-80, y=890, w=320, h=65))
    c.append(C("fm4o", "<font style='font-size:8px'>→ production_df<br>→ exceed_log<br>→ changeover_log<br>→ line_states / capacity</font>", BX("#E8F5E9","#33691E",8), x=xo, y=885, w=200, h=65))
    c.append(C("fm4n", "<font style='font-size:8px'><b>输入：</b><br>• M3 net_demand_df<br>• line_states (前日)<br>• allocated_capacity (累积)</font>", NT("#FFFDE7","#F57F17"), x=xn, y=890, w=200, h=65))
    c.append(C("fm5", "<b>7d. M5 调拨规划</b><br><font style='font-size:9px'>main(config, m1_dir, m4_dir, orchestrator,<br>  sim_date, output_dir)</font>", BB("#DCEDC8","#558B2F",11), x=xm-80, y=985, w=320, h=65))
    c.append(C("fm5o", "<font style='font-size:8px'>→ deployment_df<br>→ validation_log<br>→ allocation_detail</font>", BX("#E8F5E9","#558B2F",8), x=xo, y=985, w=200, h=55))
    c.append(C("fm5n", "<font style='font-size:8px'><b>输入：</b><br>• M1 shipment/orders<br>• M4 production<br>• orchestrator: inventory,<br>  open_deployment, in_transit,<br>  production_gr, delivery_gr</font>", NT("#FFFDE7","#F57F17"), x=xn, y=980, w=200, h=85))
    c.append(C("fm6", "<b>7e. M6 物流执行</b><br><font style='font-size:9px'>run_daily_physical_flow(config_dict,<br>  orchestrator, date, output_dir)</font>", BB("#F0F4C3","#827717",11), x=xm-80, y=1080, w=320, h=65))
    c.append(C("fm6o", "<font style='font-size:8px'>→ truck_load_df<br>→ delivery_df<br>→ physical_inventory_df</font>", BX("#E8F5E9","#827717",8), x=xo, y=1080, w=200, h=55))
    c.append(C("fm6n", "<font style='font-size:8px'><b>输入：</b><br>• M5 deployment_df<br>• orchestrator: inventory,<br>  open_deployment<br>• config: truck specs, MDQ</font>", NT("#FFFDE7","#F57F17"), x=xn, y=1075, w=200, h=80))
    c.append(C("fop", "<b>8. Orchestrator 状态更新</b><br><font style='font-size:9px'>run_daily_processing(date, shipment_df,<br>  production_df, deployment_df, delivery_df)<br>按序: M1发货→M4生产→M5调拨→M6交付</font>", HB("#FBE9E7","#D84315",11), x=xm-80, y=1175, w=320, h=80))
    c.append(C("fopn", "<font style='font-size:8px'><b>处理顺序：</b><br>① process_module1_shipments<br>② process_module4_production<br>③ process_module5_deployment<br>④ process_module6_delivery<br>⑤ cleanup_past_due()</font>", NT("#FBE9E7","#D84315"), x=xo, y=1170, w=220, h=95))
    c.append(C("fsv", "<b>9. 保存日状态</b><br><font style='font-size:9px'>save_daily_state(date) → CSV 快照</font>", BB("#E3F2FD","#1565C0",11), x=xm-60, y=1280, w=280, h=50))
    c.append(C("fdc", "<b>还有下一天?</b>", DM("#FFF3E0","#E65100"), x=xm-20, y=1355, w=200, h=80))
    c.append(C("fle", "<b>每日循环结束</b>", HB("#BBDEFB","#0D47A1",11), x=xm-20, y=1470, w=200, h=40))
    c.append(C("frp", "<b>10. 生成汇总报告</b><br><font style='font-size:9px'>SummaryReportGenerator.generate_all_reports()<br>8份报告: 订单/产能/换产/调拨/生产/交付/卡车/库存</font>", BB("#FCE4EC","#AD1457",11), x=xm-80, y=1530, w=320, h=70))
    c.append(C("fbc", "<b>11. 库存平衡验证</b><br><font style='font-size:9px'>InventoryBalanceChecker.check_all_dates()\n期初 + 流入 - 流出 = 期末</font>", BB("#E0F7FA","#00838F",11), x=xm-60, y=1620, w=280, h=60))
    c.append(C("fen", "<b>结束</b>", EL("#1565C0","#0D47A1"), x=xm, y=1710, w=160, h=50))
    c.append(C("ch1", "<b>输入依赖</b>", TX(13,"#F57F17",1), "1", x=xn, y=660, w=200, h=25))
    c.append(C("ch2", "<b>主执行流程</b>", TX(13,"#1565C0",1), "1", x=xm-20, y=660, w=200, h=25))
    c.append(C("ch3", "<b>模块输出</b>", TX(13,"#2E7D32",1), "1", x=xo, y=660, w=200, h=25))
    flow = [("f0","f1"),("f1","f2"),("f2","f3"),("f3","f4"),("f4","f5"),("f5","f6"),("f6","fls"),
            ("fls","fm1"),("fm1","fm3"),("fm3","fm4"),("fm4","fm5"),("fm5","fm6"),
            ("fm6","fop"),("fop","fsv"),("fsv","fdc"),("fle","frp"),("frp","fbc"),("fbc","fen")]
    for i,(s,t) in enumerate(flow):
        c.append(EG(f"ff{i}", "", AR("#1565C0"), s, t))
    c.append(EG("fdy", "是", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#2E7D32;fontSize=11;fontColor=#2E7D32;fontStyle=1;exitX=1;exitY=0.5;entryX=1;entryY=0.5;", "fdc", "fls"))
    c.append(EG("fdn", "否", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#C62828;fontSize=11;fontColor=#C62828;fontStyle=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;", "fdc", "fle"))
    for s,t in [("fm1","fm1o"),("fm3","fm3o"),("fm4","fm4o"),("fm5","fm5o"),("fm6","fm6o"),("fop","fopn")]:
        c.append(EG(f"fo_{s}", "", DA("#888888"), s, t))
    return PG("4. 每日仿真执行流程", "p4", c, 1550, 1800)


# ════════════════════════════════════════════════════════
#  PAGE 5: 模块间数据流
# ════════════════════════════════════════════════════════
def page5():
    c = []
    c.append(C("p5t", "<b>模块间数据流与状态管理</b>", TX(20,"#333",1), x=600, y=5, w=1200, h=35))
    c.append(C("dm1", "<b>M1 需求规划</b><br><font style='font-size:9px'>demand_planning/</font>", HB("#C8E6C9","#2E7D32",13), x=80, y=80, w=280, h=60))
    c.append(C("dm3", "<b>M3 MRP 计划</b><br><font style='font-size:9px'>mrp_planning/</font>", HB("#A5D6A7","#1B5E20",13), x=80, y=280, w=280, h=60))
    c.append(C("dm4", "<b>M4 生产计划</b><br><font style='font-size:9px'>production_planning/</font>", HB("#81C784","#33691E",13), x=80, y=480, w=280, h=60))
    c.append(C("dm5", "<b>M5 调拨规划</b><br><font style='font-size:9px'>deployment_planning/</font>", HB("#DCEDC8","#558B2F",13), x=80, y=680, w=280, h=60))
    c.append(C("dm6", "<b>M6 物流执行</b><br><font style='font-size:9px'>logistics_execution/</font>", HB("#F0F4C3","#827717",13), x=80, y=880, w=280, h=60))
    c.append(C("dor", "<b>Orchestrator (SSOT)</b>", HB("#FBE9E7","#D84315",14), x=700, y=350, w=500, h=420))
    orch_items = [
        ("dos1","unrestricted_inventory","Dict[(mat,loc)→qty]","非限制库存"),
        ("dos2","open_deployment","Dict[uid→record]","未完成调拨"),
        ("dos3","in_transit","Dict[uid→record]","在途库存"),
        ("dos4","production_gr","List[dict]","生产入库"),
        ("dos5","delivery_gr","List[dict]","交付入库"),
        ("dos6","shipment_log","List[dict]","发货日志"),
        ("dos7","production_plan_backlog","List[dict]","生产计划"),
        ("dos8","space_capacity","DataFrame","空间约束"),
    ]
    for i,(sid,name,typ,desc) in enumerate(orch_items):
        yy = 40 + i * 44
        c.append(C(sid, f"<font style='font-size:9px'><b>{name}</b><br><font color='#6a1b9a'>{typ}</font> — {desc}</font>", FI("#FFF3E0","#E65100"), "dor", 15, yy, 465, 38))
    c.append(C("df13", "<font style='font-size:8px'>orders_df<br>shipment_df<br>supply_demand_log_df</font>", BX("#E8F5E9","#2E7D32",8), x=400, y=145, w=160, h=55))
    c.append(EG("da13a", "", AR("#2E7D32"), "dm1", "df13"))
    c.append(EG("da13b", "", AR("#2E7D32"), "df13", "dm3"))
    c.append(C("df34", "<font style='font-size:8px'>net_demand_df<br>(按物料-地点-日期)</font>", BX("#C8E6C9","#1B5E20",8), x=400, y=360, w=160, h=40))
    c.append(EG("da34a", "", AR("#1B5E20"), "dm3", "df34"))
    c.append(EG("da34b", "", AR("#1B5E20"), "df34", "dm4"))
    c.append(C("df45", "<font style='font-size:8px'>production_df<br>产线产出</font>", BX("#AED581","#33691E",8), x=400, y=555, w=160, h=35))
    c.append(EG("da45a", "", AR("#33691E"), "dm4", "df45"))
    c.append(EG("da45b", "", AR("#33691E"), "df45", "dm5"))
    c.append(C("df56", "<font style='font-size:8px'>deployment_df<br>调拨计划</font>", BX("#DCEDC8","#558B2F",8), x=400, y=760, w=160, h=35))
    c.append(EG("da56a", "", AR("#558B2F"), "dm5", "df56"))
    c.append(EG("da56b", "", AR("#558B2F"), "df56", "dm6"))
    flows_to_orch = [
        ("dmo1","dm1","shipment_df → process_module1_shipments()","#2E7D32"),
        ("dmo4","dm4","production_df → process_module4_production()","#33691E"),
        ("dmo5","dm5","deployment_df → process_module5_deployment()","#558B2F"),
        ("dmo6","dm6","delivery_df → process_module6_delivery()","#827717"),
    ]
    for eid,src,lbl,clr in flows_to_orch:
        c.append(EG(eid, lbl, f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor={clr};fontSize=8;fontColor={clr};dashed=0;", src, "dor"))
    views_from_orch = [
        ("dvo3","dm3","unrestricted_inventory\nopen_deployment, in_transit\ndelivery_gr, production_plan_backlog","#D84315"),
        ("dvo5","dm5","unrestricted_inventory\nopen_deployment, in_transit\nproduction_gr, delivery_gr","#D84315"),
        ("dvo6","dm6","unrestricted_inventory\nopen_deployment","#D84315"),
    ]
    for eid,tgt,lbl,clr in views_from_orch:
        c.append(EG(eid, lbl, f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=1.5;strokeColor={clr};fontSize=7;fontColor={clr};dashed=1;dashPattern=6 4;exitX=0;exitY=0.5;", "dor", tgt))
    c.append(EG("dvo1", "unrestricted_inventory", f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=1.5;strokeColor=#D84315;fontSize=7;fontColor=#D84315;dashed=1;dashPattern=6 4;", "dor", "dm1"))
    c.append(C("cst", "<b>Excel 配置 Sheet → 模块映射</b>", CT("#F3E5F5","#7B1FA2",12), x=1350, y=80, w=550, h=500))
    sheets = [
        ("cs1","<b>全局配置</b>","SIT_Design, Global_seed, Global_Network,\nGlobal_SpaceCapacity, Global_LeadTime,\nGlobal_DemandPriority"),
        ("cs2","<b>M1 需求规划</b>","M1_InitialInventory, M1_DemandForecast,\nM1_ForecastError, M1_OrderCalendar,\nM1_AOConfig, M1_DPSConfig, M1_SupplyChoiceConfig"),
        ("cs3","<b>M3 MRP</b>","M3_SafetyStock, COValidation"),
        ("cs4","<b>M4 生产计划</b>","M4_MaterialLocationLineCfg, M4_LineCapacity,\nM4_ChangeoverMatrix, M4_ChangeoverDefinition,\nM4_ProductionReliability"),
        ("cs5","<b>M5 调拨规划</b>","M5_PushPullModel, M5_DeployConfig"),
        ("cs6","<b>M6 物流执行</b>","M6_TruckReleaseCon, M6_MaterialMD,\nM6_DeliveryDelayDistribution, M6_MDQBypassRules,\nM6_TruckTypeSpecs, M6_TruckCapacityPlan"),
    ]
    for i,(sid,title,desc) in enumerate(sheets):
        yy = 35 + i * 75
        c.append(C(sid, f"{title}<hr size='1'><font style='font-size:8px'>{desc.replace(chr(10),'<br>')}</font>", FI("#F3E5F5","#7B1FA2"), "cst", 15, yy, 515, 65))
    c.append(C("po", "<b>Orchestrator 日处理顺序</b>", CT("#FFCCBC","#BF360C",12), x=1350, y=610, w=550, h=300))
    steps = [
        ("po1","① process_module1_shipments(shipment_df)","扣减库存 → 记录 shipment_log","#2E7D32"),
        ("po2","② process_module4_production(production_df)","增加库存 → 记录 production_gr","#33691E"),
        ("po3","③ process_module5_deployment(deployment_df)","创建 open_deployment → 扣减发送方库存","#558B2F"),
        ("po4","④ process_module6_delivery(delivery_df)","关闭 open_deployment → 增加接收方库存","#827717"),
        ("po5","⑤ cleanup_past_due_open_deployments()","清理过期调拨 (grace_days)","#BF360C"),
    ]
    for i,(sid,title,desc,clr) in enumerate(steps):
        yy = 35 + i * 50
        c.append(C(sid, f"<font color='{clr}' style='font-size:10px'><b>{title}</b></font><br><font style='font-size:8px'>{desc}</font>", FI("#FFF3E0","#E65100"), "po", 15, yy, 515, 42))
    c.append(C("dfl1", "", f"rounded=0;fillColor=none;strokeColor=#2E7D32;strokeWidth=2;dashed=0;", x=1350, y=940, w=30, h=2))
    c.append(C("dfl1t", "<font style='font-size:9px'>━━ 数据输出 (写入)</font>", TL(9), x=1385, y=930, w=180, h=20))
    c.append(C("dfl2", "", f"rounded=0;fillColor=none;strokeColor=#D84315;strokeWidth=1.5;dashed=1;dashPattern=6 4;", x=1350, y=960, w=30, h=2))
    c.append(C("dfl2t", "<font style='font-size:9px'>╌╌ 状态视图 (只读)</font>", TL(9), x=1385, y=950, w=180, h=20))
    return PG("5. 模块间数据流", "p5", c, 1920, 990)


# ════════════════════════════════════════════════════════
#  PAGE 6: 配置系统与基础设施
# ════════════════════════════════════════════════════════
def page6():
    c = []
    c.append(C("p6t", "<b>配置系统·执行模式·优化基础设施</b>", TX(20,"#333",1), x=500, y=5, w=1600, h=35))
    c.append(C("s1c", "<b>配置系统 Config System</b>", CT("#F3E5F5","#7B1FA2",14), x=20, y=50, w=900, h=420))
    c.append(C("s1a", "<b>config/loader.py</b><hr size='1'><font style='font-size:9px'><b>load_config(force_reload=False)</b> → Dict[str, Any]<br>加载 default_config.yaml<br>深度合并 (deep merge) CHAINSIGHT_CONFIG 覆盖<br><br><b>get_config()</b> → 获取缓存配置<br><b>get_module_config(module_name)</b> → 模块配置<br><b>get_shared_config()</b> → 共享配置<br><b>reset_config_cache()</b> → 清除缓存</font>", FI("#F3E5F5","#7B1FA2"), "s1c", 20, 38, 400, 170))
    c.append(C("s1b", "<b>配置加载流程</b><hr size='1'><font style='font-size:9px'>① default_config.yaml (默认值)<br>  ↓ deep merge<br>② CHAINSIGHT_CONFIG 环境变量覆盖<br>  ↓ 缓存<br>③ get_config() 返回合并结果<br>  ↓<br>④ get_module_config('demand_planning')<br>   → config['modules']['demand_planning']</font>", FI("#F3E5F5","#7B1FA2"), "s1c", 440, 38, 430, 170))
    c.append(C("s1d", "<b>Excel 配置加载 (config_loader.py)</b><hr size='1'><font style='font-size:9px'>load_configuration(config_path) → config_dict<br>• 读取 BC_S5.xlsx 所有 Sheet<br>• 标准化列名 (去空格/统一格式)<br>• 转换数据类型 (日期/数值)<br>• 缺失值填充 (NaN → 0 / '')<br>• 返回 Dict[sheet_name → DataFrame]<br>→ 20+ DataFrames 供各模块使用</font>", FI("#F3E5F5","#7B1FA2"), "s1c", 20, 220, 850, 100))
    c.append(C("s1e", "<b>预校验 (config_validator.py)</b><hr size='1'><font style='font-size:9px'>run_pre_simulation_validation(config_path, output_dir)<br>• 检查必需 Sheet 是否存在<br>• 检查必需列是否完整<br>• 验证数据类型/范围<br>• M1-M6 模块专用规则验证<br>→ (passed: bool, report_path: str)</font>", FI("#F3E5F5","#7B1FA2"), "s1c", 20, 330, 850, 75))
    c.append(C("s2c", "<b>执行模式 Execution Modes</b>", CT("#E3F2FD","#1565C0",14), x=940, y=50, w=900, h=420))
    modes = [
        ("em1","File 模式 (开发/测试)","simulation_file.py",
         "输入: Excel 配置文件 (--config)\n输出: 模块日输出 (Excel), CSV 快照, 汇总报告\n适用: 本地开发·功能测试·单次运行\n流程: Excel → run_integrated_simulation() → 日循环 → 报告"),
        ("em2","DB 模式 (生产)","simulation_db.py",
         "输入: PostgreSQL 配置 (--db-config BC_S5)\n输出: 结果写入数据库 (可配置)\n适用: 生产环境·多轮运行·可扩展\n流程: DB配置 → DbRuntimeState → MemoryDataStore → 日循环 → DB写入"),
        ("em3","Parallel 模式 (性能优化)","parallel_executor/",
         "触发: CHAINSIGHT_PARALLEL=true (默认启用)\n并行: M1 ∥ M4 ∥ M5 通过 ThreadPoolExecutor\n同步点: 并行阶段完成后 → M6 (依赖 M5)\n回退: CHAINSIGHT_PARALLEL=false 串行执行"),
    ]
    for i,(eid,title,entry,desc) in enumerate(modes):
        yy = 38 + i * 125
        c.append(C(eid, f"<b>{title}</b> <font color='#888' style='font-size:9px'>({entry})</font><hr size='1'><font style='font-size:9px'>{desc.replace(chr(10),'<br>')}</font>", FI("#E3F2FD","#1565C0"), "s2c", 20, yy, 855, 115))
    c.append(C("s3c", "<b>DuckDB 加速基础设施</b>", CT("#B2EBF2","#00838F",14), x=20, y=490, w=900, h=380))
    c.append(C("dk1", "<b>duckdb_sql_wrapper.py (共享工具)</b><hr size='1'><font style='font-size:9px'>class DuckDBSQL:<br>  @classmethod merge(left, right, on, how) — 50k+ 行加速<br>  @classmethod groupby_agg(df, group, agg) — 10k+ 行加速<br>  @classmethod filter(df, where) — 5k+ 行加速<br>  @classmethod sort(df, cols) — 100k+ 行加速<br><br>策略: 自动选择 DuckDB 或 Pandas (按数据量)</font>", FI("#E0F7FA","#00838F"), "s3c", 20, 38, 420, 140))
    c.append(C("dk2", "<b>memory_data_store.py</b><hr size='1'><font style='font-size:9px'>class MemoryDataStore (Singleton):<br>  _conn: duckdb.DuckDBPyConnection (内存DB)<br>  enable(memory_limit, threads)<br>  write_module_output(module, table, date, df)<br>  read_module_output(module, table, date) → df<br>  clear_all()<br><br>用途: DB模式零磁盘IO数据传递 (50-100× faster)</font>", FI("#E0F7FA","#00838F"), "s3c", 460, 38, 420, 140))
    c.append(C("dk3", "<b>模块专用 DuckDB 加速器</b><hr size='1'><font style='font-size:9px'>• mrp_planning/duckdb_batch_calculator.py — 大规模层级计算<br>• production_planning/duckdb_batch_calculator.py — 产能批量计算<br>• deployment_planning/duckdb_batch_calculator.py — 大规模分配<br>• deployment_planning/horizon_batch_calculator.py — 多时间窗口批量<br>• logistics_execution/ — 物流优化加速</font>", FI("#E0F7FA","#00838F"), "s3c", 20, 190, 860, 100))
    c.append(C("dk4", "<b>优化配置</b><hr size='1'><font style='font-size:9px'>• optimization_config.py — OptimizationConfig 全局优化开关<br>• resource_config.py — 系统内存自动检测 → DuckDB memory_limit<br>• cpu_config.py — CPU 核数检测 → get_optimal_workers() → ThreadPool</font>", FI("#E0F7FA","#00838F"), "s3c", 20, 300, 860, 65))
    c.append(C("s4c", "<b>服务与工具支撑</b>", CT("#FCE4EC","#AD1457",14), x=940, y=490, w=900, h=380))
    c.append(C("sv1", "<b>SummaryReportGenerator</b><hr size='1'><font style='font-size:9px'>generate_all_reports(start_date, end_date) → 8份报告:<br>① 订单/发货/缺料汇总 ② 产能超额报告<br>③ 换产报告 ④ 调拨计划汇总<br>⑤ 生产计划汇总 ⑥ 交付计划汇总<br>⑦ 卡车使用报告 ⑧ 历史库存报告</font>", FI("#FCE4EC","#AD1457"), "s4c", 20, 38, 420, 110))
    c.append(C("sv2", "<b>PerformanceProfiler</b><hr size='1'><font style='font-size:9px'>cProfile 上下文管理器<br>with PerformanceProfiler('Module3', dir):<br>  run_module3()<br>→ 自动采集并输出性能报告</font>", FI("#FCE4EC","#AD1457"), "s4c", 460, 38, 410, 110))
    c.append(C("sv3", "<b>ValidationManager</b><hr size='1'><font style='font-size:9px'>add_error(module, category, message)<br>add_warning() / add_info()<br>write_report() → validation_report.txt<br>has_errors() / get_error_count()</font>", FI("#FCE4EC","#AD1457"), "s4c", 20, 158, 420, 95))
    c.append(C("sv4", "<b>SimulationTimeManager</b><hr size='1'><font style='font-size:9px'>统一日期处理 (消除时区/格式不一致)<br>get_current_date() / get_date_string()<br>get_previous_date(days) / advance_date()</font>", FI("#FCE4EC","#AD1457"), "s4c", 460, 158, 410, 95))
    c.append(C("sv5", "<b>InventoryBalanceChecker</b><hr size='1'><font style='font-size:9px'>check_all_dates(start, end) → bool<br>验证: 期初库存 + 流入 - 流出 = 期末库存<br>逐 (物料, 地点, 日期) 检查</font>", FI("#FCE4EC","#AD1457"), "s4c", 20, 263, 420, 75))
    c.append(C("sv6", "<b>normalization.py</b><hr size='1'><font style='font-size:9px'>normalize_location() / normalize_material()<br>normalize_identifiers()<br>统一标识符格式 (各模块 re-export)</font>", FI("#FCE4EC","#AD1457"), "s4c", 460, 263, 410, 75))
    return PG("6. 配置系统与基础设施", "p6", c, 1860, 890)


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
def main():
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<mxfile host="app.diagrams.net" type="device" version="24.7.0">\n'
    xml += page1() + '\n'
    xml += page2() + '\n'
    xml += page3() + '\n'
    xml += page4() + '\n'
    xml += page5() + '\n'
    xml += page6() + '\n'
    xml += '</mxfile>\n'
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ChainSight_重构后框架设计图.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Generated: {out}")
    print(f"Size: {len(xml):,} chars, {xml.count('<mxCell'):,} cells")

if __name__ == "__main__":
    main()
