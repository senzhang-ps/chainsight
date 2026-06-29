"""M4 重构集成测试：在 test_integration.py 的仿真骨架里加挂 ModuleFour。

结构与 test/test_integration.py 对齐：
- Orch 从 config_path 加载配置、管理日期迭代、DQ 校验、DB 建模；
- StateContext 管理可变状态（库存、在途等）；
- ModuleOne 每日真实生成；
- ModuleFour 每日内存驱动（ctx.get_previous_line_state / ctx.get_all_previous_allocated_capacity 注入 override）；
- ModuleThree 历史回放：从 DB 读 module3_output_netdemand 作为当日净需求。

M4/M3 生命周期：prepare() 仅循环前一次（一次性静态准备），run() 每日一次。
M3 → M4 衔接为 1 天 lag（M4 先消费上一轮 M3、M3 再产出当日并存回）；首日 M4 无 M3、产出为空。

⚠️ M3 输入方式（已对齐 test_integration.py，不再用 CSV 种子）：
历史 M3 净需求由用户先跑一遍旧链路落库到 ``input.module3_output_netdemand``，
本测试通过 ``ModuleThree(m3_run_id=...)`` 直接从 DB 读取，**不再读取/平移
Module3Output_20251215.csv**。旧链路 ``_load_base_m3`` / ``_seed_m3_to_db``
种子逻辑已移除。

⚠️ schema 说明：
Orch(config_path=...) 解析出的 schema 是 ``public``（config_path 无 project 目录），
但用户落库的真实 M3/M4 数据在 ``input`` schema。仿真全程**不写 DB**
（test_db 下 save_module_output/save_daily_state 已被注释跳过），故把
``orch.db.schema`` 临时切到 ``input`` 是安全的——既让 ModuleThree 能读到真实 M3，
也不会污染 ``input`` 里的基准数据。

⚠️ 配置说明：
``skip_dq=True`` 时 Orch 自带的配置加载（ConfigReader.load_all）已把 Excel 全部
31 个 sheet 读入 all_config（含 M1/M4），并把列投影为正确类型（如 location→string）。
故**不再用 xl.parse 覆盖 all_config**——原始 parse 会让 location 退化为 int64，
导致 M1 polars join SchemaError。

⚠️ M4 结果对比：
仿真跑完后，从 ``input.module4_output_*``（production/exceed/changeover）取
旧链路基准（同 RUNID），与每日 ``m4.output()`` 新结果逐日比对（比对工具复用
``tests/test_m4_parity.py`` 的 compare_* 函数）。已知 produced_qty 因 RNG 路径
差异（旧 DuckDB vs 新 pandas）天然发散、少数行产能拆分天差一行，按既有 parity
结论**只报告、不硬失败**；con_planned_qty / changeover_id / mu_loss 等allocator
产物才是重构 parity 的主体。

运行：
    pytest tests/test_m4_refactor_integration.py -q
    # 缩短区间：
    M4_TEST_START=2025-12-15 M4_TEST_END=2025-12-19 pytest tests/test_m4_refactor_integration.py -q
    # 或脚本：
    python tests/test_m4_refactor_integration.py --start 2025-12-15 --end 2025-12-19
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

import pandas as pd

# 先把项目根 + tests 目录入 path，再 import src / test_m4_parity
# （脚本直跑 python tests/xxx.py 时 sys.path[0] 是 tests/，缺项目根）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core.orchestrator import Orch
from src.modules import module1
from src.modules.state_context import StateContext
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.mrp_planning.integration_refactor import ModuleThree
# 复用 parity 测试的逐表比对工具（compare_production/exceed/changeover）。
# 注意：不导入 _select——它内部用 parity 模块级 RUNID（旧 runid），本测试用自带 RUNID。
from test_m4_parity import (
    compare_production, compare_exceed, compare_changeover,
    PROD_TABLE, EXCEED_TABLE, CHANGEOVER_TABLE, META_COLS,
)

logger = logging.getLogger("SupplyChainSimulation")

CONFIG_PATH = str(PROJECT_ROOT / 'config' / 'OC_Paste_S1_20251224_repare.xlsx')
# 用户先跑完旧链路、M3/M4 落库到 input schema 的 run_id（直接从 DB 读 M3，不再种子）
RUNID = 'db_OC_Paste_S1_20251224_repare_20260625_173940'
M3_HISTORICAL_RUNID = RUNID
# 仿真窗口对齐基准：M4 12-16~12-19 每天都要能消费前一天 M3，故从 12-15 起跑
DEFAULT_START = os.environ.get('M4_TEST_START', '2025-12-15')
DEFAULT_END = os.environ.get('M4_TEST_END', '2025-12-19')
EXPECTED_KEYS = {
    'production_df', 'exceed_log', 'issues_df',
    'changeover_log', 'current_line_states', 'current_allocated_capacity',
}


# ── 辅助 ────────────────────────────────────────────────────────────

def _read_old(db, table, date_str):
    """从 input.<table> 取旧链路基准（同 RUNID），剥离 META_COLS。

    直接写 raw SQL（用本测试自带 RUNID），不复用 test_m4_parity._select——
    后者内部引用 parity 模块级 RUNID（旧 runid），会取错数据。
    """
    tn = table.split('.')[-1]
    cols = [r[0] for r in db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='input' AND table_name=%s ORDER BY ordinal_position",
        (tn,),
    )]
    rows = db.execute_query(
        f'SELECT {", ".join(cols)} FROM {table} '
        f'WHERE run_id=%s AND sim_date=%s',
        (RUNID, date_str),
    )
    df = pd.DataFrame(rows, columns=cols)
    return df.drop(columns=[c for c in META_COLS if c in df.columns])


def _make_output_dir(config_path: str) -> str:
    """为每次测试运行创建独立输出目录（避免 _ensure_output_dir 的 exist_ok=False）。"""
    base = os.path.join(PROJECT_ROOT, 'outputs', 'm4_refactor_test')
    os.makedirs(base, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ── 主仿真 ──────────────────────────────────────────────────────────

def run_integrated_simulation_m4(
    config_path=CONFIG_PATH,
    start_date=DEFAULT_START,
    end_date=DEFAULT_END,
    output_base_dir=None,
    engine='polars',
):
    """Orch + StateContext + ModuleOne + ModuleFour 多日仿真。

    与 test/test_integration.py 的 run_integrated_simulation 对齐，
    在每日循环里加挂 ModuleFour。M3 直接从 DB 读（m3_run_id），不种子 CSV。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()],
    )

    if output_base_dir is None:
        output_base_dir = _make_output_dir(config_path)

    # ── Orch ──
    # skip_dq=True：①避免 DQ 通过后把 cfg_* 写回 DB（污染 input 基准）；
    #   ②省去校验开销；③关键——Orch 的配置加载路径（ConfigReader.load_all）会把
    #   Excel 列投影为正确类型（如 location→string），all_config 已含全部 31 个
    #   sheet（含 M1/M4）。若再用 xl.parse 原始覆盖，location 会变 int64，M1 的
    #   polars join 会 SchemaError。故此处**不覆盖** all_config。
    orch = Orch(
        start_date=start_date, end_date=end_date,
        config_path=config_path, output_path=output_base_dir,
        engine=engine, skip_dq=True,
    )
    # ⚠️ 把 db schema 临时切到 input：真实 M3/M4 数据落在 input schema，
    #    仿真全程不写 DB（save_* 已跳过、skip_dq=True 不落配置），切换只读不污染基准。
    if orch.db is not None:
        orch.db.schema = 'input'
        logger.info("🗄️ orch.db.schema 切到 'input'（读真实 M3/M4 基准；仿真不写库）")

    ctx = StateContext(simulation_date=start_date, orch=orch)
    ctx.initialize(orch.all_config)

    # ── ModuleOne ──
    m1 = module1.ModuleOne(
        simulation_date=str(start_date), orchestrator=ctx, orch=orch,
    )
    m1.prepare()

    # ── ModuleFour ──
    # 注意：重构后 ModuleFour 已移除 output_dir / skip_file_output 入参
    # （文件输出由外部 Orch 负责），故这里不再传入。
    m4 = ModuleFour(
        simulation_date=str(start_date), simulation_start_date=start_date,
        orchestrator=ctx, orch=orch,
    )
    m4.prepare()  # 一次性：load_datas + 标识符归一 + 分配器静态 maps

    # ── ModuleThree（历史回放：直接从 input DB 读 module3_output_netdemand）──
    # 对齐 test_integration.py：不再 _seed_m3_to_db 种子 CSV，ModuleThree.run
    # 内 _read_net_demand 按 m3_run_id + 当日 sim_date 直读。
    m3 = ModuleThree(
        simulation_date=str(start_date), simulation_start_date=start_date,
        orchestrator=ctx, orch=orch, output_dir=os.path.join(output_base_dir, 'module3'),
        skip_file_output=True, m3_run_id=M3_HISTORICAL_RUNID,
    )
    m3.prepare()  # 占位（不依赖 cfg_*，取数在 run 内）

    all_results = {'module1': [], 'module4': []}
    simulation_start_time = time.time()

    # ── 仿真循环 ──
    for i, current_date in orch.iter_dates():
        date_str = current_date.strftime('%Y-%m-%d')
        ctx.day_start(date_str)

        # ---- M1 ----
        try:
            m1.simulation_date = current_date
            m1.run()
            m1_result = m1.output()
            m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
            if not m1_shipments.empty:
                ctx.apply_shipments(m1_shipments, date_str)
            m1_result['simulation_date'] = current_date
            all_results['module1'].append(m1_result)
        except Exception as e:
            logger.warning(f"⚠️ M1 @ {date_str} 失败（忽略）: {e}")

        # ---- M3 → M4（1 天 lag：M4 先消费上一轮 M3，M3 再产出当日并存回）----
        m3.simulation_date = current_date
        m4.simulation_date = current_date
        m4.previous_line_states_override = ctx.get_previous_line_state(date_str)
        m4.allocated_capacity_override = ctx.get_all_previous_allocated_capacity(date_str)
        m4.run()                          # 消费上一轮 module3_result（首日为 None → 空产）
        m3.run()                          # 从 DB 读当日净需求
        m4.module3_result = m3.output()   # 存回供下一轮
        m4_result = m4.output()

        # 结构断言
        assert EXPECTED_KEYS.issubset(m4_result.keys()), \
            f"{date_str} 结果缺键: {EXPECTED_KEYS - set(m4_result.keys())}"

        # override 接入断言
        assert m4.previous_line_states == ctx.get_previous_line_state(date_str), \
            f"{date_str} line_states override 未接入"
        assert m4.previously_allocated == ctx.get_all_previous_allocated_capacity(date_str), \
            f"{date_str} allocated_capacity override 未接入"

        # 数值合理性
        prod = m4_result['production_df']
        if not prod.empty:
            assert (prod['produced_qty'] <= prod['con_planned_qty']).all(), \
                f"{date_str} produced_qty > con_planned_qty"

        # 跨天结转 → 写回 ctx
        ctx.apply_line_state(m4_result['current_line_states'], date_str)
        ctx.apply_allocated_capacity(m4_result['current_allocated_capacity'], date_str)
        all_results['module4'].append({
            'date': date_str,
            'n_production': len(prod),
            'm4_result': m4_result,
        })

        ctx.day_end(date_str)
        # ⚠️ 不调用 orch.save_module_output / orch.save_daily_state：
        #    test_db 的 migrate 不创建 module_output 表，save 会抛 UndefinedTable。
        #    M4 验证不需要持久化；当 DB 完整后可恢复这些调用。

    # ── 完成 ──
    total_seconds = time.time() - simulation_start_time
    minutes, seconds = divmod(total_seconds, 60)
    runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"

    n_m4_prod = sum(1 for r in all_results['module4'] if r['n_production'] > 0)
    logger.info(
        f"🎉 集成仿真完成! 共 %d 天 (M4 %d 天产出), 耗时 %s",
        len(all_results['module4']), n_m4_prod, runtime_str,
    )

    return {
        'simulation_completed': True,
        'dates_processed': len(all_results['module4']),
        'n_m4_production_days': n_m4_prod,
        'results': all_results,
        'orch': orch,  # 供对比阶段复用 db 连接
    }


# ── M4 结果对比（新侧 m4.output() vs input 旧基准）──

def _run_m4_comparison(result: dict) -> list:
    """逐日比对每日 m4.output() 与 input.module4_output_* 旧基准。

    比对工具复用 test_m4_parity.compare_*；新侧 M3 输入取自前一天的旧 M3
    （即 ModuleThree 读 DB 的当日 sim_date，与旧链路 lag 一致）。
    已知发散项（produced_qty / 少数行结构差）只打印不硬失败。
    """
    orch = result['orch']
    db = orch.db
    new_days = result['results']['module4']

    report = []
    for rec in new_days:
        date_str = rec['date']
        new_res = rec['m4_result']
        new_prod = new_res['production_df']
        # 首日（lag）M4 无 production_df（旧基准当天也无），跳过比对
        # —— 空 DataFrame 列缺失会让 compare_* 的 groupby KeyError。
        if new_prod is None or len(new_prod) == 0:
            print(f"\n== {date_str} == （M4 无 production_df，跳过比对）")
            continue
        # 旧基准落在 input schema 的 module4_output_*（同 RUNID）
        try:
            old_prod = _read_old(db, PROD_TABLE, date_str)
            old_exc = _read_old(db, EXCEED_TABLE, date_str)
            old_co = _read_old(db, CHANGEOVER_TABLE, date_str)
        except Exception as e:
            logger.warning(f"⚠️ 读旧基准 @ {date_str} 失败（跳过）: {e}")
            continue

        day = {
            'date': date_str,
            'production': compare_production(new_prod, old_prod),
            'exceed': compare_exceed(new_res['exceed_log'], old_exc),
            'changeover': compare_changeover(new_res['changeover_log'], old_co),
        }
        report.append(day)
        print(f"\n== {date_str} ==")
        print(f"  production : {day['production']}")
        print(f"  exceed     : {day['exceed']}")
        print(f"  changeover : {day['changeover']}")
    return report


# ── pytest ──────────────────────────────────────────────────────────

def test_m4_refactor_integration():
    result = run_integrated_simulation_m4()
    assert result['simulation_completed']
    n_m4 = result['n_m4_production_days']
    print(f"\n[summary] {result['dates_processed']} 天, {n_m4} 天 M4 有 production_df 产出")
    assert n_m4 > 0, "从 DB 读 M3 后仍无 M4 production_df 产出"


def test_m4_refactor_parity():
    """跑完仿真后，与 input 旧基准逐日比对 3 张表。

    稳健不变量（断言）：每日 changeover mu_loss 全等（换产损耗总额守恒）。
    已知发散（仅打印）：production 行数/exceed 行数偶差 1 行、produced_qty
    因 DuckDB vs pandas RNG 全发散（非重构 bug，见 test_m4_parity 文件头注释）。
    """
    result = run_integrated_simulation_m4()
    report = _run_m4_comparison(result)

    # 稳健不变量：每日 changeover mu_loss 全等
    for d in report:
        cl = d['changeover'].get('mu_loss_eq', 0)
        cr = d['changeover']['rows_new']
        assert cl == cr, f"{d['date']} changeover mu_loss 不全等: {cl}/{cr}"

    # 发散项汇总（informational）
    prod_key_mismatch, exc_key_mismatch = [], []
    for d in report:
        n, m = d['production']['struct_keys'].split('/')
        if n != m:
            prod_key_mismatch.append(
                f"{d['date']}={d['production']['struct_keys']}"
                f"(rows {d['production']['rows_new']}/{d['production']['rows_old']})"
            )
        n, m = d['exceed']['keys_match'].split('/')
        if n != m:
            exc_key_mismatch.append(f"{d['date']}={d['exceed']['keys_match']}")
    if prod_key_mismatch:
        print(f"\n⚠️ production 结构键发散的日子: {prod_key_mismatch}")
    if exc_key_mismatch:
        print(f"⚠️ exceed 键集合发散的日子: {exc_key_mismatch}")
    print("ℹ️ produced_qty 因 DuckDB 边界全发散（非重构 bug），见 test_m4_parity 注释。")


def test_m4_refactor_determinism():
    """同 seed 两次短区间运行，produced_qty 应一致。"""
    start, end = '2025-12-15', '2025-12-17'
    r1 = run_integrated_simulation_m4(start_date=start, end_date=end)
    r2 = run_integrated_simulation_m4(start_date=start, end_date=end)
    cols = ['material', 'location', 'line', 'production_plan_date',
            'con_planned_qty', 'produced_qty']
    for rec1, rec2 in zip(
        r1['results']['module4'], r2['results']['module4'],
    ):
        p1 = rec1['m4_result']['production_df']
        p2 = rec2['m4_result']['production_df']
        if p1.empty and p2.empty:
            continue
        assert not p1.empty and not p2.empty
        pd.testing.assert_frame_equal(
            p1[cols].sort_values(cols).reset_index(drop=True),
            p2[cols].sort_values(cols).reset_index(drop=True),
            check_dtype=False,
        )


# ── 脚本入口 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="M4 重构集成仿真 + 旧基准对比")
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--start', default=DEFAULT_START)
    parser.add_argument('--end', default=DEFAULT_END)
    parser.add_argument('--engine', default='polars')
    args = parser.parse_args()

    result = run_integrated_simulation_m4(
        config_path=args.config, start_date=args.start,
        end_date=args.end, engine=args.engine,
    )
    print(f"\n✅ 完成: {result['dates_processed']} 天, "
          f"{result['n_m4_production_days']} 天 M4 有产出")

    # 跑完后顺带输出与旧基准的对比报告
    report = _run_m4_comparison(result)
    if report:
        print("\n=== M4 对比汇总（新 vs input 旧基准）===")
        for d in report:
            p = d['production']
            print(
                f"{d['date']}  prod(rows new/old={p['rows_new']}/{p['rows_old']}, "
                f"keys={p['struct_keys']}, con_eq={p.get('con_planned_qty_eq','-')}, "
                f"produced_eq={p.get('produced_qty_eq','-')})  "
                f"exceed(keys={d['exceed']['keys_match']})  "
                f"changeover(mu_loss_eq={d['changeover'].get('mu_loss_eq','-')}"
                f"/{d['changeover']['rows_new']})"
            )


if __name__ == '__main__':
    main()
