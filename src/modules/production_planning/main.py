"""
主模块

提供Module4的主要入口函数和命令行接口。
"""

import os
import argparse
from typing import Optional

import pandas as pd

from ...utils.normalization import normalize_identifiers
from ...utils.defaults import DEFAULT_RANDOM_SEED
from .config_loader import load_config, validate_config
from .demand_loader import load_daily_net_demand
from .plan_builder import build_unconstrained_plan_for_single_day
from .capacity_allocator import (
    centralized_capacity_allocation_with_changeover,
    extract_allocated_capacity_from_plan,
    extract_line_states_from_plan,
    validate_capacity_allocation,
    calculate_changeover_metrics,
    simulate_production,
)
from .state_manager import (
    get_or_init_simulation_start,
    load_line_state,
    save_line_state,
    load_all_previous_capacity,
    save_allocated_capacity,
)
from .output_writer import write_output
from .utils import cast_identifiers_to_str, dedup_issues


def run_daily_production_planning(
    config_file: str,
    module3_output_dir: str,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    output_dir: str
) -> str:
    """运行单日生产计划。

    主要功能包括：加载配置与净需求、构建无约束计划、
    分配产能与仿真生产、保存状态与产能、写出当日输出。

    参数：
        config_file: M4配置Excel路径
        module3_output_dir: Module3每日输出目录
        simulation_date: 当前仿真日期
        simulation_start: 仿真起始日期
        output_dir: 输出目录

    返回：
        str: 生成的每日输出文件路径

    异常：
        Exception: 执行失败时
    """
    try:
        planner = DailyProductionPlanner(
            config_file=config_file,
            module3_output_dir=module3_output_dir,
            simulation_date=simulation_date,
            simulation_start=simulation_start,
            output_dir=output_dir
        )
        return planner.run()
    except Exception as e:
        date_str = simulation_date.strftime('%Y-%m-%d')
        raise


class DailyProductionPlanner:
    """每日生产计划器类。

    封装单日生产计划的完整执行流程。
    """

    def __init__(
        self,
        config_file: str,
        module3_output_dir: str,
        simulation_date: pd.Timestamp,
        simulation_start: pd.Timestamp,
        output_dir: str
    ):
        """初始化计划器。

        参数：
            config_file: 配置文件路径
            module3_output_dir: Module3输出目录
            simulation_date: 仿真日期
            simulation_start: 仿真起始日期
            output_dir: 输出目录
        """
        self.config_file = config_file
        self.module3_output_dir = module3_output_dir
        self.simulation_date = simulation_date
        self.simulation_start = simulation_start
        self.output_dir = output_dir
        self.issues = []

    def run(self) -> str:
        """执行生产计划流程。

        返回：
            str: 输出文件路径
        """
        cfg = self._load_and_validate_config()
        net_demand = self._load_net_demand()
        mlcfg = self._prepare_config(cfg)

        uncon_plan = self._build_unconstrained_plan(
            net_demand, mlcfg
        )

        plan_log, exceed_log = self._allocate_capacity(
            uncon_plan, cfg, mlcfg
        )

        plan_log = self._simulate_and_finalize(plan_log, cfg, mlcfg)
        changeover_log = calculate_changeover_metrics(
            plan_log, cfg['ChangeoverDefinition']
        )

        self._save_states(plan_log, cfg, mlcfg)

        return self._write_output(plan_log, exceed_log, changeover_log)

    def _load_and_validate_config(self) -> dict:
        """加载并校验配置。

        返回：
            dict: 配置字典
        """
        cfg = load_config(self.config_file)
        self.issues.extend(validate_config(cfg))
        return cfg

    def _load_net_demand(self) -> pd.DataFrame:
        """加载净需求数据。

        返回：
            pd.DataFrame: 净需求数据
        """
        net_demand = load_daily_net_demand(
            self.module3_output_dir,
            self.simulation_date
        )
        net_demand = cast_identifiers_to_str(
            net_demand, ['material', 'location']
        )

        if not net_demand.empty and 'requirement_date' in net_demand.columns:
            net_demand['requirement_date'] = pd.to_datetime(
                net_demand['requirement_date']
            )

        return net_demand

    def _prepare_config(self, cfg: dict) -> pd.DataFrame:
        """准备配置数据。

        参数：
            cfg: 配置字典

        返回：
            pd.DataFrame: 物料地点产线配置
        """
        mlcfg = cfg['MaterialLocationLineCfg']

        mlcfg = normalize_identifiers(mlcfg)

        return mlcfg

    def _build_unconstrained_plan(
        self,
        net_demand: pd.DataFrame,
        mlcfg: pd.DataFrame
    ) -> pd.DataFrame:
        """构建无约束计划。

        参数：
            net_demand: 净需求数据
            mlcfg: 配置

        返回：
            pd.DataFrame: 无约束计划
        """
        net_demand = normalize_identifiers(net_demand)

        return build_unconstrained_plan_for_single_day(
            net_demand, mlcfg,
            self.simulation_date, self.simulation_start,
            self.issues
        )

    def _allocate_capacity(
        self,
        uncon_plan: pd.DataFrame,
        cfg: dict,
        mlcfg: pd.DataFrame
    ) -> tuple:
        """分配产能。

        参数：
            uncon_plan: 无约束计划
            cfg: 配置字典
            mlcfg: 物料地点产线配置

        返回：
            tuple: (计划日志, 超额日志)
        """
        previous_states = load_line_state(
            self.output_dir, self.simulation_date
        )
        previous_capacity = load_all_previous_capacity(
            self.output_dir, self.simulation_date
        )

        co_mat = self._build_changeover_matrix(cfg)
        co_def = cfg['ChangeoverDefinition'].set_index(
            ['changeover_id', 'line']
        )['time'].to_dict()

        cap_df = cfg['LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])

        rate_map = mlcfg.set_index(
            ['material', 'location', 'delegate_line']
        )['prd_rate']
        rate_map.index.set_names(['material', 'location', 'line'], inplace=True)

        return centralized_capacity_allocation_with_changeover(
            uncon_plan, cap_df, rate_map, co_mat, co_def, mlcfg,
            previous_line_states=previous_states,
            simulation_date=self.simulation_date,
            previously_allocated_capacity=previous_capacity,
            issues=self.issues
        )

    def _build_changeover_matrix(self, cfg: dict) -> pd.Series:
        """构建换产矩阵。

        参数：
            cfg: 配置字典

        返回：
            pd.Series: 换产矩阵
        """
        co_mat_df = cfg['ChangeoverMatrix'].copy()
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)

        co_mat = co_mat_df.set_index(
            ['from_material', 'to_material']
        )['changeover_id']

        return co_mat.sort_index()

    def _simulate_and_finalize(
        self,
        plan_log: pd.DataFrame,
        cfg: dict,
        mlcfg: pd.DataFrame
    ) -> pd.DataFrame:
        """仿真生产并添加校验。

        参数：
            plan_log: 计划日志
            cfg: 配置字典
            mlcfg: 配置

        返回：
            pd.DataFrame: 最终计划日志
        """
        seed = cfg.get('RandomSeed', DEFAULT_RANDOM_SEED)
        plan_log = simulate_production(
            plan_log, cfg['ProductionReliability'], seed=seed
        )

        rate_map = mlcfg.set_index(
            ['material', 'location', 'delegate_line']
        )['prd_rate'].to_dict()
        co_def = cfg['ChangeoverDefinition'].set_index(
            ['changeover_id', 'line']
        )['time'].to_dict()

        previous_capacity = load_all_previous_capacity(
            self.output_dir, self.simulation_date
        )

        validation_issues = validate_capacity_allocation(
            plan_log, previous_capacity,
            self.simulation_date, rate_map, co_def
        )
        self.issues.extend(validation_issues)

        return plan_log

    def _save_states(
        self,
        plan_log: pd.DataFrame,
        cfg: dict,
        mlcfg: pd.DataFrame
    ) -> None:
        """保存状态信息。

        参数：
            plan_log: 计划日志
            cfg: 配置字典
            mlcfg: 配置
        """
        cap_df = cfg['LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])

        co_def = cfg['ChangeoverDefinition'].set_index(
            ['changeover_id', 'line']
        )['time'].to_dict()

        rate_map = mlcfg.set_index(
            ['material', 'location', 'delegate_line']
        )['prd_rate'].to_dict()

        line_states = extract_line_states_from_plan(
            plan_log, cap_df, co_def,
            self.simulation_date, rate_map
        )
        if line_states:
            save_line_state(
                self.output_dir, self.simulation_date, line_states
            )

        allocated_capacity = extract_allocated_capacity_from_plan(
            plan_log, rate_map, co_def
        )
        if allocated_capacity:
            save_allocated_capacity(
                self.output_dir, self.simulation_date, allocated_capacity
            )

    def _write_output(
        self,
        plan_log: pd.DataFrame,
        exceed_log: pd.DataFrame,
        changeover_log: pd.DataFrame
    ) -> str:
        """写出输出文件。

        参数：
            plan_log: 计划日志
            exceed_log: 超额日志
            changeover_log: 换产日志

        返回：
            str: 输出文件路径
        """
        self.issues = dedup_issues(self.issues)

        base_output = os.path.join(self.output_dir, "Module4Output.xlsx")

        output_path = write_output(
            plan_log, exceed_log, self.issues, changeover_log,
            base_output, self.simulation_date
        )

        self._check_critical_issues()

        return output_path

    def _check_critical_issues(self) -> None:
        """检查关键问题。"""
        critical = [
            x for x in self.issues
            if any(key in str(x.get('issue', ''))
                   for key in ['No line config', 'Multiple eligible lines'])
        ]

        if critical:
            date_str = self.simulation_date.strftime('%Y-%m-%d')
            msg = (
                f'发现关键校验错误 {date_str}! '
                f'请查看Validation工作表。'
            )


def main():
    """命令行入口。

    支持日度模式和旧版兼容模式。
    """
    parser = _create_argument_parser()
    args = parser.parse_args()

    try:
        if args.mode == 'daily':
            _run_daily_mode(args)
        else:
            _run_legacy_mode(args)
    except Exception as e:
        raise


def _create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    返回：
        argparse.ArgumentParser: 参数解析器
    """
    parser = argparse.ArgumentParser(
        description='Module 4: APS工业生产仿真（支持每日执行）'
    )

    parser.add_argument(
        '--config', required=True,
        help='配置Excel文件路径'
    )
    parser.add_argument(
        '--mode', choices=['daily', 'legacy'], default='daily',
        help='执行模式'
    )

    # 日常模式参数
    parser.add_argument(
        '--module3_output_dir',
        help='Module3每日输出目录（daily模式）'
    )
    parser.add_argument(
        '--simulation_date',
        help='仿真日期 YYYY-MM-DD（daily模式）'
    )
    parser.add_argument(
        '--simulation_start',
        help='仿真起始日期 YYYY-MM-DD（daily模式，首次运行必需）'
    )
    parser.add_argument(
        '--output_dir',
        help='输出目录（daily模式）'
    )

    # 旧版模式参数
    parser.add_argument(
        '--input',
        help='Legacy: 输入文件路径'
    )
    parser.add_argument(
        '--sim_start',
        help='Legacy: 仿真起始日期'
    )
    parser.add_argument(
        '--sim_end',
        help='Legacy: 仿真结束日期'
    )
    parser.add_argument(
        '--output',
        help='Legacy: 输出文件路径'
    )

    return parser


def _run_daily_mode(args) -> None:
    """运行日度模式。

    参数：
        args: 命令行参数
    """
    required = [
        args.module3_output_dir,
        args.simulation_date,
        args.output_dir
    ]
    if not all(required):
        raise ValueError(
            "Daily模式需要: --module3_output_dir, "
            "--simulation_date, --output_dir"
        )

    simulation_date = pd.to_datetime(args.simulation_date)
    simulation_start = get_or_init_simulation_start(
        args.output_dir,
        pd.to_datetime(args.simulation_start) if args.simulation_start else None
    )

    run_daily_production_planning(
        config_file=args.config,
        module3_output_dir=args.module3_output_dir,
        simulation_date=simulation_date,
        simulation_start=simulation_start,
        output_dir=args.output_dir
    )


def _run_legacy_mode(args) -> None:
    """运行旧版兼容模式。

    参数：
        args: 命令行参数
    """
    required = [args.input, args.sim_start, args.sim_end, args.output]
    if not all(required):
        raise ValueError(
            "Legacy模式需要: --input, --sim_start, --sim_end, --output"
        )

    cfg = load_config(args.input)
    issues = validate_config(cfg)

    nd = cfg.get('NetDemand', pd.DataFrame())
    nd = cast_identifiers_to_str(nd, ['material', 'location'])
    if nd.empty:
        raise ValueError("Legacy模式需要配置文件中包含NetDemand工作表")

    nd['requirement_date'] = pd.to_datetime(nd['requirement_date'])
    mlcfg = cfg['MaterialLocationLineCfg']

    co_mat = cfg['ChangeoverMatrix'].set_index(
        ['from_material', 'to_material']
    )['changeover_id'].sort_index()

    co_def = cfg['ChangeoverDefinition'].set_index(
        ['changeover_id', 'line']
    )['time'].to_dict()

    cap_df = cfg['LineCapacity']

    rate_map = mlcfg.set_index(
        ['material', 'location', 'delegate_line']
    )['prd_rate']
    rate_map.index.set_names(['material', 'location', 'line'], inplace=True)

    uncon = pd.DataFrame(columns=[
        'material', 'location', 'line', 'planned_date',
        'uncon_planned_qty', 'simulation_date', 'original_quantity'
    ])

    plan_log, exceed_log = centralized_capacity_allocation_with_changeover(
        uncon, cap_df, rate_map, co_mat, co_def, mlcfg, issues=issues
    )

    plan_log = simulate_production(
        plan_log, cfg['ProductionReliability'],
        seed=cfg.get('RandomSeed', DEFAULT_RANDOM_SEED)
    )

    changeover_log = calculate_changeover_metrics(
        plan_log, cfg['ChangeoverDefinition']
    )

    issues = dedup_issues(issues)
    write_output(plan_log, exceed_log, issues, changeover_log, args.output)

    _check_legacy_critical_issues(issues)


def _check_legacy_critical_issues(issues: list) -> None:
    """检查Legacy模式的关键问题。

    参数：
        issues: 问题列表
    """
    critical = [
        x for x in issues
        if any(key in str(x.get('issue', ''))
               for key in ['No line config', 'Multiple eligible lines'])
    ]

    if critical:
        msg = (
            '发现关键校验错误! 请查看输出文件的Validation工作表。\n' +
            '\n'.join(str(x['issue']) for x in critical)
        )
        raise Exception(msg)


if __name__ == '__main__':
    main()
