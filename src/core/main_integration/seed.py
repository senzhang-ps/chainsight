"""
seed.py

随机种子管理函数模块。

为所有模块提供统一的随机种子加载与设置功能，确保仿真结果的可重现性。
"""

import numpy as np

from ...utils.defaults import DEFAULT_RANDOM_SEED


def load_global_seed(config_dict: dict) -> int:
    """从配置中加载全局随机种子

    目的：
    - 提供稳定的随机源；当前实现从 `Global_Seed` 键读取，优先使用 `seed` 列，兼容首行首列旧格式。

    Args:
        config_dict: 配置数据字典

    Returns:
        int: 随机种子值，默认为 DEFAULT_RANDOM_SEED

    逻辑：
        - 按优先级读取 → 打印提示 → 返回默认值或实际种子
    """
    seed_key = None
    for candidate in ('Global_seed', 'Global_Seed'):
        if candidate in config_dict and not config_dict[candidate].empty:
            seed_key = candidate
            break

    if seed_key:
        seed_df = config_dict[seed_key]
        if 'seed' in seed_df.columns:
            seed_value = int(seed_df.iloc[0]['seed'])
            return seed_value
        elif len(seed_df.columns) > 0 and len(seed_df) > 0:
            # 兼容旧格式，读取首行首列的值
            seed_value = int(seed_df.iloc[0, 0])
            return seed_value
    
    return DEFAULT_RANDOM_SEED


def set_module_seeds(config_dict: dict, global_seed: int = None):
    """写入集成流程使用的统一随机种子配置

    目的：
    - 将全局种子应用到 numpy，并写入集成流程约定的模块级随机种子键，便于下游流程复用。

    Args:
        config_dict: 配置数据字典
        global_seed: 指定的全局种子；为 None 时从配置中读取

    Returns:
        int: 实际使用的全局种子

    逻辑：
        - 若未传入则加载 → 设置 numpy 种子 → 写入模块级种子键 → 打印确认信息
    """
    if global_seed is None:
        global_seed = load_global_seed(config_dict)
    
    # 设置 numpy 全局种子
    np.random.seed(global_seed)
    
    # 写入集成流程约定的模块级种子键，作为统一随机源配置
    config_dict['M1_RandomSeed'] = global_seed
    config_dict['M3_RandomSeed'] = global_seed  
    config_dict['M4_RandomSeed'] = global_seed
    config_dict['M5_RandomSeed'] = global_seed
    config_dict['M6_RandomSeed'] = global_seed
    
    return global_seed
