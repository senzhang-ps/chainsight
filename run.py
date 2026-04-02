#!/usr/bin/env python3
"""
ChainSight 主运行脚本

这是项目的主入口。所有命令行调用都通过这个脚本完成。
实际的执行逻辑位于 src/core/run.py 中。
"""

import sys
from pathlib import Path

# 确保项目可以正确导入
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# 导入并执行核心运行函数
from src.core.run import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        import traceback
        print(f"\n[FATAL] 未捕获的异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
