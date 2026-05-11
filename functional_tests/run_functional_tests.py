from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path):
    module_name = "functional_tests_dynamic." + ".".join(path.relative_to(ROOT).with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    test_files = sorted(
        p for p in (ROOT / "functional_tests").rglob("test_*.py") if "__pycache__" not in p.parts
    )
    total = 0
    failures = []

    for path in test_files:
        module = _load_module(path)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            test_id = f"{path.relative_to(ROOT)}::{name}"
            try:
                func()
                print(f"PASS {test_id}")
            except Exception as exc:
                failures.append((test_id, exc, traceback.format_exc()))
                print(f"FAIL {test_id}: {exc}")

    if failures:
        print("")
        print(f"{len(failures)} of {total} functional tests failed")
        for test_id, _exc, tb in failures:
            print("")
            print(f"--- {test_id} ---")
            print(tb.rstrip())
        return 1

    print("")
    print(f"{total} functional tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
