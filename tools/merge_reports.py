#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并分段测试报告为完整版单文件。"""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"


def merge_reports(prefix: str, glob_pattern: str, out_name: str):
    files = sorted(DOCS_DIR.glob(glob_pattern))
    if not files:
        print(f"[WARN] No files matched: {glob_pattern}")
        return

    parts = []
    for i, f in enumerate(files):
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()

        if i == 0:
            # 第一个文件完整保留
            parts.append(text.rstrip())
        else:
            # 找到 '## 2.' 的位置，从此处开始
            start = next(
                (j for j, l in enumerate(lines) if re.match(r"^## 2\.", l)), None
            )
            if start is None:
                print(f"[WARN] Could not find '## 2.' in {f.name}, skipping.")
                continue
            # 分隔符 + 原一级标题（作为区间标识）+ 正文
            section_title = lines[0]  # 原 # 标题
            body = "\n".join(lines[start:]).rstrip()
            parts.append(f"\n---\n\n{section_title}\n\n{body}")

    merged = "\n".join(parts) + "\n"
    out_path = DOCS_DIR / out_name
    out_path.write_text(merged, encoding="utf-8")
    line_count = merged.count("\n")
    print(f"  -> {out_path}")
    print(f"     {len(files)} files merged, {line_count} lines total")


print("Merging BC reports...")
merge_reports(
    prefix="BC",
    glob_pattern="BC算法优化测试报告_Day*.md",
    out_name="BC算法优化测试报告_完整版.md",
)

print("Merging OC reports...")
merge_reports(
    prefix="OC",
    glob_pattern="OC算法优化测试报告_Day*.md",
    out_name="OC算法优化测试报告_完整版.md",
)

print("Done.")
