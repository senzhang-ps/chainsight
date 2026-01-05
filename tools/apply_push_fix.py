#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用push逻辑修复到module5.py
"""

def apply_push_fix():
    """应用push逻辑修复"""
    
    # 读取原文件
    with open('module5.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到push函数的开始和结束位置
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def push_softpush_allocation('):
            start_line = i
        elif start_line is not None and line.strip() == 'return plan_rows_push':
            end_line = i + 1  # 包含return语句
            break
    
    if start_line is None or end_line is None:
        print(f"❌ 无法找到push函数的位置: start={start_line}, end={end_line}")
        return
        
    print(f"📍 找到push函数: 第{start_line+1}行 到 第{end_line}行")
    
    # 读取修复后的函数
    with open('module5_push_fixed.py', 'r', encoding='utf-8') as f:
        new_function = f.read()
    
    # 替换函数
    new_lines = lines[:start_line] + [new_function + '\n'] + lines[end_line:]
    
    # 写回文件
    with open('module5.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Push函数已修复并替换")
    print(f"   原函数: {end_line - start_line} 行")
    print(f"   新函数: {len(new_function.split(chr(10)))} 行")

if __name__ == "__main__":
    apply_push_fix()
