# -*- coding: utf-8 -*-
"""
安全表达式解析器模块

提供安全的布尔表达式解析功能，用于评估MDQ旁路规则等业务逻辑。
仅支持受控的变量名和有限的操作符，防止代码注入。

典型用法示例:
    evaluator = SafeExpressionEvaluator(['waiting_days', 'deployed_qty_ratio'])
    result = evaluator.eval('waiting_days > 5', {'waiting_days': 7})
"""

import ast
from typing import Any, Dict, Set


class SafeExpressionEvaluator:
    """
    安全表达式解析器类。
    
    仅允许受控变量名和有限操作符的布尔表达式解析，
    用于业务规则的安全评估。
    
    Attributes:
        allowed_names: 允许使用的变量名集合
    """
    
    def __init__(self, allowed_names: list) -> None:
        """
        初始化解析器。
        
        参数：
            allowed_names: 允许在表达式中使用的变量名列表
        """
        self.allowed_names: Set[str] = set(allowed_names)
    
    def eval(self, expr: str, context: Dict[str, Any]) -> bool:
        """
        解析并执行布尔表达式。
        
        参数：
            expr: 布尔表达式字符串
            context: 变量上下文字典
            
        返回：
            表达式的布尔结果
            
        异常：
            ValueError: 当表达式语法不支持时
        """
        expr = self._normalize_expression(expr)
        if not expr:
            return False
        
        node = ast.parse(expr, mode='eval')
        return self._eval_node(node.body, context)
    
    def _normalize_expression(self, expr: str) -> str:
        """
        标准化表达式：替换SQL风格的逻辑操作符。
        
        参数：
            expr: 原始表达式字符串
            
        返回：
            标准化后的Python风格表达式
        """
        expr = (expr or '').strip()
        expr = expr.replace('AND', 'and')
        expr = expr.replace('OR', 'or')
        expr = expr.replace('NOT', 'not')
        return expr
    
    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """
        递归解析AST节点。
        
        参数：
            node: AST节点对象
            context: 变量上下文字典
            
        返回：
            节点计算结果
            
        异常：
            ValueError: 当遇到不支持的语法时
        """
        if isinstance(node, ast.BoolOp):
            return self._eval_bool_op(node, context)
        elif isinstance(node, ast.Compare):
            return self._eval_compare(node, context)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._eval_node(node.operand, context)
        elif isinstance(node, ast.Name):
            return self._eval_name(node, context)
        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Unsupported syntax: {ast.dump(node)}")
    
    def _eval_bool_op(self, node: ast.BoolOp, context: Dict[str, Any]) -> bool:
        """
        解析布尔运算节点（and/or）。
        
        参数：
            node: 布尔运算AST节点
            context: 变量上下文字典
            
        返回：
            布尔运算结果
            
        异常：
            ValueError: 当遇到不支持的布尔操作符时
        """
        values = [self._eval_node(v, context) for v in node.values]
        
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        
        raise ValueError(
            f"Unsupported boolean operator: {type(node.op).__name__}"
        )
    
    def _eval_compare(self, node: ast.Compare, context: Dict[str, Any]) -> bool:
        """
        解析比较运算节点。
        
        参数：
            node: 比较运算AST节点
            context: 变量上下文字典
            
        返回：
            比较运算结果
            
        异常：
            ValueError: 当遇到不支持的比较操作符时
        """
        left = self._eval_node(node.left, context)
        results = []
        
        for op, comparator in zip(node.ops, node.comparators):
            right = self._eval_node(comparator, context)
            result = self._compare_values(op, left, right)
            results.append(result)
            left = right
        
        return all(results)
    
    def _compare_values(
        self,
        op: ast.cmpop,
        left: Any,
        right: Any
    ) -> bool:
        """
        执行值比较。
        
        参数：
            op: 比较操作符
            left: 左操作数
            right: 右操作数
            
        返回：
            比较结果
            
        异常：
            ValueError: 当遇到不支持的比较操作符时
        """
        comparison_ops = {
            ast.Eq: lambda l, r: l == r,
            ast.NotEq: lambda l, r: l != r,
            ast.Lt: lambda l, r: l < r,
            ast.LtE: lambda l, r: l <= r,
            ast.Gt: lambda l, r: l > r,
            ast.GtE: lambda l, r: l >= r,
        }
        
        op_type = type(op)
        if op_type in comparison_ops:
            return comparison_ops[op_type](left, right)
        
        raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
    
    def _eval_name(self, node: ast.Name, context: Dict[str, Any]) -> Any:
        """
        解析变量名节点。
        
        参数：
            node: 变量名AST节点
            context: 变量上下文字典
            
        返回：
            变量值
            
        异常：
            ValueError: 当变量名不在允许列表或上下文中不存在时
        """
        if node.id not in self.allowed_names:
            raise ValueError(f"Variable '{node.id}' is not allowed")
        if node.id not in context:
            raise ValueError(f"Variable '{node.id}' not found in context")
        
        return context[node.id]
