"""
Agent tools: calculator, search, Python REPL, and RAG — all wrapped with try/except.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

logger = logging.getLogger(__name__)

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_ALLOWED_UNARY: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class SafeEvalError(ValueError):
    """Raised when the expression is not allowed or fails to evaluate."""


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeEvalError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _ALLOWED_BINOPS:
            raise SafeEvalError(f"Operator not allowed: {op.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _ALLOWED_BINOPS[op](left, right)
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _ALLOWED_UNARY:
            raise SafeEvalError(f"Unary operator not allowed: {op.__name__}")
        return _ALLOWED_UNARY[op](_safe_eval_node(node.operand))
    raise SafeEvalError(f"Syntax not allowed: {type(node).__name__}")


def safe_calculator_eval(expression: str) -> float | int:
    """
    Evaluate a numeric expression using ast (no names, no calls, no attributes).
    """
    expression = expression.strip()
    if not expression:
        raise SafeEvalError("Empty expression.")
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree)


@tool("calculator")
def calculator_tool(expression: str) -> str:
    """
    Evaluate a safe arithmetic expression (numbers, +, -, *, /, //, %, **, unary +/-).
    Example: (2 + 3) * 4 - 1
    """
    try:
        result = safe_calculator_eval(expression)
        return str(result)
    except SafeEvalError as e:
        return f"[calculator error] {e}"
    except Exception as e:  # noqa: BLE001 — tool must not crash the agent
        logger.exception("Calculator unexpected error")
        return f"[calculator error] Unexpected error: {e!s}"


@tool("duckduckgo_search")
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo (privacy-focused). Use for recent events and facts."""
    try:
        return DuckDuckGoSearchRun().invoke(query)
    except Exception as e:  # noqa: BLE001
        logger.exception("DuckDuckGo failed")
        return f"[duckduckgo_search error] {e!s}"


@tool("tavily_search")
def tavily_search(query: str) -> str:
    """Search with Tavily (news/research). Requires TAVILY_API_KEY."""
    try:
        tool_inst = TavilySearchResults(max_results=4)
        # Tavily returns a list of dicts; stringify for the model
        result = tool_inst.invoke({"query": query})
        return str(result)
    except Exception as e:  # noqa: BLE001
        logger.exception("Tavily failed")
        return f"[tavily_search error] {e!s}"


@tool("python_repl")
def python_repl(code: str) -> str:
    """
    Execute Python code in a REPL (sandbox carefully in production).
    Prefer for small calculations or quick data transforms the user explicitly wants.
    """
    try:
        repl = PythonREPLTool()
        return repl.invoke(code)
    except Exception as e:  # noqa: BLE001
        logger.exception("Python REPL failed")
        return f"[python_repl error] {e!s}"


def build_all_tools(rag_tool: BaseTool) -> list[BaseTool]:
    """Return every tool passed to create_agent (RAG included as a tool)."""
    return [
        calculator_tool,
        duckduckgo_search,
        tavily_search,
        python_repl,
        rag_tool,
    ]
