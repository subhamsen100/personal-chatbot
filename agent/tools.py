"""
Agent tools — each one handles a distinct category of query.

ADK reads the docstrings and type hints to build the tool schema,
so keep them accurate and descriptive.
"""

import ast
import math
import operator
from datetime import datetime, timezone

from config import settings
from storage.vector_store import vector_store


# Tool 1: Knowledge base retrieval ─

def retrieve_from_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the user's personal document knowledge base and return relevant chunks.

    Use this tool ONLY when the user asks about content that would be in their
    uploaded documents (reports, notes, PDFs, etc.). Do NOT use it for math,
    dates, greetings, or anything the user said in the conversation.

    Args:
        query: A focused search query based on the user's question.
        top_k: Number of top chunks to return (default 5, max 10).

    Returns:
        Formatted text with retrieved document chunks and source references,
        or a message indicating nothing was found.
    """
    if vector_store.is_empty():
        return (
            "The knowledge base is empty. "
            "Please ask the user to upload documents first."
        )

    k = min(top_k, settings.top_k)
    chunks = vector_store.search(query, top_k=k)

    if not chunks:
        return "No relevant chunks found in the knowledge base for this query."

    lines = [f"Found {len(chunks)} relevant chunk(s):\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"--- Chunk {i} ---\n"
            f"Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']}\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(lines)


# Tool 2: Safe calculator ─

# Whitelist of safe AST node types and operators
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)
_SAFE_NAMES = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
        }
        op_fn = ops.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_safe_eval(node.operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named math functions are allowed.")
        fn = _SAFE_NAMES.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        args = [_safe_eval(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Name):
        val = _SAFE_NAMES.get(node.id)
        if val is None:
            raise ValueError(f"Unknown name: {node.id}")
        return val
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Use this tool for any arithmetic, algebra, or math-related question such as
    multiplication, division, powers, square roots, logarithms, trigonometry, etc.

    Supports: +, -, *, /, **, %, //, and all standard math functions
    (sqrt, log, sin, cos, tan, ceil, floor, factorial, etc.)
    and constants (pi, e, tau, inf).

    Args:
        expression: A valid mathematical expression string, e.g. "17 * 7" or "sqrt(144)".

    Returns:
        The computed result as a string, or an error description.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Return int if result is a whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 10))
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"


# Tool 3: Current date / time

def get_current_datetime() -> str:
    """
    Return the current local date and time.

    Use this tool when the user asks about the current date, time, day of the
    week, or anything that requires knowing what time it is right now.

    Returns:
        A human-readable string with the current date and time.
    """
    now = datetime.now()
    return now.strftime("%A, %d %B %Y — %H:%M:%S (local time)")
