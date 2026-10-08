"""Positive AST/call allowlist for model-generated utility repairs."""

from __future__ import annotations

import ast
from dataclasses import dataclass


ALLOWED_BUILTINS = frozenset(
    {"sorted", "len", "set", "enumerate", "sum", "range", "list", "tuple"}
)
ALLOWED_METHODS = frozenset({"split", "append", "add"})
# Methods that mutate their receiver in place. Allowed only on local
# selection structures — never on the ``chunks`` input or chunk dicts.
MUTATING_METHODS = frozenset({"append", "add", "sort", "extend", "clear", "pop", "remove", "update"})

ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.For,
    ast.While,
    ast.If,
    ast.Continue,
    ast.Break,
    ast.Pass,
    ast.Expr,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Subscript,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Slice,
    ast.Index,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.keyword,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)


@dataclass(frozen=True)
class AllowlistResult:
    allowed: bool
    reason: str = ""


def validate_source(source: str) -> AllowlistResult:
    """Reject everything not explicitly allowed before execution.

    In addition to the positive AST/call allowlist, perform a simple
    root/taint check so the input ``chunks`` list and its dictionaries
    cannot be mutated in place. Local selection structures remain fine.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return AllowlistResult(False, f"syntax_error: {exc}")

    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(functions) != 1 or functions[0].name != "select_chunks":
        return AllowlistResult(
            False,
            "utility must contain exactly one select_chunks function",
        )
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        return AllowlistResult(False, "top-level statements are not allowed")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return AllowlistResult(
                False,
                f"disallowed syntax node: {type(node).__name__}",
            )
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name not in ALLOWED_BUILTINS and call_name not in ALLOWED_METHODS:
                return AllowlistResult(False, f"disallowed call: {call_name}")
        if isinstance(node, ast.Attribute):
            if node.attr not in ALLOWED_METHODS | MUTATING_METHODS:
                return AllowlistResult(False, f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            return AllowlistResult(False, "bytes constants are not allowed")

    mutation = _reject_input_mutation(functions[0])
    if mutation is not None:
        return mutation

    return AllowlistResult(True, "allowlist passed")


def _reject_input_mutation(fn: ast.FunctionDef) -> AllowlistResult | None:
    """Taint ``chunks`` and aliases/elements derived from it; reject stores
    and mutating method calls on tainted roots."""
    arg_names = {arg.arg for arg in fn.args.args}
    if "chunks" not in arg_names:
        return AllowlistResult(False, "select_chunks must accept a chunks parameter")

    tainted: set[str] = {"chunks"}

    def root_name(node: ast.AST) -> str | None:
        while isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
        return None

    def is_tainted_expr(node: ast.AST) -> bool:
        name = root_name(node)
        return name in tainted if name is not None else False

    for node in ast.walk(fn):
        # for chunk in chunks / for chunk in tainted_alias
        if isinstance(node, ast.For):
            if is_tainted_expr(node.iter):
                for target_name in _names_bound(node.target):
                    tainted.add(target_name)
            # Also taint comprehension-like unpacking of tainted iterables.
        if isinstance(node, ast.comprehension):
            if is_tainted_expr(node.iter):
                for target_name in _names_bound(node.target):
                    tainted.add(target_name)

        # alias = chunks / alias = chunks[...] / alias = chunk
        if isinstance(node, ast.Assign):
            if is_tainted_expr(node.value):
                for target in node.targets:
                    for name in _names_bound(target):
                        tainted.add(name)
            # Store into a tainted container: chunk["x"] = ... or chunks[i] = ...
            for target in node.targets:
                if isinstance(target, ast.Subscript) and is_tainted_expr(target.value):
                    return AllowlistResult(
                        False,
                        "input mutation rejected: cannot assign into chunks or chunk dicts",
                    )
                if isinstance(target, ast.Attribute) and is_tainted_expr(target.value):
                    return AllowlistResult(
                        False,
                        "input mutation rejected: cannot set attributes on chunks inputs",
                    )

        if isinstance(node, ast.AugAssign):
            if is_tainted_expr(node.target):
                return AllowlistResult(
                    False,
                    "input mutation rejected: cannot augment-assign chunks inputs",
                )
            if isinstance(node.target, ast.Subscript) and is_tainted_expr(
                node.target.value
            ):
                return AllowlistResult(
                    False,
                    "input mutation rejected: cannot augment-assign into chunk dicts",
                )

        if isinstance(node, ast.Delete):
            for target in node.targets:
                if is_tainted_expr(target) or (
                    isinstance(target, ast.Subscript)
                    and is_tainted_expr(target.value)
                ):
                    return AllowlistResult(
                        False,
                        "input mutation rejected: cannot delete from chunks inputs",
                    )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in MUTATING_METHODS and is_tainted_expr(node.func.value):
                return AllowlistResult(
                    False,
                    f"input mutation rejected: cannot call {node.func.attr}() on chunks inputs",
                )

    return None


def _names_bound(target: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_names_bound(elt))
    elif isinstance(target, ast.Starred):
        names.extend(_names_bound(target.value))
    return names


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return type(node).__name__


__all__ = ["AllowlistResult", "validate_source"]
