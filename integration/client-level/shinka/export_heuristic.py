#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, List, Set

FEATURE_NAMES = {
    "size",
    "queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
}

ALLOWED_CALLS = {"max", "min", "abs", "float", "int"}


class PredictToCTranslator:
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.locals: Set[str] = set()

    def translate(self, predict_fn: ast.FunctionDef, source_path: Path) -> str:
        body_lines = self._emit_block(self._strip_docstring(predict_fn.body), indent=1)
        guard = f"{self.function_name.upper()}_H"
        return "\n".join(
            [
                f"#ifndef {guard}",
                f"#define {guard}",
                "",
                '#include "../shinka_generated_common.h"',
                "",
                f"/* Generated from {source_path} */",
                f"static int {self.function_name}(const ShinkaFeatures *features) {{",
                *body_lines,
                "}",
                "",
                f"#endif  /* {guard} */",
                "",
            ]
        )

    def _strip_docstring(self, statements: List[ast.stmt]) -> List[ast.stmt]:
        if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
            if isinstance(statements[0].value.value, str):
                return statements[1:]
        return statements

    def _emit_block(self, statements: Iterable[ast.stmt], indent: int) -> List[str]:
        lines: List[str] = []
        prefix = "    " * indent
        for statement in statements:
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise ValueError("Only simple assignments are supported in predict().")
                target = statement.targets[0].id
                expr = self._emit_expr(statement.value)
                if target not in self.locals:
                    self.locals.add(target)
                    lines.append(f"{prefix}double {target} = {expr};")
                else:
                    lines.append(f"{prefix}{target} = {expr};")
            elif isinstance(statement, ast.AugAssign):
                if not isinstance(statement.target, ast.Name):
                    raise ValueError("Only simple augmented assignments are supported in predict().")
                target = statement.target.id
                if target not in self.locals:
                    raise ValueError(f"Augmented assignment used before declaration: {target}")
                op = self._emit_op(statement.op)
                expr = self._emit_expr(statement.value)
                lines.append(f"{prefix}{target} {op}= {expr};")
            elif isinstance(statement, ast.If):
                condition = self._emit_expr(statement.test)
                lines.append(f"{prefix}if ({condition}) {{")
                lines.extend(self._emit_block(statement.body, indent + 1))
                if statement.orelse:
                    lines.append(f"{prefix}}} else {{")
                    lines.extend(self._emit_block(statement.orelse, indent + 1))
                lines.append(f"{prefix}}}")
            elif isinstance(statement, ast.Return):
                lines.append(f"{prefix}return (int)({self._emit_expr(statement.value)});")
            else:
                raise ValueError(f"Unsupported statement in predict(): {type(statement).__name__}")
        return lines

    def _emit_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "1" if node.value else "0"
            if node.value is None:
                raise ValueError("None is not supported in predict().")
            return repr(node.value)
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id != "features":
                raise ValueError("Only features[...] subscripts are supported.")
            key_node = node.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                if key_node.value not in FEATURE_NAMES:
                    raise ValueError(f"Unsupported feature name: {key_node.value}")
                return f"((double)features->{key_node.value})"
            raise ValueError("Feature keys must be constant strings.")
        if isinstance(node, ast.BinOp):
            left = self._emit_expr(node.left)
            right = self._emit_expr(node.right)
            op = self._emit_op(node.op)
            return f"({left} {op} {right})"
        if isinstance(node, ast.BoolOp):
            op = "&&" if isinstance(node.op, ast.And) else "||"
            return "(" + f" {op} ".join(self._emit_expr(value) for value in node.values) + ")"
        if isinstance(node, ast.UnaryOp):
            operand = self._emit_expr(node.operand)
            if isinstance(node.op, ast.Not):
                return f"(!({operand}))"
            if isinstance(node.op, ast.USub):
                return f"(-({operand}))"
            if isinstance(node.op, ast.UAdd):
                return f"(+({operand}))"
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            lhs = self._emit_expr(node.left)
            pieces: List[str] = []
            current_left = lhs
            for op, comparator in zip(node.ops, node.comparators):
                rhs = self._emit_expr(comparator)
                pieces.append(f"({current_left} {self._emit_cmp(op)} {rhs})")
                current_left = rhs
            return "(" + " && ".join(pieces) + ")"
        if isinstance(node, ast.IfExp):
            condition = self._emit_expr(node.test)
            body = self._emit_expr(node.body)
            orelse = self._emit_expr(node.orelse)
            return f"({condition} ? {body} : {orelse})"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple builtin calls are supported in predict().")
            func_name = node.func.id
            if func_name not in ALLOWED_CALLS:
                raise ValueError(f"Unsupported call in predict(): {func_name}")
            args = [self._emit_expr(arg) for arg in node.args]
            if func_name == "max":
                if len(args) != 2:
                    raise ValueError("max() must receive exactly two arguments.")
                return f"shinka_max_double({args[0]}, {args[1]})"
            if func_name == "min":
                if len(args) != 2:
                    raise ValueError("min() must receive exactly two arguments.")
                return f"shinka_min_double({args[0]}, {args[1]})"
            if func_name == "abs":
                if len(args) != 1:
                    raise ValueError("abs() must receive exactly one argument.")
                return f"shinka_abs_double({args[0]})"
            if func_name in {"float", "int"}:
                if len(args) != 1:
                    raise ValueError(f"{func_name}() must receive exactly one argument.")
                return f"({args[0]})"
        raise ValueError(f"Unsupported expression in predict(): {type(node).__name__}")

    def _emit_cmp(self, op: ast.AST) -> str:
        mapping = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        for py_type, c_op in mapping.items():
            if isinstance(op, py_type):
                return c_op
        raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")

    def _emit_op(self, op: ast.AST) -> str:
        mapping = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        for py_type, c_op in mapping.items():
            if isinstance(op, py_type):
                return c_op
        raise ValueError(f"Unsupported operator: {type(op).__name__}")


def _load_predict_function(program_path: Path) -> ast.FunctionDef:
    module = ast.parse(program_path.read_text(encoding="utf-8"), filename=str(program_path))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "predict":
            return node
    raise ValueError(f"No predict() function found in {program_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Shinka-evolved predict() function into a C header."
    )
    parser.add_argument("--program_path", type=str, required=True)
    parser.add_argument("--output_header", type=str, required=True)
    parser.add_argument("--function_name", type=str, required=True)
    args = parser.parse_args()

    program_path = Path(args.program_path).resolve()
    output_header = Path(args.output_header).resolve()
    output_header.parent.mkdir(parents=True, exist_ok=True)

    translator = PredictToCTranslator(function_name=args.function_name)
    predict_fn = _load_predict_function(program_path)
    header_text = translator.translate(predict_fn=predict_fn, source_path=program_path)
    output_header.write_text(header_text, encoding="utf-8")
    print(f"===== output file : {output_header}")


if __name__ == "__main__":
    main()
