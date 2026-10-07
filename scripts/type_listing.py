"""Index the repository and list its types."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union, TypedDict


class FunctionRecord(TypedDict):
    """A function record type."""

    file: str
    line: int
    function: str
    arguments: dict[str, str]
    return_type: Optional[str]


SKIP_DIRS = {".git", ".venv", "__pycache__", "build", "dist"}


def annotation(node: Optional[ast.expr]) -> Optional[str]:
    """Extract the annotation."""
    return ast.unparse(node) if node is not None else None


def annotated_arguments(args: ast.arguments) -> dict[str, str]:
    """Extract annotated arguments."""
    result: dict[str, str] = {}

    all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]

    for arg in all_args:
        value = annotation(arg.annotation)
        if value is not None:
            result[arg.arg] = value

    if args.vararg is not None:
        value = annotation(args.vararg.annotation)
        if value is not None:
            result[f"*{args.vararg.arg}"] = value

    if args.kwarg is not None:
        value = annotation(args.kwarg.annotation)
        if value is not None:
            result[f"**{args.kwarg.arg}"] = value

    return result


class TypeCollector(ast.NodeVisitor):
    """Collect classes from source."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[str] = []
        self.functions: list[FunctionRecord] = []

    def name(self, value: str) -> str:
        """The name of the type."""
        return ".".join([*self.scope, value])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Look at class definitions."""
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Look at function definitions."""
        self.collect_function(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Look at async function definitions."""
        self.collect_function(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def collect_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> None:
        """Collect a function."""
        arguments = annotated_arguments(node.args)
        return_type = annotation(node.returns)

        if arguments or return_type is not None:
            self.functions.append(
                {
                    "file": str(self.path),
                    "line": node.lineno,
                    "function": self.name(node.name),
                    "arguments": arguments,
                    "return_type": return_type,
                }
            )


def scan(root: Path) -> list[FunctionRecord]:
    """Scan all source files for types."""
    records: list[FunctionRecord] = []

    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as error:
            print(f"ERROR: {path}: {error}", file=sys.stderr)
            continue

        collector = TypeCollector(path)
        collector.visit(tree)
        records.extend(collector.functions)

    return records


def print_report(records: list[FunctionRecord]) -> None:
    """Display a type report."""
    return_types: dict[str, set[str]] = defaultdict(set)
    argument_types: dict[str, set[str]] = defaultdict(set)

    for record in records:
        path = record["file"]

        return_type = record["return_type"]
        if isinstance(return_type, str):
            return_types[return_type].add(path)

        arguments = record["arguments"]
        if isinstance(arguments, dict):
            for value in arguments.values():
                argument_types[value].add(path)

    print("Return types:")

    for type_name, files in sorted(return_types.items()):
        print(f"\n[{type_name}]: {len(files)} files")
        for file in sorted(files):
            print(f"  {file}")

    print("\nArgument types:")

    for type_name, files in sorted(argument_types.items()):
        print(f"\n[{type_name}]: {len(files)} files")
        for file in sorted(files):
            print(f"  {file}")

    print("\nFunction details:")

    for record in records:
        print(f"\n{record['file']}:{record['line']} {record['function']}()")

        arguments = record["arguments"]
        if arguments:
            print("  arguments:")
            for name, type_name in arguments.items():
                print(f"    {name}: {type_name}")

        return_type = record["return_type"]
        if return_type is not None:
            print(f"  returns: {return_type}")


if __name__ == "__main__":
    repository = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print_report(scan(repository))
