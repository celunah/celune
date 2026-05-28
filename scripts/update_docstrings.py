from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "celune",
    ROOT / "tests",
    ROOT / "extensions",
    ROOT / "main.py",
    ROOT / "setup.py",
]


@dataclass
class ParsedDoc:
    description: str
    args: dict[str, str]
    returns: str
    raises: dict[str, str]


@dataclass
class Replacement:
    start: int
    end: int
    text: str


SECTION_PATTERN = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Yields|Raises)\s*:\s*$"
)


def target_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS:
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
    return files


def normalize_sentence(text: str) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return "Describe this function."
    if compact[-1] not in ".!?":
        compact += "."
    return compact


def clean_inline(text: str) -> str:
    return " ".join(text.strip().split())


def parse_docstring(doc: str) -> ParsedDoc:
    lines = doc.expandtabs().splitlines()
    section = "description"
    description_lines: list[str] = []
    args: dict[str, str] = {}
    returns_lines: list[str] = []
    raises: dict[str, str] = {}
    current_name: str | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped and section != "description":
            current_name = None
            continue

        match = SECTION_PATTERN.match(stripped)
        if match:
            name = match.group(1)
            if name in {"Args", "Arguments", "Parameters"}:
                section = "args"
            elif name in {"Returns", "Yields"}:
                section = "returns"
            else:
                section = "raises"
            current_name = None
            continue

        if section == "description":
            description_lines.append(raw_line.rstrip())
            continue

        if section == "returns":
            returns_lines.append(stripped)
            continue

        if ":" in stripped:
            name, value = stripped.split(":", 1)
            key = clean_inline(name)
            val = clean_inline(value)
            current_name = key
            if section == "args":
                args[key] = val
            else:
                raises[key] = val
            continue

        if current_name:
            if section == "args":
                args[current_name] = clean_inline(f"{args[current_name]} {stripped}")
            else:
                raises[current_name] = clean_inline(
                    f"{raises[current_name]} {stripped}"
                )

    description = "\n".join(description_lines).strip()
    description = description.split("\n\n", 1)[0].strip() if description else ""
    return ParsedDoc(
        description=description,
        args=args,
        returns=clean_inline(" ".join(returns_lines)),
        raises=raises,
    )


def function_kind(parents: list[tuple[str, str]], name: str) -> str:
    if any(kind == "func" for kind, _ in parents):
        return "nested"
    if name.startswith("_"):
        return "private"
    return "public"


def iter_direct_raises(node: ast.AST) -> Iterable[str]:
    class RaiseCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: list[str] = []

        def visit_FunctionDef(self, inner: ast.FunctionDef) -> None:
            if inner is not node:
                return
            self.generic_visit(inner)

        def visit_AsyncFunctionDef(self, inner: ast.AsyncFunctionDef) -> None:
            if inner is not node:
                return
            self.generic_visit(inner)

        def visit_Lambda(self, inner: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, inner: ast.ClassDef) -> None:
            return

        def visit_Raise(self, inner: ast.Raise) -> None:
            exc = inner.exc
            if exc is None:
                self.names.append("Exception")
            elif isinstance(exc, ast.Call):
                self.names.append(expr_name(exc.func))
            else:
                self.names.append(expr_name(exc))

    collector = RaiseCollector()
    collector.visit(node)
    return collector.names


def has_non_none_return(node: ast.AST) -> bool:
    class ReturnCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_FunctionDef(self, inner: ast.FunctionDef) -> None:
            if inner is not node:
                return
            self.generic_visit(inner)

        def visit_AsyncFunctionDef(self, inner: ast.AsyncFunctionDef) -> None:
            if inner is not node:
                return
            self.generic_visit(inner)

        def visit_Lambda(self, inner: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, inner: ast.ClassDef) -> None:
            return

        def visit_Return(self, inner: ast.Return) -> None:
            if inner.value is not None:
                if not (
                    isinstance(inner.value, ast.Constant) and inner.value.value is None
                ):
                    self.found = True

    collector = ReturnCollector()
    collector.visit(node)
    return collector.found


def expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return expr_name(node.value)
    if isinstance(node, ast.Call):
        return expr_name(node.func)
    return "Exception"


def inferred_return_text(parsed: ParsedDoc) -> str:
    if parsed.returns:
        return parsed.returns
    return "Result of this function."


def arg_description(name: str, parsed: ParsedDoc) -> str:
    for key, value in parsed.args.items():
        normalized = key.lstrip("*")
        if normalized == name and value:
            return value
    return f"Value for `{name}`."


def raise_description(name: str, parsed: ParsedDoc) -> str:
    if name in parsed.raises and parsed.raises[name]:
        return parsed.raises[name]
    return f"If `{name}` needs to be raised."


def public_docstring(node: ast.AST, indent: str, parsed: ParsedDoc) -> str:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    description = normalize_sentence(
        parsed.description or ast.get_docstring(node, clean=False) or ""
    )
    lines = [
        f'{indent}"""{description}',
        "",
    ]

    params: list[str] = []
    all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    for arg in all_args:
        if arg.arg not in {"self", "cls"}:
            params.append(arg.arg)
    if node.args.vararg:
        params.append(node.args.vararg.arg)
    if node.args.kwarg:
        params.append(node.args.kwarg.arg)

    if params:
        lines.append(f"{indent}Args:")
        lines.extend(
            f"{indent}    {name}: {arg_description(name, parsed)}" for name in params
        )
        lines.append("")

    annotation = node.returns
    returns_needed = bool(parsed.returns) or has_non_none_return(node)
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        returns_needed = False
    if returns_needed:
        lines.append(f"{indent}Returns:")
        lines.append(f"{indent}    {inferred_return_text(parsed)}")
        lines.append("")

    raise_names = list(dict.fromkeys(name for name in iter_direct_raises(node) if name))
    if parsed.raises:
        for name in parsed.raises:
            if name not in raise_names:
                raise_names.append(name)
    if raise_names:
        lines.append(f"{indent}Raises:")
        lines.extend(
            f"{indent}    {name}: {raise_description(name, parsed)}"
            for name in raise_names
        )
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    lines.append(f'{indent}"""')
    return "\n".join(lines)


def private_docstring(node: ast.AST, indent: str, parsed: ParsedDoc) -> str:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    raw = parsed.description or ast.get_docstring(node, clean=False) or ""
    first = raw.strip().splitlines()[0] if raw.strip() else ""
    return f'{indent}"""{normalize_sentence(first)}"""'


def collect_replacements(source: str, tree: ast.AST) -> list[Replacement]:
    replacements: list[Replacement] = []
    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[tuple[str, str]] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.parents.append(("class", node.name))
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._handle(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._handle(node)

        def _handle(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            if not node.body:
                return
            doc_expr = node.body[0]
            if not (
                isinstance(doc_expr, ast.Expr)
                and isinstance(doc_expr.value, ast.Constant)
                and isinstance(doc_expr.value.value, str)
            ):
                self.parents.append(("func", node.name))
                self.generic_visit(node)
                self.parents.pop()
                return

            kind = function_kind(self.parents, node.name)
            parsed = parse_docstring(doc_expr.value.value)
            start = line_offsets[doc_expr.lineno - 1]
            if doc_expr.end_lineno is None:
                raise SyntaxError("unterminated docstring")

            end_lineno = doc_expr.end_lineno
            end = line_offsets[end_lineno]

            indent_match = re.match(r"\s*", lines[doc_expr.lineno - 1])
            if indent_match is None:
                raise RuntimeError("failed to determine indentation")
            indent = indent_match.group(0)

            if kind == "nested":
                if len(node.body) == 1:
                    text = f"{indent}pass\n"
                else:
                    text = ""
            elif kind == "private":
                text = private_docstring(node, indent, parsed) + "\n"
            else:
                text = public_docstring(node, indent, parsed) + "\n"

            replacements.append(Replacement(start=start, end=end, text=text))

            self.parents.append(("func", node.name))
            self.generic_visit(node)
            self.parents.pop()

    Visitor().visit(tree)
    return sorted(replacements, key=lambda item: item.start, reverse=True)


def rewrite_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    replacements = collect_replacements(source, tree)
    if not replacements:
        return False

    updated = source
    for replacement in replacements:
        updated = (
            updated[: replacement.start] + replacement.text + updated[replacement.end :]
        )

    if updated != source:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = 0
    for path in target_files():
        if rewrite_file(path):
            changed += 1
            print(path.relative_to(ROOT))
    print(f"Changed {changed} files.")


if __name__ == "__main__":
    main()
