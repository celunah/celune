from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_PACKAGES = {"celune", "tests", "extensions"}
TARGETS = [
    ROOT / "celune",
    ROOT / "tests",
    ROOT / "extensions",
    ROOT / "main.py",
    ROOT / "setup.py",
]


@dataclass(frozen=True)
class ImportStatement:
    node: ast.Import | ast.ImportFrom
    start: int
    end: int
    text: str


def target_files() -> list[Path]:
    files: list[Path] = []

    for target in TARGETS:
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(sorted(target.rglob("*.py")))

    return files


def is_stdlib_package(name: str) -> bool:
    root = name.split(".", 1)[0]

    if root in sys.builtin_module_names:
        return True

    return root in sys.stdlib_module_names


def import_package(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", 1)[0]

    if node.level:
        return "." * node.level + (node.module or "")

    return (node.module or "").split(".", 1)[0]


def is_future_import(node: ast.Import | ast.ImportFrom) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def top_package(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", 1)[0]

    if node.level:
        return ""

    return (node.module or "").split(".", 1)[0]


def import_section(node: ast.Import | ast.ImportFrom) -> int:
    if is_future_import(node):
        return -1

    if isinstance(node, ast.ImportFrom) and node.level:
        return 2

    package = top_package(node)

    if package in FIRST_PARTY_PACKAGES:
        return 2

    if package in sys.builtin_module_names or package in sys.stdlib_module_names:
        return 0

    return 1


def import_kind(node: ast.Import | ast.ImportFrom) -> int:
    return 0 if isinstance(node, ast.Import) else 1


def import_sort_key(statement: ImportStatement) -> tuple[int, str, int, int, str]:
    package = top_package(statement.node)

    return (
        import_section(statement.node),  # future, stdlib, third-party, local
        package.lower(),  # keep same package grouped
        import_kind(statement.node),  # import first, from import last
        len(statement.text.strip()),  # then length
        statement.text.lower(),  # final fallback
    )


def collect_top_imports(tree: ast.Module, lines: list[str]) -> list[ImportStatement]:
    statements: list[ImportStatement] = []

    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        text = "".join(lines[start:end])

        statements.append(
            ImportStatement(
                node=node,
                start=start,
                end=end,
                text=text,
            )
        )

    return statements


def split_import_sections(
    statements: list[ImportStatement],
) -> list[list[ImportStatement]]:
    sections: dict[int, list[ImportStatement]] = {
        -1: [],
        0: [],
        1: [],
        2: [],
    }

    for statement in statements:
        sections[import_section(statement.node)].append(statement)

    return [section for section in sections.values() if section]


def import_block_bounds(statements: list[ImportStatement]) -> tuple[int, int] | None:
    if not statements:
        return None

    return min(statement.start for statement in statements), max(
        statement.end for statement in statements
    )


def render_import_block(statements: list[ImportStatement]) -> list[str]:
    rendered_sections: list[str] = []

    for section in split_import_sections(statements):
        sorted_section = sorted(section, key=import_sort_key)
        rendered_sections.append(
            "".join(statement.text for statement in sorted_section).rstrip()
        )

    return ("\n\n".join(rendered_sections) + "\n").splitlines(keepends=True)


def rewrite_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    tree = ast.parse(source)
    statements = collect_top_imports(tree, lines)
    bounds = import_block_bounds(statements)

    if bounds is None:
        return False

    start, end = bounds
    updated_block = render_import_block(statements)

    if lines[start:end] == updated_block:
        return False

    lines[start:end] = updated_block
    path.write_text("".join(lines), encoding="utf-8")

    return True


def main() -> None:
    changed = 0

    for path in target_files():
        if rewrite_file(path):
            changed += 1
            print(path.relative_to(ROOT))

    print(f"Changed {changed} files.")


if __name__ == "__main__":
    main()
