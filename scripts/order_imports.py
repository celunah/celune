# SPDX-License-Identifier: MIT
"""Recursively order imports by package name and line length automatically."""

import sys
import ast
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional


ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_PACKAGES = {"celune", "tests", "extensions"}
PRIORITY_IMPORT_MODULES = {"os", "sys"}
TARGETS = [
    ROOT / "celune",
    ROOT / "tests",
    ROOT / "extensions",
    ROOT / "scripts",
    ROOT / "main.py",
    ROOT / "nuitka_main.py",
    ROOT / "setup.py",
]


@dataclass(frozen=True)
class ImportStatement:
    node: Union[ast.Import, ast.ImportFrom]
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


def import_package(node: Union[ast.Import, ast.ImportFrom]) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", 1)[0]

    if node.level:
        return "." * node.level + (node.module or "")

    return (node.module or "").split(".", 1)[0]


def is_future_import(node: Union[ast.Import, ast.ImportFrom]) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def top_package(node: Union[ast.Import, ast.ImportFrom]) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", 1)[0]

    if node.level:
        return ""

    return (node.module or "").split(".", 1)[0]


def import_section(node: Union[ast.Import, ast.ImportFrom]) -> int:
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


def module_name(node: Union[ast.Import, ast.ImportFrom]) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name
    if node.level:
        return "." * node.level + (node.module or "")
    return node.module or ""


def module_sort_key(statement: ImportStatement) -> tuple[int, str]:
    name = module_name(statement.node)
    return len(name), name.lower()


def sort_statement_group(
    statements: list[ImportStatement],
) -> list[ImportStatement]:
    grouped: dict[str, list[ImportStatement]] = {}
    for statement in statements:
        package = top_package(statement.node) or module_name(statement.node)
        grouped.setdefault(package, []).append(statement)

    ordered_packages = sorted(
        grouped,
        key=lambda pkg: (len(pkg), pkg.lower()),
    )
    ordered: list[ImportStatement] = []
    for package in ordered_packages:
        ordered.extend(sorted(grouped[package], key=module_sort_key))
    return ordered


def imported_targets_key(
    statement: ImportStatement,
) -> tuple[int, str, tuple[int, str]]:
    assert isinstance(statement.node, ast.ImportFrom)
    imported = ", ".join(
        alias.name + (f" as {alias.asname}" if alias.asname else "")
        for alias in statement.node.names
    )
    return len(imported), imported.lower(), module_sort_key(statement)


def sort_from_statement_group(
    statements: list[ImportStatement],
) -> list[ImportStatement]:
    grouped: dict[str, list[ImportStatement]] = {}
    for statement in statements:
        package = top_package(statement.node) or module_name(statement.node)
        grouped.setdefault(package, []).append(statement)

    ordered_packages = sorted(
        grouped,
        key=lambda pkg: min(imported_targets_key(stmt) for stmt in grouped[pkg]),
    )
    ordered: list[ImportStatement] = []
    for package in ordered_packages:
        ordered.extend(sorted(grouped[package], key=imported_targets_key))
    return ordered


def is_priority_import(statement: ImportStatement) -> bool:
    return (
        isinstance(statement.node, ast.Import)
        and all(alias.asname is None for alias in statement.node.names)
        and module_name(statement.node) in PRIORITY_IMPORT_MODULES
    )


def is_plain_import(statement: ImportStatement) -> bool:
    return isinstance(statement.node, ast.Import) and all(
        alias.asname is None for alias in statement.node.names
    )


def is_aliased_import(statement: ImportStatement) -> bool:
    return isinstance(statement.node, ast.Import) and any(
        alias.asname is not None for alias in statement.node.names
    )


def sorted_import_groups(section: list[ImportStatement]) -> list[list[ImportStatement]]:
    priority = sort_statement_group(
        [statement for statement in section if is_priority_import(statement)],
    )
    plain_and_aliased = sort_statement_group(
        [
            statement
            for statement in section
            if isinstance(statement.node, ast.Import)
            and not is_priority_import(statement)
        ],
    )
    from_imports = sort_from_statement_group(
        [
            statement
            for statement in section
            if isinstance(statement.node, ast.ImportFrom)
        ]
    )

    return [group for group in (priority, plain_and_aliased, from_imports) if group]


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


def import_block_bounds(statements: list[ImportStatement]) -> Optional[tuple[int, int]]:
    if not statements:
        return None

    return min(statement.start for statement in statements), max(
        statement.end for statement in statements
    )


def render_import_block(statements: list[ImportStatement]) -> list[str]:
    rendered_sections: list[str] = []

    for section in split_import_sections(statements):
        rendered_sections.append(
            "".join(
                statement.text
                for group in sorted_import_groups(section)
                for statement in group
            ).rstrip()
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

    import_lines = {
        line_number
        for statement in statements
        for line_number in range(statement.start, statement.end)
    }
    if any(
        line.strip() and line_number not in import_lines
        for line_number, line in enumerate(lines[start:end], start)
    ):
        return False

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
