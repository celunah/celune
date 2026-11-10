# SPDX-License-Identifier: Apache-2.0
"""Build GitHub Actions' core and CEDTS backend analysis matrix."""

import ast
import sys
import json
from pathlib import Path
from typing import Union

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _PROJECT_ROOT / "celune" / "backends" / "environment.py"


def _assignment_value(module: ast.Module, name: str) -> ast.AST:
    """Return the value assigned to one module-level name."""
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return statement.value
    raise ValueError(f"Could not find {name} in {_MANIFEST_PATH}")


def _string_value(node: ast.AST, label: str) -> str:
    """Return a string literal from one manifest expression."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise TypeError(f"Expected a string literal for {label}")
    return node.value


def _backend_components() -> list[dict[str, Union[str, int]]]:
    """Read backend IDs and Python versions directly from the manifest registry."""
    module = ast.parse(_MANIFEST_PATH.read_text(encoding="utf-8"))
    default_python = _string_value(
        _assignment_value(module, "_DEFAULT_BACKEND_PYTHON"),
        "_DEFAULT_BACKEND_PYTHON",
    )
    registry = _assignment_value(module, "BACKEND_MANIFESTS")
    if not isinstance(registry, ast.Dict):
        raise TypeError("BACKEND_MANIFESTS must remain a literal dictionary")

    components: list[dict[str, Union[str, int]]] = []
    seen_backend_ids: set[str] = set()
    for key_node, value_node in zip(registry.keys, registry.values):
        if key_node is None:
            raise ValueError("BACKEND_MANIFESTS cannot contain dictionary unpacking")
        backend_id = _string_value(key_node, "backend ID")
        if backend_id in seen_backend_ids:
            raise ValueError(f"Duplicate backend ID in manifest registry: {backend_id}")
        seen_backend_ids.add(backend_id)
        if not isinstance(value_node, ast.Call):
            raise TypeError(f"Manifest for {backend_id} must be a constructor call")

        manifest_arguments = {
            keyword.arg: keyword.value
            for keyword in value_node.keywords
            if keyword.arg is not None
        }
        declared_id = _string_value(
            manifest_arguments["backend_id"], f"{backend_id}.backend_id"
        )
        if declared_id != backend_id:
            raise ValueError(
                f"Registry key {backend_id} does not match manifest ID {declared_id}"
            )

        python_node = manifest_arguments.get("python")
        python_version = (
            default_python
            if python_node is None
            else _string_value(python_node, f"{backend_id}.python")
        )
        components.append(
            {
                "component": f"Backend: {backend_id}",
                "kind": "backend",
                "backend_id": backend_id,
                "python": python_version,
                "timeout": 60,
            }
        )

    if not components:
        raise ValueError("BACKEND_MANIFESTS must register at least one backend")
    return sorted(components, key=lambda component: component["backend_id"])


def _matrix_json() -> str:
    """Serialize the core and backend rows as a GitHub Actions matrix."""
    components = [
        {"component": "Core engine", "kind": "core", "timeout": 30},
        *_backend_components(),
    ]
    return json.dumps({"include": components}, separators=(",", ":"))


def main() -> int:
    """Print the matrix or append it to a GitHub Actions output file."""
    matrix = _matrix_json()
    if len(sys.argv) == 1:
        print(matrix)
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--github-output":
        with Path(sys.argv[2]).open("a", encoding="utf-8") as output_file:
            output_file.write(f"components={matrix}\n")
        return 0
    print("Usage: ci_backend_matrix.py [--github-output PATH]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
