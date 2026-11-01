# SPDX-License-Identifier: Apache-2.0
"""Parse and invoke explicit Python-style call expressions for the debug UI."""

import ast
import builtins
import importlib
import inspect
from collections.abc import Awaitable
from typing import Optional
from typing import cast


class InvokeParseError(ValueError):
    """Raised when an invoke expression is malformed or uses unsupported syntax."""


class InvokeResolutionError(LookupError):
    """Raised when an invoke expression name cannot be imported or resolved."""


class InvokeCallError(Exception):
    """Wrap an exception raised while executing a nested or outer call."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


async def invoke_expression(expression: str, engine: object) -> object:
    """Resolve and call one supported expression without evaluating Python source.

    Args:
        expression: Python-style callable expression entered in the inspector.
        engine: The active Celune object exposed as the ``engine`` name.

    Returns:
        object: The call result, awaiting it when it is awaitable.

    Raises:
        InvokeParseError: The expression is malformed or contains unsupported syntax.
        InvokeResolutionError: A referenced name, attribute, or callable is unavailable.
        InvokeCallError: A called function raises an exception.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise InvokeParseError(str(exc)) from exc

    if not isinstance(parsed.body, ast.Call):
        raise InvokeParseError("expression must be a function or method call")

    async def resolve(node: ast.AST) -> object:
        """Interpret one permitted expression node recursively."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return _resolve_name(node.id, engine)
        if isinstance(node, ast.Attribute):
            if _is_dotted_path(node):
                return _resolve_name(_dotted_name(node), engine)
            value = await resolve(node.value)
            try:
                return getattr(value, node.attr)
            except AttributeError as exc:
                raise InvokeResolutionError(
                    f"cannot resolve attribute {node.attr!r}"
                ) from exc
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = await resolve(node.operand)
            if not isinstance(value, (int, float, complex)) or isinstance(value, bool):
                raise InvokeParseError("signs are supported only for numeric literals")
            return +value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.List):
            return [await resolve(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            items = []
            for item in node.elts:
                items.append(await resolve(item))
            return tuple(items)
        if isinstance(node, ast.Set):
            return {await resolve(item) for item in node.elts}
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise InvokeParseError("dictionary unpacking is not supported")
            return {
                await resolve(key): await resolve(value)
                for key, value in zip(node.keys, node.values)
                if key is not None
            }
        if isinstance(node, ast.Call):
            function = await resolve(node.func)
            if not callable(function):
                raise InvokeResolutionError(
                    f"{ast.unparse(node.func)!r} is not callable"
                )
            if any(isinstance(argument, ast.Starred) for argument in node.args):
                raise InvokeParseError("argument unpacking is not supported")
            if any(keyword.arg is None for keyword in node.keywords):
                raise InvokeParseError("keyword unpacking is not supported")

            arguments = [await resolve(argument) for argument in node.args]
            keywords = {
                keyword.arg: await resolve(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            try:
                result = function(*arguments, **keywords)
                if inspect.isawaitable(result):
                    return await cast(Awaitable[object], result)
                return result
            except Exception as exc:
                raise InvokeCallError(exc) from exc

        raise InvokeParseError(f"unsupported expression element: {type(node).__name__}")

    return await resolve(parsed.body)


def _dotted_name(node: ast.AST) -> str:
    """Return a dotted identifier path or reject dynamic call targets."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    raise InvokeParseError("call target must be a dotted name")


def _is_dotted_path(node: ast.AST) -> bool:
    """Return whether an expression is composed only of dotted identifiers."""
    if isinstance(node, ast.Name):
        return True
    return isinstance(node, ast.Attribute) and _is_dotted_path(node.value)


def _resolve_name(name: str, engine: object) -> object:
    """Resolve a built-in, engine attribute, or importable dotted path."""
    if name == "engine":
        return engine
    if name.startswith("engine."):
        return _traverse_attributes(engine, name.split(".")[1:], name)
    if "." not in name:
        if hasattr(builtins, name):
            return getattr(builtins, name)
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            raise InvokeResolutionError(f"cannot resolve name {name!r}") from exc
        except ImportError as exc:
            raise InvokeResolutionError(f"cannot import {name!r}: {exc}") from exc
        except Exception as exc:
            raise InvokeResolutionError(f"cannot import {name!r}: {exc}") from exc

    parts = name.split(".")
    import_error: Optional[ImportError] = None
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and not (
                module_name == exc.name or module_name.startswith(f"{exc.name}.")
            ):
                raise InvokeResolutionError(
                    f"importing {module_name!r} failed: {exc}"
                ) from exc
            import_error = exc
            continue
        except ImportError as exc:
            raise InvokeResolutionError(
                f"importing {module_name!r} failed: {exc}"
            ) from exc
        except Exception as exc:
            raise InvokeResolutionError(
                f"importing {module_name!r} failed: {exc}"
            ) from exc
        return _traverse_attributes(module, parts[split_at:], name)

    raise InvokeResolutionError(f"cannot resolve name {name!r}") from import_error


def _traverse_attributes(value: object, attributes: list[str], name: str) -> object:
    """Resolve dotted attributes while presenting lookup failures consistently."""
    for attribute in attributes:
        try:
            value = getattr(value, attribute)
        except AttributeError as exc:
            raise InvokeResolutionError(
                f"cannot resolve attribute {attribute!r} in {name!r}"
            ) from exc
    return value
