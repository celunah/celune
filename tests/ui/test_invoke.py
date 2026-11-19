# SPDX-License-Identifier: Apache-2.0
"""Tests for safe Python-style calls in the debug inspector."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from celune.ui.invoke import invoke_expression
from celune.exceptions import (
    InvokeCallError,
    InvokeParseError,
    InvokeResolutionError,
)


def test_invoke_interprets_literals_nested_calls_and_imported_objects() -> None:
    """Resolve imports and recursively build normal Python argument values."""
    result = asyncio.run(
        invoke_expression(
            "builtins.sorted([abs(-2), max(1, 3)], reverse=True)",
            object(),
        )
    )

    assert result == [3, 2]
    assert (
        asyncio.run(
            invoke_expression("builtins.str(pathlib.Path('sample.txt'))", object())
        )
        == "sample.txt"
    )
    audio = asyncio.run(
        invoke_expression(
            "celune.audio.pad_note([65.4], duration=0.01, sample_rate=8000, "
            "attack_seconds=0, release_seconds=0, leading_silence_seconds=0, "
            "trailing_silence_seconds=0)",
            object(),
        )
    )
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (80, 2)


def test_invoke_resolves_engine_and_awaits_async_calls() -> None:
    """Expose the active engine and await returned coroutines."""

    async def combine(value: int, *, offset: int) -> int:
        return value + offset

    engine = SimpleNamespace(combine=combine)

    assert asyncio.run(invoke_expression("engine.combine(2, offset=3)", engine)) == 5


@pytest.mark.parametrize(
    "expression",
    (
        "(",
        "abs(1 + 2)",
        "sum(*[1, 2])",
        "dict(**{'value': 1})",
        "[value for value in (1, 2)]",
        "math.sqrt[0]()",
    ),
)
def test_invoke_rejects_malformed_or_unsupported_syntax(expression: str) -> None:
    """Reject syntax outside the explicitly supported expression grammar."""
    with pytest.raises(InvokeParseError):
        asyncio.run(invoke_expression(expression, object()))


@pytest.mark.parametrize(
    "expression",
    (
        "__celune_invoke_missing_module__.value()",
        "math.missing_attribute()",
        "math.pi()",
    ),
)
def test_invoke_reports_name_and_callable_resolution_failures(
    expression: str,
) -> None:
    """Reject missing imports, attributes, and non-callable targets."""
    with pytest.raises(InvokeResolutionError):
        asyncio.run(invoke_expression(expression, object()))


def test_invoke_propagates_nested_and_outer_call_exceptions() -> None:
    """Keep exceptions raised by invoked code separate from parser failures."""

    def inner_fail() -> None:
        raise ZeroDivisionError("nested failure")

    def fail() -> None:
        raise RuntimeError("failed")

    engine = SimpleNamespace(inner_fail=inner_fail, fail=fail)
    with pytest.raises(InvokeCallError) as nested_call:
        asyncio.run(invoke_expression("abs(engine.inner_fail())", engine))
    assert isinstance(nested_call.value.error, ZeroDivisionError)
    assert str(nested_call.value.error) == "nested failure"

    with pytest.raises(InvokeCallError) as outer_call:
        asyncio.run(invoke_expression("engine.fail()", engine))
    assert isinstance(outer_call.value.error, RuntimeError)
    assert str(outer_call.value.error) == "failed"
