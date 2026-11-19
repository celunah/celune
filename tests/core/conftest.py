# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures shared by the core tests."""

from collections.abc import Generator

import pytest

from .fixtures import CeluneCoreTestSupport


@pytest.fixture(scope="class", autouse=True)
def close_cached_celune(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Close the cached fake Celune instance after a core test class completes."""
    yield
    test_class = request.cls
    if test_class is not None and issubclass(test_class, CeluneCoreTestSupport):
        cached_celune = test_class._cached_celune
        if cached_celune is not None:
            test_class._cached_celune = None
            cached_celune.close()
