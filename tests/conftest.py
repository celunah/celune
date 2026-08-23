"""Celune Pytest hooks and handlers."""
# SPDX-License-Identifier: Apache-2.0

import sys
import shutil
from collections.abc import Iterator, Generator
from typing import TypedDict

import pytest


class CollectionError(TypedDict):
    """A collection error type."""

    nodeid: str
    details: str


collection_errors: list[CollectionError] = []


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Pytest collection report hook."""
    if report.failed:
        collection_errors.append(
            {
                "nodeid": report.nodeid,
                "details": str(report.longrepr),
            }
        )


@pytest.fixture(autouse=True)
def celune_test_context(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Celune's text context."""
    del request
    yield


def truecolor_separator(
    text: str,
    r: int,
    g: int,
    b: int,
    sep: str = "=",
) -> str:
    """Return a centered True Color terminal separator."""
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    content = f" {text} "
    remaining = max(0, width - len(content))
    left = remaining // 2
    right = remaining - left
    color = f"\033[38;2;{r};{g};{b}m"
    reset = "\033[0m"
    return f"{color}{sep * left}{content}{sep * right}{reset}"


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(
    session: pytest.Session,
    exitstatus: pytest.ExitCode,
) -> Generator:
    """Pytest session finish hook."""
    yield

    if not collection_errors:
        return

    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")

    if terminalreporter is not None:
        terminalreporter.write_line(
            truecolor_separator("test failure hint", 206, 186, 255)
        )
        terminalreporter.write_line("Try running `uv run pytest`.")
        terminalreporter.write_line(
            "If it still fails, the test environment may be misconfigured."
        )

    sys.exit(exitstatus)
