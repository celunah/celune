"""Celune Pytest hooks, handlers, and worker isolation."""
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
import sys
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

import pytest


class CollectionError(TypedDict):
    """A collection error type."""

    nodeid: str
    details: str


class TestFailure(TypedDict):
    """A test execution failure type."""

    nodeid: str
    phase: str
    details: str


collection_errors: list[CollectionError] = []
test_failures: list[TestFailure] = []


def _configure_worker_data_root() -> None:
    """Give each xdist worker an isolated Celune data and cache root."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker_id:
        return

    run_id = os.environ.get("PYTEST_XDIST_TESTRUNUID", "local")
    worker_data_root = Path(
        tempfile.mkdtemp(prefix=f"celune-pytest-{run_id}-{worker_id}-")
    )
    huggingface_root = worker_data_root / "huggingface"
    os.environ["HF_HOME"] = str(huggingface_root)
    os.environ["HF_HUB_CACHE"] = str(huggingface_root / "hub")
    os.environ["NUMBA_CACHE_DIR"] = str(worker_data_root / "numba")

    import celune.paths as celune_paths

    def worker_user_data_dir(
        appname: Optional[str] = None,
        appauthor: Union[Literal[False], str, None] = None,
        version: Optional[str] = None,
        roaming: bool = False,
        ensure_exists: bool = False,
        use_site_for_root: bool = False,
        **_kwargs: object,
    ) -> str:
        """Return this worker's isolated Celune data root."""
        del appname, appauthor, version, roaming, ensure_exists, use_site_for_root
        return str(worker_data_root)

    celune_paths.user_data_dir = worker_user_data_dir
    celune_paths.configure_huggingface_cache_environment()
    atexit.register(shutil.rmtree, worker_data_root, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    """Configure process-local state before test modules are collected."""
    del config
    _configure_worker_data_root()


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Pytest collection report hook."""
    if report.failed:
        collection_errors.append(
            {
                "nodeid": report.nodeid,
                "details": str(report.longrepr),
            }
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Track setup, call, and teardown failures from normal test execution."""
    if report.failed:
        test_failures.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "details": str(report.longrepr),
            }
        )


@pytest.fixture(autouse=True)
def celune_test_context(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Celune's test context."""
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

    if not collection_errors and not test_failures:
        return

    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")

    if terminalreporter is not None:
        terminalreporter.write_line(
            truecolor_separator("test failure hint", 206, 186, 255)
        )
        if collection_errors:
            terminalreporter.write_line(
                f"Collection failures recorded: {len(collection_errors)}"
            )
        if test_failures:
            terminalreporter.write_line(f"Test failures recorded: {len(test_failures)}")
            first_failure = test_failures[0]
            terminalreporter.write_line(
                "Run the first failing test directly with: `"
                f"uv run pytest {first_failure['nodeid']} -vv"
                "`."
            )
        else:
            terminalreporter.write_line(
                "Run collection diagnostics with: `uv run pytest --collect-only -vv`."
            )
        terminalreporter.write_line(
            "If it still fails, inspect the first traceback and verify the test environment."
        )

    sys.exit(exitstatus)
