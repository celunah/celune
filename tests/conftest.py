"""Celune pytest configuration and worker isolation."""
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional, Union

import pytest


pytest_plugins = ("tests.celtest",)


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


@pytest.fixture(autouse=True)
def celune_test_context(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Celune's test context."""
    del request
    yield
