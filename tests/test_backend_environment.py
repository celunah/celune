# SPDX-License-Identifier: Apache-2.0
"""Collect split backend environment tests under the historical test module name."""

from .cedts_worker_protocol import TestBackendEnvironment as _TestBackendEnvironment


class TestBackendEnvironment(_TestBackendEnvironment):
    """Collect backend environment and worker protocol tests."""
