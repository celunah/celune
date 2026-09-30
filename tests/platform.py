# SPDX-License-Identifier: Apache-2.0
"""Platform markers for tests that require native operating-system behavior."""

import sys

import pytest

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux",
)
WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="requires Windows",
)
