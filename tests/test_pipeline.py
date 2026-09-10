# SPDX-License-Identifier: Apache-2.0
"""Collect split pipeline tests under the historical test module name."""

import pytest

from .pipeline_basics import TestPipeline as _TestPipeline
from .pipeline_output import TestPipelineAsync as _TestPipelineAsync


class TestPipeline(_TestPipeline):
    """Collect baseline pipeline tests."""


@pytest.mark.anyio
class TestPipelineAsync(_TestPipelineAsync):
    """Collect persona and output pipeline tests."""
