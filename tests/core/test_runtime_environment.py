# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime and platform environment validation."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

from unittest import mock

import pytest

from celune import runtime
from celune.celune import Celune
from celune.typing.aliases import LogLevel
from celune.ui.app import (
    CeluneUI,
)
from celune.ui.headless import CeluneHeadlessUI

from ..support import CeluneTestCase


class TestRuntimeEnvironment(CeluneTestCase):
    """Tests for runtime environment checks."""

    def test_core_runtime_classes_are_final_at_runtime(self) -> None:
        """Verify final core classes reject runtime subclass creation."""
        final_classes = (Celune, CeluneUI, CeluneHeadlessUI)

        for final_class in final_classes:
            assert getattr(final_class, "__final__", False)
            with pytest.raises(TypeError, match="is final and cannot be subclassed"):
                type(f"{final_class.__name__}Subclass", (final_class,), {})

    def test_check_supported_backends_reports_cpu_cuda_and_rocm(self) -> None:
        """Verify backend labels across supported runtime branches.

        Raises:
            AssertionError: Runtime detection changes unexpectedly.
        """
        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=False),
            mock.patch(
                "celune.runtime.torch.backends.mps.is_available", return_value=False
            ),
        ):
            assert runtime.check_supported_backends() == ("CPU", False)

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", None),
            mock.patch(
                "celune.runtime.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 4090",
            ),
        ):
            assert runtime.check_supported_backends() == ("CUDA", True)

        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", "6.0"),
        ):
            assert runtime.check_supported_backends() == ("ROCm", False)

    def test_check_supported_backends_treats_missing_driver_as_unusable_cuda(
        self,
    ) -> None:
        """Verify missing-driver CUDA init failures do not crash backend detection."""
        with (
            mock.patch("celune.runtime.torch.cuda.is_available", return_value=True),
            mock.patch.object(runtime.torch.version, "hip", None),
            mock.patch(
                "celune.runtime.torch.cuda.get_device_name",
                side_effect=RuntimeError(
                    "Found no NVIDIA driver on your system. Please check that you "
                    "have an NVIDIA GPU and installed a driver from "
                    "http://www.nvidia.com/Download/index.aspx"
                ),
            ),
        ):
            assert runtime.check_supported_backends() == ("CUDA", False)

    def test_validate_runtime_rejects_unsupported_backends_without_cuda_work(
        self,
    ) -> None:
        """Verify unsupported backends fail before CUDA work begins.

        Raises:
            AssertionError: Runtime rejection behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        errors: list[str] = []
        states: list[str] = []

        def log(
            msg: str,
            severity: str = "info",
            *,
            loglevel: LogLevel = "info",
        ) -> None:
            _ = loglevel
            logs.append((msg, severity))

        with (
            mock.patch("celune.runtime.sys.version_info", (3, 12, 0)),
            mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ),
        ):
            assert not runtime.validate_runtime(
                log,
                errors.append,
                states.append,
                False,
                lambda exc, dev: str(exc),
                "info",
                "qwen3",
            )
        assert errors == ["No supported backend found"]
        assert states == ["error"]

    @pytest.mark.parametrize("backend_name", ("mini", "luxtts"))
    def test_validate_runtime_allows_cpu_tts_backends(self, backend_name: str) -> None:
        """Verify CPU-only environments allow both CPU-capable TTS backends."""
        logs: list[tuple[str, str]] = []
        errors: list[str] = []
        states: list[str] = []

        def log(
            msg: str,
            severity: str = "info",
            *,
            loglevel: LogLevel = "info",
        ) -> None:
            _ = loglevel
            logs.append((msg, severity))

        with (
            mock.patch("celune.runtime.sys.version_info", (3, 12, 0)),
            mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ),
        ):
            assert runtime.validate_runtime(
                log,
                errors.append,
                states.append,
                False,
                lambda exc, dev: str(exc),
                "info",
                backend_name,
            )

        assert not errors
        assert not states
