# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime and platform environment validation."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

from unittest import mock

import pytest

from celune import runtime
from celune.ui.app import (
    CeluneUI,
)
from celune.celune import Celune
from celune.exceptions import RuntimeCheckError
from celune.typing.aliases import LogCallback, LogLevel
from celune.ui.headless import CeluneHeadlessUI

from ..support import CeluneTestCase, FakeBackend, FakeGlow


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

    def test_validate_runtime_allows_pocket_tts_on_cpu(self) -> None:
        """Verify CPU-only environments allow Pocket TTS."""
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
                "mini",
            )

        assert not errors
        assert not states

    def test_gpu_runtime_preflight_precedes_backend_model_loading(self) -> None:
        """Reject a GPU backend before model loading and preserve diagnostics."""
        Celune._instance = None
        backend = FakeBackend()
        backend.is_fake = False
        errors: list[str] = []

        def fail_runtime_check(log: LogCallback, **kwargs: object) -> bool:
            del kwargs
            log(
                "Compute test for GPU 0 failed: RuntimeError: synthetic CUDA failure",
                "warning",
            )
            log(
                "GPU tests reported that all GPUs have failed their runtime checks.",
                "error",
            )
            return False

        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.core.lifecycle.resolve_backend_name", return_value="qwen3"
            ),
            mock.patch(
                "celune.core.loading.validate_runtime",
                side_effect=fail_runtime_check,
            ) as check,
            mock.patch(
                "celune.core.reload.resolve_backend", return_value=backend
            ) as resolve_backend,
            mock.patch.object(backend, "preload_models") as preload_models,
        ):
            engine = Celune(
                config={"vram": "high"},
                tts_backend="qwen3",
                error_callback=errors.append,
            )
            self.addCleanup(setattr, Celune, "_instance", None)
            self.addCleanup(engine.close)
            check.assert_not_called()
            with pytest.raises(RuntimeCheckError, match="synthetic CUDA failure"):
                engine.load(raise_on_error=True)

        check.assert_called_once()
        resolve_backend.assert_called_once()
        preload_models.assert_not_called()
        assert len(errors) == 1
        assert "synthetic CUDA failure" in errors[0]

    def test_cpu_backend_preflight_skips_cuda_compute_check(self) -> None:
        """Allow Mini on CPU without running CUDA device or compute checks."""
        Celune._instance = None

        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
            mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ),
            mock.patch("celune.runtime._run_compute_test") as compute_test,
            mock.patch(
                "celune.core.reload.resolve_backend", return_value=FakeBackend()
            ),
        ):
            celune = Celune(config={"vram": "low"}, tts_backend="mini")
            self.addCleanup(setattr, Celune, "_instance", None)
            self.addCleanup(celune.close)
            with mock.patch("celune.core.voice.play_signal", return_value=False):
                assert celune.load()

        compute_test.assert_not_called()
