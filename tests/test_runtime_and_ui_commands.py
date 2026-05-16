# SPDX-License-Identifier: MIT
"""Tests for runtime validation and lightweight UI commands."""

import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from celune.backends.qwen3 import Qwen3
from celune import runtime
from celune.ui.commands import process_command

from celune.ui.app import CeluneUI


class RuntimeTests(unittest.TestCase):
    """Tests for runtime environment checks."""

    def test_check_supported_backends_reports_cpu_cuda_and_rocm(self) -> None:
        """Verify backend labels across supported runtime branches.

        Returns:
            None: Assertions verify runtime detection behavior.

        Raises:
            AssertionError: Runtime detection changes unexpectedly.
        """
        with mock.patch("celune.runtime.torch.cuda.is_available", return_value=False):
            with mock.patch(
                "celune.runtime.torch.backends.mps.is_available", return_value=False
            ):
                self.assertEqual(runtime.check_supported_backends(), ("CPU", False))

        with mock.patch("celune.runtime.torch.cuda.is_available", return_value=True):
            with mock.patch.object(runtime.torch.version, "hip", None):
                self.assertEqual(runtime.check_supported_backends(), ("CUDA", True))

        with mock.patch("celune.runtime.torch.cuda.is_available", return_value=True):
            with mock.patch.object(runtime.torch.version, "hip", "6.0"):
                self.assertEqual(runtime.check_supported_backends(), ("ROCm", False))

    def test_validate_runtime_rejects_unsupported_backends_without_cuda_work(
        self,
    ) -> None:
        """Verify unsupported backends fail before CUDA work begins.

        Returns:
            None: Assertions verify runtime rejection behavior.

        Raises:
            AssertionError: Runtime rejection behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        errors: list[str] = []
        states: list[str] = []

        def log(msg: str, severity: str) -> None:
            logs.append((msg, severity))

        with mock.patch("celune.runtime.sys.version_info", (3, 12, 0)):
            with mock.patch(
                "celune.runtime.check_supported_backends", return_value=("CPU", False)
            ):
                self.assertEqual(
                    runtime.validate_runtime(
                        log,
                        errors.append,
                        states.append,
                        False,
                        lambda exc, dev: str(exc),
                        False,
                    ),
                    False,
                )
        self.assertEqual(errors, ["No supported backend found"])
        self.assertEqual(states, ["error"])


class UICommandTests(unittest.TestCase):
    """Tests for lightweight slash command behavior."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.ui = SimpleNamespace()
        self.ui.input_box = SimpleNamespace(load_text=lambda text: None)
        self.ui.safe_log = lambda msg, severity="info": self.logs.append(
            (msg, severity)
        )
        self.ui.celune = SimpleNamespace(
            backend=SimpleNamespace(),
            voice_prompt=None,
            can_use_rubberband=True,
            speed=1.0,
            reverb=SimpleNamespace(strength=0.0),
        )

    def _process_command(self, command: str, args: list[str]) -> None:
        """Process one command against the typed UI test double.

        Args:
            command: Command name without a leading slash.
            args: Parsed command arguments.

        Returns:
            None: This helper forwards to the production command handler.
        """
        process_command(cast(CeluneUI, self.ui), command, args)

    def test_xvectoronly_command_requires_qwen3_and_valid_value(self) -> None:
        """Verify the Qwen3-only toggle command and argument checks.

        Returns:
            None: Assertions verify command behavior.

        Raises:
            AssertionError: Command behavior changes unexpectedly.
        """
        self._process_command("xvectoronly", [])
        self.assertEqual(self.logs[-1][1], "warning")

        backend = object.__new__(Qwen3)
        backend.x_vector_only = False
        self.ui.celune.backend = backend
        self._process_command("xvectoronly", [])
        self.assertEqual(self.logs[-1], ("Usage: /xvectoronly <true/false>", "warning"))
        self._process_command("xvectoronly", ["maybe"])
        self.assertEqual(self.logs[-1][1], "warning")
        self._process_command("xvectoronly", ["true"])
        self.assertEqual(backend.x_vector_only, True)
        self.assertEqual(
            self.logs[-1], ("Qwen3 identity-only cloning enabled.", "info")
        )

    def test_common_commands_update_state_and_validate_inputs(self) -> None:
        """Verify prompt, speed, and reverb command paths.

        Returns:
            None: Assertions verify command behavior.

        Raises:
            AssertionError: Command behavior changes unexpectedly.
        """
        self._process_command("voiceprompt", ["gentle", "tone"])
        self.assertEqual(self.ui.celune.voice_prompt, "gentle tone")
        self._process_command("voiceprompt", ["clear"])
        self.assertIsNone(self.ui.celune.voice_prompt)

        self._process_command("speed", ["120%"])
        self.assertEqual(self.ui.celune.speed, 1.2)
        self._process_command("speed", ["200"])
        self.assertEqual(self.logs[-1][1], "warning")

        self._process_command("reverb", ["50"])
        self.assertEqual(self.ui.celune.reverb.strength, 0.5)
        self._process_command("reverb", ["150"])
        self.assertEqual(self.logs[-1][1], "warning")
