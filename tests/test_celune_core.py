# SPDX-License-Identifier: MIT
"""Tests for Celune core behavior without real models or GPU work."""

from pathlib import Path
from typing import Any, cast
from unittest import mock, TestCase

from celune.celune import Celune
from celune.exceptions import BackendError

from tests.support import FakeBackend, FakeGlow


class CeluneCoreTests(TestCase):
    """Tests for Celune orchestration without real model work."""

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    def _make_celune(self, config: dict) -> Celune:
        """Build a Celune instance with lightweight fakes.

        Args:
            config: Configuration dictionary supplied to Celune.

        Returns:
            Celune: A Celune instance with fake glow and backend objects.

        Raises:
            BackendError: Celune initialization rejects the supplied config.
        """
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.pyop_is_available", return_value=False),
        ):
            celune = Celune(config=config, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)
            return celune

    def test_constructor_validates_backend_and_chunk_size(self) -> None:
        """Verify constructor validation and derived chunk size behavior.

        Returns:
            None: Assertions verify constructor behavior.

        Raises:
            AssertionError: Constructor behavior changes unexpectedly.
        """
        with self.assertRaisesRegex(BackendError, "no backend set"):
            Celune(config={}, tts_backend=None)

        celune = self._make_celune({})
        self.assertEqual(celune.chunk_size, 8)
        self.assertEqual(getattr(celune.glow, "started"), True)
        celune.close()

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.pyop_is_available", return_value=False),
            self.assertRaisesRegex(BackendError, "invalid chunk length"),
        ):
            Celune(
                config={},
                tts_backend=FakeBackend,
                target_chunk_length=0.65,
            )

    def test_voice_loading_uses_backend_and_bundle_defaults(self) -> None:
        """Verify backend voices and bundle metadata determine defaults.

        Returns:
            None: Assertions verify voice selection behavior.

        Raises:
            AssertionError: Voice loading behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        self.assertEqual(celune.load_available_voices(), True)
        self.assertEqual(celune.voices, ("balanced", "bold"))
        self.assertEqual(celune.current_voice, "balanced")

        fake_bundle = mock.Mock()
        fake_bundle.voice_order = ("bold", "balanced")
        fake_bundle.metadata = {"default_voice": "bold"}
        fake_loader = mock.Mock(bundle=fake_bundle)
        celune.backend.uses_voice_bundles = True
        with mock.patch("celune.celune.default_loader", return_value=fake_loader):
            self.assertEqual(celune.load_voice_bundle(Path("fixture.cevoice")), True)
        self.assertEqual(celune.current_voice, "bold")

    def test_pyop_connection_starts_detached_companion(self) -> None:
        """Verify Celune connects to PYOP through the local detached API boundary.

        Returns:
            None: Assertions verify PYOP process startup and client setup.

        Raises:
            AssertionError: PYOP connection behavior changes unexpectedly.
        """
        client = mock.Mock()
        process = mock.Mock()
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch(
                "celune.celune.pyop_is_available",
                side_effect=[False, True, True],
            ) as available,
            mock.patch(
                "celune.celune.start_pyop_detached", return_value=process
            ) as start,
            mock.patch("celune.celune.stop_pyop_process") as stop,
            mock.patch("celune.celune.httpx.Client", return_value=client) as client_cls,
        ):
            celune = Celune(config={"pyop": {"enabled": True}}, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)

            self.assertIs(celune.vision, client)
            self.assertIs(celune._pyop_process, process)
            start.assert_called_once()
            client_cls.assert_called_once()
            self.assertEqual(
                available.call_args_list[0].args[0], "http://127.0.0.1:2061"
            )
            celune.close()
            stop.assert_called_once_with(process)

    def test_pyop_launcher_passes_default_model_to_detached_process(self) -> None:
        """Verify the detached PYOP API receives the default model id.

        Returns:
            None: Assertions verify launcher environment setup.

        Raises:
            AssertionError: PYOP launcher model configuration changes unexpectedly.
        """
        from celune import pyop

        with (
            mock.patch("celune.pyop.pyop_python", return_value=Path("pyop-python.exe")),
            mock.patch("celune.pyop.pyop_entrypoint", return_value="-m pyop_api"),
            mock.patch("celune.pyop.Path.exists", return_value=True),
            mock.patch("celune.pyop.sys.executable", "celune-python.exe"),
            mock.patch("celune.pyop.subprocess.Popen") as popen,
        ):
            process = pyop.start_pyop_detached()

        self.assertIs(process, popen.return_value)

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["PYOP_MODEL"], "lunahr/pyop-2b")
        self.assertEqual(env["PYOP_QUANTIZED"], "1")

    def test_pyop_launcher_hides_windows_console(self) -> None:
        """Verify the detached PYOP API does not leave a console window open."""
        from celune import pyop

        with (
            mock.patch("celune.pyop.pyop_python", return_value=Path("pyop-python.exe")),
            mock.patch("celune.pyop.pyop_entrypoint", return_value="-m pyop_api"),
            mock.patch("celune.pyop.Path.exists", return_value=True),
            mock.patch("celune.pyop.sys.executable", "celune-python.exe"),
            mock.patch("celune.pyop.sys.platform", "win32"),
            mock.patch(
                "celune.pyop.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True
            ),
            mock.patch(
                "celune.pyop.subprocess.CREATE_NEW_PROCESS_GROUP",
                0x00000200,
                create=True,
            ),
            mock.patch("celune.pyop.subprocess.Popen") as popen,
        ):
            process = pyop.start_pyop_detached()

        self.assertIs(process, popen.return_value)
        flags = popen.call_args.kwargs["creationflags"]
        self.assertEqual(flags & 0x08000000, 0x08000000)
        self.assertEqual(flags & 0x00000200, 0x00000200)
        self.assertEqual(popen.call_args.kwargs["start_new_session"], False)

    def test_pyop_talkback_config_can_disable_persona_input_mode(self) -> None:
        """Verify persona talkback can be disabled without disabling PYOP."""
        from celune.pyop import pyop_enabled, pyop_talkback_enabled

        config = {"pyop": {"enabled": True, "talkback": False}}
        self.assertEqual(pyop_enabled(config), True)
        self.assertEqual(pyop_talkback_enabled(config), False)
        self.assertEqual(pyop_talkback_enabled({"pyop": {}}), True)

    def test_think_reconnects_to_pyop_before_speech_fallback(self) -> None:
        """Verify stale Celune instances reconnect to PYOP on the next think call.

        Returns:
            None: Assertions verify lazy PYOP reconnect behavior.

        Raises:
            AssertionError: PYOP reconnect behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.vision = None
        celune.locked = False
        celune.cur_state = "idle"
        client = mock.Mock()
        celune._pyop_conn = mock.Mock(return_value=client)
        with (
            mock.patch("celune.celune.think_pipeline", return_value=True) as think,
            mock.patch.object(celune, "say", return_value=False) as say,
        ):
            self.assertEqual(celune.think("hello"), True)
            pyop_thread = celune._pyop_thread
            self.assertIsNotNone(pyop_thread)
            assert pyop_thread is not None
            pyop_thread.join(timeout=2)

        self.assertIs(celune.vision, client)
        think.assert_called_once_with(celune, "hello")
        say.assert_not_called()

    def test_logging_waiting_and_api_settings_cover_edge_cases(self) -> None:
        """Verify logging gates, readiness checks, and API fallbacks.

        Returns:
            None: Assertions verify core utility behavior.

        Raises:
            AssertionError: Core utility behavior changes unexpectedly.
        """
        logs: list[tuple[str, str]] = []
        celune = self._make_celune(
            {"api": {"port": "bad", "rate_limit_per_minute": "bad"}}
        )
        celune.log_callback = lambda msg, severity="info": logs.append((msg, severity))
        celune.log("hello")
        self.assertEqual(logs[-1], ("hello", "info"))
        celune.log_dev("hidden")
        self.assertEqual(len(logs), 1)
        celune.dev = True
        celune.log_dev("visible")
        self.assertEqual(logs[-1], ("visible", "info"))

        celune.loaded = False
        self.assertEqual(celune._wait_until_idle(timeout=0), False)
        celune.loaded = True
        celune.locked = False
        self.assertEqual(celune._wait_until_idle(timeout=0), True)

        self.assertEqual(
            celune._api_settings(),
            (True, "127.0.0.1", 2060, None, 60),
        )
        self.assertEqual(logs[-2][1], "warning")
        self.assertEqual(logs[-1][1], "warning")

    def test_load_success_and_model_failure_paths_are_stubbed(self) -> None:
        """Verify successful startup and default-model failure handling.

        Returns:
            None: Assertions verify startup behavior.

        Raises:
            AssertionError: Startup behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.setup_extensions = mock.Mock()
        celune._warmup = mock.Mock(return_value=True)
        celune._start_configured_api = mock.Mock()
        celune.backend.preload_models = mock.Mock()
        celune.backend.load_default_model = mock.Mock(return_value={"model": "ok"})
        celune.backend.model_id_for_voice = mock.Mock(return_value="fake/balanced")
        with (
            mock.patch("celune.celune.threading.Thread") as thread_cls,
            mock.patch("celune.celune.validate_runtime", return_value=True),
            mock.patch("celune.celune.play_readiness_signal", return_value=False),
        ):
            thread_cls.return_value.start = mock.Mock()
            self.assertEqual(celune.load(), True)
        self.assertEqual(celune.loaded, True)
        self.assertEqual(getattr(celune.glow, "entered"), True)
        celune.close()

        failing = self._make_celune({})
        failing.setup_extensions = mock.Mock()
        failing.backend.preload_models = mock.Mock()
        failing.backend.load_default_model = mock.Mock(side_effect=RuntimeError("boom"))
        errors: list[str] = []
        failing.error_callback = errors.append
        self.assertEqual(failing.load(), False)
        self.assertEqual(errors, ["Default model failed to load"])

    def test_unload_runtime_state_clears_models_without_cuda(self) -> None:
        """Verify model references are cleared without touching CUDA.

        Returns:
            None: Assertions verify unload behavior.

        Raises:
            AssertionError: Unload behavior changes unexpectedly.
        """
        celune = self._make_celune({})
        celune.model = cast(Any, object())
        celune.llm = cast(Any, object())
        celune.tokenizer = cast(Any, object())
        celune.backend.model = object()
        with mock.patch("celune.celune.torch.cuda.is_available", return_value=False):
            celune.unload_runtime_state(include_normalizer=True)
        self.assertIsNone(celune.model)
        self.assertIsNone(celune.llm)
        self.assertIsNone(celune.tokenizer)
        self.assertIsNone(celune.backend.model)
