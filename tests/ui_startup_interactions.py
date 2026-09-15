# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime validation and lightweight UI commands."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import time
import asyncio
import logging
import tempfile
import threading
from types import SimpleNamespace
from typing import Optional, cast
from pathlib import Path
from unittest import mock
from collections.abc import Callable

import numpy as np
import pytest
from textual import events
from textual.widget import Widget
from textual.widgets import Label, Button, RichLog, TextArea, ProgressBar

from celune.ui import app as ui_app
from celune.ui import resources as ui_resources
from celune.i18n import string
from celune.theme import colors
from celune.utils import discard
from celune.celune import Celune
from celune.config import Config
from celune.ui.app import (
    ButtonActions,
    CeluneUI,
    ProgressLabel,
    VoiceButton,
)
from tests.support import FakeBackend
from celune.ui.theme import severity_color
from celune.persona.asr import WhisperWord, WhisperSegment
from celune.typing.common import JSONSerializable
from .ui_startup_foundation import TestUIStartup as _TestUIStartup


class TestUIStartup(_TestUIStartup):
    """Exercise interactive startup and shutdown behavior."""

    def test_vc_recording_vad_plays_leading_silence(self) -> None:
        """Verify live VC converts and plays silence while its VAD gate is closed."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        converted_chunks: list[np.ndarray] = []
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        converted_chunks.append(
                            np.asarray(audio, dtype=np.float32).copy()
                        )
                        or SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                is_in_tutorial=False,
                dev=False,
                config={"input_device": "Stereo Mix (Realtek)"},
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None

        def invoke_captured_callback(
            callback: ui_app._VCAudioCallback,
            audio: np.ndarray,
        ) -> None:
            callback(
                audio,
                len(audio),
                None,
                None,
            )

        class FakeInputStream:
            """Tiny input-stream fake for VC VAD recording tests."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock()
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ),
            mock.patch("celune.ui.app.sd.InputStream", side_effect=FakeInputStream),
            mock.patch(
                "celune.ui.app.queue_streaming_sfx_audio", return_value=1
            ) as queue_stream,
            mock.patch("celune.ui.app.finish_streaming_sfx_audio"),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            if captured_callback is None or not callable(captured_callback):
                pytest.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    captured_callback,
                    np.full((48000, 2), 0.001, dtype=np.float32),
                )
                for _ in range(50):
                    if converted_chunks:
                        break
                    time.sleep(0.01)

            stop_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, stop_event))

        self.assertTrue(converted_chunks)
        self.assertTrue(np.allclose(converted_chunks[0], 0.0))
        queue_stream.assert_called_once()

    def test_vc_recording_vad_preroll_keeps_speech_onset(self) -> None:
        """Verify live VC prepends a short preroll so VAD onset is not clipped."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        converted_chunks: list[np.ndarray] = []
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        converted_chunks.append(
                            np.asarray(audio, dtype=np.float32).copy()
                        )
                        or SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                is_in_tutorial=False,
                dev=False,
                config={"input_device": "Stereo Mix (Realtek)"},
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None

        class FakeAIVAD:
            """Tiny AI VAD stub with a deterministic speech onset."""

            def has_voice(self, audio: np.ndarray, sample_rate: int) -> bool:
                """Report only the loud synthetic buffer as speech.

                Args:
                    audio: Input audio block inspected by the fake detector.
                    sample_rate: Input audio sample rate.

                Returns:
                    bool: Whether the fake detector considers speech active.
                """
                del sample_rate
                return float(np.max(np.abs(audio))) >= 0.05

            def reset(self) -> None:
                """Reset the fake detector state."""

        fake_vad = FakeAIVAD()

        def invoke_captured_callback(
            callback: ui_app._VCAudioCallback,
            audio: np.ndarray,
        ) -> None:
            callback(
                audio,
                len(audio),
                None,
                None,
            )

        class FakeInputStream:
            """Tiny input-stream fake for VC VAD preroll tests."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock(
                    side_effect=lambda: invoke_captured_callback(
                        captured_callback,
                        np.full((1000, 2), 0.004, dtype=np.float32),
                    )
                )
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector",
                return_value=fake_vad,
            ),
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ),
            mock.patch("celune.ui.app.sd.InputStream", side_effect=FakeInputStream),
            mock.patch("celune.ui.app.queue_streaming_sfx_audio", return_value=1),
            mock.patch("celune.ui.app.finish_streaming_sfx_audio"),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            if captured_callback is None or not callable(captured_callback):
                pytest.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    captured_callback,
                    np.full((1000, 2), 0.004, dtype=np.float32),
                )
                invoke_captured_callback(
                    captured_callback,
                    np.full((2000, 2), 0.05, dtype=np.float32),
                )
                time.sleep(0.05)

            stop_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, stop_event))

        speech_chunks = [
            chunk for chunk in converted_chunks if not np.allclose(chunk, 0.0)
        ]
        self.assertTrue(speech_chunks)
        self.assertGreaterEqual(len(speech_chunks[0]), 3000)
        self.assertAlmostEqual(float(speech_chunks[0][0, 0]), 0.004)
        self.assertAlmostEqual(float(speech_chunks[0][-1, 0]), 0.05)

    def test_vc_recording_flushes_active_speech_before_end_of_phrase(self) -> None:
        """Verify live VC can submit one mid-speech chunk before silence arrives."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        converted_chunks: list[np.ndarray] = []
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        converted_chunks.append(
                            np.asarray(audio, dtype=np.float32).copy()
                        )
                        or SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                is_in_tutorial=False,
                dev=False,
                config={"input_device": "Stereo Mix (Realtek)"},
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None

        def invoke_captured_callback(
            callback: ui_app._VCAudioCallback,
            audio: np.ndarray,
        ) -> None:
            callback(
                audio,
                len(audio),
                None,
                None,
            )

        class FakeInputStream:
            """Tiny input-stream fake for live VC mid-speech chunk flushing."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock()
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector",
                return_value=None,
            ),
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ),
            mock.patch("celune.ui.app.sd.InputStream", side_effect=FakeInputStream),
            mock.patch(
                "celune.ui.app.queue_streaming_sfx_audio", return_value=1
            ) as queue_stream,
            mock.patch("celune.ui.app.finish_streaming_sfx_audio"),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            if captured_callback is None or not callable(captured_callback):
                pytest.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    captured_callback,
                    np.full((40000, 2), 0.05, dtype=np.float32),
                )
                invoke_captured_callback(
                    captured_callback,
                    np.full((40000, 2), 0.05, dtype=np.float32),
                )
                time.sleep(0.05)

            assert cast(mock.Mock, ui.celune.convert_audio).call_count >= 1
            assert (
                queue_stream.call_args.kwargs["status_label_key"]
                == "pipeline.revoicing_label"
            )

            stop_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, stop_event))

        assert converted_chunks
        assert converted_chunks[0].shape[0] >= 17000

    def test_vc_recording_prefers_ai_vad_when_available(self) -> None:
        """Verify live VC can use the optional AI VAD instead of the RMS fallback."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                is_in_tutorial=False,
                dev=False,
                config={"input_device": "Stereo Mix (Realtek)"},
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None

        class FakeAIVAD:
            """Tiny AI VAD stub forcing every callback to non-speech."""

            def __init__(self) -> None:
                self.calls = 0
                self.last_shape = ()
                self.last_sample_rate = 0

            def has_voice(self, audio: np.ndarray, sample_rate: int) -> bool:
                """Report every callback as non-speech.

                Args:
                    audio: Input audio block inspected by the fake detector.
                    sample_rate: Input audio sample rate.

                Returns:
                    bool: Always ``False`` so silent VC conversion is queued.
                """
                self.calls += 1
                self.last_shape = audio.shape
                self.last_sample_rate = sample_rate
                return False

            def reset(self) -> None:
                """Reset the fake detector state."""

        fake_vad = FakeAIVAD()

        def invoke_captured_callback(
            callback: ui_app._VCAudioCallback,
            audio: np.ndarray,
        ) -> None:
            callback(
                audio,
                len(audio),
                None,
                None,
            )

        class FakeInputStream:
            """Tiny input-stream fake for VC AI VAD tests."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock()
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector",
                return_value=fake_vad,
            ),
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ),
            mock.patch("celune.ui.app.sd.InputStream", side_effect=FakeInputStream),
            mock.patch("celune.ui.app.queue_streaming_sfx_audio", return_value=1),
            mock.patch("celune.ui.app.finish_streaming_sfx_audio"),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            if captured_callback is None or not callable(captured_callback):
                pytest.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    captured_callback,
                    np.full((120000, 2), 0.2, dtype=np.float32),
                )
                time.sleep(0.05)

            stop_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, stop_event))

        self.assertGreaterEqual(fake_vad.calls, 1)
        self.assertGreaterEqual(cast(mock.Mock, ui.celune.convert_audio).call_count, 1)

    def test_vc_recording_ai_vad_exception_keeps_detector_active(self) -> None:
        """Verify one AI VAD callback failure does not disable AI VAD forever."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                convert_audio=mock.Mock(
                    side_effect=lambda audio, sample_rate, label=None, **_kwargs: (
                        SimpleNamespace(
                            audio=np.asarray(audio, dtype=np.float32).copy(),
                            sample_rate=sample_rate,
                            label=label or "Stereo Mix",
                        )
                    )
                ),
                is_in_tutorial=False,
                dev=False,
                config={"input_device": "Stereo Mix (Realtek)"},
            ),
        )
        ui.safe_log = mock.Mock()
        ui.update_resources = mock.Mock()
        captured_callback: Optional[ui_app._VCAudioCallback] = None

        class FlakyAIVAD:
            """Tiny AI VAD stub that fails once and then keeps working."""

            def __init__(self) -> None:
                self.calls = 0
                self.reset_calls = 0

            def has_voice(self, audio: np.ndarray, sample_rate: int) -> bool:
                """Raise once, then report non-speech.

                Args:
                    audio: Input audio block inspected by the fake detector.
                    sample_rate: Input audio sample rate.

                Returns:
                    bool: ``False`` after the first forced failure.

                Raises:
                    RuntimeError: The first detector call fails intentionally.
                """
                self.calls += 1
                discard(audio)
                discard(sample_rate)
                if self.calls == 1:
                    raise RuntimeError("test failure")
                return False

            def reset(self) -> None:
                """Reset the fake detector state."""
                self.reset_calls += 1

        fake_vad = FlakyAIVAD()

        def invoke_captured_callback(
            callback: ui_app._VCAudioCallback,
            audio: np.ndarray,
        ) -> None:
            callback(
                audio,
                len(audio),
                None,
                None,
            )

        class FakeInputStream:
            """Tiny input-stream fake for VC AI VAD recovery tests."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = kwargs["callback"]
                self.start = mock.Mock()
                self.stop = mock.Mock()
                self.close = mock.Mock()

        with (
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector",
                return_value=fake_vad,
            ),
            mock.patch(
                "celune.ui.app.sd.query_devices",
                return_value={
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                    "name": "Stereo Mix",
                },
            ),
            mock.patch("celune.ui.app.sd.InputStream", side_effect=FakeInputStream),
            mock.patch("celune.ui.app.queue_streaming_sfx_audio", return_value=1),
            mock.patch("celune.ui.app.finish_streaming_sfx_audio"),
        ):
            start_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, start_event))

            if captured_callback is None or not callable(captured_callback):
                pytest.fail("recording callback was not registered")
            else:
                invoke_captured_callback(
                    captured_callback,
                    np.full((2048, 2), 0.2, dtype=np.float32),
                )
                invoke_captured_callback(
                    captured_callback,
                    np.full((2048, 2), 0.2, dtype=np.float32),
                )
                time.sleep(0.05)

            stop_event = SimpleNamespace(
                key="ctrl+r",
                prevent_default=mock.Mock(),
                stop=mock.Mock(),
            )
            ui.on_key(cast(events.Key, stop_event))

        assert fake_vad.calls == 2
        assert fake_vad.reset_calls == 1

    def test_ctrl_r_wakes_sleeping_celune_without_starting_recording(self) -> None:
        """Verify CTRL+R wakes a sleeping VC runtime instead of starting capture."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                sleeping=True,
                cur_state="sleeping",
                config={},
            ),
        )
        ui.safe_status = mock.Mock()
        ui.change_input_state = mock.Mock()
        ui.wake_from_sleep = mock.Mock()
        ui._cancel_sleep_timer = mock.Mock()
        ui.toggle_vc_recording = mock.Mock(return_value=True)

        start_event = SimpleNamespace(
            key="ctrl+r",
            prevent_default=mock.Mock(),
            stop=mock.Mock(),
        )

        ui.on_key(cast(events.Key, start_event))

        ui._cancel_sleep_timer.assert_called_once()
        ui.safe_status.assert_called_once_with("Waking up")
        ui.change_input_state.assert_called_once_with(locked=True)
        ui.wake_from_sleep.assert_called_once_with()
        ui.toggle_vc_recording.assert_not_called()
        start_event.prevent_default.assert_called_once_with()
        start_event.stop.assert_called_once_with()

    def test_graceful_exit_uses_live_vc_shutdown_path(self) -> None:
        """Verify app exit stops live VC before closing the runtime."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(Celune, SimpleNamespace(close=mock.Mock()))
        ui._shutdown_live_vc_recording = mock.Mock()
        ui.exit = mock.Mock()

        ui._graceful_exit()

        assert ui.cur_state == "exiting"
        ui._shutdown_live_vc_recording.assert_called_once_with()
        cast(mock.Mock, ui.celune.close).assert_called_once_with()
        ui.exit.assert_called_once_with()

    def test_graceful_exit_fades_screen_before_unmounting(self) -> None:
        """Verify the visible screen fades before Textual is asked to exit."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui._shutdown_runtime = mock.Mock()
        ui.exit = mock.Mock()
        screen_styles = SimpleNamespace(
            opacity=1.0,
            scrollbar_size_vertical=2,
            scrollbar_size_horizontal=1,
        )
        screen = SimpleNamespace(
            styles=screen_styles,
            query=mock.Mock(),
            refresh=mock.Mock(),
            _vertical_scrollbar=SimpleNamespace(display=True),
            _horizontal_scrollbar=SimpleNamespace(display=True),
            _scrollbar_corner=SimpleNamespace(display=True),
            show_vertical_scrollbar=True,
            show_horizontal_scrollbar=True,
        )
        child_styles = SimpleNamespace(
            scrollbar_size_vertical=2,
            scrollbar_size_horizontal=1,
        )
        child = SimpleNamespace(
            styles=child_styles,
            refresh=mock.Mock(),
            _vertical_scrollbar=SimpleNamespace(display=True),
            _horizontal_scrollbar=SimpleNamespace(display=True),
            _scrollbar_corner=SimpleNamespace(display=True),
            show_vertical_scrollbar=True,
            show_horizontal_scrollbar=True,
        )
        screen.query.return_value = (child,)
        fade_complete: list[Callable[[], None]] = []
        after_refresh: list[Callable[[], None]] = []

        def capture_fade(*_args, on_complete=None, **_kwargs) -> None:
            if on_complete is not None:
                fade_complete.append(on_complete)

        def capture_after_refresh(callback: Callable[[], None]) -> None:
            after_refresh.append(callback)

        with (
            mock.patch.object(
                CeluneUI,
                "screen",
                new_callable=mock.PropertyMock,
                return_value=screen,
            ),
            mock.patch.object(ui, "_animate_opacity", side_effect=capture_fade) as fade,
            mock.patch.object(
                ui,
                "call_after_refresh",
                side_effect=capture_after_refresh,
            ),
        ):
            ui._graceful_exit()

            fade.assert_called_once_with(
                screen,
                0.0,
                on_complete=mock.ANY,
                duration=ui_app._EXIT_FADE_SECONDS,
            )
            ui._shutdown_runtime.assert_not_called()
            ui.exit.assert_not_called()
            assert len(fade_complete) == 1

            fade_complete[0]()

            assert screen.refresh.call_args == mock.call(repaint=True)
            assert screen.styles.opacity == 0.0
            for widget in (screen, child):
                assert widget.styles.scrollbar_size_vertical == 0
                assert widget.styles.scrollbar_size_horizontal == 0
                assert not widget.show_vertical_scrollbar
                assert not widget.show_horizontal_scrollbar
                assert not widget._vertical_scrollbar.display
                assert not widget._horizontal_scrollbar.display
                assert not widget._scrollbar_corner.display
            ui._shutdown_runtime.assert_not_called()
            ui.exit.assert_not_called()
            assert len(after_refresh) == 1

            after_refresh[0]()

        ui._shutdown_runtime.assert_called_once_with()
        ui.exit.assert_called_once_with()

    def test_graceful_exit_survives_shutdown_exception(self) -> None:
        """Verify a core shutdown error cannot escape the exit handler."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui._old_stderr = mock.Mock()
        ui._persist_log_entry = mock.Mock()
        ui._shutdown_runtime = mock.Mock(
            side_effect=RuntimeError("pipeline close failed")
        )
        ui.exit = mock.Mock()

        with mock.patch(
            "celune.ui.app.format_error_message",
            return_value="An internal error occurred: pipeline close failed",
        ):
            ui._graceful_exit()

        ui.exit.assert_called_once_with()
        ui._old_stderr.write.assert_called_once()
        self.assertIn("pipeline close failed", ui._old_stderr.write.call_args.args[0])
        ui._persist_log_entry.assert_any_call(
            mock.ANY,
            "error",
        )

    def test_shutdown_error_persists_traceback(self) -> None:
        """Verify shutdown diagnostics remain available after the UI closes."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui._old_stderr = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            ui._log_file_path = Path(temp_dir) / "celune.log"
            with mock.patch(
                "celune.ui.app.format_error_message",
                return_value=(
                    "An internal error occurred: pipeline close failed\ntraceback text"
                ),
            ):
                ui._report_shutdown_error(RuntimeError("pipeline close failed"))

            persisted = ui._log_file_path.read_text(encoding="utf-8")

        self.assertIn("pipeline close failed", persisted)
        self.assertIn("traceback text", persisted)

    def test_runtime_shutdown_continues_after_live_input_error(self) -> None:
        """Verify core teardown still runs when live input cleanup fails."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(Celune, SimpleNamespace(close=mock.Mock()))
        ui._shutdown_live_vc_recording = mock.Mock(
            side_effect=RuntimeError("input close failed")
        )
        ui._report_shutdown_error = mock.Mock()

        ui._shutdown_runtime()

        cast(mock.Mock, ui.celune.close).assert_called_once_with()
        ui._report_shutdown_error.assert_called_once()

    def test_unmount_survives_shutdown_exception(self) -> None:
        """Verify late unmount cleanup cannot turn normal exit into an app error."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui._shutdown_runtime = mock.Mock(
            side_effect=RuntimeError("runtime close failed")
        )
        ui._report_shutdown_error = mock.Mock()

        ui.on_unmount()

        ui._report_shutdown_error.assert_called_once()

    def test_runtime_shutdown_is_idempotent(self) -> None:
        """Verify graceful exit and unmount close the core only once."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(Celune, SimpleNamespace(close=mock.Mock()))
        ui._shutdown_live_vc_recording = mock.Mock()
        ui.exit = mock.Mock()

        ui._graceful_exit()
        ui.on_unmount()

        ui._shutdown_live_vc_recording.assert_called_once_with()
        cast(mock.Mock, ui.celune.close).assert_called_once_with()

    def test_launcher_loss_uses_graceful_exit_path(self) -> None:
        """Verify a launcher disconnect routes through normal UI cleanup."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(Celune, SimpleNamespace(close=mock.Mock()))
        ui._shutdown_live_vc_recording = mock.Mock()
        ui.exit = mock.Mock()

        with mock.patch("celune.ui.app.launcher_loss_requested", return_value=True):
            ui._check_launcher_loss()

        assert ui.cur_state == "exiting"
        ui._shutdown_live_vc_recording.assert_called_once_with()
        cast(mock.Mock, ui.celune.close).assert_called_once_with()
        ui.exit.assert_called_once_with()

    def test_start_vc_recording_logs_ambiguous_input_device_as_warning(self) -> None:
        """Verify ambiguous VC input device matches do not bubble up as exceptions."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                input_mode="voice_conversion",
                vc_backend=SimpleNamespace(),
                sleeping=False,
                cur_state="ready",
                config={"input_device": "Razer Kraken V4 - Chat"},
            ),
        )
        ui.safe_log = mock.Mock()

        with (
            mock.patch.object(ui_app, "_RUNTIME_DEPENDENCIES_LOADED", True),
            mock.patch(
                "celune.ui.app.resolve_audio_device_with_info",
                side_effect=ValueError("ambiguous input device"),
            ),
        ):
            assert not ui._start_vc_recording()

        ui.safe_log.assert_called_once_with("ambiguous input device", "warning")

    def test_gpu_usage_returns_cached_sample_without_starting_a_thread(self) -> None:
        """Verify synchronous footer rendering only reads the async sampler cache."""
        with mock.patch("celune.ui.resources._NVIDIA_SMI", "nvidia-smi"):
            previous_tasks = dict(ui_resources._NVIDIA_SMI_TASKS)
            previous_usage = ui_resources._NVIDIA_SMI_USAGE
            ui_resources._NVIDIA_SMI_TASKS.clear()
            ui_resources._NVIDIA_SMI_USAGE = 42
            try:
                assert ui_resources.gpu_usage() == 42
                assert not ui_resources._NVIDIA_SMI_TASKS
            finally:
                ui_resources._NVIDIA_SMI_TASKS.clear()
                ui_resources._NVIDIA_SMI_TASKS.update(previous_tasks)
                ui_resources._NVIDIA_SMI_USAGE = previous_usage

    def test_gpu_usage_worker_is_a_native_async_task(self) -> None:
        """Verify GPU polling is hosted by the current event loop."""

        async def start_and_stop() -> None:
            previous_tasks = dict(ui_resources._NVIDIA_SMI_TASKS)
            try:
                ui_resources._NVIDIA_SMI_TASKS.clear()
                with (
                    mock.patch("celune.ui.resources._NVIDIA_SMI", "nvidia-smi"),
                    mock.patch.object(
                        ui_resources,
                        "_gpu_usage_worker",
                        new=mock.AsyncMock(),
                    ),
                ):
                    ui_resources.start_gpu_usage_worker()
                    task = ui_resources._NVIDIA_SMI_TASKS.get(
                        asyncio.get_running_loop()
                    )
                    assert isinstance(task, asyncio.Task)
                    assert task.get_loop() is asyncio.get_running_loop()
                    ui_resources.stop_gpu_usage_worker()
                    await asyncio.sleep(0)
            finally:
                ui_resources._NVIDIA_SMI_TASKS.clear()
                ui_resources._NVIDIA_SMI_TASKS.update(previous_tasks)

        asyncio.run(start_and_stop())

    def test_gpu_usage_query_times_out(self) -> None:
        """Verify a stuck nvidia-smi query becomes an unavailable sample."""

        class FakeProcess:
            """Minimal async subprocess fake for timeout cleanup."""

            returncode: Optional[int] = None
            killed = False

            def kill(self) -> None:
                """Record process termination."""
                self.killed = True
                self.returncode = -9

            async def communicate(self) -> tuple[bytes, bytes]:
                """Raise until the timeout handler terminates the process."""
                if not self.killed:
                    raise TimeoutError
                return b"", b""

        process = FakeProcess()

        async def create_process(*_args: object, **_kwargs: object) -> FakeProcess:
            """Return the fake GPU query process."""
            return process

        with (
            mock.patch("celune.ui.resources._NVIDIA_SMI", "nvidia-smi"),
            mock.patch(
                "celune.ui.resources.asyncio.create_subprocess_exec",
                side_effect=create_process,
            ) as create,
        ):
            assert asyncio.run(ui_resources._query_gpu_usage()) is None

        create.assert_called()
        assert process.killed

    def test_textual_input_lock_update_with_persona_on_ui_thread(self) -> None:
        """Verify input state updates update with Persona."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=False),
        )
        ui.resources = cast(Label, None)
        persona_config: Config = {"talkback": True}
        ui.celune = cast(
            Celune,
            SimpleNamespace(config={"persona": cast(JSONSerializable, persona_config)}),
        )

        with mock.patch.object(ui, "_persona_loaded") as available:
            ui.change_input_state(locked=True)

        assert ui.input_box.placeholder == "Please wait"
        assert ui.style_button.actions == ButtonActions(press=False, hold=False)
        assert not ui.style_button.disabled
        available.assert_not_called()

        with (
            mock.patch.object(ui, "_persona_loaded") as available,
            mock.patch("celune.ui.app.threading.Thread") as thread_cls,
        ):
            ui.change_input_state(locked=False)

        assert ui.input_box.placeholder == string("ui.input_placeholder")
        assert ui.style_button.actions == ButtonActions(press=True, hold=True)
        assert not ui.style_button.disabled
        available.assert_called_once_with()
        thread_cls.return_value.start.assert_called_once()

    def test_change_input_state_reveals_vc_buttons_after_backend_switch(self) -> None:
        """Verify VC controls appear when the UI unlocks into voice conversion mode."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=False),
        )
        ui.vc_mode_button = Button(string("ui.vc_mode_talk"))
        ui.vc_pitch_button = Button(string("ui.vc_pitch_button", value="+0"))
        ui.resources = cast(Label, None)
        ui._input_locked = True
        ui.vc_mode_button.display = False
        ui.vc_pitch_button.display = False
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                vc_backend=SimpleNamespace(),
                vc_f0_condition=False,
                vc_pitch_shift=0,
            ),
        )

        with mock.patch("celune.ui.app.threading.Thread") as thread_cls:
            ui.change_input_state(locked=False)

        assert ui.vc_mode_button.display
        assert ui.vc_pitch_button.display
        assert not ui.vc_mode_button.disabled
        assert not ui.vc_pitch_button.disabled
        thread_cls.return_value.start.assert_called_once()

    def test_placeholder_uses_loaded_persona_not_runtime_capability(self) -> None:
        """Verify the input placeholder reflects whether Persona actually loaded."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=False),
        )
        ui.resources = cast(Label, None)
        persona_config: Config = {"enabled": True, "talkback": True}
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={
                    "vram": "high",
                    "persona": cast(JSONSerializable, persona_config),
                },
                vision=None,
            ),
        )

        ui._persona_available = ui.persona_loaded()
        assert ui.normal_input_placeholder() == string("ui.input_placeholder")

        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={
                    "vram": "high",
                    "persona": cast(JSONSerializable, persona_config),
                },
                vision=SimpleNamespace(),
            ),
        )
        ui._persona_available = ui.persona_loaded()
        assert ui.normal_input_placeholder() == string("ui.say_placeholder")

    def test_placeholder_uses_voice_changer_text_when_vc_backend_is_selected(
        self,
    ) -> None:
        """Verify the input placeholder follows VC backend selection."""
        ui = CeluneUI()
        ui.input_box = TextArea()
        ui.style_button = VoiceButton(
            "Voice",
            actions=ButtonActions(press=False, hold=False),
        )
        ui.resources = cast(Label, None)
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                config={},
                vc_backend=SimpleNamespace(),
                vision=SimpleNamespace(),
            ),
        )

        assert ui.normal_input_placeholder() == string("ui.voice_changer_placeholder")

    def test_refresh_vc_controls_hides_buttons_outside_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC-only buttons collapse away while the UI is in TTS mode."""
        ui = CeluneUI()
        ui.vc_mode_button = Button(string("ui.vc_mode_talk"))
        ui.vc_pitch_button = Button(string("ui.vc_pitch_button", value="+0"))
        ui._cancel_vc_recording = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                vc_backend=None,
                vc_f0_condition=False,
                vc_pitch_shift=0,
            ),
        )

        ui.refresh_vc_controls()

        assert not ui.vc_mode_button.display
        assert not ui.vc_pitch_button.display
        assert ui.vc_mode_button.disabled
        assert ui.vc_pitch_button.disabled
        ui._cancel_vc_recording.assert_called_once_with(announce=False)

    def test_refresh_vc_controls_shows_buttons_in_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC-only buttons return when a VC backend is active."""
        ui = CeluneUI()
        ui._input_locked = False
        ui.vc_mode_button = Button(string("ui.vc_mode_talk"))
        ui.vc_pitch_button = Button(string("ui.vc_pitch_button", value="+0"))
        ui._cancel_vc_recording = mock.Mock()
        ui.celune = cast(
            Celune,
            SimpleNamespace(
                vc_backend=SimpleNamespace(),
                vc_f0_condition=True,
                vc_pitch_shift=3,
            ),
        )

        ui.refresh_vc_controls()

        assert ui.vc_mode_button.display
        assert ui.vc_pitch_button.display
        assert ui.vc_mode_button.label == string("ui.vc_mode_sing")
        assert ui.vc_pitch_button.label == string("ui.vc_pitch_button", value="+3")
        assert not ui.vc_mode_button.disabled
        assert not ui.vc_pitch_button.disabled
        ui._cancel_vc_recording.assert_not_called()

    def test_runtime_logger_warning_is_routed_into_ui_logs(self) -> None:
        """Verify external Python logger warnings are routed into the UI logs."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info", **kwargs: captured.append(
            (msg, severity)
        )

        logger = logging.getLogger("torch.utils.flop_counter")

        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning("test warning")

        assert captured == [
            (
                "Internal runtime warning: test warning",
                "warning",
            )
        ]

    def test_runtime_huggingface_logger_error_is_routed_into_ui_logs(self) -> None:
        """Verify Hugging Face logger errors are routed into the UI log widget."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info", **kwargs: captured.append(
            (msg, severity)
        )

        logger = logging.getLogger("huggingface_hub")
        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.error("download failed because the connection dropped")

        assert captured == [
            (
                (
                    "Internal runtime error: download failed because the "
                    "connection dropped"
                ),
                "error",
            )
        ]

    def test_runtime_global_log_redirect_captures_unlisted_external_logger(
        self,
    ) -> None:
        """Verify arbitrary external loggers are captured without per-backend wiring."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info", **kwargs: captured.append(
            (msg, severity)
        )

        logger = logging.getLogger("some.third_party.backend")

        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning("backend emitted a warning")

        assert captured == [
            ("Internal runtime warning: backend emitted a warning", "warning")
        ]

    def test_runtime_logger_suppresses_filtered_messages(self) -> None:
        """Verify runtime logger redirection honors the shared suppression list."""
        ui = CeluneUI()
        captured: list[tuple[str, str]] = []
        ui.safe_log = lambda msg, severity="info", **kwargs: captured.append(
            (msg, severity)
        )

        logger = logging.getLogger("some.third_party.backend")

        ui.install_runtime_log_redirects()
        self.addCleanup(ui._remove_runtime_log_redirects)

        logger.warning("Loading weights from C:/models/checkpoint.safetensors")

        assert not captured

    def test_safe_status_marquees_long_text_for_narrow_status_label(self) -> None:
        """Verify long status text scrolls instead of clipping."""

        class FakeLabel:
            """Tiny fake status label with a constrained width."""

            def __init__(self, width: int) -> None:
                self.size = SimpleNamespace(width=width)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Update the marquee label text.

                Args:
                    value: New rendered text captured from the UI update call.
                """
                self.rendered = value

        ui = CeluneUI()
        fake_status = FakeLabel(width=14)
        ui.status = cast(Label, fake_status)
        ui.resources = cast(Label, None)

        ui.safe_status("Playing C:/Users/user/Music/really_long_filename_demo.wav")
        first = fake_status.rendered
        ui.advance_status_marquee()
        second = fake_status.rendered

        assert first != second
        assert first.startswith("  ")
        assert second.startswith("  ")

    def test_safe_status_keeps_short_text_static(self) -> None:
        """Verify short status text does not marquee."""

        class FakeLabel:
            """Tiny fake status label with a constrained width."""

            def __init__(self, width: int) -> None:
                self.size = SimpleNamespace(width=width)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Update the marquee label text.

                Args:
                    value: New rendered text captured from the UI update call.
                """
                self.rendered = value

        ui = CeluneUI()
        fake_status = FakeLabel(width=40)
        ui.status = cast(Label, fake_status)
        ui.resources = cast(Label, None)

        ui.safe_status("Playing")
        first = fake_status.rendered
        ui.advance_status_marquee()

        assert first == fake_status.rendered

    def test_speech_caption_reveals_words_with_audio_progress(self) -> None:
        """Verify speech captions reveal words and restore the progress bar."""

        class FakeCaption:
            """Small caption widget test double."""

            is_attached = False

            def __init__(self) -> None:
                self.display = False
                self.styles = SimpleNamespace(opacity=1.0)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Capture the visible caption words."""
                self.rendered = value

        class FakeProgressBar:
            """Small progress bar test double."""

            is_attached = False

            def __init__(self, parent: Optional[object] = None) -> None:
                self.display = True
                self.styles = SimpleNamespace(opacity=1.0)
                self.parent = parent

            def update(self, **_kwargs: object) -> None:
                """Accept progress updates without rendering a real Textual widget."""

        ui = CeluneUI()
        caption = FakeCaption()
        progress_row = SimpleNamespace(display=True)
        ui.caption = cast(Label, caption)
        ui.progress_label = ProgressLabel()
        ui.progress_bar = cast(ProgressBar, FakeProgressBar(progress_row))

        ui.tts_caption("One two three")
        assert not progress_row.display
        assert caption.display
        assert not ui.progress_label.display
        ui.safe_caption_progress(1, 3)
        assert caption.rendered == "One"
        ui.safe_caption_progress(2, 3)
        assert caption.rendered == "One two"
        ui.safe_caption_progress(1, 3)
        assert caption.rendered == "One two"
        assert ui._caption_progress == 2 / 3
        ui.safe_caption_progress(3, 3)
        assert caption.rendered == "One two three"
        assert not ui._caption_active

        ui.tts_idle()
        assert not ui._caption_active
        assert not caption.display
        assert progress_row.display
        assert not ui.progress_label.display
        assert cast(FakeProgressBar, ui.progress_bar).display

        with mock.patch.object(ui, "_animate_opacity") as animate_opacity:
            ui.tts_idle()
        animate_opacity.assert_not_called()
        assert cast(FakeProgressBar, ui.progress_bar).styles.opacity == 1.0

        ui.tts_caption("Overlap")
        caption.is_attached = True
        cast(FakeProgressBar, ui.progress_bar).is_attached = True
        transition_callbacks: list[Optional[Callable[[], None]]] = []

        def capture_transition(
            _widget: Widget,
            _opacity: float,
            on_complete: Optional[Callable[[], None]] = None,
            **_kwargs: object,
        ) -> None:
            transition_callbacks.append(on_complete)

        with mock.patch.object(ui, "_animate_opacity", side_effect=capture_transition):
            ui.safe_caption_progress(1, 1)
            ui.safe_progress(1, 1)

        assert ui._caption_transitioning
        assert caption.display
        assert not progress_row.display
        assert transition_callbacks[0] is not None
        with mock.patch.object(ui, "set_timer", return_value=None):
            transition_callbacks[0]()
        assert not ui._caption_transitioning
        assert not caption.display
        assert progress_row.display
        ui._reset_playback_widgets()

        caption.is_attached = False
        cast(FakeProgressBar, ui.progress_bar).is_attached = False
        ui.tts_caption("Fade")
        caption.is_attached = True
        cast(FakeProgressBar, ui.progress_bar).is_attached = True
        transition_callbacks = []
        with mock.patch.object(ui, "_animate_opacity", side_effect=capture_transition):
            ui._hide_caption_widgets()
            ui.tts_idle()

        assert caption.display
        assert not progress_row.display
        assert transition_callbacks[0] is not None
        with mock.patch.object(ui, "set_timer", return_value=None):
            transition_callbacks[0]()
        assert not caption.display
        assert progress_row.display
        ui._reset_playback_widgets()

        with mock.patch.object(ui, "set_timer", return_value=None):
            ui.tts_caption("Wake me")
        ui.safe_progress(1, 1)
        assert not progress_row.display
        ui._reset_playback_widgets()
        assert not ui._caption_active
        assert not caption.display
        assert progress_row.display
        assert cast(FakeProgressBar, ui.progress_bar).display
        assert cast(FakeProgressBar, ui.progress_bar).styles.opacity == 1.0

        attached_progress = FakeProgressBar()
        attached_progress.is_attached = True
        with mock.patch.object(
            ui, "set_timer", side_effect=lambda _delay, callback: callback()
        ):
            ui._animate_opacity(cast(ProgressBar, attached_progress), 0.0)
        assert attached_progress.styles.opacity == 0.0

        with mock.patch.object(ui, "call_from_thread") as call_from_thread:
            worker = threading.Thread(target=ui._hide_caption_widgets)
            worker.start()
            worker.join()
        call_from_thread.assert_called_once_with(ui._hide_caption_widgets)

    def test_speech_caption_uses_word_timestamps(self) -> None:
        """Verify captions use Whisper word boundaries instead of sentence interpolation."""
        segments = (
            WhisperSegment(
                text="One two three",
                start=0.0,
                end=3.0,
                words=(
                    WhisperWord("One", 0.0, 0.4),
                    WhisperWord("two", 0.8, 1.4),
                    WhisperWord("three", 1.9, 2.8),
                ),
            ),
        )

        timings = CeluneUI._caption_word_timing_ranges(
            ("One", "two", "three"),
            segments,
            3.0,
        )

        assert timings == ((0.0, 0.4), (0.8, 1.4), (1.9, 2.8))

        normalized_timings = CeluneUI._caption_word_timing_ranges(
            (r"C:\Users\user",),
            (
                WhisperSegment(
                    text="C drive Users user",
                    start=0.0,
                    end=3.0,
                    words=(
                        WhisperWord("C", 0.0, 0.4),
                        WhisperWord("drive", 0.5, 1.0),
                        WhisperWord("Users", 1.1, 1.8),
                        WhisperWord("user", 2.0, 2.8),
                    ),
                ),
            ),
            3.0,
            ("C", "drive", "Users", "user"),
        )

        self.assertEqual(normalized_timings, ((0.0, 2.8),))

        mismatched_timing_words = CeluneUI._caption_word_timing_ranges(
            ("One", "two"),
            (
                WhisperSegment(
                    text="alpha beta gamma delta epsilon",
                    start=0.0,
                    end=5.0,
                    words=tuple(
                        WhisperWord(word, index, index + 0.5)
                        for index, word in enumerate(
                            ("alpha", "beta", "gamma", "delta", "epsilon")
                        )
                    ),
                ),
            ),
            5.0,
            ("first", "second", "third", "fourth", "fifth"),
        )

        assert len(mismatched_timing_words) == 2
        assert mismatched_timing_words[0][0] == 0.0
        assert mismatched_timing_words[-1][1] == 4.5

        fewer_timing_words = CeluneUI._caption_word_timing_ranges(
            ("One", "two", "three"),
            (
                WhisperSegment(
                    text="alpha",
                    start=0.0,
                    end=1.0,
                    words=(WhisperWord("alpha", 0.0, 1.0),),
                ),
            ),
            1.0,
            ("alpha",),
        )

        assert len(fewer_timing_words) == 3

    def test_caption_transcriber_does_not_publish_progress_to_playback_bar(
        self,
    ) -> None:
        """Verify caption model loading does not alter the foreground bar."""
        ui = CeluneUI()
        ui.celune = cast(
            Celune,
            SimpleNamespace(config={"persona": {"enabled": True}}),
        )
        ui._persona_speech_model_id = mock.Mock(return_value="test/whisper")
        ui._persona_speech_language = mock.Mock(return_value=None)
        ui._run_on_ui_thread = lambda callback: callback()
        transcriber = mock.Mock()
        transcriber.transcribe_segments.return_value = ()

        class ImmediateThread:
            """Run one background analysis target synchronously."""

            def __init__(self, target: Callable[[], None], **_kwargs: object) -> None:
                self._target = target

            def start(self) -> None:
                """Run the captured target."""
                self._target()

        with (
            mock.patch.object(ui_app, "np", np, create=True),
            mock.patch.object(
                ui_app,
                "persona_enabled",
                return_value=True,
                create=True,
            ),
            mock.patch.object(
                ui_app,
                "WhisperTranscriber",
                return_value=transcriber,
                create=True,
            ) as transcriber_type,
            mock.patch.object(ui_app.threading, "Thread", ImmediateThread),
        ):
            ui.tts_caption_timing(
                "One two",
                np.ones(8, dtype=np.float32),
                48000,
            )

        transcriber_type.assert_called_once_with("test/whisper", language=None)

    def test_speech_caption_timing_refinement_does_not_hide_words(self) -> None:
        """Verify late word timings cannot regress an already rendered caption."""
        ui = CeluneUI()

        class FakeCaption:
            """Small caption widget test double."""

            def __init__(self) -> None:
                self.display = False
                self.styles = SimpleNamespace(opacity=1.0, height=0)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Capture the visible caption words."""
                self.rendered = value

        ui.caption = cast(Label, FakeCaption())
        ui.tts_caption("One two three")
        ui.safe_caption_progress(2, 3)
        assert cast(FakeCaption, ui.caption).rendered == "One two"

        ui._caption_word_timings = ((2.5, 2.8), (2.9, 3.0), (3.1, 3.2))
        ui._caption_audio_duration = 3.2
        visible_sentence, visible_words = ui._caption_words_for_progress(2 / 3)

        assert visible_words == 2
        assert visible_sentence == ("One", "two")

    def test_status_ticker_recovers_active_playback_status(self) -> None:
        """Verify the TUI ticker displays the active playback-source status."""

        class FakeLabel:
            """Tiny fake status label for the playback ticker."""

            def __init__(self) -> None:
                self.size = SimpleNamespace(width=40)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Capture the rendered status."""
                self.rendered = value

        ui = CeluneUI()
        ui.celune = cast(
            Celune,
            SimpleNamespace(_playback_source_statuses={1: "Playing fixture.wav"}),
        )
        ui.status = cast(Label, FakeLabel())

        ui.advance_status_marquee()

        assert ui._status_text == "Playing fixture.wav"
        assert "Playing fixture.wav" in cast(FakeLabel, ui.status).rendered

    def test_safe_status_repaints_terminal_accent_for_error(self) -> None:
        """Verify error status repaints the terminal shell accent to the error color."""

        class FakeLabel:
            """Simple label test double with mutable styles."""

            def __init__(self) -> None:
                self.size = SimpleNamespace(width=40)
                self.styles = SimpleNamespace(color=None, border=None, border_top=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Capture the rendered label value.

                Args:
                    value: The latest label content.
                """
                self.rendered = value

        class FakeWidget:
            """Simple widget test double with mutable styles."""

            def __init__(self) -> None:
                self.styles = SimpleNamespace(
                    color=None,
                    border=None,
                    border_top=None,
                    background=None,
                    scrollbar_color=None,
                    scrollbar_color_hover=None,
                    scrollbar_color_active=None,
                    scrollbar_background=None,
                    scrollbar_background_hover=None,
                    scrollbar_background_active=None,
                )
                self.rendered = ""

            def update(self, value: str = "") -> None:
                """Capture the rendered label value.

                Args:
                    value: The latest label content.
                """
                self.rendered = value

        ui = CeluneUI()
        ui.celune = cast(
            Celune,
            SimpleNamespace(config={}, backend=FakeBackend, is_in_tutorial=False),
        )
        ui.logs = cast(RichLog, FakeWidget())
        ui.input_box = cast(TextArea, FakeWidget())
        ui.style_button = cast(Button, FakeWidget())
        ui.resources = cast(Label, FakeWidget())
        ui.header = cast(Label, FakeWidget())
        ui.progress_bar = cast(Button, FakeWidget())
        ui.header_lines = (cast(Label, FakeWidget()), cast(Label, FakeWidget()))
        ui.status = cast(Label, FakeLabel())

        ui._fatal_error_active = True
        ui.safe_status("Could not start", "error")

        expected = severity_color(ui.active_theme_name, "error")
        assert ui.theme == "celune_error"
        assert ui.logs.styles.color is None
        assert ui.logs.styles.border is None
        assert ui.logs.styles.background is None
        assert ui.logs.styles.scrollbar_color is None
        assert ui.input_box.styles.border is None
        assert ui.input_box.styles.background is None
        assert ui.input_box.styles.scrollbar_color is None
        assert ui.style_button.styles.border is None
        assert ui.style_button.styles.background is None
        assert ui.resources.styles.color is None
        assert ui.header.styles.color is None
        assert ui.header_lines[0].styles.border_top is None
        assert ui.progress_bar.styles.color is None
        assert ui.progress_bar.styles.background is None
        assert ui.status.styles.color == expected
        error_theme = ui.get_theme("celune_error")
        assert error_theme is not None
        assert error_theme.primary == colors.ERROR_DARK_ACCENT
        assert error_theme.accent == expected
        assert error_theme.foreground == colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_BACKGROUND,
            7.0,
        )
        assert error_theme.background == colors.ERROR_BACKGROUND

    def test_nonfatal_error_status_keeps_normal_theme(self) -> None:
        """Verify ordinary error statuses do not switch into the fatal error theme."""
        ui = CeluneUI()
        ui.status = Label()

        ui.safe_status("Minor issue", "error")

        assert ui.theme == "celune"
        assert not ui._fatal_error_active

    def test_wrapped_fatal_glow_activates_error_theme(self) -> None:
        """Verify the fatal theme only activates through wrapped ``glow.fatal()``."""
        ui = CeluneUI()
        ui.status = Label()
        ui.celune = cast(
            Celune,
            SimpleNamespace(glow=SimpleNamespace(fatal=mock.Mock())),
        )

        ui.wrap_runtime_fatal_glow()
        ui.celune.glow.fatal()

        assert ui._fatal_error_active
        assert ui.theme == "celune_error"

    def test_fatal_theme_stays_pinned_after_later_nonfatal_status_updates(self) -> None:
        """Verify later routine events cannot clear the fatal UI theme once activated."""
        ui = CeluneUI()
        ui.status = Label()
        ui.celune = cast(
            Celune,
            SimpleNamespace(glow=SimpleNamespace(fatal=mock.Mock())),
        )

        ui.wrap_runtime_fatal_glow()
        ui.celune.glow.fatal()
        ui.safe_status("Idle")
        ui.safe_status("Speaking")

        assert ui._fatal_error_active
        assert ui.theme == "celune_error"

    def test_fatal_status_text_ignores_later_idle_updates(self) -> None:
        """Verify fatal UI status text is not overwritten by later normal lifecycle events."""
        ui = CeluneUI()
        ui.status = Label()
        ui.safe_status("Celune could not warm up", "error")
        ui._fatal_error_active = True

        ui.safe_status("Idle")
        ui.safe_status("Speaking")

        assert ui._status_text == "Celune could not warm up"
        assert ui.status_severity == "error"

    def test_runtime_error_themes_cover_dark_and_light_modes(self) -> None:
        """Verify both dedicated runtime error themes are registered correctly."""
        ui = CeluneUI()

        ui.register_runtime_error_themes()

        dark_error = ui.get_theme("celune_error")
        light_error = ui.get_theme("celune_light_error")
        assert dark_error is not None
        assert light_error is not None
        assert dark_error.background == colors.ERROR_BACKGROUND
        assert light_error.background == colors.ERROR_LIGHT_BACKGROUND
        assert dark_error.accent == colors.THEME.error
        assert light_error.accent == colors.THEME_LIGHT.error
        assert dark_error.primary == colors.ERROR_DARK_ACCENT
        assert light_error.primary == colors.ERROR_DARK_ACCENT
        assert dark_error.foreground == colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_BACKGROUND,
            7.0,
        )
        assert light_error.foreground == colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_LIGHT_BACKGROUND,
            7.0,
        )

    def test_resize_repaints_status_after_width_change(self) -> None:
        """Verify widening the status label re-renders the current text immediately."""

        class FakeLabel:
            """Tiny fake status label with a mutable width."""

            def __init__(self, width: int) -> None:
                self.size = SimpleNamespace(width=width)
                self.styles = SimpleNamespace(color=None)
                self.rendered = ""

            def update(self, value: str) -> None:
                """Update the marquee label text.

                Args:
                    value: New rendered text captured from the UI update call.
                """
                self.rendered = value

        ui = CeluneUI()
        fake_status = FakeLabel(width=14)
        ui.status = cast(Label, fake_status)
        ui.resources = cast(Label, None)

        message = "Playing C:/Users/user/Music/really_long_filename_demo.wav"
        ui.safe_status(message)
        narrow = fake_status.rendered

        fake_status.size = SimpleNamespace(width=96)
        ui.on_resize(cast(events.Resize, SimpleNamespace()))

        self.assertNotEqual(narrow, fake_status.rendered)
        self.assertEqual(fake_status.rendered, f"  {message}")

    def test_tutorial_typing_runs_as_a_native_async_worker(self) -> None:
        """Verify tutorial typing updates the UI loop without a worker thread."""

        async def run_typing() -> None:
            ui = CeluneUI()
            with mock.patch.object(CeluneUI, "on_text_area_changed"):
                async with ui.run_test(size=(80, 24)) as pilot:
                    ui.celune = cast(
                        Celune,
                        SimpleNamespace(sleeping=False),
                    )
                    ui.cur_state = "ready"
                    submitted = mock.Mock()
                    ui._submit_text = submitted
                    with (
                        mock.patch("celune.utils.typing_delay", return_value=0.0),
                        mock.patch.object(
                            ui,
                            "call_from_thread",
                            side_effect=AssertionError(
                                "native async typing must stay on the UI loop"
                            ),
                        ),
                    ):
                        ui.type_and_send("Hi", process_commands=True)
                        await pilot.pause()

                    assert ui.input_box.text == "Hi"
                    submitted.assert_called_once_with("Hi", True)

        asyncio.run(run_typing())
