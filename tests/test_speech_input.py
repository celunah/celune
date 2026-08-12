# SPDX-License-Identifier: MIT
"""Tests for Persona microphone speech input."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Optional, cast
from unittest import mock

import numpy as np
import pytest
from textual import events

from celune.persona.asr import (
    PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
    WhisperTranscriber,
)
from celune.typing.persona import _WhisperProcessor
from celune.ui.app import CeluneUI

from .support import CeluneTestCase


class TestSpeechInput(CeluneTestCase):
    """Verify Persona speech-input routing and transcription helpers."""

    def tearDown(self) -> None:
        """Reset the singleton UI instance after each test."""
        CeluneUI._instance = None

    def test_whisper_transcriber_uses_configured_model_lazily(self) -> None:
        """Verify transcription loads the selected Hugging Face model on demand."""
        fake_processor = mock.Mock()
        fake_features = mock.Mock()
        fake_features.to.return_value = fake_features
        fake_processor.return_value = SimpleNamespace(
            input_features=fake_features,
            attention_mask=None,
        )
        fake_processor.batch_decode.return_value = [" hello "]
        fake_model = mock.Mock()
        fake_parameter = mock.Mock()
        fake_parameter.device = "cuda:0"
        fake_model.parameters.side_effect = lambda: iter((fake_parameter,))
        fake_model.generate.return_value = [[1]]
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "transformers.AutoModelForSpeechSeq2Seq.from_pretrained",
                return_value=fake_model,
            ) as model_factory,
            mock.patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ) as processor_factory,
        ):
            transcriber = WhisperTranscriber("openai/whisper-small")

            assert (
                transcriber.transcribe(
                    np.ones((4800, 2), dtype=np.float32),
                    48000,
                )
                == "hello"
            )

        model_factory.assert_called_once_with(
            "openai/whisper-small",
            dtype=mock.ANY,
            quantization_config=mock.ANY,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        processor_factory.assert_called_once_with("openai/whisper-small")
        assert fake_processor.call_args.args[0].ndim == 1
        assert fake_processor.call_args.args[0].shape == (1600,)
        assert fake_processor.call_args.kwargs["sampling_rate"] == 16000
        assert fake_model.generate.call_args.kwargs["task"] == "transcribe"

    def test_whisper_word_timestamps_are_grouped_from_token_timestamps(self) -> None:
        """Verify token timestamps are grouped into the words Whisper decoded."""
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                decode=lambda tokens, **_kwargs: {
                    1: "Hello",
                    2: ",",
                    3: " world",
                    4: " again",
                }[tokens[0]]
            )
        )
        raw_segment = {
            "tokens": np.asarray([1, 2, 3, 4]),
            "idxs": [0, 4],
            "result": {"token_timestamps": np.asarray([0.1, 0.3, 0.5, 0.8])},
        }

        words = WhisperTranscriber._decode_words(
            cast(_WhisperProcessor, processor),
            raw_segment,
            0.0,
            1.0,
        )

        assert [(word.text, word.start, word.end) for word in words] == [
            ("Hello,", 0.1, 0.5),
            ("world", 0.5, 0.8),
            ("again", 0.8, 1.0),
        ]

    def test_whisper_segments_decode_one_dimensional_tokens_as_one_sequence(
        self,
    ) -> None:
        """Verify one segment's token tensor is decoded as a complete sequence."""
        observed_shapes: list[tuple[int, ...]] = []

        def batch_decode(tokens, **_kwargs) -> list[str]:
            observed_shapes.append(tuple(tokens.shape))
            return ["hello world"]

        processor = SimpleNamespace(batch_decode=batch_decode)
        generated = {
            "segments": [[{"start": 0.0, "end": 1.0, "tokens": np.asarray([1, 2])}]]
        }

        segments = WhisperTranscriber._decode_segments(
            cast(_WhisperProcessor, processor),
            generated,
        )

        assert observed_shapes == [(1, 2)]
        assert [(segment.text, segment.start, segment.end) for segment in segments] == [
            ("hello world", 0.0, 1.0)
        ]

    @staticmethod
    def test_ctrl_r_routes_persona_and_vc_to_separate_recorders() -> None:
        """Verify Persona CTRL+R does not alter the existing VC shortcut path."""
        ui = CeluneUI()
        ui.cur_state = "idle"
        ui.celune = SimpleNamespace(
            sleeping=False,
            cur_state="idle",
            vc_backend=None,
        )
        ui.toggle_persona_recording = mock.Mock(return_value=True)
        ui.toggle_vc_recording = mock.Mock(return_value=True)
        event = SimpleNamespace(
            key="ctrl+r",
            prevent_default=mock.Mock(),
            stop=mock.Mock(),
        )

        ui.on_key(cast(events.Key, event))

        ui.toggle_persona_recording.assert_called_once_with()
        ui.toggle_vc_recording.assert_not_called()
        event.prevent_default.assert_called_once_with()
        event.stop.assert_called_once_with()

        ui.celune.vc_backend = SimpleNamespace()
        ui.on_key(cast(events.Key, event))

        ui.toggle_vc_recording.assert_called_once_with()

    def test_persona_recording_submits_after_vad_silence(self) -> None:
        """Verify VAD silence queues final audio and submits its transcript."""
        ui = CeluneUI()
        ui.cur_state = "idle"
        ui.input_box = SimpleNamespace(text="", placeholder="", load_text=mock.Mock())
        ui.style_button = SimpleNamespace(disabled=False)
        ui.safe_log = mock.Mock()
        ui.safe_status = mock.Mock()
        ui.update_resources = mock.Mock()
        ui._cancel_sleep_timer = mock.Mock()
        ui.celune = SimpleNamespace(
            config={
                "vram": "high",
                "persona": {"talkback": True, "speech_end_delay_seconds": 0},
            },
            dev=False,
            sleeping=False,
            cur_state="idle",
            vc_backend=None,
            vision=SimpleNamespace(),
            think=mock.Mock(return_value=True),
        )

        captured_callback: Optional[Callable[..., None]] = None

        class FakeInputStream:
            """Small sounddevice input stream test double."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = cast(Callable[..., None], kwargs["callback"])

            def start(self) -> None:
                """Start the fake stream."""

            def stop(self) -> None:
                """Stop the fake stream."""

            def close(self) -> None:
                """Close the fake stream."""

        vad = SimpleNamespace(has_voice=mock.Mock(side_effect=[True, False]))
        transcriber = mock.Mock()
        transcriber.transcribe.return_value = "hello there"

        with (
            mock.patch("celune.ui.app._load_ui_runtime_dependencies"),
            mock.patch(
                "celune.ui.app.sd",
                SimpleNamespace(InputStream=FakeInputStream),
                create=True,
            ),
            mock.patch("celune.ui.app.np", np, create=True),
            mock.patch(
                "celune.ui.app.persona_talkback_enabled",
                return_value=True,
                create=True,
            ),
            mock.patch(
                "celune.ui.app.persona_config",
                return_value={
                    "speech_model_id": "fixture/whisper",
                    "speech_end_delay_seconds": 0,
                },
                create=True,
            ),
            mock.patch(
                "celune.ui.app.vc_vad_hangover_frames", return_value=0, create=True
            ),
            mock.patch(
                "celune.ui.app.PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS",
                PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
                create=True,
            ),
            mock.patch.object(
                ui,
                "call_from_thread",
                side_effect=lambda callback, *args: callback(*args),
            ),
            mock.patch(
                "celune.ui.app.resolve_audio_device_with_info",
                return_value=(
                    "microphone",
                    {
                        "max_input_channels": 1,
                        "default_samplerate": 16000,
                        "name": "Microphone",
                    },
                ),
            ),
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector",
                return_value=vad,
                create=True,
            ),
            mock.patch(
                "celune.ui.app.WhisperTranscriber",
                return_value=transcriber,
                create=True,
            ),
        ):
            assert ui._start_persona_recording()
            worker = ui._persona_recording_worker

            if captured_callback is None:
                pytest.fail("Persona recording callback was not registered")
            else:
                captured_callback(
                    np.ones((1600, 1), dtype=np.float32), 1600, None, None
                )
                captured_callback(
                    np.zeros((5000, 1), dtype=np.float32), 5000, None, None
                )

            if worker is not None:
                worker.join(timeout=2.0)

        ui.celune.think.assert_called_once_with("hello there")
        transcriber.transcribe.assert_called_once()

    def test_persona_recording_times_out_without_speech(self) -> None:
        """Verify silent Persona recording stops after the no-input timeout."""
        ui = CeluneUI()
        ui.cur_state = "idle"
        ui.input_box = SimpleNamespace(text="", placeholder="", load_text=mock.Mock())
        ui.style_button = SimpleNamespace(disabled=False)
        ui.safe_log = mock.Mock()
        ui.safe_status = mock.Mock()
        ui.update_resources = mock.Mock()
        ui._cancel_sleep_timer = mock.Mock()
        ui.celune = SimpleNamespace(
            config={
                "vram": "high",
                "persona": {"talkback": True, "speech_end_delay_seconds": 0},
            },
            dev=False,
            sleeping=False,
            cur_state="idle",
            vc_backend=None,
            vision=SimpleNamespace(),
            think=mock.Mock(return_value=True),
        )

        captured_callback: Optional[Callable[..., None]] = None

        class FakeInputStream:
            """Small sounddevice input stream test double."""

            def __init__(self, **kwargs) -> None:
                nonlocal captured_callback
                captured_callback = cast(Callable[..., None], kwargs["callback"])

            def start(self) -> None:
                """Start the fake stream."""

            def stop(self) -> None:
                """Stop the fake stream."""

            def close(self) -> None:
                """Close the fake stream."""

        vad = SimpleNamespace(has_voice=mock.Mock(return_value=False))
        transcriber = mock.Mock()
        monotonic_values = iter((0.0, 0.0, PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS))

        with (
            mock.patch("celune.ui.app._load_ui_runtime_dependencies"),
            mock.patch(
                "celune.ui.app.sd",
                SimpleNamespace(InputStream=FakeInputStream),
                create=True,
            ),
            mock.patch("celune.ui.app.np", np, create=True),
            mock.patch(
                "celune.ui.app.persona_talkback_enabled",
                return_value=True,
                create=True,
            ),
            mock.patch(
                "celune.ui.app.persona_config",
                return_value={
                    "speech_model_id": "fixture/whisper",
                    "speech_end_delay_seconds": 0,
                },
                create=True,
            ),
            mock.patch(
                "celune.ui.app.vc_vad_hangover_frames", return_value=0, create=True
            ),
            mock.patch(
                "celune.ui.app.PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS",
                PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
                create=True,
            ),
            mock.patch.object(
                ui,
                "call_from_thread",
                side_effect=lambda callback, *args: callback(*args),
            ),
            mock.patch(
                "celune.ui.app.resolve_audio_device_with_info",
                return_value=(
                    "microphone",
                    {
                        "max_input_channels": 1,
                        "default_samplerate": 16000,
                        "name": "Microphone",
                    },
                ),
            ),
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector",
                return_value=vad,
                create=True,
            ),
            mock.patch(
                "celune.ui.app.WhisperTranscriber",
                return_value=transcriber,
                create=True,
            ),
            mock.patch(
                "celune.ui.app.time.monotonic",
                side_effect=lambda: next(
                    monotonic_values, PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS
                ),
            ),
        ):
            assert ui._start_persona_recording()

            if captured_callback is None:
                pytest.fail("Persona recording callback was not registered")
            else:
                captured_callback(
                    np.zeros((1600, 1), dtype=np.float32), 1600, None, None
                )

            worker = ui._persona_recording_worker
            if worker is not None:
                worker.join(timeout=2.0)

        ui.celune.think.assert_not_called()
        transcriber.transcribe.assert_not_called()
        assert any(call.args[1] == "warning" for call in ui.safe_log.call_args_list)

    @staticmethod
    def test_persona_transcription_does_not_repeat_reported_error() -> None:
        """Verify a final transcription failure does not repeat its warning."""
        ui = CeluneUI()
        ui.safe_log = mock.Mock()
        ui.safe_status = mock.Mock()
        ui.update_resources = mock.Mock()
        ui.style_button = SimpleNamespace(disabled=False)

        ui._complete_persona_transcription(
            "",
            "",
            RuntimeError("already reported"),
            error_already_reported=True,
        )

        ui.safe_log.assert_not_called()
        ui.safe_status.assert_called_once()
