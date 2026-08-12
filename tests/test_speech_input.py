# SPDX-License-Identifier: Apache-2.0
"""Tests for Persona microphone speech input."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Optional, cast
from unittest import TestCase, mock

import numpy as np
from textual import events

from celune.persona.asr import WhisperTranscriber
from celune.ui.app import CeluneUI


class SpeechInputTests(TestCase):
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

            self.assertEqual(
                transcriber.transcribe(
                    np.ones((4800, 2), dtype=np.float32),
                    48000,
                ),
                "hello",
            )

        model_factory.assert_called_once_with(
            "openai/whisper-small",
            dtype=mock.ANY,
            quantization_config=mock.ANY,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        processor_factory.assert_called_once_with("openai/whisper-small")
        self.assertEqual(fake_processor.call_args.args[0].ndim, 1)
        self.assertEqual(fake_processor.call_args.args[0].shape, (1600,))
        self.assertEqual(fake_processor.call_args.kwargs["sampling_rate"], 16000)
        self.assertEqual(fake_model.generate.call_args.kwargs["task"], "transcribe")

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
            mock.patch("celune.ui.app.sd.InputStream", FakeInputStream),
            mock.patch(
                "celune.ui.app.create_live_voice_activity_detector", return_value=vad
            ),
            mock.patch("celune.ui.app.WhisperTranscriber", return_value=transcriber),
        ):
            self.assertEqual(ui._start_persona_recording(), True)
            worker = ui._persona_recording_worker

            if captured_callback is None:
                self.fail("Persona recording callback was not registered")
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
