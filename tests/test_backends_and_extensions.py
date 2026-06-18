# SPDX-License-Identifier: MIT
"""Tests for backend resolution and extension infrastructure."""

import sys
import tempfile
import textwrap
import importlib
import threading
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
from unittest import mock, TestCase
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch

from celune.utils import discard
from celune.backends import resolve_backend
from celune.extensions.manager import CeluneExtensionManager
from celune.extensions.base import CeluneContext, CeluneExtension
from celune.exceptions import (
    BackendError,
    ExtensionAlreadyRegisteredError,
    InvalidExtensionError,
)
from .support import (
    FakeBackend,
    make_voice_loader,
    mock_dotstts_backend,
    mock_mini_backend,
    mock_qwen3_backend,
    mock_voxcpm_backend,
)


class BackendTests(TestCase):
    """Tests for backend base behavior and backend resolution."""

    def test_base_backend_reports_models(self) -> None:
        """Verify model metadata helpers on a fake backend.

        Raises:
            AssertionError: A backend helper returns an unexpected value.
        """
        backend = FakeBackend(log=lambda _msg, _severity="info": None)
        self.assertEqual(backend.default_model_id, "fake/balanced")
        self.assertEqual(backend.all_model_ids, ["fake/balanced", "fake/bold"])
        self.assertEqual(backend.voices, ["balanced", "bold"])
        self.assertEqual(backend.model_id_for_voice("bold"), "fake/bold")

    def test_base_backend_materializes_bundle_pt_refs_when_available(self) -> None:
        """Verify CEVOICE bundles eagerly extract .pt refs alongside .wav files."""
        materialize = mock.Mock(side_effect=lambda voice, kind: Path(f"{voice}.{kind}"))
        loader = SimpleNamespace(
            bundle=SimpleNamespace(
                voice_order=("balanced", "bold"),
                voices={
                    "balanced": {"assets": {"wav": {}, "pt": {}}},
                    "bold": {"assets": {"wav": {}}},
                },
            ),
            materialize=materialize,
        )

        with mock.patch("celune.backends.base.default_loader", return_value=loader):
            backend = FakeBackend(log=lambda _msg, _severity="info": None)
            backend.validate_refs()

        self.assertEqual(
            materialize.call_args_list,
            [
                mock.call("balanced", "wav"),
                mock.call("balanced", "pt"),
                mock.call("bold", "wav"),
            ],
        )

    def test_base_backend_truncates_long_reference_wav_to_ten_seconds(self) -> None:
        """Verify the shared reference helper clips long WAV prompts to ten seconds."""
        backend = FakeBackend(log=lambda _msg, _severity="info": None)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "reference.wav"
            canonical_temp = Path(temp_dir) / "celune-temp"
            canonical_temp.mkdir(parents=True, exist_ok=True)
            sf.write(source, np.zeros(12 * 24000, dtype=np.float32), 24000)

            with mock.patch(
                "celune.backends.base.temp_data_dir", return_value=canonical_temp
            ):
                truncated = backend.truncate_reference(source)

            self.assertNotEqual(truncated, source)
            self.assertLessEqual(sf.info(truncated).duration, 10.0)
            self.assertEqual(truncated.parent, canonical_temp)
            self.assertEqual(source.exists(), True)

            backend.unload_model()
            self.assertEqual(truncated.exists(), False)

    def test_resolve_backend_accepts_instance_type_and_rejects_unknown(self) -> None:
        """Verify supported backend specifications and invalid input failures.

        Raises:
            AssertionError: Backend resolution behavior changes unexpectedly.
        """
        instance = FakeBackend(log=lambda _msg, _severity="info": None)
        self.assertIs(resolve_backend(instance), instance)
        self.assertIsInstance(resolve_backend(FakeBackend), FakeBackend)
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            resolve_backend("missing")
        with self.assertRaisesRegex(TypeError, "backend_name"):
            resolve_backend(123)  # type: ignore[arg-type]

    def test_resolve_backend_accepts_mini_backend_name(self) -> None:
        """Verify the Pocket TTS backend resolves through the backend registry."""

        class StubTTSModel:
            """Import-time stand-in for the Pocket TTS package class."""

        with mock.patch.dict(
            sys.modules,
            {"pocket_tts": SimpleNamespace(TTSModel=StubTTSModel)},
        ):
            mini = importlib.import_module("celune.backends.mini")
            mini_cls = mini.Mini

            with mock.patch.object(mini_cls, "_validate_refs"):
                backend = resolve_backend("mini")

        self.assertIsInstance(backend, mini_cls)
        self.assertEqual(backend.name, "mini")

    def test_resolve_backend_accepts_dotstts_backend_name(self) -> None:
        """Verify the dots.tts backend resolves through the backend registry."""

        with mock_dotstts_backend() as dotstts_cls:
            with mock.patch.object(dotstts_cls, "_validate_refs"):
                backend = resolve_backend("dotstts")

        self.assertIsInstance(backend, dotstts_cls)
        self.assertEqual(backend.name, "dotstts")

    def test_voxcpm2_uses_pack_cfg_scale_when_present(self) -> None:
        """Verify CEVOICE can override VoxCPM2's per-voice CFG scale."""

        with mock_voxcpm_backend() as voxcpm2_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.cfg_value = None

                def generate_streaming(
                    self, *args, **kwargs
                ) -> Iterator[npt.NDArray[np.float32]]:
                    """Generate fake VoxCPM2 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    self.cfg_value = kwargs["cfg_value"]
                    yield np.zeros((1,), dtype=np.float32)

            loader = make_voice_loader(
                "calm", {"cfg_scale": 4.2, "reference_text": "Pack reference."}
            )
            with (
                mock.patch.object(voxcpm2_cls, "_validate_refs"),
                mock.patch.object(
                    voxcpm2_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch(
                    "celune.backends.voxcpm2.default_loader", return_value=loader
                ),
            ):
                backend = voxcpm2_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(
                    backend.generate_stream(
                        model,
                        text="hello",
                        voice="calm",
                        chunk_size=1,
                    )
                )

            self.assertEqual(model.cfg_value, 4.2)

    def test_voxcpm2_requires_reference_text_for_valid_voice_identifiers(self) -> None:
        """Verify VoxCPM2 rejects packs whose voices omit the required reference text."""

        with mock_voxcpm_backend() as voxcpm2_cls:
            loader = make_voice_loader("calm", {})
            with mock.patch(
                "celune.backends.voxcpm2.default_loader", return_value=loader
            ):
                with self.assertRaisesRegex(
                    BackendError, "requires a compatible CEVOICE/CECHAR package"
                ):
                    voxcpm2_cls(log=lambda _msg, _severity="info": None)

    def test_voxcpm2_uses_truncated_reference_wav_when_present(self) -> None:
        """Verify VoxCPM2 passes reference audio through the shared truncation hook."""

        with mock_voxcpm_backend() as voxcpm2_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.reference_wav_path = None

                def generate_streaming(
                    self, *args, **kwargs
                ) -> Iterator[npt.NDArray[np.float32]]:
                    """Generate fake VoxCPM2 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    self.reference_wav_path = kwargs["reference_wav_path"]
                    yield np.zeros((1,), dtype=np.float32)

            loader = make_voice_loader(
                "calm", {"cfg_scale": 4.2, "reference_text": "Pack reference."}
            )
            with (
                mock.patch(
                    "celune.backends.voxcpm2.default_loader", return_value=loader
                ),
                mock.patch.object(
                    voxcpm2_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = voxcpm2_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(
                    backend.generate_stream(
                        model,
                        text="hello",
                        voice="calm",
                        chunk_size=1,
                    )
                )

            self.assertEqual(model.reference_wav_path, Path("trimmed.wav"))

    def test_voxcpm2_requires_a_compatible_voice_pack(self) -> None:
        """Verify VoxCPM2 refuses to initialize without a usable CEVOICE/CECHAR pack."""

        with (
            mock_voxcpm_backend() as voxcpm2_cls,
            mock.patch("celune.backends.voxcpm2.default_loader", return_value=None),
        ):
            with self.assertRaisesRegex(
                BackendError, "requires a compatible CEVOICE/CECHAR package"
            ):
                voxcpm2_cls(log=lambda _msg, _severity="info": None)

    def test_mini_requires_reference_text_for_valid_voice_identifiers(self) -> None:
        """Verify Mini rejects packs whose voices omit the required reference text."""

        with mock_mini_backend() as mini_cls:
            loader = make_voice_loader("calm", {})
            with mock.patch("celune.backends.mini.default_loader", return_value=loader):
                with self.assertRaisesRegex(
                    BackendError, "requires a compatible CEVOICE/CECHAR package"
                ):
                    mini_cls(log=lambda _msg, _severity="info": None)

    def test_mini_uses_truncated_reference_wav_when_building_prompt_state(self) -> None:
        """Verify Mini builds prompt state from the shared truncated WAV path."""

        with mock_mini_backend() as mini_cls:
            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})

            class FakeModel:
                """Fake model class for use in this test suite."""

                sample_rate = 24000

                def __init__(self) -> None:
                    self.audio_conditioning = None

                def get_state_for_audio_prompt(self, audio_conditioning: str) -> dict:
                    """Return a fake prompt state.

                    Args:
                        audio_conditioning: Prompt WAV path passed by the backend.

                    Returns:
                        dict: A fake prompt state.
                    """
                    self.audio_conditioning = audio_conditioning
                    return {}

                @staticmethod
                def generate_audio_stream(
                    model_state: dict, text_to_generate: str
                ) -> Iterator[torch.Tensor]:
                    """Generate fake Pocket TTS chunks.

                    Args:
                        model_state: Prompt state used for generation.
                        text_to_generate: Text content to synthesize.
                    """
                    discard(model_state)
                    discard(text_to_generate)
                    yield torch.zeros((1,), dtype=torch.float32)

            with (
                mock.patch("celune.backends.mini.default_loader", return_value=loader),
                mock.patch.object(
                    mini_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = mini_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            self.assertEqual(model.audio_conditioning, str(Path("trimmed.wav")))

    def test_mini_does_not_apply_reference_wav_truncation_to_reference_text(
        self,
    ) -> None:
        """Verify Mini does not treat transcript text as a WAV path during validation."""

        with mock_mini_backend() as mini_cls:
            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch("celune.backends.mini.default_loader", return_value=loader),
                mock.patch.object(
                    mini_cls,
                    "_truncate_reference",
                    side_effect=AssertionError("should not see reference text"),
                ),
            ):
                backend = mini_cls(log=lambda _msg, _severity="info": None)
                self.assertEqual(backend.voices, ["calm"])

    def test_mini_requires_a_compatible_voice_pack(self) -> None:
        """Verify Mini refuses to initialize without a usable CEVOICE/CECHAR pack."""

        with (
            mock_mini_backend() as mini_cls,
            mock.patch("celune.backends.mini.default_loader", return_value=None),
        ):
            with self.assertRaisesRegex(
                BackendError, "requires a compatible CEVOICE/CECHAR package"
            ):
                mini_cls(log=lambda _msg, _severity="info": None)

    def test_qwen3_uses_pack_reference_text_when_present(self) -> None:
        """Verify CEVOICE can override Qwen3's per-voice reference text."""

        with mock_qwen3_backend() as qwen3_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.ref_audio = None
                    self.ref_text = None

                def generate_voice_clone_streaming(
                    self, *args, **kwargs
                ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
                    """Generate fake Qwen3 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    self.ref_audio = kwargs["ref_audio"]
                    self.ref_text = kwargs["ref_text"]
                    yield np.zeros((1,), dtype=np.float32), 24000, None

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(qwen3_cls, "_validate_refs"),
                mock.patch.object(
                    qwen3_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch("celune.backends.qwen3.default_loader", return_value=loader),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            self.assertEqual(model.ref_text, "Pack reference.")

    def test_qwen3_uses_truncated_reference_wav_when_present(self) -> None:
        """Verify Qwen3 passes reference audio through the shared truncation hook."""

        with mock_qwen3_backend() as qwen3_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.ref_audio = None
                    self.ref_text = None

                def generate_voice_clone_streaming(
                    self, *args, **kwargs
                ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
                    """Generate fake Qwen3 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    self.ref_audio = kwargs["ref_audio"]
                    self.ref_text = kwargs["ref_text"]
                    yield np.zeros((1,), dtype=np.float32), 24000, None

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(qwen3_cls, "_validate_refs"),
                mock.patch("celune.backends.qwen3.default_loader", return_value=loader),
                mock.patch.object(
                    qwen3_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            self.assertEqual(model.ref_text, "Pack reference.")
            self.assertEqual(model.ref_audio, Path("trimmed.wav"))

    def test_dotstts_uses_pack_reference_text_when_present(self) -> None:
        """Verify CEVOICE can override dots.tts reference text."""

        with mock_dotstts_backend() as dotstts_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                sample_rate = 48000

                def __init__(self) -> None:
                    self.prompt_audio_path = None
                    self.prompt_text = None

                def generate_stream(self, *args, **kwargs) -> Iterator[torch.Tensor]:
                    """Generate fake dots.tts chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    self.prompt_audio_path = kwargs["prompt_audio_path"]
                    self.prompt_text = kwargs["prompt_text"]
                    yield torch.zeros((1, 4), dtype=torch.float32)

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(dotstts_cls, "_validate_refs"),
                mock.patch.object(
                    dotstts_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch(
                    "celune.backends.dotstts.default_loader", return_value=loader
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            self.assertEqual(model.prompt_text, "Pack reference.")

    def test_dotstts_uses_truncated_reference_wav_when_present(self) -> None:
        """Verify dots.tts passes reference audio through the shared truncation hook."""

        with mock_dotstts_backend() as dotstts_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                sample_rate = 48000

                def __init__(self) -> None:
                    self.prompt_audio_path = None
                    self.prompt_text = None

                def generate_stream(self, *args, **kwargs) -> Iterator[torch.Tensor]:
                    """Generate fake dots.tts chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    self.prompt_audio_path = kwargs["prompt_audio_path"]
                    self.prompt_text = kwargs["prompt_text"]
                    yield torch.zeros((1, 4), dtype=torch.float32)

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(dotstts_cls, "_validate_refs"),
                mock.patch(
                    "celune.backends.dotstts.default_loader", return_value=loader
                ),
                mock.patch.object(
                    dotstts_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            self.assertEqual(model.prompt_text, "Pack reference.")
            self.assertEqual(model.prompt_audio_path, str(Path("trimmed.wav")))

    def test_dotstts_falls_back_to_the_active_pack_voice_ids(self) -> None:
        """Verify dots.tts uses the pack voice when the backend default is absent."""

        with mock_dotstts_backend() as dotstts_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                sample_rate = 48000

                def __init__(self) -> None:
                    self.prompt_audio_path = None
                    self.prompt_text = None

                def generate_stream(self, *args, **kwargs) -> Iterator[torch.Tensor]:
                    """Generate fake dots.tts chunks.

                    Args:
                        args: Positional arguments accepted for backend compatibility.
                        kwargs: Keyword arguments carrying prompt metadata under test.
                    """
                    discard(args)
                    self.prompt_audio_path = kwargs["prompt_audio_path"]
                    self.prompt_text = kwargs["prompt_text"]
                    yield torch.zeros((1, 4), dtype=torch.float32)

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(dotstts_cls, "_validate_refs"),
                mock.patch.object(
                    dotstts_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch(
                    "celune.backends.dotstts.default_loader", return_value=loader
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello"))

            self.assertEqual(model.prompt_audio_path, str(Path("calm.wav")))
            self.assertEqual(model.prompt_text, "Pack reference.")

    def test_dotstts_requires_reference_text_for_valid_voice_identifiers(self) -> None:
        """Verify dots.tts rejects packs whose voices omit the required reference text."""

        with mock_dotstts_backend() as dotstts_cls:
            loader = make_voice_loader("calm", {})
            with mock.patch(
                "celune.backends.dotstts.default_loader", return_value=loader
            ):
                with self.assertRaisesRegex(
                    BackendError, "requires a compatible CEVOICE/CECHAR package"
                ):
                    dotstts_cls(log=lambda _msg, _severity="info": None)

    def test_dotstts_requires_a_compatible_voice_pack(self) -> None:
        """Verify dots.tts refuses to initialize without a usable CEVOICE/CECHAR pack."""

        with (
            mock_dotstts_backend() as dotstts_cls,
            mock.patch("celune.backends.dotstts.default_loader", return_value=None),
        ):
            with self.assertRaisesRegex(
                BackendError, "requires a compatible CEVOICE/CECHAR package"
            ):
                dotstts_cls(log=lambda _msg, _severity="info": None)

    def test_dotstts_manually_pumps_and_closes_backend_stream(self) -> None:
        """Verify dots.tts iterates and closes its backend stream explicitly."""

        with mock_dotstts_backend() as dotstts_cls:

            class FakeStream:
                """Minimal iterator exposing a close hook for one backend test."""

                def __init__(self) -> None:
                    self._chunks = [
                        torch.zeros((1, 1), dtype=torch.float32),
                        torch.ones((1, 1), dtype=torch.float32),
                    ]
                    self.closed = False

                def __iter__(self) -> "FakeStream":
                    return self

                def __next__(self) -> torch.Tensor:
                    if not self._chunks:
                        raise StopIteration
                    return self._chunks.pop(0)

                def close(self) -> None:
                    """Close the stream."""
                    self.closed = True

            class FakeModel:
                """Fake model class for use in this test suite."""

                sample_rate = 48000

                def __init__(self) -> None:
                    self.stream = FakeStream()

                def generate_stream(self, *args, **kwargs) -> FakeStream:
                    """Generate fake dots.tts chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.

                    Returns:
                        FakeStream: A fake stream of dots.tts chunks.
                    """
                    discard(args)
                    discard(kwargs)
                    return self.stream

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(dotstts_cls, "_validate_refs"),
                mock.patch.object(
                    dotstts_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch(
                    "celune.backends.dotstts.default_loader", return_value=loader
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                chunks = list(
                    backend.generate_stream(model, text="hello", voice="calm")
                )

            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0][1], 48000)
            self.assertEqual(chunks[1][0].tolist(), [1.0])
            self.assertEqual(model.stream.closed, True)

    def test_dotstts_suppresses_loguru_runtime_noise(self) -> None:
        """Verify dots.tts suppression also disables its Loguru logger namespace."""

        with mock_dotstts_backend() as dotstts_cls:
            fake_loguru = mock.Mock()
            with mock.patch("celune.backends.dotstts.loguru.logger", fake_loguru):
                with dotstts_cls.suppress_backend_output():
                    pass

        fake_loguru.disable.assert_called_once_with("dots_tts")
        fake_loguru.enable.assert_called_once_with("dots_tts")

    def test_qwen3_requires_reference_text_for_valid_voice_identifiers(self) -> None:
        """Verify Qwen3 rejects packs whose voices omit the required reference text."""

        with mock_qwen3_backend() as qwen3_cls:
            loader = make_voice_loader("calm", {})
            with mock.patch(
                "celune.backends.qwen3.default_loader", return_value=loader
            ):
                with self.assertRaisesRegex(
                    BackendError, "requires a compatible CEVOICE/CECHAR package"
                ):
                    qwen3_cls(log=lambda _msg, _severity="info": None)

    def test_qwen3_requires_a_compatible_voice_pack(self) -> None:
        """Verify Qwen3 refuses to initialize without a usable CEVOICE/CECHAR pack."""

        with (
            mock_qwen3_backend() as qwen3_cls,
            mock.patch("celune.backends.qwen3.default_loader", return_value=None),
        ):
            with self.assertRaisesRegex(
                BackendError, "requires a compatible CEVOICE/CECHAR package"
            ):
                qwen3_cls(log=lambda _msg, _severity="info": None)

    def test_qwen3_requires_at_least_one_valid_voice_identifier(self) -> None:
        """Verify Qwen3 rejects packs that do not expose any usable voice names."""

        loader = SimpleNamespace(
            bundle=SimpleNamespace(
                voices={"": {"reference_text": "ignored"}},
                voice_order=("",),
            ),
            materialize=lambda ref_voice, kind: Path(f"{ref_voice}.{kind}"),
        )

        with (
            mock_qwen3_backend() as qwen3_cls,
            mock.patch("celune.backends.qwen3.default_loader", return_value=loader),
        ):
            with self.assertRaisesRegex(
                BackendError, "requires a compatible CEVOICE/CECHAR package"
            ):
                qwen3_cls(log=lambda _msg, _severity="info": None)

    def test_qwen3_manually_pumps_and_closes_backend_stream(self) -> None:
        """Verify Qwen3 iterates and closes its backend stream explicitly."""

        with mock_qwen3_backend() as qwen3_cls:

            class FakeStream:
                """Minimal iterator exposing a close hook for one backend test."""

                def __init__(self) -> None:
                    self._chunks = [
                        (np.zeros((1,), dtype=np.float32), 24000, {"chunk_steps": 1}),
                        (np.ones((1,), dtype=np.float32), 24000, {"chunk_steps": 1}),
                    ]
                    self.closed = False

                def __iter__(self) -> "FakeStream":
                    return self

                def __next__(
                    self,
                ) -> tuple[npt.NDArray[np.float32], int, Optional[dict]]:
                    if not self._chunks:
                        raise StopIteration
                    return self._chunks.pop(0)

                def close(self) -> None:
                    """Close the stream."""
                    self.closed = True

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.stream = FakeStream()

                def generate_voice_clone_streaming(self, *args, **kwargs) -> FakeStream:
                    """Generate fake Qwen3 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.

                    Returns:
                        FakeStream: A fake stream of Qwen3 chunks.
                    """
                    discard(args)
                    discard(kwargs)
                    return self.stream

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(qwen3_cls, "_validate_refs"),
                mock.patch.object(
                    qwen3_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch("celune.backends.qwen3.default_loader", return_value=loader),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                chunks = list(
                    backend.generate_stream(model, text="hello", voice="calm")
                )

            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0][1], 24000)
            self.assertEqual(chunks[1][0].tolist(), [1.0])
            self.assertEqual(model.stream.closed, True)

    def test_qwen3_marks_final_chunk_when_eos_was_not_observed(self) -> None:
        """Verify Qwen3 marks exhausted generations that never surfaced EOS."""

        with mock_qwen3_backend() as qwen3_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                @staticmethod
                def generate_voice_clone_streaming(
                    *args, **kwargs
                ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
                    """Generate fake Qwen3 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    discard(kwargs)
                    yield (
                        np.zeros((1,), dtype=np.float32),
                        24000,
                        {
                            "chunk_steps": 512,
                            "total_steps_so_far": 512,
                            "is_final": True,
                        },
                    )

            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch.object(qwen3_cls, "_validate_refs"),
                mock.patch.object(
                    qwen3_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch("celune.backends.qwen3.default_loader", return_value=loader),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                chunk = next(
                    backend.generate_stream(FakeModel(), text="hello", voice="calm")
                )

            self.assertEqual(chunk[2]["missing_eos"], True)

    def test_voxcpm2_marks_final_chunk_when_stop_token_was_not_observed(self) -> None:
        """Verify VoxCPM2 marks exhausted generations that never surfaced a stop."""

        with mock_voxcpm_backend() as voxcpm2_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                @staticmethod
                def generate_streaming(
                    *args, **kwargs
                ) -> Iterator[npt.NDArray[np.float32]]:
                    """Generate fake VoxCPM2 chunks.

                    Args:
                        args: Arguments used for generation.
                        kwargs: Keyword arguments used for generation.
                    """
                    discard(args)
                    discard(kwargs)
                    for _ in range(512):
                        yield np.ones((1,), dtype=np.float32)

            loader = make_voice_loader(
                "calm", {"cfg_scale": 4.2, "reference_text": "Pack reference."}
            )
            with (
                mock.patch.object(voxcpm2_cls, "_validate_refs"),
                mock.patch.object(
                    voxcpm2_cls, "_truncate_reference", side_effect=lambda path: path
                ),
                mock.patch(
                    "celune.backends.voxcpm2.default_loader", return_value=loader
                ),
            ):
                backend = voxcpm2_cls(log=lambda _msg, _severity="info": None)
                chunks = list(
                    backend.generate_stream(
                        FakeModel(),
                        text="hello",
                        voice="calm",
                        chunk_size=1,
                    )
                )

            self.assertEqual(chunks[-1][2]["missing_eos"], True)


class ExtensionTests(TestCase):
    """Tests for extension context and manager behavior."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.dev_logs: list[tuple[str, str]] = []
        self.invocations: list[tuple[str, tuple[str, ...]]] = []
        self.play_calls: list[tuple[str, bool, float]] = []
        self.context = CeluneContext(
            log=lambda msg, severity="info": self.logs.append((msg, severity)),
            log_dev=lambda msg, severity="info": self.dev_logs.append((msg, severity)),
            say=lambda text, save=True, display_text=None: True,
            think=lambda text: True,
            play=lambda sound_path, keep=False, volume=1.0: (
                self.play_calls.append((sound_path, keep, volume)) or True
            ),
            status=lambda msg, severity="info": None,
            set_voice=lambda name: True,
            get_state=lambda: "idle",
            wait_until_ready=lambda timeout=30.0: True,
        )

    def test_context_and_extension_helpers_delegate_calls(self) -> None:
        """Verify extension helper methods delegate through their context.

        Raises:
            AssertionError: Extension delegation behavior changes unexpectedly.
        """
        extension = DemoExtension(self.context)
        self.context.expose("token", "value")
        self.assertEqual(self.context.get("token"), "value")
        self.assertEqual(extension.state, "idle")
        extension.log("hello")
        self.assertEqual(self.logs[-1], ("[Demo] hello", "info"))
        self.assertEqual(extension.say("hello"), True)
        self.assertEqual(extension.think("hello"), True)
        self.assertEqual(extension.play("tone.wav"), True)
        self.assertEqual(self.play_calls[-1], ("tone.wav", False, 1.0))
        self.assertEqual(extension.play("quiet.wav", keep=True, volume=0.25), True)
        self.assertEqual(self.play_calls[-1], ("quiet.wav", True, 0.25))
        self.assertEqual(extension.set_voice("bold"), True)

    def test_manager_registers_invokes_and_autoloads_extensions(self) -> None:
        """Verify registration, duplicate handling, and directory autoloading.

        Raises:
            AssertionError: Extension manager behavior changes unexpectedly.
        """
        manager = CeluneExtensionManager(self.context)
        manager.register(DemoExtension)
        self.assertEqual(manager.list_extensions(), ["Demo"])
        with self.assertRaises(ExtensionAlreadyRegisteredError):
            manager.register(DemoExtension)
        with self.assertRaises(InvalidExtensionError):
            manager.register(int)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_file = Path(temp_dir) / "fixture.py"
            extension_file.write_text(
                textwrap.dedent(
                    """
                    from celune.extensions.base import CeluneExtension

                    class LoadedExtension(CeluneExtension):
                        EXTENSION_NAME = "Loaded"

                        def invoke(self, *args, **kwargs):
                            return None
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)
        self.assertIn("Loaded", manager.list_extensions())

    def test_manager_invoke_and_autostart_run_in_threads(self) -> None:
        """Verify threaded extension invocation and autostart behavior.

        Raises:
            AssertionError: Threaded extension behavior changes unexpectedly.
        """
        event = threading.Event()

        class AutoExtension(DemoExtension):
            """Autostart extension used by one manager test."""

            EXTENSION_NAME = "Auto"
            AUTOSTART = True

            def autostart(self) -> None:
                event.set()

        manager = CeluneExtensionManager(self.context)
        manager.register(AutoExtension)
        manager.autostart_all()
        self.assertTrue(event.wait(timeout=1))

        invoke_event = threading.Event()

        class InvokeExtension(DemoExtension):
            """Invokable extension used by one manager test."""

            EXTENSION_NAME = "Invoke"

            def invoke(self, *args, **kwargs) -> None:
                invoke_event.set()

        manager.register(InvokeExtension)
        manager.invoke("Invoke", "x")
        self.assertTrue(invoke_event.wait(timeout=5))
        with self.assertRaises(InvalidExtensionError):
            manager.invoke("Missing")


class DemoExtension(CeluneExtension):
    """Simple extension implementation used by manager tests."""

    EXTENSION_NAME = "Demo"

    def invoke(self, *args, **kwargs) -> None:
        return None
