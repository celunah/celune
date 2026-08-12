# SPDX-License-Identifier: MIT
"""Tests for backend resolution and extension infrastructure."""

import contextlib
import importlib
import io
import re
import sys
import tempfile
import textwrap
import threading
from collections.abc import Generator, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional, Union, cast
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch

from celune.backends.tts import resolve_backend
from celune.backends.tts.gpt_sovits import GPTSoVITS, GPTSoVITSPipeline
from celune.backends.vc import resolve_vc_backend
from celune.backends.vc.passthrough import CelunePassthroughVCBackend
from celune.backends.vc.seedvc import CeluneSeedVCBackend
from celune.celune import Celune
from celune.dataclasses.pipeline import VoiceConversionRequest
from celune.exceptions import (
    ExtensionAlreadyRegisteredError,
    InvalidExtensionError,
)
from celune.extensions.base import CeluneContext, CeluneExtension
from celune.extensions.manager import CeluneExtensionManager
from celune.i18n import string
from celune.typing.aliases import AudioChunk
from celune.typing.backends import BackendModel, _SeedVCRealtimeModule
from celune.utils import discard

from .support import (
    CeluneTestCase,
    FakeBackend,
    FakeVCBackend,
    make_voice_loader,
    mock_dotstts_backend,
    mock_mini_backend,
    mock_qwen3_backend,
    mock_voxcpm_backend,
)


class TestBackend(CeluneTestCase):
    """Tests for backend base behavior and backend resolution."""

    def test_gpt_sovits_uses_huggingface_snapshot_paths_for_model_config(self) -> None:
        """Verify GPT-SoVITS model configuration stays outside its source tree."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            (root / "GPT_SoVITS/TTS_infer_pack").mkdir(parents=True)
            (root / "GPT_SoVITS/TTS_infer_pack/TTS.py").touch()
            snapshot = Path(temp_dir) / "huggingface" / "snapshot"
            snapshot.mkdir(parents=True)

            backend = GPTSoVITS(
                log=lambda _msg, _severity="info": None,
                root=str(root),
                variant="v4",
            )
            backend._model_snapshot = snapshot
            config = backend._model_config("v4")
            custom = cast(dict[str, Union[str, bool]], config["custom"])

            assert custom["t2s_weights_path"] == str(snapshot / "s1v3.ckpt")
            assert custom["vits_weights_path"] == str(
                snapshot / "gsv-v4-pretrained/s2Gv4.pth"
            )
            assert "GPT_SoVITS/pretrained_models" not in str(custom)

    def test_gpt_sovits_uses_custom_t2s_checkpoint_override(self) -> None:
        """Verify a configured GPT checkpoint replaces only the variant T2S model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            (root / "GPT_SoVITS/TTS_infer_pack").mkdir(parents=True)
            (root / "GPT_SoVITS/TTS_infer_pack/TTS.py").touch()
            custom_checkpoint = Path(temp_dir) / "custom-e20.ckpt"
            custom_checkpoint.touch()
            snapshot = Path(temp_dir) / "huggingface" / "snapshot"
            for relative_path in (
                "chinese-hubert-base/config.json",
                "chinese-roberta-wwm-ext-large/config.json",
                "gsv-v4-pretrained/s2Gv4.pth",
                "gsv-v4-pretrained/vocoder.pth",
            ):
                target = snapshot / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

            backend = GPTSoVITS(
                log=lambda _msg, _severity="info": None,
                root=str(root),
                variant="v4",
                t2s_weights_path=str(custom_checkpoint),
            )
            backend._model_snapshot = snapshot
            config = backend._model_config("v4")
            custom = cast(dict[str, Union[str, bool]], config["custom"])

            assert custom["t2s_weights_path"] == str(custom_checkpoint)
            assert custom["vits_weights_path"] == str(
                snapshot / "gsv-v4-pretrained/s2Gv4.pth"
            )
            assert backend._variant_is_available(
                snapshot,
                "v4",
                custom_checkpoint,
            )

    def test_gpt_sovits_bootstrap_uses_celune_user_data_directory(self) -> None:
        """Verify missing GPT-SoVITS source is installed below Celune user data."""
        expected_root = Path("C:/runtime-data") / "gpt_sovits"

        with (
            mock.patch(
                "celune.backends.tts.gpt_sovits.runtime_data_dir",
                return_value=expected_root.parent,
            ),
            mock.patch.object(GPTSoVITS, "_candidate_roots", return_value=iter(())),
            mock.patch.object(
                GPTSoVITS, "_download_source_tree", return_value=expected_root
            ) as download,
        ):
            assert GPTSoVITS._resolve_root(None) == expected_root

        download.assert_called_once_with(expected_root)

    def test_gpt_sovits_auto_selects_a_streaming_variant(self) -> None:
        """Verify automatic GPT-SoVITS selection accepts fragment-streaming variants."""
        backend = GPTSoVITS.__new__(GPTSoVITS)
        backend._custom_t2s_weights_path = None
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir)
            for relative_path in (
                "chinese-hubert-base/config.json",
                "chinese-roberta-wwm-ext-large/config.json",
                "s1v3.ckpt",
                "v2Pro/s2Gv2Pro.pth",
                "sv/pretrained_eres2netv2w24s4ep4.ckpt",
                "gsv-v4-pretrained/s2Gv4.pth",
                "gsv-v4-pretrained/vocoder.pth",
            ):
                target = snapshot / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

            assert backend._select_variant(snapshot, None) == "v2Pro"
            assert backend._select_variant(snapshot, "v4") == "v4"

    def test_gpt_sovits_converts_integer_pcm_to_float_audio(self) -> None:
        """Verify GPT-SoVITS int16 PCM is scaled to Celune's float audio range."""
        pcm = np.array([-32768, 0, 32767], dtype=np.int16)

        audio = GPTSoVITS._to_numpy_audio(pcm)

        np.testing.assert_allclose(audio, [-1.0, 0.0, 32767 / 32768])
        assert audio.dtype == np.float32

    def test_gpt_sovits_uses_english_for_unambiguous_latin_text(self) -> None:
        """Verify automatic language selection uses English phonemization for Latin text."""
        assert GPTSoVITS._resolve_text_language("auto", "Hello, Celune.") == "en"
        assert GPTSoVITS._resolve_text_language("auto", "Hello, 你好.") == "auto"
        assert GPTSoVITS._resolve_text_language("ja", "Hello.") == "ja"

    def test_gpt_sovits_rejects_reference_audio_shorter_than_three_seconds(
        self,
    ) -> None:
        """Verify invalid GPT-SoVITS reference duration is rejected before inference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "short.wav"
            sf.write(path, np.zeros(24000, dtype=np.float32), 24000)

            with pytest.raises(ValueError):
                GPTSoVITS._validate_reference_audio("calm", path)

    def test_gpt_sovits_trims_quiet_reference_edges(self) -> None:
        """Verify quiet reference edges are removed while spoken audio is retained."""
        backend = GPTSoVITS.__new__(GPTSoVITS)
        backend._truncated_reference_paths = set()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "calm.wav"
            audio = np.concatenate(
                [
                    np.zeros(12000, dtype=np.float32),
                    np.full(96000, 0.1, dtype=np.float32),
                    np.zeros(12000, dtype=np.float32),
                ]
            )
            sf.write(source, audio, 24000)

            trimmed = backend._trim_reference_silence(source)

            assert trimmed != source
            assert sf.info(trimmed).duration >= 4.0
            assert sf.info(trimmed).duration < 5.0
            assert trimmed in backend._truncated_reference_paths

    def test_gpt_sovits_refreshes_prompt_cache_after_voice_change(self) -> None:
        """Verify a changed voice reference clears official prompt state."""
        pipeline = SimpleNamespace(
            prompt_cache={
                "ref_audio_path": "old.wav",
                "prompt_semantic": object(),
                "refer_spec": [object()],
                "prompt_text": "Old voice.",
                "prompt_lang": "en",
            }
        )

        GPTSoVITS._refresh_prompt_cache(
            cast(GPTSoVITSPipeline, pipeline),
            Path("new.wav"),
            "New voice.",
            "en",
        )

        assert pipeline.prompt_cache["ref_audio_path"] is None
        assert pipeline.prompt_cache["refer_spec"] == []
        assert pipeline.prompt_cache["prompt_text"] is None

    def test_base_backend_reports_models(self) -> None:
        """Verify model metadata helpers on a fake backend.

        Raises:
            AssertionError: A backend helper returns an unexpected value.
        """
        backend = FakeBackend(log=lambda _msg, _severity="info": None)
        assert backend.default_model_id == "fake/balanced"
        assert backend.all_model_ids == ["fake/balanced", "fake/bold"]
        assert backend.voices == ["balanced", "bold"]
        assert backend.model_id_for_voice("bold") == "fake/bold"

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

        with mock.patch("celune.backends.tts.base.default_loader", return_value=loader):
            backend = FakeBackend(log=lambda _msg, _severity="info": None)
            backend.validate_refs()

        assert materialize.call_args_list == [
            mock.call("balanced", "wav"),
            mock.call("balanced", "pt"),
            mock.call("bold", "wav"),
        ]

    def test_base_backend_truncates_long_reference_wav_to_ten_seconds(self) -> None:
        """Verify the shared reference helper clips long WAV prompts to ten seconds."""
        backend = FakeBackend(log=lambda _msg, _severity="info": None)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "reference.wav"
            canonical_temp = Path(temp_dir) / "celune-temp"
            canonical_temp.mkdir(parents=True, exist_ok=True)
            sf.write(source, np.zeros(12 * 24000, dtype=np.float32), 24000)

            with mock.patch(
                "celune.backends.tts.base.temp_data_dir", return_value=canonical_temp
            ):
                truncated = backend.truncate_reference(source)

            assert truncated != source
            assert sf.info(truncated).duration <= 10.0
            assert truncated.parent == canonical_temp
            assert source.exists()

            backend.unload_model()
            assert not truncated.exists()

    def test_resolve_backend_accepts_instance_type_and_rejects_unknown(self) -> None:
        """Verify supported backend specifications and invalid input failures.

        Raises:
            AssertionError: Backend resolution behavior changes unexpectedly.
        """
        instance = FakeBackend(log=lambda _msg, _severity="info": None)
        assert resolve_backend(instance) is instance
        assert isinstance(resolve_backend(FakeBackend), FakeBackend)
        with pytest.raises(
            ValueError,
            match=re.escape(
                string(
                    "celune.unknown_backend",
                    backend="missing",
                    available="mini, qwen3, dotstts, voxcpm2, gpt-sovits",
                )
            ),
        ):
            resolve_backend("missing")
        with pytest.raises(TypeError, match="backend_name"):
            resolve_backend(123)  # type: ignore[arg-type]

    def test_resolve_vc_backend_accepts_instance_type_and_rejects_unknown(self) -> None:
        """Verify supported VC backend specifications and invalid input failures."""
        instance = FakeVCBackend(log=lambda _msg, _severity="info": None)
        assert resolve_vc_backend(instance) is instance
        assert isinstance(resolve_vc_backend(FakeVCBackend), FakeVCBackend)
        with pytest.raises(ValueError, match="unknown voice-conversion backend"):
            resolve_vc_backend("missing")
        with pytest.raises(TypeError, match="voice-conversion backend"):
            resolve_vc_backend(123)  # type: ignore[arg-type]

    def test_unload_model_releases_nested_runtime_members(self) -> None:
        """Verify backend unload clears nested releasable members hidden inside wrapper objects."""

        class NestedRuntime:
            """Minimal nested runtime object exposing a close hook."""

            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                """Record runtime shutdown."""
                self.closed = True

        class WrapperRuntime:
            """Minimal wrapper that keeps releasable objects in nested attributes."""

            def __init__(self) -> None:
                self.inner = NestedRuntime()
                self.cache = {"child": NestedRuntime()}

        backend = FakeBackend(log=lambda _msg, _severity="info": None)
        runtime = WrapperRuntime()
        inner = runtime.inner
        cached = runtime.cache["child"]
        backend.model = cast(BackendModel, runtime)

        backend.unload_model()

        assert inner.closed
        assert cached.closed
        assert backend.model is None

    def test_passthrough_vc_backend_returns_playable_output(self) -> None:
        """Verify the passthrough VC backend returns decoded audio unchanged."""
        backend = CelunePassthroughVCBackend(log=lambda _msg, _severity="info": None)
        source = np.ones((12, 2), dtype=np.float32)

        output = backend.convert(
            VoiceConversionRequest(
                source_audio=source,
                sample_rate=44100,
                target_voice="balanced",
                target_character="Celune",
                label="fixture audio",
            )
        )

        assert output.sample_rate == 44100
        assert output.label == "fixture audio"
        assert output.audio.shape == (12, 2)
        assert np.array_equal(output.audio, source)
        assert output.audio is not source

    def test_resolve_vc_backend_accepts_seedvc_backend_name(self) -> None:
        """Verify the Seed-VC backend resolves through the VC backend registry."""
        backend = resolve_vc_backend("seed-vc")
        assert isinstance(backend, CeluneSeedVCBackend)
        assert backend.name == "seed-vc"

    def test_seedvc_backend_requires_reference_audio(self) -> None:
        """Verify Seed-VC refuses requests without a target reference WAV."""
        backend = CeluneSeedVCBackend(log=lambda _msg, _severity="info": None)

        with pytest.raises(ValueError, match="requires at least one target reference"):
            backend.convert(
                VoiceConversionRequest(
                    source_audio=np.ones((8,), dtype=np.float32),
                    sample_rate=24000,
                    label="fixture",
                )
            )

    def test_seedvc_backend_converts_audio_with_cached_wrapper(self) -> None:
        """Verify Seed-VC wraps converted audio into Celune's VC output contract."""
        backend = CeluneSeedVCBackend(log=lambda _msg, _severity="info": None)
        captured: dict[str, Union[str, int, float, bool]] = {}

        class FakeWrapper:
            """Minimal Seed-VC wrapper stand-in for one backend test."""

            @staticmethod
            def convert_voice(**kwargs):
                """Return a generator-style conversion result for backend tests.

                Args:
                    kwargs: Wrapper arguments forwarded from the backend under test.

                Returns:
                    Generator[None, None, npt.NDArray[np.float32]]: One generator whose return value carries the
                    converted waveform.
                """
                captured.update(kwargs)
                assert Path(str(kwargs["source"])).exists()
                assert Path(str(kwargs["target"])).exists()

                def result_generator() -> Generator[None, None, AudioChunk]:
                    yield from ()
                    return np.array([0.25, -0.25], dtype=np.float32)

                return result_generator()

        backend._wrapper = FakeWrapper()

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target.wav"
            sf.write(target, np.zeros((16,), dtype=np.float32), 24000)

            output = backend.convert(
                VoiceConversionRequest(
                    source_audio=np.ones((12, 2), dtype=np.float32),
                    sample_rate=24000,
                    target_voice="balanced",
                    target_character="Celune",
                    target_references=(target,),
                    label="fixture audio",
                )
            )

        assert output.sample_rate == 22050
        assert output.label == "fixture audio"
        assert output.audio.dtype == np.float32
        assert output.audio.tolist() == [0.25, -0.25]
        assert not captured["stream_output"]
        assert not captured["f0_condition"]
        assert captured["pitch_shift"] == 0

    def test_seedvc_live_backend_uses_native_session_without_offline_wrapper(
        self,
    ) -> None:
        """Verify live VC sends blocks to Seed-VC's native session directly."""
        backend = CeluneSeedVCBackend(log=lambda _msg, _severity="info": None)
        source = np.ones((8640, 2), dtype=np.float32)
        reference = Path("reference.wav")
        request = VoiceConversionRequest(
            source_audio=source,
            sample_rate=48000,
            target_references=(reference,),
            label="live fixture",
        )
        realtime_module = ModuleType("seed_vc.real_time")
        model_set = (
            "model",
            "semantic",
            "vocoder",
            "campplus",
            "mel",
            {"sampling_rate": 22050},
        )

        with (
            mock.patch.object(
                backend,
                "_get_live_runtime",
                return_value=(realtime_module, model_set),
            ),
            mock.patch.object(backend, "_initialize_live_session") as initialize,
            mock.patch.object(
                backend,
                "_convert_live_audio",
                return_value=np.array([0.1, -0.1], dtype=np.float32),
            ) as convert,
        ):
            output = backend.convert_live(request)

        initialize.assert_called_once_with(
            realtime_module,
            model_set,
            reference,
            48000,
        )
        convert.assert_called_once_with(source, 48000)
        assert output.sample_rate == 48000
        assert output.label == "live fixture"
        np.testing.assert_array_equal(
            output.audio,
            np.array([0.1, -0.1], dtype=np.float32),
        )

    def test_seedvc_live_loader_uses_the_module_source_directory(self) -> None:
        """Verify native Seed-VC relative resources resolve from its package root."""
        backend = CeluneSeedVCBackend(log=lambda _msg, _severity="info": None)
        module = ModuleType("seed_vc.real_time")
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            source_file = source_root / "real-time-gui.py"
            source_file.touch()
            module.__file__ = str(source_file)
            captured: dict[str, Path] = {}
            load_count = 0

            def load_models(_args: SimpleNamespace) -> tuple[object, ...]:
                """Capture the working directory used by native model loading."""
                nonlocal load_count
                load_count += 1
                captured["cwd"] = Path.cwd()
                return ("model", "semantic", "vocoder", "campplus", "mel", {})

            setattr(module, "load_models", load_models)
            captured_stdout = io.StringIO()
            with (
                contextlib.redirect_stdout(captured_stdout),
                mock.patch.object(backend, "_load_live_module", return_value=module),
            ):
                backend._get_live_runtime()
                backend._get_live_runtime()

        assert captured["cwd"] == source_root
        assert load_count == 1
        assert captured_stdout.getvalue() == ""

    def test_seedvc_live_inference_keeps_native_stdout_off_the_worker_stream(
        self,
    ) -> None:
        """Verify native live inference diagnostics cannot corrupt worker framing."""
        backend = CeluneSeedVCBackend(log=lambda _msg, _severity="info": None)
        module = ModuleType("seed_vc.real_time")

        def custom_infer(*_args: object) -> torch.Tensor:
            """Emit diagnostics that the native implementation writes during inference."""
            print("target_lengths: tensor([1])")
            print("0%| | 0/10", file=sys.stderr)
            return torch.zeros(4)

        setattr(module, "custom_infer", custom_infer)
        backend._live_module = cast(_SeedVCRealtimeModule, module)
        backend._live_model_set = ("model",)
        backend._live_reference_path = Path("reference.wav")
        backend._live_reference_wav = np.zeros(4, dtype=np.float32)
        backend._live_model_sample_rate = 22050
        backend._live_block_frame = 4
        backend._live_block_frame_16k = 1
        backend._live_input_wav = torch.zeros(4)
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(captured_stdout),
            contextlib.redirect_stderr(captured_stderr),
            mock.patch.object(
                backend,
                "_apply_live_sola",
                return_value=torch.zeros(4),
            ),
        ):
            output = backend._convert_live_block(
                np.zeros(4, dtype=np.float32),
            )

        assert captured_stdout.getvalue() == ""
        assert captured_stderr.getvalue() == ""
        assert output.shape == (4,)

    def test_seedvc_backend_uses_configured_pitch_shift_for_wrapper_requests(
        self,
    ) -> None:
        """Verify Seed-VC forwards its configured pitch shift into wrapper requests."""
        backend = CeluneSeedVCBackend(
            log=lambda _msg, _severity="info": None,
            pitch_shift=-6,
        )
        captured: dict[str, Union[str, int, float, bool]] = {}

        class FakeWrapper:
            """Minimal Seed-VC wrapper stand-in for request override coverage."""

            @staticmethod
            def convert_voice(**kwargs):
                """Return a generator-style conversion result for backend tests.

                Args:
                    kwargs: Wrapper arguments forwarded from the backend under test.

                Returns:
                    Generator[None, None, npt.NDArray[np.float32]]: One generator whose return value carries the
                    converted waveform.
                """
                captured.update(kwargs)

                def result_generator() -> Generator[None, None, AudioChunk]:
                    yield from ()
                    return np.array([0.1, -0.1], dtype=np.float32)

                return result_generator()

        backend._wrapper = FakeWrapper()

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target.wav"
            sf.write(target, np.zeros((16,), dtype=np.float32), 24000)

            backend.convert(
                VoiceConversionRequest(
                    source_audio=np.ones((12, 2), dtype=np.float32),
                    sample_rate=24000,
                    target_references=(target,),
                    pitch_shift=9,
                )
            )

        assert captured["pitch_shift"] == 9

    def test_seedvc_backend_prefers_request_f0_condition_over_backend_default(
        self,
    ) -> None:
        """Verify one conversion request can override talk vs sing mode."""
        backend = CeluneSeedVCBackend(
            log=lambda _msg, _severity="info": None,
            f0_condition=False,
        )
        captured: dict[str, Union[str, int, float, bool]] = {}

        class FakeWrapper:
            """Minimal Seed-VC wrapper stand-in for f0 override coverage."""

            @staticmethod
            def convert_voice(**kwargs):
                """Return a generator-style conversion result for backend tests.

                Args:
                    kwargs: Wrapper arguments forwarded from the backend under test.

                Returns:
                    Generator[None, None, npt.NDArray[np.float32]]: One generator whose return value carries the
                    converted waveform.
                """
                captured.update(kwargs)

                def result_generator() -> Generator[None, None, AudioChunk]:
                    yield from ()
                    return np.array([0.1, -0.1], dtype=np.float32)

                return result_generator()

        backend._wrapper = FakeWrapper()

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target.wav"
            sf.write(target, np.zeros((16,), dtype=np.float32), 24000)

            output = backend.convert(
                VoiceConversionRequest(
                    source_audio=np.ones((12, 2), dtype=np.float32),
                    sample_rate=24000,
                    target_references=(target,),
                    f0_condition=True,
                )
            )

        assert captured["f0_condition"]
        assert output.sample_rate == 44100

    def test_seedvc_backend_redirects_package_checkpoint_downloads_into_hf_cache(
        self,
    ) -> None:
        """Verify Celune redirects Seed-VC downloads into the shared Hugging Face cache."""
        backend = CeluneSeedVCBackend(log=lambda _msg, _severity="info": None)
        captured: list[tuple[str, str, str]] = []

        fake_hf_utils = ModuleType("seed_vc.hf_utils")
        fake_wrapper = ModuleType("seed_vc.seed_vc_wrapper")

        def fake_hf_hub_download(
            repo_id: str,
            filename: str,
            cache_dir: str,
        ) -> str:
            captured.append((repo_id, filename, cache_dir))
            return str(Path(cache_dir) / filename)

        setattr(fake_hf_utils, "hf_hub_download", fake_hf_hub_download)
        setattr(
            fake_hf_utils,
            "load_custom_model_from_hf",
            lambda *args, **kwargs: None,
        )
        setattr(fake_wrapper, "load_custom_model_from_hf", lambda *args, **kwargs: None)
        setattr(fake_wrapper, "SeedVCWrapper", type("FakeSeedVCWrapper", (), {}))

        def import_module(name: str) -> ModuleType:
            if name == "seed_vc.hf_utils":
                return fake_hf_utils
            if name == "seed_vc.seed_vc_wrapper":
                return fake_wrapper
            return importlib.import_module(name)

        with tempfile.TemporaryDirectory() as temp_dir:
            expected_cache_dir = Path(temp_dir) / "huggingface" / "hub"

            with (
                mock.patch(
                    "celune.backends.vc.seedvc.huggingface_hub_cache_dir",
                    return_value=expected_cache_dir,
                ),
                mock.patch(
                    "celune.backends.vc.seedvc.importlib.import_module",
                    side_effect=import_module,
                ),
            ):
                backend._load_wrapper_type()
                resolved = getattr(fake_wrapper, "load_custom_model_from_hf")(
                    "funasr/campplus",
                    "campplus_cn_common.bin",
                    None,
                )

        assert Path(resolved) == expected_cache_dir / "campplus_cn_common.bin"
        assert captured == [
            (
                "funasr/campplus",
                "campplus_cn_common.bin",
                str(expected_cache_dir),
            )
        ]

    def test_resolve_backend_accepts_mini_backend_name(self) -> None:
        """Verify the Pocket TTS backend resolves through the backend registry."""

        class StubTTSModel:
            """Import-time stand-in for the Pocket TTS package class."""

        with mock.patch.dict(
            sys.modules,
            {"pocket_tts": SimpleNamespace(TTSModel=StubTTSModel)},
        ):
            mini = importlib.import_module("celune.backends.tts.mini")
            mini_cls = mini.Mini

            with mock.patch.object(mini_cls, "_validate_refs"):
                backend = resolve_backend("mini")

        assert isinstance(backend, mini_cls)
        assert backend.name == "mini"

    def test_resolve_backend_accepts_dotstts_backend_name(self) -> None:
        """Verify the dots.tts backend resolves through the backend registry."""

        with (
            mock_dotstts_backend() as dotstts_cls,
            mock.patch.object(dotstts_cls, "_validate_refs"),
        ):
            backend = resolve_backend("dotstts")

        assert isinstance(backend, dotstts_cls)
        assert backend.name == "dotstts"

    def test_voxcpm2_uses_pack_cfg_scale_when_present(self) -> None:
        """Verify CEVOICE can override VoxCPM2's per-voice CFG scale."""

        with mock_voxcpm_backend() as voxcpm2_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.cfg_value = None

                def generate_streaming(self, *args, **kwargs) -> Iterator[AudioChunk]:
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
                    "celune.backends.tts.voxcpm2.default_loader", return_value=loader
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

            assert model.cfg_value == 4.2

    def test_voxcpm2_reloads_the_checkpoint_tokenizer(self) -> None:
        """Verify VoxCPM2 replaces its package loader's incompatible tokenizer."""
        with mock_voxcpm_backend() as voxcpm2_cls:
            tokenizer = mock.Mock()
            model = SimpleNamespace(tts_model=SimpleNamespace(text_tokenizer=None))
            with mock.patch(
                "celune.backends.tts.voxcpm2.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ):
                voxcpm2_cls._install_checkpoint_tokenizer(model, "snapshot")

            tokenizer.encode.return_value = [11]
            assert model.tts_model.text_tokenizer("hello") == [11]
            tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)

    @staticmethod
    def test_voxcpm2_requires_reference_text_for_valid_voice_identifiers() -> None:
        """Verify VoxCPM2 enters fatal state when pack voices omit reference text."""

        with mock_voxcpm_backend() as voxcpm2_cls:
            loader = make_voice_loader("calm", {})
            fatal = mock.Mock()
            with mock.patch(
                "celune.backends.tts.voxcpm2.default_loader", return_value=loader
            ):
                voxcpm2_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

    def test_voxcpm2_uses_truncated_reference_wav_when_present(self) -> None:
        """Verify VoxCPM2 passes reference audio through the shared truncation hook."""

        with mock_voxcpm_backend() as voxcpm2_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                def __init__(self) -> None:
                    self.reference_wav_path = None

                def generate_streaming(self, *args, **kwargs) -> Iterator[AudioChunk]:
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
                    "celune.backends.tts.voxcpm2.default_loader", return_value=loader
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

            assert model.reference_wav_path == Path("trimmed.wav")

    @staticmethod
    def test_voxcpm2_requires_a_compatible_voice_pack() -> None:
        """Verify VoxCPM2 enters fatal state without a usable CEVOICE/CECHAR pack."""

        with (
            mock_voxcpm_backend() as voxcpm2_cls,
            mock.patch("celune.backends.tts.voxcpm2.default_loader", return_value=None),
        ):
            fatal = mock.Mock()
            voxcpm2_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

    @staticmethod
    def test_mini_requires_reference_text_for_valid_voice_identifiers() -> None:
        """Verify Mini enters fatal state when pack voices omit reference text."""

        with mock_mini_backend() as mini_cls:
            loader = make_voice_loader("calm", {})
            fatal = mock.Mock()
            with mock.patch(
                "celune.backends.tts.mini.default_loader", return_value=loader
            ):
                mini_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

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
                mock.patch(
                    "celune.backends.tts.mini.default_loader", return_value=loader
                ),
                mock.patch.object(
                    mini_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = mini_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            assert model.audio_conditioning == str(Path("trimmed.wav"))

    def test_mini_does_not_apply_reference_wav_truncation_to_reference_text(
        self,
    ) -> None:
        """Verify Mini does not treat transcript text as a WAV path during validation."""

        with mock_mini_backend() as mini_cls:
            loader = make_voice_loader("calm", {"reference_text": "Pack reference."})
            with (
                mock.patch(
                    "celune.backends.tts.mini.default_loader", return_value=loader
                ),
                mock.patch.object(
                    mini_cls,
                    "_truncate_reference",
                    side_effect=AssertionError("should not see reference text"),
                ),
            ):
                backend = mini_cls(log=lambda _msg, _severity="info": None)
                assert backend.voices == ["calm"]

    @staticmethod
    def test_mini_requires_a_compatible_voice_pack() -> None:
        """Verify Mini enters fatal state without a usable CEVOICE/CECHAR pack."""

        with (
            mock_mini_backend() as mini_cls,
            mock.patch("celune.backends.tts.mini.default_loader", return_value=None),
        ):
            fatal = mock.Mock()
            mini_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

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
                ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
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
                mock.patch(
                    "celune.backends.tts.qwen3.default_loader", return_value=loader
                ),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            assert model.ref_text == "Pack reference."

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
                ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
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
                mock.patch(
                    "celune.backends.tts.qwen3.default_loader", return_value=loader
                ),
                mock.patch.object(
                    qwen3_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            assert model.ref_text == "Pack reference."
            assert model.ref_audio == Path("trimmed.wav")

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
                    "celune.backends.tts.dotstts.default_loader", return_value=loader
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            assert model.prompt_text == "Pack reference."

    def test_dotstts_reloads_the_tokenizer_with_the_mistral_regex_fix(self) -> None:
        """Verify dots.tts applies Transformers' Mistral tokenizer compatibility flag."""
        with mock_dotstts_backend() as dotstts_cls:
            tokenizer = object()
            model = SimpleNamespace(
                pretrained_path=Path("snapshot"),
                model=SimpleNamespace(
                    tokenizer=object(),
                    core=SimpleNamespace(tokenizer=object()),
                ),
            )
            with mock.patch(
                "celune.backends.tts.dotstts.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer:
                dotstts_cls._fix_checkpoint_tokenizer(model)

            load_tokenizer.assert_called_once_with(
                "snapshot",
                local_files_only=True,
                fix_mistral_regex=True,
            )
            assert model.model.tokenizer is tokenizer
            assert model.model.core.tokenizer is tokenizer

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
                    "celune.backends.tts.dotstts.default_loader", return_value=loader
                ),
                mock.patch.object(
                    dotstts_cls, "_truncate_reference", return_value=Path("trimmed.wav")
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello", voice="calm"))

            assert model.prompt_text == "Pack reference."
            assert model.prompt_audio_path == str(Path("trimmed.wav"))

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
                    "celune.backends.tts.dotstts.default_loader", return_value=loader
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                list(backend.generate_stream(model, text="hello"))

            assert model.prompt_audio_path == str(Path("calm.wav"))
            assert model.prompt_text == "Pack reference."

    @staticmethod
    def test_dotstts_requires_reference_text_for_valid_voice_identifiers() -> None:
        """Verify dots.tts enters fatal state when pack voices omit reference text."""

        with mock_dotstts_backend() as dotstts_cls:
            loader = make_voice_loader("calm", {})
            fatal = mock.Mock()
            with mock.patch(
                "celune.backends.tts.dotstts.default_loader", return_value=loader
            ):
                dotstts_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

    @staticmethod
    def test_dotstts_requires_a_compatible_voice_pack() -> None:
        """Verify dots.tts enters fatal state without a usable CEVOICE/CECHAR pack."""

        with (
            mock_dotstts_backend() as dotstts_cls,
            mock.patch("celune.backends.tts.dotstts.default_loader", return_value=None),
        ):
            fatal = mock.Mock()
            dotstts_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

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
                    "celune.backends.tts.dotstts.default_loader", return_value=loader
                ),
            ):
                backend = dotstts_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                chunks = list(
                    backend.generate_stream(model, text="hello", voice="calm")
                )

            assert len(chunks) == 2
            assert chunks[0][1] == 48000
            assert chunks[1][0].tolist() == [1.0]
            assert model.stream.closed

    @staticmethod
    def test_dotstts_suppresses_loguru_runtime_noise() -> None:
        """Verify dots.tts suppression also disables its Loguru logger namespace."""

        with mock_dotstts_backend() as dotstts_cls:
            fake_loguru = mock.Mock()
            with (
                mock.patch("celune.backends.tts.dotstts.loguru.logger", fake_loguru),
                dotstts_cls.suppress_backend_output(),
            ):
                pass

        fake_loguru.disable.assert_called_once_with("dots_tts")
        fake_loguru.enable.assert_called_once_with("dots_tts")

    @staticmethod
    def test_qwen3_requires_reference_text_for_valid_voice_identifiers() -> None:
        """Verify Qwen3 enters fatal state when pack voices omit reference text."""

        with mock_qwen3_backend() as qwen3_cls:
            loader = make_voice_loader("calm", {})
            fatal = mock.Mock()
            with mock.patch(
                "celune.backends.tts.qwen3.default_loader", return_value=loader
            ):
                qwen3_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

    @staticmethod
    def test_qwen3_requires_a_compatible_voice_pack() -> None:
        """Verify Qwen3 enters fatal state without a usable CEVOICE/CECHAR pack."""

        with (
            mock_qwen3_backend() as qwen3_cls,
            mock.patch("celune.backends.tts.qwen3.default_loader", return_value=None),
        ):
            fatal = mock.Mock()
            qwen3_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

    @staticmethod
    def test_qwen3_requires_at_least_one_valid_voice_identifier() -> None:
        """Verify Qwen3 enters fatal state when no usable voice names exist."""

        loader = SimpleNamespace(
            bundle=SimpleNamespace(
                voices={"": {"reference_text": "ignored"}},
                voice_order=("",),
            ),
            materialize=lambda ref_voice, kind: Path(f"{ref_voice}.{kind}"),
        )

        with (
            mock_qwen3_backend() as qwen3_cls,
            mock.patch("celune.backends.tts.qwen3.default_loader", return_value=loader),
        ):
            fatal = mock.Mock()
            qwen3_cls(log=lambda _msg, _severity="info": None, fatal=fatal)

        fatal.assert_called_once_with()

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
                ) -> tuple[AudioChunk, int, Optional[dict]]:
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
                mock.patch(
                    "celune.backends.tts.qwen3.default_loader", return_value=loader
                ),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                model = FakeModel()
                chunks = list(
                    backend.generate_stream(model, text="hello", voice="calm")
                )

            assert len(chunks) == 2
            assert chunks[0][1] == 24000
            assert chunks[1][0].tolist() == [1.0]
            assert model.stream.closed

    def test_qwen3_marks_final_chunk_when_eos_was_not_observed(self) -> None:
        """Verify Qwen3 marks exhausted generations that never surfaced EOS."""

        with mock_qwen3_backend() as qwen3_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                @staticmethod
                def generate_voice_clone_streaming(
                    *args, **kwargs
                ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
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
                mock.patch(
                    "celune.backends.tts.qwen3.default_loader", return_value=loader
                ),
            ):
                backend = qwen3_cls(log=lambda _msg, _severity="info": None)
                chunk = next(
                    backend.generate_stream(FakeModel(), text="hello", voice="calm")
                )

            assert chunk[2]["missing_eos"]

    def test_voxcpm2_marks_final_chunk_when_stop_token_was_not_observed(self) -> None:
        """Verify VoxCPM2 marks exhausted generations that never surfaced a stop."""

        with mock_voxcpm_backend() as voxcpm2_cls:

            class FakeModel:
                """Fake model class for use in this test suite."""

                @staticmethod
                def generate_streaming(*args, **kwargs) -> Iterator[AudioChunk]:
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
                    "celune.backends.tts.voxcpm2.default_loader", return_value=loader
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

            assert chunks[-1][2]["missing_eos"]


class TestExtension(CeluneTestCase):
    """Tests for extension context and manager behavior."""

    def setUp(self) -> None:
        """Reset extension-manager state before each extension test."""
        self.backend_override = mock.Mock(
            side_effect=lambda backend_name: contextlib.nullcontext(
                cast(Celune, SimpleNamespace())
            )
        )

        self.cevoice_override = mock.Mock(
            side_effect=lambda bundle: contextlib.nullcontext(
                cast(Celune, SimpleNamespace())
            )
        )
        self.logs: list[tuple[str, str]] = []
        self.invocations: list[tuple[str, tuple[str, ...]]] = []
        self.play_calls: list[tuple[str, bool, float]] = []
        self.context = CeluneContext(
            log=lambda msg, severity="info", **kwargs: self.logs.append(
                (msg, severity)
            ),
            log_level="verbose",
            say=lambda text, save=True, display_text=None: True,
            think=lambda text: True,
            play=lambda sound_path, keep=False, volume=1.0: (
                self.play_calls.append((sound_path, keep, volume)) or True
            ),
            status=lambda msg, severity="info": None,
            set_voice=lambda name: True,
            get_state=lambda: "idle",
            wait_until_ready=lambda timeout=30.0: True,
            backend_override=self.backend_override,
            cevoice_override=self.cevoice_override,
        )

    def test_context_and_extension_helpers_delegate_calls(self) -> None:
        """Verify extension helper methods delegate through their context.

        Raises:
            AssertionError: Extension delegation behavior changes unexpectedly.
        """
        extension = DemoExtension(self.context)
        self.context.expose("token", "value")
        assert self.context.get("token") == "value"
        assert extension.state == "idle"
        extension.log("hello")
        assert self.logs[-1] == ("[Demo] hello", "info")
        assert extension.say("hello")
        assert extension.think("hello")
        assert extension.play("tone.wav")
        assert self.play_calls[-1] == ("tone.wav", False, 1.0)
        assert extension.play("quiet.wav", keep=True, volume=0.25)
        assert self.play_calls[-1] == ("quiet.wav", True, 0.25)
        assert extension.set_voice("bold")
        with extension.with_backend("mini"):
            pass
        with extension.with_cevoice("nova"):
            pass
        self.backend_override.assert_called_once_with("mini")
        self.cevoice_override.assert_called_once_with("nova")

    def test_manager_registers_invokes_and_autoloads_extensions(self) -> None:
        """Verify registration, duplicate handling, and directory autoloading.

        Raises:
            AssertionError: Extension manager behavior changes unexpectedly.
        """
        manager = CeluneExtensionManager(self.context)
        manager.register(DemoExtension)
        assert manager.list_extensions() == ["Demo"]
        with pytest.raises(ExtensionAlreadyRegisteredError):
            manager.register(DemoExtension)
        with pytest.raises(InvalidExtensionError):
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
        assert "Loaded" in manager.list_extensions()

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
        assert event.wait(timeout=1)

        invoke_event = threading.Event()

        class InvokeExtension(DemoExtension):
            """Invokable extension used by one manager test."""

            EXTENSION_NAME = "Invoke"

            def invoke(self, *args, **kwargs) -> None:
                invoke_event.set()

        manager.register(InvokeExtension)
        manager.invoke("Invoke", "x")
        assert invoke_event.wait(timeout=5)
        with pytest.raises(InvalidExtensionError):
            manager.invoke("Missing")


class DemoExtension(CeluneExtension):
    """Simple extension implementation used by manager tests."""

    EXTENSION_NAME = "Demo"

    def invoke(self, *args, **kwargs) -> None:
        return None
