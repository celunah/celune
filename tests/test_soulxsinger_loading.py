# SPDX-License-Identifier: Apache-2.0
"""Tests for SoulX-Singer model loading progress and preparation."""

from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import pytest

from celune.exceptions import BackendError
from celune.backends.tts import soulxsinger


def test_soulxsinger_preload_prepares_only_the_pinned_source() -> None:
    """Prepare the pinned source without downloading G2P resources."""
    backend = object.__new__(soulxsinger.SoulXSinger)
    backend.log = mock.Mock()
    backend._progress_callback = mock.Mock()
    backend.high_precision = False

    with (
        mock.patch.object(soulxsinger, "require_model_consent"),
        mock.patch.object(
            backend,
            "model_is_available_locally",
            return_value=(True, "cached"),
        ),
        mock.patch.object(soulxsinger, "load_upstream_source") as load_source,
        mock.patch.object(backend, "_ensure_f0_checkpoint") as load_f0,
    ):
        backend.preload_models()

    load_source.assert_called_once_with()
    load_f0.assert_not_called()
    backend.log.assert_not_called()


def test_soulxsinger_preload_downloads_f0_checkpoint_only_in_melody_mode() -> None:
    """Fetch the separate RMVPE checkpoint only when high precision is enabled."""
    backend = object.__new__(soulxsinger.SoulXSinger)
    backend.log = mock.Mock()
    backend._progress_callback = mock.Mock()
    backend.high_precision = True

    with (
        mock.patch.object(soulxsinger, "require_model_consent"),
        mock.patch.object(
            backend,
            "model_is_available_locally",
            return_value=(True, "cached"),
        ),
        mock.patch.object(soulxsinger, "load_upstream_source"),
        mock.patch.object(backend, "_ensure_f0_checkpoint") as load_f0,
    ):
        backend.preload_models()

    load_f0.assert_called_once_with()


def test_soulxsinger_errors_use_english_diagnostics() -> None:
    """Keep backend exception text independent of the active locale."""
    backend = object.__new__(soulxsinger.SoulXSinger)

    with pytest.raises(BackendError) as error:
        backend.load_model("unsupported")

    assert error.value.error_code == "soulx_unsupported_model"
    assert str(error.value) == "SoulX-Singer does not support model 'unsupported'"


def test_soulxsinger_reference_wav_loader_uses_soundfile_and_resamples() -> None:
    """Decode WAV prompts with soundfile and preserve upstream audio shape."""
    audio = soulxsinger.np.array(
        [[0.25, 0.75], [-0.25, -0.75]],
        dtype=soulxsinger.np.float32,
    )
    soundfile = SimpleNamespace(read=mock.Mock(return_value=(audio, 16000)))
    resample = mock.Mock(side_effect=lambda waveform, *_rates: waveform)
    torchaudio = SimpleNamespace(functional=SimpleNamespace(resample=resample))

    with mock.patch.dict(
        soulxsinger.sys.modules,
        {"soundfile": soundfile, "torchaudio": torchaudio},
    ):
        waveform = soulxsinger._load_soulx_reference_wav("reference.wav", 24000)

    soundfile.read.assert_called_once_with(
        "reference.wav",
        dtype="float32",
        always_2d=True,
    )
    resample.assert_called_once()
    resampled_waveform, source_rate, target_rate = resample.call_args.args
    assert soulxsinger.torch.equal(
        resampled_waveform,
        soulxsinger.torch.tensor([[0.25, -0.25], [0.75, -0.75]]),
    )
    assert (source_rate, target_rate) == (16000, 24000)
    assert soulxsinger.torch.equal(
        waveform,
        soulxsinger.torch.tensor([[0.5, -0.5]]),
    )


@pytest.mark.parametrize("high_precision", (False, True))
def test_soulxsinger_load_model_reports_progress_without_stage_logs(
    tmp_path: Path,
    high_precision: bool,
) -> None:
    """Report numeric loading progress without logging internal setup steps."""
    backend = object.__new__(soulxsinger.SoulXSinger)
    backend.log = mock.Mock()
    backend._progress_callback = mock.Mock()
    backend.high_precision = high_precision
    model = mock.Mock()
    processor = mock.Mock()
    model_module = SimpleNamespace(SoulXSinger=mock.Mock(return_value=model))
    processor_module = SimpleNamespace(
        DataProcessor=mock.Mock(return_value=processor),
        load_wav=mock.Mock(),
    )
    f0_extractor = mock.Mock()
    f0_module = SimpleNamespace(
        F0Extractor=mock.Mock(return_value=f0_extractor),
    )
    modules = {
        "soulxsinger.models.soulxsinger": model_module,
        "soulxsinger.utils.data_processor": processor_module,
        "preprocess.tools.f0_extraction": f0_module,
    }
    omega_conf = SimpleNamespace(
        load=mock.Mock(
            return_value={
                "audio": {"hop_size": 256, "sample_rate": 24000},
                "infer": {"n_steps": 4, "cfg": 1.5},
            }
        )
    )
    omega_module = mock.Mock(OmegaConf=omega_conf)

    with (
        mock.patch.object(soulxsinger, "require_model_consent"),
        mock.patch.object(soulxsinger.torch.cuda, "is_available", return_value=True),
        mock.patch.object(soulxsinger, "load_upstream_source", return_value=tmp_path),
        mock.patch.object(
            soulxsinger.importlib,
            "import_module",
            side_effect=modules.__getitem__,
        ) as import_module,
        mock.patch.object(
            backend,
            "model_is_available_locally",
            return_value=(True, str(tmp_path)),
        ),
        mock.patch.object(
            backend,
            "_ensure_f0_checkpoint",
            return_value=tmp_path / "rmvpe.pt",
        ) as ensure_f0,
        mock.patch.object(
            soulxsinger.torch,
            "load",
            return_value={"state_dict": {}},
        ),
        mock.patch.object(soulxsinger.sys, "path", [*soulxsinger.sys.path]),
        mock.patch.dict(soulxsinger.sys.modules, {"omegaconf": omega_module}),
    ):
        backend.prepare_model_loading()
        runtime = backend.load_model(soulxsinger.SOULX_SINGER_MODEL_ID)

    expected_imports = [
        mock.call("soulxsinger.models.soulxsinger"),
        mock.call("soulxsinger.utils.data_processor"),
    ]
    if high_precision:
        expected_imports.append(mock.call("preprocess.tools.f0_extraction"))
    assert import_module.call_args_list == expected_imports
    assert backend._progress_callback.call_args_list == [
        mock.call(step, 8) for step in range(9)
    ]
    assert processor_module.load_wav is soulxsinger._load_soulx_reference_wav
    if high_precision:
        ensure_f0.assert_called_once_with()
        f0_module.F0Extractor.assert_called_once_with(
            str(tmp_path / "rmvpe.pt"),
            device="cuda",
            is_half=True,
            target_sr=24000,
            hop_size=256,
            verbose=False,
        )
        assert runtime.f0_extractor is f0_extractor
    else:
        ensure_f0.assert_not_called()
        assert runtime.f0_extractor is None
    backend.log.assert_not_called()
