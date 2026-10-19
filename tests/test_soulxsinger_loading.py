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

    with (
        mock.patch.object(soulxsinger, "require_model_consent"),
        mock.patch.object(
            backend,
            "model_is_available_locally",
            return_value=(True, "cached"),
        ),
        mock.patch.object(soulxsinger, "load_upstream_source") as load_source,
    ):
        backend.preload_models()

    load_source.assert_called_once_with()
    backend.log.assert_not_called()


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


def test_soulxsinger_load_model_reports_progress_without_stage_logs(
    tmp_path: Path,
) -> None:
    """Report numeric loading progress without logging internal setup steps."""
    backend = object.__new__(soulxsinger.SoulXSinger)
    backend.log = mock.Mock()
    backend._progress_callback = mock.Mock()
    model = mock.Mock()
    processor = mock.Mock()
    model_module = SimpleNamespace(SoulXSinger=mock.Mock(return_value=model))
    processor_module = SimpleNamespace(
        DataProcessor=mock.Mock(return_value=processor),
        load_wav=mock.Mock(),
    )
    modules = {
        "soulxsinger.models.soulxsinger": model_module,
        "soulxsinger.utils.data_processor": processor_module,
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
            soulxsinger.torch,
            "load",
            return_value={"state_dict": {}},
        ),
        mock.patch.object(soulxsinger.sys, "path", [*soulxsinger.sys.path]),
        mock.patch.dict(soulxsinger.sys.modules, {"omegaconf": omega_module}),
    ):
        backend.prepare_model_loading()
        backend.load_model(soulxsinger.SOULX_SINGER_MODEL_ID)

    assert import_module.call_args_list == [
        mock.call("soulxsinger.models.soulxsinger"),
        mock.call("soulxsinger.utils.data_processor"),
    ]
    assert backend._progress_callback.call_args_list == [
        mock.call(step, 7) for step in range(8)
    ]
    assert processor_module.load_wav is soulxsinger._load_soulx_reference_wav
    backend.log.assert_not_called()
