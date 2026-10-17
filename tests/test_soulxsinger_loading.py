# SPDX-License-Identifier: Apache-2.0
"""Tests for SoulX-Singer model loading progress and preparation."""

import builtins
from typing import Any
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from celune.i18n import string
from celune.backends.tts import soulxsinger


def test_soulxsinger_prepares_nltk_before_worker_requests() -> None:
    """Load NLTK in the worker startup thread before CEDTS starts requests."""
    backend = object.__new__(soulxsinger.SoulXSinger)

    with mock.patch.object(soulxsinger.importlib, "import_module") as import_module:
        backend.prepare_model_loading()

    import_module.assert_called_once_with("nltk")


def test_soulxsinger_nltk_preparation_reports_resource_progress(
    tmp_path: Path,
) -> None:
    """Report progress while preparing the English phonemizer datasets."""
    nltk = mock.Mock()
    nltk.data = mock.Mock(path=[])
    nltk.data.find.side_effect = LookupError
    nltk.download.return_value = True
    report_progress = mock.Mock()
    log = mock.Mock()
    import_nltk = builtins.__import__

    def check_import_order(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "nltk":
            assert log.call_args_list == [
                mock.call(string("soulx.phonemizer_data_preparing"), "info")
            ]
            assert report_progress.call_args_list == [mock.call(0, 3)]
        return import_nltk(name, *args, **kwargs)

    with (
        mock.patch.dict(soulxsinger.sys.modules, {"nltk": nltk}),
        mock.patch.object(soulxsinger, "app_data_dir", return_value=tmp_path),
        mock.patch("builtins.__import__", side_effect=check_import_order),
    ):
        soulxsinger._ensure_nltk_data(report_progress, log)

    assert [call.args[0] for call in nltk.download.call_args_list] == [
        "averaged_perceptron_tagger_eng",
        "cmudict",
        "punkt_tab",
    ]
    assert report_progress.call_args_list == [
        mock.call(0, 3),
        mock.call(1, 3),
        mock.call(2, 3),
        mock.call(3, 3),
    ]
    assert log.call_args_list[0] == mock.call(
        string("soulx.phonemizer_data_preparing"), "info"
    )


def test_soulxsinger_preload_prepares_nltk_data() -> None:
    """Prepare phonemizer datasets before the worker loads model weights."""
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
        mock.patch.object(soulxsinger, "_load_upstream_source"),
        mock.patch.object(soulxsinger, "_ensure_nltk_data") as ensure_nltk_data,
    ):
        backend.preload_models()

    ensure_nltk_data.assert_called_once_with(backend.report_progress, backend.log)


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


def test_soulxsinger_load_model_reports_each_loading_stage(tmp_path: Path) -> None:
    """Forward model construction, checkpoint, CUDA, and processor stages."""
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
        mock.patch.object(soulxsinger, "_load_upstream_source", return_value=tmp_path),
        mock.patch.object(
            soulxsinger.importlib,
            "import_module",
            side_effect=modules.__getitem__,
        ),
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
        mock.patch.object(
            soulxsinger,
            "_load_nltk_phonemizer",
            return_value=mock.Mock(),
        ),
        mock.patch.object(soulxsinger.sys, "path", [*soulxsinger.sys.path]),
        mock.patch.dict(soulxsinger.sys.modules, {"omegaconf": omega_module}),
    ):
        backend.load_model(soulxsinger.SOULX_SINGER_MODEL_ID)

    assert backend._progress_callback.call_args_list == [
        mock.call(step, 8) for step in range(9)
    ]
    assert processor_module.load_wav is soulxsinger._load_soulx_reference_wav
    logged_messages = [call.args[0] for call in backend.log.call_args_list]
    assert string("soulx.checkpoint_loading") in logged_messages
    assert string("soulx.cuda_transfer") in logged_messages
    assert logged_messages[-1] == string("soulx.model_ready")
