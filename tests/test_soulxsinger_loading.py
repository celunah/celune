# SPDX-License-Identifier: Apache-2.0
"""Tests for SoulX-Singer model loading progress and preparation."""

from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from celune.i18n import string
from celune.backends.tts import soulxsinger


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

    with (
        mock.patch.dict(soulxsinger.sys.modules, {"nltk": nltk}),
        mock.patch.object(soulxsinger, "app_data_dir", return_value=tmp_path),
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


def test_soulxsinger_load_model_reports_each_loading_stage(tmp_path: Path) -> None:
    """Forward model construction, checkpoint, CUDA, and processor stages."""
    backend = object.__new__(soulxsinger.SoulXSinger)
    backend.log = mock.Mock()
    backend._progress_callback = mock.Mock()
    model = mock.Mock()
    processor = mock.Mock()
    model_module = SimpleNamespace(SoulXSinger=mock.Mock(return_value=model))
    processor_module = SimpleNamespace(DataProcessor=mock.Mock(return_value=processor))
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
    logged_messages = [call.args[0] for call in backend.log.call_args_list]
    assert string("soulx.checkpoint_loading") in logged_messages
    assert string("soulx.cuda_transfer") in logged_messages
    assert logged_messages[-1] == string("soulx.model_ready")
