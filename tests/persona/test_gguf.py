# SPDX-License-Identifier: Apache-2.0
"""Tests for Persona GGUF package selection and runtime fallback."""

import hashlib
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from celune.dataclasses.persona import ChatMessage, GenerateRequest
from celune.exceptions import ModelConsentRequired
from celune.persona import gguf, runtime
from celune.registry import get_model_metadata
from celune.registry.llm import PERSONA_DEFAULT_MODEL_ID


def test_gguf_artifacts_are_limited_to_reviewed_model_and_precisions() -> None:
    """Resolve only the pinned Qwen3-VL packages for matching precisions."""
    assert (
        gguf.persona_gguf_variant(PERSONA_DEFAULT_MODEL_ID, "4bit")
        == (gguf.QWEN3_VL_GGUF_VARIANTS["4bit"])
    )
    assert (
        gguf.persona_gguf_variant(PERSONA_DEFAULT_MODEL_ID, "8bit")
        == (gguf.QWEN3_VL_GGUF_VARIANTS["8bit"])
    )
    assert gguf.persona_gguf_variant("Qwen/Qwen3-VL-8B-Instruct", "4bit") is None
    assert gguf.persona_gguf_variant(PERSONA_DEFAULT_MODEL_ID, "none") is None


def test_llama_cpp_archives_include_server_and_cuda_libraries() -> None:
    """Pin both the llama-server executable and CUDA support files per platform."""
    with (
        mock.patch.object(gguf.sys, "platform", "win32"),
        mock.patch.object(gguf.platform, "machine", return_value="AMD64"),
    ):
        windows_archives = gguf.llama_cpp_archives()
    assert [archive.filename for archive in windows_archives] == [
        "llama-b12110-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]
    assert [archive.size for archive in windows_archives] == [
        287_014_657,
        391_443_627,
    ]

    with (
        mock.patch.object(gguf.sys, "platform", "linux"),
        mock.patch.object(gguf.platform, "machine", return_value="x86_64"),
    ):
        linux_archives = gguf.llama_cpp_archives()
    assert [archive.filename for archive in linux_archives] == [
        "llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz",
    ]
    assert [archive.size for archive in linux_archives] == [
        187_314_621,
        594_371_477,
    ]


def test_gguf_model_load_falls_back_once_when_server_initialization_fails() -> None:
    """Cache native fallback after one failed llama.cpp initialization."""
    log_messages: list[tuple[str, str]] = []
    persona_runtime = runtime.PersonaRuntime(
        config={"vram": "high", "inference_mode": "gguf"},
        log=lambda message, severity: log_messages.append((message, severity)),
    )

    def load_native(model_id: str, quantization: str) -> None:
        persona_runtime.backend.model_id = model_id
        persona_runtime.backend.quantization = quantization

    with (
        mock.patch("celune.persona.runtime.require_model_consent"),
        mock.patch(
            "celune.persona.gguf.download_persona_gguf",
            return_value=(Path("model.gguf"), Path("mmproj.gguf")),
        ),
        mock.patch(
            "celune.persona.gguf.ensure_llama_server",
            return_value=Path("llama-server"),
        ),
        mock.patch(
            "celune.persona.gguf.LlamaCppPersonaRuntime",
            side_effect=RuntimeError("server failed to load the model"),
        ) as llama_runtime,
        mock.patch.object(
            persona_runtime.backend,
            "load",
            side_effect=load_native,
        ) as native_load,
    ):
        persona_runtime.load(PERSONA_DEFAULT_MODEL_ID, "4bit")
        persona_runtime.load(PERSONA_DEFAULT_MODEL_ID, "4bit")

    llama_runtime.assert_called_once()
    native_load.assert_called_once_with(PERSONA_DEFAULT_MODEL_ID, "4bit")
    assert persona_runtime.effective_runtime == "native"
    assert persona_runtime.requested_runtime == "gguf"
    assert persona_runtime.inference_metadata() == {
        "model": PERSONA_DEFAULT_MODEL_ID,
        "type": "native",
        "quant": "bnb_4bit",
    }
    assert (
        sum("Persona GGUF unavailable" in message for message, _ in log_messages) == 1
    )
    assert any("using native inference" in message for message, _ in log_messages)


def test_unsupported_persona_model_uses_native_without_trying_gguf() -> None:
    """Use native inference directly for a model without a verified package."""
    persona_runtime = runtime.PersonaRuntime(
        config={"vram": "high", "inference_mode": "gguf"}
    )
    with (
        mock.patch("celune.persona.runtime.require_model_consent"),
        mock.patch("celune.persona.gguf.LlamaCppPersonaRuntime") as llama_runtime,
        mock.patch.object(persona_runtime.backend, "load") as native_load,
    ):
        persona_runtime.load("Qwen/Qwen3-VL-8B-Instruct", "4bit")

    llama_runtime.assert_not_called()
    native_load.assert_called_once_with("Qwen/Qwen3-VL-8B-Instruct", "4bit")
    assert persona_runtime.requested_runtime == "gguf"
    assert persona_runtime.effective_runtime == "native"


def test_consent_failure_does_not_trigger_native_fallback() -> None:
    """Surface model-consent decisions without retrying another runtime."""
    persona_runtime = runtime.PersonaRuntime(
        config={"vram": "high", "inference_mode": "gguf"}
    )
    metadata = get_model_metadata(PERSONA_DEFAULT_MODEL_ID)
    assert metadata is not None
    with (
        mock.patch(
            "celune.persona.runtime.require_model_consent",
            side_effect=ModelConsentRequired(metadata),
        ),
        mock.patch.object(persona_runtime.backend, "load") as native_load,
        pytest.raises(ModelConsentRequired),
    ):
        persona_runtime.load(PERSONA_DEFAULT_MODEL_ID, "4bit")

    native_load.assert_not_called()


def test_gguf_generation_failure_is_not_replayed_through_native() -> None:
    """Keep generation failures on the active runtime's existing error path."""
    persona_runtime = runtime.PersonaRuntime(
        config={"vram": "high", "inference_mode": "gguf"}
    )
    server = SimpleNamespace(
        model_id=PERSONA_DEFAULT_MODEL_ID,
        quantization="Q4_K_M",
        generate=mock.Mock(side_effect=RuntimeError("generation failed")),
        close=mock.Mock(),
        interrupt=mock.Mock(),
    )
    with (
        mock.patch("celune.persona.runtime.require_model_consent"),
        mock.patch(
            "celune.persona.gguf.download_persona_gguf",
            return_value=(Path("model.gguf"), Path("mmproj.gguf")),
        ),
        mock.patch(
            "celune.persona.gguf.ensure_llama_server",
            return_value=Path("llama-server"),
        ),
        mock.patch(
            "celune.persona.gguf.LlamaCppPersonaRuntime",
            return_value=server,
        ),
        mock.patch.object(persona_runtime.backend, "load") as native_load,
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        persona_runtime.generate(
            GenerateRequest(
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    server.generate.assert_called_once()
    native_load.assert_not_called()


def test_persona_gguf_rejects_video_attachments() -> None:
    """Reject video media rather than claiming unverified GGUF support."""
    candidate = gguf.LlamaCppPersonaRuntime.__new__(gguf.LlamaCppPersonaRuntime)
    request = GenerateRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[runtime.VideoContentItem(type="video", video="clip.mp4")],
            )
        ]
    )
    with pytest.raises(ValueError, match="does not support video"):
        candidate._messages(request)


def test_persona_gguf_reports_server_token_usage() -> None:
    """Preserve llama-server's actual prompt and completion token counts."""
    candidate = gguf.LlamaCppPersonaRuntime.__new__(gguf.LlamaCppPersonaRuntime)
    candidate.model_id = PERSONA_DEFAULT_MODEL_ID
    candidate.quantization = "Q4_K_M"
    candidate._cancellation_event = threading.Event()
    request = GenerateRequest(
        messages=[ChatMessage(role="user", content="hello")],
        max_new_tokens=24,
        temperature=0.4,
        top_p=0.8,
        repetition_penalty=1.2,
        context_space=1024,
    )
    with mock.patch.object(
        candidate,
        "_request_json",
        side_effect=[
            {"input_tokens": 5},
            {
                "choices": [{"message": {"content": "ready"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        ],
    ) as request_json:
        response = candidate.generate(request)

    assert response.text == "ready"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 2
    completion_payload = request_json.call_args_list[1].args[1]
    assert completion_payload["max_tokens"] == 24
    assert completion_payload["temperature"] == 0.4
    assert completion_payload["top_p"] == 0.8
    assert completion_payload["repeat_penalty"] == 1.2


def test_persona_gguf_cancellation_does_not_start_an_http_request() -> None:
    """Surface cancellation directly before opening a server request."""
    candidate = gguf.LlamaCppPersonaRuntime.__new__(gguf.LlamaCppPersonaRuntime)
    candidate._cancellation_event = threading.Event()
    candidate._cancellation_event.set()

    with pytest.raises(InterruptedError, match="cancelled"):
        candidate._request_json("/v1/chat/completions", {})


def test_persona_gguf_close_terminates_the_managed_server() -> None:
    """Stop a live llama-server process during Persona unload."""
    candidate = gguf.LlamaCppPersonaRuntime.__new__(gguf.LlamaCppPersonaRuntime)
    process = mock.Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    candidate._closed = False
    candidate._process = process
    candidate._log_thread = None

    candidate.close()

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    assert candidate._closed


def test_persona_close_stops_gguf_server_when_generation_holds_the_lock() -> None:
    """Do not leave llama-server running when shutdown cannot acquire Persona's lock."""
    persona_runtime = runtime.PersonaRuntime(config={"inference_mode": "gguf"})
    server = SimpleNamespace(close=mock.Mock())
    persona_runtime._gguf_runtime = cast(gguf.LlamaCppPersonaRuntime, server)
    persona_runtime.effective_runtime = "gguf"
    with persona_runtime.lock:
        persona_runtime.close()

    server.close.assert_called_once_with()
    assert persona_runtime._gguf_runtime is None
    assert persona_runtime.effective_runtime == "native"


def test_persona_gguf_archive_rejects_path_traversal(tmp_path: Path) -> None:
    """Reject archive entries that escape the llama.cpp runtime cache."""
    archive_path = tmp_path / "unsafe.zip"
    target = tmp_path / "extract"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(RuntimeError, match="unsafe path"):
        gguf._extract_runtime_archive(archive_path, target)

    assert not (tmp_path / "outside.txt").exists()


def test_persona_gguf_artifact_verification_checks_size_and_digest(
    tmp_path: Path,
) -> None:
    """Verify cached artifacts against the exact size and SHA-256 pin."""
    artifact = tmp_path / "model.gguf"
    contents = b"verified model"
    artifact.write_bytes(contents)
    digest = hashlib.sha256(contents).hexdigest()

    gguf._verify_artifact(artifact, len(contents), digest)
    with pytest.raises(RuntimeError, match="unexpected size"):
        gguf._verify_artifact(artifact, len(contents) + 1, digest)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        gguf._verify_artifact(artifact, len(contents), "0" * 64)
