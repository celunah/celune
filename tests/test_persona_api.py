# SPDX-License-Identifier: MIT
"""Tests for the shared Persona runtime helpers."""

from typing import cast
from unittest import TestCase, mock

from celune.persona import impl
from celune.persona import runtime
from celune.utils import discard


class _FakeEncoded:
    """Minimal encoded-input object supporting ``.to()``."""

    def __init__(self) -> None:
        self.device = None

    def to(self, device: str) -> "_FakeEncoded":
        """Record the requested device and return ``self``."""
        self.device = device
        return self


class _FakeTokenizer:
    """Minimal tokenizer fake for Persona backend tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.eos_token_id = 7

    def __call__(self, *, text: str, return_tensors: str) -> _FakeEncoded:
        """Record text-only encoding requests."""
        self.calls.append((text, return_tensors))
        return _FakeEncoded()

    @staticmethod
    def decode(token_ids, skip_special_tokens: bool) -> str:
        """Return a fixed decoded response string."""
        discard(token_ids)
        discard(skip_special_tokens)
        return "decoded"


class _FakeModel:
    """Minimal model fake satisfying the Persona model protocol."""

    def __init__(self) -> None:
        self.device = "cpu"

    @staticmethod
    def generate(**kwargs):
        """Unused generate stub for protocol compatibility."""
        discard(kwargs)
        raise NotImplementedError("test fake does not generate")

    @staticmethod
    def eval() -> None:
        """Unused eval stub for protocol compatibility."""


class _FakeProcessor:
    """Minimal processor fake exposing a chat template hook."""

    def __init__(self, tokenizer: _FakeTokenizer | None = None) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def apply_chat_template(*args, **kwargs) -> str:
        """Return a fixed prompt rendering for load-time support checks."""
        discard(args)
        discard(kwargs)
        return "prompt"


class PersonaApiTests(TestCase):
    """Tests for shared Persona runtime behavior."""

    def test_messages_have_vision_only_for_explicit_media_items(self) -> None:
        """Verify visual mode is enabled only for explicit media attachments."""
        self.assertEqual(
            runtime._messages_have_vision([{"role": "user", "content": "hello"}]),
            False,
        )
        self.assertEqual(
            runtime._messages_have_vision(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            ),
            False,
        )
        self.assertEqual(
            runtime._messages_have_vision(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": "file:///frame.png"}],
                    }
                ]
            ),
            True,
        )
        self.assertEqual(
            runtime._messages_have_vision(
                [
                    {
                        "role": "user",
                        "content": [{"type": "video", "video": "file:///clip.mp4"}],
                    }
                ]
            ),
            True,
        )

    def test_text_only_inputs_do_not_enter_vision_processing(self) -> None:
        """Verify no-attachment requests stay on the text-only backend path."""
        backend = runtime.PersonaBackend()
        backend.processor = None
        fake_tokenizer = _FakeTokenizer()
        backend.tokenizer = cast(runtime.PersonaTokenizer, fake_tokenizer)
        backend.model = cast(runtime.PersonaModel, _FakeModel())
        backend.supports_vision = True

        messages = cast(
            list[runtime.JSONDict],
            [{"role": "user", "content": "hello"}],
        )
        with mock.patch(
            "celune.persona.runtime._render_chat_prompt",
            return_value="User: hello\n\nAssistant:",
        ) as render:
            encoded = backend._build_inputs(messages)

        self.assertIsInstance(encoded, _FakeEncoded)
        self.assertEqual(encoded.device, "cpu")
        self.assertEqual(
            fake_tokenizer.calls,
            [("User: hello\n\nAssistant:", "pt")],
        )
        render.assert_called_once_with(fake_tokenizer, messages)

    def test_load_uses_causal_lm_for_qwen_vl_backend(self) -> None:
        """Verify Persona loading uses the causal-LM entrypoint."""
        backend = runtime.PersonaBackend()
        fake_model = mock.Mock()
        fake_model.eval.return_value = None
        fake_tokenizer = _FakeTokenizer()
        fake_processor = _FakeProcessor(tokenizer=fake_tokenizer)

        with (
            mock.patch(
                "celune.persona.runtime.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ) as processor_loader,
            mock.patch(
                "celune.persona.runtime.Qwen2_5_VLForConditionalGeneration.from_pretrained",
                return_value=fake_model,
            ) as model_loader,
            mock.patch(
                "celune.persona.runtime.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ) as tokenizer_loader,
        ):
            backend.load("Qwen/Qwen2.5-VL-3B-Instruct", "none")

        processor_loader.assert_called_once_with(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
        )
        model_loader.assert_called_once()
        self.assertEqual(
            model_loader.call_args.args,
            ("Qwen/Qwen2.5-VL-3B-Instruct",),
        )
        self.assertEqual(
            model_loader.call_args.kwargs,
            {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": runtime.torch.bfloat16,
            },
        )
        tokenizer_loader.assert_not_called()
        fake_model.eval.assert_called_once_with()
        self.assertIs(backend.processor, fake_processor)
        self.assertIs(backend.tokenizer, fake_tokenizer)
        self.assertTrue(backend.supports_vision)

    def test_persona_client_routes_backend_output_to_dev_logs(self) -> None:
        """Verify Persona backend stdout/stderr is captured into developer logs."""
        logs: list[tuple[str, str]] = []
        client = impl.PersonaClient(
            log_dev=lambda msg, severity="info": logs.append((msg, severity))
        )

        with mock.patch.object(
            client.runtime,
            "load",
            side_effect=lambda model_id, quantization: (
                print(f"loading {model_id}"),
                print(f"warn {quantization}", file=__import__("sys").stderr),
            ),
        ):
            client.load("fixture/model", "4bit")

        self.assertEqual(
            logs,
            [
                ("[PERSONA] warn 4bit", "warning"),
            ],
        )
