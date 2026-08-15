# SPDX-License-Identifier: MIT
"""Tests for the shared Persona runtime helpers."""

import contextlib
from types import SimpleNamespace
from typing import Optional, Union, cast
from unittest import TestCase, mock

from celune.constants import (
    PERSONA_DEFAULT_MODEL_ID,
    PERSONA_MODELS,
    persona_model_tier,
    remote_code_model_revision,
)
from celune.i18n import string
from celune.persona import impl, runtime
from celune.typing.aliases import RecordedKwargValue
from celune.typing.common import JSON
from celune.utils import discard


class _FakeEncoded:
    """Minimal encoded-input object supporting ``.to()``."""

    def __init__(self) -> None:
        self.device = None

    def to(self, device: str) -> "_FakeEncoded":
        """Record the requested device and return ``self``.

        Args:
            device: The device to load to.

        Returns:
            _FakeEncoded: A fake encoded class object.
        """
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
        """Return a fixed decoded response string.

        Args:
            token_ids: The token IDs to decode.
            skip_special_tokens: Whether to skip special tokens while decoding.

        Returns:
            str: The fake decoded tokens.
        """
        discard(token_ids)
        discard(skip_special_tokens)
        return "decoded"


class _FakeModel:
    """Minimal model fake satisfying the Persona model protocol."""

    def __init__(self) -> None:
        self.device = "cpu"

    @staticmethod
    def generate(**kwargs):
        """Unused generate stub for protocol compatibility.

        Args:
            kwargs: Keyword arguments to use while generating.

        Raises:
            NotImplementedError: The fake generation method was called outside of Celune's test harness.
        """
        discard(kwargs)
        raise NotImplementedError("test fake does not generate")

    @staticmethod
    def eval() -> None:
        """Unused eval stub for protocol compatibility."""


class _FakeGenerativeModel:
    """Minimal model fake that returns one short generated sequence."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.calls: list[dict[str, RecordedKwargValue]] = []

    def generate(self, **kwargs) -> runtime.torch.Tensor:
        """Record generation kwargs and return one synthetic completion.

        Args:
            kwargs: Keyword arguments to use while generating.

        Returns:
            runtime.torch.Tensor: A fake PyTorch tensor.
        """
        self.calls.append(dict(kwargs))
        return runtime.torch.tensor([[1, 2, 3]], dtype=runtime.torch.long)

    @staticmethod
    def eval() -> None:
        """Unused eval stub for protocol compatibility."""


class _FakeProcessor:
    """Minimal processor fake exposing a chat template hook."""

    def __init__(self, tokenizer: Optional[_FakeTokenizer] = None) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def apply_chat_template(*args, **kwargs) -> str:
        """Return a fixed prompt rendering for load-time support checks.

        Args:
            args: Arguments to use while generating.
            kwargs: Keyword arguments to use while generating.

        Returns:
            str: A fake formatted chat template.
        """
        discard(args)
        discard(kwargs)
        return "prompt"


class _FakeMultimodalProcessor:
    """Minimal multimodal processor fake without native chat rendering."""

    def __init__(self, tokenizer: Optional[_FakeTokenizer] = None) -> None:
        self.tokenizer = tokenizer
        self.image_processor = SimpleNamespace()
        self.calls: list[dict[str, RecordedKwargValue]] = []

    def __call__(self, **kwargs) -> _FakeEncoded:
        """Return one encoded batch for multimodal processor calls."""
        self.calls.append(dict(kwargs))
        return _FakeEncoded()


class _FakeQwenVlConfig:
    """Minimal config fake exposing the expected Qwen VL model type."""

    model_type = "qwen3_vl"


class PersonaApiTests(TestCase):
    """Tests for shared Persona runtime behavior."""

    @staticmethod
    def _text_message(role: runtime.Role, text: str) -> runtime.ChatMessagePayload:
        """Build one typed text-only Persona message payload."""
        return runtime.ChatMessagePayload(role=role, content=text)

    @staticmethod
    def _content_message(
        role: runtime.Role,
        content: list[runtime.ContentItem],
    ) -> runtime.ChatMessagePayload:
        """Build one typed multimodal Persona message payload."""
        return runtime.ChatMessagePayload(role=role, content=content)

    @contextlib.contextmanager
    def _mock_qwen_vl_load(
        self,
        *,
        processor: Optional[Union[_FakeProcessor, _FakeMultimodalProcessor]],
        model: Union[_FakeModel, _FakeGenerativeModel, mock.Mock],
        tokenizer: Optional[_FakeTokenizer] = None,
        processor_side_effect: Optional[Exception] = None,
    ):
        """Patch the shared Qwen VL loader entrypoints for one test."""
        with contextlib.ExitStack() as stack:
            config_loader = stack.enter_context(
                mock.patch(
                    "celune.persona.runtime.AutoConfig.from_pretrained",
                    return_value=_FakeQwenVlConfig(),
                )
            )
            processor_loader = stack.enter_context(
                mock.patch(
                    "celune.persona.runtime.AutoProcessor.from_pretrained",
                    return_value=processor,
                    side_effect=processor_side_effect,
                )
            )
            model_loader = stack.enter_context(
                mock.patch(
                    "celune.persona.runtime.Qwen3VLForConditionalGeneration.from_pretrained",
                    return_value=model,
                )
            )
            tokenizer_loader = stack.enter_context(
                mock.patch(
                    "celune.persona.runtime.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                )
            )
            yield {
                "config_loader": config_loader,
                "processor_loader": processor_loader,
                "model_loader": model_loader,
                "tokenizer_loader": tokenizer_loader,
            }

    def test_runtime_clamps_quantization_and_rejects_disabled_persona(self) -> None:
        """Verify VRAM presets constrain direct Persona runtime usage."""
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            runtime_xhigh = runtime.PersonaRuntime(config={"vram": "xhigh"})
            runtime_low = runtime.PersonaRuntime(config={"vram": "low"})

            with mock.patch.object(runtime_xhigh.backend, "load") as load_backend:
                runtime_xhigh.load("fixture/model", "4bit")

            load_backend.assert_called_once_with("fixture/model", "8bit")

            with self.assertRaisesRegex(
                ValueError, "not available for VRAM tier 'low'"
            ):
                runtime_low.load("fixture/model", "8bit")

    def test_messages_have_vision_only_for_explicit_media_items(self) -> None:
        """Verify visual mode is enabled only for explicit media attachments."""
        self.assertEqual(
            runtime.messages_have_vision([self._text_message("user", "hello")]),
            False,
        )
        self.assertEqual(
            runtime.messages_have_vision(
                [
                    self._content_message(
                        "user",
                        [runtime.TextContentItem(type="text", text="hello")],
                    )
                ]
            ),
            False,
        )
        self.assertEqual(
            runtime.messages_have_vision(
                [
                    self._content_message(
                        "user",
                        [
                            runtime.ImageContentItem(
                                type="image",
                                image="file:///frame.png",
                            )
                        ],
                    )
                ]
            ),
            True,
        )
        self.assertEqual(
            runtime.messages_have_vision(
                [
                    self._content_message(
                        "user",
                        [
                            runtime.VideoContentItem(
                                type="video",
                                video="file:///clip.mp4",
                            )
                        ],
                    )
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

        messages = [self._text_message("user", "hello")]
        with mock.patch(
            "celune.persona.runtime._render_chat_prompt",
            return_value="User: hello\n\nAssistant:",
        ) as render:
            encoded = backend.build_inputs(messages)

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
        revision = cast(str, remote_code_model_revision(PERSONA_DEFAULT_MODEL_ID))

        with self._mock_qwen_vl_load(
            processor=fake_processor,
            model=fake_model,
            tokenizer=fake_tokenizer,
        ) as loaders:
            backend.load("Qwen/Qwen3-VL-4B-Instruct", "none")

        loaders["config_loader"].assert_called_once_with(
            "Qwen/Qwen3-VL-4B-Instruct",
            trust_remote_code=True,
            revision=revision,
        )
        loaders["processor_loader"].assert_called_once_with(
            "Qwen/Qwen3-VL-4B-Instruct",
            trust_remote_code=True,
            revision=revision,
        )
        loaders["model_loader"].assert_called_once()
        self.assertEqual(
            loaders["model_loader"].call_args.args,
            ("Qwen/Qwen3-VL-4B-Instruct",),
        )
        self.assertEqual(
            loaders["model_loader"].call_args.kwargs,
            {
                "trust_remote_code": True,
                "revision": revision,
                "device_map": "auto",
                "dtype": runtime.torch.bfloat16,
            },
        )
        loaders["tokenizer_loader"].assert_not_called()
        fake_model.eval.assert_called_once_with()
        self.assertIs(backend.processor, fake_processor)
        self.assertIs(backend.tokenizer, fake_tokenizer)
        self.assertTrue(backend.supports_vision)

    def test_trusted_model_revision_allows_abliterated_qwen_vl_model(self) -> None:
        """Verify the pinned abliterated Qwen VL derivative is trusted."""
        model_id = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
        revision = cast(str, remote_code_model_revision(model_id))
        self.assertEqual(
            runtime.PersonaBackend._resolve_model_revision(model_id),
            revision,
        )

    def test_persona_registry_derives_supported_variants_and_tiers(self) -> None:
        """Verify each compact registry entry expands to both trusted model IDs."""
        self.assertEqual(
            tuple(
                (definition["model"], definition["organization"], definition["tier"])
                for definition in PERSONA_MODELS
            ),
            (
                ("Qwen3-VL-4B-Instruct", "Qwen", "standard"),
                ("Qwen3-VL-8B-Instruct", "Qwen", "smart"),
                ("Qwen3-VL-8B-Thinking", "Qwen", "smart"),
                ("Qwen3.5-4B", "Qwen", "standard"),
                ("Qwen3.5-9B", "Qwen", "smart"),
                ("gemma-4-E2B-it", "google", "standard"),
                ("gemma-4-E4B-it", "google", "smart"),
            ),
        )

        for definition in PERSONA_MODELS:
            official_id = f"{definition['organization']}/{definition['model']}"
            abliterated_id = f"huihui-ai/Huihui-{definition['model']}-abliterated"
            self.assertEqual(
                remote_code_model_revision(official_id),
                definition["revisions"]["official"],
            )
            self.assertEqual(
                remote_code_model_revision(abliterated_id),
                definition["revisions"]["abliterated"],
            )
            self.assertEqual(persona_model_tier(official_id), definition["tier"])
            self.assertEqual(persona_model_tier(abliterated_id), definition["tier"])

    def test_load_warns_for_unknown_remote_code_models(self) -> None:
        """Verify Persona warns before loading an unknown model."""
        backend = runtime.PersonaBackend()
        with (
            self._mock_qwen_vl_load(
                processor=_FakeProcessor(tokenizer=_FakeTokenizer()),
                model=mock.Mock(),
                tokenizer=_FakeTokenizer(),
            ) as loaders,
            mock.patch.object(runtime._LOGGER, "warning") as warning,
        ):
            backend.load("untrusted/example", "none")

        warning.assert_called_once_with(
            string("persona.unsupported_model", model_id="untrusted/example")
        )
        loaders["config_loader"].assert_called_once_with(
            "untrusted/example",
            trust_remote_code=True,
            revision="main",
        )

    def test_load_treats_multimodal_processor_as_vision_capable(self) -> None:
        """Verify multimodal processors are accepted even without chat templates."""
        backend = runtime.PersonaBackend()
        fake_model = mock.Mock()
        fake_model.eval.return_value = None
        fake_tokenizer = _FakeTokenizer()
        fake_processor = _FakeMultimodalProcessor(tokenizer=fake_tokenizer)

        with self._mock_qwen_vl_load(
            processor=fake_processor,
            model=fake_model,
            tokenizer=fake_tokenizer,
        ) as loaders:
            backend.load("Qwen/Qwen3-VL-4B-Instruct", "none")

        loaders["tokenizer_loader"].assert_not_called()
        self.assertIs(backend.processor, fake_processor)
        self.assertTrue(backend.supports_vision)

    def test_load_raises_when_processor_fails_for_vlm_model(self) -> None:
        """Verify Persona does not silently downgrade a VLM to tokenizer-only mode."""
        backend = runtime.PersonaBackend()
        fake_model = mock.Mock()
        fake_model.eval.return_value = None

        with (
            self._mock_qwen_vl_load(
                processor=None,
                model=fake_model,
                tokenizer=None,
                processor_side_effect=RuntimeError("processor boom"),
            ),
            self.assertRaisesRegex(
                ValueError,
                "Persona processor failed to load for model 'Qwen/Qwen3-VL-4B-Instruct'",
            ) as exc_info,
        ):
            backend.load("Qwen/Qwen3-VL-4B-Instruct", "none")

        self.assertIsInstance(exc_info.exception.__cause__, RuntimeError)

    def test_vision_inputs_use_qwen_vl_utils_when_processor_is_loaded(self) -> None:
        """Verify image requests go through qwen-vl-utils without capability gating."""
        backend = runtime.PersonaBackend()
        fake_processor = _FakeMultimodalProcessor(tokenizer=_FakeTokenizer())
        backend.processor = cast(runtime.PersonaProcessor, fake_processor)
        backend.tokenizer = cast(runtime.PersonaTokenizer, fake_processor.tokenizer)
        backend.model = cast(runtime.PersonaModel, _FakeModel())
        backend.supports_vision = False
        messages = [
            self._content_message(
                "user",
                [runtime.ImageContentItem(type="image", image="file:///frame.png")],
            )
        ]

        captured_kwargs: dict[str, bool] = {}

        def _fake_process_vision_info(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return [b"image"], None, {}

        fake_qwen_vl_utils = SimpleNamespace(
            process_vision_info=_fake_process_vision_info
        )
        with (
            mock.patch.dict("sys.modules", {"qwen_vl_utils": fake_qwen_vl_utils}),
            mock.patch(
                "celune.persona.runtime._render_chat_prompt",
                return_value="User: [image]\n\nAssistant:",
            ),
        ):
            encoded = backend.build_inputs(messages)

        self.assertIsInstance(encoded, _FakeEncoded)
        self.assertEqual(encoded.device, "cpu")
        self.assertEqual(len(fake_processor.calls), 1)
        self.assertEqual(fake_processor.calls[0]["text"], "User: [image]\n\nAssistant:")
        self.assertEqual(fake_processor.calls[0]["images"], [b"image"])
        self.assertEqual(
            captured_kwargs,
            {
                "return_video_kwargs": True,
                "return_video_metadata": True,
            },
        )
        self.assertNotIn("do_resize", fake_processor.calls[0])

    def test_generate_releases_transient_vram_after_vision_turn(self) -> None:
        """Verify vision requests drop temporary GPU allocations after generation."""
        backend = runtime.PersonaBackend()
        backend.processor = cast(runtime.PersonaProcessor, _FakeMultimodalProcessor())
        backend.tokenizer = cast(runtime.PersonaTokenizer, _FakeTokenizer())
        backend.model = cast(runtime.PersonaModel, _FakeGenerativeModel())
        backend.model_id = "fixture/model"
        backend.quantization = "4bit"
        backend.supports_vision = True
        request = runtime.GenerateRequest(
            messages=[
                runtime.ChatMessage(
                    "user",
                    [runtime.ImageContentItem(type="image", image="file:///frame.png")],
                )
            ]
        )

        with (
            mock.patch.object(
                backend,
                "_build_inputs",
                return_value={"input_ids": runtime.torch.tensor([[1, 2]])},
            ),
            mock.patch("celune.persona.runtime.gc.collect") as collect,
            mock.patch(
                "celune.persona.runtime.torch.cuda.is_available", return_value=True
            ),
            mock.patch("celune.persona.runtime.torch.cuda.synchronize") as sync,
            mock.patch("celune.persona.runtime.torch.cuda.empty_cache") as empty_cache,
        ):
            response = backend.generate(request)

        self.assertEqual(response.text, "decoded")
        collect.assert_called_once_with()
        sync.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_persona_client_routes_backend_output_to_verbose_logs(self) -> None:
        """Verify Persona backend stdout/stderr is captured into verbose logs."""
        logs: list[tuple[str, str]] = []
        client = impl.PersonaClient(
            log=lambda msg, severity="info", **kwargs: logs.append((msg, severity))
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

    def test_persona_client_summarizes_without_character_prompt(self) -> None:
        """Verify conversation summaries use a neutral VLM request contract."""
        client = impl.PersonaClient(config={"persona": {"model_id": "fixture/model"}})
        response = mock.Mock()
        response.json.return_value = {"response": "The user reported a TTS cutoff."}

        with mock.patch.object(client, "post", return_value=response) as post:
            summary = client.summarize_history(
                [{"role": "user", "content": "The TTS cut off."}],
                "The conversation concerns speech output.",
                300,
            )

        payload = cast(JSON, post.call_args.args[0])
        system = cast(str, payload["system"])
        user = cast(str, payload["user"])
        self.assertEqual(summary, "The user reported a TTS cutoff.")
        self.assertIn("neutral conversation summarizer", system)
        self.assertNotIn("CEVOICE", system)
        self.assertNotIn("active character", system)
        self.assertIn("Existing summary:", user)
        self.assertIn("Conversation turns:", user)
