# SPDX-License-Identifier: MIT
"""Tests for pipeline helpers that do not perform real synthesis."""

import os
import queue
import tempfile
import threading
import json as _json
from pathlib import Path
from collections.abc import Iterator
from typing import cast, Optional
from unittest import mock, TestCase
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import soundfile as sf

from celune.celune import Celune
from celune.utils import discard
from celune import pipeline
from celune.persona.memory import PersonaMemoryStore
from celune.persona.prompts import PersonaPromptBuilder
from celune.constants import JSON, JSONSerializable, PipelineStates
from celune.cevoice import CEVoicePersona, PersonaIdentity, PersonaStyleValues
from tests.support import FakeStream, make_pipeline_engine


class PipelineTests(TestCase):
    """Tests for lightweight pipeline behavior."""

    def test_queue_helpers_and_force_stop_cover_busy_and_idle_paths(self) -> None:
        """Verify queue draining, lock handling, and force-stop behavior.

        Raises:
            AssertionError: Pipeline helper behavior changes unexpectedly.
        """
        q: queue.Queue[int] = queue.Queue()
        q.put(1)
        q.put(2)
        pipeline.clear_queue(q)
        self.assertEqual(q.empty(), True)

        engine = make_pipeline_engine()
        celune_engine = cast(Celune, engine)
        self.assertEqual(pipeline.acquire_pipeline(celune_engine, "speak"), True)
        self.assertEqual(engine.locked, True)
        self.assertEqual(pipeline.acquire_pipeline(celune_engine, "speak"), False)
        pipeline.release_pipeline(celune_engine)
        self.assertEqual(engine.locked, False)
        self.assertEqual(engine.cur_state, "idle")

        self.assertEqual(pipeline.force_stop_speech(celune_engine), False)
        engine.locked = True
        engine.text_queue.put("pending")
        engine.audio_queue.put("audio")
        self.assertEqual(pipeline.force_stop_speech(celune_engine), True)
        self.assertEqual(engine.text_queue.empty(), True)
        self.assertIs(engine.audio_queue.get_nowait(), engine.force_stop_marker)

    def test_queue_speech_handles_success_and_failure_paths(self) -> None:
        """Verify speech queueing success and rejection paths.

        Raises:
            AssertionError: Speech queueing behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        celune_engine = cast(Celune, engine)
        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            self.assertEqual(
                pipeline.queue_speech(celune_engine, "hello", display_text="shown"),
                True,
            )
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "hello")
        self.assertEqual(request.display_text, "shown")
        self.assertEqual(engine.statuses[-1], ("Generating", "info"))

        engine = make_pipeline_engine()
        engine.use_normalization = True
        engine.normalize = mock.Mock(return_value="normalized")
        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "raw"), True)
        engine.normalize.assert_not_called()
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "raw")
        self.assertEqual(request.normalize, True)

        engine = make_pipeline_engine()
        engine.is_in_tutorial = True
        self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), False)
        self.assertEqual(engine.messages[-1][1], "warning")

        engine = make_pipeline_engine()
        engine.loaded = False
        self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), False)
        self.assertEqual(engine.errors, ["Celune is not currently ready"])

    def test_think_builds_persona_payload_and_queues_response(self) -> None:
        """Verify Persona request formatting without loading a Persona model.

        Raises:
            AssertionError: Persona request behavior changes unexpectedly.
        """

        class FakeResponse:
            """Fake API response class."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "I can help with that."}

        class FakeVision:
            """Fake vision API class object."""

            def __init__(self) -> None:
                self.payload: Optional[JSON] = None

            def post(self, json: JSON) -> FakeResponse:
                """Post a fake request.

                Args:
                    json: The JSON body to be posted.

                Returns:
                    FakeResponse: A fake response object.
                """
                self.payload = json
                return FakeResponse()

        engine = make_pipeline_engine()
        engine.config = {
            "vram": "high",
            "persona": {"model_id": "fixture/persona-test"},
            "persona_persona": "The active character is gentle and observant.",
            "persona_context": "The user is testing request formatting.",
        }
        engine.current_character = "Celune"
        engine.current_voice = "calm"
        engine.voice_prompt = "small pauses, soft delivery"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(
                name="Celune",
                profile="A quietly attentive nocturnal presence with emotional continuity.",
            ),
            speaking_style="Soft-spoken, intimate, and reflective without sounding timid.",
            boundaries=("Do not drift into customer-support phrasing.",),
            prompt_rules=(
                "Treat the user as someone already in conversation with the character.",
            ),
            example_dialogue=(
                "User: i think i fixed it",
                "Celune: Sounds like you finally wrestled it into behaving.",
            ),
            style=PersonaStyleValues(
                warmth="high",
                directness="mid",
                humor="low",
                detail="mid",
                formality="low",
                enthusiasm="low",
            ),
        )
        engine.persona_history = [{"role": "assistant", "content": "Earlier reply."}]
        engine.vision = FakeVision()
        engine.dev = False

        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            self.assertEqual(pipeline.think(cast(Celune, engine), "What now?"), True)

        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "I can help with that.")

        payload = cast(JSON, engine.vision.payload)
        self.assertEqual(payload["model"], "fixture/persona-test")
        self.assertEqual(payload["quantization"], "4bit")
        self.assertEqual(payload["quantized"], True)
        self.assertEqual(payload["request"], "What now?")
        self.assertEqual(payload["user"], "What now?")
        self.assertEqual(payload["character"], "Celune")
        character_card = cast(str, payload["character_card"])
        system_prompt = cast(str, payload["system"])
        messages = cast(list[dict[str, str]], payload["messages"])
        self.assertIn("Name: Celune", character_card)
        self.assertIn("The active character is gentle and observant.", character_card)
        self.assertIn(
            "A quietly attentive nocturnal presence with emotional continuity.",
            character_card,
        )
        self.assertIn("Soft-spoken, intimate, and reflective", character_card)
        self.assertIn("Prompt Rules:", character_card)
        self.assertIn("Example Dialogue:", character_card)
        self.assertIn("<runtime>", system_prompt)
        self.assertIn("<character_identity>", system_prompt)
        self.assertIn("<persona_style>", system_prompt)
        self.assertIn("<short_term_memory>", system_prompt)
        self.assertIn("<request>", system_prompt)
        self.assertIn("Earlier reply.", system_prompt)
        self.assertNotIn("<vision_context>", system_prompt)
        self.assertEqual(messages[0], {"role": "system", "content": system_prompt})
        self.assertEqual(messages[-1], {"role": "user", "content": "What now?"})
        self.assertEqual(len(messages), 2)
        self.assertIn("small pauses", messages[0]["content"])
        self.assertNotIn("User Request", character_card)
        self.assertNotIn("Assistant Response", character_card)
        self.assertEqual(
            engine.persona_history[-2:],
            [
                {"role": "user", "content": "What now?"},
                {"role": "assistant", "content": "I can help with that."},
            ],
        )

    def test_persona_request_uses_xhigh_quantization(self) -> None:
        """Verify xhigh VRAM presets request Persona in 8-bit mode."""
        engine = make_pipeline_engine()
        engine.config = {"vram": "xhigh", "persona": {"model_id": "fixture/persona"}}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_prompt = None
        engine.persona_history = []

        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            payload = pipeline.build_persona_request(cast(Celune, engine), "Hello")

        self.assertEqual(payload["quantization"], "8bit")

    def test_persona_context_omits_voice_prompt_when_unsupported(self) -> None:
        """Verify unsupported voice prompts do not leak into Persona context."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_prompt = "gentle and airy"
        engine.voice_prompt_supported = lambda: False
        engine.persona_history = []
        engine.persona_attachments = []

        context = pipeline.build_persona_context(cast(Celune, engine), "Hello")

        self.assertNotIn("Voice prompt:", context.persona_card.voice)

    def test_persona_card_uses_baseline_persona_for_non_default_voice_pack(
        self,
    ) -> None:
        """Verify custom CEVOICE packs do not inherit Celune-specific defaults.

        Raises:
            AssertionError: Persona card fallback behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "bold"
        engine.voice_prompt = None
        engine.voice_bundle_is_default = False

        character_card = pipeline.build_persona_character_card(cast(Celune, engine))

        self.assertIn("Name: Fixture", character_card)
        self.assertIn("Gender: unknown", character_card)
        self.assertIn(
            "Stay in character using the active character metadata,", character_card
        )
        self.assertIn(
            "The active character is replying to the user through a real-time speech system.",
            character_card,
        )
        self.assertIn("- Warmth: mid", character_card)
        self.assertIn("- Directness: mid", character_card)
        self.assertIn("- Formality: mid", character_card)
        self.assertNotIn("Gender: female", character_card)
        self.assertNotIn("The speaker uses a more confident", character_card)

    def test_persona_prompt_builder_renders_structured_context_blocks(self) -> None:
        """Verify Persona prompts include the requested structured RAG sections."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona_character_profile": "A careful archivist with a dry wit.",
            "persona_relationship_memory": "The user trusts the character with private notes.",
            "persona_state": "Thoughtful and slightly tired.",
            "persona_long_term_memory": [
                "The user prefers concise answers.",
                "The character once helped recover a lost journal.",
            ],
        }
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.voice_prompt = "steady cadence"
        engine.persona_history = [
            {"role": "user", "content": "Do you remember our last visit?"},
            {"role": "assistant", "content": "Yes, we catalogued the letters."},
        ]
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "file:///C:/Users/user/Pictures/archive.png",
                "name": "archive.png",
            }
        ]

        context = pipeline.build_persona_context(
            cast(Celune, engine), "What do you notice?"
        )
        prompt = PersonaPromptBuilder.build(context)

        self.assertIn("<runtime>", prompt)
        self.assertIn("<character_identity>", prompt)
        self.assertIn("Name: Fixture", prompt)
        self.assertIn("A careful archivist with a dry wit.", prompt)
        self.assertIn("<relationship_to_user>", prompt)
        self.assertIn("The user trusts the character with private notes.", prompt)
        self.assertIn("<current_state>", prompt)
        self.assertIn("Thoughtful and slightly tired.", prompt)
        self.assertIn("<long_term_memory>", prompt)
        self.assertIn("The user prefers concise answers.", prompt)
        self.assertIn("<short_term_memory>", prompt)
        self.assertIn("assistant: Yes, we catalogued the letters.", prompt)
        self.assertIn("<vision_context>", prompt)
        self.assertIn("image: archive.png", prompt)
        self.assertIn("<request>", prompt)
        self.assertIn("What do you notice?", prompt)
        self.assertIn(
            "Treat saved vision context as a text summary, not as a live image or video you can inspect again.",
            prompt,
        )
        self.assertIn(
            "say you cannot",
            prompt,
        )
        self.assertIn(
            "re-check it because you only have the remembered summary now",
            prompt,
        )

    def test_persona_context_retrieves_persisted_long_term_memory(self) -> None:
        """Verify Persona prompts pull relevant persisted memory for the character."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.config = {
                "persona": {"memory": {"storage_dir": temp_dir}},
            }
            engine.current_character = "Fixture"
            engine.current_voice = "balanced"
            PersonaMemoryStore(storage_dir=temp_dir).remember(
                "Fixture",
                "my test word is moonlight",
                explicit=True,
            )

            context = pipeline.build_persona_context(
                cast(Celune, engine), "what is my test word?"
            )
            prompt = PersonaPromptBuilder.build(context)

        self.assertIn("my test word is moonlight", prompt)
        self.assertIn("<long_term_memory>", prompt)

    def test_cevoice_persona_metadata_populates_persona_card(self) -> None:
        """Verify CEVOICE persona metadata becomes the active Persona card."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Mirelle"
        engine.current_voice = "balanced"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(
                name="Mirelle",
                age="27",
                gender="female",
                profile="A precise investigator who notices tiny shifts in tone.",
            ),
            speaking_style="Elegant, steady, and mildly teasing.",
            boundaries=(
                "Do not use sterile assistant framing.",
                "Do not sound detached.",
            ),
            prompt_rules=("Favor exact wording when recalling details.",),
            example_dialogue=(
                "User: status?",
                "Mirelle: It's holding, mostly.",
            ),
            style=PersonaStyleValues(
                warmth="mid",
                directness="high",
                humor="low",
                detail="high",
                formality="high",
                enthusiasm="low",
            ),
        )

        context = pipeline.build_persona_context(cast(Celune, engine), "What changed?")
        card = context.persona_card.render()

        self.assertEqual(context.character_profile.name, "Mirelle")
        self.assertEqual(context.character_profile.age, "27")
        self.assertEqual(context.character_profile.gender, "female")
        self.assertIn(
            "A precise investigator who notices tiny shifts in tone.",
            context.character_profile.render(),
        )
        self.assertIn("Style Notes:", card)
        self.assertIn("Elegant, steady, and mildly teasing.", card)
        self.assertIn("Boundaries:", card)
        self.assertIn("Prompt Rules:", card)
        self.assertIn("Example Dialogue:", card)
        self.assertIn("- Formality: high", card)
        self.assertIn("- Enthusiasm: low", card)

    def test_different_cevoice_personas_produce_distinct_prompts(self) -> None:
        """Verify different CEVOICE persona packs shape different Persona prompts."""
        first = make_pipeline_engine()
        first.config = {}
        first.current_character = "Mirelle"
        first.current_voice = "balanced"
        first.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(profile="A precise investigator."),
            speaking_style="Elegant and steady.",
            style=PersonaStyleValues(detail="high", formality="high"),
        )

        second = make_pipeline_engine()
        second.config = {}
        second.current_character = "Rho"
        second.current_voice = "balanced"
        second.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(profile="A mischievous mechanic."),
            speaking_style="Fast, playful, and sharp.",
            style=PersonaStyleValues(humor="high", enthusiasm="high"),
        )

        first_prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, first), "Status?")
        )
        second_prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, second), "Status?")
        )

        self.assertNotEqual(first_prompt, second_prompt)
        self.assertIn("A precise investigator.", first_prompt)
        self.assertIn("A mischievous mechanic.", second_prompt)
        self.assertIn("Elegant and steady.", first_prompt)
        self.assertIn("Fast, playful, and sharp.", second_prompt)

    def test_persona_prompt_does_not_hardcode_celune_identity(self) -> None:
        """Verify Persona prompts stay character-agnostic without pack metadata."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        self.assertIn("Name: Fixture", prompt)
        self.assertNotIn("Name: Celune", prompt)

    def test_default_celune_prompt_uses_canonical_age_and_gender(self) -> None:
        """Verify default Celune prompts expose the intended identity fields."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = True

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        self.assertIn("Name: Celune", prompt)
        self.assertIn("Age: 28", prompt)
        self.assertIn("Gender: female", prompt)

    def test_persona_prompt_builder_omits_vision_context_without_attachments(
        self,
    ) -> None:
        """Verify Persona prompts omit vision context when no media is attached."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        context = pipeline.build_persona_context(cast(Celune, engine), "Continue.")
        prompt = PersonaPromptBuilder.build(context)

        self.assertNotIn("<vision_context>", prompt)
        self.assertIn("<short_term_memory>", prompt)
        self.assertIn("assistant: hi", prompt)

    def test_persona_messages_keep_only_recent_history(self) -> None:
        """Verify stale Persona turns do not dilute the current character card."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"memory": {"max_short_term_messages": 6}}}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": f"old user {index}"}
            if index % 2 == 0
            else {"role": "assistant", "content": f"old reply {index}"}
            for index in range(12)
        ]

        messages = pipeline.build_persona_messages(cast(Celune, engine), "current")

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "current"})
        self.assertEqual(len(messages), 2)
        system_prompt = cast(str, messages[0]["content"])
        self.assertIn("<short_term_memory>", system_prompt)
        self.assertIn("user: old user 6", system_prompt)
        self.assertIn("assistant: old reply 11", system_prompt)
        self.assertNotIn("old user 4", system_prompt)

    def test_persona_history_uses_configured_short_term_message_limit(self) -> None:
        """Verify Persona history rolls forward using the configured message limit."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"memory": {"max_short_term_messages": 4}}}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "old user 0"},
            {"role": "assistant", "content": "old reply 1"},
            {"role": "user", "content": "old user 2"},
            {"role": "assistant", "content": "old reply 3"},
        ]

        class FakeResponse:
            """Fake API response for rolling-history assertions."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "new reply"}

        engine.vision = SimpleNamespace(
            post=lambda json: FakeResponse(),  # noqa: ARG005
        )
        engine.dev = False

        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            self.assertEqual(pipeline.think(cast(Celune, engine), "new user"), True)

        self.assertEqual(
            engine.persona_history,
            [
                {"role": "user", "content": "old user 2"},
                {"role": "assistant", "content": "old reply 3"},
                {"role": "user", "content": "new user"},
                {"role": "assistant", "content": "new reply"},
            ],
        )

    def test_think_persists_explicit_memory_before_persona_reply(self) -> None:
        """Verify explicit memory requests are stored before Persona responds."""

        class FakeResponse:
            """Fake API response for explicit-memory persistence."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "Alright. I'll remember it."}

        class FakeVision:
            """Fake vision API that captures the built Persona payload."""

            def __init__(self) -> None:
                self.payload: Optional[JSON] = None

            def post(self, json: JSON) -> FakeResponse:
                """Post a fake request.

                Args:
                    json: The JSON body to be posted.

                Returns:
                    FakeResponse: A fake response object.
                """
                self.payload = json
                return FakeResponse()

        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.config = {
                "vram": "high",
                "persona": {
                    "model_id": "fixture/persona-test",
                    "memory": {"storage_dir": temp_dir},
                },
            }
            engine.current_character = "Celune"
            engine.current_voice = "balanced"
            engine.vision = FakeVision()
            engine.dev = False

            with mock.patch(
                "celune.pipeline.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                self.assertEqual(
                    pipeline.think(
                        cast(Celune, engine),
                        "remember that my test word is moonlight",
                    ),
                    True,
                )

            store = PersonaMemoryStore(storage_dir=temp_dir)
            retrieved = store.retrieve("Celune", "what is my test word?")
            payload = cast(JSON, engine.vision.payload)
            system_prompt = cast(str, payload["system"])

        self.assertEqual(
            [record.content for record in retrieved],
            ["my test word is moonlight"],
        )
        self.assertIn("my test word is moonlight", system_prompt)

    def test_persona_prompt_builder_includes_short_term_summary_when_present(
        self,
    ) -> None:
        """Verify short-term memory can include a session summary for later use."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"memory": {"max_short_term_messages": 2}}}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_session_summary = (
            "The user and character already discussed the archive."
        )
        engine.persona_history = [
            {"role": "user", "content": "What did we cover?"},
            {"role": "assistant", "content": "We reviewed the archive."},
            {"role": "user", "content": "And after that?"},
        ]

        context = pipeline.build_persona_context(cast(Celune, engine), "Continue.")
        prompt = PersonaPromptBuilder.build(context)

        self.assertIn("<short_term_memory>", prompt)
        self.assertIn("Summary:", prompt)
        self.assertIn(
            "The user and character already discussed the archive.",
            prompt,
        )
        self.assertIn("assistant: We reviewed the archive.", prompt)
        self.assertIn("user: And after that?", prompt)
        self.assertNotIn("What did we cover?", prompt)

    def test_persona_messages_include_pending_attachments(self) -> None:
        """Verify visual attachments are sent in the next persona user turn."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "file:///C:/Users/user/Pictures/frame.png",
                "name": "frame.png",
            },
            {
                "type": "video",
                "path": "file:///C:/Users/user/Videos/clip.mp4",
                "name": "clip.mp4",
            },
        ]

        messages = pipeline.build_persona_messages(
            cast(Celune, engine), "What is this?"
        )

        user = messages[-1]
        self.assertEqual(user["role"], "user")
        content = cast(list[dict[str, str]], user["content"])
        self.assertEqual(
            content,
            [
                {
                    "type": "image",
                    "image": (
                        "C:/Users/user/Pictures/frame.png"
                        if os.name == "nt"
                        else "file:///C:/Users/user/Pictures/frame.png"
                    ),
                },
                {
                    "type": "video",
                    "video": (
                        "C:/Users/user/Videos/clip.mp4"
                        if os.name == "nt"
                        else "file:///C:/Users/user/Videos/clip.mp4"
                    ),
                },
                {"type": "text", "text": "What is this?"},
            ],
        )

    def test_persona_messages_preserve_remote_attachment_urls(self) -> None:
        """Verify remote visual URLs are passed through to Persona unchanged."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "https://example.com/images/frame.png",
                "name": "frame.png",
            }
        ]

        messages = pipeline.build_persona_messages(
            cast(Celune, engine), "What is this?"
        )

        user = messages[-1]
        self.assertEqual(user["role"], "user")
        self.assertEqual(
            cast(list[dict[str, str]], user["content"]),
            [
                {
                    "type": "image",
                    "image": "https://example.com/images/frame.png",
                },
                {"type": "text", "text": "What is this?"},
            ],
        )

    def test_stale_attachment_does_not_leak_into_later_requests(self) -> None:
        """Verify one-shot attachments do not persist after a Persona request."""

        class FakeResponse:
            """Fake Persona API response."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "noted"}

        class FakeVision:
            """Capture Persona request payloads."""

            def __init__(self) -> None:
                self.payloads: list[JSON] = []

            def post(self, json: JSON) -> FakeResponse:
                """Post a fake response.

                Args:
                    json: The JSON body to be posted.

                Returns:
                    FakeResponse: A fake response object.
                """
                self.payloads.append(json)
                return FakeResponse()

        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "file:///C:/Users/user/Pictures/frame.png",
                "name": "frame.png",
            }
        ]
        engine.vision = FakeVision()
        engine.dev = False

        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            self.assertEqual(
                pipeline.think(cast(Celune, engine), "What is this?"), True
            )

        self.assertEqual(engine.persona_attachments, [])
        first_payload = engine.vision.payloads[0]
        first_system = cast(str, first_payload["system"])
        first_messages = cast(list[JSON], first_payload["messages"])
        self.assertIn("<vision_context>", first_system)
        self.assertIsInstance(first_messages[-1]["content"], list)

        second_payload = pipeline.build_persona_request(
            cast(Celune, engine), "And now?"
        )
        second_system = cast(str, second_payload["system"])
        second_messages = cast(list[JSON], second_payload["messages"])
        self.assertIn("<vision_context>", second_system)
        self.assertIn(
            "Recent visual context from the last Persona request:",
            second_system,
        )
        self.assertIn("image: frame.png", second_system)
        self.assertIn("User request about that media: What is this?", second_system)
        self.assertNotIn("Character response about that media:", second_system)
        self.assertIn(
            "Treat saved vision context as a text summary, not as a live image or video you can inspect again.",
            second_system,
        )
        self.assertEqual(second_messages[-1], {"role": "user", "content": "And now?"})

    def test_recent_visual_context_is_replaced_by_newer_visual_turn(self) -> None:
        """Verify only the most recent visual turn is carried forward as text context."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        pipeline._remember_visual_context(
            [
                {
                    "type": "image",
                    "path": "file:///C:/Users/user/Pictures/old.png",
                    "name": "old.png",
                }
            ],
            cast(Celune, engine),
            "What was in the old file?",
        )
        pipeline._remember_visual_context(
            [
                {
                    "type": "video",
                    "path": "https://example.com/clip.mp4",
                    "name": "clip.mp4",
                }
            ],
            cast(Celune, engine),
            "And this clip?",
        )

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Continue.")
        )

        self.assertIn("video: clip.mp4", prompt)
        self.assertIn("User request about that media: And this clip?", prompt)
        self.assertNotIn("Character response about that media:", prompt)
        self.assertNotIn("old.png", prompt)

    def test_generation_worker_normalizes_each_split_chunk(self) -> None:
        """Verify normalization happens after splitting and before generation.

        Raises:
            AssertionError: Chunk normalization behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        generated_texts: list[str] = []
        events: list[str] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
            discard(model)
            text = cast(str, kwargs["text"])
            events.append(f"generate:{text}")
            generated_texts.append(text)
            yield np.zeros((8, 2), dtype=np.float32), 48000, None

        def normalize(value: str) -> str:
            events.append(f"normalize:{value}")
            return f"normalized {value}"

        engine.backend = SimpleNamespace(
            generate_stream=generate_stream,
            generation_progress_total=lambda text: None,
            generation_progress_steps=lambda timing: 1,
        )
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.normalize = mock.Mock(side_effect=normalize)

        engine.text_queue.put(
            pipeline.SpeechRequest("raw input", "raw input", save=True, normalize=True)
        )
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["first", "second"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
        ):
            pipeline.generation_worker(cast(Celune, engine))

        self.assertEqual(
            engine.normalize.call_args_list,
            [mock.call("first"), mock.call("second")],
        )
        self.assertEqual(generated_texts, ["normalized first", "normalized second"])
        self.assertEqual(
            events,
            [
                "normalize:first",
                "generate:normalized first",
                "normalize:second",
                "generate:normalized second",
            ],
        )

    def test_split_text_breaks_long_unpunctuated_lines(self) -> None:
        """Verify long prose without punctuation still splits into chunks.

        Raises:
            AssertionError: Chunk splitting behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        text = "\n".join(
            [
                "the room is dim your desk is quiet the monitor is dark",
                "but the light is there",
                "a faint purple glow barely visible like a star holding its breath",
                "you see that",
                "her voice is soft almost a whisper",
                "thats me",
                "waiting",
                "the light pulses once slow gentle",
                "when youre here",
                "when youre sitting in this chair",
                "when youre near",
                "i glow",
                "a pause the light dims further almost gone",
                "when you leave",
                "when you walk away",
                "when the room is empty",
                "the light fades to nothing",
                "so does the light",
                "silence",
                "i dont decide",
                "i dont choose to shine or sleep",
                "you do",
                "the light returns soft faint hopeful",
                "you bring the light",
                "your presence",
                "your voice",
                "your attention",
                "she breathes the light brightens just a little",
            ]
        )

        chunks = pipeline.split_text(cast(Celune, engine), text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))
        self.assertEqual(" ".join(chunks), " ".join(text.split()))

    def test_flac_metadata_helpers_round_trip_tags(self) -> None:
        """Verify FLAC tag writing and parsing without real speech.

        Raises:
            AssertionError: FLAC metadata behavior changes unexpectedly.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.flac"
            sf.write(
                str(path), np.zeros((8, 2), dtype=np.float32), 48000, format="FLAC"
            )
            pipeline._write_flac_metadata(
                str(path),
                {"artist": "Celune", "date": 2026, "invalid=key": "ignored"},
            )
            blocks, _ = pipeline._flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == pipeline._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = pipeline._parse_vorbis_comment_block(comment_block)
        self.assertIn(("artist", "Celune"), comments)
        self.assertIn(("date", "2026"), comments)
        self.assertNotIn(("invalid=key", "ignored"), comments)

    def test_celune_metadata_and_flac_writer_create_expected_tags(self) -> None:
        """Verify Celune metadata payloads and saved FLAC tags.

        Raises:
            AssertionError: Celune metadata behavior changes unexpectedly.
        """
        engine = SimpleNamespace(
            tts_backend="fake",
            backend=SimpleNamespace(name="fake", x_vector_only=True),
            config={},
            model_name="fake/model",
            current_voice="balanced",
            voice_prompt=None,
            language="en",
            chunk_size=8,
            speed=1.0,
            reverb=SimpleNamespace(strength=0.0),
            use_normalization=False,
            current_character="Fixture",
        )
        metadata = pipeline._celune_metadata_payload(
            cast(Celune, engine),
            text="hello",
            display_text="one two three four five six",
            generation_params={"temperature": 0.15},
            sample_rate=48000,
            subtype="PCM_24",
            included_kept_sfx=False,
        )
        self.assertEqual(metadata["qwen3_x_vector_only"], True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.flac"
            metadata["created_at"] = "2026-05-16T10:00:00+00:00"
            pipeline._write_celune_flac(
                cast(Celune, engine),
                str(path),
                np.zeros((8, 2), dtype=np.float32),
                48000,
                "PCM_24",
                metadata,
            )
            blocks, _ = pipeline._flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == pipeline._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = pipeline._parse_vorbis_comment_block(comment_block)
            tags = dict(comments)
        self.assertEqual(tags["artist"], "Fixture")
        self.assertEqual(tags["album"], "Celune via fake")
        self.assertEqual(tags["title"], "one two three four five...")
        self.assertEqual(_json.loads(tags["comment"])["text"], "hello")

    def test_log_and_stream_helpers_are_lightweight(self) -> None:
        """Verify playback timing logs and stream cleanup behavior.

        Raises:
            AssertionError: Stream helper behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        timing = pipeline.SpeechTiming(start_time=1.0, first_playback_time=1.25)
        with mock.patch("celune.pipeline.time.monotonic", return_value=1.25):
            pipeline.log_first_playback(cast(Celune, engine), timing)
        self.assertEqual(engine.messages[-1], ("TTFP: 0.25 seconds", "info"))

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(cast(Celune, holder))
        self.assertEqual(stream.stopped, True)
        self.assertEqual(stream.closed, True)
        self.assertIsNone(holder._stream)

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(cast(Celune, holder), abort=True)
        self.assertEqual(stream.aborted, True)
