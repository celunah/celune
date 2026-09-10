# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline helpers that do not perform real synthesis."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import json as _json
import tempfile
import threading
from types import SimpleNamespace
from typing import Optional, cast
from pathlib import Path
from unittest import mock
from collections.abc import Iterator

import numpy as np
import pytest

from celune import pipeline
from celune.i18n import string
from celune.utils import discard
from celune.celune import Celune
from celune.cevoice import (
    CEVoice,
    CEVoiceLoader,
    CEVoicePersona,
    PersonaIdentity,
    PersonaStyleValues,
    persona_files_from_bundle,
)
from celune.constants import PipelineStates
from celune.persona.impl import compact_persona_history
from celune.typing.agent import (
    ToolCall,
    AgentTask,
    AgentContext,
    AgentRequest,
    AgentToolSchema,
    AgentToolBehavior,
    AgentToolValueType,
    AgentToolDangerLevel,
    AgentToolArgumentSchema,
)
from celune.typing.common import JSON, JSONSerializable
from celune.typing.aliases import AudioChunk
from celune.persona.prompts import PersonaPromptBuilder, render_markdown_subsection
from celune.persona.capabilities import PersonaCapabilities

from .support import (
    make_pipeline_engine,
)
from .platform import LINUX_ONLY, WINDOWS_ONLY
from .test_persona_memory import StubEmbeddingMemoryStore
from .pipeline_basics import TestPipelineAsync as _TestPipelineAsync


@pytest.mark.anyio
class TestPipelineAsync(_TestPipelineAsync):
    """Exercise persona-aware pipeline behavior."""

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

        assert "Voice prompt:" not in context.persona_card.voice

    def test_persona_card_uses_baseline_persona_for_non_default_voice_pack(
        self,
    ) -> None:
        """Verify custom CEVOICE/CECHAR packs do not inherit Celune-specific defaults.

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

        assert "Name: Fixture" in character_card
        assert "Gender: unknown" in character_card
        assert (
            "Stay in character using the active character metadata," in character_card
        )
        assert (
            "The active character is replying to the user through a real-time speech system."
            in character_card
        )
        assert "- Warmth: mid" in character_card
        assert "- Directness: mid" in character_card
        assert "- Formality: mid" in character_card
        assert "Gender: female" not in character_card
        assert "The speaker uses a more confident" not in character_card

    def test_persona_prompt_builder_renders_structured_context_blocks(self) -> None:
        """Verify Persona prompts include the requested structured RAG sections."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona_character_profile": "A careful archivist with a dry wit.",
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

        assert "<profile>" in prompt
        assert "## Identity" in prompt
        assert "<memory>" in prompt
        assert "- The user prefers concise answers." in prompt
        assert "- The character once helped recover a lost journal." in prompt
        assert "<mood>" in prompt
        assert "Thoughtful and slightly tired." in prompt
        assert "<history>" not in prompt
        assert "assistant: Yes, we catalogued the letters." not in prompt
        assert "user: What do you notice?" not in prompt
        assert "A careful archivist with a dry wit." in prompt
        assert "Push the conversation forward naturally." in prompt
        assert (
            "Never output emojis; use plain text suitable for speech synthesis."
            in prompt
        )
        assert (
            "Treat facts in <memory> as true background context when they are relevant."
            in prompt
        )
        assert (
            "Keep facts from <memory> silent unless the current user message clearly asks for them"
            in prompt
        )
        assert "## Runtime Guidance" in prompt
        assert "Do not greet the user or restart the conversation." in prompt
        assert "## Reference Resolution" in prompt
        assert "The active character is Fixture." in prompt
        assert (
            "When the user refers to the active character by name, nickname, or matching third-person pronouns"
            in prompt
        )
        assert "he, him, his, she, her, hers, they, them, their" in prompt
        assert "What do you notice?" not in prompt
        assert "<request>" not in prompt

    def test_markdown_persona_headers_use_consistent_spacing(self) -> None:
        """Verify generated and embedded Persona Markdown headers have one blank line after them."""
        rendered = render_markdown_subsection(
            "Fixture",
            "Intro\n## Nested\n\n- first\n## Another\n- second",
        )

        assert (
            rendered
            == "## Fixture\n\nIntro\n## Nested\n\n- first\n## Another\n\n- second"
        )

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

        assert context.character_profile.name == "Mirelle"
        assert context.character_profile.age == "27"
        assert context.character_profile.gender == "female"
        assert (
            "A precise investigator who notices tiny shifts in tone."
            in context.character_profile.render()
        )
        assert context.persona_source_material.identity == (
            "Name: Mirelle\nAge: 27\nGender: female\n\n"
            "A precise investigator who notices tiny shifts in tone."
        )
        assert context.persona_source_material.speech_style == (
            "Elegant, steady, and mildly teasing.\n\n"
            "- Warmth: mid\n- Directness: high\n- Humor: low\n"
            "- Detail: high\n- Formality: high\n- Enthusiasm: low"
        )
        assert "Style Notes:" in card
        assert "Elegant, steady, and mildly teasing." in card
        assert "Boundaries:" in card
        assert "Prompt Rules:" in card
        assert "Example Dialogue:" in card
        assert "- Formality: high" in card
        assert "- Enthusiasm: low" in card

    def test_voice_persona_style_extends_shared_persona(self) -> None:
        """Verify a selected voice can refine the shared Persona response style."""
        engine = make_pipeline_engine()
        engine.backend.uses_voice_bundles = True
        engine.current_voice = "bold"
        engine.current_character = "Celune"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(name="Celune", profile="A careful archivist."),
            speaking_style="Measured and observant.",
            style=PersonaStyleValues(
                warmth="high",
                directness="mid",
                enthusiasm="low",
            ),
        )
        engine.persona_history = []
        engine.persona_attachments = []
        engine.retrieved_long_term_memory = []
        engine.config = {"persona_state": "Neutral."}
        voice_persona = CEVoicePersona(
            speaking_style="More playful and energetic.",
            prompt_rules=("Use a brighter conversational rhythm.",),
            style=PersonaStyleValues(directness="high", enthusiasm="high"),
        )
        fake_loader = SimpleNamespace(bundle=SimpleNamespace())

        with (
            mock.patch("celune.persona.impl.default_loader", return_value=fake_loader),
            mock.patch("celune.conversation.default_loader", return_value=None),
            mock.patch(
                "celune.persona.impl.persona_metadata_from_voice",
                return_value=voice_persona,
            ),
        ):
            context = pipeline.build_persona_context(
                cast(Celune, engine),
                "Hello.",
            )

        assert context.persona_card.directness == "high"
        assert context.persona_card.enthusiasm == "high"
        assert "Measured and observant." in context.persona_card.speaking_style
        assert "More playful and energetic." in context.persona_card.speaking_style
        assert (
            "Use a brighter conversational rhythm." in context.persona_card.prompt_rules
        )

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

        assert first_prompt != second_prompt
        assert "A precise investigator." in first_prompt
        assert "A mischievous mechanic." in second_prompt

    def test_persona_prompt_prefers_manifest_markdown_files_when_available(
        self,
    ) -> None:
        """Verify CECHAR v3 persona Markdown overrides legacy-derived prompt text."""
        engine = make_pipeline_engine()
        engine.config = {"persona_persona": "Legacy personality text."}
        engine.current_character = "Mirelle"
        engine.current_voice = "balanced"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(
                name="Mirelle",
                profile="Legacy identity text.",
            ),
            speaking_style="Legacy speech style.",
        )
        fake_loader = SimpleNamespace(
            bundle=SimpleNamespace(
                metadata={
                    "name": "Mirelle",
                    "assets": {
                        "identity.md": {
                            "offset": 0,
                            "length": 18,
                            "sha256": "0" * 64,
                        },
                        "personality.md": {
                            "offset": 18,
                            "length": 21,
                            "sha256": "1" * 64,
                        },
                        "speech_style.md": {
                            "offset": 39,
                            "length": 20,
                            "sha256": "2" * 64,
                        },
                    },
                },
                assets={
                    "identity.md": {},
                    "personality.md": {},
                    "speech_style.md": {},
                },
                read_bundle_asset=lambda name: {
                    "identity.md": b"Manifest identity.",
                    "personality.md": b"Manifest personality.",
                    "speech_style.md": b"Manifest speech style.",
                }[name],
            ),
        )

        with mock.patch("celune.conversation.default_loader", return_value=fake_loader):
            prompt = PersonaPromptBuilder.build(
                pipeline.build_persona_context(cast(Celune, engine), "Status?")
            )

        self.assertIn("Manifest identity.", prompt)
        self.assertIn("Manifest personality.", prompt)
        self.assertIn("Manifest speech style.", prompt)
        self.assertIn("<user_instructions>", prompt)
        self.assertIn("Legacy personality text.", prompt)
        self.assertNotIn("Legacy identity text.", prompt)
        self.assertNotIn("Ignored text.", prompt)

    def test_persona_debug_overrides_replace_manifest_markdown_files(self) -> None:
        """Verify opt-in app-data Markdown replaces matching CECHAR source files."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"debug_overrides": True}}
        engine.current_character = "Mirelle"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(name="Mirelle")
        )
        fake_loader = SimpleNamespace(
            bundle=SimpleNamespace(
                metadata={
                    "persona": {"identity": {"name": "Mirelle"}},
                },
            ),
        )

        with (
            mock.patch("celune.conversation.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.conversation.persona_files_from_bundle",
                return_value={"personality.md": "Pack personality."},
            ),
            mock.patch(
                "celune.conversation.persona_override_files",
                return_value={"personality.md": "Debug personality."},
            ),
        ):
            files = pipeline._persona_manifest_files(cast(Celune, engine))

        assert files == {"personality.md": "Debug personality."}

    def test_persona_prompt_does_not_hardcode_celune_identity(self) -> None:
        """Verify Persona prompts stay character-agnostic without pack metadata."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        assert "Name: Fixture" in prompt
        assert "Name: Celune" not in prompt

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
        self.assertIn(f"Gender: {string('persona.default_gender')}", prompt)

    def test_default_cechar_pack_adds_celune_prompt_foundation(self) -> None:
        """Verify the bundled CECHAR pack assembles only its current prompt."""
        bundle = CEVoice.open(Path("voices/default.cevoice"))
        loader = CEVoiceLoader(bundle)
        self.addCleanup(loader.close)

        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = True

        with mock.patch("celune.conversation.default_loader", return_value=loader):
            prompt = PersonaPromptBuilder.build(
                pipeline.build_persona_context(cast(Celune, engine), "Hello.")
            )

        for source_text in persona_files_from_bundle(bundle).values():
            self.assertIn(source_text, prompt)

        self.assertIn("Aliases: Cel", prompt)
        self.assertIn("Role: Lunar guardian", prompt)
        self.assertIn("Ground every claim in available evidence.", prompt)
        self.assertIn("adopt the corrected meaning immediately", prompt)
        self.assertIn("Remain consistently yourself.", prompt)
        self.assertNotIn("apply the correction immediately", prompt)
        self.assertNotIn("Celune is a quiet nocturnal presence", prompt)
        self.assertNotIn(
            "These examples demonstrate tone and conversational style", prompt
        )
        self.assertNotIn(
            "Keep the familiar nocturnal tone grounded and concise.", prompt
        )
        self.assertNotIn("## Response Behavior", prompt)
        self.assertNotIn("User: i think i fixed it", prompt)

        voice_manifest = bundle.metadata["voices"]
        self.assertIsInstance(voice_manifest, dict)
        for voice_metadata in voice_manifest.values():
            self.assertIsInstance(voice_metadata, dict)
            self.assertNotIn("persona", voice_metadata)

    def test_non_default_character_pack_does_not_inherit_celune_foundation(
        self,
    ) -> None:
        """Verify another character pack does not receive Celune-only instructions."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Mirelle"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = False

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        self.assertNotIn("You are Celune, commonly called Cel.", prompt)

    def test_agent_prompt_context_uses_existing_task_and_tool_contracts(self) -> None:
        """Verify task context and tool metadata are available without runtime authority."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = False

        request = AgentRequest("Check whether the process is running.")
        task = AgentTask(task_id="task-1", session_id="default", request=request)
        agent_context = AgentContext(
            request=request,
            mode="agent",
            persona_capabilities=PersonaCapabilities(),
            task=task,
            last_tool_result={
                "tool_call_id": "call-1",
                "output": "running",
                "error": None,
            },
        )
        schema = AgentToolSchema(
            tool_id="process_status",
            display_name="Process status",
            description="Check whether a local process is running.",
            arguments=(
                AgentToolArgumentSchema(
                    name="name",
                    value_type=AgentToolValueType.STRING,
                ),
            ),
            behavior=AgentToolBehavior.READ_ONLY,
            danger=AgentToolDangerLevel.LOW,
        )
        pending_call: ToolCall = {
            "id": "call-1",
            "name": "process_status",
            "arguments": {"name": "celune"},
        }

        context = pipeline.build_persona_context(
            cast(Celune, engine),
            request.request,
            agent_context=agent_context,
            tool_schemas=(schema,),
            pending_tool_call=pending_call,
        )
        prompt = PersonaPromptBuilder.build(context)

        self.assertIn("<agent_context>", prompt)
        self.assertIn('"tool_id": "process_status"', prompt)
        self.assertIn('"state": "queued"', prompt)
        self.assertIn('"output": "running"', prompt)
        self.assertIn("runtime remains authoritative", prompt)
        self.assertLess(prompt.index("<memory>"), prompt.index("<agent_context>"))

        payload = pipeline.build_persona_request(
            cast(Celune, engine),
            request.request,
            agent_context=agent_context,
            tool_schemas=(schema,),
            pending_tool_call=pending_call,
        )
        messages = cast(list[JSON], payload["messages"])
        self.assertEqual(payload["system"], messages[0]["content"])
        self.assertEqual(cast(str, payload["system"]).count("<agent_context>"), 1)

    def test_named_celune_custom_pack_does_not_use_default_identity(self) -> None:
        """Verify custom packs named Celune do not inherit default identity fields."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = False

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        assert "Name: Celune" in prompt

    def test_persona_context_uses_weighted_emotion_state_when_unconfigured(
        self,
    ) -> None:
        """Verify Persona state can come from weighted conversation emotion."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "I feel awful."},
            {"role": "assistant", "content": "I am staying steady."},
        ]

        fake_analyzer = SimpleNamespace(
            summarize_history=mock.Mock(
                return_value=SimpleNamespace(
                    target_state=(
                        "Target emotion: gently reassuring. "
                        "The user's recent mood leans toward sadness."
                    )
                )
            )
        )

        with mock.patch(
            "celune.conversation._persona_emotion_analyzer",
            return_value=fake_analyzer,
        ):
            context = pipeline.build_persona_context(
                cast(Celune, engine), "Please stay with me."
            )

        assert "Target emotion: gently reassuring." in context.mood_or_state
        fake_analyzer.summarize_history.assert_called_once()

    def test_persona_context_prefers_configured_state_over_emotion_analysis(
        self,
    ) -> None:
        """Verify an explicit persona_state still overrides automatic emotion blending."""
        engine = make_pipeline_engine()
        engine.config = {"persona_state": "Thoughtful and slightly tired."}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        with mock.patch("celune.conversation._persona_emotion_analyzer") as analyzer:
            context = pipeline.build_persona_context(cast(Celune, engine), "Hello.")

        assert context.mood_or_state == "Thoughtful and slightly tired."
        analyzer.assert_not_called()

    def test_persona_context_logs_emotion_fallback_reason(self) -> None:
        """Verify emotion-analysis failures are surfaced in verbose logs."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        captured: list[tuple[str, str]] = []
        engine.log = lambda msg, severity="info", **kwargs: captured.append(
            (msg, severity)
        )

        fake_analyzer = SimpleNamespace(
            last_error="lunahr/emotispace-128 could not be loaded",
            summarize_history=mock.Mock(return_value=None),
        )

        with mock.patch(
            "celune.conversation._persona_emotion_analyzer",
            return_value=fake_analyzer,
        ):
            context = pipeline.build_persona_context(cast(Celune, engine), "Hello.")

        assert context.mood_or_state == "Neutral."
        assert captured == [
            (
                (
                    "Persona emotion analysis fell back to Neutral: "
                    "lunahr/emotispace-128 could not be loaded"
                ),
                "warning",
            )
        ]

    def test_persona_prompt_builder_omits_vision_context_without_attachments(
        self,
    ) -> None:
        """Verify Persona prompts no longer serialize recent chat into the system prompt."""
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

        assert "<vision_context>" not in prompt
        assert "<history>" not in prompt
        assert "assistant: hi" not in prompt

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

        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "current"}
        assert len(messages) == 8
        assert messages[1] == {"role": "user", "content": "old user 6"}
        assert messages[-2] == {"role": "assistant", "content": "old reply 11"}
        system_prompt = cast(str, messages[0]["content"])
        assert "<history>" not in system_prompt
        assert "old user 4" not in str(messages)

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
            post=lambda json: FakeResponse(),
        )
        engine.dev = False

        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "new user")

        assert engine.persona_history == [
            {"role": "user", "content": "old user 2"},
            {"role": "assistant", "content": "old reply 3"},
            {"role": "user", "content": "new user"},
            {"role": "assistant", "content": "new reply"},
        ]

    def test_persona_history_compacts_older_turns_into_session_summary(self) -> None:
        """Verify older Persona turns are summarized and recent turns stay available."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona": {
                "memory": {
                    "max_short_term_messages": 3,
                    "context_compaction_keep_recent_messages": 2,
                    "context_summary_max_characters": 240,
                }
            }
        }
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "The archive is stored in the attic."},
            {"role": "assistant", "content": "I will keep that context in mind."},
        ]

        class FakeResponse:
            """Fake API response for context-compaction assertions."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response."""
                return {"response": "Understood."}

        engine.vision = SimpleNamespace(
            post=lambda json: FakeResponse(),
        )
        engine.dev = False

        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "What is next?")

        assert engine.persona_history == [
            {"role": "user", "content": "What is next?"},
            {"role": "assistant", "content": "Understood."},
        ]
        assert "The archive is stored in the attic." in engine.persona_session_summary
        assert engine.persona_session_summary.startswith("Conversation context:")
        assert "Earlier summary:" not in engine.persona_session_summary
        assert "user:" not in engine.persona_session_summary
        assert "assistant:" not in engine.persona_session_summary

    def test_persona_history_summary_does_not_nest_previous_summary(self) -> None:
        """Verify repeated compaction removes wrappers and duplicate summary labels."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona": {
                "memory": {
                    "max_short_term_messages": 1,
                    "context_compaction_keep_recent_messages": 1,
                    "context_summary_max_characters": 240,
                }
            }
        }
        engine.persona_session_summary = (
            "<conversation_summary>Earlier summary: Earlier summary: "
            "The archive is stored in the attic.</conversation_summary>"
        )
        engine.persona_history = [
            {"role": "user", "content": "The archive is stored in the attic."},
            {"role": "assistant", "content": "I will remember the archive location."},
        ]

        compact_persona_history(cast(Celune, engine))

        assert engine.persona_session_summary.count("archive") == 1
        assert engine.persona_session_summary.startswith("Conversation context:")
        assert "Earlier summary:" not in engine.persona_session_summary
        assert "<conversation_summary>" not in engine.persona_session_summary
        assert "</conversation_summary>" not in engine.persona_session_summary

    def test_persona_history_prefers_neutral_vlm_summary(self) -> None:
        """Verify compaction stores the VLM summary instead of raw conversation turns."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona": {
                "memory": {
                    "max_short_term_messages": 1,
                    "context_compaction_keep_recent_messages": 1,
                    "context_summary_max_characters": 240,
                }
            }
        }
        engine.persona_session_summary = "Earlier context."
        engine.persona_history = [
            {"role": "user", "content": "The TTS response was cut off."},
            {"role": "assistant", "content": "I will keep that in mind."},
        ]
        summarize_history = mock.Mock(
            return_value="The conversation concerns a TTS cutoff."
        )
        engine.vision = SimpleNamespace(summarize_history=summarize_history)

        compact_persona_history(cast(Celune, engine))

        summarize_history.assert_called_once_with(
            [{"role": "user", "content": "The TTS response was cut off."}],
            "Earlier context.",
            240,
        )
        assert (
            engine.persona_session_summary
            == "Conversation context: The conversation concerns a TTS cutoff."
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
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            engine.persona_memory_store = store

            with mock.patch(
                "celune.speech.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                assert pipeline.think(
                    cast(Celune, engine),
                    "remember that my test word is moonlight",
                )

            retrieved = store.retrieve("Celune", "what is my test word?")

        assert [record.content for record in retrieved] == ["my test word is moonlight"]

    def test_persona_memory_path_is_independent_of_markdown_debug_overrides(
        self,
    ) -> None:
        """Verify Markdown debug overrides do not change Persona memory storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for debug_overrides in (False, True):
                engine = make_pipeline_engine()
                engine.config = {
                    "persona": {"debug_overrides": debug_overrides},
                }
                with mock.patch(
                    "celune.persona.memory.persona_data_dir",
                    return_value=Path(temp_dir),
                ):
                    store = pipeline._persona_memory_store(cast(Celune, engine))

                assert store is not None
                assert (
                    store._path_for_character("Celune")
                    == Path(temp_dir) / "celune" / "memory" / "records.json"
                )

    def test_think_uses_classifier_for_unmatched_durable_user_context(self) -> None:
        """Verify unmatched durable context can be saved by the local classifier."""

        class FakeResponse:
            """Fake response for Persona and memory-classifier requests."""

            def __init__(self, payload: JSON) -> None:
                self.payload = payload

            def raise_for_status(self) -> None:
                """Fake return of raise_for_status()."""

            def json(self) -> JSON:
                """Return the configured fake response payload."""
                return self.payload

        class FakeVision:
            """Fake Persona client exposing the classifier hook."""

            def __init__(self) -> None:
                self.classifier_payload: Optional[JSON] = None

            def post(self, json: JSON) -> FakeResponse:
                """Return the normal Persona response."""
                discard(json)
                return FakeResponse({"response": "I understand."})

            def classify_memory(self, json: JSON) -> FakeResponse:
                """Return one durable fact from the classifier."""
                self.classifier_payload = json
                return FakeResponse(
                    {
                        "response": _json.dumps(
                            {
                                "memories": [
                                    {
                                        "content": "The user has a dog named Luna.",
                                        "importance": 2,
                                        "confidence": 0.94,
                                    }
                                ]
                            }
                        )
                    }
                )

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
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            engine.persona_memory_store = store

            with mock.patch(
                "celune.speech.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                assert pipeline.think(
                    cast(Celune, engine),
                    "I recently adopted a dog named Luna.",
                )

            classifier_payload = cast(FakeVision, engine.vision).classifier_payload
            assert classifier_payload is not None
            records = store.load_records("Celune")

        assert [record.content for record in records] == [
            "The user has a dog named Luna"
        ]

    def test_persona_response_speech_is_not_saved(self) -> None:
        """Verify generated Persona replies skip saved utterance artifacts."""
        engine = make_pipeline_engine()
        engine.dev = False
        response = mock.Mock()
        response.json.return_value = {"response": "Generated reply."}
        engine.vision = SimpleNamespace(post=mock.Mock(return_value=response))

        with (
            mock.patch("celune.conversation._store_persona_memories"),
            mock.patch("celune.conversation.build_persona_request", return_value={}),
            mock.patch("celune.speech.queue_speech", return_value=True) as q,
        ):
            assert pipeline.think(cast(Celune, engine), "User request.")

        q.assert_called_once_with(
            engine,
            "Generated reply.",
            save=False,
            display_text="Generated reply.",
        )

    def test_persona_prompt_builder_includes_compacted_summary_when_present(
        self,
    ) -> None:
        """Verify compacted older conversation remains available to Persona."""
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

        assert "<conversation_summary>" in prompt
        assert "The user and character already discussed the archive." in prompt

    def _assert_persona_messages_include_pending_attachments(
        self, expected_image: str, expected_video: str
    ) -> None:
        """Verify local visual attachments are converted for Persona."""
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
        assert user["role"] == "user"
        content = cast(list[dict[str, str]], user["content"])
        assert content == [
            {
                "type": "image",
                "image": expected_image,
            },
            {
                "type": "video",
                "video": expected_video,
            },
            {"type": "text", "text": "What is this?"},
        ]

    @LINUX_ONLY
    def test_persona_messages_include_pending_attachments_on_linux(self) -> None:
        """Verify Linux Persona messages use file URLs for local attachments."""
        self._assert_persona_messages_include_pending_attachments(
            "file:///C:/Users/user/Pictures/frame.png",
            "file:///C:/Users/user/Videos/clip.mp4",
        )

    @WINDOWS_ONLY
    def test_persona_messages_include_pending_attachments_on_windows(self) -> None:
        """Verify Windows Persona messages use local paths for attachments."""
        self._assert_persona_messages_include_pending_attachments(
            "C:/Users/user/Pictures/frame.png",
            "C:/Users/user/Videos/clip.mp4",
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
        assert user["role"] == "user"
        assert cast(list[dict[str, str]], user["content"]) == [
            {
                "type": "image",
                "image": "https://example.com/images/frame.png",
            },
            {"type": "text", "text": "What is this?"},
        ]

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
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "What is this?")

        assert engine.persona_attachments == []
        first_payload = engine.vision.payloads[0]
        first_messages = cast(list[JSON], first_payload["messages"])
        assert isinstance(first_messages[-1]["content"], list)

        second_payload = pipeline.build_persona_request(
            cast(Celune, engine), "And now?"
        )
        second_system = cast(str, second_payload["system"])
        second_messages = cast(list[JSON], second_payload["messages"])
        assert "<behavior>" in second_system
        assert second_messages[-1] == {"role": "user", "content": "And now?"}

    async def test_generation_worker_normalizes_each_split_chunk(self) -> None:
        """Verify normalization happens after splitting and before generation.

        Raises:
            AssertionError: Chunk normalization behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        generated_texts: list[str] = []
        events: list[str] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            text = cast(str, kwargs["text"])
            events.append(f"generate:{text}")
            generated_texts.append(text)
            yield np.ones((8, 2), dtype=np.float32) * 0.01, 48000, None

        def normalize(value: str) -> str:
            events.append(f"normalize:{value}")
            return f"normalized {value}"

        engine.backend = SimpleNamespace(
            generate_stream=generate_stream,
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
        engine.caption_timing_callback = mock.Mock()
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
            await self._run_generation_worker(cast(Celune, engine))

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
        engine.caption_timing_callback.assert_called_once()
        timing_call = engine.caption_timing_callback.call_args
        assert timing_call is not None
        self.assertEqual(timing_call.args[0], "raw input")
        self.assertEqual(timing_call.args[3], "normalized first normalized second")

    async def test_generation_worker_reloads_language_specific_model_when_needed(
        self,
    ) -> None:
        """Verify request-scoped language can trigger a backend model reload."""
        engine = make_pipeline_engine()
        backend = self._LanguageAwareBackend()
        engine.backend = backend
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "Auto"
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

        engine.text_queue.put(
            pipeline.SpeechRequest(
                "bonjour",
                "bonjour",
                language="fr",
                save=True,
            )
        )
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["bonjour"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        backend.unload_model.assert_called_once_with()
        backend.load_model.assert_called_once_with("fake/balanced", lang="fr")
        assert backend.current_language == "fr"
        assert engine.model.kwargs["lang"] == "fr"

    async def test_generation_worker_disables_smart_buffer_for_realtime_speed(
        self,
    ) -> None:
        """Verify smart buffering gets out of the way when generation is realtime."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.full((48000, 2), 0.1, dtype=np.float32)
            for _ in range(3):
                yield chunk.copy(), 48000, None

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
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
        engine.smart_buffer_generation_speed = 1.3
        engine.config = {"smart_buffer": {"enabled": True}}

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=True))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
            mock.patch(
                "celune.playback._queue_playback_chunk",
                side_effect=lambda _engine, _source_id, audio, _sr, _timing=None: (
                    queued_lengths.append(len(audio))
                ),
            ),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert queued_lengths == [48000, 48000, 48000]
        assert engine.smart_buffer_target_seconds == 0.0

    async def test_generation_worker_expands_smart_buffer_when_speed_drops(
        self,
    ) -> None:
        """Verify slower observed generation expands the smart buffer target."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.full((48000, 2), 0.1, dtype=np.float32)
            for _ in range(3):
                yield chunk.copy(), 48000, None

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
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
        engine.smart_buffer_generation_speed = 1.3
        engine.config = {"smart_buffer": {"enabled": True}}

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=True))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
            mock.patch(
                "celune.playback._queue_playback_chunk",
                side_effect=lambda _engine, _source_id, audio, _sr, _timing=None: (
                    queued_lengths.append(len(audio))
                ),
            ),
            mock.patch(
                "celune.pipeline._monotonic_time",
                side_effect=[0.0, 0.1, 0.2, 2.8, 5.6, 8.4] + [8.4] * 16,
            ),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert queued_lengths == [48000, 48000, 48000]
        assert engine.smart_buffer_generation_speed > 0.5
        assert engine.smart_buffer_generation_speed < 1.3
        assert engine.smart_buffer_target_seconds > 0.0

    async def test_generation_worker_waits_for_completion_at_very_low_speed(
        self,
    ) -> None:
        """Verify very slow generation fully buffers the utterance before playback."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.full((48000, 2), 0.1, dtype=np.float32)
            for _ in range(3):
                yield chunk.copy(), 48000, None

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
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
        engine.smart_buffer_generation_speed = 0.35
        engine.config = {"smart_buffer": {"enabled": True}}

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=True))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
            mock.patch(
                "celune.playback._queue_playback_chunk",
                side_effect=lambda _engine, _source_id, audio, _sr, _timing=None: (
                    queued_lengths.append(len(audio))
                ),
            ),
            mock.patch(
                "celune.pipeline._monotonic_time",
                side_effect=[0.0, 0.5, 2.0, 4.0, 6.0, 6.0] + [6.0] * 16,
            ),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert queued_lengths == [48000, 48000, 48000]
        assert engine.smart_buffer_target_seconds == float("inf")

    def test_playback_blocks_uses_true_50ms_chunks(self) -> None:
        """Verify mixer block splitting uses real wall-clock block lengths."""
        timing = pipeline.SpeechTiming(start_time=0.0)
        chunk = pipeline.PlaybackChunk(
            source_id=1,
            audio=np.zeros((4800, 2), dtype=np.float32),
            sample_rate=48000,
            timing=timing,
        )

        blocks = pipeline._playback_blocks(chunk)

        assert len(blocks) == 2
        first_block, first_timing = blocks[0]
        second_block, second_timing = blocks[1]
        assert first_block.shape == (2400, 2)
        assert second_block.shape == (2400, 2)
        assert first_timing is timing
        assert second_timing is None

    async def test_generation_worker_handles_save_false_without_concatenate_error(
        self,
    ) -> None:
        """Verify silence analysis does not crash when output saving is disabled."""
        engine = make_pipeline_engine()
        engine.backend = SimpleNamespace(
            generate_stream=lambda _model, **_kwargs: iter(
                [(np.full((8, 2), 0.1, dtype=np.float32), 48000, None)]
            )
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

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=False))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch(
                "celune.pipeline.is_silent_utterance", return_value=(False, 0)
            ) as silent_mock,
            mock.patch("celune.pipeline._write_celune_flac") as write_mock,
        ):
            await self._run_generation_worker(cast(Celune, engine))

        silent_mock.assert_called_once()
        write_mock.assert_not_called()
        assert engine.recently_saved is None

    async def test_generation_worker_accumulates_total_generated_speech_seconds(
        self,
    ) -> None:
        """Verify completed speech adds to the cumulative footer metric."""
        engine = make_pipeline_engine()
        engine.backend = SimpleNamespace(
            generate_stream=lambda _model, **_kwargs: iter(
                [(np.full((48000, 2), 0.1, dtype=np.float32), 48000, None)]
            )
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
        engine.total_generated_speech_seconds = 30.0

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=False))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline._write_celune_flac"),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert engine.total_generated_speech_seconds == 31.0

    async def test_generation_worker_ignores_absolute_silence_without_retrying(
        self,
    ) -> None:
        """Verify absolute-silence chunks never enter playback or trigger retries."""
        engine = make_pipeline_engine()
        generate_stream = mock.Mock(
            side_effect=lambda _model, **_kwargs: iter(
                [(np.zeros((8, 2), dtype=np.float32), 48000, None)]
            )
        )

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
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

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=False))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(True, 2)),
            mock.patch("celune.playback._queue_playback_chunk") as queue_chunk,
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert generate_stream.call_count == 1
        queue_chunk.assert_not_called()
        retry_logs = [
            message
            for message, severity in engine.messages
            if severity == "warning" and "regenerating" in message
        ]
        assert retry_logs == []
        assert not any(
            "may be unexpectedly silent" in message
            for message, severity in engine.messages
            if severity == "warning"
        )
        assert engine.text_queue.empty()

    async def test_generation_worker_skips_requeue_once_silent_retry_limit_is_reached(
        self,
    ) -> None:
        """Verify capped silent requests are not put back into the queue."""
        engine = make_pipeline_engine()
        capped_request = pipeline.SpeechRequest(
            "hello",
            "hello",
            save=False,
            silent_retry_count=3,
        )
        generate_stream = mock.Mock(
            side_effect=lambda _model, **_kwargs: iter(
                [(np.full((8, 2), 0.0005, dtype=np.float32), 48000, None)]
            )
        )

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
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
        engine.text_queue.put(capped_request)
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(True, 2)),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert generate_stream.call_count == 1
        assert any(
            "stayed silent after 3 retries" in message
            for message, severity in engine.messages
            if severity == "warning"
        )
        assert engine.text_queue.empty()
