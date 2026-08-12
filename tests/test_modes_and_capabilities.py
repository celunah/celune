# SPDX-License-Identifier: MIT
"""Tests for global operation modes and Persona capabilities."""

from types import SimpleNamespace
from typing import cast

from celune.agent import AgentOutput, agent_mode_enabled
from celune.modes import resolve_operation_mode
from celune.persona.capabilities import PersonaCapabilities
from celune.persona.impl import persona_enabled
from celune.persona.runtime import PersonaBackend
from celune.typing.common import Config
from celune.typing.persona import PersonaModel, PersonaProcessor, PersonaTokenizer

from .support import CeluneTestCase


class TestOperationMode(CeluneTestCase):
    """Verify global modes provide the requested feature gates."""

    def test_operation_modes_are_resolved_directly(self) -> None:
        """Verify the active global modes include the production agent mode."""
        self.assertEqual(resolve_operation_mode({"mode": "speak"}), "speak")
        self.assertEqual(resolve_operation_mode({"mode": "converse"}), "converse")
        self.assertEqual(resolve_operation_mode({"mode": "agent"}), "agent")

    def test_legacy_input_mode_does_not_change_global_mode(self) -> None:
        """Verify legacy input-mode values remain compatible with the new switch."""
        assert resolve_operation_mode({"mode": "voice_conversion"}) == "converse"

    def test_speak_disables_persona_and_agent_mode_enables_agent_routing(self) -> None:
        """Verify speak disables Persona while agent routing remains enabled."""
        speak_config: Config = {
            "mode": "speak",
            "vram": "high",
            "persona": {"enabled": True},
        }
        converse_config: Config = {
            "mode": "converse",
            "vram": "high",
            "persona": {"enabled": False},
        }

        self.assertFalse(persona_enabled(speak_config))
        self.assertTrue(persona_enabled(converse_config))
        self.assertTrue(agent_mode_enabled({"mode": "agent"}))
        self.assertFalse(agent_mode_enabled({"mode": "converse"}))


class TestPersonaCapabilities(CeluneTestCase):
    """Verify Persona capability declarations are explicit and architecture-aware."""

    def test_unloaded_backend_is_text_only(self) -> None:
        """Verify text remains available while optional capabilities are disabled."""
        capabilities = PersonaBackend().capabilities()

        assert capabilities == PersonaCapabilities(
            text=True,
            vision=False,
            image_uploads=False,
            emotion_probes=False,
        )

    def test_loaded_vlm_reports_multimodal_and_emotion_capabilities(self) -> None:
        """Verify a compatible loaded VLM reports its supported features."""
        backend = PersonaBackend()
        backend.model = cast(
            PersonaModel,
            SimpleNamespace(config=SimpleNamespace(hidden_size=16)),
        )
        backend.tokenizer = cast(PersonaTokenizer, SimpleNamespace())
        backend.processor = cast(PersonaProcessor, SimpleNamespace())
        backend.supports_vision = True
        backend.supports_emotion_probes = True

        capabilities = backend.capabilities()

        assert capabilities.text
        assert capabilities.vision
        assert capabilities.image_uploads
        assert capabilities.emotion_probes

    def test_agent_output_contract_has_stable_response_shape(self) -> None:
        """Verify the future agent output contract matches the public schema."""
        output: AgentOutput = {
            "tool_call": None,
            "response": "placeholder",
            "end": True,
            "paused": False,
        }

        assert set(output) == {"tool_call", "response", "end", "paused"}
