# SPDX-License-Identifier: Apache-2.0
"""Tests for global operation modes and Persona capabilities."""

from types import SimpleNamespace
from typing import cast
from unittest import mock

from celune.agent import AgentOutput, agent_mode_enabled
from celune.modes import resolve_operation_mode
from celune.celune import _agent_task_config
from celune.persona.impl import persona_enabled, persona_context_size
from celune.typing.common import Config
from celune.typing.persona import PersonaModel, PersonaProcessor, PersonaTokenizer
from celune.persona.runtime import PersonaBackend
from celune.persona.capabilities import PersonaCapabilities

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

    def test_vram_presets_gate_persona_and_agent_features(self) -> None:
        """Verify Persona and standard agent mode require the high VRAM tier."""
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
        converse_low_config: Config = {"mode": "converse", "vram": "medium"}
        agent_high_config: Config = {"mode": "agent", "vram": "high"}
        agent_xhigh_config: Config = {"mode": "agent", "vram": "xhigh"}

        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            self.assertFalse(persona_enabled(speak_config))
            self.assertTrue(persona_enabled(converse_config))
            self.assertFalse(persona_enabled(converse_low_config))
            self.assertTrue(persona_enabled(agent_high_config))
            self.assertTrue(persona_enabled(agent_xhigh_config))
            self.assertTrue(agent_mode_enabled(agent_high_config))
            self.assertTrue(agent_mode_enabled(agent_xhigh_config))
            smart_agent_config: Config = {
                "mode": "agent",
                "vram": "high",
                "persona": {"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
            }
            self.assertFalse(agent_mode_enabled(smart_agent_config))
            self.assertFalse(persona_enabled(smart_agent_config))
            self.assertFalse(agent_mode_enabled({"mode": "converse"}))

    def test_persona_and_agent_context_sizes_are_capped(self) -> None:
        """Verify configured context values cannot exceed the VRAM policy caps."""
        assert persona_context_size({"persona": {"context_size": 8192}}) == 2048
        assert persona_context_size({}) == 2048
        assert (
            _agent_task_config({"agent": {"context_size": 32768}}).context_size == 8192
        )
        assert _agent_task_config({}).context_size == 8192


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
