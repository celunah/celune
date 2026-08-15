# SPDX-License-Identifier: MIT
"""Structured prompt building for the Persona system."""

import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, cast

from ..paths import temp_data_dir
from ..typing.agent import AgentContext, AgentToolSchema, ToolCall
from ..typing.common import JSON

_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")


def _render_lines(lines: list[str]) -> str:
    """Return non-empty lines joined for one prompt section."""
    cleaned = [line.strip() for line in lines if isinstance(line, str) and line.strip()]
    if not cleaned:
        return "none"
    return "\n".join(cleaned)


def _render_optional_section(tag: str, content: str) -> str:
    """Return one tagged section only when ``content`` is non-empty."""
    stripped = content.strip()
    if not stripped or stripped == "none":
        return ""
    return f"<{tag}>\n{stripped}\n</{tag}>"


def _render_json(value: JSON) -> str:
    """Return deterministic JSON text for model-only structured context."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_agent_context(
    agent_context: AgentContext,
    tool_schemas: Sequence[AgentToolSchema],
    pending_tool_call: Optional[ToolCall],
) -> str:
    """Render existing agent contracts as non-authoritative Persona context."""
    sections = [
        (
            "Runtime context is informational only. The runtime remains authoritative "
            "for task state, tool selection, validation, permissions, approval, and limits. "
            "Do not claim that a tool ran without a structured result."
        ),
        f"Mode: {agent_context.mode}",
        (
            "Planning instruction: state the concrete local action intent in concise "
            "natural language for the registered tool selector."
            if agent_context.last_tool_result is None
            else "Response instruction: use the structured tool result to answer the "
            "user naturally; do not invent actions or results."
        ),
    ]
    task = agent_context.task
    if task is not None:
        sections.append(
            "Task:\n"
            + _render_json(
                {
                    "task_id": task.task_id,
                    "session_id": task.session_id,
                    "request": task.request.request,
                    "state": task.state.value,
                    "iterations": task.iterations,
                    "generated_tokens": task.generated_tokens,
                    "context_tokens": task.context_tokens,
                }
            )
        )
    if tool_schemas:
        sections.append(
            "Tool catalog:\n"
            + "\n".join(
                _render_json(schema.to_json())
                for schema in sorted(tool_schemas, key=lambda item: item.tool_id)
            )
        )
    if pending_tool_call is not None:
        sections.append(
            "Pending tool call:\n" + _render_json(cast(JSON, pending_tool_call))
        )
    if agent_context.last_tool_result is not None:
        sections.append(
            "Last tool result:\n"
            + _render_json(cast(JSON, agent_context.last_tool_result))
        )
    return "\n\n".join(sections)


def render_markdown_subsection(heading: str, content: str) -> str:
    """Return one Markdown subsection with consistent heading spacing.

    Args:
        heading: The heading to be rendered.
        content: The content to be rendered.

    Returns:
        str: The rendered Markdown subsection.
    """
    lines = content.strip().splitlines()
    normalized_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        normalized_lines.append(line)
        index += 1

        if _MARKDOWN_HEADING.match(line):
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index < len(lines):
                normalized_lines.append("")

    stripped = "\n".join(normalized_lines).strip()
    if not stripped:
        return ""
    return f"## {heading}\n\n{stripped}"


@dataclass(frozen=True)
class CharacterProfile:
    """Identity information for the active character."""

    name: str
    age: str = "unknown"
    gender: str = "unknown"
    profile: str = ""

    def render(self) -> str:
        """Return the character identity block.

        Returns:
            str: The formatted character identity block.
        """
        lines = [
            f"Name: {self.name.strip() or 'Unknown'}",
            f"Age: {self.age.strip() or 'unknown'}",
            f"Gender: {self.gender.strip() or 'unknown'}",
        ]
        if self.profile.strip():
            lines.extend(["", "Profile:", self.profile.strip()])
        return "\n".join(lines)


@dataclass(frozen=True)
class PersonaCard:
    """Persona and speaking-style instructions for the active character."""

    persona: str
    warmth: str
    directness: str
    humor: str
    detail: str
    context: str
    voice: str
    speaking_style: str = ""
    boundaries: tuple[str, ...] = ()
    prompt_rules: tuple[str, ...] = ()
    example_dialogue: tuple[str, ...] = ()
    formality: str = "mid"
    enthusiasm: str = "mid"

    def render(self) -> str:
        """Return the persona style block.

        Returns:
            str: The formatted style block.
        """
        lines = [
            "Persona:",
            self.persona.strip()
            or "Speak naturally as the active character with conversational continuity and emotional consistency.",
            "",
            "Speaking Style:",
            f"- Warmth: {self.warmth.strip() or 'mid'}",
            f"- Directness: {self.directness.strip() or 'mid'}",
            f"- Humor: {self.humor.strip() or 'low'}",
            f"- Detail: {self.detail.strip() or 'mid'}",
            f"- Formality: {self.formality.strip() or 'mid'}",
            f"- Enthusiasm: {self.enthusiasm.strip() or 'mid'}",
        ]
        if self.speaking_style.strip():
            lines.extend(["", "Style Notes:", self.speaking_style.strip()])
        if self.boundaries:
            lines.extend(["", "Boundaries:"])
            lines.extend(f"- {item}" for item in self.boundaries if item.strip())
        if self.prompt_rules:
            lines.extend(["", "Prompt Rules:"])
            lines.extend(f"- {item}" for item in self.prompt_rules if item.strip())
        if self.example_dialogue:
            lines.extend(["", "Example Dialogue:"])
            lines.extend(f"- {item}" for item in self.example_dialogue if item.strip())
        if self.context.strip():
            lines.extend(["", "Context:", self.context.strip()])
        if self.voice.strip():
            lines.extend(["", "Voice:", self.voice.strip()])
        return "\n".join(lines)


@dataclass(frozen=True)
class RetrievedMemoryBundle:
    """Long-term memory retrieved for the current request."""

    memories: tuple[str, ...] = ()

    def render(self) -> str:
        """Return the long-term memory block.

        Returns:
            str: The formatted long-term memory block.
        """
        return _render_lines([f"- {memory}" for memory in self.memories])


@dataclass(frozen=True)
class PersonaSourceMaterial:
    """Whitelisted character source material used to assemble the system prompt."""

    identity: str = ""
    soul: str = ""
    personality: str = ""
    speech_style: str = ""
    boundaries: str = ""
    examples: str = ""

    def profile_section(self) -> str:
        """Return the profile section assembled from static identity files.

        Returns:
            str: The rendered profile section.
        """
        return "\n\n".join(
            section
            for section in (
                render_markdown_subsection("Identity", self.identity),
                render_markdown_subsection("Soul", self.soul),
            )
            if section
        )

    def behavior_section(self) -> str:
        """Return the behavior section assembled from persona behavior files.

        Returns:
            str: The rendered behavior section.
        """
        return "\n\n".join(
            section
            for section in (
                render_markdown_subsection("Personality", self.personality),
                render_markdown_subsection("Speech Style", self.speech_style),
                render_markdown_subsection("Boundaries", self.boundaries),
                render_markdown_subsection("Examples", self.examples),
            )
            if section
        )


@dataclass(frozen=True)
class PersonaContext:
    """Structured context passed into the Persona prompt builder."""

    character_profile: CharacterProfile
    persona_card: PersonaCard
    persona_source_material: PersonaSourceMaterial
    mood_or_state: str
    conversation_summary: str = ""
    retrieved_long_term_memory: RetrievedMemoryBundle = field(
        default_factory=RetrievedMemoryBundle
    )
    user_instructions: str = ""
    agent_context: Optional[AgentContext] = None
    tool_schemas: tuple[AgentToolSchema, ...] = ()
    pending_tool_call: Optional[ToolCall] = None


class PersonaPromptBuilder:
    """Build structured runtime prompts for the Persona system."""

    @staticmethod
    def _write_debug_prompt(prompt: str) -> None:
        """Persist the current Persona prompt when the temp directory is writable."""
        with contextlib.suppress(OSError):
            (temp_data_dir(create=True) / "rag_prompt.txt").write_text(
                prompt,
                encoding="utf-8",
            )

    @staticmethod
    def build(context: PersonaContext) -> str:
        """Return the structured Persona runtime prompt.

        Args:
            context: The current character context.

        Returns:
            str: The formatted RAG prompt for persona.
        """
        behavior_lines = [
            "- Respond only as the active character.",
            "- Use the native chat history in this request as the recent conversation context.",
            "- Treat facts in <memory> as true background context when they are relevant.",
            (
                "- Keep facts from <memory> silent unless the current user message "
                "clearly asks for them or they are necessary for a natural reply."
            ),
            "- Match the emotional direction in <mood> naturally.",
            "- Do not greet the user or restart the conversation.",
            "- Do not repeat earlier turns unless the user asks for a recap.",
            "- Push the conversation forward naturally.",
            "- Stay under 3 sentences unless the user asked for detail.",
            "- Use a single paragraph unless formatting is necessary.",
            "- Never output emojis; use plain text suitable for speech synthesis.",
            "- Prefer reacting to the current conversation over reinforcing the character's identity.",
        ]
        character_name = (
            context.character_profile.name.strip() or "the active character"
        )
        reference_resolution_lines = [
            f"- The active character is {character_name}.",
            (
                "- When the user refers to the active character by name, nickname, "
                "or matching third-person pronouns (for example: he, him, his, "
                "she, her, hers, they, them, their), while discussing the character, "
                "voice, persona, or conversation, interpret those references as "
                "referring to the active character unless another person is clearly "
                "identified."
            ),
            '- The user\'s first-person pronouns ("I", "me", "my") always refer to the user.',
            "- Do not reinterpret statements about the active character as statements about the user.",
        ]
        sections = [
            _render_optional_section(
                "profile",
                context.persona_source_material.profile_section(),
            ),
            _render_optional_section(
                "conversation_summary",
                context.conversation_summary,
            ),
            _render_optional_section(
                "memory",
                context.retrieved_long_term_memory.render(),
            ),
            _render_optional_section(
                "mood",
                context.mood_or_state.strip() or "Target emotion: neutral.",
            ),
            _render_optional_section(
                "behavior",
                "\n\n".join(
                    section
                    for section in (
                        context.persona_source_material.behavior_section(),
                        render_markdown_subsection(
                            "Runtime Guidance",
                            _render_lines(behavior_lines),
                        ),
                        render_markdown_subsection(
                            "Reference Resolution",
                            _render_lines(reference_resolution_lines),
                        ),
                    )
                    if section
                ),
            ),
            _render_optional_section(
                "user_instructions",
                context.user_instructions,
            ),
            _render_optional_section(
                "agent_context",
                (
                    _render_agent_context(
                        context.agent_context,
                        context.tool_schemas,
                        context.pending_tool_call,
                    )
                    if context.agent_context is not None
                    else ""
                ),
            ),
        ]

        prompt = "\n\n".join(section for section in sections if section)
        PersonaPromptBuilder._write_debug_prompt(prompt)
        return prompt
