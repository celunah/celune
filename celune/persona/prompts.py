# SPDX-License-Identifier: MIT
"""Structured prompt building for the Persona system."""

import contextlib
import re
from dataclasses import dataclass, field

from ..paths import temp_data_dir

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


def _render_markdown_subsection(heading: str, content: str) -> str:
    """Return one Markdown subsection with consistent heading spacing."""
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
                _render_markdown_subsection("Identity", self.identity),
                _render_markdown_subsection("Soul", self.soul),
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
                _render_markdown_subsection("Personality", self.personality),
                _render_markdown_subsection("Speech Style", self.speech_style),
                _render_markdown_subsection("Boundaries", self.boundaries),
                _render_markdown_subsection("Examples", self.examples),
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
    retrieved_long_term_memory: RetrievedMemoryBundle = field(
        default_factory=RetrievedMemoryBundle
    )


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
                        _render_markdown_subsection(
                            "Runtime Guidance",
                            _render_lines(behavior_lines),
                        ),
                        _render_markdown_subsection(
                            "Reference Resolution",
                            _render_lines(reference_resolution_lines),
                        ),
                    )
                    if section
                ),
            ),
        ]

        prompt = "\n\n".join(section for section in sections if section)
        PersonaPromptBuilder._write_debug_prompt(prompt)
        return prompt
