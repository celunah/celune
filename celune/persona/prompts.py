# SPDX-License-Identifier: MIT
"""Structured prompt building for the Persona system."""

import textwrap
from pathlib import Path
from dataclasses import dataclass, field

from platformdirs import user_data_dir

from ..constants import APP_SLUG


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
class ShortTermHistory:
    """Current-run chat history for the active conversation."""

    turns: tuple[tuple[str, str], ...] = ()
    session_summary: str = ""

    def render(self, message: str) -> str:
        """Return the short-term memory block.

        Args:
            message: The user's recent message.

        Returns:
            str: The formatted short-term memory block.
        """
        lines: list[str] = []
        if self.session_summary.strip():
            lines.extend(["Summary:", self.session_summary.strip(), ""])

        lines.extend(f"{role}: {content}" for role, content in self.turns)

        if self.turns:
            last_assistant = next(
                (
                    content
                    for role, content in reversed(self.turns)
                    if role == "assistant"
                ),
                None,
            )

            # HACK: don't repeat this
            # the characters loved to repeat themselves for no apparent reason
            # thanks Qwen for me having to prompt engineer around this issue with both ChatGPT and Claude
            #
            # Qwen3-VL is also prone to this, the prompt probably sucks
            if last_assistant:
                lines.append(
                    "[The assistant has already acknowledged the complaint about repetition. "
                    "Do not acknowledge it again. Move the conversation forward.]"
                )

        lines.append(f"user: {message}")
        return _render_lines(lines)


@dataclass(frozen=True)
class VisualContext:
    """Optional visual context for the current request."""

    items: tuple[str, ...] = ()

    def render(self) -> str:
        """Return the visual context block.

        Returns:
            str: The formatted visual context block.
        """
        return _render_lines([f"- {item}" for item in self.items])


@dataclass(frozen=True)
class PersonaContext:
    """Structured context passed into the Persona prompt builder."""

    character_profile: CharacterProfile
    persona_card: PersonaCard
    relationship_memory: str
    mood_or_state: str
    retrieved_long_term_memory: RetrievedMemoryBundle = field(
        default_factory=RetrievedMemoryBundle
    )
    current_run_chat_history: ShortTermHistory = field(default_factory=ShortTermHistory)
    visual_context: VisualContext = field(default_factory=VisualContext)
    user_message: str = ""


class PersonaPromptBuilder:
    """Build structured runtime prompts for the Persona system."""

    @staticmethod
    def build(context: PersonaContext) -> str:
        """Return the structured Persona runtime prompt.

        Args:
            context: The current character context.

        Returns:
            str: The formatted RAG prompt for persona.
        """
        sections = [
            textwrap.dedent(
                f"""
                <runtime>
                You are {context.character_profile.name}. Respond only as {context.character_profile.name}.

                Rules:
                1. Read the conversation in <short_term_memory> before writing anything.
                2. Never repeat a sentence, phrase, or idea you already said in that history.
                3. Facts in <long_term_memory> are true. Use them. Do not contradict them.
                4. Stay under 3 sentences unless the user asked a detailed question.
                5. You are not an AI assistant. Do not offer help. Just talk.
                6. Never use "assist", "help you today", "what can I do for you", or similar service-session phrasing,
                unless the character identity explicitly requires it.
                </runtime>
                """
            ).strip(),
            _render_optional_section(
                "character_identity",
                context.character_profile.render(),
            ),
            _render_optional_section(
                "long_term_memory",
                context.retrieved_long_term_memory.render(),
            ),
            _render_optional_section(
                "persona_style",
                context.persona_card.render(),
            ),
            # the following two fields are unused
            _render_optional_section(
                "relationship_to_user",
                context.relationship_memory.strip() or "none",
            ),
            _render_optional_section(
                "current_state",
                context.mood_or_state.strip() or "neutral",
            ),
            _render_optional_section(
                "short_term_memory",
                context.current_run_chat_history.render(context.user_message.strip()),
            ),
            _render_optional_section(
                "vision_context",
                context.visual_context.render(),
            ),
            textwrap.dedent(
                f"""
                <response_behavior>
                - Continue the conversation from <short_term_memory> as {context.character_profile.name}.
                - Match the tone and length of the example dialogue if provided.
                - Do not greet the user. Do not ask what they need. Just respond.
                - Do not repeat anything already said in <short_term_memory>.
                - If you don't know something, say so in character — don't invent facts.
                - One topic per response. Be direct.
                </response_behavior>
                """
            ).strip(),
            f"{context.character_profile.name}:",
        ]

        # this is for inspecting your RAG prompt, in case your character goes off the guidelines
        # it is located in the following paths:
        # %localappdata%\celune\temp\rag_prompt.txt on Windows, or
        # ~/.local/share/celune/temp/rag_prompt.txt on Linux
        with open(
            Path(user_data_dir(APP_SLUG, appauthor=False)) / "temp" / "rag_prompt.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n\n".join(section for section in sections if section))

        return "\n\n".join(section for section in sections if section)
