# SPDX-License-Identifier: MIT
"""Structured prompt building for the Persona system."""

import contextlib
from dataclasses import dataclass, field

from ..paths import temp_data_dir


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

    def render_identity_summary(self) -> str:
        """Return a compact identity summary for the runtime prompt.

        Returns:
            str: A concise identity block for the active character.
        """
        name = self.name.strip() or "Unknown"
        profile = self.profile.strip()
        sep = ", " if profile else "."
        if profile:
            profile = profile[0].lower() + profile[1:]

        lines = [
            f"You are {name}{sep}{profile}",
            f"When asked for an introduction, refer to yourself as {name}.",
        ]
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

    def behavior_cues(self) -> tuple[str, ...]:
        """Return concise CEVOICE-driven behavior cues for the runtime prompt.

        Returns:
            tuple[str, ...]: Prompt lines derived from CEVOICE persona metadata.
        """
        cues: list[str] = []
        if self.speaking_style.strip():
            cues.append(self.speaking_style.strip())

        rules: list[str] = []
        for item in self.boundaries:
            stripped = item.strip()
            if stripped:
                rules.append(stripped)
        for item in self.prompt_rules:
            stripped = item.strip()
            if stripped and stripped not in rules:
                rules.append(stripped)
        if rules:
            cues.append("\n- ".join(rules[:2]))
        return tuple(cues)


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

            if last_assistant:
                lines.append(
                    "[The assistant has already acknowledged the complaint about repetition. "
                    "Do not acknowledge it again. Move the conversation forward.]"
                )

        lines.append(f"user: {message}")
        return _render_lines(lines)


@dataclass(frozen=True)
class PersonaContext:
    """Structured context passed into the Persona prompt builder."""

    character_profile: CharacterProfile
    persona_card: PersonaCard
    mood_or_state: str
    retrieved_long_term_memory: RetrievedMemoryBundle = field(
        default_factory=RetrievedMemoryBundle
    )
    current_run_chat_history: ShortTermHistory = field(default_factory=ShortTermHistory)
    user_message: str = ""


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
        name = context.character_profile.name.strip() or "Unknown"
        behavior_lines = [
            f"- Respond only as {name}.",
            "- Continue directly from <history>.",
            "- Push the conversation forward instead of returning to earlier turns.",
            "- Treat facts in <memories> as true context when they are relevant.",
            (
                "- Keep items from <memories> silent unless the current user "
                "message clearly asks for them or they are necessary for a natural reply."
            ),
            "- Do not greet the user. Do not ask what they need. Just respond.",
            "- Do not repeat anything already said in <history>.",
            "- Do not bring up older messages, stored facts, or resolved topics on your own.",
            "- Keep the reply natural, short, and emotionally consistent with <mood>.",
            "- Stay under 3 sentences, unless the user asked for detail.",
            "- Use only a single paragraph, unless formatting is necessary.",
        ]
        behavior_lines.extend(
            f"- {cue}." if not cue.endswith((".", "!", "?")) else f"- {cue}"
            for cue in context.persona_card.behavior_cues()
        )
        sections = [
            _render_optional_section(
                "profile",
                context.character_profile.render_identity_summary(),
            ),
            _render_optional_section(
                "memories",
                context.retrieved_long_term_memory.render(),
            ),
            _render_optional_section(
                "history",
                context.current_run_chat_history.render(context.user_message.strip()),
            ),
            _render_optional_section(
                "mood",
                context.mood_or_state.strip() or "neutral",
            ),
            f"<behavior>\n{_render_lines(behavior_lines)}\n</behavior>".strip(),
            f"{name}:",
        ]

        prompt = "\n\n".join(section for section in sections if section)
        PersonaPromptBuilder._write_debug_prompt(prompt)
        return prompt
