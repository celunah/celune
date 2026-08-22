# Celune

Celune is a local-first character engine for expressive text-to-speech, voice
conversion, Persona conversation, and constrained agent work. It buffers speech
for responsive playback, keeps the core runtime usable from Python, and exposes
the same engine through the terminal UI, REST API, WebUI, and extension system.

This site documents the behavior implemented in the repository, including the
custom CEVOICE/CECHAR voice-pack format and CEDTS isolated-worker protocol.

## Choose a path

| If you want to… | Start here |
| --- | --- |
| Install and launch Celune | [Requirements and installation](getting-started/installation.md) |
| Understand the first startup | [First run](getting-started/first-run.md) |
| Change runtime behavior | [Configuration](user-guide/configuration.md) |
| Speak, converse, or use the agent | [Operation modes](user-guide/modes.md) |
| Call Celune from another program | [REST API](API.md) or [Public Python API](reference/public-api.md) |
| Build a voice pack or character | [CEVOICE and CECHAR](CEVOICE.md) |
| Add a plugin-like extension | [Extensions](interfaces/extensions.md) |
| Work on a backend or worker | [Architecture](development/architecture.md), [Backends](development/backends.md), and [CEDTS](formats/cedts.md) |

## The short version

```text
input text/audio
      │
      ├─ speak ────────► TTS backend ─► buffered pipeline ─► DSP ─► playback
      ├─ converse ────► Persona ─────► speech pipeline ────────► playback
      ├─ agent ────────► planner/tools ► Persona response ──────► playback
      └─ voice input ──► VC backend ─► normalized audio ────────► playback
```

The core uses normalized `numpy.float32` audio, 48 kHz stereo output, and FLAC
for saved files and API responses. Backends may run in isolated environments;
the core communicates with them over CEDTS rather than importing backend-only
packages into the main application.

## Project conventions

The source repository is MIT-licensed. Third-party models, backend packages,
voice recordings, and other assets retain their own licenses. See
[Licensing](licensing.md) before redistributing a model or a voice pack.

The repository root keeps `README.md` for the project landing page,
`AGENTS.md` for contributor instructions, and `CLAUDE.md` for tooling metadata.
Documentation-owned Markdown lives under `docs/`; the lore-only Book of Celune
at `resources/about/about-celune.md` is intentionally not treated as a factual
technical reference.
