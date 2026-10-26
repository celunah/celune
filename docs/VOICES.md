# Voice Design

Updated: Sep 30, 2026 (Classic Remastered)

This document describes how Celune's voice identities were created, selected, and refined.

## Who is Celune

This is only a simplified explanation. More details are included in her lore.

Celune by nature appears to be a young female of approximately 28 years of age, who speaks with a low contralto tone.

Her average pitch range during speech is ~170 Hz. This is reflected across all four of her tones, however her Upbeat tone may modulate more than the rest, contributing to a higher perceived pitch of approx. 210 Hz.

The character personality is loosely based on Japanese-style philosophies, and the connected UX practices follow a Korean style. When she speaks, she tends to be slightly hesitant and keeps her responses brief, while naturally pausing in her speech. The interpretation is left to the user to decipher.

Please check [the Book of Celune](https://github.com/celunah/celune/blob/dev/resources/about/about-celune.md) for lore-accurate details and a comprehensive description of Celune.

## Pronunciation glossary

Celune can be pronounced in one of two ways:

- English-style: Seh-LOON (IPA: /sɛˈluːn/, \[ˌsɛˈluːn\])
- French-style: Say-LUNE or See-LUNE, approximation: Say-L(Y)OON or See-L(Y)OON, (IPA: /seɪˈlyn/ or /seˈlyn/)

Phonetic IPA forms of Celune are only available for the canonical English way of saying her name. French-style IPA is left simplified.

Parts in brackets may not be said equally by all speakers, as they're not native English sounds.

Her name was made by appending a "Ce-" prefix from "celestial" onto the French word "lune", meaning "moon".

## Models

All TTS models voice cloning from samples found in your CEVOICE or CECHAR voice pack.

Celune native models were removed as of version 4.0.0. If you wish to use them, please downgrade Celune to an older version.

Refer to the following models for details:

- [Qwen3-TTS](<https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>)
- [VoxCPM2](<https://huggingface.co/openbmb/VoxCPM2>)
- [Pocket TTS](<https://huggingface.co/kyutai/pocket-tts>) [(ungated)](<https://huggingface.co/lunahr/pocket-tts-ungated>)
- [dots.tts](<https://huggingface.co/dots-studio/dots.tts-mf>)
- [LuxTTS](<https://huggingface.co/YatharthS/LuxTTS>)

## Reference text

These scripts are what Celune says in the reference audio:

Non-verbal tags are included for improved expression in her delivery.

> Calm:
>
> ```text
> [sigh] There are moments in life, which need... just a little bit of calmness, right?
> ```
>
> Balanced:
>
> ```text
> I feel like something good is about to happen. Really, it's just bound to come.
> ```
>
> Bold:
>
> ```text
> [gasp] Wait, what the hell just happened? Why?
> ```
>
> Upbeat:
>
> ```text
> [laughter] Now THAT is what I call "funny".
> ```

## Reference prompts

A base prompt was used to create the main voice with [Gemini 3.8 Flash TTS](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-text-to-speech/):

```text
A feminine voice in a low register, clear and steady with mild airiness. Balanced, composed delivery with natural, restrained expression. Fully voiced, not breathy or whispered.
```

It was then guided by a style delivery prompt. They are listed below.

> Calm:
>
> ```text
> slightly hushed delivery, very subtle modulation, not a whisper
> ```
>
> Balanced:
>
> ```text
> No extra prompt
> ```
>
> Bold:
>
> ```text
> high energy, with richer modulation
> ```
>
> Upbeat:
>
> ```text
> playful with joy, richer modulation, slightly higher pitched
> ```

## Effects

Celune's voice does not incorporate any additional audio effects to ensure a clean, natural delivery.

The voices were cleaned up of background noise.

## Output format

This format is currently enforced by Celune's DSP outputs.

- 48kHz stereo, signed 24-bit PCM, FLAC
