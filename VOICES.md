# Voice Design
This document describes how Celune's voice identities were created, selected, and refined.

# Who is Celune
Celune by nature appears to be a young female of approximately 28 years of age, who speaks with a low contralto tone.

Her average pitch range during speech is ~170 Hz. This is reflected across all four of her tones, however her Upbeat tone may modulate more than the rest, contributing to a higher perceived pitch of approx. 210 Hz.

The character personality is loosely based on Japanese-style philosophies, and the connected UX practices follow a Korean style. When she speaks, she tends to be slightly hesitant and keeps her responses brief, while naturally pausing in her speech. The interpretation is left to the user to decipher.

# Pronunciation glossary
Celune can be pronounced in one of two ways:
- English-style: Seh-LOON (IPA: /sɛˈluːn/)
- French-style: Say-LUNE or See-LUNE, approximation: Say-L(Y)OON or See-L(Y)OON, (IPA: /seɪˈlyn/ or /seˈlyn/)

Parts in brackets may not be said equally by all speakers.

The name is derived from the author's username.

# Models
Qwen-based models used in Celune no longer use reference audios.

Check https://huggingface.co/collections/lunahr/celune for a list of Celune models in use, or these model pages:

[Neutral](https://huggingface.co/lunahr/Celune-1.7B-Neutral)・[Calm](https://huggingface.co/lunahr/Celune-1.7B-Calm)・[Energetic](https://huggingface.co/lunahr/Celune-1.7B-Energetic)・[Upbeat](https://huggingface.co/lunahr/Celune-1.7B-Upbeat)

However, the VoxCPM2 backend does use them. The quality of expression is greatly improved.

# Reference text
These scripts are what Celune says in the reference audio. They were modified from an original script to reduce the hallucination risk.

- Calm
`My name is... Celune... It is so... quiet.`

- Neutral
`My name is Celune, pronounced Celune. It is a pleasure to meet you.`

- Energetic
`My name is Celune! Let's do this, we have to get it done!`

- Upbeat
`Hehehe... Hi, I'm Celune. Look, I have something to tell... might as well make it fun. Shall we?`

# Reference prompts
These prompts were used to steer direction of the voice during auditioning.

- Calm
`A female voice with a soft, velvety, and hushed texture. A slow, sophisticated blend with focused vocal control.`

- Neutral
`A female voice with a warm, steady, and slightly resonant texture. Calm and articulate with clear, grounded presence.`

- Energetic
`A female voice with a rich, resonant, and decisive texture. Confident, professional, and clear with a rhythmic drive.`

- Upbeat
`A female voice with a bright, warm, and expressive texture. Upbeat, witty, and clear with a conversational flow and playful cadence.`

# Candidates
The batch size per voice is 50. One voice was selected as the best match. Voices were generated using [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).

- Energetic #16 (Seed: 590298652)
- Neutral #32 (Seed: 418977738)
- Calm #7 (Seed: 4243102495)
- Upbeat #16 (Seed: 3771593946)

# Post-processing
These edits were applied to make sure the new references match the initial reference.

- Upbeat pitch `-1.5 sem`
- Energetic/Neutral pitch `-1 sem`
- Calm pitch `-0.5 sem`

- Energetic - slow down `lune` in `Celune` by 33% (high quality paul stretch), to correct the pronunciation of the name
- Neutral - 75ms pause between `pronounced` and `Celune`, for natural pacing

# Effects
These effects and their settings give Celune's voice its identity.

## Reverb
- room size = 50%
- pre-delay = 100ms
- reverberance = 60%
- damping = 75%
- tone low = 0%
- tone high = 50%
- wet gain = -16 dB (-12 dB for Calm)
- dry gain = 0 dB
- stereo width = 85%
- wet only = no

The Upbeat voice has an increased amount of default reverb, the TTS model will replicate it during speech.

An additional amount of reverb can be applied within Celune. 

Refer to the /reverb command for details.

## High-pass
- frequency = 200 Hz
- roll-off (dB per octave) = 12 dB

## Layering
- Two tracks: wet and dry
- Max volume w/o clipping on both
- -5dB relative to dry on wet track

# Output format
This format makes Celune sound the best on your computer, especially if using VoxCPM2.

- 48kHz stereo, signed 24-bit PCM
