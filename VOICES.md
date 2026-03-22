# Reference Text
- Calm
`My name is... Celune... It is so... quiet.`

- Neutral
`My name is Celune, pronounced Celune. It is a pleasure to meet you.`

- Energetic
`My name is Celune! Let's do this, we have to get it done!`

# Candidates
- Energetic #16 (Seed: 590298652)
- Neutral #32 (Seed: 418977738)
- Calm #7 (Seed: 4243102495)

# Post-processing
- Energetic/Neutral pitch `-1 sem`
- Calm pitch `-0.5 sem`

- Energetic - slow down `lune` in `Celune` by 33% (high quality paul stretch), to correct the pronunciation of the name
- Neutral - 150ms pause between `pronounced` and `Celune`, for natural pacing

# Effects

## Reverb
- room size = 15%
- pre-delay = 75ms
- reverberance = 15%
- damping = 75%
- tone low = 0%
- tone high = 50%
- wet gain = -20 dB
- dry gain = 0 dB
- stereo width = 85%
- wet only = no 

## High-pass
- frequency = 200 Hz
- roll-off (dB per octave) = 12 dB 

## Layering
- Two tracks: wet and dry
- Max volume w/o clipping on both
- -5dB relative to dry on wet track

# Output format
- 48kHz stereo, signed 16-bit PCM