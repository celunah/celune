# Qwen3 VoiceDesign reference

## Exact model

The only permitted generation model is:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

Do not use a clone, custom-voice, or other size/type model as a fallback. If the exact model cannot be loaded, stop and explain the dependency or cache problem.

## Celune-managed cache

Run model discovery/download in a short-lived subprocess so cache environment changes cannot leak into Celune or the user's shell. From the repository, use:

```python
from celune.paths import huggingface_hub_cache_dir, temp_data_dir
from celune.backends.tts.base import cached_hf_snapshot_path, local_hf_offline_mode
```

First call the existing cache lookup with the exact model ID and the model files required by the installed Qwen package. Under `local_hf_offline_mode()`, load the returned snapshot path with `Qwen3TTSModel.from_pretrained()` or the installed faster wrapper. Only if a complete snapshot is absent, call `huggingface_hub.snapshot_download()` with `repo_id` set to the exact model ID and `cache_dir=str(huggingface_hub_cache_dir(create=True))`. Set `HF_HOME` and `HF_HUB_CACHE` to the corresponding Celune-managed paths for that subprocess only. Use `local_files_only=True` for the first lookup/load attempt, and record the resolved snapshot/revision.

Never inspect, populate, or rely on the host Hugging Face cache. Do not use a global environment mutation as a substitute for Celune's helpers.

## Runtime API check

The installed `qwen_tts` package currently exposes:

```python
model = Qwen3TTSModel.from_pretrained(snapshot_path, device_map="cuda:0", torch_dtype=...)
wavs, sample_rate = model.generate_voice_design(
    text=preview_text,
    instruct=voice_instruction,
    language=language,
)
```

The installed `faster_qwen3_tts` wrapper may expose the same method and requires CUDA. Inspect the actual signatures and model type at runtime. The loaded model must report VoiceDesign support; a successful load of the wrong model type is a failure, not a fallback opportunity. Save each returned waveform as a playable WAV using its returned sample rate, preserving the original candidate until CAC completes.

## Candidate metadata and revisions

For every candidate record JSON with at least:

- candidate label and batch/revision number;
- exact model ID, resolved snapshot path, and revision/commit;
- full VoiceDesign instruction and original brief;
- preview text and language;
- seed and all generation settings;
- WAV output path and sample rate.

Generate A/B/C with identical inputs and settings except for the recorded random/sampling seed. If the user gives descriptive feedback, append it to the original instruction and generate a new batch, up to three batches by default. Never change the preview sentence, model, or cache policy just to make candidates easier to compare.
