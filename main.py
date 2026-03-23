#!/usr/bin/env python3
# pylint: disable=R0902, R0913, R0917, W0718
"""
Celune 2.0.1 - A celestial TTS engine.
She has three tones, and can change them on the fly.
She can also run extensions.
"""

import sys

from huggingface_hub.utils import disable_progress_bars

from celune.celune import Celune
from celune.ui import CeluneUI

disable_progress_bars()


def main() -> None:
    """Instantiate and start Celune."""
    sys.stdout.write("\x1b]2;Celune\x07")
    sys.stdout.flush()

    ui = CeluneUI()
    celune = Celune(
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ref_audio="refs/neutral.wav",
        ref_text="My name is Celune, pronounced Celune. It is a pleasure to meet you.",
        log_callback=ui.tts_log,
        status_callback=ui.safe_status,
        error_callback=ui.error,
        idle_callback=ui.tts_idle,
        queue_avail_callback=ui.tts_queue_avail,
        voice_changed_callback=ui.tts_voice_changed,
    )

    celune.setup_extensions()

    ui.celune = celune
    ui.run()


if __name__ == "__main__":
    main()
