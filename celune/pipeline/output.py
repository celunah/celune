# SPDX-License-Identifier: Apache-2.0
"""Speech output naming, saving, and SFX downloads."""

from __future__ import annotations
import os
import re
import sys
import json
import pathlib
import contextlib
import subprocess
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Optional

import soundfile as sf

from .buffering import config_value_lines
from .timing import (
    MAX_YOUTUBE_DOWNLOAD_RETRIES,
)

from ..i18n import string
from ..constants import (
    APP_SLUG,
)
from ..cesn import (
    contains_cesn_note_event,
)
from ..utils import (
    discard,
    available,
    cesn_to_text,
)
from ..paths import (
    outputs_dir,
    project_root,
    temp_data_dir,
    running_compiled,
)

if TYPE_CHECKING:
    from ..celune import Celune


def format_stat_duration(seconds: float) -> str:
    """Format a duration as whole minutes and seconds for engine statistics."""
    whole_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


def _output_filename_fragment(text: str, *, singing_backend: bool) -> str:
    """Return a safe filename fragment, removing CESN tags for singing text."""
    filename_text = text
    if singing_backend and contains_cesn_note_event(text):
        filename_text = cesn_to_text(text)
    first_words = "_".join(filename_text.split()[:3]).lower()
    return re.sub(r"[^a-zA-Z0-9_]", "", first_words)


def _output_filename_prefix(*, singing_backend: bool) -> str:
    """Return the output filename prefix for speech or singing audio."""
    output_kind = "singing" if singing_backend else "speech"
    return f"{APP_SLUG}_{output_kind}"


def saved_output_speech_seconds() -> float:
    """Return cumulative duration from saved speech and singing outputs."""
    output_dir = outputs_dir()
    if not output_dir.exists():
        return 0.0

    total_seconds = 0.0
    patterns = (
        f"{_output_filename_prefix(singing_backend=False)}_*.flac",
        f"{_output_filename_prefix(singing_backend=True)}_*.flac",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            try:
                total_seconds += sf.info(path).duration
            except (OSError, RuntimeError, TypeError, ValueError):
                continue

    return total_seconds


def _youtube_sfx_temp_path() -> pathlib.Path:
    """Return the fixed temporary WAV path used for URL-backed SFX playback."""
    return temp_data_dir(create=True) / "temporary_audio.wav"


def is_youtube_sfx_url(value: str) -> bool:
    """Return whether ``value`` looks like a supported YouTube URL."""
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.netloc or "").lower().removeprefix("www.")
    return host in {"youtube.com", "youtu.be", "music.youtube.com"}


def youtube_sfx_title(url: str) -> str:
    """Return a friendly title for one YouTube URL when available."""
    query = urllib.parse.urlencode({"url": url, "format": "json"})
    endpoint = f"https://www.youtube.com/oembed?{query}"
    # noinspection PyBroadException
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return "YouTube audio"

    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return "YouTube audio"


def _summarize_youtube_download_error(output: str) -> str:
    """Extract the actionable reason from yt-dlp output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    ignored_markers = (
        "no supported javascript runtime",
        "only deno is enabled",
        "youtube extraction without a js runtime",
        "github.com/yt-dlp/yt-dlp/wiki/ejs",
    )

    for line in reversed(lines):
        if line.upper().startswith("ERROR:"):
            reason = line.split(":", 1)[1].strip()
            if reason:
                return reason

    for line in reversed(lines):
        if line.lower().startswith("warning:"):
            continue
        if not any(marker in line.lower() for marker in ignored_markers):
            return line

    return string("pipeline.download_unknown_error")


def download_youtube_sfx(
    engine: Celune, url: str
) -> Optional[tuple[pathlib.Path, str]]:
    """Download one YouTube URL as a temporary WAV file for SFX playback."""
    yt_dlp_module = "yt_dlp"
    if not available(yt_dlp_module):
        engine.log(string("pipeline.yt_dlp_missing"), "warning")
        engine.error_callback(string("pipeline.yt_dlp_required"))
        return None

    output_path = _youtube_sfx_temp_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        output_path.unlink(missing_ok=True)

    title = youtube_sfx_title(url)
    out_tmpl = str(output_path.with_suffix(".%(ext)s"))
    engine.status_callback(string("status.downloading_audio"))
    engine.log(f"[SFX] Downloading audio from {url}...")
    python_executable = sys.executable
    if running_compiled():
        if os.name == "nt":
            python_executable = str(project_root() / ".venv" / "Scripts" / "python.exe")
        else:
            python_executable = str(project_root() / ".venv" / "bin" / "python")
    command = [
        python_executable,
        "-m",
        yt_dlp_module,
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--no-playlist",
        "--no-progress",
        "--force-overwrites",
        *_youtube_download_options(engine),
        "--output",
        out_tmpl,
        url,
    ]
    failure_reason = string("pipeline.download_unknown_error")
    discard(failure_reason)

    total_attempts = MAX_YOUTUBE_DOWNLOAD_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        with contextlib.suppress(OSError):
            output_path.unlink(missing_ok=True)

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            failure_reason = string("pipeline.download_timeout")
        else:
            output = "\n".join(
                part for part in (completed.stderr, completed.stdout) if part
            )
            failure_reason = _summarize_youtube_download_error(output)
            if completed.returncode == 0 and output_path.exists():
                return output_path, title
            if completed.returncode == 0 and not output_path.exists():
                failure_reason = string("pipeline.downloader_no_file")

        if attempt < total_attempts:
            engine.log(
                f"{string('pipeline.download_failed')}: {failure_reason}; "
                f"{string('pipeline.download_retry', retry_count=attempt, max_retries=MAX_YOUTUBE_DOWNLOAD_RETRIES)}",
                "warning",
            )
            continue

        engine.log(f"{string('pipeline.download_failed')}: {failure_reason}", "warning")
        engine.error_callback(string("pipeline.download_youtube_failed_short"))
        return None

    return None


def _youtube_download_options(engine: Celune) -> list[str]:
    """Build optional yt-dlp arguments from the nested YouTube configuration."""
    value = engine.config.get("youtube")
    if not isinstance(value, dict):
        return []

    options: list[str] = []
    cookies_file = config_value_lines(value.get("cookies_file"))
    cookies_from_browser = config_value_lines(value.get("cookies_from_browser"))
    if cookies_file:
        options.extend(("--cookies", cookies_file[0]))
    elif cookies_from_browser:
        options.extend(("--cookies-from-browser", cookies_from_browser[0]))

    for key, option in (
        ("js_runtimes", "--js-runtimes"),
        ("remote_components", "--remote-components"),
    ):
        for configured_value in config_value_lines(value.get(key)):
            options.extend((option, configured_value))

    po_tokens = config_value_lines(value.get("po_token"))
    if po_tokens:
        options.extend(("--extractor-args", f"youtube:po_token={','.join(po_tokens)}"))

    player_clients = config_value_lines(value.get("player_client"))
    if player_clients:
        options.extend(
            ("--extractor-args", f"youtube:player_client={','.join(player_clients)}")
        )

    for extractor_arg in config_value_lines(value.get("extractor_args")):
        options.extend(("--extractor-args", extractor_arg))
    return options
