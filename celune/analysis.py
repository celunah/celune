# pylint: disable=R0912, R0914
"""Analyze a WAV file and generate a radar chart plus a text report."""

import sys
import pathlib
import warnings

import librosa
import matplotlib
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.projections import PolarAxes

matplotlib.use("Agg")
# ensure you have installed this font
rcParams["font.family"] = "Outfit Thin"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TEXT_CONFIG = {
    "trait_labels": {
        "Calmness": "Calmness",
        "Energy": "Energy",
        "Softness": "Softness",
        "Clarity": "Clarity",
        "Expressiveness": "Expressiveness",
        "Presence": "Presence",
        "Playfulness": "Playfulness",
    },
    "graph": {
        "title": "Celune's Voice Report",
        "status_ok": "Celune is performing normally.",
        "status_underperforming_single": "Celune's {traits} is underperforming.",
        "status_watch_single": "Celune's {traits} is driving her tone.",
        "status_watch_multi": "Celune's {traits} are driving her tone.",
        "status_high_single": "Celune's {traits} is overpowering.",
        "status_high_multi": "Celune's {traits} are overpowering.",
    },
    "report": {
        "title": "CELUNE VOICE REPORT",
        "file_label": "File",
        "raw_metrics_header": "[ RAW METRICS ]",
        "traits_header": "[ DERIVED TRAIT SCORES  (0.0 = min, 1.0 = max) ]",
        "assessment_header": "[ ASSESSMENT ]",
        "note_line_1": "NOTE: Trait scores are heuristic estimates derived from",
        "note_line_2": "signal-level features. They are not ground-truth labels.",
        "duration_label": "Duration (s)",
        "sample_rate_label": "Sample rate (Hz)",
        "rms_mean_label": "RMS mean",
        "rms_std_label": "RMS std",
        "dynamic_range_label": "Dynamic range (dB)",
        "mean_pitch_label": "Mean pitch / F0 (Hz)",
        "pitch_variance_label": "Pitch variance (Hz^2)",
        "pitch_na": "N/A",
        "voiced_ratio_label": "Voiced ratio",
        "pause_ratio_label": "Pause ratio",
        "speaking_pace_label": "Speaking pace proxy",
        "spectral_centroid_label": "Spectral centroid (Hz)",
        "zcr_label": "Zero-crossing rate",
        "hf_energy_label": "HF energy ratio (>=4kHz)",
    },
    "assessment": {
        "warning_short_audio": (
            "This report may not be comprehensive enough to determine how "
            "Celune is actually performing. Please provide a longer clip."
        ),
        "warning_pitch_failed": (
            "No voicings found. Are you sure Celune said anything? "
            "Will use default values, this may be incorrect."
        ),
        "duration": "Audio duration is {duration:.2f} seconds.",
        "pitch_mean": "Mean pitch is {hz:.1f} Hz, it {description}.",
        "pitch_unknown": "Mean pitch could not be determined from this sample.",
        "pitch_desc_very_deep": "is too low",
        "pitch_desc_low_male": "may be too low",
        "pitch_desc_mid_male_low_female": "is in range",
        "pitch_desc_mid_female": "may be too high. If measuring Celune's upbeat tone, this is safe to ignore",
        "pitch_desc_high_female": "may be too high. If measuring Celune's upbeat tone, this is safe to ignore",
        "pitch_desc_very_high": "is too high",
        "trait_level": "{trait} is {level} ({score:.2f}).",
        "level_low": "low",
        "level_moderate_low": "moderate-low",
        "level_moderate_high": "moderate-high",
        "level_high": "high",
        "overall_calm": "Overall, this sample sounds calm and controlled.",
        "overall_energetic": "Overall, this sample is energetic and expressive.",
        "overall_steady": "Overall, this sample sounds steady and measured.",
        "overall_playful": "Overall, this voice has a noticeably playful character.",
        "overall_presence": "Overall, this voice has strong presence and authority.",
        "overall_balanced": "Overall, the voice shows a balanced mix of traits.",
        "high_pause_ratio": (
            "A high pause ratio suggests deliberate pacing or significant silence."
        ),
    },
}


def _text(*keys: str) -> str:
    """Fetch a configurable text value from ``TEXT_CONFIG``."""
    value = TEXT_CONFIG
    for key in keys:
        value = value[key]
    return value


def _trait_label(trait_name: str) -> str:
    """Return the display label for a trait."""
    return _text("trait_labels", trait_name)


def _join_trait_names(trait_names: list[str]) -> str:
    """Join display trait names in a readable way."""
    display_names = [_trait_label(name) for name in trait_names]
    if len(display_names) == 1:
        return display_names[0]
    if len(display_names) == 2:
        return f"{display_names[0]} and {display_names[1]}"
    return ", ".join(display_names[:-1]) + f", and {display_names[-1]}"


def load_audio(voice: pathlib.Path) -> tuple[np.ndarray, int]:
    """Load a WAV file while preserving the native sample rate."""
    y, sr = librosa.load(str(voice), sr=None, mono=True)
    return y, sr


def compute_raw_metrics(y: np.ndarray, sr: int) -> dict:
    """Compute low-level audio descriptors from a mono signal."""
    metrics = {
        "duration_s": librosa.get_duration(y=y, sr=sr),
        "sample_rate": sr,
    }

    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]  # noqa
    metrics["rms_mean"] = float(np.mean(rms_frames))
    metrics["rms_std"] = float(np.std(rms_frames))

    rms_db = librosa.amplitude_to_db(rms_frames + 1e-9)
    metrics["dynamic_range_db"] = float(np.max(rms_db) - np.min(rms_db))

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) > 0:
        metrics["pitch_mean_hz"] = float(np.nanmean(voiced_f0))
        metrics["pitch_variance"] = float(np.nanvar(voiced_f0))
        metrics["pitch_extraction_ok"] = True
    else:
        metrics["pitch_mean_hz"] = float("nan")
        metrics["pitch_variance"] = float("nan")
        metrics["pitch_extraction_ok"] = False

    metrics["voiced_ratio"] = float(np.mean(voiced_flag))
    metrics["pause_ratio"] = 1.0 - metrics["voiced_ratio"]

    hop_duration_s = 512 / sr
    num_voiced_frames = int(np.sum(voiced_flag))
    metrics["speaking_pace_proxy"] = float(
        num_voiced_frames * hop_duration_s / max(metrics["duration_s"], 1e-6)
    )

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]  # noqa
    metrics["spectral_centroid_mean"] = float(np.mean(centroid))

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]  # noqa
    metrics["zcr_mean"] = float(np.mean(zcr))

    stft = np.abs(librosa.stft(y, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr)
    hf_mask = freqs >= 4000
    hf_energy = float(np.sum(stft[hf_mask, :] ** 2))
    total_energy = float(np.sum(stft**2)) + 1e-9
    metrics["hf_energy_ratio"] = hf_energy / total_energy

    return metrics


def _clip_norm(value: float, low: float, high: float) -> float:
    """Map ``value`` from ``[low, high]`` to ``[0.0, 1.0]`` and clip."""
    return float(np.clip((value - low) / (high - low + 1e-9), 0.0, 1.0))


def _blend_colors(color_a: str, color_b: str, mix: float) -> str:
    """Blend two colors and return the result as a hex string."""
    color_a_rgb = np.array(mcolors.to_rgb(color_a))
    color_b_rgb = np.array(mcolors.to_rgb(color_b))
    blended = (1.0 - mix) * color_a_rgb + mix * color_b_rgb
    return mcolors.to_hex(blended)  # noqa


def _summarize_trait_status(traits: dict) -> tuple[str, str]:
    """Return a short bottom status message and its associated color."""
    base_color = "#CEBAFF"
    warn_color = "#F0E68C"
    high_color = "#F07178"

    trait_weights = {
        "presence": 1.5,
        "clarity": 1.4,
        "energy": 1.3,
        "expressiveness": 1.0,
        "softness": 0.9,
        "calmness": 0.7,
        "playfulness": 0.6,
    }

    high_traits = {name: score for name, score in traits.items() if score >= 0.8}
    warning_traits = {
        name: score for name, score in traits.items() if 0.6 <= score < 0.8
    }
    low_traits = {name: score for name, score in traits.items() if score < 0.3}

    if high_traits:
        joined = _join_trait_names(list(high_traits.keys())).lower()
        template_key = (
            "status_high_single" if len(high_traits) == 1 else "status_high_multi"
        )
        message = _text("graph", template_key).format(traits=joined)
        return message, high_color

    if warning_traits:
        joined = _join_trait_names(list(warning_traits.keys())).lower()
        template_key = (
            "status_watch_single" if len(warning_traits) == 1 else "status_watch_multi"
        )
        message = _text("graph", template_key).format(traits=joined)
        return message, warn_color

    if len(low_traits) >= 4:
        worst_trait = max(
            low_traits,
            key=lambda name: (0.5 - low_traits[name]) * trait_weights.get(name, 1.0),
        )
        message = _text("graph", "status_underperforming_single").format(
            traits=worst_trait
        )
        return message, warn_color

    return _text("graph", "status_ok"), base_color


def compute_traits(m: dict) -> dict:
    """Derive heuristic voice trait scores in the ``0.0`` to ``1.0`` range."""
    traits = {}

    pitch_hz = m["pitch_mean_hz"] if m["pitch_extraction_ok"] else 200.0
    pitch_var = m["pitch_variance"] if m["pitch_extraction_ok"] else 0.0

    pitch_stability = 1.0 - _clip_norm(pitch_var, 0, 8000)
    range_stability = 1.0 - _clip_norm(m["dynamic_range_db"], 5, 30)
    pace_calm = 1.0 - abs(m["speaking_pace_proxy"] - 0.45) / 0.45
    traits["Calmness"] = float(
        np.clip(
            0.4 * pitch_stability + 0.35 * range_stability + 0.25 * max(pace_calm, 0),
            0.0,
            1.0,
        )
    )

    loudness_score = _clip_norm(m["rms_mean"], 0.005, 0.15)
    voiced_score = _clip_norm(m["voiced_ratio"], 0.3, 0.9)
    pace_score = _clip_norm(m["speaking_pace_proxy"], 0.2, 0.8)
    traits["Energy"] = float(
        np.clip(
            0.4 * loudness_score + 0.35 * voiced_score + 0.25 * pace_score,
            0.0,
            1.0,
        )
    )

    quietness = 1.0 - _clip_norm(m["rms_mean"], 0.005, 0.15)
    warmth = 1.0 - _clip_norm(m["spectral_centroid_mean"], 1000, 4000)
    traits["Softness"] = float(np.clip(0.55 * quietness + 0.45 * warmth, 0.0, 1.0))

    hf_score = _clip_norm(m["hf_energy_ratio"], 0.02, 0.15)
    noise_penalty = 1.0 - _clip_norm(m["zcr_mean"], 0.05, 0.25)
    traits["Clarity"] = float(np.clip(0.6 * hf_score + 0.4 * noise_penalty, 0.0, 1.0))

    pitch_exp = _clip_norm(pitch_var, 0, 8000)
    range_exp = _clip_norm(m["dynamic_range_db"], 5, 30)
    traits["Expressiveness"] = float(
        np.clip(0.6 * pitch_exp + 0.4 * range_exp, 0.0, 1.0)
    )

    pause_penalty = 1.0 - _clip_norm(m["pause_ratio"], 0.1, 0.6)
    traits["Presence"] = float(
        np.clip(0.5 * loudness_score + 0.5 * pause_penalty, 0.0, 1.0)
    )

    pitch_height = _clip_norm(pitch_hz, 100, 350)
    traits["Playfulness"] = float(
        np.clip(0.35 * pitch_height + 0.35 * pitch_exp + 0.30 * pace_score, 0.0, 1.0)
    )

    return traits


def generate_assessment(m: dict, traits: dict) -> list[str]:
    """Generate human-readable assessment lines for the voice sample."""
    lines = []

    if m["duration_s"] < 2.0:
        lines.append(_text("assessment", "warning_short_audio"))
    if not m["pitch_extraction_ok"]:
        lines.append(_text("assessment", "warning_pitch_failed"))

    lines.append(_text("assessment", "duration").format(duration=m["duration_s"]))

    if m["pitch_extraction_ok"]:
        hz = m["pitch_mean_hz"]
        if hz < 100:
            pitch_desc = _text("assessment", "pitch_desc_very_deep")
        elif hz < 165:
            pitch_desc = _text("assessment", "pitch_desc_low_male")
        elif hz < 210:
            pitch_desc = _text("assessment", "pitch_desc_mid_male_low_female")
        elif hz < 280:
            pitch_desc = _text("assessment", "pitch_desc_mid_female")
        elif hz < 360:
            pitch_desc = _text("assessment", "pitch_desc_high_female")
        else:
            pitch_desc = _text("assessment", "pitch_desc_very_high")
        lines.append(
            _text("assessment", "pitch_mean").format(hz=hz, description=pitch_desc)
        )
    else:
        lines.append(_text("assessment", "pitch_unknown"))

    def level(trait_score: float) -> str:
        if trait_score < 0.25:
            return _text("assessment", "level_low")
        if trait_score < 0.50:
            return _text("assessment", "level_moderate_low")
        if trait_score < 0.75:
            return _text("assessment", "level_moderate_high")
        return _text("assessment", "level_high")

    for trait, score in traits.items():
        lines.append(
            _text("assessment", "trait_level").format(
                trait=_trait_label(trait),
                level=level(score),
                score=score,
            )
        )

    calm = traits["Calmness"]
    energy = traits["Energy"]
    expr = traits["Expressiveness"]
    presence = traits["Presence"]
    play = traits["Playfulness"]

    if calm > 0.65 and energy < 0.45:
        lines.append(_text("assessment", "overall_calm"))
    elif energy > 0.65 and expr > 0.55:
        lines.append(_text("assessment", "overall_energetic"))
    elif calm > 0.55 and expr < 0.4:
        lines.append(_text("assessment", "overall_steady"))
    elif play > 0.6:
        lines.append(_text("assessment", "overall_playful"))
    elif presence > 0.7:
        lines.append(_text("assessment", "overall_presence"))
    else:
        lines.append(_text("assessment", "overall_balanced"))

    if m["pause_ratio"] > 0.5:
        lines.append(_text("assessment", "high_pause_ratio"))

    return lines


def plot_radar(traits: dict, title: str, output_path: pathlib.Path) -> None:
    """Draw a filled radar chart of trait scores and save it as a PNG."""
    base_color = "#CEBAFF"
    warn_color = "#F0E68C"
    high_color = "#F07178"
    grid_color = "#CEBAFF"
    inner_band_color = _blend_colors(base_color, "#1d1824", 0.55)

    trait_names = list(traits.keys())
    labels = [_trait_label(name) for name in trait_names]
    values = list(traits.values())
    num_labels = len(labels)
    max_value = max(values) if values else 0.0
    status_text, status_color = _summarize_trait_status(traits)

    angles = [index / float(num_labels) * 2 * np.pi for index in range(num_labels)]
    angles += angles[:1]
    values_plot = values + values[:1]

    if max_value <= 0.6:
        shape_color = base_color
    elif max_value <= 0.8:
        mix = _clip_norm(max_value, 0.6, 0.8)
        shape_color = _blend_colors(base_color, warn_color, mix)
    else:
        mix = _clip_norm(max_value, 0.8, 1.0)
        shape_color = _blend_colors(warn_color, high_color, mix)

    ax: PolarAxes
    fig, ax = plt.subplots(figsize=(7.4, 8.8), subplot_kw={"polar": True})

    fig.patch.set_facecolor("#1d1824")
    ax.set_facecolor("#1d1824")

    band_specs = [
        (0.0, 0.3, inner_band_color, 0.18),
        (0.3, 0.3, base_color, 0.14),
        (0.6, 0.2, warn_color, 0.14),
        (0.8, 0.2, high_color, 0.16),
    ]
    for start, height, color, alpha in band_specs:
        ax.bar(
            x=0.0,
            height=height,
            width=2 * np.pi,
            bottom=start,
            align="edge",
            color=color,
            edgecolor="none",
            alpha=alpha,
            zorder=0,
        )

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=grid_color, size=8)
    ax.tick_params(colors=grid_color)
    ax.spines["polar"].set_color(grid_color)
    ax.spines["polar"].set_linewidth(1.2)
    ax.grid(color=grid_color, alpha=0.18, linewidth=0.9)
    ax.tick_params(axis="x", pad=6)
    ax.set_rlabel_position(22)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, fontweight="bold", color=grid_color)

    ax.plot(angles, values_plot, color=shape_color, linewidth=2.8, linestyle="solid")
    ax.fill(angles, values_plot, color=shape_color, alpha=0.24)

    ax.set_title(
        f"{_text('graph', 'title')}\n{title}",
        size=13,
        fontweight="bold",
        pad=34,
        color=grid_color,
    )

    fig.text(
        0.5,
        0.04,
        status_text,
        ha="center",
        va="center",
        color=status_color,
        fontsize=11,
        fontweight="bold",
    )

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.05, right=0.95)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    m: dict,
    traits: dict,
    assessment: list[str],
    voice: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Write a plain-text report with metrics, traits, and assessment."""
    sep = "=" * 60

    with output_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(f"{sep}\n")
        file_handle.write(f"  {_text('report', 'title')}\n")
        file_handle.write(f"  {_text('report', 'file_label')} : {voice.name}\n")
        file_handle.write(f"{sep}\n\n")

        file_handle.write(f"{_text('report', 'raw_metrics_header')}\n")
        file_handle.write(
            f"  {_text('report', 'duration_label'):<22}: {m['duration_s']:.4f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'sample_rate_label'):<22}: {m['sample_rate']}\n"
        )
        file_handle.write(
            f"  {_text('report', 'rms_mean_label'):<22}: {m['rms_mean']:.6f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'rms_std_label'):<22}: {m['rms_std']:.6f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'dynamic_range_label'):<22}: {m['dynamic_range_db']:.2f}\n"
        )

        if m["pitch_extraction_ok"]:
            file_handle.write(
                f"  {_text('report', 'mean_pitch_label'):<22}: {m['pitch_mean_hz']:.2f}\n"
            )
            file_handle.write(
                f"  {_text('report', 'pitch_variance_label'):<22}: {m['pitch_variance']:.2f}\n"
            )
        else:
            file_handle.write(
                f"  {_text('report', 'mean_pitch_label'):<22}: {_text('report', 'pitch_na')}\n"
            )
            file_handle.write(f"  {_text('report', 'pitch_variance_label'):<22}: N/A\n")

        file_handle.write(
            f"  {_text('report', 'voiced_ratio_label'):<22}: {m['voiced_ratio']:.4f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'pause_ratio_label'):<22}: {m['pause_ratio']:.4f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'speaking_pace_label'):<22}: {m['speaking_pace_proxy']:.4f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'spectral_centroid_label'):<22}: {m['spectral_centroid_mean']:.2f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'zcr_label'):<22}: {m['zcr_mean']:.6f}\n"
        )
        file_handle.write(
            f"  {_text('report', 'hf_energy_label'):<22}: {m['hf_energy_ratio']:.6f}\n"
        )
        file_handle.write("\n")

        file_handle.write(f"{_text('report', 'traits_header')}\n")
        for trait, score in traits.items():
            bar_len = int(score * 20)
            gauge = "#" * bar_len + "-" * (20 - bar_len)
            file_handle.write(f"  {_trait_label(trait):<16} : {score:.4f}  [{gauge}]\n")
        file_handle.write("\n")

        file_handle.write(f"{_text('report', 'assessment_header')}\n")
        for line in assessment:
            file_handle.write(f"  {line}\n")
        file_handle.write("\n")

        file_handle.write(f"{sep}\n")
        file_handle.write(f"  {_text('report', 'note_line_1')}\n")
        file_handle.write(f"  {_text('report', 'note_line_2')}\n")
        file_handle.write(f"{sep}\n")


def analyze_voice(voice: pathlib.Path) -> None:
    """Analyze incoming voice artifact."""
    if not voice.exists():
        print("Invalid voice path.", file=sys.stderr)
        sys.exit(1)

    stem = voice.stem
    out_dir = voice.parent
    radar_path = out_dir / f"{stem}_radar.png"
    report_path = out_dir / f"{stem}_report.txt"

    y, sr = load_audio(voice)
    metrics = compute_raw_metrics(y, sr)
    traits = compute_traits(metrics)
    assessment = generate_assessment(metrics, traits)
    plot_radar(traits, voice.name, radar_path)
    write_report(metrics, traits, assessment, voice, report_path)
