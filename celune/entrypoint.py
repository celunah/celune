# SPDX-License-Identifier: MIT
"""CLI entrypoint helpers."""

import contextlib
import datetime
import importlib
import importlib.util
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from collections.abc import Callable

from celune import REVISION, __tagline__, __version__
from celune.constants import APP_NAME, APP_SLUG, NVIDIA_DEVICE_KEYWORDS, ExitCodes
from celune.config import config_log_level, normalize_log_level
from celune.i18n import string
from celune.watchdog import launcher_loss_requested, start_watchdog
from celune.paths import migrate_legacy_app_data, project_root, running_compiled
from celune.terminal import set_terminal_title
from celune.updater import apply_update_and_restart


def _env_flag(name: str) -> bool:
    """Interpret common truthy environment variable values."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "on", "yes", "enabled"}


# refer to the app configuration for details on these parameters
INITIAL_LOG_LEVEL = normalize_log_level(os.getenv("CELUNE_LOG_LEVEL", "info"))
INITIAL_HEADLESS = _env_flag("CELUNE_HEADLESS")
INITIAL_BACKEND = os.getenv("CELUNE_BACKEND")

# these parameters are used by the app CLI and its commands, e.g. 'celune doctor'
LAUNCHED_VIA_LAUNCHER = _env_flag("CELUNE_LAUNCHER")
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = project_root()
SETUP_PATH = PROJECT_ROOT / "setup.py"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "default_config.yaml"
SCRIPT_NAME = "main.py"
EXIT_CODES = ExitCodes
_RUNTIME: Optional[SimpleNamespace] = None
_FORCE_STARTUP_DIAGNOSTICS = False
_STARTUP_DIAGNOSTICS: list[str] = []
_STARTUP_DIAGNOSTIC_SINK: Optional[Callable[[str], None]] = None
_CELUNE_PROCESS_NAMES = frozenset(
    {"celune", "celune-bin", "celune-bin.exe", "celune.appimage", "celune.exe"}
)

start_watchdog()


def _load_ui_test_backend() -> type:
    """Load the lightweight fake backend used by the explicit UI test mode."""
    support = importlib.import_module("tests.support")
    return support.FakeBackend


def _startup_diagnostics_enabled(force: bool = False) -> bool:
    """Return whether pre-UI startup diagnostics should be printed."""
    return (
        force
        or _FORCE_STARTUP_DIAGNOSTICS
        or INITIAL_LOG_LEVEL != "info"
        or _env_flag("CELUNE_BOOT_DIAGNOSTICS")
    )


def _print_startup_diagnostic(message: str, force: bool = False) -> None:
    """Queue or forward one early-startup diagnostic line.

    Args:
        message: Diagnostic text to display during startup.
        force: Whether to include the diagnostic regardless of log level.
    """
    if not _startup_diagnostics_enabled(force=force):
        return

    _STARTUP_DIAGNOSTICS.append(message)
    if _STARTUP_DIAGNOSTIC_SINK is not None:
        _STARTUP_DIAGNOSTIC_SINK(message)


def _flush_startup_diagnostics() -> None:
    """Print queued startup diagnostics for non-interactive startup paths."""
    for message in _STARTUP_DIAGNOSTICS:
        print(string("cli.startup_diagnostic_prefix", message=message), flush=True)
    _STARTUP_DIAGNOSTICS.clear()


def normalize_argv0(argv: Optional[list[str]] = None) -> list[str]:
    """Return argv with the launcher-facing program name normalized when applicable.

    Args:
        argv: The arguments to normalize.

    Returns:
        list[str]: Normalized argument list.
    """
    resolved = list(sys.argv if argv is None else argv)
    if resolved and LAUNCHED_VIA_LAUNCHER:
        resolved[0] = "celune"
    return resolved


def _display_version() -> tuple[str, str]:
    """Return the user-facing version string together with the raw revision marker."""
    return __version__, REVISION


def _print_dependency_setup_help(package_name: str) -> None:
    """Print the shared missing-dependency guidance used by startup paths."""
    print(string("cli.dependency_missing", package_name=package_name))
    print(string("cli.dependency_required", app_name=APP_NAME))
    print()
    print(string("cli.setup_automatically", app_name=APP_NAME))
    print(string("cli.setup_cmd_setup_py"))
    print()
    print(string("cli.setup_with_uv"))
    print(string("cli.setup_cmd_uv_sync"))
    print()
    print(string("cli.install_manually"))
    print(string("cli.setup_cmd_pip_install", package_name=package_name))
    print()

    print(string("cli.full_traceback"))
    if os.name == "nt":
        print(string("cli.traceback_cmd_set_dev"))
        print(string("cli.traceback_cmd_python", script_name=SCRIPT_NAME))
    else:
        print(string("cli.traceback_cmd_dev_python", script_name=SCRIPT_NAME))
    print()


def _load_runtime() -> SimpleNamespace:
    """Import lightweight CLI helpers without importing the engine or UI."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    _print_startup_diagnostic(string("cli.startup_loading_runtime"))

    try:
        import webbrowser

        import psutil
        import yaml

        from celune.config import (
            config_bool,
            config_value,
            env_bool,
            merge_missing_defaults,
        )
        from celune.exceptions import No, UpdateError
        from celune.namedays import has_name_day
        from celune.paths import (
            config_path,
            default_config_path,
            ensure_config_path,
        )
        from celune.ui import SelectMenu
        from celune.updater import check_for_update, update_to_latest
        from celune.utils import detected_ide, indent, supports_ansi, title_case
    except ModuleNotFoundError as package:
        if package.name is not None:
            _print_dependency_setup_help(package.name)

        if INITIAL_LOG_LEVEL != "info":
            with contextlib.suppress(ModuleNotFoundError):
                from rich.traceback import install

                install()

            raise

        sys.exit(EXIT_CODES.EXIT_MISSING_DEPENDENCIES.value)

    _RUNTIME = SimpleNamespace(
        yaml=yaml,
        psutil=psutil,
        webbrowser=webbrowser,
        __version__=__version__,
        REVISION=REVISION,
        __tagline__=__tagline__,
        No=No,
        UpdateError=UpdateError,
        has_name_day=has_name_day,
        check_for_update=check_for_update,
        update_to_latest=update_to_latest,
        SelectMenu=SelectMenu,
        config_bool=config_bool,
        config_value=config_value,
        env_bool=env_bool,
        merge_missing_defaults=merge_missing_defaults,
        config_path=config_path,
        default_config_path=default_config_path,
        ensure_config_path=ensure_config_path,
        detected_ide=detected_ide,
        supports_ansi=supports_ansi,
        indent=indent,
        title_case=title_case,
        ExitCodes=ExitCodes,
    )
    _print_startup_diagnostic(string("cli.startup_runtime_ready"))
    return _RUNTIME


def _load_core_runtime() -> SimpleNamespace:
    """Import the engine and full UI runtime when it is needed."""
    runtime = _load_runtime()
    if hasattr(runtime, "Celune"):
        return runtime

    try:
        from celune.celune import Celune
        from celune.ui import (
            CeluneHeadlessBaseUI,
            CeluneHeadlessUI,
            CeluneTextualUI,
            CeluneUI,
        )
    except ModuleNotFoundError as package:
        if package.name is not None:
            _print_dependency_setup_help(package.name)

        if INITIAL_LOG_LEVEL != "info":
            with contextlib.suppress(ModuleNotFoundError):
                from rich.traceback import install

                install()

            raise

        sys.exit(EXIT_CODES.EXIT_MISSING_DEPENDENCIES.value)

    runtime.Celune = Celune
    runtime.CeluneUI = CeluneUI
    runtime.CeluneHeadlessUI = CeluneHeadlessUI
    runtime.CeluneHeadlessBaseUI = CeluneHeadlessBaseUI
    runtime.CeluneTextualUI = CeluneTextualUI
    _print_startup_diagnostic(string("cli.startup_runtime_ready"))
    return runtime


@dataclass
class DoctorCheck:
    """One environment check reported by `celune doctor`."""

    label: str
    ok: bool
    detail: str
    severity: str = "error"
    hint: Optional[str] = None


def _doctor_status(check: DoctorCheck) -> str:
    """Return a compact status label for one doctor check."""
    if check.ok:
        return "OK"
    if check.severity == "warning":
        return "WARN"
    return "FAIL"


def _doctor_add(
    checks: list[DoctorCheck],
    label: str,
    ok: bool,
    detail: str,
    severity: str = "error",
    hint: Optional[str] = None,
) -> None:
    """Append one doctor check result to the active report."""
    checks.append(
        DoctorCheck(label=label, ok=ok, detail=detail, severity=severity, hint=hint)
    )


def _doctor_import(module_name: str) -> bool:
    """Return whether a module can be resolved without importing the full app."""
    return importlib.util.find_spec(module_name) is not None


def _doctor_distro_name(system_name: str) -> str:
    """Return a human-readable distribution or OS label for the current system."""
    if system_name != "Linux":
        return system_name

    try:
        return platform.freedesktop_os_release().get("NAME", "Linux")
    except OSError:
        return "Linux"


def _doctor_openrgb_path() -> Optional[Path]:
    """Resolve OpenRGB even when it was not added to PATH."""
    for binary_name in ("openrgb", "OpenRGB"):
        binary_path = shutil.which(binary_name)
        if binary_path:
            return Path(binary_path)

    if platform.system() == "Windows":
        candidates: list[Path] = []
        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if base:
                candidates.extend(
                    [
                        Path(base) / "OpenRGB" / "OpenRGB.exe",
                        Path(base) / "OpenRGB" / "openrgb.exe",
                    ]
                )

        scoop = os.environ.get("SCOOP") or str(Path.home() / "scoop")
        candidates.extend(
            [
                Path(scoop) / "apps" / "openrgb" / "current" / "OpenRGB.exe",
                Path(scoop) / "apps" / "openrgb" / "current" / "openrgb.exe",
            ]
        )
    else:
        candidates = [
            Path("/usr/bin/openrgb"),
            Path("/usr/local/bin/openrgb"),
            Path("/opt/OpenRGB/openrgb"),
            Path("/opt/OpenRGB/OpenRGB"),
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _doctor_binary_path(binary_name: str) -> Optional[Path]:
    """Resolve one external runtime binary used by the app."""
    if binary_name == "openrgb":
        return _doctor_openrgb_path()

    binary_path = shutil.which(binary_name)
    return Path(binary_path) if binary_path else None


def _doctor_venv_python() -> Path:
    """Return the interpreter path that the native launcher expects to find."""
    if os.name == "nt":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def _doctor_running_python() -> Path:
    """Return the interpreter path currently running the doctor command."""
    return Path(sys.executable).resolve()


def _doctor_subprocess_python() -> Path:
    """Return the Python executable doctor fixups should invoke."""
    if running_compiled():
        return _doctor_venv_python()

    return _doctor_running_python()


def _doctor_same_path(left: Path, right: Path) -> bool:
    """Return whether two paths refer to the same normalized location."""
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _doctor_config_path() -> Optional[Path]:
    """Resolve the runtime config path when `platformdirs` is available."""
    try:
        from platformdirs import user_data_dir
    except ModuleNotFoundError:
        return None

    return Path(user_data_dir(APP_SLUG, appauthor=False)) / "config.yaml"


def _doctor_detect_backend(torch_module: ModuleType) -> tuple[str, bool]:
    """Mirror the app's backend detection without importing the full runtime."""
    if torch_module.cuda.is_available():
        if getattr(torch_module.version, "hip", None) is not None:
            return "ROCm", False

        if any(
            keyword in str(torch_module.cuda.get_device_name(0)).lower()
            for keyword in NVIDIA_DEVICE_KEYWORDS
        ):
            return "CUDA", True
        return "ZLUDA", True

    if (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return "MPS", False

    return "CPU", False


def _doctor_cuda_architecture(capability: tuple[int, int]) -> str:
    """Mirror Celune's CUDA architecture support mapping for doctor output."""
    major, minor = capability

    if major in [10, 11, 12] and minor == 0:
        return "Blackwell"
    if major == 9 and minor == 0:
        return "Hopper"
    if major == 8 and minor == 9:
        return "Ada Lovelace"
    if major == 8 and minor in [0, 6, 7]:
        return "Ampere"
    if major < 8:
        raise NotImplementedError("capability not supported")

    raise ValueError("invalid capability")


def _doctor_run_compute_test(torch_module: ModuleType, device: str = "cuda") -> str:
    """Mirror the app's CUDA compute smoke test and return the device used."""
    dtype = (
        torch_module.bfloat16
        if torch_module.cuda.is_bf16_supported()
        else torch_module.float16
    )

    x = torch_module.randn(2048, 2048, device=device, dtype=dtype)
    y = torch_module.randn(2048, 2048, device=device, dtype=dtype)

    torch_module.cuda.synchronize()
    z = x @ y

    torch_module.cuda.synchronize()
    for _ in range(25):
        z = x @ y
        z = torch_module.relu(z)

    torch_module.cuda.synchronize()
    check_tensor = z.mean().detach()
    result_device = str(check_tensor.device)

    return result_device


def _doctor_torch_details() -> list[DoctorCheck]:
    """Collect non-startup PyTorch and accelerator diagnostics."""
    checks: list[DoctorCheck] = []

    if not _doctor_import("torch"):
        _doctor_add(
            checks,
            "PyTorch",
            False,
            "Module 'torch' is not installed.",
            hint="Run `python setup.py` or `uv sync`.",
        )
        return checks

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        _doctor_add(
            checks,
            "PyTorch",
            False,
            f"Import failed: {exc}",
            hint="Repair the environment with `celune doctor --fix`.",
        )
        return checks

    cuda_version = getattr(torch.version, "cuda", None)
    build_variant = getattr(torch, "__version__", "<unknown>")
    _doctor_add(
        checks,
        "PyTorch",
        True,
        f"{build_variant} (CUDA build: {cuda_version or 'none'})",
    )

    try:
        backend, usable = _doctor_detect_backend(torch)
    except (RuntimeError, AssertionError) as exc:
        _doctor_add(
            checks,
            "Accelerator backend",
            False,
            f"Probe failed: {exc}",
            hint="Check your GPU drivers, PyTorch, and any compatibility layers.",
        )
        return checks

    if backend == "CUDA":
        _doctor_add(checks, "Accelerator backend", True, "CUDA detected")
    elif backend == "ZLUDA":
        _doctor_add(
            checks,
            "Accelerator backend",
            False,
            "Detected non-NVIDIA CUDA compatibility mode (likely ZLUDA or similar).",
            severity="warning",
            hint=f"{APP_NAME}'s performance may be impacted.",
        )
    elif backend == "ROCm":
        _doctor_add(
            checks,
            "Accelerator backend",
            usable,
            "ROCm detected. CUDA compatibility is required for the main backends.",
            hint="Use a CUDA-compatible environment or the Mini backend for CPU-only usage.",
        )
        return checks
    elif backend == "MPS":
        _doctor_add(
            checks,
            "Accelerator backend",
            usable,
            "MPS detected. Celune does not support MPS execution.",
            hint="Use a CUDA-compatible environment or the Mini backend for CPU-only usage.",
        )
        return checks
    else:
        _doctor_add(
            checks,
            "Accelerator backend",
            usable,
            "No CUDA-capable backend detected.",
            severity="warning",
            hint=f"{APP_NAME} Mini can run on CPU, but other backends need a CUDA-compatible runtime.",
        )
        return checks

    try:
        cuda_available = bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError) as exc:
        _doctor_add(
            checks,
            "CUDA availability",
            False,
            f"Probe failed: {exc}",
            hint="Check your GPU drivers, PyTorch, and any compatibility layers.",
        )
        return checks

    if cuda_available:
        expected = "12.8"
        _doctor_add(
            checks,
            "CUDA runtime",
            cuda_version == expected,
            f"Detected CUDA {cuda_version or 'unknown'}",
            hint=f"{APP_NAME} expects a CUDA 12.8-compatible runtime for the main GPU backends.",
        )

        try:
            device_count = int(torch.cuda.device_count())
        except (RuntimeError, AssertionError) as exc:
            _doctor_add(
                checks,
                "CUDA devices",
                False,
                f"Enumeration failed: {exc}",
                hint="Check whether the current user can access the GPU runtime.",
            )
            return checks

        _doctor_add(
            checks,
            "CUDA devices",
            device_count > 0,
            f"{device_count} detected",
            hint="Install or expose a CUDA-compatible GPU for GPU-based backends."
            "Ensure any compatibility layers work correctly.",
        )

        for index in range(device_count):
            try:
                gpu_name = str(torch.cuda.get_device_name(index))
                major, minor = torch.cuda.get_device_capability(index)
                try:
                    architecture = _doctor_cuda_architecture((major, minor))
                    supported = True
                    detail = f"{gpu_name} ({architecture}, compute capability {major}.{minor})"
                except (ValueError, NotImplementedError):
                    supported = False
                    detail = f"{gpu_name} (compute capability {major}.{minor})"
                hint = (
                    f"{APP_NAME} requires Ampere or newer for the CUDA-only backends."
                    f"If not using NVIDIA, try installing ZLUDA or another compatibility layer."
                    if not supported
                    else None
                )
                _doctor_add(
                    checks,
                    f"GPU {index}",
                    supported,
                    detail,
                    hint=hint,
                )
            except (RuntimeError, AssertionError) as exc:
                _doctor_add(
                    checks,
                    f"GPU {index}",
                    False,
                    f"Probe failed: {exc}",
                    hint="Check driver and CUDA runtime health.",
                )

        if device_count > 0:
            try:
                compute_device = _doctor_run_compute_test(torch)
                _doctor_add(
                    checks,
                    "CUDA compute test",
                    True,
                    f"Succeeded on {compute_device}",
                )
            except (RuntimeError, AssertionError) as exc:
                _doctor_add(
                    checks,
                    "CUDA compute test",
                    False,
                    f"Failed: {exc}",
                    hint="The CUDA runtime was detected, but the selected device is not usable.",
                )
    else:
        _doctor_add(
            checks,
            "CUDA runtime",
            False,
            "PyTorch did not report CUDA availability after backend detection succeeded.",
            hint="Check your GPU drivers, PyTorch, and any compatibility layers.",
        )

    return checks


def _doctor_checks() -> list[DoctorCheck]:
    """Collect a lightweight environment report without starting the app."""
    checks: list[DoctorCheck] = []
    system_name = platform.system()
    distro_name = _doctor_distro_name(system_name)

    _doctor_add(
        checks,
        "Operating system",
        system_name in {"Windows", "Linux"},
        f"{distro_name} ({platform.machine()})",
        hint=f"{APP_NAME} currently supports Windows and Linux only.",
    )

    python_ok = (3, 12) <= sys.version_info < (3, 15)
    python_detail = (
        f"{platform.python_version()} (supported: 3.12, 3.13, or 3.14)"
        if python_ok
        else f"{platform.python_version()} (unsupported: expected 3.12, 3.13, or 3.14)"
    )
    _doctor_add(
        checks,
        "Python",
        python_ok,
        python_detail,
        hint=f"{APP_NAME} currently supports Python 3.12, 3.13, and 3.14.",
    )

    for label, path, hint in (
        (
            "Repository root",
            PROJECT_ROOT,
            "Run the launcher from the cloned repository.",
        ),
        ("setup.py", SETUP_PATH, "Repair uses the repo-local `setup.py` bootstrapper."),
        (
            "default_config.yaml",
            DEFAULT_CONFIG_PATH,
            f"This file ships {APP_NAME}'s bundled defaults.",
        ),
    ):
        _doctor_add(checks, label, path.exists(), str(path), hint=hint)

    version, revision = _display_version()
    version_detail = version if not revision else f"{version} ({revision})"
    _doctor_add(checks, "Version metadata", True, version_detail)

    venv_python = _doctor_venv_python()
    current_python = _doctor_running_python()
    if _doctor_same_path(current_python, venv_python):
        _doctor_add(
            checks,
            "Python environment",
            True,
            f"virtual environment ({current_python})",
        )
    elif venv_python.exists():
        _doctor_add(
            checks,
            "Python environment",
            False,
            f"system interpreter ({current_python})",
            severity="warning",
            hint=f"Prefer the project virtual environment at {venv_python}.",
        )
    else:
        _doctor_add(
            checks,
            "Python environment",
            False,
            f"project virtual environment not found; running {current_python}",
            hint="Run `uv sync` to create or repair a compatible environment.",
        )

    _doctor_add(
        checks,
        "Launcher Python",
        venv_python.exists(),
        str(venv_python),
        hint="Run `uv sync` to create or repair a compatible environment.",
    )

    uv_path = shutil.which("uv")
    _doctor_add(
        checks,
        "uv",
        uv_path is not None,
        uv_path or "not found",
        hint=f"{APP_NAME} setup uses uv to sync Python dependencies.",
    )

    required_imports = [
        ("yaml", "PyYAML", f"{APP_NAME} needs YAML support for config loading."),
        ("psutil", "psutil", f"{APP_NAME} uses psutil for process checks."),
        (
            "platformdirs",
            "platformdirs",
            f"{APP_NAME} stores runtime data in a user data directory.",
        ),
        ("textual", "textual", f"{APP_NAME}'s interactive UI depends on Textual."),
        ("rich", "rich", f"{APP_NAME}'s terminal rendering depends on Rich."),
        (
            "readchar",
            "readchar",
            f"{APP_NAME}'s selection menus depend on readchar.",
        ),
        (
            "lingua",
            "lingua-language-detector",
            f"{APP_NAME} uses Lingua to infer utterance language.",
        ),
        ("torch", "PyTorch", f"{APP_NAME}'s speech backends require PyTorch."),
        (
            "torchaudio",
            "torchaudio",
            f"{APP_NAME}'s audio pipeline requires torchaudio.",
        ),
        (
            "sounddevice",
            "sounddevice",
            f"{APP_NAME} DSP playback requires sounddevice.",
        ),
        ("soundfile", "soundfile", f"{APP_NAME}'s audio writer requires soundfile."),
        (
            "faster_qwen3_tts",
            "faster-qwen3-tts",
            f"{APP_NAME}'s default backend requires faster-qwen3-tts.",
        ),
    ]
    optional_imports = [
        ("torchvision", "torchvision", "Persona vision support uses torchvision."),
        ("pocket_tts", "pocket-tts", f"{APP_NAME} Mini needs pocket-tts."),
        ("dots_tts", "dots.tts", "The dots.tts MeanFlow backend needs dots.tts."),
        ("voxcpm", "voxcpm", "The VoxCPM2 backend needs voxcpm."),
        ("openrgb", "openrgb-python", "Presence lighting needs openrgb-python."),
        ("matplotlib", "matplotlib", "Developer visualizations use matplotlib."),
        ("pedalboard", "pedalboard", "Reverb effects require pedalboard."),
        ("pyrubberband", "pyrubberband", "Voice speed controls require pyrubberband."),
    ]

    for module_name, package_name, hint in required_imports:
        available = _doctor_import(module_name)
        _doctor_add(
            checks,
            f"Python package: {package_name}",
            available,
            "installed" if available else "missing",
            hint=hint,
        )

    for module_name, package_name, hint in optional_imports:
        available = _doctor_import(module_name)
        _doctor_add(
            checks,
            f"Python package: {package_name}",
            available,
            "installed" if available else "missing",
            severity="warning",
            hint=hint,
        )

    required_binaries = [("sox", "SoX audio processing is required.")]
    if system_name == "Windows":
        required_binaries.extend(
            [
                ("rubberband", "Rubber Band CLI is required for time stretching."),
                ("openrgb", f"OpenRGB powers {APP_NAME}'s presence lighting features."),
            ]
        )
    elif distro_name in {"Ubuntu", "Debian"}:
        required_binaries.append(
            ("rubberband", "Rubber Band CLI is required for time stretching.")
        )
    elif distro_name == "Arch Linux":
        required_binaries.extend(
            [
                ("rubberband", "Rubber Band CLI is required for time stretching."),
                ("openrgb", f"OpenRGB powers {APP_NAME}'s presence lighting features."),
            ]
        )

    for binary_name, hint in required_binaries:
        binary_path = _doctor_binary_path(binary_name)
        _doctor_add(
            checks,
            f"Binary: {binary_name}",
            binary_path is not None,
            str(binary_path) if binary_path else "not found",
            hint=hint,
        )

    config_path = _doctor_config_path()
    if config_path is None:
        _doctor_add(
            checks,
            "Runtime config path",
            False,
            "platformdirs is not installed, so the runtime config location is unavailable.",
            hint="Install platformdirs to restore runtime path handling.",
        )
    else:
        parent_exists = config_path.parent.exists()
        _doctor_add(
            checks,
            "Runtime config path",
            True,
            str(config_path),
        )
        _doctor_add(
            checks,
            "Runtime config file",
            config_path.exists(),
            "present" if config_path.exists() else "not created yet",
            severity="warning",
            hint=(
                f"{APP_NAME} will create this file on first successful startup."
                if parent_exists
                else f"The runtime data directory has not been created yet. Run {APP_NAME} at least once to create it."
            ),
        )

    checks.extend(_doctor_torch_details())
    return checks


doctor_running_python = _doctor_running_python
doctor_torch_details = _doctor_torch_details
doctor_checks = _doctor_checks


def run_doctor(argv: list[str]) -> int:
    """Run `celune doctor` and optionally repair the environment.

    Args:
        argv: Arguments passed to `celune doctor`.

    Returns:
        int: The exit code for `celune doctor` to exit with.
    """
    doctor_args = argv[2:]
    fix = False
    if doctor_args:
        if doctor_args == ["--fix"]:
            fix = True
        else:
            print(string("cli.doctor_usage", program=argv[0]))
            print(string("cli.doctor_description", app_name=APP_NAME))
            print(string("cli.doctor_fix_help"))
            return EXIT_CODES.EXIT_UNKNOWN_ARGS.value

    print(string("cli.doctor_checking", app_name=APP_NAME))
    print()

    checks = _doctor_checks()
    failures = 0
    warnings_count = 0
    passes = 0

    for check in checks:
        status = _doctor_status(check)
        print(f"[{status}] {check.label}: {check.detail}")
        if check.hint and not check.ok:
            print(f"       {string('cli.doctor_hint', hint=check.hint)}")

        if check.ok:
            passes += 1
        elif check.severity == "warning":
            warnings_count += 1
        else:
            failures += 1

    print()
    print(
        string(
            "cli.doctor_summary",
            passes=passes,
            warnings_count=warnings_count,
            warning_suffix="" if warnings_count == 1 else "s",
            failures=failures,
        )
    )
    print()

    if failures == 0 and warnings_count == 0:
        print(string("cli.doctor_ready", app_name=APP_NAME))
    elif failures == 0 and warnings_count > 0:
        print(string("cli.doctor_performance_impacted", app_name=APP_NAME))
        print(string("cli.doctor_rerun_fix"))
        print(string("cli.doctor_fix_limits"))
    else:
        print(string("cli.doctor_will_not_work", app_name=APP_NAME))
        print(string("cli.doctor_rerun_fix"))
        print(string("cli.doctor_fix_limits"))

    if fix:
        print()
        print(string("cli.doctor_attempting_fix"))
        try:
            result = subprocess.run(
                [str(_doctor_subprocess_python()), str(SETUP_PATH)],
                cwd=PROJECT_ROOT,
                check=False,
            )
        except OSError as exc:
            print(string("cli.fix_failed", error=exc))
            return EXIT_CODES.EXIT_FAILURE.value
        return result.returncode

    return EXIT_CODES.EXIT_FAILURE.value if failures else 0


def handle_config(command_args: list[str], prog_name: str) -> None:
    """Handle `celune config` commands lazily.

    Args:
        command_args: Current command's arguments.
        prog_name: The name of the program.
    """
    runtime = _load_runtime()

    if len(command_args) == 1:
        if command_args[0] == "view":
            if not runtime.config_path().exists():
                print(string("cli.config_not_created", app_name=APP_NAME))
                print(string("cli.run_once_to_create_config", app_name=APP_NAME))
                sys.exit(EXIT_CODES.EXIT_FAILURE.value)

            print(string("cli.current_config", app_name=APP_NAME))
            print()

            try:
                with open(runtime.config_path(), encoding="utf-8") as cfg:
                    print(cfg.read().strip())
            except PermissionError:
                print(string("cli.config_could_not_be_read", app_name=APP_NAME))
                sys.exit(EXIT_CODES.EXIT_FAILURE.value)
        elif command_args[0] == "edit":
            if not runtime.config_path().exists():
                print(string("cli.config_not_created", app_name=APP_NAME))
                print(string("cli.run_once_to_create_config", app_name=APP_NAME))
                sys.exit(EXIT_CODES.EXIT_FAILURE.value)

            try:
                runtime.webbrowser.open(runtime.config_path())
            except PermissionError:
                print(string("cli.config_could_not_be_read", app_name=APP_NAME))
                sys.exit(EXIT_CODES.EXIT_FAILURE.value)
        else:
            print(string("cli.invalid_argument"))
            print()
            print(string("cli.config_usage", program=prog_name))
            print(string("cli.config_description", app_name=APP_NAME))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
    elif len(command_args) > 1:
        print(string("cli.too_many_arguments"))
        print()
        print(string("cli.config_usage", program=prog_name))
        print(string("cli.config_description", app_name=APP_NAME))
        sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
    else:
        print(string("cli.no_argument_given"))
        print()
        print(string("cli.config_usage", program=prog_name))
        print(string("cli.config_description", app_name=APP_NAME))
        sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)


def _close_existing_celune_processes(runtime: SimpleNamespace) -> None:
    """Prompt before terminating other Celune processes found by the launcher."""
    current_pids = {os.getpid(), os.getppid()}
    launcher_pid = os.getenv("CELUNE_LAUNCHER_PID")
    if launcher_pid:
        with contextlib.suppress(ValueError):
            current_pids.add(int(launcher_pid))

    existing_processes = []
    for proc in runtime.psutil.process_iter():
        with contextlib.suppress(
            runtime.psutil.AccessDenied,
            runtime.psutil.NoSuchProcess,
            runtime.psutil.ZombieProcess,
        ):
            if proc.pid in current_pids:
                continue
            if str(proc.name()).casefold() in _CELUNE_PROCESS_NAMES:
                existing_processes.append(proc)

    if not existing_processes:
        return

    print(string("cli.already_running", app_name=APP_NAME))
    choice = runtime.SelectMenu(
        [string("cli.yes"), string("cli.no")],
        [True, False],
        string(
            "cli.already_running_close_existing",
            app_name=APP_NAME,
        ),
    ).start()

    if choice is not True:
        sys.exit(EXIT_CODES.EXIT_ALREADY_RUNNING.value)

    for proc in existing_processes:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (
            runtime.psutil.NoSuchProcess,
            runtime.psutil.ZombieProcess,
        ):
            continue
        except (
            runtime.psutil.AccessDenied,
            runtime.psutil.TimeoutExpired,
        ):
            print(
                string(
                    "cli.already_running_failed_closing",
                    app_name=APP_NAME,
                )
            )
            sys.exit(EXIT_CODES.EXIT_ALREADY_RUNNING.value)


def start(
    log_level: Optional[str] = None,
    testing: bool = False,
) -> None:
    """Instantiate and start the app.

    Args:
        log_level: Optional startup log level override.
        testing: Whether the app should be started in UI test mode.

    Raises:
        No: Raised on Celune's name day unless explicitly overridden.
        Exception: Re-raised after printing a traceback in developer mode.
    """
    global _FORCE_STARTUP_DIAGNOSTICS
    global _STARTUP_DIAGNOSTIC_SINK

    _FORCE_STARTUP_DIAGNOSTICS = log_level not in {None, "info"}
    _print_startup_diagnostic(string("cli.startup_begin", app_name=APP_NAME))
    runtime = _load_runtime()
    active_log_level = normalize_log_level(log_level or INITIAL_LOG_LEVEL)

    try:
        migrate_legacy_app_data()
        if testing:
            runtime = _load_core_runtime()
            _print_startup_diagnostic(string("cli.startup_creating_ui"))
            _print_startup_diagnostic(string("cli.startup_handing_off_ui"))
            ui = runtime.CeluneUI(startup_messages=_STARTUP_DIAGNOSTICS)
            _STARTUP_DIAGNOSTIC_SINK = ui.receive_startup_diagnostic
            _print_startup_diagnostic(string("cli.startup_preparing_core"))
            celune = runtime.Celune(config={}, backend=_load_ui_test_backend())
            ui.celune = celune
            ui.prepare_theme()
            try:
                ui.run()
            finally:
                _STARTUP_DIAGNOSTIC_SINK = None
            sys.exit(EXIT_CODES.EXIT_SUCCESS.value)
        if runtime.supports_ansi():
            set_terminal_title(
                (
                    APP_NAME,
                    string("osc.state_initializing"),
                    string("osc.action_starting"),
                )
            )
        date = datetime.datetime.now(datetime.UTC)
        if runtime.has_name_day("Celine", date) and not runtime.env_bool(
            "CELUNE_OVERRIDE_CELINE_DAY"
        ):
            raise runtime.No

        ide = runtime.detected_ide()
        if ide is not None:
            print(string("cli.running_from_ide", app_name=APP_NAME, ide=ide))
            print(string("cli.ide_terminals_differ"))
            print()
            time.sleep(5)

        bundled_default_config_path = runtime.default_config_path()
        active_config_path, created_config = runtime.ensure_config_path(
            active_path=runtime.config_path(create_parent=True),
            default_path=bundled_default_config_path,
        )
        if created_config:
            print(string("cli.config_created", app_name=APP_NAME))

        with open(active_config_path, encoding="utf-8") as cfg:
            config = runtime.yaml.safe_load(cfg)
        with open(bundled_default_config_path, encoding="utf-8") as cfg:
            default_config = runtime.yaml.safe_load(cfg)

        if not isinstance(default_config, dict):
            default_config = {}
        if not isinstance(config, dict):
            config = {}

        config, config_updated = runtime.merge_missing_defaults(config, default_config)
        if config_updated:
            with open(active_config_path, "w", encoding="utf-8") as cfg:
                runtime.yaml.safe_dump(config, cfg, sort_keys=False)
            print(string("cli.config_updated_defaults", app_name=APP_NAME))

        active_log_level = normalize_log_level(
            log_level if log_level is not None else config_log_level(config)
        )
        headless = runtime.config_bool(config, "CELUNE_HEADLESS", "headless")
        configured_backend = INITIAL_BACKEND or runtime.config_value(config, "backend")
        backend = configured_backend if isinstance(configured_backend, str) else None

        if not headless and runtime.supports_ansi():
            update = runtime.check_for_update()
            if update:
                latest_label = f"{APP_NAME} {update.latest_version}"
                if not update.latest_tag:
                    warnings.warn(
                        string("cli.update_info_incomplete"),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    latest_label = APP_NAME

                choice = runtime.SelectMenu(
                    [string("cli.update_choice_yes"), string("cli.update_choice_no")],
                    [True, False],
                    "\n".join(
                        [
                            string("cli.update_found"),
                            string(
                                "cli.update_version_summary",
                                app_name=APP_NAME,
                                local_version=update.local_version,
                                local_revision=update.local_revision,
                                latest_label=latest_label,
                                latest_revision=update.latest_revision,
                            ),
                            string("cli.update_prompt"),
                        ]
                    ),
                ).start()

                if choice:
                    if running_compiled():
                        print(
                            string(
                                "cli.launcher_apply_artifact",
                                app_name=APP_NAME,
                            )
                        )
                        time.sleep(2)
                        sys.exit(runtime.ExitCodes.EXIT_PENDING_UPDATE.value)

                    print(string("cli.updating", app_name=APP_NAME))
                    try:
                        runtime.update_to_latest()
                    except runtime.UpdateError as exc:
                        detail = runtime.title_case(str(exc))
                        print(
                            string(
                                "cli.update_failed", app_name=APP_NAME, detail=detail
                            )
                        )
                        print(string("cli.continuing_current_version"))
                        time.sleep(5)
                    else:
                        print(string("cli.update_success_restart", app_name=APP_NAME))
                        time.sleep(5)
                        sys.exit(runtime.ExitCodes.EXIT_PENDING_UPDATE.value)
        elif runtime.check_for_update() and not runtime.supports_ansi():
            print(string("cli.no_ansi"))
            if running_compiled():
                print(string("cli.request_refresh_binaries"))
                time.sleep(2)
                sys.exit(runtime.ExitCodes.EXIT_PENDING_UPDATE.value)

            print(string("cli.apply_update_noninteractive"))
            try:
                runtime.update_to_latest()
            except runtime.UpdateError as exc:
                detail = runtime.title_case(str(exc))
                print(string("cli.update_failed", app_name=APP_NAME, detail=detail))
                print(string("cli.continuing_current_version"))
                time.sleep(5)
            else:
                print(string("cli.update_success_restart", app_name=APP_NAME))
                time.sleep(5)
                sys.exit(runtime.ExitCodes.EXIT_PENDING_UPDATE.value)

        if not runtime.env_bool("CELUNE_LAUNCHER"):
            launcher_exe = "celune.exe" if os.name == "nt" else "celune.appimage"
            print(string("cli.not_via_launcher", app_name=APP_NAME))
            print()
            print(string("cli.suppress_message_run_with", app_name=APP_NAME))
            print(runtime.indent(f"{launcher_exe}", spaces=4))
            print()
            print(string("cli.or_set_env_var"))
            print(runtime.indent("CELUNE_LAUNCHER=1", spaces=4))
            time.sleep(5)
        else:
            _close_existing_celune_processes(runtime)

        if not headless and runtime.supports_ansi():
            from celune.ui import CeluneUI

            def prepare_interactive_runtime():
                """Construct the engine inside the already-mounted UI worker."""
                runtime = _load_core_runtime()
                _print_startup_diagnostic(string("cli.startup_preparing_core"))
                return runtime.Celune(
                    tts_backend=backend,
                    log_callback=ui.tts_log,
                    status_callback=ui.safe_status,
                    error_callback=ui.error,
                    idle_callback=ui.tts_idle,
                    queue_avail_callback=ui.tts_queue_avail,
                    voice_changed_callback=ui.tts_voice_changed,
                    change_input_state_callback=ui.change_input_state,
                    change_voice_lock_state_callback=ui.change_voice_lock_state,
                    progress_callback=ui.safe_progress,
                    caption_progress_callback=ui.safe_caption_progress,
                    caption_callback=ui.tts_caption,
                    caption_timing_callback=ui.tts_caption_timing,
                    log_level=active_log_level,
                    config=config,
                )

            _print_startup_diagnostic(string("cli.startup_creating_ui"))
            ui = CeluneUI(
                startup_loader=prepare_interactive_runtime,
                startup_messages=_STARTUP_DIAGNOSTICS,
            )
            _STARTUP_DIAGNOSTIC_SINK = ui.receive_startup_diagnostic
            try:
                ui.run()
            finally:
                _STARTUP_DIAGNOSTIC_SINK = None
        elif headless:
            runtime = _load_core_runtime()
            _print_startup_diagnostic(string("cli.startup_preparing_headless"))
            ui_headless = runtime.CeluneHeadlessUI(config)
            _print_startup_diagnostic(string("cli.startup_preparing_core"))
            celune = runtime.Celune(
                tts_backend=backend,
                log_callback=ui_headless.headless_log,
                error_callback=ui_headless.headless_error,
                log_level=active_log_level,
                config=config,
            )
            ui_headless.celune = celune
            _flush_startup_diagnostics()

            if not celune.load():
                print(string("cli.could_not_initialize", app_name=APP_NAME))
                celune.close()
                time.sleep(5)
                sys.exit(runtime.ExitCodes.EXIT_FAILURE.value)

            print(string("cli.running_headless", app_name=APP_NAME))
            print(string("cli.headless_extensions_only", app_name=APP_NAME))
            ui_headless.run()
        else:
            _flush_startup_diagnostics()
            print(string("cli.no_ansi"))
            print(string("cli.cannot_start_normal_mode", app_name=APP_NAME))
            print(string("cli.hint"))
            print(runtime.indent(string("cli.try_another_terminal"), spaces=4))
            time.sleep(5)
            sys.exit(runtime.ExitCodes.EXIT_NO_ANSI.value)

        if launcher_loss_requested():
            sys.exit(runtime.ExitCodes.EXIT_LAUNCHER_LOST.value)
    except Exception as exc:
        if exc.__class__ != runtime.No:
            stdout = getattr(sys.stdout, "underlying_stdout", sys.stdout)
            stderr = getattr(sys.stderr, "underlying_stderr", sys.stderr)
            sys.stdout = stdout
            sys.stderr = stderr

            print(string("cli.internal_error_running", app_name=APP_NAME))
            if active_log_level != "info":
                with contextlib.suppress(ModuleNotFoundError):
                    from rich.traceback import install

                    install()

                raise
            print(str(exc) or string("cli.no_error_description"))
            print(string("cli.full_traceback_title"))
            if os.name == "nt":
                print(string("cli.traceback_cmd_set_dev"))
                print(
                    runtime.indent(
                        string(
                            "cli.traceback_cmd_python", script_name=SCRIPT_NAME
                        ).strip(),
                        spaces=4,
                    )
                )
            else:
                print(
                    runtime.indent(
                        string(
                            "cli.traceback_cmd_dev_python",
                            script_name=SCRIPT_NAME,
                        ).strip(),
                        spaces=4,
                    )
                )
            print()
            print(string("cli.additional_debugging"))
            print(runtime.indent(string("cli.set_dev_true"), spaces=4))
            sys.exit(runtime.ExitCodes.EXIT_FAILURE.value)

        print(string("cli.celine_day_1"))
        print(string("cli.celine_day_2"))
        print()
        print(string("cli.hint"))
        print(runtime.indent(string("cli.try_again_tomorrow"), spaces=4))
        print(string("cli.or_set_env_var"))
        print(runtime.indent("CELUNE_OVERRIDE_CELINE_DAY=1", spaces=4))
        time.sleep(5)
        sys.exit(
            runtime.ExitCodes.EXIT_CELINE_DAY.value
            if random.uniform(0, 1) < 0.5
            else runtime.ExitCodes.EXIT_CELINE_DAY_SIX_SEVEN.value
        )
    finally:
        _FORCE_STARTUP_DIAGNOSTICS = False


def main(argv: Optional[list[str]] = None) -> None:
    """Process arguments and perform an appropriate action.

    Args:
        argv: Arguments to be processed.
    """
    resolved_argv = normalize_argv0(argv)
    args = resolved_argv[1:]

    if args and args[0] == "__apply_update":
        if len(args) < 3:
            print(string("cli.apply_update_usage"))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)

        try:
            parent_pid = int(args[1])
        except ValueError:
            print(string("cli.invalid_launcher_pid"))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)

        launcher_path = Path(args[2]).resolve()
        try:
            sys.exit(apply_update_and_restart(parent_pid, launcher_path, args[3:]))
        except Exception as exc:
            print(
                string(
                    "cli.apply_launcher_update_failed",
                    app_name=APP_NAME,
                    error=exc,
                )
            )
            sys.exit(EXIT_CODES.EXIT_FAILURE.value)

    if not args:
        start()
    elif args[0] in {"start", "run"}:
        allowed_args = {"--verbose", "-v", "--debug", "--test", "-t"}
        value_args = [arg for arg in args[1:] if arg.startswith("--log-level=")]
        if any(arg not in allowed_args for arg in args[1:] if arg not in value_args):
            print(string("cli.invalid_argument"))
            print()
            print(string("cli.start_usage", program=resolved_argv[0], command=args[0]))
            print(string("cli.start_description", app_name=APP_NAME))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
        requested_levels = [
            arg.partition("=")[2] for arg in value_args if arg.partition("=")[2]
        ]
        if len(requested_levels) > 1:
            print(string("cli.invalid_argument"))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
        if (
            requested_levels
            and normalize_log_level(requested_levels[0]) != requested_levels[0].lower()
        ):
            print(string("cli.invalid_argument"))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
        requested_log_level = (
            requested_levels[0]
            if requested_levels
            else "debug"
            if "--debug" in args[1:]
            else "verbose"
            if any(arg in {"--verbose", "-v"} for arg in args[1:])
            else None
        )
        testing = any(arg in {"--test", "-t"} for arg in args[1:])
        start(log_level=requested_log_level, testing=testing)
    elif args[0] == "config":
        handle_config(args[1:], resolved_argv[0])
    elif args[0] == "doctor":
        if len(args) > 1 and args[1] != "--fix":
            print(string("cli.invalid_argument"))
            print()
            print(string("cli.doctor_usage", program=resolved_argv[0]))
            print(string("cli.doctor_description", app_name=APP_NAME))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
        sys.exit(run_doctor(resolved_argv))
    elif args[0] in {"help", "--help", "-h"}:
        if len(args) > 1:
            print(string("cli.too_many_arguments"))
            print()
            print(string("cli.help_usage", program=resolved_argv[0]))
            print(string("cli.help_description"))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)

        # HACK: tabs are a quick and dirty alignment trick
        # they are not guaranteed to work in all terminals equally well
        print(string("cli.help_main_usage", program=resolved_argv[0]))
        print()
        print(string("cli.help_available_commands"))
        print(string("cli.help_start", app_name=APP_NAME))
        print(string("cli.help_config", app_name=APP_NAME))
        print(string("cli.help_doctor", app_name=APP_NAME))
        print(string("cli.help_help"))
        print(string("cli.help_version", app_name=APP_NAME))
        print()
        print(string("cli.help_parameter_note", program=resolved_argv[0]))
        print(string("cli.help_subcommand_note", program=resolved_argv[0]))
        print()
        print(string("cli.help_default_start", app_name=APP_NAME))
    elif args[0] in {"version", "--version", "-v"}:
        if len(args) > 1:
            print(string("cli.too_many_arguments"))
            print()
            print(string("cli.version_usage", program=resolved_argv[0]))
            print(string("cli.version_description", app_name=APP_NAME))
            sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)

        version, revision = _display_version()
        if revision:
            print(
                f"{APP_NAME} {version.split('+', maxsplit=1)[0]} ({revision.rstrip('*')})"
            )
        else:
            print(f"{APP_NAME} {version}")
        print(__tagline__)

        if "dirty" in revision:
            print()
            print(string("cli.modified_version_note", app_name=APP_NAME))
    else:
        print(string("cli.unknown_command_or_argument"))
        print(string("cli.run_help_hint", program=resolved_argv[0]))
        sys.exit(EXIT_CODES.EXIT_UNKNOWN_ARGS.value)
