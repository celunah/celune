# SPDX-License-Identifier: Apache-2.0
"""Management of isolated dependency environments for Celune backends."""

import os
import sys
import json
import time
import shutil
import hashlib
import platform
import importlib
import subprocess
from pathlib import Path
from typing import Optional
from collections.abc import Generator
from dataclasses import asdict, dataclass
from contextlib import suppress, contextmanager

from ..i18n import string
from ..paths import backend_environments_dir

__all__ = [
    "BACKEND_MANIFESTS",
    "BackendEnvironment",
    "BackendEnvironmentError",
    "BackendEnvironmentManager",
    "BackendManifest",
    "backend_manifest",
]

_WORKER_HUGGINGFACE_REQUIREMENTS = (
    "huggingface-hub>=0.36,<1.0.0",
    "hf-xet",
    "transformers>=4.56,<5.0.0",
    "lingua-language-detector>=2.2.0,<3.0.0",
    "librosa==0.11.0",
    "llvmlite==0.47.0",
    "numpy",
    "numba==0.65.1",
    "pillow",
    "platformdirs",
    "psutil",
    "sounddevice",
    "soundfile",
    "zstandard",
)
_MAIN_BRANCH_PYTORCH_REQUIREMENTS = (
    "torch==2.11.0+cu128",
    "torchaudio==2.11.0+cu128",
    "torchvision==0.26.0+cu128",
)
_PYTORCH_INDEX_URLS = (
    "https://pypi.org/simple",
    "https://download.pytorch.org/whl/cu128",
)
_UV_OPERATION_TIMEOUT_SECONDS = 900.0
_DEFAULT_BACKEND_PYTHON = "3.13"


@dataclass(frozen=True)
class BackendManifest:
    """Describe one backend's independently installable dependency set."""

    backend_id: str
    kind: str
    requirements: tuple[str, ...]
    backend_module: str
    backend_class: str
    python: Optional[str] = None
    runtime: Optional[str] = None
    index_urls: tuple[str, ...] = ()
    revision: int = 1

    def fingerprint(self) -> str:
        """Return the stable environment fingerprint for this manifest."""
        payload = {
            "manifest": asdict(self),
            "machine": platform.machine().lower(),
            "platform": sys.platform,
            "python": self.python or _DEFAULT_BACKEND_PYTHON,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


class BackendEnvironmentError(RuntimeError):
    """Raised when a backend environment cannot be created or used."""


@dataclass(frozen=True)
class BackendEnvironment:
    """Describe an installed backend environment."""

    manifest: BackendManifest
    root: Path

    @property
    def virtualenv(self) -> Path:
        """Return the virtual environment directory."""
        return self.root / "venv"

    @property
    def python(self) -> Path:
        """Return the environment's Python executable."""
        if os.name == "nt":
            return self.virtualenv / "Scripts" / "python.exe"
        return self.virtualenv / "bin" / "python"

    @property
    def metadata_path(self) -> Path:
        """Return the environment manifest path."""
        return self.root / "manifest.json"

    @property
    def is_ready(self) -> bool:
        """Return whether the environment has a usable interpreter and manifest."""
        return self.python.is_file() and self.metadata_path.is_file()


BACKEND_MANIFESTS = {
    "mini": BackendManifest(
        backend_id="mini",
        kind="tts",
        requirements=(
            *_MAIN_BRANCH_PYTORCH_REQUIREMENTS,
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "pocket-tts>=2.1.0",
        ),
        backend_module="celune.backends.tts.mini",
        backend_class="Mini",
        index_urls=_PYTORCH_INDEX_URLS,
    ),
    "qwen3": BackendManifest(
        backend_id="qwen3",
        kind="tts",
        requirements=(
            *_MAIN_BRANCH_PYTORCH_REQUIREMENTS,
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "faster-qwen3-tts>=0.2.4",
        ),
        backend_module="celune.backends.tts.qwen3",
        backend_class="Qwen3",
        index_urls=_PYTORCH_INDEX_URLS,
    ),
    "dotstts": BackendManifest(
        backend_id="dotstts",
        kind="tts",
        requirements=(
            *_MAIN_BRANCH_PYTORCH_REQUIREMENTS,
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "dots.tts @ git+https://github.com/celunah/dots.tts",
        ),
        backend_module="celune.backends.tts.dotstts",
        backend_class="DotsTtsMF",
        python="3.12",
        index_urls=_PYTORCH_INDEX_URLS,
    ),
    "voxcpm2": BackendManifest(
        backend_id="voxcpm2",
        kind="tts",
        requirements=(
            *_MAIN_BRANCH_PYTORCH_REQUIREMENTS,
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "voxcpm>=2.0.0",
        ),
        backend_module="celune.backends.tts.voxcpm2",
        backend_class="VoxCPM2",
        python="3.12",
        index_urls=_PYTORCH_INDEX_URLS,
    ),
    "gpt-sovits": BackendManifest(
        backend_id="gpt-sovits",
        kind="tts",
        requirements=(
            *_MAIN_BRANCH_PYTORCH_REQUIREMENTS,
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "cn2an",
            "ffmpeg-python",
            "g2p-en",
            "g2pk2",
            "jieba",
            "jieba-fast",
            "ko-pron",
            "matplotlib",
            "opencc",
            "peft<0.18.0",
            "pypinyin",
            "pytorch-lightning>=2.4",
            "pyopenjtalk>=0.4.1",
            "rotary-embedding-torch",
            "split-lang",
            "tojyutping",
            "torchmetrics<=1.5",
            "torchcodec",
            "wordsegment",
            "x-transformers",
        ),
        backend_module="celune.backends.tts.gpt_sovits",
        backend_class="GPTSoVITS",
        index_urls=_PYTORCH_INDEX_URLS,
    ),
    "seed-vc": BackendManifest(
        backend_id="seed-vc",
        kind="vc",
        requirements=(
            *_MAIN_BRANCH_PYTORCH_REQUIREMENTS,
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "seed-vc @ git+https://github.com/celunah/seed-vc.git",
        ),
        backend_module="celune.backends.vc.seedvc",
        backend_class="CeluneSeedVCBackend",
        index_urls=_PYTORCH_INDEX_URLS,
    ),
}


def backend_manifest(backend_id: str) -> BackendManifest:
    """Return the manifest registered for one backend ID.

    Args:
        backend_id: The backend identifier to resolve.

    Returns:
        BackendManifest: The backend's dependency manifest.

    Raises:
        KeyError: If no manifest is registered for ``backend_id``.
    """
    return BACKEND_MANIFESTS[backend_id.strip().lower()]


@contextmanager
def _exclusive_lock(path: Path, timeout: float) -> Generator[None, None, None]:
    """Acquire a process-released cross-process lock backed by one file."""
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt  # pylint: disable=E0401

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            handle.close()
            if time.monotonic() - started >= timeout:
                raise BackendEnvironmentError(
                    string("backends.environment_lock_timeout", path=path)
                ) from None
            time.sleep(0.1)
        else:
            break

    try:
        yield
    finally:
        with suppress(OSError):
            if os.name == "nt":
                import msvcrt  # pylint: disable=E0401

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class BackendEnvironmentManager:
    """Create and locate backend environments in Celune's AppData directory."""

    def __init__(
        self,
        root: Optional[Path] = None,
        uv_executable: Optional[str] = None,
        lock_timeout: float = 300.0,
        uv_timeout: float = _UV_OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        self.root = root or backend_environments_dir(create=True)
        self.uv_executable = uv_executable or shutil.which("uv")
        self.lock_timeout = lock_timeout
        self.uv_timeout = uv_timeout

    def environment_for(self, manifest: BackendManifest) -> BackendEnvironment:
        """Return the deterministic environment location for a manifest."""
        return BackendEnvironment(
            manifest=manifest,
            root=self.root / manifest.backend_id / manifest.fingerprint(),
        )

    def ensure(
        self,
        manifest: BackendManifest,
        python_executable: Optional[str] = None,
    ) -> BackendEnvironment:
        """Install a backend environment if it is not already ready.

        Args:
            manifest: The backend dependency manifest to install.
            python_executable: Optional Python interpreter used to create the venv.

        Returns:
            BackendEnvironment: The ready backend environment.

        Raises:
            BackendEnvironmentError: If uv is unavailable or installation fails.
        """
        environment = self.environment_for(manifest)
        if environment.is_ready:
            return environment

        if self.uv_executable is None:
            raise BackendEnvironmentError(string("backends.uv_required"))

        lock_path = self.root / manifest.backend_id / ".install.lock"
        with _exclusive_lock(lock_path, self.lock_timeout):
            if environment.is_ready:
                return environment

            environment.root.parent.mkdir(parents=True, exist_ok=True)
            temporary_root = environment.root.with_name(
                f"{environment.root.name}.install-{os.getpid()}"
            )
            shutil.rmtree(temporary_root, ignore_errors=True)
            try:
                temporary_root.mkdir(parents=True, exist_ok=True)
                virtualenv = temporary_root / "venv"
                self._run_uv(
                    "venv",
                    "--python",
                    python_executable or manifest.python or _DEFAULT_BACKEND_PYTHON,
                    str(virtualenv),
                )
                install_arguments = [
                    "pip",
                    "install",
                    "--no-config",
                    "--no-cache",
                    "--python",
                    str(self._python_path(virtualenv)),
                ]
                if manifest.index_urls:
                    install_arguments.extend(
                        [
                            "--index-strategy",
                            "unsafe-best-match",
                            "--index-url",
                            manifest.index_urls[0],
                        ]
                        + [
                            item
                            for index_url in manifest.index_urls[1:]
                            for item in ("--extra-index-url", index_url)
                        ]
                    )
                self._run_uv(*install_arguments, *manifest.requirements)
                metadata = {
                    "manifest": asdict(manifest),
                    "fingerprint": manifest.fingerprint(),
                }
                temporary_root.joinpath("manifest.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                temporary_root.rename(environment.root)
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)

        if not environment.is_ready:
            raise BackendEnvironmentError(
                string(
                    "backends.environment_not_created",
                    path=environment.root,
                )
            )
        return environment

    @staticmethod
    def _python_path(virtualenv: Path) -> Path:
        """Return the interpreter path inside a virtual environment directory."""
        if os.name == "nt":
            return virtualenv / "Scripts" / "python.exe"
        return virtualenv / "bin" / "python"

    def _run_uv(self, *arguments: str) -> None:
        """Run one uv operation and convert failures into backend errors."""
        assert self.uv_executable is not None
        environment = os.environ.copy()
        # The compiled launcher and its host environment may carry core Python
        # or package-manager settings into uv. Those settings can constrain
        # backend resolution, so the isolated installer must not inherit them.
        for variable in tuple(environment):
            if variable.startswith(("PIP_", "UV_")):
                environment.pop(variable, None)
        for variable in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
        ):
            environment.pop(variable, None)
        try:
            subprocess.run(
                [self.uv_executable, *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.uv_timeout,
                env=environment,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            output = ""
            if isinstance(error, subprocess.CalledProcessError):
                output = (error.stderr or error.stdout or "").strip()
            elif isinstance(error, subprocess.TimeoutExpired):
                output = string(
                    "backends.uv_timeout",
                    seconds=self.uv_timeout,
                )
            detail = f": {output}" if output else ""
            raise BackendEnvironmentError(
                string("backends.dependencies_install_failed", detail=detail)
            ) from error
