# SPDX-License-Identifier: MIT
"""Management of isolated dependency environments for Celune backends."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..paths import backend_environments_dir, core_python_executable

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
    "transformers>=4.56,<5.0.0",
)


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
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
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
        requirements=(*_WORKER_HUGGINGFACE_REQUIREMENTS, "pocket-tts>=2.1.0"),
        backend_module="celune.backends.tts.mini",
        backend_class="Mini",
    ),
    "qwen3": BackendManifest(
        backend_id="qwen3",
        kind="tts",
        requirements=(*_WORKER_HUGGINGFACE_REQUIREMENTS, "faster-qwen3-tts>=0.2.4"),
        backend_module="celune.backends.tts.qwen3",
        backend_class="Qwen3",
    ),
    "dotstts": BackendManifest(
        backend_id="dotstts",
        kind="tts",
        requirements=(
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "dots.tts @ git+https://github.com/celunah/dots.tts",
        ),
        backend_module="celune.backends.tts.dotstts",
        backend_class="DotsTtsMF",
    ),
    "voxcpm2": BackendManifest(
        backend_id="voxcpm2",
        kind="tts",
        requirements=(*_WORKER_HUGGINGFACE_REQUIREMENTS, "voxcpm>=2.0.0"),
        backend_module="celune.backends.tts.voxcpm2",
        backend_class="VoxCPM2",
    ),
    "gpt-sovits": BackendManifest(
        backend_id="gpt-sovits",
        kind="tts",
        requirements=(
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "cn2an",
            "ffmpeg-python",
            "g2p-en",
            "g2pk2",
            "jieba-fast",
            "ko-pron",
            "opencc",
            "peft<0.18.0",
            "pypinyin",
            "pytorch-lightning>=2.4",
            "pyopenjtalk>=0.4.1",
            "rotary-embedding-torch",
            "split-lang",
            "tojyutping",
            "torchmetrics<=1.5",
            "wordsegment",
            "x-transformers",
        ),
        backend_module="celune.backends.tts.gpt_sovits",
        backend_class="GPTSoVITS",
    ),
    "seed-vc": BackendManifest(
        backend_id="seed-vc",
        kind="vc",
        requirements=(
            *_WORKER_HUGGINGFACE_REQUIREMENTS,
            "seed-vc @ git+https://github.com/celunah/seed-vc.git",
        ),
        backend_module="celune.backends.vc.seedvc",
        backend_class="CeluneSeedVCBackend",
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
    """Acquire a small cross-process lock using an exclusively created file."""
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - started >= timeout:
                raise BackendEnvironmentError(
                    f"Timed out waiting for backend environment lock: {path}"
                ) from None
            time.sleep(0.1)
        else:
            os.close(descriptor)
            break

    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


class BackendEnvironmentManager:
    """Create and locate backend environments in Celune's AppData directory."""

    def __init__(
        self,
        root: Optional[Path] = None,
        uv_executable: Optional[str] = None,
        lock_timeout: float = 300.0,
    ) -> None:
        self.root = root or backend_environments_dir(create=True)
        self.uv_executable = uv_executable or shutil.which("uv")
        self.lock_timeout = lock_timeout

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
            raise BackendEnvironmentError(
                "uv is required to install isolated backend environments"
            )

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
                    python_executable
                    or manifest.python
                    or str(core_python_executable()),
                    str(virtualenv),
                )
                install_arguments = [
                    "pip",
                    "install",
                    "--python",
                    str(self._python_path(virtualenv)),
                ]
                if manifest.index_urls:
                    install_arguments.extend(
                        ["--index-url", manifest.index_urls[0]]
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
                f"Backend environment was not created: {environment.root}"
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
        try:
            subprocess.run(
                [self.uv_executable, *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            output = ""
            if isinstance(error, subprocess.CalledProcessError):
                output = (error.stderr or error.stdout or "").strip()
            detail = f": {output}" if output else ""
            raise BackendEnvironmentError(
                f"Could not install backend dependencies{detail}"
            ) from error
