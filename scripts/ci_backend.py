# SPDX-License-Identifier: Apache-2.0
"""Run static code-quality checks for one isolated CEDTS backend."""

import os
import sys
import shutil
import argparse
import importlib
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], environment: dict[str, str]) -> None:
    """Run one static analysis command from the project root."""
    subprocess.run(command, check=True, cwd=_PROJECT_ROOT, env=environment)


def _command(name: str) -> str:
    """Resolve a static analysis executable from the active dev environment."""
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required code-quality command is unavailable: {name}")
    return executable


def _interpreter_details(python: Path) -> tuple[str, Path]:
    """Read the CEDTS interpreter version and pure-Python package directory."""
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import sys, sysconfig; "
                "print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
                "print(sysconfig.get_paths()['purelib'])"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    python_version, site_packages = result.stdout.splitlines()
    return python_version, Path(site_packages)


def _check_backend(backend_id: str, environment_root: Path) -> None:
    """Provision manifest dependencies and statically inspect its adapter."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    environment_module = importlib.import_module("celune.backends.environment")
    manifests = environment_module.BACKEND_MANIFESTS
    if backend_id not in manifests:
        available_backends = ", ".join(sorted(manifests))
        raise ValueError(
            f"Unknown backend {backend_id!r}; available backends: {available_backends}"
        )

    manifest = manifests[backend_id]
    manager = environment_module.BackendEnvironmentManager(
        root=environment_root,
        uv_executable=shutil.which("uv"),
        uv_timeout=2700.0,
    )
    backend_environment = manager.ensure(manifest)
    source_path = _PROJECT_ROOT.joinpath(
        *manifest.backend_module.split(".")
    ).with_suffix(".py")
    if not source_path.is_file():
        raise FileNotFoundError(f"Backend source file does not exist: {source_path}")

    python_version, site_packages = _interpreter_details(backend_environment.python)
    analysis_environment = os.environ.copy()
    analysis_environment["PYTHONPATH"] = os.pathsep.join(
        (str(_PROJECT_ROOT), str(site_packages))
    )
    analysis_environment["PYTHONNOUSERSITE"] = "1"
    for variable in ("PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        analysis_environment.pop(variable, None)

    ruff = _command("ruff")
    pylint = [
        sys.executable,
        "-m",
        "pylint",
        "--errors-only",
        "--disable=not-callable,no-member",
        "--rcfile",
        str(_PROJECT_ROOT / "pyproject.toml"),
        "--ignore-paths=^$",
        "--py-version",
        python_version,
        str(source_path),
    ]
    pyrefly = [
        _command("pyrefly"),
        "check",
        "--python-interpreter-path",
        str(backend_environment.python),
        "--python-version",
        python_version,
        str(source_path),
    ]
    docstring_coverage = [
        _command("docstr-coverage"),
        "-i",
        "-P",
        "--fail-under",
        "80",
        str(source_path),
    ]

    print(
        f"Checking {backend_id} adapter with Python {python_version}; "
        "adapter implementation is not imported or executed."
    )
    _run([ruff, "format", "--check", str(source_path)], analysis_environment)
    _run([ruff, "check", str(source_path)], analysis_environment)
    _run(pylint, analysis_environment)
    _run(pyrefly, analysis_environment)
    try:
        _run(docstring_coverage, analysis_environment)
    except subprocess.CalledProcessError:
        print(
            "::warning::Docstring coverage is below 80%. Try adding more missing docstrings."
        )


def main() -> int:
    """Parse backend CI arguments and run its static checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--environment-root", required=True, type=Path)
    arguments = parser.parse_args()
    _check_backend(arguments.backend, arguments.environment_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
