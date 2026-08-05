# SPDX-License-Identifier: MIT
"""Bootstrap a backend worker before importing the Celune package."""

import runpy
import site
import sys
from pathlib import Path


def main() -> None:
    """Add core fallback packages before entering the worker module."""
    if len(sys.argv) < 2:
        raise SystemExit("base site-packages path is required")
    site.addsitedir(sys.argv[1])
    project_path = Path(__file__).resolve().parent.parent
    script_directory = str(Path(__file__).resolve().parent)
    sys.path = [entry for entry in sys.path if entry != script_directory]
    sys.path.insert(0, str(project_path))
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    runpy.run_module("celune.backends.worker", run_name="__main__")


if __name__ == "__main__":
    main()
