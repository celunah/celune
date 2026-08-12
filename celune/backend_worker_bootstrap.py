# SPDX-License-Identifier: MIT
"""Bootstrap a backend worker before importing the Celune package."""

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Prepare the isolated backend environment before entering the worker module."""
    project_path = Path(__file__).resolve().parent.parent
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
    script_directory = str(Path(__file__).resolve().parent)
    sys.path = [entry for entry in sys.path if entry != script_directory]
    sys.path.insert(0, str(project_path))
    runpy.run_module("celune.backends.worker", run_name="__main__")


if __name__ == "__main__":
    main()
