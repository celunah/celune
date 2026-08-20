# SPDX-License-Identifier: MIT
"""Bootstrap a backend worker while preserving its CEDTS channel arguments."""

import sys
import runpy
from pathlib import Path


def main() -> None:
    """Prepare the isolated backend environment before entering the worker module."""
    project_path = Path(__file__).resolve().parent.parent
    script_directory = str(Path(__file__).resolve().parent)
    sys.path = [entry for entry in sys.path if entry != script_directory]
    sys.path.insert(0, str(project_path))
    runpy.run_module("celune.backends.worker", run_name="__main__")


if __name__ == "__main__":
    main()
