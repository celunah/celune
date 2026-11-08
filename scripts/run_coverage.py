# SPDX-License-Identifier: Apache-2.0
"""Run coverage.py on Celune for a detailed coverage trace."""

import sys
import runpy
from pathlib import Path

from coverage import Coverage
from rich.console import Console

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

cov = Coverage(source=["celune"], branch=True)
cov.start()

try:
    sys.argv = ["main.py", *sys.argv[1:]]
    runpy.run_path(str(root / "main.py"), run_name="__main__")
except Exception:
    print()
    con = Console()
    con.print_exception()
    sys.exit(1)
except KeyboardInterrupt:
    print()
    con = Console()
    con.print_exception()
    sys.exit(130)
finally:
    print()
    cov.stop()
    cov.save()
    cov.report(skip_empty=True)
