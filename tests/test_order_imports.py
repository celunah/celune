# SPDX-License-Identifier: MIT
"""Tests for the local import ordering script."""

import sys
import ast
import importlib.util
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "order_imports.py"
SPEC = importlib.util.spec_from_file_location("order_imports_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ORDER_IMPORTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORDER_IMPORTS
SPEC.loader.exec_module(ORDER_IMPORTS)


class OrderImportsTests(TestCase):
    """Verify custom import ordering stays stable."""

    def test_render_import_block_groups_imports_by_runtime_and_support_type(
        self,
    ) -> None:
        """Verify the local sorter matches the expected grouped layout."""
        source = """import os
import gc
import glob
import random
import hashlib
import secrets
import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Protocol
from collections.abc import Iterator
"""
        lines = source.splitlines(keepends=True)
        statements = ORDER_IMPORTS.collect_top_imports(ast.parse(source), lines)

        rendered = "".join(ORDER_IMPORTS.render_import_block(statements))

        self.assertEqual(
            rendered,
            """import os
import gc
import glob
import random
import hashlib
import secrets
import contextlib
from pathlib import Path
from collections.abc import Iterator
from abc import ABC, abstractmethod
from typing import Callable, Optional, Protocol
""",
        )

    def test_render_import_block_keeps_same_package_imports_adjacent(self) -> None:
        """Verify imports from the same package are not split apart."""
        source = """import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import numpy.typing as npt
"""
        lines = source.splitlines(keepends=True)
        statements = ORDER_IMPORTS.collect_top_imports(ast.parse(source), lines)

        rendered = "".join(ORDER_IMPORTS.render_import_block(statements))

        self.assertEqual(
            rendered,
            """import numpy as np
import numpy.typing as npt
import torch
import soundfile as sf
import sounddevice as sd
""",
        )

    def test_render_import_block_keeps_same_from_package_imports_adjacent(self) -> None:
        """Verify repeated from-imports from one package are not split apart."""
        source = """from typing import Optional
from abc import ABC
from typing import Callable
"""
        lines = source.splitlines(keepends=True)
        statements = ORDER_IMPORTS.collect_top_imports(ast.parse(source), lines)

        rendered = "".join(ORDER_IMPORTS.render_import_block(statements))

        self.assertEqual(
            rendered,
            """from abc import ABC
from typing import Callable
from typing import Optional
""",
        )
