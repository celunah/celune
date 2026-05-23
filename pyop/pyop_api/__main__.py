# SPDX-License-Identifier: MIT
"""Command-line entrypoint for the detached PYOP API."""

from __future__ import annotations

import os

import uvicorn

from . import PYOP_HOST, PYOP_PORT


def main() -> None:
    """Start the detached PYOP API server."""
    host = os.getenv("PYOP_HOST", PYOP_HOST)
    port = int(os.getenv("PYOP_PORT", str(PYOP_PORT)))
    uvicorn.run("pyop_api.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
