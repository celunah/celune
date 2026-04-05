"""Celune common utility functions."""

import subprocess


def get_revision() -> str:
    """Get current Git repo revision."""
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = "*" if status else ""
        return f"{rev}{dirty}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
