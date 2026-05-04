"""Color utility helpers."""

import random


def random_hex() -> str:
    """Return a random six-digit hex color."""
    hex_numbers = []
    for _ in range(6):
        hex_numbers.append(random.choice("123456789abcdef"))

    return f"#{''.join(hex_numbers)}"
