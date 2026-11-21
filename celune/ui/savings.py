# SPDX-License-Identifier: Apache-2.0
"""On-demand speech cost comparisons for the Textual UI."""

from decimal import Decimal, ROUND_HALF_UP

from ..i18n import string
from ..celune import Celune
from ..constants import APP_NAME, COST_EQUIVALENTS


def _format_cost(amount: float) -> str:
    """Return a compact USD estimate rounded to cents or finer below one cent."""
    if amount <= 0.0:
        return "0.00"

    decimal_amount = Decimal(str(amount))
    precision = Decimal("0.0001") if amount < 0.01 else Decimal("0.01")
    formatted = str(decimal_amount.quantize(precision, rounding=ROUND_HALF_UP))
    return formatted.rstrip("0").rstrip(".")


def _format_rate(amount: float) -> str:
    """Return the configured per-minute API price without rounding its precision."""
    return format(Decimal(str(amount)).normalize(), "f")


def format_savings_report(celune: Celune) -> str:
    """Build the localized session and cumulative API cost comparison report.

    Args:
        celune: The active Celune runtime state.

    Returns:
        str: A plain-text report with estimates and pricing caveats.
    """
    session_seconds = max(
        0.0,
        float(getattr(celune, "total_generated_speech_seconds", 0.0)),
    )
    historical_seconds = max(
        0.0,
        float(getattr(celune, "historical_generated_speech_seconds", 0.0)),
    )
    session_minutes = session_seconds / 60.0
    all_time_minutes = (historical_seconds + session_seconds) / 60.0

    rows = tuple(
        (
            model,
            string(
                "commands.savings_amount", cost=_format_cost(session_minutes * rate)
            ),
            string(
                "commands.savings_amount",
                cost=_format_cost(all_time_minutes * rate),
            ),
            string("commands.savings_per_minute", cost=_format_rate(rate)),
        )
        for model, rate in COST_EQUIVALENTS.items()
    )
    headers = (
        string("commands.savings_model"),
        string("commands.savings_session"),
        string("commands.savings_all_time"),
        string("commands.savings_base_price"),
    )
    widths = tuple(
        max((len(headers[index]), *(len(row[index]) for row in rows)))
        for index in range(len(headers))
    )
    header_line = " | ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    )
    row_lines = tuple(
        " | ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(row, widths))
        )
        for row in rows
    )
    separator = "-+-".join("-" * width for width in widths)

    return "\n".join(
        (
            string("commands.savings_title", app_name=APP_NAME),
            string("commands.savings_question", app_name=APP_NAME),
            separator,
            header_line,
            separator,
            *row_lines,
            separator,
            "",
            string("commands.savings_disclaimer_pricing"),
            string("commands.savings_disclaimer_changes"),
            string("commands.savings_disclaimer_fees", app_name=APP_NAME),
        )
    )
