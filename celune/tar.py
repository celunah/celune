# SPDX-License-Identifier: Apache-2.0
"""Command-line operations for inspecting and editing CEVOICE packs."""

import os
import re

# noinspection PyProtectedMember
from typing import Optional
from pathlib import Path, PurePosixPath, PureWindowsPath

from .cevoice import (
    SUPPORTED_PERSONA_FILENAMES,
    CEVoice,
    CEVoiceError,
    add_cevoice_entries,
    bundle_character_name,
    remove_cevoice_entries,
)
from .i18n import string
from .constants import ExitCodes


def handle_ls_command(arguments: list[str], program: str) -> int:
    """List a CEVOICE pack after validating all of its entries."""
    if len(arguments) != 1:
        return _syntax_error(program, "ls")
    return _list_pack(Path(arguments[0]))


def handle_tar_command(arguments: list[str], program: str) -> int:
    """Run one pack inspection, extraction, or mutation operation."""
    if arguments == ["--help"]:
        _print_help(program)
        return 0
    if not arguments:
        return _syntax_error(program, "tar")

    operation = arguments[0]
    insertion_position: Optional[int] = None
    if operation in {"-tf", "-xf", "-rf", "-df"}:
        pass
    elif re.fullmatch(r"-af[0-9]+", operation):
        try:
            insertion_position = int(operation[3:])
        except ValueError:
            return _syntax_error(program, "tar")
        if insertion_position < 1:
            return _syntax_error(program, "tar")
        operation = "-af"
    else:
        return _syntax_error(program, "tar")

    values = arguments[1:]
    if operation == "-tf":
        if len(values) != 1:
            return _syntax_error(program, "tar")
        return _list_pack(Path(values[0]))
    if operation == "-xf":
        if not 1 <= len(values) <= 2:
            return _syntax_error(program, "tar")
        return _extract_pack(
            Path(values[0]), Path(values[1]) if len(values) == 2 else None
        )
    if operation == "-rf":
        if len(values) < 2:
            return _syntax_error(program, "tar")
        return _remove_pack_entries(Path(values[0]), values[1:], program, indexed=False)
    if operation == "-df":
        if len(values) < 2:
            return _syntax_error(program, "tar")
        return _remove_pack_entries(Path(values[0]), values[1:], program, indexed=True)
    if operation == "-af":
        if len(values) < 2:
            return _syntax_error(program, "tar")
        return _add_pack_entries(
            Path(values[0]),
            values[1:],
            program,
            insertion_position,
        )
    return _syntax_error(program, "tar")


def _list_pack(path: Path) -> int:
    """Validate and print the ordered entries in one pack."""
    try:
        bundle = _open_validated_pack(path)
        print(string("cli.pack_name", name=bundle.path.name))
        character_name = bundle_character_name(bundle)
        if character_name:
            print(string("cli.pack_character", name=character_name))
        for index, entry in enumerate(bundle.entries, start=1):
            print(
                string(
                    "cli.pack_entry",
                    index=index,
                    name=entry.name,
                    size=entry.size,
                )
            )
        return 0
    except CEVoiceError:
        print(string("cli.pack_invalid"))
    except OSError:
        print(string("cli.pack_io_error"))
    return 1


def _extract_pack(path: Path, destination: Optional[Path]) -> int:
    """Extract a validated pack without replacing any existing destination."""
    try:
        bundle = _open_validated_pack(path)
        root = (destination or Path.cwd()).resolve()
        targets: list[tuple[Path, str]] = []
        normalized_targets: set[str] = set()
        for entry in bundle.entries:
            windows_name = PureWindowsPath(entry.name)
            relative_name = PurePosixPath(entry.name)
            if (
                windows_name.drive
                or windows_name.is_absolute()
                or not relative_name.parts
                or any(part in {"", ".", ".."} for part in relative_name.parts)
            ):
                print(string("cli.pack_invalid"))
                return 1
            target = root.joinpath(*relative_name.parts)
            resolved_target = target.resolve(strict=False)
            if resolved_target != root and root not in resolved_target.parents:
                print(string("cli.pack_invalid"))
                return 1
            normalized = os.path.normcase(str(target))
            if normalized in normalized_targets:
                print(string("cli.pack_exists"))
                return 1
            normalized_targets.add(normalized)
            if target.exists() or target.is_symlink():
                print(string("cli.pack_exists"))
                return 1
            if any(
                parent.exists() and not parent.is_dir()
                for parent in target.parents
                if parent != root and root in parent.parents
            ):
                print(string("cli.pack_exists"))
                return 1
            targets.append((target, entry.name))

        for target, _ in targets:
            if any(
                os.path.normcase(str(parent)) in normalized_targets
                for parent in target.parents
                if parent != root and root in parent.parents
            ):
                print(string("cli.pack_exists"))
                return 1

        if root.exists() and not root.is_dir():
            print(string("cli.pack_io_error"))
            return 1
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        try:
            for target, name in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    written.append(target)
                    output.write(bundle.read_entry(name))
        except OSError:
            for target in written:
                target.unlink(missing_ok=True)
            raise
        return 0
    except CEVoiceError:
        print(string("cli.pack_invalid"))
    except OSError:
        print(string("cli.pack_io_error"))
    return 1


def _add_pack_entries(
    path: Path,
    source_names: list[str],
    program: str,
    insertion_position: Optional[int],
) -> int:
    """Append or insert source files into a validated pack."""
    try:
        bundle = _open_validated_pack(path)
    except CEVoiceError:
        print(string("cli.pack_invalid"))
        return 1
    except OSError:
        print(string("cli.pack_io_error"))
        return 1

    if insertion_position is not None and insertion_position > len(bundle.entries) + 1:
        return _syntax_error(program, "tar")
    new_names: list[str] = []
    for source_name in source_names:
        source = Path(source_name)
        if not source.is_file():
            print(string("cli.pack_missing_file", path=source_name))
            return 1
        suffix = source.suffix.lower()
        if suffix not in {".wav", ".pt", ".md"} or (
            suffix == ".md" and source.name not in SUPPORTED_PERSONA_FILENAMES
        ):
            print(string("cli.pack_unsupported_file", name=source.name))
            return 1
        new_names.append(_added_entry_name(source))
    existing_names = {entry.name.casefold() for entry in bundle.entries}
    seen_names: set[str] = set()
    for name in new_names:
        if name.casefold() in existing_names or name.casefold() in seen_names:
            print(string("cli.pack_duplicate", name=name))
            return 1
        seen_names.add(name.casefold())
    try:
        add_cevoice_entries(
            path, [Path(name) for name in source_names], insertion_position
        )
        return 0
    except CEVoiceError:
        print(string("cli.pack_update_failed"))
    except OSError:
        print(string("cli.pack_io_error"))
    return 1


def _remove_pack_entries(
    path: Path,
    selectors: list[str],
    program: str,
    *,
    indexed: bool,
) -> int:
    """Resolve and remove exact names or 1-based selectors."""
    try:
        bundle = _open_validated_pack(path)
    except CEVoiceError:
        print(string("cli.pack_invalid"))
        return 1
    except OSError:
        print(string("cli.pack_io_error"))
        return 1

    entries = bundle.entries
    names: list[str] = []
    for selector in selectors:
        if indexed:
            exact = next(
                (entry.name for entry in entries if entry.name == selector),
                None,
            )
            if exact is not None:
                name = exact
            elif re.fullmatch(r"[0-9]+", selector):
                try:
                    index = int(selector)
                except ValueError:
                    return _selector_error(program, selector)
                if not 1 <= index <= len(entries):
                    return _selector_error(program, selector)
                name = entries[index - 1].name
            else:
                return _selector_error(program, selector)
        else:
            if selector not in {entry.name for entry in entries}:
                return _selector_error(program, selector)
            name = selector
        if name in names:
            return _selector_error(program, selector)
        names.append(name)
    try:
        remove_cevoice_entries(path, names)
        return 0
    except CEVoiceError:
        print(string("cli.pack_update_failed"))
    except OSError:
        print(string("cli.pack_io_error"))
    return 1


def _open_validated_pack(path: Path) -> CEVoice:
    """Open a pack through CEVoice and validate every entry checksum."""
    bundle = CEVoice.open(path)
    for entry in bundle.entries:
        bundle.read_entry(entry.name)
    return bundle


def _added_entry_name(path: Path) -> str:
    """Return the filename that a supported source will store in the pack."""
    suffix = path.suffix.lower()
    if suffix in {".wav", ".pt"}:
        return f"{path.stem}.{suffix[1:]}"
    return path.name


def _selector_error(program: str, selector: str) -> int:
    """Report a missing, repeated, or out-of-range removal selector."""
    print(string("cli.invalid_argument"))
    print(string("cli.pack_selector_missing", selector=selector))
    print(string("cli.tar_usage", program=program))
    return ExitCodes.EXIT_UNKNOWN_ARGS.value


def _syntax_error(program: str, command: str) -> int:
    """Report invalid CLI syntax using the shared unknown-argument code."""
    print(string("cli.invalid_argument"))
    print(
        string(
            "cli.ls_usage" if command == "ls" else "cli.tar_usage",
            program=program,
        )
    )
    return ExitCodes.EXIT_UNKNOWN_ARGS.value


def _print_help(program: str) -> None:
    """Print the focused `celune tar` operation reference."""
    print(string("cli.tar_help_title", program=program))
    for key in (
        "cli.tar_help_list",
        "cli.tar_help_extract",
        "cli.tar_help_remove",
        "cli.tar_help_delete",
        "cli.tar_help_append",
        "cli.tar_help_help",
    ):
        print(string(key, program=program))
