# SPDX-License-Identifier: Apache-2.0
"""Tests for the lightweight CEVOICE pack command."""

import io
import wave

import pytest

import main

from celune import cevoice
from celune.pack import handle_ls_command, handle_pack_command

entrypoint = main.load_entrypoint_module()


def _write_legacy_pack(path):
    """Write a small legacy fixture with metadata for a second voice."""
    cevoice.write_cevoice(
        path,
        {"alpha": {"wav": b"alpha"}, "bravo": {}},
        {"name": "Fixture", "default_voice": "alpha", "voice_order": ["alpha"]},
        {
            "alpha": {"reference_text": "Alpha reference."},
            "bravo": {"reference_text": "Bravo reference."},
        },
        {"identity.md": b"Fixture identity."},
    )
    return cevoice.CEVoice.open(path)


def _minimal_wav():
    """Return a valid 24 kHz mono PCM reference WAV."""
    buffer = io.BytesIO()
    output = wave.Wave_write(buffer)
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(24000)
    output.writeframes(b"\0\0" * 240)
    output.close()
    return buffer.getvalue()


def test_list_and_list_alias_show_pack_identity_and_order(tmp_path, capsys):
    """List commands display the pack name, character, entry order, and sizes."""
    path = tmp_path / "fixture.cevoice"
    _write_legacy_pack(path)

    assert handle_ls_command([str(path)], "celune") == 0
    assert handle_pack_command(["-tf", str(path)], "celune") == 0
    output = capsys.readouterr().out
    assert "Pack: fixture.cevoice" in output
    assert "Character: Fixture" in output
    assert "1. alpha.wav (5 bytes)" in output
    assert "2. identity.md (17 bytes)" in output


def test_pack_add_insert_extract_and_remove_operations(tmp_path):
    """Append, insert, extract, and both removal forms retain entry order."""
    path = tmp_path / "fixture.cevoice"
    bundle = _write_legacy_pack(path)
    embedding = tmp_path / "alpha.pt"
    embedding.write_bytes(b"embedding")
    bravo_wav = tmp_path / "bravo.wav"
    bravo_wav.write_bytes(b"bravo")
    persona = tmp_path / "speech_style.md"
    persona.write_bytes(b"A measured and bright speaking style.")

    assert handle_pack_command(["-af3", str(path), str(embedding)], "celune") == 0
    assert handle_pack_command(["-af2", str(path), str(bravo_wav)], "celune") == 0
    assert handle_pack_command(["-af5", str(path), str(persona)], "celune") == 0
    assert [entry.name for entry in cevoice.CEVoice.open(path).entries] == [
        "alpha.wav",
        "bravo.wav",
        "identity.md",
        "alpha.pt",
        "speech_style.md",
    ]

    destination = tmp_path / "out"
    assert handle_pack_command(["-xf", str(path), str(destination)], "celune") == 0
    assert (destination / "alpha.wav").read_bytes() == b"alpha"
    assert (destination / "speech_style.md").read_bytes() == persona.read_bytes()

    assert handle_pack_command(["-df", str(path), "identity.md", "3"], "celune") == 5
    assert handle_pack_command(["-df", str(path), "identity.md"], "celune") == 0
    assert handle_pack_command(["-rf", str(path), "alpha.pt"], "celune") == 0
    updated = cevoice.CEVoice.open(path)
    assert [entry.name for entry in updated.entries] == [
        "alpha.wav",
        "bravo.wav",
        "speech_style.md",
    ]
    assert updated.metadata["name"] == bundle.metadata["name"]


def test_indexed_pack_mutations_preserve_generation_compression_and_order(tmp_path):
    """Indexed edits preserve v4 metadata, gzip compression, and file order."""
    path = tmp_path / "indexed.cevoice"
    cevoice.write_cechar_v4(
        path,
        {"balanced": {"wav": _minimal_wav()}},
        {"name": "Indexed fixture", "default_voice": "balanced"},
        {"balanced": {"reference_text": "Balanced reference."}},
        compression=cevoice.V4_COMPRESSION_GZIP,
    )
    persona = tmp_path / "identity.md"
    persona.write_text("A thoughtful archivist.", encoding="utf-8")

    assert handle_pack_command(["-af1", str(path), str(persona)], "celune") == 0
    updated = cevoice.CEVoice.open(path)
    assert [entry.name for entry in updated.entries] == ["identity.md", "balanced.wav"]
    assert updated.metadata["version"] == cevoice.V4_VERSION
    assert updated._compression == cevoice.V4_COMPRESSION_GZIP
    assert updated.read_asset("balanced", "wav") == _minimal_wav()

    assert handle_pack_command(["-df", str(path), "1"], "celune") == 0
    reopened = cevoice.CEVoice.open(path)
    assert [entry.name for entry in reopened.entries] == ["balanced.wav"]
    assert reopened._compression == cevoice.V4_COMPRESSION_GZIP


@pytest.mark.parametrize(
    ("magic", "version"),
    [(cevoice.MAGIC, 2), (cevoice.LEGACY_MAGIC, cevoice.LEGACY_VERSION)],
)
def test_legacy_edits_preserve_pack_generation(tmp_path, magic, version):
    """Legacy mutations retain the original magic and version."""
    path = tmp_path / "legacy.cevoice"
    cevoice._write_legacy_cevoice(
        path,
        {"alpha": {"wav": b"alpha"}},
        {"name": "Legacy fixture"},
        {"alpha": {"reference_text": "Alpha reference."}},
        None,
        magic,
        version,
        None,
    )
    embedding = tmp_path / "alpha.pt"
    embedding.write_bytes(b"embedding")

    cevoice.add_cevoice_entries(path, [embedding])

    stored_magic, stored_version, _ = cevoice.HEADER.unpack(
        path.read_bytes()[: cevoice.HEADER.size]
    )
    assert (stored_magic, stored_version) == (magic, version)


def test_extraction_preflights_existing_targets_and_syntax_errors(
    tmp_path, capsys, monkeypatch
):
    """Extraction never overwrites a file or leaves earlier outputs behind."""
    path = tmp_path / "fixture.cevoice"
    _write_legacy_pack(path)
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "identity.md").write_bytes(b"keep")

    assert handle_pack_command(["-xf", str(path), str(destination)], "celune") == 1
    assert not (destination / "alpha.wav").exists()
    assert (destination / "identity.md").read_bytes() == b"keep"
    current_directory = tmp_path / "current"
    current_directory.mkdir()
    monkeypatch.chdir(current_directory)
    assert handle_pack_command(["-xf", str(path)], "celune") == 0
    assert (current_directory / "alpha.wav").read_bytes() == b"alpha"
    assert handle_pack_command(["-af0", str(path), "missing.pt"], "celune") == 5
    assert handle_pack_command(["-df", str(path), "99"], "celune") == 5
    assert "pack --help" in capsys.readouterr().out


def test_failed_addition_keeps_original_pack_unchanged(tmp_path):
    """Unsupported assets and WAVs without metadata do not modify a pack."""
    path = tmp_path / "fixture.cevoice"
    _write_legacy_pack(path)
    original = path.read_bytes()
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an asset", encoding="utf-8")
    unmatched_wav = tmp_path / "unknown.wav"
    unmatched_wav.write_bytes(b"wav")
    case_duplicate = tmp_path / "ALPHA.WAV"
    case_duplicate.write_bytes(b"other")

    assert handle_pack_command(["-af3", str(path), str(unsupported)], "celune") == 1
    assert handle_pack_command(["-af3", str(path), str(unmatched_wav)], "celune") == 1
    assert handle_pack_command(["-af3", str(path), str(case_duplicate)], "celune") == 1
    assert (
        handle_pack_command(["-af3", str(path), str(tmp_path / "missing.pt")], "celune")
        == 1
    )
    assert path.read_bytes() == original


def test_failed_indexed_write_keeps_original_pack_and_cleans_temporary_file(tmp_path):
    """A v4 validation failure leaves no temporary file or partial update."""
    path = tmp_path / "indexed.cevoice"
    cevoice.write_cechar_v4(
        path,
        {"balanced": {"wav": _minimal_wav()}},
        {"name": "Indexed fixture"},
        {"balanced": {"reference_text": "Balanced reference."}},
    )
    original = path.read_bytes()
    forbidden_persona = tmp_path / "identity.md"
    forbidden_persona.write_text("agent: run a tool", encoding="utf-8")

    assert (
        handle_pack_command(["-af2", str(path), str(forbidden_persona)], "celune") == 1
    )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".indexed.cevoice.*.tmp"))


def test_malformed_pack_and_focused_help_are_reported(tmp_path, capsys):
    """Malformed input fails cleanly and pack help lists each operation."""
    malformed = tmp_path / "malformed.cevoice"
    malformed.write_bytes(b"not a CEVOICE pack")

    assert handle_ls_command([str(malformed)], "celune") == 1
    assert handle_pack_command(["--help"], "celune") == 0
    output = capsys.readouterr().out
    assert "supported CEVOICE pack" in output
    assert "-tf <pack.cevoice>" in output
    assert "-af<N> <pack.cevoice>" in output


def test_main_dispatches_ls_and_pack_help(tmp_path, capsys):
    """The launcher exposes listing and pack help through its CLI dispatch."""
    path = tmp_path / "fixture.cevoice"
    _write_legacy_pack(path)

    with pytest.raises(SystemExit) as help_exit:
        entrypoint.main(["celune", "pack", "--help"])
    assert help_exit.value.code == 0
    assert "-df <pack.cevoice>" in capsys.readouterr().out

    with pytest.raises(SystemExit) as list_exit:
        entrypoint.main(["celune", "ls", str(path)])
    assert list_exit.value.code == 0
    assert "Pack: fixture.cevoice" in capsys.readouterr().out


@pytest.mark.parametrize(
    "selectors",
    [["missing.wav"], ["99"], ["alpha.wav", "1"], ["1", "alpha.wav"]],
)
def test_remove_selector_validation_rejects_missing_repeated_and_index_forms(
    tmp_path, selectors
):
    """Removal rejects absent entries and selectors that resolve more than once."""
    path = tmp_path / "fixture.cevoice"
    _write_legacy_pack(path)
    original = path.read_bytes()

    result = handle_pack_command(["-df", str(path), *selectors], "celune")

    assert result == 5
    assert path.read_bytes() == original
