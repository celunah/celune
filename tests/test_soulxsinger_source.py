# SPDX-License-Identifier: Apache-2.0
"""Tests for SoulX-Singer's pinned source checkout."""

from pathlib import Path
from unittest import mock
from tempfile import TemporaryDirectory

from celune.backends.tts import soulxsinger


def test_soulxsinger_source_clones_into_revision_directory() -> None:
    """Clone directly into the locked revision directory without renaming it."""
    with TemporaryDirectory() as directory:
        source_parent = Path(directory) / "soulxsinger-source"
        source_root = source_parent / soulxsinger.SOULX_SINGER_CODE_REVISION
        package_path = source_root / "soulxsinger" / "models" / "soulxsinger.py"
        repository = mock.Mock()
        repository.head.commit.hexsha = soulxsinger.SOULX_SINGER_CODE_REVISION

        def clone_source(_url: str, destination: Path, **_kwargs: object) -> mock.Mock:
            cloned_package = destination / "soulxsinger" / "models" / "soulxsinger.py"
            cloned_package.parent.mkdir(parents=True)
            cloned_package.touch()
            return repository

        with (
            mock.patch(
                "celune.backends.tts.soulxsinger.backend_environments_dir",
                return_value=source_parent.parent,
            ),
            mock.patch("git.Repo") as repository_type,
        ):
            repository_type.clone_from.side_effect = clone_source
            assert soulxsinger._load_upstream_source() == source_root

        repository_type.clone_from.assert_called_once_with(
            soulxsinger.SOULX_SINGER_CODE_REPOSITORY,
            source_root,
            no_checkout=True,
            multi_options=["--filter=blob:none"],
        )
        repository.git.checkout.assert_called_once_with(
            soulxsinger.SOULX_SINGER_CODE_REVISION
        )
        assert package_path.is_file()
