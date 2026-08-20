# SPDX-License-Identifier: MIT
"""Tests for Needle checkpoint provenance and preparation."""

import json
import hashlib
from pathlib import Path
from typing import Optional
from unittest import TestCase
from collections.abc import Mapping
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from celune.typing.common import JSONSerializable
from celune.agent.needle_model import NeedleModel, NeedleConfig
from celune.agent.needle_checkpoint import (
    NEEDLE_PICKLE_FILE,
    NeedleCheckpointError,
    NeedleTensorInventory,
    NeedleUnsupportedConverterError,
    prepare_needle_checkpoint,
    validate_needle_safetensors,
)


def _config_values() -> dict[str, JSONSerializable]:
    """Return a small architecture configuration for structural tests."""
    return {
        "vocab_size": 16,
        "d_model": 8,
        "num_heads": 2,
        "num_kv_heads": 1,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "max_seq_len": 8,
        "pad_token_id": 0,
        "decoder_start_token_id": 1,
        "eos_token_id": 1,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }


def _write_config(path: Path, values: Mapping[str, JSONSerializable]) -> None:
    """Write a test architecture configuration."""
    path.write_text(json.dumps(dict(values)), encoding="utf-8")


def _write_checkpoint(
    path: Path,
    values: Mapping[str, JSONSerializable],
    *,
    prefix: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> NeedleTensorInventory:
    """Write a complete synthetic checkpoint with independent tensor storage."""
    config = NeedleConfig.from_mapping(values)
    model = NeedleModel(config).to(dtype=dtype or torch.float32)
    state = {
        (f"model.{name}" if prefix else name): tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))
    return validate_needle_safetensors(path, values)


class NeedleCheckpointValidationTests(TestCase):
    """Verify strict validation and typed failures for checkpoint artifacts."""

    def test_valid_checkpoint_inventory_and_metadata_are_serializable(self) -> None:
        """Validate the complete state dictionary and JSON metadata shape."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config_path = root / "config.json"
            source_path = root / "model.safetensors"
            _write_config(config_path, values)
            inventory = _write_checkpoint(source_path, values)

            prepared = prepare_needle_checkpoint(
                "test/model",
                "revision-1",
                root / "cache",
                source_path=source_path,
                config_path=config_path,
            )

            self.assertEqual(prepared.metadata.source_format, "safetensors")
            self.assertEqual(prepared.metadata.tensor_count, inventory.tensor_count)
            self.assertEqual(
                prepared.metadata.parameter_count,
                inventory.parameter_count,
            )
            self.assertEqual(
                prepared.metadata.canonical_sha256,
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                json.loads(json.dumps(prepared.metadata.to_json()))["source_filename"],
                "model.safetensors",
            )

    def test_invalid_tensor_names_are_rejected_by_strict_validation(self) -> None:
        """Reject missing or unexpected state-dictionary entries."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config = NeedleConfig.from_mapping(values)
            state = {
                name: tensor.detach().clone()
                for name, tensor in NeedleModel(config).state_dict().items()
            }
            state.pop("embed_tokens.weight")
            state["unexpected.weight"] = torch.zeros((1, 1))
            checkpoint = root / "invalid.safetensors"
            save_file(state, str(checkpoint))

            with self.assertRaises(NeedleCheckpointError):
                validate_needle_safetensors(checkpoint, values)

    def test_invalid_tensor_shape_and_dtype_are_rejected(self) -> None:
        """Reject tensors that do not match the declared runtime state."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config = NeedleConfig.from_mapping(values)
            state = {
                name: tensor.detach().clone()
                for name, tensor in NeedleModel(config).state_dict().items()
            }
            state["embed_tokens.weight"] = torch.zeros(
                (config.vocab_size + 1, config.d_model),
                dtype=torch.float32,
            )
            state["lm_head.weight"] = state["lm_head.weight"].to(torch.float16)
            checkpoint = root / "invalid.safetensors"
            save_file(state, str(checkpoint))

            with self.assertRaises(NeedleCheckpointError):
                validate_needle_safetensors(checkpoint, values)

    def test_invalid_configuration_is_reported_as_checkpoint_error(self) -> None:
        """Translate invalid architecture dimensions into the typed error."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            invalid_values = dict(values)
            invalid_values["d_model"] = 7
            checkpoint = root / "checkpoint.safetensors"
            _write_checkpoint(checkpoint, values)

            with self.assertRaises(NeedleCheckpointError):
                validate_needle_safetensors(checkpoint, invalid_values)

    def test_source_artifact_and_hash_are_checked_before_preparation(self) -> None:
        """Reject unsupported source names and provenance hash mismatches."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config_path = root / "config.json"
            source_path = root / "model.safetensors"
            _write_config(config_path, values)
            _write_checkpoint(source_path, values)

            with self.assertRaises(NeedleCheckpointError):
                prepare_needle_checkpoint(
                    "test/model",
                    "revision-1",
                    root / "cache",
                    source_filename="unsupported.bin",
                    source_path=source_path,
                    config_path=config_path,
                )

            with self.assertRaises(NeedleCheckpointError):
                prepare_needle_checkpoint(
                    "test/model",
                    "revision-1",
                    root / "cache",
                    source_path=source_path,
                    config_path=config_path,
                    expected_source_sha256="0" * 64,
                )

    def test_legacy_pickle_requires_an_explicit_converter(self) -> None:
        """Never load the legacy JAX/Flax pickle in the runtime path."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config_path = root / "config.json"
            source_path = root / NEEDLE_PICKLE_FILE
            _write_config(config_path, values)
            source_path.write_bytes(b"legacy checkpoint")

            with self.assertRaises(NeedleUnsupportedConverterError):
                prepare_needle_checkpoint(
                    "test/model",
                    "revision-1",
                    root / "cache",
                    source_filename=NEEDLE_PICKLE_FILE,
                    source_path=source_path,
                    config_path=config_path,
                )

            def unsupported_converter(
                _source: Path,
                _destination: Path,
                _config: Mapping[str, JSONSerializable],
            ) -> None:
                """Represent a converter that cannot safely handle the source."""
                raise NotImplementedError

            with self.assertRaises(NeedleUnsupportedConverterError):
                prepare_needle_checkpoint(
                    "test/model",
                    "revision-1",
                    root / "cache",
                    source_filename=NEEDLE_PICKLE_FILE,
                    source_path=source_path,
                    config_path=config_path,
                    pickle_converter=unsupported_converter,
                )

    def test_incompatible_legacy_conversion_output_is_rejected(self) -> None:
        """Reject converter output that does not match the current Needle model."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config_path = root / "config.json"
            source_path = root / NEEDLE_PICKLE_FILE
            _write_config(config_path, values)
            source_path.write_bytes(b"legacy checkpoint")

            def incompatible_converter(
                _source: Path,
                destination: Path,
                _config: Mapping[str, JSONSerializable],
            ) -> None:
                """Write an artifact from an incompatible architecture."""
                save_file({"unexpected.weight": torch.zeros((1, 1))}, str(destination))

            with self.assertRaises(NeedleCheckpointError):
                prepare_needle_checkpoint(
                    "test/model",
                    "revision-1",
                    root / "cache",
                    source_filename=NEEDLE_PICKLE_FILE,
                    source_path=source_path,
                    config_path=config_path,
                    pickle_converter=incompatible_converter,
                )

    def test_legacy_conversion_is_cached_by_source_and_converter_version(self) -> None:
        """Reuse validated conversion output only for the exact source and converter."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = _config_values()
            config_path = root / "config.json"
            source_path = root / NEEDLE_PICKLE_FILE
            cache_path = root / "cache"
            _write_config(config_path, values)
            source_path.write_bytes(b"legacy checkpoint v1")
            calls: list[tuple[Path, Path, Mapping[str, JSONSerializable]]] = []

            def converter(
                source: Path,
                destination: Path,
                target_config: Mapping[str, JSONSerializable],
            ) -> None:
                calls.append((source, destination, target_config))
                _write_checkpoint(destination, target_config)

            first = prepare_needle_checkpoint(
                "test/model",
                "revision-1",
                cache_path,
                source_filename=NEEDLE_PICKLE_FILE,
                source_path=source_path,
                config_path=config_path,
                pickle_converter=converter,
                converter_version="converter-1",
            )
            second = prepare_needle_checkpoint(
                "test/model",
                "revision-1",
                cache_path,
                source_filename=NEEDLE_PICKLE_FILE,
                source_path=source_path,
                config_path=config_path,
                pickle_converter=converter,
                converter_version="converter-1",
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(first.path, second.path)
            self.assertEqual(first.metadata.source_format, "jax_flax_pickle")
            self.assertTrue(first.path.is_file())
            self.assertFalse(
                any(path.suffix == ".tmp" for path in first.path.parent.iterdir())
            )

            source_path.write_bytes(b"legacy checkpoint v2")
            prepare_needle_checkpoint(
                "test/model",
                "revision-1",
                cache_path,
                source_filename=NEEDLE_PICKLE_FILE,
                source_path=source_path,
                config_path=config_path,
                pickle_converter=converter,
                converter_version="converter-1",
            )
            self.assertEqual(len(calls), 2)

            prepare_needle_checkpoint(
                "test/model",
                "revision-1",
                cache_path,
                source_filename=NEEDLE_PICKLE_FILE,
                source_path=source_path,
                config_path=config_path,
                pickle_converter=converter,
                converter_version="converter-2",
            )
            self.assertEqual(len(calls), 3)
