# SPDX-License-Identifier: Apache-2.0
"""Verified preparation of the inference-only Needle checkpoint."""

from __future__ import annotations

import os
import json
import hashlib
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Union, Optional, Protocol, cast

import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

from ..paths import huggingface_hub_cache_dir
from ..typing.common import JSON, JSONSerializable
from .needle_model import NeedleModel, NeedleConfig
from ..exceptions import (
    NeedleCheckpointError,
    NeedleUnsupportedConverterError,
)

NEEDLE_MODEL_ID = "Cactus-Compute/needle"
NEEDLE_MODEL_REVISION = "5f89b4307696d669c3df1d38ae057e6e1728b107"
NEEDLE_CONFIG_FILE = "config.json"
NEEDLE_WEIGHTS_FILE = "model.safetensors"
NEEDLE_PICKLE_FILE = "needle.pkl"
NEEDLE_TOKENIZER_FILE = "tokenizer.model"
NEEDLE_RUNTIME_WEIGHTS_SHA256 = (
    "c5f9a3016e4537e492c362da5cb8ba05107d8595bec0d5ea5d8a65801db46531"
)
NEEDLE_PICKLE_SHA256 = (
    "40a32e91d1d4197bf15ba559b74f6727c342dc8746918742fc7d8e2c1f18df40"
)
NEEDLE_PREPARER_VERSION = "celune-needle-preparer-1"
NEEDLE_DIRECT_CONVERTER_VERSION = "direct-safetensors-1"


class NeedlePickleConverter(Protocol):
    """Offline process boundary for converting a legacy JAX/Flax checkpoint.

    Implementations must bridge to a separately isolated converter process. This
    protocol is not an invitation to unpickle data inside Celune's runtime.
    """

    def __call__(
        self,
        source_path: Path,
        destination_path: Path,
        target_config: Mapping[str, JSONSerializable],
        /,
    ) -> None:
        """Write a safetensors artifact without loading pickle in Celune."""
        raise NotImplementedError("protocol not defined")


@dataclass(frozen=True)
class NeedleTensorInventory:
    """Validated tensor inventory for one canonical Needle artifact."""

    names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]
    tensor_count: int
    parameter_count: int

    def to_json(self) -> JSON:
        """Serialize tensor inventory metadata into JSON-compatible data."""
        return {
            "names": cast(JSONSerializable, list(self.names)),
            "shapes": cast(
                JSONSerializable,
                {name: list(shape) for name, shape in self.shapes.items()},
            ),
            "dtypes": cast(JSONSerializable, self.dtypes),
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class NeedleCheckpointMetadata:
    """Provenance and validation metadata for one prepared artifact."""

    model_id: str
    revision: str
    source_filename: str
    source_format: str
    source_sha256: str
    source_size: int
    canonical_sha256: str
    canonical_size: int
    converter_version: str
    tensor_count: int
    parameter_count: int

    def to_json(self) -> JSON:
        """Serialize checkpoint provenance and validation metadata."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source_filename": self.source_filename,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "canonical_sha256": self.canonical_sha256,
            "canonical_size": self.canonical_size,
            "converter_version": self.converter_version,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class NeedlePreparedCheckpoint:
    """Prepared checkpoint and its configuration for the future Needle loader."""

    path: Path
    config_path: Path
    metadata: NeedleCheckpointMetadata


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local artifact."""
    if not path.is_file():
        raise NeedleCheckpointError(f"Needle checkpoint artifact is missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise NeedleCheckpointError(
            f"Needle checkpoint artifact could not be read: {path}"
        ) from exc
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, JSONSerializable]:
    """Load and validate one JSON architecture configuration."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NeedleCheckpointError(
            f"Needle configuration could not be read: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise NeedleCheckpointError("Needle configuration must be a JSON object")
    return cast(dict[str, JSONSerializable], value)


def _expected_dtype(
    config: Mapping[str, JSONSerializable],
) -> torch.dtype:
    """Resolve the declared runtime tensor dtype from the model configuration."""
    value = config.get("torch_dtype", "float32")
    if not isinstance(value, str):
        raise NeedleCheckpointError("Needle torch_dtype must be a string")
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtypes.get(value)
    if dtype is None:
        raise NeedleCheckpointError(f"unsupported Needle torch_dtype: {value}")
    return dtype


def _normalized_tensor_name(name: str) -> str:
    """Normalize a Hugging Face model prefix for Celune's state dictionary."""
    return name.removeprefix("model.")


def validate_needle_safetensors(
    path: Path,
    config: Union[NeedleConfig, Mapping[str, JSONSerializable]],
    expected_dtype: Optional[torch.dtype] = None,
) -> NeedleTensorInventory:
    """Validate names, shapes, dtypes, counts, and strict model loading."""
    if not path.is_file():
        raise NeedleCheckpointError(f"Needle safetensors artifact is missing: {path}")
    config_mapping = (
        config
        if isinstance(config, Mapping)
        else cast(Mapping[str, JSONSerializable], config.__dict__)
    )
    try:
        model_config = (
            config
            if isinstance(config, NeedleConfig)
            else NeedleConfig.from_mapping(config)
        )
    except (TypeError, ValueError) as exc:
        raise NeedleCheckpointError(
            "Needle configuration is invalid for the runtime model"
        ) from exc
    dtype = expected_dtype or _expected_dtype(config_mapping)
    target = NeedleModel(model_config).to(dtype=dtype)
    target_state = target.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_name in tuple(handle.keys()):
                name = _normalized_tensor_name(raw_name)
                if name in tensors:
                    raise NeedleCheckpointError(
                        f"Needle checkpoint has duplicate tensor name: {name}"
                    )
                tensors[name] = handle.get_tensor(raw_name)
    except NeedleCheckpointError:
        raise
    except Exception as exc:
        raise NeedleCheckpointError(
            f"Needle safetensors artifact could not be read: {path}"
        ) from exc

    missing = sorted(set(target_state) - set(tensors))
    unexpected = sorted(set(tensors) - set(target_state))
    if missing or unexpected:
        raise NeedleCheckpointError(
            "Needle checkpoint tensor names do not match the strict model state "
            f"(missing={missing}, unexpected={unexpected})"
        )

    shape_errors: list[str] = []
    dtype_errors: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for name, tensor in tensors.items():
        shape = tuple(tensor.shape)
        shapes[name] = shape
        dtypes[name] = str(tensor.dtype)
        if shape != tuple(int(value) for value in target_state[name].shape):
            shape_errors.append(
                f"{name}: expected {tuple(target_state[name].shape)}, got {shape}"
            )
        if tensor.dtype != dtype:
            dtype_errors.append(f"{name}: expected {dtype}, got {tensor.dtype}")
    if shape_errors:
        raise NeedleCheckpointError(
            "Needle checkpoint tensor shapes do not match: " + "; ".join(shape_errors)
        )
    if dtype_errors:
        raise NeedleCheckpointError(
            "Needle checkpoint tensor dtypes do not match: " + "; ".join(dtype_errors)
        )

    try:
        target.load_state_dict(tensors, strict=True)
    except RuntimeError as exc:
        raise NeedleCheckpointError(
            "Needle checkpoint failed strict state-dict loading"
        ) from exc
    names = tuple(sorted(tensors))
    return NeedleTensorInventory(
        names=names,
        shapes={name: shapes[name] for name in names},
        dtypes={name: dtypes[name] for name in names},
        tensor_count=len(names),
        parameter_count=sum(tensor.numel() for tensor in tensors.values()),
    )


def _cache_component(value: str) -> str:
    """Make a model identifier safe and stable as a cache directory name."""
    component = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip(".")
    return component or "default"


def _atomic_replace(source: Path, destination: Path) -> None:
    """Publish one validated temporary artifact atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _metadata_matches(
    metadata: Optional[NeedleCheckpointMetadata],
    *,
    model_id: str,
    revision: str,
    source_filename: str,
    source_sha256: str,
    converter_version: str,
) -> bool:
    """Return whether cached provenance still describes the requested source."""
    return metadata is not None and (
        metadata.model_id == model_id
        and metadata.revision == revision
        and metadata.source_filename == source_filename
        and metadata.source_sha256 == source_sha256
        and metadata.converter_version == converter_version
    )


def _read_metadata(path: Path) -> Optional[NeedleCheckpointMetadata]:
    """Read cached metadata, treating malformed cache entries as stale."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    string_keys = (
        "model_id",
        "revision",
        "source_filename",
        "source_format",
        "source_sha256",
        "canonical_sha256",
        "converter_version",
    )
    numeric_keys = (
        "source_size",
        "canonical_size",
        "tensor_count",
        "parameter_count",
    )
    if not all(key in value for key in string_keys + numeric_keys):
        return None
    if not all(isinstance(value[key], str) for key in string_keys):
        return None
    if not all(
        isinstance(value[key], int) and not isinstance(value[key], bool)
        for key in numeric_keys
    ):
        return None
    return NeedleCheckpointMetadata(
        model_id=cast(str, value["model_id"]),
        revision=cast(str, value["revision"]),
        source_filename=cast(str, value["source_filename"]),
        source_format=cast(str, value["source_format"]),
        source_sha256=cast(str, value["source_sha256"]),
        source_size=cast(int, value["source_size"]),
        canonical_sha256=cast(str, value["canonical_sha256"]),
        canonical_size=cast(int, value["canonical_size"]),
        converter_version=cast(str, value["converter_version"]),
        tensor_count=cast(int, value["tensor_count"]),
        parameter_count=cast(int, value["parameter_count"]),
    )


def _write_metadata(path: Path, metadata: NeedleCheckpointMetadata) -> None:
    """Write provenance metadata through an atomic replacement."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(metadata.to_json(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _atomic_replace(temporary, path)
    except OSError as exc:
        raise NeedleCheckpointError(
            f"Needle checkpoint metadata could not be published: {path}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _download(
    model_id: str,
    revision: str,
    filename: str,
    cache_root: Path,
) -> Path:
    """Resolve one exact-revision Hugging Face artifact."""
    try:
        return Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_root),
                repo_type="model",
            )
        )
    except Exception as exc:
        raise NeedleCheckpointError(
            f"Needle source artifact could not be obtained: {model_id}@{revision} "
            f"/{filename}"
        ) from exc


def prepare_needle_checkpoint(
    model_id: str = NEEDLE_MODEL_ID,
    revision: str = NEEDLE_MODEL_REVISION,
    cache_dir: Optional[Path] = None,
    *,
    source_filename: str = NEEDLE_WEIGHTS_FILE,
    source_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    expected_source_sha256: Optional[str] = None,
    pickle_converter: Optional[NeedlePickleConverter] = None,
    converter_version: str = NEEDLE_PREPARER_VERSION,
) -> NeedlePreparedCheckpoint:
    """Prepare and strictly validate the canonical inference checkpoint.

    The normal path selects the pinned upstream ``model.safetensors`` artifact.
    The legacy ``needle.pkl`` path is accepted only through an explicitly
    isolated converter callback; Celune never unpickles it in the runtime.
    """
    if source_filename not in {NEEDLE_WEIGHTS_FILE, NEEDLE_PICKLE_FILE}:
        raise NeedleCheckpointError(
            f"unsupported Needle source artifact: {source_filename}"
        )
    if not model_id or not revision or not converter_version:
        raise NeedleCheckpointError(
            "Needle model_id, revision, and converter_version must be non-empty"
        )
    cache_root = cache_dir or huggingface_hub_cache_dir(create=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved_config = config_path or _download(
        model_id, revision, NEEDLE_CONFIG_FILE, cache_root
    )
    config = _load_config(resolved_config)
    resolved_source = source_path or _download(
        model_id, revision, source_filename, cache_root
    )
    source_sha256 = _sha256(resolved_source)
    expected_hash = expected_source_sha256
    if (
        expected_hash is None
        and model_id == NEEDLE_MODEL_ID
        and revision == NEEDLE_MODEL_REVISION
    ):
        expected_hash = (
            NEEDLE_RUNTIME_WEIGHTS_SHA256
            if source_filename == NEEDLE_WEIGHTS_FILE
            else NEEDLE_PICKLE_SHA256
        )
    if expected_hash is not None and source_sha256 != expected_hash:
        raise NeedleCheckpointError(
            f"Needle source hash mismatch for {source_filename}: "
            f"expected {expected_hash}, got {source_sha256}"
        )

    if source_filename == NEEDLE_WEIGHTS_FILE:
        inventory = validate_needle_safetensors(resolved_source, config)
        metadata = NeedleCheckpointMetadata(
            model_id=model_id,
            revision=revision,
            source_filename=source_filename,
            source_format="safetensors",
            source_sha256=source_sha256,
            source_size=resolved_source.stat().st_size,
            canonical_sha256=source_sha256,
            canonical_size=resolved_source.stat().st_size,
            converter_version=NEEDLE_DIRECT_CONVERTER_VERSION,
            tensor_count=inventory.tensor_count,
            parameter_count=inventory.parameter_count,
        )
        return NeedlePreparedCheckpoint(resolved_source, resolved_config, metadata)

    if pickle_converter is None:
        raise NeedleUnsupportedConverterError(
            "needle.pkl is a legacy JAX/Flax checkpoint; provide an isolated "
            "offline converter before using it"
        )
    artifact_dir = (
        cache_root
        / "celune-needle-prepared"
        / _cache_component(model_id)
        / _cache_component(revision)
    )
    artifact_path = artifact_dir / (
        f"model-{source_sha256[:16]}-{_cache_component(converter_version)}.safetensors"
    )
    metadata_path = artifact_path.with_suffix(".json")
    cached_metadata = _read_metadata(metadata_path)
    if artifact_path.is_file() and _metadata_matches(
        cached_metadata,
        model_id=model_id,
        revision=revision,
        source_filename=source_filename,
        source_sha256=source_sha256,
        converter_version=converter_version,
    ):
        assert cached_metadata is not None
        try:
            inventory = validate_needle_safetensors(artifact_path, config)
            if (
                _sha256(artifact_path) == cached_metadata.canonical_sha256
                and inventory.tensor_count == cached_metadata.tensor_count
                and inventory.parameter_count == cached_metadata.parameter_count
            ):
                return NeedlePreparedCheckpoint(
                    artifact_path,
                    resolved_config,
                    cached_metadata,
                )
        except NeedleCheckpointError:
            pass

    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary = artifact_dir / f".{artifact_path.name}.{uuid4().hex}.tmp"
    try:
        pickle_converter(resolved_source, temporary, config)
        if not temporary.is_file():
            raise NeedleCheckpointError(
                "Needle pickle converter did not publish a safetensors artifact"
            )
        inventory = validate_needle_safetensors(temporary, config)
        canonical_sha256 = _sha256(temporary)
        metadata = NeedleCheckpointMetadata(
            model_id=model_id,
            revision=revision,
            source_filename=source_filename,
            source_format="jax_flax_pickle",
            source_sha256=source_sha256,
            source_size=resolved_source.stat().st_size,
            canonical_sha256=canonical_sha256,
            canonical_size=temporary.stat().st_size,
            converter_version=converter_version,
            tensor_count=inventory.tensor_count,
            parameter_count=inventory.parameter_count,
        )
        _atomic_replace(temporary, artifact_path)
        _write_metadata(metadata_path, metadata)
        return NeedlePreparedCheckpoint(artifact_path, resolved_config, metadata)
    except NeedleCheckpointError:
        raise
    except NotImplementedError as exc:
        raise NeedleUnsupportedConverterError(
            "the supplied Needle pickle converter does not support this artifact"
        ) from exc
    except Exception as exc:
        raise NeedleCheckpointError("Needle pickle conversion failed") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
