# SPDX-License-Identifier: Apache-2.0
"""Pinned model-weight contracts for Celune's TTS backends.

The contracts describe the upstream artifacts that a backend is allowed to
load.  Tensor inventories are represented by a count, dtype distribution, and
canonical inventory digest so a backend can validate a local snapshot without
keeping thousands of individual tensor names in the source tree.
"""

from __future__ import annotations

import json
import hashlib
from typing import Union, Optional
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping

import torch

from ...exceptions import InvalidCheckpoint, ModelContractError

__all__ = [
    "MODEL_CONTRACTS",
    "ModelComponentContract",
    "ModelContract",
    "QuantizationRule",
    "RuntimeDtypeRule",
    "TensorInventoryContract",
    "WeightArtifactContract",
    "model_contract",
    "validate_model_state",
    "validate_safetensors_artifact",
]


@dataclass(frozen=True)
class TensorInventoryContract:
    """Expected structural inventory for one serialized tensor file."""

    tensor_count: int
    parameter_count: int
    dtype_counts: tuple[tuple[str, int], ...]
    inventory_sha256: str


@dataclass(frozen=True)
class WeightArtifactContract:
    """Expected immutable metadata for one upstream weight artifact."""

    path: str
    size: int
    sha256: str
    inventory: Optional[TensorInventoryContract] = None


@dataclass(frozen=True)
class RuntimeDtypeRule:
    """Expected runtime dtype for one relative tensor-name prefix."""

    prefix: str
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class QuantizationRule:
    """Safe linear-module names for runtime weight quantization."""

    module_suffixes: tuple[str, ...]
    excluded_module_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelComponentContract:
    """Expected state inventory and runtime dtypes for one model component."""

    name: str
    artifact_path: str
    inventory: TensorInventoryContract
    runtime_dtypes: tuple[RuntimeDtypeRule, ...]
    quantization: Optional[QuantizationRule] = None


@dataclass(frozen=True)
class ModelContract:
    """Pinned Hugging Face model contract for one backend variant."""

    backend_id: str
    model_id: str
    revision: str
    artifacts: tuple[WeightArtifactContract, ...]
    components: tuple[ModelComponentContract, ...]
    variant: Optional[str] = None


def _inventory(
    tensor_count: int,
    parameter_count: int,
    dtype_counts: tuple[tuple[str, int], ...],
    inventory_sha256: str,
) -> TensorInventoryContract:
    """Build one compact inventory declaration."""
    return TensorInventoryContract(
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        dtype_counts=dtype_counts,
        inventory_sha256=inventory_sha256,
    )


def _artifact(
    path: str,
    size: int,
    sha256: str,
    inventory: Optional[TensorInventoryContract] = None,
) -> WeightArtifactContract:
    """Build one immutable upstream artifact declaration."""
    return WeightArtifactContract(
        path=path,
        size=size,
        sha256=sha256,
        inventory=inventory,
    )


def _component(
    name: str,
    artifact_path: str,
    inventory: TensorInventoryContract,
    *runtime_dtypes: RuntimeDtypeRule,
    quantization: Optional[QuantizationRule] = None,
) -> ModelComponentContract:
    """Build one runtime component declaration."""
    return ModelComponentContract(
        name=name,
        artifact_path=artifact_path,
        inventory=inventory,
        runtime_dtypes=runtime_dtypes,
        quantization=quantization,
    )


_POCKET_QUANTIZATION = QuantizationRule(
    module_suffixes=("in_proj", "out_proj", "linear1", "linear2"),
    excluded_module_names=(
        "conditioner",
        "decoder",
        "embedding",
        "embeddings",
        "flow_net",
        "input_linear",
        "input_proj",
        "mimi",
        "norm",
        "out_eos",
        "output_projs",
        "vocoder",
    ),
)
_TRANSFORMER_QUANTIZATION = QuantizationRule(
    module_suffixes=(
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ),
    excluded_module_names=(
        "decoder",
        "embedding",
        "embeddings",
        "head",
        "lm_head",
        "norm",
        "output",
        "speaker",
        "spk",
        "tokenizer",
        "vocoder",
    ),
)
_FIRERED_QUANTIZATION = QuantizationRule(
    module_suffixes=_TRANSFORMER_QUANTIZATION.module_suffixes,
    excluded_module_names=_TRANSFORMER_QUANTIZATION.excluded_module_names
    + (
        "dit",
        "dit_head",
        "patch_encoder",
        "spk_proj_dit",
        "spk_proj_llm",
        "stop_head",
    ),
)


_POCKET_ENGLISH_INVENTORY = _inventory(
    214,
    109502146,
    (("torch.bfloat16", 214),),
    "216df7d816a634d6d6a74236fdc33c44d41987ccb3196cd9733b7c317c5a50d0",
)
_POCKET_FRENCH_INVENTORY = _inventory(
    358,
    336068290,
    (("torch.bfloat16", 358),),
    "36d0ec99bbb6d30bd1b5a192174e68210722bb135cea231393073af63d900fa9",
)
_QWEN06_INVENTORY = _inventory(
    478,
    914643008,
    (("torch.bfloat16", 478),),
    "aba2004328291ecb3523b0be7c18f5ecf7e1cc14a21abbaf576fabe54519a08e",
)
_QWEN17_INVENTORY = _inventory(
    480,
    1928677440,
    (("torch.bfloat16", 480),),
    "7982f98abf01548b24e16e7b2d77abd775801bb112eb43e54ec1cb68409545c1",
)
_QWEN_TOKENIZER_INVENTORY = _inventory(
    496,
    170557441,
    (("torch.float32", 496),),
    "59366382fab43834d4c4f5806581af1904b18bc88a9e43b6db9487671c310057",
)
_DOTS_CORE_INVENTORY = _inventory(
    870,
    2199404546,
    (("torch.bfloat16", 870),),
    "b2c4681557e061cbfbb38f407e26d39069abbad8d87b0047ecd2980cb9787d96",
)
_DOTS_SPEAKER_INVENTORY = _inventory(
    938,
    7259203,
    (("torch.float32", 816), ("torch.int64", 122)),
    "9a6ef82bb4cd15f0d377fc652c0b535dc82dac3809dcc4fd78d04a54f2b99862",
)
_DOTS_VOCODER_INVENTORY = _inventory(
    951,
    180869212,
    (("torch.float32", 951),),
    "c602f55ddd8833135ed37812098e5bf44a8db49ac657c7a9726d9b9acc22122e",
)
_VOX_INVENTORY = _inventory(
    577,
    2290004544,
    (("torch.bfloat16", 577),),
    "765e02460007ee77b49b1cf71100adb6b12c2f4eb09098ef46476c8dbfb4e935",
)
_FIRERED_CORE_INVENTORY = _inventory(
    677,
    2120632897,
    (("torch.float32", 677),),
    "dfbe358e0e0ea4b554aed11bd7bb72343712bebe927a688418f708ed0df3c700",
)
_FIRERED_REDAE_INVENTORY = _inventory(
    458,
    943776322,
    (("torch.float32", 458),),
    "f7b28fb636e8f9ac3304f99825fb34e875b19f7e6baaf8737eaef609c61d9f0c",
)


_POCKET_REVISION = "d03cd73415a8d46d8eb115c7b524aebb0a729f4a"
_QWEN06_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
_QWEN17_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
_DOTS_REVISION = "c28105adc8228143392b4e346994ff613ee48a06"
_VOX_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
_FIRERED_REVISION = "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610"
_LUXTTS_REVISION = "527f245a276a0eb42ea103a7a512bcfd771eb9b6"


def _pocket_contract(
    language_variant: str,
    path: str,
    size: int,
    sha256: str,
    inventory: TensorInventoryContract,
) -> ModelContract:
    """Build a Pocket TTS language contract from the ungated repository."""
    artifact = _artifact(path, size, sha256, inventory)
    return ModelContract(
        backend_id="mini",
        model_id="lunahr/pocket-tts-ungated",
        revision=_POCKET_REVISION,
        artifacts=(artifact,),
        components=(
            _component(
                "flow_lm",
                path,
                inventory,
                RuntimeDtypeRule("", ("torch.bfloat16",)),
                quantization=_POCKET_QUANTIZATION,
            ),
        ),
        variant=language_variant,
    )


_MINI_CONTRACTS = (
    _pocket_contract(
        "english",
        "languages/english/model.safetensors",
        219029196,
        "473f47d99560bd50eb8b4509d3cacfe7f316ab20bdca86505403a2e6a936a6e9",
        _POCKET_ENGLISH_INVENTORY,
    ),
    _pocket_contract(
        "french_24l",
        "languages/french_24l/model.safetensors",
        672178676,
        "399758aa0352f47034f9d89297578efaac3aa611620e281fc9296e834ed5be5f",
        _POCKET_FRENCH_INVENTORY,
    ),
    _pocket_contract(
        "german",
        "languages/german/model.safetensors",
        219029196,
        "06aaf44c27faf74a82bb6e4c277b88f566a62e135c9550254b2f23226948dfb1",
        _POCKET_ENGLISH_INVENTORY,
    ),
    _pocket_contract(
        "italian",
        "languages/italian/model.safetensors",
        219029196,
        "3e2d714aa3ef7d95d31226c3f57226b5d57ca5b207acf574a24c6764ddac06bc7",
        _POCKET_ENGLISH_INVENTORY,
    ),
    _pocket_contract(
        "portuguese",
        "languages/portuguese/model.safetensors",
        219029196,
        "b70702d8cf5bd83f7af7016eee20dd4d2f9b59aa3091ee96df5f24da392e6941",
        _POCKET_ENGLISH_INVENTORY,
    ),
    _pocket_contract(
        "spanish",
        "languages/spanish/model.safetensors",
        219029196,
        "1a4d84c547893941a5515b8121c671b8fe079d41382d58aec9dc442961258c71",
        _POCKET_ENGLISH_INVENTORY,
    ),
)


_QWEN_COMMON_TOKENIZER = _artifact(
    "speech_tokenizer/model.safetensors",
    682293092,
    "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    _QWEN_TOKENIZER_INVENTORY,
)


_QWEN_CONTRACTS = (
    ModelContract(
        backend_id="qwen3",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        revision=_QWEN06_REVISION,
        artifacts=(
            _artifact(
                "model.safetensors",
                1829344272,
                "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",
                _QWEN06_INVENTORY,
            ),
            _QWEN_COMMON_TOKENIZER,
        ),
        components=(
            _component(
                "talker",
                "model.safetensors",
                _QWEN06_INVENTORY,
                RuntimeDtypeRule("", ("torch.bfloat16",)),
                quantization=_TRANSFORMER_QUANTIZATION,
            ),
            _component(
                "speech_tokenizer",
                "speech_tokenizer/model.safetensors",
                _QWEN_TOKENIZER_INVENTORY,
                RuntimeDtypeRule("", ("torch.float32",)),
            ),
        ),
    ),
    ModelContract(
        backend_id="qwen3",
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        revision=_QWEN17_REVISION,
        artifacts=(
            _artifact(
                "model.safetensors",
                3857413744,
                "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c",
                _QWEN17_INVENTORY,
            ),
            _QWEN_COMMON_TOKENIZER,
        ),
        components=(
            _component(
                "talker",
                "model.safetensors",
                _QWEN17_INVENTORY,
                RuntimeDtypeRule("", ("torch.bfloat16",)),
                quantization=_TRANSFORMER_QUANTIZATION,
            ),
            _component(
                "speech_tokenizer",
                "speech_tokenizer/model.safetensors",
                _QWEN_TOKENIZER_INVENTORY,
                RuntimeDtypeRule("", ("torch.float32",)),
            ),
        ),
    ),
)


_DOTS_CONTRACT = ModelContract(
    backend_id="dotstts",
    model_id="rednote-hilab/dots.tts-mf",
    revision=_DOTS_REVISION,
    artifacts=(
        _artifact(
            "model.safetensors",
            4398915254,
            "a16d5798da197bf647fc01915236873e4672e975b0341360703ec49d002c4696",
            _DOTS_CORE_INVENTORY,
        ),
        _artifact(
            "speaker_encoder.safetensors",
            29150484,
            "1cf3861c9dee79e4db34bd0b8a4155e68bed27a7c6274e168bb6ee4fed191c85",
            _DOTS_SPEAKER_INVENTORY,
        ),
        _artifact(
            "vocoder.safetensors",
            723585584,
            "c0e45c08f480df67ac4c354b465355fcc7e2f6c8765263b6dfeddd1f4671c93d",
            _DOTS_VOCODER_INVENTORY,
        ),
        _artifact(
            "latent_stats.pt",
            3197,
            "313b13af56d659ecf869d5f854508fcf823c8f957aefc6bc05244991abd6ffe1",
        ),
    ),
    components=(
        _component(
            "core",
            "model.safetensors",
            _DOTS_CORE_INVENTORY,
            RuntimeDtypeRule("", ("torch.bfloat16",)),
            quantization=_TRANSFORMER_QUANTIZATION,
        ),
        _component(
            "speaker_encoder",
            "speaker_encoder.safetensors",
            _DOTS_SPEAKER_INVENTORY,
            RuntimeDtypeRule("", ("torch.float32", "torch.int64")),
        ),
        _component(
            "vocoder",
            "vocoder.safetensors",
            _DOTS_VOCODER_INVENTORY,
            RuntimeDtypeRule("", ("torch.float32",)),
        ),
    ),
)


_VOX_CONTRACT = ModelContract(
    backend_id="voxcpm2",
    model_id="openbmb/VoxCPM2",
    revision=_VOX_REVISION,
    artifacts=(
        _artifact(
            "model.safetensors",
            4580080592,
            "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d",
            _VOX_INVENTORY,
        ),
    ),
    components=(
        _component(
            "tts_model",
            "model.safetensors",
            _VOX_INVENTORY,
            RuntimeDtypeRule("", ("torch.bfloat16",)),
            quantization=_TRANSFORMER_QUANTIZATION,
        ),
    ),
)


_FIRERED_CONTRACT = ModelContract(
    backend_id="fireredtts3",
    model_id="FireRedTeam/FireRedTTS3",
    revision=_FIRERED_REVISION,
    artifacts=(
        _artifact(
            "fireredtts3_base/model.safetensors",
            8482608484,
            "d6ceed109a04207ef48bc669fb68248d3a44d990fcf2b9641990a6134d2ebb8a",
            _FIRERED_CORE_INVENTORY,
        ),
        _artifact(
            "redae/model.safetensors",
            3775160672,
            "0723e87fdaf46d377f01aa32ce56b43f43f9b249415bc8b8f81040eaa8b14abe",
            _FIRERED_REDAE_INVENTORY,
        ),
    ),
    components=(
        _component(
            "tts_core",
            "fireredtts3_base/model.safetensors",
            _FIRERED_CORE_INVENTORY,
            RuntimeDtypeRule("backbone_llm", ("torch.bfloat16",)),
            RuntimeDtypeRule("stop_head", ("torch.bfloat16",)),
            RuntimeDtypeRule("dit", ("torch.float32",)),
            RuntimeDtypeRule("dit_head", ("torch.float32",)),
            RuntimeDtypeRule("patch_encoder", ("torch.float32",)),
            RuntimeDtypeRule("spk_proj_dit", ("torch.float32",)),
            RuntimeDtypeRule("spk_proj_llm", ("torch.float32",)),
            quantization=_FIRERED_QUANTIZATION,
        ),
        _component(
            "redae",
            "redae/model.safetensors",
            _FIRERED_REDAE_INVENTORY,
            RuntimeDtypeRule("encoder", ("torch.bfloat16",)),
            RuntimeDtypeRule("decoder", ("torch.float32",)),
        ),
    ),
)


_LUXTTS_CONTRACT = ModelContract(
    backend_id="luxtts",
    model_id="YatharthS/LuxTTS",
    revision=_LUXTTS_REVISION,
    artifacts=(
        _artifact("model.pt", 491318136, "745855037478eb888a1b568255c46e9275f3997"),
        _artifact(
            "text_encoder.onnx",
            17633735,
            "495eca2d5f8a911f5c361bcce5bd55cdd2508ccdd26ce3e9bf1d3c29eb974861",
        ),
        _artifact(
            "text_encoder_int8.onnx",
            5570211,
            "f2de9a761a85e5ddd125dee6e05bad1c7ee92c11b83b4d775dab216a6aa41379",
        ),
        _artifact(
            "fm_decoder.onnx",
            477534010,
            "4510d4f5f049f14ef80207fca695e13c820e2cea61635f402954950bc62b1e3c",
        ),
        _artifact(
            "fm_decoder_int8.onnx",
            124657100,
            "3cc2e08a96610d7ea1b227398e97cdbbe0414499741d3aec0b8113db2a2ab251",
        ),
        _artifact(
            "vocoder/vocos.bin",
            63972079,
            "116b9875a0369d6a0156d752b4548121fe75fdc81d39943e81c46ac9bfa72d11",
        ),
    ),
    components=(),
)


MODEL_CONTRACTS: tuple[ModelContract, ...] = (
    *_MINI_CONTRACTS,
    *_QWEN_CONTRACTS,
    _DOTS_CONTRACT,
    _VOX_CONTRACT,
    _FIRERED_CONTRACT,
    _LUXTTS_CONTRACT,
)


def model_contract(
    backend_id: str,
    model_id: str,
    variant: Optional[str] = None,
) -> ModelContract:
    """Return the pinned contract for one backend model variant.

    Args:
        backend_id: Celune backend identifier.
        model_id: Hugging Face repository identifier.
        variant: Optional backend-specific model variant, such as a Pocket TTS language directory.

    Returns:
        ModelContract: The matching pinned model contract.

    Raises:
        ModelContractError: No pinned contract matches the requested model.
    """
    matches = tuple(
        contract
        for contract in MODEL_CONTRACTS
        if contract.backend_id == backend_id
        and contract.model_id == model_id
        and (variant is None or contract.variant == variant)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and variant is None:
        raise ModelContractError(
            f"model contract variant is required for {backend_id}/{model_id}"
        )
    raise ModelContractError(
        f"no pinned model contract for {backend_id}/{model_id}"
        + (f" variant={variant}" if variant is not None else "")
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local weight artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_inventory(
    tensors: Mapping[str, tuple[tuple[int, ...], str]],
) -> str:
    """Hash sorted tensor names, shapes, and dtypes as a stable inventory."""
    payload = [
        {"name": name, "shape": list(shape), "dtype": dtype}
        for name, (shape, dtype) in sorted(tensors.items())
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dtype_name(dtype: object) -> str:
    """Normalize a tensor or safetensors dtype to the contract spelling."""
    value = str(dtype)
    if value.startswith("torch."):
        return value
    return {
        "BF16": "torch.bfloat16",
        "F16": "torch.float16",
        "F32": "torch.float32",
        "F64": "torch.float64",
        "I64": "torch.int64",
        "I32": "torch.int32",
        "I8": "torch.int8",
        "U8": "torch.uint8",
    }.get(value, value)


def _inventory_from_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, tuple[tuple[int, ...], str]], Counter[str], int]:
    """Build a structural inventory from a loaded PyTorch state dictionary."""
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    dtype_counts: Counter[str] = Counter()
    parameter_count = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ModelContractError("model state contains an invalid tensor entry")
        shape = tuple(int(value) for value in tensor.shape)
        dtype = _dtype_name(tensor.dtype)
        tensors[name] = (shape, dtype)
        dtype_counts[dtype] += 1
        parameter_count += tensor.numel()
    return tensors, dtype_counts, parameter_count


def _checkpoint_layer_name(tensor_name: str) -> str:
    """Return the owning layer name for a tensor parameter name."""
    for suffix in (".weight", ".bias"):
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _dtype_value(dtype: str) -> Union[str, torch.dtype]:
    """Return a PyTorch dtype for a contract dtype spelling when available."""
    return {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
    }.get(dtype, dtype)


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor uses a supported quantized representation."""
    if _dtype_name(tensor.dtype) in {
        "torch.float8_e4m3fn",
        "torch.float8_e4m3fnuz",
        "torch.float8_e5m2",
        "torch.float8_e5m2fnuz",
        "torch.int8",
        "torch.uint8",
    }:
        return True
    return tensor.__class__.__module__.startswith("torchao.")


def _allows_quantized_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    quantization: QuantizationRule,
) -> bool:
    """Return whether a tensor is an approved quantized layer parameter."""
    if not _is_quantized_tensor(tensor) or not tensor_name.endswith(".weight"):
        return False
    layer_name = _checkpoint_layer_name(tensor_name)
    if layer_name.rsplit(".", 1)[-1] not in quantization.module_suffixes:
        return False
    return not any(
        token in layer_name.split(".") for token in quantization.excluded_module_names
    )


def _invalid_state(
    *,
    backend: str,
    filename: str,
    path: str,
    reason: str,
    tensor_name: Optional[str] = None,
    shape: Optional[tuple[int, ...]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    expected: Optional[Union[str, torch.dtype]] = None,
    actual: Optional[Union[str, torch.dtype]] = None,
) -> InvalidCheckpoint:
    """Build a structured checkpoint failure for a loaded model state."""
    return InvalidCheckpoint(
        backend=backend,
        filename=filename,
        path=path,
        tensor_name=tensor_name,
        layer_name=(
            _checkpoint_layer_name(tensor_name) if tensor_name is not None else None
        ),
        shape=shape,
        dtype=dtype,
        expected=expected,
        actual=actual,
        reason=reason,
    )


def validate_model_state(
    state: Mapping[str, torch.Tensor],
    contract: TensorInventoryContract,
    *,
    name: str,
    backend: str = "unknown",
    filename: Optional[str] = None,
    path: Optional[str] = None,
    runtime_dtypes: tuple[RuntimeDtypeRule, ...] = (),
    allow_quantized: bool = False,
    quantization: Optional[QuantizationRule] = None,
) -> None:
    """Validate a loaded model state against one tensor inventory contract.

    Args:
        state: State dictionary from the loaded backend component.
        contract: Expected tensor inventory derived from the pinned artifact.
        name: Diagnostic component name.
        backend: Celune backend identifier.
        filename: Checkpoint filename for diagnostics.
        path: Checkpoint-relative path for diagnostics.
        runtime_dtypes: Optional prefix-specific runtime dtype rules.
        allow_quantized: Permit approved quantized layer weights while checking
            the unquantized structure and non-quantized dtypes.
        quantization: Contract rule that identifies approved quantized layers.

    Raises:
        InvalidCheckpoint: The state is structurally incomplete, inconsistent, or invalid.
    """
    checkpoint_filename = filename or "model.safetensors"
    checkpoint_path = path or name
    try:
        tensors, dtype_counts, parameter_count = _inventory_from_state(state)
    except ModelContractError as exc:
        raise _invalid_state(
            backend=backend,
            filename=checkpoint_filename,
            path=checkpoint_path,
            reason=str(exc),
        ) from exc
    quantized_state = bool(
        allow_quantized
        and quantization is not None
        and any(
            _allows_quantized_tensor(tensor_name, state[tensor_name], quantization)
            for tensor_name in tensors
        )
    )
    if len(tensors) != contract.tensor_count and not quantized_state:
        raise _invalid_state(
            backend=backend,
            filename=checkpoint_filename,
            path=checkpoint_path,
            reason=f"{name} tensor count mismatch: expected {contract.tensor_count}, got {len(tensors)}",
        )
    if parameter_count != contract.parameter_count and not quantized_state:
        raise _invalid_state(
            backend=backend,
            filename=checkpoint_filename,
            path=checkpoint_path,
            reason=f"{name} parameter count mismatch: expected {contract.parameter_count}, got {parameter_count}",
        )
    if not allow_quantized:
        expected_counts = dict(contract.dtype_counts)
        if dict(dtype_counts) != expected_counts:
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} dtype counts mismatch: expected {expected_counts}, got {dict(dtype_counts)}",
            )
        inventory_digest = _canonical_inventory(tensors)
        if inventory_digest != contract.inventory_sha256:
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} tensor inventory mismatch: expected {contract.inventory_sha256}, got {inventory_digest}",
            )

    for tensor_name, (_shape, dtype) in tensors.items():
        matching_rules = tuple(
            rule
            for rule in runtime_dtypes
            if rule.prefix == ""
            or tensor_name == rule.prefix
            or tensor_name.startswith(f"{rule.prefix}.")
        )
        quantized = bool(
            allow_quantized
            and quantization is not None
            and _allows_quantized_tensor(tensor_name, state[tensor_name], quantization)
        )
        if (
            matching_rules
            and not quantized
            and not any(dtype in rule.dtypes for rule in matching_rules)
        ):
            expected = sorted({item for rule in matching_rules for item in rule.dtypes})
            tensor = state[tensor_name]
            expected_value = _dtype_value(expected[0]) if len(expected) == 1 else None
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} tensor has dtype {dtype}; expected one of {expected}",
                tensor_name=tensor_name,
                shape=tensors[tensor_name][0],
                dtype=tensor.dtype,
                expected=expected_value,
                actual=tensor.dtype,
            )
        tensor = state[tensor_name]
        if (
            not quantized
            and tensor.is_floating_point()
            and not bool(torch.isfinite(tensor).all())
        ):
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} tensor contains non-finite values",
                tensor_name=tensor_name,
                shape=tensors[tensor_name][0],
                dtype=tensor.dtype,
                actual=tensor.dtype,
            )


def validate_safetensors_artifact(
    root: Union[str, Path],
    artifact: WeightArtifactContract,
    *,
    backend: str = "unknown",
) -> None:
    """Validate one local safetensors artifact and its header inventory.

    Args:
        root: Root directory of the local model snapshot.
        artifact: Expected artifact metadata.
        backend: Celune backend identifier.

    Raises:
        InvalidCheckpoint: The artifact is missing, changed, unreadable, or structurally invalid.
    """
    path = Path(root) / artifact.path
    filename = Path(artifact.path).name
    if not path.is_file():
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact is missing",
        )
    try:
        size = path.stat().st_size
        if size != artifact.size:
            raise InvalidCheckpoint(
                backend=backend,
                filename=filename,
                path=artifact.path,
                reason=f"model artifact size mismatch: expected {artifact.size}, got {size}",
            )
        digest = _sha256(path)
    except InvalidCheckpoint:
        raise
    except OSError as exc:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact could not be read",
        ) from exc
    if digest != artifact.sha256:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact checksum mismatch",
        )
    if artifact.inventory is None:
        return

    try:
        tensors = _read_safetensors_inventory(path)
    except Exception as exc:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact could not be parsed",
        ) from exc

    actual = _inventory_from_header(tensors)
    if actual != artifact.inventory:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact tensor inventory mismatch",
        )


def _inventory_from_header(
    tensors: Mapping[str, tuple[tuple[int, ...], str]],
) -> TensorInventoryContract:
    """Convert a safetensors header map into a compact inventory."""
    dtype_counts = Counter(dtype for _shape, dtype in tensors.values())
    parameter_count = sum(
        _shape_parameter_count(shape) for shape, _dtype in tensors.values()
    )
    return TensorInventoryContract(
        tensor_count=len(tensors),
        parameter_count=parameter_count,
        dtype_counts=tuple(sorted(dtype_counts.items())),
        inventory_sha256=_canonical_inventory(tensors),
    )


def _read_safetensors_inventory(
    path: Path,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read the safetensors header without materializing model tensors."""
    try:
        with path.open("rb") as handle:
            header_length_bytes = handle.read(8)
            if len(header_length_bytes) != 8:
                raise ModelContractError("safetensors header length is truncated")
            header_length = int.from_bytes(header_length_bytes, "little")
            header = json.loads(handle.read(header_length).decode("utf-8"))
    except ModelContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise ModelContractError("safetensors header is invalid") from exc

    if not isinstance(header, dict):
        raise ModelContractError("safetensors header is not an object")

    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, raw_tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_tensor, dict):
            raise ModelContractError("safetensors tensor entry is invalid")
        raw_shape = raw_tensor.get("shape")
        raw_dtype = raw_tensor.get("dtype")
        if not isinstance(raw_shape, list) or not isinstance(raw_dtype, str):
            raise ModelContractError(f"safetensors tensor metadata is invalid: {name}")
        tensors[name] = (
            tuple(int(dimension) for dimension in raw_shape),
            _dtype_name(raw_dtype),
        )
    return tensors


def _shape_parameter_count(shape: tuple[int, ...]) -> int:
    """Return the number of scalar values represented by one tensor shape."""
    count = 1
    for dimension in shape:
        count *= dimension
    return count
