"""Model loading helpers for Celune."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Optional, Union, cast, overload

import torch
from torch import nn
from torch.nn import functional as _f
from safetensors import safe_open
from safetensors.torch import safe_open as safe_open_torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .backends import CeluneBackend
from .constants import NORMALIZER_MODEL_ID, T

CheckpointPath = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class HybridCheckpointIndex:
    """Tensor dtype index for hybrid INT8/BF16 checkpoints."""

    int8: tuple[str, ...]
    bf16: tuple[str, ...]
    quantization_metadata: dict[str, torch.dtype]
    other: dict[str, torch.dtype]

    @property
    def is_hybrid_int8_bf16(self) -> bool:
        """Return whether the checkpoint contains only INT8 and BF16 tensors."""
        return bool(self.int8) and bool(self.bf16) and not self.other

    @property
    def has_int8_tensors(self) -> bool:
        """Return whether the checkpoint contains actual INT8 tensors."""
        return bool(self.int8)


@dataclass(frozen=True)
class HybridInt8LinearProfile:
    """Backend-safe INT8 Linear tensor name profile."""

    name: str
    module_prefixes: tuple[str, ...]
    weight_suffixes: tuple[str, ...]


@dataclass
class HybridTTSModel(Generic[T]):
    """Loaded TTS model plus the checkpoint dtype index used to load it."""

    model: T
    index: HybridCheckpointIndex
    incompatible_keys: Any = None
    state_dict: Optional[dict[str, torch.Tensor]] = None


class Int8ScaledLinear(nn.Module):
    """Linear layer that stores INT8 weights and can cache runtime weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        scale_shape: torch.Size,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize an INT8-backed linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether the layer has a bias parameter.
            scale_shape: Shape of the checkpoint scale tensor.
            device: Optional device for initial buffers.

        Returns:
            None: This constructor registers INT8 weight and scale buffers.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "int8_scale",
            torch.empty(scale_shape, dtype=torch.float32, device=device),
        )
        self.register_buffer("runtime_weight", None, persistent=False)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def _dequantized_weight(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return this layer's dequantized weight tensor.

        Args:
            device: Target device for the returned tensor.
            dtype: Target dtype for the returned tensor.

        Returns:
            torch.Tensor: Dequantized weight tensor.
        """
        scale_tensor = cast(torch.Tensor, self.get_buffer("int8_scale"))
        weight_tensor = cast(torch.Tensor, self.get_buffer("weight"))
        scale = scale_tensor.to(device=device, dtype=dtype)
        if scale.ndim > 0:
            scale = scale.view(-1, 1)
        return weight_tensor.to(device=device, dtype=dtype) * scale

    def enable_runtime_weight_cache(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
        offload_quantized_buffers: bool = False,
    ) -> None:
        """Cache the dequantized weight used by fast Qwen3 CUDA graph replay.

        Args:
            device: Target device for the cached tensor.
            dtype: Target dtype for the cached tensor.
            offload_quantized_buffers: Whether to move original INT8 tensors to
                CPU after the runtime weight is cached.

        Returns:
            None: This method stores a non-persistent runtime tensor.
        """
        # must not be defined in __init__, contrary to what PyCharm expects
        self.runtime_weight = self._dequantized_weight(device=device, dtype=dtype)
        if offload_quantized_buffers:
            self.weight = cast(torch.Tensor, self.get_buffer("weight")).cpu()
            self.int8_scale = cast(torch.Tensor, self.get_buffer("int8_scale")).cpu()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply the linear projection using transiently dequantized weights.

        Args:
            inp: Input tensor to project.

        Returns:
            torch.Tensor: Projected output tensor.
        """
        runtime_weight = self.get_buffer("runtime_weight")
        if runtime_weight is not None:
            weight = cast(torch.Tensor, runtime_weight)
        else:
            weight = self._dequantized_weight(device=inp.device, dtype=inp.dtype)

        bias = (
            cast(torch.Tensor, self.bias).to(dtype=inp.dtype)
            if self.bias is not None
            else None
        )

        # https://github.com/pytorch/pytorch/issues/119482
        # pylint: disable=E1102
        return _f.linear(inp, weight, bias)


def _get_module(root: nn.Module, name: str) -> nn.Module:
    """Resolve a dotted module path from a root module.

    Args:
        root: Root module to traverse.
        name: Dotted child module path.

    Returns:
        nn.Module: Resolved module.

    Raises:
        AttributeError: A named child module cannot be found.
        KeyError: A numeric module-list child cannot be found.
    """
    module: nn.Module = root
    if not name:
        return module

    for part in name.split("."):
        module = (
            # this is an internal PyTorch call, so a warning is expected
            cast(nn.Module, module._modules[part])
            if part.isdigit()
            else cast(nn.Module, getattr(module, part))
        )
    return module


def replace_int8_linear_modules(
    model: nn.Module,
    index: HybridCheckpointIndex,
    state_dict: dict[str, torch.Tensor],
    profile: HybridInt8LinearProfile,
) -> int:
    """Replace checkpoint-marked linear layers with INT8-backed modules.

    Args:
        model: Qwen3 model whose modules should be replaced.
        index: Hybrid checkpoint index that identifies INT8 weights.
        state_dict: Checkpoint state dictionary containing scale tensor shapes.
        profile: Backend profile limiting allowed INT8 Linear names.

    Returns:
        int: Number of layers replaced.

    Raises:
        ValueError: An INT8 tensor is unsupported by the backend profile.
    """
    validate_int8_linear_checkpoint(index, state_dict, profile)
    count = 0
    for weight_name in index.int8:
        module_name = weight_name.removesuffix(".weight")
        parent_name, _, child_name = module_name.rpartition(".")
        parent = _get_module(model, parent_name)
        child = getattr(parent, child_name)
        if not isinstance(child, nn.Linear):
            raise ValueError(
                f"{profile.name} INT8 tensor '{weight_name}' targets "
                f"{type(child).__name__}, expected torch.nn.Linear"
            )

        replacement = Int8ScaledLinear(
            child.in_features,
            child.out_features,
            child.bias is not None,
            state_dict[f"{module_name}.weight.int8_scale"].shape,
            device=child.weight.device,
        )
        setattr(parent, child_name, replacement)
        count += 1
    return count


QWEN3_SAFE_TRANSFORMER_INT8_PROFILE = HybridInt8LinearProfile(
    name="Qwen3-TTS",
    module_prefixes=(
        "talker.model.layers.",
        "talker.code_predictor.model.layers.",
        "talker.codec_head_model.layers.",
    ),
    weight_suffixes=(
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ),
)


VOXCPM2_SAFE_TRANSFORMER_INT8_PROFILE = HybridInt8LinearProfile(
    name="VoxCPM2",
    module_prefixes=(
        "tts_model.base_lm.layers.",
        "tts_model.residual_lm.layers.",
        "tts_model.feat_encoder.encoder.layers.",
        "tts_model.feat_encoder.layers.",
        "tts_model.feat_decoder.estimator.decoder.layers.",
        "tts_model.feat_decoder.estimator.layers.",
        "base_lm.layers.",
        "residual_lm.layers.",
        "feat_encoder.encoder.layers.",
        "feat_encoder.layers.",
        "feat_decoder.estimator.decoder.layers.",
        "feat_decoder.estimator.layers.",
    ),
    weight_suffixes=(
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ),
)


def _matches_int8_profile(name: str, profile: HybridInt8LinearProfile) -> bool:
    """Return whether an INT8 tensor name is allowed by a backend profile."""
    return name.startswith(profile.module_prefixes) and name.endswith(
        profile.weight_suffixes
    )


def validate_int8_linear_checkpoint(
    index: HybridCheckpointIndex,
    state_dict: Mapping[str, torch.Tensor],
    profile: HybridInt8LinearProfile,
) -> None:
    """Validate INT8 tensors before any inference path can use them.

    Args:
        index: Checkpoint dtype index.
        state_dict: Loaded state dictionary used to verify scale tensors.
        profile: Backend-safe module profile.

    Raises:
        ValueError: A tensor is missing scale metadata or is unsupported.
    """
    missing_scales: list[str] = []
    unsupported: list[str] = []

    for name in index.int8:
        scale_name = f"{name}.int8_scale"
        if scale_name not in state_dict:
            missing_scales.append(scale_name)
        if not _matches_int8_profile(name, profile):
            unsupported.append(name)

    if missing_scales:
        preview = ", ".join(missing_scales[:8])
        raise ValueError(
            f"{profile.name} hybrid INT8 checkpoint is missing scale tensors: {preview}"
        )

    if unsupported:
        preview = ", ".join(unsupported[:8])
        raise ValueError(
            f"{profile.name} hybrid INT8 checkpoint contains unsupported INT8 "
            f"tensors: {preview}. Only safe transformer Linear weights are "
            "accepted."
        )


def qwen3_int8_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename Qwen3 INT8 scale tensors for ``Int8ScaledLinear`` modules.

    Args:
        state_dict: Original checkpoint state dictionary.

    Returns:
        dict[str, torch.Tensor]: State dictionary with scale tensors renamed.
    """
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight.int8_scale"):
            converted[f"{name.removesuffix('.weight.int8_scale')}.int8_scale"] = tensor
        else:
            converted[name] = tensor
    return converted


def enable_qwen3_int8_runtime_cache(
    model: nn.Module,
    device: Union[torch.device, str],
    dtype: torch.dtype,
    offload_quantized_buffers: bool = False,
    module_filter: Optional[Callable[[str, Int8ScaledLinear], bool]] = None,
) -> int:
    """Enable cached dequantized runtime weights for Qwen3 INT8 modules.

    Args:
        model: Model containing INT8-backed linear layers.
        device: Target device for cached runtime weights.
        dtype: Target dtype for cached runtime weights.
        offload_quantized_buffers: Whether to move INT8 checkpoint tensors to CPU
            after the runtime weight is cached.
        module_filter: Optional predicate for deciding which modules to cache.

    Returns:
        int: Number of layers updated.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Int8ScaledLinear):
            if module_filter is not None and not module_filter(name, module):
                continue
            module.enable_runtime_weight_cache(
                device=device,
                dtype=dtype,
                offload_quantized_buffers=offload_quantized_buffers,
            )
            count += 1
    return count


def _checkpoint_files(checkpoint: CheckpointPath) -> list[Path]:
    """Resolve checkpoint files from a file or directory path.

    Args:
        checkpoint: A checkpoint file or directory containing checkpoint shards.

    Returns:
        list[Path]: Ordered checkpoint files to load or inspect.

    Raises:
        FileNotFoundError: The path does not exist or has no supported files.
    """
    path = Path(checkpoint)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")

    for pattern in ("model*.safetensors", "*.safetensors", "pytorch_model*.bin"):
        files = sorted(path.glob(pattern))
        if files:
            return files

    files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))
    if files:
        return files

    raise FileNotFoundError(f"no supported checkpoint files found in: {path}")


def _load_safetensors_file(
    path: Path, device: Union[str, torch.device]
) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors checkpoint file.

    Args:
        path: Safetensors file to load.
        device: Device used by safetensors while reading tensors.

    Returns:
        dict[str, torch.Tensor]: Tensor state dictionary from the file.
    """
    tensors: dict[str, torch.Tensor] = {}
    with safe_open_torch(path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _safetensors_dtype(dtype: str) -> torch.dtype:
    """Convert a safetensors dtype label into a PyTorch dtype.

    Args:
        dtype: Safetensors dtype label.

    Returns:
        torch.dtype: Equivalent PyTorch dtype.

    Raises:
        ValueError: The safetensors dtype label is unsupported.
    """
    dtype_map = {
        "I8": torch.int8,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    try:
        return dtype_map[dtype]
    except KeyError as e:
        raise ValueError(f"unsupported safetensors dtype: {dtype}") from e


def _inspect_safetensors_file(path: Path) -> dict[str, torch.dtype]:
    """Inspect tensor dtypes in a safetensors file without loading payloads.

    Args:
        path: Safetensors file to inspect.

    Returns:
        dict[str, torch.dtype]: Tensor names mapped to dtypes.
    """
    dtypes: dict[str, torch.dtype] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            dtypes[key] = _safetensors_dtype(f.get_slice(key).get_dtype())
    return dtypes


def _load_torch_file(
    path: Path, map_location: Union[str, torch.device]
) -> dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint file as a tensor state dictionary.

    Args:
        path: PyTorch checkpoint file to load.
        map_location: Device mapping used by ``torch.load``.

    Returns:
        dict[str, torch.Tensor]: Tensor state dictionary from the file.

    Raises:
        TypeError: The checkpoint does not contain a tensor state dictionary.
    """
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location=map_location)

    if isinstance(loaded, Mapping):
        for key in ("state_dict", "model", "module"):
            nested = loaded.get(key)
            if isinstance(nested, Mapping) and all(
                isinstance(value, torch.Tensor) for value in nested.values()
            ):
                return dict(nested)

        if all(isinstance(value, torch.Tensor) for value in loaded.values()):
            return dict(loaded)

    raise TypeError(f"checkpoint file does not contain a tensor state dict: {path}")


def _index_hybrid_dtypes(
    dtypes: Mapping[str, torch.dtype],
) -> HybridCheckpointIndex:
    """Build a hybrid checkpoint index from tensor dtypes.

    Args:
        dtypes: Tensor names mapped to dtypes.

    Returns:
        HybridCheckpointIndex: Classification of INT8, BF16, metadata, and other tensors.
    """
    int8: list[str] = []
    bf16: list[str] = []
    quantization_metadata: dict[str, torch.dtype] = {}
    other: dict[str, torch.dtype] = {}

    for name, dtype in dtypes.items():
        if dtype is torch.int8:
            int8.append(name)
        elif dtype is torch.bfloat16:
            bf16.append(name)
        elif _is_quantization_metadata(name):
            quantization_metadata[name] = dtype
        else:
            other[name] = dtype

    return HybridCheckpointIndex(
        int8=tuple(sorted(int8)),
        bf16=tuple(sorted(bf16)),
        quantization_metadata=dict(sorted(quantization_metadata.items())),
        other=dict(sorted(other.items())),
    )


def inspect_hybrid_tts_checkpoint(
    checkpoint: Union[CheckpointPath, Mapping[str, torch.Tensor]],
) -> HybridCheckpointIndex:
    """Inspect checkpoint tensor dtypes without loading safetensors payloads."""
    if isinstance(checkpoint, Mapping):
        if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            raise TypeError("checkpoint mapping must contain only torch.Tensor values")
        return _index_hybrid_tensors(checkpoint)

    dtypes: dict[str, torch.dtype] = {}
    for file in _checkpoint_files(checkpoint):
        if file.suffix == ".safetensors":
            part = _inspect_safetensors_file(file)
        else:
            part = {
                name: tensor.dtype
                for name, tensor in _load_torch_file(file, "cpu").items()
            }

        overlap = dtypes.keys() & part.keys()
        if overlap:
            names = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"duplicate tensor names across checkpoint shards: {names}"
            )
        dtypes.update(part)

    return _index_hybrid_dtypes(dtypes)


def _load_checkpoint_state_dict(
    checkpoint: Union[CheckpointPath, Mapping[str, torch.Tensor]],
    map_location: Union[str, torch.device],
) -> dict[str, torch.Tensor]:
    """Load a checkpoint path or tensor mapping into a state dictionary.

    Args:
        checkpoint: Checkpoint path, checkpoint directory, or tensor mapping.
        map_location: Device mapping used for PyTorch checkpoint files.

    Returns:
        dict[str, torch.Tensor]: Combined tensor state dictionary.

    Raises:
        TypeError: The checkpoint mapping contains non-tensor values.
        ValueError: Multiple shards contain the same tensor name.
    """
    if isinstance(checkpoint, Mapping):
        if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            raise TypeError("checkpoint mapping must contain only torch.Tensor values")
        return dict(checkpoint)

    state_dict: dict[str, torch.Tensor] = {}
    for file in _checkpoint_files(checkpoint):
        if file.suffix == ".safetensors":
            part = _load_safetensors_file(file, map_location)
        else:
            part = _load_torch_file(file, map_location)

        overlap = state_dict.keys() & part.keys()
        if overlap:
            names = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"duplicate tensor names across checkpoint shards: {names}"
            )
        state_dict.update(part)
    return state_dict


def _index_hybrid_tensors(
    state_dict: Mapping[str, torch.Tensor],
) -> HybridCheckpointIndex:
    """Build a hybrid checkpoint index from tensors.

    Args:
        state_dict: Tensor state dictionary to classify.

    Returns:
        HybridCheckpointIndex: Classification of INT8, BF16, metadata, and other tensors.
    """
    return _index_hybrid_dtypes(
        {name: tensor.dtype for name, tensor in state_dict.items()}
    )


def _is_quantization_metadata(name: str) -> bool:
    """Return whether a tensor name represents quantization metadata.

    Args:
        name: Tensor name to classify.

    Returns:
        bool: ``True`` when the tensor is quantization metadata.
    """
    return name.endswith(
        (
            ".quant_scale",
            ".weight.quant_scale",
            ".int8_scale",
            ".weight.int8_scale",
            ".scale",
            ".scales",
            ".weight_scale",
            ".weight_scales",
            ".zero_point",
            ".zero_points",
            ".weight_zero_point",
            ".weight_zero_points",
        )
    )


def _maybe_cast_bf16_tensors(
    state_dict: dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> dict[str, torch.Tensor]:
    """Optionally cast BF16 tensors while preserving INT8 tensors.

    Args:
        state_dict: Tensor state dictionary to update.
        dtype: Target dtype for BF16 tensors, or ``None`` to keep BF16.

    Returns:
        dict[str, torch.Tensor]: State dictionary with only BF16 tensors cast.
    """
    if dtype is None or dtype is torch.bfloat16:
        return state_dict

    return {
        name: tensor.to(dtype=dtype) if tensor.dtype is torch.bfloat16 else tensor
        for name, tensor in state_dict.items()
    }


def _call_model_factory(
    model_or_factory: Any,
    index: HybridCheckpointIndex,
) -> Any:
    """Resolve a model instance from an instance or factory callback.

    Args:
        model_or_factory: Model object or callable creating one.
        index: Hybrid checkpoint index passed to compatible factories.

    Returns:
        Any: Resolved model instance.
    """
    if not callable(model_or_factory) or hasattr(model_or_factory, "load_state_dict"):
        return model_or_factory

    try:
        parameters = inspect.signature(model_or_factory).parameters
    except (TypeError, ValueError):
        return model_or_factory(index)

    if parameters:
        return model_or_factory(index)
    return model_or_factory()


def _load_state_dict_into_model(
    model: Any,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool,
) -> Any:
    """Load a state dictionary into a model with assign-aware fallback.

    Args:
        model: Model that exposes ``load_state_dict``.
        state_dict: Tensor state dictionary to load.
        strict: Whether missing or unexpected keys are errors.

    Returns:
        Any: The incompatible key result returned by the model.

    Raises:
        TypeError: The model does not expose ``load_state_dict``.
        RuntimeError: Loading fails for a reason unrelated to INT8 gradients.
    """
    if not hasattr(model, "load_state_dict"):
        raise TypeError(
            "model object must provide load_state_dict or use load_state_dict_fn"
        )

    try:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    except TypeError:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if (
            "Only Tensors of floating point and complex dtype can require gradients"
            not in str(e)
        ):
            raise
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        return model.load_state_dict(state_dict, strict=strict, assign=True)


@overload
def load_hybrid_tts_checkpoint(
    checkpoint: Union[CheckpointPath, Mapping[str, torch.Tensor]],
    model_or_factory: Union[T, Callable[[HybridCheckpointIndex], T]],
    *,
    map_location: Union[str, torch.device] = "cpu",
    bf16_dtype: Optional[torch.dtype] = None,
    strict: bool = True,
    allow_extra_dtypes: bool = False,
    load_state_dict_fn: Optional[
        Callable[[T, Mapping[str, torch.Tensor], bool], Any]
    ] = None,
    keep_state_dict: bool = False,
    evaluate: bool = True,
) -> HybridTTSModel[T]:
    """Typed overload for loading into a model.

    Args:
        checkpoint: Checkpoint path, directory, or tensor mapping.
        model_or_factory: Model instance or factory callback.
        map_location: Device mapping used while reading checkpoints.
        bf16_dtype: Optional target dtype for BF16 tensors.
        strict: Whether missing or unexpected keys are errors.
        allow_extra_dtypes: Whether tensors outside INT8/BF16 are allowed.
        load_state_dict_fn: Optional custom state-dict loader.
        keep_state_dict: Whether to retain the loaded state dictionary.
        evaluate: Whether to put the model in evaluate mode.

    Returns:
        HybridTTSModel[T]: Loaded model wrapper.
    """


@overload
def load_hybrid_tts_checkpoint(
    checkpoint: Union[CheckpointPath, Mapping[str, torch.Tensor]],
    model_or_factory: None = None,
    *,
    map_location: Union[str, torch.device] = "cpu",
    bf16_dtype: Optional[torch.dtype] = None,
    strict: bool = True,
    allow_extra_dtypes: bool = False,
    load_state_dict_fn: Optional[
        Callable[[T, Mapping[str, torch.Tensor], bool], Any]
    ] = None,
    keep_state_dict: bool = True,
    evaluate: bool = True,
) -> HybridTTSModel[None]:
    """Typed overload for checkpoint inspection with retained state.

    Args:
        checkpoint: Checkpoint path, directory, or tensor mapping.
        model_or_factory: ``None`` when no model should be loaded.
        map_location: Device mapping used while reading checkpoints.
        bf16_dtype: Optional target dtype for BF16 tensors.
        strict: Reserved for overload compatibility.
        allow_extra_dtypes: Whether tensors outside INT8/BF16 are allowed.
        load_state_dict_fn: Reserved for overload compatibility.
        keep_state_dict: Whether to retain the loaded state dictionary.
        evaluate: Reserved for overload compatibility.

    Returns:
        HybridTTSModel[None]: Checkpoint wrapper without a model.
    """


def load_hybrid_tts_checkpoint(
    checkpoint: Union[CheckpointPath, Mapping[str, torch.Tensor]],
    model_or_factory: Optional[Union[T, Callable[[HybridCheckpointIndex], T]]] = None,
    *,
    map_location: Union[str, torch.device] = "cpu",
    bf16_dtype: Optional[torch.dtype] = None,
    strict: bool = True,
    allow_extra_dtypes: bool = False,
    load_state_dict_fn: Optional[
        Callable[[T, Mapping[str, torch.Tensor], bool], Any]
    ] = None,
    keep_state_dict: bool = False,
    evaluate: bool = True,
) -> Union[HybridTTSModel[T], HybridTTSModel[None]]:
    """Load a hybrid INT8/BF16 TTS checkpoint through a library-agnostic adapter.

    INT8 tensors are kept as INT8 so target libraries with quantized modules can
    consume them directly. BF16 tensors are kept as BF16 unless ``bf16_dtype`` is
    provided. Pass a model instance, or a factory that accepts the
    :class:`HybridCheckpointIndex`, so Qwen or any other backend can create the
    correct architecture before weights are assigned.
    """
    state_dict = _load_checkpoint_state_dict(checkpoint, map_location)
    index = _index_hybrid_tensors(state_dict)

    if index.other and not allow_extra_dtypes:
        preview = ", ".join(
            f"{name}={dtype}" for name, dtype in list(index.other.items())[:8]
        )
        raise ValueError(
            "checkpoint contains tensors outside INT8/BF16; "
            f"pass allow_extra_dtypes=True to permit them ({preview})"
        )

    state_dict = _maybe_cast_bf16_tensors(state_dict, bf16_dtype)

    if model_or_factory is None:
        return HybridTTSModel(
            model=None,
            index=index,
            state_dict=state_dict if keep_state_dict else None,
        )

    model = _call_model_factory(model_or_factory, index)
    incompatible_keys = (
        load_state_dict_fn(model, state_dict, strict)
        if load_state_dict_fn is not None
        else _load_state_dict_into_model(model, state_dict, strict)
    )

    evaluate_fn = getattr(model, "evaluate", None)
    if evaluate and callable(evaluate_fn):
        # we already checked if this is a callable, so this can be safely suppressed
        # pylint: disable=E1102
        evaluate_fn()

    return HybridTTSModel(
        model=model,
        index=index,
        incompatible_keys=incompatible_keys,
        state_dict=state_dict if keep_state_dict else None,
    )


def load_normalizer_components(
    log: Callable[[str, str], None],
    backend: Union[CeluneBackend, type[CeluneBackend]],
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load CeluneNorm and return its tokenizer and model.

    Args:
        log: Logging callback used to report cache and loading progress.
        backend: Backend type or instance used to resolve model cache helpers.

    Returns:
        tuple[PreTrainedTokenizerBase, PreTrainedModel]: The loaded tokenizer and
            causal language model.
    """
    available, path = backend.model_is_available_locally(NORMALIZER_MODEL_ID)
    model_ref = path if available and path is not None else NORMALIZER_MODEL_ID

    if available:
        log("Normalizer is already available in cache", "info")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    llm = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    return tokenizer, llm
