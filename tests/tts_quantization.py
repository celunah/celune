# SPDX-License-Identifier: Apache-2.0
"""Tests for TTS quantization policy and TorchAO integration boundaries."""

import importlib
from unittest.mock import patch

import torch
from torch import nn

from celune.backends.tts.contracts import (
    ModelContract,
    QuantizationRule,
    RuntimeDtypeRule,
    ModelComponentContract,
    TensorInventoryContract,
    _canonical_inventory,
    _inventory_from_state,
)
from celune.backends.tts.quantization import (
    _torchao_api,
    quantization_mode,
    quantize_component,
)
from tests.support import FakeBackend


def test_torchao_import_skips_native_enum_registration(capsys) -> None:
    """Avoid TorchAO's deprecated Enum pytree registration on current PyTorch."""
    _torchao_api()
    importlib.import_module("torchao.prototype.mx_formats")

    assert (
        "Calling register_constant() on Enum subclasses" not in capsys.readouterr().err
    )


def test_quantization_mode_selects_int8_for_sm86() -> None:
    """Use INT8 on Ampere devices without native FP8 support."""
    with (
        patch(
            "celune.backends.tts.quantization.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "celune.backends.tts.quantization.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
    ):
        assert quantization_mode(True) == "int8"


def test_quantization_mode_selects_fp8_for_sm89() -> None:
    """Use FP8 on Ada and newer devices."""
    with (
        patch(
            "celune.backends.tts.quantization.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "celune.backends.tts.quantization.torch.cuda.get_device_capability",
            return_value=(8, 9),
        ),
    ):
        assert quantization_mode(True) == "fp8"


def test_quantization_mode_keeps_bfloat16_when_disabled() -> None:
    """Never select a quantized format when configuration is false."""
    with patch(
        "celune.backends.tts.quantization.torch.cuda.is_available",
        return_value=True,
    ):
        assert quantization_mode(False) is None


def test_quantization_policy_only_targets_contract_layer_families() -> None:
    """Keep boundaries out of the contract-selected quantization set."""
    from celune.backends.tts.contracts import model_contract
    from celune.backends.tts.quantization import _is_safe_linear

    contract = model_contract(
        "mini",
        "lunahr/pocket-tts-ungated",
        variant="english",
    )
    rule = contract.components[0].quantization
    assert rule is not None
    assert _is_safe_linear(nn.Linear(4, 4), "flow_lm.layers.0.linear1", rule)
    assert not _is_safe_linear(nn.Linear(4, 4), "flow_lm.input_linear", rule)


def test_quantization_components_prefer_declared_nested_module() -> None:
    """Scope TorchAO to a named child instead of the wrapper root."""
    root = nn.Module()
    root.core = nn.Linear(4, 4)
    root.vocoder = nn.Linear(4, 4)

    runtime = type("Runtime", (), {"model": root})()
    component = ModelComponentContract(
        name="core",
        artifact_path="model.safetensors",
        inventory=TensorInventoryContract(0, 0, (), ""),
        runtime_dtypes=(),
    )
    contract = ModelContract(
        backend_id="test",
        model_id="test",
        revision="test",
        artifacts=(),
        components=(component,),
    )

    components = FakeBackend().quantization_components(runtime, contract)

    assert components["core"] is root.core


def test_pocket_contract_names_its_runtime_quantization_root() -> None:
    """Keep Pocket TTS's contract component aligned with its loaded module."""
    from celune.backends.tts.contracts import model_contract

    contract = model_contract(
        "mini",
        "lunahr/pocket-tts-ungated",
        variant="english",
    )

    assert contract.components[0].name == "flow_lm"


def test_int8_conversion_passes_post_quantization_contract_validation() -> None:
    """Exercise TorchAO conversion without using a real speech model or GPU."""
    module = nn.Module()
    module.q_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
    state = module.state_dict()
    tensors, dtype_counts, parameter_count = _inventory_from_state(state)
    component = ModelComponentContract(
        name="toy",
        artifact_path="model.safetensors",
        inventory=TensorInventoryContract(
            tensor_count=len(tensors),
            parameter_count=parameter_count,
            dtype_counts=tuple(sorted(dtype_counts.items())),
            inventory_sha256=_canonical_inventory(tensors),
        ),
        runtime_dtypes=(RuntimeDtypeRule("", ("torch.bfloat16",)),),
        quantization=QuantizationRule(module_suffixes=("q_proj",)),
    )

    assert quantize_component(module, component, mode="int8", backend="test") == 1


def test_quantization_releases_source_state_before_torchao() -> None:
    """Do not retain the pre-quantization state while TorchAO replaces weights."""
    source_state = {
        "q_proj.weight": torch.ones((4, 4), dtype=torch.bfloat16),
        "q_proj.bias": torch.ones((4,), dtype=torch.bfloat16),
    }
    tensors, dtype_counts, parameter_count = _inventory_from_state(source_state)
    component = ModelComponentContract(
        name="toy",
        artifact_path="model.safetensors",
        inventory=TensorInventoryContract(
            tensor_count=len(tensors),
            parameter_count=parameter_count,
            dtype_counts=tuple(sorted(dtype_counts.items())),
            inventory_sha256=_canonical_inventory(tensors),
        ),
        runtime_dtypes=(RuntimeDtypeRule("", ("torch.bfloat16",)),),
        quantization=QuantizationRule(module_suffixes=("q_proj",)),
    )
    released = [False]

    class TrackingState(dict):
        """Record when the first state mapping becomes unreachable."""

        def __del__(self) -> None:
            released[0] = True

    class TrackingModule(nn.Module):
        """Return a tracked mapping for the pre-quantization validation only."""

        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
            self._state_dict_calls = 0

        def state_dict(self, *args, **kwargs):
            state = super().state_dict(*args, **kwargs)
            self._state_dict_calls += 1
            if self._state_dict_calls == 1:
                return TrackingState(state)
            return state

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply the tracked linear layer to one input tensor."""
            return self.q_proj(value)

    module = TrackingModule()

    def fake_quantize(target, _config, *, filter_fn):
        assert released[0]
        assert filter_fn(target.q_proj, "q_proj")
        target.q_proj.weight = nn.Parameter(
            torch.ones((4, 4), dtype=torch.int8), requires_grad=False
        )

    with patch(
        "celune.backends.tts.quantization._torchao_api",
        return_value=(fake_quantize, object, object),
    ):
        assert quantize_component(module, component, mode="int8", backend="test") == 1


def test_quantization_does_not_move_components_between_devices() -> None:
    """Keep TorchAO in place so live CUDA models are not duplicated by staging."""
    module = nn.Module()
    module.q_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
    state = module.state_dict()
    tensors, dtype_counts, parameter_count = _inventory_from_state(state)
    component = ModelComponentContract(
        name="toy",
        artifact_path="model.safetensors",
        inventory=TensorInventoryContract(
            tensor_count=len(tensors),
            parameter_count=parameter_count,
            dtype_counts=tuple(sorted(dtype_counts.items())),
            inventory_sha256=_canonical_inventory(tensors),
        ),
        runtime_dtypes=(RuntimeDtypeRule("", ("torch.bfloat16",)),),
        quantization=QuantizationRule(module_suffixes=("q_proj",)),
    )
    moves: list[torch.device] = []

    def track_move(*, device: torch.device) -> nn.Module:
        moves.append(device)
        return module

    def fake_quantize(target, _config, *, filter_fn):
        assert filter_fn(target.q_proj, "q_proj")
        target.q_proj.weight = nn.Parameter(
            torch.ones((4, 4), dtype=torch.int8), requires_grad=False
        )

    with (
        patch.object(module, "to", side_effect=track_move),
        patch(
            "celune.backends.tts.quantization._torchao_api",
            return_value=(fake_quantize, object, object),
        ),
    ):
        assert quantize_component(module, component, mode="int8", backend="test") == 1

    assert not moves
