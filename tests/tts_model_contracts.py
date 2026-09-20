# SPDX-License-Identifier: Apache-2.0
"""Tests for the pinned TTS model-weight contracts."""

import torch
import pytest

from celune.exceptions import ModelContractError
from celune.backends.tts.contracts import (
    MODEL_CONTRACTS,
    RuntimeDtypeRule,
    TensorInventoryContract,
    model_contract,
    _canonical_inventory,
    validate_model_state,
)


def test_all_tts_contracts_have_pinned_artifacts() -> None:
    """Require every registered TTS contract to pin a non-empty artifact set."""
    assert MODEL_CONTRACTS
    for contract in MODEL_CONTRACTS:
        assert contract.revision
        assert contract.artifacts
        assert all(len(artifact.sha256) == 64 for artifact in contract.artifacts)


def test_pocket_contract_uses_the_ungated_repository_and_language_variant() -> None:
    """Resolve Pocket TTS through the explicitly ungated repository."""
    contract = model_contract(
        "mini",
        "lunahr/pocket-tts-ungated",
        variant="french_24l",
    )

    assert contract.revision == "d03cd73415a8d46d8eb115c7b524aebb0a729f4a"
    assert contract.artifacts[0].path == "languages/french_24l/model.safetensors"
    assert contract.artifacts[0].inventory is not None
    assert contract.artifacts[0].inventory.tensor_count == 358


def test_qwen_contract_requires_a_variant_when_model_id_is_ambiguous() -> None:
    """Reject a Qwen lookup that does not identify the model size."""
    with pytest.raises(ModelContractError):
        model_contract("qwen3", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", variant="other")


def test_loaded_state_is_checked_against_the_exact_inventory() -> None:
    """Accept a complete state and reject a missing tensor or wrong dtype."""
    state = {
        "layer.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "layer.bias": torch.ones((2,), dtype=torch.bfloat16),
    }
    inventory = TensorInventoryContract(
        tensor_count=2,
        parameter_count=6,
        dtype_counts=(("torch.bfloat16", 2),),
        inventory_sha256=_canonical_inventory(
            {
                "layer.weight": ((2, 2), "torch.bfloat16"),
                "layer.bias": ((2,), "torch.bfloat16"),
            }
        ),
    )

    validate_model_state(
        state,
        inventory,
        name="test",
        runtime_dtypes=(RuntimeDtypeRule("", ("torch.bfloat16",)),),
    )

    with pytest.raises(ModelContractError):
        validate_model_state(
            {"layer.weight": state["layer.weight"]},
            inventory,
            name="test",
        )
