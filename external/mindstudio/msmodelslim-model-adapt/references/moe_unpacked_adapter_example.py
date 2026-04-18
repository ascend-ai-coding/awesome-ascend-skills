#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Adapter-side unpack example for MoE fused weights.

Goal:
- Detect whether experts are packed 3D tensors.
- Unpack packed weights into per-expert Linear parameter keys.
- Replace original MoE block with unpacked Linear-expert module.
"""

from typing import Dict

import torch

from .moe_unpacked_module_example import SparseMoeBlockWithLinearExperts


def is_packed_moe_tensor(key: str, tensor: torch.Tensor) -> bool:
    """Heuristic: packed experts are usually 3D for gate_up/down projections."""
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.dim() != 3:
        return False
    return key.endswith("experts.gate_up_proj") or key.endswith("experts.down_proj")


def unpack_packed_moe_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Unpack common packed MoE keys into Linear-expert keys.

    Expected packed patterns:
    - *.experts.gate_up_proj: [num_experts, 2*intermediate, hidden]
    - *.experts.down_proj:    [num_experts, hidden, intermediate]
    """
    unpacked = dict(state_dict)

    for key, tensor in state_dict.items():
        if not is_packed_moe_tensor(key, tensor):
            continue

        if key.endswith("experts.gate_up_proj"):
            prefix = key.replace("experts.gate_up_proj", "experts.")
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                gate_w, up_w = tensor[i].chunk(2, dim=0)
                unpacked[f"{prefix}{i}.gate_proj.weight"] = gate_w
                unpacked[f"{prefix}{i}.up_proj.weight"] = up_w

        elif key.endswith("experts.down_proj"):
            prefix = key.replace("experts.down_proj", "experts.")
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                unpacked[f"{prefix}{i}.down_proj.weight"] = tensor[i]

    return unpacked


def replace_moe_module_if_needed(layer, cfg, act_fn):
    """
    Replace fused MoE module with unpacked Linear-expert module.
    Call this during layer construction/loading in model_adapter.py.
    """
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
        return layer

    # Example: original layer.mlp uses packed expert layout.
    unpacked_moe = SparseMoeBlockWithLinearExperts(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        act_fn=act_fn,
    )
    layer.mlp = unpacked_moe
    return layer


def load_layer_with_unpack_example(layer, layer_state_dict: Dict[str, torch.Tensor], strict: bool = False):
    """
    Example load sequence inside adapter:
    1) unpack packed MoE tensors
    2) load state dict into replaced module structure
    """
    unpacked_state_dict = unpack_packed_moe_weights(layer_state_dict)
    missing, unexpected = layer.load_state_dict(unpacked_state_dict, strict=strict)
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "loaded_keys": len(unpacked_state_dict),
    }

