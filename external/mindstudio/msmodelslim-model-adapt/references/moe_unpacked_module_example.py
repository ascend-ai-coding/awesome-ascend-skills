#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
MoE unpacked module example.

Purpose:
- Provide a pure-nn.Linear MoE implementation for models whose original experts
  are fused/packed in 3D tensors.
- Keep routing logic and forward behavior consistent with the source model.

Note:
- This is an adaptation example, not a drop-in module for every model.
- You must align tensor layouts and routing semantics with the original modeling file.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeExpertMLP(nn.Module):
    """Single expert with pure Linear layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, act_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TopKRouter(nn.Module):
    """Generic top-k router example."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        topk_prob, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True)
        return logits, topk_prob.to(logits.dtype), topk_idx


class SparseMoeBlockWithLinearExperts(nn.Module):
    """
    MoE block where each expert is explicit Linear layers.
    This is the target structure after unpack.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            [MoeExpertMLP(hidden_size, intermediate_size, act_fn) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        _, routing_weights, selected_experts = self.router(flat)

        out = torch.zeros_like(flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        active_experts = torch.nonzero(expert_mask.sum(dim=(-1, -2)) > 0).flatten()

        for expert_idx in active_experts.tolist():
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            expert_out = self.experts[expert_idx](flat[token_idx])
            expert_out = expert_out * routing_weights[token_idx, top_k_pos, None]
            out.index_add_(0, token_idx, expert_out.to(out.dtype))

        return out.reshape(batch_size, seq_len, hidden_size)

