#!/usr/bin/env python3
"""
Attention mechanism comparison: CUDA FlashAttention vs NPU MindIE
Demonstrates 3 attention algorithms available on Ascend
"""

import torch
import torch_npu
import os


# Original CUDA FlashAttention
def flash_attention_cuda(q, k, v):
    """Original FlashAttention implementation"""
    from flash_attn import flash_attn_func

    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    return output


# NPU MindIE Attention (3 algorithms)
def mindie_attention_npu(q, k, v, algo=None):
    """
    NPU-optimized attention with 3 algorithm options

    Args:
        q, k, v: Query, Key, Value tensors [batch, seq_len, heads, head_dim]
        algo: Algorithm selection
            0: fused_attn_score (default FA)
            1: ascend_laser_attention (high-performance, recommended)
            3: npu_fused_infer_attention_score (inference optimized)
    """
    from mindiesd import attention_forward

    if algo is None:
        algo = int(os.environ.get("ALGO", "1"))  # Default: laser_attention

    # Transpose for BNSD layout
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if algo == 0:
        # Default FA operator
        output = attention_forward(
            q_t, k_t, v_t, opt_mode="manual", op_type="fused_attn_score", layout="BNSD"
        )
    elif algo == 1:
        # High-performance FA (Ascend Laser Attention)
        output = attention_forward(
            q_t,
            k_t,
            v_t,
            opt_mode="manual",
            op_type="ascend_laser_attention",
            layout="BNSD",
        )
    elif algo == 3:
        # NPU fused inference attention
        scale = q.shape[-1] ** -0.5
        output = torch_npu.npu_fused_infer_attention_score(
            q_t,
            k_t,
            v_t,
            num_heads=q.shape[2],
            input_layout="BNSD",
            scale=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )[0]
    else:
        raise ValueError(f"Invalid ALGO={algo}. Must be 0, 1, or 3.")

    # Transpose back to BHSD layout
    return output.transpose(1, 2)


# Example usage
if __name__ == "__main__":
    # Initialize NPU
    torch.npu.set_device(0)

    # Create sample tensors
    batch, seq_len, heads, head_dim = 2, 1024, 32, 128
    q = torch.randn(
        batch, seq_len, heads, head_dim, device="npu:0", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, seq_len, heads, head_dim, device="npu:0", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch, seq_len, heads, head_dim, device="npu:0", dtype=torch.bfloat16
    )

    # Test 3 algorithms
    for algo in [0, 1, 3]:
        os.environ["ALGO"] = str(algo)
        output = mindie_attention_npu(q, k, v, algo=algo)
        print(f"✓ ALGO={algo} successful! Output shape: {output.shape}")
