# Domain 4: Attention Mechanism Adaptation

## Overview

The attention mechanism is the most complex adaptation domain. The original codebase uses a single `flash_attention()` function; the Ascend version introduces a multi-backend dispatch system, long-context attention with Ulysses + Ring, Attention Cache, and sub-head splitting.

## 4.1 Architecture Change

```
Original:
  WanSelfAttention.forward() → flash_attention(q, k, v, ...)

Ascend:
  WanSelfAttention.forward()
    → self.attention(q, k, v, ...)           # Sub-head dispatch
      → self._attention_op(q, k, v, ...)     # Multi-backend dispatch
        ├── RainFusion path (sparse attention)
        ├── ALGO=1: ascend_laser_attention
        ├── ALGO=3: npu_fused_infer_attention_score
        └── ALGO=0: fused_attn_score (default)
```

## 4.2 Multi-Backend Dispatch (`_attention_op`)

The `_attention_op` method in `WanSelfAttention` implements backend selection:

```python
ALGO = int(os.getenv('ALGO', 0))

def _attention_op(self, q, k, v, **kwargs):
    if torch.npu.is_available():
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        # Priority 1: RainFusion sparse attention
        if kwargs.get('rainfusion_config') and q.shape[1] == k.shape[1]:
            config = kwargs['rainfusion_config']
            if config['type'] == 'v1':
                return Rainfusion.rainfusion_fa(q, k, v, config, ...)
            else:
                return Rainfusion_blockwise.rainfusion_fa(q, k, v, config, ...)

        # Priority 2: ALGO-based dispatch (self-attention only)
        if ALGO == 1 and q.shape[1] == k.shape[1]:
            return attention_forward(q, k, v, op_type="ascend_laser_attention")

        if ALGO == 3 and q.shape[1] == k.shape[1]:
            if hasattr(self, 'fa_quant') and self.fa_quant:
                return self.fa_quant(q, k, v)  # Quantized attention
            return torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=self.num_heads, input_layout="BSND")

        # Default: ALGO == 0
        return attention_forward(q, k, v, op_type="fused_attn_score")
    else:
        return F.scaled_dot_product_attention(q, k, v)
```

## 4.3 Sub-Head Splitting

Controlled by `USE_SUB_HEAD` environment variable:

```python
class WanSelfAttention(nn.Module):
    def __init__(self, ...):
        self.use_sub_head = int(os.getenv('USE_SUB_HEAD', 0))
        if self.use_sub_head != 0:
            assert num_heads % self.use_sub_head == 0

    def attention(self, q, k, v, **kwargs):
        if self.use_sub_head:
            output = []
            for i in range(num_heads // self.use_sub_head):
                output.append(self._attention_op(
                    q[:, :, i*sub:i*sub+sub],
                    k[:, :, i*sub:i*sub+sub],
                    v[:, :, i*sub:i*sub+sub], **kwargs))
            return torch.cat(output, dim=2)
        return self._attention_op(q, k, v, **kwargs)
```

**Use case:** When total head count is too large for a single NPU attention kernel, split into groups.

## 4.4 xFuserLongContextAttention (`attn_layer.py`)

This is the core distributed attention implementation for sequence parallel mode, inheriting from `yunchang.LongContextAttention`.

### Key Features

1. **Ulysses + Ring hybrid**: Combines head-dimension parallelism (Ulysses) with KV sequence parallelism (Ring)
2. **FA-AllToAll overlap**: Overlaps communication with computation for higher throughput
3. **Multi-backend FA**: Uses the same ALGO-based dispatch as local attention

### Standard Forward Path (no overlap)

```
Input: q, k, v  [B, local_seq_len, num_heads, head_dim]
  ↓
1. all_to_all_4D(scatter_idx=2, gather_idx=1)  # heads→sequence
  ↓  q, k, v now: [B, full_seq_len, local_heads, head_dim]
  ↓
2. Ring all_gather for K, V (if ring_world_size > 1)
  ↓  k, v now: [B, ring_full_seq_len, local_heads, head_dim]
  ↓
3. FA computation (ALGO-dispatched)
  ↓
4. all_to_all_4D(scatter_idx=1, gather_idx=2)  # sequence→heads
  ↓
Output: [B, local_seq_len, num_heads, head_dim]
```

### Overlap Forward Path

When `OVERLAP=1` environment variable is set:

```
Stream 1 (main):     [AllToAll chunk0] → [FA chunk0] → wait(stream2) → [FA chunk1] → ...
Stream 2 (auxiliary): .................. → [AllToAll chunk1] → [AllToAll chunk2] → ...
```

This pipelines the AllToAll communication of the next chunk with the FA computation of the current chunk.

## 4.5 Attention Cache via MindIE

`mindiesd.CacheAgent` enables temporal step-skipping in the denoising loop:

```python
from mindiesd import CacheConfig, CacheAgent

# Configuration
config = CacheConfig(
    method="attention_cache",
    ratio=1.2,          # Cache ratio
    interval=4,         # Skip interval
    start_step=12,      # Start caching from step 12
    end_step=37,        # Stop caching at step 37
)
cache = CacheAgent(config)

# Inject into model blocks
for block in transformer.blocks:
    block.cache = cache
    block.args = args

# In WanAttentionBlock.forward():
# Instead of directly calling self.self_attn(...), use:
y = self.cache.apply(self.self_attn, norm_out, seq_lens, grid_sizes, freqs, self.args, ...)
```

**How it works:** The CacheAgent determines whether to compute attention or reuse cached results based on the timestep index. During "cached" steps, it interpolates from previous results instead of running full attention computation.

## 4.6 Forward Signature Changes

All attention-related forward methods gain additional parameters:

```python
# WanSelfAttention.forward()
# Original: (self, x, seq_lens, grid_sizes, freqs)
# Ascend:   (self, x, seq_lens, grid_sizes, freqs, args=None, rainfusion_config=None, t_idx=None, b_idx=None)

# WanAttentionBlock.forward()
# Original: (self, x, e, seq_lens, grid_sizes, freqs, context, context_lens)
# Ascend:   (self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, rainfusion_config, t_idx, b_idx)

# WanModel.forward()
# Original: (self, x, t, context, seq_len, y=None)
# Ascend:   (self, x, t, context, seq_len, y=None, t_idx=None)
```

New parameters:
- `args`: General arguments bag (for cache configuration)
- `rainfusion_config`: Sparse attention configuration dict
- `t_idx`: Timestep index (for cache and sparse attention scheduling)
- `b_idx`: Block index (for per-block sparse attention strategy)

## 4.7 Block Iteration Change

```python
# Original
for block in self.blocks:
    x = block(x, **kwargs)

# Ascend
for b_idx, block in enumerate(self.blocks):
    x = block(x, b_idx=b_idx, **kwargs)
```

The block index is passed for per-block sparse attention scheduling.

## Pitfalls

1. **Cross-attention always uses default FA**: ALGO=1/3 optimizations and RainFusion only apply to self-attention where `q.shape[1] == k.shape[1]`.
2. **Attention Cache requires block-level injection**: Every `WanAttentionBlock` must have `cache` and `args` attributes set.
3. **xFuserLongContextAttention padding**: When `seq_lens < padded_seq_len`, padding tokens are handled separately to avoid corrupting attention results.
