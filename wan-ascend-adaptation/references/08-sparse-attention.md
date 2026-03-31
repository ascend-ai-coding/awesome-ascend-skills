# Domain 8: Sparse Attention — RainFusion

## Overview

RainFusion is an Ascend-specific sparse attention optimization that exploits the spatiotemporal locality of video tokens to reduce attention computation. Two versions exist: v1 (window-based adaptive) and v2 (blockwise Top-K).

## 8.1 Common Configuration

```python
rainfusion_config = {
    "type": "v1",          # or "v2" for blockwise
    "sparsity": 0.64,      # Fraction of attention to skip (0-1)
    "skip_timesteps": 15,  # Use full attention for first N steps
    "atten_mask_all": None, # Computed lazily on first forward
    "grid_size": None,      # Computed lazily based on input shape
}
```

**Sparsity-speed tradeoff:**
- `sparsity=0`: Full attention (no speedup)
- `sparsity=0.64`: Good balance of quality and speed (recommended)
- `sparsity=0.9`: Maximum speedup, some quality degradation

**Skip timesteps:** The first N denoising steps use full attention (critical for layout/structure), then switch to sparse attention for refinement steps.

## 8.2 RainFusion v1 — Window-Based Adaptive

### Algorithm

```
For each attention block, for each head:
1. Compute bandwidth = 1 - sqrt(sparsity)
2. Reorder tokens: move first frame to end (global reference)
3. Try two sparse strategies:
   a. Local: Window attention in spatial dims (t,h,w) → rearranged to (h,w,t)
   b. Global: Window attention in temporal dim with sparse stride
4. For each strategy, compute mask_recall vs. full attention weights
5. Choose strategy with higher recall (or Local if recall > 0.95)
6. Execute sparse FA with chosen mask
```

### Underlying Operator

```python
import torch_atb

param = torch_atb.RazorFusionAttentionParam(
    razor_len=razor_len,        # Sparsified sequence length
    pre_tokens=pre_tokens,      # Window: tokens before current
    next_tokens=next_tokens,    # Window: tokens after current
    text_q_len=text_q_len,      # Protected text tokens (Q side)
    text_kv_len=text_kv_len,    # Protected text tokens (KV side)
)
op = torch_atb.RazorFusionAttentionOp(param)
output = op.forward(q, k, v, attention_mask)
```

### Mask Generation

```python
@staticmethod
def get_atten_mask(grid_size, sparsity):
    """Pre-compute attention masks for all heads."""
    f, h, w = grid_size
    bandwidth = 1 - math.sqrt(sparsity)

    masks = {}
    for head_idx in range(num_heads):
        # Compute local mask (spatial window)
        local_mask = compute_window_mask(f, h, w, bandwidth, mode='local')
        # Compute global mask (temporal stride)
        global_mask = compute_window_mask(f, h, w, bandwidth, mode='global')
        # Choose based on recall
        masks[head_idx] = choose_better_mask(local_mask, global_mask, ...)

    return masks
```

### Grid Size Calculation

```python
@staticmethod
def get_grid_size(input_shape, patch_size):
    """Convert latent shape to video grid dimensions."""
    # input_shape: [seq_len, channels]
    # Returns: (frames, height, width) after patch embedding
```

## 8.3 RainFusion v2 — Blockwise Top-K

### Algorithm

```
1. Rearrange tokens into blocks of 128 tokens each
   - Group by spatial blocks: (fn×hn×wn) super-blocks, each (fb×hb×wb) tokens
2. Average-pool Q and K to block-level representations
3. Compute block-level attention scores: pooled_Q × pooled_K^T
4. Top-K selection: keep ceil(num_blocks × (1-sparsity)) blocks per row
5. Always protect: first-frame blocks + text token blocks
6. Execute sparse FA using mindiesd.sparse_flash_attn_rf_v2
```

### Underlying Operator

```python
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention

output = rain_fusion_attention(
    q, k, v,
    block_mask=block_mask,  # Boolean mask: which blocks to attend
    block_size=128,
)
```

### Key Differences from v1

| Aspect | v1 | v2 (Blockwise) |
|--------|-----|---------|
| Granularity | Per-token window | Per-block (128 tokens) |
| Strategy selection | Adaptive (Local vs Global per head) | Top-K scoring |
| Extra compute | Mask recall calculation | Average pooling + Top-K |
| Underlying operator | torch_atb.RazorFusionAttention | mindiesd.sparse_flash_attn_rf_v2 |
| Memory overhead | Pre-computed masks for all heads | Dynamic block masks per forward |

## 8.4 Integration Points

### In WanModel.forward()

```python
# Lazy initialization of RainFusion config
if self.rainfusion_config and self.rainfusion_config["atten_mask_all"] is None:
    if self.rainfusion_config["type"] == "v1":
        self.rainfusion_config["grid_size"] = Rainfusion.get_grid_size(
            x[0].shape, self.patch_size)
        self.rainfusion_config["atten_mask_all"] = Rainfusion.get_atten_mask(
            grid_size=self.rainfusion_config["grid_size"],
            sparsity=self.rainfusion_config["sparsity"])
    else:
        self.rainfusion_config["grid_size"] = Rainfusion_blockwise.get_grid_size(
            x[0].shape, self.patch_size)
```

### In generate.py

```python
if args.use_rainfusion:
    rainfusion_config = {
        "type": args.rainfusion_type,
        "sparsity": args.sparsity,
        "skip_timesteps": args.sparse_start_step,
        "atten_mask_all": None,
        "grid_size": None,
    }
    transformer.rainfusion_config = rainfusion_config
```

### In _attention_op()

```python
if rainfusion_config is not None and q.shape[1] == k.shape[1]:
    # Only for self-attention
    if rainfusion_config["type"] == "v1":
        return Rainfusion.rainfusion_fa(q, k, v, rainfusion_config,
            t_idx=t_idx, b_idx=b_idx)
    else:
        return Rainfusion_blockwise.rainfusion_fa(q, k, v, rainfusion_config,
            t_idx=t_idx, b_idx=b_idx)
```

## 8.5 Timestep Scheduling

The `t_idx` parameter controls when sparse attention activates:

```python
# Inside Rainfusion.rainfusion_fa():
if t_idx < config["skip_timesteps"]:
    return full_attention(q, k, v)  # Full attention for early steps
else:
    return sparse_attention(q, k, v, mask)  # Sparse for later steps
```

**Rationale:**
- Early denoising steps (high noise): Need full attention to establish global layout
- Later steps (low noise): Local refinement only, sparse attention sufficient

## Pitfalls

1. **Self-attention only**: RainFusion is disabled for cross-attention (text conditioning).
2. **Grid size dependency**: Masks are resolution-dependent. Different video sizes require re-computation of `atten_mask_all`.
3. **First frame protection**: The first video frame always receives full attention as a global reference anchor.
4. **sparsity=0 is valid**: Setting sparsity to 0 effectively disables sparse attention without code changes.
