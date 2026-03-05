# Rainfusion Sparse Attention Pattern

Reduce attention computation and memory usage through structured sparsity for long sequences.

## Overview

Rainfusion implements two sparse attention strategies:

- **v1 (Window-based)**: Limits attention to local windows with configurable bandwidth
- **v2 (Blockwise)**: Uses blockwise masking with avgpool-based importance scoring

Use Rainfusion when sequences exceed 16K tokens or memory constraints limit dense attention.

## Two Versions Comparison

| Feature | v1 Window-Based | v2 Blockwise |
|---------|----------------|--------------|
| Sparsity strategy | Fixed window bandwidth | Dynamic blockwise masking |
| Mask generation | Pre-computed window masks | Avgpool-based score thresholding |
| Layout transformation | Local vs global attention selection | Tensor rearrangement for block alignment |
| Best for | Consistent local patterns | Variable importance regions |
| Memory reduction | 30-50% | 50-70% |
| Implementation | `rainfusion.py` | `rainfusion_blockwise.py` |

### v1: Window-Based Attention

```python
# Bandwidth calculation from sparsity
# bandwidth = 1 - sqrt(sparsity)
# sparsity=0.5 → bandwidth=0.29 (29% of tokens attend to each side)
# sparsity=0.9 → bandwidth=0.05 (5% of tokens attend to each side)
```

### v2: Blockwise Attention

```python
# Blockwise mask based on avgpool scores
# pool_size=128 blocks tokens into importance groups
# keep_len = ceil(seq_len * (1 - sparsity))
# Top-k scoring blocks selected, rest masked
```

## Code Examples

### Before: Dense Attention

```python
import torch
import torch.nn as nn

class DenseAttention(nn.Module):
    def forward(self, q, k, v, attention_mask=None):
        # Full O(n²) attention computation
        batch, seq_len, num_heads, head_dim = q.shape
        
        # Compute attention scores: [batch, heads, seq, seq]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim ** -0.5)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        
        return output
```

### After: Rainfusion v1 (Window-Based)

```python
from wan.utils.rainfusion import Rainfusion
import torch
import torch.nn as nn

class SparseAttentionV1(nn.Module):
    def __init__(self, grid_size, sparsity=0.5, skip_timesteps=20):
        super().__init__()
        # grid_size: [T, H, W] - temporal and spatial dimensions
        self.rainfusion = Rainfusion(
            grid_size=grid_size,      # e.g., [16, 30, 52]
            sparsity=sparsity,        # 0.0 to 1.0
            skip_timesteps=skip_timesteps  # Start sparse attn after N steps
        )
        self.grid_size = grid_size
        
    def forward(self, q, k, v, text_len=0, t_idx=0):
        # Generate attention mask for current sparsity
        atten_mask = Rainfusion.get_atten_mask(
            self.grid_size,
            self.rainfusion.sparsity
        )
        
        # Rainfusion handles local vs global attention selection
        output = self.rainfusion(
            query=q,           # [batch, seq, heads, dim]
            key=k,
            value=v,
            atten_mask_all=atten_mask,
            text_len=text_len,
            t_idx=t_idx
        )
        return output
```

### After: Rainfusion v2 (Blockwise)

```python
from wan.utils.rainfusion_blockwise import Rainfusion_blockwise
import torch.nn as nn

class SparseAttentionV2(nn.Module):
    def __init__(self, grid_size, sparsity=0.9, pool_size=128):
        super().__init__()
        self.rainfusion = Rainfusion_blockwise(
            grid_size=grid_size,
            sparsity=sparsity,        # Higher values = more sparse
            pool_size=pool_size,      # Block size for avgpool
            skip_timesteps=0,
            txt_len=512,              # Text token length
            txt_first=False
        )
        self.base_blockmask = None
        
    def forward(self, q, k, v, t_idx=0, batch_idx=0):
        t_b_idx = [t_idx, batch_idx]
        
        output, self.base_blockmask = self.rainfusion(
            q=q,                      # [batch, seq, heads, dim]
            k=k,
            v=v,
            t_b_idx=t_b_idx,
            base_blockmask=self.base_blockmask
        )
        return output
```

### Integration with Wan Model

```python
# wan/modules/attn_layer.py modification
from wan.utils.rainfusion_blockwise import Rainfusion_blockwise

class WanAttention(nn.Module):
    def __init__(self, dim, num_heads, use_sparse=False, grid_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        if use_sparse and grid_size is not None:
            self.rainfusion = Rainfusion_blockwise(
                grid_size=grid_size,
                sparsity=float(os.environ.get("RAINFUSION_SPARSITY", "0.9")),
                pool_size=int(os.environ.get("RAINFUSION_POOL_SIZE", "128")),
                skip_timesteps=int(os.environ.get("RAINFUSION_SKIP_STEPS", "0")),
            )
            self.use_sparse = True
        else:
            self.use_sparse = False
            
    def forward(self, x, context=None, t_idx=None):
        q = self.to_q(x)
        k = self.to_k(context if context is not None else x)
        v = self.to_v(context if context is not None else x)
        
        if self.use_sparse and t_idx is not None:
            out = self.rainfusion(q, k, v, [t_idx, 0], None)[0]
        else:
            # Fallback to dense attention
            from mindiesd.layers.flash_attn.attention_forward import attention_forward
            out = attention_forward(q, k, v, opt_mode="manual", 
                                   op_type="ascend_laser_attention", layout="BNSD")
        
        return self.to_out(out)
```

## Key Parameters

### v1 Window-Based Parameters

```python
Rainfusion(
    grid_size=[16, 30, 52],    # [T, H, W] - temporal, height, width tokens
    sparsity=0.5,              # 0.0-1.0, higher = more sparse
    skip_timesteps=20          # Dense attn for first N steps
)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `grid_size` | list[int] | - | [T, H, W] dimensions after patch embedding |
| `sparsity` | float | 0.0-1.0 | Attention sparsity ratio |
| `skip_timesteps` | int | >=0 | Dense attention steps before sparse mode |

### v2 Blockwise Parameters

```python
Rainfusion_blockwise(
    grid_size=[16, 30, 52],
    pool_size=128,             # Avgpool block size
    sparsity=0.9,              # Higher = more aggressive sparsity
    skip_timesteps=0,
    txt_len=512,               # Text context length
    txt_first=False            # Text token ordering
)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `pool_size` | int | >=64 | Block size for importance pooling |
| `sparsity` | float | 0.0-1.0 | Block selection sparsity ratio |
| `txt_len` | int | >=0 | Length of text conditioning tokens |
| `txt_first` | bool | - | Whether text tokens precede image tokens |

### Bandwidth Calculation (v1)

```python
# bandwidth determines window size for local attention
bandwidth = 1 - math.sqrt(sparsity)

# Examples:
# sparsity=0.0  -> bandwidth=1.0 (dense)
# sparsity=0.25 -> bandwidth=0.5 (50% window)
# sparsity=0.5  -> bandwidth=0.29 (29% window)
# sparsity=0.9  -> bandwidth=0.05 (5% window)
```

## Performance Benefits

### Memory Reduction

| Sequence Length | Dense (GB) | v1 Sparse (GB) | v2 Sparse (GB) |
|-----------------|------------|----------------|----------------|
| 8K | 2.0 | 1.4 (30%) | 1.2 (40%) |
| 16K | 8.0 | 4.8 (40%) | 3.2 (60%) |
| 32K | 32.0 | 16.0 (50%) | 9.6 (70%) |
| 64K | 128.0 | 51.2 (60%) | 25.6 (80%) |

### Computation Speedup

```python
# Relative speedup vs dense attention (RTX A100 equivalent)
# Higher sparsity = fewer FLOPs

sparsity_speedup = {
    0.0: 1.0,    # Dense baseline
    0.5: 1.4,    # 40% speedup
    0.7: 1.8,    # 80% speedup
    0.9: 2.5,    # 150% speedup
}
```

### Recall Ratio Selection (v1)

```python
# Rainfusion v1 dynamically selects local vs global attention
def select_attention_mode(ratio_local, ratio_global):
    """
    ratio_local: Attention mass captured by local window
    ratio_global: Attention mass captured by global stride
    
    Returns: True for local attention, False for global
    """
    WIN_RATIO = 0.95  # Threshold for local attention
    return ratio_local > WIN_RATIO or ratio_local > ratio_global

# If local window captures >95% attention mass, use local
# Otherwise use global attention with stride
```

## When to Use

### Use Rainfusion When

| Scenario | Recommendation |
|----------|----------------|
| Sequence length > 16K | v2 blockwise with sparsity=0.7-0.9 |
| OOM errors with dense attention | Start with sparsity=0.5, increase as needed |
| Video generation (high T) | v1 window-based for temporal consistency |
| Image generation (high H×W) | v2 blockwise for spatial sparsity |
| Memory-constrained inference | v2 with sparsity=0.9, pool_size=256 |

### Sparsity Selection Guide

```python
def select_sparsity(seq_len, available_memory_gb):
    """Recommend sparsity level based on constraints"""
    
    # Memory-based selection
    required_dense_gb = (seq_len ** 2) * 4 / (1024 ** 3)  # FP32 attention matrix
    
    if required_dense_gb > available_memory_gb * 0.8:
        # Need aggressive sparsity
        return 0.9
    elif required_dense_gb > available_memory_gb * 0.5:
        # Moderate sparsity
        return 0.7
    elif seq_len > 32768:
        # Long sequence, use sparsity for speed
        return 0.5
    else:
        # Can use dense
        return 0.0
```

### Version Selection

```python
def select_rainfusion_version(task_type):
    """Select v1 or v2 based on task characteristics"""
    
    if task_type in ["video_generation", "temporal_consistency"]:
        # v1 better for temporal patterns
        return "v1"
    elif task_type in ["high_resolution_image", "variable_sparsity"]:
        # v2 better for spatial importance
        return "v2"
    elif task_type == "memory_critical":
        # v2 higher memory savings
        return "v2"
    else:
        # Default to v2
        return "v2"
```

## Configuration

### Environment Variables

```bash
# Enable Rainfusion sparse attention
export USE_RAINFUSION=1

# v1 Window-based configuration
export RAINFUSION_SPARSITY=0.5
export RAINFUSION_SKIP_TIMESTEPS=20

# v2 Blockwise configuration
export RAINFUSION_SPARSITY=0.9
export RAINFUSION_POOL_SIZE=128
export RAINFUSION_SKIP_STEPS=0

# Backend selection
export RAINFUSION_VERSION=v2  # v1 or v2
```

### Runtime Configuration

```python
import os

def configure_rainfusion(grid_size):
    """Configure Rainfusion from environment variables"""
    
    if os.environ.get("USE_RAINFUSION", "0") != "1":
        return None
    
    version = os.environ.get("RAINFUSION_VERSION", "v2")
    sparsity = float(os.environ.get("RAINFUSION_SPARSITY", "0.9"))
    skip_steps = int(os.environ.get("RAINFUSION_SKIP_STEPS", "0"))
    
    if version == "v1":
        from wan.utils.rainfusion import Rainfusion
        return Rainfusion(
            grid_size=grid_size,
            sparsity=sparsity,
            skip_timesteps=skip_steps
        )
    else:
        from wan.utils.rainfusion_blockwise import Rainfusion_blockwise
        pool_size = int(os.environ.get("RAINFUSION_POOL_SIZE", "128"))
        return Rainfusion_blockwise(
            grid_size=grid_size,
            sparsity=sparsity,
            pool_size=pool_size,
            skip_timesteps=skip_steps,
            txt_len=512,
            txt_first=False
        )
```

### Model Integration Pattern

```python
# Enable Rainfusion in Wan model initialization
class WanModel:
    def __init__(self, config):
        self.grid_size = [
            config.num_frames,
            config.height // config.patch_size,
            config.width // config.patch_size
        ]
        
        # Conditionally enable Rainfusion
        if config.use_sparse_attention:
            self.rainfusion = configure_rainfusion(self.grid_size)
        
        for block in self.transformer_blocks:
            block.attn.use_sparse = config.use_sparse_attention
            if config.use_sparse_attention:
                block.attn.rainfusion = self.rainfusion
```

## Troubleshooting

### Issue: Quality degradation with high sparsity

**Cause**: Too much information masked

**Solutions**:
1. Reduce sparsity (0.9 -> 0.7)
2. Increase skip_timesteps for warm-up
3. Use v1 for temporal consistency
4. Add protect_len for important tokens (v2)

### Issue: Blockwise mask generation overhead (v2)

**Cause**: Avgpool and topk on every forward pass

**Solutions**:
1. Cache base_blockmask across timesteps
2. Increase pool_size to reduce block count
3. Use v1 for consistent patterns

### Issue: OOM during attention

**Cause**: Even sparse attention exceeds memory

**Solutions**:
1. Increase sparsity further (0.7 -> 0.9)
2. Reduce pool_size in v2
3. Enable gradient checkpointing
4. Use sequence parallelism

### Issue: Incorrect attention with text conditioning

**Cause**: Text token ordering mismatch

**Solutions**:
1. Set correct txt_len parameter
2. Adjust txt_first based on token ordering
3. Verify grid_size calculation includes text tokens

## Examples

### Basic Usage

```python
import torch
from wan.utils.rainfusion_blockwise import Rainfusion_blockwise

# Setup
batch, seq, heads, dim = 1, 16384, 16, 128
grid_size = [16, 32, 32]  # T=16, H=32, W=32

q = torch.randn(batch, seq, heads, dim).npu()
k = torch.randn(batch, seq, heads, dim).npu()
v = torch.randn(batch, seq, heads, dim).npu()

# Create Rainfusion
rf = Rainfusion_blockwise(
    grid_size=grid_size,
    sparsity=0.9,
    pool_size=128,
    txt_len=512
)

# Forward pass
output, _ = rf(q, k, v, [0, 0], None)
```

### Progressive Sparsity

```python
# Start dense, gradually increase sparsity
class ProgressiveSparseAttention:
    def __init__(self, grid_size):
        self.rainfusion = Rainfusion_blockwise(
            grid_size=grid_size,
            sparsity=0.0,  # Start dense
            skip_timesteps=0
        )
        self.current_sparsity = 0.0
        
    def step(self, t_idx, total_steps):
        # Linear increase from 0.0 to 0.9
        target_sparsity = min(0.9, t_idx / (total_steps * 0.5))
        
        if abs(target_sparsity - self.current_sparsity) > 0.1:
            self.rainfusion.sparsity = target_sparsity
            self.current_sparsity = target_sparsity
```

## References

- Implementation: `wan/utils/rainfusion.py`, `wan/utils/rainfusion_blockwise.py`
- MindIE SD: `mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2`
- ATB Operations: `torch_atb.RazorFusionAttentionParam`
