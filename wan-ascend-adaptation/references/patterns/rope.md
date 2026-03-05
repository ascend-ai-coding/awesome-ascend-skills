# Rotary Position Embedding Pattern

Replace manual RoPE with MindIE optimized implementation.

## Manual Implementation (Original)
```python
def apply_rotary_pos_emb(x, freqs):
    cos = freqs[0]
    sin = freqs[1]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rotated * sin
```

## MindIE RoPE Implementation

### Basic Usage
```python
import mindiesd

@torch.amp.autocast('npu', enabled=False)  # Disable autocast for precision
def apply_rotary_pos_emb(q, k, freqs, grid_size=None):
    cos, sin = freqs
    q_embed = mindiesd.rotary_position_embedding(q, cos, sin, grid_size)
    k_embed = mindiesd.rotary_position_embedding(k, cos, sin, grid_size)
    return q_embed, k_embed
```

### With Grid Size
```python
def apply_rope_with_grid(q, k, freqs, seq_len, head_dim):
    # Calculate grid size for spatial attention
    grid_size = int(seq_len ** 0.5)
    
    with torch.amp.autocast('npu', enabled=False):
        q = q.float()  # Ensure FP32
        k = k.float()
        
        q_embed = mindiesd.rotary_position_embedding(
            q, freqs[0], freqs[1], grid_size
        )
        k_embed = mindiesd.rotary_position_embedding(
            k, freqs[0], freqs[1], grid_size
        )
    
    return q_embed.to(q.dtype), k_embed.to(k.dtype)
```

## File Modifications

### wan/modules/model.py
```python
# Add imports
import mindiesd
import torch.amp

# Replace manual RoPE function
# def apply_rotary_pos_emb(x, freqs):  # REMOVE

# Use MindIE RoPE
@torch.amp.autocast('npu', enabled=False)
def apply_rotary_pos_emb(q, k, freqs, grid_size=None):
    cos, sin = freqs
    q_embed = mindiesd.rotary_position_embedding(q, cos, sin, grid_size)
    k_embed = mindiesd.rotary_position_embedding(k, cos, sin, grid_size)
    return q_embed, k_embed
```

## Precision Management

### Disable AutoCast
```python
# MindIE RoPE requires FP32 for numerical stability
with torch.amp.autocast('npu', enabled=False):
    output = mindiesd.rotary_position_embedding(
        x.float(),
        cos.float(),
        sin.float(),
        grid_size
    )
```

### Manual Precision Control
```python
def apply_rope_with_precision(x, freqs, grid_size, dtype=torch.float32):
    cos, sin = freqs
    
    # Cast to target precision
    x_fp32 = x.float()
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()
    
    # Apply RoPE
    output = mindiesd.rotary_position_embedding(
        x_fp32, cos_fp32, sin_fp32, grid_size
    )
    
    # Return to original dtype
    return output.to(x.dtype)
```

## Performance Optimization

### Precompute Frequency Bases
```python
def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    # Compute frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis

# Precompute once and reuse
freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
```

### Memory-Efficient RoPE
```python
class MemoryEfficientRoPE:
    def __init__(self, head_dim, max_seq_len):
        self.freqs = self._precompute_freqs(head_dim, max_seq_len)
    
    def apply(self, x, seq_len):
        # Slice only needed frequencies
        freqs = self.freqs[:seq_len]
        
        with torch.amp.autocast('npu', enabled=False):
            output = mindiesd.rotary_position_embedding(
                x.float(), freqs.real, freqs.imag, None
            )
        
        return output.to(x.dtype)
```

## Advanced Configurations

### Spatial Attention RoPE
```python
def apply_3d_rope(q, k, freqs, t, h, w):
    # 3D RoPE for video (temporal + spatial)
    grid_size = (h, w)
    
    with torch.amp.autocast('npu', enabled=False):
        # Apply temporal RoPE
        q_t = mindiesd.rotary_position_embedding(q, freqs[0], freqs[1], None)
        k_t = mindiesd.rotary_position_embedding(k, freqs[0], freqs[1], None)
        
        # Apply spatial RoPE
        q_s = mindiesd.rotary_position_embedding(q_t, freqs[2], freqs[3], grid_size)
        k_s = mindiesd.rotary_position_embedding(k_t, freqs[2], freqs[3], grid_size)
    
    return q_s.to(q.dtype), k_s.to(k.dtype)
```

## Environment Variables

```bash
# RoPE precision
export ROPE_PRECISION=float32     # Force FP32 for RoPE

# Memory optimization
export ROPE_USE_INPLACE=0        # In-place computation (experimental)
```

## Troubleshooting

### Issue: Incorrect attention due to position encoding
**Cause**: Grid size not set correctly for spatial attention
**Solution**:
```python
# Calculate grid size from sequence length
grid_size = int(seq_len ** 0.5)
output = mindiesd.rotary_position_embedding(x, cos, sin, grid_size)
```

### Issue: Precision loss in attention
**Cause**: AutoCast enabled for RoPE
**Solution**:
```python
@torch.amp.autocast('npu', enabled=False)
def apply_rope(x, freqs):
    return mindiesd.rotary_position_embedding(x, freqs[0], freqs[1], None)
```

### Issue: Out of memory with long sequences
**Cause**: Storing full frequency tensors
**Solution**:
- Use precomputed frequencies with slicing
- Enable gradient checkpointing
- Reduce max sequence length

## Configuration Templates

### Training Configuration
```yaml
rope:
  type: "mindie_rope"
  precision: "float32"
  autocast: false
  grid_size: "auto"  # Auto-calculate from seq_len
  precompute: true

environment:
  ROPE_PRECISION: "float32"
```

## References

- [MindIE RoPE API](https://github.com/Ascend/mindie)
- [Rotary Position Embedding Paper](https://arxiv.org/abs/2104.09864)
- [RoPE Implementation Guide](https://www.hiascend.com/document)
