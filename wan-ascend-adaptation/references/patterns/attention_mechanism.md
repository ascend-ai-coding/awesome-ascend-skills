# Attention Mechanism Pattern

Replace FlashAttention 2/3 with MindIE NPU attention algorithms.

## MindIE Attention Algorithms

MindIE provides 3 attention algorithms for different use cases:

### Algorithm Selection
```bash
export ALGO=0  # fused_attn_score (default, stable)
export ALGO=1  # ascend_laser_attention (high performance)  
export ALGO=3  # npu_fused_infer_attention_score (inference)
```

### Code Integration
```python
import mindiesd
import torch_npu

def attention_forward(q, k, v, attention_mask=None, **kwargs):
    algo = int(os.environ.get("ALGO", "0"))
    
    output = mindiesd.ops.attention_forward(
        q, k, v,
        attention_mask=attention_mask,
        dropout_p=0.0,
        causal=False,
        algo=algo,
        **kwargs
    )
    return output
```

## File Modifications

### wan/modules/attn_layer.py
```python
# Replace original FlashAttention import
# from flash_attn import flash_attn_func  # REMOVE

# Add MindIE imports
import mindiesd.ops.attention_forward
import torch_npu

class WanAttention:
    def forward(self, q, k, v, **kwargs):
        # Get attention algorithm from environment
        algo = int(os.environ.get("ALGO", "0"))
        
        # Use MindIE attention
        output = mindiesd.ops.attention_forward(
            q, k, v,
            attention_mask=self.attention_mask,
            dropout_p=self.dropout_p,
            causal=self.is_causal,
            algo=algo,
            **kwargs
        )
        
        return output
```

### Key Parameters
```python
# Algorithm-specific configurations
algo_configs = {
    0: {"name": "fused_attn_score", "precision": "fp32", "memory_efficient": True},
    1: {"name": "ascend_laser_attention", "precision": "bf16", "performance": "highest"},
    3: {"name": "npu_fused_infer_attention_score", "precision": "fp16", "inference": True}
}
```

## Performance Comparison

| Algorithm | Speed | Memory Quality | Best For |
|-----------|-------|----------------|----------|
| ALGO=0 | Good | Balanced | General training |
| ALGO=1 | Best | High | Training, inference |
| ALGO=3 | Fastest | Lowest | Inference only |

## Advanced Configuration

### Custom Algorithm Selection
```python
def select_attention_algorithm():
    if os.environ.get("INFER_MODE", "false").lower() == "true":
        return 3  # Inference optimization
    
    # Training mode selection
    if os.environ.get("FAST_LAYERNORM", "1") == "1":
        return 1  # High performance
    
    return 0  # Default stable
```

### Batch Size Optimization
```python
def optimize_attention_for_batch_size(batch_size):
    if batch_size <= 8:
        return 0  # Use fused_attn for small batches
    elif batch_size <= 32:
        return 1  # Use laser_attention for medium batches
    else:
        return 1  # Always use high-performance for large batches
```

## Stream Management

### Compute-Communication Overlap
```python
class AsyncAttention:
    def __init__(self):
        self.stream = torch_npu.Stream()
    
    def forward(self, q, k, v):
        # Create stream for async attention
        with torch_npu.stream(self.stream):
            output = mindiesd.ops.attention_forward(
                q, k, v,
                algo=int(os.environ.get("ALGO", "1"))
            )
        
        return output
```

## Environment Variables

```bash
# Attention algorithm selection
export ALGO=1                    # 0: fused_attn, 1: laser_attn (default), 3: infer_attn

# Additional attention tuning
export OVERLAP=1                 # Enable AllToAll overlap with FA computation
export ATTN_PRECISION=bfloat16   # Override precision (auto, float16, bfloat16)
```

## Troubleshooting

### Issue: Memory allocation failed
**Cause**: Algorithm requires too much memory
**Solutions**:
1. Switch to ALGO=0 for memory efficiency
2. Reduce batch size
3. Enable gradient checkpointing
4. Use sequence parallel

### Issue: Slow attention computation
**Cause**: Algorithm not optimized for current workload
**Solutions**:
1. Try ALGO=1 for better performance
2. Enable OVERLAP=1 for AllToAll overlap
3. Check NPU stream configuration

### Issue: Accuracy degradation with ALGO=1
**Cause**: Laser attention may have precision differences
**Solutions**:
1. Use FP32 instead of BF16 precision
2. Switch to ALGO=0 for numerical stability
3. Add precision-specific calibration

## Performance Optimization

### Algorithm-Aware Training
```python
class AdaptiveAttention:
    def __init__(self):
        self.algo = int(os.environ.get("ALGO", "1"))
        
    def forward(self, q, k, v, training=True):
        if training and self.algo == 3:
            # ALGO=3 is inference-only, fallback to ALGO=1
            return self.attention_forward(q, k, v, algo=1)
        return self.attention_forward(q, k, v, algo=self.algo)
```

### Memory Management
```python
# Enable NPU-specific memory optimization
torch_npu.set_option('ACL_MEMORY_OPTIMIZE', 1)
torch_npu.set_option('ACL_GRAPH_COMPILE_WITH_FP16', 1)
```

## Configuration Templates

### Training Configuration
```yaml
# config/training.yaml
attention:
  algorithm: 1  # Laser attention for training
  precision: "bfloat16"
  overlap_comm: true
  causal: false
  
environment:
  ALGO: "1"
  FAST_LAYERNORM: "1"
  OVERLAP: "1"
```

### Inference Configuration
```yaml
# config/inference.yaml
attention:
  algorithm: 3  # Inference optimization
  precision: "float16"
  overlap_comm: false
  causal: false
  
environment:
  ALGO: "3"
  INFER_MODE: "true"
```

## References

- [MindIE Attention API](https://github.com/Ascend/mindie)
- [FlashAttention Comparison](https://github.com/Dao-AILab/flash-attention)
- [NPU Performance Tuning Guide](https://www.hiascend.com/document)

## Examples

See [`examples/attention_comparison.py`](../examples/attention_comparison.py) for algorithm benchmarking.
