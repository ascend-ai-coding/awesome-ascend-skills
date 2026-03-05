# Distributed Training Pattern

Migrate from NCCL to HCCL backend with enhanced parallelism strategies.

## Backend Migration

### NCCL → HCCL
```python
# Original (NCCL)
import torch.distributed as dist

dist.init_process_group(backend="nccl")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# NPU adaptation (HCCL)
import torch.distributed as dist
import torch_npu

dist.init_process_group(backend="hccl")  # Use HCCL for NPU
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
torch_npu.npu.synchronize()  # Ensure completion
```

## File Modifications

### wan/distributed/parallel_mgr.py
```python
import torch_npu

def init_model_parallel_group(backend=None):
    if backend is None:
        backend = "hccl" if torch_npu.is_available() else "nccl"
    
    dist.init_process_group(backend=backend)
    
    # NPU-specific device setup
    if torch_npu.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch_npu.npu.set_device(local_rank)
    
    return dist.get_rank(), dist.get_world_size()
```

## Tensor Parallel

### Enhanced ColumnParallelLinear
```python
import torch_npu

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_per_partition = out_features // world_size
        
        # Partition weight across GPUs/NPUs
        self.weight = nn.Parameter(
            torch.empty(self.out_per_partition, in_features)
        )
    
    def forward(self, x):
        # Use NPU-optimized GEMM with all-reduce
        output = torch_npu.npu_mm_all_reduce_base(
            x, self.weight, 
            reduction='sum',
            comm_group=self.tp_group
        )
        return output
```

### Enhanced RowParallelLinear
```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_per_partition = in_features // world_size
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_partition)
        )
    
    def forward(self, x):
        # Partition input and compute
        output = F.linear(x, self.weight)
        
        # All-reduce across TP group
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        torch_npu.npu.synchronize()
        
        return output
```

## VAE Patch Parallel

### 2D Spatial Partitioning
```python
class VAEPatchParallel:
    def __init__(self, patch_size, tp_size):
        self.patch_size = patch_size
        self.tp_size = tp_size
        
        # Create communication groups
        self.row_group = dist.new_group(ranks=list(range(0, tp_size)))
        self.col_group = dist.new_group(ranks=list(range(tp_size, 2*tp_size)))
    
    def split_patches(self, x):
        # Split feature map into patches
        B, C, H, W = x.shape
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        
        # Reshape and permute
        x = x.reshape(B, C, H_patches, self.patch_size, W_patches, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, H_patches*W_patches, C, -1)
        
        return x
    
    def communicate_patches(self, x):
        # Row communication
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.row_group)
        torch_npu.npu.synchronize()
        
        # Column communication
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.col_group)
        torch_npu.synchronize()
        
        return x
```

## Sequence Parallel (Ulysses + Ring)

### Ulysses Sequence Parallel
```python
class UlyssesSequenceParallel:
    def __init__(self, world_size):
        self.world_size = world_size
        self.seq_length_per_rank = None
    
    def forward(self, x):
        seq_len = x.size(1)
        self.seq_length_per_rank = seq_len // self.world_size
        
        # Split sequence across ranks
        x = x[:, self.seq_length_per_rank * self.rank: 
                 self.seq_length_per_rank * (self.rank + 1), :]
        
        return x
    
    def backward(self, grad_output):
        # Gather gradients across ranks
        gathered_grad = torch.empty_like(grad_output)
        dist.all_gather_into_tensor(gathered_grad, grad_output)
        torch_npu.synchronize()
        
        return gathered_grad
```

## Communication Overlap

### Async AllReduce with Stream
```python
class AsyncCommunication:
    def __init__(self):
        self.comm_stream = torch_npu.Stream()
        self.compute_stream = torch_npu.Stream()
    
    def async_all_reduce(self, tensor):
        # Start all-reduce in communication stream
        with torch_npu.stream(self.comm_stream):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Continue computation in compute stream
        with torch_npu.stream(self.compute_stream):
            # Do other work while communication happens
            pass
        
        # Synchronize before using results
        torch_npu.synchronize()
        return tensor
```

## Environment Variables

```bash
# Backend selection
export TORCH_DIST_BACKEND=hccl     # Force HCCL backend

# Communication optimization
export HCCL_BLOCK_SIZE=1048576     # Block size for all-reduce
export HCCL_ALLREDUCE_TREE=ring    # All-reduce algorithm (ring, tree)

# NPU-specific
export HCCL_ENABLE_TCP=0           # Use RoCE instead of TCP
export HCCL_IB_TIMEOUT=18          # InfiniBand timeout
```

## Troubleshooting

### Issue: Distributed training hangs
**Cause**: HCCL backend not initialized properly
**Solution**:
```python
# Verify NPU device setup
if torch_npu.is_available():
    torch_npu.npu.set_device(local_rank)
    
# Check HCCL availability
print(f"HCCL available: {dist.is_hccl_available()}")
```

### Issue: Slow all-reduce performance
**Cause**: Network configuration suboptimal
**Solutions**:
```bash
# Use ring all-reduce for small tensors
export HCCL_ALLREDUCE_TREE=ring

# Increase block size for large tensors
export HCCL_BLOCK_SIZE=2097152

# Enable RoCE if available
export HCCL_ENABLE_TCP=0
export HCCL_IB_HCA=mlx5_0
```

### Issue: Deadlock in VAE patch parallel
**Cause**: Communication group mismatch
**Solution**:
```python
# Verify group membership
print(f"Row group ranks: {dist.get_process_group_ranks(self.row_group)}")
print(f"Col group ranks: {dist.get_process_group_ranks(self.col_group)}")
```

## Configuration Templates

### Multi-Node Configuration
```yaml
distributed:
  backend: "hccl"
  tensor_parallel_size: 4
  sequence_parallel: "ulysses"
  vae_patch_parallel: true
  
environment:
  TORCH_DIST_BACKEND: "hccl"
  HCCL_ALLREDUCE_TREE: "ring"
  OVERLAP: "1"
```

### Single-Node Configuration
```yaml
distributed:
  backend: "hccl"
  tensor_parallel_size: 8
  sequence_parallel: "ring"
  vae_patch_parallel: false
  
environment:
  TORCH_DIST_BACKEND: "hccl"
  OVERLAP: "1"
```

## References

- [HCCL Programming Guide](https://www.hiascend.com/document)
- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Sequence Parallel Strategies](https://arxiv.org/abs/2305.14314)
