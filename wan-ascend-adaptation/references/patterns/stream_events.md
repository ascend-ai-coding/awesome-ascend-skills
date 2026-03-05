# NPU Stream and Event Management Patterns

Patterns for asynchronous execution, overlapping computation with communication, and fine-grained synchronization on Ascend NPUs.

## Overview

Ascend NPUs support CUDA-style streams for asynchronous execution. Using streams correctly allows overlapping computation with communication, hiding latency and improving throughput in distributed training.

**Key APIs**:

- `torch.npu.Stream()` - Create async execution streams
- `torch.npu.Event()` - Record timing and synchronize operations
- `torch.npu.synchronize()` - Wait for all operations to complete
- `stream.wait_event(event)` - Cross-stream synchronization

## Stream Creation

### Default Stream

```python
import torch
import torch_npu

# Operations on default stream execute sequentially
default_stream = torch.npu.current_stream()
```

### Custom Streams for Async Execution

```python
# Create dedicated streams for different operation types
compute_stream = torch.npu.Stream()   # For attention/FFN computation
comm_stream = torch.npu.Stream()      # For all-to-all communication
```

### Stream Context Manager

```python
# Execute operations on specific stream
with torch.npu.stream(compute_stream):
    output = attention_layer(hidden_states)

# Alternative syntax
compute_stream.wait_stream(torch.npu.default_stream())
with torch.npu.stream(compute_stream):
    output = model.forward(input)
```

## Event Management

### Creating Events

```python
# Events for synchronization points
compute_done = torch.npu.Event(enable_timing=True)  # Enable for profiling
comm_ready = torch.npu.Event()
```

### Recording and Waiting

```python
# Record event on source stream
with torch.npu.stream(compute_stream):
    output = computation(x)
    compute_done.record(compute_stream)  # Mark when compute finishes

# Make target stream wait for event
comm_stream.wait_event(compute_done)   # comm_stream waits for compute

# Execute on comm_stream after compute completes
with torch.npu.stream(comm_stream):
    result = communication(output)
```

### Synchronization Patterns

```python
# Wait for specific stream
compute_stream.synchronize()

# Wait for specific event
compute_done.synchronize()

# Wait for all NPU operations
torch.npu.synchronize()
```

## Overlapping Patterns

### Pattern: Computation-Communication Overlap

**Synchronous Version (Baseline)**

```python
def forward_sync(self, hidden_states):
    # Step 1: All-to-all communication
    hidden_states = self.all_to_all(hidden_states)
    
    # Step 2: Attention computation (waits for comm)
    attn_output = self.attention(hidden_states)
    
    # Step 3: All-to-all communication (waits for attn)
    attn_output = self.all_to_all_reverse(attn_output)
    
    return attn_output
```

**Asynchronous Version with Overlap**

```python
def forward_async(self, hidden_states):
    # Create streams once (typically in __init__)
    # self.compute_stream = torch.npu.Stream()
    # self.comm_stream = torch.npu.Stream()
    
    # Events for synchronization
    comm_ready = torch.npu.Event()
    compute_done = torch.npu.Event()
    
    # Step 1: Start communication on comm_stream
    with torch.npu.stream(self.comm_stream):
        comm_output = self.all_to_all(hidden_states)
        comm_ready.record(self.comm_stream)
    
    # Step 2: Compute attention on compute_stream
    # Wait for communication to finish first
    self.compute_stream.wait_event(comm_ready)
    
    with torch.npu.stream(self.compute_stream):
        attn_output = self.attention(comm_output)
        compute_done.record(self.compute_stream)
    
    # Step 3: Reverse all-to-all
    # Wait for compute to finish
    self.comm_stream.wait_event(compute_done)
    
    with torch.npu.stream(self.comm_stream):
        output = self.all_to_all_reverse(attn_output)
    
    # Synchronize before returning to default stream
    torch.npu.synchronize()
    
    return output
```

### Pattern: Double-Buffered Overlap (Advanced)

```python
def forward_double_buffer(self, hidden_states_list):
    """
    Process multiple batches with overlapped comm/compute.
    While computing batch N, communicate batch N+1.
    """
    results = []
    
    for i, hidden_states in enumerate(hidden_states_list):
        # Alternate between two stream pairs
        streams = self.stream_pairs[i % 2]  # [(compute_0, comm_0), (compute_1, comm_1)]
        compute_s, comm_s = streams
        
        # Communication on comm_s
        with torch.npu.stream(comm_s):
            comm_out = self.all_to_all(hidden_states)
            ready_event = torch.npu.Event()
            ready_event.record(comm_s)
        
        # Computation waits for comm, then executes
        compute_s.wait_event(ready_event)
        with torch.npu.stream(compute_s):
            attn_out = self.attention(comm_out)
            done_event = torch.npu.Event()
            done_event.record(compute_s)
        
        results.append((attn_out, done_event))
    
    # Synchronize all at end
    torch.npu.synchronize()
    return [r[0] for r in results]
```

## Code Examples: Before and After

### Example 1: Attention Layer with All-to-All

**Before (Synchronous)**

```python
class AttentionLayerSync(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Sequential: comm -> compute -> comm
        x = all_to_all(x, scatter_dim=1, gather_dim=2)  # Waits for everything
        
        qkv = self.qkv(x)
        attn = flash_attention(qkv)  # Waits for all-to-all
        
        out = self.proj(attn)
        out = all_to_all_reverse(out)  # Waits for projection
        
        return out
```

**After (Asynchronous with Overlap)**

```python
class AttentionLayerAsync(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Dedicated streams for overlap
        self.compute_stream = torch.npu.Stream()
        self.comm_stream = torch.npu.Stream()
    
    def forward(self, x):
        # Event for comm->compute handoff
        comm_done = torch.npu.Event()
        compute_done = torch.npu.Event()
        
        # Start all-to-all on comm stream
        with torch.npu.stream(self.comm_stream):
            x_comm = all_to_all(x, scatter_dim=1, gather_dim=2)
            comm_done.record(self.comm_stream)
        
        # Attention on compute stream, waits for comm
        self.compute_stream.wait_event(comm_done)
        with torch.npu.stream(self.compute_stream):
            qkv = self.qkv(x_comm)
            attn = flash_attention(qkv)
            out = self.proj(attn)
            compute_done.record(self.compute_stream)
        
        # Reverse all-to-all on comm stream, waits for compute
        self.comm_stream.wait_event(compute_done)
        with torch.npu.stream(self.comm_stream):
            output = all_to_all_reverse(out)
        
        # Ensure completion before returning
        torch.npu.synchronize()
        
        return output
```

### Example 2: Pipeline Parallelism Stage

**Before**

```python
def pipeline_stage_sync(inputs):
    # Forward pass
    outputs = model(inputs)
    
    # Send to next stage
    send(outputs, next_rank)
    
    # Receive grad from next stage (blocks)
    grad_output = recv(prev_rank)
    
    # Backward pass
    grad_input = torch.autograd.grad(outputs, inputs, grad_output)
    
    return grad_input
```

**After (with recv/recv overlap)**

```python
def pipeline_stage_async(inputs):
    forward_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()
    
    # Forward on dedicated stream
    with torch.npu.stream(forward_stream):
        outputs = model(inputs)
        forward_done = torch.npu.Event()
        forward_done.record(forward_stream)
    
    # Send can start as soon as forward outputs ready
    comm_stream.wait_event(forward_done)
    with torch.npu.stream(comm_stream):
        send(outputs, next_rank)
    
    # Receive can happen in parallel with send setup
    with torch.npu.stream(comm_stream):
        grad_output = recv(next_rank)
        recv_done = torch.npu.Event()
        recv_done.record(comm_stream)
    
    # Backward waits for grad from next stage
    torch.npu.default_stream().wait_event(recv_done)
    grad_input = torch.autograd.grad(outputs, inputs, grad_output)
    
    return grad_input
```

## Best Practices

### When to Use Streams

| Scenario | Recommendation |
|----------|----------------|
| Single forward pass | Default stream is fine |
| Comm/Compute overlap | Use 2+ streams |
| Pipeline parallelism | Stream per stage |
| Inference batching | Stream per request |

### When to Synchronize

**Always synchronize**:

- Before accessing tensor data on CPU (`.cpu()`, `.numpy()`)
- At the end of training step (before optimizer step)
- When switching between training and evaluation
- Before measuring elapsed time for profiling
- At epoch/model boundaries

**Avoid unnecessary synchronize**:

- Inside tight loops (prevents overlap)
- Between independent operations on same stream
- After every layer (defeats async execution)

### Stream Lifecycle

```python
class ModelWithStreams(nn.Module):
    def __init__(self):
        super().__init__()
        # Create streams once during init
        self.compute_stream = torch.npu.Stream()
        self.comm_stream = torch.npu.Stream()
    
    def forward(self, x):
        # Reuse streams (do not recreate per forward)
        with torch.npu.stream(self.compute_stream):
            return self.layers(x)
```

### Event Reuse

```python
# Good: Reuse events
event_pool = [torch.npu.Event() for _ in range(4)]

def process_batch(batch, event_idx):
    event = event_pool[event_idx % 4]
    with torch.npu.stream(compute_stream):
        result = model(batch)
        event.record(compute_stream)
    return result, event

# Bad: Creating events in hot path
for batch in dataloader:
    event = torch.npu.Event()  # Expensive
    event.record()
```

## Common Pitfalls

### Pitfall 1: Missing Synchronization

```python
# BUG: Tensor may not be ready
with torch.npu.stream(comm_stream):
    output = all_to_all(input)

# No synchronization - undefined behavior
loss = criterion(output, target)  # May see incomplete data

# FIX: Synchronize before use on different stream
torch.npu.synchronize()
# Or use events for finer control
```

### Pitfall 2: Race Condition

```python
# BUG: Two streams modify same tensor
with torch.npu.stream(stream1):
    tensor.add_(1)

with torch.npu.stream(stream2):
    tensor.mul_(2)  # Race with add_ - undefined order

# FIX: Explicit synchronization
event = torch.npu.Event()
with torch.npu.stream(stream1):
    tensor.add_(1)
    event.record(stream1)

stream2.wait_event(event)
with torch.npu.stream(stream2):
    tensor.mul_(2)
```

### Pitfall 3: CPU Access Without Sync

```python
# BUG: Data race
with torch.npu.stream(compute_stream):
    result = model(input)

# Accessing tensor before GPU work completes
numpy_array = result.cpu().numpy()  # May be incomplete

# FIX: Synchronize before CPU access
torch.npu.synchronize()
numpy_array = result.cpu().numpy()
```

### Pitfall 4: Stream Destruction During Use

```python
# BUG: Stream goes out of scope
def process():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        result = model(input)
    return result  # Stream destroyed, ops may fail

# FIX: Keep stream alive
class Processor:
    def __init__(self):
        self.stream = torch.npu.Stream()
    
    def process(self, input):
        with torch.npu.stream(self.stream):
            return model(input)
```

### Pitfall 5: Excessive Synchronization

```python
# BAD: Synchronize after every layer
for layer in model:
    with torch.npu.stream(compute_stream):
        x = layer(x)
    torch.npu.synchronize()  # Defeats async

# GOOD: Synchronize once at end
for layer in model:
    with torch.npu.stream(compute_stream):
        x = layer(x)
torch.npu.synchronize()
```

## Debugging Tips

### Verify Synchronization

```python
# Use events to verify ordering
event1 = torch.npu.Event(enable_timing=True)
event2 = torch.npu.Event(enable_timing=True)

with torch.npu.stream(stream1):
    op1()
    event1.record()

with torch.npu.stream(stream2):
    stream2.wait_event(event1)
    op2()
    event2.record()

# Check that event2 happens after event1
print(f"Elapsed: {event1.elapsed_time(event2)} ms")
```

### Profile Stream Activity

```python
# Use NPU profiler to see stream activity
with torch.npu.profiler.profile(
    activities=[
        torch.npu.profiler.ProfilerActivity.CPU,
        torch.npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=True
) as prof:
    model(input)

print(prof.key_averages().table())
```

### Check for Missing Sync

```python
# Enable synchronous mode for debugging
torch.npu.set_sync_debug_mode("warn")
# or "full" for strict checking

model(input)  # Will warn if sync issues detected
```

## Summary

1. **Create streams once** - Reuse across iterations
2. **Use events for cross-stream sync** - More efficient than global sync
3. **Synchronize before CPU access** - Always sync before `.cpu()` or `.numpy()`
4. **Avoid sync in hot paths** - Let operations overlap
5. **Profile to verify overlap** - Use events to measure actual overlap achieved
6. **Debug with sync modes** - Enable debug mode to catch race conditions

## References

- [NPU Stream API](https://github.com/Ascend/pytorch)
- [PyTorch CUDA Streams](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [Communication-Computation Overlap](https://arxiv.org/abs/2004.02891)
