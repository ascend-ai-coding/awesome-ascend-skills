# OM Model Inference Reference

Complete guide for running inference on Ascend NPU using ais_bench.

## Table of Contents

1. [Installation](#installation)
2. [Python API](#python-api)
3. [Inference Modes](#inference-modes)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- CANN Toolkit (6.0.RC1+)
- Python 3.7+
- numpy, tqdm, wheel

### Install from Source (Recommended)

```bash
# Clone repository
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_bench

# Build aclruntime (device communication layer)
pip3 wheel ./backend/ -v

# Build ais_bench (inference interface)
pip3 wheel ./ -v

# Install both packages
pip3 install ./aclruntime-*.whl ./ais_bench-*.whl
```

### Verify Installation

```python
from ais_bench.infer.interface import InferSession
print("ais_bench installed successfully")
```

---

## Python API

### InferSession Class

Main class for single-process OM model inference.

```python
from ais_bench.infer.interface import InferSession

# Initialize
session = InferSession(
    device_id=0,           # NPU device ID (check with npu-smi info)
    model_path="model.om", # Path to OM model
    acl_json_path=None,    # Optional: path to acl.json for profiling/dump
    debug=False,           # Enable verbose logging
    loop=1                 # Number of repeated inference per call
)
```

### Get Model Information

```python
# Input tensor info
for inp in session.get_inputs():
    print(f"Input: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.datatype}")
    print(f"  Size: {inp.size} bytes")

# Output tensor info
for out in session.get_outputs():
    print(f"Output: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type: {out.datatype}")
```

### Basic Inference

```python
import numpy as np

# Prepare input (numpy array)
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = session.infer([input_data], mode='static')

# outputs is a list of numpy arrays
for i, out in enumerate(outputs):
    print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
```

### Performance Measurement

```python
# Get inference time
summary = session.summary()
times = summary.exec_time_list  # List of (start, end) tuples

# Calculate metrics
import numpy as np
if times:
    latencies = [end - start for start, end in times]
    print(f"Mean latency: {np.mean(latencies):.3f} ms")
    print(f"Throughput: {1000/np.mean(latencies):.1f} FPS")

# Reset statistics
session.reset_summaryinfo()
```

### Resource Cleanup

```python
# Release device resources for this session
session.free_resource()

# Release all AscendCL resources (call once at program end)
InferSession.finalize()
```

---

## Inference Modes

### Static Shape

For models with fixed input shapes.

```python
outputs = session.infer([input_data], mode='static')
```

### Dynamic Batch

For models compiled with `--dynamic_batch_size`.

```python
# Model was compiled with: --dynamic_batch_size="1,2,4,8"
outputs = session.infer([input_data], mode='dymbatch')
```

### Dynamic HW

For models compiled with `--dynamic_image_size`.

```python
# Model was compiled with: --dynamic_image_size="224,224;448,448"
outputs = session.infer([input_data], mode='dymhw')
```

### Dynamic Dims

For models compiled with `--dynamic_dims`.

```python
# Model was compiled with: --dynamic_dims="1,3,224,224;1,3,448,448"
outputs = session.infer([input_data], mode='dymdims')
```

### Dynamic Shape

For models compiled with `--input_shape_range`. Requires output size specification.

```python
# Specify output buffer size (in bytes)
outputs = session.infer(
    [input_data],
    mode='dymshape',
    custom_sizes=100000  # or [size1, size2] for multiple outputs
)
```

---

## Performance Optimization

### Pipeline Inference

For batch processing, use `infer_pipeline` to overlap compute and data transfer.

```python
# Prepare multiple inputs
inputs_list = [input1, input2, input3, input4]

# Pipeline inference (better throughput)
outputs_list = session.infer_pipeline(inputs_list, mode='static')

# outputs_list is a list of output lists
```

### Multi-Threaded Inference

```python
# Enable pipeline mode for compute-transfer parallelism
outputs = session.infer([input_data], mode='static')  # Already uses internal optimization
```

### Multi-Device Inference

Use `MultiDeviceSession` for multi-NPU parallelism.

```python
from ais_bench.infer.interface import MultiDeviceSession

# Initialize (doesn't load model yet)
multi_session = MultiDeviceSession(model_path="model.om")

# Run on multiple devices
devices_feeds = {
    0: [input1, input2],  # Device 0 inputs
    1: [input3, input4],  # Device 1 inputs
}
results = multi_session.infer(devices_feeds, mode='static')

# results is {device_id: [outputs...]}
```

### Profiling

```python
# Create acl.json for profiling
import json
acl_config = {
    "profiler": {
        "switch": "on",
        "output": "./profiler_results"
    }
}
with open("acl.json", "w") as f:
    json.dump(acl_config, f)

# Use in session
session = InferSession(device_id=0, model_path="model.om", acl_json_path="acl.json")
```

---

## Troubleshooting

### Common Errors

**Error: `ImportError: cannot import name 'InferSession'`**

Solution: Install both aclruntime and ais_bench packages.

```bash
pip3 install ./aclruntime-*.whl ./ais_bench-*.whl
```

**Error: `acl init failed`**

Solution: Source CANN environment.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # or cann/set_env.sh
```

**Error: `open device failed`**

Solution: Check NPU status.

```bash
npu-smi info  # Should show device info
```

**Error: `load model failed`**

Solution: Verify model path and format.

```bash
# Check model file
file model.om  # Should be "data"
ls -la model.om
```

### Memory Issues

For large models or batch processing:

```python
# Process in smaller batches
batch_size = 4
for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i+batch_size]
    outputs = session.infer_pipeline(batch, mode='static')
    # Process outputs...

# Free resources between batches if needed
session.free_resource()
```

### Performance Debugging

```python
# Enable debug mode
session = InferSession(device_id=0, model_path="model.om", debug=True)

# Check timing breakdown
outputs = session.infer([input_data], mode='static')
# Debug mode prints H2D, compute, D2H times
```

---

## Input Data Types

Supported input types for `infer()`:

| Type | Example |
|------|---------|
| `numpy.ndarray` | `np.random.randn(1,3,224,224).astype(np.float32)` |
| `torch.Tensor` | `torch.randn(1,3,224,224)` |
| `aclruntime.Tensor` | Advanced use case |

### NumPy Dtype Mapping

| Model Dtype | NumPy Dtype |
|-------------|-------------|
| float32 | `np.float32` |
| float16 | `np.float16` |
| int32 | `np.int32` |
| int64 | `np.int64` |
| uint8 | `np.uint8` |

---

## Command Line Usage

ais_bench also provides CLI tool for quick testing:

```bash
# Basic inference
python3 -m ais_bench --model model.om

# With input data
python3 -m ais_bench --model model.om --input data/ --output results/

# Performance benchmark
python3 -m ais_bench --model model.om --loop 100 --warmup_count 10

# Dynamic batch
python3 -m ais_bench --model model.om --dymBatch 4

# Debug mode
python3 -m ais_bench --model model.om --debug 1

# Multi-device
python3 -m ais_bench --model model.om --device 0,1,2,3
```

---

## References

- [ais_bench README](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README.md)
- [ais_bench API Guide](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/API_GUIDE.md)
- [CANN Inference Documentation](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/inferapplicationdev/aclcppdevg/aclcppdevg_0000.html)
