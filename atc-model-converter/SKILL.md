---
name: atc-model-converter
description: Complete toolkit for Huawei Ascend NPU model conversion and inference. (1) Convert ONNX models to .om format using ATC tool with multi-CANN version support (8.3.RC1, 8.5.0+). (2) Run Python inference on OM models using ais_bench. (3) Compare precision between CPU ONNX and NPU OM outputs. (4) End-to-end YOLO inference with Ultralytics preprocessing/postprocessing. Use when converting, testing, or deploying models on Ascend AI processors.
---

# ATC Model Converter

Complete guide for converting ONNX models to Ascend AI processor compatible format using ATC (Ascend Tensor Compiler) tool.

**Supported CANN Versions:** 8.3.RC1, 8.5.0

## ⚠️ Critical Compatibility Requirements

Before starting, ensure your environment meets these requirements:

| Component | Requirement | Why |
|-----------|-------------|-----|
| **Python** | 3.7, 3.8, 3.9, or **3.10** | Python 3.11+ incompatible with CANN 8.1.RC1 |
| **NumPy** | **< 2.0** (e.g., 1.26.4) | CANN uses deprecated NumPy API |
| **ONNX Opset** | 11 or 13 (for CANN 8.1.RC1) | Higher opset versions not supported |

**Quick Environment Setup:**
```bash
# Create Python 3.10 environment (recommended)
conda create -n atc_py310 python=3.10 -y
conda activate atc_py310

# Install compatible dependencies
pip install torch torchvision ultralytics onnx onnxruntime
pip install "numpy<2.0" --force-reinstall
pip install decorator attrs absl-py psutil protobuf sympy
```

See [FAQ.md](references/FAQ.md) for detailed troubleshooting.

---

## Quick Start

```bash
# 1. Check your CANN version and environment
./scripts/check_env.sh

# 2. Source the appropriate environment (see CANN Version Guide below)
export PATH=/home/miniconda3/envs/atc_py310/bin:$PATH  # If using conda
export PYTHONPATH=/home/miniconda3/envs/atc_py310/lib/python3.10/site-packages:$PYTHONPATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # For 8.1.RC1/8.3.RC1
# OR
source /usr/local/Ascend/cann/set_env.sh  # For 8.5.0

# 3. Basic ONNX to OM conversion
atc --model=model.onnx --framework=5 --output=output_model --soc_version=Ascend310P3

# With input shape specification
atc --model=model.onnx --framework=5 --output=output_model \
    --soc_version=Ascend310P3 \
    --input_shape="input:1,3,224,224"

# With AIPP preprocessing
atc --model=model.onnx --framework=5 --output=output_model \
    --soc_version=Ascend310P3 \
    --insert_op_conf=aipp_config.cfg
```

## CANN Version Guide

Different CANN versions have different environment setup paths:

| CANN Version | Environment Path | Ops Package Requirement |
|--------------|------------------|-------------------------|
| 8.3.RC1 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | Standard installation |
| 8.5.0 | `/usr/local/Ascend/cann/set_env.sh` | Must install matching ops package |

### Auto-detect CANN Version

```bash
# Use provided script to detect and set environment
./scripts/setup_env.sh

# Or manually check version
atc --help 2>&1 | head -5
# OR
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || \
cat /usr/local/Ascend/cann/latest/version.cfg 2>/dev/null
```

### Environment Setup by Version

**For CANN 8.3.RC1:**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/lib64:$LD_LIBRARY_PATH
```

**For CANN 8.5.0+:**
```bash
source /usr/local/Ascend/cann/set_env.sh
# For non-Ascend host development only:
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/<arch>-linux/devlib:$LD_LIBRARY_PATH
```

## OM Model Inference

After converting your model to OM format, use ais_bench for Python inference on Ascend NPU.

### Install ais_bench

```bash
# Source compilation (recommended for latest version)
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_bench

# Build packages
pip3 wheel ./backend/ -v  # Build aclruntime
pip3 wheel ./ -v          # Build ais_bench

# Install
pip3 install ./aclruntime-*.whl ./ais_bench-*.whl
```

### Basic Inference

```bash
# Print model info
python3 scripts/infer_om.py --model model.om --info

# Run inference with random input
python3 scripts/infer_om.py --model model.om --input-shape "1,3,640,640"

# Run inference with actual input
python3 scripts/infer_om.py --model model.om --input test.npy --output result.npy

# Performance benchmark
python3 scripts/infer_om.py --model model.om --warmup 10 --loop 100
```

### Python API Usage

```python
from ais_bench.infer.interface import InferSession
import numpy as np

# Initialize session
session = InferSession(device_id=0, model_path="model.om")

# Get model info
print("Inputs:", [(i.name, i.shape) for i in session.get_inputs()])
print("Outputs:", [(o.name, o.shape) for o in session.get_outputs()])

# Run inference
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.infer([input_data], mode='static')

# Get timing
print(f"Inference time: {session.summary().exec_time_list[-1]:.3f} ms")

# Cleanup
session.free_resource()
```

---

## Precision Comparison

Verify conversion accuracy by comparing ONNX (CPU) vs OM (NPU) outputs.

```bash
# Basic comparison
python3 scripts/compare_precision.py --onnx model.onnx --om model.om --input test.npy

# With custom tolerances
python3 scripts/compare_precision.py --onnx model.onnx --om model.om --input test.npy \
    --atol 1e-3 --rtol 1e-2

# Save detailed results
python3 scripts/compare_precision.py --onnx model.onnx --om model.om --input test.npy \
    --output comparison.json --save-diff ./diff/
```

### Understanding Results

| Metric | Description |
|--------|-------------|
| `cosine_similarity` | 1.0 = identical, >0.99 = very close |
| `max_abs_diff` | Maximum absolute difference |
| `outlier_ratio` | Percentage of values exceeding tolerance |
| `is_close` | Pass/fail based on atol/rtol |

**Typical acceptable thresholds:**
- FP32 models: `atol=1e-4`, `rtol=1e-3`
- FP16 models: `atol=1e-3`, `rtol=1e-2`

---

## End-to-End YOLO Inference

Run complete YOLO inference pipeline with Ultralytics preprocessing/postprocessing.

```bash
# Single image
python3 scripts/yolo_om_infer.py --model yolo.om --source image.jpg --output result.jpg

# Directory of images
python3 scripts/yolo_om_infer.py --model yolo.om --source images/ --output results/

# With custom confidence threshold
python3 scripts/yolo_om_infer.py --model yolo.om --source image.jpg --conf 0.5

# Save detection results to txt files
python3 scripts/yolo_om_infer.py --model yolo.om --source images/ --save-txt
```

### Python API for YOLO

```python
from yolo_om_infer import YoloOMInferencer

# Initialize
inferencer = YoloOMInferencer(
    model_path="yolo.om",
    device_id=0,
    conf_thres=0.25,
    iou_thres=0.45
)

# Run inference
result = inferencer("image.jpg")

# Access results
print(f"Detections: {result['num_detections']}")
print(f"Inference time: {result['timing']['infer_ms']:.1f}ms")

for det in result['detections']:
    print(f"  {det['cls_name']}: {det['conf']:.2f} at {det['box']}")

# Cleanup
inferencer.free_resource()
```

---

## Prerequisites

### Required Software

- CANN Toolkit (8.3.RC1 or 8.5.0)
- Python 3.7+ (for helper scripts)
- onnxruntime (optional, for `get_onnx_info.py`)

### Environment Variables

```bash
# Optional: Set parallel compilation for large models
export TE_PARALLEL_COMPILER=8

# Optional: Enable graph dump for debugging
export DUMP_GE_GRAPH=1

# Optional: Enable verbose logging
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

### Verify Environment

```bash
# Check ATC tool is available
which atc
atc --help | head -3

# Check NPU device info
npu-smi info -l

# Verify CANN environment
echo $ASCEND_HOME
```

## Core Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--model` | Yes | Input ONNX model path | `--model=resnet50.onnx` |
| `--framework` | Yes | Framework type (5=ONNX) | `--framework=5` |
| `--output` | Yes | Output OM model path | `--output=resnet50` |
| `--soc_version` | Yes | Ascend chip version | `--soc_version=Ascend310P3` |
| `--input_shape` | Optional | Input tensor shapes | `--input_shape="input:1,3,224,224"` |
| `--input_format` | Optional | Input format (NCHW/NHWC) | `--input_format=NCHW` |
| `--precision_mode` | Optional | Precision mode | `--precision_mode=force_fp16` |
| `--insert_op_conf` | Optional | AIPP config file path | `--insert_op_conf=aipp.cfg` |
| `--log` | Optional | Log level | `--log=info` |

## Common Workflows

### 0. Preparing ONNX Model (CRITICAL)

Before ATC conversion, ensure your ONNX model uses the correct opset version:

**Using Ultralytics (YOLO models):**
```python
from ultralytics import YOLO

model = YOLO('model.pt')
# Use opset 11 for maximum compatibility with CANN 8.1.RC1
model.export(format='onnx', imgsz=640, opset=11)
```

**Using PyTorch directly:**
```python
import torch

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=11,  # Use 11 for CANN 8.1.RC1, 13 or 17 for newer CANN
    input_names=['images'],
    output_names=['output0']
)
```

**Verify ONNX opset version:**
```bash
# Install onnx if needed
pip install onnx

# Check opset version
python3 -c "import onnx; model = onnx.load('model.onnx'); print(model.opset_import)"
```

---

### 1. Basic ONNX Conversion

Convert a simple ONNX model without shape specification:

```bash
atc --model=yolov5s.onnx \
    --framework=5 \
    --output=yolov5s \
    --soc_version=Ascend310P3
```

### 2. Conversion with Input Shape

When input shape is dynamic or needs override:

```bash
# First, inspect ONNX input names
python3 scripts/get_onnx_info.py model.onnx

# Then convert with correct input name
atc --model=model.onnx \
    --framework=5 \
    --output=model_om \
    --soc_version=Ascend310P3 \
    --input_shape="actual_input_name:1,3,640,640"
```

### 3. Conversion with AIPP

AIPP (AI Preprocessing) handles image preprocessing on NPU:

```bash
# Create AIPP config file
cat > aipp.cfg << 'AIPP_EOF'
aipp_op {
    aipp_mode: static
    input_format: YUV420SP_U8
    src_image_size_w: 640
    src_image_size_h: 640
    crop: false
    resize: false
    csc_switch: true
    matrix_r0c0: 298
    matrix_r0c1: 516
    matrix_r0c2: 0
    matrix_r1c0: 298
    matrix_r1c1: -100
    matrix_r1c2: -208
    matrix_r2c0: 298
    matrix_r2c1: 0
    matrix_r2c2: 409
    input_bias_0: 16
    input_bias_1: 128
    input_bias_2: 128
    mean_chn_0: 104
    mean_chn_1: 117
    mean_chn_2: 123
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 1.0
    var_reci_chn_1: 1.0
    var_reci_chn_2: 1.0
}
AIPP_EOF

# Convert with AIPP
atc --model=model.onnx \
    --framework=5 \
    --output=model_aipp \
    --soc_version=Ascend310P3 \
    --insert_op_conf=aipp.cfg
```

### 4. Precision Mode Selection

Control computation precision:

```bash
# Force FP32 for maximum precision
atc --model=model.onnx --framework=5 --output=model_fp32 \
    --soc_version=Ascend310P3 --precision_mode=force_fp32

# Force FP16 for better performance (may reduce precision)
atc --model=model.onnx --framework=5 --output=model_fp16 \
    --soc_version=Ascend310P3 --precision_mode=force_fp16

# Allow mix precision (default)
atc --model=model.onnx --framework=5 --output=model_mix \
    --soc_version=Ascend310P3 --precision_mode=allow_mix_precision
```

## Soc Version Reference

| Device | Soc Version | How to Check |
|--------|-------------|--------------|
| Atlas 200I DK A2 | Ascend310B4 | `npu-smi info` |
| Atlas 310P | Ascend310P1/P3 | `npu-smi info` |
| Atlas 910 | Ascend910 | `npu-smi info` |
| Atlas 910B | Ascend910B | `npu-smi info` |

**Get your soc_version:**
```bash
# Get Name field, prepend "Ascend"
npu-smi info | grep Name
# Result: Name: xxxyy → Use: Ascendxxxyy
```

## Troubleshooting

### Error: Opname not found in model

**Cause:** Wrong input name specified in `--input_shape`

**Solution:**
```bash
# Verify input names with script
python3 scripts/get_onnx_info.py model.onnx

# Use correct name in conversion
atc --model=model.onnx --input_shape="correct_name:1,3,224,224" ...
```

### Error: Invalid soc_version

**Cause:** Wrong chip version specified

**Solution:**
```bash
# Check actual chip version
npu-smi info
# Use the exact Name from output with "Ascend" prefix
```

### Conversion Too Slow

**Solution:** Enable parallel compilation
```bash
export TE_PARALLEL_COMPILER=16
atc --model=model.onnx ...
```

### Debug Graph Issues

```bash
# Enable graph dumping
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=2

# Run conversion, then check generated .pbtxt files
atc --model=model.onnx ...
# View ge_onnx*.pbtxt with Netron
```

## Version-Specific Notes

### CANN 8.3.RC1

- Environment path: `/usr/local/Ascend/ascend-toolkit/`
- Some advanced fusion options may differ
- Standard ops package installation

### CANN 8.5.0+

- Environment path: `/usr/local/Ascend/cann/`
- **Important:** Must install matching ops package for target device
- Enhanced fusion and optimization capabilities
- New parameters: `--compression_optimize_conf`, `--op_compiler_cache_mode`

See [CANN_VERSIONS.md](references/CANN_VERSIONS.md) for detailed version-specific differences.

## Advanced Parameters

See [PARAMETERS.md](references/PARAMETERS.md) for complete ATC parameter reference including:
- Fusion configuration
- Custom operator settings
- Dynamic shape handling
- Quantization options
- Memory optimization

## Resources

### scripts/
**Conversion & Environment:**
- **`check_env_enhanced.sh`** - ⭐ **RECOMMENDED** - Comprehensive compatibility check (Python, NumPy, modules, CANN)
- `get_onnx_info.py` - Inspect ONNX model inputs/outputs
- `convert_onnx.sh` - Batch conversion helper
- `check_env.sh` - Basic CANN environment check
- `setup_env.sh` - Auto-setup CANN environment

**Inference & Testing:**
- **`infer_om.py`** - ⭐ Python inference for OM models using ais_bench
- **`compare_precision.py`** - ⭐ Compare ONNX vs OM output precision
- **`yolo_om_infer.py`** - ⭐ End-to-end YOLO inference with Ultralytics pipeline

### references/
- [PARAMETERS.md](references/PARAMETERS.md) - Complete ATC parameter reference
- [AIPP_CONFIG.md](references/AIPP_CONFIG.md) - AIPP configuration guide
- [INFERENCE.md](references/INFERENCE.md) - ⭐ ais_bench inference guide (API, modes, optimization)
- [FAQ.md](references/FAQ.md) - Frequently asked questions
- [CANN_VERSIONS.md](references/CANN_VERSIONS.md) - Version-specific guidance
