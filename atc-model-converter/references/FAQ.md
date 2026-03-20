# ATC Model Conversion FAQ

> **Testing Experience**: This FAQ is based on real-world testing on Ascend 910B3 (CANN 8.1.RC1).  
> For complete session timeline and detailed troubleshooting, see [SESSION_EXPERIENCE.md](../../SESSION_EXPERIENCE.md)

## Critical Compatibility Issues (Must Read)

### Python Version Compatibility

**⚠️ CRITICAL: CANN 8.1.RC1 only supports Python 3.7 - 3.10**

**Error:** `TypeError: cannot pickle 'FrameLocalsProxy' object`

**Symptom:** ATC fails immediately with FrameLocalsProxy error

**Root Cause:** Python 3.11+ introduced FrameLocalsProxy which is incompatible with CANN 8.1.RC1's TBE compiler

**Affected Versions:**
- ❌ Python 3.11, 3.12, 3.13
- ✅ Python 3.7, 3.8, 3.9, 3.10

**Solution:**
```bash
# Create Python 3.10 Conda environment
conda create -n atc_py310 python=3.10 -y
conda activate atc_py310

# Use this environment for all ATC operations
export PATH=/home/miniconda3/envs/atc_py310/bin:$PATH
export PYTHONPATH=/home/miniconda3/envs/atc_py310/lib/python3.10/site-packages:$PYTHONPATH
```

---

### NumPy Version Compatibility

**⚠️ CRITICAL: CANN 8.1.RC1 requires NumPy < 2.0**

**Error:** `AttributeError: np.float_ was removed in the NumPy 2.0 release. Use np.float64 instead.`

**Root Cause:** CANN 8.1.RC1 uses deprecated NumPy API that was removed in NumPy 2.0

**Solution:**
```bash
# Downgrade NumPy
pip install "numpy<2.0" --force-reinstall

# Verify version
python3 -c "import numpy; print(numpy.__version__)"  # Should be 1.x.x
```

**Recommended:** NumPy 1.21 - 1.26

---

### ONNX Opset Version Compatibility

**⚠️ CRITICAL: Use correct ONNX opset version for your CANN version**

**Error:** `No parser is registered for Op [ai.onnx::22::Conv]`

**Root Cause:** PyTorch 2.0+ defaults to ONNX opset 17+, but CANN 8.1.RC1 only supports opset 11-17

**Compatibility Matrix:**

| CANN Version | Supported ONNX Opset |
|--------------|---------------------|
| 8.1.RC1 | 11, 13 |
| 8.3.RC1 | 11, 13, 17 |
| 8.5.0 | 11, 13, 17, 19 |

**Solution - Export with correct opset:**
```python
import torch

model = torch.load('model.pt', map_location='cpu')
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape to your model

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=11,  # Use 11 for CANN 8.1.RC1
    input_names=['input'],
    output_names=['output']
)
```

**Or using the provided export script:**
```bash
python3 scripts/export_onnx.py --pt_model model.pt --output model.onnx \
    --input_shape 1,3,224,224 --opset 11
```

---

### Python Module Dependencies

**⚠️ CANN requires additional Python modules not in default installation**

**Errors:**
```
ModuleNotFoundError: No module named 'decorator'
ModuleNotFoundError: No module named 'attr'
ModuleNotFoundError: No module named 'attrs'
```

**Solution:**
```bash
# Install all required dependencies
pip install decorator attrs absl-py psutil protobuf sympy
```

**Complete environment setup:**
```bash
# Create and activate Python 3.10 environment
conda create -n atc_py310 python=3.10 -y
conda activate atc_py310

# Install PyTorch and ONNX tools
pip install torch torchvision onnx onnxruntime

# Install CANN-compatible NumPy
pip install "numpy<2.0" --force-reinstall

# Install CANN required dependencies
pip install decorator attrs absl-py psutil protobuf sympy
```

---

## Common Errors and Solutions

### E10001: Invalid parameter value

**Error:** `Value [linux] for parameter [--host_env_os] is invalid`

**Cause:** Environment variable or parameter issue

**Solution:**
```bash
# Ensure you're running ATC in a proper terminal, not IDE integrated terminal
# Source CANN environment properly
source /usr/local/Ascend/cann/set_env.sh

# Run directly in shell instead of through IDE
atc --model=model.onnx --framework=5 ...
```

---

### E10016: Opname not found in model

**Error:** `Opname [xxx] specified in [--input_shape] is not found in the model`

**Cause:** Wrong input name specified in `--input_shape`

**Solution:**
```bash
# First, find the correct input names
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx')
for inp in sess.get_inputs():
    print(f'Name: {inp.name}, Shape: {inp.shape}')
"

# Then use the correct name
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --input_shape="correct_name:1,3,224,224"
```

**Note:** Input name is case-sensitive and must match exactly.

---

### Invalid soc_version

**Error:** Model compilation fails with soc_version error

**Cause:** Wrong or unsupported chip version specified

**Solution:**
```bash
# Check actual device version
npu-smi info

# For Atlas 200I DK A2
npu-smi info  # Look for Name field (e.g., "310B4")
# Use: --soc_version=Ascend310B4

# For Atlas 310P
npu-smi info  # Look for Name field
# Use: --soc_version=Ascend310P3 or Ascend310P1
```

---

### Conversion Too Slow

**Symptom:** ATC takes hours to convert a model

**Solutions:**

1. **Enable parallel compilation:**
```bash
export TE_PARALLEL_COMPILER=16  # Adjust based on CPU cores
atc --model=model.onnx ...
```

2. **Use non-ascend machine for compilation:**
```bash
# ATC can run on x86_64 without NPU for model conversion
# Install CANN Toolkit on development machine
```

3. **Enable operator cache:**
```bash
export ASCEND_CACHE_PATH=./atc_cache
atc --model=model.onnx --op_compiler_cache_mode=enable ...
```

---

### Out of Memory During Conversion

**Error:** Compilation fails with memory error

**Solutions:**

1. **Reduce parallel compiler threads:**
```bash
export TE_PARALLEL_COMPILER=4  # Lower value
```

2. **Clear operator cache:**
```bash
rm -rf ~/.cache/atc
```

3. **Use swap space:**
```bash
# Increase swap if needed
```

---

### Precision Issues After Conversion

**Symptom:** Model outputs differ significantly from original ONNX

**Solutions:**

1. **Try FP32 mode:**
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --precision_mode=force_fp32
```

2. **Check AIPP configuration:**
```bash
# Verify AIPP config is correct
# Remove AIPP temporarily to isolate issue
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3
# (without --insert_op_conf)
```

3. **Enable debug mode to compare intermediate results:**
```bash
export DUMP_GE_GRAPH=2
atc --model=model.onnx --log=debug ...
# Compare ge_onnx*.pbtxt files
```

---

### Dynamic Shape Issues

**Error:** `Dynamic shape not supported` or shape mismatch

**Solutions:**

1. **Specify static shape:**
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --input_shape="input:1,3,224,224"
```

2. **Use dynamic batch/size parameters:**
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --dynamic_batch_size="1,2,4,8"
```

---

### Missing Operators

**Error:** `Op [xxx] is not supported`

**Solutions:**

1. **Check CANN version:**
```bash
# Some ops require newer CANN
# Check documentation for supported ops
```

2. **Use custom op plugin:**
```bash
# Implement custom operator for unsupported ops
# See ATC documentation for custom op development
```

3. **Modify model to use supported ops:**
```bash
# Replace unsupported ops with equivalent supported ops
# Use ONNX graph surgery tools
```

---

### File Not Found Errors

**Error:** `No such file or directory` for model or config files

**Solutions:**

1. **Use absolute paths:**
```bash
atc --model=/home/user/models/model.onnx ...
```

2. **Check file permissions:**
```bash
ls -la model.onnx
chmod 644 model.onnx
```

3. **Verify file format:**
```bash
# Check ONNX file is valid
python3 -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"
```

---

## Environment Issues

### ATC Command Not Found

**Symptom:** `atc: command not found`

**Solution:**
```bash
# Source CANN environment
source /usr/local/Ascend/cann/set_env.sh

# Verify installation
which atc
ls /usr/local/Ascend/cann/bin/atc
```

### Python Module Import Errors

**Error:** Python import errors during conversion

**Solution:**
```bash
# Set Python path
export PYTHONPATH=/usr/local/Ascend/cann/python/site-packages:$PYTHONPATH

# Or use CANN's Python
export PATH=/usr/local/python3.7.5/bin:$PATH
```

---

## Performance Optimization

### Slow Inference After Conversion

**Solutions:**

1. **Use FP16 precision:**
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --precision_mode=force_fp16
```

2. **Enable fusion:**
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --fusion_switch_file=fusion.cfg
```

3. **Use AIPP for preprocessing:**
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --insert_op_conf=aipp.cfg
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
atc --model=model.onnx --log=debug ...
```

### Dump Intermediate Graphs

```bash
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=2
atc --model=model.onnx ...
# View ge_onnx*.pbtxt files with Netron
```

### Check Operator Compilation

```bash
# Check operator support
atc --model=model.onnx --op_debug_level=1 ...
```

### Validate Environment

```bash
# Use provided check script
./scripts/check_env_enhanced.sh
```

---

## Pre-Conversion Checklist

> **完整的端到端转换流程请参考 SKILL.md 中的 Workflow 1-3。**

Before running ATC, verify:

- [ ] **Python version**: 3.7, 3.8, 3.9, or 3.10 (NOT 3.11+)
- [ ] **NumPy version**: < 2.0 (check with `pip show numpy`)
- [ ] **Python modules**: decorator, attrs, absl-py, psutil, protobuf, sympy installed
- [ ] **ONNX opset**: 11 or 13 for CANN 8.1.RC1 (verify with Netron)
- [ ] **CANN environment**: Sourced with `source set_env.sh`
- [ ] **PYTHONPATH**: Points to correct Python site-packages
- [ ] **Input shape**: Verified with `get_onnx_info.py`
- [ ] **soc_version**: Matches hardware (check with `npu-smi info`)

---

## Quick Troubleshooting Decision Tree

```
ATC Failed?
│
├── Error: "cannot pickle 'FrameLocalsProxy'"
│   └── Solution: Use Python 3.10, not 3.11+
│
├── Error: "np.float_ was removed"
│   └── Solution: pip install "numpy<2.0"
│
├── Error: "No parser is registered for Op [ai.onnx::22::xxx]"
│   └── Solution: Re-export ONNX with opset=11
│
├── Error: "ModuleNotFoundError: No module named 'xxx'"
│   └── Solution: pip install decorator attrs absl-py psutil protobuf sympy
│
├── Error: "Opname not found in model"
│   └── Solution: Check input names with get_onnx_info.py
│
└── Error: "Invalid soc_version"
    └── Solution: Check with npu-smi info, use format "Ascendxxx"
```

---

## Version Compatibility Matrix

详见 [CANN_VERSIONS.md](CANN_VERSIONS.md) 中的完整版本兼容性矩阵。

---

## Getting Help

1. **Check documentation:** See ATC offline model compilation guide
2. **Community forums:** Huawei Cloud community and Ascend forums
3. **Model Zoo:** Reference implementations for common models
4. **Debug outputs:** Use --log=debug and graph dumps for detailed analysis
5. **Session Experience:** See [SESSION_EXPERIENCE.md](../../SESSION_EXPERIENCE.md) for detailed timeline
