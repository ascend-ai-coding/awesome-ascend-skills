# CANN Version Differences Guide

This document details the differences between CANN versions relevant to ATC model conversion.

## Version Overview

| Version | Release Date | Environment Path | Key Changes | Tested |
|---------|-------------|------------------|-------------|---------|
| 8.1.RC1 | 2023 | `ascend-toolkit` | Legacy version, strict requirements | ✅ Yes (Ascend 910B3) |
| 8.3.RC1 | 2023 | `ascend-toolkit` | Standard toolkit installation | ✅ Supported |
| 8.5.0 | 2024 | `cann` | New path, ops package required | ✅ Supported |

### Compatibility Quick Reference

| CANN Version | Python Version | NumPy Version | ONNX Opset | Python Modules Required |
|--------------|----------------|---------------|------------|-------------------------|
| 8.1.RC1 | **3.7 - 3.10** | **< 2.0** | **11, 13** | decorator, attrs, absl-py, psutil, protobuf, sympy |
| 8.3.RC1 | 3.7 - 3.10 | < 2.0 | 11, 13, 17 | decorator, attrs, absl-py |
| 8.5.0 | 3.7 - 3.10 | >= 1.21 | 11, 13, 17, 19 | decorator, attrs, absl-py |

## Environment Path Differences

### CANN 8.3.RC1

```bash
# Installation path
/usr/local/Ascend/ascend-toolkit/

# Environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Library path
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/lib64:$LD_LIBRARY_PATH
```

### CANN 8.5.0+

```bash
# Installation path
/usr/local/Ascend/cann/

# Environment setup
source /usr/local/Ascend/cann/set_env.sh

# Library path (for non-Ascend hosts)
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/<arch>-linux/devlib:$LD_LIBRARY_PATH
```

## Key Differences

### ⚠️ CANN 8.1.RC1 Special Requirements

CANN 8.1.RC1 has strict compatibility requirements that differ from newer versions:

#### Python Version (CRITICAL)
- **Supported:** Python 3.7, 3.8, 3.9, 3.10
- **NOT Supported:** Python 3.11, 3.12, 3.13
- **Error if using wrong version:** `TypeError: cannot pickle 'FrameLocalsProxy' object`
- **Solution:** Use Conda to create Python 3.10 environment

#### NumPy Version (CRITICAL)
- **Required:** NumPy < 2.0 (e.g., 1.26.4)
- **NOT Supported:** NumPy 2.0+
- **Error if using wrong version:** `AttributeError: np.float_ was removed in the NumPy 2.0 release`
- **Solution:** `pip install "numpy<2.0" --force-reinstall`

#### ONNX Opset Version (CRITICAL)
- **Supported:** Opset 11, 13
- **NOT Supported:** Opset 17+
- **Error if using wrong version:** `No parser is registered for Op [ai.onnx::22::Conv]`
- **Solution:** Export ONNX with `opset_version=11`

#### Required Python Modules
Must install these additional modules:
```bash
pip install decorator attrs absl-py psutil protobuf sympy
```

---

### 1. Ops Package Requirement

**CANN 8.1.RC1:**
- Ops package (opp) is included in standard installation
- No additional ops package required

**CANN 8.3.RC1:**
- Ops package (opp) is optional for basic functionality
- Standard installation works for most use cases

**CANN 8.5.0+:**
- **Mandatory:** Must install matching ops package for target device
- Without ops package, model compilation will fail
- Ops package must match the specific Ascend chip version

### 2. ATC Parameters

Most parameters work identically across versions. However:

| Parameter | 8.3.RC1 | 8.5.0+ | Notes |
|-----------|---------|--------|-------|
| `--compression_optimize_conf` | Limited | Full | Enhanced compression in 8.5.0+ |
| `--op_compiler_cache_mode` | Not available | Available | Speeds up re-compilation |
| `--op_compiler_cache_dir` | Not available | Available | Cache directory control |
| `--fusion_switch_file` | Basic | Enhanced | More fusion options in 8.5.0+ |

### 3. Environment Variables

**Common variables (all versions):**
```bash
export TE_PARALLEL_COMPILER=8    # Parallel compilation
export DUMP_GE_GRAPH=1           # Graph dumping
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # Verbose logging
```

**8.5.0 specific:**
```bash
export ASCEND_CACHE_PATH=./cache  # Operator cache path
```

### 4. Model Compatibility

Models converted with ATC are generally forward-compatible but not backward-compatible:

- **8.3.RC1 → 8.5.0+:** Models work, but may not use new optimizations
- **8.5.0+ → 8.3.RC1:** May not work if using new features

## Version Detection

### Check Version

```bash
# Method 1: Check version file
cat /usr/local/Ascend/cann/latest/version.cfg 2>/dev/null || \
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null

# Method 2: Check ATC help
atc --help 2>&1 | head -5

# Method 3: Use provided script
./scripts/check_env.sh
```

### Auto-Detection Script

The provided `setup_env.sh` script automatically detects your CANN version and sets up the appropriate environment:

```bash
source ./scripts/setup_env.sh
```

## Best Practices

### For Multi-Version Development

1. **Use version detection:**
   ```bash
   if [ -d "/usr/local/Ascend/cann" ]; then
       source /usr/local/Ascend/cann/set_env.sh
   else
       source /usr/local/Ascend/ascend-toolkit/set_env.sh
   fi
   ```

2. **Document required version:**
   ```bash
   # In your project README
   Required: CANN 8.3.RC1 or 8.5.0+
   Tested on: CANN 8.5.0
   ```

3. **Test on both versions** if supporting multiple versions

4. **Use provided helper scripts** for automatic version handling

## References

- [CANN 8.3.RC1 Documentation](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/atctool/atlasatc_16_0001.html)
- [CANN 8.5.0 Documentation](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatc_16_0001.html)
- [Huawei Ascend Community](https://www.hiascend.com/)
