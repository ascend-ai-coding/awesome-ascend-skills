---
name: verification
description: Verify CANN 8.3 installation and environment on Ascend NPU. Use when checking if CANN, drivers, and NPU devices are correctly installed and configured.
---

# Installation Verification

Verify CANN installation is working correctly.

## Verify NPU Driver

```bash
npu-smi info
```

**Expected output:**
```
+--------------------------------------------------------------------------------+
| npu-smi 24.1.rc1.3                 Version: 24.1.rc1.3                         |
+----------------------+---------------+------------------------------------------+
| NPU   Name           | Health        | Power(W)           Temp(C)               |
| Chip  Device         | Bus-Id        | AICore(%)          Memory-Usage(MB)      |
+======================+===============+==========================================+
| 0     910B1          | OK            | 67.0               42                    |
+----------------------+---------------+------------------------------------------+
```

## Verify CANN Version

### For Conda Installation
```bash
cat /home/miniconda3/Ascend/ascend-toolkit/latest/version.info
```

### For Yum/Offline Installation
```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.info
```

**Expected output:**
```
Version=8.3
```

## Verify Environment Variables

```bash
# Check ASCEND_TOOLKIT_HOME
echo $ASCEND_TOOLKIT_HOME

# Check PATH includes Ascend
echo $PATH | grep Ascend

# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | grep Ascend
```

## Verify NPU Detection

```bash
# List all NPU devices
npu-smi info -l

# Check specific NPU
npu-smi info -t board -i 0
```

## Test Python Import

```bash
# Test torch-npu import (if installed)
python3 -c "import torch; import torch_npu; print(torch_npu.__version__)"

# Test CANN availability
python3 -c "import torch; print(torch.npu.is_available())"
```

**Expected output:**
```
True
```

## Run Sample Program

```bash
# Navigate to samples directory
cd /usr/local/Ascend/ascend-toolkit/latest/samples

# Or for Conda
cd /home/miniconda3/Ascend/ascend-toolkit/latest/samples

# Run a simple test
cd operator/AddCustomSample
# Follow README to build and run
```

## Quick Verification Script

```bash
#!/bin/bash
echo "=== CANN Verification ==="

echo -e "\n[1] NPU Driver:"
npu-smi info | head -8

echo -e "\n[2] CANN Version:"
cat /usr/local/Ascend/ascend-toolkit/latest/version.info 2>/dev/null || \
cat /home/miniconda3/Ascend/ascend-toolkit/latest/version.info 2>/dev/null

echo -e "\n[3] Environment:"
echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"

echo -e "\n[4] Python Test:"
python3 -c "import torch; print(f'torch.npu.available: {torch.npu.is_available()}')" 2>/dev/null || echo "torch_npu not installed"

echo -e "\n=== Verification Complete ==="
```

## Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| Driver installed | `npu-smi info` | NPU info displayed |
| CANN version | `cat .../version.info` | Version=8.3 |
| Env vars set | `echo $ASCEND_TOOLKIT_HOME` | Path displayed |
| NPU available | `torch.npu.is_available()` | True |

## Official Reference

- [Verification Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0014.html)
