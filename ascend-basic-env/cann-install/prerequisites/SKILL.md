---
name: prerequisites
description: CANN 8.3.RC1 installation prerequisites and system requirements. Use when checking hardware compatibility, OS support, Python versions, or system requirements before CANN installation on Ascend NPU.
---

# CANN Installation Prerequisites

Check system requirements before installing CANN 8.3.RC1.

## Hardware Requirements

| Atlas Series | Chip | Supported Products |
|--------------|------|-------------------|
| Atlas A3 | Ascend 910x | Latest training/inference |
| Atlas A2 | Ascend 910B | Inference, fine-tuning |
| Atlas Training | Ascend 910 | Training workloads |

## Operating Systems

| OS Family | Supported Versions |
|-----------|-------------------|
| openEuler | 20.03, 22.03, 24.03 |
| Ubuntu | 18.04, 20.04, 22.04 |
| Debian | 10, 11, 12 |
| CentOS | 7.6, 8.2 |
| Kylin | V10, V4 |
| BCLinux | 8.2 |
| UOS | V20 |
| veLinux | 1.0 |

## Software Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.7.x - 3.11.4 |
| Architecture | Linux aarch64 (ARM64) |
| NPU Driver | Must be installed first |
| Conda/Miniconda | For conda installation method |

## Pre-Install Checklist

Run these checks before installation:

```bash
# 1. Check OS version
cat /etc/os-release

# 2. Check architecture (must be aarch64)
uname -m

# 3. Check Python version (3.7-3.11)
python3 --version

# 4. Check if NPU driver is installed
npu-smi info
```

## Expected Outputs

### OS Check
```
NAME="openEuler"
VERSION="22.03 (LTS-SP1)"
```

### Architecture Check
```
aarch64
```

### Python Check
```
Python 3.9.18
```

### NPU Driver Check
```
+--------------------------------------------------------------------------------+
| npu-smi 24.1.rc1.3                 Version: 24.1.rc1.3                         |
+----------------------+---------------+------------------------------------------+
| NPU   Name           | Health        | Power(W)           Temp(C)               |
| Chip  Device         | Bus-Id        | AICore(%)          Memory-Usage(MB)      |
+======================+===============+==========================================+
| 0     910B1          | OK            | 67.0               42                    |
...
```

## Dependencies to Install

**Debian/Ubuntu/veLinux:**
```bash
apt-get update
apt-get install -y make dkms gcc linux-headers-$(uname -r)
```

**openEuler/CentOS/RPM-based:**
```bash
yum makecache
yum install -y make dkms gcc kernel-headers-$(uname -r) kernel-devel-$(uname -r)
```

## Official Reference

- [Hardware Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/releasenote/hwcompatibility/hwcompatibility_001.html)
