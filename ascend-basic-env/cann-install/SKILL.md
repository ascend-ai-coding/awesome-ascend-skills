---
name: cann-install
description: CANN installation master skill for Huawei Ascend NPU. Routes to version-specific installation guides for CANN 8.3 and 8.5. Use when installing, upgrading, or troubleshooting CANN installation on Atlas A3/A2/910 chips.
---

# CANN Installation

CANN (Compute Architecture for Neural Networks) installation guides for Ascend NPU.

## Version Selection

| Version | Status | Sub-Skill |
|---------|--------|-----------|
| **8.5.0** | Latest | [v8.5/](v8.5/SKILL.md) |
| **8.3.RC1** | Stable | [v8.3/](v8.3/SKILL.md) |

## Quick Start (Recommended: 8.5.0)

```bash
# 1. Add Conda channel
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/

# 2. Install CANN toolkit
conda install ascend::cann-toolkit==8.5.0

# 3. Configure environment
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh

# 4. Install ops (choose your chip)
conda install ascend::cann-a3-ops==8.5.0     # Atlas A3
# OR
conda install ascend::cann-910b-ops==8.5.0   # Atlas A2 (910B)
# OR
conda install ascend::cann-910-ops==8.5.0    # Atlas Training (910)
```

## Version Comparison

| Feature | 8.3.RC1 | 8.5.0 |
|---------|---------|-------|
| Python Support | 3.7-3.11.4 | **3.7-3.13.x** |
| New OS | - | **vesselOS** |
| Package Naming | `*-cann-kernels` | `cann-*-ops` |
| Conda Permissions | N/A | **Requires 755** |

## Chip Support

| Chip Series | 8.3 Package | 8.5 Package |
|-------------|-------------|-------------|
| Atlas A3 | `a3-cann-kernels` | `cann-a3-ops` |
| Atlas A2 (910B) | `cann-kernels-910b` | `cann-910b-ops` |
| Atlas Training (910) | `cann-kernels-910` | `cann-910-ops` |

## Installation Workflow

1. **Check Prerequisites** → Hardware, OS, Python version
2. **Install Driver & Firmware** → NPU driver must be installed first
3. **Install CANN Toolkit** → Choose: Conda/Yum/Offline
4. **Configure Environment** → Source set_env.sh
5. **Verify Installation** → Run npu-smi info

## Official References

- [CANN Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition)
- [Download Center](https://www.hiascend.com/developer/download/community/result?module=cann)
