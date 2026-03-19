---
name: v8.5
description: CANN 8.5.0 installation master skill for Huawei Ascend NPU. Routes to prerequisites, driver/firmware, toolkit installation methods (conda/yum/offline), environment configuration, verification, and troubleshooting. Use when installing or upgrading CANN 8.5.0 on Atlas A3/A2/910 chips.
---

# CANN 8.5.0 Installation

CANN (Compute Architecture for Neural Networks) 8.5.0 installation guide for Ascend NPU.

## Quick Navigation

| Step | Task | Sub-Skill |
|------|------|-----------|
| 1 | Check Prerequisites | [prerequisites/](prerequisites/SKILL.md) |
| 2 | Install Driver & Firmware | [driver-firmware/](driver-firmware/SKILL.md) |
| 3 | Install Toolkit (Choose One) | See below |
| 4 | Configure Environment | [env-config/](env-config/SKILL.md) |
| 5 | Verify Installation | [verification/](verification/SKILL.md) |
| - | Troubleshooting | [troubleshooting/](troubleshooting/SKILL.md) |

## Toolkit Installation Methods

| Method | Sub-Skill | Best For |
|--------|-----------|----------|
| Conda (Recommended) | [toolkit-conda/](toolkit-conda/SKILL.md) | Development, isolated environments |
| Yum/APT | [toolkit-yum/](toolkit-yum/SKILL.md) | System-wide, production servers |
| Offline | [toolkit-offline/](toolkit-offline/SKILL.md) | Air-gapped systems, no internet |

## Quick Start (Conda Method)

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

## What's New in 8.5.0

| Feature | 8.3.RC1 | 8.5.0 |
|---------|---------|-------|
| Python Support | 3.7-3.11.4 | **3.7-3.13.x** |
| New OS | - | **vesselOS** |
| Package Naming | `*-cann-kernels` | `cann-*-ops` |
| Conda Permissions | N/A | **Requires 755** |

## Chip Support Matrix

| Chip Series | Ops Package | Use Case |
|-------------|-------------|----------|
| Atlas A3 | `cann-a3-ops` | Latest inference/training |
| Atlas A2 (910B) | `cann-910b-ops` | Inference, fine-tuning |
| Atlas Training (910) | `cann-910-ops` | Training workloads |

## Official References

- [CANN 8.5.0 Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850)
- [Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)
- [Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)
