---
name: v8.3
description: CANN 8.3 installation master skill for Huawei Ascend NPU. Routes to prerequisites, driver/firmware, toolkit installation methods (conda/yum/offline), environment configuration, verification, and troubleshooting. Use when installing or upgrading CANN 8.3 on Atlas A3/A2/910 chips.
---

# CANN 8.3 Installation

CANN (Compute Architecture for Neural Networks) 8.3 installation guide for Ascend NPU.

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
conda install ascend::cann-toolkit

# 3. Configure environment
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh

# 4. Install kernels (choose your chip)
conda install ascend::a3-cann-kernels      # Atlas A3
# OR
conda install ascend::cann-kernels-910b    # Atlas A2 (910B)
# OR
conda install ascend::cann-kernels-910     # Atlas Training (910)
```

## Chip Support Matrix

| Chip Series | Kernels Package | Use Case |
|-------------|-----------------|----------|
| Atlas A3 | `a3-cann-kernels` | Latest inference/training |
| Atlas A2 (910B) | `cann-kernels-910b` | Inference, fine-tuning |
| Atlas Training (910) | `cann-kernels-910` | Training workloads |

## Installation Workflow

```
┌─────────────────┐
│ Prerequisites   │ ← Check OS, Python, Hardware
└────────┬────────┘
         ▼
┌─────────────────┐
│ Driver/Firmware │ ← Install NPU driver first
└────────┬────────┘
         ▼
┌─────────────────┐
│ CANN Toolkit    │ ← Choose: Conda/Yum/Offline
└────────┬────────┘
         ▼
┌─────────────────┐
│ Env Config      │ ← Source set_env.sh
└────────┬────────┘
         ▼
┌─────────────────┐
│ Verification    │ ← npu-smi info
└─────────────────┘
```

## Official References

- [CANN 8.3 Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1)
- [Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)
- [Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html)
