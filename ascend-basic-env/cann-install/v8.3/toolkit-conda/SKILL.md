---
name: toolkit-conda
description: Install CANN 8.3 toolkit via Conda environment. Use when setting up isolated CANN environments with Conda for Atlas A3/A2/910 chips. Recommended for development.
---

# CANN Toolkit Installation (Conda)

Install CANN toolkit using Conda - **recommended for development**.

## Prerequisites

- Miniconda/Anaconda installed
- NPU driver installed (see [driver-firmware](../driver-firmware/SKILL.md))
- Internet access to Huawei mirror

## Add Conda Channel

```bash
# Add Huawei Ascend channel
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/

# Verify channel added
conda config --show channels
```

## Install CANN Toolkit

```bash
# Install CANN toolkit
conda install ascend::cann-toolkit
```

## Configure Environment

```bash
# Source environment (Conda path)
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh

# Or if using Anaconda
source /home/anaconda3/Ascend/ascend-toolkit/set_env.sh
```

## Install Kernels (Choose Your Chip)

### Atlas A3 Series
```bash
conda install ascend::a3-cann-kernels
```

### Atlas A2 Series (910B)
```bash
conda install ascend::cann-kernels-910b
```

### Atlas Training Series (910)
```bash
conda install ascend::cann-kernels-910
```

## Quick Reference Commands

```bash
# Full installation (A3)
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh
conda install ascend::a3-cann-kernels

# Full installation (910B)
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh
conda install ascend::cann-kernels-910b
```

## Verify Installation

```bash
# Check CANN version
cat /home/miniconda3/Ascend/ascend-toolkit/latest/version.info

# Verify environment
echo $ASCEND_TOOLKIT_HOME
```

## Make Environment Persistent

```bash
# Add to bashrc
echo "source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

## Installation Paths

| Component | Path |
|-----------|------|
| Toolkit | `/home/miniconda3/Ascend/ascend-toolkit/` |
| Kernels | `/home/miniconda3/Ascend/ascend-toolkit/latest/opp/` |
| set_env.sh | `/home/miniconda3/Ascend/ascend-toolkit/set_env.sh` |

## Official Reference

- [Conda Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0010.html)
