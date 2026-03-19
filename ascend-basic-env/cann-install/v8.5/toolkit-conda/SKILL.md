---
name: toolkit-conda
description: Install CANN 8.5.0 toolkit via Conda environment. Use when setting up isolated CANN environments with Conda for Atlas A3/A2/910 chips. Recommended for development.
---

# CANN Toolkit Installation (Conda)

Install CANN 8.5.0 toolkit using Conda - **recommended for development**.

## Prerequisites

- Miniconda/Anaconda installed
- NPU driver installed (see [driver-firmware](../driver-firmware/SKILL.md))
- Internet access to Huawei mirror
- **Directory permissions 755** for Conda paths

## Conda Directory Permissions

```bash
# Ensure Conda directory and parents have 755 permissions
chmod 755 /home/miniconda3
chmod 755 /home/miniconda3/envs  # if using custom envs
```

## Add Conda Channel

```bash
# Add Huawei Ascend channel
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/

# Verify channel added
conda config --show channels
```

## Install CANN Toolkit

```bash
# Install CANN toolkit 8.5.0
conda install ascend::cann-toolkit==8.5.0
```

## Configure Environment

```bash
# Source environment (Conda path)
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh

# Or if using Anaconda
source /home/anaconda3/Ascend/ascend-toolkit/set_env.sh
```

## Install Ops (Choose Your Chip)

### Atlas A3 Series
```bash
conda install ascend::cann-a3-ops==8.5.0
```

### Atlas A2 Series (910B)
```bash
conda install ascend::cann-910b-ops==8.5.0
```

### Atlas Training Series (910)
```bash
conda install ascend::cann-910-ops==8.5.0
```

## Quick Reference Commands

```bash
# Full installation (A3)
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit==8.5.0
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh
conda install ascend::cann-a3-ops==8.5.0

# Full installation (910B)
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit==8.5.0
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh
conda install ascend::cann-910b-ops==8.5.0
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

## Official Reference

- [Conda Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html)
