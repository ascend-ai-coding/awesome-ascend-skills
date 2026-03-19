---
name: toolkit-yum
description: Install CANN 8.5.0 toolkit via Yum/APT package manager. Use for system-wide CANN installation on openEuler, CentOS, and other RPM-based Linux distributions.
---

# CANN Toolkit Installation (Yum)

Install CANN 8.5.0 toolkit using Yum package manager for **system-wide installation**.

## Prerequisites

- NPU driver installed (see [driver-firmware](../driver-firmware/SKILL.md))
- Root/sudo privileges
- Internet access

## Add Repository

```bash
# Add Ascend repository
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo

# Update cache
yum makecache
```

## Install CANN Toolkit

```bash
# Install toolkit 8.5.0
sudo yum install -y Ascend-cann-toolkit-8.5.0
```

## Configure Environment

```bash
# Source environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Install Ops (Choose Your Chip)

### Atlas A3 Series
```bash
sudo yum install -y Ascend-cann-A3-ops-8.5.0
```

### Atlas A2 Series (910B)
```bash
sudo yum install -y Ascend-cann-910b-ops-8.5.0
```

### Atlas Training Series (910)
```bash
sudo yum install -y Ascend-cann-910-ops-8.5.0
```

## Quick Reference Commands

```bash
# Full installation (A3)
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo
yum makecache
sudo yum install -y Ascend-cann-toolkit-8.5.0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
sudo yum install -y Ascend-cann-A3-ops-8.5.0

# Full installation (910B)
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo
yum makecache
sudo yum install -y Ascend-cann-toolkit-8.5.0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
sudo yum install -y Ascend-cann-910b-ops-8.5.0
```

## List Available Packages

```bash
# List all CANN packages
yum list | grep -i ascend

# Search specific package
yum search Ascend-cann
```

## Verify Installation

```bash
# Check CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.info

# Verify environment
echo $ASCEND_TOOLKIT_HOME
```

## Make Environment Persistent

```bash
# Add to bashrc
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

## Installation Paths

| Component | Path |
|-----------|------|
| Toolkit | `/usr/local/Ascend/ascend-toolkit/` |
| Ops | `/usr/local/Ascend/ascend-toolkit/latest/opp/` |
| set_env.sh | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |

## Official Reference

- [Yum Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)
