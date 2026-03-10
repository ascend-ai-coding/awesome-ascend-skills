---
name: toolkit-offline
description: Install CANN 8.5.0 toolkit from offline packages. Use for air-gapped environments, systems without internet access, or when package managers are unavailable.
---

# CANN Toolkit Installation (Offline)

Install CANN 8.5.0 toolkit from offline packages for **air-gapped environments**.

## Download Packages

Download from [Ascend Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0):

### Required Packages

| Package | Description |
|---------|-------------|
| `Ascend-cann-toolkit_8.5.0_linux-aarch64.run` | CANN toolkit |
| `Ascend-cann-A3-ops_8.5.0_linux-aarch64.run` | A3 ops |
| `Ascend-cann-910b-ops_8.5.0_linux-aarch64.run` | 910B ops |
| `Ascend-cann-910-ops_8.5.0_linux-aarch64.run` | 910 ops |

## Direct Download URLs

```bash
# Toolkit
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-aarch64.run

# A3 ops
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run

# 910B ops
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run

# 910 ops
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910-ops_8.5.0_linux-aarch64.run
```

## Transfer Files

```bash
# Upload to server (example using scp)
scp Ascend-cann-*.run user@server:/home/

# Or use USB/physical media for air-gapped systems
```

## Install Toolkit

```bash
cd /home

# Add execute permission
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run

# Install
bash ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install

# Expected output:
# Install success!
```

## Configure Environment

```bash
# Source environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Install Ops (Choose Your Chip)

### Atlas A3 Series
```bash
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
```

### Atlas A2 Series (910B)
```bash
chmod +x Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
```

### Atlas Training Series (910)
```bash
chmod +x Ascend-cann-910-ops_8.5.0_linux-aarch64.run
bash ./Ascend-cann-910-ops_8.5.0_linux-aarch64.run --install
```

## Installation Options

```bash
# Install to custom path
./Ascend-cann-toolkit_*.run --install --install-path=/opt/ascend

# Silent install (no prompts)
./Ascend-cann-toolkit_*.run --install --quiet

# Verify package integrity
./Ascend-cann-toolkit_*.run --check
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
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

## Official Reference

- [Offline Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)
- [Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)
