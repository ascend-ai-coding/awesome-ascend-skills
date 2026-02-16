---
name: toolkit-offline
description: Install CANN 8.3 toolkit from offline packages. Use for air-gapped environments, systems without internet access, or when package managers are unavailable.
---

# CANN Toolkit Installation (Offline)

Install CANN toolkit from offline packages for **air-gapped environments**.

## Download Packages

Download from [Ascend Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3):

### Required Packages

| Package | Description |
|---------|-------------|
| `Ascend-cann-toolkit_8.3_linux-aarch64.run` | CANN toolkit |
| `Atlas-A3-cann-kernels_8.3_linux-aarch64.run` | A3 kernels |
| `Ascend-cann-kernels-910b_8.3_linux-aarch64.run` | 910B kernels |
| `Ascend-cann-kernels-910_8.3_linux-aarch64.run` | 910 kernels |

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
chmod +x Ascend-cann-toolkit_8.3_linux-aarch64.run

# Install
./Ascend-cann-toolkit_8.3_linux-aarch64.run --install

# Expected output:
# Install success!
```

## Configure Environment

```bash
# Source environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Install Kernels (Choose Your Chip)

### Atlas A3 Series
```bash
chmod +x Atlas-A3-cann-kernels_8.3_linux-aarch64.run
./Atlas-A3-cann-kernels_8.3_linux-aarch64.run --install
```

### Atlas A2 Series (910B)
```bash
chmod +x Ascend-cann-kernels-910b_8.3_linux-aarch64.run
./Ascend-cann-kernels-910b_8.3_linux-aarch64.run --install
```

### Atlas Training Series (910)
```bash
chmod +x Ascend-cann-kernels-910_8.3_linux-aarch64.run
./Ascend-cann-kernels-910_8.3_linux-aarch64.run --install
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

## Installation Paths

| Component | Default Path |
|-----------|--------------|
| Toolkit | `/usr/local/Ascend/ascend-toolkit/` |
| Kernels | `/usr/local/Ascend/ascend-toolkit/latest/opp/` |
| set_env.sh | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |

## Official Reference

- [Offline Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0012.html)
- [Download Center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3)
