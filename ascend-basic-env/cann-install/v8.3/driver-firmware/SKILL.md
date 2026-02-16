---
name: driver-firmware
description: Ascend NPU driver and firmware installation for CANN 8.3. Use when installing or upgrading NPU drivers, firmware, creating HwHiAiUser, or resolving driver-related issues.
---

# NPU Driver & Firmware Installation

Install NPU driver and firmware before CANN toolkit.

## Create HwHiAiUser

The default running user with UID/GID 1000:

```bash
# Create user group
groupadd HwHiAiUser

# Create user
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

> **Note:** If UID/GID 1000 is occupied, see [troubleshooting](../troubleshooting/SKILL.md).

## Install Dependencies

**Debian/Ubuntu/veLinux:**
```bash
apt-get update
apt-get install -y make dkms gcc linux-headers-$(uname -r) net-tools pciutils
```

**openEuler/CentOS/Kylin/RPM-based:**
```bash
yum makecache
yum install -y make dkms gcc kernel-headers-$(uname -r) kernel-devel-$(uname -r) net-tools pciutils
```

**CentOS 8.2 x86 (additional):**
```bash
yum install -y elfutils-libelf-devel
```

## Download Packages

Download from [Ascend Download Center](https://www.hiascend.com/developer/download/community/result?module=driver):

- `Ascend-hdk-{chip}-npu-driver_{version}_linux-aarch64.run`
- `Ascend-hdk-{chip}-npu-firmware_{version}.run`

Upload to `/home/` directory.

## Installation Order

| Scenario | Order |
|----------|-------|
| First-time install | Driver → Firmware |
| Reinstall/Upgrade | Firmware → Driver |

## Install Commands

```bash
cd /home

# Add execution permissions
chmod +x Ascend-hdk-*-npu-*.run

# Install driver (first-time)
./Ascend-hdk-{chip}-npu-driver_{version}_linux-aarch64.run --full --install-for-all

# Expected output:
# Driver package installed successfully!

# Install firmware
./Ascend-hdk-{chip}-npu-firmware_{version}.run --full

# Expected output:
# Firmware package installed successfully! Reboot now...
```

## Post-Install

```bash
# Reboot if prompted
reboot

# Verify after reboot
npu-smi info
```

## Verify Driver

```bash
npu-smi info
```

**Expected output:**
```
+--------------------------------------------------------------------------------+
| npu-smi 24.1.rc1.3                 Version: 24.1.rc1.3                         |
+----------------------+---------------+------------------------------------------+
| NPU   Name           | Health        | Power(W)           Temp(C)               |
+======================+===============+==========================================+
| 0     910B1          | OK            | 67.0               42                    |
+----------------------+---------------+------------------------------------------+
```

## Common Issues

| Issue | Solution |
|-------|----------|
| DKMS failure | Clean `/var/lib/dkms/davinci_ascend` and reinstall |
| GCC version mismatch | Link to correct GCC version |
| Kernel headers missing | Install matching kernel-devel package |
| Missing tools (lspci) | Install pciutils, net-tools |

See [troubleshooting](../troubleshooting/SKILL.md) for detailed solutions.

## Official Reference

- [Driver Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0002.html)
