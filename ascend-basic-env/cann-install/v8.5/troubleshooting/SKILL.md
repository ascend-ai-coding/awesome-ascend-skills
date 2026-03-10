---
name: troubleshooting
description: CANN 8.5.0 installation troubleshooting guide. Use when resolving installation errors, DKMS failures, missing tools, kernel header issues, HwHiAiUser creation, or permission problems on Ascend NPU.
---

# Installation Troubleshooting

Common issues and solutions for CANN 8.5.0 installation.

## Missing Tools Error

**Error:**
```
[ERROR] The list of missing tools: lspci, ifconfig
```

**Solution:**
```bash
# Debian/Ubuntu/veLinux
apt-get install -y net-tools pciutils

# RPM-based systems
yum install -y net-tools pciutils
```

## DKMS Install Failure

**Error:**
```
[ERROR] Dkms install failed, details in: /var/log/ascend_seclog/ascend_install.log
[ERROR] Driver_ko_install failed
```

**Solution 1 - Clean DKMS:**
```bash
cd /var/lib/dkms
rm -rf davinci_ascend
# Reinstall driver
```

**Solution 2 - GCC Version Mismatch:**
```bash
# Backup old GCC
mv /usr/bin/gcc /usr/bin/gcc.bak
mv /usr/bin/g++ /usr/bin/g++.bak

# Link to correct version (e.g., 7.3.0)
ln -s /usr/local/gcc7.3.0/bin/gcc /usr/bin/gcc
ln -s /usr/local/gcc7.3.0/bin/g++ /usr/bin/g++

# Reinstall driver
```

## HwHiAiUser Creation Fails

**Error:**
```
groupadd: GID '1000' already exists
```

**Solution:**
```bash
# Find user with UID 1000
cat /etc/passwd | grep 1000

# Change existing user UID (example to 1002)
usermod -u 1002 <existing_user>
groupmod -g 1002 <existing_user>

# Create HwHiAiUser
groupadd -g 1000 HwHiAiUser
useradd -g HwHiAiUser -u 1000 -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

## Conda Permission Error (8.5.0 New)

**Error:** Installation fails due to permission issues

**Solution:**
```bash
# Ensure Conda directory and all parents have 755 permissions
chmod 755 /home
chmod 755 /home/miniconda3
chmod 755 /home/miniconda3/envs
```

## Kernel Headers Not Found

**Solution:**
```bash
# Install matching kernel headers
yum install -y kernel-headers-$(uname -r) kernel-devel-$(uname -r)
```

## Upgrade from Previous Version

**Error:** Installation fails when upgrading from older CANN

**Solution:**
```bash
# Must use upgrade parameter or uninstall first
# Option 1: Use upgrade parameter
./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --upgrade

# Option 2: Uninstall old version first
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --uninstall
# Then install new version
./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
```

## Environment Variables Not Set

**Solution:**
```bash
# Source environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Make persistent
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
```

## NPU Not Detected

**Check driver:**
```bash
npu-smi info
```

**If no devices shown:**
1. Verify driver is installed: `rpm -qa | grep ascend`
2. Check kernel module: `lsmod | grep davinci`
3. Reinstall driver if needed

## Log Files

| Log | Path |
|-----|------|
| Installation log | `/var/log/ascend_seclog/ascend_install.log` |
| Driver log | `/var/log/npu/` |

## Official Reference

- [Troubleshooting Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)
