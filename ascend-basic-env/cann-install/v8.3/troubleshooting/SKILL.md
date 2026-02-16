---
name: troubleshooting
description: CANN 8.3 installation troubleshooting guide. Use when resolving installation errors, DKMS failures, missing tools, kernel header issues, HwHiAiUser creation, or permission problems on Ascend NPU.
---

# Installation Troubleshooting

Common issues and solutions for CANN installation.

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

## DKMS Package Missing (RPM Systems)

**Solution:**
```bash
# Download DKMS
wget https://mirrors.huaweicloud.com/epel/7/aarch64/Packages/d/dkms-2.7.1-1.el7.noarch.rpm

# Install
rpm -ivh dkms-2.7.1-1.el7.noarch.rpm

# If dependency error:
yum install -y elfutils-libelf-devel
rpm -ivh dkms-2.7.1-1.el7.noarch.rpm
```

## OpenSSL-devel Conflict (openEuler)

**Error:** SO file conflict between openssl-libs and openssl-SMx-libs

**Solution:**
```bash
mkdir openDown && cd openDown
yum install --downloadonly --downloaddir=. openssl-devel
rpm -Uvh openssl-devel-*.rpm --force
```

## Kernel Headers Not Found

### CentOS 7.6 x86
```bash
wget https://mirrors.bfsu.edu.cn/centos-vault/7.6.1810/os/x86_64/Packages/kernel-headers-3.10.0-957.el7.x86_64.rpm
wget https://mirrors.bfsu.edu.cn/centos-vault/7.6.1810/os/x86_64/Packages/kernel-devel-3.10.0-957.el7.x86_64.rpm
rpm -ivh kernel-headers-*.rpm kernel-devel-*.rpm
```

### CentOS 7.6 ARM
```bash
wget https://mirrors.bfsu.edu.cn/centos-vault/altarch/7.6.1810/os/aarch64/Packages/kernel-headers-4.14.0-115.el7a.0.1.aarch64.rpm
wget https://mirrors.bfsu.edu.cn/centos-vault/altarch/7.6.1810/os/aarch64/Packages/kernel-devel-4.14.0-115.el7a.0.1.aarch64.rpm
rpm -ivh kernel-headers-*.rpm kernel-devel-*.rpm
```

## Environment Variables Not Set

**Symptoms:** Commands not found, libraries not loaded

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

- [Troubleshooting Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0020.html)