---
name: env-config
description: Configure CANN 8.5.0 environment variables after installation. Use when setting up PATH, LD_LIBRARY_PATH, ASCEND_TOOLKIT_HOME, or persisting CANN environment via set_env.sh.
---

# Environment Configuration

Configure CANN 8.5.0 environment variables after installation.

## Source Environment Script

### For Conda Installation
```bash
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh
```

### For Yum/Offline Installation
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `ASCEND_TOOLKIT_HOME` | CANN toolkit installation path |
| `ASCEND_AICPU_PATH` | AICPU operator path |
| `ASCEND_OPP_PATH` | Operator development path |
| `PATH` | Includes CANN binaries |
| `LD_LIBRARY_PATH` | CANN shared libraries |

## Verify Environment

```bash
# Check key variables
echo $ASCEND_TOOLKIT_HOME
echo $PATH | grep Ascend
echo $LD_LIBRARY_PATH | grep Ascend
```

**Expected output:**
```
/usr/local/Ascend/ascend-toolkit/latest
.../usr/local/Ascend/ascend-toolkit/latest/bin:...
.../usr/local/Ascend/ascend-toolkit/latest/lib:...
```

## Make Environment Persistent

### For Current User
```bash
# Add to bashrc
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

# Reload
source ~/.bashrc
```

### For All Users
```bash
# Add to profile
sudo bash -c 'echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /etc/profile'

# Reload
source /etc/profile
```

## Conda Activation Hook

For automatic activation with Conda environment:

```bash
# Create activation script
mkdir -p ~/.conda/envs/myenv/etc/conda/activate.d
cat > ~/.conda/envs/myenv/etc/conda/activate.d/ascend.sh << 'EOF'
source /home/miniconda3/Ascend/ascend-toolkit/set_env.sh
EOF
```

## Custom Environment Setup

```bash
# Manual environment setup (if set_env.sh unavailable)
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PATH=$ASCEND_TOOLKIT_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib:$LD_LIBRARY_PATH
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Command not found | Source set_env.sh |
| Library not found | Check LD_LIBRARY_PATH |
| Wrong version loaded | Check PATH order, remove old paths |

## Related Skills

- [verification/](../verification/SKILL.md) - Verify installation
- [troubleshooting/](../troubleshooting/SKILL.md) - Debug environment issues

## Official Reference

- [Environment Configuration](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)
