# SSH Tool Comparison and Selection Guide

## Overview

This guide helps you choose the right SSH tool for password authentication based on your use case and environment.

## Quick Selection

| Your Situation | Recommended Tool |
|----------------|------------------|
| Writing a quick bash script | **sshpass** |
| Building a Python application | **paramiko** |
| Creating deployment automation | **fabric** |
| Need interactive terminal | **sshpass** |
| Need advanced SFTP operations | **paramiko** |
| Want clean, readable code | **fabric** |
| No Python available | **sshpass** |

## Tool Comparison Matrix

### Features

| Feature | sshpass | paramiko | fabric |
|---------|---------|----------|--------|
| Command execution | Yes | Yes | Yes |
| Interactive TTY | Yes | Limited | Yes |
| File upload (SCP/SFTP) | Yes | Yes | Yes |
| File download | Yes | Yes | Yes |
| Jump host / Gateway | Yes | Yes | Yes |
| Parallel execution | Manual | Yes | Yes |
| Timeout handling | Basic | Yes | Yes |
| Error handling | Exit codes only | Exceptions | Rich |
| Progress callbacks | No | Yes | Yes |
| SSH config support | Yes | Yes | Yes |
| Proxy support | Yes | Yes | Yes |
| Shell escape safety | Manual | Yes | Yes |

### Usability

| Aspect | sshpass | paramiko | fabric |
|--------|---------|----------|--------|
| Installation difficulty | Easy (package manager) | Easy (pip) | Easy (pip) |
| Learning curve | Very low | Medium | Low |
| Code verbosity | Low | High | Medium |
| Documentation | Good | Excellent | Excellent |
| Community support | High | Very high | High |
| Maintenance | Active | Active | Active |

### Performance

| Aspect | sshpass | paramiko | fabric |
|--------|---------|----------|--------|
| Connection overhead | Low | Medium | Medium |
| Command execution | Fast | Fast | Fast |
| File transfer | Fast (native scp) | Good (SFTP) | Good |
| Memory usage | Very low | Medium | Medium |
| CPU usage | Low | Medium | Medium |

### Security

| Aspect | sshpass | paramiko | fabric |
|--------|---------|----------|--------|
| Password in process list | Yes (with -p) | No | No |
| Environment variable | SSHPASS supported | N/A | N/A |
| Password file | Supported | N/A | N/A |
| Key-based auth fallback | Yes | Yes | Yes |
| Host key verification | Yes | Yes | Yes |

## Detailed Tool Descriptions

### sshpass

**What it is:** A command-line utility that non-interactively provides password to SSH.

**Strengths:**
- Works with existing SSH/SCP commands
- No Python required
- Very simple to use
- Works in any shell script
- Native TTY support

**Weaknesses:**
- Password visible in process list (with `-p` flag)
- Limited error handling
- No built-in parallel execution
- Shell injection risk if not careful

**Best for:**
- Quick one-off commands
- Shell scripts
- CI/CD pipelines
- When Python is not available

**Detailed documentation:** [tools/sshpass.md](tools/sshpass.md)

### paramiko

**What it is:** A pure-Python implementation of SSHv2 protocol.

**Strengths:**
- Full control over SSH operations
- Excellent SFTP support
- Good for complex workflows
- Pure Python (no external dependencies)
- Detailed exception handling
- Can be used in async code

**Weaknesses:**
- More verbose code
- No high-level abstractions
- Interactive TTY is limited
- Steeper learning curve

**Best for:**
- Python applications
- Complex SSH workflows
- When you need fine-grained control
- SFTP-heavy operations

**Detailed documentation:** [tools/paramiko-basic.md](tools/paramiko-basic.md), [tools/paramiko-advanced.md](tools/paramiko-advanced.md)

### fabric

**What it is:** A high-level Python library for remote execution, built on paramiko.

**Strengths:**
- Clean, readable API
- Built-in context managers (cd, prefix, etc.)
- Good error handling
- Task-based CLI support
- Gateway/jump host made easy
- Great for deployment scripts

**Weaknesses:**
- Larger dependency footprint
- Less control than raw paramiko
- Overkill for simple tasks
- Changes between versions

**Best for:**
- Deployment automation
- Configuration management
- Multi-host operations
- When you want clean code

**Detailed documentation:** [tools/fabric-basic.md](tools/fabric-basic.md), [tools/fabric-advanced.md](tools/fabric-advanced.md)

## Environment Detection

### Check Available Tools

```bash
# Check sshpass
if command -v sshpass &> /dev/null; then
    echo "sshpass: available"
else
    echo "sshpass: not installed"
fi

# Check paramiko
if python3 -c "import paramiko" 2>/dev/null; then
    echo "paramiko: available"
else
    echo "paramiko: not installed"
fi

# Check fabric
if python3 -c "import fabric" 2>/dev/null; then
    echo "fabric: available"
else
    echo "fabric: not installed"
fi

# Check SSH keys
if [ -f ~/.ssh/id_rsa ]; then
    echo "SSH key: available"
else
    echo "SSH key: not found"
fi
```

## Installation Guide

### sshpass

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y sshpass

# CentOS/RHEL/AlmaLinux
sudo yum install -y epel-release && sudo yum install -y sshpass

# Fedora
sudo dnf install -y sshpass

# macOS
brew install hudochenkov/sshpass/sshpass

# Alpine
apk add sshpass

# Arch Linux
sudo pacman -S sshpass
```

### paramiko

```bash
pip install paramiko
```

### fabric

```bash
pip install fabric
```

## Decision Tree

```
Need password auth?
├── Is Python available?
│   ├── Yes
│   │   ├── Need deployment automation? → fabric
│   │   ├── Need full control? → paramiko
│   │   └── Want simple code? → fabric
│   └── No
│       └── Use sshpass
└── Prefer SSH keys?
    └── Use native ssh (no tool needed)
```
