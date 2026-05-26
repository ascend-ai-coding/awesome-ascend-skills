---
name: external-gitcode-ascend-drivingsdk-ascend-model-migration
description: Ascend NPU model migration suite. Invoke when user wants to migrate/train
  models on Ascend NPU, setup environment, or deploy models from open-source repositories.
original-name: ascend-model-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Ascend Model Migration Suite

Complete model migration and training suite for Ascend NPU environment. This suite provides end-to-end workflow from SSH connection to model training.

## When to Invoke

- User wants to migrate open-source models to Ascend NPU
- User needs to setup training environment on Ascend NPU
- User wants to train models (BEVFormer, etc.) on Ascend NPU
- User asks about model deployment on Ascend platform

## Suite Structure

```
ascend-model-migration/
├── SKILL.md              # Main skill (this file)
├── ssh-connection/       # SSH connection skill (from ssh-dev-suite)
│   └── SKILL.md
├── model-migration/      # Model migration skill
│   └── SKILL.md
└── model-training/       # Model training skill
    └── SKILL.md

# External Dependencies (in AgentSkills/skills/)
└── ascend-mmlab-install-suite/  # MMLab environment setup
    └── SKILL.md
```

## Workflow Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ssh-connection │ ──► │ mmlab-install   │ ──► │ model-migration │ ──► │ model-training  │
│                 │     │    -suite       │     │                 │     │                 │
│  Connect to     │     │  Install mmcv,  │     │  Clone repo,    │     │  Execute        │
│  Ascend server  │     │  mmdet, mmdet3d │     │  apply patches  │     │  training script│
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Complete Workflow

### Phase 0: DrivingSDK Preparation

**DrivingSDK仓库地址**：`https://gitcode.com/Ascend/DrivingSDK`

在开始任何工作前，确认DrivingSDK已克隆到工作目录：
```bash
cd <working_directory>
git clone https://gitcode.com/Ascend/DrivingSDK
```

**重要**：DrivingSDK包含所有必要的patch文件和训练脚本，无需向用户询问patch位置。

### Phase 1: Environment Information Collection

Ask user for the following information:

```
1. Server IP address (e.g., 192.168.13.151)
2. SSH username (e.g., root)
3. Docker container name (if applicable, e.g., qianqian_0312)
4. Conda environment name (e.g., qianqian_2)
5. Working directory path (e.g., /home/zqq/0312_trae_new)
```

Use `ssh-connection` skill (this suite) to establish connection.

### Phase 2: Environment Setup

Use `AgentSkills/skills/ascend-mmlab-install-suite/` to install dependencies:

1. Verify base environment (torch, torch_npu, CANN)
2. Install MMLab dependencies:
   - mmcv-full (with NPU operators)
   - mmdetection
   - mmsegmentation
   - mmdetection3d
   - detectron2

### Phase 3: Model Migration

Use `model-migration` skill (this suite) to prepare model code:

1. Clone open-source repository
2. Apply NPU adaptation patches
3. Setup dataset and weight links

### Phase 4: Model Training

Use `model-training` skill (this suite) to launch training:

1. Select training mode (performance/accuracy)
2. Execute training script
3. Monitor training progress

## Supported Models

| Model | Status | DrivingSDK Path |
|-------|--------|-----------------|
| BEVFormer | ✅ Supported | model_examples/BEVFormer |
| More models | 🔄 Coming soon | - |

## Quick Start Example

```
User: "I want to train BEVFormer on Ascend NPU"

Agent workflow:
1. [ssh-connection] Ask for server info → Connect
2. [ascend-mmlab-install-suite] Check environment → Install MMLab deps
3. [model-migration] Clone BEVFormer → Apply patch → Link data
4. [model-training] Launch training → Monitor progress
```

## Key Principles

### NPU Compatibility Check

After installation, verify NPU compatibility:
```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
```
Should return empty list `[]` for NPU environment.

### Network Strategy

| Situation | Strategy |
|-----------|----------|
| Server can access GitHub | Direct git clone on server |
| Server network timeout | Configure proxy or clone locally → scp to server |
| Proxy available | Configure proxy: `export http_proxy=http://proxy:port` |

**重要**：所有GitHub仓库克隆都应使用GitHub官方链接，不使用镜像站点。如果网络超时，优先配置代理。

### Error Handling

Each sub-skill handles its own errors and provides recovery suggestions. Common issues:

1. **SSH connection failed**: Check IP, username, password/key
2. **NPU device not found**: Check environment variables
3. **CUDA paths conflict**: Reinstall with NPU support
4. **Patch application fails**: Check code version

## Environment Variables Reference

Common Ascend NPU environment variables:
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1200
```

Note: `ASCEND_RT_VISIBLE_DEVICES` may be required in specific environments.

## Reference Files

- **SSH connection**: `ssh-connection/SKILL.md` (this suite)
- **MMLab install suite**: `AgentSkills/skills/ascend-mmlab-install-suite/SKILL.md`
- **DrivingSDK path**: `DrivingSDK/model_examples/`
- **Model migration**: `model-migration/SKILL.md` (this suite)
- **Model training**: `model-training/SKILL.md` (this suite)
