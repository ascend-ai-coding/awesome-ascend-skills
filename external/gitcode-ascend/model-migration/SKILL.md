---
name: external-gitcode-ascend-model-migration
description: Model code migration for Ascend NPU. Invoke when user needs to clone
  open-source repo and apply NPU adaptation patches.
original-name: model-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Model Migration Skill

Clone open-source model repositories and apply NPU adaptation patches for Ascend NPU training.

## When to Invoke

- User wants to migrate open-source model to Ascend NPU
- User needs to clone model repository
- User needs to apply DrivingSDK patches
- User asks about preparing model code for NPU training

## Information to Collect

Ask user for the following:

```
1. Model name (e.g., BEVFormer)
2. Working directory path on server
3. Dataset path (if already prepared)
4. Pre-trained weights path (if available)
```

**重要**：DrivingSDK仓库地址为 `https://gitcode.com/Ascend/DrivingSDK`，包含所有必要的patch文件和训练脚本。

## Workflow

### Step 1: Clone Repository in DrivingSDK model_examples Directory

**重要**：必须在 `DrivingSDK/model_examples/<model_name>/` 目录下克隆模型仓库，而不是在外部创建新目录。

```bash
# 进入DrivingSDK的model_examples目录
cd <DrivingSDK_path>/model_examples/BEVFormer

# 克隆模型仓库（会在当前目录下创建BEVFormer子目录）
git clone https://github.com/fundamentalvision/BEVFormer.git
```

### Step 2: Copy Patch to Model Directory

```bash
# 将patch文件复制到克隆的模型目录中
cp bev_former_config.patch BEVFormer/
```

### Step 3: Checkout Specific Commit

```bash
cd BEVFormer
git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
```

### Step 4: Apply NPU Patches

```bash
# 在模型目录内应用patch
git apply --reject --whitespace=fix bev_former_config.patch
```

### Step 5: Setup Dataset and Weight Links

```bash
# 创建数据和权重目录
mkdir -p data ckpts

# 设置数据集软链接
ln -sf <dataset_path>/nuscenes data/nuscenes
ln -sf <dataset_path>/can_bus data/can_bus

# 设置预训练权重软链接
ln -sf <weight_path>/r101_dcn_fcos3d_pretrain.pth ckpts/
```

### Step 6: Return to Parent Directory for Training

```bash
# 返回到model_examples/BEVFormer目录
cd ..

# 执行训练脚本
bash test/train_performance_8p_base_fp32.sh --batch-size=1 --num-npu=8
```

## Supported Models

### BEVFormer

**Repository**: `https://github.com/fundamentalvision/BEVFormer.git`

**Commit**: `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`

**Patch**: `bev_former_config.patch`

**Dataset Required**:
- nuscenes (processed)
- can_bus

**Weights Required**:
- `r101_dcn_fcos3d_pretrain.pth`

**Commands**:
```bash
# 进入DrivingSDK的model_examples/BEVFormer目录
cd <DrivingSDK_path>/model_examples/BEVFormer

# 克隆BEVFormer仓库
git clone https://github.com/fundamentalvision/BEVFormer.git

# 复制patch到BEVFormer目录
cp bev_former_config.patch BEVFormer/

# 进入BEVFormer目录
cd BEVFormer

# 切换到指定commit
git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c

# 应用patch
git apply --reject --whitespace=fix bev_former_config.patch

# 创建数据和权重目录
mkdir -p data ckpts

# 设置数据集软链接
ln -sf <dataset_path>/nuscenes data/nuscenes
ln -sf <dataset_path>/can_bus data/can_bus

# 设置预训练权重软链接
ln -sf <weight_path>/r101_dcn_fcos3d_pretrain.pth ckpts/

# 返回上级目录执行训练
cd ..
bash test/train_performance_8p_base_fp32.sh --batch-size=1 --num-npu=8
```

## Network Strategy

| Situation | Strategy |
|-----------|----------|
| Server can access GitHub | Direct git clone on server |
| Server network timeout | Configure proxy or clone locally → scp to server |
| Proxy available | Configure proxy: `export http_proxy=http://proxy:port` |

**重要**：所有GitHub仓库克隆都应使用GitHub官方链接，不使用镜像站点。如果网络超时，优先配置代理。

### Local Clone + SCP Transfer

If server cannot access GitHub:

```bash
# On local machine
git clone https://github.com/fundamentalvision/BEVFormer.git
cd BEVFormer
git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c

# Transfer to server
scp -r BEVFormer <user>@<server>:<working_directory>/
```

## Verification

After migration, verify setup:

```bash
# Check directory structure
ls -la <model_directory>/

# Check patch applied
cd <model_directory> && git status

# Check data links
ls -la data/
ls -la ckpts/
```

## Troubleshooting

### Patch Application Fails

1. Check if correct commit is checked out
2. Check for `.rej` files for manual merge
3. Verify patch file is for correct model version

### Dataset Not Found

1. Verify dataset path exists
2. Check symbolic link is correct
3. Ask user for correct dataset path

### Weights Not Found

1. Verify weight file exists
2. Check weight file name matches config
3. Ask user for correct weight path

## Reference

- DrivingSDK model examples: `DrivingSDK/model_examples/`
- Model-specific README: `DrivingSDK/model_examples/<model>/README.md`
