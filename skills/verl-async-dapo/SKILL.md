---
name: verl-async-dapo
description: Verl 单异步 DAPO 训练配置生成器。触发场景：(1) 启动单异步 DAPO 训练 (2) 生成训练脚本 (3) 配置特性参数 (4) 训练前检查。**特性策略**：用户未指定时默认开启性能特性（flash_attn/dynamic_batch/remove_padding/gradient_checkpointing），显存特性（offload/recompute）默认关闭。OOM 时自动追加显存特性重试。**训练监控**：启动后输出 SwanLab 链接供用户自行查看，仅在错误时通知用户。**依赖 skill**：SwanLab 配置通过 swanlab-setup skill 提供。
---

# Verl 单异步 DAPO 训练

## 交互式启动流程

启动训练服务前，Agent **必须**按以下顺序询问用户：

### 1. 容器选择
```
检测到以下容器：
  [1] jins ( 运行中)
  [2] 创建新容器

请选择容器 (1/2):
```

### 2. 代理配置（如创建新容器或网络问题）
```
请提供代理配置（如不需要请回复"无"）：
  http_proxy: ?
  https_proxy: ?
```

### 3. 特性选择
```
是否有自定义特性需求？(回复"默认"使用默认配置)

性能特性（默认全开）:
  - flash_attn: Flash Attention 加速
  - dynamic_batch: 动态 Batch Size
  - remove_padding: Remove Padding 优化
  - gradient_checkpointing: 梯度检查点

显存特性（默认关闭，OOM 时自动开启）:
  - offload: 参数/优化器卸载
  - recompute: 重计算

可选特性:
  - prefix_cache: Prefix Cache
  - chunked_prefill: Chunked Prefill
```

### 4. SwanLab 配置（如未配置）
```
SwanLab 监控配置：
  Host: ?
  API Key: ?
```

## 快速开始

```bash
# 方式 1: 快速启动脚本 (推荐)
CONTAINER=jins TRAIN_STEPS=100 \
SWANLAB_HOST=http://10.143.2.129:8000 \
SWANLAB_API_KEY=your-key \
bash scripts/quick_start.sh

# 方式 2: 分步执行
# 2.1 训练前检查
bash scripts/preflight_check.sh

# 2.2 启动单异步 DAPO 训练 (使用默认特性进行训练)
export TRAIN_STEPS=4
bash scripts/run_dapo.sh

# 2.3 启动 One-Step-Off-Policy 训练
export TRAIN_STEPS=4
bash scripts/run_one_step_off_policy.sh
```

### One-Step-Off-Policy 训练

One-Step-Off-Policy 是资源隔离的单异步训练模式，Trainer 和 Rollout 使用独立的 GPU 组。

```bash
# 基本用法
export TRAIN_STEPS=100
export TRAINER_GPUS=4
export ROLLOUT_GPUS=4
bash scripts/run_one_step_off_policy.sh

# 启用 offload 特性
export ENABLE_OFFLOAD=True
bash scripts/run_one_step_off_policy.sh

# FSDP2 框架
export FRAMEWORK=fsdp
bash scripts/run_one_step_off_policy.sh
```

**配置参数**:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FRAMEWORK` | `megatron` | 框架 (megatron/fsdp) |
| `TRAINER_GPUS` | 4 | 训练 GPU 数 |
| `ROLLOUT_GPUS` | 4 | Rollout GPU 数 |
| `MAX_RESPONSE_LENGTH` | 8192 | 最大响应长度 |
| `ENABLE_OFFLOAD` | False | 是否启用 offload |

## SwanLab 配置

SwanLab 配置由 **swanlab-setup skill** 提供，`common.sh` 启动时自动加载。详见 swanlab-setup skill 的 SKILL.md。

配置按优先级自动获取：**环境变量 > 配置文件 > 交互式输入**

```bash
# 环境变量（推荐）
export SWANLAB_HOST="http://your-swanlab-host:8000"
export SWANLAB_API_KEY="your-api-key"
```

**重要**：必须在 `trainer.logger` 参数中包含 `"swanlab"` 才能启用监控：

```bash
# 正确的 logger 配置（启用 SwanLab 监控）
trainer.logger='["console", "swanlab"]'

# 仅控制台输出（不启用监控）
trainer.logger='["console"]'
```

SwanLab 环境变量会自动被 verl tracking 模块读取：
- `SWANLAB_API_KEY`: API 密钥
- `SWANLAB_LOG_DIR`: 日志目录 (默认: swanlog)
- `SWANLAB_MODE`: 模式 (cloud/local, 默认: cloud)

## 代理配置

```bash
export http_proxy="http://your-proxy:8080"
export https_proxy="http://your-proxy:8080"
```

## 训练监控策略

**启动后行为**:
1. 输出 SwanLab 链接供用户自行查看
2. 后台静默运行，不实时推送进度
3. **仅在错误时通知用户**（OOM、崩溃、配置错误等）

**用户自行监控命令**:
```bash
# 查看当前进度
docker exec {container} grep "Training Progress" /verl/train.log | tail -1

# 实时跟踪日志
docker exec {container} tail -f /verl/train.log

# 检查是否出错
docker exec {container} grep -E "Error|error|OOM|Exception" /verl/train.log | tail -10
```

## 特性管理策略

### 特性分类

| 类型 | 特性 | 默认状态 | 说明 |
|------|------|----------|------|
| **性能特性** | `flash_attn` | ✅ 默认开 | Flash Attention 加速 |
| **性能特性** | `dynamic_batch` | ✅ 默认开 | 动态 Batch Size |
| **性能特性** | `remove_padding` | ✅ 默认开 | Remove Padding 优化 |
| **性能特性** | `gradient_checkpointing` | ✅ 默认开 | 梯度检查点 |
| **显存特性** | `offload` | ❌ 默认关 | 参数/优化器卸载 (OOM 时自动追加) |
| **显存特性** | `recompute` | ❌ 默认关 | 重计算 (OOM 时自动追加) |
| **可选特性** | `prefix_cache` | ❌ 默认关 | Prefix Cache |
| **可选特性** | `chunked_prefill` | ❌ 默认关 | Chunked Prefill |

### 特性优先级

```
用户显式指定 > 默认性能特性 > OOM 自动追加显存特性
```

**重要**：
- 用户未指定时 → 使用默认性能特性（全开）
- OOM 时 → 自动追加显存特性（即使用户指定了其他特性）
- 用户显式指定的特性永不移除，只在 OOM 时追加

### OOM 自动恢复流程

```
启动训练 (默认性能特性)
    ↓
OOM 错误?
    ├─ 是 → 追加 offload → 重试
    │   └─ 仍 OOM → 追加 recompute → 重试
    │       └─ 仍 OOM → 报告失败，建议调整参数
    └─ 否 → 训练成功（静默完成）
```

## 配置参数

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FRAMEWORK` | `megatron` | 框架 (megatron/fsdp) |
| `MODEL_PATH` | Qwen3-8B | 模型路径 |
| `TRAIN_FILE` | dapo-math-17k.parquet | 训练数据 |
| `VAL_FILE` | aime-2024.parquet | 验证数据 |
| `TRAIN_STEPS` | 100 | 训练步数 |
| `TRAINER_GPUS` | 4 | 训练 GPU 数 |
| `ROLLOUT_GPUS` | 4 | Rollout GPU 数 |
| `USER_FEATURES` | (空) | 用户显式指定特性，逗号分隔 |
| `MAX_OOM_RETRIES` | 2 | OOM 最大重试次数 |
| `SWANLAB_HOST` | (从配置获取) | SwanLab 服务器地址 |
| `SWANLAB_API_KEY` | (从配置获取) | SwanLab API 密钥 |

## 详细参考

- **完整工作流程**: See [references/workflow.md](references/workflow.md)
- **参数详解**: See [references/params.md](references/params.md)
- **故障排查**: See [references/troubleshooting.md](references/troubleshooting.md)