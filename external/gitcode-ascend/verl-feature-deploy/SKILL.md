---
name: external-gitcode-ascend-verl-feature-deploy
description: Verl 分布式训练服务一键拉起与配置。触发场景：(1) 用户要启动 Verl 训练任务或部署 RLHF/DAPO 训练环境 (2) 在
  NPU 集群上拉起 Verl 训练容器 (3) 配置 Ray 集群和 SwanLab 监控 (4) 根据 7 位二进制掩码灵活配置加速特性。支持 Qwen3-8B
  等 Megatron 模型的 DAPO 训练全流程。
original-name: verl-deploy
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# Verl Deploy - 训练服务一键拉起

在 NPU 集群上拉起 Verl 分布式训练服务，并灵活配置加速特性，支持 DAPO/GRPO 等 RLHF 算法。

## 核心原则

1. **用户配置优先**：用户明确指定 > 自动检测 > 默认值
2. **配置必须验证**：路径、镜像等运行前检测有效性，不猜测
3. **用户确认是阻断点**：执行前展示配置清单并等待确认
4. **严格区分宿主机/容器内**：docker cp 在宿主机执行，进入容器后不再使用 docker 前缀命令

---

## 整体流程

```
1. 环境预检查 → 2. 用户交互 → 3. 配置确认 → 4. 镜像准备+容器拉起 → 5. SwanLab 配置 → 6. 生成双脚本 → 7. 复制+执行 → 8. 验证
```

---

## 阶段 1：环境预检查

执行以下 bash 命令自动探测机器环境：

```bash
# NPU 信息
npu-smi info
npu-smi info -t board 2>/dev/null

# Docker 和镜像
docker ps -a | grep verl
docker images | grep -E "verl|ascend"

# 模型权重
find /mnt/public /mnt2 /mnt/project -maxdepth 4 -type d -name "Qwen*" 2>/dev/null

# 数据集
find /mnt /mnt2 -name "*.parquet" 2>/dev/null | head -20

# 网卡和 IP
hostname -I
```

将检测结果记录，供阶段 2 使用。

---

## 阶段 2：用户交互收集信息

通过 AskUserQuestion 分轮次收集，每轮提供预填值（来自阶段 1 检测结果）。

### 轮次 1：基础配置

- 训练模式：单机 / 多机
- NPU 卡号（预填检测结果，用户可改）
- 模型路径（列出检测到的模型供选择，或手动指定路径）
- 训练数据集路径
- 测试数据集路径
- Checkpoint 保存路径

### 轮次 2：镜像与容器配置

- 容器处理方式：使用已有容器 / 新建容器
- 如新建：
  - 镜像选择：列出本地已有 verl 镜像 / 手动输入 / 从 quay.io 拉取
  - 默认拉取地址：`quay.io/ascend/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0`

### 轮次 3：SwanLab 配置

- 是否开启 SwanLab 日志（是/否）

如果开启，逐项询问（提供默认值）：
- SwanLab Host：**无默认值，必填**（如 `http://192.168.1.100:8000`）
- SwanLab API Key：**无默认值，必填**
- SwanLab Mode：默认 `cloud`
- SwanLab Workspace：默认 `TrainingMaster`
- SwanLab Log Dir：默认 `/mnt/project/h4380/examples/verl/swanlab_cloud_fsdp_log/`
- Project Name：默认 `verl_hlm`

### 轮次 4：特性配置

输入 7 位二进制特性掩码：

```
位[0]位[1]位[2]位[3]位[4]位[5]位[6]
 ↓    ↓    ↓    ↓    ↓    ↓    ↓
RmvPad DynBSZ Offload PfxCache Recompute SwapOpt VPP
```

默认值：`0000000`（全部关闭）

特性说明：

| 位 | 名称 | 类型 | 说明 |
|----|------|------|------|
| 0 | Remove Padding | 简单开关 | 减少无效 padding 计算 |
| 1 | Dynamic Batch Size | 简单开关 | 按实际长度切分 batch |
| 2 | Offload | 简单开关 | 参数/梯度卸载到 CPU |
| 3 | Prefix Cache | 追加行 | 复用 KV 缓存前缀 |
| 4 | Recompute | 追加 3 行 | 重计算替代缓存 |
| 5 | Swap Optimizer | 追加行 | 优化器状态换入换出 |
| 6 | VPP | 追加行 | 虚拟流水线并行 |

**互斥警告**：bit5(Swap Optimizer) 与 bit2(Offload) 的 optimizer_offload 不能同时开启，脚本会自动将 optimizer_offload 设为 False。

### 轮次 5：训练超参

提供默认值，用户可覆盖：
- 训练步数（默认 4）
- Batch Size（默认 32）
- 学习率（默认 1e-6）
- TP 张量并行（默认 4）
- PP 流水线并行（默认 2）
- 每 prompt 生成响应数（默认 8）

---

## 阶段 3：配置确认（阻断点）

展示完整配置清单，包括：

- 硬件配置（NPU 卡号、数量）
- 容器/镜像信息
- 资源路径（模型、数据集、Checkpoint）
- 训练超参
- 特性掩码及各特性开关状态
- SwanLab 配置（如开启）

等待用户确认后继续。**用户确认前不执行任何修改操作。**

---

## 阶段 4：镜像准备 + 容器拉起

### 已有容器

确认容器名，检查状态，如已停止则启动：

```bash
docker ps -a | grep <container_name>
docker start <container_name>  # 如已停止
```

### 新建容器

1. **确认/拉取镜像**：

```bash
# 用户选择本地镜像 → 直接使用
# 用户选择拉取 → 执行
docker pull quay.io/ascend/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0
```

2. **拉起容器**：使用 `scripts/verl_docker_run.sh`

```bash
bash scripts/verl_docker_run.sh <镜像ID> <容器名>
```

3. **验证容器就绪**：

```bash
docker exec <container_name> bash -c "npu-smi info && python --version"
```

---

## 阶段 5：SwanLab 配置（如开启）

进入容器后，以容器内形式操作（无 docker 前缀）。

**注意**：此阶段需要先进入容器再操作。如果后面阶段 7 才正式进入容器，也可以在阶段 7 进入容器后、启动训练前执行 SwanLab 配置。

### 步骤 1：检查并安装 swanlab

```bash
python -c "import swanlab" 2>/dev/null && echo "swanlab 已安装" || pip install swanlab
```

### 步骤 2：登录

```bash
echo '<SWANLAB_API_KEY>' | swanlab login --host <SWANLAB_HOST>
```

### 步骤 3：验证登录

```bash
cat ~/.swanlab/config.json
```

文件存在且包含 API Key 信息即表示登录成功。

---

## 阶段 6：生成双脚本

使用 `scripts/generate_training.sh` 生成两个脚本。

### 生成结果

| 脚本 | 职责 | 对应参考 |
|------|------|----------|
| `start_verl.sh` | 上层：NPU/网络配置、Ray 集群启动、SwanLab 环境变量、调用下层 | start_flex.sh |
| `run_training.sh` | 下层：训练超参、特性配置、Hydra python 训练命令 | test_dapo_qwen3_8b_megatron_flex.sh |

调用关系：`start_verl.sh` → `bash run_training.sh`

### 生成命令

```bash
bash scripts/generate_training.sh \
  --mask <FEATURE_MASK> \
  --output-dir /tmp/verl_scripts/ \
  --model-path <MODEL_PATH> \
  --train-file <TRAIN_FILE> \
  --test-file <TEST_FILE> \
  --ckpts-dir <CKPTS_DIR> \
  --npu-devices "0,1,2,3,4,5,6,7" \
  --master-addr <IP> \
  --socket-ifname <IFNAME> \
  --train-steps <STEPS> \
  --train-tp <TP> \
  --train-pp <PP> \
  --swanlab <yes|no> \
  --swanlab-host <HOST> \
  --swanlab-api-key <KEY> \
  --swanlab-mode <MODE> \
  --swanlab-workspace <WS> \
  --swanlab-log-dir <DIR> \
  --project-name <NAME>
```

### 特性处理机制

#### Type A（bit0-2）：简单变量替换

模板中已存在占位符，sed 直接替换：
- `USE_REMOVE_PADDING_PLACEHOLDER` → True/False
- `USE_DYNAMIC_BSZ_PLACEHOLDER` → True/False
- `PARAM_OFFLOAD_PLACEHOLDER` / `OPTIMIZER_OFFLOAD_PLACEHOLDER` / `GRAD_OFFLOAD_PLACEHOLDER` → 各自 True/False

Offload + Swap Optimizer 互斥：同时开启时自动将 `OPTIMIZER_OFFLOAD` 强制设为 False。

#### Type B（bit3-6）：注释控制

模板中 Type B 特性行默认以 `# ` 注释，根据 bit 位取消注释：

```bash
# bit3: Prefix Cache
[ "$bit3" = "1" ] && sed -i 's/^# \(actor_rollout_ref.rollout.enable_prefix_caching=True\)/\1/' "$script"

# bit4: Recompute（3 行）
if [ "$bit4" = "1" ]; then
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full\)|\1|' "$script"
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block\)|\1|' "$script"
    sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=8\)|\1|' "$script"
fi

# bit5: Swap Optimizer
[ "$bit5" = "1" ] && sed -i 's|^# \(+actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True\)|\1|' "$script"

# bit6: VPP
[ "$bit6" = "1" ] && sed -i 's|^# \(actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2\)|\1|' "$script"

# 清理未启用的注释行
sed -i '/^#.*\(+\?actor_rollout_ref\|enable_prefix_caching\)/d' "$script"
```

#### SwanLab 处理

模板默认 `trainer.logger='["console","swanlab"]'`。如 SwanLab 未开启：

```bash
sed -i "s|trainer.logger='\[\"console\",\"swanlab\"\]'|trainer.logger='[\"console\"]'|g" "$script"
```

---

## 阶段 7：复制脚本到容器 + 执行

### 步骤 1（宿主机）：docker cp

> **关键**：必须在宿主机环境执行 docker cp。确认当前不在容器内。

```bash
docker cp /tmp/verl_scripts/start_verl.sh <container_name>:/verl/start_verl.sh
docker cp /tmp/verl_scripts/run_training.sh <container_name>:/verl/run_training.sh
```

### 步骤 2：进入容器

```bash
docker exec -it <container_name> bash
```

### 步骤 3（容器内）：SwanLab 配置 + 启动训练

进入容器后，以容器内形式执行（无 docker 前缀）：

```bash
# 激活环境
source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh

# （如开启 SwanLab）注入环境变量 + 安装登录
export SWANLAB_HOST="<用户提供的地址>"
export SWANLAB_API_KEY="<用户提供的 API Key>"
export SWANLAB_MODE="<cloud 或 local>"
export SWANLAB_WORKSPACE="<工作空间>"
export SWANLAB_LOG_DIR="<日志目录>"
export PROJECT_NAME="<项目名>"

python -c "import swanlab" 2>/dev/null || pip install swanlab
echo "$SWANLAB_API_KEY" | swanlab login --host "$SWANLAB_HOST"
cat ~/.swanlab/config.json  # 验证

# 启动训练
cd /verl
bash start_verl.sh
```

`start_verl.sh` 自动完成：Ray 集群启动 → 调用 `run_training.sh` → 开始训练。

---

## 阶段 8：验证训练启动

在容器内检查：

```bash
ps aux | grep main_dapo
tail -20 logs/verl_qwen3_8b_*.log
```

---

## 并行策略约束

| 约束 | 说明 |
|------|------|
| `gen_tp == train_tp` | 推理和训练的张量并行必须相等 |
| `(train_tp x train_pp) % NPU_PER_NODE == 0` | tp x pp 必须能被每节点 NPU 数整除 |

---

## 可用脚本

| 脚本 | 用途 |
|------|------|
| `scripts/generate_training.sh` | 根据参数生成双脚本 |
| `scripts/feature_mask.sh` | 7 位二进制特性掩码解析 |
| `scripts/pre_check.sh` | 环境预检查 |
| `scripts/verl_docker_run.sh` | 容器拉起 |

## 参考文档

- `references/feature-guide.md` — 7 个加速特性详细说明
- `references/troubleshooting.md` — 常见问题与解决方案（Q1-Q13）
- `references/ops-commands.md` — 常用运维命令（Docker/NPU/Ray/日志）
