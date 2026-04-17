---
name: mindspeed-llm-training
description: MindSpeed-LLM 分布式训练启动指南，用于华为昇腾 NPU。覆盖预训练、指令微调（Full/LoRA/QLoRA）、偏好对齐（DPO）的训练启动配置，包含入口脚本选择、并行策略（TP/PP/CP）、训练参数、可选评估。当用户需要在昇腾 NPU 上启动 MindSpeed-LLM 训练任务时使用。
keywords:
    - mindspeed
    - mindspeed-llm
    - training
    - 训练
    - fine-tuning
    - 微调
    - lora
    - qlora
    - sft
    - pretraining
    - 预训练
    - distributed training
    - 分布式训练
    - dpo
    - evaluation
    - 评估
---

# MindSpeed-LLM 分布式训练启动

本 Skill 指导用户在华为昇腾 NPU 上启动 MindSpeed-LLM 分布式训练任务。

## 入口脚本

| 入口脚本 | 用途 |
|----------|------|
| `pretrain_gpt.py` | 预训练（从头训练或继续预训练） |
| `posttrain_gpt.py` | **指令微调（SFT、LoRA、QLoRA、Full）** |
| `posttrain_gpt.py` | 偏好对齐（DPO、GRPO） |
| `train_fsdp2.py` | FSDP2 后端训练 |

> **重要**：所有微调任务必须使用 `posttrain_gpt.py`，**不要**使用 `pretrain_gpt.py`。只有 `posttrain_gpt.py` 才能正确路由 `--is-instruction-dataset` 到 packed 指令数据加载器。

## 快速开始

### LoRA 微调（Qwen2.5-7B）

```bash
# 使用示例脚本（推荐）
bash examples/mcore/qwen25/tune_qwen25_7b_4k_lora_ptd.sh

# 或手动启动
bash run_finetune.sh
```

### 预训练（Qwen2.5-7B）

```bash
bash examples/mcore/qwen25/pretrain_qwen25_7b_32k_ptd.sh
```

## 示例脚本

脚本位于 `examples/mcore/<model_name>/`：

| 脚本模式 | 用途 |
|----------|------|
| `pretrain_<model>_*.sh` | 预训练启动 |
| `tune_<model>_*_full*.sh` | 全参数微调 |
| `tune_<model>_*_lora*.sh` | LoRA 微调 |
| `generate_<model>_*.sh` | 推理 |
| `evaluate_<model>_*.sh` | 评估 |

### 支持的模型

| 模型族 | 规模 | 脚本目录 |
|--------|------|----------|
| Qwen2.5 | 0.5B - 72B | `examples/mcore/qwen25/` |
| Qwen3 | 0.6B - 32B | `examples/mcore/qwen3/` |
| LLaMA 3/3.1 | 8B, 70B | `examples/mcore/llama3/` |
| DeepSeek-V2/V3 | 各规模 | `examples/mcore/deepseek/` |
| ChatGLM4 | 9B | `examples/mcore/glm4/` |
| Mistral | 7B | `examples/mcore/mistral/` |
| Baichuan2 | 7B, 13B | `examples/mcore/baichuan2/` |
| Qwen3-MoE | 235B | `examples/mcore/qwen3_moe/` |
| Mixtral | 8x7B | `examples/mcore/mixtral/` |

## 训练参数

### 并行策略

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--tensor-model-parallel-size` | 张量并行度（TP） | 1-8 |
| `--pipeline-model-parallel-size` | 流水线并行度（PP） | 1-8 |
| `--context-parallel-size` | 上下文并行度（CP） | 1-2 |
| `--micro-batch-size` | 每 NPU 微批大小 | 1-4 |
| `--global-batch-size` | 全局批大小 | 8-64 |

### 推荐并行配置

| 模型规模 | NPU 数 | 推荐 TP | 推荐 PP |
|----------|--------|---------|---------|
| < 3B | 1-8 | 1 | 1 |
| 7B-14B | 8 | 1-2 | 1-4 |
| 32B-72B | 8-16 | 4-8 | 2-4 |
| 100B+ | 16+ | 8 | 4+ |

### 通用训练参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--seq-length` | 序列长度 | 2048-32768 |
| `--train-iters` | 训练迭代数 | 1000-5000 |
| `--lr` | 学习率 | 1e-5 ~ 1e-6 |
| `--bf16` | BFloat16 精度 | 始终推荐 |
| `--use-flash-attn` | Flash Attention | 始终推荐 |
| `--use-fused-rmsnorm` | 融合 RMSNorm | 始终推荐 |
| `--use-fused-swiglu` | 融合 SwiGLU | 始终推荐 |
| `--use-distributed-optimizer` | 分布式优化器 | 多卡推荐 |
| `--save-interval` | Checkpoint 保存间隔 | 500-1000 |
| `--eval-interval` | 评估间隔 | 500-1000 |

### 微调专用参数

```bash
--finetune                    # 启用微调模式
--stage sft                   # 监督微调阶段
--is-instruction-dataset      # 使用 packed 指令数据加载器
--prompt-type qwen            # 对话模板（qwen, qwen3, llama3, glm4 等）
--no-load-optim               # 不加载优化器状态
--no-load-rng                 # 不加载随机数状态
--no-pad-to-seq-lengths       # 不填充到 seq-length（节省计算）
--reset-attention-mask        # 打包数据需重置注意力掩码
--enable-thinking             # Qwen3 思维链模式（需配合 --prompt-type qwen3）
```

### LoRA 参数

```bash
--lora-r 8                    # LoRA 秩（典型：8-16）
--lora-alpha 16               # LoRA 缩放因子（推荐 α/r ≈ 2）
--lora-fusion                 # 启用 CCLoRA（计算通信重叠）
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

### QLoRA 参数

```bash
--qlora                       # 启用 QLoRA
--lora-r 8
--lora-alpha 16
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

> QLoRA 使用 4-bit 量化基础模型 + LoRA 适配器，显著减少显存占用。
> **前提**：QLoRA 训练前，权重转换（HF→MG）时须加 `--qlora-nf4` 生成 NF4 量化权重。
> **注意**：QLoRA 不支持 `--lora-fusion`，开启无性能收益。

## 训练启动脚本模板

```bash
#!/bin/bash

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# LoRA 微调示例
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    --use-mcore-models \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --seq-length 4096 \
    --train-iters 2000 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --bf16 \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-distributed-optimizer \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type qwen \
    --no-load-optim \
    --no-load-rng \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load ./model_weights/qwen25_7b_mcore/ \
    --data-path ./finetune_dataset/alpaca \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --save ./lora_output/ \
    --save-interval 1000 \
    --log-interval 1
```

## DPO 偏好对齐

```bash
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    --stage dpo \
    --is-pairwise-dataset \
    --is-instruction-dataset \
    --dpo-beta 0.1 \
    --dpo-loss-type sigmoid \
    --data-path ./dataset/pairwise_data \
    # ... 其他参数同 SFT
```

## 可选：模型评估

训练完成后可运行评估：

```bash
bash examples/mcore/qwen25/evaluate_qwen25_7b_mmlu.sh
```

支持的评估基准：MMLU、GSM8K、BBH、C-Eval、CMMLU、HumanEval、HellaSwag、BoolQ、NeedleBench、AGI-Eval。

## FSDP2 后端（高级）

使用 FSDP2 替代 Megatron 原生分布式：

```bash
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    --fsdp2 \
    --fsdp2-reshard-after-forward \
    # ... 其他参数
```

## 常见问题

**Q: `AssertionError: .idx and .bin files cannot be found`**

两个常见原因：

1. **入口脚本错误**：微调必须使用 `posttrain_gpt.py`（不是 `pretrain_gpt.py`）
2. **数据路径错误**：`--data-path` 应为前缀（如 `./finetune_dataset/alpaca`），不要包含 `_packed`

**Q: `$'\r': command not found`**

训练脚本有 Windows 换行符：

```bash
sed -i "s/\r//g" your_script.sh
```

**Q: 训练卡住或 HCCL 超时**

```bash
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**Q: 多机训练如何配置**

```bash
# 节点 0（主节点）
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 torchrun ...

# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 torchrun ...
```

## 相关 Skill

- [mindspeed-llm-env-setup](../mindspeed-llm-env-setup/SKILL.md) - 环境搭建
- [mindspeed-llm-data-prep](../mindspeed-llm-data-prep/SKILL.md) - 数据预处理
- [mindspeed-llm-weight-prep](../mindspeed-llm-weight-prep/SKILL.md) - 权重转换
- [hccl-test](../../hccl-test/SKILL.md) - 多卡通信测试

## 参考资源

- [详细训练配置](references/training-config.md) - 完整参数列表、高级并行策略、性能优化
- [MindSpeed-LLM 仓库](https://gitcode.com/ascend/MindSpeed-LLM)
