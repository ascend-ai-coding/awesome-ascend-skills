---
name: mindspeed-llm-pipeline
description: MindSpeed-LLM 端到端训练部署流水线，用于华为昇腾 NPU。串联环境搭建、数据预处理、权重转换、分布式训练、推理验证、评估、权重导出的完整流程。覆盖参数一致性校验、阶段间数据流转、模型特定配置（Qwen、LLaMA、DeepSeek）。用户需要在昇腾 NPU 上完成 MindSpeed-LLM 训练部署全流程时使用。
keywords:
    - mindspeed
    - mindspeed-llm
    - pipeline
    - 流水线
    - end-to-end
    - 端到端
    - deployment
    - 部署
    - workflow
    - 工作流
    - qwen
    - llama
    - deepseek
---

# MindSpeed-LLM 端到端训练部署流水线

本 Skill 串联 MindSpeed-LLM 从环境搭建到模型导出的完整部署流程，确保各阶段参数一致、数据正确流转。

## 使用顺序

```
1. 环境搭建  →  2. 模型下载  →  3. 权重转换(HF→MG)
→  4. 数据预处理  →  5. 训练启动
→  6. 推理验证(可选)  →  7. 评估(可选)  →  8. 权重导出(MG→HF，可选)
```

## 前置要求

| 依赖 | 说明 | 参考 |
|------|------|------|
| CANN + torch_npu | NPU 运行环境 | [mindspeed-llm-env-setup](../mindspeed-llm-env-setup/SKILL.md) |
| MindSpeed + Megatron-LM | 分布式训练框架 | [mindspeed-llm-env-setup](../mindspeed-llm-env-setup/SKILL.md) |

## 快速开始：Qwen2.5-7B LoRA 微调

以下示例展示从零到训练完成的完整流程。

### Step 1：环境初始化

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 验证
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

详细安装步骤见 [mindspeed-llm-env-setup](../mindspeed-llm-env-setup/SKILL.md)。

### Step 2：下载模型权重

```bash
mkdir -p ./model_from_hf/Qwen2.5-7B-Instruct && cd ./model_from_hf/Qwen2.5-7B-Instruct

# 下载配置文件和分词器
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json

# 下载权重分片
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00002-of-00004.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00003-of-00004.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00004-of-00004.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model.safetensors.index.json
cd ../..
```

验证：`ls model_from_hf/Qwen2.5-7B-Instruct/` 应包含 `config.json`、`tokenizer.json`、4 个 `.safetensors` 文件。

### Step 3：权重转换（HF → Megatron）

```bash
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --add-qkv-bias \
    --load-dir ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --save-dir ./model_weights/qwen25_7b_mcore/ \
    --tokenizer-model ./model_from_hf/Qwen2.5-7B-Instruct/tokenizer.json \
    --model-type-hf llama2 \
    --params-dtype bf16
```

验证：`ls model_weights/qwen25_7b_mcore/` 应包含 `mp_rank_*/` 目录和 `latest_checkpointed_iteration.txt`。

详细参数见 [mindspeed-llm-weight-prep](../mindspeed-llm-weight-prep/SKILL.md)。

### Step 4：数据预处理

```bash
# 下载训练数据
mkdir -p dataset && cd dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# 预处理为 packed 格式
python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --output-prefix ./finetune_dataset/alpaca \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type qwen \
    --workers 4 \
    --log-interval 1000
```

验证：`ls finetune_dataset/` 应包含 `alpaca_packed_*_document.bin` 和 `.idx` 文件。

详细格式见 [mindspeed-llm-data-prep](../mindspeed-llm-data-prep/SKILL.md)。

### Step 5：训练启动

```bash
#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    --use-mcore-models \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
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

详细参数见 [mindspeed-llm-training](../mindspeed-llm-training/SKILL.md)。

### Step 6：推理验证（可选）

训练完成后可使用示例脚本验证模型质量：

```bash
bash examples/mcore/qwen25/generate_qwen25_7b_ptd.sh
```

### Step 7：评估（可选）

```bash
bash examples/mcore/qwen25/evaluate_qwen25_7b_mmlu.sh
```

### Step 8：权重导出（可选）

训练后将 Megatron 权重转回 HuggingFace 格式以便部署：

```bash
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/qwen25_7b_mcore/ \
    --save-dir ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --tokenizer-model ./model_from_hf/Qwen2.5-7B-Instruct/tokenizer.json \
    --model-type-hf llama2 \
    --params-dtype bf16
```

> MG→HF 时使用 v1 需设 TP=PP=1；v2 无需设置并行参数。输出在 `--save-dir` 下的 `mg2hf/` 子目录。

## 阶段间数据流转

```
model_from_hf/Qwen2.5-7B-Instruct/     ← Step 2 输出
    ↓ convert_ckpt.py (HF→MG)
model_weights/qwen25_7b_mcore/          ← Step 3 输出
    ↓ 作为 --load 输入
    ↓
dataset/*.parquet                        ← Step 4 输入
    ↓ preprocess_data.py
finetune_dataset/alpaca_packed_*         ← Step 4 输出
    ↓ 作为 --data-path 输入
    ↓
lora_output/                             ← Step 5 输出
    ↓ convert_ckpt.py (MG→HF，可选)
model_from_hf/.../mg2hf/                 ← Step 8 输出
```

## 参数一致性规则

以下参数 **必须** 在权重转换、训练、推理之间保持一致：

| 参数 | 转换 | 训练 | 推理 |
|------|:----:|:----:|:----:|
| TP（`tensor-parallel-size`） | 设定 | 匹配 | 匹配 |
| PP（`pipeline-parallel-size`） | 设定 | 匹配 | 匹配 |
| `--add-qkv-bias` | 按模型 | 匹配 | 匹配 |
| `--group-query-attention` | 按模型 | 匹配 | 匹配 |
| `--num-query-groups` | 按模型 | 匹配 | 匹配 |
| `--position-embedding-type` | 按模型 | 匹配 | 匹配 |
| 模型架构参数 | — | 必须正确 | 匹配训练 |

> 参数不一致会导致权重加载失败、NaN loss 或输出乱码。

## 模型特定配置速查

| 模型 | `--model-type-hf` | `--add-qkv-bias` | `--num-query-groups` | 层数 |
|------|-------------------|:-----------------:|:--------------------:|:----:|
| Qwen2.5-7B | `llama2` | 是 | 4 | 28 |
| Qwen2.5-72B | `llama2` | 是 | 8 | 80 |
| Qwen3-8B | `qwen3` | 否 | 8 | 36 |
| LLaMA-3-8B | `llama2` | 否 | 8 | 32 |
| LLaMA-3-70B | `llama2` | 否 | 8 | 80 |
| DeepSeek-7B | `llama2` | 否 | 32 | 32 |

## 预检清单

开始部署前逐项确认：

- [ ] NPU 可用：`python -c "import torch_npu; print(torch.npu.is_available())"`
- [ ] CANN 环境激活：`echo $ASCEND_HOME_PATH`
- [ ] MindSpeed-LLM 目录包含 `megatron/` 子目录
- [ ] HF 权重下载完整（config.json + tokenizer + safetensors）
- [ ] 确定 TP/PP 配置并记录
- [ ] 确认模型层数可被 PP 整除
- [ ] 数据集已下载

## 常见问题

**Q: 训练报错 `.idx and .bin files cannot be found`**

1. 微调必须使用 `posttrain_gpt.py`（不是 `pretrain_gpt.py`）
2. `--data-path` 应为前缀（如 `./finetune_dataset/alpaca`），不含 `_packed`

**Q: 权重转换后训练报错 Shape mismatch**

TP/PP 或模型架构参数不一致。对照上方参数一致性表逐项检查。

**Q: 训练 loss 为 NaN**

- 检查 `--initial-loss-scale`（bf16 建议 4096）
- 确认 `--clip-grad 1.0`
- 确认 `--add-qkv-bias` 是否匹配模型

**Q: `$'\\r': command not found`**

脚本有 Windows 换行符：`sed -i "s/\\r//g" your_script.sh`

**Q: 多机训练超时**

```bash
export HCCL_CONNECT_TIMEOUT=1800
# 确保 MASTER_ADDR 不是 localhost
```

## 相关 Skill

- [mindspeed-llm-env-setup](../mindspeed-llm-env-setup/SKILL.md) - Phase 1: 环境搭建
- [mindspeed-llm-data-prep](../mindspeed-llm-data-prep/SKILL.md) - Phase 4: 数据预处理
- [mindspeed-llm-weight-prep](../mindspeed-llm-weight-prep/SKILL.md) - Phase 3/8: 权重转换
- [mindspeed-llm-training](../mindspeed-llm-training/SKILL.md) - Phase 5: 训练启动

## 参考资源

- [完整流水线参考](references/pipeline-guide.md) - 预训练/DPO 流水线、模型架构参数、多机部署
- [MindSpeed-LLM 快速开始](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/quick_start.md)
- [MindSpeed-LLM 仓库](https://gitcode.com/ascend/MindSpeed-LLM)
