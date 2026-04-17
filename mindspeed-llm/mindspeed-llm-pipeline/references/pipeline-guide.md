# MindSpeed-LLM 流水线完整参考

## 预训练流水线

预训练使用 `pretrain_gpt.py` 入口，数据格式为 `_text_document.bin/.idx`。

### 数据预处理

```bash
python ./preprocess_data.py \
    --input ./dataset/enwiki.parquet \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B/ \
    --output-prefix ./dataset/enwiki \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --workers 4 \
    --log-interval 1000
```

输出：`enwiki_text_document.bin` + `.idx`

### 预训练启动

```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --use-mcore-models \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 4 \
    --global-batch-size 64 \
    --seq-length 4096 \
    --train-iters 100000 \
    --lr 5e-4 \
    --min-lr 5e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.1 \
    --bf16 \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-distributed-optimizer \
    --data-path ./dataset/enwiki_text_document \
    --split 99,1,0 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B/ \
    --load ./model_weights/qwen25_7b_mcore/ \
    --save ./ckpt/qwen25_pretrain/ \
    --save-interval 500 \
    --eval-interval 500 \
    --log-interval 1
```

### 预训练续训

```bash
# 不加 --finetune（否则会跳过优化器加载）
# 不加 --no-load-optim、--no-load-rng
--load ./ckpt/qwen25_pretrain/
--use-distributed-optimizer    # 必需
```

## DPO 偏好对齐流水线

### 数据预处理（Pairwise 格式）

```bash
python ./preprocess_data.py \
    --input ./dataset/orca_rlhf.jsonl \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --output-prefix ./pairwise_dataset/orca_rlhf \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStylePairwiseHandler \
    --prompt-type qwen \
    --map-keys '{"prompt":"question", "query":"", "system":"system"}' \
    --workers 4
```

输出：`orca_rlhf_packed_chosen_*` 和 `orca_rlhf_packed_rejected_*` 文件对。

### DPO 训练

```bash
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    --use-mcore-models \
    --finetune \
    --stage dpo \
    --is-pairwise-dataset \
    --is-instruction-dataset \
    --dpo-beta 0.1 \
    --dpo-loss-type sigmoid \
    --prompt-type qwen \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --seq-length 4096 \
    --train-iters 1000 \
    --lr 5e-6 \
    --bf16 \
    --use-flash-attn \
    --no-load-optim \
    --data-path ./pairwise_dataset/orca_rlhf \
    --load ./model_weights/qwen25_7b_mcore/ \
    --save ./dpo_output/
```

## 模型架构参数

### Qwen2.5-7B

```bash
--num-layers 28 \
--hidden-size 3584 \
--ffn-hidden-size 18944 \
--num-attention-heads 28 \
--kv-channels 128 \
--num-query-groups 4 \
--padded-vocab-size 152064 \
--rotary-base 1000000 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--swiglu \
--group-query-attention \
--add-qkv-bias \
--disable-bias-linear \
--untie-embeddings-and-output-weights \
--position-embedding-type rope
```

### Qwen3-8B

```bash
--num-layers 36 \
--hidden-size 4096 \
--ffn-hidden-size 12288 \
--num-attention-heads 32 \
--kv-channels 128 \
--num-query-groups 8 \
--padded-vocab-size 151936 \
--rotary-base 1000000 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--swiglu \
--group-query-attention \
--disable-bias-linear \
--untie-embeddings-and-output-weights \
--position-embedding-type rope \
--qk-layernorm
```

### LLaMA-3-8B

```bash
--num-layers 32 \
--hidden-size 4096 \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--kv-channels 128 \
--num-query-groups 8 \
--padded-vocab-size 128256 \
--rotary-base 500000 \
--normalization RMSNorm \
--norm-epsilon 1e-5 \
--swiglu \
--group-query-attention \
--disable-bias-linear \
--untie-embeddings-and-output-weights \
--position-embedding-type rope
```

### LLaMA-3-70B

```bash
--num-layers 80 \
--hidden-size 8192 \
--ffn-hidden-size 28672 \
--num-attention-heads 64 \
--kv-channels 128 \
--num-query-groups 8 \
--padded-vocab-size 128256 \
--rotary-base 500000 \
--normalization RMSNorm \
--norm-epsilon 1e-5 \
--swiglu \
--group-query-attention \
--disable-bias-linear \
--untie-embeddings-and-output-weights \
--position-embedding-type rope
```

## LoRA 合并 + 导出流水线（仅 v1）

> **注意**：LoRA 权重合并仅支持 v1（`convert_ckpt.py`）。v2（`convert_ckpt_v2.py`）不支持 LoRA 权重转换。

训练完成后将 LoRA 权重合并回基础模型并导出 HF 格式：

```bash
# Step 1: LoRA 合并导出 HF
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/qwen25_7b_mcore/ \
    --save-dir ./model_from_hf/merged_output/ \
    --lora-load ./lora_output/ \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --tokenizer-model ./model_from_hf/Qwen2.5-7B-Instruct/tokenizer.json

# Step 2: 复制 config.json 和 tokenizer（MG→HF 不会自动复制）
cp ./model_from_hf/Qwen2.5-7B-Instruct/config.json ./model_from_hf/merged_output/mg2hf/
cp ./model_from_hf/Qwen2.5-7B-Instruct/tokenizer* ./model_from_hf/merged_output/mg2hf/
```

## 多机部署

### 节点配置

```bash
# 节点 0（主节点）
NPUS_PER_NODE=8
MASTER_ADDR=192.168.1.100     # 主节点真实 IP
MASTER_PORT=6000
NNODES=2
NODE_RANK=0

# 节点 1（工作节点）
NPUS_PER_NODE=8
MASTER_ADDR=192.168.1.100     # 同一主节点 IP
MASTER_PORT=6000
NNODES=2
NODE_RANK=1
```

### 多机检查清单

- [ ] SSH 免密登录：`ssh-copy-id root@nodeN`
- [ ] 所有节点 CANN/torch_npu 版本一致
- [ ] NPU 健康：`npu-smi info`
- [ ] HCCL 通信测试：`hccl_test -b 8 -e 256M -d float16 -o allreduce`
- [ ] 模型和数据路径一致（或使用 `--no-shared-storage`）
- [ ] `MASTER_ADDR` 为真实 IP（不是 localhost）

## 目录结构约定

```
MindSpeed-LLM/
├── model_from_hf/              # HF 格式权重
│   └── Qwen2.5-7B-Instruct/
│       ├── config.json
│       ├── tokenizer.json
│       └── model-*.safetensors
├── model_weights/              # Megatron 格式权重
│   └── qwen25_7b_mcore/
│       ├── mp_rank_00/
│       └── latest_checkpointed_iteration.txt
├── dataset/                    # 原始数据
│   └── train.parquet
├── finetune_dataset/           # 预处理后的数据
│   ├── alpaca_packed_input_ids_document.bin
│   └── alpaca_packed_input_ids_document.idx
├── lora_output/                # 训练输出
│   └── iter_XXXXXX/
└── examples/mcore/             # 示例脚本
    ├── qwen25/
    ├── qwen3/
    ├── llama3/
    └── deepseek/
```

## 推理验证

训练完成后用生成脚本验证模型质量：

```bash
# 基础模型推理
bash examples/mcore/qwen25/generate_qwen25_7b_ptd.sh

# LoRA 模型推理（需指定 LoRA 参数）
python inference.py \
    --use-mcore-models \
    --load ./model_weights/qwen25_7b_mcore/ \
    --lora-load ./lora_output/ \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --tokenizer-name-or-path ./model_from_hf/Qwen2.5-7B-Instruct/ \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --task greedy_search \
    --max-new-tokens 512 \
    --use-kv-cache \
    --use-flash-attn
```

## 评估基准

| 基准 | 领域 | 脚本 |
|------|------|------|
| MMLU | 多学科知识 | `evaluate_*_mmlu.sh` |
| GSM8K | 数学推理 | `evaluate_*_gsm8k.sh` |
| C-Eval | 中文知识 | `evaluate_*_ceval.sh` |
| HumanEval | 代码生成 | `evaluate_*_humaneval.sh` |
| BBH | 推理能力 | `evaluate_*_bbh.sh` |
| BoolQ | 阅读理解 | `evaluate_*_boolq.sh` |

```bash
# 评估参数
--task mmlu \
--data-path ./eval_data/mmlu/test \
--evaluation-batch-size 8 \
--max-new-tokens 256 \
--prompt-type qwen            # 微调模型需指定
```

## 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| Shape mismatch 加载失败 | TP/PP 或模型参数不一致 | 对照参数一致性表 |
| NaN loss | loss scale 不匹配或缺少 bias 参数 | 检查 `--initial-loss-scale`、`--add-qkv-bias` |
| `.idx/.bin not found` | 入口脚本错误或数据路径含后缀 | 用 `posttrain_gpt.py`，路径为前缀 |
| 训练卡住 | HCCL 超时 | `export HCCL_CONNECT_TIMEOUT=1800` |
| MG→HF 缺少 config.json | MG→HF 只输出权重 | 手动复制 config/tokenizer |
| 推理输出乱码 | `--prompt-type` 不匹配 | 与训练时一致 |
| OOM | 批次太大或未启用优化 | 减 MBS/GBS，启用 Flash Attention |

## 官方参考

- [快速开始](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/quick_start.md)
- [安装指南](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/install_guide.md)
- [预训练文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/pretrain/)
- [微调文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/finetune/)
- [权重转换文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/checkpoint/)
