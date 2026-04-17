# MindSpeed-LLM 训练配置完整参考

## 并行策略详解

### 张量并行（TP）

将模型层内权重按列/行切分到多卡：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--tensor-model-parallel-size` | TP 度 | 1-8 |
| `--sequence-parallel` | 序列并行（减少激活内存） | 推荐开启 |

### 流水线并行（PP）

将模型层分配到不同阶段：

| 参数 | 说明 |
|------|------|
| `--pipeline-model-parallel-size` | PP 度 |
| `--num-layers-per-virtual-pipeline-stage` | VPP 每阶段层数（减少气泡） |
| `--num-layer-list` | 动态 PP：自定义每阶段层数 |

> VPP 和动态 PP 不能同时使用。

### 上下文并行（CP）

长序列切分到多卡：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--context-parallel-size` | CP 度 | 1 |
| `--context-parallel-algo` | 算法 | `ulysses_cp_algo` |
| `--use-cp-send-recv-overlap` | 通信重叠 | 否 |
| `--attention-mask-type` | 注意力掩码类型 | `causal` |

算法选择：
- `ulysses_cp_algo`：默认，适合通用场景
- `megatron_cp_algo`：Ring Attention，适合超长序列

### 专家并行（EP）— MoE 专用

| 参数 | 说明 |
|------|------|
| `--expert-model-parallel-size` | EP 度 |
| `--moe-grouped-gemm` | MoE 分组 GEMM 优化 |

### 2D 张量并行（高级）

| 参数 | 说明 |
|------|------|
| `--tp-2d` | 启用 2D TP |
| `--tp-x` | X 轴切分数 |
| `--tp-y` | Y 轴切分数，TP = X × Y |
| `--enable-overlap-ag-with-matmul` | AllGather 与 matmul 重叠 |
| `--enable-overlap-matmul-with-rs` | ReduceScatter 与 matmul 重叠 |

### 推荐并行配置

| 模型规模 | NPU 数 | TP | PP | CP | 备注 |
|----------|--------|----|----|----|----|
| < 3B | 1-8 | 1 | 1 | 1 | 纯 DP |
| 7B-14B | 8 | 1-2 | 1-4 | 1 | LoRA 可 TP=1 |
| 32B-72B | 8-16 | 4-8 | 2-4 | 1-2 | 长序列加 CP |
| 100B+ | 16+ | 8 | 4+ | 2-4 | 考虑 2D-TP |

## 优化器与学习率

### 优化器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--optimizer` | 优化器类型 | `adam` |
| `--lr` | 初始学习率 | 1e-5 |
| `--min-lr` | 最小学习率 | 1e-6 |
| `--weight-decay` | 权重衰减 | 0.01 |
| `--adam-beta1` | Adam β1 | 0.9 |
| `--adam-beta2` | Adam β2 | 0.95 |
| `--adam-eps` | Adam ε | 1e-8 |
| `--clip-grad` | 梯度裁剪 | 1.0 |

### 学习率调度

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lr-decay-style` | 调度策略 | `cosine` |
| `--lr-warmup-fraction` | 预热比例 | 0.03 |
| `--lr-decay-iters` | 衰减迭代数 | 等于 train-iters |

### 分布式优化器

```bash
--use-distributed-optimizer    # 将优化器状态分散到各卡，节省显存
```

> 多卡训练始终推荐。断点续训时必须使用。

## 混合精度

| 参数 | 说明 |
|------|------|
| `--bf16` | BFloat16 精度（推荐） |
| `--fp16` | Float16 精度 |
| `--params-dtype` | 权重精度 |
| `--o2-optimizer` | 半精度优化器（节省 4N/DP 字节） |
| `--o2-gradient` | 半精度梯度（需同时用 `--no-gradient-accumulation-fusion`） |

## 融合算子（性能优化）

| 参数 | 说明 | 推荐 |
|------|------|------|
| `--use-flash-attn` | Flash Attention | 始终开启 |
| `--use-fused-rmsnorm` | 融合 RMSNorm | 始终开启 |
| `--use-fused-swiglu` | 融合 SwiGLU | 始终开启 |
| `--use-fused-rotary-pos-emb` | 融合 RoPE | 推荐开启 |
| `--use-mc2` | MC2 通信计算重叠 | 高级场景 |

## 重计算（激活检查点）

| 参数 | 说明 |
|------|------|
| `--recompute-granularity` | 粒度：`selective`（推荐）或 `full` |
| `--recompute-method` | 方法：`block`（连续层）或 `uniform`（均匀分布） |
| `--recompute-num-layers` | 重计算的层数 |

> `selective` 仅重计算注意力，计算开销小。`full` 重计算所有激活，显存节省最大但速度慢 20-30%。

## Checkpoint 配置

### 保存参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--save` | 保存路径 | 必需 |
| `--save-interval` | 保存间隔（步数） | 500-1000 |
| `--no-save-optim` | 不保存优化器状态 | 微调时可用 |
| `--no-save-rng` | 不保存随机状态 | 微调时可用 |

### 加载参数

| 参数 | 说明 |
|------|------|
| `--load` | 加载路径 |
| `--no-load-optim` | 不加载优化器状态 |
| `--no-load-rng` | 不加载随机状态 |
| `--finetune` | 微调模式（重置训练状态） |

### 断点续训注意事项

- 预训练续训 **不要** 加 `--finetune`（会跳过优化器加载）
- 必须使用 `--use-distributed-optimizer`
- 保持 TP/PP 配置不变

## 数据加载参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--data-path` | 数据前缀路径 | 必需 |
| `--tokenizer-type` | 分词器类型 | `PretrainedFromHF` |
| `--tokenizer-name-or-path` | 分词器路径 | HF 模型目录 |
| `--seq-length` | 序列长度 | 2048-32768 |
| `--micro-batch-size` | 微批大小 | 1-4 |
| `--global-batch-size` | 全局批大小 | 8-64 |

### 序列打包

| 参数 | 说明 |
|------|------|
| `--pack` | 数据预处理时启用打包 |
| `--reset-attention-mask` | 训练时重置注意力掩码 |
| `--reset-position-ids` | 训练时重置位置 ID |

## 微调专用参数

### SFT（指令微调）

```bash
--finetune
--stage sft
--is-instruction-dataset
--prompt-type qwen           # 对话模板
--no-load-optim
--no-load-rng
```

### LoRA

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--lora-r` | LoRA 秩 | 8-16 |
| `--lora-alpha` | 缩放因子 | 16-32（通常 2×r） |
| `--lora-fusion` | CCLoRA 通信计算重叠 | 推荐 |
| `--lora-target-modules` | 目标模块 | `linear_qkv linear_proj linear_fc1 linear_fc2` |

### QLoRA

| 参数 | 说明 |
|------|------|
| `--qlora` | 启用 QLoRA |
| `--qlora-nf4` | 权重转换时使用 NF4 量化 |

> QLoRA 使用 4-bit 量化基础模型 + LoRA 适配器，显存占用约为 LoRA 的 35%。
> **注意**：QLoRA 不支持 `--lora-fusion`，开启无性能收益。

## DPO 偏好对齐

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stage dpo` | DPO 训练阶段 | - |
| `--is-pairwise-dataset` | 使用 pairwise 数据格式 | - |
| `--dpo-beta` | KL 正则系数 | 0.1 |
| `--dpo-loss-type` | 损失类型 | `sigmoid` |
| `--dpo-label-smoothing` | 标签平滑 | 0.0 |
| `--pref-ftx` | SFT 损失权重 | 0.0 |
| `--ref-model` | 参考模型路径 | None（用训练模型） |

## 环境变量

| 变量 | 推荐值 | 说明 |
|------|--------|------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | 必需 |
| `HCCL_CONNECT_TIMEOUT` | `1800` | HCCL 连接超时秒数（默认 120） |
| `ASCEND_LAUNCH_BLOCKING` | `0` | 0=异步（性能好，内存消耗高，有 OOM 风险）；1=同步（调试用，性能差） |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 减少内存碎片 |
| `TASK_QUEUE_ENABLE` | `2` | 任务队列优化 |
| `COMBINED_ENABLE` | `1` | 相邻算子融合 |

## 显存优化策略总结

| 技术 | 参数 | 显存节省 | 速度影响 |
|------|------|----------|----------|
| Flash Attention | `--use-flash-attn` | 中间张量 | 无 |
| 选择性重计算 | `--recompute-granularity selective` | 激活 | +10% |
| 完全重计算 | `--recompute-granularity full` | 大量激活 | +20-30% |
| 分布式优化器 | `--use-distributed-optimizer` | 优化器状态 | 通信开销小 |
| O2 优化器 | `--o2-optimizer` | 优化器+梯度 | 精度略降 |
| 序列打包 | `--pack` | 填充浪费 | 预处理慢 |
| QLoRA | `--qlora` | 约 65% 权重 | 精度损失 |
| 梯度累积 | `--gradient-accumulation-steps` | 峰值批次 | 训练时间长 |

## 性能分析（Profiling）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--profile` | 启用性能分析 | 否 |
| `--profile-step-start` | 开始步骤 | 0 |
| `--profile-step-end` | 结束步骤 | -1（训练结束） |
| `--profile-level` | 详细程度 | `level0` |
| `--profile-with-memory` | 内存分析 | 否 |
| `--profile-save-path` | 输出目录 | `./profile` |

## 多机训练配置

```bash
# 节点 0（主节点）
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 MASTER_PORT=6000 \
  torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
  --master_addr 192.168.1.100 --master_port 6000 posttrain_gpt.py ...

# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 MASTER_PORT=6000 \
  torchrun --nproc_per_node 8 --nnodes 2 --node_rank 1 \
  --master_addr 192.168.1.100 --master_port 6000 posttrain_gpt.py ...
```

### 多机检查清单

- [ ] SSH 免密登录配置
- [ ] 所有节点 CANN/torch_npu 版本一致
- [ ] NPU 设备健康（`npu-smi info`）
- [ ] HCCL 通信测试通过
- [ ] 模型和数据路径一致
- [ ] TP/PP/EP 配置一致

## 常见参数冲突

| 冲突 | 解决 |
|------|------|
| `--o2-gradient` | 必须同时用 `--no-gradient-accumulation-fusion` |
| `--tp-2d` + `--sequence-parallel` | 不兼容，二选一 |
| VPP + 动态 PP | 不兼容，二选一 |
| `--finetune` + 预训练续训 | 续训不加 `--finetune` |
| `--qlora` + `--train-from-scratch` | 不兼容 |

## FSDP2 后端参数

| 参数 | 说明 |
|------|------|
| `--fsdp2` | 启用 FSDP2 后端 |
| `--fsdp2-reshard-after-forward` | 前向后重分片参数 |

## 官方参考

- [预训练文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/pretrain/)
- [微调文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/finetune/)
- [DPO 文档](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/solutions/dpo/)
- [性能优化](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/features/)
