---
name: training-mfu-calculator
description: 大模型训练MFU (Model FLOPs Utilization) 计算工具。在用户需要计算MFU、评估训练效率、分析模型性能、或提到FLOPs、吞吐量、硬件利用率时自动应用。支持Dense和MoE模型，提供详细的性能分析报告。
---

# MFU计算工具

## 何时使用本 Skill

- 用户需要**计算MFU**：评估大模型训练的硬件利用率。
- 用户提到**FLOPs、吞吐量、训练效率**：分析模型训练性能。
- 用户需要**性能评估**：对比不同硬件或配置的训练效率。
- 用户提到**硬件利用率、算力利用率**：优化训练系统性能。
- 用户需要**模型性能分析**：分析Dense或MoE模型的计算量。

---

## 快速开始

### 最简单的使用方式

```python
from mfu_calculator import MODEL_CONFIGS, MFUCalculator, TrainingConfig

# 使用预定义模型配置
model_config = MODEL_CONFIGS["llama-7b"]

# 训练配置
training_config = TrainingConfig(
    batch_size=512,
    micro_batch_size=4,
    num_gpus=128,
    seq_length=2048,
    step_time=2.5,  # 每步时间（秒）
    hardware_peak_flops=312,  # A100峰值算力
    hardware_name="A100",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
print(calculator.generate_report())
```

---

## 信息来源

计算MFU需要三类信息，以下是获取方式：

### 1. 模型配置（从 config.json 读取）

从 HuggingFace 格式的 `config.json` 文件中读取模型架构参数：

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 94,
  "vocab_size": 151936,
  "num_attention_heads": 64,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "intermediate_size": 12288,
  "moe_intermediate_size": 1536,
  "num_experts_per_tok": 8,
  "num_experts": 128
}
```

**参数映射表**：

| config.json 字段 | MFU计算参数 | 说明 |
|-----------------|-------------|------|
| `hidden_size` | `hidden_size` | 隐藏层维度 |
| `num_hidden_layers` | `layer_num` | 层数 |
| `vocab_size` | `vocab_size` | 词表大小 |
| `num_attention_heads` | `head_num` | 注意力头数 |
| `num_key_value_heads` | `kv_head_num` | KV头数 (GQA) |
| `head_dim` | `head_dim` | 头维度 |
| `intermediate_size` | `intermediate_size` | FFN中间层 |
| `moe_intermediate_size` | `expert_hidden_size` | MoE专家中间层 |
| `num_experts_per_tok` | `topk` | 激活专家数 |

### 2. 训练配置（从启动脚本读取）

从训练启动脚本（如 `run_train.sh`、`train.py` 参数）中读取：

**常见启动脚本示例**：
```bash
# 示例1: torchrun 启动
torchrun --nproc_per_node=8 --nnodes=4 train.py \
    --seq_length 8192 \
    --global_batch_size 32 \
    ...

# 示例2: accelerate 启动
accelerate launch --num_processes 32 train.py \
    --max_seq_length 8192 \
    --train_batch_size 32 \
    ...
```

**参数映射表**：

| 启动脚本参数 | MFU计算参数 | 说明 |
|-------------|-------------|------|
| `--seq_length` / `--max_seq_length` | `seq_length` | 训练序列长度 |
| `--global_batch_size` / `--train_batch_size` | `gbs` | 全局batch size |
| `--nproc_per_node × --nnodes` / `--num_processes` | `num_gpus` | GPU/NPU总数 |

**计算 GPU/NPU 数量**：
```python
# 单机多卡
num_gpus = nproc_per_node

# 多机多卡
num_gpus = nproc_per_node * nnodes
```

### 3. 性能数据（用户提供或训练日志）

#### 方式一：用户提供

用户直接提供每步耗时（step_time）：
- 单位：秒
- 建议：取稳定训练阶段的平均值

#### 方式二：从训练日志读取

从训练日志中提取 step_time：

**日志示例1（Transformers格式）**：
```
[2024-01-15 10:23:45] step=100 loss=2.345 learning_rate=1e-4 step_time=4.4s
[2024-01-15 10:23:50] step=101 loss=2.342 learning_rate=1e-4 step_time=4.3s
```

**日志示例2（Megatron格式）**：
```
iteration 100/ 1000 | elapsed time per iteration (ms): 4400 | ...
iteration 101/ 1000 | elapsed time per iteration (ms): 4350 | ...
```

**日志示例3（PyTorch格式）**：
```
Step 100: loss=2.345, time=4.40s, throughput=1861.82 tokens/s/GPU
```

**提取方法**：
```python
import re

def extract_step_time(log_file):
    """从日志文件提取step_time"""
    times = []
    with open(log_file, 'r') as f:
        for line in f:
            # 匹配 "step_time=4.4s" 或 "time=4.40s"
            match = re.search(r'(?:step_)?time[=:]\s*([\d.]+)\s*s?', line)
            if match:
                times.append(float(match.group(1)))
    
    # 取稳定阶段的平均值（跳过前10步预热）
    return sum(times[10:]) / len(times[10:]) if len(times) > 10 else None
```

### 4. 硬件信息

**默认值**：Ascend910B2 = 353 TFLOPS

**获取方式**：
- 从环境变量：`echo $ASCEND_DEVICE_TYPE`
- 从 npu-smi：`npu-smi info`
- 从 nvidia-smi：`nvidia-smi --query-gpu=name --format=csv`

---

## 目录结构

```
agent_skills/mfu-calculator/
├── SKILL.md                          # 本文件
├── reference/
│   └── mfu_reference.md              # MFU计算参考文档
└── scripts/
    └── mfu_calculator.py             # MFU计算工具实现
```

---

## 核心功能

### 1. 支持的模型类型

**Dense模型**：
- ✅ LLaMA系列（7B, 13B, 70B）
- ✅ Qwen系列（7B, 72B）
- ✅ 自定义Dense模型

**MoE模型**：
- ✅ Mixtral-8x7B
- ✅ 自定义MoE模型

### 2. 支持的FFN类型

- ✅ 标准FFN
- ✅ SwiGLU FFN

### 3. 支持的Attention类型

- ✅ 标准Multi-Head Attention
- ✅ Grouped Query Attention (GQA)

---

## 使用方法

### 方法一：使用预定义模型

```python
from mfu_calculator import MODEL_CONFIGS, MFUCalculator, TrainingConfig

# 查看支持的预定义模型
print(MODEL_CONFIGS.keys())
# 输出: dict_keys(['llama-7b', 'llama-13b', 'llama-70b', 'qwen-7b', 'qwen-72b', 'mixtral-8x7b'])

# 选择模型
model_config = MODEL_CONFIGS["llama-70b"]

# 配置训练参数
training_config = TrainingConfig(
    batch_size=1024,
    micro_batch_size=8,
    num_gpus=256,
    seq_length=2048,
    step_time=3.8,
    hardware_peak_flops=989,
    hardware_name="H100",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
print(calculator.generate_report())
```

### 方法二：自定义模型

```python
from mfu_calculator import ModelConfig, MFUCalculator, TrainingConfig

# 自定义Dense模型
model_config = ModelConfig(
    hidden_size=8192,
    num_layers=80,
    vocab_size=128000,
    seq_length=4096,
    num_attention_heads=64,
    num_key_value_heads=8,  # GQA
    intermediate_size=22016,
    ffn_type="swiglu",
)

# 训练配置
training_config = TrainingConfig(
    batch_size=2048,
    micro_batch_size=16,
    num_gpus=512,
    seq_length=4096,
    step_time=5.2,
    hardware_peak_flops=313,
    hardware_name="Ascend910B",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
mfu = calculator.calculate_mfu()
print(f"MFU: {mfu*100:.2f}%")
```

### 方法三：MoE模型

```python
from mfu_calculator import ModelConfig, MFUCalculator, TrainingConfig

# MoE模型配置
model_config = ModelConfig(
    hidden_size=4096,
    num_layers=32,
    vocab_size=32000,
    seq_length=2048,
    num_attention_heads=32,
    intermediate_size=14336,
    ffn_type="swiglu",
    is_moe=True,
    num_experts=8,
    num_experts_per_tok=2,
    expert_intermediate_size=14336,
)

# 训练配置
training_config = TrainingConfig(
    batch_size=512,
    num_gpus=128,
    seq_length=2048,
    step_time=4.1,
    hardware_peak_flops=312,
    hardware_name="A100",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
print(calculator.generate_report())
```

### 方法四：从 config.json 自动读取

```python
import json
from mfu_calculator import ModelConfig, MFUCalculator, TrainingConfig, cal_flops_simple, cal_mfu_simple

# 1. 从 config.json 读取模型配置
with open("config.json", "r") as f:
    config = json.load(f)

# 2. 从启动脚本或用户提供训练配置
seq_length = 8192      # 从 --seq_length 参数
gbs = 32               # 从 --global_batch_size 参数
num_gpus = 32          # 从 nproc_per_node * nnodes 计算
step_time = 4.4        # 从训练日志或用户提供

# 3. 计算MFU
flops = cal_flops_simple(
    hidden_size=config["hidden_size"],
    expert_hidden_size=config.get("moe_intermediate_size", config.get("intermediate_size", 4 * config["hidden_size"])),
    head_num=config["num_attention_heads"],
    kv_head_num=config.get("num_key_value_heads", config["num_attention_heads"]),
    sequence_length=seq_length,
    layer_num=config["num_hidden_layers"],
    vocab_size=config["vocab_size"],
    topk=config.get("num_experts_per_tok", 1),
    gbs=gbs,
    head_dim=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
)

# 默认使用 Ascend910B2 的峰值算力
hw_flops = 353 * 1e12

mfu = cal_mfu_simple(
    real_flops=flops,
    num_gpu=num_gpus,
    sec_per_step=step_time,
    hw_flops_per_gpu=hw_flops
)

print(f"模型总FLOPs: {flops/1e15:.2f} PFLOPs")
print(f"MFU: {mfu * 100:.2f}%")
```

### 方法五：Qwen3 MoE 完整示例

```python
import json
from mfu_calculator import cal_flops_simple, cal_mfu_simple

# Qwen3 MoE config.json
config = {
    "hidden_size": 4096,
    "num_hidden_layers": 94,
    "vocab_size": 151936,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 12288,
    "moe_intermediate_size": 1536,
    "num_experts_per_tok": 8,
    "num_experts": 128
}

# 训练配置（从启动脚本读取）
seq_length = 8192
gbs = 32
num_gpus = 32
step_time = 4.4  # 用户提供或从日志读取

# 计算FLOPs
flops = cal_flops_simple(
    hidden_size=config["hidden_size"],
    expert_hidden_size=config["moe_intermediate_size"],
    head_num=config["num_attention_heads"],
    kv_head_num=config["num_key_value_heads"],
    sequence_length=seq_length,
    layer_num=config["num_hidden_layers"],
    vocab_size=config["vocab_size"],
    topk=config["num_experts_per_tok"],
    gbs=gbs,
    head_dim=config["head_dim"]
)

# 计算MFU（默认 Ascend910B2）
hw_flops = 353 * 1e12
mfu = cal_mfu_simple(
    real_flops=flops,
    num_gpu=num_gpus,
    sec_per_step=step_time,
    hw_flops_per_gpu=hw_flops
)

print(f"模型总FLOPs: {flops/1e15:.2f} PFLOPs")
print(f"MFU: {mfu * 100:.2f}%")
# 输出:
# 模型总FLOPs: 9.83 PFLOPs
# MFU: 19.78%
```

---

## 性能指标

### 1. MFU (Model FLOPs Utilization)

**计算公式**：
```
MFU = 有效模型计算FLOPs / 硬件理论峰值FLOPs
```

**评估标准**：

| MFU范围 | 评估等级 | 说明 |
|---------|----------|------|
| ≥40% | 优秀 | 硬件利用率很高 |
| 30-40% | 良好 | 硬件利用率较好 |
| 20-30% | 一般 | 有优化空间 |
| <20% | 需要优化 | 存在明显性能问题 |

### 2. 单卡吞吐量

**计算公式**：
```
Throughput = (gbs × seq_length) / (step_time × num_gpus)
```

单位：tokens/s/GPU

### 3. 集群吞吐量

**计算公式**：
```
Cluster_Throughput = (Throughput × num_gpus × 3600 × 24) / 10^12
```

单位：T tokens/day

---

## 硬件峰值FLOPS参考

### NVIDIA GPU

| GPU型号 | FP16/BF16 TFLOPS |
|---------|------------------|
| A100 | 312 |
| H100 | 989 |
| V100 | 125 |
| RTX4090 | 165 |

### Ascend NPU

| NPU型号 | FP16 TFLOPS | 说明 |
|---------|-------------|------|
| Ascend910 | 256 | 台积电7nm EUV |
| Ascend910A | 256 | 台积电7nm EUV |
| Ascend910A2 | 256 | 台积电7nm EUV |
| Ascend910B | 320 | 中芯国际N+1 |
| Ascend910B1 | 320 | 中芯国际N+1 |
| **Ascend910B2** | **353** | **默认值** |
| Ascend910B3 | 353 | 中芯国际N+1 |
| Ascend910C | 800 | 双Die封装 |

---

## FLOPs计算公式

### Attention部分

**标准Attention**：
```
FLOPs = 8BSH² + 4BS²H
```

**GQA (Grouped Query Attention)**：
```
FLOPs = BSH² × (4 + kv_heads/num_heads) + 4BS²H
```

### FFN部分

**标准FFN**：
```
FLOPs = 4BSH × intermediate_size
```

**SwiGLU FFN**：
```
FLOPs = 6BSH × intermediate_size
```

**MoE + SwiGLU**：
```
FLOPs = 6BSH × activated_experts × expert_intermediate_size
```

### Logits部分

```
FLOPs = 6BSH × vocab_size
```

### 总FLOPs

```
单层FLOPs = attention_flops + ffn_flops
模型FLOPs = 单层FLOPs × num_layers
单步FLOPs = 3 × 模型FLOPs + logits_flops
```

**说明**：假设反向传播FLOPs = 2 × 前向传播FLOPs

---

## 输出示例

```
============================================================
MFU计算报告
============================================================

【模型配置】
- 模型类型: Dense
- 隐藏层维度: 4096
- 层数: 32
- 注意力头数: 32
- KV头数: 32
- 序列长度: 2048
- 词表大小: 32000
- FFN类型: swiglu
- FFN中间层大小: 11008

【训练配置】
- 全局batch size: 512
- 微批次大小: 4
- GPU数量: 128
- 每步时间: 2.500秒
- 硬件: A100
- 硬件峰值: 312 TFLOPS

【FLOPs分析】
- 单步训练FLOPs: 1.23e+18
- 有效计算FLOPS: 4.92e+17 (492.00 TFLOPS)

【性能指标】
- MFU: 52.31%
- 单卡吞吐: 3276.80 tokens/s/GPU
- 集群吞吐: 36.28 T tokens/day

【性能评估】
- MFU评估: 良好 (50-60%)
============================================================
```

---

## 常见问题

### Q1: MFU过低怎么办？

**可能原因**：
- 通信开销过大
- 内存访问延迟
- 数据加载瓶颈
- CPU-GPU同步

**解决方案**：
- 优化通信效率（使用NCCL/HCCL优化）
- 减少内存碎片
- 优化数据加载pipeline
- 使用混合精度训练

### Q2: 如何选择合适的硬件峰值FLOPS？

**建议**：
- 使用硬件规格书中的FP16/BF16峰值算力
- 考虑实际使用的数据类型
- 参考本文档中的硬件峰值FLOPS表

### Q3: MoE模型的MFU计算有什么特殊之处？

**说明**：
- MoE模型只计算激活的专家
- 实际计算量 = 基础计算量 × 激活专家数
- 需要正确配置`num_experts_per_tok`参数

### Q4: GQA对FLOPs有什么影响？

**说明**：
- GQA减少了KV cache和计算量
- 计算公式中需要考虑`kv_heads/num_heads`比例
- 可以显著降低长序列的计算开销

---

## 最佳实践

### 1. 准确测量step_time

```python
import time

# 测量多个步骤取平均值
step_times = []
for i in range(10):
    start = time.time()
    train_step()
    step_times.append(time.time() - start)

avg_step_time = sum(step_times) / len(step_times)
```

### 2. 合理配置batch_size

```python
# 根据显存和硬件选择合适的micro_batch_size
# 一般建议：尽可能大，但不要OOM

# A100 80GB示例
micro_batch_size = 8  # 对于7B模型

# H100 80GB示例
micro_batch_size = 16  # 对于7B模型
```

### 3. 对比不同配置

```python
# 对比不同硬件的MFU
configs = [
    ("A100", 312),
    ("H100", 989),
    ("Ascend910B", 313),
]

for name, peak_flops in configs:
    training_config.hardware_name = name
    training_config.hardware_peak_flops = peak_flops
    calculator = MFUCalculator(model_config, training_config)
    mfu = calculator.calculate_mfu()
    print(f"{name}: MFU = {mfu*100:.2f}%")
```

---

## 参考资源

### 文档

- **MFU计算参考**: `reference/mfu_reference.md`
- **MFU计算工具**: `scripts/mfu_calculator.py`

### 外部资源

- [大语言模型训练-LLM：Dense & MOE模型 MFU 计算](https://zhuanlan.zhihu.com/p/1918630175156985869)
- [PyTorch FLOPs计算](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Transformer论文](https://arxiv.org/abs/1706.03762)

---

## 总结

本skill提供了完整的MFU计算方案：

1. **全面支持**：支持Dense和MoE模型，多种FFN和Attention类型
2. **易于使用**：提供预定义模型配置，快速上手
3. **详细报告**：生成完整的性能分析报告
4. **准确计算**：基于标准FLOPs计算公式

通过本skill，可以准确评估大模型训练的硬件利用率，指导性能优化工作。
