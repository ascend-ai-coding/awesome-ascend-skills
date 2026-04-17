# MFU计算参考文档

## 一、MFU基本概念

### 1.1 什么是MFU？

MFU (Model FLOPs Utilization) 是衡量大模型训练效率的核心指标，表示在一个训练步骤中，GPU或其他加速器实际执行的有效模型计算FLOPs与该设备理论最大FLOPs的比值。

**公式**：
```
MFU = 有效模型计算FLOPs / 硬件理论峰值FLOPs
```

### 1.2 MFU的意义

- **反映硬件利用率**：MFU越高，说明硬件算力利用越充分
- **评估训练效率**：可以横向对比不同训练系统的效率
- **指导性能优化**：帮助定位性能瓶颈

### 1.3 MFU评估标准

| MFU范围 | 评估等级 | 说明 |
|---------|----------|------|
| ≥40% | 优秀 | 硬件利用率很高 |
| 30-40% | 良好 | 硬件利用率较好 |
| 20-30% | 一般 | 有优化空间 |
| <20% | 需要优化 | 存在明显性能问题 |

---

## 二、FLOPs计算方法

### 2.1 Attention部分

**标准Attention**：
```
Q, K, V线性层: 8BSH²
  - Q: 2BSH²
  - K: 2BSH²
  - V: 2BSH²
  - Output: 2BSH²

Attention计算: 4BS²H

总计: 8BSH² + 4BS²H
```

**GQA (Grouped Query Attention)**：
```
Q, K, V线性层: BSH² × (4 + kv_heads/num_heads)
  - Q: 2BSH²
  - K: 2BSH² × (kv_heads/num_heads)
  - V: 2BSH² × (kv_heads/num_heads)
  - Output: 2BSH²

Attention计算: 4BS²H

总计: BSH² × (4 + kv_heads/num_heads) + 4BS²H
```

### 2.2 FFN部分

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
FLOPs = 6BSH × activated_expert_num × expert_intermediate_size
```

### 2.3 Logits部分

```
FLOPs = 6BSH × vocab_size
```

### 2.4 总FLOPs计算

**单层FLOPs**：
```
layer_flops = attention_flops + ffn_flops
```

**模型FLOPs**：
```
model_flops = layer_flops × num_layers
```

**单步训练FLOPs**：
```
step_flops = 3 × model_flops + logits_flops
```

**说明**：假设反向传播FLOPs = 2 × 前向传播FLOPs

---

## 三、MFU计算流程

### 3.1 计算步骤

1. **计算单步FLOPs**：
   ```
   step_flops = calculate_step_flops(batch_size)
   ```

2. **计算有效FLOPS**：
   ```
   effective_flops = step_flops / step_time
   ```

3. **计算MFU**：
   ```
   mfu = effective_flops / (hardware_peak_flops × 1e12)
   ```

### 3.2 完整公式

```
MFU = (step_flops / step_time) / (hardware_peak_flops × 1e12)
```

---

## 四、其他性能指标

### 4.1 单卡吞吐量

```
Throughput = (gbs × seq_length) / (step_time × num_gpus)
```

单位：tokens/s/GPU

### 4.2 集群吞吐量

```
Cluster_Throughput = (Throughput × num_gpus × 3600 × 24) / 10^12
```

单位：T tokens/day

---

## 五、硬件峰值FLOPS参考

### 5.1 NVIDIA GPU

| GPU型号 | FP16/BF16 TFLOPS | 说明 |
|---------|------------------|------|
| A100 | 312 | 标准版 |
| A100-80G | 312 | 80GB显存版 |
| H100 | 989 | 最新旗舰 |
| H100-80G | 989 | 80GB显存版 |
| V100 | 125 | 上一代旗舰 |
| RTX4090 | 165 | 消费级旗舰 |

### 5.2 Ascend NPU

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

## 六、预定义模型配置

### 6.1 LLaMA系列

**LLaMA-7B**：
- hidden_size: 4096
- num_layers: 32
- vocab_size: 32000
- num_attention_heads: 32
- intermediate_size: 11008
- ffn_type: swiglu

**LLaMA-13B**：
- hidden_size: 5120
- num_layers: 40
- vocab_size: 32000
- num_attention_heads: 40
- intermediate_size: 13824
- ffn_type: swiglu

**LLaMA-70B**：
- hidden_size: 8192
- num_layers: 80
- vocab_size: 32000
- num_attention_heads: 64
- num_key_value_heads: 8 (GQA)
- intermediate_size: 28672
- ffn_type: swiglu

### 6.2 Qwen系列

**Qwen-7B**：
- hidden_size: 4096
- num_layers: 32
- vocab_size: 151936
- num_attention_heads: 32
- intermediate_size: 11008
- ffn_type: swiglu

**Qwen-72B**：
- hidden_size: 8192
- num_layers: 80
- vocab_size: 151936
- num_attention_heads: 64
- num_key_value_heads: 8
- intermediate_size: 24576
- ffn_type: swiglu

### 6.3 MoE模型

**Mixtral-8x7B**：
- hidden_size: 4096
- num_layers: 32
- vocab_size: 32000
- num_attention_heads: 32
- intermediate_size: 14336
- ffn_type: swiglu
- is_moe: True
- num_experts: 8
- num_experts_per_tok: 2
- expert_intermediate_size: 14336

---

## 七、使用示例

### 7.1 使用预定义模型

```python
from mfu_calculator import MODEL_CONFIGS, MFUCalculator, TrainingConfig

# 使用LLaMA-7B配置
model_config = MODEL_CONFIGS["llama-7b"]

# 训练配置
training_config = TrainingConfig(
    batch_size=512,
    micro_batch_size=4,
    num_gpus=128,
    seq_length=2048,
    step_time=2.5,
    hardware_peak_flops=312,
    hardware_name="A100",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
print(calculator.generate_report())
```

### 7.2 自定义模型

```python
from mfu_calculator import ModelConfig, MFUCalculator, TrainingConfig

# 自定义模型配置
model_config = ModelConfig(
    hidden_size=6144,
    num_layers=48,
    vocab_size=128000,
    seq_length=4096,
    num_attention_heads=48,
    num_key_value_heads=8,
    intermediate_size=16384,
    ffn_type="swiglu",
)

# 训练配置
training_config = TrainingConfig(
    batch_size=1024,
    micro_batch_size=8,
    num_gpus=256,
    seq_length=4096,
    step_time=4.5,
    hardware_peak_flops=313,
    hardware_name="Ascend910B",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
mfu = calculator.calculate_mfu()
print(f"MFU: {mfu*100:.2f}%")
```

### 7.3 MoE模型

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
    batch_size=256,
    micro_batch_size=2,
    num_gpus=64,
    seq_length=2048,
    step_time=3.2,
    hardware_peak_flops=989,
    hardware_name="H100",
)

# 计算MFU
calculator = MFUCalculator(model_config, training_config)
print(calculator.generate_report())
```

---

## 八、注意事项

### 8.1 计算准确性

1. **序列长度影响**：Attention计算的FLOPs与序列长度平方成正比，长序列会显著增加计算量
2. **GQA优化**：使用GQA可以减少KV cache和计算量
3. **MoE稀疏性**：MoE模型只计算激活的专家，实际计算量取决于激活专家数

### 8.2 性能优化建议

1. **提高MFU的方法**：
   - 优化通信效率（使用NCCL/HCCL优化）
   - 减少内存碎片
   - 使用混合精度训练
   - 优化数据加载pipeline

2. **常见性能瓶颈**：
   - 通信开销过大
   - 内存访问延迟
   - 数据加载瓶颈
   - CPU-GPU同步

### 8.3 硬件差异

1. **GPU vs NPU**：
   - 峰值算力不同
   - 内存带宽差异
   - 通信库差异（NCCL vs HCCL）

2. **不同代际硬件**：
   - A100 vs H100：H100算力约为A100的3倍
   - Ascend910 vs Ascend910B：B版本算力提升约20%

---

## 九、参考资料

1. [大语言模型训练-LLM：Dense & MOE模型 MFU 计算](https://zhuanlan.zhihu.com/p/1918630175156985869)
2. [PyTorch FLOPs计算](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
3. [Transformer论文](https://arxiv.org/abs/1706.03762)
4. [MoE论文](https://arxiv.org/abs/2101.03961)
