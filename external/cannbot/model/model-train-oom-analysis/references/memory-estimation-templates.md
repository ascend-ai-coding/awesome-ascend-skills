# NPU 训练静态内存估算模板

本文档提供训练内存估算的通用模板和常见模型的参考值，用于在训练前评估内存需求或诊断 OOM 问题。

## 内存组成总览

```
训练峰值内存 ≈ ① 参数内存 + ② 梯度内存 + ③ 优化器状态 + ④ 激活内存 + ⑤ 临时缓冲区
```

## 各组成部分估算方法

### ① 参数内存

```
Dense 模型:  P × dtype_bytes / (TP × PP)
MoE 模型:   P_dense × dtype_bytes / (TP × PP) + P_expert × dtype_bytes / (TP × EP × PP)
```

- `P` = 参数总量（如 8B = 80亿）
- `dtype_bytes` = 数据类型字节数（bf16=2, fp32=4, fp8=1）

### ② 梯度内存

```
≈ 参数内存  (梯度与参数同 dtype、同切分方式)
```

### ③ 优化器状态内存

**AdamW**（最常见）：

```
= P / 并行度 × 4B × 2    (fp32 的一阶矩 m 和二阶矩 v)
```

- 若启用 FSDP (`data_parallel_shard_degree > 1`): 状态在 DP 维度上进一步分片
- 若启用 `swap_optimizer = true`: 优化器状态卸载到 CPU，NPU 上几乎为 0

### ④ 激活内存

激活内存与 `batch_size (B)`、`seq_len (S)`、`hidden_dim (H)`、`num_layers (L)` 和激活重计算策略强相关：

| 重计算策略 | `[activation_checkpoint]` 配置 | 激活内存估算 |
|-----------|-------------------------------|------------|
| 无重计算 | `mode = "none"` | ≈ `B × S × H × L × dtype_bytes × K`（K≈10~15） |
| 全量重计算 | `mode = "full"` | ≈ `B × S × H × dtype_bytes`（仅保存模型输入） |
| 选择性重计算 | `mode = "selective"` | 介于两者之间，取决于 `selective_ac_option` |

其中 `K` 是每个 Transformer 层保存的中间张量数量系数（取决于模型架构）。

### ⑤ 临时缓冲区

```
≈ 总内存 × 5%~15%
```

包括：HCCL 通信缓冲、NPUWorkspaceAllocator、梯度聚合桶、临时计算空间等。

## 常见模型估算参考

### Llama3 8B (bf16, AdamW)

| 组成部分 | 计算方式 | 单卡 (无并行) | TP=8 |
|---------|---------|-------------|------|
| 参数内存 | 8B × 2B | 16 GB | 2 GB |
| 梯度内存 | ≈ 参数内存 | 16 GB | 2 GB |
| 优化器状态 (m+v) | 8B × 4B × 2 | 64 GB | 8 GB |
| 激活内存 (selective AC) | 取决于 B×S | 变动 | 变动 |
| **静态总计 (不含激活)** | | **96 GB** | **12 GB** |

**配置示例**（llama3-8b，参数示意，实际配置见 `config_registry.py`）：

- `local_batch_size = 2`, `seq_len = 2048`
- `activation_checkpoint.mode = "selective"`
- `swap_optimizer = true` → 优化器内存卸载到 CPU

### DeepSeek V32 671B (bf16, AdamW, MoE)

| 组成部分 | 说明 | 估算值 |
|---------|------|--------|
| Dense 参数 | ~14B dense 参数 | 28 GB / (TP × PP) |
| Expert 参数 | ~657B expert 参数 (256 experts) | 1314 GB / (TP × EP × PP) |
| 优化器状态 | 取决于 FSDP 分片、swap_optimizer | 取决于具体并行策略 |
| 激活内存 | 强依赖 AC 策略、B、S | 需实际测量 |

**配置示例**（DeepSeek-V3.2 671B，参数示意，实际配置见 `config_registry.py`）：

- `local_batch_size = 4`, `seq_len = 2048`
- `activation_checkpoint.mode = "selective"`
- 256 experts, `expert_parallel_degree` / `expert_tensor_parallel_degree` 需合理设置

> [!NOTE]
> 以上为理论估算参考值。实际内存占用受内存碎片、框架开销、通信缓冲等因素影响，建议结合 memory snapshot 实测验证。

## 内存预算检查公式

```
估算总内存 ≤ NPU 总显存 × 0.95  (预留 5% 安全余量)
```

若估算结果超出预算，按以下优先级调整（参见 SKILL.md Step 4 优化建议矩阵）：

1. 启用 activation checkpoint
2. 启用 swap_optimizer
3. 增大并行度（TP/PP/EP）
4. 减小 batch_size / seq_len
