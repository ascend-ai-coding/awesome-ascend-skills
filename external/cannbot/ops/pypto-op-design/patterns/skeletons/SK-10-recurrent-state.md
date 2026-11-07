---
type: pattern/skeleton
title: Recurrent State Machine
description: 状态跨序列步骤传递的递归计算骨架。
tags:
- recurrent
flow_pattern:
- V
- C
- V
examples:
- GatedDeltaRule
- SumLSTM
---

## SK-10: Recurrent State Machine

**适用场景**: 序列依赖、状态跨时间步传递的算子。

**CV 排布**: 纯 V (LSTM) 或 CVC (Delta Rule)

### 泛化变体（适用算子族）

虽然当前样本只来自 `GatedDeltaRule` 和 `SumLSTM`，但本骨架的"跨 chunk 状态 + 块级递推"结构覆盖整个**序列递归算子族**，具体包括：

| 变体 | 状态形态 | 块内计算 | CV 模式 | 典型实现 |
|------|---------|---------|---------|---------|
| **LSTM / GRU** | `[hidden_dim]` 标量门控状态 | 门控乘加 + tanh/sigmoid | 纯 V | SumLSTM |
| **Gated Delta Rule** | `[key_dim, value_dim]` 矩阵状态 | 块级矩阵求逆 + 状态更新 | CVC（块求逆走 C） | GatedDeltaRule |
| **State Space Model (SSM/Mamba/RWKV)** | `[hidden_dim, state_dim]` 矩阵状态 | 离散化 SSM 递推 + element-wise | 纯 V 或 CVC | 未在仓内，可参考 |
| **Linear Attention (循环形式)** | `[head_dim, head_dim]` outer-product 状态 | 块级 outer-product + 加和 | CVC | 未在仓内，可参考 |
| **任意带"前向单向递推"的算子** | 任意 | 任意 | 任意 | — |

**统一框架**：3 层 loop（batch → 并行维 → 序列块），状态在序列 loop 外初始化、内更新、外写回；并行维（head/group）总是 `parallel=True`；序列块大小常用 `L=128` 或 `L=16`（对应 unroll 第一项）。本骨架的核心不在具体的门控形式，而在**"序列维不可并行 + 并行维必须并行 + 状态显式传递"** 这一三元组。

展开因子候选为 16、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def recurrent_kernel(query, key, value, state, output, last_state_out, ...):
    for b_idx in pypto.loop(batch_size):                    # Loop: Batch
        seq_len = dynamic_from_prefix_sum

        for nv_idx in pypto.loop(nv):                       # Loop: Parallel dim
            state_tile = state[b_idx, nv_idx]               # Init state from global

            for s_idx, uf in pypto.loop_unroll(seq_len, unroll_list=[16]):
                q_tile = view(query, ..., valid_shape=[chunk_len, ...])
                k_tile = view(key, ...)
                v_tile = view(value, ...)

                # V/C: State Update (LSTM gates / Delta Rule)
                ... state update computation ...

                state_tile[:] = new_state                   # State transfer

            last_state_out[b_idx, nv_idx] = state_tile      # State writeback
```

### 关键编码特征

| 特征 | 说明 |
|------|------|
| **状态变量** | 在序列 loop 外初始化，loop 内更新，loop 外写回 |
| **chunk 边界** | 使用 `is_loop_end` 检测尾部，`fillpad` 处理对齐 |
| **串行依赖** | 状态传递使得序列 loop **不可并行化** |
| **双向输出** | 每个 chunk 的中间结果 + 最终状态 |

### 开箱性能优化提示

> 实证来源：`models/qwen3_next/gated_delta_rule_impl.py:351-458`（aligned 版本）、`:480-489`（unaligned 版本）

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.vec_nbuffer_setting` | **必须细粒度配** | 按操作索引逐项给值，例：`{0: 4, 1: 16, 2: 8, 3: 8, ...}` | 递归状态机的 V 算子拓扑复杂，统一值会冲突 |
| `runtime_options.stitch_function_max_num` | **必配** | **`2`（极低！）** | 防止状态依赖被错误融合到大子图，与 SK-01 (`128`) 完全相反 |
| `runtime_options.device_sched_parallelism` | 必配 | `8` | 8 路并行调度，配合向量维并行 |
| 向量维并行 | **必配** | 内层 loop `pypto.loop(..., parallel=True)` | 状态在序列维不可并行，但跨 head/nv 维可并行 |
| 序列 loop 展开 | 按依赖选择单值 | 16 为候选，需要保持迭代顺序和状态更新 | 展开因子不自动等于数据块长度，核对步长与尾块 |
| `set_semantic_label` | 必配 | 标记 chunk 边界 | 让编译器识别递归块，应用正确调度策略 |
| `set_vec_tile_shapes(16, 16, 128, 128)` | 推荐 | 统一 4D tile | 与 unroll 第一项 16 配套 |
| 状态写回 | 强制 | loop 外初始化 + loop 内 `state[:] = new_state` + loop 后 `assemble` 写回 | 状态生命周期清晰，不允许编译器优化掉 |
| 动态 seq_len 变体 | 配置升级 | vec_nbuffer 条目数翻倍（~17 项），同时保留 `stitch_function_max_num=2` | unaligned 版本的标准做法 |

**性能建议**：递归状态沿序列方向存在依赖，不能直接并行。可以评估较小子图，以及无依赖的向量或 head 维度并行；具体配置需要测量。

---
