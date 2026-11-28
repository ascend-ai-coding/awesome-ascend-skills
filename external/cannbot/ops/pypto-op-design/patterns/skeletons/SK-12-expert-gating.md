---
type: pattern/skeleton
title: Expert Gating (Pure Vec)
description: MoE 专家打分和选择的纯向量骨架。
tags:
- moe
- vector
flow_pattern:
- V
examples:
- GLMSelectExperts
---

## SK-12: Expert Gating (Pure Vec)

**适用场景**: MoE 专家选择的纯向量操作。

**CV 排布**: 纯 V

展开因子候选为 256、64、16、4、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def expert_gating_kernel(logits, bias, weight_out, index_out, ...):
    for bs_idx in pypto.loop(bs_loop):                   # Loop: per-token
        # V: Score Prep (sigmoid + bias)
        scores = sigmoid(logits[bs_idx])
        scores = add(scores, bias)

        # V: TopK Select (group + expert two-stage)
        ... topk_group -> mask -> topk_expert ...

        # V: Renormalize
        weights = div(topk_weights, sum(topk_weights))

        weight_out[bs_idx:bs_idx+1, :] = weights
        index_out[bs_idx:bs_idx+1, :] = topk_ids
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **纯 V 操作** | 全程无 MatMul，仅包含 sigmoid、topk、gather、scatter |
| **两阶段 TopK** | 组级 topk_group → mask → 专家级 topk（AT-15） |
| **Sigmoid + Bias** | scores = sigmoid(logits) + bias |
| **per-token 独立** | `pypto.loop` 逐 token 处理（粒度固定为 1，无需 `unroll_factor` 特化，不用 `loop_unroll`） |
| **重归一化** | topk_weights = weights / sum(weights) |
| **双输出** | weights 和 indices 分别写入两个输出 tensor |
| **典型算子** | GLMSelectExperts |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/hc_pre_impl.py:90-150`（Sinkhorn 路由）、`models/qwen3_next/gated_delta_rule_impl.py:480-489`（动态路由）

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}` 起步；Sinkhorn 迭代多时升至**逐项配置**（17+ 条目） | TopK + 重归一化 V 拓扑较深 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | 与 SK-08 一致 |
| `runtime_options.device_sched_mode` | 推荐 | `0`（顺序）— Sinkhorn 路由型；`1`（并行）— 直接 TopK 型 | 取决于路由策略 |
| `runtime_options.device_sched_parallelism` | 推荐 | `8`（动态路由） | 多专家并行 |
| 循环展开 | 按需、单值 | 256 是候选，不是固定起点 | 与较小因子比较资源、编译与运行成本 |
| `pypto.loop` 逐 token | 必配 | 外层 loop | per-token 独立路由（粒度固定为 1，无需 `unroll_factor` 特化，不用 `loop_unroll`） |
| Sinkhorn 迭代数 | 经验 | `iters=20`, `eps=1e-6` | 软路由收敛 |
| 双输出 indices/weights | 强制 | 用 `[:]=` 切片写而非 `pypto.assemble` | 短轴存储更高效 |
| `set_vec_tile_shapes` 自适应 | 必配 | 按 expert/group 数动态设 | 不同 topk 配置需不同 tile |
| `combine_axis=True` | 推荐 | jit 首行 | |

**性能建议**：根据 TopK 扫描开销评估循环展开。硬 TopK 和 Sinkhorn 软路由的语义不同，不能仅为性能替换；修改算法必须先确认需求和参考计算允许。

---
