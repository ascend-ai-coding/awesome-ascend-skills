---
type: pattern/skeleton
title: Fused Pre-Attn (Two-Phase)
description: 将预处理投影和注意力计算融合为单一 kernel 的骨架。
tags:
- attention
- projection
flow_pattern:
- V1
- C1
- V2
- C2
- V3
- C3
examples:
- GLMAttnFusion
- Qwen3PreAttnFused
---

## SK-05: Fused Pre-Attn (Two-Phase)

**适用场景**: 将 Pre-Processing（Norm + QKV Proj + RoPE）和 Flash Attention 融合为单一 kernel。

**CV 排布**:
- Phase 1: V(Norm) → C(Quant Linear) → V(Dequant+Split+Norm+RoPE+Cache)
- Phase 2: SK-01 Online Flash Attention

### 骨架结构

```python
def fused_pre_attn_kernel(input, residual, norm_params, weights, quant_params,
                          cos, sin, block_table, kv_cache, ..., attn_output):
    # ==============================
    # Phase 1: Pre-Processing (V-C-V)
    # ==============================
    for bs_idx, uf in pypto.loop_unroll(bs_loop, unroll_list=[...]):
        x_fp32 = cast(view(input, ...), FP32)
        res_fp32 = cast(view(residual, ...), FP32)
        x_add = add(x_fp32, res_fp32)

        # V: Norm
        normed = rms_norm(x_add, gamma, eps)

        # C: Quant Linear (INT8 W8A8)
        x_int8, x_scale = quantize(normed)
        y_int32 = matmul(x_int8, w_int8, INT32)
        y = dequant(y_int32, x_scale, w_scale)

        # V: QKV Split + RoPE
        q = rms_norm_per_head(y[:, :q_dim], q_gamma)
        k = rms_norm_per_head(y[:, q_dim:q_dim+k_dim], k_gamma)
        v = y[:, q_dim+k_dim:]
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        # V: Cache Write
        scatter_update(kv_cache, block_table, k, v)

        # 中间 buffer 暂存 Q（Phase2 消费）
        q_tmp[bs_offset:] = cast(q, BF16)
        residual_out[bs_offset:] = cast(x_add, BF16)

    # ==============================
    # Phase 2: Flash Attention (SK-01 C1-V1-C2)
    # ==============================
    for b_idx in pypto.loop(batch_size):
        for h_idx in pypto.loop(num_heads):
            for q_idx in pypto.loop(q_tiles):
                allocate oi/li/mi accumulators
                for kv_idx in pypto.loop(kv_tiles):
                    k_tile = gather(k_cache, block_table, ...)
                    v_tile = gather(v_cache, block_table, ...)
                    ... SK-01 core (C1-V1-C2 + online softmax) ...
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **两阶段设计** | Phase1 预处理（V+C+V）+ Phase2 Flash Attention（C1-V1-C2） |
| **Phase1 循环** | `loop_unroll` 处理 batch，逻辑独立无跨迭代依赖 |
| **Phase2 循环** | 复用 SK-01 的 4~5 层嵌套循环结构 |
| **量化投影** | Phase1 的 C 阶段通常为 INT8 W8A8 量化（AT-11） |
| **中间 buffer** | Phase1 输出暂存到 `pypto.tensor()` 供 Phase2 消费 |
| **Cache 写入** | Phase1 内包含 AT-16 scatter_update 写入 KV cache |
| **典型算子** | GLMAttnFusion, Qwen3PreAttnFused |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/lightning_indexer_prolog_quant_v4_impl.py:150-302`、`models/deepseek_v4/hc_pre_impl.py:117-150`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.cube_l1_reuse_setting` | 必配 | Phase1: `{-1: 2, 1: 1}`；Phase2 复用 SK-01 配置 | 两阶段独立配置 |
| `pass_options.vec_nbuffer_setting` | 必配 | Phase1: `{0: 2}`；Phase2: `{-1: 4}` | Phase1 V 阶段轻量，Phase2 重 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | |
| `runtime_options.device_sched_mode` | 推荐 | `0`（HC 风格 顺序）或 `1`（GLM 风格 并行） | 取决于 Phase1↔Phase2 是否数据独立 |
| Phase1 `loop_unroll` | 必配 | 候选因子 256、64、16、4、1，初始设计只选一个值，其余留作调优候选 | Phase1 计算密度低，展开降低 prologue 占比；先单值验证尾块 |
| 注意力循环策略 | 复用 SK-01 | 使用 `pypto.loop`，或从 4、2、1 中选择一个展开因子 | 核对循环依赖及尾块 |
| Phase1 输出中间 buffer | **强制** | 命名 `pypto.tensor([B,N,D], BF16, "q_tmp")` 显式分配 | Phase2 消费 Phase1 输出，必须显式 buffer 防止编译器误判生命周期 |
| KV Cache scatter | 推荐放 Phase1 末 | `AT-16` 写入紧贴 RoPE 之后 | 与 Phase2 的 KV 读完全解耦 |
| `combine_axis=True` | 必配 | jit 首行 | |

**该骨架特有的性能方向**：**Phase1/Phase2 配置解耦** + **中间 buffer 命名显式化**。瓶颈通常在 Phase1↔Phase2 数据接力——Phase1 的 q_tmp / residual_out 必须用命名 `pypto.tensor()`，否则 Phase2 读不到稳定地址。

---
