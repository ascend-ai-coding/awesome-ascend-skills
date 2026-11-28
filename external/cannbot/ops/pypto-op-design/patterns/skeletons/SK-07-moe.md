---
type: pattern/skeleton
title: MOE (Gate + Select + FFN)
description: 门控、专家选择和前馈计算融合的 MoE 骨架。
tags:
- moe
flow_pattern:
- V1
- C1
- V2
- C2
- V3
- C3
- V4
examples:
- GLMMoEFusion
- GMMSwiGLUQuant
---

## SK-07: MOE (Gate + Select + FFN)

**适用场景**: 混合专家模型的完整推理流程。

**CV 排布**: V(Gate Score) → C(Gate Linear) → V(Expert Select) → V(Quant) → C(Up Proj) → V(Dequant+SwiGLU) → V(Quant) → C(Down Proj) → V(Dequant)

展开因子候选为 32、16、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def moe_fusion_kernel(input, gate_weight, expert_weights, ...):
    for bs_idx, uf in pypto.loop_unroll(bs_loop, unroll_list=[32]):
        hidden_tile = pypto.view(input, ..., valid_shape=[...])

        # C: Gate Score
        [C] logits = matmul(hidden_tile, gate_weight, FP32, b_trans=True)
        [V] weights = sigmoid(logits)

        # V: Expert Select (TopK routing)
        [V] topk_ids, topk_weights = topk_routing(weights, ...)

        # V: Pre-Expert Quant (optional)
        [V] x_int8, x_scale = quantize(hidden_tile)

        # C: Expert Up Linear
        [C] up_int32 = matmul(x_int8, w13[expert], INT32)

        # V: Post-Expert Dequant (optional)
        [V] up = dequant(up_int32, x_scale, w13_scale)

        # V: Expert Activation (SwiGLU)
        [V] gate, value = split(up)
        [V] swiglu = swiglu(gate, value)

        # Down Projection
        [V] sw_int8, sw_scale = quantize(swiglu)
        [C] down_int32 = matmul(sw_int8, w2[expert], INT32)
        [V] down = dequant(down_int32, sw_scale, w2_scale)

        # V: Weighted Sum
        [V] ffn_out = down * topk_weights -> sum

        [V] assemble(ffn_out, [bs_offset, 0], output)
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **长 CV 管线** | 7+ 阶段：V→C→V→V→C→V→V→C→V，C/V 频繁切换 |
| **Expert Gating** | 使用 AT-15 两阶段 topk（组级 + 专家级） |
| **Per-expert 权重** | MatMul 的权重按 expert 索引动态选择 `w13[expert]` |
| **量化路由** | FFN 内部 2 次 Quant→MatMul→Dequant 管线 |
| **Score 加权** | 输出乘以 `topk_weights` 再 sum，需广播处理 |
| **Loop 策略** | 外层 `loop_unroll`，内层 expert 循环用 Python `range` |
| **典型算子** | GLMMoEFusion, GMMSwiGLUQuant |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/compressor_impl.py:411-610`、`models/deepseek_v4/hc_pre_impl.py:117-150`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| 外层 loop_unroll | 按需 | 候选因子为 32、16、8、4、1，每次选一个 | 按 token 分块，核对专家选择与后续计算的依赖 |
| `pypto.set_pass_options(sg_set_scope=(1, True, False))` | 推荐 | 在专家分支前 | 显式控制子图边界，防止 Gate→Expert 错误融合 |
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}` 起步；MoE 涉及量化时升 `{0: 4, -1: 6}` | 长 V 管线（V→C→V→C→V）需要更多 nbuffer |
| `pass_options.cube_l1_reuse_setting` | 必配 | `{-1: 2}` 通用，专家权重不复用时 `{0: 1}` | 每个专家权重独立，L1 复用有限 |
| `runtime_options.device_sched_mode` | 推荐 | `1`（并行） | 多专家并行调度 |
| 专家循环 | 强制 | **Python `range(N)` 而非 `pypto.loop`** | N 编译期已知，Python range 让 PyPTO 静态展开 |
| `topk_weights` 加权 | 推荐 | 在最末一次 V 阶段融合 sum | 不要拆出独立 sum kernel |
| Per-token Quant | 必配 | `AT-05 (FP→INT8) + AT-06 (INT→FP)` 围绕每个 MatMul | INT8 W8A8 标准链路 |
| 路由计算 | 强制 | 保持规格要求的硬 TopK 路由；仅在规格明确要求 Sinkhorn 时使用对应变体 | 路由算法不得偏离规格，否则与 Golden 不一致 |
| `combine_axis=True` | 必配 | jit 首行 | |

**该骨架特有的性能方向**：**专家循环静态展开** + **量化对围绕每次 MatMul**。瓶颈通常出在专家选择和加权 sum 拆得太散——必须用 Python range 让编译器静态展开，且最末次 V 阶段直接融合加权 sum。

---
