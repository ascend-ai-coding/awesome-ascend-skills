---
type: pattern/skeleton
title: FFN / SwiGLU
description: 两路线性投影与 SwiGLU 激活融合的前馈网络骨架。
tags:
- ffn
flow_pattern:
- C1
- C2
- V1
examples:
- FusedSwiGLU
---

## SK-06: FFN / SwiGLU

**适用场景**: 前馈网络，两个并行 MatMul + SwiGLU 激活。

**CV 排布**: C(Gate) → C(Value) → V(SwiGLU)

### 骨架结构

```python
def fused_swiglu_kernel(input, weight_gate, weight_fc, bias_gate, bias_fc, output):
    for idx in pypto.loop(tile_count):                              # Loop: Tile
        x_tile = view(input, [TILE, K], [offset, 0], valid_shape=[tile_len, K])

        # C: Gate Linear
        set_cube_tile_shapes(c_tiles)
        gate = matmul(x_tile, weight_gate, DT_FP32, extend_params={"bias_tensor": bias_gate})

        # C: Value Linear
        value = matmul(x_tile, weight_fc, DT_FP32, extend_params={"bias_tensor": bias_fc})

        # V: SwiGLU Activation
        set_vec_tile_shapes(v_tiles)
        silu = mul(gate, sigmoid(gate))
        y = mul(silu, value)
        y_bf16 = cast(y, DT_BF16)

        output[offset:offset+tile_len, :] = y_bf16
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **并行 C→C** | Gate 和 Value 两个 MatMul 共享同一输入 tile，独立计算 |
| **单层 Loop** | 仅 batch/token 一维循环 |
| **C→V 顺序** | 两个 C 阶段在前，V（SwiGLU）在后，需 `set_cube_tile_shapes` 两次 |
| **TileShape 相同** | Gate 和 Value 的 M/N/K 相同，C 阶段可复用同一 TileShape |
| **SwiGLU** | 使用 AT-07，`silu(gate) * value` |
| **偏置支持** | 可选的 bias 通过 `extend_params={"bias_tensor": b}` 融合进 matmul |
| **典型算子** | FusedSwiGLU |

### 开箱性能优化提示

> 实证来源：`pypto-gym/ops/pypto_tensor/experimental/ops_transformer/fused_swiglu/fused_swiglu_impl.py:23-65`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 2, 0: 4}` | SwiGLU 阶段 V 双缓冲；首维更激进 4 |
| `pass_options.cube_l1_reuse_setting` | 必配 | `{-1: 2}` | Gate / Value 两个 C 阶段共享 L1 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | |
| `runtime_options.device_sched_mode` | **必配** | `3` | FFN 专用调度模式，与 SK-01 (mode=1) 不同 |
| `combine_axis=True` | 必配 | jit 首行 | |
| `set_cube_tile_shapes` | 必配 | `[128,128], [128,256], [128,128]` 起步 | Gate/Value 两个 MatMul 共用一组 tile |
| MatMul bias 融合 | **必配** | 用 `extend_params={"bias_tensor": b}` 在 matmul 内融合 bias | 不要单独写 `add(out, bias)` |
| Tile_m | 推荐 | `512` | FFN 的 M（token 维度）经验值 |
| `view + valid_shape` | 必配 | 处理尾块 | 见 AT-20 |
| SwiGLU 写法 | 强制 | `mul(silu(gate), value)` 单条向量链 | 避免中间张量 |

**该骨架特有的性能方向**：**device_sched_mode=3 FFN 专用调度** + **bias 融合到 matmul**。瓶颈通常在 Gate/Value 两个 MatMul 调度——device_sched_mode=3 专门优化双 MatMul 并行；bias 必须用 matmul 的 extend_params 融合，否则会拆出独立的 V 阶段。

---
