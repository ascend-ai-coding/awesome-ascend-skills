---
type: pattern/atom
title: K-Split MatMul（大 K 投影）
description: 将大 K 维度投影沿 K 轴经 `pypto.view` 拆分为两部分，各自 `matmul(→FP32)` 后相加，以 K 轴并行缓解 M 轴不可切导致的 cube 利用率饱和。
tags:
- matmul
flow_pattern:
- C
examples:
- MLAPrologQuant
- decode 小 M 大 K 投影
---

## AT-22: K-Split MatMul（大 K 投影）

**描述**: 将大 K 维度投影沿 K 轴经 `pypto.view` 拆分为两部分，各自 `matmul(→FP32)` 后相加，以 K 轴并行缓解 M 轴不可切导致的 cube 利用率饱和。

**CV 排布**: C（2× MatMul + 1× add）

**适用场景**: M < cube M-tile 下限（如 M=8 < 16，M 轴不可并行），仅 N/K 可切；或单 matmul 中间量超 UB 限制 tile_bs 增大。

**输入**:
- `x: Tensor[M, K]` — 输入（BF16/FP16），K 较大（如 7168）
- `w: Tensor[K, N]` — 权重（BF16，可 NZ 格式）

**输出**:
- `y: Tensor[M, N]` — 投影结果（FP32）

**计算流**:
```
KH = K // 2
x1 = pypto.view(x, [M, KH], [0, 0]);  x2 = pypto.view(x, [M, KH], [0, KH])
w1 = pypto.view(w, [KH, N], [0, 0]);  w2 = pypto.view(w, [KH, N], [KH, 0])
y = pypto.matmul(x1, w1, pypto.DT_FP32) + pypto.matmul(x2, w2, pypto.DT_FP32)
```

**要点**:
- 确定性：FP32 累加 + FP32 相加，不依赖嵌套 loop 或 `enable_split_k`（后者引入非确定性，max_error_ratio=0 输出不可用）。
- UB：单 matmul 中间量 FP32 扩展由 `[M, K]` 减半至 `[M, K/2]`，为增大 tile_bs 的前提之一。
- 精度：K 拆分仅改变归约顺序，与单 matmul 的数值差异在 FP32 噪声内，处于 max_error_ratio 容差内。

**实例化参数**:
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `split_factor` | 拆分份数 | 2（K=7168→3584×2） |
| `out_dtype` | matmul 输出 dtype | DT_FP32 |

**使用算子**: MLAPrologQuant（Q 下投影 `x@w_dq`，K=7168 拆半）。与 tile_bs 多值 unroll、NZ 权重、`transposed_batchmatmul` 联用，见 SK-04 性能方向 ③。



---
