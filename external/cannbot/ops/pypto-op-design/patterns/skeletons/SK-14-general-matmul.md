---
type: pattern/skeleton
title: General Multi-MatMul
description: 不匹配专用模式时使用的通用多矩阵乘骨架。
tags:
- matmul
flow_pattern:
- C
- V
examples:
- GLMGate
- GMMRouting
- QuantGroupedMM
- QuantMatMulReduce
---

## SK-14: General Multi-MatMul (兜底骨架)

**适用场景**: 以**多个 matmul** 为核心计算的算子，且不匹配 SK-01~SK-07 等特定骨架时的通用兜底。
涵盖：分组矩阵乘、量化矩阵乘、多路并行线性投影、多专家批量 matmul、
matmul 间夹少量向量操作（dequant/quant/reduce/scatter）等。

**CV 排布**: 多个 C 阶段显式展开，C 之间可夹少量 V 操作。模式：`C...[V]...C...[V]...C`

> **编码约束**：禁止使用 `for mm_idx in range(N)` + `optional_v_stage()` 循环分发 matmul，
> 每个 C/V 阶段必须显式写出，JIT 编译器需要在编译期确定完整计算图。

### 骨架结构

```python
@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-1: 8},
    },
    runtime_options={"stitch_function_max_num": 128},
)
def general_multi_matmul_kernel(A, B_list, output, config):
    pypto.experimental.set_operation_options(combine_axis=True)

    # ── ○ 可选 静态预处理 ─────────────────────────
    pypto.set_vec_tile_shapes(v_static_tile)                # 预处理阶段: 独立设置
    ...  # 动态轴提取 / 参数 cast / 常量计算

    # ── ○ 可选 Shape 前置变换 ─────────────────────
    pypto.set_vec_tile_shapes(v_shape_tile)                 # Shape 变换阶段: 独立设置
    ...  # 3D→2D reshape / transpose / 合轴拆轴 / 静态 shape 推导

    # ── ○ 可选 外层 Loop ─────────────────────────
    # 变体 A: 无 loop（全量串行 matmul）
    # 变体 B: parallel loop → for g_idx in pypto.loop(num_groups, parallel=True):
    # 变体 C/D: 串行 loop → for g_idx in pypto.loop(tile_count):
    ...  # 分组切片: a_tile = group_slice(A, g_idx)

    # ── ○ 可选 C_1 ───────────────────────────────
    pypto.set_cube_tile_shapes(c1_tile)                    # 每个 MatMul 必须独立设置
    ...  # MatMul (如 Gate Projection / QKV Proj / QuantMatMul)

    # ── ○ 可选 V_inter ────────────────────────────
    pypto.set_vec_tile_shapes(v1_tile)
    ...  # dequant / quant / split / activation（保持轻量，重 V 操作走 SK-15）

    # ── ○ 可选 C_2 ───────────────────────────────
    pypto.set_cube_tile_shapes(c2_tile)                    # 必须重设，M/N/K 通常不同
    ...  # MatMul (如 Up Projection / Second Proj)

    # ── ○ 可选 V_inter ────────────────────────────
    pypto.set_vec_tile_shapes(v2_tile)
    ...  # SwiGLU / GELU / concat / cast

    # ── ○ 可选 C_N (可重复 1~N 次) ────────────────
    pypto.set_cube_tile_shapes(cN_tile)                    # 每个 MatMul 前必设
    ...  # MatMul (如 Down Projection / Output Proj)

    # ── ○ 可选 V_post（后聚合）────────────────────
    pypto.set_vec_tile_shapes(v_post_tile)
    ...  # reduce / cast / scatter / index_put_ 写回

    # ── ○ 可选 结果存储 ───────────────────────────
    ...  # group_store / index_put_(accumulate=True) / assemble 写回
```

### 变体速查

| 变体 | Loop 结构 | MatMul 模式 | 典型场景 | 实例 |
|------|----------|------------|---------|------|
| **A: 无 Loop** | 无显式 loop | 2+ 串行 matmul（全量） | 多路投影管线 | QuantMatMulReduceSum |
| **B: 并行 Loop** | `loop(N, parallel=True)` | 每组 1~2 个 matmul | 多专家分组并行 | GMMFinalizeRouting, QuantGroupedMM |
| **C: 串行 Loop** | `loop(tiles)` | 每组 1 个 matmul | 逐 tile 批量投影 | GLMGate |
| **D: 串行多 MatMul** | `loop(tiles)` | 每组 2~3 个 matmul | gate+up 并行→down | PanguFusedLayer(FFN 部分), FusedSwiGLU |

### 典型编排模式

```
# 模式 1: 串行管线（前一 matmul 输出 → 后一 matmul 输入）
C1(A, W1) → [V: dequant] → C2(mm1_out, W2) → [V: dequant] → C3(mm2_out, W3)
实例: 多级量化投影 (MLA: q_a_proj → norm → q_b_proj)

# 模式 2: 并行多路（同一输入 → 多个独立 matmul → 合并）
C1(A, W_gate) ─┐
                ├→ [V: SwiGLU/concat] → C3(merged, W_down)
C2(A, W_up)  ──┘
实例: FFN (gate_proj + up_proj → SwiGLU → down_proj)

# 模式 3: 分组并行（不同输入分片 → 各自 matmul → scatter/gather）
for g in parallel:
    C_g(A[g], W[g]) → [V: scale] → scatter(output)
实例: GMM (per-expert matmul + logit scaling)

# 模式 4: 循环内多 C（KV 分片场景，每组多步 matmul）
for group:
    C1(Q, K[group]) → [V: softmax] → C2(P, V[group])
实例: 通用稀疏 attention（非标准 FA 场景）
```

### 阶段积木速查

| 积木 | 类型 | 可选操作 | 可重复 | 依赖 |
|------|------|---------|--------|------|
| C_N | C | MatMul (BF16 / INT8 / MXFP8 via scaled_mm) | **是 (1~N)** | — |
| V_inter | V | dequant / quant / split / activation | 是 | 相邻 C 阶段 |
| V_post | V | reduce / cast / scatter / index_put_ | 否 | 最后一个 C 阶段 |

> V_inter 必须保持轻量（仅 quant/dequant/split/activation），重 V 操作应走 SK-15。

### 关键编码特征

| 特征 | 说明 |
|------|------|
| **每 matmul 独立 TileShape** | 不同 matmul 的 M/N/K 维度通常不同，需各自 `set_cube_tile_shapes`，复用会触发表达式爆炸 |
| **parallel=True** | 分组独立时启用并行循环，编译器自动并行化 |
| **K 轴分组** | 沿 K 轴按组切分 A/B，每组独立 matmul |
| **scaled_mm** | MXFP8 量化 matmul 的统一接口：`scaled_mm(A, B, FP32, scale_a, scale_b)` |
| **index_put_ 累积** | 多组写入同一输出 tensor 时使用 `accumulate=True` |
| **C 间 V 保持轻量** | 两个 matmul 之间的 V 操作仅做 dequant/quant/activation，重 V 走 SK-15 |
| **post-loop V** | loop 后一次性做 batch add / reduce / cast |
| **NZ 格式权重** | 权重可能使用 NZ(fractal) 格式，需在签名中声明 |

### 适用条件

- `has_matmul == True` 且 **matmul_count >= 1**
- 不匹配 SK-01(FA), SK-02(SinglePass), SK-03(Norm+Linear), SK-04(Prolog), SK-06(FFN), SK-07(MOE) 等特定骨架
- 典型特征：多个分组/并行/串行 matmul，C 之间夹少量 V 操作（非完整 CV 融合管线）
- 与 SK-15(CV Fusion) 的区别：SK-14 的 C 阶段占主导，V 阶段仅做量化/反量化/激活等轻量后处理

### 开箱性能优化提示

> 推断来源：GLMGate/GMMRouting/QuantGroupedMM 等多 MatMul 算子共性；MXFP8 量化通过 `scaled_mm` 接口

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.cube_l1_reuse_setting` | **必配** | 多键分轴：`{-1: 2, 0: 4, 1: 1}` | 每个 matmul 的 L1 复用策略不同，需精细 |
| `pass_options.cube_nbuffer_setting` | 必配 | `{-1: 2}` 起步；分组 MM 升 `{1: 2}` | 多 cube 阶段双缓冲 |
| `pass_options.vec_nbuffer_setting` | 推荐 | `{-1: 2}` 即可（V 阶段轻量） | C 间 V 仅做量化等轻操作 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | |
| `runtime_options.device_sched_mode` | 推荐 | `1`（并行） | 多 MatMul 并行调度 |
| 每个 MatMul 独立 `set_cube_tile_shapes` | **强制** | 每个 matmul 前必设 | M/N/K 维度不同，复用前一个的 tile 会触发表达式爆炸 |
| `parallel=True` | 推荐 | 分组循环（专家/group） | 编译器自动并行化 |
| K 轴分组切分 | 推荐 | `view + valid_shape` 按组切片 | 见 AT-20 |
| `scaled_mm` 接口 | 必配（量化） | MXFP8 用统一接口 | 不要拆出独立 quant/dequant |
| `index_put_(target, idx, v, accumulate=True)` | 推荐 | 多组写同一输出 | 比拆 assemble 高效 |
| C 间 V 操作 | **保持轻量** | 仅 quant/dequant/split/activation | 重 V 操作应分骨架做（SK-15） |
| post-loop V 聚合 | 推荐 | 循环后一次性 `add/reduce/cast` | 避免循环内重复 V 启动 |
| NZ 权重格式 | 推荐 | 签名声明 `format=NZ` | 提升 cube 加载效率 |
| `combine_axis=True` | 必配 | jit 首行 | |

**性能建议**：分别检查各矩阵乘的 TileShape。分组之间没有数据依赖时，可以评估 parallel=True，比较编译规模、资源占用和运行耗时。

---
