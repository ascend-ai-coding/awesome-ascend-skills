---
type: pattern/skeleton
title: Linear Projection (Norm→MatMul)
description: 以归一化和矩阵乘为核心的线性投影骨架。
tags:
- projection
flow_pattern:
- V1
- C1
examples:
- MLAProlog
- Qwen3PreAttn
---

## SK-03: Linear Projection (Norm→MatMul)

**适用场景**: 单阶段或多阶段线性投影，通常带前置 RMSNorm。

**CV 排布**: V(RMSNorm) → C(MatMul) [→ V(后处理)]

展开因子候选为 128、64、32、16、8、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def linear_projection_kernel(input, weight, gamma, output, ...):
    gamma_fp32 = pypto.cast(gamma, DT_FP32)
    weight_bf16 = ...

    for idx in pypto.loop(tile_count):                  # Loop: Tile (or loop_unroll)
        x_tile = pypto.view(input, [TILE, K], [offset, 0], valid_shape=[tile_len, K])

        # V: Pre-Norm (RMSNorm)
        pypto.set_vec_tile_shapes(v_tiles)
        normed = rms_norm(x_tile, gamma_fp32, eps)
        normed_bf16 = pypto.cast(normed, DT_BF16)

        # V: Quant (optional)
        # ...

        # C: Linear Projection
        pypto.set_cube_tile_shapes(c_tiles)
        projected = pypto.matmul(normed_bf16, weight, dtype, b_trans=True)

        # V: Dequant / Post-Proc (optional)
        # ...

        pypto.assemble(cast(projected, output_dtype), [offset, 0], output)
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **V→C 排布** | 先 Vec（Norm），再 Cube（MatMul），可选后接 Vec |
| **单层 Loop** | 仅 batch/token 一维循环，loop 内完成 V+C |
| **TileShape 切换** | V 阶段和 C 阶段前各自 `set_vec/cube_tile_shapes` |
| **后处理 V 阶段** | 可选：split、reshape、per-head norm、RoPE（非必须） |
| **量化变体** | V→C 可扩展为 V→Quant→C→Dequant（AT-11） |
| **典型算子** | MLAProlog（部分）、Qwen3PreAttn |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/lightning_indexer_prolog_quant_v4_impl.py:150-302`、`models/deepseek_v4/mla_prolog_v4_impl.py:393-413`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.cube_l1_reuse_setting` | 必配 | `{-1: 2, 1: 1}` 分轴 | 权重轴用 1（不复用），激活轴用 2（双缓冲），匹配"权重静态、激活动态"特性 |
| `pass_options.vec_nbuffer_setting` | 推荐 | `{0: 2}`（最 Norm 阶段） | RMSNorm 在 V 阶段，nbuffer=2 即可 |
| `pypto.set_cache_policy(NONE_CACHEABLE, True)` | 条件配 | 静态权重张量；**仅当权重在 loop 内被单次消费** | 权重只读一次时不需占用 L2，避免与激活竞争 cache；**⚠️ 若权重跨 loop 迭代复用（如 mhc_pre 变体的 phi 被 8 次 unroll 复用且可驻留 L2），标记 NONE_CACHEABLE 会强制每迭代回 HBM 重读，反而劣化——勿用** |
| token 循环展开 | 按需 | 每次选一个因子，128 仅为候选 | 比较编译成本和运行耗时，验证余数处理 |
| `pypto.set_semantic_label("...")` | 推荐 | 每阶段一个标签 | 帮助编译器识别阶段边界，便于 Pass 调度 |
| `pypto.reshape(..., inplace=True)` | 推荐 | tile 内 reshape | 避免临时张量分配 |
| TileShape | 推荐 | V 阶段 `set_vec_tile_shapes(4, hidden)`；C 阶段 `set_cube_tile_shapes([M,K],[K,N],[M,N])` | V/C 切换前各设一次 |

**该骨架特有的性能方向**：**权重 NONE_CACHEABLE + loop_unroll 自适应**。瓶颈通常在权重 cache 抢占激活——用 `set_cache_policy(NONE_CACHEABLE)` 标记权重就能恢复激活吞吐。

#### ⚠️ 形态变体：大归约维 ND × 极小输出宽 K（mhc_pre 类）

当算子为「单 matmul + norm」但形状落在 **输出宽 ≤ 64、归约维 N·D ≥ 数万、FP32 计算**（典型：`mhc_pre` 的 `matmul [B,28672]×[28672,24]`）时，用下列变体结构替代上方开箱提示：

| 维度 | 变体配置 | 原理 |
|------|---------|------|
| 循环 | `loop_unroll(0, BS, 1, unroll_list=[16])`（M=unroll_length） | 平铺 BT-loop 顾此失彼：BT 大则 cube 只有一个任务无并行，BT 小则 vec 归约被切碎。loop_unroll 让 vec 连续处理 16 行整 D，cube 按 M=16 出多个任务——两侧并行同时成立 |
| vec tile | D 轴大 tile：`(8,2048)` / `(1,2048)` / `(1,N,2048)` | 计算集中在尾轴，单任务连续扫得越长，DMA/计算比越好；小行 tile 会把长归约切碎 |
| 权重布局 | wrapper 预转置 `phi.T.contiguous()`，matmul 不加 `b_trans` | 转置 host 侧只做一次；`b_trans` 是每个 cube 任务重复付跨步寻址代价 |
| cube | `[16,16],[512,1024],[128,128]` + `enable_split_k=True` | 输出极小，cube 并行只能切归约维（分核算部分和再归并）；M=16 与 unroll_length 对齐 |
| 权重 cache | 不设 NONE_CACHEABLE | phi 被各 unroll 迭代反复复用且可驻留 L2；禁缓存等于强制每轮回 HBM 重读 |
| 装饰器 | `stitch_function_max_num=128`、`device_sched_mode=2`、`cube_nbuffer={-1:4}`、`vec_nbuffer={"DEFAULT":4,...}`、`sg_set_scope` 分段、`combine_axis=True` | stitch 把多次 unroll 迭代缝成大图做流水调度；动态调度只在图内任务足够多时优于静态分配 |
| UB 约束 | 勿提前 cast 大中间量（FP32 全 tile > UB 会 spill） | x 的 FP32 形态是 BF16 两倍，跨 matmul 驻留 UB 装不下；用时重读 BF16 即时 cast 反而更快 |

**识别条件**：`matmul_count==1 AND 输出宽 ≤ 64 AND 归约维 N·D ≥ 2^14 AND FP32 计算`，命中即直接用变体，跳过 BT sweep。

**警示**：变体各配置的有效性都依赖 loop_unroll 这个结构前提，拆开单项套到平铺 BT-loop 上只会劣化——要么整体用，要么不用。

---
