---
type: pattern/skeleton
title: General Pure Vector
description: 不匹配专用模式时使用的通用纯向量骨架。
tags:
- vector
flow_pattern:
- V
examples:
- BNReduce
- AvgPool2d
- ScatterNdSub
- MOEGatingTopK
---

## SK-13: General Pure Vector (兜底骨架)

**适用场景**: 不包含任何 matmul 的纯向量算子，且不匹配 SK-08/09/10/12 等特定骨架时的通用兜底。
涵盖：多阶段 reshape/transpose 管线、窗口化 reduce、in-place scatter、复杂多分支向量逻辑等。

**CV 排布**: 纯 V（零 Cube 操作）。全程仅 `set_vec_tile_shapes`，禁用 `set_cube_tile_shapes`。

### 骨架结构

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-1: N}},
    runtime_options={"stitch_function_max_num": 128},
)
def general_vector_kernel(inputs..., outputs..., scalar_params):
    # ── ○ 可选 静态预处理 ─────────────────────────
    ...  # 动态轴提取: N = x.shape[0]  # SymbolicScalar
    pypto.set_vec_tile_shapes(v_static_tile)               # 预处理阶段: 独立设置
    param_fp32 = cast(static_param, DT_FP32)
    param_expanded = expand_clone(reshape(param_fp32, [1, 1, D]), [1, N, D])

    # ── ○ 可选 Shape 前置变换 ─────────────────────
    pypto.set_vec_tile_shapes(v_shape_tile)                # Shape 变换阶段: 独立设置
    ...  # 3D→2D reshape / transpose / permute / 合轴拆轴 / 静态 shape 推导

    # ── ○ 可选 V_pre（loop 外全量 Dataflow，变体 A）──
    pypto.set_vec_tile_shapes(v_pre_tile)                  # 阶段 1: 必须重设
    ...  # reshape → transpose → reduce 管线，无 loop

    # ── ○ 可选 单 Loop（变体 B）────────────────────
    for idx in pypto.loop(tile_count):                     # or loop_unroll
        tile = pypto.view(input, [TILE_M, TILE_N], [offset_m, offset_n],
                          valid_shape=[valid_m, valid_n])

        pypto.set_vec_tile_shapes(v1_tile)                 # 阶段 1: Cast + Preprocess
        tile_fp32 = cast(tile, DT_FP32)
        ...

        pypto.set_vec_tile_shapes(v2_tile)                 # 阶段 2: Core Compute（必须重设）
        ...  # element-wise / reduce / topk / scatter / gather

        result = cast(computed, output_dtype)
        pypto.assemble(result, [offset], output)

    # ── ○ 可选 多层嵌套 Loop（变体 C）──────────────
    for b in pypto.loop(batch_count):
        for oh in range(OH):
            for ow in range(OW):
                pypto.set_vec_tile_shapes(v_tile)
                ...
                pypto.assemble(result, [offset], output)

    # ── ○ 可选 后聚合 ─────────────────────────────
    pypto.set_vec_tile_shapes(v_post_tile)                 # 阶段 N: 必须重设
    ...  # reshape / reduce / move 写回
    output_final.move(reshape(output, final_shape))
```

### 变体速查

| 变体 | Loop 结构 | 说明 | 实例 |
|------|----------|------|------|
| **A: 无 Loop Dataflow** | 无显式 loop | 全量 reshape→transpose→reduce 管线 | BNTrainingReduce, MHC_Post/Pre/Res |
| **B: 单 Loop Batch** | `loop_unroll(BS)` | 逐 batch tile，tile 内独立计算 | ScatterNdSub, MOEGatingTopK |
| **C: 多层嵌套** | `loop(BC)` → `range(OH)` → `range(OW)` | 多维输出逐位置计算 | AvgPool2d |

### 关键编码特征

| 特征 | 说明 |
|------|------|
| **多阶段 vec_tile 切换** | 每个计算阶段前必须重设 `set_vec_tile_shapes`（如 reshape→reduce→scatter 各一次） |
| **reshape/transpose 管线** | 纯 V 算子常用 reshape→transpose→reshape 重排数据 |
| **expand_clone 显式广播** | 用 `expand_clone` 手动实现 broadcast，不要依赖自动广播 |
| **index_put_ 原地写入** | ScatterNdSub 风格：`index_put_(target, (indices,), values, accumulate=True)` |
| **assemble + move** | 输出通过 assemble 写入临时 buffer，最后 `.move()` 写回 |
| **禁用 Cube TileShape** | 全程仅需 `set_vec_tile_shapes`，配 `set_cube_tile_shapes` 会被忽略并报警告 |

### 适用条件

- `has_matmul == False`
- 不匹配 SK-08(简单 batch unroll)、SK-09(多轴分块)、SK-10(递归)、SK-12(Expert Gating) 等特定骨架
- 典型特征：包含 reshape/transpose/reduce/scatter/topk/gather 等复杂向量操作

### 开箱性能优化提示

> 推断来源：BNReduce/AvgPool2d/ScatterNdSub 等纯 V 算子的共性配置；与 SK-08 同源但允许多阶段 tile 切换

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}` 起步；多阶段 reshape/transpose 时升 `{0: 4, -1: 6}` | 多 V 阶段串接 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | |
| 多 vec_tile 切换 | **必配** | 每阶段（reshape→reduce→scatter）前各 `set_vec_tile_shapes` 一次 | 不切换会触发 18000 表达式上限 |
| `expand_clone` 显式广播 | 强制 | 不要依赖自动广播 | PyPTO 仅支持单轴广播，多轴必须 `expand_clone` |
| `index_put_(target, (idx,), v, accumulate=True)` | 推荐 | 原地散点累加 | 比 `assemble + 累加` 高效 |
| `assemble + .move()` | 必配（多输出） | 临时 buffer → `.move()` 写回主张量 | 解耦 DAG，防止 view/assemble 环路 |
| 循环展开因子 | 按需 | 64、16、4、1 分别作为单值候选 | 验证尾块并比较耗时 |
| `pypto.reshape(..., inplace=True)` | 推荐 | 临时形变 | |
| `set_cube_tile_shapes` | **禁用** | 无 cube 操作 | 配了会被编译器忽略并报警告 |

**该骨架特有的性能方向**：**多阶段 vec_tile 切换 + expand_clone 显式广播**。瓶颈通常在 reshape/transpose 管线被编译器统一融合导致表达式爆炸——每个 reshape/transpose/reduce 阶段必须单独 `set_vec_tile_shapes`，把每段限制在 18000 表达式内。

---
