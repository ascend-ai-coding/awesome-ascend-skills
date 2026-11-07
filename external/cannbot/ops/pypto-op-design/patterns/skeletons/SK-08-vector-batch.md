---
type: pattern/skeleton
title: Vector Element-wise (Batch Unroll)
description: 独立 token 上执行批量向量计算的骨架。
tags:
- vector
flow_pattern:
- V1
examples:
- InplaceAddRmsNorm
- mhc_pre/post
---

## SK-08: Vector Element-wise (Batch Unroll)

**适用场景**: 逐 token 的纯向量操作（归一化、门控等），每个 token 独立计算。

**CV 排布**: 纯 V

展开因子候选为 64、16、4、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def vector_batch_kernel(input, params..., output):
    params_fp32 = cast(params, FP32)
    set_vec_tile_shapes(v_rows, v_cols)

    for bs_idx, uf in pypto.loop_unroll(0, BS, 1, unroll_list=[64]):
        tile = input[bs_idx:bs_idx+uf, :]                  # Load tile (Python slice)

        # V: Element-wise Compute
        tile_fp32 = cast(tile, FP32)
        ... element-wise ops (mul, add, sum, rsqrt, etc.) ...
        result = cast(computed, output_dtype)

        pypto.assemble(result, [bs_idx, 0], output)        # Store tile
```

### 关键编码特征

| 特征 | 说明 |
|------|------|
| **loop_unroll** | 使用 `loop_unroll` 而非 `loop`，tile_size 动态变化 |
| **切片加载** | `x[bs:bs+uf, :]` 直接 Python 切片 |
| **assemble 存储** | `assemble(tile, [bs_offset, 0], output)` |
| **无 TileShape 切换** | 全程一种 vec_tile 配置 |
| **无跨 loop 状态** | 每个 tile 完全独立 |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/mla_prolog_v4_impl.py:312-360` 的纯向量段、`models/deepseek_v4/hc_pre_impl.py:149-150`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}` | 纯 V 的标准双缓冲 |
| `pass_options.cube_l1_reuse_setting` | **不配** | 无 cube 操作 | 配了无效，反而占用编译器配额 |
| Batch 循环展开 | 按需、单值 | 64 为候选，实际值按 batch 范围选择 | 显式验证不能整除时的处理 |
| `runtime_options.stitch_function_max_num` | 推荐 | `128` | |
| `combine_axis=True` | 推荐 | jit 首行 | 尾轴 broadcast 内联 brcb，见 F-15 |
| `set_vec_tile_shapes(rows, cols)` | 必配 | 循环外设置一次 | 全程不切换，保持向量化稳定 |
| 静态参数预 cast | 必配 | `params_fp32 = pypto.cast(params, FP32)` 放 jit 体首部 | 避免每次迭代重复 cast |
| `pypto.reshape(..., inplace=True)` | 推荐 | 临时 reshape | 避免缓存分配 |
| 切片访问 | 推荐 | `x[bs:bs+uf, :]` 直接 Python 切片 | 让编译器静态推导，比 `view + valid_shape` 略快（短批场景） |

**性能建议**：分别比较不同的单值展开因子和静态参数预转换。短批量同样需要验证，不能假定大展开因子会自动选择较小的处理路径。

---
