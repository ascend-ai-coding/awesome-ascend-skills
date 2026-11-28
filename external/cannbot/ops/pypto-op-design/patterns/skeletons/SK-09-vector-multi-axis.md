---
type: pattern/skeleton
title: Vector Tiling (Multi-Axis)
description: 在多个维度分块执行向量计算的骨架。
tags:
- vector
flow_pattern:
- V
examples:
- InterleaveRope
- ApplyAdamWV2
---

## SK-09: Vector Tiling (Multi-Axis)

**适用场景**: 需要在多个维度上分块的向量操作（如 RoPE 在 batch+seq 两个维度分块，优化器在参数维度分块）。

**CV 排布**: 纯 V

展开因子候选为 32、16、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def vector_tiling_kernel(input, cos, sin, output):
    for b_idx in pypto.loop(B):                            # Loop: Outer axis
        for s_idx, uf in pypto.loop_unroll(S, unroll_list=[32]):
            x_tile = pypto.view(input, [uf, N, D], [b_offset, 0, s_offset],
                                valid_shape=[uf, N, valid_s])
            cos_tile = pypto.view(cos, [uf, 1, D//2], [b_offset, 0, s_offset],
                                  valid_shape=[uf, 1, valid_s])
            sin_tile = similar

            # V: Multi-Axis Vector Compute
            set_vec_tile_shapes(v_rows, v_cols)
            ... compute ...

            pypto.assemble(result, [b_offset, 0, s_offset, 0], output)
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **多轴嵌套** | 外层 `pypto.loop`（batch/param 维度），内层 `loop_unroll`（序列维度） |
| **内层 unroll** | 内层 loop 使用 `loop_unroll`，tile_size 动态变化 |
| **view 加载** | 多层嵌套场景用 `pypto.view` + `valid_shape` 逐 tile 加载 |
| **TileShape 固定** | 全程一种 vec_tile 配置，无切换 |
| **Pipeline Loop** | 优化器场景使用 `submit_before_loop=True` + `set_cache_policy(NONE_CACHEABLE)` |
| **无跨 loop 状态** | 各 tile 计算独立，无累积器 |
| **典型算子** | InterleaveRope, ApplyAdamWV2, ApplyRMSProp |

### 特殊: Pipeline Loop（优化器用）

```python
for k_idx in pypto.loop(k_loops, submit_before_loop=True):   # pipeline overlap
    tile = view(param, [M, N_TILE], [0, k_off], valid_shape=[M, valid_n])
    set_cache_policy(NONE_CACHEABLE)    # 输入不做缓存
    ... compute ...
```

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/mla_prolog_v4_impl.py:312-360`（RoPE 多轴）、`models/qwen3_next/gated_delta_rule_impl.py:411-423`（向量+循环耦合）

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}` 起步；多轴重叠时升至 `{0: 4, -1: 6}` | 多 view tile 并发加载需要更多 nbuffer |
| 循环展开与 Vector TileShape | 配合评估 | 每轮数据块大小与所选展开因子一致，计算 TileShape 另按 API 设置 | 16 可作为单值候选，并验证尾块 |
| `submit_before_loop=True` | **必配（优化器场景）** | 内层 loop 标记 | 启用 pipeline overlap，把权重加载和计算重叠 |
| `set_cache_policy(NONE_CACHEABLE)` | **必配（优化器场景）** | 大参数张量上 | 避免大参数挤占 cache 一致性带宽 |
| 多 view 同步加载 | 推荐 | x_tile / cos_tile / sin_tile 同 loop 内 view | 让编译器识别为同 batch 提交 |
| 内层 loop 嵌套深度 | 限制 | ≤ 3 层 | 超过 3 层会触发 18000 表达式上限 |
| K_TILE 大小 | 推荐 | `2048`（优化器）/ `128`（RoPE） | 参数维分块经验值 |
| `combine_axis=True` | 推荐 | 仅当存在尾轴-1 broadcast 二元运算时 | 尾轴 broadcast 内联 brcb，见 F-15 |

**该骨架特有的性能方向**：**submit_before_loop + NONE_CACHEABLE pipeline 双开**（优化器特化）。瓶颈通常在大张量加载阻塞计算——这两个开关一起才能形成 pipeline overlap；单开 `submit_before_loop` 但保留默认 cache 策略会因 cache 冲突反而变慢。

---
