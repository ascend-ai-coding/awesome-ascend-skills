---
type: "PyPTO Performance Optimization Card"
title: "按 UB 预算批量搬行"
description: "相邻行在 GM 上规则排布且同生命周期时，把逐行 [1, N] 搬运合并为一次 [ROWS, N] 的 pl.load/pl.store，减少 copy 发射与同步次数。"
status: "stable"
tags: ["pypto-pro", "mem", "batch", "copy"]
item_id: "mem-02"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits"
      scope: "相邻独立行用一次二维 UB 搬运与 VF 行循环摊薄 descriptor 和同步成本。"
      boundary: "批大小由全部 live tiles 与槽位预算决定；行物理连续、相同生命周期和尾批有效行数不能只由逻辑 shape 推断。"
bound_hint: "MTE2、MTE3"
applicability: "循环中反复按单行或单个小张量搬入/写出，相邻行在 GM 连续排布且下游同阶段消费，UB 预算可容纳多行 Tile"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.load/pl.store 的二维 Tile 整块搬运与 make_tile_group 轮转"
---
# 技术卡片 mem-02：按 UB 预算批量搬行

- **适用 bound**：访存 / MTE
- **一句话**：把逐行 `[1, N]` 的 `pl.load`/`pl.store` 合并成一次 `[ROWS, N]` 整块搬运，copy 次数与同步开销按 ROWS 倍下降。

## 何时用（诊断特征）

- 循环中反复按单行/小张量搬入写出，相邻行在 GM 上规则连续排布，且下游在同一阶段消费。
- trace 有密集的小 copy，MTE 利用率不高但发射与同步开销明显。
- 输入行完整落入 UB 后可连续处理，或多个输出可在 UB 累积后一次写回。

## 何时不适用

- 行间不连续（stride 不规则、gather/scatter 语义）或后续不会消费所搬行时，批量搬运变为无效流量。
- 行宽过大使 `[ROWS, N]` 超出 UB 预算，或尾块很碎时退回单行/通用路径。

## 原理

每次 `pl.load`/`pl.store` 都有描述符与同步开销；行宽小、行数多时，这些固定开销主导 MTE 时间线。批量化用一次二维 Tile 搬运覆盖 ROWS 个连续行：传输描述符和同步次数降为 1/ROWS，且 MTE 处理更长的连续块。ROWS 由 UB 预算反推：`ROWS × rowBytes × 槽位数 + 其余常驻 ≤ UB 可用`。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的逐行 `y = x * 2` kernel（行宽 128 元素 fp16，ROWS=64）；尺寸、地址与循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：每行一次 [1, N] 搬运
# row_type = pl.TileType(shape=[1, N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
# for r in pl.range(core_id, rows, num_cores):
#     xr = x_row.next(); yr = y_row.next()
#     pl.load(xr, x, [r, 0])
#     pl.mul(yr, xr, 2.0)
#     pl.store(y, yr, [r, 0])

# after：一次搬运 ROWS 行
batch_type = pl.TileType(shape=[ROWS, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec)
x_batch = pl.make_tile_group(type=batch_type, addrs=0x00000, mutex_ids=[0, 1])
y_batch = pl.make_tile_group(type=batch_type, addrs=0x20000, mutex_ids=[2, 3])

with pl.section_vector():
    batches = x.shape[0] // ROWS
    for b in pl.range(core_id, batches, num_cores):
        xb = x_batch.next()
        yb = y_batch.next()
        pl.load(xb, x, [b * ROWS, 0])
        pl.mul(yb, xb, 2.0)
        pl.store(y, yb, [b * ROWS, 0])
```

## 性能与验证指标

比较 MTE copy 次数、同步次数与同条件 `Task Duration`；重点覆盖小行宽、多行重复场景。正确性按常规精度回归，尾批（行数不整除 ROWS）用 `set_validshape` 或 mem-01 的主/尾拆分覆盖。

## 技术限制与风险

- 合并的行必须同生命周期、物理连续；batch 的 offset、长度与对齐由同一份布局计算。
- Tile 元素数与字节数不要混用；ROWS 增大时核对槽位总量不挤占索引/临时量空间。
- 行数不整除 ROWS 的尾批不能按整批长度越界读写 GM。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`
