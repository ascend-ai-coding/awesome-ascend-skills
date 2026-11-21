---
type: "PyPTO Performance Optimization Card"
title: "连续完整结果一次性直写"
description: "输出区间已连续且本轮结果完整时，用一次整块 pl.store 写回 GM，删除逐段 pl.move+pl.store 的搬出路径。"
status: "stable"
tags: ["pypto-pro", "mem", "store", "contiguous"]
item_id: "mem-04"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#12-overlapping-writes-program-order-is-not-guaranteed-even-within-one-pipe"
      scope: "把分段搬出改为连续整块 store 前，证明最终地址连续、结果完整且没有未完成或别名写入。"
      boundary: "同 pipe 程序序不能证明写入安全；跨核重叠、原子语义、尾部和 pad 责任不能因合并 store 消失。"
bound_hint: "MTE3"
applicability: "结果在 UB 中已按输出地址连续排布却仍按行/段逐个搬出，且后续无中间变换、结果只需写入最终 GM 区间"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.store 整块搬运与 pl.move 的 UB→UB offset 读"
---
# 技术卡片 mem-04：连续完整结果一次性直写

- **适用 bound**：访存 / MTE
- **一句话**：结果在 UB 中已按输出地址连续排布且本轮完整时，一次 `pl.store` 整块写回；删除逐段 `pl.move` 取段 + `pl.store` 的搬出循环。

## 何时用（诊断特征）

- 计算结果在 UB 中已按输出地址连续排布，却仍按行/段逐个小块 copy out。
- gather/scatter 的非连续分支覆盖了连续输出 case，普通 case 也承担逐段 move 与多次 MTE 发射。
- 后续没有对中间输出作变换，结果只需要写入最终 GM 区间。

## 何时不适用

- 输出有 hole、重复写、交错 layout 或跨核重叠/原子累加语义：必须保留 gather/strided 模板。
- 非连续路径本身已由 DMA stride 高效表达时，不要强行先在 UB 重排再直写。
- 连续区间超出 UB 当前结果范围时，分少量连续 chunk 直写，不要为凑一次 copy 额外重排整块数据。

## 原理

对连续完整的结果，最佳搬出路径是一次或少量大块 `pl.store`。逐段搬出每段都有 `pl.move`（UB→UB 复制）与 `pl.store` 的描述符、同步开销；直写把这些固定开销降为 1 次。连续性作为 tiling 路由条件：连续 case 直接从 UB 写最终 GM，非连续 case 保留逐段模板。收益来自少掉的 copy 描述、UB 内复制与同步，而不是改变最终字节数。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 `y = x * 2` kernel（每批 R 行）；尺寸、地址与循环边界须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

# before：逐行 pl.move 取段 + pl.store 搬出
# for b in pl.range(core_id, batches, num_cores):
#     xb = x_g.next(); acc = acc_g.current()
#     pl.load(xb, x, [b * R, 0])
#     pl.mul(acc, xb, 2.0)
#     for r in pl.range(0, R):
#         row = row_g.next()
#         pl.move(row, acc, offset=[r, 0])
#         pl.store(y, row, [b * R + r, 0])

# after：连续完整区间一次直写
batch_type = pl.TileType(shape=[R, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec)
x_g = pl.make_tile_group(type=batch_type, addrs=0x00000, mutex_ids=[0, 1])
acc_g = pl.make_tile_group(type=batch_type, addrs=0x20000, mutex_ids=[4])

with pl.section_vector():
    for b in pl.range(core_id, batches, num_cores):
        xb = x_g.next()
        acc = acc_g.current()
        pl.load(xb, x, [b * R, 0])
        pl.mul(acc, xb, 2.0)
        pl.store(y, acc, [b * R, 0])      # 一次性整块写回
```

## 性能与验证指标

比较连续与非连续输出两组的 MTE copy 数、UB 内 move 次数与同条件 `Task Duration`。连续直写路径应覆盖完整结果区间；有 hole、重复写或交错 layout 的 case 必须回退并单独验证正确性。

## 技术限制与风险

- 连续性需同时满足目的地址递增、元素无洞、无别名冲突、本轮结果完整四项，不能只检查逻辑索引连续。
- 写回长度只覆盖有效元素；尾块、pad 与空输出不得按对齐长度越界写 GM（配合 mem-01 的尾块路径）。
- 输出存在跨核重叠或原子累加语义时不能改为普通直写。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`
