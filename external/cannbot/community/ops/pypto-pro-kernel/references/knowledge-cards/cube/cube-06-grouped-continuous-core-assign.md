---
type: "PyPTO Performance Optimization Card"
title: "GroupedMatmul 组间连续核分配（全局线性 tile 空间）"
description: "多组 matmul 合并为单次 launch，把所有组的输出 tile 编入一个全局线性空间，核按 block_idx 步幅遍历该空间，下一组的 tile 紧跟上一组分配，避免每组都从 0 核重新分配造成的组尾空闲核浪费。"
status: "stable"
tags: ["pypto-pro", "cube", "grouped-matmul", "scheduling", "multi-core"]
item_id: "cube-06"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: same_mechanism
      target:
        skill: pypto-pro-kernel
        id: "contract:grouped-ownership"
      scope: "将多组独立 GEMM 的输出 tile 连续编号，避免每组重启 worker 分配造成余数工作反复落在低编号 worker 上。"
      boundary: "本卡示例使用固定 tiles_per_group；目标契约还讨论运行时组长、动态前缀和空组。关联不扩大本卡覆盖范围，也不证明当前负载失衡或存在性能收益。"
bound_hint: "MAC"
applicability: "单 kernel 承载多组 matmul（MoE 分组、batch matmul 等），各组 tile 数不被核数整除，组数较多"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.get_block_idx/pl.get_block_num、三维 GM 张量的组维寻址与单次 launch"
---
# 技术卡片 cube-06：GroupedMatmul 组间连续核分配（全局线性 tile 空间）

- **适用 bound**：调度（组尾空闲核累积浪费）
- **一句话**：所有组的 tile 进同一个线性空间，核连续分配、跨组回绕，组间不重启核计数。

## 何时用（诊断特征）

- 多组 matmul（每组独立的 A/B/out，可按组维索引寻址）。
- 单组 tile 数大概率不被核数整除，每组独立分配时组尾空闲核随组数累积。
- 各组 tile 粒度相同或可被同一映射函数覆盖。

## 何时不适用

- 单组 tile 数远大于核数且整除：组内已满载，组间分配方式无关。
- 各组 M/N/K 差异大到 tile 粒度无法统一：先统一基本块或分组编译。
- 组间存在顺序依赖：线性化并行不成立。

## 原理

每组独立从 0 核起分配时，每组的组尾余数都变成空闲核。把 `G × tiles_per_group` 个 tile 编入全局线性空间，核按 `tid = core_id + round × num_cores` 遍历，`g = tid // tiles_per_group`、`t = tid % tiles_per_group` 还原组号与组内 tile——等价于下一组接着上一组的核继续分配并循环回绕，空闲只出现在全局末尾一次。

## 怎么改（before / after）

以下为嵌入片段，截取自已上板验证的 grouped matmul kernel（每组 M×N×K 单 K 块，`tiles_per_group = (M/128)×(N/128)`）；组维布局、Tile 尺寸与地址须按当前 DESIGN 核验。

```python
import pypto_pro.language as pl

cid = pl.get_block_idx()
ncore = pl.get_block_num()

# before：每组独立从 0 核起分配，组尾余数核空闲且随组数累积
# with pl.section_cube():
#     for g in pl.range(0, a.shape[0]):
#         for t in pl.range(cid, TPG, ncore):
#             am = a_l1.next(); bm = b_l1.next()
#             pl.load(am, a, [g, 0, 0])
#             pl.load(bm, b, [g, 0, t * TILE])
#             pl.move(a_left.current(), am)
#             pl.move(b_right.current(), bm)
#             pl.matmul(acc.current(), a_left.current(), b_right.current())
#             pl.store(out, acc.current(), [g, 0, t * TILE])

# after：G × TPG 编入全局线性空间，核按 block_idx 步幅连续遍历
with pl.section_cube():
    total = a.shape[0] * TPG                 # 组间连续编号的全局 tile 空间
    for tid in pl.range(cid, total, ncore):
        g = tid // TPG                       # 组 g 的 tile 紧跟上一组
        t = tid % TPG
        am = a_l1.next()
        bm = b_l1.next()
        al = a_left.current()
        br = b_right.current()
        ac = acc.current()
        pl.load(am, a, [g, 0, 0])
        pl.load(bm, b, [g, 0, t * TILE])
        pl.move(al, am)
        pl.move(br, bm)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [g, 0, t * TILE])
```

`a/b/out` 以组维作为前导 batch 维（`[G, M, K]`、`[G, K, N]`、`[G, M, N]`）寻址，`pl.load`/`pl.store` 的 offsets 首位固定组下标；启动侧一次 `kernel[stream, block_dim](a, b, out)`。组内各 tile 的 L1→L0→MMAD→drain 写法与单组 matmul 完全一致，本卡只改任务编号。

## 性能与验证指标

观察活跃核数分布与 Task Duration；负载均衡目标为各核 tile 数差 ≤1。正确性按常规精度回归，重点核对组号/组内序号的还原映射（跨组边界 tile 不写错组）。

## 技术限制与风险

- 全局与组内索引混用是常见缺陷：组内偏移必须以组起点为基准。
- 组内各 tile 的 K 循环、尾块处理与单组 matmul 完全一致，本卡只改任务编号。
- M 轴分组且各组 M 由 device 数据决定（group_list 类）时，组边界不在编译期可知，须另行设计动态映射，不在本卡范围。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/system_variables/get_block_idx.md`、`docs/zh/api/pro_api/SIMD-API/system_variables/get_block_num.md`
- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`（高维张量 offsets）
