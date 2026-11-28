---
type: "PyPTO Performance Optimization Card"
title: "扁平二维 work grid 分解（铺满 AI Core）"
description: "存在大量独立工作项但当前仅按串行维度映射、只占用部分 AI Core 时，把工作项按二维 (M, N) 网格展平分配给全部 AI Core。"
status: "stable"
tags: ["pypto-pro", "pipeline", "scheduling"]
item_id: "pipe-06"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#count-the-complete-output-grid-before-assigning-workers"
      scope: "把所有独立输出轴展成完整线性任务网格，按实际 block 数分配并恢复坐标。"
      boundary: "递推或有依赖轴必须留在 owner 内；任务尾守卫、改变的访问局部性与最忙 worker 路径须另证。"
bound_hint: "PIPELINE、SCALAR"
applicability: "算子由大量相互独立的工作项构成，当前映射只占用部分 AI Core，逐核泳道存在大段空窗"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖 pl.get_block_idx/pl.get_block_num 与多维 GM 张量的组维寻址"
---

# 技术卡片 pipe-06：扁平二维 work grid 分解（铺满 AI Core）

- **适用 bound**：PIPELINE、SCALAR
- **一句话**：把独立工作项（chunk、head、batch、分片）展平成二维 (M, N) 网格，分配给全部 AI Core，消除核闲置。

## 何时用（诊断特征）

- 工作项相互独立（无跨项依赖），但当前只按某一个串行维度映射，参与计算的 AI Core 数显著少于物理核数。
- 逐核泳道存在大段空窗，或总工作量按核数分摊后每核远超单核容量。
- 存在唯一的串行维度被错误地用作并行映射，导致其余维度未参与分摊。

## 何时不适用

- 工作项之间存在串行依赖（如状态递推的 chunk 维），不能把串行维度展平并行。
- 工作项数少于物理核数，展平不增加并行度。
- 展平后相邻工作项的 GM 访问失去局部性，反使访存退化。

## 原理

并行度由可展平的独立工作项总数决定。把 (工作项, 头/分片) 两维展平后，逻辑块索引按全局线性编号
分配给物理核，核数与工作项数解耦，空闲核参与分摊；串行维度保留在核内层循环。核负载均衡由
展平后的任务均分保证。

## 怎么改（before / after）

以下为结构嵌入片段，展示映射方式改动；所用 `pl.*` 形态（`pl.get_block_idx`/`pl.get_block_num`、
`pl.range` 步幅遍历、`pl.min`）来自历史结构片段，不能继承为当前目标的完整验收。片段是教学级
最小示意：例题设定 `nc` 个相互独立的工作项（如 chunk），每项再分 `n_bh` 个头/分片，
`n_tasks = nc * n_bh`；`process_item` 代表按 `(c, bh)` 解码并处理一个工作项。维度解码、
任务均分与边界守卫须按当前 kernel 核验后嵌入。

**before（只按工作项维映射，头/分片维逐核串行，部分核闲置）：**

```python
# launch grid = nc：核号即工作项号；nc 小于物理核数时，其余核全程空转
with pl.section_cube():
    c = pl.get_block_idx()                  # 0..nc-1，只覆盖工作项维
    for bh in pl.range(0, n_bh):            # 头/分片维未参与核间分摊，逐核串行
        process_item(c, bh)
```

**after（两维展平为全局线性任务号，铺满全部 AI Core）：**

```python
# launch grid = 物理核数；核数与工作项数解耦，空闲核参与分摊
with pl.section_cube():
    cid = pl.get_block_idx()
    for it in pl.range(0, n_iter):
        task = cid + it * pl.get_block_num()   # 扁平全局任务号
        c = task % nc                           # 工作项维（如 chunk）
        bh = task // nc                         # 头/分片维
        if task < n_tasks:                      # n_tasks 不整除核数时的边界守卫
            process_item(c, bh)
```

**关键陷阱**：`task < n_tasks` 守卫不可省——展平后总遍历次数按核数向上取整，缺守卫的末轮会对
越界 `(c, bh)` 读写 GM；多组 matmul 的组尾空闲场景（各组 tile 数不整除核数）优先看 cube-06 的
全局线性 tile 空间，两者机制同源但适用对象不同。

## 性能与验证指标

比较有效核数、逐核负载离散度、空窗占比与同条件 `Task Duration`；正确性按完整套件回归，覆盖
任务数不整除核数时的边界守卫与尾任务均分。

## 技术限制与风险

- 串行维度不得展平并行；展平仅针对无依赖维度。
- 任务数不整除核数时须守卫 `task < n_tasks`，避免越界读写。
- 展平可能改变 GM 访问顺序与局部性；须复核 L2 命中与访存字节。

## 参考资料

- [当前 GEMM grid/worker mapping](../../../../pypto-pro-kernel/references/api/gemm.md#count-the-complete-output-grid-before-assigning-workers) 与 [batch-row folding](../../../../pypto-pro-kernel/references/api/gemm.md#fold-shared-rhs-batch-rows-when-their-physical-mapping-permits-it)：折叠前先证明原坐标与物理连续性。
