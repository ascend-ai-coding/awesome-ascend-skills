---
type: "PyPTO Performance Optimization Card"
title: "归约累加链多路展开"
description: "把单累加器跨迭代 RAW 链改成多路独立累加和树形合并，并为每组生成独立 tail mask。"
status: "stable"
tags: ["pypto-pro", "vec", "reduction", "unroll"]
sources: [{"id": "mul-add-dst-api", "resource": "https://pypto.gitcode.com/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/mul_add_dst.html", "title": "vf.mul_add_dst 官方 API"}]
item_id: "vec-08"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#choose-the-horizontal-reduction-frequency"
      scope: "用独立 lane/寄存器累加器延后横向归约，并比较依赖链长度与寄存器压力。"
      boundary: "独立累加、双块折叠和分层合并不是同一改法；尾部 MERGING 支持及重关联误差须单独证明。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-destination-state"
      scope: "mul_add_dst 必须读取已初始化的各路累加器，并保留尚需的 inactive lane 状态。"
      boundary: "隐式目标契约不代表当前版本支持 MERGING，不能把 ZEROING 作为保留旧累加值的替代。"
bound_hint: "VEC"
applicability: "结合性能指标及源码、生成物或 trace 判断单累加器 RAW 链是瓶颈且 spill 不是主因，归约长度足以摊销多累加器与最终合并"
target_api_gate: "仅限 Ascend 950PR 或 950DT；示例所需 mul_add_dst MERGING 当前文档不支持，采用该示例前须另证；另核验 reduce、动态 pl.range 与展开度"
---
# 技术卡片 vec-08：归约累加链多路展开

- **适用 bound**：VEC / 无 bound
- **一句话**：单累加器跨迭代串行时，改成 N 路独立寄存器累加，再树形合并；每个数据组独立生成 tail mask，并用 `vf.mul_add_dst` 的 MERGING 语义保持 inactive lane。

## 何时用（诊断特征）

- `@pl.vector_function` 内是 sum/dot-product 的单寄存器 loop-carried dependency。
- VECTOR 周期性空泡而 MTE 不忙，表现为等待累加器写回。
- 归约长度足以摊平额外 load 与最终合并成本。

先区分两类空泡：本卡解决的是单个累加器的 RAW 串行链；trace/生成物出现 spill→reload 时先核对其成本，若寄存器压力是主因，应先走 vec-12，避免增加累加器放大压力。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- spill/reload 主导当前瓶颈，或主因不是单累加器 RAW 依赖链。
- 归约过短，额外 load、累加器和树形合并成本无法摊销。
- 各数据组的 tail mask 不正确、尾部 inactive lane 的历史累加值未被保留，或重关联精度不合格。

## 原理

`acc = vf.add(acc, tmp, preg)` 的相邻迭代有 RAW 真依赖。四路独立累加器允许多组 load/multiply-accumulate 交错发射，最后树形合并。对于 `acc += a*b`，对应接口是 `vf.mul_add_dst(a, b, mask)`，语义为 `dst = a*b + dst`；`vf.mul_dst_add` 是 `dst = dst*a + b`，不能替换本卡累加式。

末组 tail 的 inactive lane 必须保留累加器旧值，本例以 `mode=pl.MergeMode.MERGING` 表达这一要求（能力缺口见下）。每组从 `total` 和自己的 offset 重新算 mask；一个尾 mask 不能复用给所有组。

## 怎么改（before / after）

**能力门控示意**：2026-09-05 核查的官方 `mul_add_dst` 文档仍将 `MERGING` 标为不支持。[^mul-add-dst-api] 以下示例依赖 MERGING，目标版本支持未确认时仅用于理解方法；不依赖 MERGING 的多路累加实现可按当前 API 能力验证。含 inactive lane 的尾部累加不得直接改为 ZEROING。示例的物理容量、padding 与展开度须按实际输入和目标布局确定。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

LANES = 64

@pl.vector_function
def dot_unroll4_vf(a_tile, b_tile, out_tile, total: pl.DT_INT64):
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc0 = vf.full(0.0, full, dtype=pl.DT_FP32)
    acc1 = vf.full(0.0, full, dtype=pl.DT_FP32)
    acc2 = vf.full(0.0, full, dtype=pl.DT_FP32)
    acc3 = vf.full(0.0, full, dtype=pl.DT_FP32)
    groups = (total + LANES - 1) // LANES
    main = groups // 4

    for i in pl.range(0, main):
        base = i * 4
        live0 = pl.min(LANES, total - (base + 0) * LANES)
        live1 = pl.min(LANES, total - (base + 1) * LANES)
        live2 = pl.min(LANES, total - (base + 2) * LANES)
        live3 = pl.min(LANES, total - (base + 3) * LANES)
        m0 = vf.update_mask(live0, dtype=pl.DT_FP32)
        m1 = vf.update_mask(live1, dtype=pl.DT_FP32)
        m2 = vf.update_mask(live2, dtype=pl.DT_FP32)
        m3 = vf.update_mask(live3, dtype=pl.DT_FP32)
        a0 = vf.load_align(a_tile, (base + 0) * LANES)
        a1 = vf.load_align(a_tile, (base + 1) * LANES)
        a2 = vf.load_align(a_tile, (base + 2) * LANES)
        a3 = vf.load_align(a_tile, (base + 3) * LANES)
        b0 = vf.load_align(b_tile, (base + 0) * LANES)
        b1 = vf.load_align(b_tile, (base + 1) * LANES)
        b2 = vf.load_align(b_tile, (base + 2) * LANES)
        b3 = vf.load_align(b_tile, (base + 3) * LANES)
        acc0 = vf.mul_add_dst(a0, b0, m0, mode=pl.MergeMode.MERGING)
        acc1 = vf.mul_add_dst(a1, b1, m1, mode=pl.MergeMode.MERGING)
        acc2 = vf.mul_add_dst(a2, b2, m2, mode=pl.MergeMode.MERGING)
        acc3 = vf.mul_add_dst(a3, b3, m3, mode=pl.MergeMode.MERGING)

    for group in pl.range(main * 4, groups):
        live = pl.min(LANES, total - group * LANES)
        mask = vf.update_mask(live, dtype=pl.DT_FP32)
        a = vf.load_align(a_tile, group * LANES)
        b = vf.load_align(b_tile, group * LANES)
        acc0 = vf.mul_add_dst(a, b, mask, mode=pl.MergeMode.MERGING)

    sum01 = vf.add(acc0, acc1, full)
    sum23 = vf.add(acc2, acc3, full)
    lane_sum = vf.add(sum01, sum23, full)
    dot = vf.reduce_sum(lane_sum, full)
    vf.store_align(out_tile, dot, full, dist=pl.StoreDist.FIRST_ELEMENT)
```

前置不变量：`total > 0`；`a_tile`/`b_tile` 物理容量至少为 `ceil(total/64)*64` 个 FP32，末组 padding 可安全 load；`out_tile` 至少有一个 FP32。累加器从零初始化；当前尾 mask 未选中的 lane 保留此前有效组的累加值，从未参与累加的 lane 仍为零，因此最终用 full mask 合并和 reduce 不会把尾部垃圾计入 dot。

若目标只是逐 lane 的跨组和而非标量 dot，应改函数名并定义向量输出 shape/mask；不能叫 dot 却省略 `reduce_sum`。

## 重算归约的双块折叠变体

分清三种不同候选：独立累加链用于缩短依赖链（见上文）；双块折叠先按对合并输入再归约，
减少归约发出次数；分层合并处理块级partial的状态与顺序。它们可以组合，但不互相等价。
以下仅为算法级示意，`load_block`、`reduce_block`和`pairwise_merge`不是PyPTO-Pro API：

```text
partials = []
for first in range(0, block_count, 2):
    left = load_block(first)
    right = load_block(first + 1) if first + 1 < block_count else zeros_like(left)
    partials.append(reduce_block(left + right))   # 每对块一次归约
while len(partials) > 1:
    partials = pairwise_merge(partials)           # 分层合并，奇数项原样保留
result = partials[0] if partials else additive_identity
```

源块尾部必须先屏蔽/补零；不同块的partial要保存在互不覆盖的状态中，不能提前复用仍被消费的
存储。Main/Fold阶段的组织还须满足当前工作流的启动、同步和输出合同；此伪代码不是已验证的
完整两遍均值/方差或Main/Fold实现，也不承诺某个展开度必然更快。

### 独立的补偿累加示意

下列VF片段描述Kahan式补偿状态更新，与上面的成对折叠和分层合并是不同机制；
它本身不减少块归约次数，也不能作为双块折叠的实现或性能证据。`datablock=True`
只选择归约粒度，不会自动生成Main/Fold或分层合并：

```python
adjusted = vf.sub(delta, compensation, full)
next_accum = vf.add(accum, adjusted, full)
rounded = vf.sub(next_accum, accum, full)
next_compensation = vf.sub(rounded, adjusted, full)
compensation = vf.move(next_compensation, full)
accum = vf.move(next_accum, full)
partials = vf.reduce_sum(accum, full, datablock=True)
```

分块数、块内累加范围与合并顺序按数值预算推导；窄整数状态按最坏输入推导上界后再决定何时
发出部分结果；改变归约顺序须通过完整精度回归。本节是概念方法，不以未保留的外部实机样例
声明已验证；局部累加链、成对归约或补偿累加中的一项通过，也不能验证完整Main/Fold。

## 性能与验证指标

比较 `Task Duration(us)`、VECTOR 空泡与 spill/reload。**待实测**：四路只是候选，不是固定最优。

## 技术限制与风险

- 动态长度必须有运行期主循环和尾段；最后一个数据组独立生成 mask。
- 尾组使用 MERGING 保留 inactive lane；ZEROING 会清掉同一累加器先前组的值。
- 多路/树形合并改变浮点累加顺序，使用适当中间精度并通过完整精度回归。
- 展开度过大可能 spill；回退时降低 N 或配合 vec-12。

## 参考资料

- 核查线索：目标版本的 `Vf.mul_add_dst`、lightning-indexer VF ST 的首轮 `mul`/后续 `mul_add_dst` 累加，以及 softmax ST 的四路累加与树形合并。

[^mul-add-dst-api]: [vf.mul_add_dst 官方 API](https://pypto.gitcode.com/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/mul_add_dst.html)。
