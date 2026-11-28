---
type: "PyPTO Performance Optimization Card"
title: "小定长循环显式展开"
description: "把 VF 内编译期已知的极小循环显式展开，以删除循环控制并暴露指令级并行。"
status: "stable"
tags: ["pypto-pro", "vec", "loop", "unroll"]
item_id: "vec-10"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/constraints.md#for-must-use-plrange"
      scope: "显式展开前先区分编译期 trip count 与运行期 pl.range，逐分支保留 offset、mask 和覆盖范围。"
      boundary: "循环语法契约不推荐固定展开度，也不证明独立指令增多必然比 spill 或代码体积成本更有利。"
bound_hint: "VEC"
applicability: "trip count 编译期已知且很小，展开后的数据、offset 与 mask 等价"
target_api_gate: "仅限 Ascend 950PR 或 950DT；只使用已核验 VF 基础算术，是否源码展开由当前 parser、生成物和 TilingKey 决定"
---
# 技术卡片 vec-10：小定长循环显式展开

- **适用 bound**：VEC / 无 bound
- **一句话**：VF 内编译期已知的极小循环显式展开，消除循环控制并暴露指令级并行。

## 何时用（诊断特征）

- 循环次数很小且由算法常量/TilingKey 编译期确定（如 K=3）。
- VECTOR 有可由独立操作填充的空泡，循环控制占比可见。
- 展开后不会显著增加寄存器压力或 I-cache 压力。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- trip count 动态或较大，必须保留运行期主循环和尾段。
- 展开后寄存器 spill、I-cache 或代码体积成本抵消收益。
- 不同展开分支所需的有效 lane 或 mask 粒度不同，却复用了同一个 mask。

## 原理

显式列出各迭代，去掉计数和分支，并让独立 `vf.mul` 同时可调度；树形 `vf.add` 再合并结果。

## 怎么改（before / after）

以下嵌入片段仅表达三组固定布局与显式展开。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def k3_vf(src_tile, weight_tile, out_tile, valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    # before：for k in pl.range(0, 3): acc = add(acc, mul(...))
    x0, w0 = vf.load_align(src_tile, 0), vf.load_align(weight_tile, 0)
    x1, w1 = vf.load_align(src_tile, 64), vf.load_align(weight_tile, 64)
    x2, w2 = vf.load_align(src_tile, 128), vf.load_align(weight_tile, 128)
    p0 = vf.mul(x0, w0, preg)
    p1 = vf.mul(x1, w1, preg)
    p2 = vf.mul(x2, w2, preg)
    sum01 = vf.add(p0, p1, preg)
    out = vf.add(sum01, p2, preg)
    vf.store_align(out_tile, out, preg)
```

这个片段的输入布局不变量是：`src_tile` 与 `weight_tile` 都含三个相邻、各 64 个 FP32 元素且起点满足对齐要求；三组使用同一个有效长度，`out_tile` 写回一组逐 lane 累加结果。若三组的尾长不同，必须分别生成 mask，不能复用 `preg`。

## 性能与验证指标

比较 `Task Duration(us)`、循环控制与 spill。**待实测**：只对极小定长循环预期有益；展开越大越可能被代码膨胀抵消。

## 技术限制与风险

- 只用于编译期定长的小循环；动态 trip count 用 vec-08 的运行期主循环+尾段。
- 展开后检查寄存器 spill 与 I-cache；回退时降低展开度。
- offset、mask 与每个展开分支必须覆盖原循环相同的数据和边界。
- 只有三组共享 dtype、对齐与有效长度时才可直接复用示例中的一个 `preg`；否则逐组构造 mask。
