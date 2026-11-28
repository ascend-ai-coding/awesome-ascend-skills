---
type: "PyPTO Performance Optimization Card"
title: "寄存器溢出时按条件拆分 VF"
description: "在生成物出现 spill/reload 时，按可证明条件拆成活跃值集合不同的专用 VF。"
status: "stable"
tags: ["pypto-pro", "vec", "register-pressure", "specialization"]
item_id: "vec-12"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-loop-control"
      scope: "在 VF 热循环外选择真正删除无关活跃值的专用路径，减少重复分派及寄存器存活范围。"
      boundary: "循环分支外提不等价于已消除 spill；small/full VF 的逻辑、覆盖及压力变化仍需生成物与数值验证。"
bound_hint: "VEC"
applicability: "目标路径生成物或 trace 出现 spill/reload，分支条件可证明，且专用 VF 能真正删除整段逻辑和活跃值"
target_api_gate: "仅限 Ascend 950PR 或 950DT；通过 TilingKey 或合法控制流分流，RegTraitNumTwo 仅作后端风险模型"
---
# 技术卡片 vec-12：寄存器溢出时按条件拆分 VF

- **适用 bound**：VEC / 无 bound
- **一句话**：trace/编译结果出现 preg spill/reload 时，选一个能真正删除整段逻辑与活跃值的 shape/TilingKey 分支，把大 VF 拆成寄存器集合不同的专用 VF。

## 何时用（诊断特征）

- VF 段出现成对 spill/reload，打断 VECTOR 流。
- 增大展开度或融合后性能下降，访存指标却不是主因。
- int64/uint64/complex 等可能使用多物理寄存器表示，源码变量数看似不多但实际压力更高。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 没有可归属到目标路径的 spill/reload，或性能下降由访存、同步和算法工作量主导。
- small 路径仍创建 full 路径的活跃值，两个 VF 生成同构代码，或边界条件不能覆盖全部 shape。
- 需要把 RegTraitNumTwo 或未经 value ST 的高低位 de_interleave 当作公开 Python API。

## 原理

寄存器压力取决于同时存活的物理值，而不是 Python 变量总数。在 arch3510 **没有原生 int64 计算单元**的 AscendC 实现中，用 `RegTensor<T, RegTraitNumTwo>` 表达 int64：一个逻辑 int64 占两个 32-bit 子寄存器，因此 `value1`/`level1` 两个大-R 逻辑值会额外占 4 个 preg。

`RegTraitNumTwo` 是后端 C++ 表示，不能把该类型名写成可调用 Pro Python API；对应的公开能力须按目标版本核验。本卡仅把它用作“后端/trace 风险模型”：以实际生成代码和 spill/reload 为准。拆分必须让 small-R 路径根本不创建 `value1`/`level1` 及其依赖，而不是复制两个相同的 load/store 函数。

## 怎么改（before / after）

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def reduce_max_small_r_vf(value_tile, out_value_tile,
                          dim_r: pl.DT_INT64):
    # 前置条件：0 < dim_r <= 64；没有 value1/level1。
    preg = vf.update_mask(dim_r, dtype=pl.DT_FP32)
    value0 = vf.load_align(value_tile, 0)
    best0 = vf.reduce_max(value0, preg)
    vf.store_align(out_value_tile, best0, preg,
                   dist=pl.StoreDist.FIRST_ELEMENT)

@pl.vector_function
def reduce_max_full_r_vf(value_tile, out_value_tile,
                         dim_r: pl.DT_INT64):
    # 前置条件：64 < dim_r <= 128；跨块路径确实保留 value1/level1。
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    tail = pl.min(dim_r - 64, 64)
    tail_mask = vf.update_mask(tail, dtype=pl.DT_FP32)
    value0 = vf.load_align(value_tile, 0)
    value1 = vf.load_align(value_tile, 64)
    level0 = vf.reduce_max(value0, full)
    level1 = vf.reduce_max(value1, tail_mask)
    best = vf.max(level0, level1, full)
    vf.store_align(out_value_tile, best, full,
                   dist=pl.StoreDist.FIRST_ELEMENT)

# typed @pl.jit 内按 dim_r 分流；若可编译期专门化，优先放入 TilingKey：
# if dim_r <= 64: reduce_max_small_r_vf(...)
# else: reduce_max_full_r_vf(...)
```

这段是数值闭合的 **value-only reduce-max** 嵌入片段：small 路径掩蔽首块，full 路径对第二块使用独立尾掩码，并只写归约结果所在的 lane0。它只用来表达 argmax-with-value 优化中的活跃值差异，不是假装完整 argmax-with-index；索引比较和 tie-break 必须按具体算子补齐。这里把 `dim_r == 64` 明确归入 small 路径，避免 full 路径构造零长度尾掩码。关键验收点是 small-R 生成代码里不存在 full 路径的 `value1`/`level1` 及其 spill，而非函数名不同。

另一个辅助降压手段是：对后端双寄存器表示的高/低位，尝试用寄存器内 `vf.de_interleave` 合并布局，替代 `Mul(twoMask) + ReduceSum(twoMask)` 的中间值与掩码链。该替换依赖具体 int64 表示与 lane 映射；本卡未提供可直接复制的通用 value ST，所以保留为 capability-gated 候选，必须用完整 value/index oracle 验证后再采用。

## 性能与验证指标

可精确归属目标 `Op Name` 的 trace/生成物（当前工具链可得时）必须确认目标路径的 spill/reload 消失，再比较 `Task Duration(us)`。**待实测**；不能只数 Python 局部变量。

## 技术限制与风险

- 两条 VF 路径必须功能等价并覆盖边界；完整正确性测试必须逐路径命中。
- small-R 必须实际删除 full-R 活跃值；两个同构 VF 不构成优化。
- `RegTraitNumTwo` 是 AscendC/后端 C++ 表示，不是可直接调用的 PyPTO-Pro Python API。
- 多路展开导致压力时先降低展开度；TilingKey 过多会增加编译缓存。
- `de_interleave` 的高低位替换未经当前算子数值证明时不得采用。

## 参考资料

- 专用化方法： argmax-with-value 的 `dimR < VL` 分流、删除 `value1/level1` 省 4 preg、以及 `DeInterleave` 替代中间 mask/reduce 链。
- PyPTO-Pro：`vf.de_interleave` 公开 API；实际物理寄存器占用以当前后端生成代码/trace 为准。
