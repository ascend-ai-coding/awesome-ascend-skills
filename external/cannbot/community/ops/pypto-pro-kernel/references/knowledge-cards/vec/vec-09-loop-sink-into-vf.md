---
type: "PyPTO Performance Optimization Card"
title: "外层循环下沉进 VF"
description: "把 kernel 中逐行调用 VF 的循环下沉到单个 vector_function，使跨行不变量 setup 只执行一次。"
status: "stable"
tags: ["pypto-pro", "vec", "scalar", "loop"]
item_id: "vec-09"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits"
      scope: "将逐行调用 VF 的组织改为一次 VF 调用内处理多行，摊薄调用及跨行不变的 setup 开销。"
      boundary: "本卡聚焦循环下沉与 setup 复用；目标章节还包含 DMA 行批量、metadata 和 worker 分配。仅下沉循环不等于完成整个批处理方案，逐行状态及 mask 仍需正确维护。"
bound_hint: "VEC、SCALAR"
applicability: "kernel 按行重复调用同一 VF，行间无依赖，setup 跨行不变且合并后的寄存器与 Tile 生命周期可容纳"
target_api_gate: "仅限 Ascend 950PR 或 950DT；确认 vector_function 内动态 pl.range、offset 与逐行 mask 语义后采用"
---
# 技术卡片 vec-09：外层循环下沉进 VF

- **适用 bound**：VEC / Scalar
- **一句话**：把 kernel 中逐行调用 VF 的循环折进一个 `@pl.vector_function`，让 mask、广播常量与地址 setup 只初始化一次。

## 何时用（诊断特征）

- `@pl.jit` kernel 在 `pl.range` 中每行调用同一 VF，每次只处理一行。
- trace 上 VF 反复启停，mask 创建、常量广播与地址 setup 周期性重复。
- 各行彼此无依赖，且 setup 跨行不变。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 行间存在数据依赖，setup、valid 或 dtype 随行变化且不能在循环内正确重建。
- 下沉导致活跃寄存器、代码体或 Tile 生命周期扩大并引发 spill 或容量回退。
- 行数太少或单行太长，无法用当前 shape 集和生成物证明 VF 调用/setup 是可归属瓶颈。

## 原理

VF 可以直接接收动态行数并在内部使用 `pl.range`。把不变量 setup 移到 VF 循环外，可把 N 次 VF 调用和 N 次初始化降为一次调用、一次初始化。

## 怎么改（before / after）

以下为 VF 嵌入片段，行间独立性和逐行 valid 必须由具体 kernel 提供。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def rows_vf(src_tile, out_tile, rows: pl.DT_INT64,
            row_pitch: pl.DT_INT64, valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    scale = vf.full(0.5, preg, dtype=pl.DT_FP32)  # setup 一次
    for row in pl.range(0, rows):
        x = vf.load_align(src_tile, row * row_pitch)
        y = vf.mul(x, scale, preg)
        vf.store_align(out_tile, y, preg, row * row_pitch)

# before 形态（kernel 内）：for row in pl.range(...): one_row_vf(..., row)
# after（kernel 内）：rows_vf(src_tile, out_tile, rows, row_pitch, valid)
```

## 性能与验证指标

比较 `Task Duration(us)` 与 setup/SCALAR 占比。**待实测**：行数越多、单行越短，摊薄 VF 调用与 setup 的收益预期越大。

## 技术限制与风险

- 跨行 setup 必须真正不变，各行之间不得有未表达的数据依赖。
- 下沉后一次 VF 的活跃寄存器、代码体或 Tile 生命周期不能导致 spill/容量回退。
- 尾行 valid 不相同时，在循环内按该行重新生成 mask；不能错误复用首行 mask。
- 小到不足以摊平 VF 启动的 case 必须实测决定是否保留独立路径。

“单次尾段 lane <32”可作为不值得另启 VF 的历史经验假设，但 32 不是 API 语义边界；是否合并尾段必须使用当前 shape 集和 950 生成物/trace 验证。

## 参考资料

- 核查线索：目标版本的官方 CV fused 与 FA 示例中，`@pl.vector_function` 内使用 `pl.range` 批量处理数据的方式。
