---
type: "PyPTO Performance Optimization Card"
title: "`vf.reduce_*` 硬件树形归约"
description: "用 vf.reduce_sum、reduce_max 或 reduce_min 替代逐元素或单累加器标量归约。"
status: "stable"
tags: ["pypto-pro", "vec", "reduction", "tree"]
item_id: "vec-07"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-reduction-result"
      scope: "硬件 reduce 的 lane0 结果、后续广播和 FIRST_ELEMENT 写回应分别遵循结果布局。"
      boundary: "多寄存器归约、重关联误差、对齐和实际收益不由单次 reduce 契约证明。"
bound_hint: "VEC"
applicability: "当前归约由标量循环或线性依赖链实现，结果只需单值或可广播，且 mask 与 dtype 满足 reduce 接口"
target_api_gate: "仅限 Ascend 950PR 或 950DT；reduce、full 与 FIRST_ELEMENT 的语义、对齐和 dtype 支持均须核验"
---
# 技术卡片 vec-07：`vf.reduce_*` 硬件树形归约

- **适用 bound**：VEC
- **一句话**：用 `vf.reduce_sum`/`vf.reduce_max`/`vf.reduce_min` 替代逐元素或单累加器标量归约。

## 何时用（诊断特征）

- reduction 是逐元素 Python/kernel 标量循环，或反复取单元素累加。
- VECTOR 出现单累加器 RAW 依赖空泡，SCALARLDST 同时偏高。
- 归约结果只需一个值或后续可在寄存器中广播。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 树形重关联无法满足当前 SPEC 的浮点精度或特殊值语义。
- 调用方逐元素使用本例的归约值却未广播，或单值写回却未使用 FIRST_ELEMENT store。
- mask dtype、有效元素数、Tile 对齐或多寄存器归约边界尚未证明。

## 原理

`vf.reduce_*` 在有效 mask 内做硬件树形归约，结果写在返回寄存器 lane0；依赖链从线性趋近树形深度，并避免标量 round-trip。

## 怎么改（before / after）

以下为嵌入片段，外层 Tile、dtype 和 tail 上下文须另行核验。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def reduce_sum_vf(src_tile, out_tile, valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    src = vf.load_align(src_tile, 0)
    total = vf.reduce_sum(src, preg)
    # 若下一步逐元素消费，广播 lane0；若只需标量结果可直接 FIRST_ELEMENT store
    total_all = vf.full(total, preg)
    vf.store_align(out_tile, total_all, preg)
```

归约跨多个寄存器时，可先对每个寄存器做局部 `vf.reduce_sum`，再按 vec-08 使用多路累加/树形合并；行宽远小于 VL 时参见 vec-06 的 packed 归约。

## 性能与验证指标

比较 `Task Duration(us)`、SCALARLDST 与 VECTOR 空泡。**待实测**：归约长度越大、原标量同步越密，预期收益越大。

## 技术限制与风险

- 浮点求和重关联须按 SPEC 选择累加精度（本例为 FP32），并通过完整精度回归。
- 本卡默认整寄存器归约的值位于 lane0；逐元素消费前 `vf.full`，只存一个元素时使用已验证的 `pl.StoreDist.FIRST_ELEMENT` 变体。
- mask dtype 和有效元素数必须匹配源寄存器。
- 本方法对应的 AscendC `ReduceSum`/`Brcb` 方案将相关 buffer 按 64B 对齐。在 Pro 中，使用 `vf.load_align` 的 TileGroup 基址/行 pitch 仍须按当前 API 的对齐合同布局；若新增 lane0 标量 scratch 或广播 Tile，不能因为换了前端就丢掉这一内存对齐检查。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/Reg-API/reduction/reduce_sum.md`；`Vf.reduce_max/min` 同族接口。
