---
type: "PyPTO Performance Optimization Card"
title: "`vf.mul_dst_add` 融合乘加"
description: "在语义、dtype 和 mask 兼容时，用 vf.mul_dst_add 替换未被编译器融合的相邻乘加。"
status: "stable"
tags: ["pypto-pro", "vec", "fma", "instruction-fusion"]
item_id: "vec-03"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-destination-state"
      scope: "mul_dst_add 读取赋值目标的旧值，融合前必须初始化目标并核对乘数、加数与 mask。"
      boundary: "mul_dst_add 与 mul_add_dst 的隐式 dst 位置不同；有其它消费者的乘积不能直接消除。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/vf.md#215-extreme-value-exp-semantics-and-mul_dst_add-as-true-fma-hardware-evidence-2026-09-30"
      scope: "核对 mul_dst_add 的单次舍入 FMA 语义后再替换分离的 mul 与 add。"
      boundary: "FMA 与两次舍入可能产生不同结果；关联不放宽数值验收，也不说明编译器尚未自动融合。"
bound_hint: "VEC"
applicability: "存在 x 乘 weight 加 bias，编译结果未自动融合，中间乘积无其他消费者且 dtype 与 mask 兼容"
target_api_gate: "仅在当前 Ascend 950PR 或 950DT PyPTO-Pro 公开 Vf.mul_dst_add 语义和生成 vmadd 均已复核时采用"
---
# 技术卡片 vec-03：`vf.mul_dst_add` 融合乘加

- **适用 bound**：VEC
- **一句话**：在 `@pl.vector_function` 中将 `vf.mul` + `vf.add` 替换为公开的 `vf.mul_dst_add`，减少指令与中间寄存器。

## 何时用（诊断特征）

- hot VF 中存在相邻的 `x * weight + bias`，三者 shape、dtype 与 mask 兼容。
- `aiv_vec_ratio` 高，乘和加是主要有效计算，且中间乘积没有其他消费者。
- trace/编译结果确认两条操作没有被编译器自动融合。

AscendC 历史经验以“Compute 占总时间 >60%”作为筛选线，并观察到 A5 上 high-level vector intrinsic 的 mask/repeat/sync 固定开销可能盖过有效计算。这两点在 PyPTO-Pro 中只能当作候选信号：是否已自动融合、VF 启动和最终收益均要查当前生成物/trace，不得把历史经验线当成 Pro 的硬合同。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 目标原值、两个源操作数或加数的次序不符合 vf.mul_dst_add 的隐式 dst 语义。
- 中间乘积有其它消费者，dtype 或 mask 不兼容，或编译器已经生成等价融合指令。

## 原理

PyPTO-Pro `vf.mul_dst_add(src0, src1, preg)` 映射硬件 `vmadd`，语义为“当前目标寄存器 × `src0` + `src1`”。Python 赋值的左值同时代表调用前的目标输入与调用后的结果，因此目标必须先初始化。它与 `vf.mul_add_dst` 的参数语义不同，不能混用。

A5 AscendC 方案通过 `GetVecLen()/sizeof(dtype)` 推导 lane，避免把 FP32 的 64 或 block 元素数当成跨 dtype/跨代常量。不能把 AscendC 的 `GetVecLen()` 当成本卡的公开 PyPTO-Pro Python 参数；固定 lane 只能来自 950 支持表、dtype 专用 TilingKey/常量与生成物验证，不能从 FP32 示例外推。

## 怎么改（before / after）

以下为嵌入片段，须放入已核验的 VF 与外层 Tile 上下文。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def mul_add_vf(x_tile, weight_tile, bias_tile, out_tile,
               valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x = vf.load_align(x_tile, 0)
    weight = vf.load_align(weight_tile, 0)
    bias = vf.load_align(bias_tile, 0)

    # before:
    # product = vf.mul(x, weight, preg)
    # out = vf.add(product, bias, preg)

    # after：x 是已初始化的隐式 dst，语义为 x = x * weight + bias
    x = vf.mul_dst_add(weight, bias, preg)
    vf.store_align(out_tile, x, preg)
```

外层 kernel 使用 `@pl.jit(auto_mutex=True)`、Vec Tile/TileGroup、`pl.load`/`pl.store` 调用该 VF。

## 性能与验证指标

AscendC 外部历史案例的 high-level `Mul+Add` → MicroAPI `MulDstAdd` 曾报告约 **-40% `aiv_time`**，仅作 `unverified_external_historical` 方向性线索，原始报告与测试条件待补，**不是 PyPTO-Pro 本卡已实测值**。实际采用时通过完整正确性测试，并比较同 manifest/目标 kernel/采集协议下的 `Task Duration(us)`。

## 技术限制与风险

- 必须逐项核对语义：目标原值是乘数，`src0` 是另一乘数，`src1` 是加数；参数颠倒会静默产生错误结果。
- 只有中间乘积无其他消费者、dtype/mask 兼容时才能融合。
- 使用 `pl.DYNAMIC`/tiling 传入有效长度，不硬编码 lane 数；mask 由实际 valid 元素数生成。
- “尾段单次 lane <32 往往不值得单独启 VF”也仅是启动开销启发式；不要无条件删除尾路径，必须用本 case 和当前芯片实测决定。
- 修改后必须跑全量正确性与同条件性能回归。

## 参考资料

- PyPTO-Pro：`python/pypto_pro/language/_vf_api.py::Vf.mul_dst_add`；官方 ST `python/tests/st/pypto_pro/frontend/vf_api/test_vf_basic_ops.py::_vf_kernel_11_copy_madd_0`。
