---
type: "PyPTO Performance Optimization Card"
title: "标量 round-trip 改为寄存器广播与批量计算"
description: "把逐行取标量、计算和写回改为寄存器广播与批量运算，并严格区分单标量、八值 block 和多行布局能力。"
status: "stable"
tags: ["pypto-pro", "vec", "scalar", "broadcast"]
item_id: "vec-04"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-reduction-result"
      scope: "同 VF 中用归约 lane0 的寄存器广播替代 getval/setval 标量往返。"
      boundary: "单值结果、逐 lane 广播和跨阶段持久化是不同路径；本关联不证明八值 BLK 广播或 packed 多行布局。"
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-outer-reduce.md#metadata-and-invariant-inputs"
      scope: "跨阶段需保留的逐行统计量可按批量 UB metadata 与 BRC_B32 消费。"
      boundary: "BRC_B32 只广播一个 B32 元素，不是八值 BLK；物理对齐、真实尾行和原统计量合同仍由调用方保证。"
bound_hint: "VEC、SCALAR"
applicability: "每行归约后存在标量往返或不必要的 Tile store→load；消费者可用同 VF full、跨阶段单 B32 广播，或已通过八哨兵 value ST 的 BLK 路径"
target_api_gate: "仅限 Ascend 950PR 或 950DT；Vf.full 与 BRC_B32 可作保守路径，BLK 和 stride-zero 多行布局须另证"
---
# 技术卡片 vec-04：标量 round-trip 改为寄存器广播与批量计算

- **适用 bound**：VEC / Scalar
- **一句话**：把 per-row 的“取标量→算→写回标量”改成“reduce→寄存器广播/8 值 block 广播→批量倒数与缩放→masked 写回”，消灭热循环里的 scalar pull 与 V_S/S_V 同步。

## 何时用（诊断特征）

- SCALAR/SCALARLDST 忙，VF/行循环中反复把 lane0 写 Tile、同步、`getval`，计算后再 `setval`。
- dynamic quant、per-row scale、normalization、softmax 在每行归约后执行标量倒数/乘法。
- 每行一次 VF setup，VECTOR 流水被 V_S/S_V 同步反复排空。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 改写会丢失跨阶段、跨核或其它消费者仍需的归约结果，或破坏其持久化与同步。
- 把 BRC_B32 单值广播误当成八值 Brcb/BLK，或采用 BLK/stride-zero 路径却未用哨兵测试证明其 lane-to-row 与布局。
- 改变 reduction 顺序、Cast 位宽或广播布局后无法满足当前算子的精度和完整 shape/tail 合同。

## 原理

本卡使用默认整寄存器归约，归约值位于 lane0；如果消费者仍在同一 VF，用 `vf.full(reg, preg)` 直接广播到有效 lane，就不产生 Tile store→load 或 Python scalar。若 scale 跨阶段持久化，则把一组 scale 存入 Tile，再按消费布局选择广播：

- `pl.LoadDist.BRC_B32` 只读取 **一个 B32 元素**并广播整寄存器，适合单 scale；
- AscendC `Brcb`/`DIST_BLK` 读取 **一个 32B block（8 个 FP32）**并把这 8 值 pattern 分发到 VL，适合一次处理 8 行 scale；PyPTO-Pro 公开枚举有 `pl.LoadDist.BLK`，但缺少该场景的官方 value ST，所以只能在编译产物与 8 个不同哨兵的数值测试通过后启用，不能用 `BRC_B32` 冒充。

## 怎么改（before / after）

A 路径为嵌入片段；B 路径在 value ST 通过前为能力门控伪码。

### A. 同一 VF 内：reduce、倒数、广播、逐元素缩放

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def normalize_row_vf(src_tile, out_tile, scale_tile,
                     valid: pl.DT_INT64, const_scale: pl.DT_FP32,
                     scale_offset: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x = vf.load_align(src_tile, 0)
    row_max = vf.reduce_max(x, preg)              # lane0
    max_all = vf.full(row_max, preg)              # lane0 -> 全有效 lane
    numerator = vf.full(const_scale, preg, dtype=pl.DT_FP32)
    inv_scale_all = vf.div(numerator, max_all, preg)
    out = vf.mul(x, inv_scale_all, preg)
    vf.store_align(out_tile, out, preg)
    vf.store_align(scale_tile, inv_scale_all, preg,
                   scale_offset, dist=pl.StoreDist.FIRST_ELEMENT)
```

该路径把 `reduce → const/max → stride-0 scale → masked 写回` 直接在寄存器里完成；无需先落 UB 再 BRC。`scale_offset` 是当前行在 `scale_tile` 中的元素 offset，调用方必须逐行传入，避免每行覆盖槽位 0。如果跨阶段复用已保存的单个 scale，可用：

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def reuse_one_scale_vf(src_tile, scale_tile, out_tile,
                       valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x = vf.load_align(src_tile, 0)
    scale_all = vf.load_align(scale_tile, 0, dist=pl.LoadDist.BRC_B32)
    out = vf.mul(x, scale_all, preg)
    vf.store_align(out_tile, out, preg)
```

### B. Brcb-8 / stride-0 的 Pro 能力边界

八行方案的完整数据流为：lane0 max 按 8 行成组，`Brcb` 得到 8 个 row max，批量 `Div(constScale, max)`，随后用 stride-0 pattern 对每行向量做 `Mul`，并以 mask 一次写回 8 个 reciprocal scale。各步骤的能力边界如下：

| AscendC / 数据操作 | PyPTO-Pro 方法 |
| --- | --- |
| lane0 `ReduceMax` | `vf.reduce_max`，默认整寄存器归约值位于 lane0 |
| 单行 lane0 广播 | `vf.full`；跨阶段单 B32 用 `pl.LoadDist.BRC_B32` |
| `Brcb` / `DIST_BLK` 的 8 个 FP32 pattern | 候选 `vf.load_align(..., dist=pl.LoadDist.BLK)`；必须先以 8 个互不相同值验证结果 lane pattern，未通过前标 capability gap |
| 批量 `const / max` | `vf.div`；8 值 pattern 的 lane 对应关系必须由上述 ST 固定 |
| stride-0 把 8 个 scale 映射到多行 | 需要当前 load/store datablock stride 能表达目标 packed layout，并用 value ST 验证；不能用单个 `BRC_B32` 替代 |
| masked scale 写回 | 每行 `FIRST_ELEMENT` 是确定退路；8 行一次写回只有显式 store layout/value ST 通过后启用 |

因此，A 是源码与 ST 依据较充分的方法候选，B 是八行方案的能力门控候选；两者在本卡中仍只是表达方法的伪代码，不是可直接复制运行的实现。若 B 的 value ST 尚未落地，就优先采用单 VF 内 `vf.full` 或逐行 `BRC_B32` 这一保守候选，并在当前算子上重新完成编译与数值验证，同时诚实标注尚未获得 8 行合并收益。

### C. 配套位宽、常量与对齐知识

- `int32→half` 中间 Cast 可在 dynamic quant 已证明值域为 `[-127,127]` 或 `[-7,7]` 时降为 `int16→half`；AscendC 外部历史案例曾观察到该改写的 Cast 吞吐可提高一倍，但这只是 `unverified_external_historical` 特定芯片观察值，原始报告与测试条件待补，必须对全部 case 证明值域并在当前 Pro 产物复测，不能从典型输入外推。
- 常量初始化（如 AscendC 的 `DuplicateConst`）移到 Process/VF setup，避免在每行热循环重复构造，并争取与 MTE2 重叠。
- 本方法对应的 AscendC `Brcb`/`WholeReduceMax` 方案将相关 buffer 按 **64B 对齐**；PyPTO-Pro 普通 `vf.load_align`/`store_align` 至少遵循当前文档的 32B 地址对齐。走 BLK/Brcb 等价路径时取更严格的 64B 约束，并由编译/上板验证。
- 多 dtype 路径在 TilingKey/专用 VF 处分流；不要在热循环里运行期判 dtype。

## 性能与验证指标

比较 `Task Duration(us)`、SCALAR/SCALARLDST、V_S/S_V 同步、VF 内 store→load 和每拍有效行数。**待实测**：先验证单行寄存器路径，再单独评估 8 行 block 广播/stride-0 路径。

## 技术限制与风险

- `vf.reduce_*` 只保证 lane0 为归约值；逐元素使用前必须广播。
- `BRC_B32` 是一个 B32 标量广播，不是八值 `Brcb`/`BLK`。
- 跨阶段确需 Tile store→load 时保留官方要求的 `vf.mem_bar`；同 VF 寄存器接力不添加伪 barrier。
- 改变 reduction 顺序、Cast 位宽或广播布局后必须过完整数值回归。

## 参考资料

- **方法场景**：dynamic-quant DB symmetric merge 的 per-row `GetValue/SetValue → Brcb 8 lane + Div + stride0 Mul + masked scale 写回`，以及 int16、常量前移和 64B 对齐。
- PyPTO-Pro：`Vf.full`、`reduce_max`、`div`、`LoadDist.BRC_B32/BLK`；后端 `EmitVFLoadAlign`；官方 lightning-indexer ST 对 `BRC_B32` 有实际用例。
