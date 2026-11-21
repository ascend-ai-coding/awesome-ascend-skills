---
type: "PyPTO Performance Optimization Card"
title: "不变量广播一次 + 多行 VL merge"
description: "把跨行不变量移出热循环，并仅在 packed 布局、蝶式归约和 tiling 成套验证时合并多行到一个 VL。"
status: "stable"
tags: ["pypto-pro", "vec", "broadcast", "packing"]
item_id: "vec-06"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-outer-reduce.md#metadata-and-invariant-inputs"
      scope: "循环不变的统计量或输入可在复用范围内一次加载，再按消费行广播。"
      boundary: "关联只覆盖不变输入复用；八值 BLK、VL merge、蝶式归约和尾 pack 仍须各自的布局与数值证明。"
bound_hint: "VEC、MTE2"
applicability: "外提时数据须在复用循环内不变；VL merge 时须证明 R、packed 布局及 Tile、tiling、mask、dispatch 一致"
target_api_gate: "仅限 Ascend 950PR 或 950DT；BRC_B32 是保守路径；BLK 须八哨兵 value ST；VL merge 须完整 packed/蝶式布局与尾 mask value ST"
---
# 技术卡片 vec-06：不变量广播一次 + 多行 VL merge

- **适用 bound**：VEC / 访存
- **一句话**：跨行不变量在 VF setup 中只 load 一次；单行远小于寄存器宽度时，把多行紧凑装进一个 VL 并做组内蝶式归约。单标量广播、8 值 block 广播和多行 merge 是三层不同能力，不能互相冒充。

## 何时用（诊断特征）

- VF 热循环反复 load 同一段 mean/rstd/gamma/scale。
- 小 A 或归约维 R 远小于 VL，窄 mask 的 `reduce` 逐行发射，lane 利用率低。
- MTE2 重复 load，或 VECTOR 指令数随行数线性增长，而每条只点亮少量 lane。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 拟外提的数据在复用循环内变化，或所用路径的 lane-to-row、row pitch、valid shape 与 CopyIn/CopyOut 无法保持一致。
- 采用 VL merge 时，R 对应布局、蝶式配对或尾 pack mask 未通过完整 value ST。
- 只实现逐行循环或单 B32 广播，却试图把收益归因于八值 block 或真正的 VL merge。

## 原理

| 层次 | 数据语义 | PyPTO-Pro 表达 | 状态 |
| --- | --- | --- | --- |
| 单 B32 标量广播 | 读一个 FP32，复制到整寄存器 | `pl.LoadDist.BRC_B32` | 有官方 ST/API 依据；本卡表达仍为示意，采用前重验 |
| 一个 32B block 广播 | 读 8 个 FP32，把 8 值 pattern 分发到 VL | 候选 `pl.LoadDist.BLK` | 枚举/后端存在；启用前补 8 哨兵 value ST |
| 多行 VL merge | `merge_num=VL/row_width` 行紧凑打包，一条指令同时算多行 | packed Tile + `active=rows_in_pack*row_width`；组内蝶式 shuffle | 算子专用布局；需完整 value ST/trace |

## 怎么改（before / after）

A 路径为嵌入片段；B、C 路径为能力门控伪码。

### A. 依据较充分的方法候选：单个跨行标量只广播一次

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def rows_shared_scalar_vf(src_tile, gamma_tile, out_tile,
                          row_count: pl.DT_INT64,
                          row_pitch: pl.DT_INT64,
                          valid_per_row: pl.DT_INT64):
    gamma = vf.load_align(gamma_tile, 0, dist=pl.LoadDist.BRC_B32)
    for row in pl.range(0, row_count):
        preg = vf.update_mask(valid_per_row, dtype=pl.DT_FP32)
        x = vf.load_align(src_tile, row * row_pitch)
        y = vf.mul(x, gamma, preg)
        vf.store_align(out_tile, y, preg, row * row_pitch)
```

这段伪代码只用于表达“不变量 load 外提”，不叫 VL merge，也不保证可直接编译或运行；`gamma_tile[0]` 必须真的是所有行共享的标量。若 gamma 是 8 个或每行不同的值，不能采用这一方法。

### B. `DIST_BLK` 八值广播的能力门控

该路径一次载入一个 block（8 个 FP32），并在 R 循环外对 mean/rstd/gamma/dbeta/dgamma 各做一次。PyPTO-Pro 可探索 `vf.load_align(tile, offset, dist=pl.LoadDist.BLK)`，但在实际算子中把它采用为实现候选前必须：

1. 输入 block 填入 8 个互不相同的值；
2. 数值 ST 明确验证 64 lane 中的 8 值重复 pattern，而不只看 dtype/codegen；
3. 编译产物确认是 `vlds(..., BLK)`；
4. 消费者的 lane-to-row 映射与 packed Tile 完全一致。

门控未过时，优先选择逐行 `BRC_B32` 或同 VF 的 `vf.full` 这一保守候选，并在当前算子上验证正确性；同时记录未获得 block 广播收益。

### C. 真正的多行 VL merge 与蝶式组内归约

VL merge 必须按下面的完整布局算法实现，而不是“for row 逐行算”：

```text
前置：row_width=R，R 为 2 的幂，R <= VL；Tile 内多行按 R 连续紧凑排布
merge_num = VL // R
for each pack:
    rows_in_pack = min(merge_num, rows_remaining)
    active = rows_in_pack * R
    load 一整个 packed register，mask=active
    重复 log2(R) 级：de_interleave → add/max → interleave
    得到每个 R-lane 行组自己的 reduce 值，并在组内广播
    一条 masked 算术同时处理 rows_in_pack 行
    store 同一 packed layout
```

蝶式每一级的配对距离、`de_interleave`/`interleave` 输入顺序和输出 lane 映射必须由算子 value ST 验证。若 R 不是 2 的幂，不能套用该蝶式；要么 padding 到有证明的布局，要么另写 mask/尾逻辑。尾 pack 使用 `active=rows_in_pack*R`，不得把 padding 当有效值。

这条路径通常还需同步修改 runtime TilingData/TilingKey、CopyIn/CopyOut 与 dispatch 阈值，让小 A/R case 真正进入新布局；只改 VF 不会自动形成 merge。

## 性能与验证指标

比较 `Task Duration(us)`、重复 load 数、有效 lane、VECTOR 指令数和 MTE/V overlap。**待实测**：分别报告 A、B、C 的增量收益，不能把单标量广播收益记到 VL merge。

## 技术限制与风险

- 只有跨行不变的数据才能外提；逐行变化值不得广播一次。
- `BRC_B32` 只广播一个 B32；八值 block 必须用经过 value ST 的 BLK 路径。
- packed layout、row pitch、valid shape、CopyIn/CopyOut、VF mask 与 dispatch 必须成套修改。
- 蝶式组内归约默认要求 R 为 2 的幂；尾 pack mask 必须是 `rows_in_pack*R`。
- BLK/packed 路径必须遵循其更严格的对齐要求，并由当前文档、编译结果和上板共同确认。

## 参考资料

- **方法场景**：batch-norm-grad 的广播一次、hc-pre 的小行 pack + 蝶式归约，以及 host tiling 联动。
- PyPTO-Pro：`LoadDist.BRC_B32/BLK`、`vf.de_interleave`/`interleave`、`vf.update_mask`；后端 `EmitVFLoadAlign`。
