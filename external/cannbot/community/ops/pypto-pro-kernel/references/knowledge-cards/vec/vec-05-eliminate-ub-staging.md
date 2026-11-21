---
type: "PyPTO Performance Optimization Card"
title: "消除中间 Vec Tile 暂存（寄存器内直算）"
description: "删除仅供紧邻 VF 步骤消费的 Vec Tile store-load 往返、scratch 和不必要的 mem_bar。"
status: "stable"
tags: ["pypto-pro", "vec", "memory", "register"]
item_id: "vec-05"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-row-reduce.md#choose-where-dependent-phases-share-state"
      scope: "中间值仅被紧邻计算消费时，比较同 VF 寄存器接力或同 JIT 的 UB 状态共享，删除临时 Tile 往返。"
      boundary: "仍承担跨循环状态、其它输出或 bank 布局作用的 Tile 不能删除；融合后的寄存器压力须实测。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee"
      scope: "保留下来的同 VF store→load 依赖须按 VF 内存排序契约处理。"
      boundary: "auto_mutex 的 Tile 调用间依赖不能代替 VF 内 barrier；直接寄存器接力不因本关联额外添加 barrier。"
bound_hint: "VEC、MTE2、MTE3"
applicability: "中间 Vec Tile 仅供紧邻 VF 步骤使用且无跨循环、输出或 bank-conflict 角色，生成物存在 store-load"
target_api_gate: "仅限 Ascend 950PR 或 950DT；使用已核验的 Vf.full、reduce、shuffle、mem_bar 与 Cast，布局能力须补 value ST"
---
# 技术卡片 vec-05：消除中间 Vec Tile 暂存（寄存器内直算）

- **适用 bound**：VEC / 访存
- **一句话**：能在向量寄存器里接力的结果不写回 Vec Tile 再读回，省掉 store→load、scratch 与不必要的 `vf.mem_bar`。

## 何时用（诊断特征）

- VF 中出现 `store_align(scratch) → mem_bar(VST_VLD) → load_align(scratch)`，scratch 只给下一步消费。
- 先把原 dtype 转成 FP32 中转 Tile，下一步又 load 回寄存器。
- 错位/重排先在 Vec Tile 上搬一次，再做 VF 算术；中间 Tile 没有跨阶段或跨循环复用。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 中转 Tile 承担跨循环状态、其它输出、跨趟缓存或规避 bank conflict 的角色。
- 改写后仍有 VF store-to-load 依赖却删除其 VST_VLD mem_bar。
- RoPE 两半 offset、sin/cos 布局、interleave 结果和两处 store 不能成套验证。

## 原理

`vf.*` 的结果本就在向量寄存器中。把 cast、reduction、broadcast、shuffle 与后续算术串在同一个 `@pl.vector_function` 中，可以删除 Vec Tile 往返；只有数据确实从 VF store 后被后续 vector load 读取时才需要 `vf.mem_bar(mode=pl.MemBarMode.VST_VLD)`。以下 A/B/C 分别处理预转换、归约和重排暂存。

## 怎么改（before / after）

B、C 路径为嵌入片段；RoPE 布局须按具体算子补齐。

### A. pre-cast staging → 唯一使用点附近转换

- 删除 `xScaleFloatBuf_` 等只做预转换的 FP32 中转 buffer 及初始化。
- 调用侧直接传原 dtype Tile；能用寄存器内 Cast 时在唯一消费者就近转换，但必须遵循 vec-02 的 UNPK/PACK capability/value-ST 门控。
- 当前需要确定正确结果时，用官方 ST 覆盖的 Tile 级 `pl.cast` 批量转换；这不会实现“零 FP32 Tile”，但不会伪造寄存器布局。
- AscendC 可用 C++ dtype 重载取代热循环中的 `if constexpr` 分支；在 Pro 中对应为 TilingKey/专用 VF 分流，不要把 C++ 重载或命名空间细节伪装成 Python API，也不要在热循环里堆运行期 dtype 分支。所有输入 dtype 组合都必须分别命中精度回归。

### B. 行归约结果不落 scratch

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def reduce_and_consume_vf(src_tile, out_tile, valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x = vf.load_align(src_tile, 0)
    row_max = vf.reduce_max(x, preg)
    max_all = vf.full(row_max, preg)
    out = vf.sub(x, max_all, preg)
    vf.store_align(out_tile, out, preg)
```

before 是 `reduce → store FIRST_ELEMENT → LocalMemBar/VST_VLD → BRC_B32 load`；after 用 `vf.full` 将 lane0 就地广播，删去 scratch、一次 store/load 和 barrier。外部历史案例记录漏掉该 LocalMemBar 时曾出现 `maxAbsErr=inf`（`unverified_external_historical`，原始报告与测试条件待补），所以若归约值需跨趟缓存或寄存器放不下，仍保留 Tile 往返，并在真实 VST→VLD 依赖间保留 `vf.mem_bar`。

### C. RoPE rotate-half：offset load 后完整写回两半

下面是语义闭合的逐 lane RoPE rotate-half 片段。输入 Tile 含前后两个各 64 个 FP32 的对齐段，输出也含两个段；`valid` 是每半区的有效长度。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def rope_rotate_half_vf(x_tile, cos_tile, sin_tile, out_tile,
                        valid: pl.DT_INT64):
    half = 64
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    x_front = vf.load_align(x_tile, 0)
    x_back = vf.load_align(x_tile, half)
    cos_front = vf.load_align(cos_tile, 0)
    cos_back = vf.load_align(cos_tile, half)
    sin_front = vf.load_align(sin_tile, 0)
    sin_back = vf.load_align(sin_tile, half)

    front_cos = vf.mul(x_front, cos_front, preg)
    front_sin = vf.mul(x_back, sin_front, preg)
    out_front = vf.sub(front_cos, front_sin, preg)

    back_sin = vf.mul(x_front, sin_back, preg)
    back_cos = vf.mul(x_back, cos_back, preg)
    out_back = vf.add(back_sin, back_cos, preg)
    vf.store_align(out_tile, out_front, preg, 0)
    vf.store_align(out_tile, out_back, preg, half)
```

该路径计算 `out_front=cos*x_front-sin*x_back`、`out_back=sin*x_front+cos*x_back`，两半都必须完整写回。它消除了使用 Tile 重排的实现为 rotate-half 做的 UB→UB 前后半搬移、整块 `Muls(-1)` 预处理和中间 LocalTensor。

交错 RoPE 也可采用另一种布局路径：用 `vf.de_interleave` 拆偶奇，给需要翻号的一半乘 `-1`，再 `vf.interleave` 重组；两路返回寄存器都必须继续参加公式或分别写回。其具体 sin/cos 排布依算子而异，因此不把不完整 shuffle 骨架标成可直接运行的 RoPE 实现。

## 性能与验证指标

比较 `Task Duration(us)`、VF store/load、barrier、Vec 内存预算和 spill。**待实测**：往返越频繁，删除后收益越可能明显。

## 技术限制与风险

- 中转 Tile 在跨循环状态、其他输出或防 bank conflict 中仍被使用时不得删除。
- 本卡默认整寄存器归约的值位于 lane0，逐元素使用前调用 `vf.full`。
- 真正的 VF store→load 依赖必须保留 `vf.mem_bar`；Tile 层 MTE/V 依赖由 mutex/`auto_mutex` 管理，二者不可互相冒充。
- 寄存器内融合会提高活跃寄存器数；出现 spill 时配合 vec-12。
- RoPE 的两半 offset、sin/cos 布局与两处 store 必须成套验证，不能只验证前半。

## 参考资料

- **方法场景**：qli pre-cast staging（A）、hc-pre reduce+duplicate（B）、RoPE rotate-half/interleave（C）。
- PyPTO-Pro：`Vf.full`、`reduce_max`、`de_interleave`、`interleave`、`mem_bar`；vec-02 的 Cast 能力门控。
