---
type: "PyPTO Performance Optimization Card"
title: "VF 尾块统一 mask"
description: "让主体和尾块共用一份 VF 计算体，并在每拍从 total 与 offset 重新生成有效 mask。"
status: "stable"
tags: ["pypto-pro", "vec", "tail", "mask"]
sources: [{"id": "vf-mask-load", "resource": "../../../../pypto-pro-kernel/references/api/vf.md#repeated-runtime-masks-in-vf-loops", "title": "完整寄存器容量与 full-mask 外提"}, {"id": "vf-load-align-api", "resource": "https://pypto.gitcode.com/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_align.html", "title": "vf.load_align 官方 API"}]
item_id: "vec-13"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-prefix-mask"
      scope: "逐拍按 total-offset 重建当前 dtype 的有效前缀，并让 offset 与 mask 使用相同元素粒度。"
      boundary: "update_mask 不回写 Python 计数器；统一尾路径不能覆盖已满足前提的 full-mask 外提义务或更早的越界访问。"
bound_hint: "VEC、SCALAR"
applicability: "full 与 tail 代码体重复，逐拍 active 可由 total 减 offset 精确重算，offset 与 mask 使用同一元素粒度，且可与 fixed full-mask 加独立 tail 的当前控制比较"
target_api_gate: "仅限 Ascend 950PR 或 950DT；须核验 vf.update_mask 不回写 Python 标量，lane 常量按 dtype 与生成物确定"
---
# 技术卡片 vec-13：VF 尾块统一 mask

- **适用 bound**：VEC / Scalar
- **一句话**：主体和尾块共用一份 VF 计算体，每拍由“剩余元素数”调用 `vf.update_mask` 生成 mask，不另抄尾块循环。

## 何时用（诊断特征）

- VF 为 full blocks 与一个/两个 tail 分别复制 load/compute/store。
- 尾块分支多、维护易错或短向量中分支/setup 占比较高。
- 一拍处理一个或多个寄存器宽度，最后一拍可能只有部分有效元素。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 主体 mask 可复用而逐拍重新生成只增加成本；先保留 [当前 mask owner](../../../../pypto-pro-kernel/references/api/vf.md#repeated-runtime-masks-in-vf-loops) 的固定主体控制，按完整调用实测再决定。
- 不能从绝对 offset 精确求出每拍有效元素，或输入/输出寻址需要独立 tail 分支。
- 两 VL 的 part1/part2、逐行 total 或 offset 粒度无法成套重建，零长度 part 仍会访问数据。
- 把 AscendC UpdateMask 的引用递减语义直译成 PyPTO-Pro，或跨 dtype/SoC 硬编码 64 lane。

## 原理

PyPTO-Pro `vf.update_mask(scalar, dtype=...)` 从给定标量值生成当前 mask；它**不沿用 AscendC `UpdateMask(cnt)` 的 C++ 引用递减语义**。因此每次迭代显式计算 `remaining = total - offset`，再用 `pl.min(remaining, LANES)` 生成本拍 mask。主体与尾块走同一代码体。

这个差异须核查到后端层：AscendC `UpdateMask<float>(uint32_t&)` 用 `uint32_t` 引用按 VL 递减，所以同一个计数器可连续消费多拍，且每行/段必须重置。本卡描述的 CCE 实现对**每次** `vf.update_mask(expr)` 先生成新的 `uint32_t` 临时量，再用 `POST_UPDATE`；这个临时量的递减不会写回 Python 源标量，目标版本以公开语义与生成物为准。因此 Pro 代码必须从 `total` 和 offset 重算 active，不能直译 `p1/p2` 可变引用模板。

## 怎么改（before / after）

以下单 VL Python 为嵌入片段。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

LANES_FP32 = 64

@pl.vector_function
def unified_tail_vf(src_tile, out_tile, total: pl.DT_INT64):
    loops = (total + LANES_FP32 - 1) // LANES_FP32
    for i in pl.range(0, loops):
        offset = i * LANES_FP32
        active = pl.min(total - offset, LANES_FP32)
        preg = vf.update_mask(active, dtype=pl.DT_FP32)
        x = vf.load_align(src_tile, offset)
        y = vf.exp(x, preg)
        vf.store_align(out_tile, y, preg, offset)
```

本片段的 FP32 普通 NORM load 要求输入起始地址 32B 对齐，物理容量至少为 `ceil(total/64)*64` 个元素，末组 padding 可安全读取；mask 不裁剪完整 VL 的读取范围。[^vf-load-align-api][^vf-mask-load]

一拍处理两个独立半区时，为每个半区分别计算 `active0 = pl.min(pl.max(total - offset, 0), LANES)`、`active1 = pl.min(pl.max(total - offset - LANES, 0), LANES)`，再分别调用 `vf.update_mask`；不要移植会原地递减的 C++ counter 模板，也不要把数学伪函数 `clamp` 当作 PyPTO-Pro API。

### 两-VL `part1/part2` 预算

RoPE `BatchInterleaveModeVF` 一拍处理两个 VL，用 `loop_size=2*VL`、`d_loop_count=ceil(d_len/loop_size)` 把尾部预算到 `part1_num/part2_num`：前 `d_loop_count-1` 拍两个 part 都满 VL；最后一拍的 `tail_num` 若大于 VL，part1 满、part2 取剩余，否则尾数全归 part1、part2 为空。它用一份 load/compute/store 取代 `tailTwoVL`/`tailOneVL` 两份尾体。

以下两 VL 文本预算为能力门控伪码：

```text
loop_count = ceil(total / (2*VL))
offset(i) = i * 2*VL
active_part1(i) = clamp(total - offset(i),      0, VL)
active_part2(i) = clamp(total - offset(i) - VL, 0, VL)
```

这与上述 `part1/part2` 预算覆盖同一元素集，但每拍直接从绝对 offset 求 active，无需可变 `uint32_t&`。转为真实两-VL VF 时，若 `active_part2==0`，应用外层分支跳过 part2 load/compute/store；不要将零长度当成已验证的 mask 边界。外层再套行循环时，`offset/active_part1/active_part2` 也必须在每行从该行 `total` 重算，不得复用上一行已递减的 `p1/p2`。

## 性能与验证指标

比较 `Task Duration(us)`、代码体重复量与尾块 case。**待实测**：在任务与当前 API 合法的方案之间，与 full-mask 外提＋独立 masked tail 作同口径 A/B，检查 mask/setup 增量是否抵消短向量/尾块的预期收益。[^vf-mask-load]

## 技术限制与风险

- 每拍从 `total` 与 `offset` 重新计算 active；不能假设 `vf.update_mask` 修改 Python 标量。
- 每个行/段重新建立自己的 `total`、offset 和两个 part active；不得复用上一行已消费的 C++ 计数器思路。
- offset 与 mask 必须使用同一元素粒度，最后一拍不得越界或漏尾。
- 两-VL 路径的 part1/part2 预算与 `offset=i*2*VL`/`offset+VL` 必须成套；尾小于等于 1 VL 时 part2 不得访问越界数据。
- `LANES_FP32=64` 只适用于已确认的目标布局；其他 dtype/SoC 由当前 API/编译结果或 tiling 常量给出。
- 全部尾长（0、1、lane-1、lane、lane+1、最大值）必须命中完整正确性测试。

## 参考资料

- 方法参照：RoPE arch35 `BatchInterleaveModeVF` 的两-VL `part1/part2` 预算与每行 `uint32_t` 重置；对照 `InterleaveModeVF` 的 `tailTwoVL/tailOneVL` 重复尾体。
- 核查线索：目标版本的 `Vf.update_mask`、官方 CV fused/softmax 动态有效长度用例，以及 CCE 后端 `EmitVFUpdateMask` 的每次临时 `uint32_t + POST_UPDATE`；用生成物确认“不写回 Python 标量”。

[^vf-mask-load]: [当前完整寄存器容量、mask 与外提比较](../../../../pypto-pro-kernel/references/api/vf.md#repeated-runtime-masks-in-vf-loops)。
[^vf-load-align-api]: [vf.load_align 官方 API](https://pypto.gitcode.com/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_align.html)。
