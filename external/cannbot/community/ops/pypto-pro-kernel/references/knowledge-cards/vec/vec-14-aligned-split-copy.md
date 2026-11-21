---
type: "PyPTO Performance Optimization Card"
title: "对齐分段 Tile 布局"
description: "把紧凑 GM 行的真实子段分别搬入独立对齐 Tile，使 physical padding 与真实 valid extent 分离。"
status: "stable"
tags: ["pypto-pro", "vec", "alignment", "memory"]
item_id: "vec-14"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#6-row-major-tile-alignment"
      scope: "分段后的物理 Tile 基址、行 pitch 与 VF load/store 地址分别满足当前路径的对齐要求。"
      boundary: "GM 的紧凑段距与片上的对齐段距不同；性能取决于额外搬运和当前生成物，不能外推单拍保证。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#fixed-and-dynamic-valid-regions"
      scope: "每个独立段用物理 padding 加真实矩形 valid region 表示。"
      boundary: "一个矩形 valid shape 不能表示中间 hole；所有槽位的 padding 均计入容量。"
bound_hint: "MTE2、MTE3"
applicability: "非 32B 段起点在生成物或 trace 中造成对齐退化，且独立 Tile 及 padding 可被 Vec 容量容纳"
target_api_gate: "仅限 Ascend 950PR 或 950DT；须核验 pl.load/store 元素 offset、Tile physical/valid shape、TileGroup 地址与对齐合同"
---
# 技术卡片 vec-14：对齐分段 Tile 布局

- **适用 bound**：VEC / 访存
- **一句话**：紧凑 GM 行的两个真实段分别搬入独立、32B 对齐的 `[1, aligned_seg]` Vec Tile；physical shape 容纳 padding，valid shape 只覆盖真实元素，VF offset 与 GM offset 分开计算。

## 何时用（诊断特征）

- VF 以不同 offset load 前后半、偶奇段或多个子段。
- `segment_len * sizeof(dtype)` 不是 32B 整数倍，连续 Vec 布局会让第二段起点非对齐。
- 编译/trace 显示后段对齐 load/store 退化，或同长度后段明显更慢。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- 原段已对齐，或新增分段、padding 和搬运的定额成本在短段或多行下更高。
- 单个矩形 valid shape 无法表达中间 hole，却仍把 padding 当作有效元素。
- CopyIn、VF offset、CopyOut 和 tiling 的 aligned_seg 公式不一致，或动态零长度段没有外层跳过。

## 原理

对 FP32 的 `SEG_LEN=60`，一段是 240B；向上对齐到 32B 后是 256B，即 `ALIGNED_SEG=64`。GM 仍可保持两个 60 元素段紧凑相邻；`pl.load` 的 GM offset 用元素坐标 `[0,0]` 与 `[0,60]`，分别装入两个独立 `[1,64]` 物理 Tile 的 offset 0。每个 Tile 的 runtime valid shape 是 `[1,60]`。

AscendC 访存优化的一个微架构假设是：32B 对齐的 RVEC load/store 走对齐形态，跨边界非对齐访问可能被拆成多拍；两次/多次 `DataCopyPad` 的定额成本可能低于 VF 中持续非对齐的代价。对应的 PyPTO-Pro 方法是两次独立 `pl.load` 和两个对齐 group；“单拍/多拍”必须在当前 950 生成物或 trace 中复核，不作为 Python API 本身的保证。

不要用单个 `[1,128]` Tile 加 `[1,120]` valid shape 表示 `[0:60] + hole[60:64] + [64:124]`：valid shape 是矩形前缀，不能表达中间 hole，会把 padding 错当有效数据。

## 怎么改（before / after）

以下是固定两段布局的 PyPTO-Pro 结构示意。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

ALIGN_BYTES = 32
FP32_BYTES = 4
SEG_LEN = 60
ALIGNED_SEG = (
    (SEG_LEN * FP32_BYTES + ALIGN_BYTES - 1) // ALIGN_BYTES
    * ALIGN_BYTES // FP32_BYTES
)

@pl.vector_function
def split_add_vf(front_tile, back_tile, out_tile,
                 valid: pl.DT_INT64):
    preg = vf.update_mask(valid, dtype=pl.DT_FP32)
    front = vf.load_align(front_tile, 0)
    back = vf.load_align(back_tile, 0)
    out = vf.add(front, back, preg)
    vf.store_align(out_tile, out, preg)

@pl.jit(auto_mutex=True)
def split_add_kernel(
    src: pl.Tensor[[1, 120], pl.DT_FP32],
    dst: pl.Tensor[[1, SEG_LEN], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[1, ALIGNED_SEG], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    # 每槽 64 * 4 = 256B；三个 group 的地址互不重叠且 32B 对齐。
    front_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    back_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])
    out_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0400, 0x0500], mutex_ids=[4, 5])

    with pl.section_vector():
        front_tile = front_group.next()
        back_tile = back_group.next()
        out_tile = out_group.next()
        pl.set_validshape(front_tile, [1, SEG_LEN])
        pl.set_validshape(back_tile, [1, SEG_LEN])
        pl.set_validshape(out_tile, [1, SEG_LEN])
        pl.load(front_tile, src, [0, 0])
        pl.load(back_tile, src, [0, SEG_LEN])
        split_add_vf(front_tile, back_tile, out_tile, SEG_LEN)
        pl.store(dst, out_tile, [0, 0])
```

这是固定 shape 的结构较完整**示意代码**：它展示两个 source group 的 physical/valid shape、两次 GM load、VF 与 GM store 的主要数据流，但不保证可直接编译或运行，也不是可原样交付的小核。`pl.load`/`pl.store` offset 是 Tensor 的绝对元素坐标，不是字节地址；TileGroup 的 `addrs` 才是 Vec 内存字节地址。实际算子必须重新规划地址、shape、tail、外层循环和入口，并通过当前工具链与设备验证。

若输入第二段是动态尾，不能让两个段盲目共享一个 `valid`：分别计算 `front_valid`/`back_valid`，给两个 source Tile 各设 valid shape，并在算法允许时使用独立 mask。runtime valid shape 必须为正且不超过 physical shape；零长度应在外层分支跳过相应 load/VF，而不是 `set_validshape(..., [1,0])`。

若目标不是两段相加，而是把两段分别 CopyOut 回紧凑 GM，则分配两个 `[1,64]` 输出 Tile，并分别 `pl.store(...,[0,0])`、`pl.store(...,[0,SEG_LEN])`；仍不要把 hole 编进一个矩形 valid shape。

## 性能与验证指标

比较 `Task Duration(us)`、对齐/非对齐 VF load/store 与新增 Tile 搬运。**待实测**：段长非 32B 对齐且 VF 访存密集时才可能覆盖 padding/多搬运成本。

## 技术限制与风险

- CopyIn、VF offset、CopyOut 与 tiling 的 `aligned_seg` 必须来自同一公式。
- 段本来已对齐时不多拆；短段/多行下额外搬运可能更慢。
- padding 计入所有 TileGroup 槽位的 Vec 内存预算；valid shape 只覆盖真实数据。
- 必须测试最小段、恰好对齐段、非对齐段和动态尾，确认 padding 不参与计算。

## 参考资料

- 数据流：GM 紧凑两段 → Vec 中分别对齐 → 计算/再紧凑写回。
- PyPTO-Pro：`pl.load`/`pl.store` 元素 offset、`TileType` physical/valid shape、`pl.set_validshape`、`make_tile_group` 槽位地址。
