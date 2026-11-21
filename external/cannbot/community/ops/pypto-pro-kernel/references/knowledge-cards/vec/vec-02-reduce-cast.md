---
type: "PyPTO Performance Optimization Card"
title: "减少、批量或就近执行 Cast"
description: "在精度合同允许时删除往返 Cast，否则批量转换，并把寄存器内 UNPK/PACK 路径置于能力和数值 ST 双重门控下。"
status: "stable"
tags: ["pypto-pro", "vec", "cast", "dtype"]
item_id: "vec-02"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-fp16-widening"
      scope: "必须用 FP32 计算时，核对 FP16 UNPK 搬入、数值 Cast、lane 顺序与计算 mask，避免不必要的 FP32 中转 Tile。"
      boundary: "该契约的 FP16 路径不替代本卡 BF16、整数或特定工具链的能力门控；删除 Cast 还须满足原精度和舍入合同。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:vf-fp32-narrowing"
      scope: "收窄后按物理寄存器容器、PACK 方式与有效元素数设计输出。"
      boundary: "端口压缩不执行数值转换；不继承其它 dtype 的舍入、饱和或当前性能结论。"
bound_hint: "VEC"
applicability: "存在冗余 Cast、可批量或就近转换的机会，或不必要的转换中转 Tile，且 dtype、rounding、布局与值域允许相应改写"
target_api_gate: "仅限 Ascend 950PR 或 950DT；Tile 级 pl.cast 需当前 ST 支持，UNPK/PACK 须有明确分布模式与端到端 value ST"
---
# 技术卡片 vec-02：减少、批量或就近执行 Cast

- **适用 bound**：VEC
- **一句话**：能保持同 dtype 就删除往返 Cast；必须用 FP32 计算时，先采用已由 PyPTO-Pro ST 覆盖的 Tile 级批量 Cast，再把“UNPK + 寄存器计算 + PACK”作为需要后端能力与数值 ST 双重通过的进一步优化。

## 何时用（诊断特征）

- trace 中类型转换指令占比较高，计算主体只有少数几条指令却夹着一进一出两次 Cast。
- 代码为转换单独分配中间 Vec Tile，并出现不必要的 store→load 往返。
- 多个逐块转换可合并为一次更大粒度的转换，或可在唯一使用点附近完成。

历史经验中，Cast 指令占比 **超过 20%** 可作介入提示，部分算子曾达到 **30%–50%**；这些数值仅为 `unverified_external_historical` 定位启发式，原始报告与测试条件待补，不是跨算子性能合同，仍以当前 manifest、芯片和 trace 实测为准。

## 何时不适用

以下是不适用情形或需要权衡的风险；经验阈值仅作诊断参考：

- SPEC 要求的中间精度、rounding 或输出 dtype 不允许删除或改写 Cast。
- 采用 after C 时，目标 UNPK/PACK 分布、B16/B32 组合、mask 粒度或 dense 顺序尚未被 value ST 和编译产物共同证明。
- 涉及整数窄化但缺少覆盖全部输入 case 的值域证明，或以历史阈值放宽当前项目精度门。

## 原理

优化按能力边界分三档，不能把三档混成一个未经验证的 VF 示例：

1. **after A：删 Cast。** 精度允许全链路 FP16/BF16 时，直接用该 dtype 计算。
2. **after B：批量 Cast。** 累加、softmax 分母等仍需 FP32 时，在唯一使用点前后各做一次 Tile 级 `pl.cast`；它会占用 FP32 Tile，但语义明确，官方 ST 有 FP16→FP32 与 FP32→FP16 用例。
3. **after C：寄存器内就近 Cast。** AscendC 对应技术用 `DIST_UNPACK_B16` 搬入、在寄存器内升到 FP32，计算后降为 B16，再用 `DIST_PACK_B32` 搬出，从而不落 FP32 中转 UB。PyPTO-Pro 的同一映射尚缺数值 ST 证明，见下方能力门控；采用前须核验目标版本。

## 怎么改（before / after）

after B 为结构示意，after C 为能力门控伪码。

### 语义明确的 PyPTO-Pro 方法示意（after B）

```python
import pypto_pro.language as pl

TILE_N = 64

@pl.jit(auto_mutex=True)
def cast_compute_kernel(
    src: pl.Tensor[[1, TILE_N], pl.DT_FP16],
    bias: pl.Tensor[[1, TILE_N], pl.DT_FP32],
    dst: pl.Tensor[[1, TILE_N], pl.DT_FP16],
):
    f16_type = pl.TileType(
        shape=[1, TILE_N], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
    )
    f32_type = pl.TileType(
        shape=[1, TILE_N], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
    )
    src_f16_group = pl.make_tile_group(
        type=f16_type, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    src_f32_group = pl.make_tile_group(
        type=f32_type, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])
    bias_f32_group = pl.make_tile_group(
        type=f32_type, addrs=[0x0400, 0x0500], mutex_ids=[4, 5])
    out_f32_group = pl.make_tile_group(
        type=f32_type, addrs=[0x0600, 0x0700], mutex_ids=[6, 7])
    out_f16_group = pl.make_tile_group(
        type=f16_type, addrs=[0x0800, 0x0900], mutex_ids=[8, 9])

    with pl.section_vector():
        src_f16 = src_f16_group.next()
        src_f32 = src_f32_group.next()
        bias_f32 = bias_f32_group.next()
        out_f32 = out_f32_group.next()
        out_f16 = out_f16_group.next()
        pl.set_validshape(src_f16, [1, TILE_N])
        pl.set_validshape(src_f32, [1, TILE_N])
        pl.set_validshape(bias_f32, [1, TILE_N])
        pl.set_validshape(out_f32, [1, TILE_N])
        pl.set_validshape(out_f16, [1, TILE_N])
        pl.load(src_f16, src, [0, 0])
        pl.load(bias_f32, bias, [0, 0])
        pl.cast(src_f32, src_f16, mode=pl.RoundMode.CAST_ROUND)
        pl.add(out_f32, src_f32, bias_f32)
        pl.cast(out_f16, out_f32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(dst, out_f16, [0, 0])
```

该伪代码按 64 个连续 FP16 元素表达 dense 转换方法，不依赖寄存器偶/奇半区布局；它不保证作为独立 kernel 可直接编译或运行。五个 Tile 显式使用相同的正 runtime valid shape，物理 shape 仍为 `[1,64]`。官方 ST `test_quant_lightning_indexer_vf.py` 覆盖 FP16→FP32 `pl.cast`，`test_cast_dedup_double_buffer.py` 以数值 oracle 覆盖 FP32→FP16 `pl.cast`；这些证据只支持 API/方法选择，不替代本算子的完整编译与数值验证。动态尾版本同样要同步设置五个 Tile；示意固定 `TILE_N=64`，因此没有展示尾块。

### after C：能力与 value-ST 门控

after C 的目标是以下 AscendC 指令序列的等价数据流：

```text
B16 dense UB --DIST_UNPACK_B16--> B16 register --Cast--> FP32 register
FP32 compute
FP32 register --Cast--> B16 register --DIST_PACK_B32--> B16 dense UB
```

以下能力和证据缺口使 after C 仍是能力门控候选，不能据此提供可直接采用的等价 Python 代码：

- `pl.LoadDist.UNPK`/`UNPK_B16` 与 `pl.StoreDist.PACK` 枚举存在，但现有 `test_vf_basic_ops.py` 的 kernel 20/21 与 26/29 对 UNPK/PACK 只检查 codegen/dtype，没有 B16 dense 数值 oracle；同文件 kernel 24 的普通 `vf.astype` FP32→FP16→FP32 则有数值 oracle，但它不证明 UNPK/PACK 分布。
- 后端按寄存器 dtype 将泛化 `StoreDist.PACK` 选择为 `PK_B16` 或 `PK_B32`；after C 要求“降精后的 B16 寄存器 + `PACK_B32`”，公开枚举没有可显式指定的 `StoreDist.PACK_B32`。不能仅凭 AST 通过便声称生成了目标指令。
- 普通 `vf.astype` 也不是 dense 一一转换：官方 `astype.md` 的 FP16 示例明确展示 widening 只消费偶位置，回转后奇位置未定义。mask 还必须按官方规则与源操作数粒度一致。

因此，只有同时满足以下条件，after C 才可从方法伪代码升级为经当前算子验证的实现候选：

1. 当前后端/API 能明确表达 B16 UNPK 与 B32 PACK，编译产物确认分布模式；
2. 新增端到端数值 ST，至少覆盖 `valid=1/63/64`，并用偶、奇位置不同哨兵证明 dense 顺序无丢失；
3. 对 widening/narrowing 分别验证 mask dtype、`CastLayout.ZERO/ONE` 与尾 mask；当前官方接口说明以源操作数粒度解释 mask，转换方向改变时不得盲目复用同一个 preg；
4. FP16/BF16 两条路径分别过精度回归。

若后续选择“显式拆偶/奇半区”而非 UNPK/PACK，也必须用两个 FP32 半区、独立的 `even_count=(valid+1)//2`/`odd_count=valid//2` mask，在降精后按已验证布局合并；补齐 widening 奇半区的数值 ST 前，只能作为设计候选。

## 性能与验证指标

比较 `Task Duration(us)`、转换指令占比与 Vec Tile 流量。**待实测**：after B 先完成当前算子的编译与数值验证；after C 只有通过上述门控并证明省掉中转 Tile/往返后才计入收益。

## 技术限制与风险

- 涉及删除 Cast 或改变 rounding/layout 的改动必须通过全量精度回归。
- 跨 dtype 的历史参考阈值为 fp16 `2^-10`、bf16 `2^-7`、fp32 `2^-13`；它们必须先与当前算子 `SPEC.md` 及精度与测试合同核对，项目标准更严时从严，不得拿历史阈值放宽已有门槛。
- FP16/BF16 reduction、softmax 分母等通常仍需 FP32 累加。
- 降整数位宽（如 int32→int16）前证明所有输入 case 值域安全。
- 不把 `LoadDist.UNPK`/`StoreDist.PACK` 的存在误当作目标 B16/B32 组合已经数值验证。

## 参考资料

- **方法场景**：RoPE arch35 的 B16↔B32 Cast 融进 load/store、避免 FP32 中转 UB，对应 after C。
- PyPTO-Pro：`language/_api.py::cast`；官方 `test_quant_lightning_indexer_vf.py`、`test_cast_dedup_double_buffer.py`；`astype.md`；后端 `backend_cce_vf_ops.cpp` 的 `EmitVFLoadAlign`/`EmitVFStoreAlign`/`EmitVFCast`。
