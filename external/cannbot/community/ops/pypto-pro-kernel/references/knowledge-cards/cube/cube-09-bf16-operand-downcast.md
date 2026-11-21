---
type: "PyPTO Performance Optimization Card"
title: "Cube 串行链操作数降精度：fp32→bf16（acc 保持 fp32 累加）"
description: "在精度合同允许时，把 cube 串行链的 L1/L0 操作数 tile 从 fp32 降为 bf16（L0C/acc 端不动），利用 bf16 matmul 更低的单次调用成本；中间结果写回 L1 时必须走 cast 到 bf16 ND + move 到 bf16 NZ 两步。"
status: "stable"
tags: ["pypto-pro", "cube", "matmul", "dtype", "precision"]
item_id: "cube-09"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/numerics.md#preserve-intermediate-precision"
      scope: "分别追踪 L1/L0 操作数 dtype、FP32 Acc 和每轮写回的 BF16 舍入位置。"
      boundary: "FP32 累加不能消除低精度操作数及循环写回误差；本卡只在原合同允许且全量数值回归通过时适用。"
bound_hint: "MAC|MTE1|MTE2|FIXPIPE"
applicability: "kernel 由多次 pl.matmul 串行链主导（迭代求逆、级数展开、多步矩阵乘链），操作数当前为 fp32，当前设备同规模对比测试显示 bf16 单次调用成本显著更低，且精度合同允许操作数按低精度写回、golden 容差门内可复验"
target_api_gate: "Ascend 950PR（A5，DAV_3510）实测；依赖已核验的 pl.TileType(MemorySpace.Mat/Left/Right, layout=NZ)、pl.cast(pl.RoundMode)、pl.move、pl.insert 与 fp32 acc（L0C）语义；其它 SoC/工具链版本必须重新做同规模对比测试，不得沿用本卡数字"
---

# 技术卡片 cube-09：Cube 串行链操作数降精度：fp32→bf16（acc 保持 fp32 累加）

- **适用 bound**：Cube / compute
- **一句话**：串行 matmul 链里把 L1/L0 **操作数**降成 bf16、**累加端**保持 fp32，吃到 bf16 matmul 更低的单次调用成本（本例设备同规模对比测试约 7.5×）；中间结果写回 L1 走「cast 到 bf16 ND → move 到 bf16 NZ」两步。

## 何时用（诊断特征）

以下特征可从当前源码、生成物、对比测试或 profiler 直接核对：

- **源码**：每个工作项执行一串 10 次以上 `pl.matmul`（迭代求逆、级数展开、多步矩阵乘链等），matmul 次数 × 单次调用成本主导 kernel 时长；操作数 tile（L1/L0，`MemorySpace.Mat/Left/Right`）声明为 `DT_FP32`。
- **同规模 dtype 对比测试**：同一 M/N/K、同一最小 matmul 链，只把操作数 dtype 从 fp32 换成 bf16，实测单次调用耗时差。本例设备（Ascend 950PR，单核 128×128×128 链）：fp32 约 5.4µs/次，bf16 约 0.7µs/次（约 7.5×；fp16 与 bf16 相当）。
- **逐段删除对比**（可选）：把链中若干次 matmul 删掉再计时，得到每次调用的真实成本构成。本例 fp32 实现中每次调用约 7.1µs，其中 matmul 计算本体约 4.2µs、其余约 2.8µs 是同步与数据搬运——确认大头在 matmul 本体而非纯同步。
- **精度合同**：操作数每轮以 bf16 写回（新增一次舍入）在容差内可接受，golden 对拍可复验。

## 何时不适用

- 精度门不容许操作数舍入：合同冻结全程 fp32 契约，或值域对 bf16 舍入敏感（大条件数、强消减、中间量动态范围极大）。
- 链很短（一两次 matmul）或瓶颈不在 matmul（纯搬运/同步/vec 主导时降 dtype 无收益）。
- dtype 成本差未在**当前设备与工具链版本**上实测证实——成本差是设备属性，禁止跨 SoC 引用历史数字。
- 中间量有要求 fp32 的外部消费者（落盘/GM 契约），且无法把舍入合法移到消费侧。

## 原理

- matmul 走哪条 dtype 路径由 **L1/L0 操作数 tile 的 dtype** 决定（官方 matmul 文档：lhs/rhs 支持 FP16/BF16/FP32 等，累加器精度通常高于输入）；acc（L0C）端独立，保持 fp32 累加不受影响。因此改动只触碰操作数端，累加精度语义不变，数值变化集中在"每轮写回引入的 bf16 舍入"。
- fp32 操作数路径并非调度浪费：本例实测 fp32 路径下 cube 实际执行量约为语义计算量的 13.7 倍，忙时吞吐已接近该 cube 峰值——即 fp32 路径就是本设备的有效执行形态，性能差距来自 dtype 路径本身。官方 Tensor 级 matmul 文档同样明确："将 BF16 升级到 FP32 再进行 matmul 计算不会有精度提升，反而会产生额外的数据搬移开销"。
- 降 bf16 后计算分量被压缩，但逐段删除对比测出的**其余开销**（同步、acc→Vec 搬运、MTE、insert，本例约 2.8µs/次）不会被本方法消除——串行链若被这部分主导，需配合减少 matmul 调用次数或阶段数的结构优化，不要期待 dtype 一改到底。

## 怎么改（before / after）

以下为**嵌入片段**：所用 `pl.*` API 均在目标 Ascend 950PR 工具链的已验证 kernel 代码中核验存在该形态；片段是教学级最小示意，嵌入 kernel 需自行管理组地址、mutex 与跨核事件，本卡只展示 dtype 与写回结构。例题设定：`BT=K_DIM=128`，`t_cv` 为 fp32 acc→Vec 结果，`t_op16`/`t_op16nz` 为 bf16 ND/NZ 的 Vec 暂存组（同 mutex 分时复用），`l1_a0` 等为 Mat space 操作数。

**before（fp32 操作数 + fp32 直拷写回）：**

```python
# 操作数组：DT_FP32
l1_a0_group = pl.make_tile_group(
    type=pl.TileType(shape=[BT, K_DIM], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
    addrs=L1_A0, mutex_ids=[15])
# 每轮写回：fp32 → fp32，无 dtype 变换
_m2_copy_vf(t_a, t_cv, ro)        # fp32 ND 拷贝（acc→Vec 结果搬到工作 tile）
pl.move(t_bnz, t_a)               # fp32 ND → NZ
pl.insert(l1_a0, t_bnz, [ro, 0])  # NZ → L1（fp32 操作数）
```

**after（bf16 操作数 + 两步写回，acc 端不动）：**

```python
# 操作数组：DT_FP32 → DT_BF16（链内全部操作数组同改）
l1_a0_group = pl.make_tile_group(
    type=pl.TileType(shape=[BT, K_DIM], dtype=pl.DT_BF16,
                     target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
    addrs=L1_A0, mutex_ids=[15])
# 每轮写回：两步——先 cast 到 bf16 ND，再 move 到 bf16 NZ
pl.cast(t_op16, t_cv, mode=pl.RoundMode.CAST_ROUND)  # fp32 ND → bf16 ND（舍入位置）
pl.move(t_op16nz, t_op16)                            # bf16 ND → bf16 NZ
pl.insert(l1_a0, t_op16nz, [ro, 0])                  # NZ → L1（bf16 操作数）
# acc / L0C 保持 fp32 累加，不做任何改动
```

**关键陷阱（两步写回是硬约束）**：`pl.cast` 直接以 bf16 **NZ** tile 为目标（fp32 ND → bf16 NZ 一步完成）会**静默损坏数据**——本例实测表现为求逆结果对角元翻倍（值仍"合法"，只有精度对比才暴露）。必须先 cast 到 bf16 **ND** tile，再 `pl.move` 到 bf16 **NZ** tile。若保留 fp32 原实现作回退基准（改 import 即可切回），可显著降低回归风险。

## 性能与验证指标

**预期变化指标**：

- 同规模 matmul 单次调用耗时：bf16 相对 fp32 的比值（改动前先测，作为收益上限依据）。
- kernel 端到端窗口时长（Task Duration）：预期下降，降幅约等于「matmul 调用次数 × 单次调用成本差」。
- aic_mac（cube 乘加占用）占比：预期下降；下降后若时长未同步下降，说明剩余瓶颈已转移到串行链其它开销，需换下一优化方向。
- 正确性：全量精度回归必须通过——操作数舍入是数值语义变化，不是无损改写。

**验证方法**：

1. 改动前先在当前设备做最小链 dtype 对比测试，确认成本差存在（成本差是设备属性，换设备必须重测）；
2. 修改操作数组 dtype 与写回路径后，先跑全量正确性（对照 golden），再按同一采集协议对比改动前后性能；
3. 把"操作数每轮以 bf16 写回"的新舍入位置记入设计文档，避免后续维护时误判精度来源。

**已有实验证据**（单算子一次实验的材料，未在其它算子/SoC 复现，不能作为当前算子的收益承诺）：在一例 6 轮迭代求逆链 kernel 上，操作数降 bf16 后该 kernel 窗口约 8953µs → 2764µs（约 3.24×），整算子端到端约 10376µs → 3956µs（约 2.62×）；全量 10 个精度用例全部通过（l2_rel 2.18–2.25e-3，门限 3e-3）。

## 技术限制与风险

- **数值语义改变**：每轮写回引入 bf16 舍入（原 fp32 链没有），必须对照 golden 全量复验并更新设计文档中的舍入位置说明，不可只跑单用例；容差贴近门限时留意余量。
- **两步写回**：一步直 cast 到 NZ 的数据损坏是静默的，代码能跑、数值错——落地时把"写回路径全部两步"作为代码检视检查项。
- **设备属性**：dtype 成本差、实际执行量倍数均为本例设备实测；换 SoC/工具链必须重新做对比测试。
- **不要顺手扩 tile**：bf16 使 L1/L0 操作数占用减半，但用释放的容量做双缓冲或扩大 shape 是独立优化项，需独立验证。
- **acc 端保持 fp32**：若进一步降累加精度，是另一个精度决策，不在本卡范围。

## 参考资料

- PyPTO 仓库：matmul API（操作数 dtype、累加器精度约定）：`docs/zh/api/pro_api/SIMD-API/cube_computation/matmul.md`
- PyPTO 仓库：cast 与舍入模式：`docs/zh/api/pro_api/SIMD-API/tile_computation/type_conversion/cast.md`、`docs/zh/api/pro_api/SIMD-API/basic_data_structures/RoundMode.md`
- PyPTO 仓库：move（跨内存层级搬运与 ND/NZ 布局转换）：`docs/zh/api/pro_api/SIMD-API/memory_data_movement/move.md`
- PyPTO 仓库：TileType（dtype/layout 定义）：`docs/zh/api/pro_api/SIMD-API/basic_data_structures/TileType.md`
- PyPTO 仓库：Tensor 级 matmul 注意事项（"避免不必要的 cast：BF16 升级 FP32 无精度提升，反而增加搬移开销"）：`docs/zh/api/tensor_api/operation/pypto-matmul.md`
- PyPTO 仓库：性能优化通用流程（先精度回归、再同协议对比）：`docs/zh/guide/programming_guide/pro/debug/performance_optimization.md`
