---
type: "PyPTO Performance Optimization Card"
title: "量化/统计类 epilogue 直接产出消费者最终物理布局，省去中转 repack pass"
description: "当量化/归约类中间结果的内部计算布局（如按有效元素紧排）与下游 GM 契约布局（如按固定分组 pad）不同、且当前用逐行/逐元素方式做二次搬运重排时，改为在计算阶段直接产出目标布局，一次跨轴 store 落位，省去整段中转 repack pass。"
status: "stable"
tags: ["pypto-pro", "vec", "layout", "quantization", "memory"]
item_id: "vec-16"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:gm-plane-offsets"
      scope: "让 epilogue 按消费者最终 GM 轴映射写出前，明确 Tensor 轴、Tile 轴和 store order。"
      boundary: "轴映射契约不保证任意 pad/gap 布局都能一步表达；本卡 N1 对齐限制、补零与写入完整性仍须保持。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/datamove.md#fixed-and-dynamic-valid-regions"
      scope: "删除独立 repack pass 时逐项恢复其原有 valid region、padding 与输出边界责任。"
      boundary: "物理容量不能冒充有效元素；目标布局中有 gap 时，直接写出不自动初始化未写区域。"
bound_hint: "MTE3"
applicability: "存在一个量化/归约类中间结果（如逐块 scale/统计量），其计算天然产出的内部紧凑布局与下游 GM 契约要求的 pad/分组布局不同，且当前实现为此专设了一个逐行或逐元素的二次重排步骤（把内部布局搬到中转缓冲、再搬到目标布局）；该重排步骤的输出布局若可以直接由上一步计算按目标 stride/order 产出，则该整段重排可省略"
target_api_gate: "Ascend 950PR 实测；依赖已核验的 `pl.store`/`pl.load` 的 `order` 参数（完成逻辑轴到物理布局的跨轴映射，含 pad/跳步）；能否把某个具体计算的输出直接摆到目标 stride 由该计算本身的向量化实现决定，不是通用保证，需逐个验证"
---

# 技术卡片 vec-16：量化/统计类 epilogue 直接产出消费者最终物理布局，省去中转 repack pass

- **适用 bound**：访存 / MTE3（输出写出侧）
- **一句话**：量化/归约结果如果最终要以某个"契约布局"（比如按 64 元素一组 pad）写到 GM，而计算本身按"紧凑布局"（比如按 32 元素一组）产出，与其算完紧凑布局再逐行搬到 pad 布局，不如让最后一步计算直接按 pad 布局的目标 stride 写出。

## 何时用（诊断特征）

以下特征可从当前源码、对比实现或 profiler 直接核对：

- 源码里存在一个专门的"重排"/"repack"步骤，形如逐行（`for row in range(m)`）或逐元素地把一个内部计算缓冲的内容搬到另一个布局不同的缓冲，再统一写出；这个步骤不做任何数值计算，纯粹是布局转换。
- 触发原因通常是：计算阶段（如某个归约/量化 kernel）为了自身效率按一种紧凑粒度产出结果（比如"每 32 个元素一个统计量"），但下游消费者（GM 契约、后续算子读取协议）要求另一种粒度或 pad 规则（比如"每 64 个元素一组、每组占 2 个字节位"）。
- 与同一计算逻辑的另一份实现对比（比如自研实现 vs. 参考实现），若参考实现有这个重排 pass 而自研实现没有，可以用 profiler（如 `aiv_vec_time`）直接量出这个 pass 在对比实现 vector 忙时中的占比。

## 何时不适用

- 下游布局本身无法用一次 `pl.store`/`pl.load` 的 `order`/stride 表达（比如目标布局需要跨行交织、条件性 pad 规则依赖运行时才能确定的元素数，且无法归约成单一 stride/order 组合）。
- **计算粒度与存储契约粒度之间存在真实的、无法消除的 gap**：本卡验证场景（N1 必须是 64 的倍数）能一步到位，本质是把接口收窄到让"计算天然产出的紧排宽度"与"存储契约要求的 pad 宽度"数值相等（即 gap 恰好为 0），并非"一步写出"这个手段本身能消解任意 gap。如果目标场景的规模不保证能让两个宽度相等（比如允许 N 取任意值、计算粒度是 32 而存储契约粒度是 64 时 N 不是 64 的倍数），gap 就是真实存在的 padding，仍然需要一次重排/补零，不能强行一步写出；此时要么显式约束输入范围以消除 gap（如本卡做法），要么这条优化不适用。
- 计算阶段本身没有自由选择输出物理布局的空间（比如输出布局被更上游的硬件通路固定死，如某些累加器→UB 的搬运通路只支持一种固定 layout）。
- 重排步骤同时承担除布局转换外的其它工作（比如同时做数值裁剪、二次归约），删除前需要先把这些工作拆分清楚，不能连同数值逻辑一起删。
- 该 pass 已经被证明与其它计算阶段完全重叠（不在关键路径上），此时消除它对端到端时延没有收益，只是减少了整体资源占用——是否值得改动取决于目标是降时延还是降资源占用。

## 原理

- 逐行/逐元素重排步骤的成本主要来自"每次搬运都是一次独立的小颗粒度向量操作"（每行/每组只有几十字节），这类操作的**固定开销**（mask 设置、地址计算、跨行不复用寄存器）远高于其搬运的数据量本身应该花费的时间；当循环次数等于 M（行数）或分组数时，这个固定开销会随 M 线性增长，在大 M 场景下累积成一个不可忽视的 VECTOR 忙时来源。
- `pl.store`/`pl.load` 的 `order` 参数支持把 tile 的逻辑轴按任意顺序映射到目标 tensor 的物理轴，天然支持"内部紧凑排布 → GM 分组 pad 排布"这类跨轴映射，只要目标布局能表达成一个固定的 stride/order 组合。如果计算阶段的最后一步（比如量化 scale 计算的最后一次输出）本身就是按行/按元素产出结果，那么让它直接按目标 stride 写出、而不是先写到一个"计算方便"的中转布局再重排，就能把"重排"这个独立 pass 完全消掉，而不是把它做得更快。
- `pl.cast` 的源/目的 Tile 物理 shape 与行跨度可以不同（按相同逻辑坐标逐元素对应）：当紧排结果与目标布局只差"行距 + dtype 承载宽度"（如 u32 承载的紧排 → 32B 行距的 u8）时，本来就要做的 dtype 收窄 cast 可以顺带完成行距变换，repack 与额外中转缓冲一起消失。
- 这不是"减少每次重排的开销"，而是"让重排这件事本身不需要发生"——区别于常见的对齐/合并优化（那些优化重排本身），这里是通过让上游计算的输出契约与下游消费者的输入契约提前对齐来消除重排的必要性。

### 参考实现为什么必须做这一步（根因，非猜测，已读参考实现源码核实）

参考实现的重排 pass 不是习惯性冗余，而是两个约束叠加的必然结果：

1. **计算粒度与存储契约粒度天生不同且不总相等**：量化数学本身把 scale 粒度定死在"每 32 个元素一个值"（该粒度是归约算法的硬编码常量，与下游存储格式无关），计算阶段天然产出每行 `ceil(N/32)` 个紧排字节；而下游 GM 契约按"每 64 个元素一对、每对占 2 字节"存储，要求每行 `ceil(N/64)*2` 字节。参考实现原文的注释原话是："source contains ceil(N / 32) valid scales, while yScale reserves ceil(N / 64) * 2 slots"——**只有 N 恰好是 64 的倍数时这两个数才相等**；N 不是 64 倍数时，后者严格大于前者，中间是必须补零的真实 gap，不能假装它不存在。
2. **最终写出 GM 用的搬运原语，要求源/目的行宽一致**：参考实现最后一步用的硬件搬运原语，其参数模型是"一个行宽（blockLen）同时约束源和目的，两侧只能在行间距（stride）上不同，行本身的字节数必须相等"。如果直接把计算阶段那个每行 `ceil(N/32)` 字节的紧排结果去搬到每行 `ceil(N/64)*2` 字节的 GM 目标，源、目的行宽对不上，这一条搬运指令的参数模型表达不了"读窄行、写宽行、中间补零"这个语义。于是必须先在中转缓冲里把紧排结果按行重新摆成目的行宽（多余部分补零），才能用这个定长搬运原语一次性写出去。

也就是说，**这一步重排的必要性由"计算粒度与存储粒度是否相等"决定，而不是实现选择**——只要允许 N 取任意值，这个 gap 就是真实存在的，跳不过去。

## 怎么改（before / after）

以 MX 量化 epilogue 的 `y_scale` 输出为例（GM 契约 `[M, ceil(N/64), 2]`，每 32 个元素共享一个 E8M0 字节）。变量名与真实 kernel（`grouped_matmul_activation_quant_impl_pro_1.py`）一致：`mxs` 是 scale 计算的紧排输出 tile（UINT32 承载，每行 `SCALE_PER_ROW` 个元素），`yso` 是 32B 行距的 u8 输出 tile，`m_off/row_start/rows_local/nt_w/valid_n1` 是当前 tile 的调度参数。

**before**（按 AscendC 的三段式结构写 PyPTO-Pro：紧排计算 → 逐行 repack 到 pad 中转缓冲 → 统一写出）：

```python
# ── 额外的两个中转缓冲（对标 AscendC quantScaleOutput_ / 逐行搬运的源窗口）──
mxs_u8 = pl.make_tile_group(                     # 紧排 u8 缓冲：行距 = 每行真实 scale 字节数
    type=pl.TileType(shape=[MT // 2, SCALE_PER_ROW], dtype=pl.DT_UINT8,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
    addrs=0x34000, mutex_ids=[27])
scale_row = pl.make_tile(                        # 单行中转 tile：逐行 repack 的搬运源
    pl.TileType(shape=[1, SCALE_PER_ROW], dtype=pl.DT_UINT8,
                target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]), addr=0x3A000)

# ① 计算阶段：紧排产出 E8M0 字节（u32 承载先收窄成 u8，布局仍是紧排、行间无 pad）
mx_scale_vf(me, mxs, hs, (total_scale + VL_B16 - 1) // VL_B16)
pl.set_validshape(mxs, [rows_local, valid_scale])
pl.set_validshape(mxs_u8, [rows_local, valid_scale])
pl.cast(mxs_u8, mxs, mode=pl.RoundMode.CAST_RINT)

# ② repack pass（对标 AscendC TransScaleLayout → TransScaleVf）：先整块清零
#    32B 行距的 pad 中转缓冲，再逐行把紧排字节搬进去——每行都是一次独立的
#    小颗粒度搬运（源行距 valid_scale 字节、目的行距 32B），固定开销 × 行数
pl.set_validshape(scale_row, [1, valid_scale])
pl.set_validshape(yso, [rows_local, 32])          # yso 为 u8 tile，32 元素 = 32B 行距
pl.expands(yso, 0)                               # 对标 Duplicate(..., 0, mSize * ONE_BLK_SIZE)
for row in pl.range(0, rows_local):
    pl.move(scale_row, mxs_u8, [row, 0])         # 从紧排缓冲取第 row 行（窗口读）
    pl.insert(yso, scale_row, [row, 0])          # 落进 pad 缓冲第 row 行行首

# ③ 行宽已与写出原语对齐，统一写出（对标 CopyScaleToGm 的 DataCopyPad）
pl.store(y_scale, yso, [m_off + row_start, nt_w * 4, 0], order=[0, 2])
```

**after**（当前 pro 实现：`pl.cast` 直接落到目标行距的输出 tile，repack 整段消失）：

```python
# 无 mxs_u8 / scale_row 两个中转缓冲、无整块清零、无逐行循环：pl.cast 的目的
# tile 行距可以与源不同（文档明确"物理 shape 和行跨度可不同，按相同逻辑坐标
# 逐元素对应"），dtype 收窄（u32→u8）与行距变换（紧排 → 32B pad 行距）在
# 同一条 cast 里完成——repack 被融合进本来就要做的那次 cast
mx_scale_vf(me, mxs, hs, (total_scale + VL_B16 - 1) // VL_B16)

valid_scale = (valid_n1 + MX_BLK - 1) // MX_BLK
pl.set_validshape(mxs, [rows_local, valid_scale])
pl.set_validshape(yso, [rows_local, valid_scale])
pl.cast(yso, mxs, mode=pl.RoundMode.CAST_RINT)   # 直落 32B 行距输出 tile

# 一次跨轴 store 直写 GM 契约布局（与 before 的 ③ 完全相同）
pl.store(y_scale, yso, [m_off + row_start, nt_w * 4, 0], order=[0, 2])
```

两种写法最后的 `pl.store` 完全相同，差别集中在中间段：before 需要 2 个额外中转缓冲 + 1 次整块清零 + 行数次 move/insert 小搬运（外加一次只做 dtype 收窄的 cast）；after 把行距变换融合进本来就要执行的 u32→u8 cast，一条指令完成，repack 这件事不再发生。

## 性能与验证指标

**理论性能差异**（两种实现方式的对比，与具体算子/shape 无关）：

- 被消除的 repack pass 的成本结构：逐行小搬运的**固定开销**（mask 设置、地址计算、指令发射、跨行不复用寄存器）× 行数，随 M 线性增长，而每行有效数据量只有几十字节——固定开销远大于数据本身应花的搬运时间；此外还多出一次 pad 缓冲整块清零和 1~2 个中转缓冲的 UB 占用。
- 收益量级：省掉的是一段纯布局搬运，AIV vector 忙时的下降幅度约等于该 pass 在原实现 epilogue vector 忙时中的占比；融合方案不引入新的计算量（行距变换搭的是本来就要执行的 dtype 收窄 cast 的"便车"）。
- 端到端 Task Duration：仅当 vector 侧未被 cube（或其它并行流水）完全遮盖时才会体现——vector 被完全遮盖的 shape 下总时延不变，但 VECTOR 资源占用仍然下降，并在 vector 占比更高的 shape（如更小的 K、更大的 N、更小的单组 M）下转化为端到端收益。
- 资源副作用为正：少 1~2 个 UB 中转缓冲，缓解 epilogue 阶段的 UB 预算压力。

**验证指标**：

- 性能主指标：`aiv_vec_time`（AIV vector 流水忙时）前后对比；用 `aic_mac_time` 与 `aiv_vec_time` 的量级关系判断 vector 是否被 cube 遮盖，决定端到端时延是否预期变化。
- 测量方法：两种实现 A/B 交替执行并中途反转顺序，多轮统计取稳定区间，消除热飘移与执行顺序伪影。
- 正确性：新旧实现的 GM 输出必须逐位/逐字节一致（本质是同一份数据换产出路径，无数值改动）；尾块（valid 列数不足满宽）与 pad 区域清零语义需单独用例覆盖，确认直写路径正确处理了原本由 repack 隐式提供的补零行为。

## 技术限制与风险

- 需要先证明"计算阶段的输出布局可以自由选择"——如果目标布局的 stride/order 组合无法被当前 `pl.store`/`pl.load` 表达（比如需要条件性 pad 长度），本项不适用，不能强行拼凑一个近似布局。
- 本卡已验证的实现（`grouped_matmul_activation_quant_impl_pro_1.py`）能一步写出，前提是显式约束了 `N1` 必须为 64 的倍数（对齐参考实现的存储契约粒度），这让"计算紧排宽度"与"存储契约 pad 宽度"数值相等、gap 恰好为 0；这个约束是消除重排的必要条件，不是这条优化手段自带的能力。迁移到其它算子/shape 时，必须先确认能否接受同等约束（或该场景下 gap 本来就为 0），不能假设"一步写出"对任意 N 都成立——否则会产出未补零的脏数据或越界写。
- 消除重排 pass 后，原本可能由该 pass 隐式提供的对齐/pad 语义（比如把无效 tail 区域清零）需要确认由新的直写路径正确覆盖，尾块场景需单独验证。
- 本卡的量化收益是特定算子、特定 shape 下的对比结果，不能作为其它 kernel 或其它 shape 的收益承诺；是否被 cube/其它流水遮盖需要针对目标 shape 单独判断（比较 `aic_mac_time` 与 `aiv_vec_time` 量级）。

## 参考资料

- PyPTO 仓库：store/load 的 `order` 跨轴映射语义见 `docs/zh/api/pro_api/SIMD-API/memory_data_movement/store.md`、`docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md`。
- PyPTO 仓库：cast 的源/目的行跨度语义（"物理 shape 和行跨度可以不同，按相同逻辑坐标逐元素对应"）见 `docs/zh/api/pro_api/SIMD-API/tile_computation/type_conversion/cast.md`。
- 参考实现的重排 pass 出处（CANN 发行包内公开头文件，非 PyPTO-Pro 文档，仅作为 before 侧结构示意与根因分析的事实依据）：
  - `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/blaze/epilogue/block/block_epilogue_gelu_tanh_mx_quant.h`
    的 `TransScaleLayout` 方法（先 `Duplicate` 整块清零再调用逐行重排；其调用的 tile 层注释原文
    "The source contains ceil(N / 32) valid scales, while yScale reserves ceil(N / 64) * 2 slots"
    显式给出宽度差）与 `CopyScaleToGm` 方法（最终 `DataCopyPad` 调用，源/目的共用同一个
    `blockLen` 行宽）；
  - `.../blaze/epilogue/tile/arch35/mx_quant.h` 的 `TransScaleVf`（逐行 `LoadUnAlign` +
    `StoreAlign NORM_B8` 循环，源行距 `scaleBlockN`、目的行距 `ONE_BLK_SIZE`）。
