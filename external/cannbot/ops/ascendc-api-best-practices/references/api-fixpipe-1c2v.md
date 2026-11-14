# 1C:2V 核模型与 Fixpipe 三条通路（DAV_3510）

> **平台**：Ascend 950PR/950DT（`__NPU_ARCH__ == 3510`）；部分接口在 5102 上同构，未单独验证。
> **依据**：本机 CANN `cann-9.2.0-beta.2` 的公开头文件、随包官方算子实现，以及 Ascend950PR 真机实测（附录 A）。**未能坐实的内容不写入本文。**
> **分工**：参数跨代差异（V220 ↔ Arch3510 字段对照）见 [api-cross-gen-fixpipe.md](./api-cross-gen-fixpipe.md)；跨核 flag 的通用规则见 [api-crosscore-sync.md](./api-crosscore-sync.md)；容量/核比例表在 `ops/npu-arch/references/npu-hardware-params.md`。本篇只讲「1C:2V 核模型 ↔ Fixpipe 三条通路 ↔ 双 AIV 分发与同步」这一层。

---

## 目录

1. [核模型：1 AIC + 2 AIV](#1-核模型1-aic--2-aiv)
2. [Fixpipe 三条通路与接口分层](#2-fixpipe-三条通路与接口分层)
3. [参数、单位与对齐约束](#3-参数单位与对齐约束)
4. [双 AIV 分发：四种策略](#4-双-aiv-分发四种策略)
5. [1C:2V 同步协议](#5-1c2v-同步协议)
6. [断言与校验清单](#6-断言与校验清单)
7. [最小样例](#7-最小样例)
8. [易错点](#8-易错点)
9. [与相邻文档的分工](#9-与相邻文档的分工)

---

## 1. 核模型：1 AIC + 2 AIV

DAV_3510 一个 AI Core 组 = **1 个 AIC（Cube）+ 2 个 AIV（Vector）**，Cube:Vector = 1:2，每个 AIV 有**独立 UB**。软件侧可用值为 `TOTAL_UB_SIZE = 248 * 1024`；"每 AIV 物理 256 KB / 每组 512 KB"出自 `ops/npu-arch/references/npu-hardware-params.md`（白皮书**表 4-2**，"按组计"为推论，本机无法独立验证）。
>
> 注意：1:2 是默认混精模式；代码里还留有 **MIX 1:1** 分支（注释 "handle MIX 1:1 specifically ... loose coupled mode in david"，由 `__MIX_CORE_AIC_RATION__ == 1` 开关）。

本文只关心**软件可见的核身份**：

| 维度 | AIC | AIV |
|---|---|---|
| `GetSubBlockNum` | `1` | `get_subblockdim` = **2** |
| `GetTaskRation`（官方接口名，返回 `int64_t`） | `1` | `get_subblockdim` = **2** |
| `GetSubBlockIdx` | `get_subblockid`；CPU-debug 实现固定返回 0，真机由硬件返回、头文件未承诺恒 0 | `get_subblockid` ∈ **{0, 1}** |
| `GetBlockIdx` | `get_block_idx` | `get_block_idx * get_subblockdim + get_subblockid`（`subblockdim = 2`） |
| 可访问 UB | **不直接读写**，但可**写** AIV 的 UB，有两条通路：Fixpipe（§2）与 `DataCopyL1ToUB<T, subBlockId>`（`subBlockId` 是**非类型模板参数** `uint8_t subBlockId = 0`，用于选 lane；只在 `__NPU_ARCH__ == 3510` 下生效，有 `count` 与 `DataCopyParams` 两个重载） | 是，lane 各自一份 |
| L1 / L0C | 各 512 KB / 256 KB，组内一份，AIC 侧持有 | **L0C 无通路**；L1 只能经 vector 侧 `CopyUB2L1` **写**（在 AIC 上 return）；反向 L1→UB 由 AIC 侧 `CopyL12UB` 发起（在 AIV 上 return） |

工程要点：

- **AIV 的 `GetBlockIdx` 是拉平的**：框架按 AIC 数量下发 block，AIV 侧看到的 blockIdx 是 `AIC_block * 2 + lane`。Tiling 里凡是"AIV 核数"的地方都要按 `2 × AIC 核数` 推导，且不能把 AIV blockIdx 直接当 AIC blockIdx 用。
- **两个 AIV 的 UB 相互独立**，地址空间互不可见。AIC 把数据放进某个 AIV 的 UB 有两条通路：① Fixpipe L0C→UB（§2，`subBlockId` 选 lane，§4）；② `DataCopyL1ToUB`（L1→UB，同样带 `subBlockId` 模板参数）。两者都不是"随便写"，lane 选择语义一致。
- 组内配对固定：AIC *i* ↔ AIV *2i*、*2i+1*。跨组同步不只有硬栅栏：除了 `SyncAll`（硬件）之外，还有基于 workspace 的软同步 `SyncAll(gm, ub, usedCores)`、`IBSet/IBWait`，以及 3510 的 `GroupBarrier`。组内（AIC ↔ 自己的 2 个 AIV）则用 intra-block flag，见 §5。

### 1.1 950PR 与 950DT 的一致性（已核验）

核验方法：本机 `cann-9.2.0-beta.2/x86_64-linux/data/platform_config/` 下 **35 个 950 SKU**（PR 9 个 / DT 26 个）按 `[section]::key` 二元组逐键对比（共 394 个键），**只有 16 个键取值不同**（其中 `SoC_version` 就是 SKU 名本身，实质差异 15 项；另外 `l2_size`、`ai_cpu_cnt`、`vir_type_list` 也不同，属 SKU 级）。

| 类别 | 是否一致 | 取值 |
|---|---|---|
| 架构代码 | ✅ | 全 SKU `NpuArch=3510`、`AIC_version=AIC-C-310`、`CCEC_AIC/AIV_version=dav-c310-*`、`Chip_type=15`、`io_die_num=2` |
| Cube/Vector 内部资源 | ✅ | 与代码常量一致：`TOTAL_L1_SIZE=512*1024`、`TOTAL_L0C_SIZE=256*1024`；INI 侧 `l1_size=524288`、`l0_c_size=262144`、`l0_a/l0_b=65536`、`fb0_size=4096`、`fb1_size=2048`、`bt_size=4096`、`vec_calc_size=128`、`vector_reg_width=256`、`wide_reg_width=768`、`predicate_reg_width=32` |
| **AIV 的 UB 容量** | ✅ | `[AICoreSpec] ub_size` = `[VectorCoreSpec] ub_size` = **253952 B（248 KB）**，35 个 SKU 全部相同；`ubblock_size=32`、`ubbank_num=16`、`ubbank_size=4096`、`ubbank_group_num=8`、`ubsubbank_num=64` |
| 内建 dtype 支持 | ✅ 跨 SKU | 两个 map **跨 35 个 SKU** 逐项相同（CANN 侧未按 PR/DT 区分）；但两表本身不同（AICore 表项更少，VectorCore 表项更多），不要互相当作等价 |
| 核数 | ❌ SKU 级 | AIC ∈ {4, 8, 28, 32, 36}，AIV = 2 × AIC ∈ {8, 16, 56, 64, 72}；与 PR/DT 之分无关 |
| 频率 | ❌ SKU 级 | `cube_freq` / `vec_freq` ∈ {1650, 1800}，**PR 与 DT 各有 1650 和 1800 档** |
| 内存子系统 | ❌ **SKU 级**（非族级） | `ddr_rate`（AIC 侧）：PR 4–31 / DT 14–80；按核档看也不是定值——PR@28 核 ∈ {23,27,31}、PR@32 核 ∈ {27,31}；DT@28/32 核 ∈ {54,61,70,80}、DT@36 核 ∈ {61,80}。只有每族最高端单 SKU 分别取 31 / 80。`memory_size`：旗舰档 PR 128 GiB / DT 144 GiB |

结论：

1. **计算微架构在 950PR 与 950DT 之间相同**（同一 `NpuArch=3510`、`AIC-C-310`、`dav-c310-*`，L1/L0C/FB/BT/UB 容量与位宽全族一致）——本文所有 1C:2V / Fixpipe 结论对两者通用，不需要按 SKU 分支。
2. 族间差异集中在**内存子系统**（`ddr_rate`、`memory_size`），此外 `ai_cpu_cnt`（5/6/7）与 `vir_type_list`（3 种）也随 SKU 变化。
3. **不要用"是否 DT"推断核数、频率或 L2**——这些是 SKU 级参数，同族内部就有多档。
4. 产品定位（PR 偏 Prefill + 推荐、DT 偏 Decode + 训练）与 "128→144 GB / 1.6→4 TB/s" 属**厂商发布口径**（外部材料，本机无法验证），且与 INI 不完全对应：DT 的 `memory_size` 里并没有 128 GiB 档（128 GiB 只在 PR 出现），`ddr_rate` 是无量纲值。引用时标明来源。

---

## 2. Fixpipe 三条通路与接口分层

### 2.1 dispatch 规则（同一份模板，三种目的）

3510 上同名 `Fixpipe` 有**两套重载**，分派机制完全不同，选错参数类型就换了实现路径：

- **`FixpipeParamsArch3510<config.format>` 版**（整体被 `__NPU_ARCH__ == 3510 || 5102` 包住）：分派是**编译期**的，只看 dst 类型 + `config.isToUB`；
- **`FixpipeParamsV220` 版**：分派是**运行期**的，按 `GetPhyType(dst.GetPosition)` 决定走 GM / L1 / UB。

下表只描述 Arch3510 的重载（本文其余部分默认用这套）：

| dst 类型 | `config.isToUB` | 实际实现 | 落地指令 |
|---|---|---|---|
| `GlobalTensor<T>` | 忽略 | `FixpipeL0C2GMImpl` | `copy_matrix_cc_to_gm` |
| `LocalTensor<T>` | `false` | `FixpipeL0C2L1Impl` | `copy_matrix_cc_to_cbuf`（写 L1） |
| `LocalTensor<T>` | `true` | `FixpipeL0C2UBImpl` | `copy_matrix_cc_to_ub`（写 UB） |

分派点：dst 是 `LocalTensor` 时由 `if constexpr (config.isToUB)` 决定走 UB 还是 L1；dst 是 `GlobalTensor` 时走单独的 GM 重载。

> **两层名字不要搞混**：上表列的是**分发壳**（`FixpipeL0C2*Impl`，内含 `if ASCEND_IS_AIC`）；
> 真正干活的是叶子函数 `FixpipeL0cToL1`、`FixpipeL0cToUB`、
> **`FixpipeL0cToOut`**（由 `FixpipeL0C2GMImpl` 调用）。
> 源码里**没有** `FixpipeL0cToGM` 这个函数——按字面搜索会一无所获。

三条通路**都只在 `if ASCEND_IS_AIC` 内执行**——**AIV 侧调用 `Fixpipe` 是静默空操作**，不会报错。

> ⚠️ 没有任何一层会替你兜住：`ASCENDC_CHECK_TPOSITION` 只校验"dst 位于 L1 或 UB"、**不校验与 `isToUB` 是否一致**，而且它在设备编译下是空宏（§6）。把 UB 的 `LocalTensor` 传给 `isToUB = false` 的 config，真机不报错、会走 L1 通路把 UB 地址当 L1 地址用。
>
> 另外这个检查**只在 LocalTensor 路径**（其中 `CheckTensorPos` 本身也是被 `ASCENDC_CPU_DEBUG` 包裹的函数）；GlobalTensor 路径**没有**位置校验。

### 2.2 `FixpipeConfig`：3510 才有 `isToUB`

```cpp
// 公开头里的定义（节选）
struct FixpipeConfig {
 CO2Layout format;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
 bool isToUB; // ← 仅 3510/5102 存在，其余架构该成员不存在
#endif
 ...
};
```

- 头文件自带的四个常量只有 `CFG_NZ` / `CFG_ROW_MAJOR` / `CFG_COLUMN_MAJOR`（+ 5102 的 `CFG_NZ_FIX`），**没有"UB 版"公开常量**。
- 随包内部头文件有 `AscendC::Impl::CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true}`、`Impl::CFG_NZ_UB`，但属 `Impl` 内部命名空间，不建议业务代码依赖。
- **AIV 侧调用是空操作**：三条通路的函数体全部包在 `if ASCEND_IS_AIC` 内，AIV 编译时整段被消除。曾有人用平台 INI 的 `[VectorCoreintrinsicDtypeMap]` 里登记了 `Intrinsic_fix_pipe_l0c2out`来反推"AIV 也能发 Fixpipe"——**不成立**：同一项在 A2/A3 的平台配置里同样同时出现在 AICore 与 VectorCore 两张表中，而 A2/A3 上 Fixpipe 明确只能由 AIC 发起，说明那只是一张能力登记表。

- **正确做法：自己定义命名空间作用域 `constexpr`**（`Fixpipe` 的模板形参是 `const FixpipeConfig&`；官方 `CFG_*` 本身就是命名空间作用域 `constexpr`，**内部链接即可**，不需要 `extern`）：

```cpp
// 放在命名空间作用域，供 Fixpipe 作为 non-type template 实参
constexpr AscendC::FixpipeConfig kCfgNdToUb = {AscendC::CO2Layout::ROW_MAJOR, true}; // L0C → UB（NZ2ND）
constexpr AscendC::FixpipeConfig kCfgNzToUb = {AscendC::CO2Layout::NZ, true}; // L0C → UB（保持 NZ）
constexpr AscendC::FixpipeConfig kCfgDnToUb = {AscendC::CO2Layout::COLUMN_MAJOR, true};// L0C → UB（NZ2DN）
```

> ⚠️ `FixpipeConfig(CO2Layout, bool)` 这个双参构造在**所有架构都无条件存在**，但在非 3510/5102 上 `isToUB` 被丢弃（初始化列表里没有该成员）。所以同一份代码在 2201 上能编过，却**静默退化成 L0C→L1**——跨代移植时必须显式隔离。

### 2.3 参数类型必须跟随 `config.format`

3510 的四个重载形参类型都是 `FixpipeParamsArch3510<config.format>`：

| 接口 | 说明 |
|---|---|
| `Fixpipe(LocalTensor dst, LocalTensor src, FixpipeParamsArch3510<fmt> p)` | L0C→L1 / L0C→UB（由 `isToUB` 选） |
| `Fixpipe(LocalTensor dst, LocalTensor src, LocalTensor<uint64_t> deqTensor, ...)` | 同上 + 向量量化（deq/relu 表经 L1→FIXBUF） |
| `Fixpipe(GlobalTensor dst, LocalTensor src, FixpipeParamsArch3510<fmt> p)` | L0C→GM |
| `Fixpipe(GlobalTensor dst, LocalTensor src, LocalTensor<uint64_t> deqTensor, ...)` | L0C→GM + 向量量化 |

`fmt` 必须是 `config.format`：`config = {ROW_MAJOR, true}` ↔ `FixpipeParamsArch3510<CO2Layout::ROW_MAJOR>`。`FixpipeParamsC310<fmt>` 是 `FixpipeParamsArch3510<fmt>` 的**空派生别名**，二者完全等价。

### 2.4 接口分层速查

| 层 | 形态 | 双 AIV 相关入口 |
|---|---|---|
| basic_api | `Fixpipe<T, U, CFG>(dst, src, FixpipeParamsArch3510<fmt>)` | `p.dualDstCtl` + `p.subBlockId` |
| tensor_api | `Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}).with(CopyL0C2UBTrait{...}), dstUb, srcL0c, Te::FixpipeParams{unitFlag, subBlockId})` | **`dualDstCtl` 在 Trait 里**（`CopyL0C2UBTrait::dualDstCtl`），**`subBlockId` 在 `FixpipeParams` 里** —— 两者分属不同结构体，实现侧最终拼到一起 |
| matmul 高阶 | `MatmulImpl::SetSubBlockIdx` + `CopyCubeOut` 自动填参 | 高阶侧 `subBlockId = GetSubBlockIdx` |

---

## 3. 参数、单位与对齐约束

字段定义见公开头 `kernel_struct_fixpipe.h`。与 `FixpipeParamsV220` 的跨代差异见 [api-cross-gen-fixpipe.md](./api-cross-gen-fixpipe.md)，此处只补单位与 1C:2V 相关项。

### 3.1 单位

| 字段 | 单位 | 取值公式 |
|---|---|---|
| `nSize` | 元素 | N 方向实际宽度；NZ2ND 时 = 输出每行元素数 |
| `mSize` | 元素 | M 方向实际行数 |
| `srcStride` | **1 个 BLOCK_CUBE = 16 元素（fp32 = 64 B = 2 个 32 B C0）** | L0C 源相邻 Z 块跨度 = `Align(baseM, 16)`，**按物理 L0C 的 M 算，不是按本次拷出的 `mSize`** |
| `dstStride` | ROW_MAJOR → **元素/行**；NZ → **32 B 块**（`DEFAULT_C0_SIZE / sizeof(DstT)` 折算）；COLUMN_MAJOR → **列 pitch（元素）** | ROW_MAJOR 填目标行间距：紧凑 UB 输出时**必须等于 `nSize`**（含对齐后的值），写进更宽矩阵时填实际行宽；NZ 填 Z 块跨度；COLUMN_MAJOR 填 DN 输出每列跨的行数 |
| `unitFlag` | — | `0` / `2`（`NON_FINAL_ACCUMULATION`）/ `3`（`FINAL_ACCUMULATION`，**均为官方算子代码里的私有常量**，见 §8 #13） |
| `dualDstCtl` | — | `0b00` / `0b01`(SPLIT_M) / `0b10`(SPLIT_N) |
| `subBlockId` | **`bool`** | `false` → AIV0 的 UB，`true` → AIV1 的 UB |

**`srcStride` 不是 `M/16`**。官方两处配方取的都是"按 16 对齐后的行数"：

```cpp
// 官方 cmct（L0C→UB，非 quant 主路）
fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE); // 单位 CO_SIZE
// 官方 matmul 通用
params_.srcStride = CeilAlign(mSize, BLOCK_CUBE);
```

> 历史误会：把"填 mSize 对 16 向上取整"读成 `ceil(M/16)` 会少 16 倍。**M=64 → `srcStride = 64`，不是 4。**
>
> **`srcStride` 描述的是 L0C 的物理几何，与本次拷出多少行无关**。官方 `DoubleCopyOutTwoFixpipe` 里两条指令的 `mSize` 从 `halfBaseM` 改成 `baseM - halfBaseM`，而 `srcStride` 始终是 `Align(baseM, 16)` 保持不动。所以只拷部分行（`mSize < baseM`）时，`srcStride` 仍按完整的 `baseM` 填。
>
> 算错的典型现象：**每行前 16 列是对的，第 16 列之后整块错位**——那就是 `srcStride` 少填了。
>
> 单位口径还有一个坑：实现里固定按 `BLOCK_CUBE = 16` 相乘，但旁边注释写的是 "in unit of C0_size"；而 MX/scale 的 matmul 路径用 dtype 相关的 `Align(mSize, c0Size)`。对 fp32 而言 16 元素 = 64 B ≠ 1 个 C0（32 B），两条配方在 M 不是 16 的倍数时可能给出不同值（如 M=100：`CeilAlign(100,16)=112` vs `Align(100,8)=104`）——**按 16 对齐最保守，非常规 M 需实测**。注意**本表的 `C0` 一律指 32 B**：`srcStride` 的单位是 `BLOCK_CUBE = 16 元素`，与 C0 不是一个概念，源码注释里的 "C0_size" 是笔误，别按 32 B 折算。

### 3.2 对齐与取值约束（`CheckFixpipeParams`）

| 条件 | 约束 |
|---|---|
| 类型组合 | src 只能是 `float` / `int32_t`；dst 与 src 必须命中官方白名单组合（如 `float→half/bf16/fp8/hifloat8/int8/uint8/float`、`int32→half/bf16/int8/uint8/int32`） |
| `isChannelSplit = true` | `nSize ∈ [1, 4095]` 且 `nSize % 8 == 0` |
| `format = ROW_MAJOR`（NZ2ND） | `nSize ∈ [1, 4095]`；**非 GM 目的**时 `nSize * sizeof(DstT) % 32 == 0`（GM 目的放宽）。→ 写 UB 时 `nSize` 必须是 `Align(N, 32/sizeof(DstT))`（fp32 按 8、fp16 按 16 对齐），且 **UB 缓冲要按 `mSize × nSize`（对齐后的）申请**，不是 `M × N` |
| `format = COLUMN_MAJOR`（NZ2DN） | `mSize ∈ [1, 32767]`；非 GM 目的时 `mSize * sizeof(DstT) % 32 == 0` |
| `format = NZ` | `nSize ∈ [1, 4095]` 且 `nSize % 16 == 0` |
| 通用 | `mSize ∈ [1, 65535]`；`dstStride != 0` |
| 多 ND（`ROW_MAJOR`） | 仅当 `params.ndNum > 1` 时校验：`srcNdStride ∈ [0, 65535]`、`dstNdStride ∈ [1, 2^32-1]` |
| 多 DN（`COLUMN_MAJOR`） | 仅当 `params.dnNum > 1` 时校验：`dstDnMatrixStride ∈ [1, 2^32-1]` |
| 地址 | 源 L0C 起始按 16×sizeof(float)=64 B 对齐；目的 UB/L1 起始按 32 B 对齐（CPU-debug 断言） |
| L0C→UB 专项 | **`isChannelSplit` 必须为 `false`**（`FixpipeL0cToUB` 内 `ASCENDC_DEBUG_ASSERT`） |

> ⚠️ 上表的**运行时检查全部只在 CPU-debug 构建生效**，真机上一条都不会报错（宏展开见 §6）。它们的作用是把硬件契约写成可读的报错，不是兜底。

### 3.3 64×64 FP32 算例

| 字段 | 取值 | 理由 |
|---|---|---|
| `format` / `isToUB` | `ROW_MAJOR` / `true` | NZ → ND 输出到 UB |
| `nSize` | `64` | 输出每行 64 个 float；64×4 B = 256 B % 32 == 0 ✔ |
| `mSize` | `64` | 64 行 |
| `srcStride` | **`64`** | `CeilAlign(64, 16)`，单位 16 元素 |
| `dstStride` | `64` | 每行 64 个元素 |
| `dualDstCtl` | `0`（显式写） | 单目标 |
| `subBlockId` | `false` | 只写 AIV0 |

UB 占用：64×64×4 B = **16 KB / lane**，远低于 248 KB。

---

## 4. 双 AIV 分发：四种策略

`dualDstCtl` 的取值约束来自 `FixpipeL0cToUB` 的三条 `ASCENDC_ASSERT`（文案原文、标识符已简化）；**`subBlockId` 没有任何断言**，它的语义只能从官方调用的行内注释（`// aiv0` / `// aiv1`）反推：

```cpp
ASCENDC_ASSERT(!(dualDstCtl != 0b00 && (format == COLUMN_MAJOR || quantPre != NoQuant || reluEn)),
 "Dual destination mode can be enabled only when normal DMA or NZ2ND is enabled with any other Fixpipe functions bypassed.");
ASCENDC_ASSERT(!(dualDstCtl == 0b01 && mSize % 2 != 0),
 "dual destination mode, split in M dimension, M/2 * N is written to each UB, M must be number of 2.");
ASCENDC_ASSERT(!(dualDstCtl == 0b10 && calNSize % 32 != 0),
 "dual destination mode, split in N dimension, M * N/2 is written to each UB, N must be number of 32.");
```

> ⚠️ 这三条 **只在 CPU-debug 构建生效**，设备编译会展开成空（见 §6）。"约束存在" ≠ "真机会拦"。

| 策略 | `dualDstCtl` | `subBlockId` | 每个 AIV 收到什么 | 附加约束 | 适用 |
|---|---|---|---|---|---|
| **A. 单目标 stripe** | `0` | 每条指令显式 0/1 | 被指定的那一个 lane 独占该 tile | 无（可同时带 quant/relu） | AIC 交替喂两个 AIV。官方用例：量化类算子里用 `subBlockId = (subBlockId_++) & 0x1` 交替喂两个 lane |
| **B. 两条指令对分** | `0` | 第 1 条 `false`、第 2 条 `true` | lane0 拿 `CeilDiv(M,2)` 行、lane1 拿 `M − CeilDiv(M,2)` 行（**M 为奇数时不等分**）；源偏移 `CeilDiv(M,2) * 16` 元素 | 自行拆 `mSize` 与源偏移；官方在 `M == 1` 时跳过第二条指令 | **需要随路 quant 时的 1C:2V** |
| **C. SplitM 单指令** | `0b01` | 不用（官方 `DoubleCopyOutDualSplitM` 里确实没有 `subBlockId` 赋值） | 各拿 **M/2 × N** | `mSize` 偶数（官方取 `Align(baseM, SPLIT_M_ALIGN=2)`）；`NoQuant`；`reluEn=false`；非 `COLUMN_MAJOR` | **C 输出为 float（无随路 quant）时的 1C:2V 主路**；half/bf16 输出走策略 B |
| **D. SplitN 单指令** | `0b10` | 可留默认；**官方实现仍传 `subBlockId = id`** | 各拿 **M × N/2** | `nSize % 32 == 0`；`NoQuant`；`reluEn=false`；非 `COLUMN_MAJOR`；**NZ 输出不支持**；高阶 matmul 侧还要求 A/B 同源（ibShare） | N 方向对半切 |

补充说明：

- 表内 `N` 指 `nSize`（官方会先 `Align(baseN, C0)`）；断言实际检查 `calNSize`，直通路径下 `calNSize == nSize`。
- SplitN 官方实现还改了 `dstStride`：`params.dstStride = stride >> 1`，与 SplitM 的用法不同。
- 官方两处对 SplitN 的 N 对齐口径不一致：`FixpipeL0cToUB` 断言 `%32`，而高阶 matmul 的 `SplitMNParamsCheck` 文案写 `N % 16 == 0` 且条件疑似取反。以 `%32` 为准。
- "C 是主路"有前提：官方 `DoubleCopyOut` 按 C 类型分流，`float → DUAL_DST_SPLIT_M`，否则走两条 Fixpipe；配套的 `enable2UB = IsSameType<CType,float>`。

官方写法：

```cpp
// 策略 B：两条 Fixpipe —— 注释直接写明 lane
AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
uint64_t halfBaseM = Cmct::Gemm::CeilDiv(baseM, SPLIT_M_ALIGN);
fixpipeParams.dualDstCtl = 0;
fixpipeParams.unitFlag = 0;
fixpipeParams.subBlockId = 0; // aiv0
Fixpipe<C_T, L0cType, CFG_ROW_MAJOR_UB>(dstLocal, c1Local, fixpipeParams);
if (baseM == 1) { return; }
// 注意：官方是在“已偏移过的” c1Local 上又叠加了一次 l0cOffset，自写时需实测确认语义
AscendC::LocalTensor<L0cType> c1LocalNext = c1Local_[l0cOffset + halfBaseM * BLOCK_CUBE];
fixpipeParams.mSize = baseM - halfBaseM;
fixpipeParams.subBlockId = 1; // aiv1
Fixpipe<C_T, L0cType, CFG_ROW_MAJOR_UB>(dstLocal, c1LocalNext, fixpipeParams);
```

```cpp
// 策略 C：一条 SplitM（赋值后立即下发；官方注释写明 "cvRatio=1:2 默认 splitM"）
fixpipeParams.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M);
Fixpipe<C_T, L0cType, CFG_ROW_MAJOR_UB>(dstLocal, c1Local, fixpipeParams);
```

### 4.1 `subBlockId` 到底是什么

- **`bool` 类型，取值只有两态**，因为组内 AIV 只有 2 个。语义是"目标 lane 选择位"：`false → AIV0`、`true → AIV1`——依据是官方注释，**不是断言**。
- 经 `uint8_t` 接口传入会被静默收窄：`SetSubBlockId(uint8_t id) { params_.subBlockId = id; }`，`2` 与 `1` 等价。
- **是"目标选择"、不是"自身身份"**：官方由单个 AIC 连续下发 `subBlockId=0` 与 `1` 两条指令（策略 B），这是代码级证据；**"单个 AIC 能跨 lane 写"已真机实测证实**（附录 A Q1）：`subBlockId=false` 时 lane1 的 UB 逐字节仍是预置的毒化值，`true` 时反之——严格互补，说明是真路由而非写两份。
- 高阶 matmul 的 `SetSubBlockId(...)` 取的是 `MatmulSubBlockInfo::GetSubBlockIdx`，该值由用户 `Matmul::SetSubBlockIdx` 传入、**缺省 0**，**不是**硬件核身份——所以"默认写自己这一 lane"不成立，默认写的是 lane0。
- `dualDstCtl != 0` 时官方代码仍会赋值 `subBlockId`，说明 API 层二者不互斥；**实测：该模式下硬件完全忽略 `subBlockId`**（附录 A Q3，四种组合逐字节相同），目的端只由 `dualDstCtl` 决定。

### 4.2 双目标下的目的地址（**已实测：必须按 `>>1` 预算**）

官方在 L0C→UB（且 A/B 同源 ibShare）通路上对目的偏移统一做了 `>>1`：

```cpp
// ND 输出：只对行项折半
int64_t dstOffset = ((curRow * baseM * stride) >> 1) + curCol * baseN;
// NZ 输出：整体折半
dstOffset = dstOffset >> 1; // splitM or splitN
// SplitN 还会同时改 dstStride
params.dstStride = stride >> 1; // splitN
```

机制的两种候选解释（"两 lane 合并地址空间" vs "每 lane 自成地址空间、只装 M/2 行"）**已由真机实验判定为后者**（附录 A Q2）：

- `dualDstCtl=1`、`mSize=64` 时，每个 lane 只收到 **32 行**（对比单目标时的 64 行，确认真的 SplitM 了）；
- 把第二块 tile 分别放到 `off=1024`（折半）与 `off=2048`（原值），footprint 立刻分叉：

| AIC 传的地址 | lane0/lane1 footprint（`row-runs`） |
|---|---|
| `off = 1024`（`>>1`） | 2048 cells，`[(0,31)]` —— **稠密无洞** |
| `off = 2048`（不折半） | 2048 cells，`[(0,15),(32,47)]` —— 16–31 行出现空洞 |

> ⚠️ **陷阱**：只有一个 tile 且 `curRow == 0` 时，折半与原值都是 offset 0，两种假设**不可区分**——必须用第二个 M tile 才能证伪。只做单 tile 实验会得出"随便填都对"的错误结论。

---

## 5. 1C:2V 同步协议

Fixpipe 把数据写进某个 AIV 的 UB 之后，必须让该 AIV 知道"数据到了"。1C:2V 用的是 **intra-block（组内）同步**，不是 cross-block。

### 5.1 两种 flag 空间

| modeId | Set 侧落地 | Wait 侧落地 | flagId 空间 | 可见范围 |
|---|---|---|---|---|
| 除 4 之外的任意值 | `ffts_cross_core_sync(mode, GetffstMsg(mode, flagId))` | `wait_flag_dev(pipe, flagId)`（不经 `GetffstMsg`） | Set 侧 `flagId & 0xf`（**仅低 4 bit**）；Wait 侧原样下发 | 跨 block（组间/核间） |
| `4`（`INTRA_MODE`） | `set_intra_block` | `wait_intra_block` | 归纳为 **16 id × 2 lane**（`INTRA_NUM=16`、`SYNC_FLAG_ID_MAX=16`，最大观测 id = 15+16 = 31）；**无常量或断言声明上界** | 仅本组 AIC ↔ 其 2 个 AIV（无文档背书） |

> mode 本身只有 2 bit（`GetffstMsg` 里 `mode & 0x3`），所以 5/6/7 会被折叠成 1/2/3；判断分支只有「等于 4」与「其它」。

依据：`NotifyEventImpl` 与 `WaitEventImpl` 的 `modeId == 4` 分支、以及 `GetffstMsg` 的 mode 掩码。

### 5.2 lane 偏移公式（本文最需要补的知识点）

```cpp
__aicore__ inline uint8_t GetIntraFlagId(uint8_t instID, uint8_t eventType, uint8_t subBlockId)
{
 return (instID + eventType + subBlockId * INTRA_NUM); // INTRA_NUM = 16
}
```

即 **lane k 的 flagId = (instID + eventType) + k × 16**。注意 `instID` 与 `eventType` 是**相加**共用一个 4 bit 空间，两者之和超过 15 会与别的组合别名（如 `(1,2)` 与 `(2,1)` 都得到 3），设计时要保证 `instID + eventType ≤ 15`。官方 KFC 握手正是这么写的：

```cpp
CrossCoreSetFlag<KFC_INTRA_MODE, PIPE_S>(KFC_SYNC_ID);
CrossCoreSetFlag<KFC_INTRA_MODE, PIPE_S>(KFC_SYNC_ID + 16);
```

官方 Matmul 高阶 API 里也有"两条 Set + lane 号 = 0/1"的写法。注意这两行位于 `#if ASCENDC_CPU_DEBUG` 且 `enableMixDualMaster` 分支内，**只能作为语义佐证，不能当设备侧实现**：

```cpp
CrossCoreSetFlag<INTRA_MODE, PIPE_FIX>(GetIntraFlagId(instID, VEC_WAIT_FIXP, 0U));
if constexpr (A_TYPE::ibShare && B_TYPE::ibShare) {
 CrossCoreSetFlag<INTRA_MODE, PIPE_FIX>(GetIntraFlagId(instID, VEC_WAIT_FIXP, 1U)); // 1 means sub_block 1
}
```

AIV 接收侧等待在**本核要用的 pipe** 上：

```cpp
if constexpr (GetPhyType(C_TYPE::pos) == Hardware::UB) {
 CrossCoreWaitFlag<INTRA_MODE, PIPE_V>(waitFixpId); // 等完直接做 Vector 计算
} else {
 CrossCoreWaitFlag<INTRA_MODE, PIPE_MTE2>(waitFixpId); // 等完直接搬 GM
}
```

> **“谁需要写 +16”的准确规律（已按官方全部调用点核对）**：**AIC 侧收发都显式写两槽 `base` 与 `base + 16`；AIV 侧收发只用 `base`**（lane 通道由硬件隐式区分）。AIV 侧写 +16 与官方所有写法都相反，会导致等待错槽。
>
> 另外，客户端里还有两条**不同机制**的通路，不要和 intra flag 混为一谈：
> - `USE_SSBUF` 路径：`waitFixpId = WAIT_FIXP + instIdx`——**不含 lane 偏移**（AIV 侧）；
> - 非 SSBUF 路径：`devEvtID = instIdx * 2 + GetSubBlockIdxImpl`，走 `WaitEvent(devEvtID)`，那是 **device event id（mode 0 的空间）**，与 intra flagId 不是同一套编号；且当 `ibShare && lane==1` 时 `intraId = devEvtID - 1`。
>
> AIC 侧显式写两槽的例子：KFC 握手里 `KFC_SYNC_ID` 与 `KFC_SYNC_ID + 16` 各发一次；AIC 侧既 wait 也 set 两个槽。

### 5.3 pipe 选择决定"这段代码在哪个核上真正执行"

`modeId == 4` 的实现用 pipe 类别做编译期分派：

| pipe | vector 侧（AIV 执行） | cube 侧（AIC 执行） |
|---|---|---|
| `PIPE_S` | ✔ | ✔ |
| `PIPE_MTE2` | ✔ | ✔ |
| `PIPE_V` / `PIPE_MTE3` | ✔ | ✘ |
| `PIPE_MTE1` / `PIPE_FIX` / `PIPE_M` | ✘ | ✔ |

（3510 的 cube pipe 集合为 `PIPE_S/PIPE_MTE1/PIPE_MTE2/PIPE_FIX/PIPE_M`，**`PIPE_MTE3` 不在其中**。上表已用 `ccec --cce-aicore-arch=dav-c310-{cube,vec}` 对 `__NPU_ARCH__==3510` 做过 `static_assert` 实测，全部通过。）

推论：`CrossCoreSetFlag<4, PIPE_FIX>` 写在 AIV 分支里是**静默空操作**；`CrossCoreSetFlag<4, PIPE_V>` 写在 AIC 分支里同样空操作。因为 `g_coreType` 是 `constexpr`，这类错误分支会被**编译期直接消除**，运行时不会有任何提示。

> 官方实践：**AIC 发送侧用 `PIPE_FIX`（Fixpipe 完成点），AIV 接收侧用 `PIPE_V`（UB 输出）或 `PIPE_MTE2`（GM 输出）**。`PIPE_S` 虽然在两侧都有效，但不是官方 L0C→UB 握手的取值。

### 5.4 推荐握手模板

```cpp
// AIC 侧：Fixpipe 完成后，显式发两个槽（base → lane0，base+16 → lane1）
if ASCEND_IS_AIC {
 AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p; // 3510 同时存在 V220 重载，类型要写全
 // ... 填 p（见 §3.3）
 Fixpipe<float, float, kCfgNdToUb>(ubTensor, l0cTensor, p); // subBlockId 决定写哪个 lane
 CrossCoreSetFlag<4, PIPE_FIX>(kFixFlagId); // → AIV0
 CrossCoreSetFlag<4, PIPE_FIX>(kFixFlagId + 16); // → AIV1
}

// AIV 侧：**只等 base**（lane 通道由硬件隐式区分），等在本核要用的 pipe 上
if ASCEND_IS_AIV {
 CrossCoreWaitFlag<4, PIPE_V>(kFixFlagId); // 不要写 + lane*16，与官方全部写法相反
 // ... 读本 lane 的 UB 做 Vector 计算
}
```

约束与注意：

- **`+16` 只在 AIC 侧写**：AIC 收发都要显式写 `base` / `base+16` 两槽；AIV 侧收发一律只用 `base`。
- **类型要写全**：3510 同时存在 `FixpipeParamsV220`与 `FixpipeParamsArch3510<config.format>`两套重载，二者都能编过但落到不同实现（`p` 只写 `auto`/不写类型容易误选）。
- **flagId 要有界常量**：禁止用 tile index 推导（会被截断并撞进保留区，见 [api-crosscore-sync.md](./api-crosscore-sync.md) §3）。
- **避开占用区**：库内已占 `11-14`（`SYNC_AIC_FLAG=11` / `SYNC_AIV_FLAG=12` / `SYNC_AIC_AIV_FLAG=13` / `SYNC_AIV_ONLY_ALL=14`）；**`15` 与它的镜像 `31` 只在用到 Matmul/KFC 时被占**（`KFC_SYNC_ID=15`），kernel 内无 mode 4 使用方时 `15` 仍可用——与 [api-crosscore-sync.md](./api-crosscore-sync.md) §3 的 `[8,9,10,15]` 同源；用 `AscendC::Matmul`/KFC 时 `0-3` 也被枚举占用。保守可用集是 `[8,10]`（无 Matmul/KFC 时可加 `15`）。
- **反向依赖**（AIV 读完、UB 可被覆盖）同样需要一条 AIV→AIC 的 flag；按上面的规律，AIV 侧仍只用 `base`，AIC 侧要等两槽。
- **实测补充（重要）**：只靠 `CrossCoreSetFlag` / `CrossCoreWaitFlag` **不足以保证 UB 数据对 AIV 可见**——同一个 binary 连跑两次，一次读到完整数据、一次读到空（附录 A）。实测加上双向 `SyncAll<false>` 会合后才稳定。flag 与 MTE/FIX pipe 之间的可见性边界官方没有文字说明。
- **"Set/Wait 必须严格配对"是硬件语义**，代码里没有任何断言或校验兜底（与 §6 一致）；配对错误在真机上只会表现为卡死或读到旧数据。

---

## 6. 断言与校验清单

| 检查 | 生效条件 |
|---|---|
| dtype 白名单 | **编译期，无条件** |
| 双目标模式功能组合、SplitM 偶 M、SplitN `%32` | **仅 CPU-debug**（`raise(SIGABRT)`）；设备侧展开为空 |
| `isChannelSplit` 必须 false（L0C→UB） | CPU-debug 生效；设备侧**仅当编译时定义 `ASCENDC_DEBUG` 且未定义 `ASCENDC_DUMP=0`** 才生效 |
| nSize/mSize 范围与 `%16`/`%8`/`%32`、`dstStride != 0`、源地址对齐 | **仅 CPU-debug**（整块被 `#if ASCENDC_CPU_DEBUG == 1` 包住） |
| 目的地址 32 B 对齐 | **仅 CPU-debug**（不在 `CheckFixpipeParams` 内） |
| dst 必须在 L1/UB（`ASCENDC_CHECK_TPOSITION`） | **仅 CPU-debug**；设备侧空宏 |

> **结论（重要）：默认编译选项下，真机上没有任何运行时参数校验兜底。** 唯一自动生效的是编译期 dtype 白名单；`ASCENDC_ASSERT` / `ASCENDC_CHECK_VALUE_RANGE` / `ASCENDC_CHECK_TPOSITION` / `KERNEL_LOG` 在设备编译下全部展开为空。因此参数填错（nSize 不满足 32B 对齐、`dstStride = 0`、双目标带 quant、SplitM 奇数 M…）**不会报错，只会静默算错，或越界写坏 UB 内相邻数据（UB 无边界保护）**。
>
> 唯一的例外是 `-DASCENDC_DEBUG`：此时**只有** `isChannelSplit` 这一条 `ASCENDC_DEBUG_ASSERT` 会在设备侧真正 trap，其余检查仍然不存在。
>
> 想让设备侧真的拦：编译时定义 `ASCENDC_DEBUG`（且不要 `ASCENDC_DUMP=0`），`ASCENDC_DEBUG_ASSERT` 系列会变成设备侧 assert（`printf` + `trap`）。但 `ASCENDC_ASSERT` 仍为空——**双目标那三条约束无论如何都不会在真机生效**。用户也可自行用 `ascendc_assert`（默认开启、依赖 `enable_asc_assert` 的元数据），但它不会自动校验 API 入参。

---

### 6.1 断言消息速查（CPU-debug 构建下按报错定位）

| 报错关键字 | 触发的条件 |
|---|---|
| `isChannelSplit must be set as false` | L0C→L1 或 L0C→UB 时 `isChannelSplit == true` |
| `Dual destination mode can be enabled only when…` | `dualDstCtl != 0` 同时用了 COLUMN_MAJOR / 量化 / relu |
| `… split in M dimension … M must be number of 2` | `dualDstCtl == 0b01` 且 `mSize` 为奇数 |
| `… split in N dimension … N must be number of 32` | `dualDstCtl == 0b10` 且 `calNSize % 32 != 0` |
| `… isChannelSplit is true, its valid range is 1 ~ 4095 and must be divisible by 8` | `isChannelSplit` 下 `nSize` 越界或未按 8 对齐 |
| `… format is NZ2ND, its valid range is 1 ~ 4095` | ROW_MAJOR 下 `nSize` 越界 |
| `… when NZ2ND, its valid value * sizeof(DstT) must be divisible by 32B` | ROW_MAJOR 且非 GM，`nSize * sizeof(DstT) % 32 != 0` |
| `… format is NZ2DN, its valid range is 1 ~ 32767` | COLUMN_MAJOR 下 `mSize` 越界 |
| `… when NZ2DN, its valid value * sizeof(DstT) must be divisible by 32B` | COLUMN_MAJOR 且非 GM，对齐不满足 |
| `… format is NZ, its valid range is 1 ~ 4095 and must be divisible by 16` | NZ 下 `nSize` 未按 `BLOCK_CUBE` 对齐或越界 |
| `Failed to check dstStride value in Fixpipe` | `dstStride == 0` |
| `Failed to check src start address alignment in Fixpipe` | L0C 源地址未按 64 B（`16 * sizeof(float)`）对齐 |
| `Failed to check dst start address alignment in Fixpipe` | L1/UB 目标地址不满足 `dstAbsAddr * sizeof(DstT) % ONE_BLK_SIZE(32) == 0` |
| `Failed to check dtype in Fixpipe` | `DstT`/`SrcT` 不在编译期类型白名单内 |
| `Failed to check quantPre value in Fixpipe` | `quantPre` 与 dtype 组合不匹配 |

> 再提醒一次：**这些报错只在 CPU-debug 构建下出现**，真机上一条都不会打印（§6 正文）。

## 7. 最小样例

### 7.1 单目标（只喂 AIV0）

```cpp
constexpr AscendC::FixpipeConfig kCfgNdToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

__aicore__ inline void Copy64x64L0cToUb(const AscendC::LocalTensor<float>& ub,
 const AscendC::LocalTensor<float>& l0c)
{
 if ASCEND_IS_AIV { return; } // Fixpipe 只在 AIC 侧执行，AIV 调用是空操作（§2.1）
 AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p{};
 p.nSize = 64;
 p.mSize = 64;
 p.srcStride = 64; // = CeilAlign(mSize,16)，M=64 就写 64（不是 M/16 = 4，见 §3.1）
 p.dstStride = 64; // ND 每行元素数
 p.dualDstCtl = 0; // 显式声明单目标（不依赖默认值）
 p.subBlockId = false; // bool：false → AIV0
 p.unitFlag = 3; // FINAL_ACCUMULATION；它不是公开常量（§8 #13），写 3 或自定义
 p.deqScalar = 0; // p{} 不做零初始化；只有 deqScalar/reluScalar/vectorRelu 无默认值（§8 #4）。NoQuant 下不被读
 AscendC::Fixpipe<float, float, kCfgNdToUb>(ub, l0c, p);
}
```

> 该写法只把数据放进 AIV0 的 UB。若算子按 1C:2V 切分，AIV1 需要第二条指令（策略 B）或直接改 SplitM（策略 C）。

### 7.2 1C:2V 主路：SplitM 单指令

```cpp
__aicore__ inline void CopySplitMToBothAivUbs(const AscendC::LocalTensor<float>& ubPerAiv,
 const AscendC::LocalTensor<float>& l0c,
 uint16_t m, uint16_t n)
{
 AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p{};
 p.nSize = n;
 p.mSize = m; // 必须偶数
 p.srcStride = static_cast<uint16_t>(((m + 15U) / 16U) * 16U); // CeilAlign(m,16)：CeilAlign 属 adv_api/matmul，普通 kernel 未必可见；m ≤ 65520 才不溢出 uint16（§8 #19）
 p.dstStride = n;
 p.dualDstCtl = 1; // DUAL_DST_SPLIT_M：每 lane 写 m/2 × n
 p.quantPre = QuantMode_t::NoQuant; // 双目标强制。QuantMode_t 是全局 typedef，不能写 AscendC::
 p.reluEn = false; // 双目标强制
 p.unitFlag = 0; // 官方 DoubleCopyOutDualSplitM 取 0；单次完结取 3（见 §8 #13）
 // 目的地址：两 lane 各写同一相对基址；多 tile 时 AIC 侧 offset 必须按 >>1 折算（§4.2 已实测，证据见 A.2）
 AscendC::Fixpipe<float, float, kCfgNdToUb>(ubPerAiv, l0c, p);
}
```

### 7.3 1C:2V 两条指令版（float → half，用 F322F16）

```cpp
__aicore__ inline void TwoFixpipeToBothAivUbs(const AscendC::LocalTensor<half>& ubHalf,
 const AscendC::LocalTensor<float>& l0c,
 uint16_t m, uint16_t n)
{
 if ASCEND_IS_AIV { return; }
 AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p{};
 p.nSize = n;
 p.dstStride = n;
 p.srcStride = static_cast<uint16_t>(((m + 15U) / 16U) * 16U);
 p.dualDstCtl = 0;
 p.quantPre = QuantMode_t::F322F16; // 全局 typedef，不能写 AscendC::；纯转换用 F322F16（不取 deqScalar）
 p.unitFlag = 0; // 与官方 DoubleCopyOutTwoFixpipe 一致

 const uint64_t halfM = (m + 1U) / 2U; // 等价于官方 CeilDiv(baseM, 2)：奇数 M 时 lane0 多一行
 p.mSize = static_cast<uint16_t>(halfM);
 p.subBlockId = false; // → AIV0
 AscendC::Fixpipe<half, float, kCfgNdToUb>(ubHalf, l0c, p);

 p.mSize = static_cast<uint16_t>(m - halfM);
 p.subBlockId = true; // → AIV1
 // 源偏移 = halfM * BLOCK_CUBE(=16) 个元素，与官方 c1Local_[l0cOffset + halfBaseM*BLOCK_CUBE] 一致
 AscendC::Fixpipe<half, float, kCfgNdToUb>(ubHalf, l0c[halfM * 16], p);
}
```

> 若确实需要**随路量化**而不是单纯 dtype 转换，把 `quantPre` 换成
> `QuantMode_t::QF322F16_PRE` 这类标量量化模式，并**必须给真实系数** `p.deqScalar`
> （该模式会走 `set_quant_pre(deqScalar)`；填 0 的结果是全 0）。

### 7.4 三个样例的共同前提

- **只在 AIC 侧调用**：`Fixpipe` 的全部实现都包在 `if ASCEND_IS_AIC` 内（§2.1），AIV 分支里调用是静默空操作。
- **参数写顺序的屏障不用自己加**：`FixpipeL0C2UBImpl` 在 `set_quant_pre` 之后、发指令之前已插 `PipeBarrier<PIPE_FIX>`（紧挨着发指令那一行）。但这**只保证「标量参数写 → fixpipe 指令」的顺序**；`Mmad`(PIPE_M) → `Fixpipe`(PIPE_FIX) 的先后是否还需 `SetFlag/WaitFlag<HardEvent::M_FIX>`，官方 matmul stage 层没有显式同步（靠指令序），属**需实测**项。
- **写完必须做 AIC→AIV 握手**：Fixpipe 只是把数据放进 UB，AIV 还需要 `CrossCoreWaitFlag` 才知道数据到了（§5.4）；两个 lane 要各发一次。
- **参数错误不会报错**：真机上没有运行时校验（§6），样例里的取值必须自己算准。
- **已编译验证**：本节三个样例用 `ccec -xcce -std=c++17 -D__DAV_C310__ --cce-aicore-arch=dav-c310-{cube,vec}` 实测 **exit 0**；把 `unitFlag = 3` 换成 `FINAL_ACCUMULATION` 则报 `use of undeclared identifier`（反证 §8 #13）。
- 样例里的 `m`/`n` 都应是**已经过 §3.2 约束检查**的值（`nSize` 满足 32B 对齐等）。

---

## 8. 易错点

| # | 易错点 | 后果 | 正解 |
|---|---|---|---|
| 1 | 把 `subBlockId` 当 int / 当"任意 lane 数组下标" | 只是 `bool`，超过 1 的值被截断为 true | 用 `false/true` 或 `lane == 1` |
| 2 | 忘写 `dualDstCtl`，指望默认值 | 默认 0 = 单目标，另一个 AIV 收不到数据 | 显式 `p.dualDstCtl = 0;` 或目标模式 |
| 3 | `srcStride` 填 `M/16`；或只拷部分行时按 `mSize` 算 | 源偏移小 16 倍，读到错块；`srcStride` 是 L0C 物理几何，必须用完整 `baseM` | 填 `((M + 15) / 16) * 16`（M=64 → 64）；官方写成 `CeilAlign(M,16)`，但那个 `CeilAlign` 在普通 kernel 里不可见（见本表第 16 条） |
| 4 | `p{}` 后直接用标量量化 | `deqScalar` **没有默认成员初始化器**，构造函数是用户提供的空体 → `p{}` **不会清零**（host 复刻实验实测为栈垃圾且每次不同）→ 脏值进 `set_quant_pre`。`reluScalar`/`vectorRelu` 同样未初始化，但**在 3510 路径上**无消费点（`dav_m510` 的实现里会读，别处不要假设安全），不影响结果 | 量化路径显式赋 `deqScalar`（其余两个可忽略：`deqScalar` 只在标量量化模式下才被读） |
| 5 | 把 UB 张量传给 `isToUB=false` 的 config | 走 L1 通路，UB 地址被当 L1 用 | config 与 dst 位置必须自洽；建议封装成唯一入口 |
| 6 | 在 AIV 分支调用 `Fixpipe` | 静默空操作 | Fixpipe 只能在 AIC 侧发起 |
| 7 | 用 `PIPE_FIX` 在 AIV 侧 Set/Wait | 空操作 → 同步丢失 | AIV 侧用 `PIPE_V` / `PIPE_MTE2`（官方取值） |
| 8 | 只给一个 lane 发 flag，却让两个 AIV 都等 | 未收到 flag 的 lane 卡死 | 每个 lane 各发一次（`base` / `base+16`） |
| 9 | 双目标 + 随路 quant/relu | **真机不报错、静默算错**（断言只在 CPU-debug 生效，§6） | 走策略 B（两条指令）或去掉 quant/relu |
| 10 | SplitM 的 `mSize` 为奇数 | 真机静默算错（CPU-debug 才 abort） | M 补到偶数，或改策略 B（官方按 `CeilDiv(M,2)` 不均分） |
| 11 | 跨代复用到 A2/A3 | `isToUB` 被静默丢弃 → 变成写 L1 | 用 `__NPU_ARCH__` 隔离，别共用 `constexpr` |
| 12 | 直接用 `FixpipeParamsC310` / `Arch3510` 名字判断是否同代 | 二者是同一结构体（派生别名），不是两代 | 见 §2.3 |
| 13 | 直接写 `p.unitFlag = FINAL_ACCUMULATION;` | **编译失败**：它不是 AscendC 公开常量，只在官方算子代码里定义（值 `3`；同一处还定义了 `NON_FINAL_ACCUMULATION=2`） | 写 `3`（最后一次累积）/ `2`（非最后一次）/ `0`（L0C 乒乓），或自己定义同名常量 |
| 14 | 依赖断言兜底（"参数错会报错"） | 真机**静默算错/越界写**（§6） | 自己在前端校验参数，别指望运行时断言 |
| 15 | 写 `GetTaskRatio` | 它不是官方文档接口，与全仓 22 处 `GetTaskRation` 用法分裂；且 `GetTaskRation` 和 `GetTaskRatio` 都能编过（impl 层并列定义，该 impl 头又被 intf 头无条件包含），编译期不会暴露 | 一律用 `GetTaskRation`（返回 `int64_t`） |
| 16 | 在普通 kernel 里用 `AscendC::CeilAlign` | 它只在 matmul 的 adv_api 头里定义，普通 kernel 看不到；可能编不过 | 自己写 `((m + 15) / 16) * 16` |
| 17 | AIV 侧照抄"+lane×16"推 flagId | 官方 client 侧用的是另一套映射（`waitFixpId = base + instIdx` 或 `devEvtID = instIdx*2 + lane`），照抄会等错 id | 两边协议自己定并实测（§5.2） |
| 18 | SplitN + NZ 输出 | 官方文案声明不支持，但该断言有前提：仅当 `DUAL_DST_SPLIT_N && ibShare && PhyPosIsUB(C) && IsSupportL0CToUB` 才触发；非 ibShare 的组合不会被拦。且真机本就不执行这类断言 | N 方向对半切只用 ROW_MAJOR |
| 19 | 用 `uint16_t` 装对齐后的 M 作 `srcStride` | `mSize ∈ [65521, 65535]`（`CheckFixpipeParams` 允许的上界）时 `((m + 15) / 16) * 16 ≥ 65536` 回绕成 0/16，源偏移整块错位 | 显式断言 `baseM ≤ 65520`（一次性 L0C 拷出不可能这么大），中间量用 `uint32_t` 再校验 |

---

## 9. 与相邻文档的分工

本层内容保持独立成篇，与相邻文档的边界如下（避免同一事实维护两处口径）：

| 内容 | 归属 | 说明 |
|---|---|---|
| V220 与 Arch3510 的字段对照、`DataCopyCO12DstParams`、`unitFlag` 与 Mmad 的配合 | [api-cross-gen-fixpipe.md](./api-cross-gen-fixpipe.md) | 该文负责「跨代怎么换参数」，本篇负责「1C:2V 怎么分发、怎么同步」 |
| 跨核 flag 的通用规则（`mode`、flagId 分配、SyncAll 保留区） | [api-crosscore-sync.md](./api-crosscore-sync.md) | 本篇 §5 只补 intra-block（mode 4）在 1C:2V 下的具体用法 |
| Fixpipe 三条通路的编译期分派、两套重载、参数单位与对齐 | 本篇 §2–§3 | — |
| 双 AIV 四种分发策略、`subBlockId` 语义、真机实测 | 本篇 §4 + 附录 A | 唯一有硬件证据的地方 |
| 断言与校验清单、断言消息速查 | 本篇 §6–§6.1 | — |
| 核数 / 容量 / 频率等硬件参数表 | `ops/npu-arch/references/npu-hardware-params.md` | 本篇 §1.1 只保留「PR 与 DT 计算微架构一致」这一结论 |

---

## 附录 A：真机实测记录（Ascend950PR 957c）

**目的**：验证 §4 中三条只有代码证据、没有硬件证据的主张。**结论：三条全部得到证实。**

### A.0 环境

| 项 | 值 |
|---|---|
| 芯片 | `Ascend950PR`，`NPU Name 957c`，`Chip Version V100`，软件 `25.7.rc1.7` |
| 拓扑 | `ai_core_cnt=28` / `cube_core_cnt=28` / `vector_core_cnt=56`，`cube_vector_combine=split` → **1 AIC : 2 AIV** |
| MIX 编译 | `__global__ __mix__(1,2)` |
| 工程 | 复用 `ops/ascendc-direct-invoke-template/` 的 ASC + CMake 路线，工程建在 `/tmp/fxprobe`（仓库文件未改动） |
| 编译 | `cmake .. -DCMAKE_PREFIX_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64/cmake && make -j4` |
| 交叉验证 | AIV 侧 `GetSubBlockIdx` 实测返回 `0/1`，与 lane 一一对应 |

**模板里两个坑（本机实测）**：

| 项 | 模板写法 | 本机应改为 |
|---|---|---|
| arch | `--npu-arch=dav-2201` | **`dav-3510`**（平台配置里 `NpuArch=3510`） |
| ACL include | `${ASCEND_HOME_PATH}/aarch64-linux/include` | `x86_64-linux/include` |

### A.1 Q1：单个 AIC 能否跨 lane 写

方法：两个 AIV 先把自己的 32 KB UB 窗口毒化成 `-1.0f`，`SyncAll<false>` 会合后 AIC 才发 Fixpipe，二次会合后各自 dump；统计窗口内等于 golden（64.0）的位置。

```
variant 100 (subBlockId=0): lane0: 4096 cells, row-runs=[(0,63)], 64 cols
 lane1: NO WRITE (poison intact)
variant 101 (subBlockId=1): lane0: NO WRITE (poison intact)
 lane1: 4096 cells, row-runs=[(0,63)], 64 cols
```

**结论**：严格互补、各为完整 64×64（`dstStride=64` 无洞）。是**路由**，不是写两份。

### A.2 Q2：SplitM 的目的地址要不要 `>>1`

方法：`dualDstCtl=1`、`mSize=64`（每 lane 应收 32 行），第二块 tile 分别用折半/原值地址。

```
variant 103 (off = 1024, 即 >>1): lane0/lane1: 2048 cells, row-runs=[(0,31)] 稠密
variant 104 (off = 2048, 不折半): lane0/lane1: 2048 cells, row-runs=[(0,15),(32,47)] 16..31 空洞
```

**结论**：官方 `>>1` 是硬性算法（假设 2 成立）。**必须用第二个 M tile 才能分辨**——单 tile 且 `curRow=0` 时两种假设都是 offset 0。

### A.3 Q3：`dualDstCtl=1` 时 `subBlockId` 是否还有效

方法：固定 SplitM，`subBlockId` 取 `false`/`true` 各跑一次，四种组合（1×64 行 / 2×32 行 × 两种数据）× 两 lane 全比 sha256。

```
subBlockId 0(v102) vs 1(v105): lane0/lane1 identical=True
subBlockId 0(v103) vs 1(v106): lane0/lane1 identical=True
```

**结论**：**被硬件完全忽略**，目的端只由 `dualDstCtl` 决定。与 Q1 的单目标情形形成对照。

### A.4 踩坑清单（实测踩出来的，可直接省几小时）

| 坑 | 现象 | 正解 |
|---|---|---|
| `if ASCEND_IS_AIC { }` 加括号 | 编译报 "type name requires a specifier or qualifier" | 宏本身展开成 `constexpr(...)`，**不能加括号** |
| `MM`/`NN`/`KK` 作常量名 | 与 AscendC 内符号歧义 | 换名 |
| 只靠 `CrossCoreSetFlag<2,PIPE_FIX>` + `CrossCoreWaitFlag` | 同一 binary 两次跑，结果一次对一次空 | 可见性不足，需双向 `SyncAll<false>` 会合 |
| UB 上电内容 | 不毒化就无法区分"没写"与"写了 0" | 先整体写成哨兵值再 dump |
| L1→L0A/L0B 用 V1 `LoadData2DParams` | 结果错误 | 用 `LoadData2DParamsV2`（V1 的参数结构在这个场景下不对） |
| 64×64 fp32 的 Fixpipe 参数 | — | `mSize=64, nSize=64, srcStride=64, dstStride=64, params.ndNum=1`（与 §3.1/§3.3 一致，实测可跑通） |

**未覆盖**：该实验的 A 操作数 L1→L0A 装载有缺陷（只能 materialize 16 个互不相同的行），因此"SplitM 具体把哪半源行给哪个 lane"只有位置证据（每 lane 恰 `mSize/2` 行），没有源行身份证据。三条结论本身是位置/字节问题，不受此影响。
