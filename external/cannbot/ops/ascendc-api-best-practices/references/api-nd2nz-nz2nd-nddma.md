# ND2NZ / NZ2ND / NDDMA 与布局变换搬运指南

> **平台**：ND2NZ / NZ2ND / Slice 在 Atlas A2/A3（DAV_2201）与 Ascend 950（DAV_3510）均可用，但 `Nd2NzParams` 字段宽度跨代不同（见「跨代字段宽度」）；**NDDMA 仅 DAV_3510（950PR / 950DT）可用，Atlas A2/A3 不支持**。
> **参数真源**：本机 CANN 头文件 `asc/include/basic_api/kernel_struct_data_copy.h` 与官方文档 `docs/zh/api/SIMD-API/basic_api/`（ND2NZ 有 GM→L1 / UB→L1 / GM→UB 三个方向，参数表略有差异）。

布局变换搬运指的是「搬运的同时完成形状 / 布局重排」的一族接口，与 [`api-datacopy.md`](api-datacopy.md) 的 GM↔UB 纯搬运、[`api-loaddata.md`](api-loaddata.md) 的 L1→L0 加载区分开。

---

## 目录

1. [通路概览](#通路概览)
2. [ND2NZ 搬运](#nd2nz-搬运)
3. [NZ2ND 搬运](#nz2nd-搬运)
4. [Slice 搬运](#slice-搬运)
5. [NDDMA 搬运](#nddma-搬运)
6. [高维 stride 策略](#高维-stride-策略)
7. [参数单位速查](#参数单位速查)

---

## 通路概览

Cube 数据通路上共三段，本文件覆盖首段：

| 段 | 通路 | 接口族 | 文档 |
|---|---|---|---|
| ① | GM → L1 | ND2NZ `DataCopy`（`Nd2NzParams`） | **本文件** |
| ② | L1 → L0A / L0B | `LoadData2D` / `LoadData3D` / MX / 变体 | [api-loaddata.md](api-loaddata.md) |
| ③ | L0C → GM / L1 | `Fixpipe` / `DataCopyCO12DstParams` | [api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md) |

搬运算子接口族在 **DAV_2201（Atlas A2/A3，如 910B3）** 上的支持情况：

| 接口族 | 典型用途 | DAV_2201 |
|---|---|:--:|
| `DataCopy(..., count)` | 连续、32B 对齐搬运 | ✅ |
| `DataCopy(..., DataCopyParams)` | 2D / 块状 stride 搬运 | ✅ |
| `DataCopyPad` | 非 32B 对齐搬运 | ✅ |
| Slice `DataCopy` | 多维切片 | ✅ 偏易用型 |
| ND2NZ `DataCopy` | ND → NZ | ✅ |
| NZ2ND `DataCopy` | NZ → ND | ✅ |
| NDDMA | 真正 1~5 维通用 stride / pad | ❌ |
| `LoadData` | L1 → L0A/B 矩阵搬运 | ✅ |
| `LoadData` 3D | 卷积 img2col | ✅（V2；V1 仅 Atlas 训练/推理系列） |
| `DataCopy` / `Fixpipe` | L0C → GM / L1 | ✅ |
| UB→UB `DataCopy` | UB 内数据搬运 | ✅ |
| `Copy` | UB 内 Vector 型搬运 | ✅（mask / repeat 形态）；**count 形态仅 3510** |

> **易用性 vs 极致性能**：官方把 Slice / ND2NZ / NZ2ND 标为偏「易用性设计」的接口。对极致性能算子，优先把高维问题降解为「维度合并 → 最大连续区 → `DataCopyParams` → 少量外层循环」（见「高维 stride 策略」）。

---

## ND2NZ 搬运

`ND → NZ`（GM → L1，把 ND 矩阵在搬运过程中排成 Cube 要求的 NZ 分形布局）：

```cpp
DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, Nd2NzParams params);
```

### 字段与单位

| 字段 | 单位 | 含义 | 取值范围 |
|---|---|---|---|
| `ndNum` | 个 | ND 矩阵数量；`0` 表示 NOP | [0, 4095] |
| `nValue` | 行 | 源矩阵行数 N；`0` 表示 NOP | [0, 16384] |
| `dValue` | 元素 | 源矩阵列数 D；`dValue * sizeof(T)` 非 32B 对齐时目的矩阵补 0；`0` 表示 NOP | [0, 65535] |
| `srcNdMatrixStride` | **element** | 相邻 ND 矩阵起始地址偏移；`ndNum = 1` 时无意义填 0 | [0, 65535] |
| `srcDValue` | **element** | 源矩阵**一行包含的元素个数**（即行 pitch） | [1, 65535] |
| `dstNzC0Stride` | **C0_SIZE（32 字节）** | 目的 NZ 中相邻 Z 矩阵起始地址偏移 | [1, 16384] |
| `dstNzNStride` | **C0_SIZE（32 字节）** | 目的 NZ 中 Z 矩阵相邻行起始地址偏移 | [1, 16384] |
| `dstNzMatrixStride` | **element** | 相邻 NZ 矩阵起始地址偏移；`0` 表示每个 NZ 矩阵都覆盖第一个 | [0, 16384] |

> ⚠️ **最容易错的两点**：① `dstNzC0Stride` / `dstNzNStride` 的单位是 **C0_SIZE（32 字节）**，不是 element；② `srcDValue` 是**行 pitch（一行包含的元素个数）**，不是「有效列数」。写成本 skill 其余文档里的「gap / stride」语义会整体错位。

典型写法（`ND [32, 48]`，GM 实际 pitch = 64）：

```cpp
Nd2NzParams p;
p.ndNum             = 1;
p.nValue            = 32;   // 行数
p.dValue            = 48;   // 列数（有效）
p.srcDValue         = 64;   // 行 pitch（element）
p.srcNdMatrixStride = 0;    // 单矩阵
p.dstNzC0Stride     = 32;   // = CeilAlign(nValue, 16)，单位 C0_SIZE
p.dstNzNStride      = 1;    // 单位 C0_SIZE
p.dstNzMatrixStride = 0;    // 单矩阵
DataCopy(l1Dst, gmSrc, p);
```

> ⚠️ **`dstNzC0Stride` 要按 `CeilAlign(nValue, 16)` 计算**（官方示例即 `dstNzC0Stride = CeilAlign(m, fractalShape[0])`，其中 `m` = 行数 = `nValue`）。取值偏小会导致目的 L1 布局**重叠写入**，而官方明确：**「搬运至 L1 Buffer 的数据不能重叠，如果存在重叠写入，硬件不会产生任何警告或错误，并且不保证重叠数据的写入顺序」**——属静默错误。

DMA 在读源时即知道「每行有效 48 element，下一行从 +64 element 开始」，并在搬运过程中直接排成 NZ——**搬运同时完成 Cube 所要求的数据布局转换**，这是这类接口的主要价值，而不是「比 DataCopy 快一点」。

### C0 与 dtype

C0 恒为 32 字节，因此 C0 内的元素数随 dtype 变化（fp16/bf16 = 16、fp8/int8 = 32、fp4 = 64、int32 = 8）。ND 的两行数据经 ND2NZ 后，在 NZ 中以 C0 为内层 DataBlock 重排。**完整的 C0 元素数表、按 C0 计数的字段换算与踩坑见 [api-loaddata.md](api-loaddata.md) 的「NZ Layout C0 大小随 dtype 变化」一节**，本文件不重复。

### 跨代字段宽度

`Nd2NzParams` 在 DAV_2201 与 DAV_3510 上**字段类型不同**（字段名相同）：

| 字段 | DAV_2201（A2/A3） | DAV_3510（950） |
|---|---|---|
| `ndNum` / `nValue` / `dstNzC0Stride` / `dstNzNStride` | `uint16_t` | `uint16_t` |
| `dValue` | `uint16_t` | `uint32_t` |
| `srcNdMatrixStride` / `srcDValue` | `uint16_t` | `uint64_t` |
| `dstNzMatrixStride` | `uint16_t` | `uint32_t` |

跨代移植时聚合初始化与溢出行为会变化，需确认目标分支（头文件以 `#if (__NPU_ARCH__ == 3510)` 分叉）。

---

## NZ2ND 搬运

`NZ → ND`（**UB → GM**，逻辑位置 `VECOUT → GM`，把 NZ 分形还原为 ND）：

```cpp
DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, Nz2NdParamsFull params);
```

> 官方文档标题即「UBToGM 随路转换 NZ2ND 搬运」，并注明该接口为**软仿接口**（从易用性角度设计）。源操作数是 UB 上的 `LocalTensor`，**不是 L0C**——L0C 的 NZ→ND 走 `Fixpipe` 的随路转换（见 [api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md)）。
> 平台：Atlas A2/A3（DAV_2201）与 Ascend 950（DAV_3510）均支持。

| 字段 | 单位 | 含义 | 取值范围 |
|---|---|---|---|
| `ndNum` | 个 | 传输 NZ 矩阵的数目 | [0, 4095] |
| `nValue` | 行 | NZ 矩阵行数 | [1, 8192] |
| `dValue` | 列 | NZ 矩阵列数；**必须为 16 的倍数** | [1, 8192] |
| `srcNdMatrixStride` | **256（16×16）个元素** | 源相邻 NZ 矩阵的偏移（头 → 头） | [1, 512] |
| `srcNStride` | **16 个元素** | 源同一 NZ 矩阵内相邻 Z 排布的偏移（头 → 头） | [0, 4096] |
| `dstDStride` | **element** | 目的 ND 矩阵相邻行的偏移（头 → 头） | [1, 65535] |
| `dstNdMatrixStride` | **element** | 目的 ND 矩阵中来自源相邻 NZ 矩阵的偏移（头 → 头） | [1, 65535] |

> ⚠️ **这一族里有两个字段的单位不是 element**：`srcNdMatrixStride` 的单位是 **256 个元素**（= 16 个 DataBlock × 16 个元素），`srcNStride` 的单位是 **16 个元素**。不要按「其余 stride 都是 element」处理。

`dstDStride` 直接以 element 表示目的 ND 的 row pitch。例如最终 GM 要得到 `[M, N]`、实际每行 pitch = 160 elements：

```cpp
p.dstDStride = 160;
```

这与 `DataCopyParams` 的 gap 语义**不同**（普通 `DataCopyParams` 的 gap 是「上一段尾 → 下一段头」，见 [api-datacopy.md](api-datacopy.md)）。

> **注意区分两个同名结构体**：`Nz2NdParamsFull`（本节的独立 `DataCopy` 接口参数，定义在 `kernel_struct_data_copy.h`）与 `Nz2NdParams`（`Fixpipe` 随路转换用的参数，定义在 `impl/basic_api/utils/kernel_utils_struct_dma_params.h`）**不是同一个东西**。且 `Nz2NdParams` 跨代字段数不同——**DAV_3510 是 3 字段** `{ndNum, srcNdStride, dstNdStride}`，**DAV_2201 是 5 字段** `{nz2ndEn, ndNum, srcNdStride, dstNdStride, originalNSize}`。见 [api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md)。

---

## Slice 搬运

多维切片（易用型接口，适合 `x[a:b:s, c:d:t]` 这类逻辑描述）：

```cpp
DataCopy(dst, src, SliceInfo dstSliceInfo[], SliceInfo srcSliceInfo[], uint32_t dimValue);
```

| 字段 | 单位 | 含义 |
|---|---|---|
| `startIndex` | element | 起点 |
| `endIndex` | element | 终点 |
| `stride` | element | 切片之间跳过多少 |
| `burstLen` | **32B** | 每段切片多长 |
| `shapeValue` | element | 原始该维长度 |

```cpp
SliceInfo src[2] = {
    {16, 70, 7, 3, 87},   // 第一维：start=16, end=70, 跨 7 element, 每段 3×32B
    {0,   2, 1, 1,  3}    // 第二维（行维度）
};
```

多维情况下（`dimValue > 1`），官方要求**非最内层的 `burstLen` 必须配置为 1**，不支持配置成其他值（`burstLen` 仅在 `dimValue = 1` 时生效）。

官方同时给出取值约束：`endIndex - startIndex + 1 = (N - 1) × stride + N × burstLen × (32 / sizeof(T))`（`N` 为正整数），且横向切片元素数 × `sizeof(T)` 必须是 32 字节的倍数。

**来源**：官方文档 `memory_vector_compute/data_move/DataCopy_GMAndUB_slice.md`（`burstLen` 参数行 + 「约束说明」节）。

---

## NDDMA 搬运

NDDMA 是最接近 `numpy ndarray: shape + strides` 语义的 DMA，可自然表达 slice / transpose / broadcast / padding / 多维 stride。

```cpp
template <typename T, uint8_t dim, const NdDmaConfig& config>
DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, NdDmaParams<T, dim> params);
```

```cpp
template <uint8_t dim>
struct NdDmaLoopInfo {
    uint64_t loopSrcStride[dim];
    uint32_t loopDstStride[dim];
    uint32_t loopSize[dim];
    uint8_t  loopLpSize[dim];   // 左 padding
    uint8_t  loopRpSize[dim];   // 右 padding
};
```

**所有 stride / size 单位均为 element。** 类型与范围：`loopSrcStride` 为 `uint64_t`（[0, 2^40)）、`loopDstStride` 为 `uint32_t`（[0, 2^20)）、`loopSize` 为 `uint32_t`（[1, 2^20)）。

### 平台门禁（关键）

| 平台 | NDDMA |
|---|---|
| DAV_3510（Ascend 950PR / 950DT） | ✅ 支持 |
| DAV_2201（Atlas A2/A3，含 910B3） | ❌ 不支持 |

**DAV_2201（910B3 所在的 Atlas A2/A3 路线）不能使用 NDDMA。** 代码中如需条件编译，参考 [npu-arch](../../npu-arch/SKILL.md) 的平台判定方式。

### 使用前必须刷新 Cache

```cpp
__aicore__ inline void NdDmaDci();
```

官方要求：**在使用 NDDMA `DataCopy` 进行数据搬运前，必须调用 `NdDmaDci()` 刷新 NDDMA Cache**，否则可能读到旧缓存。这是最容易漏掉的一步。

### 别名：`MultiCopy*` 是旧名

`MultiCopyConfig` / `MultiCopyParams` 是 `NdDmaConfig` / `NdDmaParams` 的**旧名别名**（头文件中为 `using MultiCopyConfig = NdDmaConfig; // reserve old name`）。生产代码大量使用旧名，看到 `MultiCopy*` 与 `NdDma*` 应视为同一套接口，不要当成两组能力。

### 配置约束

`NdDmaConfig` 的 `loopLpSize` / `loopRpSize` 为 `uint16_t`，**必须小于 256**；取固定值 `unsetPad`（`0xFFFF`）时表示改由 `NdDmaLoopInfo` 按维度设置。

> ⚠️ 不要与 `NdDmaLoopInfo` 中的**同名**成员混淆：`NdDmaLoopInfo::loopLpSize` / `loopRpSize` 是 `uint8_t[dim]`，逐维设置，范围 [0, 255]。前者是「所有维度统一值」，后者是「逐维值」。

### 地址数学

无 pad 的 N 维 DMA：

- `src_offset = Σ index[k] × loopSrcStride[k]`
- `dst_offset = Σ index[k] × loopDstStride[k]`

二维 `src shape = [16, 32]` 按 row-major：

```cpp
loopSize      = {32, 16};
loopSrcStride = {1,  32};
```

**【推断】** NDDMA 参数维度习惯**从内层 → 外层**描述，所以 `dim0 = width`、`dim1 = height`，`src(x,y)` 的 `offset = x × 1 + y × 32`。官方文档对维度语义只给出「数组有效索引范围为 `[0, dim-1]`」与 2D 场景图，**未在文字中规定内外层顺序**，此处为按命名与图示的推断，落地前请用官方 2D 场景图核对。

### `stride = 0` 是广播

```cpp
loopSrcStride = {1, 0};   // 外层 stride 为 0 → 第 0/1/2… 行都指向同一源 = broadcast
```

注意与普通 `DataCopyParams` 的 `gap = 0`（表示**连续**）语义**完全相反**，不要跨接口套用。

---

## 高维 stride 策略

DAV_2201（无 NDDMA）上处理「任意维度、任意 stride」的答案：

```text
原 tensor → 分析连续维 → 合并连续维 → 尽可能压成 [outer, blockLen]
          → DataCopyParams → 无法表达的更外层维度用少量循环
```

以 `shape = [A, B, C, D]`、`stride = [sA, sB, sC, 1]` 为例：

- `sC == D` → C、D 连续，先合并：`[C, D] → [C×D]`
- `sB == C×D` → 继续合并：`[B, C, D] → [B×C×D]`

最终尽量得到 `[outer dimensions..., contiguous_len]`，把最后一个不连续维的 `blockCount` / `blockLen` / `srcGap` 交给一次 DMA，仅更外层才：

```cpp
for (...)
    DataCopy(...);
```

优化目标不是「让一个 `DataCopy` 表达所有维度」，而是：

> **尽可能增加一次 DMA 所表达的连续 / 规律数据量，减少 DMA 指令数量。**

末端不是 32B 时用 `DataCopyPad`，而不是 NDDMA。

---

## 参数单位速查

本族接口的字段单位**全集**（跨接口对照见 [api-quickref.md](api-quickref.md) 的「跨接口 stride 统一速查」；本表只列本族字段）：

| 接口 | 字段 | 单位 | 连续值 |
|---|---|---|---:|
| ND2NZ | `srcDValue` | element（行 pitch） | `dValue` |
| ND2NZ | `srcNdMatrixStride` / `dstNzMatrixStride` | element | — |
| ND2NZ | `dstNzC0Stride` / `dstNzNStride` | **C0_SIZE（32 字节）** | — |
| NZ2ND | `srcNdMatrixStride` | **256（16×16）个元素** | — |
| NZ2ND | `srcNStride` | **16 个元素** | — |
| NZ2ND | `dstDStride` / `dstNdMatrixStride` | element | `dValue` |
| NDDMA | `loopSrcStride` / `loopDstStride` | element | `1` |
| NDDMA broadcast | `loopSrcStride` | element | **`0`（不前进 = 广播）** |
| Slice | `startIndex` / `endIndex` / `stride` / `shapeValue` | element | — |
| Slice | `burstLen` | **32B** | — |

> 记法：本族**不是**「只有 ND2NZ 的 C0 字段例外」——`dstNzC0Stride` / `dstNzNStride` 用 C0_SIZE（32B），NZ2ND 的 `srcNdMatrixStride` / `srcNStride` 分别以 **256 / 16 个元素**为单位，其余 stride 类字段才是 element。与 GM↔UB `DataCopyPad` 的「GM 侧 byte / UB 侧 32B DataBlock」也是两套不同规则，不要互相套用。

---

## 相关文档

| 文档 | 内容 |
|---|---|
| [api-datacopy.md](api-datacopy.md) | GM↔UB / UB↔UB 纯搬运、`DataCopyParams` gap 语义、32B 对齐与截断 |
| [api-loaddata.md](api-loaddata.md) | L1→L0 加载、C0 元素数表、`LoadData2DParamsV2` 单位 |
| [api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md) | L0C 回写、`FixpipeParamsV220` / `FixpipeParamsArch3510` |
| [api-quickref.md](api-quickref.md) | 跨接口 stride 统一速查表与选型决策树 |
