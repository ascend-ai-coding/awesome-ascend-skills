# Fixpipe / L0C 回写参数指南（V220 与 Arch3510）

> **平台**：`FixpipeParamsV220` 用于 DAV_2201（A2/A3）；`FixpipeParamsArch3510` 用于 DAV_3510（950）。两套参数**同名不同义、单位不同**，照搬旧参数必错。
>
> **适用**：DAV_2201（A2/A3）→ DAV_3510（950）迁移时 L0C 回写参数体系的切换。
> **参数真源**：官方文档 `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/`（Fixpipe / FixpipeParamsArch3510 全字段）。
> 迁移决策（何时弃高阶走低阶直跑、Fixpipe 与 Mmad 的 unitFlag 配合）见迁移技能 `ascendc-cross-gen-migration` 的 cube-migration-guide 改动 3。

## 目录

1. [L0C → GM 回写：DataCopyCO12DstParams](#l0c--gm-回写datacopyco12dstparams)
2. [Fixpipe：L0C → 格式转换 → GM / L1](#fixpipel0c--格式转换--gm--l1)
3. [FixpipeConfig（输出格式配置，950 增强）](#fixpipeconfig输出格式配置950-增强)
4. [FixpipeParamsArch3510 关键字段（与 V220 的差异点是易错区）](#fixpipeparamsarch3510-关键字段与-v220-的差异点是易错区)

## L0C → GM 回写：DataCopyCO12DstParams

Cube 中 `Mmad → L0C`，计算完成后**不能直接用普通 GM↔UB `DataCopyParams` 理解**（见 [api-datacopy.md](api-datacopy.md)）：

```cpp
DataCopy(GlobalTensor<T> dst, LocalTensor<U> src, DataCopyCO12DstParams p);
```

核心字段：`nSize`、`mSize`、`srcStride`、`dstStride`、`unitFlag` 等。

| 字段 | 单位 | 含义 |
|---|---|---|
| `nSize` / `mSize` | — | 源 NZ 矩阵的 N / M 大小 |
| `srcStride` | **`C0_Size = 16 × sizeof(srcType)`** | 源 NZ 中相邻 Z 排布的距离 |
| `dstStride` | **随输出格式变化**（见下表） | 目的 Z 排布 stride 或 ND row pitch |

`dstStride` 的意义取决于是否 NZ2ND：

| 方向 | 单位 | 含义 |
| --- | --- | --- |
| NZ → NZ | 32B | 表示目标 Z 排布 stride |
| NZ → ND | element | 表示目的矩阵 row pitch |

> ⚠️ 不能看到 `stride` 就默认「单位都是 32B」——`dstStride` 的单位随输出格式变化。

## Fixpipe：L0C → 格式转换 → GM / L1

功能上把它理解成 `L0C → FixPipe → 格式转换 / Cast / Quant / ReLU → GM / L1`：

```cpp
FixpipeParamsV220 p;

p.mSize = M;
p.nSize = N;
p.srcStride = ...;
p.dstStride = ...;

Fixpipe(dstGM, srcL0C, p);
```

DAV_2201（Atlas A2 / A3，含 910B）常见就是 `FixpipeParamsV220`。官方支持矩阵：**A2/A3 仅支持含 `FixpipeParamsV220` 的接口，950 同时支持 V220 与 `FixpipeParamsArch3510`**。另可配置 `ndNum`、`srcNdStride`、`dstNdStride`、`quantPre`、`deqScalar`、`reluEn`、`unitFlag`、`channelSplit`。

### unitFlag：Cube 与 Fix 的细粒度 overlap

`unitFlag` 用于让 Cube 与 Fix 流水做细粒度 overlap：`Mmad 算完一个分形 → FixPipe 立即搬出这个分形 → Mmad 继续算下一个`。**官方要求 Mmad 和 Fixpipe 两侧配合配置 `unitFlag`。**

随路 NZ→ND 转换可用 `Nz2NdParams`，但**它的字段数跨代不同**：**DAV_3510 是 3 字段** `{ndNum, srcNdStride, dstNdStride}`（即上面 Arch3510 节的形态）；**DAV_2201 是 5 字段** `{nz2ndEn, ndNum, srcNdStride, dstNdStride, originalNSize}`，2201 上开启随路转换需设 `nz2ndEn = true`。见 [api-nd2nz-nz2nd-nddma.md](api-nd2nz-nz2nd-nddma.md)。

## FixpipeConfig（输出格式配置，950 增强）

```cpp
struct FixpipeConfig {
    CO2Layout format;   // NZ=0(保持NZ) / ROW_MAJOR(开启NZ2ND→ND) / COLUMN_MAJOR(仅950，NZ2DN→DN)
    bool isToUB;        // 目的地址是否在 UB（L0C→UB 通路）
};
```

## FixpipeParamsArch3510 关键字段（与 V220 的差异点是易错区）

| 字段 | 说明 |
|---|---|
| nSize | N 方向大小；不开启 channelSplit 时**必须为 16 的倍数**（channelSplit 时为 8 的倍数） |
| mSize | M 方向大小；mSize=0 表示 NOP |
| srcStride | 相邻 Z 排布起始地址偏移，单位 **C0_SIZE**（16×sizeof(T)），填 mSize 对 16 向上取整 |
| dstStride | 开启 NZ2ND/NZ2DN 时 = 目的**每一行元素个数（单位 element）**；不开转换时 = 相邻 Z 排布偏移。**注意：`FixpipeParamsArch3510.dstStride` 单位是 element，而 `FixpipeParamsV220.dstStride` 是 datablock（32B）——两者单位不同！** |
| params | 随路格式转换参数：NZ（普通搬运）→ `Nz2NdParams{ndNum, srcNdStride, dstNdStride}`（**此为 3510 的 3 字段形态**）/ 950 新增 `Nz2DnParams{dnNum, srcNzMatrixStride, dstDnMatrixStride, srcNzC0Stride}` |
| dualDstCtrl / subBlockId | **仅 L0C→UB 通路有效**（双目标 / 子块模式） |
| unitFlag | Mmad 与 Fixpipe 细粒度并行（每算完一个分形立即搬出）；开启时 Mmad 和 Fixpipe 的 unitFlag 须同为 2 或 3 |

> **易错点速记**：本表 `FixpipeParamsArch3510.dstStride` 单位是 **element**，而 `FixpipeParamsV220.dstStride` 是 datablock/32B（**两个不同结构体，不是同代结论冲突**）；`nSize` 不开启 channelSplit 时须为 **16 的倍数**；`dualDstCtrl` / `subBlockId` **仅 L0C→UB 通路**有效。
