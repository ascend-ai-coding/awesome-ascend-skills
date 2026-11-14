# LoadData 系列 API 使用指南

L1 → L0 数据加载的 API 字段、对齐约束、NZ Layout 与 dtype 相关的关键差异。

---

## 目录

1. [API 概览](#api-概览)
2. [LoadData2D V1 字段与单位](#loaddata2d-v1-字段与单位)
3. [LoadData2D 搬运举例](#loaddata2d-搬运举例)
4. [LoadData3D 与 LoadData 变体](#loaddata3d-与-loaddata-变体)
5. [NZ Layout C0 大小随 dtype 变化](#nz-layout-c0-大小随-dtype-变化)
6. [LoadData_2D_MX 系列（MX 块量化格式）](#loaddata_2d_mx-系列mx-块量化格式)
7. [Multi-matrix 拼接到 L1 的 head offset 公式](#multi-matrix-拼接到-l1-的-head-offset-公式)
8. [参数单位速查](#参数单位速查)
9. [详细文档与示例位置](#详细文档与示例位置)

---

## API 概览

| API 系列 | 用途 | 典型数据类型 |
|---------|------|------------|
| `LoadData_2D` / `LoadData_2D_V2` | L1 → L0 加载，基础数据通路 | 整数 / 浮点 |
| `LoadData_2D_MX` | L1 → L0 加载 MX 块量化格式数据（配套 scale 同步加载） | mxfp8 / mxfp4 等 |
| `LoadData_3D` | L1 → L0 加载，卷积 im2col 模式 | 卷积场景 |

> 同一 API 在不同平台 / CANN 版本可能有 `LoadData_2D.md` / `LoadData_2D_MX.md` / `LoadData_2D_V2.md` / `LoadData_2D_BitMode.md` 等多个变体，功能略有差异。使用前用通配符搜 `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "LoadData*.md"` 列出全部变体。

---

## LoadData2D V1 字段与单位

`LoadData2DParams`（配套接口 `LoadData_2D`）是最基础的一版，**一个基本搬运单元是 512B 分形**：

```cpp
struct LoadData2DParams {
    uint16_t startIndex;
    uint8_t  repeatTimes;
    uint16_t srcStride;
    uint8_t  sid;
    uint16_t dstGap;
    bool     ifTranspose;
    uint8_t  addrMode;
};
```

| 参数 | 类型 | 单位 | 含义 |
|------|------|------|------|
| `startIndex` | `uint16_t` | **512B** | 从第几个 fractal 开始 |
| `repeatTimes` | `uint8_t` | 次 | 搬几个 fractal（**≤ 255**） |
| `srcStride` | `uint16_t` | **512B** | 相邻源 fractal **头 → 头**距离；**`= 1` 表示连续** |
| `dstGap` | `uint16_t` | **512B** | 前一 fractal **尾 → 下一 fractal 头**的 gap；`= 0` 表示连续 |
| `ifTranspose` | `bool` | — | 分形转置 |
| `addrMode` | `uint8_t` | — | 地址递增 / 递减 |
| `sid` | `uint8_t` | — | 保留，一般 0 |

> ⚠️ **950 例外**：官方明确「特别针对 Ascend 950PR/Ascend 950DT，设置 `dstGap` 参数无效」。跨代移植时若依赖 `dstGap` 做目的侧跳写，在 950 上会失效（来源：官方 `LoadData_2D.md`「约束说明」节）。

> ⚠️ **与 `DataCopyParams` 语义相反**：`LoadData2D` 的 `srcStride` 是 **head → head**，所以 `= 1` 才是连续；而 `DataCopyParams` 的 `srcGap` 是 **tail → next head**，`= 0` 才是连续。不要跨接口套用（见 [api-datacopy.md](api-datacopy.md) 的「DataCopyParams 与 gap 地址公式」）。

---

## LoadData2D 搬运举例

L1 中有 `Fractal0, Fractal1, Fractal2, Fractal3 …`，每个 512B。要搬 `F0 F2 F4 F6`：

```cpp
LoadData2DParams p;
p.startIndex  = 0;
p.repeatTimes = 4;
p.srcStride   = 2;   // 头 → 头跨 2 个分形
p.dstGap      = 0;   // 目的 L0 连续
```

- 源：`F0 [F1] F2 [F3] F4 [F5] F6`（方括号内为跳过的分形）
- 目的 L0：`F0 F2 F4 F6`（连续）

---

## LoadData3D 与 LoadData 变体

### LoadData3D：卷积专用

`LoadData3DParamsV2<T>` 实际是 `L1 NC1HWC0 Feature Map → ImageToColumn → L0A/L0B`，**不仅是搬数据，而是在搬运过程中做 img2col**：

| 参数 | 含义 |
|---|---|
| `padList[4]` | left / right / top / bottom pad |
| `l1H` / `l1W` | Feature Map H / W |
| `channelSize` | C |
| `kExtension` / `mExtension` | 输出 K / M 方向长度 |
| `kStartPt` | K 起点 |
| `strideW` / `strideH` | 卷积 stride |
| `filterW` / `filterH` | kernel 大小 |
| `dilationFilterW` / `dilationFilterH` | dilation |
| `padValue` | pad value |

V1（`LoadData3DParamsV1`）还有 `c1Index`、`fetchFilterW/H`、`leftTopW/H`、`jumpStride`、`repeatMode`、`repeatTime`。这些参数本质是在告诉 MTE1「怎样从 NC1HWC0 Feature Map 中生成下一批 Cube 需要的二维分形」——**它不是普通 Tensor Copy 的替代品**。

### LoadData 变体

| 变体 | 说明 | 平台（官方支持矩阵） |
|---|---|---|
| `LoadData_2D` | 基础 L1 → L0 加载（V1，见上） | A2 / A3 / 950 |
| `LoadData_2D_V2` | 同上，V2 参数形态（见「NZ Layout C0 大小随 dtype 变化」→「影响」） | **仅 950** |
| `LoadData_2D_MX` | MX 块量化格式，数据本体 + scale 一次加载 | **仅 950** |
| `LoadData_2D_BitMode` | 位模式加载 | **仅 950** |
| `LoadDataWithStride` | 仍围绕 img2col，进一步允许指定输出 K 轴偏移 | **仅 950** |
| `LoadDataWithTranspose` | 搬运 + fractal transpose（`LoadData2dTransposeParams` / `LoadData2dTransposeParamsV2`） | A2 / A3 / 950 |
| `LoadDataWithSparse` | 同时处理稀疏权重和索引数据 | A2 / A3（**不支持 950**） |
| `LoadData_3D` | 卷积 img2col（V1 / V2 见上） | V2：A2 / A3 / 950；**V1：仅 Atlas 训练 / 推理系列 AI Core** |

> ⚠️ **平台差异极大，不要按「都是 LoadData」默认可用**：V2 / MX / BitMode / WithStride 是 **950 专属**，`LoadDataWithSparse` 反过来 **不支持 950**，`LoadData3DParamsV1` 只存在于 Atlas 训练 / 推理系列。

它们都属于 **Cube 输入搬运**，而非普通 Vector GM↔UB DMA。新增 LoadData 路径前必须查阅对应 API 文档并跑通官方 example。

**来源**：各接口官方文档的「产品支持情况」节（`LoadData_2D.md` / `LoadData_2D_V2.md` / `LoadData_2D_MX.md` / `LoadData_2D_BitMode.md` / `LoadDataWithStride.md` / `LoadDataWithTranspose.md` / `LoadDataWithSparse.md` / `LoadData_3D.md`）。

---

## NZ Layout C0 大小随 dtype 变化

L1 NZ Layout 中 C0 维度按字节宽度对齐（典型 32 字节 = 1 个 C0 datablock），所以**每个 C0 包含的元素数随 dtype 变化**：

| dtype | sizeof | C0 元素数 | C0 字节数 |
|-------|--------|----------|---------|
| fp16 / bf16 | 2 字节 | **16 元素**（常量 `CUBE_BLOCK`） | 32 字节 |
| fp8（mxfp8 / fp8_e4m3 / fp8_e5m2） | 1 字节 | **32 元素**（`FP8_C0_ELEMS`） | 32 字节 |
| fp4（mxfp4） | 0.5 字节 | **64 元素** | 32 字节 |
| int8 | 1 字节 | 32 元素 | 32 字节 |
| int32 | 4 字节 | 8 元素 | 32 字节 |

### 影响

**两套参数结构体的字段单位不同，不要混用**（MX 路径下 `loadDataParams`（`LoadData2DParamsV2`）与 `loadMxDataParams`（`LoadData2DMxParams`）并列传入）：

**`LoadData2DParamsV2`**（官方接口 `LoadData_2D_V2`，**仅 950**）：

| 字段 | 类型 | 单位 | 含义 |
|------|------|------|------|
| `mStartPosition` | `uint32_t` | **16 个元素**（与 dtype 无关） | M 轴起始位置；取值范围 [0, 255]（官方 `LoadData_2D_V2.md`；**头文件中无 range 注释**，故以官方文档为准） |
| `mStep` | `uint16_t` | **16 个元素** | M 轴搬运长度；取值范围 [0, 255]（官方 `LoadData_2D_V2.md`），`mStep=0` 为 NOP |
| `kStartPosition` | `uint32_t` | **32 字节** | K 轴起始位置，等于 `K 维元素数 / C0_elements_for_dtype`；取值范围 [0, 255]（官方 `LoadData_2D_V2.md`） |
| `kStep` | `uint16_t` | **32 字节** | K 轴搬运长度，同上换算；取值范围 [0, 255]，`kStep=0` 为 NOP |
| `srcStride` | **`int32_t`** | **512 字节**（= 一个数据分形，与 dtype 无关） | K 方向相邻分形起始地址的间隔。**注意不是 32 字节的 C0 单位** |
| `dstStride` | **`uint16_t`** | **512 字节** | 同上。**类型与 `srcStride` 不对称**：以 512B 为单位时 `dstStride` 上限 65535 × 512B ≈ 32MB，大 K 场景需核算 |

> ⚠️ `srcStride`（`int32_t`）与 `dstStride`（`uint16_t`）**类型不同**，不要当成同一类型的一对参数；两者的取值范围与溢出行为不一致。

> 例：`srcStride = 5` 表示相邻分形间隔 `5 × 512 = 2560` 字节。分形恒为 512 字节、与 dtype 无关：b16 为 16×16、b8 为 16×32、b4 为 16×64、b32 为 16×8（元素宽度不同，乘积都是 512B）。

**`LoadData2DMxParams`**（官方接口 `LoadData_2D_MX`，**仅 950**，仅 MX 量化路径）：

| 字段 | 单位 | 含义 |
|------|------|------|
| `xStartPosition` / `xStep` | **1 个分形 = 32 字节** | X（M）轴起始位置 / 搬运长度；取值范围 [0, 255]（官方 `LoadData_2D_MX.md`；**头文件中无 range 注释**，故以官方文档为准） |
| `yStartPosition` / `yStep` | **32 字节** | Y（K）轴起始位置 / 搬运长度；`yStep=0` 为 NOP |
| `srcStride` / `dstStride` | **32 字节** | X 方向相邻分形起始地址的间隔 |

> **为什么 stride 单位是 512 字节**：数据分形大小恒为 512 字节、与 dtype 无关（b16 为 16×16、b8 为 16×32、b4 为 16×64、b32 为 16×8，元素宽度不同但乘积都是 512B），故 `LoadData2DParamsV2` 的 stride 以 512 字节为单位；MX 路径则以 32 字节分形为单位。

### 典型踩坑

照搬 fp16 路径的字段公式到 fp8：

```cpp
// fp16 路径（正确）：
load.kStep = nActSize / CUBE_BLOCK;             // CUBE_BLOCK = 16（fp16 C0 元素数）
load.kStartPosition = nOff / CUBE_BLOCK;

// fp8 路径误用 fp16 公式：
load.kStep = nActSize / CUBE_BLOCK;             // ❌ fp8 C0 是 32 不是 16
load.kStartPosition = nOff / CUBE_BLOCK;

// fp8 路径正确：
load.kStep = nActSize / FP8_C0_ELEMS;           // ✅ FP8_C0_ELEMS = 32
load.kStartPosition = nOff / FP8_C0_ELEMS;
```

误用 fp16 公式后：
- LoadData 读取偏移错误（每条偏移到正确位置的 1/2 处）
- 输出 NaN / inf，但 kernel 不崩（仍是合法地址）
- 不同 dtype 切换时极易触发

### 防御措施

为每个 dtype 定义独立常量并显式用于公式：

```cpp
constexpr uint32_t CUBE_BLOCK = 16;           // fp16/bf16 C0 元素数
constexpr uint32_t FP8_C0_ELEMS = 32;         // fp8 C0 元素数
constexpr uint32_t FP4_C0_ELEMS = 64;         // fp4 C0 元素数
```

并在公式中显式使用对应常量，不混用。

---

## LoadData_2D_MX 系列（MX 块量化格式）

LoadData_2D_MX 用于 MX 块量化格式（mxfp8 / mxfp4 等）的 L1 → L0 加载，数据本体 + scale 一次加载到 L0_A_MX / L0_B_MX。

### K_BASE 与 yStep 整除性

LoadData2DMxParams 的 `yStep` 单位是 32 字节 = 1 个 MX 块量化沿 K 方向的 32 个元素。

K 方向多 K-pass 时，**K_BASE 必须满足 yStep 单位整除性**：

```
K_BASE % (32 / sizeof(dtype)) == 0
```

具体到不同 dtype：
- fp8（mxfp8）：K_BASE % 32 == 0
- fp4（mxfp4）：K_BASE % 64 == 0

K_BASE 不满足整除性会导致 `yStartPosition` 分数化错位，该 K-pass 内 scale 读取偏移错误。

### 单 K-pass vs 多 K-pass 选择

- K 维元素数 ≤ 32（对 fp8） → 必须单 K-pass（`Mmad(.k = K, ...)` 一次完成）
- K 维元素数 > 32 → 可多 K-pass，但 K_BASE 必须满足整除性

### N-segment 切分时 scale 的 xStartPosition 必须跟随 N-offset

当 B 矩阵需要按 N 方向分块加载到 L0B（N 方向尺寸超过 L0B 单次容量时触发），每个 N-segment 必须独立计算 scale 的起始位置：

```cpp
// ❌ scale 多 N-segment 复用同一起始位置
mxLoad.xStartPosition = 0;                          // 第二个 N-segment 也读 N-segment 0 的 scale

// ✅ scale xStartPosition 跟随 nOff
mxLoad.xStartPosition = nOff / CUBE_BLOCK;          // 单位 = M-axis 分形数
```

误用后第一个 N-segment（nOff=0）输出正常，后续 N-segment 输出 NaN / inf。

---

## Multi-matrix 拼接到 L1 的 head offset 公式

某些算子（multi-head MatMul、multi-head Attention 等）把多个矩阵拼接到同一片 L1 NZ buffer，做一次 LoadData 出去到 L0。每个矩阵在 L1 的起始 offset 必须按 NZ fractal 物理布局正确计算。

### 数据载体的 head offset

每个矩阵占 `(mEff / 16) × (K_dim / C0_elements)` 个 NZ fractal。head 间堆叠时的 element offset：

```cpp
head_offset_data = head_idx * mEff * C0_elements_for_dtype;
```

按 dtype 落到具体公式：

```cpp
// fp16 / bf16 数据载体
head_offset_fp16 = head_idx * mEff * CUBE_BLOCK;       // CUBE_BLOCK = 16

// fp8 数据载体
head_offset_fp8 = head_idx * mEff * FP8_C0_ELEMS;      // FP8_C0_ELEMS = 32
```

### scale 载体的 head offset（B16 视图 + Dn2Nz 加载）

MX 类格式的 scale 通常用 B16 视图（每 2 个 e8m0 合并为 1 个 half）+ Dn2Nz 加载到 L1。**scale 载体在 L1 的 M-fractal element count 与数据载体不同**：

```cpp
// scale 载体的 head offset（不同于数据载体！）
head_offset_scale = head_idx * mEff * scaleK_b16;      // scaleK_b16 = (K 维元素数 / 32) / 2
```

### 典型踩坑

- 把 fp16 公式直接搬到 fp8 数据载体：`head_idx * mEff * 16` 应改为 `head_idx * mEff * 32`
- 把数据载体公式搬到 scale 载体：写 `head_idx * mEff * CUBE_BLOCK` 或 `head_idx * mEff * FP8_C0_ELEMS` 都错，scale 应该用 `head_idx * mEff * scaleK_b16`
- 数据载体公式系数与 scale 载体公式系数**必须独立推导**，不可混用

### 症状

- 单 head 路径全部正确
- multi-head（≥ 2）出现 head 维奇偶 NaN 模式（`[OK, BAD, BAD, OK]` / `[OK, BAD, OK, BAD]` 等）
- 一处公式修对后模式变化（说明仍有其他公式错位）

---

## 参数单位速查

```cpp
// ==================================================
// LoadData2D V1 —— L1 -> L0
// ==================================================
LoadData2DParams p;
p.startIndex  = ...;   // 512B fractal ID
p.repeatTimes = ...;   // 次（≤ 255）
p.srcStride   = ...;   // 512B, head -> head（= 1 表示连续）
p.dstGap      = ...;   // 512B, tail -> next head（= 0 表示连续）
p.ifTranspose = false;
p.addrMode    = ...;
p.sid         = 0;
LoadData(dstL0, srcL1, p);
```

| 结构体 | 字段 | 单位 |
|---|---|---|
| `LoadData2DParams`（V1） | `startIndex` / `srcStride` / `dstGap` | **512B** |
| `LoadData2DParamsV2` | `mStartPosition` / `mStep` | 16 个元素 |
| `LoadData2DParamsV2` | `kStartPosition` / `kStep` | 32 字节 |
| `LoadData2DParamsV2` | `srcStride`（`int32_t`）/ `dstStride`（`uint16_t`） | **512B** |
| `LoadData2DMxParams` | `xStartPosition` / `xStep` | 1 个分形 = 32 字节 |
| `LoadData2DMxParams` | `yStartPosition` / `yStep` | 32 字节 |
| `LoadData2DMxParams` | `srcStride` / `dstStride` | 32 字节 |

> 三套结构体的 stride 单位**并非全部相同**：V1 与 V2 同为 512B，MX 为 32B；且 V2 的 `srcStride` / `dstStride` 类型不对称——迁移路径时必须逐字段核对，不能整段照搬。

跨接口的完整 stride 速查见 [api-quickref.md](api-quickref.md)；L0C 回写侧的单位见 [api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md)。

---

## 详细文档与示例位置

| 资源 | 路径 |
|------|------|
| API 文档（用通配符列出所有变体） | `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "LoadData_2D*.md"` |
| LoadData2DParamsV2 字段对照 | `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "LoadData_2D_V2*.md"` |
| Mmad 配套 API 文档 | `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "Mmad*.md"` |
| MX 路径 end-to-end 示例 | `asc-devkit/examples/01_simd_cpp_api/03_basic_api/03_matrix_compute/load_data_2dmx_l12l0/` |
| 平台同步基座示例 | `asc-devkit/examples/01_simd_cpp_api/06_compatibility_guide/matmul_s4/`（A5 平台 mode=4 同步） |

新增 LoadData 路径前必须查阅 API 文档 + 跑通对应 example，不要凭"和 fp16 一样"的假设直接照搬公式。
