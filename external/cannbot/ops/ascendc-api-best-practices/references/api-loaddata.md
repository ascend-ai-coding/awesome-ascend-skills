# LoadData 系列 API 使用指南

L1 → L0 数据加载的 API 字段、对齐约束、NZ Layout 与 dtype 相关的关键差异。

---

## 目录

1. [API 概览](#api-概览)
2. [NZ Layout C0 大小随 dtype 变化](#nz-layout-c0-大小随-dtype-变化)
3. [LoadData2DMx 系列（MX 块量化格式）](#loaddata2dmx-系列mx-块量化格式)
4. [Multi-matrix 拼接到 L1 的 head offset 公式](#multi-matrix-拼接到-l1-的-head-offset-公式)
5. [详细文档与示例位置](#详细文档与示例位置)

---

## API 概览

| API 系列 | 用途 | 典型数据类型 |
|---------|------|------------|
| `LoadData2D` / `LoadData2DV2` | L1 → L0 加载，基础数据通路 | 整数 / 浮点 |
| `LoadData2DMx` | L1 → L0 加载 MX 块量化格式数据（配套 scale 同步加载） | mxfp8 / mxfp4 等 |
| `LoadData3D` | L1 → L0 加载，卷积 im2col 模式 | 卷积场景 |

> 同一 API 在不同平台 / CANN 版本可能有 `LoadData2DMx.md` / `LoadData2DMx-35.md` 等多个变体，功能略有差异。使用前用通配符搜 `asc-devkit/docs/api/context/LoadData*.md` 列出全部变体。

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

LoadData2DParamsV2 / LoadData2DMxParams 中以下字段以"datablock = C0 字节单元 = 32 字节"为单位，而不是元素单位：

| 字段 | 单位 | 含义 |
|------|------|------|
| `kStep` | datablock（32 字节） | K 方向步进，等于 `K 维元素数 / C0_elements_for_dtype` |
| `kStartPosition` | datablock（32 字节） | K 方向起始偏移，等于 `K_offset / C0_elements_for_dtype` |
| `mStep` | M-fractal（16 行，与 dtype 无关） | M 方向步进 |
| `srcStride` / `dstStride` | 16 字节（L1 / L0 fractal stride） | 与 dtype 无关 |

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

## LoadData2DMx 系列（MX 块量化格式）

LoadData2DMx 用于 MX 块量化格式（mxfp8 / mxfp4 等）的 L1 → L0 加载，数据本体 + scale 一次加载到 L0_A_MX / L0_B_MX。

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

## 详细文档与示例位置

| 资源 | 路径 |
|------|------|
| API 文档（用通配符列出所有变体） | `asc-devkit/docs/api/context/LoadData*.md` |
| LoadData2DParamsV2 字段对照 | `asc-devkit/docs/api/context/LoadData2DParamsV2.md` |
| Mmad 配套 API 文档 | `asc-devkit/docs/api/context/Mmad*.md` |
| MX 路径 end-to-end 示例 | `asc-devkit/examples/.../load_data_2dmx_l12l0/` |
| 平台同步基座示例 | `asc-devkit/examples/.../matmul_s4/`（A5 平台 mode=4 同步） |

新增 LoadData 路径前必须查阅 API 文档 + 跑通对应 example，不要凭"和 fp16 一样"的假设直接照搬公式。
