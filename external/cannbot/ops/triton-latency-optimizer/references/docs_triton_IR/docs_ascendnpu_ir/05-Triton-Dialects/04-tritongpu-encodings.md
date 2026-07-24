# TritonGPU 布局编码详解

本文档详细描述 TritonGPU 方言中定义的所有布局编码属性。布局编码是 TritonGPU 的核心概念，决定了张量数据在 GPU 线程间的分布方式。所有定义均从 TableGen 源码精确提取。

源码参考：[TritonGPUAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)

## 1. 布局编码基础

### 1.1 布局函数定义

TritonGPU 张量通过布局属性（layout attribute）确定数据在 GPU 线程间的分布。形式化地，布局是一个函数 L，将多维张量索引 i ∈ Z^d 映射到一组整数 T，对应允许访问索引 i 处数据的 CUDA 线程索引集合。

布局分为两大类：
- **分布式布局（Distributed）**：数据分布在寄存器中，每个线程拥有部分数据
- **共享布局（Shared）**：数据存储在共享内存中，所有线程可访问

### 1.2 布局编码接口

#### LayoutEncodingTrait

所有布局编码共有的接口，提供 CTA 级布局信息。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `getCTAsPerCGA()` | `SmallVector<unsigned>` | 每 CGA 的 CTA 数量 |
| `getCTAOrder()` | `SmallVector<unsigned>` | CTA 顺序（最快变化轴在前） |
| `getCTASplitNum()` | `SmallVector<unsigned>` | CTA 分割数 |
| `getRank()` | `unsigned` | 布局的秩 |

#### DistributedEncodingTrait

分布式布局的扩展接口，描述 4 级计算层次：CTAs Per CGA → Warps Per CTA → Threads Per Warp → Values Per Thread。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `getRepOrder()` | `SmallVector<unsigned>` | 重复块的顺序 |
| `getTotalElemsPerThread(shape)` | `unsigned` | 每线程总元素数 |
| `getElemsPerThread(shape)` | `SmallVector<unsigned>` | 每线程各维元素数 |
| `toLinearLayout(shape)` | `LinearLayout` | 转换为线性布局 |

#### SharedEncodingTrait

共享内存布局的接口。

| 方法 | 返回类型 | 默认值 | 说明 |
|------|----------|--------|------|
| `getAlignment()` | `int32_t` | 16 | 默认对齐字节数 |

## 2. CTA 布局

### 2.1 CTALayoutAttr

描述 CTA（Cooperative Thread Array，即线程块）在 CGA（Cooperative Thread Group，即线程组集群）中的分布。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$CTAsPerCGA` | `ArrayRef<unsigned>` | 每 CGA 中的 CTA 数量 |
| `$CTASplitNum` | `ArrayRef<unsigned>` | 张量分割数 |
| `$CTAOrder` | `ArrayRef<unsigned>` | CTA 顺序 |

**语义说明**：

- 张量被 `CTASplitNum` 分割，分配给 `CTAsPerCGA` 个线程块
- 每个 CTA 处理形状为 `tensor_shape / CTASplitNum` 的子张量
- 当 `CTAsPerCGA = [1,...,1]` 时，`CTAOrder` 被规范化为 `[n-1,...,0]`

**示例**：

| 场景 | 张量形状 | CTAsPerCGA | CTASplitNum | 说明 |
|------|----------|------------|-------------|------|
| 示例 0 | [64, 128] | [1, 2] | [1, 2] | 2 个 CTA，各处理 [64, 64] |
| 示例 1 | [64, 128] | [1, 2] | [1, 1] | 多播，2 个 CTA 处理完整张量 |
| 示例 2 (A) | [M,K] | [SplitM, SplitN] | [SplitM, 1] | A 矩阵，dim1 多播 |
| 示例 2 (B) | [K,N] | [SplitM, SplitN] | [1, SplitN] | B 矩阵，dim0 多播 |
| 示例 2 (C) | [M,N] | [SplitM, SplitN] | [SplitM, SplitN] | C 矩阵，无多播 |

## 3. 分布式布局编码

### 3.1 BlockedEncodingAttr

连续分块布局，每个 warp 拥有目标张量的连续部分。通常用于促进 Load/Store 的内存合并访问。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$sizePerThread` | `ArrayRef<unsigned>` | 每线程元素数 |
| `$threadsPerWarp` | `ArrayRef<unsigned>` | 每 warp 线程数 |
| `$warpsPerCTA` | `ArrayRef<unsigned>` | 每 CTA 的 warp 数 |
| `$order` | `ArrayRef<unsigned>` | 维度顺序（最快变化轴在前） |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局（文本 IR 中可省略，默认单 CTA） |

```mlir
#ttg.blocked<{sizePerThread = {2, 2}, threadsPerWarp = {8, 4}, warpsPerCTA = {1, 2}, CTAsPerCGA = {1, 1}, CTASplitNum = {1, 1}}>
```

**示例**：`sizePerThread = {2, 2}`, `threadsPerWarp = {8, 4}`, `warpsPerCTA = {1, 2}` 的 16x16 张量在 2 个 warp（64 线程）上的分布：

```
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
```

### 3.2 LinearEncodingAttr

基于 LinearLayout 的通用布局编码。LinearLayout 是一种统一的布局表示方式，可表达所有其他分布式布局。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$linearLayout` | `LinearLayout` | 线性布局对象 |

```mlir
#ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  warp = [[16, 0], [32, 0]],
  block = []}>
```

额外方法：

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `getTotalElemsPerThread(shape)` | `unsigned` | 每线程总元素数 |
| `getElemsPerThread(shape)` | `SmallVector<unsigned>` | 每线程各维元素数 |
| `getContigPerThread()` | `SmallVector<unsigned>` | 每线程连续元素数 |
| `getContigPerWarp()` | `SmallVector<unsigned>` | 每 warp 连续元素数 |
| `getOrder()` | `SmallVector<unsigned>` | 维度顺序 |
| `getThreadsPerWarp()` | `SmallVector<unsigned>` | 每 warp 线程分布 |
| `getWarpsPerCTA()` | `SmallVector<unsigned>` | 每 CTA warp 分布 |

### 3.3 NvidiaMmaEncodingAttr

NVIDIA Tensor Core MMA 输出的布局编码。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$versionMajor` | `unsigned` | Tensor Core 代次（1=Volta, 2=Turing/Ampere） |
| `$versionMinor` | `unsigned` | 代次内的布局变体 |
| `$warpsPerCTA` | `ArrayRef<unsigned>` | 每 CTA 的 warp 分布 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |
| `$instrShape` | `ArrayRef<unsigned>` | 指令形状 |

版本对应关系：

| versionMajor | 架构 | 隐式 warpTileSize |
|--------------|------|-------------------|
| 1 | Volta (SM70) | [16, 16] |
| 2 | Turing/Ampere (SM75/80+) | [16, 8] |

额外方法：

| 方法 | 说明 |
|------|------|
| `isVolta()` | 是否为 Volta 架构 |
| `isTuring()` | 是否为 Turing 架构 |
| `isAmpere()` | 是否为 Ampere 架构 |
| `isHopper()` | 是否为 Hopper 架构 |
| `getRepForOperand(shape, bitwidth, kWidth, opIdx)` | 获取操作数的重复块 |
| `getRepOrderForOperand(opIdx)` | 获取操作数的重复顺序 |

### 3.4 AMDMfmaEncodingAttr

AMD MFMA (Matrix Fused Multiply-Add) 矩阵核心输出的布局编码，用于 CDNA 架构 GPU。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$version` | `unsigned` | GPU 架构版本（1=gfx908, 2=gfx90a, 3=gfx942, 4=gfx950） |
| `$warpsPerCTA` | `ArrayRef<unsigned>` | 每 CTA 的 warp 分布 |
| `$tilesPerWarp` | `ArrayRef<unsigned>` | 每 warp 的 MFMA tile 数 |
| `$MDim` | `unsigned` | MFMA 输出的 M 维度 |
| `$NDim` | `unsigned` | MFMA 输出的 N 维度 |
| `$isTransposed` | `bool` | 结果张量是否已转置 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |
| `$elementType` | `std::optional<Type>` | 元素类型（默认 f32） |

`tilesPerWarp` 参数允许每个 warp 计算连续的 MFMA tile，而非跨步分布。

### 3.5 AMDWmmaEncodingAttr

AMD WMMA (Warp Matrix Multiply-Accumulate) 矩阵核心输出的布局编码，用于 RDNA 架构 GPU。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$version` | `unsigned` | 指令版本（1=gfx11, 2=gfx12） |
| `$isTransposed` | `bool` | 是否转置 |
| `$warpsPerCTA` | `ArrayRef<unsigned>` | 每 CTA 的 warp 分布 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |

每个 warp 处理的 tile 形状固定为 [16, 16]。

### 3.6 SliceEncodingAttr

沿指定维度压缩父布局的切片布局。用于 `expand_dims` 的逆操作。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$dim` | `unsigned` | 被压缩的维度 |
| `$parent` | `DistributedEncodingTrait` | 父布局编码 |

**语义**：给定父布局 L_parent，沿 dim 压缩后：
- dim=0：L = {L_parent(0,0), L_parent(1,0), ...}，每个线程拥有 dim=0 上所有行的数据
- dim=1：L = {L_parent(0,0), L_parent(0,1), ...}，每个线程拥有 dim=1 上所有列的数据

额外方法：`paddedShape(shape)`：返回填充后的形状

### 3.7 DotOperandEncodingAttr

Dot 操作数布局编码。`tt.dot a, b, c` 中 a 和 b 必须使用此布局（MMA v1/v2），MMA v3 中操作数通常使用共享布局。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$opIdx` | `unsigned` | 操作数索引（0=A, 1=B） |
| `$parent` | `Attribute` | Dot 结果的布局编码 |
| `$kWidth` | `unsigned` | K 维度上每线程连续存储的元素数（默认 0） |

`kWidth` 的确定规则：
- 对于非 Ampere/Hopper 的 MMA，kWidth = 0
- 对于 Ampere/Hopper MMA，kWidth = max(32 / bitwidth, 1)

## 4. 共享布局编码

### 4.1 SwizzledSharedEncodingAttr

使用 XOR Swizzle 避免共享内存银行冲突的布局。通过 vec、perPhase、maxPhase 三个参数控制 Swizzle 模式。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$vec` | `unsigned` | 向量化宽度（元素对齐） |
| `$perPhase` | `unsigned` | 每相位的行数 |
| `$maxPhase` | `unsigned` | 最大相位值 |
| `$order` | `ArrayRef<unsigned>` | 维度顺序（最快变化轴在前） |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |

**Swizzle 公式**：元素 (r, c) 的 Swizzle 后列坐标为 `(c / vec) ^ (r / perPhase) % maxPhase`

**Swizzle 示例**：

1. 基本模式 `vec=1, perPhase=1, maxPhase=4`：
   ```
   [ 0,  1,  2,  3]   ; xor with 0
   [ 5,  4,  7,  6]   ; xor with 1
   [10, 11,  8,  9]   ; xor with 2
   [15, 14, 13, 12]   ; xor with 3
   ```

2. 多行相位 `vec=1, perPhase=2, maxPhase=4`：每 2 行使用相同 Swizzle

3. 最大相位限制 `vec=1, perPhase=1, maxPhase=2`：xor 值限制在 0-1

4. 向量化 `vec=2`：元素以 2 为单位进行 Swizzle

### 4.2 PaddedSharedEncodingAttr

使用填充避免共享内存银行冲突的布局。在指定间隔后插入填充元素。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$intervals` | `ArrayRef<unsigned>` | 填充间隔列表 |
| `$paddings` | `ArrayRef<unsigned>` | 填充大小列表 |
| `$order` | `ArrayRef<unsigned>` | 维度顺序 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |

**内存偏移公式**：对于 1D 张量中索引 i 的元素：
```
offset = i + Σ(i / interval_k) * pad_k
```

**示例**：

1. 单间隔填充 `[2:+2]`：
   ```
   [e0, e1, p0, p1, e2, e3, p2, p3, ...]
   ```

2. 双间隔填充 `[2:+1, 4:+2]`：
   ```
   [e0, e1, p0, e2, e3, p1, p2, p3, e4, e5, p4, ...]
   ```

额外方法：`getPaddedSize(shape)` 计算给定形状下包含填充的总元素数。

### 4.3 NVMMASharedEncodingAttr

NVIDIA MMAv3/MMAv5 共享内存输入布局，匹配 WGMMA 指令的共享内存格式。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$swizzlingByteWidth` | `unsigned` | Swizzle 字节宽度（0/32/64/128） |
| `$transposed` | `bool` | 连续维度是否为最外维 |
| `$elementBitWidth` | `unsigned` | 元素位宽 |
| `$fp4Padded` | `bool` | 是否为 FP4 混合精度的特殊填充布局 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |

Swizzle 字节宽度的确定规则：
- 连续维度字节 >= 128 且整除 128 → 128
- 连续维度字节 >= 64 且整除 64 → 64
- 连续维度字节 >= 32 且整除 32 → 32
- 否则 → 0（无 Swizzle）

额外方法：`getPerPhase()`, `getMaxPhase()`, `getVec()`

### 4.4 AMDRotatingSharedEncodingAttr

AMD 旋转 Swizzle 共享内存布局。与 `SwizzledSharedEncodingAttr` 类似，但 Swizzle 模式在每 `maxPhase*perPhase` 行后旋转变化，而非简单重复。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$vec` | `unsigned` | 向量化宽度 |
| `$perPhase` | `unsigned` | 每相位行数 |
| `$maxPhase` | `unsigned` | 最大相位值 |
| `$order` | `ArrayRef<unsigned>` | 维度顺序 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |

**旋转 Swizzle 公式**：
```
outRowId = inRowId
phase    = (inRowId / perPhase) % maxPhase
blockNo  = (inRowId / (perPhase * maxPhase)) % maxPhase
combinedPhase = phase ^ blockNo
outColId = inColId ^ combinedPhase
```

**示例** `vec=1, perPhase=1, maxPhase=2`：
```
row 0: [ 0,  1,  2,  3]   ; phase=0, blockNo=0 (xor 0)
row 1: [ 5,  4,  7,  6]   ; phase=1, blockNo=0 (xor 1)
row 2: [ 9,  8, 11, 10]   ; phase=0, blockNo=1 (xor 1)
row 3: [12, 13, 14, 15]   ; phase=1, blockNo=1 (xor 0)
```

### 4.5 FractalSharedEncodingAttr（昇腾适配）

**条件编译**：`BSPRIV_DAVINCI_BISHENGIR`

昇腾 Cube 仿真的自定义 Fractal 共享内存布局。

| 参数 | 类型 | 说明 |
|------|------|------|
| `$fractalM0` | `int64_t` | Fractal M0 块大小 |
| `$fractalN0` | `int64_t` | Fractal N0 块大小 |
| `$layoutType` | `FractalLayoutType` | Fractal 布局类型 |
| `$CTALayout` | `CTALayoutAttr` | CTA 布局 |

**FractalLayoutType 枚举**：

| 枚举值 | 整数值 | 字符串 | 说明 |
|--------|--------|--------|------|
| `zN` | 0 | `zN` | 内部 Z 形（列先行后），外部 N 形（M 块先 N 块后） |
| `nZ` | 1 | `nZ` | 内部 N 形（行先列后），外部 Z 形（N 块先 M 块后） |

此编码实现了 `SharedEncodingTrait` 和 `LayoutEncodingTrait` 接口。

## 5. 共享内存空间属性

### 5.1 SharedMemorySpaceAttr

标记内存描述符指向共享内存的属性。

| 项目 | 内容 |
|------|------|
| 助记符 | `shared_memory` |
| 说明 | 指示内存描述符指向共享内存 |

## 6. 布局编码选择策略

### 6.1 布局选择原则

| 场景 | 推荐布局 | 原因 |
|------|----------|------|
| 全局内存 Load/Store | `BlockedEncodingAttr` | 促进内存合并访问 |
| Dot 操作数（MMA v1/v2） | `DotOperandEncodingAttr` | 匹配 Tensor Core 输入格式 |
| Dot 结果 | `NvidiaMmaEncodingAttr` / `AMDMfmaEncodingAttr` | 匹配 Tensor Core 输出格式 |
| 共享内存中间存储 | `SwizzledSharedEncodingAttr` / `NVMMASharedEncodingAttr` | 避免银行冲突 |
| 昇腾 Cube 操作数 | `FractalSharedEncodingAttr` | 匹配 DaVinci Cube 矩阵格式 |

### 6.2 布局转换路径

典型的布局转换路径：

```
BlockedEncoding (Load)
    ↓ ttg.convert_layout
DotOperandEncoding (Dot 输入)
    ↓ tt.dot
NvidiaMmaEncoding (Dot 输出)
    ↓ ttg.convert_layout
BlockedEncoding (Store)
```

其中 `ttg.convert_layout` 的实现可能涉及：
1. 寄存器间直接重排（同一 warp 内）
2. 通过共享内存中转（跨 warp 通信）
3. 空操作（布局兼容时）

## 7. 编码参数速查表

| 编码 | 关键参数 | 适用硬件 |
|------|----------|----------|
| `BlockedEncodingAttr` | sizePerThread, threadsPerWarp, warpsPerCTA, order | 通用 |
| `LinearEncodingAttr` | linearLayout | 通用 |
| `NvidiaMmaEncodingAttr` | versionMajor, versionMinor, warpsPerCTA, instrShape | NVIDIA |
| `AMDMfmaEncodingAttr` | version, warpsPerCTA, MDim, NDim, tilesPerWarp | AMD CDNA |
| `AMDWmmaEncodingAttr` | version, isTransposed, warpsPerCTA | AMD RDNA |
| `SliceEncodingAttr` | dim, parent | 通用 |
| `DotOperandEncodingAttr` | opIdx, parent, kWidth | 通用 |
| `SwizzledSharedEncodingAttr` | vec, perPhase, maxPhase, order | 通用 |
| `PaddedSharedEncodingAttr` | intervals, paddings, order | 通用 |
| `NVMMASharedEncodingAttr` | swizzlingByteWidth, transposed, elementBitWidth | NVIDIA Hopper+ |
| `AMDRotatingSharedEncodingAttr` | vec, perPhase, maxPhase, order | AMD |
| `FractalSharedEncodingAttr` | fractalM0, fractalN0, layoutType | 昇腾 DaVinci |
