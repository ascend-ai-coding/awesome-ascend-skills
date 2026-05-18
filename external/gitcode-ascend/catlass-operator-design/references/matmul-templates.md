# Catlass 矩阵乘法模板清单

## 模板速查表

| 模板名 | 类型 | 核心特性 | 适用场景 |
|--------|------|---------|---------|
| 00_basic_matmul | Common | 流水优化(Multi Buffer) | 通用场景，性能基线 |
| 04_padding_matmul | Common | Multi Buffer + Padding(ND) | Stride 非 512B 对齐 |
| 06_optimized_matmul | Common | Multi Buffer + Preload + Padding(NZ) + ShuffleK + 小M指令替换 | 泛化性强，复杂场景 |
| 09_splitk_matmul | MultiCoreSplitK | Multi Buffer | 基本任务块少、K轴大 |
| 21_basic_matmul_preload_zN | Common | Multi Buffer + Preload + ShuffleK | zN格式，中等复杂度 |
| 22_padding_splitk_matmul | MultiCoreSplitK | Multi Buffer + Padding(ND) | SplitK + 非对齐Stride |
| 25_matmul_full_loadA | Common | Multi Buffer + L1常驻 | A矩阵可全载入L1 |
| 31_small_matmul | Common | Multi Buffer + Scalar消减 | 小Shape，基本任务块≤AIC数 |
| 34_single_core_splitk_matmul | SingleCoreSplitK | Multi Buffer + Padding(NZ) + 写出优化 | 单核切K场景 |

## 理论模板

| 模板 | 分核策略 | 特点 |
|------|---------|------|
| Common | M、N 方向分核 | 标准分块，每个任务块搬运 `m₁K + Kn₁` |
| MultiCoreSplitK | M、N + K 方向分核 | 更易负载均衡，但有 ReduceAdd 开销 |
| SingleCoreSplitK | 大块 m₁n₁ + K切分 | 减少读取，atomicAdd GM 累加 |

## 工程优化手段

| 优化 | 作用 | 对应模板 |
|------|------|---------|
| Multi Buffer | L1/L0 多 buffer 流水并行 | 所有模板 |
| Preload | GM→L1 与计算重叠，减少 MTE2 空泡 | 06, 21 |
| Padding (ND/NZ/BlockND) | 解决 Stride 非对齐、ND2NZ 带宽损失 | 04, 06, 22, 34 |
| ShuffleK | CoreIdx 偏移起始序号，避免同地址冲突 | 06, 21 |
| 小M指令替换 | M<8 时逐行搬运 | 06 |
| L1常驻 | tile 块常驻 L1，减少重复读取 | 25, 34 |
| Scalar消减 | 小 Shape 消减冗余标量计算 | 31 |
| 写出优化 | 解决 dstStride 对齐、NZ2ND 转换损失 | 34 |

## 模板选择指南

1. **先用 00_basic_matmul 建立性能基线**，调 TileShape 后测性能
2. 按场景选择：
   - 小 Shape（任务块 ≤ AIC 数、K 较小）→ **31_small_matmul**
   - 任务块少且 K 大 → **09_splitk_matmul** 或 **22_padding_splitk_matmul**
   - 泛化场景 → **06_optimized_matmul** 或 **21_basic_matmul_preload_zN**
   - A 可 L1 常驻 → **25_matmul_full_loadA**
3. Stride 非 512B 对齐时考虑 Padding 前处理，但注意 Padding 开销

## DispatchPolicy 速查

| DispatchPolicy | 对应优化 |
|---------------|---------|
| `MmadAtlasA2Pingpong` | Multi Buffer（默认） |
| `MmadAtlasA2Preload` | Multi Buffer + Preload |
| `MmadAtlasA2Small` | 小 Shape 优化 |
| `MmadAtlasA2SingleCoreSplitk` | 单核切K |
| `MmadAtlasA2FullLoadA` | L1 全载 |

## 工程化模板参考

`catlass/examples/advanced/basic_matmul_aclnn` 是算子工程结构的**核心参考**，包含：

| 文件 | 路径 | 用途 |
|------|------|------|
| tiling.h | `advanced/basic_matmul_aclnn/op_host/catlass_basic_matmul_tiling.h` | Tiling 数据结构（宏定义方式） |
| def+infershape+tiling | `advanced/basic_matmul_aclnn/op_host/catlass_basic_matmul.cpp` | OpDef 注册、InferShape、TilingFunc 合一 |
| op_kernel | `advanced/basic_matmul_aclnn/op_kernel/catlass_basic_matmul.cpp` | Device 调用：`CatlassBasicMatmulTemplate` 封装 Kernel，`TILING_KEY_IS` 分支实例化 |
| test_aclnn | `advanced/basic_matmul_aclnn/basic_matmul_aclnn.cpp` | aclnn 两段式调用示例 |

该示例展示了 00_basic_matmul 从 standalone example 到 op_host/op_kernel 分离工程的**适配方式**。

## 详细参考

本文件为速查表，每个模板的详细理论模板、工程优化点、约束说明请参考 catlass 官方文档：
- `catlass/docs/2_Design/01_kernel_design/04_matmul_summary.md` — 全量模板清单与详细分析（500+ 行）
- `catlass/docs/2_Design/01_kernel_design/03_dispatch_policies.md` — DispatchPolicy 参数详解
- `catlass/docs/2_Design/01_kernel_design/02_swizzle.md` — Swizzle 策略图解
