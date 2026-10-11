# HFusion→HIVM 转换规则

本文档详细描述 HFusion 方言到 HIVM 方言的转换 Pass。

源码参考：[Conversion/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Conversion/Passes.td)

## 1. 转换概述

HFusion→HIVM 转换是 BishengIR 编译流水线的关键步骤，将高级算子融合图转换为硬件指令映射图。此转换决定了算子如何映射到具体的硬件执行单元（Cube/Vector/DMA）。

```
┌─────────────────────────────────────────────────┐
│               HFusion Dialect                    │
│   融合后的命名算子 + linalg.generic               │
│   调度信息 + tiling 信息                          │
└──────────────────────┬──────────────────────────┘
                       │
                       │ ConvertHFusionToHIVM
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│               HIVM Dialect                       │
│   硬件指令映射（Cube/Vector/DMA）                  │
│   内存规划 + 同步 + 多缓冲                        │
└─────────────────────────────────────────────────┘
```

## 2. ConvertHFusionToHIVM

### 2.1 Pass 定义

| 项目 | 内容 |
|------|------|
| Pass 名称 | `ConvertHFusionToHIVM` |
| CLI 参数 | `convert-hfusion-to-hivm` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::createConvertHFusionToHIVMPass()` |
| 依赖方言 | `func::FuncDialect`, `hivm::HIVMDialect`, `memref::MemRefDialect`, `arith::ArithDialect`, `affine::AffineDialect`, `scf::SCFDialect`, `vector::VectorDialect`, `linalg::LinalgDialect` |

### 2.2 Pass 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mm-map-mode` | `MmMapMode` | `core_op` | 矩阵乘法映射模式 |

### 2.3 mm-map-mode 选项详解

`mm-map-mode` 控制矩阵乘法算子如何映射到 HIVM 指令：

| 模式 | 枚举值 | 说明 |
|------|--------|------|
| `core_op` | `CoreOp` | 映射为 HIVM Core 操作（Cube 单元），使用 Fractal 数据布局 |
| `macro_instr` | `MacroInstr` | 映射为 HIVM 宏指令，使用更高级的抽象 |

**core_op 模式**：
- 矩阵乘法映射为 Cube Core 操作
- 数据布局转换为 Fractal 格式（Z 形/N 形）
- 适用于大规模矩阵乘法

**macro_instr 模式**：
- 矩阵乘法映射为宏指令
- 提供更高级的抽象，允许编译器进行更激进的优化
- 适用于需要灵活调度的场景

## 3. 转换语义

### 3.1 算子映射规则

| HFusion 算子 | HIVM 映射 | 执行单元 |
|-------------|-----------|----------|
| 矩阵乘法 (core_op) | `hivm.cube_matmul` | Cube |
| 矩阵乘法 (macro_instr) | `hivm.macro_matmul` | Cube |
| 逐元素运算 | `hivm.vector_*` | Vector |
| 归约运算 | `hivm.vector_reduce` | Vector |
| 数据搬运 | `hivm.dma_copy` | DMA |
| 广播/转置 | `hivm.vector_broadcast/transposed_copy` | Vector/DMA |

### 3.2 内存布局转换

HFusion→HIVM 转换中，数据布局从 ND 格式转换为硬件特定的格式：

| 场景 | 输入布局 | 输出布局 | 说明 |
|------|----------|----------|------|
| Cube 矩阵乘法 A | ND (M×K) | Fractal (zN) | 行优先→Z 形 Fractal |
| Cube 矩阵乘法 B | ND (K×N) | Fractal (nZ) | 列优先→N 形 Fractal |
| Cube 矩阵乘法 C | Fractal (zN) | ND (M×N) | Z 形 Fractal→行优先 |
| Vector 运算 | ND | ND | 保持 ND 布局 |

### 3.3 同步操作插入

转换过程中自动插入同步操作：

1. **Cube→Vector 依赖**：插入 `hivm.sync` 确保 Cube 计算完成
2. **DMA→计算依赖**：插入 `hivm.sync` 确保数据搬运完成
3. **多缓冲切换**：插入 `hivm.sync` 确保前一个缓冲使用完毕

## 4. 辅助转换 Pass

### 4.1 ConvertHFusionToVector

将 HFusion 操作转换为 vector 方言操作（寄存器模式）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-hfusion-to-vector` |
| 构造函数 | `mlir::createConvertHFusionToVectorPass()` |
| 依赖方言 | `vector::VectorDialect`, `linalg::LinalgDialect`, `memref::MemRefDialect`, `arith::ArithDialect`, `scf::SCFDialect` |

此 Pass 用于 SIMT 路径，将 HFusion 算子直接转换为 vector 操作，跳过 HIVM 的 Cube 映射。

### 4.2 ConvertTensorToHIVM

将 tensor 操作直接转换为 HIVM 操作（跳过 HFusion 层）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-tensor-to-hivm` |
| 构造函数 | `mlir::createTensorToHIVMConversionPass()` |
| 依赖方言 | `tensor::TensorDialect`, `hivm::HIVMDialect` |

## 5. 转换后的 HIVM 结构

### 5.1 典型 HIVM 函数结构

```
func @kernel(...) {
  %buf_a = hivm.local_alloc ...     ; 分配 Cube 输入缓冲
  %buf_b = hivm.local_alloc ...     ; 分配 Cube 输入缓冲
  %buf_c = hivm.local_alloc ...     ; 分配 Cube 输出缓冲

  %dma_a = hivm.dma_copy %a -> %buf_a  ; DMA 搬运 A
  %dma_b = hivm.dma_copy %b -> %buf_b  ; DMA 搬运 B
  hivm.sync [%dma_a, %dma_b]           ; 等待 DMA 完成

  %cube = hivm.cube_matmul %buf_a, %buf_b -> %buf_c  ; Cube 矩阵乘
  hivm.sync [%cube]                     ; 等待 Cube 完成

  %vec = hivm.vector_add %buf_c, %bias ; Vector 逐元素加
  hivm.dma_copy %vec -> %output         ; DMA 写回
}
```

### 5.2 多缓冲模式

```
func @kernel_multibuf(...) {
  %buf_a0 = hivm.local_alloc ...    ; 缓冲 A 第 0 片
  %buf_a1 = hivm.local_alloc ...    ; 缓冲 A 第 1 片

  %dma0 = hivm.dma_copy %a0 -> %buf_a0  ; 搬运第 0 片
  hivm.sync [%dma0]

  %cube0 = hivm.cube_matmul %buf_a0, %buf_b -> %buf_c0
  %dma1 = hivm.dma_copy %a1 -> %buf_a1  ; 搬运第 1 片（与计算重叠）
  hivm.sync [%cube0, %dma1]

  %cube1 = hivm.cube_matmul %buf_a1, %buf_b -> %buf_c1
  ...
}
```

## 6. 与后续 Pass 的衔接

ConvertHFusionToHIVM 完成后，进入 HIVM 变换阶段：

1. **HIVM Tensor 优化**：布局推断、内存规划、同步优化
2. **HIVM Lowering**：HIVM→TritonGPU（SIMD）或 HIVM→LLVM（SIMT）
3. **后端代码生成**：LLVM→二进制

详细内容参见：
- [04-hivm-transforms.md](04-hivm-transforms.md) — HIVM 变换 Pass
- [05-hivm-to-backend.md](05-hivm-to-backend.md) — HIVM→后端转换
