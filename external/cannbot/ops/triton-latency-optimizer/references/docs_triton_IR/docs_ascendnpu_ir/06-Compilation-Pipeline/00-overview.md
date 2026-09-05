# IR 方言间转换总览

本文档概述 AscendNPU-IR 项目中 IR 方言间的转换数据流，包括前端到后端的三条编译路径。

## 1. 编译数据流全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend Input                                │
│   Triton Python / Torch / Linalg / Arith / Math / GPU / Tensor     │
└────────────┬──────────────────────────────┬────────────────────────┘
             │                              │
             │  Triton 路径                  │  BishengIR 路径
             ▼                              ▼
┌────────────────────────┐    ┌────────────────────────────────────────┐
│   Triton IR (tt)       │    │   HFusion Dialect                      │
│   设备无关张量计算       │    │   算子融合与调度                        │
└───────────┬────────────┘    └──────────────┬─────────────────────────┘
            │                                │
            │ TritonToTritonGPU              │ HFusion Transforms
            │ Conversion                     │ (40+ Pass)
            ▼                                ▼
┌────────────────────────┐    ┌────────────────────────────────────────┐
│   TritonGPU IR (ttg)   │    │   HIVM Dialect                         │
│   GPU 布局编码           │    │   硬件指令映射                          │
└───────────┬────────────┘    └──────────────┬─────────────────────────┘
            │                                │
            │ TritonGPUToLLVM                │ HIVM Transforms (70+ Pass)
            │ Conversion                     │
            ▼                                ▼
┌────────────────────────┐    ┌────────────────────────────────────────┐
│   LLVM IR              │    │   HIVM → 后端转换                       │
│   NVIDIA GPU 代码生成   │    │   ├─ SIMD: HIVM→TritonGPU→LLVM         │
└────────────────────────┘    │   ├─ SIMT: HIVM→LLVM (DPX)             │
                              │   └─ MIX:  Cube+Vector 分裂             │
                              └────────────────────────────────────────┘
```

## 2. 三条编译路径

### 2.1 SIMD 路径（Cube 矩阵运算）

```
Frontend → HFusion → HIVM → TritonGPU → LLVM
```

- **适用场景**：矩阵乘法等 Cube 密集型运算
- **关键转换**：`ConvertHIVMToTritonGPU`
- **布局编码**：使用 `FractalSharedEncodingAttr` 适配昇腾 Cube 格式
- **特点**：利用 TritonGPU 的布局编码系统表达 Cube 矩阵运算

### 2.2 SIMT 路径（Vector 向量运算）

```
Frontend → HFusion → HIVM → LLVM (via DPX)
```

- **适用场景**：逐元素运算、归约等 Vector 密集型运算
- **关键转换**：`ConvertTritonAscendGPUToLLVM`、`GPUToDPX`
- **特点**：通过 Ascend DPX 方言将 GPU 操作降低到昇腾向量指令

### 2.3 MIX 路径（Cube + Vector 混合）

```
Frontend → HFusion → HIVM → SplitMixKernel → {Cube Kernel, Vector Kernel}
```

- **适用场景**：同时包含 Cube 和 Vector 运算的混合 kernel
- **关键 Pass**：`SplitMixKernel` 将混合函数分裂为独立的 Cube 和 Vector 函数
- **特点**：Cube 和 Vector 函数分别编译，通过共享 workspace 传递数据

## 3. 方言转换 Pass 一览

### 3.1 前端 → HFusion 转换

| Pass 名称 | CLI 参数 | 源操作 | 目标操作 |
|-----------|----------|--------|----------|
| `ConvertArithToHFusion` | `convert-arith-to-hfusion` | `arith` | `hfusion` |
| `ConvertMathToHFusion` | `convert-math-to-hfusion` | `math` | `hfusion` |
| `ConvertLinalgToHFusion` | `convert-linalg-to-hfusion` | `linalg` | `hfusion` |
| `ConvertGPUToHFusion` | `convert-gpu-to-hfusion` | `gpu` | `hfusion` |
| `ConvertTensorToHFusion` | `convert-tensor-to-hfusion` | `tensor` | `hfusion` |
| `ConvertTorchToHFusion` | `convert-torch-to-hfusion` | `torch` | `hfusion` |

源码参考：[Conversion/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Conversion/Passes.td)

### 3.2 HFusion → HIVM 转换

| Pass 名称 | CLI 参数 | 关键选项 |
|-----------|----------|----------|
| `ConvertHFusionToHIVM` | `convert-hfusion-to-hivm` | `mm-map-mode`: `core_op` / `macro_instr` |

### 3.3 HIVM → 后端转换

| Pass 名称 | CLI 参数 | 说明 |
|-----------|----------|------|
| `ConvertHIVMToTritonGPU` | `convert-hivm-to-tritongpu` | HIVM → TritonGPU (SIMD 路径) |
| `ArithToHIVMLLVMConversionPass` | `convert-arith-to-hivm-llvm` | arith → LLVM (双缓冲支持) |
| `ConvertTritonAscendGPUToLLVM` | `convert-triton-ascend-gpu-to-llvm` | TritonGPU → LLVM (昇腾) |
| `LowerMesh` | `lower-mesh` | Mesh → HCCL/LCCL |
| `LowerMemRefExt` | `lower-memref-ext` | memref_ext → memref |
| `ConvertHFusionToVector` | `convert-hfusion-to-vector` | HFusion → vector (寄存器模式) |

### 3.4 Triton IR 编译路径

| Pass 名称 | CLI 参数 | 说明 |
|-----------|----------|------|
| `ConvertTritonToTritonGPU` | `convert-triton-to-tritongpu` | tt → ttg |
| `AllocateSharedMemory` | `allocate-shared-memory` | 共享内存分配 |
| `TritonGPUGlobalScratchAllocationPass` | `tritongpu-global-scratch-memory-allocation` | 全局 scratch 分配 |
| `TritonGPUAllocateWarpGroups` | `tritongpu-allocate-warp-groups` | Warp 组分配 |

## 4. Pipeline 入口

### 4.1 HFusion Pipeline

```cpp
void buildHFusionPipelines(OpPassManager &pm,
                           const HFusionPipelineOptions &options);
```

源码参考：[HFusion/Pipelines/Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/Pipelines/Passes.h)

### 4.2 HIVM Pipeline

```cpp
void buildConvertToHIVMPipeline(mlir::OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options);

void buildHIVMTensorOptimizations(
    OpPassManager &pm, const HIVMPipelineOptions &hivmPipelineOptions);

void buildLowerHIVMPipelines(OpPassManager &pm,
                             const HIVMPipelineOptions &hivmPipelineOptions);
```

源码参考：[HIVM/Pipelines/Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Pipelines/Passes.h)

## 5. 转换依赖关系

```
ConvertArithToHFusion ──┐
ConvertMathToHFusion ───┤
ConvertLinalgToHFusion ─┤──→ HFusionOpFusion ──→ AutoSchedule ──→ AutoVectorize
ConvertGPUToHFusion ────┤
ConvertTensorToHFusion ─┘
                                                    │
                                                    ▼
                                         ConvertHFusionToHIVM
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │               │               │
                              SIMD 路径        SIMT 路径        MIX 路径
                                    │               │               │
                          ConvertHIVMTo     ArithToHIVM      SplitMixKernel
                          TritonGPU        LLVMConversion    │
                                    │               │       ┌──────┴──────┐
                                    ▼               ▼       │             │
                          TritonGPU Passes    GPUToDPX   Cube 部分   Vector 部分
                                    │               │       │             │
                                    ▼               ▼       ▼             ▼
                          ConvertTritonAs-    LLVM IR   SIMD 路径    SIMT 路径
                          cendGPUToLLVM
                                    │
                                    ▼
                                 LLVM IR
```

## 6. 关键设计决策

### 6.1 为什么需要 HFusion 层

HFusion 方言作为前端和 HIVM 之间的中间层，提供了：

1. **算子融合**：将多个小算子融合为一个大算子，减少内存带宽需求
2. **自动调度**：根据硬件特性自动选择最优的执行策略
3. **自动向量化**：将算子转换为向量指令，提高计算效率
4. **多 kernel 支持**：允许将计算图分裂为多个 kernel

### 6.2 为什么需要 HIVM 层

HIVM 方言作为硬件指令映射层，提供了：

1. **硬件抽象**：将高级算子映射到具体的硬件指令（Cube/Vector/DMA）
2. **内存管理**：规划内存分配、多缓冲和同步
3. **数据布局**：推断和优化数据布局（Fractal/ND 等）
4. **同步注入**：自动插入同步操作，确保数据一致性

### 6.3 Triton 路径与 BishengIR 路径的关系

- **Triton 路径**：直接从 Triton IR 编译到 LLVM IR，用于标准 Triton kernel
- **BishengIR 路径**：通过 HFusion → HIVM 编译，用于昇腾 NPU 优化的 kernel
- **交汇点**：SIMD 路径中 HIVM 可转换为 TritonGPU，复用 Triton 的布局编码系统
