# HIVM→后端转换

本文档详细描述 HIVM 方言到后端（LLVM IR / TritonGPU）的转换 Pass。

源码参考：[Conversion/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Conversion/Passes.td)

## 1. 转换路径总览

HIVM 到后端有三条转换路径，对应三种执行模式：

```
                    ┌─────────────────────────┐
                    │      HIVM Dialect        │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         SIMD 路径      SIMT 路径      MIX 路径
              │              │              │
    ConvertHIVMTo    ArithToHIVM    SplitMixKernel
    TritonGPU        LLVMConv.           │
              │              │      ┌─────┴─────┐
              ▼              ▼      │           │
    TritonGPU Passes  GPUToDPX   Cube 部分  Vector 部分
              │              │      │           │
              ▼              ▼      ▼           ▼
    ConvertTritonAs-  LLVM IR   SIMD 路径  SIMT 路径
    cendGPUToLLVM
              │
              ▼
           LLVM IR
```

## 2. SIMD 路径：HIVM → TritonGPU → LLVM

### 2.1 ConvertHIVMToTritonGPU

将 HIVM 操作转换为 TritonGPU 操作，复用 Triton 的布局编码系统。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `ConvertHIVMToTritonGPU` |
| CLI 参数 | `convert-hivm-to-tritongpu` |
| 构造函数 | `mlir::createConvertHIVMToTritonGPUPass()` |
| 依赖方言 | `triton::gpu::TritonGPUDialect`, `triton::TritonDialect`, `arith::ArithDialect`, `scf::SCFDialect`, `tensor::TensorDialect`, `memref::MemRefDialect`, `hivm::HIVMDialect` |

**转换语义**：

| HIVM 操作 | TritonGPU 操作 | 说明 |
|-----------|---------------|------|
| `hivm.cube_matmul` | `tt.dot` + `ttg.convert_layout` | Cube 矩阵乘法映射为 Triton dot |
| `hivm.dma_copy` | `ttg.async_copy_global_to_local` | DMA 搬运映射为异步拷贝 |
| `hivm.vector_*` | `ttg.local_load/store` + arith | Vector 操作映射为共享内存操作 |
| `hivm.sync` | `ttg.async_wait` | 同步映射为异步等待 |

**布局编码映射**：

| HIVM 布局 | TritonGPU 布局 | 说明 |
|-----------|---------------|------|
| Fractal (zN/nZ) | `FractalSharedEncodingAttr` | 昇腾 Cube 格式 |
| ND | `BlockedEncodingAttr` | 标准 ND 格式 |

### 2.2 ConvertTritonAscendGPUToLLVM

将 TritonGPU 操作通过昇腾适配路径降低到 LLVM IR。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `ConvertTritonAscendGPUToLLVM` |
| CLI 参数 | `convert-triton-ascend-gpu-to-llvm` |
| 构造函数 | `mlir::createConvertTritonAscendGPUToLLVMPass()` |

此 Pass 是标准 `ConvertTritonGPUToLLVM` 的昇腾适配版本，处理 TritonGPU 操作到昇腾 LLVM 内联汇编的映射。

## 3. SIMT 路径：HIVM → LLVM (via DPX)

### 3.1 ArithToHIVMLLVMConversionPass

将 arith 操作转换为 HIVM LLVM 操作，支持双缓冲。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `ArithToHIVMLLVMConversionPass` |
| CLI 参数 | `convert-arith-to-hivm-llvm` |
| 构造函数 | `mlir::createArithToHIVMLLVMConversionPass()` |
| 依赖方言 | `arith::ArithDialect`, `LLVM::LLVMDialect` |

### 3.2 GPUToDPX

将 GPU 操作转换为 DPX（Data Processing eXtension）操作，用于昇腾 Vector 单元。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `GPUToDPX` |
| CLI 参数 | `convert-gpu-to-dpx` |
| 构造函数 | `mlir::createGPUToDPXPass()` |
| 依赖方言 | `gpu::GPUDialect`, `dpx::DPXDialect` |

### 3.3 FuncToLLVM

函数到 LLVM 的转换。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `FuncToLLVM` |
| CLI 参数 | `convert-func-to-llvm` |
| 构造函数 | `mlir::createFuncToLLVMConversionPass()` |

## 4. MIX 路径：Cube + Vector 分裂

### 4.1 SplitMixKernel

将 Mix kernel 分裂为独立的 AICube 和 AIVector 函数。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `hivm-split-mix-kernel` |
| 作用域 | `ModuleOp` |
| 构造函数 | `mlir::hivm::createSplitMixKernelPass()` |

**分裂语义**：

输入 Mix kernel：
```mlir
func @mix_kernel(workspace) {tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} {
  %t = cube_op ins() outs(workspace)
  %r = vector_op ins(%t) ...
}
```

输出分裂后：
```mlir
func @mix_kernel_cube(workspace) {tcore_type = #hivm.tcore_type<CUBE>} {
  %t = cube_op ins() outs(workspace)
  annotation.mark %t  ; 标记避免 DCE
}

func @mix_kernel_vector(workspace) {tcore_type = #hivm.tcore_type<VECTOR>} {
  %r = vector_op ins(workspace) ...
}
```

分裂后的 Cube 和 Vector 函数分别通过 SIMD 和 SIMT 路径编译。

## 5. 辅助转换 Pass

### 5.1 LowerMesh

将 Mesh 操作降低为 HCCL/LCCL 通信操作。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `LowerMesh` |
| CLI 参数 | `lower-mesh` |
| 构造函数 | `mlir::createLowerMeshPass()` |

### 5.2 LowerMemRefExt

将 memref_ext 操作降低为标准 memref 操作。

| 项目 | 内容 |
|------|------|
| Pass 名称 | `LowerMemRefExt` |
| CLI 参数 | `lower-memref-ext` |
| 构造函数 | `mlir::createLowerMemRefExtPass()` |
| 依赖方言 | `memref::MemRefDialect`, `memref_ext::MemRefExtDialect` |

### 5.3 ConvertHFusionToVector

将 HFusion 操作直接转换为 vector 方言操作（跳过 HIVM 层的 SIMT 路径）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-hfusion-to-vector` |
| 构造函数 | `mlir::createConvertHFusionToVectorPass()` |

## 6. 转换后的 LLVM IR 结构

### 6.1 SIMD 路径输出

SIMD 路径最终生成的 LLVM IR 包含：
- Cube 矩阵乘法内联汇编
- 共享内存管理代码
- 布局转换代码（Fractal ↔ ND）
- 同步操作

### 6.2 SIMT 路径输出

SIMT 路径最终生成的 LLVM IR 包含：
- Vector 向量指令内联汇编
- 标量操作代码
- 内存访问代码
- DPX 指令

### 6.3 MIX 路径输出

MIX 路径生成两个独立的 kernel：
- **Cube kernel**：通过 SIMD 路径编译
- **Vector kernel**：通过 SIMT 路径编译
- 通过共享 workspace 传递数据

## 7. 转换依赖关系

```
ConvertHIVMToTritonGPU
    │
    ├── TritonGPU Transforms (coalesce, pipeline, etc.)
    │
    ├── AllocateSharedMemory
    │
    └── ConvertTritonAscendGPUToLLVM
            │
            └── LLVM IR

ArithToHIVMLLVMConversionPass
    │
    ├── GPUToDPX
    │
    ├── FuncToLLVM
    │
    └── LLVM IR

SplitMixKernel
    │
    ├── Cube 函数 → ConvertHIVMToTritonGPU → ...
    │
    └── Vector 函数 → ArithToHIVMLLVMConversionPass → GPUToDPX → ...
```
