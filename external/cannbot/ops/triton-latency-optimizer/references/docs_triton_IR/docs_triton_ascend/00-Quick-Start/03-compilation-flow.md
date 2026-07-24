# 03 - 编译流程全景

## 概述

本文档全面介绍 Triton-Ascend 的编译流水线，从 Python 源码到 NPU 设备二进制的完整转换过程。Triton-Ascend 的核心编译路径为 `Python AST → TTIR → TTIR 优化 → Ascend Passes → Linalg IR → BiSheng Compiler → 设备二进制`，同时支持一条间接路径 `TTIR → HFusion → HIVM → LLVM` 用于特定算子优化。理解编译流程对于调试编译错误、优化 kernel 性能和扩展编译器功能至关重要。

**关键词**：编译流水线、TTIR、Linalg IR、Ascend Passes、AutoBlockify、TritonToStructured、TritonToLinalg、HFusion、HIVM、BiSheng Compiler、编译缓存

---

## 关键概念

| 概念 | 说明 |
|------|------|
| TTIR | Triton Tensor IR，Triton 的高层中间表示，基于 `tt` 方言 |
| Linalg IR | Linear Algebra IR，MLIR 的线性代数方言，是 Ascend 编译路径的核心 IR |
| Ascend Passes | Triton-Ascend 特有的 MLIR Pass 集合，负责 TTIR 到 Linalg IR 的转换和优化 |
| HFusion | 昇腾硬件加速器融合方言，用于将算子映射到 NPU 专用硬件 |
| HIVM | 昇腾异构指令虚拟机方言，管理多核流水线中的同步与数据依赖 |
| BiSheng Compiler | 毕昇编译器，将 Linalg IR / HFusion IR 编译为 NPU 可执行二进制 |
| bishengir-compile | BiSheng Compiler 的命令行工具，负责最终的 NPU 二进制生成 |
| mlir-opt | MLIR 优化工具，用于 Linalg IR 到 LLVM IR 的 lowering |

---

## 完整编译流水线图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Triton-Ascend 编译流水线                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Python Kernel Source (@triton.jit)                                     │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐                                                       │
│  │ Python AST    │  Python 解析器生成 AST                                 │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │ TTIR 生成     │  Triton 编译器前端将 AST 转换为 TTIR                    │
│  │ (tt 方言)     │                                                       │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────┐                                                       │
│  │ TTIR 优化     │  Inliner, Combine, Canonicalizer, CSE, LICM,          │
│  │ (make_ttir)   │  ReorderBroadcast, LoopUnroll                         │
│  └──────┬───────┘                                                       │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Ascend Passes (ttir_to_linalg)                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ AutoBlockify     │  自动将逻辑核数映射到物理核数              │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐  (可选) Auto Scheduling                  │       │
│  │  │ DAG Sync/Scope/ │  多核流水线同步与缓冲区分配                │       │
│  │  │ SSBuffer        │                                         │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToStructured│  指针线性化、mask 规范化                 │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌──────────────────────┐                                    │       │
│  │  │ DiscreteMaskAccess   │  离散掩码访存转换                    │       │
│  │  │ Conversion           │                                    │       │
│  │  └────────┬─────────────┘                                    │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToAnnotation│  编译提示指令转换                       │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToUnstructured│  离散轴展开为标量循环                  │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToHIVM     │  块同步操作转换                          │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToHFusion  │  硬件加速器融合转换                      │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToLLVM     │  内联汇编转换                            │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ BubbleUpOperation│  Extract Op 上移优化                     │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToStructured│  (二次) 指针线性化                       │       │
│  │  └────────┬────────┘                                         │       │
│  │           ▼                                                  │       │
│  │  ┌─────────────────┐                                         │       │
│  │  │ TritonToLinalg   │  TTIR → Linalg IR 核心转换               │       │
│  │  └────────┬────────┘                                         │       │
│  └──────────┼───────────────────────────────────────────────────┘       │
│             ▼                                                           │
│  ┌──────────────┐                                                       │
│  │ Linalg IR     │  包含 linalg, memref, arith, scf 等方言              │
│  │ (ttadapter)   │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ├─── 路径 A: A2/A3 系列 (linalg_to_bin_A2_A3) ──→ BiSheng      │
│         │    bishengir-compile --enable-hivm-compile=true               │
│         │    --enable-hfusion-compile=true                               │
│         │    --enable-triton-kernel-compile=true                         │
│         │                                                               │
│         ├─── 路径 B: 910_95 系列 (linalg_to_bin_910_95) ──→ BiSheng    │
│         │    bishengir-compile --enable-hfusion-compile=true             │
│         │    --enable-triton-kernel-compile=true                         │
│         │                                                               │
│         └─── 路径 C: SIMT Only (ttir_to_npubin) ──→ BiSheng            │
│              bishengir-compile --enable-triton-ir-compile                │
│              --pure-simt                                                │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │ NPU Binary    │  triton_xxx_kernel.o (设备侧可执行文件)               │
│  └──────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 每个阶段的详细说明

### 阶段 1：Python AST → TTIR

当调用 `add_kernel[grid](...)` 时，Triton 的 JIT 编译器将 `@triton.jit` 装饰的 Python 函数解析为 AST，然后转换为 Triton Tensor IR (TTIR)。

**TTIR 示例**（向量加法）：

```mlir
tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                           %arg2: !tt.ptr<f32>, %arg3: i32) {
  %0 = tt.get_program_id %arg3 : i32
  %1 = arith.muli %0, %c1024 : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %3 = arith.addi %1, %2 : tensor<1024xi32>
  %4 = arith.cmpi slt, %3, %arg3 : tensor<1024xi1>
  %5 = tt.load %arg0, %3, %4 : tensor<1024xf32>
  %6 = tt.load %arg1, %3, %4 : tensor<1024xf32>
  %7 = arith.addf %5, %6 : tensor<1024xf32>
  tt.store %arg2, %3, %7, %4 : tensor<1024xf32>
}
```

### 阶段 2：TTIR 优化 (make_ttir)

TTIR 生成后，经过一系列标准 MLIR 优化 Pass：

| Pass | 功能 |
|------|------|
| `add_inliner` | 内联函数调用 |
| `add_combine` | 合并冗余的 Triton 操作 |
| `add_canonicalizer` | 标准化 IR 表示 |
| `add_reorder_broadcast` | 重排广播操作以优化性能 |
| `add_cse` | 公共子表达式消除 |
| `add_licm` | 循环不变量外提 |
| `add_symbol_dce` | 死符号消除 |
| `add_loop_unroll` | 循环展开 |

源码参考：[compiler.py - make_ttir](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L72-L93)

### 阶段 3：Ascend Passes (ttir_to_linalg)

这是 Triton-Ascend 编译流程的核心阶段，将 TTIR 转换为 Linalg IR。Pass 执行顺序如下：

源码参考：[compiler.py - ttir_to_linalg](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L95-L168)

#### 3.1 AutoBlockify

```
ascend.passes.ttir.add_auto_blockify(pm, auto_blockify_size)
```

当环境变量 `TRITON_ALL_BLOCKS_PARALLEL` 设置时，AutoBlockify Pass 会自动将逻辑核数调整为物理核数，减少调度开销。`auto_blockify_size` 参数控制调整粒度，默认为 1（不调整）。

#### 3.2 Auto Scheduling（可选）

当 `add_auto_scheduling` 选项启用时，依次执行：

| Pass | 功能 |
|------|------|
| `add_dag_sync` | 分析数据依赖关系图，插入同步指令 |
| `add_dag_scope` | 划分同步作用域 |
| `add_dag_ssbuffer` | 分配同步缓冲区 |

这三个 Pass 实现了多核流水线的自动调度，用于存算并行优化。

#### 3.3 TritonToStructured

```
ascend.passes.ttir.add_triton_to_structure(pm, enable_mask_fallback_conversion, optimize_dynamic_offset)
```

处理指针表达式和 mask 表达式中的整除取余，通过升维方法去除整除取余后重新生成 load/store 等 OP。

**核心 Converter 列表**：

| Converter | 功能 |
|-----------|------|
| RewriteAddPtrOp | 分析指针表达式，将偏移计算分解为 `PtrState` 对象 |
| CreateAddpr | 根据 `PtrState` 重新构造指针计算，消除整除和取模 |
| RewriteLoadOp | 分析 load 操作的 mask 表达式，分解为 `MaskState` |
| BuildMask | 根据 `MaskState` 重新构造 mask，消除整除和取模 |
| CreateLoad | 使用新指针和新 mask 重新创建 load 操作 |
| RewriteStoreOp | 分析 store 操作的 mask 表达式 |
| CreateStore | 使用新指针和新 mask 重新创建 store 操作 |
| RewriteAtomicRWMOp | 处理原子读写修改操作的指针问题 |
| RewriteAtomicCASOp | 处理原子比较并交换操作的指针线性化 |
| RewriteWhile | 处理 while 循环体内的指针叠加操作 |
| RewriteFor | 处理 for 循环体内的指针叠加操作 |

#### 3.4 DiscreteMaskAccessConversion

```
ascend.passes.ttir.add_discrete_mask_access_conversion(pm, compile_on_910_95, force_simt_template)
```

将基于离散索引掩码的内存访问模式进行分析与转换，为后续将离散轴展开为循环做准备。

| Converter | 功能 |
|-----------|------|
| DiscreteMaskStoreConversion | 非连续 mask 的 store 转化为 load→select→store 序列 |
| DiscreteMaskLoadConversion | 非连续 mask 的 load 转化为 load→select 序列 |
| DiscreteMaskAtomicAddConversion | 非连续 mask 的 atomic_add 转化为 select→atomic_add 序列 |

#### 3.5 TritonToAnnotation

```
ascend.passes.ttir.add_triton_to_annotation(pm)
```

处理 Ascend NPU 特有的编译提示指令 (`al.compile_hint`)，将其转换为后端的 Annotation 方言，用于指导后续优化。

#### 3.6 TritonToUnstructured

```
ascend.passes.ttir.add_triton_to_unstructure(pm, compile_on_910_95, force_simt_template)
```

将包含离散轴的张量操作转换为基于显式标量循环的标量访存。

| Converter | 功能 |
|-----------|------|
| UnstructuredMemAccessConverter\<LoadOp\> | 将 LoadOp 转化为多重循环标量加载 |
| UnstructuredMemAccessConverter\<StoreOp\> | 将 StoreOp 转化为多重循环标量存储 |
| UnstructuredMemAccessConverter\<AtomicRMWOp\> | 将 AtomicRMWOp 转化为多重循环标量 Atomic 操作 |
| UnstructuredMemAccessConverter\<AtomicCASOp\> | 将 AtomicCASOp 转化为多重循环标量 Atomic 操作 |

#### 3.7 TritonToHIVM

```
ascend.passes.ttir.add_triton_to_hivm(pm)
```

处理 Triton 的块同步操作，将其转换为 HIVM 方言中的跨核心同步指令：

| Triton 操作 | HIVM 指令 | 说明 |
|------------|----------|------|
| `al.sync_block_all` | HIVM 全局块同步 | 向所有接收核广播事件信号 |
| `al.sync_block_set` | HIVM 同步设置 | 发送核发出事件信号 |
| `al.sync_block_wait` | HIVM 同步等待 | 接收核等待事件信号 |

#### 3.8 TritonToHFusion

```
ascend.passes.ttir.add_triton_to_hfusion(pm)
```

将 TTIR 转换为 Ascend NPU 硬件加速器 HFusion 方言。例如将 `triton::HistogramOp` 转换为 `hfusion::HistogramOp`，使能在 NPU 专用硬件上高效执行。

#### 3.9 TritonToLLVM

```
ascend.passes.ttir.add_triton_to_llvm(pm)
```

将 Triton 中的内联汇编操作 (`tl.inline_assembly`) 转换为 LLVM 方言的内联汇编，最终映射为 Ascend NPU 的 CCE 硬件固有函数（Intrinsics）。

#### 3.10 BubbleUpOperation

```
ascend.passes.ttir.add_bubble_up_operation(pm)
```

对 `extract op / extract_slice` 进行顺序上移优化，优化数据局部性，某些场景能消除不必要的循环。

#### 3.11 TritonToStructured（二次执行）

在 TritonToLinalg 之前再次执行 TritonToStructured，处理前面 Pass 可能引入的新指针表达式。

#### 3.12 TritonToLinalg（核心转换）

```
ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops, enable_nd2nz_on_vector, enable_select_analysis, compile_on_910_95)
```

将 TTIR 转换为 Linalg IR，是整个编译流程中最关键的转换步骤。

**核心 Converter 列表**：

| Converter | 转换目标 |
|-----------|---------|
| StoreConverter | `triton::StoreOp` → `memref::copy` |
| AddPtrConverter | `triton::AddPtrOp` → `memref::ReinterpretCast` |
| GetProgramIDConverter | `triton::GetProgramIdOp` → 函数参数 |
| GetNumProgramsConverter | `triton::GetNumProgramsOp` → 函数参数 |
| LoadConverter | `triton::LoadOp` → `memref::copy` + `bufferization::ToTensorOp` |
| AtomicRMWConverter | `triton::AtomicRMWOp` → `linalg::GenericOp` |
| AtomicCASConverter | `triton::AtomicCASOp` → `linalg::GenericOp` |
| MakeRangeConverter | `triton::MakeRangeOp` → `linalg::GenericOp` |
| SplatConverter | `triton::SplatOp` → `linalg::FillOp` |
| ReduceConverter | `triton::ReduceOp` → `linalg::ReduceOp` |
| ArgMinConverter | `triton::ArgMinOp` → `linalg::ReduceOp` |
| ArgMaxConverter | `triton::ArgMaxOp` → `linalg::ReduceOp` |
| BroadcastConverter | `triton::BroadcastOp` → `linalg::BroadcastOp` |
| TransposeConverter | `triton::TransOp` → `linalg::TransposeOp` |
| ReshapeConverter | `triton::ReshapeOp` → `tensor::ReshapeOp` |
| ExpandDimsConverter | `triton::ExpandDimsOp` → `tensor::ExpandShapeOp` |
| MatmulConverter | `triton::DotOp` → `linalg::MatmulOp` |
| DotScaledConverter | `triton::DotScaledOp` → `linalg::MatmulOp` |
| CatConverter | `triton::CatOp` → `tensor::InsertSliceOp` |
| ScanConverter | `triton::ScanOp` → `func::CallOp` |
| GatherConverter | `triton::GatherOp` → `func::FuncOp` |
| SortOpConverter | `triton::SortOp` → `func::FuncOp` |

**Linalg IR 示例**（向量加法）：

```mlir
func.func @add_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                      %arg2: memref<?xf32>, %arg3: index) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %0 = affine.apply #map(%arg3)
  %1 = affine.apply #map1(%0)
  %2 = memref.subview %arg0[%1, 0] [1, %c1024] [1, 1]
  %3 = bufferization.to_tensor %2 : tensor<1x1024xf32>
  %4 = memref.subview %arg1[%1, 0] [1, %c1024] [1, 1]
  %5 = bufferization.to_tensor %4 : tensor<1x1024xf32>
  %6 = linalg.generic {indexing_maps = [...], iterator_types = [...]}
    ins(%3, %5 : tensor<1x1024xf32>, tensor<1x1024xf32>)
    outs(... : tensor<1x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
  } -> tensor<1x1024xf32>
  ...
}
```

### 阶段 4：Linalg IR → NPU Binary

根据目标硬件不同，有三条编译路径：

#### 路径 A：A2/A3 系列 (Ascend910B)

```
linalg_to_bin_enable_npu_compile_A2_A3(linalg, metadata, opt)
```

调用 `bishengir-compile` 并启用 HIVM 编译：

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --enable-auto-multi-buffer=True \
  --enable-auto-bind-sub-block=True \
  -o kernel
```

源码参考：[compiler.py - linalg_to_bin_A2_A3](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L607-L797)

#### 路径 B：910_95 系列

```
linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, opt)
```

910_95 系列不支持 FFTS，使用不同的编译选项：

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910_9589 \
  --enable-hfusion-compile=true \
  --enable-triton-kernel-compile=true \
  -o kernel
```

源码参考：[compiler.py - linalg_to_bin_910_95](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L403-L604)

#### 路径 C：SIMT Only 模式

```
ttir_to_npubin(mod, metadata, opt)
```

直接从 TTIR 编译，跳过 Linalg IR，使用 SIMT（单指令多线程）模式：

```bash
bishengir-compile kernel.ttir.mlir \
  --target=Ascend910B3 \
  --enable-hivm-compile=false \
  --enable-triton-ir-compile \
  --pure-simt \
  --num-warps=4 \
  --threads-per-warp=32 \
  -o kernel
```

源码参考：[compiler.py - ttir_to_npubin](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L947-L989)

---

## 两条编译路径对比

### 直接路径（SIMD，默认）

```
TTIR → Ascend Passes → Linalg IR → bishengir-compile (--enable-hivm-compile) → NPU Binary
```

- 适用于大多数 Triton kernel
- 通过 Linalg IR 表达计算语义
- BiSheng Compiler 负责 bufferization、tiling、指令调度等底层优化
- 支持 HFusion（算子融合）和 HIVM（多核同步）

### 间接路径（SIMT Only）

```
TTIR → bishengir-compile (--enable-triton-ir-compile --pure-simt) → NPU Binary
```

- 适用于需要 SIMT 执行模式的 kernel
- 跳过 Linalg IR，直接从 TTIR 编译
- 由 BiSheng Compiler 直接处理 Triton IR
- 支持自定义线程配置（num_warps, threads_per_warp）

**路径选择逻辑**（在 `add_stages` 方法中）：

```python
def add_stages(self, stages, options):
    if self.target.backend == "npu":
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        if options.force_simt_only:
            # SIMT Only 路径
            stages["npubin"] = lambda src, metadata: ttir_to_npubin(src, metadata, options)
            return
        # SIMD 路径
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options)
        if options.compile_on_910_95:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(...)
        else:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(...)
```

---

## 编译缓存机制

Triton-Ascend 使用基于哈希的编译缓存，避免重复编译相同的 kernel。

### 缓存路径

默认缓存路径：`~/.triton/cache`

可通过环境变量自定义：

```bash
export TRITON_CACHE_DIR=/path/to/custom/cache
```

### 缓存键

缓存键由以下因素决定：
- kernel 源码
- kernel 参数签名
- 编译选项（NPUOptions 的所有字段）
- 目标架构

### 调试输出

开启 debug 模式后，编译器会将每个阶段的 IR 输出到缓存目录：

```bash
export TRITON_DEBUG=1
```

输出的中间文件包括：

| 文件名 | 内容 |
|--------|------|
| `kernel.ttir.mlir` | 优化后的 TTIR |
| `kernel.ttadapter.mlir` | Ascend Passes 处理后的 Linalg IR |
| `kernel.llir.mlir` | LLVM MLIR（仅 CPU 后端） |
| `kernel.ll` | LLVM IR（仅 CPU 后端） |
| `kernel.npuir.mlir` | BiSheng Compiler 输出的 NPU IR |

---

## 关键编译选项

### NPUOptions 完整列表

以下选项可通过 `@triton.jit` 的 `npu_options` 参数或 autotune 配置传入：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `debug` | `False` | 启用调试输出 |
| `arch` | 自动检测 | 目标 NPU 架构 |
| `num_warps` | `4` | 每个 program 的 warp 数量 |
| `num_stages` | `1` | 流水线阶段数 |
| `warp_size` | `32` | 每个 warp 的线程数 |
| `auto_blockify_size` | `1` | AutoBlockify 调整粒度 |
| `compile_on_910_95` | 自动检测 | 是否编译为 910_95 目标 |
| `multibuffer` | 自动 | 启用/禁用 ping-pong 流水线（存算并行） |
| `enable_auto_bind_sub_block` | `None` | 启用/禁用自动子块绑定（CV 融合算子） |
| `enable_hivm_auto_cv_balance` | `None` | 启用/禁用自动 CV 负载均衡 |
| `sync_solver` | `None` | 启用/禁用同步求解器 |
| `unit_flag` | `None` | 启用/禁用同步单元标志 |
| `inject_barrier_all` | `None` | 启用/禁用全操作屏障注入 |
| `inject_block_all` | `None` | 启用/禁用全操作块同步注入 |
| `enable_nd2nz_on_vector` | `False` | 启用/禁用 ND→NZ 布局转换 |
| `enable_linearize` | - | 启用/禁用线性化 Pass |
| `compile_mode` | `"simd"` | 编译模式：`"simd"` / `"unstructured_in_simt"` / `"simt_only"` |
| `parallel_mode` | `"simd"` | 并行模式：`"simd"` / `"simt"` / `"mix_simd_simt"` |
| `add_auto_scheduling` | `False` | 启用/禁用自动调度 |
| `bisheng_options` | 含 libdevice | 传递给 BiSheng Compiler 的额外选项 |
| `stream` | `None` | 指定 NPU stream |

源码参考：[compiler.py - NPUOptions](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L806-L911)

### 环境变量

| 环境变量 | 说明 |
|---------|------|
| `ASCEND_HOME_PATH` | CANN 安装路径（必须设置） |
| `TRITON_ASCEND_ARCH` | 手动指定目标 NPU 架构 |
| `TRITON_CACHE_DIR` | 编译缓存目录 |
| `TRITON_ALL_BLOCKS_PARALLEL` | 启用 AutoBlockify 自动调整逻辑核数 |
| `TRITON_PRINT_AUTOTUNING` | 打印 autotune 最优参数 |
| `TRITON_COMPILE_ONLY` | 仅编译不执行 |
| `TRITON_DISABLE_FFTS` | 禁用 FFTS 特性 |
| `TRITON_ENABLE_LIBDEVICE` | 启用 libdevice 链接 |
| `TRITON_DEBUG` | 启用调试输出，dump 中间 IR |

---

## NPU 适配要点

1. **编译路径自动选择**：编译器根据 `is_compile_on_910_95` 自动选择 A2/A3 或 910_95 编译路径，无需手动干预。

2. **BiSheng Compiler 版本兼容**：`bishengir-compile` 的 API 可能有变化，Triton-Ascend 通过 `_check_bishengir_api_change()` 和 `_check_bishengir_is_regbased()` 检测并适配。

3. **UB 溢出检测**：BiSheng Compiler 编译时会计算 UB 使用量，如果超出限制会报错。可通过 `--enable-print-memory-allocated-size` 选项打印详细内存分配信息。

4. **回调函数提取元数据**：编译成功后，通过 `libkernel.so` 中的回调函数获取 task_type、workspace_size、lock_num 等运行时元数据。

5. **kernel 名称长度限制**：CANN 运行时限制 kernel 名称长度不超过 50 字符（含 `\n`），实际限制为 49 字符。超长名称会被截断。

---

## 常见问题

### Q1: 编译时报错 `MLIRCompilationError`，如何调试？

**A**: 开启 debug 模式查看中间 IR：

```bash
export TRITON_DEBUG=1
python your_kernel.py
```

检查缓存目录中的 `kernel.ttir.mlir` 和 `kernel.ttadapter.mlir`，定位出错阶段。

### Q2: 如何查看 BiSheng Compiler 的编译命令？

**A**: 开启 debug 模式后，编译器会打印完整的命令行：

```bash
export TRITON_DEBUG=1
# 输出中搜索 [DEBUG] cmd_list:
```

### Q3: 编译时间过长怎么办？

**A**:
- 确认编译缓存正常工作（检查 `~/.triton/cache` 目录）
- 使用 ccache 加速 C++ 编译：`TRITON_BUILD_WITH_CCACHE=true`
- 减小 autotune 的配置数量

### Q4: 如何手动指定目标架构？

**A**: 通过环境变量：

```bash
export TRITON_ASCEND_ARCH=Ascend910B3
```

合法值包括：Ascend910B1~B4, Ascend910_9362~9392, Ascend910_9579~9599, Ascend310B1~B4

### Q5: `compile_mode` 的三种模式有何区别？

**A**:
- `"simd"`（默认）：使用 SIMD 编译路径，经过 Linalg IR，适合大多数算子
- `"unstructured_in_simt"`：将非结构化访存转为 SIMT 模式处理，适合离散访存场景
- `"simt_only"`：纯 SIMT 模式，直接从 TTIR 编译，跳过 Linalg IR

### Q6: 如何理解 `multibuffer` 选项？

**A**: `multibuffer` 控制是否启用 double buffer（存算并行）。启用后，数据搬运和计算可以重叠执行，但会将可用 UB 容量减半。910_95 系列默认关闭，其他系列默认开启。

### Q7: 编译成功但运行结果不正确，如何排查？

**A**:
1. 检查 TTIR 是否正确（`kernel.ttir.mlir`）
2. 检查 Linalg IR 是否正确（`kernel.ttadapter.mlir`）
3. 使用 `tl.device_print` 在 kernel 中打印中间结果
4. 与 CPU 后端结果对比（设置 `target.backend = "cpu"`）

---

## 相关文档

- [01 - 环境搭建与验证](./01-environment-setup.md)：搭建 Triton-Ascend 开发环境
- [02 - 第一个 Triton-Ascend Kernel](./02-first-kernel.md)：编写并运行第一个 kernel
- [源码参考 - compiler.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py)：编译器主入口
- [源码参考 - architecture_design_and_core_features.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/architecture_design_and_core_features.md)：架构设计与核心特性
- [源码参考 - triton_ascend.cc](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/triton_ascend.cc)：Ascend Pass 注册
- [源码参考 - utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/utils.py)：编译工具函数
