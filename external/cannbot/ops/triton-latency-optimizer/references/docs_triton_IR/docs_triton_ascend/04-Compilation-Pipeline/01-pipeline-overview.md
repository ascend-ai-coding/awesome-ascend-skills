# 编译流水线总览

## 概述

Triton-Ascend 编译流水线是将 Python 编写的 Triton Kernel 逐步编译为昇腾 NPU 可执行二进制的核心流程。整个流水线从 Python AST 出发，经过多个 IR 阶段和 Pass 转换，最终生成面向 Ascend NPU 硬件的设备二进制文件（`.o` 文件）。

Triton-Ascend 提供了两条主要编译路径：**SIMD 路径**（通过 Linalg → HFusion → HIVM → LLVM）和 **SIMT 路径**（通过 TTGIR → LLVM），以及两者的混合编译路径。路径选择由编译选项 `compile_mode` 控制。

## 关键概念

| 概念 | 说明 |
|------|------|
| TTIR | Triton IR，Triton 前端生成的高层中间表示，基于 MLIR 的 Triton Dialect |
| TTGIR | TritonGPU IR，包含 GPU 线程布局编码的中间表示 |
| Linalg IR | 线性代数 IR，MLIR 标准的 Dialect，用于表达结构化计算 |
| HFusion IR | 昇腾算子融合 IR，用于 CV（Cube-Vector）流水线优化 |
| HIVM IR | 昇腾设备指令层 IR，描述 Cube/Vector 核心的执行指令 |
| BiSheng Compiler | 昇腾编译器，将 Linalg/HFusion IR 编译为 NPU 二进制 |
| SIMD 路径 | 单指令多数据路径，通过 Cube/Vector 核心分工执行 |
| SIMT 路径 | 单指令多线程路径，通过 SIMT 线程模型执行 |
| mix_mode | 混合执行模式，取值为 `aiv`（Vector）、`aic`（Cube）、`mix_simd_simt`（混合） |
| NPUOptions | NPU 编译选项数据类，控制编译流水线行为 |

## 完整编译流水线图

```
Python Kernel (@triton.jit)
    │
    ▼
Python AST
    │
    │  [code_generator.py: ast_to_ttir()]
    │  - AST 遍历与 IR 生成
    │  - builtin/constexpr 处理
    │  - Ascend 扩展注入
    │
    ▼
TTIR (Triton IR)
    │
    │  [make_ttir(): TTIR 优化]
    │  - InlinerPass / CombinePass / CanonicalizerPass
    │  - ReorderBroadcastPass / CSEPass / LICMPass
    │  - LoopUnrollPass / SymbolDCEPass
    │
    ▼
优化后的 TTIR
    │
    ├─── 路径 A: SIMD 路径 (compile_mode="simd") ──────────────────┐
    │                                                                │
    │  [ttir_to_linalg(): Ascend Passes]                            │
    │  - AutoBlockify (自动分块)                                     │
    │  - DAGSync/DAGScope/DAGSSBuffer (亲和性优化, 可选)             │
    │  - TritonToStructured (指针/mask 线性化)                       │
    │  - DiscreteMaskAccessConversion (离散 mask 转换)               │
    │  - TritonToAnnotation (注解转换)                               │
    │  - TritonToUnstructured (间接轴转循环)                         │
    │  - TritonToHIVM/HFusion/LLVM (Dialect 转换)                   │
    │  - BubbleUpOperation (操作上浮)                                │
    │  - TritonToStructured (二次结构化)                              │
    │  - TritonToLinalg (转 Linalg IR)                               │
    │                                                                │
    │  ──────────────────────────────────────────────────────────────│
    │                                                                ▼
    │  Linalg IR
    │    │
    │    │  [linalg_to_bin_*(): BiSheng Compiler]
    │    │  - HFusion 编译
    │    │  - HIVM 编译
    │    │  - LLVM 降低
    │    │
    │    ▼
    │  NPU 二进制 (kernel.o)
    │
    ├─── 路径 B: SIMT 路径 (compile_mode="simt_only") ─────────────┐
    │                                                                │
    │  [ttir_to_npubin(): TTIR 直接编译]                             │
    │  - bishengir-compile --enable-triton-ir-compile                │
    │  - --pure-simt --num-warps --threads-per-warp                  │
    │  - BiShengIR 内部: TTIR → TTGIR → LLVM → 二进制               │
    │                                                                │
    │  ──────────────────────────────────────────────────────────────│
    │                                                                ▼
    │  NPU 二进制 (kernel.o)
    │
    └─── 路径 C: 混合路径 (compile_mode="unstructured_in_simt") ───┐
                                                                     │
       [ttir_to_linalg() + BiSheng Compiler 混合模式]               │
       - SIMD 部分走 HFusion/HIVM 路径                              │
       - SIMT 部分通过 HIVMToTritonGPU 转换                         │
       - 两条路径在 LLVM IR 层汇合                                   │
                                                                     ▼
     NPU 二进制 (kernel.o)
```

## 两条编译路径对比

### 路径 A：SIMD 路径（默认路径）

```
TTIR → Ascend Passes → Linalg IR → HFusion IR → HIVM IR → LLVM IR → 设备二进制
```

| 维度 | 说明 |
|------|------|
| 执行模型 | SIMD（单指令多数据），Cube/Vector 核心分工 |
| 适用场景 | 大部分 Triton Kernel，特别是向量计算密集型 |
| mix_mode | `aiv`（Vector 核心）或 `aic`（Cube 核心） |
| 关键优化 | CV 流水线、算子融合、内存复用 |
| 编译器 | BiSheng Compiler (bishengir-compile) |
| 编译选项 | `compile_mode="simd"`（默认） |

### 路径 B：SIMT 路径

```
TTIR → TTGIR → LLVM IR → 设备二进制
```

| 维度 | 说明 |
|------|------|
| 执行模型 | SIMT（单指令多线程），线程模型执行 |
| 适用场景 | 控制流密集、动态索引、离散访存模式 |
| 关键优化 | 布局传播、线程局部性、内存合并 |
| 编译器 | BiSheng Compiler (bishengir-compile --enable-triton-ir-compile) |
| 编译选项 | `compile_mode="simt_only"`，`force_simt_only=True` |

### 混合编译路径

```
HIVM IR
    ├── [SIMD 部分] → HIVM → LLVM (Cube/Vector 核心执行)
    └── [SIMT 部分] → HIVMToTritonGPU → TTGIR → LLVM (SIMT 线程执行)
```

混合路径通过 `compile_mode="unstructured_in_simt"` 启用。在该模式下：
1. `AutoScope` 自动识别 SIMT 代码区域
2. `InsertMemSemanticForSimtVF` 为 SIMT 区域插入内存语义
3. `OutlineScope` 将 SIMT 区域外联为独立模块
4. `SplitSimtModule` 将 SIMT 模块从主模块中分离
5. 分离后的 SIMT 模块通过 `HIVMToTritonGPU` 转换进入 Triton 编译路径

## 每个阶段的输入/输出格式

| 阶段 | 输入 | 输出 | 格式说明 |
|------|------|------|----------|
| AST → TTIR | Python AST | TTIR Module | MLIR Module，包含 `tt.func`、`tt.load` 等 Triton Dialect 操作 |
| TTIR 优化 | TTIR Module | 优化后 TTIR Module | 同上，经过优化 Pass 处理 |
| Ascend Passes | TTIR Module | Linalg IR 字符串 | 包含 Linalg Dialect 操作的 MLIR 文本 |
| Linalg → Binary | Linalg IR 字符串 | NPU 二进制 (bytes) | `.o` 目标文件，由 BiSheng Compiler 生成 |
| TTIR → Binary (SIMT) | TTIR Module | NPU 二进制 (bytes) | 直接由 BiSheng Compiler 编译 |
| CPU 路径 | Linalg IR 字符串 | CPU 汇编文本 | 通过 mlir-opt → LLVM IR → llc → 汇编 |

## 编译缓存机制

Triton-Ascend 使用基于哈希的编译缓存机制，避免重复编译相同的 Kernel。

### 缓存键计算

缓存键由以下因素决定：

```python
key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
```

| 因素 | 说明 |
|------|------|
| `triton_key()` | Triton 版本 + 编译器/后端/语言模块的 SHA256 哈希 |
| `src.hash()` | 源代码哈希（AST 源或 IR 文件内容） |
| `backend.hash()` | 后端目标信息（如 NPU 架构） |
| `backend_options.hash()` | 编译选项哈希（NPUOptions 所有字段） |
| `env_vars` | 缓存失效环境变量 |

### 缓存查找流程

1. 计算缓存键的 SHA256 哈希
2. 在缓存目录中查找 `metadata.json`
3. 若命中缓存，直接返回 `CompiledKernel` 对象
4. 若未命中，执行完整编译流水线
5. 编译完成后，将各阶段 IR 和元数据写入缓存

### 缓存相关环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TRITON_ALWAYS_COMPILE` | 0 | 设为 1 强制每次重新编译 |
| `TRITON_KERNEL_DUMP` | 0 | 设为 1 启用 IR 转储 |
| `TRITON_KERNEL_OVERRIDE` | 0 | 设为 1 启用 IR 覆盖 |

### 调试输出

当 `opt.debug=True` 时，编译器会在每个阶段将中间 IR 保存到缓存目录：

| 阶段 | 文件名 | 内容 |
|------|--------|------|
| TTIR 优化后 | `kernel.ttir.mlir` | 优化后的 TTIR |
| Ascend Passes 后 | `kernel.ttadapter.mlir` | Linalg IR |
| LLVM-MLIR | `kernel.llir.mlir` | LLVM Dialect MLIR |
| LLVM IR | `kernel.ll` | LLVM IR 文本 |
| 链接后 | `kernel_linked.ll` | 链接 libdevice 后的 LLVM IR |
| CPU 汇编 | `kernel.s` | CPU 汇编代码 |
| NPU IR | `kernel.npuir.mlir` | BiSheng Compiler 输出的 NPU IR |

## 编译入口与阶段注册

编译流水线的阶段注册在 [AscendBackend.add_stages()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L1058) 中完成：

```python
def add_stages(self, stages, options):
    if self.target.backend == "npu":
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        if options.force_simt_only:
            stages["npubin"] = lambda src, metadata: ttir_to_npubin(src, metadata, options)
            return
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
        if options.compile_on_910_95:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(src, metadata, options)
        else:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(src, metadata, options)
    else:
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options)
        stages["llir"] = lambda src, metadata: linalg_to_llir(src, metadata, options)
        stages["cpuasm"] = lambda src, metadata: llir_to_cpuasm(src, metadata, options)
```

### NPU 后端阶段

| 阶段名 | 函数 | 输入 → 输出 |
|--------|------|-------------|
| `ttir` | `make_ttir()` | TTIR Module → 优化后 TTIR Module |
| `ttadapter` | `ttir_to_linalg()` | TTIR Module → Linalg IR 字符串 |
| `npubin` | `linalg_to_bin_*()` | Linalg IR 字符串 → NPU 二进制 bytes |

### SIMT 路径阶段

| 阶段名 | 函数 | 输入 → 输出 |
|--------|------|-------------|
| `ttir` | `make_ttir()` | TTIR Module → 优化后 TTIR Module |
| `npubin` | `ttir_to_npubin()` | TTIR Module → NPU 二进制 bytes |

### CPU 后端阶段

| 阶段名 | 函数 | 输入 → 输出 |
|--------|------|-------------|
| `ttir` | `make_ttir()` | TTIR Module → 优化后 TTIR Module |
| `ttadapter` | `ttir_to_linalg()` | TTIR Module → Linalg IR 字符串 |
| `llir` | `linalg_to_llir()` | Linalg IR 字符串 → LLVM IR 文本 |
| `cpuasm` | `llir_to_cpuasm()` | LLVM IR 文本 → CPU 汇编文本 |

## 编译主流程

编译入口函数 [compile()](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/compiler.py#L224) 的核心流程：

1. **创建后端**：根据 `target` 选择 `AscendBackend`
2. **解析选项**：将用户选项解析为 `NPUOptions` 或 `CPUOptions`
3. **缓存查找**：计算缓存键，检查是否命中
4. **加载 Dialect**：加载 Triton、Buffer、Ascend 等 Dialect
5. **生成 IR**：调用 `src.make_ir()` 从 AST 生成 TTIR Module
6. **逐阶段编译**：按 `stages` 字典顺序执行每个编译阶段
7. **缓存写入**：将各阶段结果和元数据写入缓存
8. **返回结果**：返回 `CompiledKernel` 对象

## NPU 适配要点

1. **路径选择**：根据 `compile_mode` 选择 SIMD/SIMT/混合路径
2. **硬件平台区分**：910B/910_95 使用 `linalg_to_bin_enable_npu_compile_910_95()`，A2/A3 使用 `linalg_to_bin_enable_npu_compile_A2_A3()`
3. **元数据解析**：`_parse_linalg_metadata()` 从 Linalg IR 中提取 `mix_mode`、`kernel_name`、`tensor_kinds` 等关键信息
4. **BiSheng Compiler**：最终二进制生成依赖 BiSheng Compiler，需正确配置编译选项
5. **UB 内存管理**：BiSheng Compiler 输出 UB 占用信息（`required_ub_bits`），用于 Inductor autotune

## 常见问题

### Q: 编译缓存导致修改不生效？

设置 `TRITON_ALWAYS_COMPILE=1` 强制重新编译，或清理缓存 `rm -r ~/.triton/cache`。

### Q: 如何选择 SIMD 还是 SIMT 路径？

- 默认使用 SIMD 路径（`compile_mode="simd"`），适合大部分场景
- 控制流密集、动态索引场景使用 SIMT 路径（`compile_mode="simt_only"`）
- 混合场景使用 `compile_mode="unstructured_in_simt"`

### Q: 编译失败如何调试？

1. 设置 `TRITON_KERNEL_DUMP=1` 转储各阶段 IR
2. 设置 `MLIR_ENABLE_DUMP=1` 转储 MLIR 优化前后的 IR
3. 设置 `TRITON_DEBUG=1` 启用调试输出
4. 检查缓存目录中的中间文件

### Q: 910_95 和 A2/A3 的编译有何区别？

910_95 使用 `linalg_to_bin_enable_npu_compile_910_95()`，支持更多 CV 流水线选项（如 `enable_hivm_auto_cv_balance`、`sync_solver` 等）。A2/A3 使用 `linalg_to_bin_enable_npu_compile_A2_A3()`，支持 `enable_ubuf_saving`、`tile_mix_vector_loop` 等选项。

## 相关文档

- [02-ttir-generation.md](02-ttir-generation.md) - AST → TTIR 生成详解
- [03-ttir-optimization.md](03-ttir-optimization.md) - TTIR 优化 Pass 详解
- [04-ascend-passes.md](04-ascend-passes.md) - Ascend 特有 Pass 详解
- [05-triton-to-linalg.md](05-triton-to-linalg.md) - Triton IR → Linalg IR 转换
- [06-linalg-to-binary.md](06-linalg-to-binary.md) - Linalg IR → 设备二进制
- [07-compile-options.md](07-compile-options.md) - 编译选项与环境变量参考
