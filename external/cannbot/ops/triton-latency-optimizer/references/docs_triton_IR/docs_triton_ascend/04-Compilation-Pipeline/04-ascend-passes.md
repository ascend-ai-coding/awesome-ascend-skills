# Ascend 特有 Pass 详解

## 概述

Ascend 特有 Pass 是 Triton-Ascend 编译流水线中针对昇腾 NPU 硬件特性的优化和转换 Pass。这些 Pass 在 TTIR 优化之后、Linalg IR 生成之前执行，由 [ttir_to_linalg()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L95) 函数驱动。

这些 Pass 负责：自动分块、指针/mask 线性化、离散访存模式转换、注解处理、间接轴转循环、DAG 亲和性优化、操作上浮等，是 Triton IR 适配昇腾 NPU 硬件的关键环节。

## 关键概念

| 概念 | 说明 |
|------|------|
| AutoBlockify | 自动分块，将逻辑核映射到物理核 |
| TritonToStructured | 结构化转换，线性化指针/mask，消除整除取余 |
| DiscreteMaskAccessConversion | 离散 mask 访存模式转换 |
| TritonToAnnotation | 注解转换，处理 `al.compile_hint` |
| TritonToUnstructured | 非结构化转换，将间接轴转为显式循环 |
| TritonAffinityOpt | DAG 亲和性优化，包含 DAGSync/DAGScope/DAGSSBuffer |
| BubbleUpOperation | 操作上浮优化，优化 extract/extract_slice 位置 |
| add_auto_scheduling | 自动调度选项，控制 DAG 亲和性优化是否启用 |

## Pass 执行序列

[ttir_to_linalg()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L95) 中定义的 Ascend Pass 序列：

```python
def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
    # ...
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    # 1. AutoBlockify
    ascend.passes.ttir.add_auto_blockify(pm, auto_blockify_size)

    # 2. DAG 亲和性优化（可选，由 add_auto_scheduling 控制）
    if metadata["add_auto_scheduling"]:
        ascend.passes.ttir.add_dag_sync(pm)
        ascend.passes.ttir.add_dag_scope(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        ascend.passes.ttir.add_dag_ssbuffer(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

    # 3. TritonToStructured
    ascend.passes.ttir.add_triton_to_structure(pm, enable_mask_fallback_conversion, optimize_dynamic_offset)

    # 4. DiscreteMaskAccessConversion
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, compile_on_910_95, force_simt_template)

    # 5. TritonToAnnotation
    ascend.passes.ttir.add_triton_to_annotation(pm)

    # 6. TritonToUnstructured
    ascend.passes.ttir.add_triton_to_unstructure(pm, compile_on_910_95, force_simt_template)

    # 7. TritonToHIVM
    ascend.passes.ttir.add_triton_to_hivm(pm)

    # 8. TritonToHFusion
    ascend.passes.ttir.add_triton_to_hfusion(pm)

    # 9. TritonToLLVM
    ascend.passes.ttir.add_triton_to_llvm(pm)

    # 10. BubbleUpOperation
    ascend.passes.ttir.add_bubble_up_operation(pm)

    # 11. TritonToStructured（二次）
    ascend.passes.ttir.add_triton_to_structure(pm, enable_mask_fallback_conversion, optimize_dynamic_offset)

    # 12. TritonToLinalg
    ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops, enable_nd2nz_on_vector,
                                             enable_select_analysis, compile_on_910_95)
    pm.run(mod)
    return str(mod)
```

## Pass 详解

### 1. AutoBlockify - 自动分块

#### 功能

AutoBlockify Pass 将多个逻辑核（program id）映射到单个物理核上执行，通过合并逻辑核减少调度开销。当逻辑核数大于物理核数时，该 Pass 自动调整逻辑核数量为物理核数的整数倍。

#### 源码位置

- 头文件：[AutoBlockify.h](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/include/AutoBlockify/AutoBlockify.h)
- 实现：[AutoBlockify.cpp](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/lib/AutoBlockify/AutoBlockify.cpp)

#### 算法原理

1. **检查是否需要分块**：遍历 `tt.func` 中的 `GetProgramIdOp`，判断是否可分块
2. **计算分块 ID**：将原始 `program_id` 重新计算为分块后的 ID
3. **替换 program_id**：用分块 ID 替换原始 `GetProgramIdOp`
4. **生成边界检查**：为分块 ID 生成上下界 mask
5. **后处理**：执行 CSE 和 Canonicalize 清理 IR

#### 编译选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_blockify_size` | int | 1 | 分块大小，1 表示不启用。受 `TRITON_ALL_BLOCKS_PARALLEL` 环境变量控制 |

当 `TRITON_ALL_BLOCKS_PARALLEL` 未设置时，`auto_blockify_size` 被强制设为 1（不启用）。

#### IR 转换示例

```mlir
// Before: 4 个逻辑核
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %pid = tt.get_program_id x : i32
  %offset = arith.muli %pid, %BLOCK_SIZE : i32
  // ...
}

// After: AutoBlockify(size=2)，2 个物理核各执行 2 个逻辑核
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %blockified_id = ... : i32  // 合并后的 ID
  %pid = arith.divsi %blockified_id, %yz_num : i32
  %pid_x = arith.remsi %pid, %x_num : i32
  // 边界检查 mask
  %mask = arith.andi %upper_mask, %lower_mask : i1
  // ...
}
```

#### NPU 适配要点

- 启用条件：`TRITON_ALL_BLOCKS_PARALLEL=1` 且 `auto_blockify_size > 1`
- 限制：Kernel 的逻辑必须对执行顺序不敏感，否则可能导致死锁
- 启用后允许 grid > 65535

---

### 2. TritonToStructured - 线性化指针/mask

#### 功能

TritonToStructured Pass 处理指针表达式和 mask 表达式中的整除取余操作，通过升维方法去除整除取余后重新生成 load/store 等操作。这是 SIMD 编译路径的关键 Pass。

#### 编译选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_mask_fallback_conversion` | bool | False | 启用 mask 回退转换 |
| `optimize_dynamic_offset` | bool | False | 优化动态偏移 |

#### 核心转换器

| 转换器 | 功能 | 局限性 |
|--------|------|--------|
| RewriteAddPtrOp | 分析 AddPtrOp 指针表达式，分解为 PtrState 对象 | 原始迭代轴必须能被分裂轴整除 |
| CreateAddpr | 根据 PtrState 重新构造 AddPtrOp | 依赖 RewriteAddPtrOp 成功 |
| RewriteLoadOp | 分析 load 的 mask 表达式，分解为 MaskState | 同 RewriteAddPtrOp |
| BuildMask | 根据 MaskState 重新构造 mask | 仅处理规范化 mask |
| CreateLoad | 用新指针和新 mask 重建 load | 依赖前置步骤 |
| RewriteStoreOp | 分析 store 的 mask 表达式 | 同 RewriteLoadOp |
| CreateStore | 用新指针和新 mask 重建 store | 依赖前置步骤 |
| RewriteAtomicRWMOp | 处理原子读写修改的指针 | 继承 RewriteAddPtrOp 局限性 |
| RewriteAtomicCASOp | 处理原子比较并交换的指针 | |
| RewriteWhile | 处理 while 循环内的指针叠加 | 不支持循环内含 if |
| RewriteFor | 处理 for 循环内的指针叠加 | |

#### IR 转换示例

```mlir
// Before: 包含整除取余的指针
%offset_x = arith.divsi %idx, %1024 : i32
%offset_y = arith.remsi %idx, %1024 : i32
%ptr = tt.addptr %base, %offset_x : !tt.ptr<f32>, i32
%mask = arith.andi %mask_x, %mask_y : tensor<1024xi1>
%load = tt.load %ptr, %mask : tensor<1024xf32>

// After: 升维消除整除取余
%ptr_new = tt.addptr %base, %new_offset : !tt.ptr<f32>, tensor<1024xi32>
%mask_new = ... // 升维后的 mask
%load = tt.load %ptr_new, %mask_new : tensor<1024xf32>
```

#### 二次执行

TritonToStructured 在 Pass 序列中执行两次：
1. 第一次：处理原始 TTIR 中的指针/mask
2. 第二次（在 BubbleUpOperation 之后）：处理由其他 Pass 生成的新指针/mask

---

### 3. DiscreteMaskAccessConversion - 离散 mask 访存模式转换

#### 功能

将 Triton 中基于离散索引 mask 的内存访问模式进行分析与转换，为后续将离散轴展开为循环做准备。该 Pass 识别出无法被后端硬件高效处理的非规律性或稀疏访问模式。

#### 编译选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compile_on_910_95` | bool | 自动检测 | 是否在 910_95 平台编译 |
| `force_simt_template` | bool | False | 强制使用 SIMT 模板 |

#### 核心转换器

| 转换器 | 描述 |
|--------|------|
| DiscreteMaskStoreConversion | 非连续 mask 的 store → load + select + store |
| DiscreteMaskLoadConversion | 非连续 mask 的 load → load + select(other) |
| DiscreteMaskAtomicAddConversion | 非连续 mask 的 atomic_add → select + atomic_add |

#### IR 转换示例

```mlir
// Before: 离散 mask 的 load
%load = tt.load %ptr, %discrete_mask : tensor<1024xf32>

// After: 转换为全量 load + select
%full_load = tt.load %ptr : tensor<1024xf32>
%other = arith.constant dense<0.0> : tensor<1024xf32>
%result = arith.select %discrete_mask, %full_load, %other : tensor<1024xi1>
```

```mlir
// Before: 离散 mask 的 store
tt.store %ptr, %value, %discrete_mask

// After: 转换为 load + select + store
%old = tt.load %ptr : tensor<1024xf32>
%merged = arith.select %discrete_mask, %value, %old : tensor<1024xf32>
tt.store %ptr, %merged
```

---

### 4. TritonToAnnotation - 注解转换

#### 功能

处理 Ascend NPU 特有的编译提示指令 `al.compile_hint`，将其转换为后端的 Annotation Dialect，用于指导后续的硬件特定优化或资源配置。

#### 核心转换器

| 转换器 | 描述 |
|--------|------|
| TritonAnnotationConversion | 将 `triton::AnnotationOp` 转换为 `annotation::MarkOp` |

#### IR 转换示例

```mlir
// Before: Triton 注解
%0 = triton.annotation "hint_name"(%input) : (tensor<1024xf32>) -> tensor<1024xf32>

// After: Annotation Dialect
%0 = annotation.mark "hint_name"(%input) : (tensor<1024xf32>) -> tensor<1024xf32>
```

---

### 5. TritonToUnstructured - 间接轴转循环

#### 功能

将经过 DiscreteMaskAccessConversion 识别出的、包含离散轴的张量操作，转换为基于显式标量循环的标量访存。这是处理 SIMT 混合模式的关键 Pass。

#### 编译选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compile_on_910_95` | bool | 自动检测 | 是否在 910_95 平台编译 |
| `force_simt_template` | bool | False | 强制使用 SIMT 模板 |

#### 核心转换器

| 转换器 | 描述 |
|--------|------|
| UnstructuredMemAccessConverter\<LoadOp\> | Load → 多重循环标量加载 |
| UnstructuredMemAccessConverter\<StoreOp\> | Store → 多重循环标量存储 |
| UnstructuredMemAccessConverter\<AtomicRMWOp\> | AtomicRMW → 多重循环标量 Atomic |
| UnstructuredMemAccessConverter\<AtomicCASOp\> | AtomicCAS → 多重循环标量 Atomic |

#### IR 转换示例

```mlir
// Before: 向量化 load
%load = tt.load %ptr, %mask : tensor<1024xf32>

// After: 标量循环加载
scf.for %i = %c0 to %c1024 step %c1 {
  %elem_mask = tensor.extract %mask[%i] : tensor<1024xi1>
  scf.if %elem_mask {
    %elem_ptr = tt.addptr %ptr, %i : !tt.ptr<f32>, i32
    %elem = tt.load %elem_ptr : f32
    // ...
  }
}
```

---

### 6. TritonAffinityOpt - DAG 亲和性优化

#### 功能

TritonAffinityOpt 是一组 Pass，用于分析 Triton IR 的数据流图（DAG），根据操作的硬件亲和性（Cube/Vector）进行优化，包括同步插入、作用域划分和共享存储缓冲区管理。

#### 源码位置

- 头文件：[Passes.h](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/include/TritonAffinityOpt/Passes.h)
- DAG 实现：[DAG.cpp](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/lib/TritonAffinityOpt/DAG.cpp)
- DAGSync：[DAGSync.cpp](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/lib/TritonAffinityOpt/DAGSync.cpp)
- DAGScope：[DAGScope.cpp](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/lib/TritonAffinityOpt/DAGScope.cpp)
- DAGSSBuffer：[DAGSSBuffer.cpp](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/lib/TritonAffinityOpt/DAGSSBuffer.cpp)

#### 子 Pass 列表

| Pass | TableGen 名称 | 说明 |
|------|--------------|------|
| DAGSync | `dag-sync` | DAG 同步插入，在 Cube/Vector 操作间插入同步和数据搬运 |
| DAGScope | `dag-scope` | DAG 作用域划分，将操作按 Cube/Vector 亲和性划分到不同 scope |
| DAGSSBuffer | `dag-ssbuf` | DAG 共享存储缓冲区，将 Vector 操作转换为共享存储缓冲区操作 |

#### DAGSync 详解

DAGSync Pass 分析操作间的数据依赖，在 Cube 和 Vector 操作之间插入同步指令（`hivm.SyncBlockSetOp`/`hivm.SyncBlockWaitOp`）和数据搬运操作。

核心功能：
1. **构建 DAG**：分析函数内所有操作的数据流关系
2. **识别核心类型**：判断每个操作属于 Cube 还是 Vector 核心
3. **插入同步**：在跨核心类型的操作间插入同步指令
4. **数据搬运**：在需要时插入 GM↔UB 的数据搬运操作
5. **内存效果分析**：基于 SharedMemoryAliasAnalysis 处理内存依赖

#### DAGScope 详解

DAGScope Pass 将操作按 Cube/Vector 亲和性划分到不同的 `scope::ScopeOp` 中：

1. **封装 Scope**：将函数体封装为 AIV（Vector）和 AIC（Cube）两个 scope
2. **操作分类**：根据 DAG 分析结果将操作移动到对应的 scope
3. **同步增强**：为 scope 间的数据依赖添加同步点

#### DAGSSBuffer 详解

DAGSSBuffer Pass 将 Vector 操作转换为使用共享存储缓冲区的操作，优化 UB 内存使用。

#### 启用条件

DAG 亲和性优化由 `add_auto_scheduling` 编译选项控制：

```python
if metadata["add_auto_scheduling"]:
    ascend.passes.ttir.add_dag_sync(pm)
    ascend.passes.ttir.add_dag_scope(pm)
    passes.common.add_cse(pm)
    passes.common.add_canonicalizer(pm)
    ascend.passes.ttir.add_dag_ssbuffer(pm)
    passes.common.add_cse(pm)
    passes.common.add_canonicalizer(pm)
```

#### IR 转换示例

```mlir
// Before: 无作用域划分
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %load = tt.load %arg0 : tensor<1024xf32>
  %dot = tt.dot %a, %b, %c : tensor<64x64xf32>
  %add = arith.addf %dot, %load : tensor<1024xf32>
  tt.store %out, %add
}

// After: DAG 亲和性优化
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  scope.scope @aiv {
    %load = tt.load %arg0 : tensor<1024xf32>
    hivm.sync_block_set @aiv, @aic, %event_id
    // ...
  }
  scope.scope @aic {
    hivm.sync_block_wait @aiv, @aic, %event_id
    %dot = tt.dot %a, %b, %c : tensor<64x64xf32>
    hivm.sync_block_set @aic, @aiv, %event_id2
  }
  scope.scope @aiv {
    hivm.sync_block_wait @aic, @aiv, %event_id2
    %add = arith.addf %dot, %load : tensor<1024xf32>
    tt.store %out, %add
  }
}
```

---

### 7. BubbleUpOperation - 操作上浮优化

#### 功能

对 `tensor.extract` 和 `tensor.extract_slice` 操作进行顺序上移优化。将 extract 操作移动到更早的位置，可以优化数据局部性，某些场景能消除转换后产生的不必要的循环。

#### 核心转换器

| 转换器 | 描述 |
|--------|------|
| BubbleUpExtract\<tensor::ExtractOp\> | extract op 顺序上移优化 |
| BubbleUpExtract\<tensor::ExtractSliceOp\> | extract_slice 顺序上移优化 |

#### IR 转换示例

```mlir
// Before: extract 在循环内
scf.for %i = %c0 to %c1024 step %c1 {
  %val = tt.load %ptr[%i] : tensor<4xf32>
  %elem = tensor.extract %val[%idx] : tensor<4xf32>
  // use %elem
}

// After: extract 上移到循环外（如果 %idx 不依赖循环变量）
%full = tt.load %ptr : tensor<1024x4xf32>
%row = tensor.extract_slice %full[%idx, 0] [1, 4] [1, 1] : tensor<4xf32>
scf.for %i = %c0 to %c1024 step %c1 {
  // use %row directly
}
```

---

### 8. 其他 Dialect 转换 Pass

以下 Pass 在 Ascend Pass 序列中执行，负责将 Triton IR 转换为不同的后端 Dialect：

#### TritonToHIVM

处理 Triton 的块同步操作，将其转换为 HIVM Dialect 的跨核心同步指令：

| 转换器 | 描述 |
|--------|------|
| TritonCustomOpToHIVMSyncOpConversion | `sync_block_all` → HIVM 全局同步 |
| | `sync_block_set` → HIVM 同步设置 |
| | `sync_block_wait` → HIVM 同步等待 |

#### TritonToHFusion

将 Triton IR 转换为 HFusion Dialect：

| 转换器 | 描述 |
|--------|------|
| TritonHistogramToHFusionConversion | `triton::HistogramOp` → `hfusion::HistogramOp` |

#### TritonToLLVM

将 Triton 的内联汇编操作转换为 LLVM Dialect：

| 转换器 | 描述 |
|--------|------|
| ElementwiseInlineAsmOpConversion | `triton::ElementwiseInlineAsmOp` → `LLVM::InlineAsmOp` |

## NPU 适配要点

1. **Pass 序列不可随意调整**：Pass 之间存在严格的依赖关系，如 DiscreteMaskAccessConversion 必须在 TritonToUnstructured 之前
2. **TritonToStructured 执行两次**：第一次处理原始 IR，第二次处理 BubbleUpOperation 后的新 IR
3. **DAG 亲和性优化可选**：由 `add_auto_scheduling` 控制，默认关闭
4. **force_simt_template 影响**：当 `compile_mode="unstructured_in_simt"` 时，`force_simt_template=True`，影响 DiscreteMaskAccessConversion 和 TritonToUnstructured 的行为
5. **910_95 平台差异**：部分 Pass 的行为在 910_95 平台上有差异

## 常见问题

### Q: 为什么 TritonToStructured 要执行两次？

第一次处理原始 TTIR 中的指针/mask 表达式。BubbleUpOperation 可能产生新的指针/mask 表达式，需要第二次 TritonToStructured 来处理。

### Q: 什么时候需要启用 add_auto_scheduling？

当 Kernel 包含 Cube（矩阵乘）和 Vector（逐元素）混合操作时，启用 `add_auto_scheduling` 可以自动插入同步和数据搬运，优化 CV 流水线。纯向量计算 Kernel 不需要启用。

### Q: DiscreteMaskAccessConversion 和 TritonToUnstructured 有何关系？

DiscreteMaskAccessConversion 先识别离散 mask 模式并转换为 select 模式，TritonToUnstructured 再将离散轴展开为标量循环。两者配合处理无法被硬件高效执行的访存模式。

### Q: AutoBlockify 何时生效？

需要同时满足：`TRITON_ALL_BLOCKS_PARALLEL=1` 环境变量已设置，且 `auto_blockify_size > 1`。Kernel 的逻辑必须对执行顺序不敏感。

## 相关文档

- [01-pipeline-overview.md](01-pipeline-overview.md) - 编译流水线总览
- [03-ttir-optimization.md](03-ttir-optimization.md) - TTIR 优化 Pass 详解
- [05-triton-to-linalg.md](05-triton-to-linalg.md) - Triton IR → Linalg IR 转换
- [07-compile-options.md](07-compile-options.md) - 编译选项与环境变量参考
