# 内存规划 — PlanMemory Pass

## 概述

`hivm-plan-memory` Pass 是内存管理子系统的核心，负责将逻辑缓冲区（`memref.alloc`）映射到物理内存偏移。它通过分析缓冲区的生命周期和依赖关系，实现内存复用（Inplace）和物理偏移分配。

## Pass 定义

- **Pass 名**：`hivm-plan-memory`
- **作用域**：`ModuleOp`
- **构造函数**：`mlir::hivm::createPlanMemoryPass()`
- **依赖方言**：`hivm::HIVMDialect`
- **源码参考**：[Passes.td:L130-L161](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L130-L161)

## 选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `mem-plan-mode` | `hivm::MemPlanMode` | `LOCAL_MEM_PLAN` | 内存规划模式 |
| `enable-global-workspace-reuse` | `bool` | `false` | 启用全局工作空间复用 |
| `enable-print-memory-allocated-size` | `bool` | `false` | 打印已分配内存大小 |
| `restrict-inplace-as-isa` | `bool` | `false` | 限制内存 inplace 与 ISA 一致 |
| `simt-vf-dynamic-size` | `int` | `216` | SIMT VF 的动态 UB 大小（KB） |
| `disable-tightly-coupled-buffer-reuse` | `bool` | `false` | 禁用紧耦合缓冲区复用 |

## 内存规划模式

### local-mem-plan（默认）

为 `memref.alloc` 分配局部内存偏移。每个 `memref.alloc` 被替换为 `hivm.hir.pointer_cast(offset)`，其中 `offset` 是在对应地址空间内的字节偏移。

```mlir
// 变换前
%copy_in_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
%dst1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>

// 变换后
%0 = arith.constant 0 : i64
%copy_in_ub = hivm.hir.pointer_cast(%0) : memref<16x16x16xf16, #hivm.address_space<ub>>
%1 = arith.constant 8192 : i64
%dst1 = hivm.hir.pointer_cast(%1) : memref<16x16x16xf16, #hivm.address_space<ub>>
```

### global-workspace-plan

为 `memref_ext.alloc_workspace` 分配全局工作空间偏移。适用于需要跨函数共享的大块全局内存。

## 核心算法

### 1. 生命周期分析

分析每个缓冲区的定义点和使用点，确定其活跃范围（liveness interval）。

### 2. Inplace 复用

当两个缓冲区的活跃范围不重叠时，它们可以共享同一块物理内存。这是 PlanMemory 的核心优化。

从测试用例 [plan-memory.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/plan-memory.mlir) 可以看到典型的 Inplace 场景：

```mlir
// 线性分配：4 个 UB 缓冲区，Inplace 复用后只需 2 个偏移
%copy_in_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
%dst1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
%dst2 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
%copy_out_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>

// PlanMemory 后：
// %copy_in_ub  → pointer_cast(0)     — load 使用后不再需要
// %dst1        → pointer_cast(8192)   — vadd 使用后不再需要
// %dst2        → pointer_cast(0)      — 复用 %copy_in_ub 的空间
// %copy_out_ub → pointer_cast(8192)   — 复用 %dst1 的空间
```

### 3. 控制流处理

PlanMemory 需要处理各种控制流结构中的缓冲区：

- **scf.for**：循环内的缓冲区需要考虑迭代间的冲突。循环内分配的缓冲区在不同迭代间不能 Inplace 复用。
- **scf.if**：if/else 分支中的缓冲区需要考虑分支间的冲突。
- **scf.while**：while 循环的 before/after 区域需要特殊处理。
- **cf.cond_br**：基本块分支中的缓冲区传递。

### 4. 子视图与形状变换

PlanMemory 支持 `memref.subview`、`memref.collapse_shape`、`memref.expand_shape`、`memref.reshape`、`memref.view`、`memref.reinterpret_cast` 等操作，正确追踪子视图与基础分配的关系。

### 5. Inplace 类型判断

PlanMemory 对不同操作类型有不同的 Inplace 策略：

- **vcast（类型转换）**：当源和目标类型大小相同时可以 Inplace；大小不同时需要独立偏移。
- **带 broadcast 的操作**：需要额外的 temp_buffer，Inplace 策略取决于 broadcast 维度。
- **带 temp_buffer 的操作**：temp_buffer 与主缓冲区独立分配。

## 内存溢出检测

PlanMemory 在分配完成后检测 UB 是否溢出。Ascend910B 系列的 UB 大小为 1572864 bits (192KB)。

```mlir
// expected-error {{ub overflow, requires 2560000 bits while 1572864 bits available!}}
func.func @test_one_mem_not_enough(%arg0_gm : memref<80000xf32, #hivm.address_space<gm>>, ...) {
  %arg0_ub = memref.alloc() : memref<80000xf32, #hivm.address_space<ub>>
  ...
}
```

## MemPlanMode 枚举

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `LOCAL_MEM_PLAN` | 0 | 为 `memref.alloc` 分配局部内存偏移 |
| `GLOBAL_WORKSPACE_PLAN` | 1 | 为 `memref_ext.alloc_workspace` 分配全局工作空间偏移 |

源码参考：[HIVMAttrs.td:L770-L777](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L770-L777)

## 相关 Pass

### hivm-normalize-loop-iterator

- **Pass 名**：`hivm-normalize-loop-iterator`
- **功能**：在 PlanMemory 之前规范化循环迭代器的特殊状态，确保内存规划的正确性。
- **依赖方言**：`hivm::HIVMDialect`, `scf::SCFDialect`
- **源码参考**：[Passes.td:L653-L658](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L653-L658)

## 源码参考

- Pass 定义：[Passes.td:L130-L161](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L130-L161)
- MemPlanMode 枚举：[HIVMAttrs.td:L770-L777](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L770-L777)
- 内存规划测试：[plan-memory.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/plan-memory.mlir)
