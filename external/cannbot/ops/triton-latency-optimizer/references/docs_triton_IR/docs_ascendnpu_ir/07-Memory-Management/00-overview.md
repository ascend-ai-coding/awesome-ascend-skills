# 内存管理子系统架构

## 概述

AscendNPU-IR 的内存管理子系统负责将高层 Tensor 抽象逐步映射到 NPU 硬件的物理地址空间。整个流程经历了从逻辑 Tensor 到 MemRef 再到物理地址的完整变换链，是编译管线中最核心的子系统之一。

## 变换流程总览

```
Tensor (高层抽象)
    │
    ▼  Bufferization (Tensor → MemRef)
MemRef + memref.alloc (逻辑缓冲区)
    │
    ▼  Memory Scope Inference (推断地址空间)
MemRef + #hivm.address_space<ub/cbuf/cc/...>
    │
    ▼  Memory Planning (PlanMemory, 分配物理偏移)
MemRef + hivm.hir.pointer_cast(offset)
    │
    ▼  Memory Alignment (对齐约束)
Aligned MemRef
    │
    ▼  Backend Lowering
物理地址访问
```

## 核心阶段

### 1. Bufferization — Tensor 到 MemRef 的转换

将基于 Tensor 的 SSA 值转换为基于 MemRef 的缓冲区操作。此阶段消除了 Tensor 语义，引入 `memref.alloc` 分配逻辑缓冲区。

详见 [01-bufferization.md](01-bufferization.md)

### 2. 内存作用域推断 — 地址空间分配

`hivm-infer-mem-scope` Pass 为每个 `memref.alloc` 推断其应位于的地址空间（UB/L1/L0C 等），通过 `#hivm.address_space<...>` 属性标注。

### 3. 内存规划 — 物理偏移分配

`hivm-plan-memory` Pass 是内存管理的核心，负责将逻辑缓冲区映射到物理内存偏移。支持两种模式：
- **local-mem-plan**：为 `memref.alloc` 分配局部内存偏移
- **global-workspace-plan**：为 `memref_ext.alloc_workspace` 分配全局工作空间偏移

详见 [02-memory-planning.md](02-memory-planning.md)

### 4. 多缓冲区 — 流水线优化

通过 `hivm-mark-multi-buffer` 和 `hivm-enable-multi-buffer` Pass 实现双/多缓冲区，使 DMA 传输与计算重叠，提升流水线效率。

详见 [03-multi-buffer.md](03-multi-buffer.md)

### 5. 额外缓冲区 — 临时缓冲区分配

`hivm-alloc-extra-buffer` 和 `hivm-outline-alloc-in-VF` Pass 为需要额外临时存储的操作分配缓冲区。

详见 [04-extra-buffer.md](04-extra-buffer.md)

### 6. 内存对齐 — 硬件约束满足

`hivm-align-alloc-size`、`hivm-mark-stride-align`、`hivm-enable-stride-align`、`hivm-lift-lowest-stride` Pass 确保内存访问满足硬件对齐要求。

详见 [05-memory-alignment.md](05-memory-alignment.md)

### 7. 紧耦合缓冲区 — Cube-Vector 数据传递

`hivm-insert-cv-tight-coupled-buffer` Pass 在 Mix CV 场景下插入紧耦合缓冲区，实现 Cube 和 Vector 核心间的高效数据传递。

详见 [06-tightly-coupled-buffer.md](06-tightly-coupled-buffer.md)

### 8. 工作空间管理 — 全局内存分配

`hivm-insert-infer-workspace-size-func`、`hivm-bind-workspace-arg`、`insert-workspace-for-mix-cv` Pass 管理全局工作空间的分配、推断和绑定。

详见 [07-workspace-management.md](07-workspace-management.md)

## 地址空间体系

HIVM 定义了以下地址空间，对应 NPU 硬件的实际存储层级：

| 地址空间 | 枚举值 | MLIR 表示 | 说明 |
|----------|--------|-----------|------|
| Zero | 0 | `#hivm.address_space<zero>` | 默认/未指定 |
| GM | 1 | `#hivm.address_space<gm>` | 全局内存 |
| L1 | 2 | `#hivm.address_space<cbuf>` | L1 缓冲区（Cube Buffer） |
| L0A | 3 | `#hivm.address_space<ca>` | L0A 缓冲区（矩阵 A） |
| L0B | 4 | `#hivm.address_space<cb>` | L0B 缓冲区（矩阵 B） |
| L0C | 5 | `#hivm.address_space<cc>` | L0C 缓冲区（矩阵 C 累加器） |
| UB | 6 | `#hivm.address_space<ub>` | 统一缓冲区（Unified Buffer） |

源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L188)

## 内存规划中的 Inplace 复用

PlanMemory Pass 的核心优化之一是 Inplace 复用——当两个缓冲区的生命周期不重叠时，它们可以共享同一块物理内存。从 [plan-memory.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/plan-memory.mlir) 测试用例可以看到：

- 基本线性分配：`pointer_cast(0)`, `pointer_cast(8192)`, ... 按顺序分配偏移
- Inplace 复用：生命周期不重叠的缓冲区共享偏移，如 `pointer_cast(0)` 被多个缓冲区复用
- 循环内冲突：循环内分配的缓冲区需要独立偏移，不能与循环外缓冲区复用
- scf.if/scf.for/scf.while 中的缓冲区：需要特殊处理 yield 操作数的内存规划

## 内存溢出检测

PlanMemory Pass 会在分配完成后检测 UB 是否溢出。例如：

```mlir
// expected-error@+1 {{ub overflow, requires 2560000 bits while 1572864 bits available!}}
func.func @test_one_mem_not_enough(%arg0_gm : memref<80000xf32, #hivm.address_space<gm>>, ...) {
  %arg0_ub = memref.alloc() : memref<80000xf32, #hivm.address_space<ub>>
  ...
}
```

Ascend910B 系列的 UB 大小为 1572864 bits (192KB)，当所需总内存超过此限制时会报错。

## 源码参考

- Pass 定义：[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td)
- 属性定义：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 内存规划测试：[plan-memory.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/plan-memory.mlir)
- 硬件规格定义：[NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
