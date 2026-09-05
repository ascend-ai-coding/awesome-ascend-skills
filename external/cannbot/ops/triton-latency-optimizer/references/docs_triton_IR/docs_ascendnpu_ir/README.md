# AscendNPU-IR 知识库

## 定位说明

本知识库是 **AscendNPU-IR** 项目的 IR 层文档，聚焦于编译器中间表示的语义、结构和变换规则。

与 [docs_triton_ascend](../docs_triton_ascend/) 的关系：
- **docs_triton_ascend**：面向 Triton 开发者，侧重 Triton 语言到 NPU 的端到端编译流程和使用方法
- **docs_ascendnpu_ir**（本库）：面向编译器开发者，侧重 IR 的精确定义、操作语义和变换算法

两者互补：docs_triton_ascend 回答"如何使用"，docs_ascendnpu_ir 回答"如何实现"。

## 目录结构

```
docs_ascendnpu_ir/
├── 00-Architecture/                    # NPU 硬件架构
│   ├── 01-npu-hardware-overview.md     # NPU 硬件总览
│   ├── 02-memory-hierarchy.md          # 内存层级体系
│   ├── 03-pipeline-execution-model.md  # 流水线执行模型
│   └── 04-data-layout.md              # 数据布局
├── 01-HIVM-Dialect/                    # HIVM 方言（核心）
│   ├── 00-overview.md                  # HIVM 总览
│   ├── 01-DMA-Operations/             # DMA 操作
│   │   ├── 00-overview.md
│   │   ├── 01-load.md
│   │   ├── 02-store.md
│   │   ├── 03-nd2nz.md
│   │   ├── 04-nz2nd.md
│   │   ├── 05-copy.md
│   │   ├── 06-fixpipe.md
│   │   ├── 07-atomic.md
│   │   ├── 08-gather-scatter.md
│   │   ├── 09-indirect-access.md
│   │   └── 10-padding-quantization.md
│   ├── 02-Vector-Operations/          # 向量操作
│   │   ├── 00-overview.md
│   │   ├── 01-unary-ops.md
│   │   ├── 02-binary-ops.md
│   │   ├── 03-ternary-ops.md
│   │   ├── 04-cast-ops.md
│   │   ├── 05-compare-ops.md
│   │   ├── 06-shift-ops.md
│   │   ├── 07-reduction-ops.md
│   │   ├── 08-data-movement.md
│   │   ├── 09-cumulative-sort.md
│   │   ├── 10-special-ops.md
│   │   └── 11-scalar-lowering.md
│   ├── 03-Macro-Operations/           # 宏操作（矩阵乘法）
│   │   ├── 00-overview.md
│   │   ├── 01-mmad-l1.md
│   │   ├── 02-batch-mmad-l1.md
│   │   ├── 03-matmul.md
│   │   ├── 04-mix-matmul.md
│   │   └── 05-mix-group-matmul.md
│   ├── 04-Synchronization/            # 同步操作
│   │   ├── 00-overview.md
│   │   ├── 01-pipe-sync.md
│   │   ├── 02-block-sync.md
│   │   ├── 03-unit-flag.md
│   │   └── 04-sync-injection.md
│   ├── 05-Custom-Operations/          # 自定义操作
│   │   ├── 00-overview.md
│   │   ├── 01-custom-op.md
│   │   └── 02-custom-macro-op.md
│   ├── 06-Attributes-Types/           # 属性与类型
│   │   ├── 00-overview.md
│   │   ├── 01-enumerations.md
│   │   ├── 02-parameterized-attrs.md
│   │   ├── 03-type-system.md
│   │   ├── 04-interfaces.md
│   │   └── 05-traits.md
│   └── 07-Intrin-Operations/          # 内置指令操作
│       └── 00-overview.md
├── 02-HACC-Dialect/                    # HACC 方言
│   ├── 00-overview.md
│   ├── 01-function-management.md
│   ├── 02-device-specification.md
│   ├── 03-kernel-args.md
│   ├── 04-host-device-binding.md
│   └── 05-transforms.md
├── 03-HFusion-Dialect/                 # HFusion 方言
│   ├── 00-overview.md
│   ├── 01-elementwise-ops.md
│   ├── 02-reduction-ops.md
│   ├── 03-matmul-ops.md
│   ├── 04-data-movement-ops.md
│   ├── 05-memory-ops.md
│   ├── 06-special-ops.md
│   ├── 07-attributes-enums.md
│   └── 08-transforms.md
├── 04-Other-Dialects/                  # 其他方言
│   ├── 01-scope-dialect.md
│   ├── 02-symbol-dialect.md
│   ├── 03-memrefext-dialect.md
│   ├── 04-mathext-dialect.md
│   ├── 05-hmap-dialect.md
│   ├── 06-annotation-dialect.md
│   └── 07-ascend-dpx-dialect.md
├── 05-Triton-Dialects/                 # Triton 相关方言
│   ├── 00-overview.md
│   ├── 01-triton-ops.md
│   ├── 02-triton-types-attrs.md
│   ├── 03-tritongpu-ops.md
│   ├── 04-tritongpu-encodings.md
│   └── 05-gluon-dialect.md
├── 06-Compilation-Pipeline/            # 编译管线
│   ├── 00-overview.md
│   ├── 01-frontend-to-hfusion.md
│   ├── 02-hfusion-transforms.md
│   ├── 03-hfusion-to-hivm.md
│   ├── 04-hivm-transforms.md
│   ├── 05-hivm-to-backend.md
│   ├── 06-triton-ir-compilation.md
│   ├── 07-pipeline-options.md
│   └── 08-pass-dependencies.md
├── 07-Memory-Management/               # 内存管理
│   ├── 00-overview.md
│   ├── 01-bufferization.md
│   ├── 02-memory-planning.md
│   ├── 03-multi-buffer.md
│   ├── 04-extra-buffer.md
│   ├── 05-memory-alignment.md
│   ├── 06-tightly-coupled-buffer.md
│   └── 07-workspace-management.md
└── 08-Reference/                       # 参考信息
    ├── 01-ir-operation-index.md
    ├── 02-attribute-enum-index.md
    ├── 03-hardware-specs-ir.md
    └── 04-glossary.md
```

## 按方言快速导航

| 方言 | 说明 | 入口文档 |
|------|------|----------|
| HIVM | NPU 底层虚拟机 IR，直接映射硬件操作 | [01-HIVM-Dialect/00-overview.md](01-HIVM-Dialect/00-overview.md) |
| HACC | 主机-设备交互和硬件规格标注 | [02-HACC-Dialect/00-overview.md](02-HACC-Dialect/00-overview.md) |
| HFusion | 混合融合 IR，高层可融合操作 | [03-HFusion-Dialect/00-overview.md](03-HFusion-Dialect/00-overview.md) |
| Scope | 作用域管理 | [04-Other-Dialects/01-scope-dialect.md](04-Other-Dialects/01-scope-dialect.md) |
| Symbol | 符号化维度管理 | [04-Other-Dialects/02-symbol-dialect.md](04-Other-Dialects/02-symbol-dialect.md) |
| MemRefExt | MemRef 扩展（workspace 分配） | [04-Other-Dialects/03-memrefext-dialect.md](04-Other-Dialects/03-memrefext-dialect.md) |
| MathExt | 数学扩展操作 | [04-Other-Dialects/04-mathext-dialect.md](04-Other-Dialects/04-mathext-dialect.md) |
| HMAP | 多处理器通信 | [04-Other-Dialects/05-hmap-dialect.md](04-Other-Dialects/05-hmap-dialect.md) |
| Annotation | 编译器标注 | [04-Other-Dialects/06-annotation-dialect.md](04-Other-Dialects/06-annotation-dialect.md) |
| AscendDPX | SIMT 设备操作 | [04-Other-Dialects/07-ascend-dpx-dialect.md](04-Other-Dialects/07-ascend-dpx-dialect.md) |
| Triton | Triton 语言操作 | [05-Triton-Dialects/01-triton-ops.md](05-Triton-Dialects/01-triton-ops.md) |
| TritonGPU | Triton 后端操作 | [05-Triton-Dialects/03-tritongpu-ops.md](05-Triton-Dialects/03-tritongpu-ops.md) |
| Gluon | Triton 布局推导 | [05-Triton-Dialects/05-gluon-dialect.md](05-Triton-Dialects/05-gluon-dialect.md) |

## 按场景查询指南

### 场景 1：理解某个 HIVM 操作的语义和参数

查阅 [01-HIVM-Dialect/](01-HIVM-Dialect/00-overview.md)，按操作类型选择子目录：
- DMA 操作（load/store/copy/fixpipe） → [01-DMA-Operations/](01-HIVM-Dialect/01-DMA-Operations/00-overview.md)
- 向量操作（vadd/vmul/vcast...） → [02-Vector-Operations/](01-HIVM-Dialect/02-Vector-Operations/00-overview.md)
- 矩阵乘法（mmadL1/matmul/mix_matmul） → [03-Macro-Operations/](01-HIVM-Dialect/03-Macro-Operations/00-overview.md)
- 同步操作（pipe_barrier/sync_block/set_flag） → [04-Synchronization/](01-HIVM-Dialect/04-Synchronization/00-overview.md)

### 场景 2：查找某个操作名对应的文档

查阅 [08-Reference/01-ir-operation-index.md](08-Reference/01-ir-operation-index.md)，按操作名搜索。

### 场景 3：查找某个枚举属性的所有可能值

查阅 [08-Reference/02-attribute-enum-index.md](08-Reference/02-attribute-enum-index.md)，按枚举名搜索。

### 场景 4：了解 NPU 型号的硬件规格参数

查阅 [08-Reference/03-hardware-specs-ir.md](08-Reference/03-hardware-specs-ir.md)，按型号搜索。

### 场景 5：理解内存管理流程（Bufferization → PlanMemory → 物理地址）

查阅 [07-Memory-Management/00-overview.md](07-Memory-Management/00-overview.md) 获取总览，然后按需深入：
- Tensor → MemRef 转换 → [01-bufferization.md](07-Memory-Management/01-bufferization.md)
- 物理偏移分配 → [02-memory-planning.md](07-Memory-Management/02-memory-planning.md)
- 流水线多缓冲区 → [03-multi-buffer.md](07-Memory-Management/03-multi-buffer.md)
- Cube-Vector 数据传递 → [06-tightly-coupled-buffer.md](07-Memory-Management/06-tightly-coupled-buffer.md)

### 场景 6：理解编译管线的 Pass 执行顺序

查阅 [06-Compilation-Pipeline/08-pass-dependencies.md](06-Compilation-Pipeline/08-pass-dependencies.md) 获取 Pass 依赖关系。

### 场景 7：理解 Triton IR 如何编译到 HIVM IR

查阅 [06-Compilation-Pipeline/06-triton-ir-compilation.md](06-Compilation-Pipeline/06-triton-ir-compilation.md)。

### 场景 8：查找某个技术术语的中英文对照

查阅 [08-Reference/04-glossary.md](08-Reference/04-glossary.md)。

### 场景 9：了解哪些向量操作会被标量降级及优化建议

查阅 [01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md](01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)，包含 15 个操作的标量降级条件、判断矩阵和 Triton 算子优化速查。

### 场景 10：了解 Reg-based 与 Mem-based 架构的区别

查阅 [00-Architecture/01-npu-hardware-overview.md](00-Architecture/01-npu-hardware-overview.md) 中的「架构分类：Reg-based 与 Mem-based」章节，包含 SIMT 支持差异、同步机制差异、编译器行为差异。

## 与 docs_triton_ascend 的交叉引用映射

| docs_triton_ascend 主题 | docs_ascendnpu_ir 对应文档 |
|-------------------------|---------------------------|
| NPU 硬件架构概述 | [00-Architecture/01-npu-hardware-overview.md](00-Architecture/01-npu-hardware-overview.md) |
| 内存层级 | [00-Architecture/02-memory-hierarchy.md](00-Architecture/02-memory-hierarchy.md) |
| 数据布局（ND/NZ/Fractal） | [00-Architecture/04-data-layout.md](00-Architecture/04-data-layout.md) |
| Triton 到 NPU 编译流程 | [06-Compilation-Pipeline/06-triton-ir-compilation.md](06-Compilation-Pipeline/06-triton-ir-compilation.md) |
| 内存管理（Bufferization/PlanMemory） | [07-Memory-Management/00-overview.md](07-Memory-Management/00-overview.md) |
| 流水线同步（Pipe Barrier/Unit Flag） | [01-HIVM-Dialect/04-Synchronization/00-overview.md](01-HIVM-Dialect/04-Synchronization/00-overview.md) |
| 向量操作标量降级与性能优化 | [01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md](01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md) |
| Reg-based/Mem-based 架构分类 | [00-Architecture/01-npu-hardware-overview.md](00-Architecture/01-npu-hardware-overview.md) |
| 硬件规格参数 | [08-Reference/03-hardware-specs-ir.md](08-Reference/03-hardware-specs-ir.md) |
| 术语对照 | [08-Reference/04-glossary.md](08-Reference/04-glossary.md) |

## 文档模板说明

本知识库的文档遵循以下模板结构：

### 操作文档模板
1. 概述 — 操作的简要说明
2. 操作定义 — 从 .td 文件提取的 TableGen 定义
3. 操作数与结果 — 参数表（名称、类型、约束）
4. 属性 — 操作的属性列表
5. IR 示例 — MLIR assembly 格式的具体示例
6. 约束与限制 — 硬件约束、类型约束等
7. 源码参考 — 绝对路径链接

### Pass 文档模板
1. 概述 — Pass 的简要说明
2. Pass 定义 — 名称、作用域、构造函数
3. 选项 — 命令行选项表
4. 算法 — 核心算法描述
5. IR 变换示例 — 变换前后的 MLIR 对比
6. 源码参考 — 绝对路径链接
