# 术语表

本术语表覆盖 AscendNPU-IR 项目的核心概念，提供中英文对照和简要说明。

## 硬件架构

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| AI Core | AI 核心 | Ascend NPU 的基本计算单元，包含 Cube Core 和 Vector Core |
| Cube Core | Cube 核心（矩阵计算单元） | 执行矩阵乘法（MAD）运算的核心，对应 HIVM 的 CUBE CoreType |
| Vector Core | Vector 核心（向量计算单元） | 执行元素级向量运算的核心，对应 HIVM 的 VECTOR CoreType |
| AI Vector (AIV) | 纯向量 AI Core | 仅包含 Vector Core 的 AI Core，不包含 Cube Core |
| Mix | 混合模式 | Cube Core 和 Vector Core 协同工作的模式，对应 HIVM 的 MIX CoreType |
| SubBlock | 子块 | AI Core 内的逻辑分区，用于 Mix CV 场景下的并行执行 |
| CoreDim | 核心维度 | 描述核心数量的维度参数 |

## 内存体系

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| Memory Space | 地址空间 | NPU 内存的逻辑分区，如 GM/UB/L1/L0A/L0B/L0C |
| GM (Global Memory) | 全局内存 | 所有 AI Core 共享的外部存储，对应 AddressSpace.GM |
| UB (Unified Buffer) | 统一缓冲区 | Vector Core 使用的片上存储，对应 AddressSpace.UB |
| L1 (Cube Buffer / CBUF) | L1 缓冲区 | Cube Core 使用的片上存储，对应 AddressSpace.L1 |
| L0A | L0A 缓冲区 | 矩阵乘法输入 A 的缓冲区，对应 AddressSpace.L0A |
| L0B | L0B 缓冲区 | 矩阵乘法输入 B 的缓冲区，对应 AddressSpace.L0B |
| L0C | L0C 缓冲区 | 矩阵乘法累加结果缓冲区，对应 AddressSpace.L0C |
| DCache | 数据缓存 | SIMT VF 模式下的数据缓存，Ascend950 系列支持 |
| FFTS (Fast Thread Schedule System) | 快速线程调度系统 | 硬件级线程调度机制，通过 set_ffts_base_addr 配置 |

## 数据布局

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| Data Layout | 数据布局 | 数据在内存中的排列方式，如 ND/nZ/zN/Fractal |
| ND Layout | ND 布局 | 标准 N-D 维度布局，行优先排列 |
| nZ Layout | nZ 布局 | NPU 特有的分块布局，按 Z 形排列 |
| zN Layout | zN 布局 | NPU 特有的分块布局，按 Z 形转置排列 |
| Fractal Layout | Fractal 布局 | 矩阵乘法专用的分形布局 |
| DOTA_ND / DOTB_ND / DOTC_ND | 矩阵乘法输入/输出布局 | 矩阵乘法操作数 A/B/C 的专用 ND 布局 |

## 编译器 IR

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| IR (Intermediate Representation) | 中间表示 | 编译器内部的程序表示形式 |
| Dialect | 方言 | MLIR 中一组相关操作和属性的集合，如 HIVM/HACC/HFusion |
| Operation (Op) | 操作 | IR 中的基本计算单元，如 hivm.hir.vadd |
| Attribute (Attr) | 属性 | 编译期已知的常量信息，附加在操作或值上 |
| Type | 类型 | 值的数据类型，如 memref/tensor/i32/f16 |
| Trait | 特征 | 操作的编译期属性标记，如 Pure/SinglePipeOpTrait |
| Interface | 接口 | 操作必须实现的方法集合，如 BufferizableOpInterface |
| MemRef | 内存引用 | MLIR 的缓冲区类型，表示对内存区域的引用 |
| Tensor | 张量 | MLIR 的张量类型，值语义，不可变 |
| SSA (Static Single Assignment) | 静态单赋值 | 每个变量只赋值一次的 IR 形式 |

## 方言名称

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| HIVM (Huawei Intermediate Virtual Machine) | HIVM 方言 | NPU 底层虚拟机 IR，直接映射硬件操作 |
| HACC (Huawei Accelerator) | HACC 方言 | 主机-设备交互和硬件规格标注 |
| HFusion | HFusion 方言 | 混合融合 IR，高层可融合操作表示 |
| Scope | Scope 方言 | 作用域管理，定义局部变量作用域 |
| Symbol | Symbol 方言 | 符号化维度管理，支持动态形状 |
| MemRefExt | MemRefExt 方言 | MemRef 扩展操作，如 workspace 分配 |
| MathExt | MathExt 方言 | 数学扩展操作，如 ilogb/ldexp |
| HMAP (Huawei Multi-Processor Access Protocol) | HMAP 方言 | 多处理器通信操作，如 all_to_all_v |
| Annotation | Annotation 方言 | 编译器标注操作，如 annotation.mark |
| AscendDPX | AscendDPX 方言 | SIMT 编程模型的设备操作 |
| Triton | Triton 方言 | Triton 编程语言的操作表示 |
| TritonGPU | TritonGPU 方言 | Triton 的 GPU/NPU 后端操作 |
| Gluon | Gluon 方言 | Triton 布局自动推导 |

## 内存管理

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| Bufferization | 缓冲区化 | 将 Tensor IR 转换为 MemRef IR 的过程 |
| Memory Planning | 内存规划 | 为逻辑缓冲区分配物理内存偏移的过程 |
| Inplace | 原地复用 | 多个缓冲区共享同一物理内存的优化 |
| Multi-Buffer | 多缓冲区 | 为同一逻辑缓冲区分配多个物理实例，实现流水线重叠 |
| Double Buffer | 双缓冲区 | Multi-Buffer 的特例，分配 2 个物理实例 |
| Tightly Coupled Buffer | 紧耦合缓冲区 | Cube-Vector 直接数据传递的缓冲区，避免 GM 中转 |
| Extra Buffer | 额外缓冲区 | 操作执行所需的临时缓冲区（temp_buffer） |
| Workspace | 工作空间 | 跨函数共享的全局内存分配区域 |
| Memory Alignment | 内存对齐 | 确保缓冲区分配和 stride 满足硬件对齐约束 |
| Stride Alignment | 步幅对齐 | 确保 memref 的 stride 满足硬件访问对齐要求 |

## 流水线与同步

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| Pipeline | 流水线 | NPU 的指令执行流水线，如 S/V/M/MTE1/MTE2/MTE3/FIX |
| Pipe Barrier | 流水线屏障 | 流水线级间的同步操作 |
| Unit Flag | 单元标志 | 硬件级事件通知机制，用于流水线同步 |
| Event | 事件 | 同步事件的标识符，EVENT_ID0-7 |
| Sync Block | 同步块 | 跨 AI Core 的同步机制 |
| Block Sync | 块同步 | AI Core 间的同步操作 |
| CV Pipelining | Cube-Vector 流水线化 | Cube 和 Vector 核心的流水线调度 |

## 编译管线

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| VFMode (Vector Function Mode) | 向量函数模式 | VF 的执行模式：SIMD/SIMT/MIX |
| SIMD | 单指令多数据 | 向量函数的 SIMD 执行模式 |
| SIMT | 单指令多线程 | 向量函数的 SIMT 执行模式，类似 GPU 编程模型 |
| Tiling | 切分 | 将大计算任务分解为适合硬件的小任务 |
| Fixpipe | 定点管道 | L0C 到 UB 的数据搬运和类型转换操作 |
| ND2NZ / NZ2ND | ND 与 NZ 布局转换 | 标准 ND 布局与 NPU 分块 NZ 布局之间的转换 |
| Pointer Cast | 指针转换 | PlanMemory 后将 memref.alloc 替换为物理偏移的操作 |

## 数据类型与操作

| 英文术语 | 中文术语 | 说明 |
|----------|----------|------|
| Structured Op | 结构化操作 | 使用 iterator_type 描述迭代语义的 HIVM 操作 |
| Macro Op | 宏操作 | 封装复杂计算模式的 HIVM 操作，如 mmadL1/matmul |
| Elementwise Op | 逐元素操作 | 对每个元素独立执行相同计算的操作 |
| Reduction Op | 归约操作 | 沿指定维度聚合元素的操作 |
| Broadcast | 广播 | 将小维度数据扩展到大维度的操作 |
| VCast | 向量类型转换 | Vector Core 上的数据类型转换操作 |
| Descale | 反缩放 | 量化矩阵乘法中的反量化操作 |
