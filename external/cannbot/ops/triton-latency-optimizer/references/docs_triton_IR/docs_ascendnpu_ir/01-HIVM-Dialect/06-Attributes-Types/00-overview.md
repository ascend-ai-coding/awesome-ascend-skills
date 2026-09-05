# HIVM 属性类型系统总览

> 关键词：Attributes, Types, Enumerations, DataLayout, AddressSpace, Pipe, CoreType

## 概述

HIVM 属性类型系统是 AscendNPU-IR 中描述硬件特性、存储层次、执行单元等关键信息的核心机制。通过 MLIR 的属性系统，HIVM 将 NPU 的硬件约束编码到 IR 中，使编译器能够在类型检查和 Lowering 阶段正确处理这些约束。

HIVM 属性类型系统分为以下几类：

1. **枚举属性**：描述有限的离散值集合，如 IteratorType、DataLayout、AddressSpace、Pipe 等
2. **参数化属性**：携带额外参数的复合属性，如 DataLayoutAttr（含 transpose/fractalSizes）、AddressSpaceAttr、BlockMappingAttr 等
3. **类型约束**：MemRef/Tensor/Vector 在 HIVM 中的特殊约束
4. **接口**：HIVMStructuredOpInterface、HIVMCoreTypeInterface、HIVMUnitFlagEnabledInterface 等
5. **Trait**：操作行为约束，如 MacroOpTrait、CoreTypeTrait、ElementwiseNaryOpTrait 等

## 属性分类

### 枚举属性（30+ 个）

枚举属性是 HIVM 中最常用的属性类型，每个枚举对应一个硬件特性或操作模式。详见 [01-enumerations.md](01-enumerations.md)。

### 参数化属性

| 属性 | 助记符 | 参数 | 说明 |
|------|--------|------|------|
| DataLayoutAttr | `data_layout` | data_layout, transpose?, fractalSizes? | 数据布局映射 |
| AddressSpaceAttr | `address_space` | address_space | 地址空间映射 |
| PipeAttr | `pipe` | pipe | Pipeline 标识 |
| TCoreTypeAttr | `tcore_type` | tcoretype | 操作 Core 类型 |
| TFuncCoreTypeAttr | `func_core_type` | funcCoreType | 函数 Core 类型 |
| TModuleCoreTypeAttr | `module_core_type` | moduleCoreType | 模块 Core 类型 |
| EventAttr | `event` | event | Event ID |
| UnitFlagAttr | `unit_flag` | unit_flag | UnitFlag 模式 |
| SyncBlockModeAttr | `sync_block_mode` | sync_mode | 同步 Block 模式 |
| SyncBlockInstrModeAttr | `sync_block_instr_mode` | sync_instr_mode | 同步指令模式 |
| ReduceOpAttr | `reduce_op` | reduce_op | 归约操作类型 |
| BlockMappingAttr | `block` | order? | Block 映射 |
| SubBlockMappingAttr | `sub_block` | sub_block | Sub-Block 映射 |

详见 [02-parameterized-attrs.md](02-parameterized-attrs.md)。

### 标记属性

| 属性 | 助记符 | 说明 |
|------|--------|------|
| MultiBufferAttr | `multi_buffer` | 多缓冲标记 |
| TPartOfMixAttr | `part_of_mix` | Mix Kernel 组成部分 |
| VFAttr | `vector_function` | Vector 函数标记 |
| HasAliaScopesAttr | `has_alias_scopes` | 别名作用域标记 |
| TightlyCoupledBufferAttr | `tightly_coupled_buffer` | 紧耦合缓冲区（含 id 参数） |
| MemUniqueAttr | `mem_unique` | 唯一内存规划 |
| ParallelLoopAttr | `parallel_loop` | 可并行化循环标记 |
| UnlikelyConditionAttr | `unlikely_condition` | 不可能条件标记 |
| SharedMemoryAttr | `shared_memory` | SIMT VF 共享内存 |

## 文档列表

| 文档 | 内容 |
|------|------|
| [01-enumerations.md](01-enumerations.md) | 所有枚举属性速查（完整枚举值列表） |
| [02-parameterized-attrs.md](02-parameterized-attrs.md) | 参数化属性详细定义 |
| [03-type-system.md](03-type-system.md) | MemRef/Tensor/Vector 在 HIVM 中的约束 |
| [04-interfaces.md](04-interfaces.md) | HIVM 接口方法列表 |
| [05-traits.md](05-traits.md) | 所有 Trait 定义和适用操作 |

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 测试用例：[attribute.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/attribute.mlir)
