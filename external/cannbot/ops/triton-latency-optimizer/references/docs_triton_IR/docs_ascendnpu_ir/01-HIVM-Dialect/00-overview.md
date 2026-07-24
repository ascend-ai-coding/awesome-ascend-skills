# HIVM 方言总览

> 关键词：HIVM、Hybrid Intelligence Virtual Machine、方言定义、操作分类

## 概述

HIVM（Hybrid Intelligence Virtual Machine，混合智能虚拟机）是 AscendNPU-IR 项目中的核心 IR 方言，用于描述华为 Ascend NPU 上的计算操作。HIVM 方言定义在 BiShengIR 编译栈中，作为硬件无关与硬件相关 IR 之间的桥梁，承载了从高级语义到低级硬件指令的映射。

HIVM 方言的设计目标是：
- 提供统一的 IR 抽象，覆盖 Ascend NPU 的 Cube（矩阵计算）、Vector（向量计算）和 Mix（混合计算）三种核心类型
- 支持 DMA 数据搬运、向量计算、矩阵乘法宏操作、同步控制、自定义操作和底层内建指令
- 通过 Pipeline 和 Address Space 属性精确描述硬件资源约束

## 方言定义

### 基本属性

| 属性 | 值 |
|------|------|
| 方言名称 | `hivm` |
| C++ 命名空间 | `::mlir::hivm` |
| 操作前缀 | `hir.` |
| TableGen 定义文件 | [HIVMBase.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td) |

### 依赖方言

HIVM 方言声明了以下依赖方言（定义于 [HIVMBase.td:L33-L37](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L33-L37)）：

| 依赖方言 | 用途 |
|----------|------|
| `arith::ArithDialect` | 算术运算基础 |
| `bishengir::memref_ext::MemRefExtDialect` | MemRef 扩展操作 |
| `math::MathDialect` | 数学运算 |
| `memref::MemRefDialect` | 内存引用操作 |
| `hacc::HACCDialect` | 硬件加速器配置 |
| `tensor::TensorDialect` | 张量操作 |

### 操作基类

HIVM 定义了两个核心操作基类（[HIVMBase.td:L52-L68](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L52-L68)）：

- **`HIVM_Op`**：基础操作基类，所有 HIVM 操作的前缀为 `hir.`
- **`HIVM_StructuredOp`**：结构化操作基类，继承 `HIVM_Op` 并实现 `HIVMStructuredOpInterface`、`MemoryEffectsOpInterface`、`FlattenInterface`、`LibraryFunctionOpInterface` 等接口

## 操作分类总表

### DMA 操作（数据搬运）

定义文件：[HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)

| 操作名 | IR 语法 | 一句话描述 |
|--------|---------|-----------|
| `LoadOp` | `hir.load` | 从 GM 加载数据到本地缓冲区（UB） |
| `StoreOp` | `hir.store` | 从本地缓冲区（UB）存储数据到 GM |
| `CopyOp` | `hir.copy` | 本地内存层级间的数据拷贝 |
| `FixpipeOp` | `hir.fixpipe` | L0C 到其他内存层级的搬运，支持随路量化/激活 |
| `ND2NZOp` | `hir.nd2nz` | ND 到 NZ 布局转换的数据搬运 |
| `NZ2NDOp` | `hir.nz2nd` | L1 到 GM 的 NZ 到 ND 布局转换搬运 |
| `L12UBOp` | `hir.l12ub` | L1 到 UB 的数据搬运 |
| `AtomicCasOp` | `hir.atomic_cas` | 原子比较并交换操作 |
| `AtomicXchgOp` | `hir.atomic_xchg` | 原子交换操作 |

### 向量操作

定义文件：[HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td)

| 操作名 | IR 语法 | 一句话描述 |
|--------|---------|-----------|
| `VExpOp` | `hir.vexp` | 逐元素指数运算 |
| `VAbsOp` | `hir.vabs` | 逐元素绝对值运算 |
| `VLnOp` | `hir.vln` | 逐元素自然对数运算 |
| `VReluOp` | `hir.vrelu` | 逐元素 ReLU 激活 |
| `VRsqrtOp` | `hir.vrsqrt` | 逐元素平方根倒数 |
| `VSqrtOp` | `hir.vsqrt` | 逐元素平方根 |
| `VTanhOp` | `hir.vtanh` | 逐元素双曲正切 |
| `VSinOp` | `hir.vsin` | 逐元素正弦 |
| `VCosOp` | `hir.vcos` | 逐元素余弦 |
| `VErfOp` | `hir.verf` | 逐元素误差函数 |
| `VRecOp` | `hir.vrec` | 逐元素倒数 |
| `VNotOp` | `hir.vnot` | 逐元素取反 |
| `VCastOp` | `hir.vcast` | 逐元素类型转换 |
| `VAddOp` | `hir.vadd` | 逐元素加法 |
| `VMulOp` | `hir.vmul` | 逐元素乘法 |
| `VMulExtOp` | `hir.vmulext` | 逐元素乘法取高 32 位 |
| `VSubOp` | `hir.vsub` | 逐元素减法 |
| `VDivOp` | `hir.vdiv` | 逐元素除法 |
| `VMaxOp` | `hir.vmax` | 逐元素最大值 |
| `VMinOp` | `hir.vmin` | 逐元素最小值 |
| `VOrOp` | `hir.vor` | 逐元素或运算 |
| `VAndOp` | `hir.vand` | 逐元素与运算 |
| `VXorOp` | `hir.vxor` | 逐元素异或运算 |
| `VModOp` | `hir.vmod` | 逐元素取模 |
| `VModUIOp` | `hir.vmodui` | 逐元素无符号取模 |
| `VShLOp` | `hir.vshl` | 逐元素左移 |
| `VShROp` | `hir.vshr` | 逐元素右移 |
| `VCmpOp` | `hir.vcmp` | 逐元素比较 |
| `VPowOp` | `hir.vpow` | 逐元素幂运算 |
| `VSelOp` | `hir.vsel` | 逐元素选择 |
| `VBrcOp` | `hir.vbrc` | 向量广播 |
| `VReduceOp` | `hir.vreduce` | 向量规约 |
| `VTransposeOp` | `hir.vtranspose` | 向量转置 |
| `VArangeOp` | `hir.varange` | 向量范围生成 |
| `VInterleaveOp` | `hir.vinterleave` | 向量交织 |
| `VDeinterleaveOp` | `hir.vdeinterleave` | 向量解交织 |
| `VFlipOp` | `hir.vflip` | 向量翻转 |
| `VMulextendedOp` | `hir.vmulextended` | 扩展乘法（高低 16 位） |
| `VPadOp` | `hir.vpad` | 向量填充 |
| `VConcatOp` | `hir.vconcat` | 向量拼接 |
| `VGatherOp` | `hir.vgather` | 向量收集 |
| `VCumprodOp` | `hir.vcumprod` | 累积乘积 |
| `VCumsumOp` | `hir.vcumsum` | 累积求和 |
| `VSortOp` | `hir.vsort` | 向量排序 |

### 宏操作（矩阵乘法）

定义文件：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)

| 操作名 | IR 语法 | 一句话描述 |
|--------|---------|-----------|
| `MmadL1Op` | `hir.mmadL1` | L1 输入的矩阵乘加操作 |
| `BatchMmadL1Op` | `hir.batchMmadL1` | L1 输入的批量矩阵乘加操作 |
| `MatmulOp` | `hir.matmul` | GM 输入的全局矩阵乘法 |
| `MixMatmulOp` | `hir.mix_matmul` | 支持后向量融合的混合矩阵乘法 |
| `MixGroupMatmulOp` | `hir.mix_group_matmul` | 支持分组专家的混合矩阵乘法 |

### 同步操作

定义文件：[HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)

| 操作名 | IR 语法 | 一句话描述 |
|--------|---------|-----------|
| `SetFlagOp` | `hir.set_flag` | 设置同步标志 |
| `WaitFlagOp` | `hir.wait_flag` | 等待同步标志 |
| `PipeBarrierOp` | `hir.pipe_barrier` | Pipeline 屏障 |
| `SyncBlockOp` | `hir.sync_block` | 块间同步 |
| `SyncBlockSetOp` | `hir.sync_block_set` | 设置块同步标志 |
| `SyncBlockWaitOp` | `hir.sync_block_wait` | 等待块同步标志 |
| `CreateSyncBlockLockOp` | `hir.create_sync_block_lock` | 创建块同步锁 |
| `SyncBlockLockOp` | `hir.sync_block_lock` | 获取块同步锁 |
| `SyncBlockUnlockOp` | `hir.sync_block_unlock` | 释放块同步锁 |

### Custom / 基础操作

定义文件：[HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td)

| 操作名 | IR 语法 | 一句话描述 |
|--------|---------|-----------|
| `GetBlockIdxOp` | `hir.get_block_idx` | 获取当前线程块索引 |
| `GetBlockNumOp` | `hir.get_block_num` | 获取线程块数量 |
| `GetSubBlockIdxOp` | `hir.get_sub_block_idx` | 获取子块索引 |
| `GetSubBlockNumOp` | `hir.get_sub_block_num` | 获取子块数量 |
| `SetAtomicOp` | `hir.set_atomic` | 设置原子操作模式 |
| `SetMaskNormOp` | `hir.set_mask_norm` | 设置掩码规范化 |
| `SetCtrlOp` | `hir.set_ctrl` | 设置控制位 |
| `LoadScalarOp` | `hir.load_scalar` | 加载标量 |
| `DCCIOp` | `hir.dcci` | 数据缓存清理/无效化 |
| `ConvertLayoutOp` | `hir.convert_layout` | 布局转换 |
| `PointerCastOp` | `hir.pointer_cast` | 指针类型转换 |
| `BitcastOp` | `hir.bitcast` | 位重解释 |
| `SetFFTSBaseAddrOp` | `hir.set_ffts_base_addr` | 设置 FFTS 基地址 |
| `CustomOp` | `hir.custom` | 自定义操作 |
| `CustomMacroOp` | `hir.custom_macro` | 自定义宏操作 |
| `GatherLoadOp` | `hir.gather_load` | 稀疏内存加载 |
| `ScatterStoreOp` | `hir.scatter_store` | 稀疏内存存储 |
| `LocalLoadOp` | `hir.local_load` | 从 UB 加载张量（SIMD 到 SIMT） |
| `LocalStoreOp` | `hir.local_store` | 存储张量到 UB（SIMT 到 SIMD） |
| `IndirectLoadOp` | `hir.indirect_load` | 间接内存加载 |
| `IndirectStoreOp` | `hir.indirect_store` | 间接内存存储 |
| `GatherTOp` | `hir.gatherT` | 按轴收集操作 |
| `IndexPutOp` | `hir.index_put` | 按轴散列写操作 |
| `ScatterTOp` | `hir.scatterT` | 按轴散列存储操作 |
| `EmbeddingGatherOp` | `hir.embedding_gather` | 嵌入查找操作 |
| `DebugOp` | `hir.debug` | 设备端调试 |
| `InitDebugOp` | `hir.init_debug` | 初始化调试 |
| `FinishDebugOp` | `hir.finish_debug` | 结束调试 |

### 内建指令

定义文件：[HIVMIntrinOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td)

| 操作名 | IR 语法 | 一句话描述 |
|--------|---------|-----------|
| `GetBlockIdxInstrOp` | `hir.intr.hivm.GET.BLOCK.IDX` | 获取块索引指令 |
| `GetBlockNumInstrOp` | `hir.intr.hivm.GET.BLOCK.NUM` | 获取块数量指令 |
| `GetSubBlockIdxInstrOp` | `hir.intr.hivm.GET.SUBBLOCKID` | 获取子块索引指令 |
| `GetSubBlockNumInstrOp` | `hir.intr.hivm.GET.SUBBLOCKDIM` | 获取子块维度指令 |
| `SetFlagImmInstrOp` | `hir.intr.hivm.SET.FLAG.IMM` | 立即数设置标志指令 |
| `WaitFlagImmInstrOp` | `hir.intr.hivm.WAIT.FLAG.IMM` | 立即数等待标志指令 |
| `SetFlagRegInstrOp` | `hir.intr.hivm.SET.FLAG.REG` | 寄存器设置标志指令 |
| `WaitFlagRegInstrOp` | `hir.intr.hivm.WAIT.FLAG.REG` | 寄存器等待标志指令 |
| `PipeBarrierInstrOp` | `hir.intr.hivm.BARRIER` | Pipeline 屏障指令 |
| `SetFftsBaseAddrInstrOp` | `hir.intr.hivm.SET.FFTS.BASE.ADDR` | 设置 FFTS 基地址指令 |
| `SetCrossCoreInstrOp` | `hir.intr.hivm.SET.CROSS.CORE` | FFTS 跨核同步指令 |
| `WaitFlagDevInstrOp` | `hir.intr.hivm.WAIT.FLAG.DEV.REG` | FFTS 块/子块同步指令 |
| `SetMaskNormInstrOp` | `hir.intr.hivm.SET.MASK.NORM` | 设置掩码规范化指令 |
| `GetCtrlInstrOp` | `hir.intr.hivm.GET.CTRL` | 获取控制位指令 |
| `SetCtrlInstrOp` | `hir.intr.hivm.SET.CTRL` | 设置控制位指令 |
| `SBitSet0InstrOp` | `hir.intr.hivm.SBITSET0` | 设置状态位为 0 |
| `SBitSet1InstrOp` | `hir.intr.hivm.SBITSET1` | 设置状态位为 1 |
| `DCCIDstInstrOp` | `hir.intr.hivm.DCCI.DST` | 数据缓存清理 GM 指令 |
| `DCCIDstUBInstrOp` | `hir.intr.hivm.DCCI.DST.UB` | 数据缓存清理 UB 指令 |

## 核心属性与枚举

### Address Space（地址空间）

定义于 [HIVMAttrs.td:L170-L197](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L170-L197)

| 枚举值 | IR 字面量 | 含义 |
|--------|----------|------|
| `Zero` | `zero` | 默认地址空间 |
| `GM` | `gm` | 全局内存 |
| `L1` | `cbuf` | L1 缓存（Cube Buffer） |
| `L0A` | `ca` | L0A 缓存（Cube A 矩阵） |
| `L0B` | `cb` | L0B 缓存（Cube B 矩阵） |
| `L0C` | `cc` | L0C 缓存（Cube C 矩阵/累加器） |
| `UB` | `ub` | 统一缓冲区（Vector Buffer） |

### PIPE（Pipeline）

定义于 [HIVMAttrs.td:L203-L244](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L244)

| 枚举值 | 含义 |
|--------|------|
| `PIPE_S` | Scalar Pipeline |
| `PIPE_V` | Vector Pipeline |
| `PIPE_M` | Cube Pipeline |
| `PIPE_MTE1` | MTE1 Pipeline（L1 搬运） |
| `PIPE_MTE2` | MTE2 Pipeline（GM 到本地搬运） |
| `PIPE_MTE3` | MTE3 Pipeline（本地到 GM 搬运） |
| `PIPE_FIX` | Fix Pipeline（L0C 搬出） |
| `PIPE_MTE4` | MTE4 Pipeline |
| `PIPE_MTE5` | MTE5 Pipeline |

### Core Type（核心类型）

定义于 [HIVMAttrs.td:L298-L317](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L317)

| 枚举值 | 含义 |
|--------|------|
| `CUBE` | Cube 核心（矩阵计算） |
| `VECTOR` | Vector 核心（向量计算） |
| `CUBE_OR_VECTOR` | Cube 或 Vector 核心 |
| `CUBE_AND_VECTOR` | Cube 和 Vector 核心（混合） |

## 与 HFusion / Triton 方言的关系

```
Triton Python API
       |
       v
  Triton Dialect (tt)        -- 前端 IR，描述 Triton 语义
       |
       v
  HFusion Dialect            -- 融合 IR，描述算子融合与 Tiling
       |
       v
  HIVM Dialect (hir)         -- 硬件映射 IR，描述 NPU 操作
       |
       v
  LLVM / CCE 指令            -- 最终代码生成
```

- **Triton 方言**：用户编写的 Triton Python 代码首先被编译为 Triton 方言 IR，描述高级语义（如 `tt.load`、`tt.store`、`tt.dot`）
- **HFusion 方言**：Triton 方言经过 Tiling 和融合优化后转换为 HFusion 方言，描述算子融合策略和 Tile 级别的计算
- **HIVM 方言**：HFusion 方言进一步 lowering 为 HIVM 方言，精确映射到 Ascend NPU 的硬件操作，包括 DMA 搬运、Pipeline 分配、同步控制等

## 方言在编译栈中的位置

```
+--------------------------------------------------+
|              Triton Python Source                 |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|           Triton Dialect (tt)                     |
|  tt.load, tt.store, tt.dot, tt.make_range ...    |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|           HFusion Dialect                         |
|  融合策略、Tiling、Bufferization                  |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|           HIVM Dialect (hir)                      |
|  hir.load, hir.store, hir.mmadL1, hir.vadd ...   |
|  Pipeline 分配、同步插入、内存规划                 |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|           CCE / LLVM IR                           |
|  库函数调用、内建指令                              |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|           Ascend NPU 二进制                       |
+--------------------------------------------------+
```

HIVM 方言处于编译栈的中间层，是连接高级语义与底层硬件的关键桥梁。在此阶段：
- 数据搬运被映射为具体的 DMA 操作（Load/Store/Copy/Fixpipe 等）
- 计算操作被分配到 Cube 或 Vector 核心
- Pipeline 归属被确定，同步操作被插入
- 内存层级（GM/L1/UB/L0A/L0B/L0C）被显式标注

## 相关文档

- DMA 操作详解：[01-DMA-Operations/00-overview.md](01-DMA-Operations/00-overview.md)
- 源码参考：
  - [HIVMBase.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td) - 方言基础定义
  - [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) - 属性与枚举定义
  - [HIVMInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td) - 接口定义
