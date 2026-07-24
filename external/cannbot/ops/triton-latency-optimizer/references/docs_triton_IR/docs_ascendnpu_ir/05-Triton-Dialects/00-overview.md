# Triton 方言体系总览

## 1. 方言层次结构

AscendNPU-IR 项目中的 Triton 编译器定义了三层核心方言，分别对应编译的不同抽象层级：

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (Python)                  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Triton Dialect (tt)                     │
│   设备无关的 IR 层，表达张量计算语义                    │
└────────────────────────┬────────────────────────────┘
                         │ TritonToTritonGPU Conversion
                         ▼
┌─────────────────────────────────────────────────────┐
│            TritonGPU Dialect (ttg)                   │
│   GPU 并行布局编码，数据分布与共享内存管理               │
└────────────────────────┬────────────────────────────┘
                         │ TritonGPUToLLVM Conversion
                         ▼
┌─────────────────────────────────────────────────────┐
│                 LLVM Dialect                         │
│   底层代码生成                                       │
└─────────────────────────────────────────────────────┘
```

此外，还存在一个实验性方言：

```
┌─────────────────────────────────────────────────────┐
│              Gluon Dialect (gluon)                   │
│   实验性布局推断方言，自动将 auto 编码替换为具体编码     │
└─────────────────────────────────────────────────────┘
```

## 2. 命名空间与操作前缀

| 方言 | 命名空间 | 操作前缀 | 类型前缀 | 属性前缀 | 源码位置 |
|------|----------|----------|----------|----------|----------|
| Triton | `::mlir::triton` | `tt.` | `!tt.` | `#tt.` | [TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonOps.td) |
| TritonGPU | `::mlir::triton::gpu` | `ttg.` | `!ttg.` | `#ttg.` | [TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td) |
| Gluon | `::mlir::triton::gluon` | `gluon.` | - | - | [GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Gluon/IR/GluonOps.td) |

## 3. 方言间关系

### 3.1 Triton (tt) → TritonGPU (ttg) 转换

`ConvertTritonToTritonGPU` Pass 将设备无关的 Triton IR 转换为带有 GPU 布局编码的 TritonGPU IR。核心转换语义：

- **类型增强**：`tensor<...>` → `tensor<..., #encoding>`，为每个张量附加分布式布局编码
- **指针类型转换**：`tt.ptr<tensor<...>>` → `tt.ptr<tensor<..., #encoding>>`
- **操作映射**：大部分 `tt.*` 操作直接映射到同名的 TritonGPU 版本，但增加了布局约束
- **新增操作**：引入 `ttg.convert_layout` 用于布局转换，`ttg.local_alloc/dealloc/load/store` 用于共享内存管理

转换选项（源自 [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td)）：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target` | string | `""` | GPU 目标，如 `cuda:80`, `hip:gfx942` |
| `num-warps` | int32 | `4` | warp 数量 |
| `threads-per-warp` | int32 | `32` | 每 warp 线程数 |
| `num-ctas` | int32 | `1` | CGA 中 CTA 数量 |
| `enable-source-remat` | bool | `false` | 启用源重物化 |
| `shared-memory-size` | int32 | `122880` | SIMT 共享内存大小（昇腾条件编译） |

### 3.2 TritonGPU (ttg) → LLVM 转换

`TritonGPUToLLVM` 转换 Pass 链将 TritonGPU IR 降低为 LLVM IR，包含以下子 Pass：

| Pass 名称 | CLI 参数 | 说明 |
|-----------|----------|------|
| `AllocateSharedMemory` | `allocate-shared-memory` | 共享内存分配与偏移标注 |
| `TritonGPUGlobalScratchAllocationPass` | `tritongpu-global-scratch-memory-allocation` | 全局 scratch 内存分配 |
| `TritonGPUAllocateWarpGroups` | `tritongpu-allocate-warp-groups` | Warp 组分配 |

昇腾适配额外引入了 `ConvertTritonAscendGPUToLLVM` Pass，将 TritonGPU 操作通过 Ascend DPX 方言降低到 LLVM。

### 3.3 Gluon 方言的角色

Gluon 方言目前仅包含一个操作 `gluon.set_auto_layout`，用于将张量的 `auto` 布局编码替换为具体的编码类型。这是布局推断的基础设施，允许编译器在后续 Pass 中自动选择最优布局。

## 4. 操作分类总览

### 4.1 Triton (tt) 操作分类

| 类别 | 操作 | 说明 |
|------|------|------|
| 类型转换 | `tt.int_to_ptr`, `tt.ptr_to_int`, `tt.bitcast`, `tt.fp_to_fp` | 指针/数值类型转换 |
| 算术运算 | `tt.clampf`, `tt.precise_sqrt`, `tt.precise_divf`, `tt.mulhiui` | 特殊算术操作 |
| 指针算术 | `tt.addptr`, `tt.advance` | 指针偏移计算 |
| 内存访问 | `tt.load`, `tt.store` | 全局内存读写 |
| 原子操作 | `tt.atomic_rmw`, `tt.atomic_cas` | 原子读-改-写 / 比较-交换 |
| 形状操作 | `tt.splat`, `tt.unsplat`, `tt.expand_dims`, `tt.reshape`, `tt.broadcast`, `tt.cat`, `tt.join`, `tt.split`, `tt.trans` | 张量形状变换 |
| SPMD | `tt.get_program_id`, `tt.get_num_programs` | 程序网格信息 |
| 矩阵乘法 | `tt.dot`, `tt.dot_scaled` | 矩阵乘法（含缩放） |
| 归约/扫描 | `tt.reduce`, `tt.scan` | 归约与前缀扫描 |
| 映射 | `tt.map_elementwise` | 标量子区域映射 |
| 范围生成 | `tt.make_range` | 整数范围生成 |
| 直方图 | `tt.histogram` | 直方图统计 |
| 收集 | `tt.gather` | 按索引收集元素 |
| 调试 | `tt.print`, `tt.assert` | 设备端调试 |
| 张量指针 | `tt.make_tensor_ptr`, `tt.make_tensor_descriptor` | 张量指针/描述符构造 |
| 描述符操作 | `tt.descriptor_load`, `tt.descriptor_store`, `tt.descriptor_reduce`, `tt.descriptor_gather`, `tt.descriptor_scatter` | 基于 TMA 描述符的内存操作 |
| 函数 | `tt.func`, `tt.call`, `tt.return` | 函数定义与调用 |
| 外部调用 | `tt.extern_elementwise`, `tt.elementwise_inline_asm` | 外部函数与内联汇编 |

### 4.2 TritonGPU (ttg) 操作分类

| 类别 | 操作 | 说明 |
|------|------|------|
| 布局转换 | `ttg.convert_layout` | 张量布局编码转换 |
| 异步拷贝 | `ttg.async_copy_global_to_local`, `ttg.async_wait`, `ttg.async_commit_group` | 异步全局到共享内存拷贝 |
| 共享内存 | `ttg.local_alloc`, `ttg.local_dealloc`, `ttg.local_load`, `ttg.local_store` | 共享内存分配/释放/读写 |
| 内存描述符视图 | `ttg.memdesc_index`, `ttg.memdesc_subslice`, `ttg.memdesc_trans`, `ttg.memdesc_reshape`, `ttg.memdesc_reinterpret` | 内存描述符子视图操作 |
| 流水线 | `ttg.predicate_stage`, `ttg.mask` | 软件流水线谓词与掩码 |
| 类型转换 | `ttg.fp4_to_fp` | FP4 到浮点数上转换 |
| 全局内存 | `ttg.global_scratch_alloc` | 全局 scratch 内存分配 |
| Warp 特化 | `ttg.warp_specialize`, `ttg.warp_specialize.partitions`, `ttg.warp_yield`, `ttg.warp_return` | Warp 组特化执行 |

## 5. 类型体系总览

Triton 方言定义了以下核心类型（源自 [TritonTypes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonTypes.td)）：

| 类型 | 助记符 | 说明 |
|------|--------|------|
| `PointerType` | `ptr` | 指针类型，包含 pointee 类型和地址空间 |
| `TensorDescType` | `tensordesc` | 张量描述符类型，用于 TMA 操作 |

类型约束别名：

| 约束名 | 定义 | 含义 |
|--------|------|------|
| `TT_Float` | `F8E4M3FN \| F8E4M3FNUZ \| F8E5M2 \| F8E5M2FNUZ \| F16 \| BF16 \| F32 \| F64` | 浮点类型 |
| `TT_Int` | `I1 \| I4 \| I8 \| I16 \| I32 \| I64` | 整数类型 |
| `TT_Ptr` | `ptr<AnyType>` | 标量指针 |
| `TT_PtrLike` | `TT_Ptr \| TT_PtrTensor` | 指针或指针张量 |
| `TT_TensorPtr` | `ptr<TT_Tensor>` | 张量指针 |
| `TT_Tensor` | `RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>` | 张量类型 |
| `TT_FpIntTensor` | `RankedTensorOf<[TT_Float, TT_Int]>` | 浮点/整数张量 |

## 6. 属性体系总览

Triton 方言定义了以下枚举属性（源自 [TritonAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)）：

| 属性 | 枚举值 | 用途 |
|------|--------|------|
| `CacheModifier` | NONE, CA, CG, WB, CS, WT, CV | 缓存修饰符 |
| `EvictionPolicy` | NORMAL, EVICT_FIRST, EVICT_LAST | 驱逐策略 |
| `PaddingOption` | PAD_ZERO, PAD_NAN | 填充选项 |
| `RMWOp` | AND, OR, XOR, ADD, FADD, MAX, MIN, UMAX, UMIN, XCHG | 原子读-改-写操作类型 |
| `MemSemantic` | RELAXED, ACQUIRE, RELEASE, ACQUIRE_RELEASE | 内存语义 |
| `MemSyncScope` | GPU, CTA, SYSTEM | 内存同步范围 |
| `ProgramIDDim` | X, Y, Z | 程序 ID 维度 |
| `RoundingMode` | RTZ, RTNE | 舍入模式 |
| `PropagateNan` | NONE, ALL | NaN 传播策略 |
| `InputPrecision` | TF32, TF32x3, IEEE | Dot 操作输入精度 |
| `ScaleDotElemType` | E4M3, E5M2, E2M3, E3M2, E2M1, BF16, FP16 | 缩放 Dot 元素类型 |
| `DescriptorReduceKind` | ADD, MIN, MAX, INC, DEC, AND, OR, XOR | 描述符归约类型 |

## 7. TritonGPU 布局编码总览

TritonGPU 方言定义了丰富的布局编码属性（源自 [TritonGPUAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)）：

| 编码 | 助记符 | 类别 | 说明 |
|------|--------|------|------|
| `BlockedEncodingAttr` | `blocked` | 分布式 | 连续分块布局，促进内存合并 |
| `LinearEncodingAttr` | `linear` | 分布式 | 线性布局，基于 LinearLayout |
| `NvidiaMmaEncodingAttr` | `nvidia_mma` | 分布式 | NVIDIA Tensor Core MMA 输出布局 |
| `AMDMfmaEncodingAttr` | `amd_mfma` | 分布式 | AMD MFMA 矩阵核心布局 |
| `AMDWmmaEncodingAttr` | `amd_wmma` | 分布式 | AMD WMMA 矩阵核心布局 |
| `SliceEncodingAttr` | `slice` | 分布式 | 沿某维度压缩的切片布局 |
| `DotOperandEncodingAttr` | `dot_op` | 分布式 | Dot 操作数布局 |
| `SwizzledSharedEncodingAttr` | `swizzled_shared` | 共享 | Swizzle 避免银行冲突的共享内存布局 |
| `PaddedSharedEncodingAttr` | `padded_shared` | 共享 | 填充避免银行冲突的共享内存布局 |
| `NVMMASharedEncodingAttr` | `nvmma_shared` | 共享 | MMAv3/v5 共享内存输入布局 |
| `AMDRotatingSharedEncodingAttr` | `amd_rotating_shared` | 共享 | AMD 旋转 Swizzle 共享内存布局 |
| `FractalSharedEncodingAttr` | `fractal_shared` | 共享 | **昇腾适配** Fractal 共享内存布局 |
| `CTALayoutAttr` | `cta_layout` | CTA | CTA 级布局描述 |

## 8. 源码参考

| 文件 | 内容 |
|------|------|
| [TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonOps.td) | Triton 操作 TableGen 定义 |
| [TritonTypes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonTypes.td) | Triton 类型 TableGen 定义 |
| [TritonAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td) | Triton 属性 TableGen 定义 |
| [TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td) | TritonGPU 操作 TableGen 定义 |
| [TritonGPUAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td) | TritonGPU 布局编码属性定义 |
| [GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/Gluon/IR/GluonOps.td) | Gluon 操作 TableGen 定义 |
