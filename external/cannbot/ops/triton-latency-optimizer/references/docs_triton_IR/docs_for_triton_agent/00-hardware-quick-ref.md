# Ascend910_95 硬件速查手册（面向 Triton Agent）

## 触发条件

当 Triton Agent 遇到以下场景时，需要参考本文档：

- 目标设备为 Ascend910_95 / 950PR / 950DT 系列（架构代号 `dav-c310`）
- 需要确定内存分配策略（UB/L1/L0C 容量、对齐约束）
- 需要选择正确的 Pipeline 或数据通路（尤其是 L0C->UB 直通、UB->L1 通路）
- 需要判断 Reg-based vs Mem-based 架构行为差异（SIMT VF 模式、同步机制、归约降级）
- 需要使用紧耦合缓冲区（TightlyCoupledBuffer）
- 需要确认对齐要求以避免硬件异常

---

## 核心知识：规格速查表

### AI Core 架构

```
每个 AI Core = 1 Cube (AIC) + 2 Vector (AIV) + Scalar
VectorCoreCount = 2 * CubeCoreCount = 2 * AiCoreCount
```

### Ascend910_95 系列规格表

| 型号 | AI Core | Cube Core | Vector Core | UB | DCache | L1 | L0A | L0B | L0C | Arch |
|------|---------|-----------|-------------|------|--------|-----|-----|-----|-----|------|
| Ascend910_950z | 4 | 4 | 8 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9579 | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_957b | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_957d | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| **Ascend910_9581** | **32** | **32** | **64** | **248KB** | **32~120KB** | **512KB** | **64KB** | **64KB** | **256KB** | **dav-c310** |
| Ascend910_9589 | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_958a | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_958b | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9599 | 36 | 36 | 72 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |

> UB 248KB = 256KB - 8KB（预留 8KB 给编译器），源码值 `UbSize=2031616 bits`

### 内存层次速查

| 层次 | IR 标识符 | 枚举值 | 大小 (910_95) | 大小 (910B) | 对齐 | 所属单元 |
|------|-----------|--------|--------------|------------|------|----------|
| GM | `gm` | 1 | HBM/L2 | HBM/L2 | - | 共享 |
| L1 | `cbuf` | 2 | **512KB** | 512KB | **32B** | Cube |
| L0A | `ca` | 3 | 64KB | 64KB | - | Cube A 端 |
| L0B | `cb` | 4 | 64KB | 64KB | - | Cube B 端 |
| L0C | `cc` | 5 | **256KB** | 128KB | **512B** | Cube C 端 |
| UB | `ub` | 6 | **248KB** | 192KB | **32B** | Vector |
| DCache | - | - | **32~120KB** | 无 | - | SIMT Vector |

IR 声明示例：

```mlir
memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
memref<?x?x?x?xf32, #hivm.address_space<cc>>
memref<256x256xf16, #hivm.address_space<gm>>
memref<128x128xf32, #hivm.address_space<ub>>
```

### 专用硬件缓冲区

| 缓冲区 | 大小 | 对齐 | 访问方式 |
|--------|------|------|----------|
| BT Buffer (BiasTable) | 1KB | 64B | `copy_cbuf_to_bt` 从 L1 拷贝 |
| FP Buffer (FixPipe) | 7KB | 128B | `hivm.fixpipe` 隐式使用 |

### 对齐要求（所有架构一致）

| 存储空间 | 对齐 | 源码值 (bits) |
|----------|------|--------------|
| UB | **32B** | 256 |
| L1 | **32B** | 256 |
| L0C | **512B** | 4096 |

---

## 910_95 特别注意

### 与 910B 的关键差异速查

| 特性 | 910B (dav-c220) | 910_95 (dav-c310) |
|------|-----------------|-------------------|
| 架构类型 | **Mem-based** (A2/A3) | **Reg-based** (A5) |
| SIMT VF 模式 | 不支持 | **支持** |
| UB | 192KB | 248KB（预留 8KB） |
| L0C | 128KB | **256KB** |
| DCache | 无 | **32~120KB** |
| L0C -> UB 直通 | 不支持 | **支持**（FixPipe） |
| UB -> L1 通路 | 不支持 | **支持**（PIPE_MTE3） |
| 紧耦合缓冲区 | 不支持 | **支持** |
| Fixpipe Dual Dst | 不支持 | **支持**（ROW_SPLIT / COLUMN_SPLIT） |
| Fixpipe NZ2DN | 不支持 | **支持** |
| 同步方式 | FFTS（基于内存） | SetFlag/WaitFlag（基于寄存器） |
| 归约标量降级 | i64/argmax/argmin 会降级 | 基本归约不降级 |
| vcmp(NE) 规范化 | vnot(vcmp(EQ)) | 不做规范化 |

### 数据通路 ASCII 图（910_95 特有）

```
+-----------------------------------------------------------------------------+
|                          Global Memory (GM / HBM)                           |
+---------------------------------------+-------------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
              +-----v-----+                           +-----v-----+
              |   MTE2    |                           |   MTE2    |
              |  GM -> L1 |                           |  GM -> UB |
              |  (双向)   |                           |  (单向)   |
              +-----+-----+                           +-----+-----+
                    |                                       |
                    v                                       v
        +-----------------------+                   +----------------------+
        |          L1           |                   |         UB           |
        |    (cbuf, 512KB)      |                   |  (ub, 248KB,预留8KB)  |
        |    Cube输入缓存       |                   |    Vector工作区      |
        +-----------+-----------+                   +--------+-------------+
                    |                                        |
        +-----------+-----------+                            |
        |           |           |                            |
  +-----v-----+ +---v---+ +-----v-----+                      |
  |   MTE1    | | MTE1  | |   MTE1    |                      |
  | L1 -> L0A | |L1->L0B| |L1 -> BT Buf|                     |
  +-----+-----+ +---+---+ +-----+-----+                      |
        |           |           |                            |
        v           v           v                            |
  +-----------+ +-----------+ +-----------+                   |
  |    L0A    | |    L0B    | | BT Buffer |                   |
  | (ca,64KB) | | (cb,64KB) | |   (1KB)   |                   |
  | 矩阵A输入  | | 矩阵B输入  | | Bias数据  |                   |
  +-----+-----+ +-----+-----+ +-----+-----+                   |
        |             |             |                          |
        +-------------+-------------+                          |
                      |                                        |
                      v                                        |
            +------------------+                               |
            |      Cube        |                               |
            |   (MatMul)       |                               |
            +--------+---------+                               |
                     |                                         |
                     v                                         |
            +------------------+                               |
            |      L0C         |                               |
            |   (cc, 256KB)    |                               |
            |   矩阵乘法结果    |                               |
            +--------+---------+                               |
                     |                                         |
         +-----------+-----------+-----------+                  |
         |           |           |           |                  |
   +-----v-----+ +---v---+ +-----v-----+     |                  |
   |    FIX    | |  FIX  | |    FIX    |     |                  |
   | L0C -> GM | |L0C->L1| | L0C -> UB |<----+                  |
   |           | |       | | (950特有) |                        |
   +-----+-----+ +---+---+ +-----+-----+                        |
         |           |           |                               |
         v           v           v                               |
   +-----------+ +-----------+ +----------------------+          |
   |    GM     | |    L1     | |  UB (紧耦合缓冲区)    |<---------+
   +-----------+ +-----------+ +----------+-----------+          |
                                          |                     |
                                    +-----v-----+               |
                                    |   MTE3    |               |
                                    |  UB -> GM |               |
                                    +-----+-----+               |
                                          |                     |
                                          v                     |
                                    +-----------+               |
                                    |    GM     |               |
                                    +-----------+               |
```

**910_95 vs 910B 关键通路差异**：

```
910_95: GM -> L1 -> L0A/L0B -> L0C -> UB -> V -> UB -> GM
                                           ^^^^^^^^^^^^
                                           L0C直通UB（省去GM中转）

910B:   GM -> L1 -> L0A/L0B -> L0C -> GM -> UB -> V -> UB -> GM
                                       ^^^^^^^^
                                       必须经过GM中转
```

### Reg-based vs Mem-based 架构差异

910_95 是 **Reg-based (A5)** 架构，910B 是 **Mem-based (A2/A3)** 架构。

判断函数：
```cpp
bool isRegBasedArch(TargetDevice targetDevice) {
  return isAscend310B(targetDevice) || isAscend950(targetDevice);
}
bool isMemBasedArch(TargetDevice targetDevice) {
  return isAscend910B(targetDevice) || isAscend910_93(targetDevice);
}
```

**核心区别**：

| 维度 | Mem-based (910B) | Reg-based (910_95) |
|------|-------------------|---------------------|
| SIMT VF 模式 | 不支持，编译器跳过 InferVFMode | **支持**，运行 InferVFMode 推断 SIMD/SIMT/MIX |
| 数据访问模型 | 全部基于 UB 缓冲区（SIMD） | SIMT 基于寄存器，SIMD 基于 UB |
| SIMT 编译路径 | 无 | SIMT VF 拆分后走 Triton GPU 编译路径 |
| DCache | 无 | 有（32~120KB） |
| 同步方式 | FFTS（基于内存） | SetFlag/WaitFlag（基于寄存器） |
| 跨核同步 | SetCrossCoreInstrOp | IntraBlockSet / IntraBlockRegInstrOp |
| Pipe Barrier | 对所有 Pipe 生成 barrier | **跳过 PIPE_V 的 barrier** |
| 归约降级 | i64 归约和整数 argmax/argmin 标量降级 | 基本归约不降级（除 argmax/argmin 对齐问题） |
| vcmp(NE) | 规范化为 vnot(vcmp(EQ)) | 不做规范化 |
| 内存规划 | SIMT/MIX 下不需要动态调整 UB | SIMT/MIX 下需动态调整 UB（考虑 DCache） |
| 入口配置 | `configureEntryForMembaseArch` | `configureEntryForRegbaseArch` |

### Pipeline 枚举速查

| 枚举 | IR 标识符 | 数值 | 硬件单元 | 数据流 | 典型操作 |
|------|-----------|------|----------|--------|----------|
| PIPE_S | `PIPE_S` | 0 | Scalar | 标量计算 | 循环控制、条件判断 |
| PIPE_V | `PIPE_V` | 1 | Vector | UB -> UB | `vadd`, `vmul`, `vcast`, `vreduce` |
| PIPE_M | `PIPE_M` | 2 | Cube | L0A/L0B -> L0C | `mmadL1` 中的矩阵乘法 |
| PIPE_MTE1 | `PIPE_MTE1` | 3 | MTE1 DMA | L1 -> L0A/L0B/BT | `mmadL1` 中的数据加载, `l12ub` |
| PIPE_MTE2 | `PIPE_MTE2` | 4 | MTE2 DMA | GM <-> L1/UB | `load`, `nd2nz` |
| PIPE_MTE3 | `PIPE_MTE3` | 5 | MTE3 DMA | UB -> GM/L1 | `store`, `nz2nd`, UB->L1 |
| PIPE_FIX | `PIPE_FIX` | 10 | FixPipe | L0C -> GM/L1/UB | `fixpipe` |

> 完整枚举还包括：PIPE_ALL(6), PIPE_MTE4(7), PIPE_MTE5(8), PIPE_V2(9), VIRTUAL_PIPE_MTE2_L1A(11), VIRTUAL_PIPE_MTE2_L1B(12), PIPE_NUM(13), PIPE_UNASSIGNED(99)

**源-目标到 Pipeline 映射**：

| 源 | 目标 | Pipeline | IR 操作 |
|----|------|----------|---------|
| GM | L1 | PIPE_MTE2 | `hivm.nd2nz` |
| GM | UB | PIPE_MTE2 | `hivm.load` |
| L1 | GM | PIPE_MTE2 | `copy_cbuf_to_gm` |
| L1 | L0A | PIPE_MTE1 | 内部指令 |
| L1 | L0B | PIPE_MTE1 | 内部指令 |
| L1 | UB | PIPE_MTE1 | `hivm.l12ub` |
| L0A/L0B | L0C | PIPE_M | Cube 计算 |
| L0C | GM | PIPE_FIX | `hivm.fixpipe` |
| L0C | L1 | PIPE_FIX | `hivm.fixpipe` |
| **L0C** | **UB** | **PIPE_FIX** | **`hivm.fixpipe`（仅 950）** |
| UB | UB | PIPE_V | `hivm.copy` |
| UB | GM | PIPE_MTE3 | `hivm.store` |
| **UB** | **L1** | **PIPE_MTE3** | **`hivm.copy`（仅 950）** |

### 紧耦合缓冲区（910_95 特有）

IR 表示：
```mlir
#hivm.tightly_coupled_buffer<id : optional<i32>>
```

两种模式：

| 模式 | 数据流向 | 说明 |
|------|---------|------|
| MoveToUb | L0C -> UB | Cube 结果直通 Vector 工作区 |
| MoveToL1 | UB -> L1 | Vector 处理结果回传 Cube 输入缓存 |

Pipeline 选择逻辑：
```
if (isAscend950(target)) {
    if (enableLayoutOptimization) {
        InsertCVDataMovement      // A5 新布局优化路径
    } else {
        InsertCVTightCoupledBuffer // 传统紧耦合缓冲区路径
    }
} else {
    InsertLoadStoreForMixCV       // 非 950: 必须经过 GM 中转
}
```

### VFMode 枚举（仅 Reg-based 架构使用）

| 枚举 | IR 标识符 | 数值 | 说明 |
|------|-----------|------|------|
| SIMD | `#hivm.vf_mode<SIMD>` | 0 | 传统 Vector 执行，基于 UB |
| SIMT | `#hivm.vf_mode<SIMT>` | 1 | 类 GPU 线程级并行，基于寄存器 |
| MIX | `#hivm.vf_mode<MIX>` | 2 | 混合模式，通过 `--enable-simd-simt-mix-compile` 启用 |

### 核心类型枚举速查

**操作级 TCoreType**：

| 枚举 | IR 标识符 | 说明 |
|------|-----------|------|
| CUBE | `#hivm.tcore_type<CUBE>` | 在 Cube 核心执行 |
| VECTOR | `#hivm.tcore_type<VECTOR>` | 在 Vector 核心执行 |
| CUBE_OR_VECTOR | `#hivm.tcore_type<CUBE_OR_VECTOR>` | 可在任一核心执行 |
| CUBE_AND_VECTOR | `#hivm.tcore_type<CUBE_AND_VECTOR>` | 需 Cube+Vector 同时执行 |

**函数级 TFuncCoreType**：

| 枚举 | IR 标识符 | 说明 |
|------|-----------|------|
| AIC | `#hivm.func_core_type<AIC>` | 运行在 AI Cube 核心 |
| AIV | `#hivm.func_core_type<AIV>` | 运行在 AI Vector 核心 |
| MIX | `#hivm.func_core_type<MIX>` | 混合使用 Cube+Vector |

---

## 相关文档链接

- [01-npu-hardware-overview.md](../docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md) -- NPU 硬件架构总览（完整型号规格、架构分类详解）
- [02-memory-hierarchy.md](../docs_ascendnpu_ir/00-Architecture/02-memory-hierarchy.md) -- 内存层次详解（完整数据通路、随路操作、910B ASCII 图）
- [03-pipeline-execution-model.md](../docs_ascendnpu_ir/00-Architecture/03-pipeline-execution-model.md) -- Pipeline 执行模型（同步机制、Trait 定义、Event ID）
- [04-data-layout.md](../docs_ascendnpu_ir/00-Architecture/04-data-layout.md) -- 数据布局详解（ND/NZ/zN/nZ/Fractal、布局转换操作）
- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) -- 型号规格 TableGen 源文件
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) -- IR 属性枚举源文件
