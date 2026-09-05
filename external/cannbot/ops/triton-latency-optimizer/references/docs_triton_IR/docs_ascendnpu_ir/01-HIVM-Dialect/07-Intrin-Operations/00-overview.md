# HIVM 内建指令操作

> 关键词：IntrinOp, GET.BLOCK.IDX, SET.FLAG.IMM, WAIT.FLAG.IMM, BARRIER, SET.FFTS.BASE.ADDR, SET.CROSS.CORE, INTRA.BLOCK

## 概述

HIVM 内建指令操作（Intrinsic Operations）是 HIVM 方言中最底层的操作，直接对应 NPU 硬件指令。这些操作从 [HIVMIntrinOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td) 定义，继承自 `LLVM_IntrOpBase`，在 Lowering 的最终阶段由高层 HIVM 操作转换而来。

内建指令操作通常不由用户直接编写，而是由编译器 Pass 自动生成。了解这些操作有助于理解 HIVM IR 到硬件指令的映射关系。

## 指令分类

### 1. Block 索引指令

| 指令 | 高层操作 | 说明 |
|------|---------|------|
| `hivm.GET.BLOCK.IDX` | `hir.get_block_idx` | 获取当前 Block 索引 |
| `hivm.GET.BLOCK.NUM` | `hir.get_block_num` | 获取 Block 总数 |
| `hivm.GET.SUBBLOCKID` | `hir.get_sub_block_idx` | 获取当前 Sub-Block 索引 |
| `hivm.GET.SUBBLOCKDIM` | `hir.get_sub_block_num` | 获取 Sub-Block 总数 |

### 2. Pipe 同步指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.FLAG.IMM` | set_pipe(I64Attr), wait_pipe(I64Attr), event_id(I64Attr) | 设置 Event Flag（立即数模式） |
| `hivm.WAIT.FLAG.IMM` | set_pipe(I64Attr), wait_pipe(I64Attr), event_id(I64Attr) | 等待 Event Flag（立即数模式） |
| `hivm.SET.FLAG.REG` | set_pipe(I64Attr), wait_pipe(I64Attr), event_id(I64) | 设置 Event Flag（寄存器模式） |
| `hivm.WAIT.FLAG.REG` | set_pipe(I64Attr), wait_pipe(I64Attr), event_id(I64) | 等待 Event Flag（寄存器模式） |
| `hivm.BARRIER` | pipe(I64Attr) | Pipeline 屏障 |

### 3. FFTS 同步指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.FFTS.BASE.ADDR` | addr(I<64>) | 设置 FFTS 基地址寄存器 |
| `hivm.SET.CROSS.CORE` | pipe(I64Attr), config(I<64>) | FFTS 跨核同步指令 |
| `hivm.WAIT.FLAG.DEV.REG` | flag_id(I<64>) | FFTS Block/Sub-Block 同步等待（寄存器） |
| `hivm.WAIT.FLAG.DEV.PIPE.IMM` | pipe(I64Attr), flag_id(I64Attr) | FFTS 同步等待（Pipe + Flag 立即数） |
| `hivm.WAIT.FLAG.DEV.PIPE.REG` | pipe(I64Attr), flag_id(I<64>) | FFTS 同步等待（Pipe 立即数 + Flag 寄存器） |

### 4. Intra-Block 同步指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.INTRA.BLOCK.mode` | pipe(I64Attr), sync_id(I<64>) | Block 内同步设置（寄存器模式） |
| `hivm.WAIT.INTRA.BLOCK.mode` | pipe(I64Attr), sync_id(I<64>) | Block 内同步等待（寄存器模式） |
| `hivm.SET.INTRA.BLOCKI.mode` | pipe(I64Attr), sync_id(I64Attr) | Block 内同步设置（立即数模式） |
| `hivm.WAIT.INTRA.BLOCKI.mode` | pipe(I64Attr), sync_id(I64Attr) | Block 内同步等待（立即数模式） |

### 5. 控制指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.MASK.NORM` | 无 | 设置 Mask 为正常模式 |
| `hivm.GET.CTRL` | 无 | 获取控制寄存器值 |
| `hivm.SET.CTRL` | config(I<64>) | 设置控制寄存器 |
| `hivm.SBITSET0` | x(I<64>), idx(I<64>) | 设置状态位为 0 |
| `hivm.SBITSET1` | x(I<64>), idx(I<64>) | 设置状态位为 1 |

### 6. 缓存指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.DCCI.DST` | ptr(LLVMPointer), entire(I<64>), dst(I<64>) | 清除/无效化 GM 数据缓存 |
| `hivm.DCCI.DST.UB` | ptr(LLVMPointer), entire(I<64>), dst(I<64>) | 清除/无效化 UB 数据缓存 |

## 指令详细说明

### SET.FLAG.IMM / WAIT.FLAG.IMM

立即数模式的 Pipe 同步指令，Event ID 编码在指令中。

```
// 参数
set_pipe:  I64Attr  -- 发送信号的 Pipe（立即数）
wait_pipe: I64Attr  -- 接收信号的 Pipe（立即数）
event_id:  I64Attr  -- Event ID（立即数）

// 无返回值
```

### SET.FLAG.REG / WAIT.FLAG.REG

寄存器模式的 Pipe 同步指令，Event ID 来自寄存器值。

```
// 参数
set_pipe:  I64Attr  -- 发送信号的 Pipe（立即数）
wait_pipe: I64Attr  -- 接收信号的 Pipe（立即数）
event_id:  I<64>    -- Event ID（寄存器值）

// 无返回值
```

### SET.CROSS.CORE

FFTS 跨核同步指令，发送数据（包括模式和 Flag ID）到 FFTS 目标地址。

```
// 参数
pipe:   I64Attr  -- Pipe 类型（立即数）
config: I<64>    -- 配置值（寄存器）

// config 编码格式：
// config = (0x0001 | ((mode & 0x0f) << 4) | ((flagID & 0x0f) << 8))
//
// mode:
//   0: inter block synchronization
//   1: inter subblock synchronization
//   2: intra block synchronization
//
// flagID: 8-bit flag ID，每个 flag ID 有一个计数器
```

### INTRA.BLOCK 同步指令

Block 内 Cube Core 和 Vector Core 之间的同步指令。

**ID 映射规则**（Mix-Mode Block，1 CUBECORE + 2 VECCORE）：

| SET 来源 | 目标 | ID 映射 |
|----------|------|---------|
| VECCORE0 ID 0-15 | CUBECORE | ID 0-15 |
| VECCORE1 ID 0-15 | CUBECORE | ID 16-31 |
| CUBECORE ID 0-15 | VECCORE0 | ID 0-15 |
| CUBECORE ID 16-31 | VECCORE1 | ID 0-15 |

**同步机制**：
- 每个 sync ID 有一个 4-bit 计数器
- SET 操作：递增对方 Core 对应 ID 的计数器，不阻塞当前 Pipeline
- WAIT 操作：如果对应 ID 的计数器为 0 则阻塞，否则递减计数器

## 高层操作到内建指令的映射

| 高层操作 | 内建指令 | 说明 |
|---------|---------|------|
| `hir.get_block_idx` | `hivm.GET.BLOCK.IDX` | Block 索引查询 |
| `hir.get_block_num` | `hivm.GET.BLOCK.NUM` | Block 数量查询 |
| `hir.set_flag` (静态 Event ID) | `hivm.SET.FLAG.IMM` | 立即数模式同步 |
| `hir.set_flag` (动态 Event ID) | `hivm.SET.FLAG.REG` | 寄存器模式同步 |
| `hir.wait_flag` (静态 Event ID) | `hivm.WAIT.FLAG.IMM` | 立即数模式等待 |
| `hir.wait_flag` (动态 Event ID) | `hivm.WAIT.FLAG.REG` | 寄存器模式等待 |
| `hir.pipe_barrier` | `hivm.BARRIER` | Pipeline 屏障 |
| `hir.set_ffts_base_addr` | `hivm.SET.FFTS.BASE.ADDR` | FFTS 基地址设置 |
| `hir.sync_block_set` | `hivm.SET.CROSS.CORE` / `hivm.SET.INTRA.BLOCKI.mode` | 跨核/核内同步 |
| `hir.sync_block_wait` | `hivm.WAIT.FLAG.DEV.REG` / `hivm.WAIT.INTRA.BLOCKI.mode` | 跨核/核内等待 |
| `hir.set_mask_norm` | `hivm.SET.MASK.NORM` | Mask 模式设置 |
| `hir.set_ctrl` | `hivm.SET.CTRL` | 控制寄存器设置 |
| `hir.dcci` | `hivm.DCCI.DST` / `hivm.DCCI.DST.UB` | 缓存操作 |

## 常见问题

**Q: 为什么有 IMM 和 REG 两种模式？**
A: IMM（Immediate）模式将参数编码在指令中，适用于编译时已知的常量参数，执行效率更高。REG（Register）模式从寄存器读取参数，适用于运行时计算的动态参数。

**Q: SET.CROSS.CORE 的 config 如何编码？**
A: `config = 0x0001 | ((mode & 0x0f) << 4) | ((flagID & 0x0f) << 8)`，其中 mode 为同步模式（0=跨Block，1=跨Sub-Block，2=Block内），flagID 为 8-bit Flag ID。

**Q: INTRA.BLOCK 同步的 ID 映射规则是什么？**
A: 在 Mix-Mode Block 中，CUBECORE 有 32 个 ID，每个 VECCORE 有 16 个 ID。VECCORE0 的 ID 0-15 映射到 CUBECORE 的 ID 0-15，VECCORE1 的 ID 0-15 映射到 CUBECORE 的 ID 16-31，反之亦然。

## 相关文档

- 源码参考：[HIVMIntrinOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td)
- 高层同步操作：[04-Synchronization/01-pipe-sync.md](../04-Synchronization/01-pipe-sync.md)
- 跨核同步：[04-Synchronization/02-block-sync.md](../04-Synchronization/02-block-sync.md)
