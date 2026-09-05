# HIVM 枚举属性速查

> 关键词：Enumeration, IteratorType, DataLayout, AddressSpace, Pipe, CoreType, Event, UnitFlag, SyncBlockMode, ReduceOperation, AtomicKind

## 概述

本文档列出 HIVM 方言中所有枚举属性的完整枚举值，是 Agent 最常查阅的参考。所有枚举值均从 [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) 精确提取。

枚举属性在 MLIR IR 中的格式为 `#hivm.<mnemonic><<ENUM_VALUE>>`，例如 `#hivm.pipe<PIPE_M>`。

---

## IteratorType（12 值）

源码：[HIVMAttrs.td#L51-L78](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L51-L78)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| kParallel | 0 | `parallel` | 并行迭代 |
| kBroadcast | 1 | `broadcast` | 广播迭代 |
| kTranspose | 2 | `transpose` | 转置迭代 |
| kReduction | 3 | `reduction` | 归约迭代 |
| kInterleave | 4 | `interleave` | 交织迭代 |
| kDeinterleave | 5 | `deinterleave` | 解交织迭代 |
| kInverse | 6 | `inverse` | 逆序迭代 |
| kPad | 7 | `pad` | 填充迭代 |
| kConcat | 8 | `concat` | 拼接迭代 |
| kGather | 9 | `gather` | 收集迭代 |
| kCumulative | 10 | `cumulative` | 累积迭代 |
| kOpaque | 99 | `opaque` | 不透明迭代 |

---

## DataLayout（7 值）

源码：[HIVMAttrs.td#L84-L101](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L84-L101)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| DOTA_ND | 1 | `dotA_ND` | 矩阵 A 的 ND 布局（Cube 矩阵乘） |
| DOTB_ND | 2 | `dotB_ND` | 矩阵 B 的 ND 布局 |
| DOTC_ND | 3 | `dotC_ND` | 矩阵 C 的 ND 布局 |
| nZ | 4 | `nZ` | nZ 格式（列优先分块） |
| zN | 5 | `zN` | zN 格式（行优先分块） |
| ND | 6 | `ND` | ND 格式（标准行优先） |
| Fractal | 7 | `Fractal` | Fractal 格式 |

---

## AddressSpace（7 值）

源码：[HIVMAttrs.td#L171-L188](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L188)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| Zero | 0 | `zero` | 默认地址空间 |
| GM | 1 | `gm` | 全局内存 |
| L1 | 2 | `cbuf` | L1 缓存（CBuffer） |
| L0A | 3 | `ca` | L0A 缓存（Cube A 矩阵） |
| L0B | 4 | `cb` | L0B 缓存（Cube B 矩阵） |
| L0C | 5 | `cc` | L0C 缓存（Cube C 矩阵/累加器） |
| UB | 6 | `ub` | 统一缓冲区 |

---

## Pipe（15 值）

源码：[HIVMAttrs.td#L203-L236](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L236)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| PIPE_S | 0 | Scalar Pipe |
| PIPE_V | 1 | Vector Pipe |
| PIPE_M | 2 | Cube Matrix Pipe |
| PIPE_MTE1 | 3 | Memory Transfer Engine 1（L1 搬运） |
| PIPE_MTE2 | 4 | Memory Transfer Engine 2（GM→片内搬运） |
| PIPE_MTE3 | 5 | Memory Transfer Engine 3（片内→GM 搬运） |
| PIPE_ALL | 6 | 所有 Pipe |
| PIPE_MTE4 | 7 | Memory Transfer Engine 4 |
| PIPE_MTE5 | 8 | Memory Transfer Engine 5 |
| PIPE_V2 | 9 | Vector Pipe 2 |
| PIPE_FIX | 10 | Fixpipe（L0C→UB/GM 数据转换） |
| VIRTUAL_PIPE_MTE2_L1A | 11 | 虚拟 Pipe：MTE2 到 L1A |
| VIRTUAL_PIPE_MTE2_L1B | 12 | 虚拟 Pipe：MTE2 到 L1B |
| PIPE_NUM | 13 | Pipe 数量标记 |
| PIPE_UNASSIGNED | 99 | 未分配 Pipe |

---

## TCoreType（4 值）

源码：[HIVMAttrs.td#L298-L309](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L309)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| CUBE | 1 | Cube Core（矩阵计算） |
| VECTOR | 2 | Vector Core（向量计算） |
| CUBE_OR_VECTOR | 3 | 可在 Cube 或 Vector 上执行 |
| CUBE_AND_VECTOR | 4 | 需要在 Cube 和 Vector 上同时执行 |

---

## TFuncCoreType（4 值）

源码：[HIVMAttrs.td#L250-L261](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L250-L261)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| AIC | 1 | AI Cube Kernel |
| AIV | 2 | AI Vector Kernel |
| MIX | 3 | 混合 Cube+Vector Kernel |
| AIC_OR_AIV | 4 | AIC 或 AIV Kernel |

---

## TModuleCoreType（3 值）

源码：[HIVMAttrs.td#L271-L276](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L271-L276)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| AIC | 1 | 所有函数为 AIC |
| AIV | 2 | 所有函数为 AIV |
| MIX | 3 | 包含 AIC 和 AIV 函数 |

---

## PadMode（3 值）

源码：[HIVMAttrs.td#L330-L341](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L330-L341)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| PadNull | 0 | 无填充 |
| PadFirstElem | 1 | 使用第一个元素填充 |
| PadValue | 2 | 使用指定值填充 |

---

## EvictionPolicy（2 值）

源码：[HIVMAttrs.td#L356-L364](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L356-L364)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| EvictFirst | 0 | 先驱逐 |
| EvictLast | 1 | 后驱逐 |

---

## RoundMode（7 值）

源码：[HIVMAttrs.td#L378-L406](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L378-L406)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| RINT | 0 | `rint` | 就近舍入， ties to even |
| ROUND | 1 | `round` | 就近舍入， ties away from zero |
| FLOOR | 2 | `floor` | 向负无穷舍入 |
| CEIL | 3 | `ceil` | 向正无穷舍入 |
| TRUNC | 4 | `trunc` | 向零舍入 |
| ODD | 5 | `odd` | 向奇数舍入（Von Neumann） |
| TRUNCWITHOVERFLOW | 6 | `truncwithoverflow` | 截断并溢出 |

---

## UnsignedMode（4 值）

源码：[HIVMAttrs.td#L408-L425](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L408-L425)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| SI2SI | 0 | `si2si` | 有符号→有符号 |
| SI2UI | 1 | `si2ui` | 有符号→无符号 |
| UI2SI | 2 | `ui2si` | 无符号→有符号 |
| UI2UI | 3 | `ui2ui` | 无符号→无符号 |

---

## TypeFn / Cast（3 值）

源码：[HIVMAttrs.td#L431-L446](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L431-L446)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| cast_signed | 0 | 有符号类型转换 |
| cast_unsigned | 1 | 无符号类型转换 |
| bitcast | 2 | 位转换（不改变位模式） |

---

## CompareMode（6 值）

源码：[HIVMAttrs.td#L452-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L452-L473)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| EQ | 0 | `eq` | 等于 |
| NE | 1 | `ne` | 不等于 |
| LT | 2 | `lt` | 小于 |
| GT | 3 | `gt` | 大于 |
| GE | 4 | `ge` | 大于等于 |
| LE | 5 | `le` | 小于等于 |

---

## Event（8 值）

源码：[HIVMAttrs.td#L479-L498](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L479-L498)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| EVENT_ID0 | 0 | Event ID 0 |
| EVENT_ID1 | 1 | Event ID 1 |
| EVENT_ID2 | 2 | Event ID 2 |
| EVENT_ID3 | 3 | Event ID 3 |
| EVENT_ID4 | 4 | Event ID 4 |
| EVENT_ID5 | 5 | Event ID 5 |
| EVENT_ID6 | 6 | Event ID 6 |
| EVENT_ID7 | 7 | Event ID 7 |

---

## UnitFlag（4 值）

源码：[HIVMAttrs.td#L512-L523](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L523)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| DISABLED | 0 | 禁用 UnitFlag |
| RESERVED | 1 | 保留 |
| ENABLED_WITHOUT_UPDATE | 2 | 启用但不更新标志 |
| ENABLED_WITH_UPDATE | 3 | 启用并更新标志 |

---

## SyncBlockMode（6 值）

源码：[HIVMAttrs.td#L540-L555](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L540-L555)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| ALL_CUBE | 0 | 所有 Cube Core 同步 |
| ALL_VECTOR | 1 | 所有 Vector Core 同步 |
| ALL_SUB_VECTOR | 2 | 所有 Sub-Vector 同步 |
| BARRIER_CUBE | 3 | Cube-Cube 屏障 |
| BARRIER_VECTOR | 4 | Vector-Vector 屏障 |
| ALL | 5 | 所有 AIC/AIV 同步 |

---

## SyncBlockInstrMode（3 值）

源码：[HIVMAttrs.td#L569-L578](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L569-L578)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| INTER_BLOCK_SYNCHRONIZATION | 0 | 跨 Block 同步 |
| INTER_SUBBLOCK_SYNCHRONIZATION | 1 | 跨 Sub-Block 同步 |
| INTRA_BLOCK_SYNCHRONIZATION | 2 | Block 内同步（默认） |

---

## ReduceOperation（12 值）

源码：[HIVMAttrs.td#L596-L623](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L596-L623)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| none | 0 | 无归约 |
| sum | 1 | 求和 |
| prod | 2 | 求积 |
| max | 3 | 最大值 |
| min | 4 | 最小值 |
| max_with_index | 5 | 最大值及索引 |
| min_with_index | 6 | 最小值及索引 |
| any | 7 | 任意为真 |
| all | 8 | 全部为真 |
| xori | 9 | 异或（整数） |
| ori | 10 | 或（整数） |
| andi | 11 | 与（整数） |

---

## AtomicKind（9 值）

源码：[HIVMAttrs.td#L637-L658](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L637-L658)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| NONE | 0 | `none` | 无原子操作 |
| ADD | 1 | `add` | 原子加 |
| MAX | 2 | `max` | 原子最大值 |
| MIN | 3 | `min` | 原子最小值 |
| AND | 4 | `and` | 原子与 |
| OR | 5 | `or` | 原子或 |
| XOR | 6 | `xor` | 原子异或 |
| CAS | 7 | `or` | 比较并交换 |
| XCHG | 8 | `xor` | 原子交换 |

---

## AlignKind（3 值）

源码：[HIVMAttrs.td#L670-L678](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L670-L678)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| ALIGN | 0 | `align` | 对齐 |
| UNALIGNED | 1 | `unaligned` | 未对齐 |
| UNKNOWN | 2 | `unknown` | 未知 |

---

## AxisKind（3 值）

源码：[HIVMAttrs.td#L688-L696](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L688-L696)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| FIRST | 0 | `first` | 第一轴 |
| MIDDLE | 1 | `middle` | 中间轴 |
| LAST | 2 | `last` | 最后轴 |

---

## VFMode（3 值）

源码：[HIVMAttrs.td#L948-L954](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L948-L954)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| SIMD | 0 | SIMD 模式（向量化） |
| SIMT | 1 | SIMT 模式（多线程） |
| MIX | 2 | 混合 SIMD+SIMT 模式 |

---

## DataCacheKind（4 值）

源码：[HIVMAttrs.td#L736-L747](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L736-L747)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| ALL | 0 | `all` | 所有缓存 |
| UB | 1 | `ub` | UB 缓存 |
| OUT | 2 | `out` | 输出缓存 |
| ATOMIC | 3 | `atomic` | 原子缓存 |

---

## DCCIMode（2 值）

源码：[HIVMAttrs.td#L753-L763](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L753-L763)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| SINGLE_CACHE_LINE | 0 | `single_cache_line` | 单缓存行 |
| ALL_CACHE_LINES | 1 | `all_cache_lines` | 所有缓存行 |

---

## FixpipePreQuantMode（5 值）

源码：[HIVMAttrs.td#L783-L796](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L796)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| NO_QUANT | 0 | 不量化 |
| F322F16 | 1 | FP32→FP16 |
| S322I8 | 9 | INT32→INT8 |
| QF322F32_PRE | 15 | 量化 FP32→FP32 预处理 |
| F322BF16 | 16 | FP32→BF16 |

---

## FixpipePreReluMode（4 值）

源码：[HIVMAttrs.td#L803-L814](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L803-L814)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| NO_RELU | 0 | 不激活 |
| NORMAL_RELU | 1 | 标准 ReLU |
| LEAKY_RELU | 2 | Leaky ReLU |
| P_RELU | 3 | P-ReLU |

---

## FixpipeDualDstMode（3 值）

源码：[HIVMAttrs.td#L821-L830](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L821-L830)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| NO_DUAL | 0 | 单目的地模式 |
| ROW_SPLIT | 1 | M 维度拆分，M/2 x N 写入每个 UB |
| COLUMN_SPLIT | 2 | N 维度拆分，M x N/2 写入每个 UB |

---

## FixpipeDMAMode（3 值）

源码：[HIVMAttrs.td#L847-L854](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L847-L854)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| NZ2ND | 0 | `nz2nd` | nZ 格式→ND 格式 |
| NZ2DN | 1 | `nz2dn` | nZ 格式→DN 格式 |
| NZ2NZ | 2 | `normal` | 正常模式（nZ→nZ） |

---

## DeinterleaveMode（3 值）

源码：[HIVMAttrs.td#L865-L874](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L865-L874)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| CHANNEL_0 | 0 | 通道 0 |
| CHANNEL_1 | 1 | 通道 1 |
| ALL_CHANNELS | 999 | 所有通道 |

---

## DescaleMode（3 值）

源码：[HIVMAttrs.td#L886-L895](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L886-L895)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| DescaleNull | 0 | 不使用反量化 |
| DescalePerChannel | 1 | 按 Channel 反量化 |
| DescalePerTensor | 2 | 按 Tensor 反量化 |

---

## MatmulBiasMode（5 值）

源码：[HIVMAttrs.td#L903-L914](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L903-L914)

| 枚举名 | 值 | 说明 |
|--------|---|------|
| NoBias | 0 | 无 bias |
| PerChannelAdd | 1 | Per-channel 加法 bias |
| PostPerChannelAddWithSplitK | 2 | Split-K 后 Per-channel 加法 bias |
| ElementwiseAdd | 3 | 逐元素加法 bias |
| MMInitPerChannelAddWithSplitK | 4 | Split-K 初始化 Per-channel 加法 bias |

---

## MemoryEffect（3 值）

源码：[HIVMAttrs.td#L1046-L1055](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1046-L1055)

| 枚举名 | 值 | 字符串 | 说明 |
|--------|---|--------|------|
| READ | 0 | `read` | 读内存 |
| WRITE | 1 | `write` | 写内存 |
| READ_WRITE | 2 | `read_write` | 读写内存 |

---

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 测试用例：[attribute.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/attribute.mlir)
