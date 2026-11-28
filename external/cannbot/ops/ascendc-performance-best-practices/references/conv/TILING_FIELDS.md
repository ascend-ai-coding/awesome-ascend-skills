# Conv2DV2 tiling / tiling_data 字段解析

依据源码：

- `op_host/op_tiling/conv2d_v2_tiling.h` → `Conv2DTilingData`（下发到 device 的完整结构）
- `op_host/op_tiling/arch35/conv2d_v2_api_tilingdata.h` → `TConv2DTiling`（API tiling 子集）
- `op_kernel/arch35/conv2d_v2_tilingkey.h` → `tiling_key` 模板维

TTK 把 tiling bytes 按 **int32 little-endian** 打印；其中 `uint64` 字段会拆成 `(lo, hi)` 两个 int32。可用 `decode_tiling.py` 还原成命名字段。

## tiling_key

`tiling_key` 是 AscendC 模板参数的编码组合（见 `Conv2DV2Key::ASCENDC_TPL_ARGS_DECL`），主要维度：

| 维度 | 含义 |
|------|------|
| FmapTiling | AL1 是否 fullload / 仅 M fullload / other |
| WeightTiling | BL1 是否 fullload / 仅 N fullload / other |
| L1PingPong | AL1/BL1 ping-pong 开闭 |
| L0PingPong | AL0/BL0 ping-pong 开闭 |
| OutputOrder | HW mode / M mode |
| IterOrder | M 优先 / N 优先迭代 |
| GroupType | normal / ori-group / opt-group |
| EnableSmallChannel | C04 小通道路径 |
| WeightUbTrans | weight UB 转置 |
| FmapCopyMode | Load3D / DMA |
| InnerBatch | 单 batch / 1x1 multi / multi |
| NoPad / SmallWeight / SmallKernel | pad / 小 weight / 小 kernel 特化 |

示例：`448 (0x1c0)`、`8698 (0x21fa)`、`16777664 (0x10001c0)` 对应不同模板路径。

## tiling_data 字段（按结构体顺序）

### Shape / 单核切分

| 字段 | 类型 | 含义 |
|------|------|------|
| orgHi/orgWi/orgHo/orgWo | u64 | 原始输入/输出 H、W |
| orgHixWi | u64 | Hin×Win |
| singleCoreBatch/Ho/Wo | u64 | 单核处理的 batch / Ho / Wo |
| unionDataXt | u64 | 联合扩展数据（含部分打包标志） |
| hin/win/hout/wout | u64 | 运行态输入输出 H/W（与 org* 对应） |
| orgCi/orgCo | u32 | 原始 Cin/Cout |
| singleCoreCi/Co | u32 | 单核 Cin/Cout |
| batch/cin/cout/kh/kw | u32 | 运行态 N/Cin/Cout/Kh/Kw |

### L1 / L0 tile

| 字段 | 含义 |
|------|------|
| hoL1/woL1 | L1 上 feature 的 Ho/Wo tile |
| kAL1/kBL1 | L1 上 A/B 的 K tile |
| khL1/kwL1 | L1 上 kernel H/W tile |
| nBL1 | L1 上 weight N(Cout) tile |
| hoL0/woL0/kL0/nL0 | L0A/L0B/L0C 对应 tile |
| pBufferFlag | ping-pong 开关位图（L1/L0） |
| aL1SpaceSize | AL1 占用字节/空间 |
| multiNBL1 | BL1 上 N 方向倍数 |

### 卷积属性

| 字段 | 含义 |
|------|------|
| kernelH/W | 卷积核高宽 |
| strideH/W | 步长 |
| dilationH/W | 膨胀 |
| padTop/Bottom/Left/Right | padding |
| groups/enlarge | 分组卷积相关 |
| innerBatch | 内层 batch 合并数 |

### 多核切分（与 block_dim 相关）

| 字段 | 含义 |
|------|------|
| batchDim | batch 维核数 |
| groupDim | group 维核数 |
| nDim | Cout/N 维核数 |
| hoDim/woDim | 输出 H/W 维核数 |
| cinOpt/coutOpt/groupOpt | group-opt 路径优化后的通道/组 |

### 标志位

| 字段 | 含义 |
|------|------|
| hasBias | 是否有 bias |
| hf32Enable/hf32TransMode | HF32 |
| biasFullLoadFlag | bias 全载 |
| iterateMNOrder | M/N 迭代顺序 |
| quant*/relu*/clip*/dualOutput | Extend/Quant 路径扩展（普通 Conv2DV2 通常为 0） |
| offsetx/roundMode | 量化偏置与舍入 |

## 常用解读组合

性能分析时优先看：

1. **tiling_key** → 走哪条模板（fullload / pingpong / HW or M）
2. **hoL1, woL1, kAL1, kBL1, nBL1, hoL0, woL0, kL0, nL0** → 实际 tile
3. **batchDim, nDim, hoDim, woDim** → 多核怎么拆（和 `block_dim` 对照）
4. **pBufferFlag** → 是否开 ping-pong
5. **hasBias / hf32Enable** → 功能路径

### 本次 20 case 的观察

多数用例 `woL1=0` / `woL0=0`：表示走 **M-mode**（`OutputOrder=M`），把 `Ho×Wo` 折到 M 维，用 `hoL1/hoL0` 表达 M tile，而不是独立的 HW tile。  
`pBufferFlag` 常见取值：`7`（较全 ping-pong）、`15/19/27/31`（不同 L1/L0 组合）。
