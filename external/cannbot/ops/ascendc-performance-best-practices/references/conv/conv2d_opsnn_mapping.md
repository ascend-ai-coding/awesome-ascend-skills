# Conv2D ops-nn → 直调映射参考

> 本文件供 Conv Step T 的 T-A（TTK 采集）和 T-B（ops-nn 研读）阶段使用。
> 覆盖从 ops-nn 注册框架到直调工程所需的全部映射知识：tiling_key 位域解码、pBufferFlag 到直调 l1PP/l0PP 转换、tiling_data 95 字段到 Conv2DCase 25 字段的子集映射、AscendC Cube API 参数位域、ops-nn 源码文件索引与注册框架机制替代。
> 所有结论均经 TTK 采集数据验证。

---

## 1. tiling_key 解码

tiling_key 是 16 个模板参数字段的位域拼接，从低位到高位按 `ASCENDC_TPL_ARGS_DECL` 声明顺序排列。

### 1.1 字段位宽表

| # | 字段 | 位宽 | bit位 | 取值定义 |
|---|------|------|-------|---------|
| 1 | FmapTiling | 2 | 0-1 | 0=FULLLOAD_AL1, 1=ONLY_M_FULLLOAD_AL1_AL0, 2=OTHER |
| 2 | WeightTiling | 2 | 2-3 | 0=FULLLOAD_BL1, 1=ONLY_N_FULLLOAD_BL1_BL0, 2=OTHER |
| 3 | L1PingPong | 2 | 4-5 | 0=ALL_CLOSE, 1=AL1_OPEN, 2=BL1_OPEN, 3=ALL_OPEN |
| 4 | L0PingPong | 2 | 6-7 | 0=ALL_CLOSE, 1=AL0_OPEN, 2=BL0_OPEN, 3=ALL_OPEN |
| 5 | OutputOrder | 1 | 8 | 0=HW_MODE, 1=M_MODE |
| 6 | IterOrder | 1 | 9 | 0=M_FIRST, 1=N_FIRST |
| 7 | GroupType | 2 | 10-11 | 0=NORMAL, 1=ORI_GROUP, 2=OPT_GROUP, 3=OPT_SIMPLIFIED |
| 8 | EnableSmallChannel | 1 | 12 | 0=CLOSE, 1=OPEN |
| 9 | WeightUbTrans | 1 | 13 | 0=CLOSE, 1=OPEN |
| 10 | FmapCopyMode | 2 | 14-15 | 0=Load3D, 1=DMA |
| 11 | InnerBatch | 2 | 16-17 | 0=SINGLE, 1=1X1_MULTI, 2=MULTI |
| 12 | DisContinuous | 2 | 18-19 | 0=CLOSE, 1=INPUT_HWNC |
| 13 | BatchOne | 1 | 20 | 0=MULTI_BATCH, 1=ONE_BATCH |
| 14 | NoPad | 1 | 21 | 0=HAS_PAD, 1=NO_PAD |
| 15 | SmallWeight | 2 | 22-23 | 0=NOT_SMALL, 1=FULLLOAD_KL1_NL0, 2=SMALLER_THAN_BL0 |
| 16 | SmallKernel | 1 | 24 | 0=NOT_SMALL_KERNEL, 1=SMALL_KERNEL |

NCHW 格式下，`CONV2D_SCALAR_OPT_SEL` 把 BatchOne/NoPad/SmallWeight/SmallKernel 固定为 0（FRACTAL_Z 格式才有非零值）。bit 24 的 SmallKernel 在 NCHW 下也可为 1（SmallKernel 路径）。

### 1.2 解码函数

```python
def decode_tiling_key(key):
    fields = [
        ("FmapTiling", 2), ("WeightTiling", 2), ("L1PingPong", 2), ("L0PingPong", 2),
        ("OutputOrder", 1), ("IterOrder", 1), ("GroupType", 2), ("EnableSmallChannel", 1),
        ("WeightUbTrans", 1), ("FmapCopyMode", 2), ("InnerBatch", 2), ("DisContinuous", 2),
        ("BatchOne", 1), ("NoPad", 1), ("SmallWeight", 2), ("SmallKernel", 1),
    ]
    offset = 0
    result = {}
    for name, bw in fields:
        result[name] = (key >> offset) & ((1 << bw) - 1)
        offset += bw
    return result
```

### 1.3 典型 key 解码示例

| key | hex | FmapTiling | WeightTiling | L1PP | L0PP | Order | Iter | 对应 SEL 宏 |
|-----|-----|------------|-------------|------|------|-------|------|------------|
| 466 | 0x1d2 | 2(OTHER) | 0(FULLLOAD_BL1) | 1(AL1_OPEN) | 3(ALL_OPEN) | M | M_FIRST | ONLY_BL1_FULLLOAD |
| 501 | 0x1f5 | 1(ONLY_M) | 1(ONLY_N) | 3(ALL_OPEN) | 3(ALL_OPEN) | M | M_FIRST | ABL1_FULLLOAD_M_FIRST |
| 506 | 0x1fa | 2(OTHER) | 2(OTHER) | 3(ALL_OPEN) | 3(ALL_OPEN) | M | M_FIRST | NO_FULLLOAD_ALL_OPEN |
| 1000 | 0x3e8 | 0(FULLLOAD_AL1) | 2(OTHER) | 2(BL1_OPEN) | 3(ALL_OPEN) | M | N_FIRST | ONLY_AL1_FULLLOAD |
| 8698 | 0x21fa | 2(OTHER) | 2(OTHER) | 3(ALL_OPEN) | 3(ALL_OPEN) | M | M_FIRST | NO_FULLLOAD_ALL_OPEN + WeightUbTrans |
| 16777664 | 0x10001c0 | 0(FULLLOAD_AL1) | 0(FULLLOAD_BL1) | 0(ALL_CLOSE) | 3(ALL_OPEN) | M | M_FIRST | ABL1_FULLLOAD_M_FIRST + SmallKernel |
| 16777472 | 0x1000100 | 0(FULLLOAD_AL1) | 0(FULLLOAD_BL1) | 0(ALL_CLOSE) | 0(ALL_CLOSE) | M | M_FIRST | ABL1_FULLLOAD_M_FIRST + SmallKernel + L0全关 |

**来源**：`ops-nn/conv/conv2d_v2/op_kernel/arch35/conv2d_v2_tilingkey.h` 的 `ASCENDC_TPL_ARGS_DECL` + `ops-nn/conv/common/op_kernel/arch35/conv_tilingkey.h` 的枚举常量。

### 1.4 T-B 读取要点

T-B subagent 按 tiling_key 在 ops-nn 中追踪 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL_*` 宏，定位实际编译进该 key 的头文件/类/循环结构。关键文件见 [§5 ops-nn 文件索引](#5-ops-nn-文件索引)。

---

## 2. pBufferFlag 解码

pBufferFlag 是 tiling_data 中的一个 u32 字段，5-bit 位域，控制 L0A/L0B/L0C/AL1/BL1 的双缓冲开关。

### 2.1 位域定义

| bit | mask | 含义 | 来源常量 |
|-----|------|------|---------|
| 0 | 0x01 | L0A double buffer | `L0A_L0B_PB_FLAG_MASK=0x03` 低 bit |
| 1 | 0x02 | L0B double buffer | `L0A_L0B_PB_FLAG_MASK=0x03` 高 bit |
| 2 | 0x04 | L0C double buffer | 实际使用较少，L0C PP 由 tiling_key L0PingPong 控制时此 bit 可为 1 |
| 3 | 0x08 | AL1 double buffer | `L1_PB_OFFSET=3` |
| 4 | 0x10 | BL1 double buffer | `WEIGHT_L1_PB_OFFSET=4` |

**来源**：`ops-nn/conv/common/op_host/op_tiling/arch35/conv_template_utils.h:196-199`

### 2.2 转换到直调 Conv2DCase 编码

直调 `Conv2DCase` 用独立的 `l1PingPong`（bit0=AL1, bit1=BL1）和 `l0PingPong`（bit0=AL0, bit1=BL0）。

```python
def pBufferFlag_to_cannbench(pb):
    l1PingPong = (pb >> 3) & 3   # bit3=AL1, bit4=BL1
    l0PingPong = pb & 3          # bit0=L0A, bit1=L0B
    return l1PingPong, l0PingPong
```

### 2.3 验证说明

上述转换已经多批 TTK 采集数据验证：`pBufferFlag_to_cannbench` 对所有出现过的 pBufferFlag 值（4/7/11/15/19/27/31）均产出与 tiling_key 中 L1PingPong/L0PingPong 字段一致的 `l1PingPong`/`l0PingPong` 值。

### 2.4 常见误读

**pBufferFlag=11 不是 l1PP=3**：bit3=1（AL1 PP），bit4=0（BL1 单缓冲），所以 l1PP=1（只 AL1 PP）。这与 SplitN 的「BL1 单缓冲 + hoistWeight」一致。误读为 l1PP=3 会导致 T-B 错误地认为官方在 SplitN case 上同时开了 AL1+BL1 双缓冲，而直调 `ValidateCase` 要求 SplitN/SplitM 时 AL1 必须单缓冲（见 `conv2d_tiling.template` 的 `ValidateCase` 约束）。

---

## 3. tiling_data 字段定义

tiling_data 是 `TConv2DTiling` 结构体的二进制序列化，按 int32 little-endian 传输。uint64 字段拆成 (lo, hi) 两个 int32。完整结构体 95 字段、376 字节。

### 3.1 完整字段分类

| 类别 | 字段 | 类型 | 含义 |
|------|------|------|------|
| **Shape** | orgHi/orgWi/orgHo/orgWo | u64 | 原始输入/输出 H/W |
| | orgHixWi | u64 | Hin×Win |
| | singleCoreBatch/Ho/Wo | u64 | 单核 batch/Ho/Wo |
| | unionDataXt | u64 | Load3D xt 寄存器预编码值 |
| | hin/win/hout/wout | u64 | 运行态 H/W |
| | orgCi/orgCo | u32 | 原始 Cin/Cout |
| | singleCoreCi/Co | u32 | 单核 Cin/Cout |
| | batch/cin/cout/kh/kw | u32 | 运行态 N/Cin/Cout/Kh/Kw |
| **L1 tile** | hoL1/woL1 | u32 | L1 feature 的 Ho/Wo tile（M-mode 下 woL1=0，hoL1 表达 M tile） |
| | kAL1/kBL1 | u32 | L1 A/B 的 K tile（已含 kh×kw，即 kAL1=CinTileA×kh×kw） |
| | khL1/kwL1 | u32 | L1 kernel H/W tile（通常 0） |
| | nBL1 | u32 | L1 weight N(Cout) tile |
| **L0 tile** | hoL0/woL0 | u32 | L0 feature 的 Ho/Wo tile（M-mode 下 woL0=0） |
| | kL0 | u32 | L0 K tile |
| | nL0 | u32 | L0 N tile |
| **标志** | pBufferFlag | u32 | ping-pong 位图（见 [§2](#2-pbufferflag-解码)） |
| | groups/enlarge | u32 | 分组卷积 |
| | aL1SpaceSize | u32 | AL1 占用字节（host 预计算） |
| | multiNBL1 | u32 | BL1 N 方向倍数（nBL1/nL0，仅 DMA/WeightUbTrans 路径使用） |
| **L0 步长** | cinAInCore/cinATailInCore | u32 | 核内 Cin 满块/尾块 |
| | cinBInCore/cinBTailInCore | u32 | |
| | mStep/kStep/nStep | u32 | L0 M/K/N 步长预计算 |
| | fmapKStride/weightKStride | u32 | K 维 stride 预计算 |
| | cinOffsetBlockInGM/coutOffsetBlock | u32 | GM 偏移步长 |
| | nL1DivBlockSize | u32 | L1 N 分块大小 |
| **卷积属性** | kernelH/kernelW | u32 | 卷积核 H/W |
| | strideH/strideW | u32 | 步长 |
| | dilationH/dilationW | u32 | 膨胀 |
| | padTop/Bottom/Left/Right | u32 | padding |
| | innerBatch | u32 | 内层 batch 合并数 |
| **多核** | batchDim/groupDim/nDim | u32 | batch/group/N 维核数 |
| | hoDim/woDim | u32 | 输出 H/W 维核数 |
| | cinOpt/coutOpt/groupOpt | u32 | group-opt 优化后通道/组 |
| **u8 标志** | iterateMNOrder | u8 | M/N 迭代顺序 |
| | biasFullLoadFlag | u8 | bias 全载 |
| | hf32Enable/hf32TransMode | u8 | HF32 |
| | hasBias/hasScale | u8 | bias/scale 使能 |
| | dualOutput | u8 | 双输出 |
| | quantMode0/1, reluMode0/1, clipMode0/1 | u8 | 量化/激活/裁剪 |
| | fixedShiftValue | u8 | 定点移位 |
| | offsetx | i8 | 量化偏置 |
| | roundMode | i8 | 舍入模式 |

**来源**：`ops-nn/conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_api_tilingdata.h` + `ttk_conv2d/decode_tiling.py`

### 3.2 直调 kernel 需要的字段子集

直调 kernel 的 `Conv2DCase` 结构体只需要约 25 个字段（从 tiling_data 中提取）：

| Conv2DCase 字段 | tiling_data 来源 | 说明 |
|-----------------|-----------------|------|
| n/ci/hi/wi/co/kh/kw | batch/cin/hi/wi/cout/kh/kw | 运行态 shape |
| strideH/W, padT/B/L/R, dilH/W | 同名 | 卷积属性 |
| batchDim/mDim/nDim | batchDim/hoDim/nDim | 多核切分（mDim=hoDim） |
| coreM | singleCoreHo | 单核 M |
| mAL1/kAL1/kBL1/nBL1 | hoL1/kAL1/kBL1/nBL1 | L1 tile |
| mL0/kL0/nL0 | hoL0/kL0/nL0 | L0 tile |
| l1PingPong/l0PingPong | pBufferFlag 转换 | 见 [§2.2](#22-转换到直调-conv2dcase-编码) |
| dtype | input_dtypes | FP16/FP32/BF16 |
| innerBatch/kAccum/hf32 | 同名或推导 | 扩展特性 |

**其余 70 个字段是 ops-nn 的 host 预计算值**（如 mStep/kStep/nStep/fmapKStride 等，用于减少 kernel 侧 Scalar 指令）。直调在 kernel 侧动态计算这些值，不需要从 tiling_data 传入。T-B subagent 不应把 host 预计算字段搬进直调 `Conv2DCase`。

---

## 4. AscendC API 参数位域参考

> 以下内容在 T-C 直接拷贝 `.template` 时通常不需要（模板已封装），但在 T-C 从设计文档生成代码、或调试 API 报错时需要参照。

### 4.1 数据搬运 GM→L1

| API | 参数结构 | 用途 | 关键参数 |
|-----|---------|------|---------|
| `DataCopy<T>(dst, src, Dn2NzParams)` | Dn2NzParams | NCHW fmap/weight GM→L1 ND→NZ | dnNum, nValue, dValue, srcDnMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride |
| `DataCopy<T>(dst, src, Nd2NzParams)` | Nd2NzParams | NHWC 或 1×1 weight GM→L1 ND→NZ | ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride |
| `DataCopyPad<T>(dst, src, DataCopyParams, DataCopyPadParams)` | DataCopyParams + DataCopyPadParams | Bias GM→L1 带右侧 padding | blockCount, blockLen, leftPad=0, rightPad=补齐量 |
| `DataCopy<T>(dst, src, DataCopyParams)` | DataCopyParams | L1→C2(BT) bias 搬运 | blockCount, blockLen(单位32B) |

**1×1 weight 特化**：kh×kw==1 时用 `Nd2NzParams`（weight 已是 [Co,Ci,1,1]），kh×kw>1 时用 `Dn2NzParams`（需要把 [Co,Ci,kh,kw] 重排为 NZ）。

**c04 模式**：Ci<4 时用 `DataCopy<T,true>`（enableSmallC0=true），C0_SIZE 对齐到 4*sizeof(T)。

### 4.2 数据搬运 L1→L0

| API | 参数结构 | 用途 | 关键参数 |
|-----|---------|------|---------|
| `LoadData<A2, A1, T>(dst, src, Load3DBitModeParam)` | Load3DBitModeParam | Fmap L1→L0A img2col | SetConfig0(xm), SetConfig1(xt) |
| `LoadData<B2, B1, T>(dst, src, Load2DBitModeParam)` | Load2DBitModeParam | Weight L1→L0B 2D搬运 | SetMStartPosition, SetKStartPosition, SetMStep, SetKStep, SetSrcStride, SetDstStride |

**Load3D Config0 (xm) 位域**：

| bit位 | 字段 | 含义 |
|-------|------|------|
| 0-15 | currentKL0 (kStep) | K 维传输长度 |
| 16-31 | mExtension (mStep) | M 维传输长度 |
| 32-47 | posK | K 维起点 |
| 48-63 | mStartPt (posM) | M 维起点 |

**Load3D Config1 (xt) 位域**：

| bit位 | 字段 | 含义 |
|-------|------|------|
| 0-5 | strideW | W 步长 [1,63] |
| 6-11 | strideH | H 步长 [1,63] |
| 12-19 | kernelW | 卷积核 W [1,255] |
| 20-27 | kernelH | 卷积核 H [1,255] |
| 28-35 | dilationW | W 膨胀 [1,255] |
| 36-43 | dilationH | H 膨胀 [1,255] |
| 44 | kernelW_high | kernelW+256 标志 |
| 45 | kernelH_high | kernelH+256 标志 |
| 48-63 | channelSize | 源通道数 [1,65535] |

**Load2D 参数**：

| 方法 | 含义 | 典型取值 |
|------|------|---------|
| SetMStartPosition | M 轴起点(单位16) | nOffset/16（SplitN 时选 nL0 子块） |
| SetKStartPosition | K 轴起点(单位32B) | localK/k0 |
| SetMStep | M 方向长度(单位16) | curN/16 |
| SetKStep | K 方向长度(单位32B) | curK/k0 |
| SetSrcStride | 源 K 间隔(单位512B) | nBL1/16 |
| SetDstStride | 目的 K 间隔(单位512B) | curN/16 |
| SetIfTranspose | 是否转置 | false |

### 4.3 Cube 计算

| API | 参数结构 | 关键字段 |
|-----|---------|---------|
| `Mmad<L0cT, FmapT, WeightT>(dst, srcA, srcB, MmadParams)` | MmadParams | m, n, k, cmatrixInitVal, cmatrixSource, unitFlag |

**MmadParams 字段**：

| 字段 | 含义 | 取值 |
|------|------|------|
| m | 左矩阵 A 高度 | AlignUp(curM, 16) |
| n | 右矩阵 B 宽度 | AlignUp(curN, 16) |
| k | A 宽度=B 高度 | curK |
| cmatrixInitVal | 是否初始化 L0C | 首段=true(清零/加载bias), 后续=false(累加) |
| cmatrixSource | C 初始值来源 | 首段=true(从 BT 加载 bias), 后续=false |
| unitFlag | Mmad-Fixpipe 并行控制 | 0=DISABLE, 2=ENABLE_ONLY(K中间), 3=ENABLE_WITH_FLIP(K末次) |

### 4.4 输出 L0C→GM

| API | 参数结构 | 关键字段 |
|-----|---------|---------|
| `Fixpipe<T, float, CFG_COLUMN_MAJOR>(dst, src, FixpipeParamsC310)` | FixpipeParamsC310 | nSize, mSize, srcStride, dstStride, quantPre, deqScalar |

**FixpipeParamsC310 字段**：

| 字段 | 含义 | 取值 |
|------|------|------|
| nSize | N 方向大小 | curN |
| mSize | M 方向大小 | curM |
| srcStride | 源相邻 Z 偏移(单位C0_SIZE) | AlignUp(curM, 16) |
| dstStride | 目的 ND 每行元素数 | Ho×Wo (M-mode NCHW) |
| quantPre | 量化模式 | NoQuant(fp32), F322F16(fp16), F322BF16(bf16) |
| deqScalar | 反量化 scale | 1065353216 (=1.0f 的 uint64, 无 scale 时) |
| unitFlag | 与 Mmad 配合 | ENABLE_WITH_FLIP(3) |

### 4.5 Flag 同步

| HardEvent | 源→目标 | 用途 |
|-----------|---------|------|
| MTE2_MTE1 | GM→L1 完成后通知 L1→L0 | AL1/BL1 数据就绪 |
| MTE1_MTE2 | L1→L0 完成后释放 L1 buffer | AL1/BL1 可覆写 |
| M_MTE1 | Mmad 完成后释放 L0 buffer | L0A/L0B 可覆写 |
| MTE1_M | L1→L0 完成后通知 Mmad | L0A/L0B 数据就绪 |
| M_FIX | Mmad 完成后通知 Fixpipe | L0C 可输出 |
| FIX_M | Fixpipe 完成后释放 L0C | L0C 可覆写 |

**合法同步组合矩阵**：

| 源＼目标 | M | MTE1 | MTE2 | MTE3 | FIX |
|---------|---|------|------|------|-----|
| M | PipeBarrier | M_MTE1 | M_MTE2 | — | M_FIX |
| MTE1 | MTE1_M | PipeBarrier | MTE1_MTE2 | MTE1_MTE3 | MTE1_FIX |
| MTE2 | MTE2_M | MTE2_MTE1 | PipeBarrier | MTE2_MTE3 | — |
| MTE3 | — | MTE3_MTE1 | MTE3_MTE2 | PipeBarrier | — |
| FIX | FIX_M | FIX_MTE1 | — | — | PipeBarrier |

### 4.6 Event ID 分配

```
AL1 pingpong:  MTE2_MTE1 event {0, 1}
BL1 pingpong:  MTE2_MTE1 event {2, 3}
Bias:          MTE2_MTE1 event {4}
L0 pingpong:   M_MTE1 / MTE1_M event {0, 1}
Mmad→Fixpipe:  M_FIX / FIX_M event {2, 3}
```

AL1/BL1 用独立 event pair，使两条 L1 流以不同 K 速率运行互不干扰。

### 4.7 Load3D 配置 API

| API | 参数 | 用途 |
|-----|------|------|
| `Load3DSetFMatrixCal(hiLoadL1, wiLoadL1, padList[4])` | hi/wi/pad{L,R,T,B} | 设置 FMatrix 属性 |
| `Load3DSetPaddingCal(offsetx)` | offsetx | 设置 pad 值（量化场景） |
| `SetLoadDataRepeatWithStride(LoadDataRepeatParamWithStride)` | repeatStride, repeatTime, repeatMode, dstStride | 设置 repeat 参数 |

### 4.8 HF32

| API | 参数 | 用途 |
|-----|------|------|
| `SetHF32Mode(enable)` | 0=DISABLE, 1=ENABLE | fp32→HF32 转换，性能翻倍有精度损失 |
| `SetHF32TransMode(mode)` | 0=NEAREST_EVEN, 1=NEAREST_ZERO | 舍入模式 |

---

## 5. ops-nn 文件索引

T-B subagent 按 tiling_key 读 ops-nn 时，以下文件按用途索引：

### 5.1 设备侧 kernel 源码

| 文件 | 路径（相对 `ops-nn/conv/conv2d_v2/`） | 用途 |
|------|------|------|
| 入口 | `op_kernel/conv2d_v2.cpp` | kernel 入口 + 模板分发 |
| Conv2dBase | `op_kernel/arch35/conv2d_v2.h` | Init/Process/RunConv2dKernel |
| Conv2dIntf | `op_kernel/arch35/conv2d_v2_intf.h` | flag/event/kPreLoadAFlag |
| Conv2d API | `op_kernel/arch35/conv2d_v2_api.h` | API 类 |
| API Impl | `op_kernel/arch35/conv2d_v2_api_impl.h` | queueAL1/queueBL1 |
| Instr Impl | `op_kernel/arch35/conv2d_v2_instr_impl.h` | LoadAL1ToolsMMode（核心 flag 同步） |
| Instr Base | `op_kernel/arch35/conv2d_v2_instr_base_impl.h` | Load3D/Load2D/Mmad/Fixpipe |
| DMA Impl | `op_kernel/arch35/conv2d_v2_instr_dma_impl.h` | DMA 搬运 |
| Common | `op_kernel/arch35/conv2d_v2_common_func.h` | 通用函数 |
| TilingKey | `op_kernel/arch35/conv2d_v2_tilingkey.h` | tiling_key 字段定义 + SEL 宏 |
| Conv Common Key | `common/op_kernel/arch35/conv_tilingkey.h`（相对 `ops-nn/conv/`） | 枚举常量 |
| Template Utils | `common/op_host/op_tiling/arch35/conv_template_utils.h`（相对 `ops-nn/conv/`） | pBufferFlag mask 定义 |
| TilingData | `op_host/op_tiling/arch35/conv2d_v2_api_tilingdata.h` | TConv2DTiling 结构体 |

### 5.2 host 侧 tiling 源码

| 文件 | 路径（相对 `ops-nn/conv/conv2d_v2/`） | 用途 |
|------|------|------|
| Base Tiling | `op_host/op_tiling/arch35/conv2d_v2_base_tiling_basic_block.cpp` | BasicBlock 多核切分 + L1/L0 tiling 搜索 |
| Key Select | `op_host/op_tiling/arch35/conv2d_v2_base_tiling_tilingkey.cpp` | tiling_key 字段决策逻辑 |

### 5.3 ops-nn 关键机制 → 直调替代

T-B 读 ops-nn 时注意：ops-nn 的注册框架机制在直调中有不同替代方案，不要把注册宏/queue 搬进直调。

| ops-nn 机制 | 直调替代 |
|------------|---------|
| `TQue<QuePosition::A1, depth>` | 手动 `SetFlag/WaitFlag<MTE2_MTE1/MTE1_MTE2>` + `al1Bufs_` |
| `AllocTensor/EnQue/DeQue/FreeTensor` | `LocalTensor<T>(TPosition::A1, offset, size)` + flag |
| `GET_TILING_DATA` | `Conv2DCase` 结构体参数传递 |
| `SetSysWorkspace` | 不需要（直调无系统 workspace） |
| `KERNEL_TASK_TYPE` | `__cube__` 属性 |
| `ASCENDC_TPL_*` 模板分发 | runtime `if/switch` 分支 |

**set_flag/wait_flag 比 queue 性能好得多**：queue 的 AllocTensor/EnQue/DeQue/FreeTensor 有状态机开销和缓冲区映射开销，手动 flag 直接操作硬件信号量，Scalar 开销最低。

### 5.4 queueAL1 / queueBL1 语义

| queue | depth | 决定条件 | 直调等价 |
|-------|-------|---------|---------|
| queueAL1 | 2 | `kPreLoadAFlag = (l1PingPong==AL1_OPEN) && (weightTiling==FULLLOAD_BL1)` | al1Bufs_ = al1PP_ ? 2 : 1 |
| queueBL1 | 1 | 始终单缓冲 | bl1Bufs_ = bl1PP_ ? 2 : 1（直调）或 1（ops-nn 语义） |

**关键**：ops-nn 的 `kPreLoadAFlag=true` 对应 pBufferFlag bit3=1（AL1 双缓冲），bit4=0（BL1 单缓冲）。直调用 `l1PingPong` 的 bit0/bit1 分别控制 AL1/BL1，更灵活。

### 5.5 SplitN 在 ops-nn 中的循环结构

ops-nn 在 `IterateMFirstMMode` 的 nL0Iter 循环层实现 SplitN：
- AL1 在 nL0 推进时不重新加载（复用同一 fmap 块）
- `multiNBL1 = nBL1 / nL0` 决定 nL0 迭代次数
- BL1 在 nL0 子块间不重新加载（BL1 窗口覆盖 nBL1，L0B 用 MStartPosition 选偏移）

直调模板的 `ComputeWindow` 在 nSub 循环层实现同样逻辑，AL1 加载一次，nSub 步长=nL0。

