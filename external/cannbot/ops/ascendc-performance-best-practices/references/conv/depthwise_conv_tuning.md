# Depthwise Conv2D 模板翻译

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 本文件是对已实现模板 [`dav310/dwconv2d_kernel.template`](templates/dav310/dwconv2d_kernel.template) 的逐层翻译。
> 说明 kernel 的循环结构、每个循环用到的 AscendC API、以及流水线 flag / CrossCore 同步控制。
> 对应的接口说明见 [dwconv2d_mix.md](templates/dwconv2d_mix.md)；tiling 契约见 [`dwconv2d_tiling.template`](templates/dwconv2d_tiling.template)。

---

## 1. 整体架构

```
DepthwiseConv2dKernel<DTYPE, L1_PINGPONG>（__mix__(1,2)）
├── Init()        派生量计算 + L1 布局 + 核映射（AIV/AIC 共用）
└── Process()     AIV 路径 + AIC 路径（CrossCore 握手）
    ├── AIV: SetupGroup → TransNCHW2NZ → BL1
    └── AIC: SetupGroup → LoadAL1 → ComputeL0 → DoCopyOut
        ├── ComputeL0()      M/N/K 三层循环（L0 双缓冲）
        │   ├── DoLoadAL0()   L1→L0A（Load3D）
        │   ├── DoLoadBL0()   L1→L0B（Load2D）
        │   ├── DoMmad()      Cube 矩阵乘
        │   └── DoCopyOut()   L0C→GM
        └── bias: GM→L1→BT
```

源文件：`dwconv2d_kernel.template:104-806`

---

## 2. Process() AIV 路径（权重 NZ 变换）

AIV 负责 per-group 权重的 NCHW→NZ 变换，写入 BL1 供 AIC 消费。

```
AIV Process:
  for g (singleGroupOpt):              ← per-group 循环
    [if BL1_PINGPONG] 仅 vecId == g%2 处理
    SetupGroup(curGroup)               ← 算 enlarge/enlargeTail、GM 偏移

    ubNd = Duplicate(0)                ← 清零 ND buffer
    DataCopy(ubNd, filterGm, NdDmaParams)  ← 权重 GM→UB（NCHW 紧凑布局）
    TransNCHW2NZ()                     ← NCHW→NZ 变换（Gather）
    DataCopy(bl1, ubNz, nzCp)         ← UB→BL1

    CrossCoreSetFlag(PIPE_MTE3, syncMte3Mte1)  ← 通知 AIC BL1 就绪
```

### 2.1 TransNCHW2NZ

权重从 NCHW 紧凑布局重排为 Cube 需要的 NZ 分形布局，通过 `Gather` + 索引表实现。

```
TransNCHW2NZ():
  SetIndex()                          ← 构建索引表（每 K0 元素步长=khkw）
  __VEC_SCOPE__:
    for ci1Opt:                        ← Cin/K0 块循环
      for khkw:                        ← kh×kw 循环
        for co:                        ← Cout/N0 循环
          Gather(gatherReg, srcAddr+srcOffset, indexReg)
          Reg::StoreAlign(dstAddr+dstOffset, gatherReg, mask)
```

源文件：`dwconv2d_kernel.template:294-378, 464-549`

### 2.2 SetIndex

构建 `Gather` 的索引表：`[0, khkw, 2*khkw, ..., (K0-1)*khkw]`，按 `kUbSize_` 步长重复填充 256 元素。

```c
// Reg API 实现
Reg::LoadAlign<IndexT>(indexReg, indexAddr);
for (repeat = 0..repeatTimes):
    Reg::Adds<IndexT>(indexReg, indexReg, nStride, maskReg);
    Reg::StoreAlign<IndexT>(indexAddr + dstOffset, indexReg, maskReg);
```

源文件：`dwconv2d_kernel.template:464-498`

---

## 3. Process() AIC 路径（Cube 计算）

AIC 消费 AIV 写入的 BL1 权重，加载 fmap 到 AL1，执行 Cube Mmad。

```
AIC Process:
  SetFlag(初始化所有 L0/L1/bias event)   ← 标记初始状态为已释放

  for g (singleGroupOpt):              ← per-group 循环
    SetupGroup(curGroup)
    CrossCoreWaitFlag(PIPE_MTE1, aicSyncBase)  ← 等 AIV 的 BL1 就绪

    for bIter (batchCount):            ← batch 维
      for mAL1Iter (step=hoL1):       ← M 维 L1 切分
        Wait(MTE1_MTE2, AL1[l1BufId])
        LoadAL1(l1BufId, mGlobal, curMAL1)  ← fmap GM→AL1（NCHW→NZ）
        Wait(MTE2_MTE1, AL1[l1BufId])
        ComputeL0(al1, bl1, ...)      ← M/N/K 三层循环
        SetFlag(MTE1_MTE2, AL1[l1BufId])
        [if AL1_PINGPONG] al1PP++

    [if !isLastGroup]:
      CrossCoreSetFlag(PIPE_MTE1, aicSyncBase+1)  ← 通知 AIV 可以写下一组

  Wait(所有 L0/L1/bias event)
```

源文件：`dwconv2d_kernel.template:381-461`

---

## 4. ComputeL0（M/N/K 三层循环）

```
ComputeL0(al1, bl1, mGlobal, curMAL1, nIter, curN, kTotal):
  [if enableBias_]:
    InitConstValue(biasL1, 0)          ← 清零 bias L1
    DataCopyPad(biasL1, biasGm)        ← bias GM→L1
    DataCopy(biasBT, biasL1[nIter])    ← bias L1→BT(C2)

  for mL0It (step=hoL0):              ← M 维 L0 切分
    SetLoadDataRepeatWithStride(...)   ← Load3D repeat 参数
    for nL0It (step=nL0):             ← N 维 L0 切分
      Wait(FIX_M, cl0BufId)           ← 等 L0C 释放
      for kIt (step=kL0):            ← K 维 L0 切分
        Wait(M_MTE1, mte1Flag)        ← 等 L0A/L0B 释放
        DoLoadAL0(al1, al0, ...)      ← L1→L0A（Load3D img2col）
        DoLoadBL0(bl1, bl0, ...)      ← L1→L0B（Load2D）
        SetFlag/Wait(MTE1_M, mte1Flag)
        DoMmad(cl0, al0, bl0, ...)    ← Cube 矩阵乘
        SetFlag(M_MTE1, mte1Flag)     ← 释放 L0A/L0B
        pingPongFlag ^= 1

      SetFlag/Wait(M_FIX, cl0BufId)   ← Mmad→Fixpipe 同步
      DoCopyOut(cl0, ...)             ← L0C→GM 输出
      SetFlag(FIX_M, cl0BufId)        ← 释放 L0C
      l0cPP++
```

**L0A/L0B 双缓冲**：`pingPongFlag` 在 0/1 间 toggle，`mte1Flag = pingPongFlag + 6`（event ID 基址）。
**L0C 双缓冲**：`cl0BufId = l0cPP & 1`，两个 L0C 半区交替。

源文件：`dwconv2d_kernel.template:607-728`

---

## 5. 数据搬运 API

### 5.1 GM→L1（AIC 侧）

| 操作 | API | 参数结构 | 源码位置 |
|------|-----|---------|---------|
| fmap GM→AL1（NCHW→NZ） | `DataCopy(al1Dst, fmapGm, Dn2NzParams)` | dnNum=1, nValue=curHiLoadL1×Win, dValue=enlargeActual | `LoadAL1:552-605` |
| bias GM→L1（带右 pad） | `DataCopyPad(biasL1, biasGm, DataCopyParams, DataCopyPadParams)` | blockLen=curN×sizeof(T), rightPad | `ComputeL0:631-648` |
| bias L1→BT(C2) | `DataCopy(biasBT, biasL1[nIter], DataCopyParams)` | blockCount=AlignUp(curN,GN0)×sizeof(T)/32 | `ComputeL0:650-658` |

### 5.2 GM→UB→BL1（AIV 侧）

| 操作 | API | 参数结构 | 源码位置 |
|------|-----|---------|---------|
| 权重 GM→UB（NCHW 紧凑） | `DataCopy<DTYPE, 3, kDefaultMultiCopyConfig>(ubNd, filterGm, NdDmaParams)` | 3 维 loop: loop0=srcKSize, loop1=C_PER_GROUP, loop2=enlargeActual | `Process AIV:339-350` |
| UB→BL1（NZ 布局） | `DataCopy(bl1, ubNz, DataCopyParams)` | blockCount=kTotal/K0, blockLen=co1Opt×GN0×K0×sizeof(T)/32 | `Process AIV:371-376` |
| UB 清零 | `Duplicate<DTYPE>(ubNd, 0, nzBufElems)` | | `Process AIV:333` |

### 5.3 L1→L0

| 操作 | API | 关键参数 | 源码位置 |
|------|-----|---------|---------|
| fmap L1→L0A（img2col） | `LoadData<A2, A1, DTYPE>(al0, al1, Load3DBitModeParam)` | SetConfig0(xm): kStep/mStep/posK/posM; SetConfig1(xt): stride/kernel/dilation/cin | `DoLoadAL0:731-742` |
| weight L1→L0B（2D） | `LoadData<B2, B1, DTYPE>(bl0, bl1, Load2DBitModeParam)` | MStartPosition=nOff/GN0, KStartPosition, MStep, KStep, SrcStride, DstStride | `DoLoadBL0:744-762` |

### 5.4 Cube 计算

| 操作 | API | 关键参数 | 源码位置 |
|------|-----|---------|---------|
| 矩阵乘 | `Mmad(cl0, al0, bl0, MmadParams)` | m=safeMAlign, n=curNL0, k=curK, cmatrixInitVal, cmatrixSource | `DoMmad:765-779` |

### 5.5 L0C→GM 输出

| 操作 | API | 关键参数 | 源码位置 |
|------|-----|---------|---------|
| 输出（量化/格式转换） | `Fixpipe<DTYPE, L0cT, CFG_COLUMN_MAJOR>(outputGm, cl0, FixpipeParamsC310)` | nSize, mSize, srcStride, dstStride, quantPre, deqScalar | `DoCopyOut:782-802` |

### 5.6 HF32

| 操作 | API | 说明 | 源码位置 |
|------|-----|------|---------|
| 进入/退出 HF32 | `SetHF32Mode(hf32Enable)` | FP32 专用 | `Init:251-257` |
| 舍入模式 | `SetHF32TransMode(hf32TransMode)` | 0=NEAREST_EVEN, 1=NEAREST_ZERO | `Init:251-257` |

---

## 6. AIV/AIC CrossCore 同步

### 6.1 CrossCore Flag 机制

AIV 写完 BL1 后通过 `CrossCoreSetFlag` 通知 AIC，AIC 通过 `CrossCoreWaitFlag` 等待。

```
AIV 侧:
  CrossCoreSetFlag<G_CV_ENHANCE_MODE, PIPE_MTE3>(syncMte3Mte1)

AIC 侧:
  CrossCoreWaitFlag<G_CV_ENHANCE_MODE, PIPE_MTE1>(aicSyncBase)
  [处理完当前 group 后]
  CrossCoreSetFlag<G_CV_ENHANCE_MODE, PIPE_MTE1>(aicSyncBase + 1)
```

- `G_CV_ENHANCE_MODE = 4`：enhance 模式
- `G_CV_SYNC_STRIDE = 16`：BL1 乒乓时每组 AIC 用不同 syncBase（`g % 2 * 16`）
- `G_CV_SYNC_ID_MTE3_MTE1 = 0`：MTE3→MTE1 方向
- `G_CV_SYNC_ID_MTE1_MTE3 = 1`：MTE1→MTE3 方向

源文件：`dwconv2d_kernel.template:48-51, 311-313, 367-378, 408-413, 448-449`

### 6.2 BL1 乒乓与 AIV 交替

BL1 乒乓开启时（`L1_PINGPONG & 2`），两个 AIV 按 `g % 2` 交替处理 group：

```
AIV vec0: g=0, 2, 4, ...   → BL1[0]
AIV vec1: g=1, 3, 5, ...   → BL1[1]
AIC:      按 aicSyncBase = g%2 * 16 等待对应 AIV
```

BL1 单缓冲时（`!(L1_PINGPONG & 2)`），vec1 无独立 BL1 空间，vec0 串行处理所有 group。

源文件：`dwconv2d_kernel.template:303-378`

---

## 7. Flag 同步与 Event ID

### 7.1 AIC 侧 Event ID

```
AL1 pingpong:      MTE1_MTE2 / MTE2_MTE1 event {0, 1}
Bias:              MTE1_MTE2 / MTE2_MTE1 event {2}
Bias BT:           M_MTE1 event {3}
L0A/L0B pingpong: M_MTE1 / MTE1_M event {6, 7}  (base=6)
L0C pingpong:     M_FIX / FIX_M event {0, 1}
```

### 7.2 流水线模式

#### ComputeL0（M/N/K 三层循环，L0 双缓冲）

```
for mL0It (step=hoL0):
  for nL0It (step=nL0):
    Wait(FIX_M, cl0BufId)               ← 等 L0C 释放
    for kIt (step=kL0):
      mte1Flag = pingPongFlag + 6
      Wait(M_MTE1, mte1Flag)            ← 等 L0A/L0B 释放
      DoLoadAL0 + DoLoadBL0             ← MTE1 加载（与上一轮 Mmad 并行）
      SetFlag(MTE1_M, mte1Flag)
      Wait(MTE1_M, mte1Flag)            ← M 等数据就绪
      DoMmad                            ← Cube 计算
      SetFlag(M_MTE1, mte1Flag)         ← 释放 L0A/L0B
      pingPongFlag ^= 1

    SetFlag/Wait(M_FIX, cl0BufId)       ← Mmad→Fixpipe 同步
    DoCopyOut                           ← L0C→GM 输出
    SetFlag(FIX_M, cl0BufId)            ← 释放 L0C
    l0cPP++
```

#### LoadAL1（AL1 乒乓预取）

```
AL1 加载:
  Wait(MTE1_MTE2, l1BufId)             ← 等 AL1 buffer 释放
  DataCopy(al1Dst, fmapGm, Dn2NzParams) ← GM→AL1
  SetFlag(MTE2_MTE1, l1BufId)          ← 通知数据就绪
  Load3DSetFMatrixCal(...)             ← FMatrix 属性
```

AIC 主循环中 AL1 乒乓交替：`l1BufId = al1PP & 1`，每轮 `al1PP++`。

源文件：`dwconv2d_kernel.template:390-461, 607-728`

---

## 8. Load3D 参数位域

### 8.1 Config0 (xm)

| bit 位 | 字段 | 含义 |
|--------|------|------|
| 0-15 | currentKL0 (kStep) | K 维传输长度 |
| 16-31 | mExtension (mStep) | M 维传输长度 |
| 32-47 | posK | K 维起点 |
| 48-63 | mStartPt (posM) | M 维起点 |

### 8.2 Config1 (xt)

| bit 位 | 字段 | 含义 |
|--------|------|------|
| 0-5 | strideW | W 步长 [1,63] |
| 6-11 | strideH | H 步长 [1,63] |
| 12-19 | kernelW | 卷积核 W [1,255] |
| 20-27 | kernelH | 卷积核 H [1,255] |
| 28-35 | dilationW | W 膨胀 [1,255] |
| 36-43 | dilationH | H 膨胀 [1,255] |
| 44 | kernelW_high | kernelW+256 标志 |
| 45 | kernelH_high | kernelH+256 标志 |
| 48-63 | channelSize | 源通道数 [1,65535] |

源文件：`dwconv2d_kernel.template:63-77, 596-604, 736-741`

### 8.3 FMatrix 设置

| API | 参数 | 源码位置 |
|-----|------|---------|
| `Load3DSetFMatrixCal(curHiLoadL1, orgWin, padList[4])` | padList={padLeft, padRight, padTopL1_, 255} | `LoadAL1:591-594` |
| `Load3DSetPaddingCal(0)` | offsetx=0 | 同上 |

### 8.4 Load2D 参数

| 方法 | 含义 | 典型取值 |
|------|------|---------|
| `SetMStartPosition` | M 轴起点(单位16) | nOff/GN0 |
| `SetKStartPosition` | K 轴起点(单位K0) | kOff/K0 |
| `SetMStep` | M 方向长度(单位16) | CeilDiv(curN,GN0) |
| `SetKStep` | K 方向长度(单位K0) | CeilDiv(curK,K0) |
| `SetSrcStride` | 源 K 间隔 | CeilDiv(coutOpt,GN0) |
| `SetDstStride` | 目的 K 间隔 | mStep |
| `SetIfTranspose` | 是否转置 | false |

源文件：`dwconv2d_kernel.template:744-762`

---

## 9. MmadParams 字段

| 字段 | 含义 | 取值 |
|------|------|------|
| m | 左矩阵 A 高度 | safeMAlign（AlignUp(curM, GM0)） |
| n | 右矩阵 B 宽度 | curN |
| k | A 宽度=B 高度 | curK |
| cmatrixInitVal | 是否初始化 L0C | `isFirst && !useBias`（首段且无 bias 时清零） |
| cmatrixSource | C 初始值来源 | `isFirst && useBias`（首段且有 bias 时从 BT 加载） |
| unitFlag | Mmad-Fixpipe 并行控制 | false |

源文件：`dwconv2d_kernel.template:765-779`

---

## 10. FixpipeParamsC310 字段

| 字段 | 含义 | 取值 |
|------|------|------|
| nSize | N 方向大小 | curN |
| mSize | M 方向大小 | curM |
| srcStride | 源相邻 Z 偏移 | mAligned |
| dstStride | 目的 ND 每行元素数 | hout×wout |
| quantPre | 量化模式 | F322F16(fp16), F322BF16(bf16), NoQuant(fp32) |
| deqScalar | 反量化 scale | 1065353216 (=1.0f) |
| reluEn | ReLU 使能 | false |

源文件：`dwconv2d_kernel.template:782-802`

---

## 11. 关键状态变量

| 变量 | 含义 | 决定条件 |
|------|------|---------|
| `AL1_PINGPONG` | AL1 双缓冲（编译期） | `L1_PINGPONG & 1` |
| `BL1_PINGPONG` | BL1 双缓冲（编译期） | `L1_PINGPONG & 2` |
| `isVec0_` | 当前 AIV 是否为 vec0 | `rawIdx % taskRation == 0` |
| `vecId_` | AIV 编号（0 或 1） | `rawIdx % taskRation` |
| `enableBias_` | 是否有 bias | `tiling.hasBias != 0` |
| `enlargeActual_` | 当前 group 的 enlarge | 末组用 `enlargeTail`，否则 `enlarge` |
| `cinAligned_` | Cin 对齐到 K0 | `AlignUp(cinOpt, K0_VAL)` |
| `kTotal_` | K 总长 | `cinAligned_ × khkw` |
| `pingPongFlag` | L0A/L0B 乒乓 | 0/1 交替 |
| `l0cPP` | L0C 乒乓 | 0/1 交替 |
| `al1PP` | AL1 乒乓计数 | 每轮 `++`，`& 1` 取 buffer |

源文件：`dwconv2d_kernel.template:107-108, 137-159, 401`

---

## 12. AIV 与 AIC 的 L1 地址共识

AIV 和 AIC 各自在 `Init` 中按相同公式计算 L1 布局，无需交换地址：

```
L1 布局:
  AL1:  [al1BufCount × al1HalfSize]     al1BufCount = AL1_PINGPONG ? 2 : 1
  BL1:  [bl1BufCount × bl1HalfBytes]    bl1BufCount = BL1_PINGPONG ? 2 : 1
  Bias: [AlignUp(afterBl1, 32)]
```

AIV 写 BL1 的偏移 = `bl1BaseBytes_ + vecId_ × bl1HalfBytes_`（乒乓时）。
AIC 读 BL1 的偏移 = `bl1BaseBytes_ + (g % 2) × bl1HalfBytes_`（乒乓时）。

源文件：`dwconv2d_kernel.template:238-248, 326-327, 410-411`
