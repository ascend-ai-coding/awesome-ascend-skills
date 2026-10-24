# Conv2D Cube 模板翻译

> 本文件是对已实现模板 [`dav310/conv2d_kernel.template`](templates/dav310/conv2d_kernel.template) 的逐层翻译。
> 说明 kernel 的循环结构、每个循环用到的 AscendC API、以及流水线 flag 同步控制。
> 对应的接口说明见 [conv2d_cube.md](templates/conv2d_cube.md)；tiling 契约见 [`conv2d_tiling.template`](templates/conv2d_tiling.template)。

---

## 1. 整体架构

```
Conv2DKernel<T>（__cube__）
├── Init()        派生量计算 + L1 布局 + 核映射
└── Process()     多核主循环 → ComputeOutputTile / ComputeWindow
    ├── ComputeOutputTile()   非 splitM/splitN 路径
    ├── ComputeWindow()       splitM/splitN 路径（AL1 全载 + 单缓冲）
    │   └── ComputeSubTile()  window 内子块计算
    └── ComputeStep()         K L0 内循环（L0 双缓冲 toggle）
        └── FlushTile()       L0C→GM 输出（含 split-K flush）
```

源文件：`dav310/conv2d_kernel.template:135-826`

---

## 2. Process() 主循环

```
Process:
  for batch:                              ← batch 维（每核负责连续 batch 段）
    for nOff (step=nBL1):                 ← N 维外循环（BL1 窗口）
      LoadBias(nStart, actualN)            ← bias GM→L1
      [hoistWeight] LoadWeight(...)        ← 权重外提（BL1 全载+单缓冲时）
      for mOff (step=mAL1):               ← M 维循环（AL1 窗口）
        if useWindow_ (splitM || splitN):
          ComputeWindow(...)
        else:
          ComputeOutputTile(...)
    WaitFlag(FIX_M)                       ← 等最后一个 Fixpipe 完成
```

**核映射**：`blockIdx → (batchCore, mCore, nCore)`，batch 优先绑核（`batchCore = block / (nDim*mDim)`）。

源文件：`conv2d_kernel.template:202-260`

---

## 3. ComputeOutputTile（非 window 路径）

splitM 和 splitN 均关闭时走此路径。AL1/BL1 各自按 K 步切分，支持乒乓预取。

```
ComputeOutputTile(batch, mStart, curM, nStart, actualN):
  SetupFmatrix()                          ← FMatrix 属性 + Load3D repeat 参数
  SetFlag(M_MTE1, L0[0/1])                ← L0 初始状态：标记为 M 已释放
  LoadFmapBlock(0, 0, ...)                ← 首块 AL1 加载
  SetFlag(MTE2_MTE1, AL1[0])
  [if !hoistWeight]:
    LoadWeightBlock(0, 0, ...)            ← 首块 BL1 加载
    SetFlag(MTE2_MTE1, BL1[0])

  for s (step=cinStep):                  ← K 步循环（Cin 切分）
    aBlk = s / stepsPerA_; bBlk = s / stepsPerB_
    if aOpen:                             ← AL1 块边界
      Wait(MTE2_MTE1, AL1[aBlk%bufs])
      [if al1PP] 预取下一块 AL1
    if bOpen:                             ← BL1 块边界
      Wait(MTE2_MTE1, BL1[bBlk%bufs])
      [if bl1PP] 预取下一块 BL1

    ComputeStep(...)                      ← K L0 内循环 + Mmad

    if aClose:
      SetFlag(MTE1_MTE2, AL1[aBlk%bufs])
      [if !al1PP] 串行加载下一块
    if bClose:
      SetFlag(MTE1_MTE2, BL1[bBlk%bufs])
      [if !bl1PP] 串行加载下一块

  Wait(MTE1_MTE2, AL1/BL1 all)           ← 等所有 L1 buffer 释放
  Wait(M_MTE1, L0[0/1])
  FlushTile(...)                          ← 输出 L0C→GM
```

源文件：`conv2d_kernel.template:469-578`

---

## 4. ComputeWindow（splitM/splitN 路径）

splitM（`mAL1 > mL0`）或 splitN（`nBL1 > nL0`）时走此路径。AL1 一次性加载全 K，单缓冲常驻，被多个 nL0/mL0 子块复用。

```
ComputeWindow(batch, mStart, curM, nStart, actualN):
  CalcFmapRange()                         ← 算有效输入行 + pad
  LoadFmapBlock(0, 0, ...)               ← AL1 加载一次（全 K）
  SetFlag/Wait(MTE2_MTE1, AL1[0])        ← 等加载完成
  Load3DSetFMatrixCal(...)               ← FMatrix 属性

  for nSub (step=nL0):                   ← SplitN N 子循环
    [if splitN] LoadBiasToBt(nSub, curN)  ← 每个 nL0 子块重载 BT
    for sub (step=mL0):                  ← SplitM M 子循环
      ComputeSubTile(..., nSub, curN)

  SetFlag/Wait(MTE1_MTE2, AL1[0])        ← 释放 AL1
```

**关键**：AL1 加载后全程只读，无 MTE2/M 重叠。收益是 nSub×sub 个子块复用同一 fmap。

源文件：`conv2d_kernel.template:580-617`

### 4.1 ComputeSubTile

```
ComputeSubTile(batch, mStart, curM, ..., nOffset, curN):
  [if !hoistWeight]:
    LoadWeightBlock(0, 0, nWinStart, nWinCount)  ← BL1 加载
    SetFlag(MTE2_MTE1, BL1[0])

  for s (step=cinStep):                  ← K 步循环（BL1 切分，AL1 已全载）
    if bOpen: [if bl1PP] 预取下一块 BL1
    ComputeStep(..., nOffset)            ← K L0 内循环 + Mmad
    if bClose: [if !bl1PP] 串行加载下一块 BL1

  Wait(MTE1_MTE2, BL1 all)
  Wait(M_MTE1, L0[0/1])
  FlushTile(...)
```

源文件：`conv2d_kernel.template:621-699`

---

## 5. ComputeStep（K L0 内循环）

所有路径最终进入此函数。L0A/L0B 双缓冲 toggle，Mmad 与 L1→L0 搬运并行。

```
ComputeStep(cl0, mp, s, aBlk, bBlk, ...):
  for off (step=kL0):                    ← K L0 内循环
    l0Buffer = l0Toggle_ (0/1 交替)
    Wait(M_MTE1, L0[l0Buffer])           ← 等 M 释放此 L0 buffer
    LoadAl0(al0, al1, kBase+off, curK)   ← L1→L0A（Load3D img2col）
    LoadBl0(bl0, bl1, kBaseB+off, curK, nOffset, curN)  ← L1→L0B（Load2D）
    SetFlag(MTE1_M, L0[l0Buffer])
    Wait(MTE1_M, L0[l0Buffer])           ← 等数据就绪

    [if firstOfSegment && fixPending_]: Wait(FIX_M)  ← 等上一段输出完成
    mp.k = curK
    Mmad(cl0, al0, bl0, mp)              ← Cube 矩阵乘
    SetFlag(M_MTE1, L0[l0Buffer])        ← 释放 L0 给下一轮
    l0Toggle_ ^= 1

    [if splitK && accumK >= flushK_]:
      FlushTile(...)                     ← 中途 flush L0C→GM（atomic add）
```

源文件：`conv2d_kernel.template:701-768`

---

## 6. FlushTile / CopyOut（L0C→GM 输出）

```
FlushTile(cl0, batch, mStart, curM, nStart, actualN, firstSegment):
  SetFlag/Wait(M_FIX)                    ← Mmad→Fixpipe 同步
  CopyOut(cl0, ..., accumulate=!firstSegment)
  SetFlag(FIX_M)                         ← Fixpipe 完成后释放 L0C
  fixPending_ = true
```

```
CopyOut(cl0, ..., accumulate):
  FixpipeParamsC310:
    nSize=actualN, mSize=curM
    srcStride=AlignUp(curM,16)
    dstStride=Ho*Wo (M-mode NCHW)
    quantPre=DTypeTag<T>::quant (NoQuant/F322F16/F322BF16)
  [if accumulate] SetAtomicAdd<T>()      ← split-K 分段折叠
  Fixpipe<T, float, CFG_COLUMN_MAJOR>(yGm, cl0, fp)
  [if accumulate] SetAtomicNone()
```

源文件：`conv2d_kernel.template:435-467, 770-779`

---

## 7. 数据搬运 API

### 7.1 GM→L1

| 操作 | API | 参数结构 | 源码位置 |
|------|-----|---------|---------|
| fmap GM→AL1（NCHW→NZ） | `DataCopy(al1, xGm, Dn2NzParams)` | dnNum=1, nValue=rows×Wi, dValue=Cin | `LoadFmapBlock:283-307` |
| weight GM→BL1（1×1） | `DataCopy(bl1, wGm, Nd2NzParams)` | ndNum=1, nValue=Co, dValue=Cin | `LoadWeight:327-337` |
| weight GM→BL1（kh×kw>1） | `DataCopy(bl1, wGm, Dn2NzParams)` | dnNum=Co, nValue=khkw, dValue=Cin | `LoadWeight:338-349` |
| bias GM→L1（带右 pad） | `DataCopyPad(biasL1, biasGm, DataCopyParams, DataCopyPadParams)` | blockLen=actualN×sizeof(T), rightPad=对齐量 | `LoadBias:352-362` |
| bias L1→BT(C2) | `DataCopy(biasBt, biasL1, DataCopyParams)` | blockCount=AlignUp(actualN,64)/32 | `LoadBiasToBt:367-376` |

### 7.2 L1→L0

| 操作 | API | 关键参数 | 源码位置 |
|------|-----|---------|---------|
| fmap L1→L0A（img2col） | `LoadData<A2, A1, T>(al0, al1, Load3DBitModeParam)` | SetConfig0(xm): kStep/mStep/posK/posM; SetConfig1(xt): stride/kernel/dilation/cin | `LoadAl0:406-415` |
| weight L1→L0B（2D） | `LoadData<B2, B1, T>(bl0, bl1, Load2DBitModeParam)` | MStartPosition=nOffset/16, KStartPosition, MStep, KStep, SrcStride=nBL1/16, DstStride | `LoadBl0:420-433` |

### 7.3 Cube 计算

| 操作 | API | 关键参数 | 源码位置 |
|------|-----|---------|---------|
| 矩阵乘 | `Mmad(cl0, al0, bl0, MmadParams)` | m=AlignUp(curM,16), n=AlignUp(curN,16), k=curK, cmatrixInitVal, cmatrixSource, unitFlag | `ComputeStep:739-754` |

### 7.4 L0C→GM 输出

| 操作 | API | 关键参数 | 源码位置 |
|------|-----|---------|---------|
| 输出（量化/格式转换） | `Fixpipe<T, float, CFG_COLUMN_MAJOR>(yGm, cl0, FixpipeParamsC310)` | nSize, mSize, srcStride, dstStride, quantPre, deqScalar | `CopyOut:435-467` |

### 7.5 HF32

| 操作 | API | 说明 | 源码位置 |
|------|-----|------|---------|
| 进入/退出 HF32 | `SetHF32Mode(enable)` | FP32 专用，1=启用 | `Process:207-210` |
| 舍入模式 | `SetHF32TransMode(0)` | 0=NEAREST_EVEN | `Process:207-210` |

---

## 8. Flag 同步与 Event ID

### 8.1 Event ID 分配

```
AL1 pingpong:  MTE2_MTE1 event {0, 1}
BL1 pingpong:  MTE2_MTE1 event {2, 3}
Bias:          MTE2_MTE1 event {4}
L0 pingpong:   M_MTE1 / MTE1_M event {0, 1}
Mmad→Fixpipe:  M_FIX / FIX_M event {2, 3}
```

AL1/BL1 用独立 event pair，使两条 L1 流以不同 K 速率运行互不干扰。

源文件：`conv2d_kernel.template:49-56`

### 8.2 流水线模式

#### ComputeOutputTile（AL1/BL1 乒乓预取）

```
每次 K step:
  aOpen: Wait(MTE2_MTE1, AL1[aBlk%bufs])
         [if al1PP] 预取下一块: Wait(MTE1_MTE2) → LoadFmap → Set(MTE2_MTE1)
  bOpen: Wait(MTE2_MTE1, BL1[bBlk%bufs])
         [if bl1PP] 预取下一块
  ComputeStep: L0 PP toggle
  aClose: Set(MTE1_MTE2, AL1[aBlk%bufs])
          [if !al1PP] 串行加载下一块
  bClose: Set(MTE1_MTE2, BL1[bBlk%bufs])
          [if !bl1PP] 串行加载下一块
```

#### ComputeStep（L0 双缓冲 toggle）

```
每次 kL0 迭代:
  l0Buffer = l0Toggle_ (0/1 交替)
  Wait(M_MTE1, L0[l0Buffer])      ← 等 M 释放
  LoadAl0 + LoadBl0                ← MTE1 加载（与上一轮 Mmad 并行）
  Set(MTE1_M, L0[l0Buffer])
  Wait(MTE1_M, L0[l0Buffer])      ← M 等数据就绪
  [if firstOfSegment && fixPending_]: Wait(FIX_M)
  Mmad                             ← Cube 计算
  Set(M_MTE1, L0[l0Buffer])       ← 释放给下一轮
  l0Toggle_ ^= 1
```

#### ComputeWindow（AL1 单缓冲常驻）

```
init: Set(MTE1_MTE2, AL1[0]) → Wait → LoadFmap(全K) → Set(MTE2_MTE1, AL1[0]) → Wait
      (AL1 加载一次后全程只读)

for nSub:
  LoadBiasToBt(nSub)
  for sub:
    ComputeSubTile:
      [BL1 PP 交叉预取] (与 ComputeOutputTile 的 BL1 逻辑相同)
      ComputeStep: L0 PP toggle

cleanup: Set(MTE1_MTE2, AL1[0]) → Wait
```

**window 路径无 MTE2/M 重叠**：AL1 单缓冲加载完成后才计算。收益是 AL1 在 nSub×sub 个子块间复用。

---

## 9. Load3D 参数位域

### 9.1 Config0 (xm)

| bit 位 | 字段 | 含义 |
|--------|------|------|
| 0-15 | currentKL0 (kStep) | K 维传输长度 |
| 16-31 | mExtension (mStep) | M 维传输长度 |
| 32-47 | posK | K 维起点 |
| 48-63 | mStartPt (posM) | M 维起点 |

### 9.2 Config1 (xt)

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

源文件：`conv2d_kernel.template:32-46, 393-404`

### 9.3 FMatrix 设置

| API | 参数 | 源码位置 |
|-----|------|---------|
| `Load3DSetFMatrixCal(validRows, wi, pads[4])` | pads={padLeft, padRight, localPadTop, 255} | `SetupFmatrix:378-391`, `ComputeWindow:597-600` |
| `Load3DSetPaddingCal(0)` | offsetx=0 | 同上 |
| `SetLoadDataRepeatWithStride(LoadDataRepeatParamWithStride)` | repeatStride=0, repeatTime=1, repeatMode=0, dstStride=mAlign/16 | 同上 |

### 9.4 Load2D 参数

| 方法 | 含义 | 典型取值 |
|------|------|---------|
| `SetMStartPosition` | M 轴起点(单位16) | nOffset/16（SplitN 时选 nL0 子块） |
| `SetKStartPosition` | K 轴起点(单位32B) | localK/k0 |
| `SetMStep` | M 方向长度(单位16) | curN/16 |
| `SetKStep` | K 方向长度(单位32B) | curK/k0 |
| `SetSrcStride` | 源 K 间隔(单位512B) | nBL1/16 |
| `SetDstStride` | 目的 K 间隔(单位512B) | curN/16 |
| `SetIfTranspose` | 是否转置 | false |

源文件：`conv2d_kernel.template:420-433`

---

## 10. MmadParams 字段

| 字段 | 含义 | 取值 |
|------|------|------|
| m | 左矩阵 A 高度 | AlignUp(curM, 16) |
| n | 右矩阵 B 宽度 | AlignUp(curN, 16) |
| k | A 宽度=B 高度 | curK |
| cmatrixInitVal | 是否初始化 L0C | 首段=true(清零), 后续=false(累加) |
| cmatrixSource | C 初始值来源 | 首段=true(从 BT 加载 bias), 后续=false |
| unitFlag | Mmad-Fixpipe 并行控制 | 0=DISABLE |

源文件：`conv2d_kernel.template:473-476, 627-630, 739-754`

---

## 11. FixpipeParamsC310 字段

| 字段 | 含义 | 取值 |
|------|------|------|
| nSize | N 方向大小 | actualN |
| mSize | M 方向大小 | curM |
| srcStride | 源相邻 Z 偏移(单位C0_SIZE) | AlignUp(curM, 16) |
| dstStride | 目的 ND 每行元素数 | Ho×Wo (M-mode NCHW) |
| quantPre | 量化模式 | NoQuant(fp32), F322F16(fp16), F322BF16(bf16) |
| deqScalar | 反量化 scale | 1065353216 (=1.0f, 无 scale 时) |
| reluEn | ReLU 使能 | false |
| unitFlag | 与 Mmad 配合 | 0=DISABLE |

源文件：`conv2d_kernel.template:443-467`

---

## 12. 关键状态变量

| 变量 | 含义 | 决定条件 |
|------|------|---------|
| `al1PP_` | AL1 双缓冲 | `l1PingPong & 1` |
| `bl1PP_` | BL1 双缓冲 | `l1PingPong & 2` |
| `hoistWeight_` | 权重外提 | `!bl1PP_ && numBBlocks_==1`（BL1 全载+单缓冲） |
| `splitM_` | M 维窗口切分 | `mAL1 != mL0` |
| `splitN_` | N 维窗口切分 | `nBL1 != nL0` |
| `useWindow_` | 走 window 路径 | `splitM_ \|\| splitN_` |
| `splitK_` | K 维分段 flush | `kAccum != 0` |
| `flushK_` | K flush 阈值 | `splitK_ ? kAccum : 0xffffffff` |
| `hf32_` | HF32 模式 | `hf32 != 0`（仅 FP32） |
| `l0Toggle_` | L0 buffer 乒乓 | 0/1 交替 |
| `fixPending_` | 有未完成的 Fixpipe | FlushTile 后置 true，Wait(FIX_M) 后置 false |

源文件：`conv2d_kernel.template:811-826`
