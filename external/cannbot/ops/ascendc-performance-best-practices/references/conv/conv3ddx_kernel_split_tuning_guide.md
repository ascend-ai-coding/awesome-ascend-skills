# Conv3DDX kernel split 模板翻译

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 本文件是对模板 [`dav310/conv3ddx_ks_kernel.template`](templates/dav310/conv3ddx_ks_kernel.template)
> 及其 ops-nn 源模板（`conv/conv3d_backprop_input_v2` arch35，tiling 优先级 98）的逐层翻译。
> 说明 kernel 的循环结构、子核轮转、每个循环用到的 AscendC API、AIC/AIV 事件同步控制。
> 接口说明见 [conv3ddx_kernel_split.md](templates/conv3ddx_kernel_split.md)；tiling 契约见
> [`conv3ddx_ks_tiling.template`](templates/conv3ddx_ks_tiling.template)。
> **§11 内联了本模板依赖的全部内积基座方法**（Cube 映射 / 数据通路 / tiling 搜索 / 事件），
> 本文档自包含，无需外部模板文档。

---

## 1. 整体架构

```
Conv3ddxKsKernel<T>（__mix__(1,2)：1 AIC block + 2 AIV block）
├── Init()                 子核描述派生 + L1 布局 + 核映射（AIC/AIV 共入）
└── Process()
    ├── [AIC] CalBasicBlockLoop()       核间轮转（batch/dout/M/N）
    │   └── IterateAllForKernelSplit()  基本块 × 4 子核轮转
    │       └── Iterate()               一个 (Din, mIdx, nIdx) 相位面块
    │           ├── UpdateSubKernel()   splitIndex 切换 + stepK/kIter/pad 重算
    │           ├── LoadToA1()          GM(dedy)→AL1 连续 NZ（无洞）
    │           ├── LoadToB1()/B2()     filter 全载 + Load2d 相位选行
    │           ├── ComputeKIter()      kd 归约 × K L0 循环 + Mmad
    │           └── StorePhaseBlock()   L0C→UB/GM(ws) 三段对齐 fixpipe
    └── [AIV] RearrangeLoop()           凑齐 strideW 相位块后 interleave → GM
```

ops-nn 源文件对照（`op_kernel/arch35/`）：

| 本模板函数 | ops-nn 源 |
|-----------|-----------|
| `UpdateSubKernel` | `conv3d_bp_kernel_split_func.h: IterateForKernelSplit / RecalcStepForKernelSplit / RecalcBaseUseMForKernelSplit / UpdateHoIdxForKernelSplit` |
| `LoadToA1` | `impl/conv_bp_sub_func_load_gm_to_l1a.h: CalcLoadToA1Dn2NzParams4KernelSplit` |
| `LoadToB2`（全载选行） | `impl/conv_bp_input_sub_func_load_l1_to_l0.h: LoadToB2ForKernelSplitB1FullLoad / UpdateLoadToB2ParamsKForKernelSplit` |
| `StorePhaseBlock` | `impl/conv_bp_input_sub_func_kernelsplit_hw.h: LoadL0c2OutForKernelSplitHW` |
| `RearrangeLoop` | 同上：`InterleaveUbOutForKernelSplit / LoadUbToGmForKernelSplit / Rearrange2GmForL0cToWorkSpace` |
| 块级调度 | `conv3d_backprop_input_v2/conv3d_dx_kernel_split_block.h: CalBasicBlockCnt/CalBasicBlockIdx` |

---

## 2. Process() 主循环（AIC 侧）

```
Process:
  if (blockIdx >= usedCoreNum) return        ← 多余核直接退出
  for j in [0, calRound + (blockIdx < tailCnt)):   ← 拖尾核多算一轮
    basicIdx = blockIdx + j * usedCoreNum
    decode(basicIdx) → (batchIdx, dIdx, mIdx, nIdx)   ← D-M-N 轴序解码
    ComputeBlock(batchIdx, dIdx, mIdx, nIdx)
```

**核映射**：`totalCnt = N × Do × mCnt × nCnt`，其中 `mCnt = ceil(splitHiWi / singleCoreM)`
（注意 M 是**相位面**尺寸 `ceil(Hi/sH)×ceil(Wi/sW)`，不是 Hi×Wi）。
输出 GM offset 换算：`curSingleCoreM = (singleCoreM/splitWi)×sH×Wi`
（相位面行数 × stride 还原到原始 dx 行距）。

**轴序选择**（ops-nn `InitBasicBlockLoopDirect`）：

```
dk > 1                    → LOOP_MDN（M 优先，复用 D 滑窗 overlap）
else kh/sH > 1            → LOOP_DMN
else mCnt > nCnt          → LOOP_DMN / LOOP_DNM（沿窄的方向走位）
```

---

## 3. 子核轮转（UpdateSubKernel —— 本模板的灵魂）

每个基本块内 4 个子核按 **W 先 H 后** 步进，一次 `Iterate()` 处理一个子核：

```
Iterate():
  splitWIndex_ = (splitWIndex_+1) % strideW
  rearrangeWIndex_ 同步步进                        ← 用于 UB 相位分格
  wWrapped → splitHIndex_ = (splitHIndex_+1) % strideH
  rearrangeHIndex_ 同步步进
  splitIndex_ = splitHIndex_ * strideW + splitWIndex_
  # ---- 子核切换联动刷新 ----
  RecalcStepForKernelSplit():                      ← 子核尺寸不同 → stepK 动态变
      curStepKa = ceil(channelSize × splitHkWk[splitIndex] / baseK)
      curStepKb 同理；若 stepKb % curStepKb == 0 则沿用 tiling 的 stepKb
  kIter_/tailK_ 重算                               ← K 长度随子核变化
  load3d_.filterW/filterH = splitWk/splitHk[splitIndex]
  RecalcBaseUseMForKernelSplit():                  ← 尾相位 M 收缩
      subKernelM = (Hi/sH) × splitWi
      curMIdx ≥ subKernelM → needCompute=false（整块跳过）
      curMIdx+baseUseM > subKernelM → baseUseM 截断、mmad_.m 对齐 16
  UpdateHoIdxForKernelSplit():                     ← bpPadUp=2 时切换需 hoIdx+1
```

**4 个子核的尺寸表**（stride=2；尺寸公式与"较长组"归属见主文档 §2.3——
`K^(a) = floor((K−1−a)/S)+1`，`baseIter = K mod S` 个组长 ceil(K/S)，其余组长 floor(K/S)）：

| splitIndex | 相位 (sh,sw) | Hk' | Wk' | kernel3×3 例 | kernel4×4 例 |
|-----------|-------------|-----|-----|--------------|--------------|
| 0 | (0,0) | ceil(Kh/2) | ceil(Kw/2) | 2×2 | 2×2 |
| 1 | (0,1) | ceil(Kh/2) | floor(Kw/2) | 2×1 | 2×2 |
| 2 | (1,0) | floor(Kh/2) | ceil(Kw/2) | 1×2 | 2×2 |
| 3 | (1,1) | floor(Kh/2) | floor(Kw/2) | 1×1 | 2×2 |

起始相位由 pad 决定：`splitWStartIndex = padLeft % strideW`（pad≠0 时第一个子核
从非 0 相位开始，kernel 循环起点相应偏移）。

> SplitH（§10A）没有此轮转：相位维度进核间切分，每核只有一个固定子核
> （`splitIndex_=0`，`splitHkList_[0]` 按相位序号二选一注入）。

**子核 backprop pad**（每子核独立，进 Load3D padList；反向 pad 由正向 pad、kernel
逆序和 output padding 推导，正确性见主文档 §2.3 不变量 3）：

```
CalcSubKernelBackpadBegin(K, s, bpPad, idx):
    subSize  = (K + s − idx − 1) / s
    miss     = (K − idx − bpPad) >> 1
    return (subSize − 1) − miss
奇数 Wi 修正：(bpPadLeft∈{0,2} 且 wk∈{2,4}) 或 (bpPadLeft==1 且 wk==3) 时
             Wi 为奇 → subPadRight[0/2] += 1，否则 subPadRight[1/3] += 1
```

**stride 自适应滑窗**（每次执行一个子核的调度动作序列）：

```
1. 选择当前相位对应的、已预加载的紧凑数据块
2. 根据该相位和反向 pad，计算 dY 实际起始位置与有效窗口大小
3. 边界越界部分做 pad 语义处理（不能把越界地址当真实 dY 读取）
4. 执行该子 kernel 的 MMAD
5. 按模式提交紧凑结果（SplitHW → UB/ws 分格暂存；SplitH → 有效行直写）
```

---

## 4. A 矩阵装载（GM→AL1：连续化，无洞）

```
LoadToA1(dedy NCDHW → AL1 NZ):
  dnNum = 1; nValue = curHoSize × realWo          ← 相位面整段连续
  dValue = curCoutSize; srcDValue = Do×Ho×Wo      ← Cout 层间 stride
  dstNzNStride = 1; dstNzMatrixStride = 0          ← ★无 stride 洞（对比前放大的 strideW 洞）
  DataCopy(a1Buf, dedyGm[src], Dn2NzParams)
```

- `curHoSize = CalFmapH(baseM)`：相位面 M 块对应的 dedy 行数（CalFmapH 几何见 §11.2），
  `hiCal = ceil(baseM/splitWi)`、`hoCal = (hiCal−1) + subHk`，上限 `Ho`；
- `realWo` 修正：反向 pad=0 且子核 subPad<0 时跳过脏数据（woOffset 调整、realWo−−）；
  负 pad 场景按子核 leftCrop/rightCrop 裁剪，W 不连续时退化逐行搬
  （`dnNum=curHoSize, srcDnMatrixStride=Wo`）；
- 尾块防越界：最后一个 batch/depth 块按 dn2nz 的 howo 总偏移收缩 curHoSize。

**与前放大路径的本质差异**：前放大用 `dstNzNStride=strideW` 在 L1 写带空洞的数据
（插零靠地址空洞实现，空洞占搬运带宽），kernel split 的 kernel 拆分已把插零消掉，
A1 恢复稠密连续 —— 这是 MTE2 ×stride² 的来源。

---

## 5. B 矩阵装载（全载 + 相位选行）

### 5.1 GM→BL1（一次全载）

```
LoadToB1(filter NCDHW → BL1 NZ):
  dnNum = Cout1G; nValue = Kh×Kw（原核，不拆）; dValue = baseN(Cin 块)
  BL1 布局 = [Cout 分形][KhKw 行][Cin0×16]
```

### 5.2 BL1→L0B（Load2d mStartPosition 相位选行 —— 不重搬）

```
UpdateLoadToB2ParamsKForKernelSplit(kPos):
  kRepeat = baseK / splitHkWkC0[splitIndex]              ← 该子核下 L0B 装几个 Cout 分形
  mStartPosition = ((splitHk−1)·strideH·Kw + (splitWk−1)·strideW
                    + (splitHIndex·Kw + splitWIndex)) · blockBaseN
                       ↑ 子核首行在原核 KhKw 展开中的行号（相位偏移）
  kStep = splitWk[splitIndex]                            ← 子核宽（1 或 2）
  startAddrOffset = hkWk × kPos × kRepeat                ← 跳过 kPos 个 Cout
  srcStride = −blockBaseN（逆序）                         ← 反卷积 filter 翻转
  LoadData(l0b, bl1[startAddrOffset], Load2DParams)
```

**原理**：B1 是原核完整 Kh×Kw×Cout 的 NZ 驻留；子核 (sh,sw) 的行集合是原核展开中
一个固定 stride 网格（`W^(a,b)[u,v] = W[a+u·Sh, b+v·Sw]` 的行采样）。Load2d 的
`MStartPosition/KStep/SrcStride` 组合直接在驻留缓冲里"跳格取样"，一条指令完成子核
B 矩阵组装 —— 子核切换的 B 侧成本从"重搬 GM"降为"改 3 个标量"。

非全载路径：B 的 GM offset 按相位平移 `kSCoreIdx × kernelSplitStrideB`
（NCDHW filter 下 `kernelSplitStrideB = Kw`），L1→L0B 按 HkWk 分形逐组逆序
（`mStartPosition = (splitHkWkEnd−1−kStart)×blockBaseN` 倒着取行，跨 baseK 的
HkWk 组拆多条指令）。

---

## 6. ComputeKIter（kd 归约 × K L0 循环 + Mmad）

```
ComputeKIter(mIdx, nIdx, dIdx):                    ← L0 乒乓事件同 §11.6
  for kd in [0, Kd):                                ← Dk 外层归约
    dTmp = DinIdx + padFront − kd·dilD
    valid = 0 ≤ dTmp < Do·strideD 且 dTmp % strideD == 0   ← D 轴稀疏跳过
    curDout = dTmp / strideD
    for kIdx in [0, kIter_):
      LoadToA1(kIdx)        [Ping-Pong 时预取下一块]
      LoadToB2(kIdx)
      Wait(M_MTE1, l0PingPong) → Load3D(L1→L0A) + Load2d(L1→L0B) → Set(MTE1_M)
      Wait(MTE1_M)
      Mmad(l0c, l0a, l0b, {m=align16(baseUseM), n=baseN, k=baseK,
                          cmatrixInitVal=(kd==0 && kIdx==0)})
      Set(M_MTE1); l0PingPong ^= 1
```

- B1 全载时的 DeQue/FreeTensor 由 `isLoadB1_/isFreeB1_` 全载标志控制：
  第一个基本块 DeQue 一次，最后一个 dk/相位块才 Free（dk>1 时跨 kd 驻留）；
- A1 全载（kSCoutFullLoad 场景）同理：`isA1FullLoadFlag_`，负 W pad 时各子核
  裁剪区间不同、切换后需重载（`CheckFreeA1ForKernelSplit`）。

## 7. StorePhaseBlock（L0C→UB/GM 三段对齐 fixpipe）

M 按相位面切（baseM 对齐 splitWi 的整数倍），但 UB/fixpipe 指令要求 16 元素
（fp16 32B）对齐 —— 把每行输出切 **head/mid/tail** 三段独立对齐：

```
CalcCutInWIndex():                     ← 依 baseUseM/splitWi 计算首尾块宽度
  headWi = (splitWi 对齐尾) 左侧残余；midHi = 完整行数；tailWi = 右侧残余
StorePhaseBlock():
  wsDstOffset = hwSize × baseUseN × rearrangeWIndex     ← ★相位块按 rearrangeWIndex 分格
  fixpipe.srcStride = AlignUp16(baseUseM)               ← L0C 内 M 行距
  head: mSize=realHeadWi, 单段                         ← srcOffset += realHeadWi<<4（c0=16 固定）
  mid : dnNum=midHi, mSize=srcWi, src/dstStride=行距     ← 循环整行
  tail: mSize=realTailWi, 单段
```

- `hwSize = alignUp(headWi) + midHi×alignUp(srcWi) + alignUp(tailWi)`：相位块在
  UB 中的行容量（含对齐 padding）；
- **workspace 路径**（`useWorkspace=1`）：UB 放不下 4 块（`hwSize×baseN×2B × 4 > UB`）
  或 `Wi%16≠0` 时，fixpipe 先到 GM workspace（per-core 分片
  `GetAicBlockIdx()×(hwAlign<<1)`），AIV 再从 GM 读回。

## 8. RearrangeLoop（AIV 侧 interleave 重排）

### 8.1 交叉重排的数学形式

SplitHW 的每个 Cube 结果在 H/W 上都是"按相位采样"的紧凑块。直接写 GM 会产生
隔点/隔行访问，带宽利用率低；先交叉再写：

```
src0 = [x0, x2, x4, x6, ...]          dst[2t]   = src0[t]
src1 = [x1, x3, x5, x7, ...]     →    dst[2t+1] = src1[t]
dst  = [x0, x1, x2, x3, ...]
```

W 方向在 UB 内 Interleave 交叉；H 方向由 GM 写地址的 `rearrangeHIndex×Wi` 行偏移
还原。先 W 后 H 即把 4 个相位块还原为连续 dx tile；只改布局不改数值。

### 8.2 4 块 workspace 的 Ping/Pong 流水（workspace 路径）

一次交叉重排需要**两轮配对**的 Cube 结果（W 相位 0/1）。为避免 AIC 等待 AIV，
DDR 中准备 4 块、每块大小等于一个 Cube 输出 tile 的 workspace，两批轮转：

| 配对批次 | Cube 结果 | 写入 workspace | AIV 操作 |
|---------|-----------|---------------|---------|
| 批次 0 | C0(Ping)、C1(Pong) | WS0、WS1 | 两块都就绪后，VEC Ping 重排并写 GM |
| 批次 1 | C2(Ping)、C3(Pong) | WS2、WS3 | 两块都就绪后，VEC Pong 重排并写 GM |
| 批次 2 | 后续 Cube 结果 | 复用 WS0/WS1 | 等 VEC Ping 完成通知后再复用 |

```
AIC:  C0(Ping) → WS0 ─┐
      C1(Pong) → WS1 ─┼─> AIV: VEC Ping 重排 → GM
      C2(Ping) → WS2 ─┐
      C3(Pong) → WS3 ─┼─> AIV: VEC Pong 重排 → GM
      C4/C5       → WS0/WS1（前一批完成后复用）
```

同步协议必须保证"两个输入 workspace 均写完才启动 AIV"与"对应 AIV 重排结束才允许
AIC 覆盖复用"，由 CrossCoreSet/Wait 双向事件完成（§9）。

### 8.3 AIV 单次重排流程

```
[AIV] 每凑齐 strideW 个相位块（rearrangeWIndex == last）触发一次：
  CrossCoreWait(FLAG_MTE2_VEC_ID / PIPE_FIX)      ← 等 AIC 的 fixpipe/写 GM(ws) 完成
  if useWorkspace: LoadWorkspaceToUb()             ← GM(ws)→UB（MTE2，两块相位）
  Interleave(src0=UB[0], src1=UB[dataLen], dst=UB[2·dataLen]):
      vfLen = VRegSize/sizeof(T)
      repeatTimes = ceil(dataLen/vfLen)
      Reg::Interleave(dst0, dst1, src0, src1)      ← VF 寄存器内建，逐段交叉
      kernel=1×1 特判：第二块为全零（Duplicate 后再 Interleave）
  SetFlag(V_MTE3) → Wait
  DataCopyUbToGm():                                ← UB→GM(dx NCDHW) 隔行直写
      mid 段: blockCount=midHi, blockLen=Wi, dstStride=Wi   ← 原始行距
      head/tail 段单独搬；奇数 Wi 时 rearrangeW==1 的首块末元素跳过
  CrossCoreSet(FLAG_FIXP_ID / PIPE_MTE3)           ← 反向通知 AIC
```

workspace 路径下 baseUseN 过大时按 `UB/(dataLen×4B)` 分 loopIter 轮，
轮间 `MTE3_MTE2` 事件串行。

## 9. 事件同步（AIC↔AIV 握手拓扑）

```
AIC（PIPE_FIX 侧）                         AIV（PIPE_V / PIPE_MTE2 侧）
─────────────────────────────────────────────────────────────────
rearrangeW==0:   CrossCoreWait(FLAG_FIXP_ID)      ← 等上一行相位重排完（UB/ws 可复用）
rearrangeW==last: CrossCoreSet(FLAG_MTE2_VEC_ID)  → CrossCoreWait(FLAG_MTE2_VEC_ID)
                                                Rearrange + interleave + UB→GM
                                                CrossCoreSet(FLAG_FIXP_ID) ──→ 回到左上
```

- **每 strideW 个相位块握手一次**（不是每块）——AIV 的 interleave 粒度是整行；
- workspace 方案 AIV 首指令是 MTE2（GM→UB），等待类型用 `PIPE_FIX→PIPE_MTE2`；
  UB 直通方案用 `PIPE_FIX→PIPE_V`；
- 核内 MTE1_M / M_MTE1（L0 乒乓）事件时序见 §11.6；
- `Init()` 先空下一个 `CrossCoreSetFlag`，保证 `End()` 的 `CrossCoreWaitFlag` 成对
  （kernel split 必备防御，否则未计算核直接 Wait 会挂死）；
- launch 侧需 `SetScheduleMode(1)`（batch mode 独占核），多流场景防死锁。

## 10. Load3D 参数位域（kernel split 增量）

沿用 Conv2D Cube 模板的 Config0/Config1 位域（见
[conv2d_cube_tuning_guide.md §9](conv2d_cube_tuning_guide.md)），kernel split 的差异：

| 字段 | kernel split 取值 | 说明 |
|------|-------------------|------|
| kernelW/kernelH | `splitWk/splitHk[splitIndex]` | 子核尺寸，每次切换刷新 |
| strideW/strideH | **1** | 相位拆分后等效 stride=1（★与前放大路径最大的不同：无需插零） |
| dilationW/H | 1 | 模板准入要求 dilation=1 |
| padList | `[subPadLeft, subPadRight, subPadUp, 255]` | 子核独立 backprop pad（负值钳 0） |
| channelSize | Cout 分形数（`channelSize_`） | L1 单次驻留的 Cout 通道数 |
| mStartPt | 相位面块内 W 向起点 | `curMIdx % splitWi` |
| kStartPt | `l0aKIdx × baseK` | K 方向子块起点 |


---

## 10A. SplitH 变体翻译（H 单拆，kernelSplitMode=2）

SplitH 只按 H 相位拆分、相位维度进核间切分、输出直写无 AIV。结构与选型差异见
[conv3ddx_kernel_split.md §10](templates/conv3ddx_kernel_split.md)；本节翻译其
**执行流与 ops-nn 源对照**。

### 10A.1 Process() 主循环差异

```
Process（SplitH）:
  for j in [0, calRound + (blockIdx < tailCnt)):
    basicIdx = blockIdx + j × usedCoreNum
    decode(basicIdx) → (batchIdx, dIdx, mIdx, nIdx, kSCoreIdx)   ← 多解一个 H 相位维
    kernelIdx = (kSCoreIdx + kernelStartIdx) % strideH            ← 相位恢复（kernelStartIdx=padUp%sH）
    ReCalcMInfo(kSTailIdx, kernelIdx)                             ← 尾相位 M 收缩（见下）
    ComputeBlockSplitH(kernelIdx, ...)
```

ops-nn 源：`conv3d_dx_kernel_split_block.h: CalBasicBlockCoreForSplitH`（SplitHW 走
`CalBasicBlockCore`；两者在 `CalBasicBlock` 按 kernelSplitMode 分派）。

**尾相位 M 收缩**（Hi%sH≠0 时尾相位面行数少 1，与 SplitHW 的核内
RecalcBaseUseMForKernelSplit 对应、但发生在核间）：

```
kSTailIdx = Hi % strideH                      // ≠0 时存在尾相位
kSTailM   = (Hi/strideH) × Wi                 // 尾相位 M 总长
kernelIdx ≥ kSTailIdx 的核：
    kSTailMCnt = ceil(kSTailM/singleCoreM)    // mCnt 换小
    mCoreIdx ≥ kSTailMCnt → continue 跳过      // 越界基本块直接不算
    尾块 mCoreUse = kSTailM − (kSTailMCnt−1)×singleCoreM
```

### 10A.2 子核参数注入（每核一个固定子核）

SplitHW 的子核轮转（§3）在 SplitH 中退化为**核启动时一次注入**：

```
SetIterateParams(mCoreUse, splitBaseHk, baseHkIter, splitTailHk, kernelIdx):
  curSplitHk  = (kSCoreIdx < baseHkIter) ? splitBaseHk : splitTailHk   // 较长/较短相位组二选一
  curBackPropPadUp = CalcSubKernelBackPadUp(Kh, sH, bpPadUp, kSCoreIdx)
  SetSingleShapeParams(curSplitHk, curBackPropPadUp)    // 写 splitHkList_[0]/subPadUp
  CalcBlockKSOffset(kernelIdx)                          // B 偏移：kSCoreIdx × kernelSplitStrideB_
```

ops-nn 源：`conv3d_dx_kernel_split_block.h: SetIterateParams / CalcSubKernelBackPadUp /
InitKernelSplitStrideB`（NDHWC filter 下 `kernelSplitStrideB_ = Wk×Cin`）；
`conv3d_bp_kernel_split_func.h: SetSingleShapeParams / InitParamsForKernelSplitH`
（`splitIndex_=0`、`splitWkList_[0]=Kw`、`splitWi_=Wi`——单子核视角）。

### 10A.3 数据通路差异（对照 §4/§5/§7）

| 通路 | SplitHW | SplitH |
|------|---------|--------|
| A GM→L1 | dnNum=1 整段连续（`dstNzNStride=1`） | **strideH 强制 1**（H 向无洞）；W 向保持 `dstNzNStride=strideW` 留洞/逐行（ops-nn `CalcLoadToA1Dn2NzParams` 的 `TPL_SPLIT_KERNEL_H` 分支） |
| CalFmapH 行宽 | splitWi=ceil(Wi/sW) | **Wi**（W 未压缩）；上限仍 Ho |
| B L1→L0B | 全载 mStartPosition 相位选行 / 非全载相位平移 | `splitIndex_=0` 单子核：常规逆序 Load2d，B 偏移按 kSCoreIdx 平移 |
| L0C 输出 | 三段对齐 → UB/ws 分格 → AIV Interleave → GM | 三段对齐 **fixpipe 直写 GM**，`dstDnMatrixStride = Wi×(strideH−1) + Wi`（每写一行相位面跳 strideH−1 行原始 dx） |

**H 方向交错还原 = fixpipe 目的地址隔行 stride**，不需要 Interleave/AIV——
W 未拆故 x 方向天然连续。NCDHW 输出：

```
skipDstOffset = Wi × (strideH − 1)
dstOffset = cinOff + diOff + mSize + (mSize/Wi)×skipDstOffset
head: 单段；mid: dnNum=midHi, dstDnMatrixStride=skipDstOffset+Wi；tail: 单段
```

ops-nn 源：`impl/conv_bp_input_sub_func_store_l0c.h: LoadL0c2GmDnForKernelSplitH`（NCDHW，
COLUMN_MAJOR）/ `LoadL0c2GmNdForKernelSplitH`（NDHWC，ROW_MAJOR）；分派在
`store_l0c_dispatch.h`。

### 10A.4 事件同步差异

SplitH 的 `LoadL0c2Gm` 走 `kernelSplitMode != TPL_SPLIT_KERNEL_HW` 分支——
无 Rearrange2Gm、无 CrossCoreCSeitVForKS 四原语握手，AIC 单核完成全部工作；
核内事件（L0 乒乓、FIX_M）与 §11.6 一致。因此 SplitH 落地可用 `__cube__` 单核型
（无需 `__mix__(1,2)`），除非叠加 vecTrans 等需要 AIV 的特性。

---

## 11. 内积基座方法（本模板自包含依赖）

Kernel split 的 kernel/tiling 骨架 = "**子核替代原核、stride 归一为 1、M 换相位面**"
之后的内积（stride=1 反卷积）实现。本节完整内联其全部依赖方法，本文档自包含。

### 11.1 Cube 映射与三段式数据通路

反卷积输入梯度 `dx[i] = Σ_k dedy[o] × w[k]`（`i + pad = o·stride + k`）的 Cube 映射
（一次 MMAD = baseM×baseN×baseK）：

| 矩阵 | GM 来源 | M/N/K 含义 |
|------|---------|-----------|
| A（左） | dedy | M = dx 空间展平（Hi×Wi，kernel split 下为相位面 splitHiWi）；K = Cout×HkWk×K0（单层 kd 内） |
| B（右） | filter | N = Cin；K 与 A 相同 |
| C | dx | M × N |

**Dk 是 MMAD 之外的独立归约循环**：`kValue = Hk×Wk×Cout1G×k0`（kernel_d 不算在
L0 的 K 值里），每个 `kd` 单独做一轮完整 K 归约，结果累加进同一块 L0C
（`cmatrixInitVal` 只在首个有效 kd 置真）。D 轴有效性判定：
`dTmp = id + padFront − kd·dilD`，要求 `0 ≤ dTmp < Do·strideD` 且
`dTmp % strideD == 0`，否则跳过该 kd。

分形常量：`m0 = n0 = 16`，`k0 = 16`（fp16/bf16）/ 8（fp32）。

**三段式数据通路**（"搬运即变换"，无额外 vector kernel）：

```
段1 GM→L1：dedy NCDHW → DataCopy Dn2Nz 进 AL1（NZ 分形布局）
            filter NCDHW → DataCopy Dn2Nz 进 BL1
段2 L1→L0A：Load3D 一条指令完成"滑窗展开 + NZ→Zz 分形重排"（im2col 语义；
            pad/dilation 在此处理；stride 的插零由段1 的地址空洞实现）
段3 L1→L0B：Load2d 逆序装载（filter 空间翻转 = correlation→convolution；
            逆序 = 翻转，转置 = OI→IO 的 K/N 角色换位）
```

kernel split 对三段的影响：段1 的 A 侧空洞消失（连续化，§4）；段2 的 kernel/pad
换子核值（§3/§10）；段3 的起点换相位（§5.2）。

### 11.2 CalFmapH 几何（M 块 ↔ dedy 行数换算）

给定 L1 内 M 方向能容纳的元素数 baseM，推算需加载的 dedy 行数——A1 容量估算的
通用公式：

```
hiCal = ceil(baseM / Wi)        （M 展平后占的 dx 行数；不整除补 2 行防跨界）
hoCal = (hiCal − 1) + khDilation（反卷积滑窗反向：dx 每行对应 dedy 一行 + 核膨胀头）
curHoSize = min(hoCal, hoExpand) （hoExpand=(ho−1)·stride+1 为 dedy 实际行数上界，
                                  stride>1 时起截断作用）
```

kernel split 变体：行宽用 `splitWi`（相位面）、`khDilation` 换子核尺寸
（`subHk`，dilation=1）、上限是 `ho`（不再需要 hoExpand——拆分后等效 stride=1）。

### 11.3 基本块（baseM/baseN/baseK）经验值与对齐规则

| 场景 | baseM | baseN | baseK | 理由 |
|------|-------|-------|-------|------|
| kernel=1（DkHkWk==1） | 256 | 256 | 大 | 纯带宽型，用计算访存比最高的 256×256 方块 |
| kernel>1 | 512 | 128 | 小 | 右矩阵（filter）格式转换/逆序是瓶颈，**M 大 N 小**，用更长 MMAD 掩盖 LoadToB2 |

对齐规则：
- `baseN ≤ nValue`（Cin 对齐 16）、`baseM ≤ align(mValue,16)`、`baseK ≤ kValue`；
- baseM/baseN 还受 L0A/L0B 容量上界钳位：`baseM×baseK×B字节×a1Db ≤ 64KB` 等；
- **baseK 对齐 HkWk 分形**：从 L0 容量上限向下找第一个满足 `baseK % lenHkWkC0 == 0`
  的值——保证 L1→L0 搬运按分形对齐，避免跨分形撕裂；kernel split 下加强为
  **同时整除最大与最小子核分形**（`baseK % maxSubHkWkC0 == 0 && baseK % minSubHkWkC0 == 0`）。

### 11.4 stepK 搜索与 L1 容量模型

`stepKa/stepKb` 是 L1 一次驻留的 baseK 基本块数（乒乓下的流水深度）。两策略竞争取优：

- **EqualL1（对半分）**：L1 平分给 A1/B1，各自算出能容纳的 Cout 分形数
  `cout1A1 = (L1/2/al1Pbuffer)/a1Unit`、`cout1B1` 同理，再对齐二者（保证 K 循环同拍），
  换算 `stepKa = ceil(cout1A1 × lenHkWkC0 / baseK)`；负载均衡细节：cout1×Pbuffer ≥
  Cout1G 时按 `ceil(Cout1G/Pbuffer)` 压缩，防止单 buffer 装载过慢；
- **Ladder（阶梯降级）**：从最大 stepK（受 Cout1G×lenHkWkC0 与 L1 上限）同步递减
  stepKa=stepKb 直到容量合法。

容量模型（合法性判据）：

```
aL1Size = CalFmapH(baseM) × Wo × coutNum × A字节 × al1Pbuffer   （coutNum = stepKa·baseK/kernelHW）
bL1Size = bl1Pbuffer × B字节 × baseN × copyLine × lenHkWkC0      （copyLine ≈ ceil(stepKb·baseK/lenHkWkC0)）
aL1Size + bL1Size < L1_SIZE(512KB)
且 (stepKa·baseK) 与 lenHkWkC0 整除对齐
```

**LegalProtection 兜底链**（容量超限时的降级序）：
① 缩基本块（先降 baseK 以 k0 步长重搜 stepK；再 baseK=k0 时交替收缩 baseM/baseN）
→ ② 逐级关 L1 乒乓（BL1→AL1 组合枚举）→ ③ 仍失败报错。

### 11.5 核间切分与负载均衡

核间默认**不切 K**（`singleCoreCout=CoutG` 全载、`singleCoreDin=1`），只在
M/N/Din（batch/group）三维切：

```
totalCnt = batch × group × CeilDiv(Di, 1) × CeilDiv(hwI, singleCoreM) × CeilDiv(nValue, singleCoreCin)
coreNum  = min(totalCnt, 物理核数)      calRound = totalCnt/coreNum，tailCnt 均摊到前 tailCnt 个核
```

负载均衡微调：
- 分不满核（totalCnt ≤ coreNum）或 nCnt 是核数因子时：baseN 重排为
  `ceilAlign(nValue/nCnt, 16)`——避免尾块 N 只有 16 造成的长尾；
- 分不满核时 `singleCoreM = ceilAlign(hwI/mCnt, Wi)`——M 方向按**整行对齐**切块
  （L1 的 fmap 装载以行为单位；kernel split 下按 splitWi 相位行对齐）；
- 轴序遍历方向：dk>1 用 M-D-N（优先复用 D 滑窗 overlap），否则沿窄的方向走位。

### 11.6 L0 乒乓事件时序（MTE1_M / M_MTE1）

L0A/L0B 双缓冲与 MMAD 的握手（核内事件，与所有内积路径一致）：

```
每次 kL0 迭代:
  l0Buf = l0Toggle_ (0/1 交替)
  Wait(M_MTE1, l0Buf)          ← 等 M（上一轮 MMAD）释放此 L0 buffer
  LoadAl0 + LoadBl0            ← MTE1 加载（与上一轮 MMAD 并行）
  Set(MTE1_M, l0Buf)
  Wait(MTE1_M, l0Buf)          ← M 等数据就绪
  Mmad                          ← Cube 计算
  Set(M_MTE1, l0Buf)           ← 释放给下一轮
  l0Toggle_ ^= 1
```

L1 层再加 `MTE2_MTE1 / MTE1_MTE2`（GM→L1 与 L1→L0 的乒乓预取），形成 MTE2（GM→L1）、
MTE1（L1→L0）、M（MMAD）三级流水；A/B 两流用**独立 event pair** 使两条 L1 流以不同
K 速率运行互不干扰。

---

## 12. 设备侧执行流程（分阶段）

**初始化**：
1. 解析 tiling，初始化子核参数、各相位 pad 列表及循环方向；
2. 按 totalCnt 确定 calRound 调度范围；
3. 设置 GM 地址与跨核通信资源；
4. 判定 A1/B1 全载复用条件；
5. 初始化 L1 队列（inQueL1A/inQueL1B）与 L0A/L0B/L0C Ping/Pong 缓冲。

**外层与空间块调度**：

```
while calRound < totalCnt:
    计算当前块索引与调度参数
    SplitHW：M 维已含 H/W 拆分信息，由 kernel-split 迭代器切换相位
    设置 singleShape、startIdx；计算当前块 GM 读写偏移
    遍历 M/N/D：每次空间迭代由 IterateForKernelSplit 切换子核/相位，
    同步更新 stepK、baseUseM、baseUseN、curHoSize
```

**K 方向搬运与计算**（§6 伪代码）；**结果输出与清理**：K 循环结束后按模式分支
（SplitHW → workspace/UB 提交 + AIV 重排；SplitH → 有效行连续写回），所有 calRound
完成后释放全载 B1 buffer、FreeAllEvent 释放队列与事件。

## 13. 实现与验证检查表

| 检查项 | 应满足的条件 |
|--------|-------------|
| 子 kernel 覆盖 | 所有相位组元素数之和 = Kh×Kw；任何元素只出现一次 |
| 子 kernel 维度 | 与 `K^(a) = floor((K−1−a)/S)+1` 公式一致；K<S 时的空组不得进入 MMAD |
| pad/边界 | 每相位使用对应反向 pad、起点与有效尺寸；output padding 只影响合法输出边界，不得越界读取 |
| 相位调度 | SplitH 切 H、SplitHW 切 H/W；M、stepK、curHoSize 在切换后同步更新 |
| 数据复用 | 全载 A1/B1 只在安全生命周期内复用，释放在最后一次消费者之后 |
| L0 Ping/Pong | 当前 MMAD、下一轮 L1→L0 搬运、L0C 输出不得共用未同步 buffer |
| SplitH 输出 | 仅写有效行，W 方向地址连续；隔行 stride=Wi×(strideH−1) 正确还原 H 相位（§10A.3） |
| SplitHW 输出 | 交叉重排后 GM 中 H/W 坐标连续且唯一；AIV 读取的两个 workspace 均已完成写入 |
| 数值对比 | 与前放大路径在边界、不同 pad、不同 output padding、各相位尾块上逐元素一致 |

## 14. 关键状态变量

| 变量 | 含义 | 决定条件 |
|------|------|---------|
| `splitIndex_` | 当前子核编号 0..3 | `splitHIndex_×strideW + splitWIndex_` |
| `rearrangeWIndex_/rearrangeHIndex_` | 相位块在输出格中的 W/H 位 | 与 splitIndex 同步步进 |
| `splitWi_` | 相位面行宽 | `ceil(Wi/strideW)` |
| `channelSize_` | L1 单次 Cout 通道数 | `min(stepKa×baseK/maxSubHkWk, alignUp(CoutG,C0))` |
| `b1FullLoad` | B1 全载驻留 | L1 容量判定（主文档 §8.5） |
| `isLoadB1_/isFreeB1_` | 全载 B1 的 DeQue/Free 时机 | 全载标志 + dk/相位边界 |
| `needComputeFlag_` | 当前相位块是否有效 | 尾相位 M 越界 / Dk 全无效 |
| `kSCoreIdx_`（仅 SplitH） | 当前核的 H 相位序号 | basicIdx / batchDepthMNCnt 解码（§10A.1） |
| `useWorkspace` | L0C→GM(ws)→UB 中转 | `4×hwSize×baseN×B > UB` 或 `Wi%16≠0` |

---

## 15. 模板代码 ↔ ops-nn 源码索引

ops-nn 源仓：`conv/conv3d_backprop_input_v2/`（arch35 / DAV_3510 路径）。
下表建立每个模板部件与 ops-nn 原实现的对照，便于回溯原始实现与验证行为差异。

### 15.1 文件级对照

| 模板文件（本目录） | 落地 | ops-nn 对应源 |
|-------------------|------|--------------|
| `templates/conv3ddx_ks_tiling.template` | `op_kernel/conv3ddx_ks_tiling.h` | `op_host/op_tiling/conv3d_backprop_input_v2_tiling_data_arch35.h`（TilingData 字段）+ `op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_tiling.h`（KernelSplitPara） |
| `templates/conv3ddx_ks_tiling_impl.template` | `op_host/conv3ddx_ks_tiling.cpp`（g++） | `op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_tiling.cpp`（优先级 98）+ 其继承的 `conv3d_backprop_input_v2_inner_product_tiling.{h,cpp}`（优先级 101，基座方法） |
| `templates/dav310/conv3ddx_ks_kernel.template` | `op_kernel/conv3ddx_ks_kernel.h`（bisheng） | `op_kernel/arch35/conv3d_backprop_input_v2/conv3d_dx_kernel_split_block.h`（块级调度）+ `convolution_3d_backprop/conv3d_bp_kernel_split_func.h`（子核轮转）+ `convolution_3d_backprop/conv3d_bp_func_arch35.h`（计算主循环）+ `convolution_3d_backprop/impl/*`（数据通路子函数） |
| `templates/dav310/conv3ddx_ks_kernel_entry.template` | `op_kernel/conv3ddx_ks_kernel.cpp`（bisheng） | `op_kernel/conv3d_backprop_input_v2.cpp`（`__global__` 模板入口，tiling key 五元组分派） |
| `templates/dav310/conv3ddx_ks_launch.template` | `op_kernel/conv3ddx_ks_launch.h` | —（direct-launch 胶水，ops-nn 走注册式无对应物） |

### 15.2 tiling 契约（conv3ddx_ks_tiling.template）

| 模板部件 | ops-nn 源（文件 : 符号） | 备注 |
|---------|------------------------|------|
| `Conv3ddxKsCase` | `conv3d_backprop_input_v2_tiling_data_arch35.h : Conv3DBackpropInputArch35TilingData` | POD 契约同构；模板裁剪 bias/scale/quant/SplitK/C04 字段 |
| `ConvKsSplitMode{HW=1,H=2}` | `kernel_split_tiling.cpp : KERNEL_SPLIT_HW/KERNEL_SPLIT_H` | 数值一致 |
| `kKsSubKernelNum=4` / `kKsMaxPhases=16` | `conv3d_bp_kernel_split_func.h : SUB_KERNEL_NUM` / kh≤16 约束 | H 相位数=strideH |
| `ValidateCase`（公共段） | `kernel_split_tiling.cpp : CheckDtypeCompatibility/CheckBasicConstraints/CheckShapeValue` | dtype/groups/dilation/pad 对称/shape 下限 |
| `ValidateCase`（HW 段） | `TryKernelSplitHW/CheckKernelSizeStrideMatch/CheckShapeConditions/IsBaseShapeFitKernelSplitHW` | stride 2×2、kernel 1~4、wi≥12、cin≥16、MTE2 负向经验 |
| `ValidateCase`（H 段） | `TryKernelSplitH/CheckKernelSplitHEnable/IsBaseShapeFitKernelSplitH` | cin/cout≥16、wi≥32、kh∈[sH,16]、L1 最小容量（cout1=1） |
| `KsCalcBackpropPads` | `conv3d_backprop_input_v2_inner_product_tiling.cpp : SetBackpropPadInfo`（bpPad 推导） | 反向 pad 由正向 pad/shape 推导 |
| `KsSubBackpadBegin/End` | `conv3d_bp_kernel_split_func.h : CalcSubKernelBackpadBegin/CalcSubKernelBackpadEnd` | s=2 特化；奇数 Wi 修正含在内 |
| `KsSubBackpadUpH` / `KsFillSubPadListH` | `conv3d_dx_kernel_split_block.h : CalcSubKernelBackPadUp` | H 模式相位 pad（missPointNum 公式） |
| `KsFillSubPadList` | `conv3d_bp_kernel_split_func.h : CalcSubKernelPadListForKernelSplit` | 4 子核 pad 表 |
| `ToCaseView`（H 派生段） | `kernel_split_tiling.cpp : SetParamForKernelSplit(true)` + `CalBasicBlockCoreForSplitH`（kernelStartIdx） | splitHkBase/Tail/baseHkIter/kernelStartIdx/kSTail* |

### 15.3 host tiling 实现（conv3ddx_ks_tiling_impl.template）

| 模板函数/段 | ops-nn 源（符号） | 备注 |
|------------|------------------|------|
| `Conv3ddxKsAdmitMode` | `CheckKernelSplitEnable`（HW→H 兜底顺序） | 返回 0/1/2 三态 |
| 基本块初值（512/128、256/256） | `InitBaseMNK`（kernel_split 重写版） | 经验值同源 |
| baseK 搜索（HW：倍数契约） | `CalculateOptimalBaseK` + `AdjustBaseKForKernelSplit` | `baseK % lenHkWkC0 == 0` |
| baseK 搜索（H：整除契约） | `ShrinkBaseKForKernelSplit` | `cout1×subHkWkC0 % baseK == 0`——**与 HW 方向相反** |
| B1 全载判定 | `IsBaseShapeFitKernelSplitHW`（aBFullLoad/cout1B1） | H 模式不启用（B 走 GM 行取样重组） |
| stepK 对半分搜索 | `EqualL1MatchStepMNK/Core`（继承自 inner_product） | EqualL1 简化版；Ladder 未抽 |
| L0 容量钳位 | `AdjustBaseMNCommon`（继承） | 简化版 |
| 核间切分（HW） | `SetSingleCoreInfo`（else 分支）+ `SetSingleCoreInfoCore` | kSCnt=1、M 对齐 splitWi |
| 核间切分（H） | `SetSingleCoreInfo`（isKernelSplitOnlyH 分支） | kSCnt=strideH、M 对齐 Wi |
| 尾相位字段 | `CalBasicBlockCoreForSplitH : kSTailIdx/kSTailM` + `ReCalcMInfo : kSTailMCnt` | |
| workspace 判定 | `UpdateWorkSpaceSize` | 仅 HW；4 块 UB 预算 |
| `CalFmapH` | `CalFmapHForKernelSplit`（H：行宽 curWi=Wi） | 几何公式同源 |

### 15.4 kernel（conv3ddx_ks_kernel.template）

| 模板方法 | ops-nn 源（文件 : 符号） |
|---------|------------------------|
| `Conv3ddxKsKernel`（类骨架） | `conv3d_dx_kernel_split_block.h : Conv3dDxKsBlock`（继承 Conv3dDxOswBlock/Conv3dDxBase） |
| `Init` / 核映射 | `Conv3dDxKsBlock : Init/InitTilingData/CalBasicBlockCnt` |
| `InitSubKernelTables`（HW） | `conv3d_bp_kernel_split_func.h : InitParamsForKernelSplitHW`（splitHk/Wk 十字表 + splitWi + 起始相位） |
| `InitSubKernelTables`（H） | 同文件 `: InitParamsForKernelSplitH`（splitIndex_=0、splitWi_=Wi） |
| `InjectSubKernelForCore`（H） | `conv3d_dx_kernel_split_block.h : SetIterateParams`（curSplitHk 二选一 + CalcSubKernelBackPadUp） |
| `DecodeBasic`（HW） | `Conv3dDxOswBlock : CalBasicBlockIdx`（D-M-N 轴序） |
| `DecodeBasic`（H 相位维） | 同上 + `CalBasicBlockCoreForSplitH`（kSCoreIdx 解码 + kernelIdx 相位恢复） |
| `AdvanceSubKernel` | `conv3d_bp_kernel_split_func.h : IterateForKernelSplit`（W 先 H 后步进 + rearrange 索引） |
| `ComputePhase` | `conv3d_bp_func_arch35.h : Iterate→Compute→ComputeForNoTilingHWk→ComputeForKIter`（kd 归约 + K L0 循环 + Mmad + L0 乒乓） |
| `LoadToA1`（HW 连续） | `impl/conv_bp_sub_func_load_gm_to_l1a.h : CalcLoadToA1Dn2NzParams4KernelSplit` |
| `LoadToA1`（H：W 留洞） | 同文件 `: CalcLoadToA1Dn2NzParams`（TPL_SPLIT_KERNEL_H 分支：strideH 强制 1） |
| `LoadToB1Full`（HW） | 同文件 `: LoadToB1Dn2NzTransposeForKernelSplitHW`（原核整驻留） |
| `LoadToB1Block`（H 行取样） | 同文件 `: LoadToB1Dn2NzTransposeForKernelSplitH`（splitHk 行循环、srcGmOffset += wk×strideH） |
| `LoadAl0`（Load3D 子核 im2col） | `impl/conv_bp_input_sub_func_load_l1_to_l0.h : InitLoadToA2Params/UpdateLoadToA2ParamsM/K`（filterW/H=子核、stride=1、subPad padList） |
| `LoadBl0`（HW 全载相位选行） | 同文件 `: UpdateLoadToB2ParamsKForKernelSplit + LoadToB2ForKernelSplitB1FullLoad`（mStartPosition 相位偏移） |
| `LoadBl0`（非全载逆序） | 同文件 `: LoadToB2ReverseOnlyCommon`（HkWk 分形逐组逆序） |
| `StorePhaseBlock`（HW 三段 fixpipe） | `impl/conv_bp_input_sub_func_kernelsplit_hw.h : LoadL0c2OutForKernelSplitHW→LoadL0c2Ub/GMForKernelSplitInner` |
| `StorePhaseBlockDirectGm`（H 直写） | `impl/conv_bp_input_sub_func_store_l0c.h : LoadL0c2GmDnForKernelSplitH`（NCDHU）/ `LoadL0c2GmNdForKernelSplitH`（NDHWC）——skipDstOffset=Wi×(sH−1) 隔行 |
| `RearrangeTile` / `InterleavePhase` | `conv_bp_input_sub_func_kernelsplit_hw.h : Rearrange2Gm→InterleaveUbOutForKernelSplit` + `impl/conv_bp_input_sub_func_vector_intrinsics.h : InterleaveKernelSplitNormal/1x1`（Reg::Interleave VF 内建） |
| `LoadWorkspaceToUb` | 同文件 `: LoadWorkSpaceDataToUb`（GM(ws)→UB + MTE2_V 事件） |
| `DataCopyUbToGm` | 同文件 `: LoadUbToGmForKernelSplit→DataCopyUbToGmForKernelSplit`（head/mid/tail + 奇数 Wi 首块修正） |
| `WriteZeroPhase`（1×1 特判） | `InterleaveKernelSplit1x1`（第二相位清零语义的 GM 直写等价） |
| 事件常量（kSyncAivToAic 等） | `conv3d_bp_kernel_split_func.h : CrossCoreCSeitVForKS/CWaitVForKS/VSeitCForKS/VWaitCForKS`（FLAG_MTE2_VEC_ID/FLAG_FIXP_ID + PIPE 区分） |

### 15.5 已知简化（模板 vs ops-nn）

| 简化点 | ops-nn 完整行为 | 模板处理 |
|--------|----------------|---------|
| L1 乒乓预取 | stepKa/stepKb 内 l0aKIdx/l0bKIdx 轮转 + `*PreLoad`、TQue DeQue/FreeTensor | 骨架按每 kIdx 串行装载展示通路（§6 注释） |
| B1 全载生命周期 | `isLoadB1_/isFreeB1_` 跨基本块/dk 驻留，`FreeB1Tensor` 统一释放 | 每 tile 重载（简化） |
| Ladder stepK 搜索 | `LadderMatchStepMNK` 与 EqualL1 竞争取优 | 仅 EqualL1 |
| LegalProtection | 缩基本块 → 关乒乓四级降级链 | 乒乓容量验证 + 关 A1/B1 乒乓简化 |
| kSCoutFullLoad | A/B 均全载的独立模板（优先级 97） | 未抽（§4.4 扩展落点） |
| bias/scale/quant/SplitK/C04 | 完整支持 | 裁剪（无 bias 前提） |
| vecTrans（AIV 前置转置 filter） | NCDHW filter 大 kernel 时前置 NZ 化 | 未抽 |
| NDHWC filter/dx | SplitH 完整支持（LoadL0c2GmNdForKernelSplitH 等） | 契约限 NCDHW，NDHWC 为 §6.4 落点 |
| tiling key 编译期分派 | (loadB2, kernelSplit, group, loadB1) 四元组模板特化 | 单一 runtime splitMode 字段 |
