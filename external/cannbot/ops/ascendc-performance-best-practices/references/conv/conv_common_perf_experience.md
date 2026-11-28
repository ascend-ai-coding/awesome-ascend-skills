# 卷积类性能优化策略集

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> §1–§9 主要说明正向 Conv2D / Depthwise 的 Cube 路线；Conv3DDW 的矩阵轴见 [StreamK 指南](conv3ddw_streamk_tuning_guide.md)。形状特征选型、Vector 路线与 DSL 关键片段见 [§10](#10-形状特征选型与-dsl-实现经验)。

## 1. 算子语义分析

卷积类算子的核心计算是：对每个输出位置，将一个局部感受野内的输入与卷积核做加权求和。在 NPU Cube 路径上，这被映射为矩阵乘法：

- **M 维**：输出空间维度（Ho × Wo），即每个输出点
- **N 维**：输出通道维度（Cout）
- **K 维**：输入通道 × 卷积核尺寸（Cin × kH × kW）

不同卷积变体的差异主要体现在 K 维的构成和规模上，进而影响循环轴数量、单次 Mmad 计算量、Scalar 开销占比。

### 1.1 标准 Conv2D

标准 Conv2D 的通用实现需要处理任意 Cin/Cout/kH/kW 组合，循环轴非常多：

```
// 标准 Conv2D 循环轴（8+ 层）
for groupOpt       // 组间循环
  for batch         // batch 循环
    for kAL1Iter    // K 维 L1 切分
      for khAL1Iter // kh 维 L1 切分
        for kwAL1Iter // kw 维 L1 切分
          for mAL1Iter // M 维 L1 切分（ho×wo）
            for mL0Iter  // M 维 L0 切分
              for nL0Iter  // N 维 L0 切分
                for kL0Iter  // K 维 L0 切分
```

标准 Conv2D 的单次 Mmad 的 kL0 较大（典型值 256+），Mmad 执行时间长，Scalar 开销占比低，对 Scalar 优化不敏感。

### 1.2 DepthwiseConv2D 与标准 Conv2D 的差异

DepthwiseConv2D 每个输入通道独立卷积（groups = C），具有以下固有特征：

1. **Weight 全载**：每组 weight 的 `[cout/groups, cin/groups, kh, kw]` 在 `SetupGroup` 阶段一次性加载到 L1，整个 group 内复用——消除了 `kAL1Iter`、`khAL1Iter`、`kwAL1Iter` 三个循环轴
2. **单组计算量小**：`cin/groups = 1`，通过 enlarge 将多组拼接后 `kTotal = cinAligned × kh × kW`（典型值 144），单次 `kL0`（典型值 16-48）远小于标准 Conv2D（典型值 256+），导致单次 Mmad 时间很短
3. **N 维固定**：`coutOpt = 16`（enlarge 后），`nL0 = 16`，无需 N 维切分循环

简化后循环轴：

```
// DepthwiseConv2D 循环轴（4 层）
for group          // 组间循环（AIV/AIC 跨核交替）
  for batch         // batch 循环
    for mAL1Iter    // M 维 L1 切分
      for mL0Iter   // M 维 L0 切分
        for nL0Iter  // N 维 L0（通常只1次）
          for kL0Iter // K 维 L0（kTotal/kL0 次，如 9 次）
```

### 1.3 Scalar 敏感性

由于单次 Mmad 的 kL0 很小（16-48），一次最内轴 k 循环的 Mmad 执行时间可能只有几微秒，非常容易被 Scalar 指令打断。标准 Conv2D 的单次 Mmad 时间长（kL0=256），Scalar 开销占比低；DepthwiseConv2D 必须通过以下手段最小化 Scalar 计算：

- **干掉 TQue**：内循环中用静态构造或 TBuf + 手动 flag 替代 TQue，消除 buffer 状态机轮转和 event ID 动态分配的 Scalar 开销
- **减小全局变量访问**：将 tiling 中频繁访问的字段拷贝到局部变量，减少 GM Load 指令；Load3D 配置寄存器预计算并缓存
- **固定循环轴**：消除不必要的循环层级，减少循环变量维护和地址计算

---

## 2. 架构概览

### 2.1 AIC+AIV 混合核

采用 `__mix__(1, 2)` 混合核模式，AIC 和 AIV 分工：

```
AIV: GM(weight NCHW) → UB(nd) → V(Gather) → UB(nz) → L1(BL1)
AIC: GM(fmap) → L1(AL1) → L0A → Mmad → L0C → GM(output)
     L1(BL1) → L0B → Mmad
     GM(bias) → L1(BiasL1) → C2(BiasBT) → Mmad
```

**为什么需要 AIV**：DepthwiseConv2D 的 weight 是 `[C, 1, kH, kW]`，每个通道只有 1 个输入通道，无法直接满足 Cube 的 K 维对齐要求（K0=16 for fp16）。AIV 负责将多个 group 的 weight 拼接（enlarge）并重排为 NZ 格式。

### 2.2 存储层级与数据流

```
GM (Global Memory)
  │
  │ MTE2 (GM → L1/UB)
  ▼
L1 [AL1_ping | AL1_pong | BL1_ping | BL1_pong | BiasL1]
  │                    │
  │ MTE1 (L1 → L0A/L0B)  │ MTE1 (L1 → C2)
  ▼                    ▼
L0A/L0B [half_0 | half_1]   C2 (BiasBT)
  │
  │ CUBE (L0A × L0B → L0C)
  ▼
L0C [half_0 | half_1]
  │
  │ FIXP (L0C → GM)
  ▼
GM (Output)

UB [ndBuf | nzBuf | indexBuf]  ← AIV 侧
```

### 2.3 事件同步模型

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `MTE1_MTE2` | MTE1 完成 → 允许 MTE2 覆写 | L1 buffer 释放控制 |
| `MTE2_MTE1` | MTE2 完成 → 允许 MTE1 读取 | L1 数据就绪通知 |
| `M_MTE1` | CUBE 完成 → 允许 MTE1 覆写 | L0 buffer 释放控制 |
| `MTE1_M` | MTE1 完成 → 允许 CUBE 计算 | L0 数据就绪通知 |
| `M_FIX` | CUBE 完成 → 允许 FIXP 输出 | L0C 数据就绪 |
| `FIX_M` | FIXP 完成 → 允许下一轮 CUBE | L0C buffer 释放 |
| `V_MTE2` / `MTE2_V` | V 完成 → 允许 MTE2 / 反向 | UB buffer 同步（AIV 侧） |
| `V_MTE3` / `MTE3_V` | V 完成 → 允许 MTE3 / 反向 | UB→L1 搬运同步（AIV 侧） |
| `CrossCoreSet/WaitFlag` | AIV↔AIC 跨核同步 | BL1 生产者-消费者同步 |

---

## 3. Weight NZ 变换设计

### 3.1 问题

Cube 的 L0B 要求 NZ（FractalZ）格式：`[K1, N1, K0, N0]`，其中 K0=16、N0=16（fp16）。DepthwiseConv2D 的 weight 原始格式是 NCHW：`[enlarge, cout/groups, cin/groups, kh×kw]`。

### 3.2 变换流程

```
GM(NCHW) --MTE2--> UB(nd) --V(Gather)--> UB(nz) --MTE3--> L1(BL1)
```

**Step 1: GM → UB（多维 ND DataCopy）**

将 NCHW 格式的 weight 按 3 维 loop 搬入 UB：

```cpp
NdDmaParams<DTYPE, 3> copyParams;
// loop0: cin/groups × khkw (连续拷贝)
// loop1: cout/groups (srcStride = srcKSize, dstStride = kUbSize)
// loop2: enlarge (srcStride = cout/groups × srcKSize, dstStride = cout/groups × kUbSize + srcKSize)
DataCopy<DTYPE, 3, kDefaultMultiCopyConfig>(ubNd, filterGm_[0], copyParams);
```

**Step 2: UB 上 ND → NZ（Reg API Gather）**

先构建 gather 索引（按 `kUbSize` 间隔），再用 `Gather` 按 NZ 的 stride 要求重排：

```cpp
// 构建索引：[0, khkw, 2*khkw, ..., (K0-1)*khkw]，后续 repeat 通过 Adds 累加 nStride
Reg::Gather<DTYPE, DTYPE, IndexT>(gatherReg, srcAddr + srcOffset, indexReg, gatherMaskReg);
Reg::StoreAlign<DTYPE>(dstAddr + dstOffset, gatherReg, vstsMaskReg);
```

**Step 3: UB → L1（NZ 格式 DataCopy）**

```cpp
DataCopyParams nzCp;
nzCp.blockCount = kTotal_ / K0_VAL;
nzCp.blockLen = co1Opt * GN0 * K0_VAL * sizeof(DTYPE) / BLOCK_BYTES;
DataCopy(bl1, ubNz, nzCp);
```

---

## 4. L1/L0 PingPong 双缓冲设计

### 4.1 L1 布局

```
L1 地址空间:
[AL1_ping][AL1_pong][BL1_ping][BL1_pong][BiasL1]

偏移计算:
al1HalfSize_  = hiAL1 * orgWin_ * cinAligned_
bl1HalfBytes_ = nzBufElems_ * sizeof(DTYPE)
```

- AL1 PingPong：MTE2 搬下一轮 fmap 到 AL1[pong] 的同时，MTE1 从 AL1[ping] 搬到 L0A
- BL1 PingPong：AIV 交替写入 BL1[0]/BL1[1]，AIC 交替读取，通过 CrossCoreFlag 同步

### 4.2 L0 PingPong

```cpp
uint16_t pingPongFlag = 0;
// 每次 kIt 迭代后切换
al0 = LocalTensor(A2, pingPongFlag * L0A_HALF, ...);
bl0 = LocalTensor(B2, pingPongFlag * L0B_HALF, ...);
// ... LoadData + Mmad ...
pingPongFlag ^= 1;
```

- L0A/L0B PingPong：MTE1 搬下一轮数据到 L0[next] 的同时，CUBE 处理 L0[cur]
- L0C PingPong：CUBE 写 L0C[next] 的同时，FIXP 输出 L0C[cur]

### 4.3 L0C PingPong

```cpp
uint16_t cl0BufId = l0cPP & 1;
WaitFlag<HardEvent::FIX_M>(cl0BufId);    // 等前一轮 Fixpipe 完成
// ... Mmad 写 L0C ...
SetFlag<HardEvent::M_FIX>(cl0BufId);     // 通知 Fixpipe 可输出
WaitFlag<HardEvent::M_FIX>(cl0BufId);    // 等 Fixpipe 完成
DoCopyOut(cl0, ...);                      // Fixpipe → GM
SetFlag<HardEvent::FIX_M>(cl0BufId);     // 通知 L0C 可复用
l0cPP++;
```

---

## 5. Load3D 配置缓存设计

### 5.1 问题

Load3D 指令需要两个 64 位配置寄存器（xt 和 xm），xt 包含 strideH、strideW、kh、kw、dilationH、dilationW、cinAligned 等参数。这些参数在同一 AL1 加载块内不变，但每次 DoLoadAL0 调用都重新拼装会引入额外 Scalar 计算。

### 5.2 优化

在 `LoadAL1` 中预计算 xt 并缓存为成员变量，`DoLoadAL0` 中直接使用；xm 每次更新（含 mStep + posM）：

```cpp
// LoadAL1 中：计算一次
load3dXt_ = ((strideW & MASK) << offset) | ((kw & MASK) << offset) | ...;

// DoLoadAL0 中：xm 每次更新，xt 复用
load3dXmTmp_ = ((loadMValue & MASK) << G_MSTEP_OFFSET) | ((mL0Pos & MASK) << G_POSM_OFFSET);
param.SetConfig0(curK | (posK << G_POSK_OFFSET) | load3dXmTmp_);
param.SetConfig1(load3dXt_);
```

---

## 6. Bias via CmatrixSource 设计

### 6.1 原理

利用 MMAD 的 bias table 机制，将 bias 直接加载到 L0C（C2 位置），在第一次 K 迭代时由 MMAD 硬件自动加到累加结果上，避免在 Fixpipe 后再做一次单独的 bias add。

### 6.2 数据流

```
GM(bias) --MTE2--> L1(BiasL1) --MTE1--> C2(BiasBT) --M--> L0C(首次K迭代)
```

### 6.3 关键代码

```cpp
// bias GM → L1 (MTE2)
DataCopyPad(biasL1, biasGm_[0], ...);

// bias L1 → C2 (MTE1)
DataCopy(biasBT, biasL1[nIter], cp);

// Mmad 中：
mp.cmatrixInitVal = isFirst && !useBias;  // 不清零
mp.cmatrixSource  = isFirst && useBias;   // 从 bias table 加载
```

---

## 8. 跨核同步设计

### 8.1 机制

AIV 写完 BL1 后通知 AIC，AIC 读完 BL1 后通知 AIV 可复用。使用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`：

```
AIV: 写 BL1 → CrossCoreSetFlag(PIPE_MTE3, syncMte3Mte1)
AIC: CrossCoreWaitFlag(PIPE_MTE1, aicSyncBase) → 读 BL1 → CrossCoreSetFlag(PIPE_MTE1, aicSyncBase+1)
AIV: CrossCoreWaitFlag(PIPE_MTE3, syncMte1Mte3) → 写下一轮 BL1
```

### 8.2 PingPong ID 划分

BL1 使用 PingPong，跨核 flag 的 ID 空间按 `G_CV_SYNC_STRIDE`（16）划分给各 AIV：

```
AIV0: ID 0~15
AIV1: ID 16~31

aicSyncBase = (g % G_NUM_AIV) * G_CV_SYNC_STRIDE
```

同一 AIV 内用相邻 ID 区分 PingPong 的两组同步（aicSyncBase 和 aicSyncBase+1）。

---

## 9. Buffer 管理方式选择

### 9.1 三种方式

| 方式 | 地址分配 | 同步管理 | Scalar 开销 | 适用频率 |
|------|---------|---------|------------|---------|
| 静态构造 + 手动 flag | pos + 偏移量 | 手动 SetFlag/WaitFlag | 最低 | 高频内循环 |
| TPipe + TBuf + 手动 flag | TPipe InitBuffer | 手动 SetFlag/WaitFlag | 低 | 需与 TQue 混用 |
| TPipe + TQue | TPipe InitBuffer | TQue 自动管理 | 高 | 低频外循环 |

详细转换方法参见 [buffer_management_conversion_guide.md](../common/scalar/buffer_management_conversion_guide.md)。

### 9.2 各 buffer 最优选择

| Buffer | TPosition | 推荐方式 | 原因 |
|--------|-----------|---------|------|
| AL1 | A1 | TQue 或静态构造 | 外循环，TQue 开销可忽略 |
| BL1 | B1 | TBuf + CrossCoreFlag | 跨核共享，TQue 的 flag 不跨核 |
| BiasL1 | A1 | TBuf + 手动 flag | 与 AL1 同 TPosition，TBuf 避免地址冲突 |
| BiasBT | C2 | TQue | 外循环，FreeTensor 需延迟到 MMAD 完成后 |
| L0A | A2 | 静态构造 或 TBuf + 手动 flag | kIt 内循环，TQue scalar 开销显著 |
| L0B | B2 | 静态构造 或 TBuf + 手动 flag | 同 L0A |
| L0C | CO1 | 静态构造 或 TBuf + 手动 flag | nL0It 循环，TQue scalar 开销显著 |
| UB nd/nz | VECIN | TQue AllocTensor/FreeTensor + 手动 flag | TQue 的 enQueEvt 不匹配 V→MTE2/MTE3 |

### 9.3 性能影响

L0 层（L0A/L0B/L0C）在 kIt 内循环中使用 TQue 会引入显著 scalar 开销（buffer 状态机轮转、event ID 动态分配/释放），导致 50%-250% 性能退化。改为静态构造或 TBuf + 手动 flag 后性能恢复。

---

## 10. 形状特征选型与 DSL 实现经验

本节按形状特点补充优化策略。关键代码为 `annotated_excerpt`，按参数化方式整理，接入时补齐缓冲、视图、初始化和同步，不作为完整 Kernel 模板。

| 实现语言 | 关键片段文件 | 使用范围 |
|---------|-------------|---------|
| CANNBot-DSL | [conv_cannbotdsl_fragments.py](templates/conv_cannbotdsl_fragments.py) | 窗口/逐点 Cube、分片归约、AIV 写 L1、三乘积与 Vector 分相；接口沿用 0.8.2 / 0.8.3 写法 |
| AscendC | [conv_ascendc_fragments.h](templates/dav310/conv_ascendc_fragments.h) | 通道压紧写回、连续输入面搬运、槽复用、共享行 FMA；DAV_3510 / CANN 9.3 接口 |

### 10.1 选型与关键参数

**优化决策顺序**：先从数学语义减少无效工作，再确定片上复用和任务切分，随后重叠搬运与计算，最后精简热循环。改变算法或数据流时，同时重算物理容量和数值误差；局部指令减少不一定缩短完整调用。

| 优化思想 | 判断与实施 | 主要取舍 |
|---------|-----------|---------|
| 消除无效计算 | 跳过纯 padding 区域；逐点卷积省去展开；通道独立时避免计算跨组乘积 | Cube 补齐、对角外乘积和事后裁剪都计入实际工作量，不能只统计有用 FLOPs |
| 让数据靠近下一次使用 | 先选权重、输入窗口或梯度的复用轴，再排 M/N/K 循环；跨多次 MMAD 的结果留 L0C | 较大 tile 减少重复搬运，也可能挤掉双缓冲或增加尾块浪费 |
| 按有效工作量均衡任务 | 按有效输出深度、窗口大小和归约长度切分；对末轮单独细分 | 平均分配任务编号不等于负载均衡；更多分片增加 partial 和合并开销 |
| 用独立资源重叠流水 | AIV 处理 Cast/Pack，AIC 计算；预取提前到当前消费者之前 | 缓冲深度只提供空间，读写顺序、预取距离和释放时机决定能否重叠 |
| 把变换放到必经通路 | 搬入时完成 Cast/布局转换，AIV 直接准备 Cube L1，输出恢复留在 FP32 寄存器链 | 融合后需重新核对可见性、物理 stride 和寄存器压力 |
| 避免短任务被调度淹没 | 合并小 DMA，批量处理通道面，固定短循环和地址模式 | 大量展开、满核启动或过细 tile 可能增加 Scalar、指令及同步开销 |

**临界路径选择**：搬运受限时优先减少字节数和小 DMA 次数；Cube 受限时优先提高有效矩阵块与 K 复用；Vector/Scalar 受限时优先合并寄存器链、预计算索引和减少循环维护。已经被其他管线隐藏的阶段继续加速，可能不会改变总耗时。融合的收益来自少一次落 GM、少一次转换或少一次等待，不能只根据 launch 数判断。

| 场景特点 | 优化路线 | 关键参数与约束 |
|---------|---------|---------------|
| Conv3DDW：输出块少、归约长 | LOAD3D 空间窗口 + batch/depth/window 分片 + FP32 partial 归约 | 独立选择 Co/Ci tile、窗口行数、Cube K、分片数；平衡并行收益与 partial 搬运、归约成本 |
| Conv3DDW：逐点卷积、输入与梯度采样一致 | 直接计算 dY×Xᵀ，按 K 段预取 | 省去窗口展开；小通道或分组后矩阵过小时，比较 Vector 点积与 Cube 补齐开销 |
| Conv2D：通道归约充分 | LOAD3D + Cube，AIV 准备 L1，权重在单次调用内复用 | 点卷积可转 Matmul；小规模评估 Vector，计入输入转换与权重预排布成本 |
| Conv2D：FP32 输入值域可控 | 高低位拆分 + 三乘积 + FP32 累加 | 依据输入和权重值域选择缩放；误差包含舍入、低低项省略和长归约累积 |
| Conv2D：局部窗口规则、空间块充分 | 评估 Winograd 或按 dilation 相位拆分 | 总成本包含输入/权重变换、Cube、逆变换和输出恢复；相位拆分重新推导 padding 与采样坐标 |
| Conv2D：后处理散写集中、通道步长引起 bank 冲突 | 调整物理通道步长，压紧后连续写回 | padding 按 bank 映射选择，计入 UB 容量和额外压紧成本 |
| Depthwise：通道独立、归约短 | Vector 空间并行、原生 16 位搬运 + FP32 FMA、系数与窗口行复用 | stride/dilation 分相重排；大规模另比较 Toeplitz / 多真实通道 Cube，计入准备成本 |
| Depthwise：stride/dilation 使相邻输出取数不连续 | 连续载入后分离采样相位，寄存器 gather | 预计算移位索引和有效列掩码；避免跨行回绕，计入额外读取量 |

派发依据完整 shape、dtype、布局及卷积属性，tile 与核数独立选择。输入、权重和 bias 每次重新读取，编译缓存仅复用程序。

| 参数 | 影响 | 调整方向 |
|------|------|---------|
| `MT/NT/KT` | Cube 利用率、L0 容量与尾块比例 | 先核对对齐和补齐浪费，再增大基本块；不以单次 MMAD 大小代替完整成本 |
| `WindowRows/Kmax` | 窗口重叠搬运与 L1 占用 | 增大窗口减少重复 halo，保留双缓冲容量与通道复用空间 |
| `Parts` | 归约并行度、partial 写回和最终合并 | 输出块不足时增加；每片计算过短时缩减 |
| `RowPitch/ChannelPad` | 对齐、bank 映射、输入/输出物理大小 | 分开计算逻辑尺寸与物理 stride，包含保护区和紧凑写回缓冲 |
| `RowsUnroll/Planes/Blocks` | 输入行、系数复用与寄存器/核间负载 | 逐项增加；限制寄存器压力，避免小任务使用过多核 |

### 10.2 Conv3DDW：窗口分片与片上累加

空间窗口由 LOAD3D 展开，卷积核深度为独立任务轴；沿 batch、有效输出深度和窗口序列划分归约任务，与仅拆尾轮的 [StreamK](conv3ddw_streamk_tuning_guide.md) 分开选型。固定深度后，矩阵轴为 `M=CoTile`、`N=CiTile×Kh×Kw`、`K=窗口内输出空间位置`，batch/depth 在外层累加。

```text
GM(dY/X) → L1 双缓冲 → L0A/L0B 双缓冲 → L0C(FP32)
                                      → GM(partial) → Vector 合并/重排 → dW
```

**切分与有效区间**：先根据 stride、dilation、padding 求当前卷积核深度对应的有效输出深度，再将有效 batch/depth/window 序列交给 `reduction_range`。只覆盖 padding 的任务写零贡献，不能读未初始化的 partial。输出块已经足够并行时，优先保持单核完整归约，减少额外写回。

**核心循环**：`cube_window_buffers` 在任务循环外分配缓冲；外层预取下一窗口的梯度与输入，当前窗口调用 `spatial_window_mmad`。该函数将梯度装入 L0A、LOAD3D 将输入窗口装入 L0B，并循环累加同一 L0C。逐点且采样一致时使用 `pointwise_cube_reduce`，以“预取下一块、消费上一块、最后排空”的顺序直接归约，不展开空间窗口。

| 片段 | 保留的关键行为 |
|------|---------------|
| `spatial_window_mmad` | K 分块、L0 乒乓、首次初始化、后续窗口持续累加 |
| `pointwise_cube_reduce` | L1/L0 双缓冲和首尾流水控制；区间须按 K 块对齐 |
| `reduce_fp32_parts` | FP32 partial 按固定分片顺序合并，仅在最后恢复输出类型 |

**数据复用**：梯度块可在多个输入通道子块间复用，输入窗口可在多个输出通道子块间复用。扩大行窗减少相邻窗口重复搬运；扩大通道块提高 Cube 利用率，同时消耗更多 L1/L0C。LOAD3D 的 N 轴展开包含通道分形顺序，最终还原到 `dW[Co,Ci,Kd,Kh,Kw]` 时不能直接按紧凑 Ci×Kh×Kw 解释。

**尾轮与不规则通道**：卷积核不同深度上的有效归约长度可能不同，先去掉无贡献深度，再按有效位置数平衡分片；不要对所有深度机械分配相同工作量。通道主体与尾块分别选布局和块大小，避免为了少量尾通道把所有任务补齐到过大的矩阵。归约已经足够短时，小矩阵 Cube 的补齐成本可能超过 Vector 点积的地址与 FMA 成本。

**容量与累加**：令 `b` 为输入元素字节数，`Xcap` 已包含物理对齐和保护区，基本预算为：

```text
L1    = 2 × b × (MT × Kmax + Xcap)
L0A   = 2 × b × MT × KT
L0B   = 2 × b × NT × KT
L0C   = 4 × MT × NT
```

其他常驻权重、bias 和临时区另计。每个分片首个 MMAD 初始化 L0C，整个分片结束后再写 partial；两阶段 Kernel 在同一 stream 上顺序执行。若融合成一个混合核，所有参与者的同步次数和空任务行为必须一致，不能直接删除阶段间同步。

### 10.3 Conv2D：数值拆分与后处理布局

**窗口与权重布局**：通道归约充足时，用 LOAD3D 在 L0 展开窗口，避免完整 im2col 落 GM。权重预排布为 NZ / FRACTAL_Z 后，在空间块间复用；容量不足时按输入/输出通道分块。点卷积直接走矩阵乘，减少无必要的窗口配置。权重变换按本次调用执行，不能把权重地址相同当作内容不变。

**AIV/AIC 流水**：AIV 负责当前特征窗口的 Cast/Pack，直接写入 AIC 使用的 L1；两个 AIV 通过 `publish_aiv_l1_slice` 写不重叠的切片，AIC 在对应生产步读取一次。RingBuffer 的写/读游标保持同一逻辑步数，切片偏移与物理容量使用统一单位。权重搬运、特征转换和 Cube 的缓冲深度分别计算，避免一侧准备不足导致另一侧空等。

**数值拆分**：`split_fp32` 在 VF 内以二进制缩放后做 RN 舍入并保留残差。X/W 分别使用 `sx/sw` 缩放，近似乘积为 `(Xh·Wh + Xh·Wl + Xl·Wh) / (sx·sw)`。`three_product_mmad` 的布局与调用顺序为：

```text
A = [Ah, Al]，B = [Bl, Bh]，均沿 Cube K 轴拼接
第一次：Ah × Bh
第二次：[Ah, Al] × [Bl, Bh] = Ah × Bl + Al × Bh
全部 K 段结束：恢复 sx×sw，再加 FP32 bias
```

两个交叉项合并为一次 Cube 调用，三个乘积累加到同一 FP32 L0C。`restore_scale_and_bias` 保留恢复缩放与 bias 的寄存器链。缩放由输入值域决定，需同时限制 FP16 溢出、残差下溢、低低项省略及长归约误差；只有最后一次恢复输出 dtype，避免中途反复舍入。

**Winograd 与相位拆分**：规则窗口和充足空间块适合评估变换路线。选择输出块尺寸时同时计算输入变换、权重变换、频域 Cube、逆变换和恢复布局的成本；较大变换块会增加临时区和数值误差。膨胀卷积可按采样相位拆成紧凑网格，但需重新推导 stride/padding 与输出交错位置。输入在 UB 预补零可减少内部循环边界判断；二进制缩放改善表示范围，不能独自消除变换中的抵消误差。

**后处理布局**：对散写热点调整物理通道步长，再压紧到 NCHW。`CompactPaddedChannelsFp32` 用移位计算通道号，适用于空间元素数为 2 的幂且无向量尾块的布局；其他布局补齐索引与尾块处理。`CompactAndStoreFp32` 保留“等槽释放→压紧→V/MTE3 同步→DMA→归还槽”的顺序，外层初始化并回收事件。仅对完整覆盖的区域消除边界掩码，额外 padding 与紧凑缓冲均计入 UB。

**布局即调度**：逻辑尺寸相同，物理 stride 仍会改变 DMA 粒度、bank 访问和 Cube 装配次数。分别规划输入、计算和输出布局，让转换只发生在复用边界；权重一次预排布后供多个空间块使用，输出在寄存器/UB 中恢复后连续写回。对只读取一次的数据评估缓存旁路；对即将复用的数据保留缓存，不能统一设置同一缓存策略。

### 10.4 Depthwise：搬运复用与 DSL 表达

输入面连续、逐行 DMA 开销明显时，`LoadContiguousPlanesFp32` 将逐行搬运合成整面搬运，再用非对齐行加载和掩码恢复边界。保护区和尾部预先初始化，DMA 补齐值设为零，实际读取的全部 lane 落在已分配范围内；保留每个输出的 kh/kw FMA 顺序。

**原生搬运与行复用**：FP16/BF16 输入先按原 dtype 搬入 UB，`native_fma` 在寄存器中转为 FP32 并累加，减少 GM 中间结果与 UB 往返。小卷积核的系数跨空间位置复用，按通道处理多行或多个 batch 面。`row_pair_fma` 交错推进两个输出行；AscendC 的 `AccumulateSharedRowFp32` 让一条输入行参与多个输出的累加。展开保持每个输出原有的 kh/kw 顺序，并限制同时存活的行与累加器数。

**stride/dilation 分相**：连续载入后，`load_fp32_phases` 分离偶/奇采样相位，`phase_fma` 在相位寄存器上完成邻域 gather、边界选择与 FMA。移位索引预先约束到寄存器范围，再用有效列掩码清掉跨行回绕；不能只用计算 mask 掩盖非法地址。不同采样相位共享输入搬运，但输出按原空间顺序合并，避免新增整图重排。

**输入双缓冲**：`AcquireInputSlot → LoadContiguousPlanesFp32 → WaitInputReady → VF → ReleaseInputSlot` 管理输入槽。外层预置空闲事件并在退出时回收；输出 DMA 使用独立的完成事件。首轮只预取，末轮只排空；多核任务按通道面或行带切分，输出所有权保持唯一。

**Vector/Cube 取舍**：通道独立、归约很短时，Vector 避免 Cube 补齐与权重矩阵化成本。空间和通道规模足够大时，可评估 Toeplitz 或多个真实通道的 Cube 打包；要合并考虑打包、无效乘积、输出裁剪与中间缓冲。小任务优先增加每个任务的有效工作量，再选择 block 数，而不是固定满核启动。

**复用与指令级并行**：同一系数服务多个输出，同一输入行更新相邻输出，可以减少重复加载并用独立累加链隐藏 FMA 依赖。展开量由同时存活的行、相位寄存器和累加器决定，超过寄存器预算会转为 spill。先保持一次 load、多次寄存器计算、一次 store 的计算链，再增加行展开或 batch 面数。

**DSL 表达与代码生成**：通过 `Buffer/RingBuffer` 表达缓冲存活期，通过 Tensor 视图明确逻辑 shape、物理 stride 和字节偏移，通过 `cbd.vf` 保持 Cast、重排和累加的寄存器数据流。`range_constexpr` 适合短且固定的卷积轴；大规模空间循环保留紧凑循环，避免寄存器溢出和指令膨胀。Python 构图期计算不计为设备 Scalar 开销；检查生成指令中的重复加载、UB 中转和同步，确认修改进入实际分支。跨族表达规则见 [公共实现知识](../common/arch35/cannbotdsl/index.md)。
