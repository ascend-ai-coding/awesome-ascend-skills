# 卷积类性能优化策略集

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

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
