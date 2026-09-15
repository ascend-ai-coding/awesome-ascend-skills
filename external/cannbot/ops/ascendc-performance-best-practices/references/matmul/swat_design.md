# SWAT 调度模式技术文档

## 1. 背景

在昇腾 AI 处理器上执行大规模矩阵乘（尤其是量化场景如 MXFP4/MXFP8）时，Cube 侧的算力利用率很大程度上取决于**调度器如何把 `M × N` 的逻辑分块分配给有限数量的 AIC 核**。常见痛点有三类：

1. **非方阵分块 + B 侧流式加载**：当 A、B 都不常驻 L1（non-full-load），B 片段要反复从 GM/L2 流入。如果相邻核按朴素的"先行后列"顺序推进 N，B 就会频繁失效，导致 MTE2 带宽成为瓶颈。
2. **尾部薄片**：`M % baseM`、`N % baseN` 经常产生一个明显小于 `baseM/baseN` 的边缘 tile。它与相邻满块一起落到同一个 serpentine 行窗口里，窗口的关键路径被满块拖到 `baseM`，尾薄块的"省算"收益被完全吃掉。
3. **最后一轮空闲核**：`totalCnt = mCnt × nCnt` 往往不是 `blockNum`（AIC 核数）的整数倍，最后一轮里只有前 `endBlockIdx+1` 个核有任务，其余核空转，拖高整体时延。

**SWAT** 就是针对 non-full-load 路径提出的组合调度策略，名字可以读成「**S**erpentine **W**indow + **A**daptive **T**ail」。它用三个相互独立、又互相约束边界的机制一次性解决上述三个问题：

| 机制 | 定位 | 对应问题 |
|---|---|---|
| A. Serpentine 行窗口遍历 | 全局遍历顺序 | B 侧流式带宽复用 |
| B. 尾部基块合并（尺寸重排） | 稳态区与尾区之间的桥梁 | M/N 边缘薄片 |
| C. 尾块二维再切分 | 最后一轮局部处理 | 末轮空闲核 |

三者都沿用一个共同的常量 `WINDOW_LEN = 4`（serpentine 行窗口高度），从而保持 host tiling、device scheduler、device kernel 三层的切分边界对齐。

SWAT 下有两种子模式：

- `NO_FULL_LOAD_MODE`：A、B 都流式，本文分析的主要版本。
- `A_FULL_LOAD_MODE`：A 常驻 L1，只流式 B，机制 C 被简化。

模式通过一个编译期标签选择：

```cpp
template <uint64_t FULL_LOAD_MODE_>
struct QuantMatmulMxSwatScheduler {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

constexpr uint64_t NO_FULL_LOAD_MODE = 0UL;
constexpr uint64_t A_FULL_LOAD_MODE = 1UL;
```

## 2. 调度器整体结构

SWAT 调度器为每个 AIC 核独立维护一份"我下一轮做哪一个 tile"的迷你状态机，状态由 host tiling 决定的 `Params` 初始化：

```cpp
struct Params {
    int64_t baseM;              // 稳态基础 tile M
    int64_t baseN;              // 稳态基础 tile N
    int64_t mTailTile;          // 机制 C：M 方向二维子切分倍率
    int64_t nTailTile;          // 机制 C：N 方向二维子切分倍率
    int64_t mBaseTailSplitCnt;  // 机制 B：M 尾部合并区 tile 数
    int64_t nBaseTailSplitCnt;  // 机制 B：N 尾部合并区 tile 数
    int64_t mTailMain;          // 机制 B：M 合并区"主尺寸"
    int64_t nTailMain;          // 机制 B：N 合并区"主尺寸"
};
```

关键运行时字段（节选）：

```cpp
int64_t m_, n_, k_;
int64_t baseM_, baseN_;
int64_t mCnt_, nCnt_, totalCnt_;           // 各方向 tile 数
int64_t mBaseNormCnt_, nBaseNormCnt_;      // 机制 B: 保持 baseM/baseN 的 tile 数
int64_t mBaseTailMain_, nBaseTailMain_;    // 机制 B: 合并区主尺寸
int64_t mBaseTailLast_, nBaseTailLast_;    // 机制 B: 合并区末块尺寸
int64_t mCoreNum_, mTailCoreNum_;          // 机制 A: 行窗口高度 / 末窗口高度
int64_t mainRow_;                          // 机制 A: 稳态行窗口层数
int64_t blockIdx_, blockNum_;              // AIC 自身索引 / 总数
int64_t startBlockIdx_, endBlockIdx_;      // 活跃核范围
int64_t roundIdx_, round_;                 // 我要跑几轮 / 当前第几轮
int64_t mTailTile_, nTailTile_, totalTailTile_; // 机制 C

static constexpr int64_t WINDOW_LEN = 4;
```

kernel 层只看到一个循环接口：

```cpp
while (bs.GetTileIdx(blockIdx)) {
    int64_t mPos = Get<MNK_M>(blockIdx);
    int64_t nPos = Get<MNK_N>(blockIdx);
    BlockShape singleShape = bs.GetBlockShape(blockIdx);
    if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
        return;  // 机制 C 产生的空子块
    }
    // 以 (mPos, nPos) 与 singleShape 切 GM 张量，送给 BlockMmad
    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
}
```

调度器约定把 `BlockCoord` 的 M/N 槽用作**GM 起点坐标**，把 K/B 槽当作**逻辑 tile 索引**寄存处，这样 `GetBlockShape` 后面还能重建出形状。

## 3. 机制 A：Serpentine 行窗口遍历

### 3.1 目的

在 non-full-load 路径下 B 是流式的，如果所有核"从左扫到右"再"从左扫到右"，相邻两次扫描复用不到 B 已加载的片段。Serpentine 把每 `WINDOW_LEN` 行打成一个行窗口，窗口内多核并行走 M，**沿 N 推进时奇数行窗口反向**，让两次推进的 N 边界继续落在同一个 B 片段，从而最大化 L1/L2 命中。

### 3.2 构造期建模

```cpp
m_ = shape.m;  n_ = shape.n;  k_ = shape.k;
baseM_ = params.baseM;  baseN_ = params.baseN;
mCnt_ = CeilDiv(m_, baseM_);
nCnt_ = CeilDiv(n_, baseN_);
totalCnt_ = mCnt_ * nCnt_;

mCoreNum_   = Min(WINDOW_LEN, mCnt_);      // 行窗口高度，封顶 4
mainRow_    = mCnt_ / mCoreNum_ - 1;       // 稳态区窗口层数
mTailCoreNum_ = mCnt_ - mCoreNum_ * mainRow_;  // 末窗口实际高度

endBlockIdx_ = (totalCnt_ - 1) % blockNum_;
round_       = CeilDiv(totalCnt_, blockNum_);
if (blockIdx_ > endBlockIdx_) {
    round_ -= 1;
}
```

- `mCoreNum_` = 行窗口内并行处理 M 的核数。
- `mainRow_` = 完整窗口层数，最后一层交给 `mTailCoreNum_` 处理（不满 4 也不退化）。
- `round_` = 当前核需要跑几轮。

### 3.3 运行期映射

`GetTileIdx` 把线性 `tileIdx` 还原到 `(rowIdx, mTileIdx, nTileIdx)`，最后对奇数行反转 N：

```cpp
int64_t rowIdx = tileIdx / nCnt_ / mCoreNum_;
int64_t mTileIdx = 0, nTileIdx = 0;
if (rowIdx < mainRow_) {
    mTileIdx = rowIdx * mCoreNum_ + tileIdx % mCoreNum_;
    nTileIdx = (tileIdx / mCoreNum_) % nCnt_;
} else {
    rowIdx = mainRow_;
    int64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
    mTileIdx = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
    nTileIdx = (tailIdx / mTailCoreNum_) % nCnt_;
}
if (rowIdx & 1) {
    nTileIdx = nCnt_ - 1 - nTileIdx;   // 奇数行反向
}
```

关键不变量：**行窗口的边界在整个调度中是神圣的**——机制 B 的合并区、机制 C 的切分区都必须沿 `WINDOW_LEN` 对齐，不能打碎窗口。

## 4. 机制 B：尾部基块合并（尺寸重排，不改数量）

### 4.1 目标

如果 `m % baseM` 产生一个明显小于 `baseM` 的薄片，行窗口的关键路径被它周围的满块拖到 `baseM`。机制 B 把**最后若干个 tile（含薄片）的 M 尺寸重新等分**，让它们变得均匀。**tile 数量保持不变**，变的只是尺寸分布。

### 4.2 Host 侧搜索

```cpp
if (mTailSize > 0UL && isInnerAxisAlign) {
    uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16,
                                       runInfo_.mBlockCnt);
    uint64_t windowSize    = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
    uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
    uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
    uint64_t perfRes       = (mainWindowNum + 1UL) * runInfo_.baseM;
    uint64_t mergeWindowNum = 1UL;

    for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
         mergeLen += windowSize, ++mergeWindowNum) {
        uint64_t newTailMain =
            Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL),
                  BASIC_BLOCK_SIZE_16);
        uint64_t curPerf =
            (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
        if (curPerf <= perfRes) {
            perfRes = curPerf;
            runInfo_.mTailMain       = newTailMain;
            runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
        }
    }
}
```

拆解：

- **`windowSize / mainWindowNum / tailWindowSize`**：让尾区至少覆盖一整个 serpentine 窗口，范围 `[windowSize, 2·windowSize-1]`。
- **搜索起点 `mergeLen = tailWindowSize - 1, mergeWindowNum = 1`**：先把整个尾窗口里"`mergeLen` 个 `baseM` + 1 个 `mTailSize`"合并成 `mergeLen+1` 个 `newTailMain`。
- **步长 `mergeLen += windowSize, ++mergeWindowNum`**：每次再吞掉一个 serpentine 窗口进入合并区，保证边界对齐。
- **新尺寸公式**：

  ```
  newTailMain = Align16( ceil((mergeLen * baseM + mTailSize) / (mergeLen + 1)) )
              ≈ baseM - (baseM - mTailSize) / (mergeLen + 1)
  ```
  `mergeLen` 越大，`newTailMain` 越接近 `baseM`。
- **上界 `baseTailCntMax = (baseM - mTailSize) / 16`**：再往上 `newTailMain` 被 16 对齐拉到 `baseM`，合并不再有收益。
- **成本函数**：`curPerf = (未合并窗口数) * baseM + mergeWindowNum * newTailMain`，即所有 serpentine 窗口的关键路径之和。用 `<=` 接受新解，平局偏好更深合并（tile 更均匀）。

### 4.3 Device 端重建形状

```cpp
mBaseNormCnt_ = mCnt_ - params.mBaseTailSplitCnt;
int64_t mMergeSize = m_ - mBaseNormCnt_ * baseM_;
mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mMergeSize : params.mTailMain;
mBaseTailLast_ = mMergeSize - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;
// N 方向同理
```

最终 M 方向得到：

- `mBaseNormCnt_` 个 `baseM_`
- `mBaseTailSplitCnt - 1` 个 `mBaseTailMain_`
- 1 个 `mBaseTailLast_`
- 总和 = `m_`；总数 = `mCnt_`（**保持不变**）

形状查询：

```cpp
int64_t singleCoreM = baseM_;
if (mTileIdx >= mBaseNormCnt_) {
    singleCoreM = mTileIdx < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_;
}
```

GM 偏移修正（合并区 tile 不再是 `baseM` 的整数倍）：

```cpp
int64_t mPos = mTileIdx * baseM_ + mSplitAddrOffset;
if (mTileIdx > mBaseNormCnt_) {
    mPos -= (mTileIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
}
```

### 4.4 一个具体例子

`baseM=256, m=2404, mCnt=10`：

- `mTailSize = 100, mainWindowNum = 1, tailWindowSize = 6, baseTailCntMax = 9`。
- 迭代 1：`mergeLen=5 → newTailMain=Align16(230)=240`，`curPerf=1·256+1·240=496 < 512`，接受 `mBaseTailSplitCnt=6, mTailMain=240`。
- 迭代 2：`mergeLen=9 → newTailMain=256`（饱和），`curPerf=512`，拒绝。

最终 M 排布（仍是 **10** 个 tile）：

```
[256][256][256][256][240][240][240][240][240][180]
 └─ mBaseNormCnt_=4 ─┘└──── mBaseTailSplitCnt=6 ────┘
```

薄片从 100 涨到 180，窗口关键路径从 256 降到 240。

## 5. 机制 C：尾块二维再切分（末轮空闲核救援）

### 5.1 目标

最后一轮只有 `endBlockIdx_ + 1` 个核有活，其他核空转。机制 C 把最后一轮里**所有原 tile**（共 `tailOriCnt = endBlockIdx_ + 1` 个）各自在 M/N 方向切成 `mTailTile_ × nTailTile_` 个子块，让总活跃核数扩展到 `tailOriCnt × totalTailTile_`。

注意：这里"尾"指的是"最后一整轮里的 tile"，不只是最末那一个。

### 5.2 Host 侧搜索

优先切更长的那一边，总子块数受总核数和 cube 单元约束：

```cpp
uint64_t mTile = 1UL, nTile = 1UL;
uint64_t preSplit = 1UL, secSplit = 1UL;
uint64_t& preSplitValid = mTailSize >= nTailSize ? mTile : nTile;
uint64_t& secSplitValid = mTailSize >= nTailSize ? nTile : mTile;
uint64_t tileMax    = aicNum / tailBlockCnt;
uint64_t mTileMax   = std::min(tileMax, CeilDiv(baseM, CUBE_BLOCK));
uint64_t nTileMax   = std::min(tileMax, CeilDiv(baseN, CUBE_BLOCK));
while ((CalUsedCoreNum(runInfo_, preSplit + 1, secSplit) <= aicNum && preSplit < preSplitMax) ||
       (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1) <= aicNum && secSplit < secSplitMax)) {
    if (可扩 pre) preSplitValid = ++preSplit;
    if (可扩 sec) secSplitValid = ++secSplit;
}
```

### 5.3 Device 端激活新核

```cpp
if ((endBlockIdx_ + 1) * params.mTailTile * params.nTailTile <= AscendC::GetBlockNum()) {
    mTailTile_     = params.mTailTile;
    nTailTile_     = params.nTailTile;
    totalTailTile_ = params.mTailTile * params.nTailTile;

    uint64_t tailOriCnt = AscendC::Std::min(totalCnt_, endBlockIdx_ + 1);
    int64_t  newEndBlockIdx = endBlockIdx_ + tailOriCnt * (totalTailTile_ - 1);

    if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
        round_ += 1;  // 本来不该有活的核，现在多跑一轮
    }
    if (blockIdx_ > newEndBlockIdx) {
        mTailTile_ = 1; nTailTile_ = 1; totalTailTile_ = 1;  // 仍闲，自降避免走子块分支
    }
    endBlockIdx_ = newEndBlockIdx;
}
```

`newEndBlockIdx` 里**为什么是 `tailOriCnt * (totalTailTile_ - 1)` 而不是 `tailOriCnt * totalTailTile_`**：

- 切分前已经有 `tailOriCnt` 个核活跃；
- 切分后每个原 tile 新增 `totalTailTile_ - 1` 个子块，需要 `tailOriCnt × (totalTailTile_ - 1)` 个**新**核；
- `newEndBlockIdx` 是最后一个被激活的核索引，自然等于 `endBlockIdx_` + 新激活数；
- 总活跃数最终 = `tailOriCnt + tailOriCnt · (totalTailTile_ - 1) = tailOriCnt · totalTailTile_`，但索引末尾不能重复计那已激活的 `tailOriCnt` 个。

### 5.4 核 → (原 tile, 子块) 映射

```cpp
// GetTileIdx 最后一轮
int64_t newBlockIdx = (curRoundIdx == round_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
int64_t tileIdx     = newBlockIdx + curRoundIdx * blockNum_;

// GetTileIdx / GetBlockShape 共用的子块坐标
int64_t singleCoreMSplit = CeilDiv(singleCoreM, mTailTile_);
int64_t singleCoreNSplit = CeilDiv(singleCoreN, nTailTile_);
int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
int64_t nSplitIdx = (blockIdx_ % totalTailTile_) / mTailTile_;
int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
    return {0, 0, 0, 0};  // 越界子块，kernel 直接跳过
}
```

- `blockIdx_ / totalTailTile_` → "我是第几个原 tile"
- `blockIdx_ % totalTailTile_` → "我是这个原 tile 的第几个子块"

### 5.5 具体例子

`totalCnt=5, blockNum=24, mTailTile=nTailTile=2 ⇒ totalTailTile=4`：

- 门槛 `5 × 4 = 20 ≤ 24`，通过。
- `tailOriCnt=5, newEndBlockIdx=4+5×3=19`。

| blockIdx_ | `/4` 原 tile | `%4` 子块 | 状态 |
|---|---|---|---|
| 0~3 | 0 | 0~3 | 干原 tile 0 的 4 个子块 |
| 4~7 | 1 | 0~3 | 干原 tile 1 |
| 8~11 | 2 | 0~3 | 干原 tile 2 |
| 12~15 | 3 | 0~3 | 干原 tile 3 |
| 16~19 | 4 | 0~3 | 干原 tile 4 |
| 20~23 | — | — | 仍空闲，自降为 1×1 |

激活核数从 5 提升到 20。

## 6. 三机制协同与完整取数流程

### 6.1 层级协同

```
Host Tiling
  ├── CalcBasicBlock / AdjustBasicBlock  → baseM, baseN, baseK
  ├── OptimizeEdgeBasicBlock             → 机制 B: mBaseTailSplitCnt, mTailMain
  ├── CalcTailBasicBlock                 → 机制 C: mTailTile, nTailTile
  └── CalcPathSpecificL1 / CalScaleFactors → depthA1/B1, stepK, scaleKL1

Device Scheduler
  ├── 构造期：算 mCoreNum_, mainRow_, round_, endBlockIdx_; 应用机制 B、C 的参数
  ├── GetTileIdx：serpentine(A) + 合并区形状(B) + 末轮子块偏移(C)
  └── GetBlockShape：重建 (singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset)

Device Kernel
  └── while (GetTileIdx) { 切 GM 张量 → BlockMmad }
```

三机制共享 `WINDOW_LEN = 4`，保证彼此边界不打架：

- 机制 A 的 serpentine 行窗口高度 = `min(WINDOW_LEN, mCnt)`。
- 机制 B 的合并区以 `windowSize = WINDOW_LEN` 为步长扩张。
- 机制 C 的切分只在 `curRoundIdx == round_ - 1` 时生效，不侵入前面的 serpentine 稳态。

### 6.2 `GetTileIdx` 单步做的事（按顺序）

```cpp
__aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
{
    if (roundIdx_ >= round_) return false;                              // 1. 判终止

    int64_t curRoundIdx = roundIdx_;
    int64_t newBlockIdx = (curRoundIdx == round_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
    int64_t tileIdx = newBlockIdx + curRoundIdx * blockNum_;            // 2. 机制 C 降维
    /* startBlockIdx_ / endBlockIdx_ 归一化修正 */

    /* 3. 机制 A: 行窗口拆分 + 奇数行 N 反转 */
    int64_t rowIdx = tileIdx / nCnt_ / mCoreNum_;
    int64_t mTileIdx, nTileIdx;
    if (rowIdx < mainRow_) { /* 稳态窗口 */ }
    else                   { /* 末窗口用 mTailCoreNum_ */ }
    if (rowIdx & 1) nTileIdx = nCnt_ - 1 - nTileIdx;

    /* 4. 机制 B: 合并区 tile 形状 */
    int64_t singleCoreM = mTileIdx >= mBaseNormCnt_
                          ? (mTileIdx < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_) : baseM_;
    int64_t singleCoreN = /* N 方向同理 */;

    /* 5. 机制 C: 末轮子块偏移 */
    int64_t mSplitAddrOffset = 0, nSplitAddrOffset = 0;
    if (totalTailTile_ > 1 && curRoundIdx == round_ - 1) {
        /* CeilDiv + blockIdx_ % totalTailTile_ 计算 mSplitAddrOffset/nSplitAddrOffset */
    }

    /* 6. 合并区 GM 偏差修正 */
    int64_t mPos = mTileIdx * baseM_ + mSplitAddrOffset;
    if (mTileIdx > mBaseNormCnt_) mPos -= (mTileIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
    int64_t nPos = /* N 方向同理 */;

    /* 7. 封装 BlockCoord: M/N 存 GM 起点，K/B 存逻辑 tile 索引 */
    Get<MNK_M>(blockCoord) = mPos;
    Get<MNK_N>(blockCoord) = nPos;
    Get<MNK_K>(blockCoord) = mTileIdx;
    Get<MNK_B>(blockCoord) = nTileIdx;
    roundIdx_++;
    return true;
}
```

`GetBlockShape` 在调用方读完 GM 坐标后用同一套逻辑重算 `(singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset)`，其中机制 C 越界的子块返回 `{0,0,0,0}`，由 kernel 层显式跳过。

## 7. 小结

SWAT 的设计哲学是 **"host 搜索 + device 重建"**：host tiling 把 `baseM/baseN/mTailTile/nTailTile/mBaseTailSplitCnt/nBaseTailSplitCnt/mTailMain/nTailMain` 这 8 个量作为最小描述集传下来，device 端按 `WINDOW_LEN` 为公共边界，把三机制逐层叠加出完整的核内任务序列。

三个机制各自的"不变量"值得牢记：

- **机制 A**：行窗口高度 ≤ `WINDOW_LEN`，奇数行 N 反向，稳态区与末窗口分别处理。
- **机制 B**：`mCnt_/nCnt_` 保持不变；只把最后 `mBaseTailSplitCnt` 个 tile 的尺寸从"`mergeLen` 个满块 + 1 个薄块"重排成"`mergeLen` 个 `mTailMain` + 1 个稍薄末块"；成本函数是所有 serpentine 窗口关键路径之和。
- **机制 C**：只作用于最后一轮；末轮所有 tile 被切成 `mTailTile × nTailTile` 个子块；门槛是 `(endBlockIdx_ + 1) × totalTailTile_ ≤ blockNum_`；`newEndBlockIdx` 的 `-1` 是因为不能重复计已激活的原 `tailOriCnt` 个核。

最终，SWAT 在 non-full-load 场景下同时压住了 **B 侧流式带宽、M/N 尾部碎片、末轮核浪费** 三个痛点，用一套统一的窗口化框架把 serpentine 主遍历、尾块重排、尾块再切分融合到同一个 `GetTileIdx/GetBlockShape` 管线里。
