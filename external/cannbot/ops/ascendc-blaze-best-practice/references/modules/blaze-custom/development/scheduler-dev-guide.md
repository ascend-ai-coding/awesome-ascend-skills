# Scheduler 层扩展开发指南

> **适用路径**：blaze_custom（路径 A）

---

## §1 接口规范

**构造函数**：`MyBlockScheduler(const ProblemShape& shape, const Params& params)`

内部计算 `mCnt_ = CeilDiv(m, baseM)`、`nCnt_`、`totalCnt_`、`round_` 等状态。

**必需 6 接口**：

| 接口 | 职责 |
|------|------|
| `GetTileNum()` | 当前核需处理的 tile 轮数 |
| `GetTileIdx(BlockCoord&)` | 按轮次返回 tile 坐标，false = 结束 |
| `GetBlockShape(BlockCoord)` | tile 实际 M/N 尺寸（含尾块修正） |
| `GetTileL1Shape(mL1, nL1, kL1)` | L1 级 tile 尺寸 |
| `GetTileL0Shape(mL0, nL0)` | L0 级 tile 尺寸 |
| `GetBlockCoord` | 通过 `GetTileIdx` 输出 (mIdx, nIdx) |

**Params 结构体**：

```cpp
struct Params {
    int64_t baseM, baseN;
    int64_t mTailTile, nTailTile;           // 尾块拆分数（默认 1）
    int64_t mBaseTailSplitCnt, nBaseTailSplitCnt;
    int64_t mTailMain, nTailMain;
};
```

---

## §2 蛇形遍历模板

Z-scan 遍历 M-N 网格，奇数行反转 N 方向提升 L2 cache 复用：

```cpp
bool GetTileIdx(BlockCoord& blockCoord) {
    if (roundIdx_ >= round_) return false;
    int64_t tileIdx = blockIdx_ + roundIdx_ * blockNum_;
    int64_t rowIdx = tileIdx / nCnt_ / mCoreNum_;
    int64_t mTileIdx, nTileIdx;
    if (rowIdx < mainRow_) {
        mTileIdx = rowIdx * mCoreNum_ + tileIdx % mCoreNum_;
        nTileIdx = (tileIdx / mCoreNum_) % nCnt_;
    } else {
        int64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
        mTileIdx = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
        nTileIdx = (tailIdx / mTailCoreNum_) % nCnt_;
    }
    if (rowIdx & 1) nTileIdx = nCnt_ - 1 - nTileIdx;
    Get<MNK_M>(blockCoord) = mTileIdx * baseM_;
    Get<MNK_N>(blockCoord) = nTileIdx * baseN_;
    roundIdx_++;
    return true;
}
```

`WINDOW_LEN = 4` 控制蛇形行宽（`mCoreNum_ = Min(4, mCnt_)`）。

---

## §3 尾块处理模式

| 策略 | 实现 | 适用 |
|------|------|------|
| 尾块合并 | `mBaseTailSplitCnt > 1` | 尾块 < baseM/2 |
| 尾块拆分 | `mTailTile > 1` | 尾块较大且核数有余 |
| 尾块独立 | 默认（`mTailTile=1`） | 通用默认 |

尾块尺寸：`mBaseTail_ = m - (mCnt_ - 1) * baseM_`。`GetBlockShape` 按 `mTileIdx >= mBaseNormCnt_` 判断尾块区域并修正。

---

## §4 Group Scheduler 扩展

`GroupMatmulBlockSchedulerSplitM` 支持跨 group 迭代：

```cpp
void UpdateNextProblem(const TupleShape& problemShape) {
    m_ = Get<MNK_M>(problemShape);  n_ = Get<MNK_N>(problemShape);
    mCnt_ = CeilDiv(m_, baseM_);  nCnt_ = CeilDiv(n_, baseN_);
    totalCnt_ = mCnt_ * nCnt_;
    roundIdx_ = 0;  // 重置轮次
}
void UpdateBaseM(uint32_t baseM) { baseM_ = baseM; }
```

Kernel 层外层遍历 group，内层 `GetTileIdx` 遍历当前 group 的 tile。

> 模块索引 → `references/modules/blaze-custom/scheduler-modules.md`
