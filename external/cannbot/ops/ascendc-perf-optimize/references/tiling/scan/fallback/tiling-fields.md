# Scan / Cum 族 — TilingData 字段语义

> 流式模板的 packed TilingData（`<<<>>>` 直调按值传参）。接入生产框架时可用 `BEGIN_TILING_DATA_DEF` 定义并扩展 borrow/fold/sklanskyIters 字段（参考 cumsum `CumsumSklanskyTilingData`）。

---

## 1. 流式模板 TilingData（ScanTilingData）

| 字段 | 类型 | 语义 | 计算 |
|------|------|------|------|
| `M` | int64 | 扫描轴前维度乘积 | `∏ shape[0:dim]` |
| `R` | int64 | 扫描轴长度 | `shape[dim]` |
| `N` | int64 | 扫描轴后维度乘积 | `∏ shape[dim+1:]` |
| `mBlk` | int64 | 每 work-item 行数（M 方向） | N 大→1；N 小→`W_TARGET/N`（clamp≤M） |
| `nTileLen` | int64 | 每 work-item 列数（N 方向） | N 大→`W_TARGET`；N 小→N |
| `mOuter` | int64 | M 方向 work-item 数 | `ceil(M/mBlk)` |
| `nOuter` | int64 | N 方向 work-item 数 | `ceil(N/nTileLen)` |
| `totalWork` | int64 | work-item 总数 | `mOuter*nOuter` |

> 向量宽 `W = mBlk*nTileLen`（目标 `W_TARGET≈1024`，按 `VL=64` 对齐）。`blockDim = min(totalWork, core_num)`；`totalWork < core_num` 时缩小分块补偿并行度。

## 2. 有效长度 vs 对齐长度

| 用途 | 用有效长度 | 用对齐长度 |
|------|:---:|:---:|
| DataCopyPad 搬运（Compact 打包 W） | ✅ | ❌ |
| UB 内偏移 / buffer 分配 | ❌ | ✅（W 按 VL 对齐） |

## 3. 跨核借 R（core partition）扩展字段

| 字段 | 语义 |
|------|------|
| `rBorrow` | R 分段数（≈ core_num） |
| `partLen` | 每段长度 `ceil(R/rBorrow)` |
| `W` | M*N（触发场景下极小，常为 1） |
| workspace `[rBorrow, W]` | 段总计（P 值）+ 段进位（P 值 + int32 索引），host 方式 B 用 |

## 4. 生产框架扩展（无索引 oneway/twoway，参考 cumsum）

| 字段组 | 语义 |
|--------|------|
| `exclusive` / `reverse` | 是否 exclusive 扫描 / 反向扫描 |
| `mBorrow`/`nBorrow`/`rBorrow` | 借 M/N/R 的分核计数 |
| `{M,R,N}BlockPara`（blockCount/blockFactor/blockTailFactor） | 各轴分核切分 |
| `{main,tail}Core{Main,Tail}UbPara`（ubCount/ubFactor/ubTailFactor） | 各轴 UB 分块 |
| `FoldPara`（foldCount/foldLen/sklanskyIter）×4 | twoway 折叠参数（主/尾核 × 主/尾 chunk） |
| `sklanskyItersCount` + `sklanskyItersTiling`×N | 跨核 Sklansky 层数与每层分组/进位偏移（方式 A） |
