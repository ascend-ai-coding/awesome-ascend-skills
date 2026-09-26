# 进阶：通用切分（任意 ubSplitAxis / 大 shape / 尾块非对齐）

> 主文档：[broadcast_design.md](broadcast_design.md)　|　参考代码：[`code/broadcast_common.h`](code/broadcast_common.h)（`ComputeTiling`）
>
> 入门样例（[broadcast_add](code/broadcast_add_tiling.cpp)）已切换到本文的通用切分。本文解释为什么不能写死 `ubSplitAxis=0`，以及通用切分的完整公式——这是把指导用到真实大 shape（如 cann-bench case 4 int32 8192×8192、case 10 fp16 363×367×373）的关键。

## 1. 为什么不能写死 ubSplitAxis=0

写死「尾轴整条进 UB、沿 0 轴切」只在 `尾轴元素数 ≤ elemNum` 时成立。真实 case 的尾轴常远超 UB：

| 场景 | 问题 |
|------|------|
| 尾轴 rowLen > elemNum（如 8192 fp32） | 一整行放不进 UB，必须**切到尾轴内部** |
| 大 shape 无广播（BRC_NONE） | 同样会尾轴超 UB，需与广播输入一样的多轴切分 |
| 高维 shape | 切分轴可能落在中间某轴，`dimProductBeforeUbInner` 不再等于 `ubOuter` |

## 2. 通用切分算法（`ComputeTiling`）

切分对象是**输出 shape**。从尾轴向外累乘，找能塞进 UB 的最内切分轴；若连尾轴一行都放不下，则切尾轴本身。

```cpp
// 1) 选切分轴：从尾轴向外，尽量把内层整轴塞进 UB
int splitAxis = 0; int64_t innerBelow = 1;       // innerBelow = ∏ outputDims[splitAxis+1 .. rank-1]
for (int ax = rank - 1; ax >= 0; ax--) {
    if (innerBelow * outputDims[ax] > elemNum) { splitAxis = ax; break; }
    innerBelow *= outputDims[ax]; splitAxis = ax;
}
// 2) 切分大小
if (innerBelow > elemNum) {                       // 连一行都放不下 → 切尾轴本身
    ubSplitAxis = rank - 1; innerBelow = 1; ubFormer = elemNum;     // 每块 elemNum 个元素
} else {
    ubSplitAxis = splitAxis; ubFormer = elemNum / innerBelow;       // 一次放几行
}
ubFormer = clamp(ubFormer, 1, outputDims[ubSplitAxis]);
ubOuter  = ceil(outputDims[ubSplitAxis] / ubFormer);
ubTail   = outputDims[ubSplitAxis] - (ubOuter - 1) * ubFormer;
```

### 关键：`dimProductBeforeUbInner` 通用公式

```
dimProductBeforeUbInner = (∏ outputDims[0 .. ubSplitAxis-1]) × ubOuter      // = UB tile 总数
```

- `ubSplitAxis = 0`：∏(空)=1 → 退化为 `ubOuter`（即入门样例的特例）。
- `ubSplitAxis = rank-1`（切尾轴）：`∏ outputDims[0..rank-2] × ubOuter`。

这个值喂给 `GetAxesIndices`/`GetGmOffset`，**ubSplitAxis≠0 时必须用通用公式**，否则下标还原全错。完整实现见 `ComputeTiling`。

### 多核

```
totalTiles  = dimProductBeforeUbInner;
blockFormer = ceil(totalTiles / coreNum);        // 先按核数算每核最多几个 tile
blockNum    = ceil(totalTiles / blockFormer);    // 再回收用不上的核
blockTail   = totalTiles - (blockNum-1)*blockFormer;   // ∈[1, blockFormer]
```

> 顺序很关键：若先 `blockNum=min(coreNum,totalTiles)` 再 `blockFormer=ceil(totalTiles/blockNum)`，会出现 `blockTail<0`（如 totalTiles=100、coreNum=64 → blockFormer=2、blockTail=-26），非尾核处理越界 tile。必须先定 blockFormer 再回收核。

## 3. 尾块非 32B 对齐

`ubTail = splitDim - (ubOuter-1)*ubFormer` 往往**不是 alignEle 的倍数**（如质数 shape、大 shape 展平）。处理方式：

- **搬运**：`DataCopyPad` 的 `blockLen` 以**字节**计，内部自动 pad 到 32B；非对齐尾块**直接传真实长度即可**，无需手动 mask/padding（`DataCopyPadCompact` 已如此）。
- **计算**：`Add`/`Broadcast` 等矢量指令的 `count` 传真实元素数，AscendC 处理非对齐尾数；不要按对齐数多算，否则踩到相邻 tile。
- **输出**：`DataCopyPad` 写回同样传真实 `tile` 字节，不会越界写。

> 即：连续布局下尾块非对齐**全部由 DataCopyPad 兜底**，开发者不需要额外分支。仅当用 `DataCopy`（非 Pad）搬非对齐数据时才需自己 padding。

## 4. 大 shape 无广播（BRC_NONE）

无广播的大 shape（cann-bench case 4/5/6）和广播输入**共用同一套切分**：`ComputeTiling` 只看输出 shape，与是否广播无关。BRC_NONE 输入在 kernel 走 `CopyInPlain`，其 `len = rows * inputStrides[ubSplitAxis]`，因 stride 与输出一致即为 `tile`，天然支持尾轴被切。所以"无广播但 shape 很大"无需特殊处理——切分通用化后自动覆盖。

## 5. NDDMA 在 ubSplitAxis≠0 的参数装配

`MakeNddmaParams` / `BroadcastNddma` 已按 `ubSplitAxis` 参数化，`ubSplitSize` 语义是「**切分轴方向本 tile 的元素/行数**」：

- `ubSplitAxis = 0`：`ubSplitSize = rows`（行数）。
- `ubSplitAxis = rank-1`（切尾轴）：`ubSplitSize = ubFormer`（尾轴一段的元素数），`axisInsideUb = NDDMA_DIM-1`，只填切分轴一维 + 高位补 1。

`axisInsideUb = NDDMA_DIM - (shapeLen - ubSplitAxis)`：rank 越大、切分轴越靠外，需补的 size=1 高位轴越少。**前置条件 `rank = shapeLen - ubSplitAxis ≤ NDDMA_DIM(5)`**；超过时 Host 选型回退 ③（`PickBroadcastMode`），官方 atvoss 则走 with-loop NDDMA（本样例未实现）。

## 6. cann-bench case 对照

| case | shape / dtype | 切分结果（ComputeTiling） |
|------|--------------|------------------------|
| 4 | int32 8192×8192 | 尾轴 8192 > elemNum → 切尾轴(ubSplitAxis=1)，ubFormer=elemNum；无广播走 BRC_NONE |
| 7/8/9 | 质数非对齐 shape | ubTail 非对齐 → DataCopyPad 自动兜底 |
| 10 | fp16 363×367×373 | 尾轴 373 若 ≤ elemNum 则切中间轴(ubSplitAxis=1)，dimProductBeforeUbInner=363×ubOuter |

> 这些 case 现在都能由通用切分覆盖；写死 ubSplitAxis=0 的旧样例会在 case 4/10 上因「一行放不进 UB」失败。
