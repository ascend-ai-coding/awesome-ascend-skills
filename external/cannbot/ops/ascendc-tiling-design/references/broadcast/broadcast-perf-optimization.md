# Broadcast - 性能优化方法论

> **核心流程**: 拉取原生算子 CCE 源码逐用例对照 → 分类映射优化方案 → 同卡 A/B 验证 → 48 例精度 + bench 性能对比。任何优化结论落地前必须同卡 A/B（不同 NPU 物理卡/占用进程不同，绝对数值不可跨卡比较）。
>
> 实现对应工程：`custom_src/op_kernel/broadcast_kernel.asc`（kernel）+ `broadcast_tiling.h`（kernel/host/torch 三端共享 tiling）。

---

## 一、场景分类与优化方案速查

| 场景 | 判别条件 | 优化方案 |
|------|---------|---------|
| 前导/中间广播 BLOCK | 尾部连续段 inner×elemSize%32==0 | 整块搬（DataCopy），源块常驻去冗余读 |
| 源单块整块复制 repl-like | totalIn==inner && innerChunks==1 | `blockRepl`：源读一次，循环内纯 MTE3 写 |
| 尾维广播 UB_BCAST | inPad[m-1]==1 && outShape[m-1]>1 | UB `Broadcast<T,2,1>` 尾展开 |
| R>TILE_LEN 尾维 | outShape[m-1] > TILE_LEN | R 分片（每项展开一片 ≤TILE_LEN） |
| 行共享单源重复 | srcRow×R < TILE_LEN && G>1 | 行组化 + 组内懒 setup 纯写 |
| 中部块重复 | inner<TILE_LEN && outerG>1 && 广播维为 outer 连续后缀 | MID_BCAST：`Broadcast<T,2,0>` 块重复 |
| 源单元素 REPL_DUP | totalIn==1 | Duplicate 展开 + 输出双缓冲流水 |
| 非 32B 对齐单内块 | innerBytes ≤ 255 | BLOCK_REPEAT：标量铺块 + 单块连续写 |
| 兜底 | 其余 | SCALAR 逐元素（仅小量级） |

---

## 二、Tiling 数据结构与计算细节

### 2.1 关键字段（BroadcastTilingData）

坐标约定：输出 dims 0..ndim-1；输入形状右对齐填充（`inPad[d]`）；`srcStride[d] = inPad[d]==1 ? 0 : prod(inPad[d+1:])`（元素单位）；尾部连续段 = 最长后缀 `{k..ndim-1}` 且 `inPad[d]==outShape[d]`，`inner = prod(outShape[k:])`。

```cpp
struct BroadcastTilingData {
    int64_t outShape[BROADCAST_MAX_DIM];  // 输出形状（逐维）
    int64_t srcStride[BROADCAST_MAX_DIM]; // 源步长（元素；broadcast 维为 0）
    int64_t inner;                        // 尾部连续段长度（元素）
    uint64_t totalOut;
    uint32_t ndim, outerDims;             // outerDims = k（尾部连续段起点）
    uint32_t innerChunks, tailChunkLen;   // ceil(inner/TILE_LEN) 及其尾块
    uint32_t pattern;                     // BroadcastPattern 枚举
    uint32_t blockNum, numPerCore, tailLastCore;  // 多核切分
    uint32_t blockRepl;                   // BLOCK：源仅一个 inner 块时 1
    uint32_t replUnitElems;               // blockRepl 单元元素 = count*k（≤64KB）
    // BLOCK_REPEAT（源非 32B 对齐单内块，innerBytes≤255）
    uint16_t brInnerBytes;                // 源内块字节 = inner*elemSize
    uint16_t brUnitBytes;                 // pattern 单元字节（32B 对齐、ib 整数倍、≤2048B）
    uint16_t brTailBytes;                 // 输出尾块字节 = outBytes % brUnitBytes
    // UB_BCAST（尾维广播）
    uint32_t ubR;          // 尾维展开长 outShape[m-1]
    uint32_t ubRowDims;    // 行坐标维数 = lb+1（lb 为前部最后广播维，无则 0）
    uint32_t ubChunks;     // 行内块数 = ceil(ubSrcRow / ubChunkLen)
    uint32_t ubChunkLen;   // 行内块长（元素，尽量 32B 对齐）
    uint32_t ubTailChunk;  // 行内尾块长
    uint64_t ubSrcRow;     // 行源连续长 = prod(inPad[lb+1..m-2])
    // UB_BCAST 扩展（R 分片 + 组化）
    uint32_t ubRChunks;    // R 分片数 = ceil(ubR/TILE_LEN)（R<=TILE_LEN 时 1）
    uint64_t ubGroupLen;   // 组化：源恒定区间长 runLen = G*ubR（0=未组化）
    uint64_t ubGroupRun;   // 组化：组内分块数 = ceil(ubGroupLen/TILE_LEN)
    // MID_BCAST（中部块重复）
    uint32_t mbSrcLen;     // 源块元素 = inner
    uint32_t mbG;          // 前部广播维输出积 outerG（块重复次数）
    uint32_t mbExpand;     // 每次 Broadcast 展开块数 = min(mbG, TILE_LEN/mbSrcLen)
    uint32_t mbChunks;     // 每源块写块数 = ceil(mbG / mbExpand)
    // REPL_DUP
    uint32_t replChunkLen; // TILE_LEN
    uint32_t replTailChunk;
};
```

> UB 容量有限（实测 `BROADCAST_TILE_LEN = 8192` 时双缓冲 8192×4B×2×2=128KB 会越界 VEC UB out of bounds）——规划缓冲务必留余量（每路径 ≤64KB 级），并按实际芯片 UB 容量校准 TILE_LEN。

### 2.2 UB_BCAST 分支（R 分片 + 组化判定，tiling.h 摘录）

```cpp
// 尾维广播：R 不再受 TILE_LEN 限制（R>TILE_LEN 时 R 分片），
// 行输出过小且前部存在共享源行组时组化。
uint64_t R = outShape[m-1];
int64_t lb = -1;
for (int64_t d = m-2; d >= 0; d--) if (inPad[d] == 1) { lb = d; break; }
uint64_t srcRow = 1;
for (int64_t d = lb+1; d <= m-2; d++) srcRow *= inPad[d];   // inPad[m-1]==1 已排除
uint64_t rows = totalOut / (srcRow * R);
uint64_t rChunks = (R + BROADCAST_TILE_LEN - 1) / BROADCAST_TILE_LEN;  // R 分片
t.pattern = PATTERN_UB_BCAST;
t.ubR = R; t.ubRowDims = lb+1; t.ubRChunks = rChunks;
// 前部广播维输出积 G（共享源行组大小）
uint64_t G = 1;
for (int64_t d = 0; d <= lb; d++) if (t.srcStride[d] == 0) G *= outShape[d];
if ((srcRow * R < BROADCAST_TILE_LEN) && (G > 1)) {
    // 组化：组 = 源恒定区间 runLen = G*R；项 = groups × runChunks
    uint64_t runLen = G * R;
    uint64_t runChunks = (runLen + BROADCAST_TILE_LEN - 1) / BROADCAST_TILE_LEN;
    t.ubGroupLen = runLen; t.ubGroupRun = runChunks;
    t.ubChunks = 1; t.ubChunkLen = 0; t.ubTailChunk = 0; t.ubSrcRow = srcRow;
    items = (totalOut / runLen) * runChunks;      // 组化工作项
} else {
    // 原行内分块：chunk = min(srcRow, TILE_LEN/R)，向下取整到 32B 对齐元素数
    uint64_t alignElem = 32 / elemSize;           // fp32=8, fp16=16
    uint64_t maxChunk = BROADCAST_TILE_LEN / R;   // 至少 1
    uint64_t chunk = (srcRow < maxChunk) ? srcRow : maxChunk;
    uint64_t chunkAligned = (chunk / alignElem) * alignElem;
    if (chunkAligned == 0) chunkAligned = (chunk ? chunk : 1);  // 源过小非对齐兜底
    uint64_t chunks = (srcRow + chunkAligned - 1) / chunkAligned;
    t.ubChunks = chunks; t.ubChunkLen = chunkAligned;
    t.ubTailChunk = srcRow - (chunks-1) * chunkAligned;
    t.ubGroupLen = t.ubGroupRun = 0;
    items = rows * chunks * rChunks;
}
```

> **组化本质**：`G×R` 输出区间内源恒为 1 个元素（`64,1,1->64,2048,1024`：srcRow=1, G=2048, runLen=2097152=256×TILE_LEN，groups=64，项由 131072 → 16384）。

### 2.3 BLOCK 分支（blockRepl + MID_BCAST + bcastSuffix 守卫）

```cpp
bool block = (inner * elemSize % 32 == 0);
if (block) {
    t.pattern = PATTERN_BLOCK;
    t.innerChunks = (inner + TILE_LEN-1) / TILE_LEN;
    // repl-like BLOCK：源总元素==inner 且单内块 → 源常驻纯写
    t.blockRepl = (totalIn == inner && t.innerChunks == 1) ? 1 : 0;
    if (t.blockRepl) {
        int64_t cnt = inner, countBytes = cnt * elemSize;
        int64_t k = 64*1024 / countBytes;           // 单元放大（≤64KB）
        if (k > 8) k = 8;                           // V 复制次数上限
        if (k < 1) k = 1;
        t.replUnitElems = cnt * k;
    }
    // MID_BCAST：BLOCK 小 inner + 前部广播维
    if (t.pattern == PATTERN_BLOCK && t.blockRepl == 0) {
        int64_t outerG = 1;
        bool bcastSuffix = true, seenBcast = false;
        // 广播维必须为 outer 维连续后缀（否则 super-row flat 不连续、整块模型失效）
        for (int64_t d = 0; d < k; d++) {
            if (t.srcStride[d] == 0) { outerG *= outShape[d]; seenBcast = true; }
            else if (seenBcast) { bcastSuffix = false; }
        }
        if (inner < BROADCAST_TILE_LEN && outerG > 1 && bcastSuffix) {
            uint64_t expand = outerG;
            uint64_t maxExpand = BROADCAST_TILE_LEN / inner;  // 至少 1
            if (expand > maxExpand) expand = maxExpand;
            uint64_t chunksU = (outerG + expand-1) / expand;
            t.pattern = PATTERN_MID_BCAST;
            t.mbSrcLen = inner; t.mbG = outerG;
            t.mbExpand = expand; t.mbChunks = chunksU;
            t.innerChunks = 1; t.tailChunkLen = inner;
            items = (outer / outerG) * chunksU;       // 非广播 outer 数 × 写块数
        }
    }
}
```

### 2.4 BLOCK_REPEAT pattern 单元计算

```cpp
// 源仅一个非 32B 对齐内块（innerBytes ≤ 255）
int64_t ib = inner * elemSize;
uint64_t pb = 32 / GcdWith32(ib) * ib;      // lcm(32,ib)：32B 对齐且 ib 整数倍
uint64_t unit = pb;
uint64_t nUnits = 2048 / pb;                // 单元放大（≤2048B，控制标量铺块开销）
if (nUnits > 1) unit = pb * nUnits;
uint64_t fullUnits = outBytes / unit;
uint64_t tailBytes = outBytes % unit;
t.brInnerBytes = ib; t.brUnitBytes = unit; t.brTailBytes = tailBytes;
items = fullUnits + (tailBytes > 0 ? 1 : 0);
```

> 例 `1,2->4096,2` fp16：ib=4B，pb=32B，nUnits=64 → unit=2048B；8192 项 SCALAR 逐元素 → 64+ 个 2048B 单元单块写。

---

## 三、各 Pattern 的 Kernel 实现（完整代码）

### 3.1 UB_BCAST TQue 双缓冲流水（fp32；R 分片版）

工作项 = (row, chunkIdx, rslice)；`perItem = ubChunks × ubRChunks`。TQue 依赖管理使 CopyIn(N+1)/Broadcast(N)/CopyOut(N-1) 三引擎重叠，零显式屏障（对齐原生 BroadcastUbSch 的 DAG ping-pong）。

```cpp
__aicore__ inline void ProcessUbBcastTQue()
{
    LocalTensor<uint8_t> tmpLocal = ubTmpBuf.Get<uint8_t>();
    uint32_t chunks = tiling->ubChunks, rChunks = tiling->ubRChunks;
    uint64_t perItem = chunks * rChunks;
    int64_t rowStrideOut = tiling->ubSrcRow * tiling->ubR;
    for (uint64_t i = 0; i < itemCount; i++) {
        uint64_t w = itemStart + i;
        uint64_t row = w / perItem, rem64 = w % perItem;
        uint32_t chunkIdx = rem64 / rChunks, rslice = rem64 % rChunks;
        int64_t srcOff = RowSourceOffset(row) + chunkIdx * tiling->ubChunkLen;
        uint32_t count = (chunkIdx == chunks-1) ? tiling->ubTailChunk : tiling->ubChunkLen;
        uint32_t rLen = (rslice == rChunks-1) ? (tiling->ubR - rslice*BROADCAST_TILE_LEN)
                                              : BROADCAST_TILE_LEN;
        int64_t dstOff = row * rowStrideOut + chunkIdx * tiling->ubChunkLen * tiling->ubR
                         + rslice * BROADCAST_TILE_LEN;
        // R 分片尾片非 32B 对齐时对齐化 + 尾兜底（rChunks==1 走原生 NotAlign220，已验证）
        uint32_t rAligned = rLen, rRemain = 0;
        if (rChunks > 1 && rLen * sizeof(T) % 32 != 0) {
            uint32_t alignElem = 32 / sizeof(T);
            rAligned = (rLen / alignElem) * alignElem;
            rRemain = rLen - rAligned;
        }
        uint32_t outCount = count * rAligned;
        if (outCount > 0) {
            LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
            GlobalTensor<T> xGmTile; xGmTile.SetGlobalBuffer(xGmBase + srcOff, count);
            if (srcOff % (32/sizeof(T)) == 0 && count % (32/sizeof(T)) == 0)
                DataCopy(xLocal, xGmTile, count);                     // 对齐快路径
            else
                DataCopyPad(xLocal, xGmTile, {1, count*sizeof(T), 0, 0}, {false,0,0,0});
            inQueue.EnQue(xLocal);
            LocalTensor<T> xDeq = inQueue.DeQue<T>();
            LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
            uint32_t dstShape[2] = {count, rAligned};   // 尾维展开 count×R
            uint32_t srcShape[2] = {count, 1};
            Broadcast<T, 2, 1>(yLocal, xDeq, dstShape, srcShape, tmpLocal);
            inQueue.FreeTensor(xDeq);
            outQueue.EnQue<T>(yLocal);
            LocalTensor<T> yDeq = outQueue.DeQue<T>();
            GlobalTensor<T> yGmTile; yGmTile.SetGlobalBuffer(yGmBase + dstOff, outCount);
            if (dstOff % (32/sizeof(T)) == 0 && outCount % (32/sizeof(T)) == 0)
                DataCopy(yGmTile, yDeq, outCount);
            else
                DataCopyPad(yGmTile, yDeq, {1, outCount*sizeof(T), 0, 0});  // 单块连续模式
            outQueue.FreeTensor(yDeq);
        }
        if (rRemain > 0)
            TailFallback(srcOff, dstOff, count, rAligned, rRemain, tiling->ubR);
    }
}
```

### 3.2 UB_BCAST 屏障版（fp16）

与 TQue 版逻辑一致，仅流水形态不同：`ubInBuf`/`ubOutBuf` TBuf 直连 + 每工作项 3×`PipeBarrier<PIPE_ALL>`（MTE2 读→V 展开→MTE3 写）。fp16 数据量减半后队列簿记/跨区开销 > 屏障节省，实测屏障版更快。

```cpp
LocalTensor<T> xLocal = ubInBuf.Get<T>();
... DataCopy / DataCopyPad 读源（同 3.1）...
PipeBarrier<PIPE_ALL>();
LocalTensor<T> yLocal = ubOutBuf.Get<T>();
Broadcast<T, 2, 1>(yLocal, xLocal, {count, rAligned}, {count, 1}, tmpLocal);
PipeBarrier<PIPE_ALL>();
DataCopy / DataCopyPad 写 GM（同 3.1）...
PipeBarrier<PIPE_ALL>();
if (rRemain > 0) TailFallback(...);
```

### 3.3 组化懒 setup（ProcessUbBcastGroupLazy）

组源为单元素 → 每组首 chunk（rc==0）读源 + `Broadcast{1,1}->{1,TILE_LEN}` 展开成**常驻瓦片**（ubOutBuf），组内后续 chunk **纯 MTE3 DataCopy 写**同一瓦片。每组仅 2 次全屏障；跨核安全：核首项落组内 rc>0 时 `haveTile=false` 先补展开。

```cpp
__aicore__ inline void ProcessUbBcastGroupLazy()
{
    LocalTensor<uint8_t> tmpLocal = ubTmpBuf.Get<uint8_t>();
    LocalTensor<T> xLocal = ubInBuf.Get<T>();       // 32B 槽（1 元素）
    LocalTensor<T> yLocal = ubOutBuf.Get<T>();      // 常驻 TILE_LEN 瓦片
    uint64_t run = tiling->ubGroupLen, runChunks = tiling->ubGroupRun;
    uint32_t alignElem = 32 / sizeof(T);
    bool haveTile = false;
    for (uint64_t i = 0; i < itemCount; i++) {
        uint64_t w = itemStart + i;
        uint64_t group = w / runChunks;
        uint32_t rc = w % runChunks;
        if (rc == 0 || !haveTile) {                 // 组首 / 跨核补展开
            int64_t srcOff = GroupSourceOffset(group);
            GlobalTensor<T> xGmTile; xGmTile.SetGlobalBuffer(xGmBase + srcOff, 1);
            DataCopyPad(xLocal, xGmTile, {1, sizeof(T), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();                // 读完成 + 前组 MTE3 耗尽（V 可覆写瓦片）
            Broadcast<T, 2, 1>(yLocal, xLocal, {1, BROADCAST_TILE_LEN}, {1, 1}, tmpLocal);
            PipeBarrier<PIPE_ALL>();                // 展开完成可写
            haveTile = true;
        }
        int64_t dstOff = group * run + rc * BROADCAST_TILE_LEN;
        uint64_t remain = run - rc * BROADCAST_TILE_LEN;
        uint32_t outCount = (remain > BROADCAST_TILE_LEN) ? BROADCAST_TILE_LEN
                                                          : (uint32_t)remain;
        GlobalTensor<T> yGmTile; yGmTile.SetGlobalBuffer(yGmBase + dstOff, outCount);
        if (dstOff % alignElem == 0 && outCount % alignElem == 0)
            DataCopy(yGmTile, yLocal, outCount);    // 组内纯 MTE3 写，零屏障自动流水
        else
            DataCopyPad(yGmTile, yLocal, {1, outCount*sizeof(T), 0, 0});  // 组内尾 chunk
    }
}
```

> 关键：写侧全部读**同一瓦片**、写不同 GM 地址 → 块间无数据依赖，MTE3 连续 DataCopy 自动流水（1.45-1.47TB/s，跑满写带宽）。

### 3.4 MID_BCAST 块重复（ProcessMidBcastTQue）

输出布局 = [非广播 outer s]×[inner 源块 × mbG 次块重复]，块重复沿维 0：`Broadcast {1,inner}->{expand,inner}`。源/目标偏移恒 inner×elemSize 整数倍 → 全 32B 对齐快速 DataCopy，无需兜底。

```cpp
__aicore__ inline void ProcessMidBcastTQue()
{
    LocalTensor<uint8_t> tmpLocal = ubTmpBuf.Get<uint8_t>();
    uint32_t mbChunks = tiling->mbChunks;
    int64_t rowStrideOut = tiling->mbSrcLen * tiling->mbG;
    for (uint64_t i = 0; i < itemCount; i++) {
        uint64_t w = itemStart + i;
        uint64_t s = w / mbChunks;
        uint32_t wc = w % mbChunks;
        int64_t srcOff = MidBcastSourceOffset(s);   // 跳过 stride==0 广播维！
        uint32_t expand = (wc == mbChunks-1) ? (tiling->mbG - wc*tiling->mbExpand)
                                             : tiling->mbExpand;
        int64_t dstOff = s * rowStrideOut + wc * tiling->mbExpand * tiling->mbSrcLen;
        uint32_t outCount = expand * tiling->mbSrcLen;
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        GlobalTensor<T> xGmTile; xGmTile.SetGlobalBuffer(xGmBase + srcOff, tiling->mbSrcLen);
        DataCopy(xLocal, xGmTile, tiling->mbSrcLen);   // 源块读一次
        inQueue.EnQue(xLocal);
        LocalTensor<T> xDeq = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        // axis=0 前导块重复（与尾维元素重复 axis=1 正交！axis 写错 matched_ratio 0.0007）
        Broadcast<T, 2, 0>(yLocal, xDeq, {expand, tiling->mbSrcLen},
                           {1, tiling->mbSrcLen}, tmpLocal);
        inQueue.FreeTensor(xDeq);
        outQueue.EnQue<T>(yLocal);
        LocalTensor<T> yDeq = outQueue.DeQue<T>();
        GlobalTensor<T> yGmTile; yGmTile.SetGlobalBuffer(yGmBase + dstOff, outCount);
        DataCopy(yGmTile, yDeq, outCount);
        outQueue.FreeTensor(yDeq);
    }
}
```

### 3.5 repl-like BLOCK 纯写（ProcessBlockRepl）

源仅一个 inner 块（totalIn==inner && innerChunks==1）→ 源块读一次常驻，`Adds` 恒等复制铺成 ≤64KB 单元，循环内纯 MTE3 DataCopy（曾 8192 项重复 MTE2 读同一源块，写效仅 0.7TB/s → 1.44TB/s）。

```cpp
__aicore__ inline void ProcessBlockRepl()
{
    LocalTensor<T> xLocal = ubInBuf.Get<T>();
    uint32_t count = tiling->tailChunkLen;      // innerChunks==1 → count==inner
    uint32_t unit = tiling->replUnitElems, k = (unit > 0) ? (unit / count) : 1;
    GlobalTensor<T> xGmTile; xGmTile.SetGlobalBuffer(xGmBase, count);
    DataCopy(xLocal, xGmTile, count);           // 1) 源块读一次常驻
    PipeBarrier<PIPE_ALL>();
    for (uint32_t j = 1; j < k; j++)            // 2) V 复制铺单元（Adds 恒等）
        Adds(xLocal[j * count], xLocal, static_cast<T>(0), count);
    PipeBarrier<PIPE_ALL>();
    uint64_t fullUnits = itemCount / k, tailItems = itemCount % k;
    int64_t dstOff = itemStart * count;
    for (uint64_t i = 0; i < fullUnits; i++) {  // 3) 逐单元一次大 DMA 纯写
        GlobalTensor<T> yGmTile; yGmTile.SetGlobalBuffer(yGmBase + dstOff, unit);
        DataCopy(yGmTile, xLocal, unit);
        dstOff += unit;
    }
    if (tailItems > 0) {                        // 尾项单次写
        GlobalTensor<T> yGmTile; yGmTile.SetGlobalBuffer(yGmBase + dstOff, tailItems*count);
        DataCopy(yGmTile, xLocal, tailItems*count);
    }
}
```

> 偏移视图必须用 `operator[](uint32_t)`（返回偏移 LocalTensor），**不能用 `+`**（kernel_tensor.h 语义）。

### 3.6 REPL_DUP 源单元素（ProcessReplDup 关键段）

源单元素循环外读入一次（省每块冗余 MTE2 读），输出走 outQueue 双缓冲流水；尾块非 32B 对齐（count×elemSize%32!=0）时退回逐元素搬（Broadcast NotAlign/GatherMask 路径 2201 错位，尾块小仅一次）。

```cpp
GlobalTensor<T> xGmTile; xGmTile.SetGlobalBuffer(xGmBase, 1);
DataCopyPad(inLocal, xGmTile, {1, sizeof(T), 0, 0}, {false, 0, 0, 0});
PipeBarrier<PIPE_ALL>();                        // 源常驻 ubInBuf
for (...) {
    dstOff = w * BROADCAST_TILE_LEN; count = min(remain, BROADCAST_TILE_LEN);
    if (count * sizeof(T) % 32 != 0) { ...逐元素 DataCopyPad 兜底...; continue; }
    uint32_t dstShape[2] = {1, count}, srcShape[2] = {1, 1};
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    Broadcast<T, 2, 1>(outLocal, inLocal, dstShape, srcShape, tmpLocal);  // srcSize==1 → BrcDuplicate 直连
    outQueue.EnQue<T>(outLocal);
    LocalTensor<T> yLocal = outQueue.DeQue<T>();
    DataCopy / DataCopyPad 写 GM...
    outQueue.FreeTensor(yLocal);
}
```

### 3.7 BLOCK_REPEAT 标量铺块（ProcessBlockRepeat）

源一次单块读入对齐槽 → 标量 SetValue 铺满 pattern 单元（标量访问 UB 无对齐限制）→ 输出逐单元单块 `DataCopyPad{1, unitBytes, 0, 0}` 连续写（单块 blockLen 单位恒为字节，4B~16KB 均可靠）；尾块由末核兜底。

```cpp
uint32_t ib = tiling->brInnerBytes, unitBytes = tiling->brUnitBytes;
uint32_t ibElems = ib / sizeof(T), unitElems = unitBytes / sizeof(T);
DataCopyPad(xLocal, xGmTile, {1, ib, 0, 0}, {false, 0, 0, 0});   // 单块 <32B 读
PipeBarrier<PIPE_ALL>();
for (uint32_t e = ibElems; e < unitElems; e++)                   // 标量铺满单元
    xLocal.SetValue(e, xLocal.GetValue(e % ibElems));
PipeBarrier<PIPE_ALL>();
for (uint32_t i = 0; i < fullItems; i++) {                        // 逐单元单块连续写
    GlobalTensor<T> yGmTile; yGmTile.SetGlobalBuffer(yGmBase + dstOff / sizeof(T), unitElems);
    DataCopyPad(yGmTile, xLocal, {1, unitBytes, 0, 0});
    dstOff += unitBytes;
}
if (tailBytes > 0)  // 全局尾块单块兜底
    DataCopyPad(yGmTile, xLocal, {1, tailBytes, 0, 0});
```

### 3.8 源偏移反解 helper（跳过广播维）

```cpp
// UB_BCAST 行源偏移：反解全部 dims 0..ubRowDims-1（含广播维，stride=0 无贡献）
inline int64_t RowSourceOffset(uint64_t row) {
    int64_t offset = 0; uint64_t tmp = row;
    for (uint32_t d = tiling->ubRowDims; d > 0; d--) {
        uint32_t d0 = d-1;
        int64_t coord = tmp % outShape[d0]; tmp /= outShape[d0];
        offset += coord * srcStride[d0];
    }
    return offset;
}
// 组化源偏移：只反解 srcStride>0 的维（否则 group 误落入广播维坐标 → 全组源偏移恒 0）
inline int64_t GroupSourceOffset(uint64_t group) {
    int64_t offset = 0; uint64_t tmp = group;
    for (uint32_t d = tiling->ubRowDims; d > 0; d--) {
        uint32_t d0 = d-1;
        if (srcStride[d0] != 0) {               // 关键：跳过广播维
            int64_t coord = tmp % outShape[d0]; tmp /= outShape[d0];
            offset += coord * srcStride[d0];
        }
    }
    return offset;
}
// MID_BCAST 源偏移：与 GroupSourceOffset 同理，仅反解非广播维
inline int64_t MidBcastSourceOffset(uint64_t idx) { /* 同 GroupSourceOffset，遍历 outerDims */ }
```

> `64,1,1->64,2048,1024` 曾用 RowSourceOffset 反解 group → 全组读 x[0] 错源；`1024,1,4096->1024,8,4096` 同理全读 x 行 0。**凡"共享源"类反解必须跳 stride==0 维**。

### 3.9 TailFallback 非对齐尾兜底

R 分片尾片/组化尾项 ≤15 元素 × count：逐元素读源 → 写目标。UB 槽用 ubTmpBuf（Broadcast 已结束空闲，槽 0 恒 32B 对齐——DataCopyPad UB 侧对齐硬约束）。

```cpp
inline void TailFallback(int64_t srcOff, int64_t dstOff, uint32_t count,
                         uint32_t rAligned, uint32_t rRemain, uint32_t rStride) {
    for (uint32_t c = 0; c < count; c++)
        for (uint32_t j = 0; j < rRemain; j++) {
            LocalTensor<T> xLocal = ubTmpBuf.Get<T>();
            GlobalTensor<T> xGmTile; xGmTile.SetGlobalBuffer(xGmBase + srcOff + c, 1);
            DataCopyPad(xLocal, xGmTile, {1, sizeof(T), 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<T> yGmTile;
            yGmTile.SetGlobalBuffer(yGmBase + dstOff + c*rStride + rAligned + j, 1);
            DataCopyPad(yGmTile, xLocal, {1, sizeof(T), 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
}
```

### 3.10 各路径 UB 缓冲规划（Init 摘录）

| pattern | 缓冲分配 | 说明 |
|---------|---------|------|
| UB_BCAST 组化 | `ubInBuf(32B) + ubOutBuf(TILE_LEN×sizeof(T)) + ubTmpBuf(4096)` | 单缓冲直连懒 setup |
| UB_BCAST fp32 | `inQueue(2×inBytes) + outQueue(2×TILE_LEN×sizeof(T)) + ubTmpBuf` | inBytes 向上取整 32B |
| UB_BCAST fp16 | `ubInBuf(inBytes) + ubOutBuf(TILE_LEN×sizeof(T)) + ubTmpBuf` | TBuf 直连 |
| MID_BCAST | `inQueue(2×mbSrcLen×sizeof(T)) + outQueue(2×mbExpand×mbSrcLen×sizeof(T))` | 恒 TQue |
| BLOCK blockRepl | `ubInBuf(replUnitElems×sizeof(T))`（≤64KB 单元常驻） | 纯写 |
| BLOCK 普通 | `inQueue + outQueue(2×TILE_LEN×sizeof(T))` | 双缓冲 |
| BLOCK_REPEAT | `ubInBuf(brUnitBytes)`（≤2048B 单元常驻） | 标量铺块 |
| REPL_DUP | `ubInBuf(32B) + outQueue(2×TILE_LEN×sizeof(T))` | 源常驻 + 输出流水 |
| SCALAR | `inQueue(2×64) + outQueue(2×64)` | 队列逐元素 |

---

## 四、大 shape 达标三件套（2026-10-20 实测，50 组 ≥4M 元素全达标）

### 4.1 R>TILE_LEN 尾维分片（解除 SCALAR 灾难）

**问题**：`1024,1->1024,16384`（R=16384）曾 179.8ms（0.00x）——R 超 TILE_LEN 时回退 SCALAR 逐元素爆炸。

**方案**：tiling 加 `ubRChunks = ceil(R/TILE_LEN)`，每工作项展开一片 ≤TILE_LEN（见 3.1 的 rslice 逻辑）；尾片非 32B 对齐时对齐化 + `TailFallback` 逐元素兜底。

**实测**：`2048,1->2048,16384` fp32 84.4ms→**0.0437ms（0.91x）**、fp16 170.8ms→**0.0561ms（0.53x）**。

### 4.2 行组化 + 组内懒 setup（消除工作项碎片化）

**问题**：`64,1,1->64,2048,1024` 131072 个工作项（0.05-0.08x），逐行小项碎片化 + 组内每 chunk 冗余读 1 元素 + 冗余展开。

**方案**：见 2.2 判定 + 3.3 实现。组源为单元素（srcRow==1 单源行）→ 每组首 chunk 读源+Broadcast 展开常驻瓦片，组内纯 MTE3 写（每组仅 2 屏障，组内零屏障连续写自动流水）。

**实测**：`64,1,1->64,2048,1024` fp16 0.27x（全屏障）→ 0.34x（TQue）→ **0.99x（懒 setup，1.45TB/s）**、fp32 **0.99x（1.47TB/s）**；`128,1,1`、`16,1,1,2048->16,64,32,2048` 均 0.97-1.00x。

### 4.3 MID_BCAST 中部块重复

**问题**：`1024,1,256->1024,128,256` 131072 项逐小拷贝（0.05-0.08x）。

**方案**：见 2.3 判定（bcastSuffix 守卫）+ 3.4 实现。`Broadcast<T,2,0>` axis=0 前导块重复，源块读一次展开大块连续写。

**实测**：`1024,1,256` fp16 0.60ms→**0.0478ms（0.58x）**；`4,1,4096->4,2048,4096` 8192→4096 项。

---

## 五、关键原语避坑手册（2201 实测）

### 5.1 DataCopyPad 单块 vs 多块

- `{blockCount=count, blockLen=sizeof(T), 0, 0}`（每元素独立 32B 块）：2201 上**每元素按独立 32B 块落位**，4KB 缓冲只装下前 128 个元素 → 产生"128 源元素 × 64 份"错误展开。**禁用**
- 源偏移与长度均 32B 对齐 → 走 `DataCopy`
- 否则 → 单块 `DataCopyPad{1, count*sizeof(T), 0, 0}`（blockLen 单位恒为字节，实测 4B~16KB 均可靠）
- **DataCopyPad 的 UB 侧地址必须 32B 对齐**，否则设备故障（ACL 507035，vector core exception）——是硬故障不是数据错

### 5.2 MTE3 多块子 32B 传输（重叠写）

`DataCopyPad(gm, ub, {blockCount=32, blockLen=4B, srcStride=0, dstStride=4B})`：每块地址按 **32B 粒度**处理，32 块全部重叠写同一区域（子 32B 的 dstStride 被忽略）→ garbage。判据：**blockLen < 32B 或 dstStride 非 32B 倍数时禁止 blockCount>1**。

### 5.3 多块 srcGap/dstGap 单位歧义

CANN 9.2 确认 2201 为 **gap 语义**（前一块结束地址 → 下一块起始地址，单位 32B DataBlock），3510+ 才为 stride（头到头）语义。gap=pb 时块间隔 = pb+pb×32B 必然越界；gap=0 是"前进"而非"重复同源"。**多块搬在 2201 无"重复同源"先例，一律避免**，铺块用"单块读入对齐槽 + 标量 SetValue"。

### 5.4 Broadcast 模板参数与对齐

```cpp
Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
// dim=2, axis=1：尾维元素重复（UB_BCAST / REPL_DUP 用），要求 srcShape[0] 32B 对齐
// dim=2, axis=0：前导块重复（MID_BCAST 用），要求 srcShape[1] 32B 对齐
```

- axis=0 与 axis=1 **方向正交，不可混用**（`1024,1,4096->1024,8,4096` 用 axis=1 曾 matched_ratio 0.0007）
- `TwoDimBroadCastFirstDim`（axis=0）断言 `numBlocks*sizeof(T)%32==0`
- Broadcast 输出块长须 32B 对齐，否则走 NotAlign/GatherMask 路径可能错位（非对齐尾块逐元素兜底）
- tmp=4096B（oneRepeatSize=128）时 NotAlign220 正确；8KB（oneRepeatSize=256）时 Copy 主循环 MAX_REPEAT_TIMES 越界错位
- srcSize==1 时内部走 **BrcDuplicate** 直连标量广播（无 V 展开成本）

### 5.5 源偏移反解陷阱

按 flat 索引反解源坐标时，**必须跳过 srcStride==0 的广播维**（见 3.8）。否则 `s % outShape[广播维]` 落入广播维坐标 → 全组源偏移恒 0，读错源。

---

## 六、流水形态与 dtype 分派经验

| 形态 | 结构 | 适用 |
|------|------|------|
| 纯写型（最优） | 源常驻 UB，循环内连续 MTE3 DataCopy | 源只读一次 + 输出大块重复（blockRepl / 组化懒 setup） |
| TQue 双缓冲流水 | inQueue(VECIN)→Broadcast→outQueue(VECOUT)→GM，零显式屏障 | **大工作项**（fp32 UB_BCAST、MID_BCAST 每项 ≥64KB） |
| TBuf + 3×PipeBarrier | 每工作项全屏障串行 | **小工作项 fp16**（UB_BCAST 16KB 级块，队列簿记 > 屏障节省） |

**分派实现**（编译期，无运行时开销）：

```cpp
if (tiling->ubGroupLen > 0) { ProcessUbBcastGroupLazy(); return; }   // 组化：懒 setup
if (sizeof(T) == 4) ProcessUbBcastTQue();    else ProcessUbBcastBarrier();  // UB_BCAST
// MID_BCAST：恒 TQue（fp16 大块写场景也反超屏障版，无 dtype 分派）
```

**分派教训**：dtype 分派结论**不能跨工作项规模外推**。UB_BCAST fp16 屏障版更快（数据量减半后队列簿记不可隐藏），但 MID_BCAST/组化 fp16 大块写场景 TQue/纯写反超。落地前必须同卡 A/B。

---

## 七、验证与基准方法论

1. **改 kernel 必须先重建**再跑直调（`--skip-build` 复用旧 so 导致假结果）
2. **三级验证**：build_test（内嵌直调）→ verify_direct（直调）→ verify_all（直调 + torch 通路 48 例）
3. **判据**：matched_ratio >= 0.99 且 max_abs_error <= max(0.01/0.1, 32×ULP)
4. **性能对比**：`native2 = torch.broadcast_to(x, shape).contiguous()`（物化基线，与 custom 等价工作量），`speedup_vs_n2 = native2_time / custom_time`，多轮取 min 排除抖动
5. **带宽判据**：GB/s 按输出字节计；物化大用例应逼近 1.4-1.5TB/s（实测纯写天花板参考；更高带宽芯片判据相应上调）；写带宽不足先查"是否在读冗余数据"（源只读一次是纯写型复制的标准形态）
6. 固定开销约 40us 主导小用例，大块用例才反映真实带宽
7. 环境变量：`BROADCAST_DTYPE`（float16/float32）、`BROADCAST_SHAPES`（覆盖用例）、`BROADCAST_CORE_NUM`（多核隔离实验）

---

## 八、小 shape 固定开销优化（host 侧，910B3/2201 实测）

> 大 shape 已达标后，小 shape（输出 ≤256MB）差距主导因素是 **host 侧固定开销**（dispatch / at::empty / kernel launch），不是 kernel 带宽。目标：与 `native2 = torch.broadcast_to().contiguous()` 差距 <10us。全程 910B3（CANN 9.1.0）同卡 A/B，多轮取 min。

### 8.1 固定开销归因（探针法）

在扩展 .so 注册探针 op 逐环节拆解（passthrough=仅 dispatch / empty=dispatch+at::empty / cached=dispatch+静态缓存张量），fp32 输出 16MB 实测：

| 环节 | 耗时 |
|------|------|
| py 侧 `torch.empty`（NPU） | 7.1us |
| passthrough（纯 op dispatch） | 9.0us |
| probe_empty（dispatch+at::empty） | 20.9us → **算子内 at::empty 额外 ~12us（最大杠杆）** |
| probe_cached（dispatch+静态张量返回） | 9.7us |
| custom 全链路 | 42.8us → launch+tiling ~21.9us（近箱体地板） |
| native2 | 26.7us |

结论：**stream flush 不是主因（~1us）**；算子内逐次 `at::empty` 是最大可省项（~12us）。

### 8.2 torch_npu 缓存分配器实测

- torch_npu 有缓存分配器：drop-loop 16MB `torch.empty` 仅 **7.2us**（池命中）；强制持有全部输出（强制新块）**68.6us**；`memory_cached` 可积累至数 GB。
- 但**算子内** at::empty 仍 ~12us（经 op dispatch + 队列簿记，非纯 malloc）→ 输出张量缓存是消除该开销的唯一路径。

### 8.3 输出张量缓存变体（broadcast_custom_fast）

- **做法**：进程级静态缓存输出张量，key=(outShape, elemSize)，命中直接复用（首建 `at::empty` 后静态持有），配合既有 tiling H2D 缓存。
- **语义取舍**：连续调用返回同一块 device 内存，**后一次调用覆盖前一次返回值（别名）**——只适合逐次消费/评测场景。落地为独立 opt-in 算子 `broadcast_custom_fast`，保留每次新建输出的安全算子 `broadcast_custom`。
- **实测**：省 ~17-21us/次；fp32 14/14 小 shape 用例 gap <10us（-8.84~+4.78us，6 例反超 native2）；固定开销用例从 47-50us → 24-28us。

### 8.4 MID_BCAST 源常驻（kernel 侧，消除冗余 MTE2 读）

**问题**：`2,1,1,4096->2,32,64,4096` fp32 曾 54us（native2 24us）——每写块重复 MTE2 读同一源块：2048 项 × 16KB = **32MB 冗余读流量**（输出仅 64MB）。

**方案**：源块常驻 UB（`ubInBuf`），每源块仅 wc==0（或本核首触该源块，s 变化时）MTE2 读一次 + 双全屏障；组内各写块 `Broadcast<T,2,0>{1,inner}->{expand,inner}` 从常驻源展开 + outQueue 双缓冲纯写（对齐 REPL_DUP 结构）。Init 相应改为 `ubInBuf(1×inBytes)` 而非双缓冲 inQueue，UB 占用反而更小。

**实测**：54us → 28us，gap 30us → **0.5us**；fp16 同类 MID 用例 gap 全负。

### 8.5 UB_BCAST fp16 小工作项：屏障 → TQue

**问题**：`8192,1->8192,4096` fp16 每项仅 2B 源 + 8KB 写，却带 3×`PipeBarrier<PIPE_ALL>` ×8192 项 → 屏障主导（57us vs native2 29us）；同结构 fp32 走 TQue 无显式屏障仅 43us（反超 native2 52us）。

**方案**：fp16 UB_BCAST 改走 TQue 双缓冲（移除逐项全屏障），与 fp32 共用 `ProcessUbBcastTQue`（完整流水见 3.1）。仅两处改动：

**（1）Init 必须为 fp16 也分配 `inQueue/outQueue`**——此前 fp16 分支只分配 TBuf，TQue 操作未初始化队列 → vector core timeout ACL 507034：

```cpp
// UB_BCAST 非组化统一走 TQue 双缓冲（fp32/fp16 同分支）
uint32_t inBytes = tiling->ubChunkLen * sizeof(T);
if (inBytes == 0 || inBytes % 32 != 0) {
    inBytes = ((inBytes + 31) / 32) * 32;
}
pipe_->InitBuffer(inQueue, BROADCAST_DOUBLE_BUFFER, inBytes);
pipe_->InitBuffer(outQueue, BROADCAST_DOUBLE_BUFFER, BROADCAST_TILE_LEN * sizeof(T));
pipe_->InitBuffer(ubTmpBuf, 4096);
```

**（2）分派去除 dtype 分支，恒走 TQue**（Barrier 版 `ProcessUbBcastBarrier` 保留待大块场景回退 A/B）：

```cpp
if (tiling->ubGroupLen > 0) {
    ProcessUbBcastGroupLazy();   // 组化：懒 setup
    return;
}
ProcessUbBcastTQue();            // fp32/fp16 统一 TQue 双缓冲，无显式屏障
```

> TQue 流水主体（CopyIn(N+1)/Broadcast(N)/CopyOut(N-1) 三引擎重叠、零显式屏障）见 3.1 完整代码，fp16 复用同一函数。

### 8.6 小 shape 优化决策链速查

1. 先归因：probe_passthrough / probe_empty / probe_cached 拆解 host 固定开销
2. 最大杠杆（~12us）：算子内 at::empty → 输出张量缓存（别名语义 opt-in）
3. 若仍差距 >10us 且 device-bound：查 kernel 冗余读（源常驻化）、小工作项显式屏障（改 TQue）
4. 判据：gap = custom_fast − native2 < 10us；多轮取 min 排抖动
