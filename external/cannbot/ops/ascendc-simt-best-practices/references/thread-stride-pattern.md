# 线程排布统一编程模式

## 模式一：全局 grid-stride

SIMT 算子最基本的线程排布 for 循环模式，每个线程以全局 stride 间隔遍历数据：

```cpp
for (uint64_t index = (uint64_t)(
         AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx());
     index < count;
     index += (AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
    // 每个线程处理 stride 间隔的元素
}
```

## 模式二：核内分块 + 线程步进

采用**核内分块**：tiling 侧把输出按连续段分给各核（每核段长 `perCoreElements`、末核段长 `lastCoreElements`；分核与对齐见 [`ascendc-simt-tiling-design`](../../ascendc-simt-tiling-design/SKILL.md)），核内线程再在该段内 stride 步进。

```cpp
// kernel 侧：每核认领连续一段 [yIndexBase, yIndexBase + currentCoreElements)
// yIndexBase = blockIdx * perCoreElements；末核段长改用 lastCoreElements
// 线程索引可用 AscendC::Simt::GetThreadIdx()/GetThreadNum()，或内置 threadIdx.x/blockDim.x
for (INDEX_T index = threadIdx.x; index < currentCoreElements; index += blockDim.x) {
    INDEX_T yIndex = yIndexBase + index;      // 全局输出下标 = 核段基址 + 段内偏移
    // ... 坐标分解与 gather
}
```

- **核间**：每核处理一段**连续**数据，地址局部性好，无需全局大数乘法。
- **核内**：步进 `blockDim.x`（核内线程数），**不是** `threadNum × blockNum` 的全局 stride。
- **对齐与下限**：段长对齐、每核下限等 tiling 细节见 [`ascendc-simt-tiling-design`](../../ascendc-simt-tiling-design/SKILL.md)。
- **优于模式一之处**：核间连续访问，DCache 局部性好；末核以 `lastCoreElements` 单独收尾，负载均匀可控。

> 选型：规则逐元素、无跨核副作用 → 模式一；大 shape、需核间连续性或负载均匀 → 模式二。

## 索引计算 API

| API | 说明 |
|-----|------|
| `AscendC::Simt::GetThreadIdx()` | 当前线程索引 |
| `AscendC::Simt::GetThreadNum()` | 纯线程总数 |
| `AscendC::Simt::GetBlockIdx()` | 当前核索引 |
| `AscendC::Simt::GetBlockNum()` | 核总数 |

## 索引计算分解

```
模式一（全局）：初始 index = blockIdx * threadNum + threadIdx
                步进 stride = threadNum * blockNum
模式二（核内）：初始 yIndex = blockIdx * perCoreElements + threadIdx
                步进 stride = blockDim.x（仅核内线程数）
```

- 模式一初始 index：将线程在全局中的起始位置线性化；步进为「所有核所有线程总数」，确保每个元素只被一个线程处理。
- 模式二初始 yIndex：核段基址 + 段内偏移；步进仅核内线程数，核间已由 tiling 分段。

## 三种"线程↔数据"映射粒度

| 映射 | 形态 | 适用 | 代表算子 |
|------|------|------|---------|
| 一线程一元素 | 线程=输出元素，坐标分解后 gather/scatter | 逐元素、离散访存 | gather、scatter、pooling 前向 |
| 一线程一段/一行 | 线程独占一段，段内串行（段内顺序依赖时即扫描线） | 行级归约、前缀扫描 | segment_sum、embedding_bag、cummin、ctc_loss |
| 每 warp 一个输出 | lane 分摊规约轴 + shuffle | 规约类 | cosine_similarity、trace |

```cpp
// ① 一线程一元素 + 坐标分解（二维 gather，SIMT 最典型形态）
//    语义：output[r][c] = src[indices[r]][c]，数据视为 [gatherDim, innerSize]
for (INDEX_T index = threadIdx.x; index < currentCoreElements; index += blockDim.x) {
    INDEX_T yIndex  = yIndexBase + index;                    // 核内段基址 + 段内偏移 = 全局输出下标
    INDEX_T gatherI = Simt::UintDiv(yIndex, magic, shift);   // yIndex / innerSize → 行号（magic 快除）
    INDEX_T innerI  = yIndex - gatherI * innerSize;          // yIndex % innerSize → 列号（乘减代替取模）
    y[yIndex] = x[indices[gatherI] * innerSize + innerI];    // 间接寻址：源行起始 + 行内偏移
}

// ② 一线程一段/一行（segment_sum：threadIdx.y=段、threadIdx.x=列，段内串行累加，段内无顺序依赖）
//    该形态通常配 2D 线程块：dim3{列数, 段数}——x 维走可并行的连续列，y 维走需串行的段
asc_vf_call<Compute<TX, Index>>(dim3{threadNum, threadBlock}, ...);
for (uint32_t i = threadIdx.y; i < uniqueIdNum; i += blockDim.y) {
    TX res = 0;
    for (int32_t j = uniqueIdPosAddr[i]; j < nextIdPos; j++) {   // 段内串行累加
        res += xAddr[(baseOffset + j) * colSize + threadIdx.x];
    }
    outputAddr[outputOffset] = res;      // 中间段直接写；首/尾段才 asc_atomic_add
}

// ③ 每 warp 一个输出（一个 warp 协作完成一个输出元素的规约）
//    粒度：外层循环变量是"warp 级"的，一次迭代 32 个 lane 共同产出 1 个输出元素
const uint32_t laneId   = threadIdx.x % warpSize;      // 本线程在 warp 内的编号 0~31
const uint32_t warpId   = threadIdx.x / warpSize;      // 本线程所属 warp 在本核内的编号
const uint32_t numWarps = blockDim.x / warpSize;
for (IdxT outIdx = blockIdx.x * numWarps + warpId;      // 每个 warp 认领一个输出元素
     outIdx < totalOutputElements;
     outIdx += gridDim.x * numWarps) {                  // 再按全网格 warp 数步进（grid-stride）
    float acc = 0.0f;
    // 32 个 lane 分摊规约轴：lane0 算 r=0,32,64...；lane1 算 r=1,33,65...
    for (IdxT r = laneId; r < reduceSize; r += warpSize) {
        acc += x1[baseOffsetX1 + r * reduceStride] * x2[baseOffsetX2 + r * reduceStride];
    }
    acc = asc_reduce_add(acc);                          // warp 内树形归约（约 5 步 shuffle）
    // 归约后 warp 内所有 lane 都拿到相同结果，因此只需 lane0 落盘一次
    if (laneId == 0) { output[outIdx] = (T)acc; }
}
```

## Select 算子示例

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(2048) inline void OpSelectSimt(
    int32_t needCoreNum, int64_t perCoreElements, int64_t lastCoreElements,
    __gm__ uint8_t* condition, __gm__ T* x1, __gm__ T* x2, __gm__ T* y) {
    // 模式二：每核认领连续一段 [yIndexBase, yIndexBase + currentCoreElements)
    uint64_t yIndexBase = AscendC::Simt::GetBlockIdx() * (uint64_t)perCoreElements;
    uint64_t currentCoreElements = (AscendC::Simt::GetBlockIdx() == (uint64_t)(needCoreNum - 1))
        ? (uint64_t)lastCoreElements : (uint64_t)perCoreElements;
    for (uint64_t index = threadIdx.x; index < currentCoreElements; index += blockDim.x) {
        uint64_t yIndex = yIndexBase + index;   // 全局输出下标 = 核段基址 + 段内偏移
        y[yIndex] = condition[yIndex] ? x1[yIndex] : x2[yIndex];
    }
}
```