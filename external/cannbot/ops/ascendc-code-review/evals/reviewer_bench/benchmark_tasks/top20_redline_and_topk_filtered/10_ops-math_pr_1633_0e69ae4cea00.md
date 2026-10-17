# 代码检视报告

## 检视概览
- **仓库**: ops-math
- **PR编号**: 1633
- **PR作者**: wangxun21
- **代码文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h, conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h, conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simt.h
- **代码侧别**: Kernel侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 14 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 14 | 10 | 100% |

---

## 发现问题

### 文件: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h（Kernel侧）

#### [1] 人工检视意见

- **提出人**: llimwang
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h
- **行号**: 114
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: 索引越界写风险
  > 
  > 用户输入的 index 值 `dstIndex` 直接作为 `deDuplicateIndices` 的内存偏移，未校验是否在 `[0, maxIndex]` 范围内。若用户传入越界索引（负数或 > maxIndex），会导致越界写 workspace 内存，可能覆盖 writeBackIndices 数据或造成内存损坏。
  > 
  > **建议**: 在 `AtomicMax` 调用前增加范围校验：
  > ```cpp
  > if (dstIndex < 0 || dstIndex >= tilingData_->maxIndex + 1) {
  >     continue;
  > }
  > ```

- **代码片段**（行114）:
```cpp
 104 |         int64_t endIndex = tensorCumsumList[tensorIndex + 1] - tensorCumsumList[tensorIndex] - 1;
 105 |         if (tensorIndex == startTensorIndex) {
 106 |             startIndex = startOffset;
 107 |         }
 108 |         if (tensorIndex == endTensorIndex) {
 109 |             endIndex = endOffset;
 110 |         }
 111 | 
 112 |         for (int32_t index = startIndex + static_cast<int32_t>(Simt::GetThreadIdx<1>()); index <= endIndex;
 113 |              index += static_cast<int32_t>(Simt::GetThreadNum<1>())) {
 114 |             int32_t dstIndex = inputTensor[index];
 115 |             int32_t dstValue = tensorCumsumList[tensorIndex] + index;
 116 |             AtomicMax(deDuplicateIndices + dstIndex, dstValue);
 117 |         }
 118 |     }
 119 | }
 120 | 
 121 | __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void WriteBackIndices(
 122 |     __gm__ volatile int32_t *src, __gm__ int32_t *dst, int64_t startIndex, int64_t count, int64_t totalTensorSum)
 123 | {
```

---

#### [2] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h
- **行号**: 116
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_indices_deduplicate.h 第 116 行使用 `AtomicMax` 对 `deDuplicateIndices` 进行原子操作，但未检查 `dstIndex` 是否超出数组边界。如果 indices 中的值超出预期范围，可能导致内存越界访问。
  > 
  > **建议**: 在 AtomicMax 操作前添加边界检查。

- **代码片段**（行116）:
```cpp
 106 |             startIndex = startOffset;
 107 |         }
 108 |         if (tensorIndex == endTensorIndex) {
 109 |             endIndex = endOffset;
 110 |         }
 111 | 
 112 |         for (int32_t index = startIndex + static_cast<int32_t>(Simt::GetThreadIdx<1>()); index <= endIndex;
 113 |              index += static_cast<int32_t>(Simt::GetThreadNum<1>())) {
 114 |             int32_t dstIndex = inputTensor[index];
 115 |             int32_t dstValue = tensorCumsumList[tensorIndex] + index;
 116 |             AtomicMax(deDuplicateIndices + dstIndex, dstValue);
 117 |         }
 118 |     }
 119 | }
 120 | 
 121 | __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void WriteBackIndices(
 122 |     __gm__ volatile int32_t *src, __gm__ int32_t *dst, int64_t startIndex, int64_t count, int64_t totalTensorSum)
 123 | {
 124 |     for (int index = static_cast<int32_t>(Simt::GetThreadIdx<0>()); index < count;
 125 |          index += static_cast<int32_t>(Simt::GetThreadNum<0>())) {
```

---

#### [3] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h
- **行号**: 128
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_indices_deduplicate.h 第 124-131 行 WriteBackIndices 函数中第 128 行检查 `dstIndex >= 0 && dstIndex < totalTensorSum`，但未检查 `src[dstValue]` 的访问是否越界。
  > 
  > **建议**: 添加对 `dstValue` 的边界检查。

- **代码片段**（行128）:
```cpp
 118 |     }
 119 | }
 120 | 
 121 | __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void WriteBackIndices(
 122 |     __gm__ volatile int32_t *src, __gm__ int32_t *dst, int64_t startIndex, int64_t count, int64_t totalTensorSum)
 123 | {
 124 |     for (int index = static_cast<int32_t>(Simt::GetThreadIdx<0>()); index < count;
 125 |          index += static_cast<int32_t>(Simt::GetThreadNum<0>())) {
 126 |         int32_t dstValue = index + startIndex;
 127 |         int32_t dstIndex = src[dstValue];
 128 |         if (dstIndex >= 0 && dstIndex < totalTensorSum) {
 129 |             dst[dstIndex] = dstValue;
 130 |         }
 131 |     }
 132 | }
 133 | 
 134 | template <typename T>
 135 | __aicore__ inline void DynamicStitchIndicesDeDuplicate<T>::Process()
 136 | {
 137 |     if (blockIdx_ < tilingData_->clrBlockNum) {
```

---

#### [4] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h
- **行号**: 144
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_indices_deduplicate.h 第 144 行和 dynamic_stitch_scatter_simt.h 第 126 行使用 `for (int i = startTensorIndex_; i <= endTensorIndex_ + 1; i++)` 循环，当 `endTensorIndex_` 为 INT_MAX 时，`endTensorIndex_ + 1` 会导致整数溢出。
  > 
  > **建议**: 使用 `int64_t` 类型或添加溢出检查。

- **代码片段**（行144）:
```cpp
 134 | template <typename T>
 135 | __aicore__ inline void DynamicStitchIndicesDeDuplicate<T>::Process()
 136 | {
 137 |     if (blockIdx_ < tilingData_->clrBlockNum) {
 138 |         InitGlobalMemory(wsGm_, curClrBlockWsSize_, -1);
 139 |     }
 140 |     SyncAll();
 141 | 
 142 |     if (blockIdx_ < tilingData_->usedCoreNum) {
 143 |         LocalTensor<int64_t> tensorCumsumListLocalTensor = assistBuffer_.Get<int64_t>();
 144 |         for (int i = startTensorIndex_; i <= endTensorIndex_ + 1; i++) {
 145 |             tensorCumsumListLocalTensor.SetValue(i, tilingData_->tensorCumsumList[i]);
 146 |         }
 147 |         event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
 148 |         SetFlag<HardEvent::S_V>(eventIdSToV);
 149 |         WaitFlag<HardEvent::S_V>(eventIdSToV);
 150 |         Simt::VF_CALL<DeduplicateIndices>(Simt::Dim3{curTensorNum_, THREAD_NUM / curTensorNum_, 1},
 151 |             inTensorsPtr_,
 152 |             deDuplicateIndices_,
 153 |             startTensorIndex_,
```

---

### 文件: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h（Kernel侧）

#### [5] 人工检视意见

- **提出人**: llimwang
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h
- **行号**: 108
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: 索引上界未检查
  > 
  > SIMD Scatter 中仅检查了 `index < 0`，但未检查 `index > maxIndex`。虽然经过去重步骤后索引理论上有效，但缺少防御性检查，若去重结果异常可能导致输出 tensor 越界写。
  > 
  > **建议**: 增加上界检查：
  > ```cpp
  > if (index < 0 || index > tilingData_->maxIndex) {
  >     continue;
  > }
  > ```

- **代码片段**（行108）:
```cpp
  98 | 
  99 | template <typename T>
 100 | __aicore__ inline void DynamicStitchScatterSimd<T>::Scatter(GlobalTensor<T> inputDataGm, int64_t startOffset, int count)
 101 | {
 102 |     LocalTensor<int32_t> indices = indicesInQue_.DeQue<int32_t>();
 103 |     event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
 104 |     SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
 105 |     WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
 106 |     for (int i = 0; i < count; i++) {
 107 |         int32_t index = indices.GetValue(i);
 108 |         if (index < 0) {
 109 |             continue;
 110 |         }
 111 | 
 112 |         DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(tilingData_->ubFactor * sizeof(T)), 0, 0, 0};
 113 |         DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
 114 |         for (int64_t sliceLoop = 0; sliceLoop < tilingData_->ubLoopTimes; sliceLoop++) {
 115 |             LocalTensor<T> data = dataOutQue_.AllocTensor<T>();
 116 |             if (sliceLoop == (tilingData_->ubLoopTimes - 1)) {
 117 |                 dataCopyParams.blockLen = tilingData_->ubTailFactor * sizeof(T);
```

---


#### [6] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h
- **行号**: 106
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_scatter_simd.h 第 106 行和第 136 行使用 `int` 类型作为循环变量（`for (int i = 0; i < count; i++)`），但 `curTensorNum_` 和相关变量是 `int64_t` 类型。当数量超过 INT_MAX 时可能导致溢出。
  > 
  > **建议**: 统一使用 `int64_t` 类型作为循环变量。

- **代码片段**（行106）:
```cpp
  96 |     indicesInQue_.EnQue<int32_t>(indices);
  97 | }
  98 | 
  99 | template <typename T>
 100 | __aicore__ inline void DynamicStitchScatterSimd<T>::Scatter(GlobalTensor<T> inputDataGm, int64_t startOffset, int count)
 101 | {
 102 |     LocalTensor<int32_t> indices = indicesInQue_.DeQue<int32_t>();
 103 |     event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
 104 |     SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
 105 |     WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
 106 |     for (int i = 0; i < count; i++) {
 107 |         int32_t index = indices.GetValue(i);
 108 |         if (index < 0) {
 109 |             continue;
 110 |         }
 111 | 
 112 |         DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(tilingData_->ubFactor * sizeof(T)), 0, 0, 0};
 113 |         DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
 114 |         for (int64_t sliceLoop = 0; sliceLoop < tilingData_->ubLoopTimes; sliceLoop++) {
 115 |             LocalTensor<T> data = dataOutQue_.AllocTensor<T>();
```

---

#### [7] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h
- **行号**: 44
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_scatter_simd.h 第 44 行 Scatter 函数签名中参数 `int count` 使用 `int` 类型，但调用处传入的 `count` 可能是 `int64_t` 类型（第 151 行 `count = endOffset - startOffset + 1`），存在类型截断风险。
  > 
  > **建议**: 将参数类型改为 `int64_t count`。

- **代码片段**（行44）:
```cpp
  34 | template <typename T>
  35 | class DynamicStitchScatterSimd {
  36 | public:
  37 |     __aicore__ inline DynamicStitchScatterSimd(TPipe *pipe, const DynamicStitchTilingData *tiling)
  38 |         : pipe_(pipe), tilingData_(tiling){};
  39 |     __aicore__ inline void Init(GM_ADDR indices, GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
  40 |     __aicore__ inline void Process();
  41 | 
  42 | private:
  43 |     __aicore__ inline void CopyInIndices(int64_t startIndex, int64_t count);
  44 |     __aicore__ inline void Scatter(GlobalTensor<T> inputDataGm, int64_t startOffset, int count);
  45 |     
  46 | private:
  47 |     const DynamicStitchTilingData *tilingData_;
  48 |     TPipe *pipe_;
  49 | 
  50 |     TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> indicesInQue_;
  51 |     TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> dataOutQue_;
  52 | 
  53 |     GlobalTensor<int32_t> wsGm_;
```

---

#### [8] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h
- **行号**: 107
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_scatter_simd.h 第 107 行获取 `index` 后只检查 `index < 0` 就 continue，但未检查 `index` 是否超出输出 tensor 的边界。当 index 超出 y 的有效范围时可能导致内存越界。
  > 
  > **建议**: 添加上界检查 `index >= 0 && index < outputSize`。

- **代码片段**（行107）:
```cpp
  97 | }
  98 | 
  99 | template <typename T>
 100 | __aicore__ inline void DynamicStitchScatterSimd<T>::Scatter(GlobalTensor<T> inputDataGm, int64_t startOffset, int count)
 101 | {
 102 |     LocalTensor<int32_t> indices = indicesInQue_.DeQue<int32_t>();
 103 |     event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
 104 |     SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
 105 |     WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
 106 |     for (int i = 0; i < count; i++) {
 107 |         int32_t index = indices.GetValue(i);
 108 |         if (index < 0) {
 109 |             continue;
 110 |         }
 111 | 
 112 |         DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(tilingData_->ubFactor * sizeof(T)), 0, 0, 0};
 113 |         DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
 114 |         for (int64_t sliceLoop = 0; sliceLoop < tilingData_->ubLoopTimes; sliceLoop++) {
 115 |             LocalTensor<T> data = dataOutQue_.AllocTensor<T>();
 116 |             if (sliceLoop == (tilingData_->ubLoopTimes - 1)) {
```

---


### 文件: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simt.h（Kernel侧）


#### [9] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simt.h
- **行号**: 136
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: 第 136 行条件 `THREAD_NUM / (curTensorNum_ * sliceSize_) > 0` 可能在 curTensorNum_ * sliceSize_ 超过 THREAD_NUM 时结果为 0，但这种场景下应该仍使用 Simt::VF_CALL 执行，而非切换到备选分支。
  > 
  > **建议**: 重新评估此条件逻辑，确保在所有情况下都能正确选择执行路径。

- **代码片段**（行136）:
```cpp
 126 |         for (int i = startTensorIndex_; i <= endTensorIndex_ + 1; i++) {
 127 |             tensorCumsumListLocalTensor.SetValue(i, tilingData_->tensorCumsumList[i]);
 128 |         }
 129 |         event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
 130 |         SetFlag<HardEvent::S_V>(eventIdSToV);
 131 |         WaitFlag<HardEvent::S_V>(eventIdSToV);
 132 |         __gm__ T* yGmAddr = (__gm__ T*)yGm_.GetPhyAddr();
 133 |         __gm__ int32_t* workspaceGmAddr = (__gm__ int32_t*)workspaceGm_.GetPhyAddr();
 134 |         __ubuf__ int64_t* tensorCumsumAddr = (__ubuf__ int64_t*)tensorCumsumListLocalTensor.GetPhyAddr();
 135 | 
 136 |         if (THREAD_NUM / (curTensorNum_ * sliceSize_) > 0) {
 137 |             Simt::VF_CALL<SimtDataCopy<T>>(
 138 |                 Simt::Dim3{curTensorNum_, THREAD_NUM / (curTensorNum_ * sliceSize_), sliceSize_}, xGmAddr_,
 139 |                 startTensorIndex_, endTensorIndex_, startOffset_, endOffset_, tensorCumsumAddr, workspaceGmAddr,
 140 |                 yGmAddr, sliceSize_);
 141 |         } else {
 142 |             Simt::VF_CALL<SimtDataCopy<T>>(
 143 |                 Simt::Dim3{FIRST_DIM_THREAD_NUM, THREAD_NUM / (FIRST_DIM_THREAD_NUM * sliceSize_), sliceSize_},
 144 |                 xGmAddr_, startTensorIndex_, endTensorIndex_, startOffset_, endOffset_, tensorCumsumAddr,
 145 |                 workspaceGmAddr, yGmAddr, sliceSize_);
```

---

#### [10] 人工检视意见

- **提出人**: xieshengwei1024
- **作者**: wangxun21
- **文件**: conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simt.h
- **行号**: 84
- **评论时间**: 2026-03-17
- **Commit**: 0e69ae4cea00
- **问题描述**:

  > **问题**: dynamic_stitch_scatter_simt.h 第 84 行检查 `dstIndex < 0` 后 continue，但同样未检查上界。如果 dstIndex 超出 yGmAddr 的范围，可能导致越界写入。
  > 
  > **建议**: 添加上界检查。

- **代码片段**（行84）:
```cpp
  74 |         }
  75 | 
  76 |         if (tensorIndex == endTensorIndex) {
  77 |             curEndIndex = endOffset;
  78 |         }
  79 | 
  80 |         for (int32_t index = curStartIndex + static_cast<int32_t>(Simt::GetThreadIdx<1>()); index <= curEndIndex;
  81 |              index += static_cast<int32_t>(Simt::GetThreadNum<1>())) {
  82 |             int64_t workspaceIndex = tensorCumsum[tensorIndex] + index;
  83 |             int32_t dstIndex = workspaceGmAddr[workspaceIndex];
  84 |             if (dstIndex < 0) {
  85 |                 continue;
  86 |             }
  87 | 
  88 |             for (int32_t sliceIndex = static_cast<int32_t>(Simt::GetThreadIdx<2>()); sliceIndex < sliceSize;
  89 |                  sliceIndex += static_cast<int32_t>(Simt::GetThreadNum<2>())) {
  90 |                 int64_t yGmBaseIndex = dstIndex * sliceSize;
  91 |                 int64_t xGmBaseIndex = index * sliceSize;
  92 |                 yGmAddr[yGmBaseIndex + sliceIndex] = inputTensor[xGmBaseIndex + sliceIndex];
  93 |             }
```

---

## 被检视代码

> 本报告基于 PR 1633 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_indices_deduplicate.h`
- `conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simd.h`
- `conversion/dynamic_stitch/op_kernel/arch35/dynamic_stitch_scatter_simt.h`
