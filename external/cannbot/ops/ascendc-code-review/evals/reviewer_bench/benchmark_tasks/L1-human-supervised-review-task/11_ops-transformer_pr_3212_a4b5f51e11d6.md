# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 3212
- **PR作者**: jisongyuan@h-partners.com
- **代码文件**: attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp, attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_nhsd.h
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 12 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 12 | 11 | 100% |

---

## 发现问题

### 文件: attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp（Tiling侧）

---

#### [3] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: jisongyuan@h-partners.com
- **文件**: attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp
- **行号**: 394
- **评论时间**: 2026-03-27
- **Commit**: a4b5f51e11d6
- **问题描述**:

  > 这里检查的是`inputKeyCacheInShape_.GetDim(DIM_1)`，即第1维（index从0开始），但错误信息写的是"dim2"。cache的layout是[numBlocks, numHead, blockSize, headSize]，numHead在DIM_1。建议把日志改成"dim1 of keyCache should be same as numHead"。

- **代码片段**（行394）:
```cpp
 384 |     params_.numHead = inputKeyShape_.GetDim(DIM_1);
 385 |     params_.kHeadSize = inputKeyShape_.GetDim(DIM_2);
 386 |     params_.vHeadSize = inputValueShape_.GetDim(DIM_2);
 387 |     int64_t numBlocks = inputKeyCacheInShape_.GetDim(DIM_0);
 388 |     params_.blockSize = inputKeyCacheInShape_.GetDim(DIM_2);
 389 |     bool isAlign = ((params_.kHeadSize * params_.typeByteK) % ALIGN == 0 &&
 390 |                         (params_.vHeadSize * params_.typeByteV) % ALIGN == 0);
 391 |     OP_CHECK_IF((!isAlign), OP_LOGE(context_, "kHeadSize and vHeadSize should be align to 32."),
 392 |                 return ge::GRAPH_FAILED);
 393 |     OP_CHECK_IF((params_.numHead != inputKeyCacheInShape_.GetDim(DIM_1)),
 394 |                 OP_LOGE(context_, "dim2 of keyCache should be same as numHead."), return ge::GRAPH_FAILED);
 395 |     OP_CHECK_IF((params_.numHead > NUM_HEAD_MAX),
 396 |                 OP_LOGE(context_, "num head must less than 4095."), return ge::GRAPH_FAILED);
 397 |     OP_CHECK_IF((static_cast<uint64_t>(numBlocks) * params_.blockSize < params_.numTokens),
 398 |                 OP_LOGE(context_, "numBlocks * blockSize should larger than numTokens."), return ge::GRAPH_FAILED);
 399 |     OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_2) != inputValueCacheInShape_.GetDim(DIM_2)),
 400 |                 OP_LOGE(context_, "dim2 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
 401 |     OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_1) != inputValueCacheInShape_.GetDim(DIM_1)),
 402 |                 OP_LOGE(context_, "dim1 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
 403 |     OP_CHECK_IF((inputKeyShape_.GetDim(DIM_0) != inputValueShape_.GetDim(DIM_0)),
```

---

#### [4] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: jisongyuan@h-partners.com
- **文件**: attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp
- **行号**: 297
- **评论时间**: 2026-03-27
- **Commit**: a4b5f51e11d6
- **问题描述**:

  > `CheckInputDimNumNHSD`里的5条错误信息都有"should be is"的语法错误，多了一个"is"。比如"slot_mapping should be is 1 dim"应改为"slot_mapping should be 1 dim"，其他几条同理。

- **代码片段**（行297）:
```cpp
 287 |     return ge::GRAPH_SUCCESS;
 288 | }
 289 | 
 290 | ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDimNumNHSD()
 291 | {
 292 |     size_t kDimNum = inputKeyShape_.GetDimNum();
 293 |     size_t kCacheDimNum = inputKeyCacheInShape_.GetDimNum();
 294 |     size_t slotDimNum = slotMappingShape_.GetDimNum();
 295 |     size_t vDimNum = inputValueShape_.GetDimNum();
 296 |     size_t vCacheDimNum = inputValueCacheInShape_.GetDimNum();
 297 |     OP_CHECK_IF((slotDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "slot_mapping should be is 1 dim."),
 298 |                 return ge::GRAPH_FAILED);
 299 |     OP_CHECK_IF((kCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "key_cache should be is 4 dim."),
 300 |                 return ge::GRAPH_FAILED);
 301 |     OP_CHECK_IF((kDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "key should be is 3 dim."),
 302 |                 return ge::GRAPH_FAILED);
 303 |     OP_CHECK_IF((vCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "value_cache should be is 4 dim."),
 304 |                 return ge::GRAPH_FAILED);
 305 |     OP_CHECK_IF((vDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "value should be is 3 dim."),
 306 |                 return ge::GRAPH_FAILED);
```

---

#### [5] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: jisongyuan@h-partners.com
- **文件**: attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp
- **行号**: 396
- **评论时间**: 2026-03-27
- **Commit**: a4b5f51e11d6
- **问题描述**:

  > 代码检查条件是`params_.numHead > NUM_HEAD_MAX`（即numHead=4095是允许的），但日志写的是"must less than 4095"暗示4095不允许。建议改成"num head must not exceed 4095"或"must be less than or equal to 4095"。

- **代码片段**（行396）:
```cpp
 386 |     params_.vHeadSize = inputValueShape_.GetDim(DIM_2);
 387 |     int64_t numBlocks = inputKeyCacheInShape_.GetDim(DIM_0);
 388 |     params_.blockSize = inputKeyCacheInShape_.GetDim(DIM_2);
 389 |     bool isAlign = ((params_.kHeadSize * params_.typeByteK) % ALIGN == 0 &&
 390 |                         (params_.vHeadSize * params_.typeByteV) % ALIGN == 0);
 391 |     OP_CHECK_IF((!isAlign), OP_LOGE(context_, "kHeadSize and vHeadSize should be align to 32."),
 392 |                 return ge::GRAPH_FAILED);
 393 |     OP_CHECK_IF((params_.numHead != inputKeyCacheInShape_.GetDim(DIM_1)),
 394 |                 OP_LOGE(context_, "dim2 of keyCache should be same as numHead."), return ge::GRAPH_FAILED);
 395 |     OP_CHECK_IF((params_.numHead > NUM_HEAD_MAX),
 396 |                 OP_LOGE(context_, "num head must less than 4095."), return ge::GRAPH_FAILED);
 397 |     OP_CHECK_IF((static_cast<uint64_t>(numBlocks) * params_.blockSize < params_.numTokens),
 398 |                 OP_LOGE(context_, "numBlocks * blockSize should larger than numTokens."), return ge::GRAPH_FAILED);
 399 |     OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_2) != inputValueCacheInShape_.GetDim(DIM_2)),
 400 |                 OP_LOGE(context_, "dim2 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
 401 |     OP_CHECK_IF((inputKeyCacheInShape_.GetDim(DIM_1) != inputValueCacheInShape_.GetDim(DIM_1)),
 402 |                 OP_LOGE(context_, "dim1 of keyCache should be same as ValueCache."), return ge::GRAPH_FAILED);
 403 |     OP_CHECK_IF((inputKeyShape_.GetDim(DIM_0) != inputValueShape_.GetDim(DIM_0)),
 404 |                 OP_LOGE(context_, "dim0 of key should be same as Value."), return ge::GRAPH_FAILED);
 405 |     OP_CHECK_IF((inputKeyShape_.GetDim(DIM_1) != inputValueShape_.GetDim(DIM_1)),
```

---

### 文件: attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_nhsd.h（Kernel侧）

---

#### [10] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: jisongyuan@h-partners.com
- **文件**: attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_nhsd.h
- **行号**: 11
- **评论时间**: 2026-03-27
- **Commit**: a4b5f51e11d6
- **问题描述**:

  > 文件头注释中的文件名写的是`scatter_pa_kv_cache_normal.h`，但实际文件名是`scatter_pa_kv_cache_nhsd.h`。建议更正为实际文件名。

- **代码片段**（行11）:
```cpp
   1 | /**
   2 |  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
   3 |  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
   4 |  * CANN Open Software License Agreement Version 2.0 (the "License").
   5 |  * Please refer to the License for details. You may not use this file except in compliance with the License.
   6 |  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   7 |  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   8 |  * See LICENSE in the root of the software repository for the full text of the License.
   9 |  */
  10 | /*!
  11 |  * \file scatter_pa_kv_cache_normal.h
  12 |  * \brief
  13 |  */
  14 | 
  15 | #ifndef ASCEND_SCATTER_PA_KV_CACHE_NHSD_H
  16 | #define ASCEND_SCATTER_PA_KV_CACHE_NHSD_H
  17 | 
  18 | #include "kernel_operator.h"
  19 | #include "op_kernel/platform_util.h"
  20 | 
```

---

## 被检视代码

> 本报告基于 PR 3212 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling.cpp`
- `attention/scatter_pa_kv_cache/op_kernel/scatter_pa_kv_cache_nhsd.h`
