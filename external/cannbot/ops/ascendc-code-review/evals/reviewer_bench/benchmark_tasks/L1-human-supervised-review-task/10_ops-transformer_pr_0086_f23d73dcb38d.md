# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 86
- **PR作者**: yyoean
- **代码文件**: 5 个文件
- **代码侧别**: Host侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 11 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 11 | 10 | 100% |

---

## 发现问题


---

### 文件: mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h（Kernel侧）

#### [2] 人工检视意见

- **提出人**: liuboxi
- **作者**: yyoean
- **文件**: mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h
- **行号**: 170
- **评论时间**: 2025-10-20
- **Commit**: f23d73dcb38d
- **问题描述**:

  > 变量命名风格保持整个代码统一，类成员变量使用下划线结尾。

- **代码片段**（行170）:
```cpp
 160 |     uint32_t bsKAlign_{0};
 161 |     uint32_t startRankId_{0};
 162 |     uint32_t endRankId_{0};
 163 |     uint32_t sendRankNum_{0};
 164 |     uint32_t halfWinSize_{0};
 165 |     uint32_t dataSpaceSize_{0};
 166 |     uint32_t bufferId_{0};
 167 |     uint32_t tokenNumPerCore_{0};
 168 |     uint32_t tokenIndex_{0};
 169 |     uint32_t waitCostSize_{0};
 170 |     bool isWaitCost=false;
 171 | 
 172 |     TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moeQueue_;
 173 |     TBuf<> expertIdsBuf_;
 174 |     TBuf<> expandScalesBuf_;
 175 |     TBuf<> rowTmpFloatBuf_;
 176 |     TBuf<> sumFloatBuf_;
 177 |     TBuf<> sendCountBuf_;
 178 |     TBuf<> indexCountsBuf_;
 179 |     TBuf<> tokenBuf_;
```

---

#### [3] 人工检视意见

- **提出人**: liuboxi
- **作者**: yyoean
- **文件**: mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h
- **行号**: 281
- **评论时间**: 2025-10-20
- **Commit**: f23d73dcb38d
- **问题描述**:

  > 2为魔鬼数字，含义不明确，使用sizeof(uint64_t) / sizeof(int32_t)替代

- **代码片段**（行281）:
```cpp
 271 |     tpipe_->InitBuffer(tokenBuf_, axisHExpandXTypeSize_);                    // 7168 * 2 = 14336
 272 |     tpipe_->InitBuffer(rowTmpFloatBuf_, axisHFloatSize_);                    // 7168 * 4 = 28672
 273 |     tpipe_->InitBuffer(sumFloatBuf_, axisHFloatSize_);                       // 7168 * 4 = 28672
 274 |     tpipe_->InitBuffer(sendCountBuf_, RoundUp(moeExpertNum_, B32_PER_BLOCK) * sizeof(int32_t));
 275 |     tpipe_->InitBuffer(indexCountsBuf_, Std::max(expertIdsBufSize, REPEAT_BYTES));  // 32 * 8 * 4 = 1024
 276 |     tpipe_->InitBuffer(batchWriteItemBuf_, BATCH_WRITE_ITEM_SIZE * worldSize_);
 277 |     if (isWaitCost) {
 278 |         tpipe_->InitBuffer(waitCostBuf_, waitCostSize_ * sizeof(uint64_t));
 279 |         waitCostU64Tensor_ = waitCostBuf_.Get<uint64_t>();
 280 |         waitCostU32Tensor_ = waitCostU64Tensor_.template ReinterpretCast<int32_t>();
 281 |         Duplicate<int32_t>(waitCostU32Tensor_, 0, waitCostSize_ * 2);
 282 |     }
 283 |     batchWriteItemLocalB64 = batchWriteItemBuf_.Get<uint64_t>();
 284 |     batchWriteItemLocalB32 = batchWriteItemLocalB64.template ReinterpretCast<uint32_t>();
 285 | }
 286 | 
 287 | template <TemplateMC2TypeA2Class>
 288 | __aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::TokenActiveMaskCal()
 289 | {
 290 |     LocalTensor<int8_t> xActiveMaskInt8Tensor;
```

---

#### [4] 人工检视意见

- **提出人**: liuboxi
- **作者**: yyoean
- **文件**: mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h
- **行号**: 488
- **评论时间**: 2025-10-20
- **Commit**: f23d73dcb38d
- **问题描述**:

  > 50为魔鬼数字，需要定义常量表示时钟周期

- **代码片段**（行488）:
```cpp
 478 | }
 479 | 
 480 | template <TemplateMC2TypeA2Class>
 481 | __aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::WaitDispatch()
 482 | {
 483 |     if (startRankId_ >= worldSize_) {
 484 |         SyncAll<true>();
 485 |         return;
 486 |     }
 487 |     SyncFunc<AscendC::HardEvent::MTE2_S>();
 488 |     auto start = GetSystemCycle() / 50;
 489 |     for (uint32_t waitFlagNum = 0; waitFlagNum < sendRankNum_;) {
 490 |         waitFlagNum = 0;
 491 |         for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
 492 |             uint32_t tokenIdx = (rankId + 1) * localMoeExpertNum_ - 1;
 493 |             GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET +
 494 |                             (recvCountLocal_(tokenIdx) + expertWindowOffsetLocal_(tokenIdx)) * axisHExpandXTypeSize_;
 495 |             flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
 496 |             DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
 497 |                 flagGlobal_);
```

---

### 文件: mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h（Kernel侧）

#### [7] 人工检视意见

- **提出人**: liuboxi
- **作者**: yyoean
- **文件**: mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h
- **行号**: 338
- **评论时间**: 2025-10-20
- **Commit**: f23d73dcb38d
- **问题描述**:

  > idWaitCost在350才被真正赋值，这里只会使用初始化的值False判断，339行永远不会执行。

- **代码片段**（行338）:
```cpp
 328 | __aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Init(
 329 |     GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR waitCost, GM_ADDR XOut,
 330 |     GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData, GM_ADDR contextGM)
 331 | {
 332 |     tpipe_ = pipe;
 333 |     expandXGM_ = expandX;
 334 |     expertIdsGM_ = expertIds;
 335 |     expandIdxGM_ = expandIdx;
 336 |     sendCountGM_ = sendCount;
 337 |     scalesGM_ = scales;
 338 |     if (isWaitCost) {
 339 |         waitCostGM_ = waitCost;
 340 |     }
 341 |     XOutGM_ = XOut;
 342 |     rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
 343 |     axisBS_ = tilingData->moeDistributeCombineInfo.bs;
 344 |     axisH_ = tilingData->moeDistributeCombineInfo.h;
 345 |     axisK_ = tilingData->moeDistributeCombineInfo.k;
 346 |     aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
 347 |     moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
```

---

#### [10] 人工检视意见

- **提出人**: liuboxi
- **作者**: yyoean
- **文件**: mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.h
- **行号**: 739
- **评论时间**: 2025-10-20
- **Commit**: f23d73dcb38d
- **问题描述**:

  > 该行代码被注释，打点duration并未使用，输出结果错误。

- **代码片段**（行739）:
```cpp
 729 |         sumOfFlag = statusSumOutTensor.GetValue(0);
 730 |     }
 731 |     
 732 |     auto end = GetSystemCycle() / 50; //zly
 733 |     auto duration = end - start; //zly
 734 |     //auto curServerId = rankId_ / SERVER_RANK_SIZE;
 735 |     //auto id = curServerId * SERVER_RANK_SIZE + destRankIdx;
 736 |     
 737 |     if (isWaitCost){
 738 |         Duplicate<int32_t>(waitCostU32Tensor_, 0, waitCostSize_ * 2);
 739 |         //waitCostU32Tensor_.SetValue(id*2, duration);
 740 | 	    AscendC::SetAtomicAdd<int32_t>();
 741 | 	    AscendC::DataCopy(waitCostU32GMTensor_, waitCostU32Tensor_, waitCostSize_*2);
 742 |         AscendC::SetAtomicNone();
 743 |     }
 744 | 
 745 |     SyncAll<true>();
 746 | }
 747 | 
 748 | template <TemplateMC2TypeClass>
```

---

## 被检视代码

> 本报告基于 PR 86 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `mc2/moe_distribute_combine/op_graph/moe_distribute_combine_proto.h`
- `mc2/moe_distribute_combine/op_kernel/moe_distribute_combine.h`
- `mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h`
- `mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h`
- `mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.h`
