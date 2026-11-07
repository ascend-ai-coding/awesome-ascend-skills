# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 4699
- **PR作者**: tang-hao-hw-gitcode
- **代码文件**: 5 个文件
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 69 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 69 | 54 | 100% |

---

## 发现问题

### 文件: attention/common/op_host/split_core_v2.cpp（Tiling侧）

---


#### [2] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_host/split_core_v2.cpp
- **行号**: 742
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】批量的注释代码需要删除

- **代码片段**（行742）:
```cpp
 732 |     for (uint32_t i = minCore; i <= maxCore; ++i) {
 733 |         CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
 734 |         if (tmpResult.maxCost < result.maxCost) {
 735 |             CopyTmpResult(tmpResult, result);
 736 |         }
 737 |         ClearTmpResult(tmpResult);
 738 |     }
 739 | 
 740 |     // 判断是否要开FD
 741 |     //  splitContext.splitParam.streamK = false;
 742 |     //  CalcSplitPlan(result.usedCoreNum, INT64_MAX, splitContext, tmpResult);
 743 |     //  uint32_t gapCoreNum = 27U; // 实验值
 744 |     //  uint32_t gapBlockNum = 9U; // 实验计算值： 9 * BlockCost(64,256) = 10 * fdblockCost
 745 |     //  uint32_t maxFdSize = *std::max_element(result.fdRes.mSize.begin(), result.fdRes.mSize.end());
 746 | 
 747 |     // if (result.usedCoreNum - tmpResult.usedCoreNum < gapCoreNum){
 748 |     //     if (tmpResult.maxCost - result.maxCost < gapBlockNum *
 749 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 750 |     //         CopyTmpResult(tmpResult,result);
 751 |     //     }
```

---

#### [3] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_host/split_core_v2.cpp
- **行号**: 744
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】批量注释代码需要删除：//  uint32_t gapCoreNum = 27U; // 实验值

- **代码片段**（行744）:
```cpp
 734 |         if (tmpResult.maxCost < result.maxCost) {
 735 |             CopyTmpResult(tmpResult, result);
 736 |         }
 737 |         ClearTmpResult(tmpResult);
 738 |     }
 739 | 
 740 |     // 判断是否要开FD
 741 |     //  splitContext.splitParam.streamK = false;
 742 |     //  CalcSplitPlan(result.usedCoreNum, INT64_MAX, splitContext, tmpResult);
 743 |     //  uint32_t gapCoreNum = 27U; // 实验值
 744 |     //  uint32_t gapBlockNum = 9U; // 实验计算值： 9 * BlockCost(64,256) = 10 * fdblockCost
 745 |     //  uint32_t maxFdSize = *std::max_element(result.fdRes.mSize.begin(), result.fdRes.mSize.end());
 746 | 
 747 |     // if (result.usedCoreNum - tmpResult.usedCoreNum < gapCoreNum){
 748 |     //     if (tmpResult.maxCost - result.maxCost < gapBlockNum *
 749 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 750 |     //         CopyTmpResult(tmpResult,result);
 751 |     //     }
 752 |     // }
 753 |     // else {
```

---

#### [4] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_host/split_core_v2.cpp
- **行号**: 746
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】批量的注释代码需要删除

- **代码片段**（行746）:
```cpp
 736 |         }
 737 |         ClearTmpResult(tmpResult);
 738 |     }
 739 | 
 740 |     // 判断是否要开FD
 741 |     //  splitContext.splitParam.streamK = false;
 742 |     //  CalcSplitPlan(result.usedCoreNum, INT64_MAX, splitContext, tmpResult);
 743 |     //  uint32_t gapCoreNum = 27U; // 实验值
 744 |     //  uint32_t gapBlockNum = 9U; // 实验计算值： 9 * BlockCost(64,256) = 10 * fdblockCost
 745 |     //  uint32_t maxFdSize = *std::max_element(result.fdRes.mSize.begin(), result.fdRes.mSize.end());
 746 | 
 747 |     // if (result.usedCoreNum - tmpResult.usedCoreNum < gapCoreNum){
 748 |     //     if (tmpResult.maxCost - result.maxCost < gapBlockNum *
 749 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 750 |     //         CopyTmpResult(tmpResult,result);
 751 |     //     }
 752 |     // }
 753 |     // else {
 754 |     //     if (tmpResult.maxCost - result.maxCost < (gapBlockNum + 1) *
 755 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
```

---

#### [5] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_host/split_core_v2.cpp
- **行号**: 751
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】批量的注释代码需要删除： //         CopyTmpResult(tmpResult,result);

- **代码片段**（行751）:
```cpp
 741 |     //  splitContext.splitParam.streamK = false;
 742 |     //  CalcSplitPlan(result.usedCoreNum, INT64_MAX, splitContext, tmpResult);
 743 |     //  uint32_t gapCoreNum = 27U; // 实验值
 744 |     //  uint32_t gapBlockNum = 9U; // 实验计算值： 9 * BlockCost(64,256) = 10 * fdblockCost
 745 |     //  uint32_t maxFdSize = *std::max_element(result.fdRes.mSize.begin(), result.fdRes.mSize.end());
 746 | 
 747 |     // if (result.usedCoreNum - tmpResult.usedCoreNum < gapCoreNum){
 748 |     //     if (tmpResult.maxCost - result.maxCost < gapBlockNum *
 749 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 750 |     //         CopyTmpResult(tmpResult,result);
 751 |     //     }
 752 |     // }
 753 |     // else {
 754 |     //     if (tmpResult.maxCost - result.maxCost < (gapBlockNum + 1) *
 755 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 756 |     //         CopyTmpResult(tmpResult,result);
 757 |     //     }
 758 |     // }
 759 | 
 760 |     // 3、存在FD任务，对FD进行负载均衡分配
```

---

#### [6] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_host/split_core_v2.cpp
- **行号**: 755
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效分支请删除

- **代码片段**（行755）:
```cpp
 745 |     //  uint32_t maxFdSize = *std::max_element(result.fdRes.mSize.begin(), result.fdRes.mSize.end());
 746 | 
 747 |     // if (result.usedCoreNum - tmpResult.usedCoreNum < gapCoreNum){
 748 |     //     if (tmpResult.maxCost - result.maxCost < gapBlockNum *
 749 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 750 |     //         CopyTmpResult(tmpResult,result);
 751 |     //     }
 752 |     // }
 753 |     // else {
 754 |     //     if (tmpResult.maxCost - result.maxCost < (gapBlockNum + 1) *
 755 |     //     CalcCost(std::min(maxFdSize,param.mBaseSize),param.s2BaseSize)){
 756 |     //         CopyTmpResult(tmpResult,result);
 757 |     //     }
 758 |     // }
 759 | 
 760 |     // 3、存在FD任务，对FD进行负载均衡分配
 761 |     if (result.fdRes.fdNum > 0U) {
 762 |         SplitFD(result);
 763 |     }
 764 | 
```

---


### 文件: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h（Kernel侧）

---


#### [8] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 181
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【提示】1024多次出现，可以提取宏

- **代码片段**（行181）:
```cpp
 171 |             l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
 172 |             l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
 173 |             l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
 174 |         }
 175 | 
 176 |         // L0A B C 当前写死，能否通过基础api获取
 177 |         l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
 178 |         l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
 179 |         l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
 180 |         // L0A B C当前写死，要改成通过计算获取
 181 |         mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
 182 |         mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
 183 | 
 184 |         if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
 185 |                       mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
 186 |             mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
 187 |         } else {
 188 |             mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
 189 |         }
 190 |     }
```

---

#### [9] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 182
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【提示】1024多次出现，可以提取宏

- **代码片段**（行182）:
```cpp
 172 |             l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
 173 |             l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
 174 |         }
 175 | 
 176 |         // L0A B C 当前写死，能否通过基础api获取
 177 |         l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
 178 |         l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
 179 |         l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
 180 |         // L0A B C当前写死，要改成通过计算获取
 181 |         mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
 182 |         mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
 183 | 
 184 |         if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
 185 |                       mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
 186 |             mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
 187 |         } else {
 188 |             mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
 189 |         }
 190 |     }
 191 | 
```

---

#### [10] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 513
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > d的256、128建议用统一的宏

- **代码片段**（行513）:
```cpp
 503 |     }
 504 | 
 505 |     __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
 506 |     {
 507 |         if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
 508 |             if (runInfo.isChangeBatch) {
 509 |                 UpdateKey(runInfo.bIdx);
 510 |             }
 511 |         }
 512 | 
 513 |         if constexpr (dBaseSize > 256) {
 514 |             IterateBmm1NdL1SplitK(outputBuf, runInfo);
 515 |             return;
 516 |         }
 517 | 
 518 |         if constexpr (useDn) {
 519 |             if constexpr (dBaseSize > 128) {
 520 |                 IterateBmm1DnSplitK(outputBuf, runInfo);
 521 |             } else {
 522 |                 IterateBmm1Dn(outputBuf, runInfo);
```

---


#### [11] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 582
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】写死模板参数，可拓展性差，建议使用基本块size相关变量替代，避免基本块变化时散弹式修改:MatmulN<Q_T, KV_T, T, 64, 128, 256,

- **代码片段**（行582）:
```cpp
 572 |         mm1A.Wait<HardEvent::MTE2_MTE1>();
 573 |         mm1B.Wait<HardEvent::MTE2_MTE1>();
 574 | 
 575 |         Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
 576 |         mm1ResL0C.Wait<HardEvent::FIX_M>();
 577 | 
 578 |         MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
 579 |                                     (uint32_t)(constInfo.dSize + constInfo.dSizeRope), false, true);
 580 |         if constexpr (dBaseSize > 128) {
 581 |             if constexpr (s2BaseSize == 256) {
 582 |                 MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
 583 |                     mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 584 |                     param);
 585 |             } else {
 586 |                 MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
 587 |                     mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 588 |                     param);
 589 |             }
 590 |         } else {
 591 |             // TODO 整改 内部IF分支太多
```

---

#### [12] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 586
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】写死模板参数，可拓展性差，建议使用基本块size相关变量替代，避免基本块变化时散弹式修改:

- **代码片段**（行586）:
```cpp
 576 |         mm1ResL0C.Wait<HardEvent::FIX_M>();
 577 | 
 578 |         MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
 579 |                                     (uint32_t)(constInfo.dSize + constInfo.dSizeRope), false, true);
 580 |         if constexpr (dBaseSize > 128) {
 581 |             if constexpr (s2BaseSize == 256) {
 582 |                 MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
 583 |                     mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 584 |                     param);
 585 |             } else {
 586 |                 MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
 587 |                     mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 588 |                     param);
 589 |             }
 590 |         } else {
 591 |             // TODO 整改 内部IF分支太多
 592 |             MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
 593 |                 mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 594 |                 param);
 595 |         }
```

---

#### [13] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 592
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】写死模板参数，可拓展性差，建议使用基本块size相关变量替代，避免基本块变化时散弹式修改:

- **代码片段**（行592）:
```cpp
 582 |                 MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
 583 |                     mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 584 |                     param);
 585 |             } else {
 586 |                 MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
 587 |                     mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 588 |                     param);
 589 |             }
 590 |         } else {
 591 |             // TODO 整改 内部IF分支太多
 592 |             MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
 593 |                 mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
 594 |                 param);
 595 |         }
 596 | 
 597 |         if (unlikely(runInfo.isLastS2Loop)) {
 598 |             mm1A.Set<HardEvent::MTE1_MTE2>();
 599 |         }
 600 |         mm1B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
 601 |         mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
```

---

#### [14] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 660
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】写死模板参数，可拓展性差，建议使用基本块size相关变量替代，避免基本块变化时散弹式修改:

- **代码片段**（行660）:
```cpp
 650 |             } else {
 651 |                 CopyKeySlice(mm1BTensor, k * baseK, realK, runInfo);
 652 |             }
 653 |             mm1B.Set<HardEvent::MTE2_MTE1>();  // 通知
 654 |             mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
 655 |             mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B
 656 | 
 657 |             MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size, realK, false,
 658 |                                         true, k == 0, k == 0);
 659 | 
 660 |             MatmulFull<Q_T, KV_T, T, 128, 128, baseK, ABLayout::MK, ABLayout::KN>(
 661 |                 mm1A.GetTensor<Q_T>()[k * l1BaseKOffset], mm1BTensor, mmL0ABuffers, mmL0BBuffers,
 662 |                 mm1ResL0C.GetTensor<T>(), param);
 663 | 
 664 |             mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
 665 |         }
 666 |         if (unlikely(runInfo.isLastS2Loop)) {
 667 |             mm1A.Set<HardEvent::MTE1_MTE2>();
 668 |         }
 669 |         mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
```

---

#### [15] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h
- **行号**: 711
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】写死模板参数，可拓展性差，建议使用基本块size相关变量替代，避免基本块变化时散弹式修改:

- **代码片段**（行711）:
```cpp
 701 | 
 702 |         mm1A.Wait<HardEvent::MTE2_MTE1>();
 703 |         mm1B.Wait<HardEvent::MTE2_MTE1>();
 704 | 
 705 |         Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
 706 |         mm1ResL0C.Wait<HardEvent::FIX_M>();
 707 | 
 708 |         MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
 709 |                                     (uint32_t)(constInfo.dSize), false, true);
 710 | 
 711 |         MatmulK<KV_T, Q_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
 712 |             mm1A.GetTensor<KV_T>(), mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), param);
 713 | 
 714 |         if (unlikely(runInfo.isLastS2Loop)) {
 715 |             mm1B.Set<HardEvent::MTE1_MTE2>();
 716 |         }
 717 |         mm1A.Set<HardEvent::MTE1_MTE2>();
 718 | 
 719 |         mm1ResL0C.Set<HardEvent::M_FIX>();
 720 |         mm1ResL0C.Wait<HardEvent::M_FIX>();
```

---

### 文件: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h（Kernel侧）

---

#### [17] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 181
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】注释代码请删除：// AscendC::DumpTensor(this->lseMaxFdGm,__LINE__,512);

- **代码片段**（行181）:
```cpp
 171 |     }
 172 | 
 173 |     __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
 174 |                                             GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
 175 |                                             GlobalTensor<uint64_t> actualSeqLengthsGmQ,
 176 |                                             GlobalTensor<uint64_t> actualSeqLengthsGm, __gm__ uint8_t *key,
 177 |                                             __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
 178 |     {
 179 |         this->lseMaxFdGm = lseMaxFdGm;
 180 |         this->lseSumFdGm = lseSumFdGm;
 181 |         // AscendC::DumpTensor(this->lseMaxFdGm,__LINE__,512);
 182 |         // AscendC::DumpTensor(this->lseSumFdGm,__LINE__,512);
 183 |         this->accumOutGm = accumOutGm;
 184 |         this->attentionOutGm = attentionOutGm;
 185 |         this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
 186 |         this->actualSeqLengthsGm = actualSeqLengthsGm;
 187 | 
 188 |         this->keyPtr = key;
 189 | 
 190 |         qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);
```

---

#### [18] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 182
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】注释代码请删除：// AscendC::DumpTensor(this->lseSumFdGm,__LINE__,512);

- **代码片段**（行182）:
```cpp
 172 | 
 173 |     __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
 174 |                                             GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
 175 |                                             GlobalTensor<uint64_t> actualSeqLengthsGmQ,
 176 |                                             GlobalTensor<uint64_t> actualSeqLengthsGm, __gm__ uint8_t *key,
 177 |                                             __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
 178 |     {
 179 |         this->lseMaxFdGm = lseMaxFdGm;
 180 |         this->lseSumFdGm = lseSumFdGm;
 181 |         // AscendC::DumpTensor(this->lseMaxFdGm,__LINE__,512);
 182 |         // AscendC::DumpTensor(this->lseSumFdGm,__LINE__,512);
 183 |         this->accumOutGm = accumOutGm;
 184 |         this->attentionOutGm = attentionOutGm;
 185 |         this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
 186 |         this->actualSeqLengthsGm = actualSeqLengthsGm;
 187 | 
 188 |         this->keyPtr = key;
 189 | 
 190 |         qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);
 191 |         kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualSeqLenKVSize, constInfo.s2Size);
```

---

#### [21] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 474
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】注释代码请删除：// AscendC:DumpTensor(sinkBrcbBuf,__LINE__,64);

- **代码片段**（行474）:
```cpp
 464 | 
 465 |         LocalTensor<T> tmpSinkCastBuf = fdSinkTmpBuf.Get<T>();
 466 |         Cast(tmpSinkCastBuf, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo.gSize);
 467 |         AscendC::PipeBarrier<PIPE_V>();
 468 | 
 469 |         SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + cntM % 2);
 470 | 
 471 |         LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.Get<T>();
 472 |         Brcb(sinkBrcbBuf, tmpSinkCastBuf, (constInfo.gSize + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
 473 |              {1, BLOCK_ELEMENT_NUM});
 474 |         // AscendC:DumpTensor(sinkBrcbBuf,__LINE__,64);
 475 |         AscendC::PipeBarrier<PIPE_V>();
 476 |     }
 477 |     __aicore__ inline void SinkMax(uint32_t startRow, uint32_t dealRowCount)
 478 |     {
 479 |         constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
 480 |         int64_t gIdx = 0;
 481 |         LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.Get<T>();
 482 |         LocalTensor<T> sinkExpBuf = fdSinkExpBuf.Get<T>();
 483 | 
```

---

#### [23] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 696
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效printf请删除

- **代码片段**（行696）:
```cpp
 686 |     }
 687 | 
 688 | public:
 689 |     __aicore__ inline void FlashDecode(FDparamsX &fd)
 690 |     {
 691 |         // printf("fd.fdCoreEnable:%u\n",fd.fdCoreEnable);
 692 |         if (!fd.fdCoreEnable) {
 693 |             return;
 694 |         }
 695 |         // printf("fd.fdBN2Idx:%u\n",fd.fdBN2Idx);
 696 |         // printf("fd.fdMIdx:%u\n",fd.fdMIdx);
 697 |         // printf("fd.fdS2SplitNum:%u\n",fd.fdS2SplitNum);
 698 |         // printf("fd.mStart:%u\n",fd.mStart);
 699 |         // printf("fd.mLen:%u\n",fd.mLen);
 700 |         // printf("fd.fdWorkspaceIdx:%u\n",fd.fdWorkspaceIdx);
 701 | 
 702 |         uint32_t fdBalanceMBaseSize = 8U;
 703 |         uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
 704 |         uint32_t fdBalanceMTailSize =
 705 |             (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;
```

---

#### [24] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 698
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效printf请删除

- **代码片段**（行698）:
```cpp
 688 | public:
 689 |     __aicore__ inline void FlashDecode(FDparamsX &fd)
 690 |     {
 691 |         // printf("fd.fdCoreEnable:%u\n",fd.fdCoreEnable);
 692 |         if (!fd.fdCoreEnable) {
 693 |             return;
 694 |         }
 695 |         // printf("fd.fdBN2Idx:%u\n",fd.fdBN2Idx);
 696 |         // printf("fd.fdMIdx:%u\n",fd.fdMIdx);
 697 |         // printf("fd.fdS2SplitNum:%u\n",fd.fdS2SplitNum);
 698 |         // printf("fd.mStart:%u\n",fd.mStart);
 699 |         // printf("fd.mLen:%u\n",fd.mLen);
 700 |         // printf("fd.fdWorkspaceIdx:%u\n",fd.fdWorkspaceIdx);
 701 | 
 702 |         uint32_t fdBalanceMBaseSize = 8U;
 703 |         uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
 704 |         uint32_t fdBalanceMTailSize =
 705 |             (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;
 706 | 
 707 |         uint32_t reduceGlobaLoop = 0;
```

---

#### [25] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 700
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效printf请删除

- **代码片段**（行700）:
```cpp
 690 |     {
 691 |         // printf("fd.fdCoreEnable:%u\n",fd.fdCoreEnable);
 692 |         if (!fd.fdCoreEnable) {
 693 |             return;
 694 |         }
 695 |         // printf("fd.fdBN2Idx:%u\n",fd.fdBN2Idx);
 696 |         // printf("fd.fdMIdx:%u\n",fd.fdMIdx);
 697 |         // printf("fd.fdS2SplitNum:%u\n",fd.fdS2SplitNum);
 698 |         // printf("fd.mStart:%u\n",fd.mStart);
 699 |         // printf("fd.mLen:%u\n",fd.mLen);
 700 |         // printf("fd.fdWorkspaceIdx:%u\n",fd.fdWorkspaceIdx);
 701 | 
 702 |         uint32_t fdBalanceMBaseSize = 8U;
 703 |         uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
 704 |         uint32_t fdBalanceMTailSize =
 705 |             (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;
 706 | 
 707 |         uint32_t reduceGlobaLoop = 0;
 708 |         uint32_t reduceMLoop = 0;
 709 | 
```

---

#### [26] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 718
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效printf请删除

- **代码片段**（行718）:
```cpp
 708 |         uint32_t reduceMLoop = 0;
 709 | 
 710 |         uint32_t tmpFdS1gOuterMStart = 0;
 711 |         uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
 712 |         taskInfo.bIdx = fd.fdBN2Idx / constInfo.n2Size;
 713 |         taskInfo.n2Idx = fd.fdBN2Idx % constInfo.n2Size;
 714 |         taskInfo.gS1Idx = fd.fdMIdx * s1BaseSize;
 715 |         taskInfo.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
 716 |         uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
 717 |         uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;
 718 |         // printf("tmpFdS1gOuterMStart:%u\n",tmpFdS1gOuterMStart);
 719 |         // printf("tmpFdS1gOuterMEnd:%u\n",tmpFdS1gOuterMEnd);
 720 |         // printf("s1BaseSize:%u\n",s1BaseSize);
 721 |         // printf("taskInfo.actualCombineLoopSize:%u\n",taskInfo.actualCombineLoopSize);
 722 |         for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
 723 |              fdS1gOuterMIdx++) { // 左闭右闭
 724 |             uint32_t actualGSplitSize = fdBalanceMBaseSize;
 725 |             if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
 726 |                 actualGSplitSize = fdBalanceMTailSize;
 727 |             }
```

---

#### [27] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h
- **行号**: 738
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效printf请删除

- **代码片段**（行738）:
```cpp
 728 |             uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;
 729 | 
 730 |             LocalTensor<T> lseExp = fdLseExpBuf.Get<T>();
 731 |             LocalTensor<T> reduceOut = fdReduceBuf.Get<T>();
 732 |             WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
 733 |             CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
 734 |             SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
 735 |             WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
 736 |             if (unlikely(learnableSinkFlag)) {
 737 |                 CopySinkIn(reduceMLoop);
 738 |             }
 739 |             for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; preLoadIdx++) {
 740 |                 LocalTensor<T> mm2Res =
 741 |                     (reduceGlobaLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
 742 |                 WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
 743 |                 CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
 744 |                 SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
 745 |             }
 746 |             ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
 747 |             CalcPreNextTokens();
```

---

### 文件: attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h（Kernel侧）

---

#### [38] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h
- **行号**: 280
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】正式代码需要删除注释代码段

- **代码片段**（行280）:
```cpp
 270 |         // if constexpr (HAS_MASK) {
 271 |         constInfo.sparseMode =
 272 |             fiaAttenMaskParams.sparseMode; // TODO，后续sparseType、attenMaskCompressMode引用全部改成sparseMode
 273 |         constInfo.preTokens = fiaAttenMaskParams.preTokens;
 274 |         constInfo.nextTokens = fiaAttenMaskParams.nextTokens;
 275 |         constInfo.attenMaskBatch = fiaAttenMaskParams.attenMaskBatch;
 276 |         constInfo.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
 277 |         constInfo.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;
 278 |         constInfo.isRowInvalidOpen = fiaAttenMaskParams.isRowInvalidOpen;
 279 |         constInfo.isExistRowInvalid = fiaAttenMaskParams.isExistRowInvalid;
 280 |         // }
 281 | 
 282 |         if ASCEND_IS_AIV {
 283 |             if constexpr (VecFaBlockType::hasPse) {
 284 |                 constInfo.pseShiftByBatch = fiaPseParams.pseShiftByBatch;
 285 |                 constInfo.pseS1Size = fiaPseParams.pseS1Size;
 286 |                 constInfo.pseS2Size = fiaPseParams.pseS2Size;
 287 |                 constInfo.pseStride = s2BaseSize;
 288 |             }
 289 |         }
```

---

#### [40] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h
- **行号**: 775
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】正式代码需要删除注释代码段

- **代码片段**（行775）:
```cpp
 765 |             }
 766 |         }
 767 | 
 768 |         // PRINTF("CalcParams loop:%d, mloop:%d, isValid:%d, isChangeBatch:%d, isFirstS2Loop:%d, isLastS2Loop:%d,
 769 |         // bIdx:%d, n2Idx:%d, gS1Idx:%d, s1Idx:%d, s2Idx:%d\n",
 770 |         //     info.loop, info.mloop, info.isValid, info.isChangeBatch, info.isFirstS2Loop, info.isLastS2Loop,
 771 |         //     info.bIdx, info.n2Idx, info.gS1Idx, info.s1Idx, info.s2Idx);
 772 |         // PRINTF("CalcParams actS1Size:%d, actS2Size:%d, actMSize:%d, actMSizeAlign32:%d, actVecMSize:%d,
 773 |         // vecMbaseIdx:%d, actSingleLoopS2Size:%d, isS2SplitCore:%d\n",
 774 |         //     info.actS1Size, info.actS2Size, info.actMSize, info.actMSizeAlign32, info.actVecMSize, info.vecMbaseIdx,
 775 |         //     info.actSingleLoopS2Size, info.isS2SplitCore);
 776 |         // PRINTF("CalcParams faTmpOutWsPos:%d, preTokensLeftUp:%d, nextTokensLeftUp:%d, qPaddingBeginOffset:%d,
 777 |         // kvPaddingBeginOffset:%d\n",
 778 |         //     info.faTmpOutWsPos, info.preTokensLeftUp, info.nextTokensLeftUp, info.qPaddingBeginOffset,
 779 |         //     info.kvPaddingBeginOffset);
 780 |     }
 781 | 
 782 |     __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
 783 |                                           uint32_t &s2Cur)
 784 |     {
```

---

### 文件: attention/common/op_kernel/arch35/fia_public_define_arch35.h（Kernel侧）


#### [45] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_public_define_arch35.h
- **行号**: 231
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效结构体请删除

- **代码片段**（行231）:
```cpp
 221 | 
 222 | //     // 以下是需要用公式计算的信息
 223 | //     uint32_t s1StartIdx = 0;
 224 | //     uint32_t s1EndIdx = 0;
 225 | //     uint32_t s1Count = 0;
 226 | //     uint32_t gStartIdx = 0;
 227 | //     uint32_t gEndIdx = 0;
 228 | //     uint32_t gCount = 0;
 229 | // };
 230 | 
 231 | // struct MSplitInfo {
 232 | //     uint32_t nBufferIdx = 0U;
 233 | //     uint32_t nBufferStartM = 0U;
 234 | //     uint32_t nBufferDealM = 0U;
 235 | //     uint32_t vecStartM = 0U;
 236 | //     uint32_t vecDealM = 0U;
 237 | // };
 238 | 
 239 | // enum class TASK_DEAL_MODE : uint32_t
 240 | // {
```

---

#### [46] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_public_define_arch35.h
- **行号**: 239
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效类请删除

- **代码片段**（行239）:
```cpp
 229 | // };
 230 | 
 231 | // struct MSplitInfo {
 232 | //     uint32_t nBufferIdx = 0U;
 233 | //     uint32_t nBufferStartM = 0U;
 234 | //     uint32_t nBufferDealM = 0U;
 235 | //     uint32_t vecStartM = 0U;
 236 | //     uint32_t vecDealM = 0U;
 237 | // };
 238 | 
 239 | // enum class TASK_DEAL_MODE : uint32_t
 240 | // {
 241 | //     DEAL_ZERO = 0,
 242 | //     SKIP = 1,
 243 | //     CREATE_TASK = 2
 244 | // };
 245 | 
 246 | // template <LayOutTypeEnum LAYOUT, typename CONST_INFO_T>
 247 | // __aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t &gIdx, uint32_t &s1Idx, const CONST_INFO_T &constInfo)
 248 | // {
```

---

#### [47] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_public_define_arch35.h
- **行号**: 246
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效函数请删除

- **代码片段**（行246）:
```cpp
 236 | //     uint32_t vecDealM = 0U;
 237 | // };
 238 | 
 239 | // enum class TASK_DEAL_MODE : uint32_t
 240 | // {
 241 | //     DEAL_ZERO = 0,
 242 | //     SKIP = 1,
 243 | //     CREATE_TASK = 2
 244 | // };
 245 | 
 246 | // template <LayOutTypeEnum LAYOUT, typename CONST_INFO_T>
 247 | // __aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t &gIdx, uint32_t &s1Idx, const CONST_INFO_T &constInfo)
 248 | // {
 249 | //     // GS1
 250 | //     if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NBSD ||
 251 | //                   LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
 252 | //         gIdx = gS1Idx / constInfo.qSeqSize;
 253 | //         s1Idx = gS1Idx % constInfo.qSeqSize;
 254 | //     } else {
 255 | //         // S1G
```

---

#### [48] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_public_define_arch35.h
- **行号**: 261
- **评论时间**: 2026-04-27
- **Commit**: bb229a6b2a2a
- **问题描述**:

  > 【一般】无效函数请删除

- **代码片段**（行261）:
```cpp
 251 | //                   LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
 252 | //         gIdx = gS1Idx / constInfo.qSeqSize;
 253 | //         s1Idx = gS1Idx % constInfo.qSeqSize;
 254 | //     } else {
 255 | //         // S1G
 256 | //         s1Idx = gS1Idx / constInfo.gSize;
 257 | //         gIdx = gS1Idx % constInfo.gSize;
 258 | //     }
 259 | // }
 260 | 
 261 | // __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
 262 | // {
 263 | //     sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
 264 | //     sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
 265 | //     return sInnerToken;
 266 | // }
 267 | 
 268 | 
 269 | } // namespace AttentionCommon
 270 | 
```

---

## 被检视代码

> 本报告基于 PR 4699 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `attention/common/op_host/split_core_v2.cpp`
- `attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h`
- `attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h`
- `attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h`
- `attention/common/op_kernel/arch35/fia_public_define_arch35.h`
