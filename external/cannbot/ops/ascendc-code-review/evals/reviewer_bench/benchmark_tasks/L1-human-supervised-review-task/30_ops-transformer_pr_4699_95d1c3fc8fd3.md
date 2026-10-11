# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 4699
- **PR作者**: tang-hao-hw-gitcode
- **代码文件**: 4 个文件
- **代码侧别**: Kernel侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 33 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 33 | 30 | 100% |

---

## 发现问题

### 文件: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h（Kernel侧）

---

#### [2] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 12
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】 * \file flash_attention_noquant_block_vec_base.h与实际文件名字不符合：attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h

- **代码片段**（行12）:
```cpp
   2 |  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
   3 |  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
   4 |  * CANN Open Software License Agreement Version 2.0 (the "License").
   5 |  * Please refer to the License for details. You may not use this file except in compliance with the License.
   6 |  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   7 |  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   8 |  * See LICENSE in the root of the software repository for the full text of the License.
   9 |  */
  10 | 
  11 | /*!
  12 |  * \file flash_attention_noquant_block_vec_base.h
  13 |  * \brief
  14 |  */
  15 | #ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_
  16 | #define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_
  17 | 
  18 | #include "kernel_operator.h"
  19 | 
  20 | #include "flash_attention_score_common_regbase.h"
  21 | #include "adv_api/activation/softmax.h"
```

---

#### [3] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 15
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】宏FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_与实际文件名字不符合

- **代码片段**（行15）:
```cpp
   5 |  * Please refer to the License for details. You may not use this file except in compliance with the License.
   6 |  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   7 |  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   8 |  * See LICENSE in the root of the software repository for the full text of the License.
   9 |  */
  10 | 
  11 | /*!
  12 |  * \file flash_attention_noquant_block_vec_base.h
  13 |  * \brief
  14 |  */
  15 | #ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_
  16 | #define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_
  17 | 
  18 | #include "kernel_operator.h"
  19 | 
  20 | #include "flash_attention_score_common_regbase.h"
  21 | #include "adv_api/activation/softmax.h"
  22 | #include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
  23 | #include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
  24 | #include "vf/vf_flashupdate_new.h"
```

---

#### [4] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 407
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】无效分支请删除

- **代码片段**（行407）:
```cpp
 397 |             }
 398 |         } else {
 399 |             if (constInfo.isSoftmaxLseEnable) {
 400 |                 SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
 401 |             }
 402 |         }
 403 | 
 404 |         if (constInfo.learnableSinkFlag) {
 405 |             // if (constInfo.isGqa) {
 406 |             this->Vec1SinkComputeGSFused(runInfo, sumUb, maxUb);
 407 |             // } else {
 408 |             //     this->Vec1SinkCompute(runInfo, sumUb, maxUb);
 409 |             // }
 410 |         }
 411 |     }
 412 | 
 413 |     __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
 414 |                                              RunInfoX &runInfo)
 415 |     {
 416 |         if (unlikely(runInfo.actVecMSize == 0)) {
```

---

#### [5] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 496
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】无效函数请删除

- **代码片段**（行496）:
```cpp
 486 |         sinkCopyParams.blockCount = 1;                                   // 进行一次连续拷贝
 487 |         sinkCopyParams.blockLen = runInfo.actVecMSize * sizeof(INPUT_T); // 实际需要拷贝的字节数
 488 |         sinkCopyParams.srcStride = 0;                                    // 源地址连续
 489 |         sinkCopyParams.dstStride = 0;                                    // 目的地址连续
 490 | 
 491 |         DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
 492 |         DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
 493 |         sinkQue.EnQue(sinkUbBf16);
 494 |     }
 495 | 
 496 |     // __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar,
 497 |     //                                                SoftMaxShapeInfo &softmaxShapeInfo)
 498 |     // {
 499 |     //     event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
 500 |     //     SetFlag<HardEvent::V_S>(eventIdVToS);
 501 |     //     WaitFlag<HardEvent::V_S>(eventIdVToS);
 502 |     //     bool isUpdateNeedCheck = false;
 503 |     //     SetMaskCount();
 504 |     //     SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
 505 |     //     for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
```

---

#### [6] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 518
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】无效函数请删除

- **代码片段**（行518）:
```cpp
 508 |     //         if (checkValue == negativeIntScalar) {
 509 |     //             isUpdateNeedCheck = true;
 510 |     //             break;
 511 |     //         }
 512 |     //     }
 513 |     //     SetMaskNorm();
 514 |     //     ResetMask();
 515 |     //     return isUpdateNeedCheck;
 516 |     // }
 517 | 
 518 |     // __aicore__ inline void InvalidLineProcess(RunInfoX runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
 519 |     // {
 520 |     //     if (constInfo.softMaxCheckRes) {
 521 |     //         SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(runInfo.actVecMSize), static_cast<uint32_t>(1),
 522 |     //                                           static_cast<uint32_t>(runInfo.actVecMSize), static_cast<uint32_t>(1)};
 523 |     //         bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VALUE_FP32, softmaxShapeInfo);
 524 |     //         if (!res) {
 525 |     //             // constInfo.softMaxCheckRes = false;
 526 |     //         } else {
 527 |     //             if (unlikely(runInfo.isLastS2Loop)) {
```

---

#### [7] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 567
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 正式代码todo请删除

- **代码片段**（行567）:
```cpp
 557 |                                    .s2LeftPaddingSize = 0,
 558 |                                    .actualBIdx = runInfo.bIdx};
 559 |             bool qsEqualOne = (constInfo.s1Size == 1);
 560 |             copyPSEGmToUb(pseShiftUbTensor, pseShiftGmTensor, pseCoord, qsEqualOne);
 561 |             pseStride = constInfo.pseStride;
 562 |         }
 563 | 
 564 |         LocalTensor<uint8_t> attenMaskUb;
 565 |         LocalTensor<uint8_t> attenMaskUbPre;
 566 |         if constexpr (hasAtten == true) {
 567 |             // TODO，attenMaskInQue后续可以改成TBuf，用set/wait同步
 568 |             attenMaskUb = this->attenMaskInQue[runInfo.loop % DB].template AllocTensor<uint8_t>();
 569 |             attenMaskUbPre = this->attenMaskInQue[1 - runInfo.loop % DB].template AllocTensor<uint8_t>();
 570 |             AttenMaskCopyIn(attenMaskUb, attenMaskUbPre, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
 571 |         }
 572 | 
 573 |         LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
 574 |         LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
 575 |         LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
 576 |         LocalTensor<T> pScaleUb;
```

---

#### [8] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 764
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】空的else分支请删除

- **代码片段**（行764）:
```cpp
 754 |     }
 755 | 
 756 |     template <typename VEC2_RES_T>
 757 |     __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut,
 758 |                                      LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
 759 |                                      int64_t dSizeAligned64)
 760 |     {
 761 |         if (constInfo.isPostQuantPerChnl) {
 762 |             // postQuantScaleShape (N2, dV)
 763 |             // TODO: 复用输入Buffer 重构!!!, 当前使用一个固定2K的buffer搬运Scale
 764 | 
 765 |         } else {
 766 |             // PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, vec2ResUb, constInfo.postQuantScaleValue,
 767 |             //                                           constInfo.postQuantOffsetValue, mDealSize,
 768 |             //                                           constInfo.dSizeV, dSizeAligned64);
 769 |         }
 770 |     }
 771 | 
 772 |     /* PostQuant 必须重构 */
 773 |     template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
```

---

#### [10] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 992
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】正式代码的printf需要删除

- **代码片段**（行992）:
```cpp
 982 |     {
 983 |         if constexpr (hasAtten) {
 984 |             int64_t s1FirstValidToken = Min(Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
 985 |             int64_t s1LastValidToken = Min(Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
 986 |             s1LastValidToken = Max(s1LastValidToken - 1, 0);
 987 |             bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
 988 |             bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
 989 |                                        ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
 990 |                                         hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
 991 | 
 992 |             // printf("########################\n");
 993 |             // printf("bIdx:%d, n2Idx:%d, gIdx:%d, s1Idx:%d, gS1Idx:%d\n", runInfo.bIdx, runInfo.n2Idx, runInfo.gIdx,
 994 |             // runInfo.s1Idx, runInfo.gS1Idx); printf("batchNeedRowInvalid:%d\n", batchNeedRowInvalid);
 995 | 
 996 |             if (!batchNeedRowInvalid) {
 997 |                 return;
 998 |             }
 999 | 
1000 |             // printf("s1FirstValidToken:%d, s1LastValidToken:%d\n", s1FirstValidToken, s1LastValidToken);
1001 | 
```

---

#### [11] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 994
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】注释代码请删除

- **代码片段**（行994）:
```cpp
 984 |             int64_t s1FirstValidToken = Min(Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
 985 |             int64_t s1LastValidToken = Min(Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
 986 |             s1LastValidToken = Max(s1LastValidToken - 1, 0);
 987 |             bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
 988 |             bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
 989 |                                        ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
 990 |                                         hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
 991 | 
 992 |             // printf("########################\n");
 993 |             // printf("bIdx:%d, n2Idx:%d, gIdx:%d, s1Idx:%d, gS1Idx:%d\n", runInfo.bIdx, runInfo.n2Idx, runInfo.gIdx,
 994 |             // runInfo.s1Idx, runInfo.gS1Idx); printf("batchNeedRowInvalid:%d\n", batchNeedRowInvalid);
 995 | 
 996 |             if (!batchNeedRowInvalid) {
 997 |                 return;
 998 |             }
 999 | 
1000 |             // printf("s1FirstValidToken:%d, s1LastValidToken:%d\n", s1FirstValidToken, s1LastValidToken);
1001 | 
1002 |             bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
1003 |             blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;
```

---

#### [12] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 1005
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】调试代码请删除

- **代码片段**（行1005）:
```cpp
 995 | 
 996 |             if (!batchNeedRowInvalid) {
 997 |                 return;
 998 |             }
 999 | 
1000 |             // printf("s1FirstValidToken:%d, s1LastValidToken:%d\n", s1FirstValidToken, s1LastValidToken);
1001 | 
1002 |             bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
1003 |             blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;
1004 | 
1005 |             // printf("blockNeedRowInvalid:%d\n", blockNeedRowInvalid);
1006 |             // printf("########################\n");
1007 | 
1008 |             if (blockNeedRowInvalid) {
1009 |                 LocalTensor<float> maxTensor =
1010 |                     softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
1011 |                 if constexpr (!POST_QUANT) {
1012 |                     RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
1013 |                                               static_cast<uint32_t>(dSizeAligned64));
1014 |                 } else {
```

---

#### [13] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 1006
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】调试代码请删除

- **代码片段**（行1006）:
```cpp
 996 |             if (!batchNeedRowInvalid) {
 997 |                 return;
 998 |             }
 999 | 
1000 |             // printf("s1FirstValidToken:%d, s1LastValidToken:%d\n", s1FirstValidToken, s1LastValidToken);
1001 | 
1002 |             bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
1003 |             blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;
1004 | 
1005 |             // printf("blockNeedRowInvalid:%d\n", blockNeedRowInvalid);
1006 |             // printf("########################\n");
1007 | 
1008 |             if (blockNeedRowInvalid) {
1009 |                 LocalTensor<float> maxTensor =
1010 |                     softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
1011 |                 if constexpr (!POST_QUANT) {
1012 |                     RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
1013 |                                               static_cast<uint32_t>(dSizeAligned64));
1014 |                 } else {
1015 |                     uint32_t dStride =
```

---


#### [20] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h
- **行号**: 1455
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】正式代码的注释代码段需删除

- **代码片段**（行1455）:
```cpp
1445 |                 DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
1446 |             } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
1447 |                 OffsetCalculator<GmFormat::NGTD> offsetCalculator;
1448 |                 offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, actualSeqLengthsGmQ,
1449 |                                       constInfo.actualSeqLenSize);
1450 |                 DealActSeqLenIsZero<GmFormat::NGTD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
1451 |             }
1452 |         }
1453 |     }
1454 | 
1455 |     // template <GmFormat GM_FORMAT, typename OUT_T>
1456 |     // __aicore__ inline void UpdateAttenOutZero(FaGmTensor<OUT_T, GM_FORMAT> &dstTensor, GmCoord &gmCoord)
1457 |     // {
1458 |     //     if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
1459 |     //         ProcessS1G(dstTensor, gmCoord);
1460 |     //     } else if constexpr (GM_FORMAT == GmFormat::BNGSD || GM_FORMAT == GmFormat::NGTD) {
1461 |     //         ProcessContinuous(dstTensor, gmCoord);
1462 |     //     }
1463 |     // }
1464 | 
```

---

### 文件: attention/common/op_kernel/arch35/fia_tiling_data_noquant_gqa.h（Kernel侧）

---

#### [29] 人工检视意见

- **提出人**: yang-binrong
- **作者**: tang-hao-hw-gitcode
- **文件**: attention/common/op_kernel/arch35/vf/vf_post_quant_gs1.h
- **行号**: 18
- **评论时间**: 2026-04-27
- **Commit**: 95d1c3fc8fd3
- **问题描述**:

  > 【一般】无效函数请删除

- **代码片段**（行18）:
```cpp
   8 |  * See LICENSE in the root of the software repository for the full text of the License.
   9 |  */
  10 | /*!
  11 |  * \file vf_post_quant_gs1.h
  12 |  * \brief Post-quantization VF implementation for both GS1 and S1G formats (arch35)
  13 |  */
  14 | #ifndef VF_POST_QUANT_V2_H
  15 | #define VF_POST_QUANT_V2_H
  16 | #include "kernel_tensor.h"
  17 | namespace FaVectorApi {
  18 | // static constexpr MicroAPI::::CastTrait castTraitP0 = {
  19 | //     MicroAPI::RegLayout::ZERO,
  20 | //     MicroAPI::SatMode::NO_SAT,
  21 | //     MicroAPI::MaskMergeMode::ZEROING,
  22 | //     RoundMode::CAST_RINT
  23 | // };
  24 | __simd_vf__ inline void CastBf16ToFp32VF(__ubuf__ float *dstFp32Ub, __ubuf__ bfloat16_t *srcBf16Ub,
  25 |                                          uint32_t elementCount)
  26 | {
  27 |     RegTensor<bfloat16_t> vregSrc;
```

---

### 文件: attention/common/op_kernel/memory_copy_arch35.h（Kernel侧）

---

## 被检视代码

> 本报告基于 PR 4699 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h`
- `attention/common/op_kernel/arch35/fia_tiling_data_noquant_gqa.h`
- `attention/common/op_kernel/arch35/vf/vf_post_quant_gs1.h`
- `attention/common/op_kernel/memory_copy_arch35.h`
