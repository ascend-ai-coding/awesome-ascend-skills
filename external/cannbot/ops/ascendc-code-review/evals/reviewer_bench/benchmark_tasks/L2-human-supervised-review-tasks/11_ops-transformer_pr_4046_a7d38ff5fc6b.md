# 代码检视报告

## 检视概览
- **仓库**: ops-transformer
- **PR编号**: 4046
- **PR作者**: liwenguihw
- **代码文件**: attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h, attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **代码侧别**: Tiling侧
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: 14 条
- **检视时间**: 2026-07-29

## 检视统计

| 状态 | 条数 | 占比 |
|-----|------|------|
| 14 | 11 | 100% |

---

## 发现问题

### 文件: attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h（Tiling侧）

#### [1] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h
- **行号**: 93
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > MAX_KV_STACK_LEN 从 512 缩为 256，同步 kernel_common.hpp line 46。WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN 随之减半。这会影响所有基于 MAX_KV_STACK_LEN 做 tiling 切分的场景——如果某些旧用例以 kv stack 长度 > 256 为前提，本 PR 会把它们切到崩溃边界。请补充：1) 当前 tiling 里是否对 kv stack 超过 256 的场景有兜底或 fallback；2) UT/性能回归是否覆盖 kv stack > 256 的场景。
  > l2 级别问题

- **代码片段**（行93）:
```cpp
  83 |     TILING_DATA_FIELD_DEF(uint32_t, tailStartBatch)
  84 |     TILING_DATA_FIELD_DEF(uint32_t, tailStartN2)
  85 |     TILING_DATA_FIELD_DEF(uint32_t, tailKvNBlockTile)
  86 |     TILING_DATA_FIELD_DEF_STRUCT(coreNode, coreInfo)
  87 |     TILING_DATA_FIELD_DEF_STRUCT(splitNode, splitInfo)
  88 |     END_TILING_DATA_DEF
  89 |     
  90 |     const uint32_t SIZE_OF_16BIT = 2;
  91 |     const uint32_t SIZE_OF_32BIT = 4;
  92 |     const uint32_t N_SPLIT_HELPER = 2;
  93 |     const uint32_t MAX_KV_STACK_LEN = 256;
  94 |     const uint32_t Q_TILE_CEIL = 128;
  95 |     const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
  96 |     const uint32_t BASE_KV_SIZE = 128;
  97 |     const uint32_t PRELANCH_NUM = 3;
  98 |     const int64_t SPARSE_MODE_INT_MAX = 2147483647;
  99 |     const uint32_t TAIL_TASK_DIVISOR = 2;
 100 | 
 101 |     enum class MaskType : uint32_t {
 102 |         NO_MASK = 0,
```

---

### 文件: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp（Kernel侧）

#### [2] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 1106
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > DownCastP 新增的 bool isAlibiFullMask 参数没有默认值，但另一个未修改的 SubCoreCompute 重载（line 1169 起）在 line 1202 仍以三参数方式调用 DownCastP(sUbOffset, rowNumCurLoop, columnNumRound)，这条调用路径通过 line 1494 的 SubCoreCompute<false> 被实例化，会直接报编译错误。请给 isAlibiFullMask 加默认值 = false，或同时更新 line 1202 的调用点传入 false。
  > l2 级别问题

- **代码片段**（行1106）:
```cpp
1096 | 
1097 |         // gl = gl + exp(sink-lm)
1098 |         if constexpr (SINK_MODE == SinkMode::ENABLE) {
1099 |             if (isLastStackTile) {
1100 |                 UpdateRowSumWithSink(rowOffset, curLoop.rowNumCurLoop);
1101 |             }
1102 |         }
1103 |     }
1104 | 
1105 |     __aicore__ inline
1106 |     void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, bool isAlibiFullMask)
1107 |     {
1108 |         // *** lp = castfp32to16(ls)
1109 |         if (std::is_same<ElementOutput, bfloat16_t>::value) {
1110 |             if (isAlibiFullMask) {
1111 |                 AscendC::Cast<ElementOutput, float, false>(
1112 |                     lpUbTensorFullMask[sUbOffset],
1113 |                     lsUbTensor[sUbOffset],
1114 |                     AscendC::RoundMode::CAST_RINT,
1115 |                     (uint64_t)0,
```

---

#### [3] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 1273
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > 这里调用 CopyPUbToGm 只传了 5 个参数，但 CopyPUbToGm 的新签名（line 1150）最后一个 bool isAlibiFullMask 没有默认值，属于编译错误。另外更严重的是功能 bug：本函数刚在 line 1266 以 isAlibiFullMask 调用了 DownCastP，下转后的数据写入 lpUbTensorFullMask；而此处 CopyPUbToGm 默认分支（line 1158）使用 lpUbTensor，导致 full mask 路径实际拷到 GM 的是陈旧/未初始化的 lpUbTensor 数据。正确调用应为 CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad, isAlibiFullMask)。

- **代码片段**（行1273）:
```cpp
1263 |             AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
1264 |         }
1265 | 
1266 |         DownCastP(sUbOffset, rowNumCurLoop, columnNumRound, isAlibiFullMask);
1267 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
1268 | 
1269 |         CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
1270 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
1271 | 
1272 |         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
1273 |         CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
1274 |         if (isAlibiFullMask) {
1275 |  	             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
1276 |  	    } else {
1277 |                 if constexpr (!doTriUMask) {
1278 |                     AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
1279 |                     if (isLastNoMaskStackTile && isLastRowLoop) {
1280 |                         if(!startsWithMaskThenNomaskFlag) {
1281 |                         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
1282 |                     }
```

---

#### [4] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 1209
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > 这条调用同样只传了 5 个参数，CopyPUbToGm 新签名要求 6 个参数且 isAlibiFullMask 没有默认值，会编译报错。该重载对应非 full mask 路径，需要显式传入 false。建议与 issue 2 一起修复。

- **代码片段**（行1209）:
```cpp
1199 |             AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
1200 |         }
1201 | 
1202 |         DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
1203 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
1204 | 
1205 |         CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
1206 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
1207 | 
1208 |         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
1209 |         CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
1210 |         if constexpr (!doTriUMask) {
1211 |             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
1212 |             if (isLastNoMaskStackTile && isLastRowLoop) {
1213 |                 AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
1214 |                 AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
1215 |             }
1216 |         } else {
1217 |             AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
1218 |         }
```

---

#### [5] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 130
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > FULL32_UB_TENSOR_OFFSET_FULL_MASK = 11 * UB_UINT8_BLOCK_SIZE，与同一 init() 中的 MASK16_UB_TENSOR_OFFSET、FULL16_UB_TENSOR_OFFSET 完全重叠；line 116 的 FULL16_UB_TENSOR_OFFSET_FULL_MASK = 4*BLOCK 也与 LP/MASK/MASK32/FULL32 的偏移重叠。UB 空间别名本身不是禁忌，但这里 fullUbTensor32FullMask 在 full mask 分支与 CalcExp/Applyfull 的整个生命周期内都要持有数据（line 2246 Applyfull 使用 fullUbTensor32FullMask），同时 maskUbTensor16/fullUbTensor16 在其它分支也会读写该区域，需要在注释里明确哪些分支是互斥的、何时释放，并最好在代码里用 static_assert 或运行期断言把互斥关系固定下来，否则未来改动会很容易互相踩脏数据。

- **代码片段**（行130）:
```cpp
 120 | 
 121 |         constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
 122 |         constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
 123 |         constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
 124 |         constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
 125 |         constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;
 126 |         constexpr uint32_t SEL_MASK_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET;
 127 | 
 128 |         constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
 129 |         constexpr uint32_t FULL16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;
 130 |         constexpr uint32_t FULL32_UB_TENSOR_OFFSET_FULL_MASK = 11 * UB_UINT8_BLOCK_SIZE;
 131 |         scaleValue = scaleValue_;
 132 |         lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
 133 |         lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
 134 |         lpUbTensorFullMask = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET_FULL_MASK);
 135 |         maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
 136 |         maskUbTensorUint8 = resource.ubBuf.template GetBufferByByte<uint8_t>(MASK_UB_TENSOR_OFFSET);
 137 |         maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
 138 |         maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
 139 |         fullUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementFull>(FULL16_UB_TENSOR_OFFSET);
```

---

#### [6] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 564
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > CopyFullGmToUb1 这个命名不合适。原函数叫 CopyFullGmToUb，新增一个带 1 后缀的版本会让读者完全猜不到它和原函数的区别。它实际是一次性将 full mask 一整块整核数据搬到 fullUbTensor16FullMask，与原函数按 loop 切块搬不同，建议改名为 CopyFullGmToUbWholeBlock 或类似能体现语义的名字。

- **代码片段**（行564）:
```cpp
 554 |     void CalcGmFullShift(int64_t &offsetFull, const LayoutInput &layoutMask,
 555 |     uint32_t rowOffer, uint32_t kvSStartIdx, uint32_t maskOffsetThisSubBlock)
 556 |     {
 557 |         uint32_t fullBlockStart = rowOffer;
 558 |         uint32_t gmOffsetFullRow = fullBlockStart + maskOffsetThisSubBlock ;
 559 |         uint32_t gmOffsetFullColumn = kvSStartIdx;
 560 |         offsetFull = layoutMask.GetOffset(MatrixCoord(gmOffsetFullRow, gmOffsetFullColumn));
 561 |     }
 562 | 
 563 |     __aicore__ inline
 564 |     void CopyFullGmToUb1(
 565 |         AscendC::GlobalTensor<ElementFull> gFull,uint32_t tokenNumPerHeadThisSubBlock,
 566 |         uint32_t columnNum, uint32_t columnNumRound, uint32_t maskStride,
 567 |         uint32_t qNStartIdxVec, uint32_t qNThisSubBlock, uint32_t BIdx,
 568 |         uint32_t qHeads, int64_t offsetFull, int64_t pseQ, int64_t pseKv)
 569 |     {
 570 |         uint32_t innerUbRowOffset = 0;
 571 |         int64_t gFullOffset = 0;
 572 |         uint32_t loopNum = (qNThisSubBlock >= HEAD_NUM_2) ? (qNStartIdxVec + qNThisSubBlock - 1) : qNStartIdxVec + 1;
 573 |         for (uint32_t headIdx = qNStartIdxVec;headIdx < loopNum; headIdx++) {
```

---

#### [7] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 2003
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > 从这里到 line 2152 左右，整个新 operator() 的一份注释掉的旧实现（约 150 行）被原样保留，且中间还夹着 AscendC::printf 调试语句。合入前请清理掉，这类整块 commented-out 代码会严重污染后续 diff 审查，并且 git 已经能保留历史版本。

- **代码片段**（行2003）:
```cpp
1993 |                     (delayedRowLoopIdx == 0),
1994 |                     (delayedRowLoopIdx == rowLoopNum - 1),
1995 |                     columnNumRound,
1996 |                     pingpongFlag,
1997 |                     curStackTileMod,
1998 |                     curSinkLoop,
1999 |                     isLastStackTile,
2000 |                     false,
2001 |                     false);
2002 |             }
2003 |         }
2004 |     }
2005 | 
2006 |     // __aicore__ inline
2007 |     // void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
2008 |     //     AscendC::GlobalTensor<ElementSink> gSink, AscendC::GlobalTensor<ElementFull> gFull,
2009 |     //     const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
2010 |     //     const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
2011 |     //     uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t rowOffer, uint32_t kvSStartIdx,
2012 |     //     uint32_t kvSEndIdx, uint32_t qNStartIdx, uint32_t BIdx, uint32_t qHeads, int64_t pseQ, int64_t pseKv, bool isLastStackTile)
```

---

#### [8] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 583
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > 新函数里散落了多处注释掉的 AscendC::printf 和 AscendC::DumpTensor 调试代码（line 583、604、605、713、718，以及 operator() 里 2174-2178、2218、2292），部分后面还带着中文注释表示 失败/成功/可以正常 之类的 debug 笔记。这些属于联调遗留，合入前请全部删除，保持 kernel 代码干净——这种 debug 代码留在 release 仓库很容易被后续同事误认为是可用 API。

- **代码片段**（行583）:
```cpp
 573 |         for (uint32_t headIdx = qNStartIdxVec;headIdx < loopNum; headIdx++) {
 574 |             gFullOffset = BIdx * qHeads * pseQ * pseKv + headIdx * pseQ * pseKv + offsetFull;
 575 |             // AscendC::printf("gFullOffset:%d\n",gFullOffset);
 576 |             AscendC::DataCopyPad(
 577 |                     fullUbTensor16FullMask[innerUbRowOffset], gFull[gFullOffset],
 578 |             AscendC::DataCopyExtParams(
 579 |                         tokenNumPerHeadThisSubBlock, columnNum * sizeof(ElementFull),
 580 |                         (maskStride - columnNum) * sizeof(ElementFull),
 581 |                         (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
 582 |             AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
 583 |             // AscendC::DumpTensor(fullUbTensor16FullMask[innerUbRowOffset],0,columnNum); // 失败
 584 |             innerUbRowOffset += tokenNumPerHeadThisSubBlock * columnNumRound;
 585 |         }
 586 |         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
 587 |     }
 588 | 
 589 |         __aicore__ inline
 590 |     void CopyFullUbToUb(
 591 |         uint32_t columnNumRound, uint32_t qSBlockSize, uint32_t tokenNumPerHead,
 592 |         uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum, uint32_t epiTokenNum,
```

---

#### [9] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 1274
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > 从这里到 line 1288 的 if (isAlibiFullMask) / else 分支混用了 Tab 和空格缩进，例如 line 1275 和 line 1288 前面是 Tab+空格组合，其它行是纯空格，渲染出来层次混乱。整段请按项目其它位置的 4 空格缩进重写。另外 else 分支内部的 if constexpr 多嵌套了一层缩进（line 1277 起），但语义上它只是 isAlibiFullMask==false 时的原有逻辑，应与 if 分支在同一层级。

- **代码片段**（行1274）:
```cpp
1264 |         }
1265 | 
1266 |         DownCastP(sUbOffset, rowNumCurLoop, columnNumRound, isAlibiFullMask);
1267 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
1268 | 
1269 |         CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
1270 |         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
1271 | 
1272 |         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
1273 |         CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
1274 |         if (isAlibiFullMask) {
1275 |  	             AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
1276 |  	    } else {
1277 |                 if constexpr (!doTriUMask) {
1278 |                     AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
1279 |                     if (isLastNoMaskStackTile && isLastRowLoop) {
1280 |                         if(!startsWithMaskThenNomaskFlag) {
1281 |                         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
1282 |                     }
1283 |                         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
```

---


#### [10] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 589
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > __aicore__ inline 这一行缩进了 8 个空格，与下一行 void CopyFullUbToUb( 的 4 空格不对齐，也与文件里其它成员函数声明不一致。请统一缩进到 4 空格。

- **代码片段**（行589）:
```cpp
 579 |                         tokenNumPerHeadThisSubBlock, columnNum * sizeof(ElementFull),
 580 |                         (maskStride - columnNum) * sizeof(ElementFull),
 581 |                         (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
 582 |             AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
 583 |             // AscendC::DumpTensor(fullUbTensor16FullMask[innerUbRowOffset],0,columnNum); // 失败
 584 |             innerUbRowOffset += tokenNumPerHeadThisSubBlock * columnNumRound;
 585 |         }
 586 |         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
 587 |     }
 588 | 
 589 |         __aicore__ inline
 590 |     void CopyFullUbToUb(
 591 |         uint32_t columnNumRound, uint32_t qSBlockSize, uint32_t tokenNumPerHead,
 592 |         uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum, uint32_t epiTokenNum,
 593 |         uint32_t &qNStartUbIdx, uint32_t qNThisSubBlock, uint32_t rowNumCurLoop)
 594 |     {
 595 |         uint32_t innerUbRowOffset = 0;
 596 |         int64_t gFullOffset = 0;
 597 |         if (proTokenNum != 0) {
 598 |             // AscendC::printf("qNStartUbIdx:%d\n",qNStartUbIdx);
```

---

#### [11] 人工检视意见

- **提出人**: Allan_Yu
- **作者**: liwenguihw
- **文件**: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp
- **行号**: 608
- **评论时间**: 2026-04-11
- **Commit**: a7d38ff5fc6b
- **问题描述**:

  > innerUbRowOffset += proTokenNum * columnNumRound; 这一行逻辑上属于 if (proTokenNum != 0) 分支内部，但缩进只有 8 空格，看起来像是脱出 if 块之外。请与上面 UpCastMask 调用保持同样 12 空格缩进，避免读代码时误判作用域。

- **代码片段**（行608）:
```cpp
 598 |             // AscendC::printf("qNStartUbIdx:%d\n",qNStartUbIdx);
 599 |             gFullOffset = qNStartUbIdx * qSBlockSize * columnNumRound + proTokenIdx * columnNumRound;
 600 |             AscendC::DataCopy(
 601 |                     lpUbTensorFullMask[innerUbRowOffset], fullUbTensor16FullMask[gFullOffset],
 602 |                     proTokenNum * columnNumRound);
 603 |             AscendC::PipeBarrier<PIPE_V>();
 604 |             // AscendC::DumpTensor(lpUbTensorFullMask[innerUbRowOffset],1,columnNumRound);
 605 |             // AscendC::DumpTensor(fullUbTensor16FullMask[gFullOffset],2,columnNumRound); // 成功
 606 |             UpCastMask<float, ElementFull>(fullUbTensor32FullMask[innerUbRowOffset], lpUbTensorFullMask[innerUbRowOffset],
 607 |                 rowNumCurLoop, columnNumRound);
 608 |         innerUbRowOffset += proTokenNum * columnNumRound;
 609 |         }
 610 |         for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
 611 |             if (qNThisSubBlock >= HEAD_NUM_2) {
 612 |                 if (proTokenNum > 0 && headIdx == 0) {
 613 |                     qNStartUbIdx++;
 614 |                 }
 615 |                 if (headIdx > 0) {
 616 |                     qNStartUbIdx++;
 617 |                 }
```

---

## 被检视代码

> 本报告基于 PR 4046 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`
- `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp`
