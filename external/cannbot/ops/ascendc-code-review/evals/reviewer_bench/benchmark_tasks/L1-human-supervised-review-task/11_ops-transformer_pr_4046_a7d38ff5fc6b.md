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
---

### 文件: attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp（Kernel侧）

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


## 被检视代码

> 本报告基于 PR 4046 的人工检视评论生成（已过滤 PR 作者自己的评论、回复及修复后的 commit）

- `attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`
- `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp`
