/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// MC2 Blaze Kernel 包装 —— QuantMatmulMxKernelSwat
// ----------------------------------------------------------------------------
// 该类把标准 Blaze BlockMmad 接入 MC2 通算融合流水:
//   - AIV 通过 UDMA 把本 rank 的 M 段 Put 给所有其他 rank；
//   - AIC 在通信 buffer 上做完整 Matmul——遍历所有 rank，按 (rank, mPos, nPos) 切片；
//   - rank == rankId 时从本卡 GM（localAGmAddr_）读 A，其他 rank 从通信 buffer（aGmAddr_）读 A；
//   - 所有 rank 的部分和在 L0C 上累加（remoteRankCnt 控制 L0C 偏移），最后一次触发 fixpipe 输出 GM。
//
// 与 ascendc-blaze-best-practice 的关系：
//   复用 Blaze 基底（BlockMmad、BlockScheduler），通过遍历所有 rank 接入通算流水。
//   单卡 Matmul（无通信）等价于 ascendc-blaze-best-practice 的常规用法。
//
// 改造场景：
//   [MODIFY] N1  改 dtype（MX FP8 → BF16）：去 scale 相关字段 + 换 BlockMmad 模板
//                （block_mmad_qbmm_mx.h → block_mmad.h），详见 references/matmul_blaze.md §6.3
//   [MODIFY] C1  加 bias：Init 中读 biasGmAddr_ + ProcessSingleBatch 中 Slice bias
//   [MODIFY] A1  改 L2Cache 策略：SetL2Cache / SetScaleL2Cache 中调 CacheMode
//
// [PITFALL] rank == rankId 时必须从 localAGmAddr_（本卡 GM）读 A——通信 buffer 中
//           本 rank 的段未被 Put（AllToAllComm::PutToAllRanks 跳过 remoteRank == rankId）。
// [PITFALL] layout 维度含 rankSize *：A/B 矩阵在 MC2 中是"按 rank 切分后逻辑拼起来"的。
//           Slice 时按 rank 切，actualMPos = rank * oriM + mPos（local A），
//           actualCommMPos = rank * M + mPos（通信 buffer）。
// [PITFALL] remoteRankCnt 控制 L0C 累加位置——必须从 0 起算且与 rank 顺序对应；
//           BlockMmad 在最后一个 remoteRankCnt 触发 fixpipe，提前返回会丢数据。
//
// 进阶细节：references/matmul_blaze.md + references/mc2_architecture.md §8
// ============================================================================

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
  template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;
// B 原工程 qbmm_mx_kernel.h 在 Kernel 命名空间内直接使用 Get<...>(shape)，
// 但 Blaze/tensor_api 把 Get 定义在 AscendC::Te::Get。补充 using 声明，避免未解析符号。
using AscendC::Te::Get;

/**
 * @brief SWAT MX 量化矩阵乘内核实现
 * 该类负责具体的矩阵乘块调度和计算：遍历所有 rank，在 L0C 上累加部分和。
 */

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMatmulMxKernelSwat {
public:
  __aicore__ inline QuantMatmulMxKernelSwat()
  {}
  __aicore__ inline ~QuantMatmulMxKernelSwat()
  {}

  static constexpr bool weightNz = BlockMmad::weightNz;
  static constexpr bool transA = BlockMmad::transA;
  static constexpr bool transB = BlockMmad::transB;

  using BlockMmadParams = typename BlockMmad::Params;
  using L1Params = typename BlockMmad::L1Params;
  using AType = typename BlockMmad::AType;
  using BType = typename BlockMmad::BType;
  using CType = typename BlockMmad::CType;
  using BiasType = typename BlockMmad::BiasType;
  using LayoutA = typename BlockMmad::LayoutA;
  using LayoutB = typename BlockMmad::LayoutB;
  using LayoutC = typename BlockMmad::LayoutC;
  static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
  static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
  static constexpr int32_t SCALE_C0 = 2;

  using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

  using BlockSchedulerParams = typename BlockScheduler::Params;
  using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
  using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
  using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
  using MakeLayoutScaleA = Std::conditional_t<
    transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
    Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
  using MakeLayoutScaleB = Std::conditional_t<
    transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
    Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;
  /**
   * @brief Tiling 配置
   */
  struct QBMMTiling {
    enum BiasMode : uint32_t {
      BIAS_DISABLED = 0,
      BIAS_ENABLED = 1
    };
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t dbL0C;
    uint32_t isBias;
  };

  /**
   * @brief 本地 Rank 相关参数
   */
  struct LocalParams {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t originalM;    // 单卡负责的总 M 行数
    GM_ADDR  localAGmAddr;        // 本卡 GM 的 A（rank==rankId 时用）
    GM_ADDR  localScaleAGmAddr;   // 本卡 GM 的 ScaleA
    uint32_t splitKNum;
  };

  /**
   * @brief 顶层参数结构
   */
  struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    QBMMTiling qbmmParams;
    LocalParams localParams;
  };

public:
  __aicore__ inline void Init(const Params &params);
  __aicore__ inline void Run(const Params &params);
  __aicore__ inline void operator()(const Params &params)
  {
    Run(params);
  }

private:
  __aicore__ inline void ResetGmAddr(const Params &params);
  __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler& bs,
                        uint64_t restBatch, bool isTailRound);

  template <typename TensorB, typename TensorScaleB, typename TensorC>
  __aicore__ inline void SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
    TensorScaleB &gmScaleB, TensorC& gmC);

  template <typename TensorScaleB>
  __aicore__ inline void SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB);

private:
  BlockMmad mmadOp_;

  __gm__ AType *aGmAddr_;               // 远程数据基址（通信缓冲区）
  __gm__ AType *localAGmAddr_;          // 本地数据基址
  __gm__ BType *bGmAddr_;
  __gm__ CType *cGmAddr_;               // 输出基址（已根据流水步偏移）
  __gm__ BiasType *biasGmAddr_ = nullptr;
  __gm__ ::fp8_e8m0_t *scaleAGmAddr_;     // 远程 Scale 基址
  __gm__ ::fp8_e8m0_t *localScaleAGmAddr_;
  __gm__ ::fp8_e8m0_t *scaleBGmAddr_;

  bool isBias_{false};
  bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Run(const Params &params)
{
  Init(params);

  BlockScheduler bs(params.problemShape, params.schParams);

  BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
  mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, params.qbmmParams.dbL0C > 1,
    params.localParams.splitKNum);

  ProcessSingleBatch(params, bs, 0, true);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetScaleL2Cache(
  const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
  if (Te::Get<MNK_B>(problemShape) != 1) {
    return;
  }
  if constexpr (transB) {
    const int64_t scaleKRowBytes =
      Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
      MXFP_MULTI_BASE_SIZE;
    const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    // 0x7f: 128B cache line alignment for mx scale GM streaming
    const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0 &&
      (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
      scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
  } else {
    const int64_t scaleNStrideBytes = Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
    const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
    // 0x7f: 128B cache line alignment for mx scale GM streaming
    const bool scaleAlignForL2Stream = (scaleNStrideBytes & kCacheLineAlignMask) == 0 &&
      (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
      scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
  }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(
  const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
  TensorScaleB &gmScaleB, TensorC& gmC)
{
  const bool fullMTile = curBaseM >= Te::Get<MNK_M>(problemShape);
  if (!fullMTile) {
    return;
  }

  SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

  if constexpr (weightNz) {
    gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
  } else {
    if constexpr (transB) {
      bool bAlignForL2Stream = (Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
      gmB.SetL2CacheHint(
        bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
      bool bAlignForL2Stream =
        (Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
      gmB.SetL2CacheHint(
        bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }
  }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Init(const Params &params)
{
  if ASCEND_IS_AIV {
    return;
  }
  if (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED) {
    isBias_ = true;
  }
  ResetGmAddr(params);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ResetGmAddr(const Params &params)
{
  if ASCEND_IS_AIV {
    return;
  }
  aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
  bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
  cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
  localAGmAddr_ = reinterpret_cast<__gm__ AType*>(params.localParams.localAGmAddr);
  localScaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.localParams.localScaleAGmAddr);
  scaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
  scaleBGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
  if (isBias_) {
    biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
  }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ProcessSingleBatch(
  const Params &params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
  auto rankId = params.localParams.rankId;
  auto rankSize = params.localParams.rankSize;
  auto oriM = params.localParams.originalM;
  auto scaleKLen =
    Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
    MXFP_MULTI_BASE_SIZE;

  // 构建各 Tensor 的全局布局
  // [地址语义] gmA 是通信 buffer，按 rankSize * headMSize 行布局（每 rank 一段 headMSize）；
  //            gmALocal 是本卡 GM 的完整 A（按 rankSize * oriM 行布局）；
  //            二者尺寸不同：buffer 只装当前流水步的 headMSize 行，local A 装全 M。
  auto layoutA = MakeLayoutA{}(rankSize * Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_K>(params.problemShape));
  auto layoutALocal = MakeLayoutA{}(rankSize * oriM, Te::Get<MNK_K>(params.problemShape));
  auto layoutScaleA = MakeLayoutScaleA{}(rankSize * oriM, scaleKLen);

  auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
  auto layoutScaleB = MakeLayoutScaleB{}(rankSize * scaleKLen, Te::Get<MNK_N>(params.problemShape));
  auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
  auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

  // 创建 Tensor 句柄
  auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
  auto gmALocal = Te::MakeTensor(
    Te::MakeMemPtr<Te::Location::GM>(localAGmAddr_), layoutALocal); // 本卡 GM 的 A（rank==rankId 用）
  auto gmScaleA = Te::MakeTensor(
    Te::MakeMemPtr<Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
  auto gmScaleALocal = Te::MakeTensor(
    Te::MakeMemPtr<Te::Location::GM>(localScaleAGmAddr_), layoutScaleA);
  auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
  auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
  auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
  auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

  // 尾块更新逻辑
  auto& mTailTile = params.schParams.mTailTile;
  auto& nTailTile = params.schParams.nTailTile;
  if (needUpdateTail_ ||
    (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
              AscendC::GetBlockNum())) {
    needUpdateTail_ = true;
    bs.UpdateTailTile(mTailTile, nTailTile);
  }
  SetL2Cache(
    params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB, gmScaleB,
    gmC);

  BlockCoord blockIdx;
  int64_t mPos = 0L;
  int64_t nPos = 0L;
  constexpr int64_t kPos = 0L;
  // 遍历当前块的调度任务
  while (bs.GetTileIdx(blockIdx)) {
    BlockShape singleShape =
      bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
    if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
      return;
    }

    bs.GetTileCoord(blockIdx, mPos, nPos);
    // 切分输出块：地址基址已在外部按流水步偏移，此处仅按调度器位置切局部块
    auto gmBlockC =
      gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
    auto gmBlockBias =
      gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

    // 遍历所有 rank，在 L0C 上累加部分和
    //   - rank == rankId：A 从本卡 GM（gmALocal）读，避开通信 buffer 中未填充的本 rank 段
    //   - rank != rankId：A 从通信 buffer（gmA）读，buffer 中各 rank 段连续存放
    //   - B / ScaleB 始终从本卡 GM 读，按 rank 切 K 轴段
    //   - remoteRankCnt 控制 L0C 累加位置，最后一次（remoteRankCnt == rankSize - 1）触发 fixpipe
    uint32_t remoteRankCnt = 0;
    for (uint64_t rank = 0; rank < rankSize; rank++) {
      auto actualMPos = rank * oriM + mPos;        // local A / scaleA 的 M 偏移（按 oriM 分段）
      auto actualCommMPos = rank * Get<MNK_M>(params.problemShape) + mPos;  // 通信 buffer 的 M 偏移（按 headMSize 分段）

      // 默认从通信 buffer 读 A；rank == rankId 时改从本卡 GM 读
      auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(actualCommMPos, kPos),
          AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
      auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
          AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
      if (rank == rankId) {
        gmBlockA = gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
          AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
        gmBlockScaleA = gmScaleALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
      }

      auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
        AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
      auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos),
        AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));

      // L0C 上累加：remoteRankCnt=0 时 L0C reset，后续累加；最后一次触发 fixpipe
      mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, remoteRankCnt);
      remoteRankCnt++;
    }
  }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
