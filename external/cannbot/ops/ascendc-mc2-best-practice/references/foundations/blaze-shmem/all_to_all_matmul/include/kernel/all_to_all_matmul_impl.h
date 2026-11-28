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
// MC2 通算融合主类 —— AllToAll（通信）+ Blaze Matmul（计算）流水编排
// ----------------------------------------------------------------------------
// 该头文件是 MC2 算子的"大脑"，决定 AIV/AIC 如何协同：
//   - AIV（Vector 核）：跑 AllToAllProcess()，下发 UDMA Put，把 M 段发给所有 rank；
//   - AIC（Cube 核）：跑 MatmulProcess()，在通信 buffer 上做完整 Blaze Matmul
//                   （遍历所有 rank，在 L0C 上累加部分和）；
//   - 通过 CrossCoreSetFlag/WaitFlag 在 AIV↔AIC 间传递 buffer 就绪/释放信号；
//   - 通过 4-buffer 流水（bufferSize=4）掩盖通信延迟。
//
// 改造场景：
//   [MODIFY] N1  改通算融合类型（AllToAll+Matmul → AllReduce+Matmul 等）：
//                改 AllToAllProcess 内部 + 替换 allToAllComm_ 成员类型
//   [MODIFY] C1  改流水深度（bufferSize 4 → 8）：
//                改 host 侧 tilingData.commTilingData.bufferSize + 同步调 SHMEM 空间
//   [MODIFY] A1  改 AIC/AIV 协同顺序：改 Process() 中的 ASCEND_IS_AIV/AIC 分支
//
// [PITFALL] CrossCoreSetFlag 和 CrossCoreWaitFlag 必须按 idx 一一配对（1:1）。
//           多次 Wait 同一 flagId 会死锁；Set 了不 Wait 会让 AIV 写覆盖未读数据。
// [PITFALL] PIPE_MTE3（AIV 输出）↔ PIPE_MTE2（AIC 输入）是 AIV→AIC 的固定方向；
//           反向（AIC→AIV）用 PIPE_FIX → PIPE_MTE3。modeId 0x2 是 AIV↔AIC 跨核同步模式。
//
// 进阶细节：references/mc2_architecture.md（AIV/AIC 分工） + references/matmul_blaze.md
// ============================================================================

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "all_to_all_comm_udma.h"
#include "shmem.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "qbmm_mx_kernel.h"
#include "../tiling/quant_matmul_tiling_data.h"
#include "../tiling/all_to_all_matmul_tiling_data.h"

namespace AllToAllQuantMatmulImpl {

using namespace AscendC;
using namespace AllToAllImpl;
using namespace Blaze::Gemm;

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief All2All-Matmul 核心实现类
 * 实现思路：
 * 1. AIC 负责本地计算 (LocalMatmul) 和远程数据计算 (RemoteMatmul)。
 * 2. AIV 负责背景通信 (AllToAllProcess)，通过 UDMA 将数据分发到各卡。
 * 3. 通过流水线 (4 buffer) 掩盖通信开销。
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
class AllToAllQuantMatmulImpl {
public:
  __aicore__ inline AllToAllQuantMatmulImpl() {};

  /**
   * @brief 初始化算子状态和参数
   */
  __aicore__ inline void Init(__gm__ void* shmemSpace,
                 GM_ADDR aGM, GM_ADDR scaleAGM,
                 GM_ADDR bGM, GM_ADDR scaleBGM,
                 GM_ADDR cGM,
                 const allToAllMatmulTilingData *tilingData);
  /**
   * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
   */
  __aicore__ inline void Process();

  using TypeA = AType;
  using TypeB = BType;
  using TypeC = CType;
  using TypeScaleA = ::fp8_e8m0_t;
  using TypeScaleB = ::fp8_e8m0_t;
  static constexpr int32_t SCALE_C0 = 2;

  // Layout 定义
  using LayoutA = typename AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
  using LayoutB = typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
  using LayoutC = AscendC::Te::NDExtLayoutPtn;
  using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
    AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
    AscendC::Std::Int<SCALE_C0>>;
  using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
    AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
    AscendC::Std::Int<SCALE_C0>>;

  // 组件定义
  using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
    NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>;
  using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
  using BiasType = float;
  using LayoutBias = AscendC::Te::NDExtLayoutPtn;
  using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
  using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;

  // 参数类型
  using Params = typename QuantMatmulKernelImpl::Params;
  using BlockMmadParams = typename BlockMmad::Params;
  using L1Params = typename QuantMatmulKernelImpl::L1Params;
  using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
  using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
  using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

  QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
  __aicore__ inline void InitBaseParams(const allToAllMatmulTilingData *tilingData);
  __aicore__ inline void SetupParams(const QuantMatmulTilingData* mmTile, Params& out);
  __aicore__ inline void AllToAllProcess();  // AIV 通信任务
  __aicore__ inline void MatmulProcess();    // AIC 通算计算任务
  __aicore__ inline uint64_t CalcMOffset(uint32_t loopIdx) const;
  __aicore__ inline uint64_t GetCurMSize(uint32_t loopIdx) const;

  AllToAllComm<AType> allToAllComm_;

  GM_ADDR aGm_;
  GM_ADDR scaleAGm_;
  GM_ADDR bGm_;
  GM_ADDR scaleBGm_;
  GM_ADDR cGm_;

  uint32_t rankId_{0};
  uint64_t axisM_{0};     // M 轴总大小
  uint64_t axisKa_{0};    // K 轴大小
  uint32_t axisN_{0};     // N 轴大小
  uint32_t rankSize_{0};    // 卡数
  uint64_t commMSize_{0};   // 每次通信的 M 大小
  int32_t  commTurn_{0};     // 总流水步数
  uint64_t bufferSize_{0};   // 缓冲区大小

  uint64_t headMSize_{0};   // 标准块 M 大小
  uint32_t tileCnt_{0};     // 块数量
  uint64_t scaleKaSize_{0};   // Scale 的 K 轴字节长度

  const allToAllMatmulTilingData* tilingData_{nullptr};
};

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::Init(
  __gm__ void* shmemSpace, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
  const allToAllMatmulTilingData *tilingData)
{
  tilingData_ = tilingData;
  InitBaseParams(tilingData);
  aGm_ = aGM;
  scaleAGm_ = scaleAGM;
  bGm_ = bGM;
  scaleBGm_ = scaleBGM;
  cGm_ = cGM;

  // 初始化通信参数
  allToAllComm_.InitParams(axisM_, axisKa_, commMSize_, bufferSize_, aGM, scaleAGM, shmemSpace);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
  const allToAllMatmulTilingData *tilingData)
{
  rankSize_ = aclshmem_n_pes();
  rankId_ = aclshmem_my_pe();

  headMSize_ = tilingData->tileQbmmTilingData.m;
  tileCnt_ = tilingData->commTilingData.tileCnt;

  axisM_ = tileCnt_ * headMSize_;
  axisKa_ = tilingData->tileQbmmTilingData.k;
  axisN_ = tilingData->tileQbmmTilingData.n;
  // 计算 MXFP8 格式下 Scale 每一行的字节数
  scaleKaSize_ = CeilDiv(axisKa_, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

  commTurn_ = tileCnt_;
  commMSize_ = headMSize_;
  bufferSize_ = tilingData->commTilingData.bufferSize;
}

/**
 * @brief 设置量化矩阵乘内核参数
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::SetupParams(
  const QuantMatmulTilingData* mmTile, Params& out)
{
  ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
  BlockMmadParams mmadParams;
  // splitKNum = rankSize：遍历所有 rank 在 L0C 上累加（参与 rank 数 = rankSize）
  LocalParams localParams{rankId_, rankSize_, axisM_, aGm_, scaleAGm_, rankSize_};
  L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
    mmTile->nBufferNum};

  mmadParams.bGmAddr = bGm_;
  mmadParams.scaleBGmAddr = scaleBGm_;

  // 调度器参数
  BlockSchedulerParams schedulerParams{
    mmTile->baseM, mmTile->baseN, mmTile->mTailTile, mmTile->nTailTile,
    mmTile->mBaseTailSplitCnt, mmTile->nBaseTailSplitCnt, mmTile->mTailMain, mmTile->nTailMain};
  // 基础 Tiling 参数
  QBMMTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

  out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

/**
 * @brief AIC 负责的通算计算主流程
 *
 * wait_flag 始终在 kernel 外部、每次 kernel 调用前发出。
 * 等待 AIV 把当前 bufferId 的数据 Put 完 → 跑 Blaze Matmul（遍历所有 rank 累加）→ 释放 buffer。
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::MatmulProcess()
{
  Params params;
  SetupParams(&tilingData_->tileQbmmTilingData, params);
  for (int32_t mLoopIdx = 0; mLoopIdx < tileCnt_; ++mLoopIdx) {
    // [设计要点] bufferId 用位掩码（mLoopIdx & (bufferSize_ - 1)）做环形索引。
    //            bufferSize 必须是 2 的幂（4/8/16），否则位掩码失效。
    uint32_t bufferId = mLoopIdx & (bufferSize_ - 1);
    uint64_t mOffset = CalcMOffset(mLoopIdx);

    // 更新当前流水步的地址偏移
    params.mmadParams.aGmAddr = allToAllComm_.GetDataAddrGm(bufferId);
    params.mmadParams.scaleAGmAddr = allToAllComm_.GetScaleAddrGm() + mOffset * scaleKaSize_;
    params.mmadParams.cGmAddr = cGm_ + mOffset * axisN_ * sizeof(CType);
    params.localParams.localAGmAddr = aGm_ + mOffset * axisKa_;
    params.localParams.localScaleAGmAddr = scaleAGm_ + mOffset * scaleKaSize_;

    // 等 AIV 把当前 bufferId 的数据 Put 完
    // [PITFALL] CrossCoreWaitFlag<0x2, PIPE_MTE2> 与 AIV 侧 CrossCoreSetFlag<0x2, PIPE_MTE3>
    //           的 idx 必须一致（都是 mLoopIdx）；不一致会死锁或拿到错位数据。
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx);
    quantMatmulKernelImpl_(params);

    // 通知通信层：当前计算已完成，空间可释放（针对流水线控制）
    // [设计要点] 流水深度控制：AIC 算完第 i 轮后，AIV 才能写第 (i + bufferSize) 轮的同一 bufferId。
    //            commTurn_ > bufferSize_ 时才有"释放"概念（前 bufferSize 轮 AIV 不需要等）。
    if ((commTurn_ > bufferSize_) && (mLoopIdx < (commTurn_ - bufferSize_))) {
      CrossCoreSetFlag<0x2, PIPE_FIX>(mLoopIdx);
    }
  }
}

/**
 * @brief AIV 负责的异步通信流程
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::AllToAllProcess()
{
  // [设计要点] 先行发送全局 Scale：Scale 数据量小（M * CeilDiv(K, 64) * 2 字节），
  //            不值得做 M 轴流水，AIV 启动时一次性 Put 给所有 rank。
  allToAllComm_.PutScaleToAllRanks(0, axisM_);
  for (uint32_t mLoopIdx = 0; mLoopIdx < commTurn_; ++mLoopIdx) {
    uint32_t bufferId = mLoopIdx & (bufferSize_ - 1);
    uint64_t curMSize = GetCurMSize(mLoopIdx);
    uint64_t mOffset = CalcMOffset(mLoopIdx);

    // 流水线深度控制，防止覆盖尚未被 AIC 读取的旧数据
    // [PITFALL] 必须先 WaitFlag（等 AIC 算完）再 BarrierAll（等其他卡释放），
    //           顺序反了会让本卡 AIV 覆盖未读数据。
    if (mLoopIdx >= bufferSize_) {
      CrossCoreWaitFlag<0x2, PIPE_MTE3>(mLoopIdx - bufferSize_);
      // 确保所有卡释放了当前的buffer，才能开始写入下一个数据块
      allToAllComm_.BarrierAll();
    }

    // 分发当前数据块
    allToAllComm_.PutToAllRanks(mOffset, curMSize, bufferId);

    // 设置 Flag，通知 AIC 数据已就绪
    // [PITFALL] CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx) 与 AIC 侧
    //           CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx) 的 idx 必须一致。
    CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx);
  }
}

/**
 * @brief 算子执行入口
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::Process()
{
  // [设计要点] 同一份 kernel 二进制同时跑在 AIV 和 AIC 上，靠编译期分支隔离。
  //            ASCEND_IS_AIV / ASCEND_IS_AIC 是 bisheng 编译器提供的宏。
  //            AIV 跑通信，AIC 跑通算计算，二者通过 CrossCore Flag 同步。
  if ASCEND_IS_AIV {
    AllToAllProcess(); // AIV 通信
  }

  if ASCEND_IS_AIC {
    MatmulProcess();    // AIC 通算计算（带同步等待）
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::CalcMOffset(uint32_t loopIdx) const
{
  return loopIdx * headMSize_;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t AllToAllQuantMatmulImpl<AType, BType, CType, TransA, TransB>::GetCurMSize(uint32_t loopIdx) const
{
  return headMSize_;
}

} // namespace AllToAllQuantMatmulImpl

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E4M3(__gm__ void* shmemSpace,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulImpl::AllToAllQuantMatmulImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(shmemSpace, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E5M2(__gm__ void* shmemSpace,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulImpl::AllToAllQuantMatmulImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(shmemSpace, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E5M2(__gm__ void* shmemSpace,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulImpl::AllToAllQuantMatmulImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(shmemSpace, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E4M3(__gm__ void* shmemSpace,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulImpl::AllToAllQuantMatmulImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(shmemSpace, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}
