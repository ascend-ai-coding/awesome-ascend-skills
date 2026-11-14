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
// MC2 通信层 —— AllToAll over UDMA（URMA 协议）
// ----------------------------------------------------------------------------
// 该头文件实现"所有 rank 互发 M 段数据"的 AllToAll 语义，承载通算融合中
// "通"的部分。所有跨卡通信走 SHMEM/UDMA，**禁止使用 HCCL 高阶 API**
// （Hccl::AllReduce / Hccl::AllGather / ... 禁止清单见 references/foundations/blaze-shmem/comm_shmem.md §5）。
//
// 改造场景：
//   [MODIFY] N1  改通信原语（AllToAll → AllReduce/ReduceScatter/AllGather）：
//                重写 PutToAllRanks / 新增 ReduceBuffer 等，参考 references/comm_shmem.md §4
//   [MODIFY] C1  改 Put 策略（如只发给特定 rank）：改 PutToAllRanks 的 Block 分配
//   [MODIFY] C2  改 SHMEM 空间布局：改 GetDataAddrGm / GetScaleAddrGm 的偏移计算
//
// [PITFALL] aclshmemx_udma_put_nbi 的 dst 是"本 rank 视角下本 rank 的 SHMEM 地址"，
//           不是 remoteRank 的地址。SHMEM 内部做地址翻译。详见 comm_shmem.md §2。
// [PITFALL] aclshmemx_udma_quiet(remoteRank) 必须在每次 Put 后调用，否则不保序。
// [PITFALL] BarrierAll 用 aclshmem_barrier_all()，不能用 HCCL 同步原语。
// [PITFALL] 每个 Block 负责发往对应的 remoteRank（不是单 Block 串行）——多 Block 并发
//           才能让 UDMA 引擎并行下发，避免单 Block 成瓶颈。
//
// 进阶细节：references/comm_shmem.md
// ============================================================================

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "blaze/gemm/utils/common_utils.h"
#include "shmem.h"

namespace AllToAllImpl {

using namespace AscendC;
using namespace Blaze::Gemm;

/**
 * @brief AllToAll 通信管理类，负责通过 UDMA 协议在多卡间交换数据和量化系数（Scale）
 *
 * @tparam XType 数据类型（通常为 fp8 或其他量化类型）
 */
template<typename XType>
class AllToAllComm {
public:
  __aicore__ inline AllToAllComm() {};

  /**
   * @brief 初始化通信参数
   *
   * @param M 输入矩阵 A 的 M 轴总大小（单卡对应的行数）
   * @param kPerRank 每个 Rank 负责的 K 轴长度
   * @param commMSize 每次通信切分的 M 轴块大小（用于流水线）
   * @param bufferSize 流水线 buffer 数（典型为 4）
   * @param inputDataAddr 输入数据 A 的 GM 地址
   * @param inputScaleAddr 输入量化系数 ScaleA 的 GM 地址
   * @param shmemSpace 共享内存空间指针，用于跨卡 UDMA 访问
   */
  __aicore__ inline void InitParams(uint64_t M, uint64_t kPerRank, uint64_t commMSize,
    uint64_t bufferSize,GM_ADDR inputDataAddr, GM_ADDR inputScaleAddr, __gm__ void* shmemSpace);

  /**
   * @brief 将当前 Rank 的数据段分发到所有其他 Rank
   *
   * @param srcMOffset 当前数据段在本地 M 轴的起始偏移
   * @param mSize 当前传输的 M 轴块行数
   * @param bufferId bufferId 缓存索引
   */
  __aicore__ inline void PutToAllRanks(uint64_t srcMOffset, uint64_t mSize, uint32_t bufferId);

  /**
   * @brief 将当前 Rank 的量化系数（Scale）分发到所有其他 Rank
   *
   * @param srcMOffset 起始偏移
   * @param mSize 传输行数
   */
  __aicore__ inline void PutScaleToAllRanks(uint64_t srcMOffset, uint64_t mSize);

  /**
   * @brief 全局屏障同步，确保所有异步通信任务完成
   */
  __aicore__ inline void BarrierAll();

  /**
   * @brief 获取指定 bufferId 索引的数据 GM 地址（在共享内存中）
   */
  __aicore__ inline GM_ADDR GetDataAddrGm(uint32_t bufferId);

  /**
   * @brief 获取 Scale 数据在共享内存中的 GM 地址
   */
  __aicore__ inline GM_ADDR GetScaleAddrGm();

private:
  /**
   * @brief 将特定数据段 Put 到远程指定的 Rank
   */
  __aicore__ inline void PutSegmentToRank(uint32_t remoteRank, uint64_t srcMOffset,
    uint64_t mSize, uint32_t bufferId);

  /**
   * @brief 将特定 Scale 段 Put 到远程指定的 Rank
   */
  __aicore__ inline void PutScaleToRank(uint32_t remoteRank, uint64_t srcMOffset,
    uint64_t mSize);

  uint64_t rankSize_{0};    // 总卡数
  uint64_t M_{0};       // 单卡负责的总行数
  uint64_t commMSize_{0};   // 通信切块行数
  uint32_t rankId_{0};    // 当前卡号
  uint64_t bufferSize_;

  uint64_t bytesPerMRow_{0};     // 数据每行的字节数
  uint64_t scaleBytesPerMRow_{0};  // Scale 每行的字节数
  uint64_t bufferBlockSize_{0};  // 单个 bufferId 块的总字节数（rankSize * commMSize * bytesPerMRow）

  GM_ADDR inputDataAddr_{nullptr};
  GM_ADDR inputScaleAddr_{nullptr};
  __gm__ void* shmemContextGM_{nullptr};
};

template<typename XType>
__aicore__ inline void AllToAllComm<XType>::InitParams(
  uint64_t M, uint64_t kPerRank, uint64_t commMSize, uint64_t bufferSize,
  GM_ADDR inputDataAddr, GM_ADDR inputScaleAddr, __gm__ void* shmemSpace)
{
  shmemContextGM_ = shmemSpace;
  bufferSize_ = bufferSize;
  inputDataAddr_ = inputDataAddr;
  inputScaleAddr_ = inputScaleAddr;

  rankSize_ = aclshmem_n_pes();
  rankId_ = aclshmem_my_pe();
  M_ = M;
  commMSize_ = commMSize;

  bytesPerMRow_ = kPerRank * sizeof(XType);
  // 计算 MXFP8 格式下 Scale 的字节数（按块量化）
  scaleBytesPerMRow_ = CeilDiv(kPerRank, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
  // 单个 bufferId 缓存块承载所有卡发送过来的当前流水步的数据
  bufferBlockSize_ = rankSize_ * commMSize_ * bytesPerMRow_;
}

template<typename XType>
__aicore__ inline void AllToAllComm<XType>::PutToAllRanks(uint64_t srcMOffset, uint64_t mSize,
  uint32_t bufferId)
{
  // [设计要点] 每个 Block 负责发送到对应的 remoteRank（Block i → remoteRank i）。
  // 这样 UDMA 引擎能并行下发多条 Put，避免单 Block 串行 Put 所有 rank 成为瓶颈。
  // [MODIFY C1] 若改为 ReduceScatter（只发给一个目标 rank），改为：
  if ((AscendC::GetBlockIdx() < rankSize_) && (AscendC::GetBlockIdx() != rankId_)) {
    PutSegmentToRank(AscendC::GetBlockIdx(), srcMOffset, mSize, bufferId);
  }
  // 等待所有卡完成 Put 任务
  BarrierAll();
}

template<typename XType>
__aicore__ inline void AllToAllComm<XType>::PutScaleToAllRanks(uint64_t srcMOffset, uint64_t mSize)
{
  if ((AscendC::GetBlockIdx() < rankSize_) && (AscendC::GetBlockIdx() != rankId_)) {
    PutScaleToRank(AscendC::GetBlockIdx(), srcMOffset, mSize);
  }
  // 注意：PutScaleToAllRanks的同步由 PutToAllRanks 中的 aclshmemx_udma_quiet 保证
}

template<typename XType>
__aicore__ inline void AllToAllComm<XType>::PutSegmentToRank(uint32_t remoteRank, uint64_t srcMOffset,
  uint64_t mSize, uint32_t bufferId)
{
  // 远程 Rank 的目标窗口地址
  // [地址语义] remoteWinAddr 是"本 rank 视角下本 rank 的 SHMEM 基地址"，对应 bufferId 块。
  //            UDMA 引擎会把这个地址映射到 remoteRank 的 SHMEM；本 rank 不需要知道对端物理地址。
  GM_ADDR remoteWinAddr = GetDataAddrGm(bufferId);

  // [地址语义] dstDataOffset 用 rankId_（自己的 rank）：
  //            对端按"我从哪个 rank 来"放到对应位置，避免多 rank 数据互相覆盖。
  //            例：rank 2 发给 rank 0/1/3，对端都在 rankId_ * commMSize 偏移处收 rank 2 的数据。
  uint64_t dstDataOffset = rankId_ * commMSize_ * bytesPerMRow_;
  // [地址语义] srcDataOffset 用 remoteRank：
  //            本 rank 的 A 矩阵逻辑上按 remoteRank 分段，第 i 段是要发给 remoteRank=i 的数据。
  //            srcMOffset 是当前流水步的块内偏移。
  uint64_t srcDataOffset = remoteRank * M_ * bytesPerMRow_ + srcMOffset * bytesPerMRow_;
  uint64_t dataSize = mSize * bytesPerMRow_;

  // 非阻塞 UDMA 发送
  // [PITFALL] ubuf 参数（第三个）在 MC2 大块 Put 场景固定为 nullptr；
  //            小消息聚合场景才需要 UB 中转，本工程不涉及。
  aclshmemx_udma_put_nbi(
    remoteWinAddr + dstDataOffset,
    inputDataAddr_ + srcDataOffset,
    (__ubuf__ uint8_t*)nullptr,
    dataSize,
    remoteRank
  );
  // 确保对该远程 Rank 的传输已完成（数据已到达对端 PE）
  // [PITFALL] aclshmemx_udma_quiet 确保数据到达对端，不是"只下发不等对端"。
  aclshmemx_udma_quiet(remoteRank);
}

template<typename XType>
__aicore__ inline void AllToAllComm<XType>::PutScaleToRank(uint32_t remoteRank, uint64_t srcMOffset, uint64_t mSize)
{
  GM_ADDR remoteScaleAddr = GetScaleAddrGm();

  // Scale 空间不使用 bufferId，通常直接一次性或同步流水发送
  uint64_t dstScaleOffset = rankId_ * M_ * scaleBytesPerMRow_;
  uint64_t srcScaleOffset = remoteRank * M_ * scaleBytesPerMRow_ + srcMOffset * scaleBytesPerMRow_;
  uint64_t scaleSize = mSize * scaleBytesPerMRow_;

  aclshmemx_udma_put_nbi(
    remoteScaleAddr + dstScaleOffset,
    inputScaleAddr_ + srcScaleOffset,
    (__ubuf__ uint8_t*)nullptr,
    scaleSize,
    remoteRank
  );
}

template<typename XType>
__aicore__ inline void AllToAllComm<XType>::BarrierAll()
{
  // 全卡同步等待
  aclshmem_barrier_all();
}

template<typename XType>
__aicore__ inline GM_ADDR AllToAllComm<XType>::GetDataAddrGm(uint32_t bufferId)
{
  // 获取共享内存在本地卡对应的 GM 基地址
  // [PITFALL] aclshmem_ptr 第二个参数是 rankId_（**本卡自己**），不是 remoteRank！
  //           本卡视角下本卡的 SHMEM 基地址，就是其他卡 Put 过来的数据落点。
  // [MODIFY C2] 若改 SHMEM 布局（如新增 bias 段），偏移计算同步改。
  GM_ADDR baseAddr = (GM_ADDR)(aclshmem_ptr(shmemContextGM_, rankId_));
  return baseAddr + bufferId * bufferBlockSize_;
}

template<typename XType>
__aicore__ inline GM_ADDR AllToAllComm<XType>::GetScaleAddrGm()
{
  // Scale 存放在数据 bufferId 缓存之后
  // [设计要点] Scale 不参与 M 轴流水（数据量小），所有 rank 一次性 Put 后存在这里。
  //            布局：[Data buffers (bufferSize 个)][Scale buffers]
  return (GM_ADDR)(aclshmem_ptr(shmemContextGM_, rankId_)) + bufferSize_ * bufferBlockSize_;
}

} // namespace AllToAllImpl
