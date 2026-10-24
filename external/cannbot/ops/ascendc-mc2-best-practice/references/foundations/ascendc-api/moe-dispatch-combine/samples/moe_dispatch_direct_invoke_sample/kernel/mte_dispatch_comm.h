/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mte_dispatch_comm.h
 * \brief
 */
#ifndef MOE_DISPATCH_MTE_COMM_H
#define MOE_DISPATCH_MTE_COMM_H

#if __has_include("common/op_kernel/moe_distribute_base.h")
#include "common/op_kernel/moe_distribute_base.h"
#else
#include "moe_dispatch_base_compat.h"
#endif
#include "utils.h"
#include "basic_api/kernel_basic_intf.h"

namespace MoeDispatchMTECommImpl {

using namespace AscendC;
using namespace MoeDispatchImpl;

/**
 * @brief 通信侧常量定义
 *
 * 当前 sample 只保留 dispatch 最小闭环：
 * - payload 按 32B 对齐
 * - 状态区布局：先分 expert，再分 srcRank，每个 slot 32B 对齐（8 个 int32_t）
 * - 每个 slot: [flag, token_count, 0, 0, 0, 0, 0, 0]
 * - 通信 token 末尾固定携带 3 个 int32 的 triple
 */
constexpr static uint32_t UB_ALIGN_BYTES = 32U;
constexpr static uint32_t STATE_SLOT_INT32 = 8U;      // 每个 slot 8 个 int32_t = 32B 对齐
constexpr static uint32_t STATE_READY_OFFSET = 0;     // flag 在 slot 中的偏移
constexpr static uint32_t STATE_TOKEN_OFFSET = 1;     // token_count 在 slot 中的偏移

template<typename DataType>
class MteDispatchComm {
public:
    __aicore__ inline MteDispatchComm() {};
    __aicore__ inline void InitHcclContextByAddr(GM_ADDR mc2Context, uint32_t rankDim);
    __aicore__ inline void InitDispatchWindow(uint64_t bs, uint64_t h, uint64_t localExpertNum);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void ClearLocalStatus();
    __aicore__ inline void SetRemoteStatus(const uint32_t *expertCounts);
    __aicore__ inline void WaitRemoteStatus(uint32_t *recvCounts, uint32_t *expertCounts);
    __aicore__ inline GM_ADDR GetDispatchDataAddr(uint32_t dstRank, uint32_t srcRank, uint32_t localExpert, uint32_t slot) const;
    __aicore__ inline GM_ADDR GetLocalWindowDataAddr(uint32_t srcRank, uint32_t localExpert, uint32_t slot) const;

    __gm__ Mc2Kernel::HcclOpParam *hcclContext_ {nullptr};
    uint32_t rankDim_ {0};
    uint32_t rankId_ {0};
    uint32_t payloadBytes_ {0};
    uint32_t tokenMetaOffset_ {0};
    uint32_t commTokenBytes_ {0};
    uint32_t commTokenElems_ {0};
    uint32_t localExpertNum_ {0};
    uint64_t sourceRankRegionBytes_ {0};
    uint64_t expertRegionBytes_ {0};
    uint32_t bs_ {0};
    uint32_t h_ {0};

private:
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId) const;
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId) const;
    __aicore__ inline GM_ADDR GetDispatchStateAddr(uint32_t dstRank, uint32_t srcRank, uint32_t localExpert) const;

    TBuf<> stateWriteBuf_;
    TBuf<> stateReadBuf_;
    TBuf<> stateResetBuf_;
    LocalTensor<int32_t> stateResetTensor_;
};

/**
 * @brief 使用 host 传入的 mc2Context 初始化通信上下文
 *
 * 当前 sample 不再直接调用 GetHcclContext，而是复用 host 侧
 * HcclAllocComResourceByTiling 创建好的通信资源上下文。
 */
template<typename DataType>
__aicore__ inline void MteDispatchComm<DataType>::InitHcclContextByAddr(GM_ADDR mc2Context, uint32_t rankDim)
{
    hcclContext_ = (__gm__ Mc2Kernel::HcclOpParam*)mc2Context;
    rankDim_ = rankDim;
    rankId_ = Mc2Kernel::GetRankId(hcclContext_);
}

/**
 * @brief 固化 dispatch window 的最小布局参数
 *
 * 当前 sample 的窗口排布是：
 * 1. 先按 source rank 分区
 * 2. 每个 source rank 下再按 local expert 分区
 * 3. 每个 expert 分区内按 slot 顺序连续存放 comm token
 *
 * comm token 布局是：
 * - 对齐后的 payload
 * - 末尾 triple: [srcRank, srcTokenIdx, srcTopkIdx]
 */
template<typename DataType>
__aicore__ inline void MteDispatchComm<DataType>::InitDispatchWindow(uint64_t bs, uint64_t h, uint64_t localExpertNum)
{
    bs_ = static_cast<uint32_t>(bs);
    h_ = static_cast<uint32_t>(h);
    localExpertNum_ = static_cast<uint32_t>(localExpertNum);
    payloadBytes_ = static_cast<uint32_t>(AlignUp(h_ * sizeof(DataType), UB_ALIGN_BYTES));
    tokenMetaOffset_ = payloadBytes_ / sizeof(int32_t);
    commTokenBytes_ = payloadBytes_ + 3 * sizeof(int32_t);
    commTokenElems_ = commTokenBytes_ / sizeof(DataType);
    expertRegionBytes_ = static_cast<uint64_t>(bs_) * commTokenBytes_;
    sourceRankRegionBytes_ = static_cast<uint64_t>(localExpertNum_) * expertRegionBytes_;
}

/**
 * @brief 初始化状态读写所需的 UB buffer
 *
 * - stateWriteBuf_：写远端状态时使用
 * - stateReadBuf_：轮询本地状态区时使用
 * - stateResetBuf_：状态消费完成后重置本地状态区
 */
template<typename DataType>
__aicore__ inline void MteDispatchComm<DataType>::InitBuffer(TPipe *tPipe)
{
    // 新的状态区布局：先分 expert，再分 srcRank，每个 slot 8 个 int32_t (32B 对齐)
    // 总大小 = localExpertNum * rankDim_ * STATE_SLOT_INT32
    uint32_t totalStateElems = localExpertNum_ * rankDim_ * STATE_SLOT_INT32;
    // 写入时只写一个 slot
    tPipe->InitBuffer(stateWriteBuf_, STATE_SLOT_INT32 * sizeof(int32_t));
    // 读取时需要读取所有 expert 和 rank 的状态
    tPipe->InitBuffer(stateReadBuf_, totalStateElems * sizeof(int32_t));
    // 重置时需要重置所有状态
    tPipe->InitBuffer(stateResetBuf_, totalStateElems * sizeof(int32_t));
    stateResetTensor_ = stateResetBuf_.Get<int32_t>();
    Duplicate<int32_t>(stateResetTensor_, 0, totalStateElems);  // 0 表示未就绪
}

/**
 * @brief 获取指定 rank 的数据窗口起始地址
 */
template<typename DataType>
__aicore__ inline GM_ADDR MteDispatchComm<DataType>::GetWinDataAddrGm(uint32_t rankId) const
{
    return Mc2Kernel::GetBaseWindAddrByRankId(hcclContext_, rankId, rankId_);
}

/**
 * @brief 获取指定 rank 的状态窗口起始地址
 */
template<typename DataType>
__aicore__ inline GM_ADDR MteDispatchComm<DataType>::GetWinStatusAddrGm(uint32_t rankId) const
{
    return Mc2Kernel::GetBaseWindStateAddrByRankId(hcclContext_, rankId, rankId_);
}

/**
 * @brief 计算一个 dispatch token 在目标 rank window 中的地址
 *
 * 地址布局公式：
 * rankDataBase + sourceRankOffset + expertOffset + tokenOffset
 */
template<typename DataType>
__aicore__ inline GM_ADDR MteDispatchComm<DataType>::GetDispatchDataAddr(
    uint32_t dstRank, uint32_t srcRank, uint32_t localExpert, uint32_t slot) const
{
    return GetWinDataAddrGm(dstRank) + static_cast<uint64_t>(srcRank) * sourceRankRegionBytes_ +
           static_cast<uint64_t>(localExpert) * expertRegionBytes_ +
           static_cast<uint64_t>(slot) * commTokenBytes_;
}

/**
 * @brief 计算指定状态 slot 在目标 rank 状态区中的地址
 *
 * 状态区布局：先分 expert，再分 srcRank，每个 slot 32B 对齐
 * slot 地址 = GetWinStatusAddrGm(dstRank) + (expert * rankDim_ + srcRank) * STATE_SLOT_INT32 * sizeof(int32_t)
 */
template<typename DataType>
__aicore__ inline GM_ADDR MteDispatchComm<DataType>::GetDispatchStateAddr(
    uint32_t dstRank, uint32_t srcRank, uint32_t localExpert) const
{
    // 先分 expert，再分 srcRank
    return GetWinStatusAddrGm(dstRank) +
           (static_cast<uint64_t>(localExpert) * rankDim_ + static_cast<uint64_t>(srcRank)) * 
           STATE_SLOT_INT32 * sizeof(int32_t);
}

/**
 * @brief 获取当前 rank 本地接收窗口中的 token 地址
 *
 * 这是回搬阶段的快捷接口，本质上等价于把 dstRank 固定成当前 rank。
 */
template<typename DataType>
__aicore__ inline GM_ADDR MteDispatchComm<DataType>::GetLocalWindowDataAddr(
    uint32_t srcRank, uint32_t localExpert, uint32_t slot) const
{
    return GetDispatchDataAddr(rankId_, srcRank, localExpert, slot);
}

/**
 * @brief 清空当前 rank 的状态区
 *
 * 状态区使用 0 作为"未就绪"标记。
 * 每一轮 dispatch 结束后都需要重新清状态，避免下一轮直接读到旧值。
 * 当前 sample 默认单核独占消费整块状态区，因此可以直接整块清零。
 * 多核后若状态等待按段切分，则清理动作也必须按段或按统一收口时机执行，
 * 不能在某个核刚消费完本段后就把整块状态区提前清空。
 */
template<typename DataType>
__aicore__ inline void MteDispatchComm<DataType>::ClearLocalStatus()
{
    GlobalTensor<int32_t> selfStateTensor;
    selfStateTensor.SetGlobalBuffer((__gm__ int32_t*)Mc2Kernel::GetStatusDataSpaceGm(hcclContext_));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(localExpertNum_ * rankDim_ * STATE_SLOT_INT32 * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(selfStateTensor, stateResetTensor_, copyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}


/**
 * @brief 向每个目标 rank 发布"当前 source rank 实际发送数量"
 *
 * 状态区布局：先分 expert，再分 srcRank，每个 slot 32B 对齐
 * slot = [flag, token_count, 0, 0, 0, 0, 0, 0]
 * 写入顺序：先写 token 数量，最后写 flag = 1
 * 当前 sample 默认由单核写完所有 `(dstRank, expert)` 状态槽。
 * 多核后必须先明确每个状态槽由哪个核负责发布，或先做跨核汇总后再由指定核统一发布；
 * 否则多个核沿用这段逻辑会重复写同一状态槽。
 */
template<typename DataType>
__aicore__ inline void MteDispatchComm<DataType>::SetRemoteStatus(const uint32_t *expertCounts)
{
    PipeBarrier<PIPE_ALL>();
    
    for (uint32_t dstRank = 0; dstRank < rankDim_; ++dstRank) {
        for (uint32_t exp = 0; exp < localExpertNum_; ++exp) {
            // 准备 slot 数据：[flag=0, token_count, 0, 0, 0, 0, 0, 0]
            LocalTensor<int32_t> slotTensor = stateWriteBuf_.Get<int32_t>();
            Duplicate<int32_t>(slotTensor, 0, STATE_SLOT_INT32);
            SyncFunc<AscendC::HardEvent::V_S>();
            
            // 写入 token_count（索引 1）
            uint32_t tokenCount = expertCounts[dstRank * localExpertNum_ + exp];
            slotTensor.SetValue(STATE_TOKEN_OFFSET, static_cast<int32_t>(tokenCount));
            
            // 最后写入 flag = 1（索引 0）
            slotTensor.SetValue(STATE_READY_OFFSET, 1);

            // 计算 slot 地址：先分 expert，再分 srcRank
            __gm__ int32_t *stateAddr = (__gm__ int32_t*)GetDispatchStateAddr(dstRank, rankId_, exp);
            GlobalTensor<int32_t> remoteStateTensor;
            remoteStateTensor.SetGlobalBuffer(stateAddr);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(STATE_SLOT_INT32 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(remoteStateTensor, slotTensor, copyParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
    }
    PipeBarrier<PIPE_ALL>();
}

/**
 * @brief 轮询本地状态区，直到所有 source rank 的所有 expert 的状态都就绪
 *
 * 状态区布局：先分 expert，再分 srcRank，每个 slot 32B 对齐
 * slot = [flag, token_count, 0, 0, 0, 0, 0, 0]
 * 返回值通过 recvCounts 输出，表示"当前 rank 从各 source rank 应该消费多少个 token"。
 * 返回值通过 expertCounts 输出，表示"当前 rank 的各 local expert 从各 source rank 应该消费多少个 token"。
 * 当前 sample 读的是整块状态区，所以返回后 `recvCounts` / `expertCounts` 就是完整结果。
 * 多核后若等待阶段只轮询自己负责的状态段，则这两个输出也会退化成局部结果，
 * 只能在本段消费、局部统计或后续汇总中使用，不能默认当作整卡全量结果继续跨阶段复用。
 */
template<typename DataType>
__aicore__ inline void MteDispatchComm<DataType>::WaitRemoteStatus(uint32_t *recvCounts, uint32_t *expertCounts)
{
    __gm__ int32_t *statusAddr = (__gm__ int32_t*)Mc2Kernel::GetStatusDataSpaceGm(hcclContext_);
    GlobalTensor<int32_t> selfStateTensor;
    selfStateTensor.SetGlobalBuffer(statusAddr);
    LocalTensor<int32_t> stateTensor = stateReadBuf_.Get<int32_t>();
    bool ready = false;
    
    while (!ready) {
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        uint32_t totalStateBytes = localExpertNum_ * rankDim_ * STATE_SLOT_INT32 * sizeof(int32_t);
        DataCopyExtParams copyParams{1, totalStateBytes, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(stateTensor, selfStateTensor, copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        
        ready = true;
        for (uint32_t srcRank = 0; srcRank < rankDim_; ++srcRank) {
            uint32_t totalTokens = 0;
            for (uint32_t exp = 0; exp < localExpertNum_; ++exp) {
                uint32_t slotOffset = (exp * rankDim_ + srcRank) * STATE_SLOT_INT32;
                int32_t flag = stateTensor.GetValue(slotOffset + STATE_READY_OFFSET);
                int32_t tokenCount = stateTensor.GetValue(slotOffset + STATE_TOKEN_OFFSET);
                
                if (flag != 1) {
                    ready = false;
                    break;
                }
                expertCounts[srcRank * localExpertNum_ + exp] = static_cast<uint32_t>(tokenCount);
                totalTokens += static_cast<uint32_t>(tokenCount);
            }
            if (!ready) {
                break;
            }
            recvCounts[srcRank] = totalTokens;
        }
    }
    ClearLocalStatus();
}

} // namespace MoeDispatchMTECommImpl

#endif // MOE_DISPATCH_MTE_COMM_H