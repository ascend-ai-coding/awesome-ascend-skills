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
 * \file moe_dispatch.h
 * \brief
 */
#ifndef MOE_DISPATCH_KERNEL_H
#define MOE_DISPATCH_KERNEL_H

#include "adv_api/reduce/sum.h"
#include "tiling_data.h"
#include "utils.h"
#include "mte_dispatch_comm.h"
#include "basic_api/kernel_basic_intf.h"

namespace MoeDispatchImpl {

using namespace AscendC;
using namespace MoeDispatchMTECommImpl;

/**
 * @brief MoE dispatch 最小闭环 sample
 *
 * 功能：
 * 1. 根据 expertIds 计算 token 的目标 rank / local expert / slot
 * 2. 将 token 打包为 payload + triple，写入目标 rank window
 * 3. 发布发送数量到远端状态区
 * 4. 等待所有 source rank 的状态就绪
 * 5. 从本地接收窗口回搬 expandX / expandIdx / expertTokenNums / epRecvCounts
 *
 * 当前 sample 的限制：
 * - 只保留 EP-only 路径
 * - topK 输入为 1
 * - 每个 rank 只有 1 个 local expert
 * - 不覆盖量化、shared expert、TP 等复杂分支
 */
template<typename DataType>
class MoeDispatchOp {
public:
    struct RouteInfo {
        uint32_t tokenIdx {0};
        uint32_t topkIdx {0};
        uint32_t expertId {0};
        uint32_t dstRank {0};
        uint32_t localExpert {0};
        uint32_t dstSlot {0};
    };

    __aicore__ inline MoeDispatchOp() {};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandX, GM_ADDR expandIdx,
        GM_ADDR expertTokenNums, GM_ADDR epRecvCounts, GM_ADDR tilingGM, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline RouteInfo DecodeRoute(uint32_t linearIdx);
    __aicore__ inline uint32_t CalcCurExpertCnt(uint32_t dstExpertId, uint32_t linearIdx);
    __aicore__ inline void InitBlockRange(uint32_t totalTokens, uint32_t usedBlockNum);
    __aicore__ inline void FillTriple(LocalTensor<DataType> &commTokenLocal, const RouteInfo &route);
    __aicore__ inline void WritePackedToken(const RouteInfo &route, GlobalTensor<DataType> &dstTensor);
    __aicore__ inline void SendTokens();
    __aicore__ inline void WriteStatus();
    __aicore__ inline void WaitAllSources();
    __aicore__ inline void CopyWindowToOutputs();
    __aicore__ inline void FinalizeOutputs();
    __aicore__ inline void ResetCounts();
    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt);

    __gm__ MoeDispatchTilingData *tilingData_ {nullptr};
    MteDispatchComm<DataType> comm_;

    GlobalTensor<DataType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<DataType> expandXGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsGMTensor_;
    GlobalTensor<int32_t> epRecvCountsGMTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenQueue_;
    TBuf<> expertIdsBuf_;
    TBuf<> dstExpertBuf_;
    TBuf<> diffExpertBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> epRecvCountsOutBuf_;
    TBuf<> expertTokenNumsOutBuf_;
    LocalTensor<int32_t> expertIdsLocal_;
    LocalTensor<int32_t> dstExpertLocal_;
    LocalTensor<int32_t> diffExpertLocal_;
    LocalTensor<int32_t> epRecvCountsOutLocal_;
    LocalTensor<int64_t> expertTokenNumsOutLocal_;
    LocalTensor<float> workLocalTensor_;

    uint32_t bs_ {0};
    uint32_t h_ {0};
    uint32_t topK_ {1};
    uint32_t epWorldSize_ {0};
    uint32_t epRankId_ {0};
    uint32_t moeExpertNum_ {0};
    uint32_t localExpertNum_ {1};
    uint32_t maxRecvTokens_ {0};
    uint32_t blockStart_ {0};
    uint32_t blockEnd_ {0};
    uint32_t blockIdx_ {0};
    uint32_t blockNum_ {1};
    uint32_t recvCounts_[MAX_RANK_NUM] {0};
    uint32_t expertSendCounts_[MAX_RANK_NUM * MAX_EXPERT_NUM] {0};  // [dstRank * localExpertNum + expert]
    uint32_t expertRecvCounts_[MAX_RANK_NUM * MAX_EXPERT_NUM] {0};  // [srcRank * localExpertNum + expert]
};

/**
 * @brief 计算 ReduceSum 所需的临时 workspace 大小
 */
template<typename DataType>
__aicore__ inline int32_t MoeDispatchOp<DataType>::ReduceSumWorkNeedSize(int32_t calCnt)
{
    int32_t elementsPerBlock = 32 / sizeof(int32_t);
    return ((calCnt + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
}

/**
 * @brief 重置发送计数、接收计数和 expert token 统计
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::ResetCounts()
{
    for (uint32_t i = 0; i < MAX_RANK_NUM; ++i) {
        recvCounts_[i] = 0;
    }

    for (uint32_t i = 0; i < MAX_RANK_NUM * MAX_EXPERT_NUM; ++i) {
        expertSendCounts_[i] = 0;
        expertRecvCounts_[i] = 0;
    }
}

/**
 * @brief 将线性 token 下标翻译成 dispatch 路由信息
 *
 * 当前 sample 输入的 topK=1，因此 linearIdx 与 tokenIdx 一一对应。
 * `expertIds` 是只读路由输入；多核化时，每个核都可以保留“读取完整 expertIds 视图”的做法。
 * 这类只读输入允许重复读取或完整装载到 UB，风险点不在于“每核都看到了它”，而在于后续阶段是否把局部统计误当成全局统计。
 */
template<typename DataType>
__aicore__ inline typename MoeDispatchOp<DataType>::RouteInfo MoeDispatchOp<DataType>::DecodeRoute(uint32_t linearIdx)
{
    RouteInfo route;
    route.tokenIdx = linearIdx / topK_;
    route.topkIdx = linearIdx % topK_;
    route.expertId = static_cast<uint32_t>(expertIdsLocal_.GetValue(route.tokenIdx * topK_ + route.topkIdx));
    route.dstRank = route.expertId / localExpertNum_;
    route.localExpert = route.expertId % localExpertNum_;
    return route;
}

/**
 * @brief 按 blockIdx 为当前 block 划分处理区间 `[blockStart_, blockEnd_)`
 *
 * 前 `usedBlockNum` 个 block 均分 `totalTokens`，多出来的余数优先分给前几个 block。
 * 若 `blockIdx_ >= usedBlockNum`，则当前 block 拿到空区间。
 *
 * 当前 sample 固定传 `usedBlockNum = 1`，因此只有 block 0 处理全部 token，
 * 其他 block 直接跳过数据路径。
 * 当处理多核场景时，可能不同的函数对分核的方式有不同需求，不能只使用通过 token 控制的 blockStart_ 和 blockEnd_ 来控制数据处理区间，而是需要在每个函数内部根据实际需求进行分核策略的设计和实现。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::InitBlockRange(uint32_t totalTokens, uint32_t usedBlockNum)
{
    if (blockIdx_ >= usedBlockNum) {
        blockStart_ = totalTokens;
        blockEnd_ = totalTokens;
        return;
    }

    uint32_t tokenNumPerBlock = totalTokens / usedBlockNum;
    uint32_t remainderTokenNum = totalTokens % usedBlockNum;
    blockStart_ = tokenNumPerBlock * blockIdx_;
    if (blockIdx_ < remainderTokenNum) {
        blockStart_ += blockIdx_;
        blockEnd_ = blockStart_ + tokenNumPerBlock + 1;
    } else {
        blockStart_ += remainderTokenNum;
        blockEnd_ = blockStart_ + tokenNumPerBlock;
    }
}

/**
 * @brief 计算当前 token 在目标 expert 上的发送槽位，即当前 token 是第几个发送给这个 expert 的
 *
 * - 先把前缀 expertIds 与目标 expertId 做向量化比较
 * - 再通过 ReduceSum 得到“前缀中不等于目标 expert 的数量”
 * - 最终得到当前 expert 的前缀计数
 *
 * 这个 helper 依赖的是“完整输入前缀视图 + 全局 linearIdx”。
 * 因此多核化时，只要每个核仍保留完整只读 expertIds 视图，它既可以用于本核 token 的 slot 计算，
 * 也可以在状态阶段把 linearIdx 设到某个末尾位置来重算目标 expert 的总量。
 * 若后续把 expertIds 缩成核内局部切片，则这里的语义也会随之改变，不能再直接复用当前逻辑。
 */
template<typename DataType>
__aicore__ inline uint32_t MoeDispatchOp<DataType>::CalcCurExpertCnt(uint32_t dstExpertId, uint32_t linearIdx)
{
    if (linearIdx == 0) {
        return 0;
    }

    Duplicate<int32_t>(dstExpertLocal_, static_cast<int32_t>(dstExpertId), linearIdx);
    PipeBarrier<PIPE_V>();
    Sub(diffExpertLocal_, expertIdsLocal_, dstExpertLocal_, linearIdx);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> diffFp32 = diffExpertLocal_.template ReinterpretCast<float>();
    LocalTensor<float> dstExpertFp32 = dstExpertLocal_.template ReinterpretCast<float>();
    Abs(dstExpertFp32, diffFp32, linearIdx);
    PipeBarrier<PIPE_V>();
    Mins(diffExpertLocal_, dstExpertLocal_, 1, linearIdx);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(dstExpertFp32, diffFp32, workLocalTensor_, linearIdx);
    SyncFunc<AscendC::HardEvent::V_S>();

    int32_t curOtherExpertCnt = dstExpertLocal_.GetValue(0);
    if (static_cast<int32_t>(linearIdx) > curOtherExpertCnt) {
        return static_cast<uint32_t>(static_cast<int32_t>(linearIdx) - curOtherExpertCnt);
    }
    return 0;
}

/**
 * @brief 在 comm token 尾部写入 triple 元信息
 *
 * triple 顺序固定为：
 * [srcEpRankId, srcTokenIdx, srcTopkIdx]
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::FillTriple(LocalTensor<DataType> &commTokenLocal, const RouteInfo &route)
{
    LocalTensor<int32_t> commTokenInt32 = commTokenLocal.template ReinterpretCast<int32_t>();
    commTokenInt32.SetValue(comm_.tokenMetaOffset_, static_cast<int32_t>(epRankId_));
    commTokenInt32.SetValue(comm_.tokenMetaOffset_ + 1, static_cast<int32_t>(route.tokenIdx));
    commTokenInt32.SetValue(comm_.tokenMetaOffset_ + 2, static_cast<int32_t>(route.topkIdx));
}

/**
 * @brief 组织一个完整的通信 token 并写入目标 window
 *
 * 执行顺序：
 * 1. 从输入 x 读取 payload
 * 2. 在本地 buffer 尾部补 triple
 * 3. 将 payload + triple 整体写入目标 rank 的 window 槽位
 *
 * `x` 也是只读输入。多核化时可以让每个核按自己的 token 区间重复读取，
 * 或在容量允许时缓存本核工作集；这不会像局部统计那样引入跨核语义漂移。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::WritePackedToken(const RouteInfo &route, GlobalTensor<DataType> &dstTensor)
{
    LocalTensor<DataType> commTokenLocal = tokenQueue_.AllocTensor<DataType>();
    Duplicate<DataType>(commTokenLocal, (DataType)0, comm_.commTokenElems_);

    const uint32_t payloadBytes = static_cast<uint32_t>(h_ * sizeof(DataType));
    DataCopyExtParams copyInParams{1, payloadBytes, 0, 0, 0};
    DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
    
    GM_ADDR srcAddr = (GM_ADDR)(xGMTensor_.GetPhyAddr(route.tokenIdx * h_));
    
    DataCopyPad(commTokenLocal, xGMTensor_[route.tokenIdx * h_], copyInParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    FillTriple(commTokenLocal, route);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams copyOutParams{1, comm_.commTokenBytes_, 0, 0, 0};
    DataCopyPad(dstTensor, commTokenLocal, copyOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    tokenQueue_.FreeTensor(commTokenLocal);
}

/**
 * @brief 初始化 kernel 参数和本地 UB 资源
 *
 * 初始化步骤：
 * 1. 读取 tiling 中的 bs / h / epWorldSize / maxRecvTokens
 * 2. 解析 host 传入的 mc2Context，获取 rankId / window 基址
 * 3. 固化 dispatch window 布局
 * 4. 绑定输入输出 GlobalTensor
 * 5. 初始化用于向量化 slot 计数的 UB buffer
 * 6. 初始化状态同步缓冲区
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expandX, GM_ADDR expandIdx,
    GM_ADDR expertTokenNums, GM_ADDR epRecvCounts, GM_ADDR tilingGM, TPipe *tPipe)
{
    tilingData_ = (__gm__ MoeDispatchTilingData*)tilingGM;
    bs_ = static_cast<uint32_t>(tilingData_->tilingInfo.bs);
    h_ = static_cast<uint32_t>(tilingData_->tilingInfo.h);
    epWorldSize_ = static_cast<uint32_t>(tilingData_->tilingInfo.epWorldSize);
    maxRecvTokens_ = static_cast<uint32_t>(tilingData_->tilingInfo.maxRecvTokens);
    topK_ = static_cast<uint32_t>(tilingData_->tilingInfo.topK);
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    blockNum_ = static_cast<uint32_t>(GetBlockNum());

    comm_.InitHcclContextByAddr(mc2Context, epWorldSize_);
    epRankId_ = Mc2Kernel::GetRankId(comm_.hcclContext_);
    localExpertNum_ = 1;
    moeExpertNum_ = epWorldSize_ * localExpertNum_;
    comm_.InitDispatchWindow(bs_, h_, localExpertNum_);

    xGMTensor_.SetGlobalBuffer((__gm__ DataType*)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    expandXGMTensor_.SetGlobalBuffer((__gm__ DataType*)expandX);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expandIdx);
    expertTokenNumsGMTensor_.SetGlobalBuffer((__gm__ int64_t*)expertTokenNums);
    epRecvCountsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)epRecvCounts);

    InitBlockRange(bs_ * topK_, 1U); // 当前硬编码为1，可根据传入的 aivNum 和实际需求调整分核策略和 usedBlockNum

    tPipe->Reset();
    tPipe->InitBuffer(tokenQueue_, 1U, comm_.commTokenBytes_);
    uint32_t expertIdsCount = bs_ * topK_;
    uint32_t expertIdsAlignedCount = ((expertIdsCount + 7U) / 8U) * 8U; // int32 元素数按 32B 对齐后的 UB 容量
    uint32_t expandIdxCount = maxRecvTokens_ * 3U;
    uint32_t rankAlignedCount = ((epWorldSize_ + 7U) / 8U) * 8U; // 每 rank 统计项按 32B 对齐后的 UB 容量
    tPipe->InitBuffer(expertIdsBuf_, expertIdsAlignedCount * sizeof(int32_t));
    tPipe->InitBuffer(dstExpertBuf_, expertIdsAlignedCount * sizeof(int32_t));
    tPipe->InitBuffer(diffExpertBuf_, expertIdsAlignedCount * sizeof(int32_t));
    tPipe->InitBuffer(workLocalBuf_, ReduceSumWorkNeedSize(static_cast<int32_t>(expertIdsAlignedCount)) * sizeof(float));
    tPipe->InitBuffer(epRecvCountsOutBuf_, rankAlignedCount * sizeof(int32_t));
    tPipe->InitBuffer(expertTokenNumsOutBuf_, 8U * sizeof(int64_t));
    expertIdsLocal_ = expertIdsBuf_.Get<int32_t>();
    dstExpertLocal_ = dstExpertBuf_.Get<int32_t>();
    diffExpertLocal_ = diffExpertBuf_.Get<int32_t>();
    epRecvCountsOutLocal_ = epRecvCountsOutBuf_.Get<int32_t>();
    expertTokenNumsOutLocal_ = expertTokenNumsOutBuf_.Get<int64_t>();
    workLocalTensor_ = workLocalBuf_.Get<float>();

    Duplicate<int32_t>(expertIdsLocal_, 0, expertIdsAlignedCount);
    Duplicate<int32_t>(epRecvCountsOutLocal_, 0, rankAlignedCount);
    Duplicate<int64_t>(expertTokenNumsOutLocal_, 0, 8U);
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyExtParams expertIdsCopyParams{1, static_cast<uint32_t>(expertIdsCount * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expertIdsPadParams{false, 0, 0, 0};
    DataCopyPad(expertIdsLocal_, expertIdsGMTensor_, expertIdsCopyParams, expertIdsPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    comm_.InitBuffer(tPipe);
    ResetCounts();
}

/**
 * @brief 发送当前核负责的所有 token
 *
 * 对每个 token：
 * 1. 解析路由信息
 * 2. 计算目标 expert 的前缀槽位 dstSlot，这是分核后每个核需要独立计算的关键步骤
 * 3. 更新发送统计
 * 4. 将 token 打包并写入目标 rank window
 *
 * 注意：`expertSendCounts_` 在单核 sample 中等价于“本卡发送统计”，
 * 但多核后它只代表“当前核发送统计”。因此它可以继续作为本核局部累计值，
 * 但后续 `WriteStatus` 不能再默认把它当作整卡总量直接发布。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::SendTokens()
{
    for (uint32_t linearIdx = blockStart_; linearIdx < blockEnd_; ++linearIdx) {
        RouteInfo route = DecodeRoute(linearIdx);
        route.dstSlot = CalcCurExpertCnt(route.expertId, linearIdx);
        expertSendCounts_[route.dstRank * localExpertNum_ + route.localExpert]++;

        GM_ADDR dataAddr = comm_.GetDispatchDataAddr(route.dstRank, epRankId_, route.localExpert, route.dstSlot);

        GlobalTensor<DataType> inputTensor;
        inputTensor.SetGlobalBuffer((__gm__ DataType*)(xGMTensor_.GetPhyAddr(linearIdx * h_)));

        GlobalTensor<DataType> remoteTokenTensor;
        remoteTokenTensor.SetGlobalBuffer(
            (__gm__ DataType*)dataAddr,
            comm_.commTokenElems_);
        WritePackedToken(route, remoteTokenTensor);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

/**
 * @brief 从本地接收窗口回搬 token，并拆成 expandX / expandIdx
 *
 * 当前 sample 由于单卡上只有一个专家，回搬顺序按 source rank 和 slot 顺序展开，
 * 因此输出中的 token 顺序对应“按来源 rank 分桶后的接收顺序”。
 * 真实的算子中，单卡多专家，此时需按专家依次回搬 token，每个专家内部按来源 rank 和 slot 顺序展开，最终输出中 token 的顺序对应“先按专家分桶再按来源 rank 分桶后的接收顺序”。
 * 实际上，相当于对数据区接收到的结果进行了一次稳定排序并连续化，此时的偏移计算，依赖于 expertTokenNums 和 expertRecvCounts_ 等数据。
 * 当前 sample 里的 `outIdx` 是单核串行写回指针；多核后不能让每个核都从 0 开始写，
 * 必须先确定每个核的全局输出起始 offset，再并发写回各自的连续段。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::CopyWindowToOutputs()
{
    uint32_t outIdx = 0;
    for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
        for (uint32_t slot = 0; slot < recvCounts_[srcRank]; ++slot) {
            LocalTensor<DataType> commTokenLocal = tokenQueue_.AllocTensor<DataType>();
            GlobalTensor<DataType> localTokenTensor;
            GM_ADDR localDataAddr = comm_.GetLocalWindowDataAddr(srcRank, 0, slot);
            localTokenTensor.SetGlobalBuffer((__gm__ DataType*)localDataAddr);

            DataCopyExtParams copyInParams{1, comm_.commTokenBytes_, 0, 0, 0};
            DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
            DataCopyPad(commTokenLocal, localTokenTensor, copyInParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();

            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(h_ * sizeof(DataType)), 0, 0, 0};
            DataCopyPad(expandXGMTensor_[outIdx * h_], commTokenLocal, copyOutParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>();

            // 直接搬运 triple 到 GM（不需要 GetValue/SetValue）
            LocalTensor<int32_t> tokenI32 = commTokenLocal.template ReinterpretCast<int32_t>();
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams expandIdxCopyParams{1, static_cast<uint32_t>(3 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(expandIdxGMTensor_[outIdx * 3], tokenI32[comm_.tokenMetaOffset_], expandIdxCopyParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>();

            tokenQueue_.FreeTensor(commTokenLocal);
            outIdx++;
        }
    }
}

/**
 * @brief 向远端 rank 发布当前轮的发送数量
 *
 * 当前 sample 直接发布 `expertSendCounts_`，这是建立在“单核时该数组就是整卡统计”的前提上。
 * 多核后这里需要显式改成“先汇总再发布”或“基于完整输入视图重算后再发布”，
 * 不能继续把发送阶段顺手得到的局部数组当作全局状态使用。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::WriteStatus()
{
    comm_.SetRemoteStatus(expertSendCounts_);
}

/**
 * @brief 等待所有 source rank 的状态就绪后，再进入本地回搬阶段
 *
 * 单核 sample 中，`recvCounts_` 和 `expertRecvCounts_` 在返回后就是完整接收统计。
 * 多核后若等待阶段按状态段切分，这两个数组通常只会填充当前核负责段，
 * 后续回搬和最终输出不能再默认把它们当作整卡总表使用。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::WaitAllSources()
{
    comm_.WaitRemoteStatus(recvCounts_, expertRecvCounts_);
    PipeBarrier<PIPE_ALL>();
}

/**
 * @brief 写出 dispatch 的统计输出并清理尾部无效槽位
 *
 * 当前 sample 默认由单核把最终统计一次性写到 GM。
 * 多核后若这些统计只覆盖本核负责段，则这里必须改成“先汇总再写最终输出”
 * 或“指定唯一收口核写最终输出”，不能让多个核各自直接覆盖同一份最终结果。
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::FinalizeOutputs()
{
    // epRecvCounts[srcRank] = 从 srcRank 收到的 token 总数（所有 expert 的和）
    for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
        epRecvCountsOutLocal_.SetValue(srcRank, static_cast<int32_t>(recvCounts_[srcRank]));
    }
    
    // expertTokenNums[exp] = 本卡 local expert exp 收到的 token 总数（所有 srcRank 的和）
    // expertRecvCounts_[srcRank * localExpertNum_ + exp] 表示从 srcRank 发给 expert exp 的 token 数量
    for (uint32_t exp = 0; exp < localExpertNum_; ++exp) {
        uint64_t expertTotal = 0;
        for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
            expertTotal += static_cast<uint64_t>(expertRecvCounts_[srcRank * localExpertNum_ + exp]);
        }
        expertTokenNumsOutLocal_.SetValue(exp, static_cast<int64_t>(expertTotal));
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams recvCountsCopyParams{1, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(epRecvCountsGMTensor_, epRecvCountsOutLocal_, recvCountsCopyParams);

    DataCopyExtParams expertTokenCopyParams{1, static_cast<uint32_t>(localExpertNum_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokenNumsGMTensor_, expertTokenNumsOutLocal_, expertTokenCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

/**
 * @brief kernel 主入口
 *
 * 当前 sample 只在 AIV 核执行，完整阶段顺序为：
 * SendTokens -> WriteStatus -> WaitAllSources -> CopyWindowToOutputs -> FinalizeOutputs
 */
template<typename DataType>
__aicore__ inline void MoeDispatchOp<DataType>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }

    // 当前单核 sample 单核运行，因此直接让 block 0 处理全部 token，其他 block 跳过数据路径
    // 当处理多核场景时，可能不同的函数对分核的方式有不同需求，不能只使用通过token控制的blockStart_和blockEnd_来控制数据处理区间
    // 而是需要在每个函数内部根据实际需求进行分核策略的设计和实现
    // 特别是：发送阶段可按 token 分核，但状态写入、状态等待、输出回搬、最终统计不一定复用这组边界。
    // 多核实现时，不要把 `blockStart_ >= blockEnd_` 当成所有阶段都能直接 return 的全局条件。
    if (blockStart_ >= blockEnd_) {
        return;
    }

    SendTokens();
    WriteStatus();
    WaitAllSources();
    CopyWindowToOutputs();
    FinalizeOutputs();
}

} // namespace MoeDispatchImpl

#endif // MOE_DISPATCH_KERNEL_H