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
 * \file euclidean_norm_base.h
 * \brief EuclideanNorm 算子 Base kernel 类实现（新范式）。
 *
 * 数学：y = sqrt( sum( x^2 ) along axes )
 *
 * 新范式架构（reduction 二分归约范式）：
 *   - 统一使用 TBuf + Mutex Lock/Unlock 段式流水同步（不使用 TQue；Ascend950）
 *   - Reduce 直写 cacheBuf[cacheID × levelStride]，消除 tmpBuf 中转
 *   - DoCaching 在 cacheBuf 内部吸收低层后覆写
 *   - 不需要 ClearCacheTree（DoCaching 覆盖写保证正确性）
 *   - Buffer 体系：preInBuf(DT) + preReduceResult(fp32) + preReduceResultTail(fp32) + cacheBuf + outBuf(DT)
 *
 * Reducer = sum：
 *   pad_value = 0（PAD_CLEAR_VALUE）
 *   PreElewise = Cast(b16→fp32) + Square（VF 融合，链长 2 ≤ 7）
 *   PostElewise = Sqrt + Cast(fp32→DT)（VF 融合，链长 2 ≤ 7）
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_BASE_H_
#define OPS_NORM_EUCLIDEAN_NORM_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "adv_api/reduce/reduce.h"
#include "euclidean_norm_tiling_data.h"
#include "euclidean_norm_tiling_key.h"

namespace NsEuclideanNorm {

using namespace AscendC;

constexpr uint32_t VL_BYTES = Ops::Base::GetVRegSize();
constexpr uint32_t REP_F32 = VL_BYTES / sizeof(float);
constexpr uint16_t REP_F32_U16 = static_cast<uint16_t>(REP_F32);
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
constexpr uint32_t UB_BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float);
constexpr uint64_t UINT64_BITS = 64;
constexpr uint64_t UINT64_TOP_BIT_IDX = UINT64_BITS - 1;

// A/R 轴模式化后偶位 A、奇位 R，相邻同类型轴（同为 A 或同为 R）的固定间距。
constexpr int32_t AXIS_INTERVAL = 2;
// b16（fp16/bf16）单元素字节数。
constexpr size_t BYTES_PER_B16_ELEM = 2;
// ReduceSum srcShape 的 2D 维度（AR/RA 两 pattern 均为二维）。
constexpr size_t REDUCE_SHAPE_DIM = 2;
// FindNearestPower2 数学边界：v ≤ 2 时最近二次幂为 1。
constexpr uint64_t NEAREST_POW2_SMALL_BOUND = 2;

// DataCopyPad 轴层级索引：[0]最内块长 / [1]blockCount / [2]loop1 / [3]loop2 / [4..]外层软循环。
constexpr int32_t BLOCK_COUNT_AXIS_IDX = 1;
constexpr int32_t LOOP1_AXIS_IDX = 2;
constexpr int32_t LOOP2_AXIS_IDX = 3;
constexpr int32_t OUTER_LOOP_AXIS_BASE = 4;

constexpr AscendC::Reg::CastTrait CAST_TRAIT_TO_FP32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_NONE};

constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_FP16{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_INT32{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};

constexpr float PAD_CLEAR_VALUE = 0.0f; // pad_value（sum reducer）

// PreElewise VF 实现：按 dtype 分路装载并扩位到 fp32，自乘得 x²（中间值驻寄存器不落 UB）。
template <typename DType>
__simd_vf__ inline void CastSquareVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                         uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == BYTES_PER_B16_ELEM);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        // 掩码按剩余量生成（UpdateMask 为引用消耗语义，每轮自动递减 VL）
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        // dtype 三路：fp32 直装 / b16 DIST_UNPACK_B16 扩位装载 / int32 整装后 Cast
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::LoadAlign(iReg, src + off);
            AscendC::Reg::Cast<float, int32_t, CAST_TRAIT_TO_FP32>(f32Reg, iReg, mask);
        }

        // fp32 域自乘得 x²（中间精度恒 fp32，b16 不直接平方）
        AscendC::Reg::Mul(f32Reg, f32Reg, f32Reg, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

// tail-R 的 ExtensionPad 清零 VF：每个 A bundle 行内，从对齐后的有效末尾清到 padded 行宽。
__simd_vf__ inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                 uint32_t extLanes, uint16_t aU16, uint16_t repPerA)
{
    AscendC::Reg::RegTensor<float> idReg;
    // pad 固化值 0（sum reducer 的单位元）
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);

    // 外层按 A bundle 行跳 stride，内层按 VL 分段清 [extStart, paddedWidth)
    for (uint16_t aIdx = 0; aIdx < aU16; ++aIdx) {
        int32_t aOff = static_cast<int32_t>(aIdx) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off =
                aOff + static_cast<int32_t>(extStart) + static_cast<int32_t>(r) * static_cast<int32_t>(REP_F32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, idReg, mask);
        }
    }
}

// tail-A 的 ExtensionPad 清零 VF：rSplit 在 UB 最外层，清 [startElem, startElem+totalClear) 一段连续区。
__simd_vf__ inline void ClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t totalClear,
                                                 uint16_t repCount)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

// BurstPad 清零 VF：清每行 fp32 block 尾部的 pad 窗口 [partialStart, padEnd)，按行距跨行重复。
__simd_vf__ inline void ClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCntU16, int32_t rowStrideI,
                                                     int32_t windowOff, uint32_t padEnd, uint32_t partialStartInBlock)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);

    // pad 窗口掩码 = [0, padEnd) ∩ 非[0, partialStart)：跳过 block 内 valid 前缀只清 pad
    uint32_t cntEnd = padEnd;
    uint32_t cntStart = partialStartInBlock;
    auto maskEnd = AscendC::Reg::UpdateMask<float>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<float>(cntStart);
    auto allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart, padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask);

    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

// Phase A 尾块配对 VF：preRes += preResTail（元素级 Merge，Merge 后尾块数据即作废释放）。
__simd_vf__ inline void MergeTmpBufVfImpl(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf, uint32_t totalElems,
                                          uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, mainBuf + off);
        AscendC::Reg::LoadAlign(bReg, tailBuf + off);
        AscendC::Reg::Add(aReg, aReg, bReg, mask);
        AscendC::Reg::StoreAlign(mainBuf + off, aReg, mask);
    }
}

// 二分缓存树吸收 VF：当前层（levelOff）就地累加全部低层后覆写自身。
__simd_vf__ inline void DoCachingVfImpl(__ubuf__ float* cacheBuf, uint32_t laneN, uint32_t levelStride,
                                        int32_t levelOff, uint16_t repeatTime, uint16_t cacheLevelCnt)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);

        // 逐低层累加（GetCacheID 序列保证低层先写就绪）
        for (uint16_t j = 0; j < cacheLevelCnt; ++j) {
            int32_t lowerLevelOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + lowerLevelOff);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
        }
        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

// PostElewise VF 实现：树根逐元素 Sqrt 后按 dtype 缩位 Cast 写 outBuf。
template <typename DType>
__simd_vf__ inline void PostElewiseVfImpl(__ubuf__ float* rootPtr, __ubuf__ DType* outPtr, uint32_t laneN,
                                          uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == BYTES_PER_B16_ELEM);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
        AscendC::Reg::Sqrt(f32Reg, f32Reg, mask);

        // dtype 三路：fp32 直写 / b16 CAST_RINT + DIST_PACK_B32 打包写 / int32 CAST_TRUNC
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::Cast<DType, float, CAST_TRAIT_FROM_FP32_FP16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<DType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
        } else {
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::Cast<int32_t, float, CAST_TRAIT_FROM_FP32_INT32>(iReg, f32Reg, mask);
            AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
        }
    }
}

// GM→UB 轴映射描述：UB 展开的一根轴对应一根 GM 轴的 actual/padded 双值与 GM 步长。
struct UBAxisDesc {
    int32_t gmIdx;     // 对应规整后模式中的 GM 轴下标
    int64_t actualNum; // 实际有效元素数（参与搬运）
    int64_t paddedNum; // UB 侧 padded 元素数（含对齐补齐）
    int64_t gmStride;  // 该轴在 GM 上的元素步长
};

template <typename DType>
class EuclideanNormBaseKernel {
public:
    using DT = DType;

    __aicore__ inline EuclideanNormBaseKernel() {}

    // Base 初始化：缓存 TilingData 指针、现算二分树/输出步长派生量、绑定 GM、
    // 分配 5 个 UB buffer（preIn/preRes/preResTail/cache/out）。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td, TPipe* pipe);
    // Base 主流程：多核均分 A 迭代（大核+1 轮），每轮 DoOneAChunk→PostElewise→CopyOut；
    // 三段流水（MTE2 搬入 → V 计算 → MTE3 搬出）以同 id 的 Mutex Lock/Unlock 段链式串行，
    // 跨迭代 WAR 由段链顺序天然覆盖。
    __aicore__ inline void Process();

protected:
    // 核号 → 本核 A 迭代区间 [aLoopStart, aLoopEnd)（大小核均分映射）。
    __aicore__ inline void UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd);
    // A 迭代序号 → 外层 A 轴索引 aIdx[] + 切分轴 chunk 序号 aSplitChunkIdx。
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx);
    // R 迭代序号 → 外层 R 轴索引 rOuterIdx[] + chunk 序号 + chunk 实际长度；返回外层 R 的 GM 偏移。
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen);

    // 单个 A chunk 的完整 R 二分归约（主块 + 尾块配对 + DoCaching）。
    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);
    // 后处理：cacheBuf 树根逐元素 Sqrt + 缩位 Cast 回 DT 写 outBuf（VF 融合，链长 2）。
    __aicore__ inline void PostElewise(int64_t aLen);
    // 输出搬运：outBuf 有效段（aLen×innerAProd）DataCopyPad 直写 GM y（tail-R/tail-A 两路径）。
    __aicore__ inline void CopyOut(int64_t outerOutOff, int64_t aLen);

    // 构造 GM→UB 的轴映射表（actual/padded 双值 + GM 步长），供 DoCopyInTile 组装 DataCopyPad 参数。
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]);
    // 输入搬运：按轴映射表组装 extParams/loopParams（K≥3 开 Loop 模式），外层 for 覆盖 >4 维。
    __aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen, __ubuf__ DT* preIn);

    // PreElewise 封装：preIn(DT) → preRes(fp32) 的 x²。
    __aicore__ inline void CastSquareVf(__ubuf__ DT* src, __ubuf__ float* dst);
    // ExtensionPad 清零封装（partial chunk 的 R 向 stale 区，tail-R/tail-A 布局分路）。
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen);
    // BurstPad 清零封装（tail-R 的 burst 尾轴 block 内 pad）。
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen);
    // 尾块配对封装：mainBuf += tailBuf。
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf);
    // 缓存树吸收封装：cacheID 层就地累加全部低层。
    __aicore__ inline void DoCachingVf(uint16_t cacheID);

    // 最内 A 轴下标（最大偶下标）。
    __aicore__ inline int32_t LastAAxis() const;
    // 最内 R 轴下标（最大奇下标）。
    __aicore__ inline int32_t LastRAxis() const;
    // 二分树写入层级：rIdx 的低位连续 1 个数 - 1。
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    // 不大于 v 的最大 2 的幂。
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    // ⌊log2(v)⌋。
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;
    // R 切分轴第 rChunkIdx 块的实际长度（末块可能 partial）。
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;

    // ── TilingData 指针与 Init 现算派生量 ──
    const EuclideanNormTilingData* td_ = nullptr;
    bool isTailR_ = false; // tail 类型：Init 由 axisNum 奇偶现算
    // 二分缓存树参数（由 rLoopCntTotal 现算）：主段块数/尾块数/树层数
    int64_t rSplitChunkCnt_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    // 输出 GM 各 A 轴步长（由 axisShape 现算，供 CopyOut 解码输出偏移）
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    // ── GM 绑定与 UB buffer（5 个 TBuf，无 TQue）──
    GlobalTensor<DT> xGm_;
    GlobalTensor<DT> yGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> preInBuf_;            // 输入（DT 域）
    TBuf<QuePosition::VECCALC> preReduceResult_;     // fp32 中间结果（Reduce src）
    TBuf<QuePosition::VECCALC> preReduceResultTail_; // 尾块 + 兼作 Reduce sharedTmpBuffer
    TBuf<QuePosition::VECCALC> cacheBuf_;            // 二分缓存树
    TBuf<QuePosition::VECCALC> outBuf_;              // 输出（DT 域）
    // ── 核内同步 MutexID（Process 内 AllocMutexID 申请 / 末尾释放，禁止硬编码）──
    uint8_t mutexId_ = 0;
};

template <typename DType>
__aicore__ inline int32_t EuclideanNormBaseKernel<DType>::LastAAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType>
__aicore__ inline int32_t EuclideanNormBaseKernel<DType>::LastRAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

template <typename DType>
__aicore__ inline uint64_t EuclideanNormBaseKernel<DType>::FindNearestPower2(uint64_t v) const
{
    if (v == 0) {
        return 0;
    }
    if (v <= NEAREST_POW2_SMALL_BOUND) {
        return 1;
    }
    const uint64_t num = v - 1;
    const uint64_t pow = UINT64_TOP_BIT_IDX - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}

template <typename DType>
__aicore__ inline uint64_t EuclideanNormBaseKernel<DType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

template <typename DType>
__aicore__ inline uint16_t EuclideanNormBaseKernel<DType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}

template <typename DType>
__aicore__ inline int64_t EuclideanNormBaseKernel<DType>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = td_->axisShape[td_->rSplitIdx];
    const int64_t start = rChunkIdx * td_->rUbFactor;
    return (start + td_->rUbFactor > rAxisSize) ? (rAxisSize - start) : td_->rUbFactor;
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td,
                                                            TPipe* pipe)
{
    td_ = td;
    // tail 类型：偶数轴→tail-R（最内轴为 R），奇数轴→tail-A
    isTailR_ = (td_->axisNum % AXIS_INTERVAL == 0);

    // 二分树参数：rLoopCntTotal 拆成「2 的幂主段 + 尾段」，cacheCount = log2(主段) + 1 层
    rSplitChunkCnt_ = Ops::Base::CeilDiv(td->axisShape[td->rSplitIdx], td->rUbFactor);
    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(td->rLoopCntTotal)));
    bisectionTail_ = td->rLoopCntTotal - bisectionPos_;
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    // 输出 GM 步长：从最内 A 轴向外累乘（输出无 R 维，仅 A 轴参与）
    int64_t outStrideAcc = 1;
    for (int32_t i = td->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            outStride_[i] = outStrideAcc;
            outStrideAcc *= td->axisShape[i];
        }
    }

    // 绑定 GM 与分配 5 个 UB buffer（preIn/preRes/preResTail 同尺寸，cache 固定 16KB，out 独立）
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));

    pipe_ = pipe;
    pipe_->InitBuffer(preInBuf_, td->preBufSize);
    pipe_->InitBuffer(preReduceResult_, td->preBufSize);
    pipe_->InitBuffer(preReduceResultTail_, td->preBufSize);
    pipe_->InitBuffer(cacheBuf_, td->cacheBufUbSize);
    pipe_->InitBuffer(outBuf_, td->postBufSize);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
        aLoopStart = blockIdx * td_->aBigCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aBigCoreLoopCnt;
    } else {
        aLoopStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aSmallCoreLoopCnt;
    }
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::UnravelALoop(int64_t aLoopIdx, int64_t aIdx[],
                                                                    int64_t& aSplitChunkIdx)
{
    int64_t aLoopRem = aLoopIdx;
    aSplitChunkIdx = aLoopRem % td_->aSplitChunkCnt;
    aLoopRem /= td_->aSplitChunkCnt;
    for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        aIdx[k] = aLoopRem % td_->axisShape[k];
        aLoopRem /= td_->axisShape[k];
    }
}

template <typename DType>
__aicore__ inline int64_t EuclideanNormBaseKernel<DType>::UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[],
                                                                       int64_t& rChunkIdx, int64_t& rLen)
{
    rChunkIdx = rIdx % rSplitChunkCnt_;
    int64_t rLoopRem = rIdx / rSplitChunkCnt_;
    int64_t gmOff = 0;
    for (int32_t k = td_->rSplitIdx - AXIS_INTERVAL; k >= 1; k -= AXIS_INTERVAL) {
        rOuterIdx[k] = rLoopRem % td_->axisShape[k];
        rLoopRem /= td_->axisShape[k];
        gmOff += rOuterIdx[k] * td_->axisStride[k];
    }
    rLen = RLenOfChunk(rChunkIdx);
    return gmOff + rChunkIdx * td_->rUbFactor * td_->axisStride[td_->rSplitIdx];
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    // 超出启用核数的核直接退出
    if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
        return;
    }

    // 解码本核 A 迭代区间（大小核均分映射）
    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    UnravelBlockLoop(aLoopStart, aLoopEnd);

    const int64_t aSplitAxisSize = td_->axisShape[td_->aSplitIdx];
    const int64_t aSplitStride = td_->axisStride[td_->aSplitIdx];
    const int64_t aSplitOutStr = outStride_[td_->aSplitIdx];

    // 申请核内同步 MutexID：三段流水（MTE2 搬入 → V 计算 → MTE3 搬出）以同 id 的
    // Lock/Unlock 段链式串行，跨迭代 WAR（上轮 V/MTE3 读 → 下轮 MTE2/V 覆写）由段链顺序天然覆盖
    mutexId_ = AscendC::AllocMutexID();

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        // 解码外层 A 轴索引与切分轴 chunk 序号
        int64_t aIdx[MAX_PATTERN_RANK] = {0};
        int64_t aSplitChunkIdx = 0;
        UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

        // 外层 A 轴贡献的 GM 输入偏移与输出偏移
        int64_t chunkGmOff = 0;
        int64_t chunkOutOff = 0;
        for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
            chunkGmOff += aIdx[k] * td_->axisStride[k];
            chunkOutOff += aIdx[k] * outStride_[k];
        }

        // A chunk 起点、有效长度（末 chunk 可能 partial）
        const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
        const int64_t aEnd = aChunkStart + td_->aUbFactor;
        const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : td_->aUbFactor;

        chunkGmOff += aChunkStart * aSplitStride;
        chunkOutOff += aChunkStart * aSplitOutStr;

        // 每 A chunk 三段：R 二分归约（内部多段）→ 树根 Sqrt+Cast（V 段）→ 搬出（MTE3 段）
        DoOneAChunk(chunkGmOff, aLen);

        AscendC::Mutex::Lock<PIPE_V>(mutexId_);
        PostElewise(aLen);
        AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId_);
        CopyOut(chunkOutOff, aLen);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId_);
    }

    AscendC::ReleaseMutexID(mutexId_);
}

// 单个 A chunk 的完整 R 二分归约：主段 tile 装载→Cast+Square→（tail 配对 Merge）→ReduceSum 直写
// cacheBuf[cacheID×levelStride]→DoCaching 就地吸收低层。跨 rIdx 迭代的 preIn/preRes WAR 由
// 同 id 的 Mutex Lock/Unlock 段链式串行覆盖（MTE2 搬入段 ↔ V 计算段交替衔接）。
template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    for (int64_t rIdx = 0; rIdx < bisectionPos_; ++rIdx) {
        // ── 主块：解码 R 迭代 → 装载 → Cast+Square ──
        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkIdxMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = UnravelRLoop(rIdx, rOuterIdx, rChunkIdxMain, rLenMain);

        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(preInBuf_.Get<DT>().GetPhyAddr());
        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.Get<float>().GetPhyAddr());

        // 搬入段（MTE2）：主块
        AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
        DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain, preIn);
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

        // 计算段（V）：主块平方与 pad 清零（等搬入段完成）
        AscendC::Mutex::Lock<PIPE_V>(mutexId_);
        CastSquareVf(preIn, preRes);
        // partial chunk：清 R 向对齐补齐的 ExtensionPad（stale 数据会污染 ReduceSum）
        if (rLenMain < td_->rUbFactor) {
            ClearChunkExtensionVf(preRes, rLenMain);
        }
        // tail-R：清 burst 尾轴 block 内 pad（dummy 填充值不可信，必须清）
        if (isTailR_) {
            ClearInnerBurstTailPadVf(preRes, rLenMain);
        }
        AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

        // ── Phase A 尾块配对：rIdx 落在尾段时，主尾块元素级 Merge 后一次 reduce ──
        if (rIdx < bisectionTail_) {
            int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
            int64_t rChunkIdxTail = 0;
            int64_t rLenTail = 0;
            const int64_t rOffTail = UnravelRLoop(rIdx + bisectionPos_, rOuterIdxTail, rChunkIdxTail, rLenTail);

            __ubuf__ float* preResTail =
                reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());

            // 搬入段（MTE2）：尾块（复用 preIn，等主块 V 段读完）
            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(outerGmOff + rOffTail, aLen, rLenTail, preIn);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            // 计算段（V）：尾块平方 + pad 清零 + 元素级 Merge
            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            CastSquareVf(preIn, preResTail);
            if (rLenTail < td_->rUbFactor) {
                ClearChunkExtensionVf(preResTail, rLenTail);
            }
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preResTail, rLenTail);
            }
            MergeTmpBufVf(preRes, preResTail);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }

        // ── Reduce 直写缓存树当前层，DoCaching 就地吸收低层 ──
        const uint16_t cacheID = GetCacheID(rIdx);
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = Ops::Base::CeilAlign(laneA, UB_BLOCK_F32);
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

        // 计算段（V）：ReduceSum + DoCaching（与上方的 V 段同链串行，天然有序）；
        // srcShape 按 padded 值：tail-R 用 AR（沿内层 R 归约），tail-A 用 RA（沿外层 R 归约）
        AscendC::Mutex::Lock<PIPE_V>(mutexId_);
        if (isTailR_) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {laneA,
                                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                                   laneA};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        }
        DoCachingVf(cacheID);
        AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
    }
}

template <typename DType>
__aicore__ inline int32_t EuclideanNormBaseKernel<DType>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[])
{
    int32_t k = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DT));

    if (isTailR_) {
        // tail-R：UB 内 [A_bundle, R_bundle]——先收 R 轴（内层），再收 A 轴（外层）
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;                // 切分轴：chunk 实际长度
                padded = td_->rUbFactorAlign; // padded = 对齐后的因子
            } else if (i == lastR) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem); // burst 尾轴按 32B 对齐
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;           // 切分轴：chunk 实际长度
                padded = td_->aUbFactor; // A 侧无需对齐
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    } else {
        // tail-A：UB 内 [R_bundle, A_bundle]——先收 A 轴（内层），再收 R 轴（外层）
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else if (i == lastA) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem); // burst 尾轴按 32B 对齐
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    }
    return k;
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                                                    __ubuf__ DT* preIn)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t axisCnt = BuildUBAxes(aLen, rLen, ubAxes);

    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    const int64_t dtBytes = static_cast<int64_t>(sizeof(DT));
    // 第 1 轴（最内）→ blockLen；dstStride = padded 行宽 − 对齐搬运量（UB 侧 gap，单位 32B）
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].actualNum * dtBytes);

    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    const int64_t copyPadBytes =
        Ops::Base::CeilAlign(static_cast<int64_t>(extParams.blockLen), static_cast<int64_t>(UB_BLOCK_BYTES));
    const int64_t target0Bytes = ubAxes[0].paddedNum * dtBytes;
    extParams.dstStride = (target0Bytes - copyPadBytes) / static_cast<int64_t>(UB_BLOCK_BYTES);

    // 第 2 轴 → blockCount；srcStride = GM 侧 gap（尾→头，减一次 blockLen）
    if (axisCnt > BLOCK_COUNT_AXIS_IDX) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[BLOCK_COUNT_AXIS_IDX].actualNum);
        extParams.srcStride =
            ubAxes[BLOCK_COUNT_AXIS_IDX].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen);
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }

    // UB 每层字节步长：ubStride[k] = ubStride[k-1] × paddedNum[k-1]
    int64_t ubStride[MAX_PATTERN_RANK] = {0};
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < axisCnt; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedNum;
    }

    // 第 3/4 轴 → loop1/loop2（stride 为 advance 语义：头→头，= axisStride × dt）
    if (axisCnt > LOOP1_AXIS_IDX) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[LOOP1_AXIS_IDX].actualNum);
        loopParams.loop1SrcStride =
            static_cast<uint64_t>(ubAxes[LOOP1_AXIS_IDX].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[LOOP1_AXIS_IDX]);
        loopParams.loop2Size = 1;
    }
    if (axisCnt > LOOP2_AXIS_IDX) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[LOOP2_AXIS_IDX].actualNum);
        loopParams.loop2SrcStride =
            static_cast<uint64_t>(ubAxes[LOOP2_AXIS_IDX].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[LOOP2_AXIS_IDX]);
    }

    // K≥3 开 Loop 模式（Set/ResetLoopModePara 必须成对）
    const bool useLoopMode = (axisCnt > LOOP1_AXIS_IDX);
    if (useLoopMode) {
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    }

    // 第 5 轴及更外 → 软件循环展开（K≤4 时 outerProd=1，循环体只执行一次）
    int64_t outerProd = 1;
    for (int32_t k = OUTER_LOOP_AXIS_BASE; k < axisCnt; ++k) {
        outerProd *= ubAxes[k].actualNum;
    }

    auto preInLocal = preInBuf_.Get<DT>();
    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        // 软循环序号 → 各外层轴索引 → GM/UB 增量偏移（混合进制解码）
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t outerRem = outerFlat;
        for (int32_t k = OUTER_LOOP_AXIS_BASE; k < axisCnt; ++k) {
            const int64_t axisSize = ubAxes[k].actualNum;
            const int64_t axisIdx = outerRem % axisSize;
            outerRem /= axisSize;
            addGmOffElem += axisIdx * ubAxes[k].gmStride;
            addUbOffBytes += axisIdx * ubStride[k];
        }
        const int64_t ubOffElems = addUbOffBytes / dtBytes;
        DataCopyPad(preInLocal[ubOffElems], xGm_[baseGmOff + addGmOffElem], extParams, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::CastSquareVf(__ubuf__ DT* src, __ubuf__ float* dst)
{
    // 元素总数 = padded UB tile 全量（aUnit × rPadded），repeat 按 VL 分段
    const uint32_t totalElems =
        static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t repeatTime =
        static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<CastSquareVfImpl<DT>>(src, dst, totalElems, repeatTime);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
{
    // 整块装载（rLen == rUbFactor）时无 ExtensionPad
    if (rLen >= td_->rUbFactor) {
        return;
    }

    if (isTailR_) {
        // tail-R：清每个 A bundle 行的 [对齐后有效末尾, padded 行宽)
        const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart = Ops::Base::CeilAlign(rLenInner, UB_BLOCK_F32);
        const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA;
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = Ops::Base::CeilDiv(extLanes, REP_F32);
        const uint16_t aU16 = static_cast<uint16_t>(aBundleEntries);

        asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes, aU16, static_cast<uint16_t>(repPerA));
    } else {
        // tail-A：rSplit 在 UB 最外层，清连续 stale 区 [rLen×cellElems, rUbFactor×cellElems)
        const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->innerRProdAlign);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount = Ops::Base::CeilDiv(totalClear, REP_F32);

        asc_vf_call<ClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount));
    }
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
{
    const uint32_t bsInput = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(DT));
    const int32_t lastR = LastRAxis();
    // 有效 R 列数：rSplit==lastR 时为本 chunk 实际长度，否则为整轴长度
    const uint32_t validR =
        (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) : static_cast<uint32_t>(td_->axisShape[lastR]);
    if (validR % bsInput == 0) {
        return;
    }
    // UB 行宽：rSplit==lastR 时为 R_bundle 宽，否则为 burst 尾轴 padded 宽
    const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                   Ops::Base::CeilAlign(static_cast<uint32_t>(td_->axisShape[lastR]), bsInput);
    // rSplit!=LastR：按全量行清（partial chunk 需要的行在 UB 中不连续——A entry 在最外层，
    // 行号含 aStride 跳跃），多清的 stale 行其 BurstPad 位置同为脏数据
    const uint32_t rowCnt =
        (td_->rSplitIdx == lastR) ?
            static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
            static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign * td_->innerRProdAlign) /
                rowStride;

    const uint32_t padEndInRow = Ops::Base::CeilAlign(validR, bsInput);
    const uint32_t partialBlockIdx = validR / UB_BLOCK_F32;
    const uint32_t partialStartInBlock = validR % UB_BLOCK_F32;
    const uint32_t padEnd = padEndInRow - partialBlockIdx * UB_BLOCK_F32;

    asc_vf_call<ClearInnerBurstTailPadVfImpl>(base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
                                              static_cast<int32_t>(partialBlockIdx * UB_BLOCK_F32), padEnd,
                                              partialStartInBlock);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf)
{
    const uint32_t totalElems =
        static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t repeatTime =
        static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<MergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::DoCachingVf(uint16_t cacheID)
{
    // 层宽 = laneA 对齐到 fp32 block；当前层起点 = cacheID × levelStride
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    asc_vf_call<DoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::PostElewise(int64_t aLen)
{
    // 树根 = 最高层起点（cacheCount-1 层）
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;
    __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.Get<DT>().GetPhyAddr());

    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

    asc_vf_call<PostElewiseVfImpl<DT>>(rootPtr, outPtr, laneN, repeatTime);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::CopyOut(int64_t outerOutOff, int64_t aLen)
{
    auto outLocal = outBuf_.Get<DT>();

    DataCopyExtParams outParams;
    if (isTailR_) {
        // 路径 1（tail-R）：outBuf 按 A 连续, 一次搬 aLen × innerAProd
        int64_t innerAProd = 1;
        for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= LastAAxis(); k += AXIS_INTERVAL) {
            innerAProd *= td_->axisShape[k];
        }
        outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(DT)));
        outParams.blockCount = 1;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = td_->axisShape[lastA];
        if (td_->aSplitIdx == lastA) {
            // 路径 2（tail-A 且切分轴=最内 A）：单 burst
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = 1;
        } else {
            // 路径 3（tail-A 且切分轴在外）：多 burst——blockLen=最内 A 轴长，blockCount=块数，
            int64_t innerAProd = 1;
            for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= lastA; k += AXIS_INTERVAL) {
                innerAProd *= td_->axisShape[k];
            }
            outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
        }
    }
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[outerOutOff], outLocal, outParams);
}

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_BASE_H_
