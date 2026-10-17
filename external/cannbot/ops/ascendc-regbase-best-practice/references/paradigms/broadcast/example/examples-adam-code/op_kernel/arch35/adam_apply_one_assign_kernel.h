/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AdamApplyOneAssign Kernel — AdamKernel<T, RANK>
 */
#ifndef ADAM_APPLY_ONE_ASSIGN_KERNEL_H_
#define ADAM_APPLY_ONE_ASSIGN_KERNEL_H_
#include "kernel_operator.h"
#include "adam_apply_one_assign_tiling_struct.h"
#include "adam_apply_one_assign_struct.h"

// ============================================================
// Kernel 侧辅助函数 (int64_t * 版本, 无 std::vector)
// ============================================================

// GetCoreRange: 计算当前核负责的 tile 序号区间 [start, end)——
// 主核（coreId < mainCoreNum）各处理 mainTiles 块，尾核各处理 mainTiles-1 块。
__aicore__ inline void GetCoreRange(int64_t coreId, int64_t mainTiles, int64_t mainCoreNum, int64_t& start,
                                    int64_t& end)
{
    if (coreId < mainCoreNum) {
        start = coreId * mainTiles;
        end = start + mainTiles;
    } else {
        start = mainCoreNum * mainTiles + (coreId - mainCoreNum) * (mainTiles - 1);
        end = start + mainTiles - 1;
    }
}

// GetUBSplitRange: 返回本次 UB 分段的元素长度——最后一段为尾块 ubTail，其余为 ubFactor。
__aicore__ inline int64_t GetUBSplitRange(int64_t ubBlockIdx, int64_t ubOuter, int64_t ubFactor, int64_t ubTail)
{
    return (ubBlockIdx == ubOuter - 1) ? ubTail : ubFactor;
}

// FlatToEffectiveCoord: 把 flat tile 序号展开为坐标系坐标——外轴（0..ubSplitIdx-1）
// 逐维取模分解，切分轴坐标 = ubBlockIdx × ubFactor，内侧轴保持 0。
__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t ubSplitIdx,
                                            int64_t ubFactor, int64_t ubOuter, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++) {
        effCoord[d] = 0;
    }
    int64_t ubBlockIdx = flat % ubOuter;
    int64_t outer = flat / ubOuter;
    for (int64_t d = ubSplitIdx - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[ubSplitIdx] = ubBlockIdx * ubFactor;
    return true;
}

// CalcOffset / CalcTransferCount: 输入与输出规则相同，共用一份实现。
// CalcOffset: 按坐标与 stride 计算 GM 偏移（元素个数）——广播维 stride=0，坐标不贡献偏移。
__aicore__ inline int64_t CalcOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) {
        offset += effCoord[d] * strides[d];
    }
    return offset; // 元素个数，gmIn_[]/gmOut_[] 的 index
}

// CalcTransferCount: 计算本 tile 的搬运元素数——split 轴大小（广播时为 1，否则 ubBlockLength）
// × 内侧轴连乘。
__aicore__ inline int64_t CalcTransferCount(const int64_t* normalShape, int64_t rank, int64_t ubSplitIdx,
                                            int64_t ubBlockLength)
{
    int64_t splitElems = (normalShape[ubSplitIdx] == 1) ? 1 : ubBlockLength;
    int64_t innerElems = 1;
    for (int64_t d = ubSplitIdx + 1; d < rank; d++) {
        innerElems *= normalShape[d];
    }
    return splitElems * innerElems;
}

template <typename T>
__simd_vf__ inline void MulAddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count,
                                 uint32_t oneRepeatSize, uint16_t repeatTimes);

template <typename T, int64_t RANK>
class AdamKernel {
    static constexpr int64_t MAX_RANK = 8; // 最大支持的 rank（坐标缓冲大小）
    static constexpr int64_t MAX_NDDMA_DIMS = 5;
    static constexpr int64_t NDDMA_DIMS = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(T);

    AscendC::TPipe pipe_;
    const AdamTilingData<RANK>* td_ = nullptr;
    AscendC::GlobalTensor<T> gmIn_[MAX_INPUT_SLOTS];
    AscendC::GlobalTensor<T> gmOut_[MAX_OUTPUT_SLOTS];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES];
    AscendC::MultiCopyParams<T, NDDMA_DIMS> nddmaParams_[MAX_INPUT_SLOTS];
    int64_t nddmaOuterIters_[MAX_INPUT_SLOTS];
    int64_t nddmaDims_ = 0;

public:
    // Init: 绑定全部输入/输出 GM 地址，按 perBufBytes 初始化 PHYS_NODES 个 TBuf，
    // 并预计算每个输入的 NDDMA 参数（loopSize/loopSrcStride/loopDstStride）
    // 及 rank 超 5 维时的 outer 迭代数。
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const AdamTilingData<RANK>* td)
    {
        td_ = td;
        for (int32_t i = 0; i < MAX_INPUT_SLOTS; i++) {
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        }
        for (int32_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        }
        for (int32_t i = 0; i < PHYS_NODES; i++) {
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);
        }

        const int64_t* dstShape = td_->maxBroShape;
        int64_t k = td_->split.ubSplitIdx;
        nddmaDims_ = (RANK - k <= NDDMA_DIMS) ? (RANK - k) : NDDMA_DIMS;
        for (int32_t inp = 0; inp < MAX_INPUT_SLOTS; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < NDDMA_DIMS; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
                inner *= (d == k) ? td_->split.ubFactor : dstShape[d];
                nd++;
            }
            for (; nd < NDDMA_DIMS; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
            }
            // outer loop 只覆盖 flat 层和 NDDMA 之间的 gap: d = k .. RANK-nddmaDims-1
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = k; d < RANK - nddmaDims_; d++) {
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.ubFactor : dstShape[d];
            }
        }
    }

    // Process: 多核 tile 主循环——GetCoreRange 取本核区间后逐 tile 展开坐标，
    // 按 CopyInBrc（NDDMA 广播搬运）→ Vector 计算（Mul/Sqrt/Div/Sub/MulAdd VF 链）→ CopyOut
    // 序列完成 Adam 更新；Mutex 保护 buffer 跨流水线复用，三个输出依次写回 GM。
    __aicore__ inline void Process()
    {
        uint8_t mutexId = AscendC::AllocMutexID();

        int64_t start = 0;
        int64_t end = 0;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.mainTiles, td_->multicore.mainCoreNum, start, end);

        // 输入槽位索引 (gmIn_[]) — 命名对齐 ops-nn def: input0~input4 / mul0_x~mul3_x / add2_y
        constexpr int32_t inInput0 = 0; // input0 (dataGrad) — square(S1) 与 mul1(S8) 的源
        constexpr int32_t inInput1 = 1; // input1 (dataV)    — mul2(S3) 操作数
        constexpr int32_t inInput2 = 2; // input2 (dataM)    — mul0(S7) 操作数
        constexpr int32_t inInput3 = 3; // input3 (dataVar)  — sub(S12) 操作数
        constexpr int32_t inInput4 = 4; // input4            — mul4(S11) 乘子
        constexpr int32_t inMul0X = 5;  // mul0_x (dataInputMul)  — mul0(S7) 乘子
        constexpr int32_t inMul1X = 6;  // mul1_x (dataInputMul1) — mul1(S8) 乘子
        constexpr int32_t inMul2X = 7;  // mul2_x (dataInputMul2) — mul2(S3) 乘子
        constexpr int32_t inMul3X = 8;  // mul3_x (dataInputMul3) — mul3(S2) 乘子
        constexpr int32_t inAdd2Y = 9;  // add2_y (dataInputAdd2) — add2(S6) 操作数

        // UB 槽位索引 (buf_[]) — 槽位多角色复用, 生命周期见设计文档 UB 生命周期表
        constexpr int32_t ubLoadData = 0; // data 类输入(grad/m/v/input4/var)装载槽, 复用为 sqrtResult
        constexpr int32_t ubLoadMulX = 1; // 乘子(mul0_x/mul1_x)装载槽, 复用为 add2Result/mul4Result
        constexpr int32_t ubMid = 2;      // 中间量槽: squareResult → truedivResult → output2
        constexpr int32_t ubOutM = 3;     // output1(m 路)累加槽: mul1Result 经 MulAddDst 就地累加
        constexpr int32_t ubOutV = 4;     // output0(v 路)累加槽: mul3Result 经 MulAddDst 就地累加

        // 输出槽位索引 (gmOut_[]) — 命名对齐 def: Output("out0"/"out1"/"out2")
        constexpr int32_t out0 = 0; // out0 (output0) — v 路更新: v·mul2_x + grad²·mul3_x
        constexpr int32_t out1 = 1; // out1 (output1) — m 路更新: m·mul0_x + grad·mul1_x
        constexpr int32_t out2 = 2; // out2 (output2) — var 更新: var − truediv·input4

        int64_t innerCount = 1;
        for (int64_t d = td_->split.ubSplitIdx + 1; d < RANK; d++) {
            innerCount *= td_->maxBroShape[d];
        }

        int64_t coord[MAX_RANK] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t ubBlockLength =
                GetUBSplitRange(flat % td_->split.ubOuter, td_->split.ubOuter, td_->split.ubFactor, td_->split.ubTail);
            int64_t count = ubBlockLength * innerCount;
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.ubSplitIdx, td_->split.ubFactor,
                                 td_->split.ubOuter, coord);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inInput0, ubLoadData, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            AscendC::Mul(buf_[ubMid].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubLoadData].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inMul1X, ubLoadMulX, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            AscendC::Mul(buf_[ubOutM].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubLoadMulX].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inInput2, ubLoadData, ubBlockLength);
            CopyInBrc(coord, inMul0X, ubLoadMulX, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            uint16_t repeatTimesS3 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[ubOutM].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubLoadMulX].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubLoadData].Get<T>().GetPhyAddr(), count, VL, repeatTimesS3);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inMul3X, ubOutV, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            AscendC::Mul(buf_[ubOutV].Get<T>(), buf_[ubMid].Get<T>(), buf_[ubOutV].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inInput1, ubLoadData, ubBlockLength);
            CopyInBrc(coord, inMul2X, ubMid, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            uint16_t repeatTimesS5 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[ubOutV].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubLoadData].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[ubMid].Get<T>().GetPhyAddr(), count, VL, repeatTimesS5);
            AscendC::Sqrt(buf_[ubLoadData].Get<T>(), buf_[ubOutV].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inAdd2Y, ubMid, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            AscendC::Add(buf_[ubLoadMulX].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubMid].Get<T>(), count);
            AscendC::Div(buf_[ubMid].Get<T>(), buf_[ubOutM].Get<T>(), buf_[ubLoadMulX].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inInput4, ubLoadData, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            AscendC::Mul(buf_[ubLoadMulX].Get<T>(), buf_[ubMid].Get<T>(), buf_[ubLoadData].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
            CopyInBrc(coord, inInput3, ubLoadData, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

            AscendC::Mutex::Lock<PIPE_V>(mutexId);
            AscendC::Sub(buf_[ubMid].Get<T>(), buf_[ubLoadData].Get<T>(), buf_[ubLoadMulX].Get<T>(), count);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId);

            AscendC::Mutex::Lock<PIPE_MTE3>(mutexId);
            CopyOutOne(coord, out0, ubOutV, ubBlockLength);
            CopyOutOne(coord, out1, ubOutM, ubBlockLength);
            CopyOutOne(coord, out2, ubMid, ubBlockLength);
            AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId);
        }

        AscendC::ReleaseMutexID(mutexId);
    }

private:
    // CopyInBrc: 输入的 NDDMA 广播搬运——按本 tile 块长更新 loopSize 后 DataCopy 到 UB；
    // RANK ≤ 5 时单次覆盖所有维，> 5 时按 nddmaOuterIters_ 逐段搬运超出 NDDMA 窗口的外侧维。
    __aicore__ inline void CopyInBrc(const int64_t* coord, int32_t inputIdx, int32_t slot, int64_t ubBlockLength)
    {
        int64_t k = td_->split.ubSplitIdx;
        int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
        const int64_t* dstShape = td_->maxBroShape;

        auto params = nddmaParams_[inputIdx];
        int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < NDDMA_DIMS; nd++) {
            if (nd == kNd) {
                params.loopInfo.loopSize[nd] = ubBlockLength;
            }
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};

        if constexpr (RANK <= MAX_NDDMA_DIMS) {
            AscendC::DataCopy<T, NDDMA_DIMS, cfg>(buf_[slot].Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].Get<T>();
            int64_t elemBase = off;
            for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
                int64_t elemAdj = 0;
                int64_t tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, NDDMA_DIMS, cfg>(buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
            }
        }
    }

    // CopyOutOne: 用 DataCopyPad 把 UB 结果写回 GM——不要求 blockLen 32B 对齐，
    // 硬件只写有效字节，blockLen = count × sizeof(T)。
    __aicore__ inline void CopyOutOne(const int64_t* coord, int32_t outputIdx, int32_t slot, int64_t ubBlockLength)
    {
        int64_t off = CalcOffset(coord, td_->outputStrides[outputIdx], RANK);
        int64_t cnt = CalcTransferCount(td_->outputShapes[outputIdx], RANK, td_->split.ubSplitIdx, ubBlockLength);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].Get<T>(), extParams);
    }
};

// MulAddVF: RegBase VF 封装的乘累加——dst += src0 × src1，
// 按 repeat 分次 Load/MulAddDst/Store，用于 m/v 路的就地累加。
template <typename T>
__simd_vf__ inline void MulAddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count,
                                 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
#endif // ADAM_APPLY_ONE_ASSIGN_KERNEL_H_
