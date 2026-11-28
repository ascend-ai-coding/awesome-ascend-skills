// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "c_api/asc_simd.h"

// 卷积 AscendC 关键片段（annotated_excerpt）。
// 整理自用户提供的通道压紧、窗口复用与连续搬运实现。
// DAV_3510 / CANN 9.3；仅含关键操作，调用方负责分配、初始化与事件生命周期。
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace conv_perf_fragments {

// 空间元素数为 2 的幂；源/目标 UB 分离，目标 32B 对齐。
// padded 每通道步长 HW+Padding，compact 每通道步长 HW，均按 FP32 元素计。
// 外层先等源数据就绪；压紧后完成 V->MTE3 同步，MTE3 读完才复用 compact。
template <int Log2HW, int Channels, int Padding>
__aicore__ inline void CompactPaddedChannelsFp32(__ubuf__ float* padded, __ubuf__ float* compact)
{
    using namespace AscendC;
    static_assert(Log2HW >= 6 && Log2HW < 25 && Channels > 0 && Padding >= 0, "Aligned FP32 channel layout");
    constexpr int HW = 1 << Log2HW;
    static_assert(int64_t(Channels) * HW / 64 <= 65535, "VF loop bound");
    __VEC_SCOPE__
    {
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<int32_t> lane, offset, channel, index;
        Reg::RegTensor<float> value;
        Reg::Arange(lane, 0);
        for (uint16_t block = 0; block < uint16_t(Channels * HW / 64); ++block) {
            Reg::Adds(offset, lane, int32_t(block * 64), mask);
            Reg::ShiftRights(channel, offset, int16_t(Log2HW), mask);
            Reg::Muls(channel, channel, int32_t(Padding), mask);
            Reg::Add(index, offset, channel, mask);
            Reg::Gather(value, padded, (Reg::RegTensor<uint32_t>&)index, mask);
            Reg::StoreAlign(compact + block * 64, value, mask);
        }
    }
}

// 双缓冲输出的一次消费：packed 必须已获分配，并与 padded 不重叠。
// 外层启动前 SetFlag<MTE3_V>(slotEvent)；结束时 WaitFlag<MTE3_V> 回收全部槽。
// slotEvent 管理 MTE3->V 的槽复用，readyEvent 管理 V->MTE3 的本次数据就绪。
template <int Log2HW, int Channels, int Padding>
__aicore__ inline void CompactAndStoreFp32(__ubuf__ float* padded, AscendC::LocalTensor<float> packed,
                                           AscendC::GlobalTensor<float> dst, uint64_t dstOffset, event_t slotEvent,
                                           event_t readyEvent)
{
    using namespace AscendC;
    WaitFlag<HardEvent::MTE3_V>(slotEvent);
    CompactPaddedChannelsFp32<Log2HW, Channels, Padding>(padded, (__ubuf__ float*)packed.GetPhyAddr());
    SetFlag<HardEvent::V_MTE3>(readyEvent);
    WaitFlag<HardEvent::V_MTE3>(readyEvent);
    DataCopy(dst[dstOffset], packed, Channels * (1 << Log2HW));
    SetFlag<HardEvent::MTE3_V>(slotEvent);
}

// 连续 FP32 输入面批量搬运：src 面间距 planeElements，dst 间距 ubStrideElements。
// Host 令 rightPad=(8-planeElements%8)%8，ubStrideElements >= planeElements+rightPad，
// dst 起点/物理面间距为 32B 对齐；保护区和 SIMD 超出有效行的读取区域预先初始化。
// 外层在流水启动前调用 asc_set_copy_pad_val(0.0f)，保证 DMA 补齐值为零。
// 调用前等待旧消费者释放 dst；消费前配对等待 readyEvent，不在此复用事件号。
__aicore__ inline void LoadContiguousPlanesFp32(__ubuf__ float* dst, __gm__ float* src, uint16_t planes,
                                                uint32_t planeElements, uint32_t ubStrideElements, uint8_t rightPad,
                                                event_t readyEvent)
{
    asc_copy_gm2ub_align(dst, src, planes, planeElements * sizeof(float), 0, rightPad, true,
                         static_cast<asc_load_l2_cache_mode>(0), planeElements * sizeof(float),
                         ubStrideElements * sizeof(float));
    asc_sync_notify(PIPE_MTE2, PIPE_V, readyEvent);
}

// 核内输入双缓冲的生命周期；两个方向的事件号由调用方统一分配。
// 外层预置 V->MTE2 freeEvent；每次先 AcquireInputSlot，再 LoadContiguousPlanesFp32，
// 待 readyEvent 就绪后计算，计算结束 ReleaseInputSlot；退出时回收所有 freeEvent。
__aicore__ inline void AcquireInputSlot(event_t freeEvent)
{
    asc_sync_wait(PIPE_V, PIPE_MTE2, freeEvent);
}

__aicore__ inline void WaitInputReady(event_t readyEvent)
{
    asc_sync_wait(PIPE_MTE2, PIPE_V, readyEvent);
}

__aicore__ inline void ReleaseInputSlot(event_t freeEvent)
{
    asc_sync_notify(PIPE_V, PIPE_MTE2, freeEvent);
}

// 在 C API SIMD VF 内调用：同一输入行参与三个相邻输出的累加。
// 调用方已处理左右采样及边界；先 kh、再 kw 推进，每个输出保持原 FMA 顺序。
__simd_callee__ inline void AccumulateSharedRowFp32(vector_float& firstSum, vector_float& secondSum,
                                                    vector_float& thirdSum, vector_float row, float firstWeight,
                                                    float secondWeight, float thirdWeight, vector_bool mask)
{
    asc_axpy(firstSum, row, firstWeight, mask);
    asc_axpy(secondSum, row, secondWeight, mask);
    asc_axpy(thirdSum, row, thirdWeight, mask);
}

} // namespace conv_perf_fragments
#endif
