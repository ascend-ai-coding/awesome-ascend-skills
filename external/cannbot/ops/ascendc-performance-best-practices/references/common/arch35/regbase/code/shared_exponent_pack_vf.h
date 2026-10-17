/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * shared-exponent-pack：离线可读的 Reg VF 注解摘录
 *
 * 这不是独立头文件。它摘录真实的编排、指数修正和 dtype 分派；寄存器结构、
 * 常量、LoadData/ProcessQuantization* 以及 Tiling 成员必须由目标工程提供。
 */

#if 0 // annotated excerpt; preserve the surrounding class and constants when integrating

// 顶层顺序不能打散：先从输入构造 max exponent，再生成 shared exponent/reciprocal，
// 写 scale，最后用同一组寄存器状态量化并 Pack 输出。
template <const bool canInterleave, const bool canMaxLowBound>
__aicore__ inline void Compute(
    __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr,
    __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* lhsYAddr,
    __ubuf__ uint8_t* rhsYAddr, uint16_t loop0, uint16_t loop1)
{
    __VEC_SCOPE__ {
        ComputeRegisters lhsRegs;
        ComputeRegisters rhsRegs;
        AuxRegisters auxRegs;
        InitializeConstants<canInterleave>(auxRegs);

        ComputeMaxExponents<canInterleave>(
            lhsRegs, rhsRegs, auxRegs, lhsXAddr, rhsXAddr, loop0, loop1);

        if constexpr (!IsSame<DTYPE_X, float>::value) {
            ComputeScalesAndSharedExp<canMaxLowBound>(lhsRegs, auxRegs);
            // B16 输入的 lhs/rhs scale 进入同一个 uint8 输出向量：LOWEST/HIGHEST
            // 决定各占哪一半，Or 只负责合并，三步不能拆开。
            AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(
                (AscendC::Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, lhsRegs.mxScale);
            if constexpr (!canInterleave) {
                ComputeScalesAndSharedExp<canMaxLowBound>(rhsRegs, auxRegs);
                AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::HIGHEST>(
                    (AscendC::Reg::RegTensor<uint8_t>&)rhsRegs.mxScale, rhsRegs.mxScale);
                AscendC::Reg::Or(lhsRegs.mxScale, lhsRegs.mxScale, rhsRegs.mxScale, auxRegs.p0);
            }
            AscendC::Reg::StoreAlign<uint8_t>(
                mxScaleAddr, (AscendC::Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, auxRegs.p2);
        } else {
            // FP32 路径的 lhs/rhs scale 分地址回写，不能照搬 B16 路径的 HIGHEST+Or。
            ComputeScalesAndSharedExp<canMaxLowBound>(lhsRegs, auxRegs);
            AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(
                (AscendC::Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, lhsRegs.mxScale);
            AscendC::Reg::StoreAlign<uint8_t>(
                mxScaleAddr, (AscendC::Reg::RegTensor<uint8_t>&)lhsRegs.mxScale, auxRegs.p2);
            if constexpr (!canInterleave) {
                ComputeScalesAndSharedExp<canMaxLowBound>(rhsRegs, auxRegs);
                AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(
                    (AscendC::Reg::RegTensor<uint8_t>&)rhsRegs.mxScale, rhsRegs.mxScale);
                AscendC::Reg::StoreAlign<uint8_t>(mxScaleAddr + alignedPostAxisSize_,
                    (AscendC::Reg::RegTensor<uint8_t>&)rhsRegs.mxScale, auxRegs.p2);
            }
        }

        QuantizeValues<canInterleave>(lhsRegs, rhsRegs, auxRegs,
            lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    }
}

// ComputeMaxExponents 的完整实现依赖输入 dtype 和 canInterleave 分支。
// 它必须产出后续函数使用的 absMax/expMax 字段，不能改成普通 ReduceMax 后丢掉位级表示。
template <const bool canInterleave>
__aicore__ inline void ComputeMaxExponents(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs,
    __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr,
    uint16_t loop0, uint16_t loop1);

// dtype 分派放在编译期，避免在 VF 热循环里为 FP32/B16 的不同位级路径增加运行期分支。
template <const bool canMaxLowBound>
__aicore__ inline void ComputeScalesAndSharedExp(
    ComputeRegisters& regs, AuxRegisters& auxRegs)
{
    if constexpr (IsSame<DTYPE_X, float>::value) {
        ComputeScalesAndSharedExpFP32<canMaxLowBound>(regs, auxRegs);
    } else {
        ComputeScalesAndSharedExpNonFP32<canMaxLowBound>(regs, auxRegs);
    }
}

// reciprocal 不是通用 Div：这里直接变换指数位，并显式修正 Inf、特殊指数和 0。
// 任一 Select 被删掉都会改变特殊值语义，必须和目标 golden 做位级验证。
__aicore__ inline void ComputeReciprocalScale(
    ComputeRegisters& regs, AuxRegisters& auxRegs, bool isModeOne)
{
    if (isModeOne) {
        AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(
            auxRegs.infMask, regs.expMaxU16, auxRegs.maxEleU16, auxRegs.p0);
    } else {
        AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(
            auxRegs.zeroMask, regs.expMaxU16, auxRegs.zeroU16, auxRegs.p0);
    }
    AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(
        auxRegs.specialMask, regs.expMaxU16, auxRegs.biasU16, auxRegs.p0);
    AscendC::Reg::Sub(regs.expMaxU16, auxRegs.biasU16, regs.expMaxU16, auxRegs.p0);
    AscendC::Reg::Select<uint16_t>(
        regs.expMaxU16, auxRegs.nanU16, regs.expMaxU16, auxRegs.infMask);
    AscendC::Reg::Select<uint16_t>(
        regs.expMaxU16, auxRegs.specialExpU16, regs.expMaxU16, auxRegs.specialMask);
    if (!isModeOne) {
        AscendC::Reg::Select<uint16_t>(
            regs.expMaxU16, auxRegs.zeroU16, regs.expMaxU16, auxRegs.zeroMask);
    }
}

// FP32 的 RNE 不是“随便右移”：0x7FFF + 截断位最低 bit 实现 ties-to-even。
// 这段必须与随后的 Pack、指数掩码和特殊值 Select 作为整体迁移。
template <const bool canMaxLowBound>
__aicore__ inline void ComputeScalesAndSharedExpFP32(
    ComputeRegisters& regs, AuxRegisters& auxRegs)
{
    if constexpr (calcMode == MODE_ONE || calcMode == MODE_THREE) {
        ComputeScaleFromAbsMaxFP32<false, canMaxLowBound>(
            auxRegs, regs.absMaxU32, regs.mxScale0U32, regs.mxScale0Add1U32, regs.mxScale);
        AscendC::Reg::ShiftLefts(regs.expMaxU16, regs.mxScale, BF16_SHR_NUM, auxRegs.p0);
        ComputeReciprocalScale(regs, auxRegs, true);
    } else {
        if constexpr (calcMode == MODE_TWO) {
            AscendC::Reg::ShiftRights(
                regs.roundBiasU32, regs.absMaxU32, FP32_PACK_SHR_NUM, auxRegs.p1);
            AscendC::Reg::And(
                regs.roundBiasU32, regs.roundBiasU32, auxRegs.oneU32, auxRegs.p1);
            AscendC::Reg::Adds(regs.roundBiasU32, regs.roundBiasU32, 0x7FFF, auxRegs.p1);
            AscendC::Reg::Add(
                regs.absMaxU32, regs.absMaxU32, regs.roundBiasU32, auxRegs.p1);
            AscendC::Reg::ShiftRights(
                regs.absMaxU32, regs.absMaxU32, FP32_PACK_SHR_NUM, auxRegs.p1);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(
                regs.absMaxU16, regs.absMaxU32);
            AscendC::Reg::And(
                regs.expMaxU16, regs.absMaxU16, auxRegs.maxEleU16, auxRegs.p0);
        }
        // 其余模式在原实现中有不同的 expMax 来源，但最终都必须经过特殊值修正。
        ComputeReciprocalScale(regs, auxRegs, false);
    }
}

// 最后的 dtype 分派仍在同一 VF 中；不要先把中间 FP32/B16 结果落 UB 再开一条 Pack pass。
template <const bool canInterleave>
__aicore__ inline void QuantizeValues(
    ComputeRegisters& lhsRegs, ComputeRegisters& rhsRegs, AuxRegisters& auxRegs,
    __ubuf__ DTYPE_X* lhsXAddr, __ubuf__ DTYPE_X* rhsXAddr,
    __ubuf__ uint8_t* lhsYAddr, __ubuf__ uint8_t* rhsYAddr,
    uint16_t loop0, uint16_t loop1)
{
    if constexpr (IsSame<DTYPE_X, float>::value) {
        QuantizeValuesFP32<canInterleave>(lhsRegs, rhsRegs, auxRegs,
            lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    } else {
        QuantizeValuesNonFP32<canInterleave>(lhsRegs, rhsRegs, auxRegs,
            lhsXAddr, rhsXAddr, lhsYAddr, rhsYAddr, loop0, loop1);
    }
}

#endif
