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
 * two-pass-amax-cast：离线可读的 Reg VF 注解摘录
 *
 * 这不是独立头文件。它保留关键指令顺序，并统一使用
 * AscendC::Reg 主命名空间和公开 load/store API。
 * xDtype/yDtype、CastTrait、VL、rowCount、tailNum、maxValue 等均来自目标类和 Tiling，
 * 接入时必须从目标工程补齐，不能为让摘录单独编译而猜默认值。
 */

#if 0 // annotated excerpt; paste the functions into a verified target class context

// 为什么在 load 时立刻转 FP32：amax、scale 和除法共用同一 FP32 数值域；
// 若先落一个 B16 中间 UB，再单独 Cast，会增加搬运并改变 smooth 的融合位置。
__aicore__ inline void DataCopyInputVF(
    __ubuf__ xDtype* xAddr, __ubuf__ xDtype* smoothAddr,
    AscendC::Reg::RegTensor<float>& vregRes, AscendC::Reg::MaskReg pregMask)
{
    AscendC::Reg::RegTensor<xDtype> vregX;
    AscendC::Reg::RegTensor<xDtype> vregSmooth;
    AscendC::Reg::RegTensor<float> vregSmoothFp32;

    AscendC::Reg::LoadAlign<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xAddr);
    AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vregRes, vregX, pregMask);
    if constexpr (hasSmooth) {
        AscendC::Reg::LoadAlign<xDtype, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregSmooth, smoothAddr);
        AscendC::Reg::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, pregMask);
        AscendC::Reg::Mul(vregRes, vregRes, vregSmoothFp32, pregMask);
    }
}

// 第一遍只求统计量，并把最终 scale/offset Duplicate 到整条寄存器。
// 为什么不把 scale 写 UB 后再读回来：第二遍紧接着消费它，寄存器传递可省一次往返；
// scaleAddr/offsetAddr 的非对齐回写只是算子输出，不是两遍之间的通信手段。
__aicore__ inline void ComputeScaleVF(
    __ubuf__ xDtype* xAddr, __ubuf__ xDtype* smoothAddr,
    __ubuf__ float* scaleAddr, __ubuf__ float* offsetAddr,
    AscendC::Reg::RegTensor<float>& vregDupScale,
    AscendC::Reg::RegTensor<float>& vregDupOffset, int32_t indexRow)
{
    AscendC::Reg::RegTensor<float> vregInput;
    AscendC::Reg::RegTensor<float> vregAbs;
    AscendC::Reg::RegTensor<float> vregScale;
    AscendC::Reg::RegTensor<float> vregMinX;
    AscendC::Reg::RegTensor<float> vregMaxX;
    AscendC::Reg::RegTensor<float> vregReduceMaxX;
    AscendC::Reg::RegTensor<float> vregReduceMinX;
    AscendC::Reg::RegTensor<float> vregMaxSubMin;
    AscendC::Reg::RegTensor<float> vregMaxDivScale;
    AscendC::Reg::RegTensor<float> vregNegMaxDivScale;
    AscendC::Reg::RegTensor<float> vregOffset;
    AscendC::Reg::RegTensor<float> vregReduceMaxXTail;
    AscendC::Reg::RegTensor<float> vregReduceMinXTail;
    AscendC::Reg::RegTensor<float> vregFinalMax;
    AscendC::Reg::RegTensor<float> vregFinalMin;

    AscendC::Reg::MaskReg pregLoop;
    AscendC::Reg::MaskReg pregAll =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg pregTail;
    AscendC::Reg::MaskReg pregLane0 =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::UnalignRegForStore scaleCursor;
    AscendC::Reg::UnalignRegForStore offsetCursor;

    uint16_t vfLoop = (rowCount + VL - 1) / VL;
    uint32_t remaining = rowCount;
    AscendC::Reg::Duplicate(vregMaxX, NEG_INFINITY, pregAll);

    if constexpr (isSymmetrical) {
        for (uint16_t j = 0; j < vfLoop; ++j) {
            pregLoop = AscendC::Reg::UpdateMask<float>(remaining);
            DataCopyInputVF(xAddr + indexRow * rowCount + j * VL,
                smoothAddr + j * VL, vregInput, pregLoop);
            AscendC::Reg::Abs(vregAbs, vregInput, pregLoop);
            // 用全 mask 累积是安全的前提：无效 lane 已由 zeroing Cast 和初始化值控制。
            AscendC::Reg::Max(vregMaxX, vregAbs, vregMaxX, pregAll);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(vregReduceMaxX, vregMaxX, pregAll);
        AscendC::Reg::Muls(vregScale, vregReduceMaxX, maxValue, pregAll);
        AscendC::Reg::Duplicate(vregDupScale, vregScale, pregAll);
        AscendC::Reg::StoreUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            scaleAddr, vregScale, scaleCursor, 1);
    } else {
        AscendC::Reg::Duplicate(vregMinX, POS_INFINITY, pregAll);
        // min/max 的无效 lane 不能用 0 参与；所以完整拍与最后尾拍分开归约。
        for (uint16_t j = 0; j + 1 < vfLoop; ++j) {
            pregLoop = AscendC::Reg::UpdateMask<float>(remaining);
            DataCopyInputVF(xAddr + indexRow * rowCount + j * VL,
                smoothAddr + j * VL, vregInput, pregLoop);
            AscendC::Reg::Max(vregMaxX, vregInput, vregMaxX, pregAll);
            AscendC::Reg::Min(vregMinX, vregInput, vregMinX, pregAll);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(vregReduceMaxX, vregMaxX, pregAll);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MIN>(vregReduceMinX, vregMinX, pregAll);

        uint32_t tailRemaining = tailNum;
        pregTail = AscendC::Reg::UpdateMask<float>(tailRemaining);
        DataCopyInputVF(xAddr + indexRow * rowCount + (vfLoop - 1) * VL,
            smoothAddr + (vfLoop - 1) * VL, vregInput, pregTail);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(vregReduceMaxXTail, vregInput, pregTail);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MIN>(vregReduceMinXTail, vregInput, pregTail);
        AscendC::Reg::Max(vregFinalMax, vregReduceMaxX, vregReduceMaxXTail, pregLane0);
        AscendC::Reg::Min(vregFinalMin, vregReduceMinX, vregReduceMinXTail, pregLane0);

        AscendC::Reg::Sub(vregMaxSubMin, vregFinalMax, vregFinalMin, pregLane0);
        AscendC::Reg::Muls(vregScale, vregMaxSubMin, offsetDivValue, pregLane0);
        AscendC::Reg::Duplicate(vregDupScale, vregScale, pregAll);
        AscendC::Reg::StoreUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            scaleAddr, vregScale, scaleCursor, 1);
        AscendC::Reg::Div<float, &mode>(vregMaxDivScale, vregFinalMax, vregScale, pregLane0);
        AscendC::Reg::Muls(vregNegMaxDivScale, vregMaxDivScale, NEGATIVE_ONE, pregLane0);
        AscendC::Reg::Adds(vregOffset, vregNegMaxDivScale, offsetValue, pregLane0);
        AscendC::Reg::Duplicate(vregDupOffset, vregOffset, pregAll);
        AscendC::Reg::StoreUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            offsetAddr, vregOffset, offsetCursor, 1);
    }
    AscendC::Reg::StoreUnAlignPost<float>(scaleAddr, scaleCursor, 0);
    if constexpr (!isSymmetrical) {
        AscendC::Reg::StoreUnAlignPost<float>(offsetAddr, offsetCursor, 0);
    }
}

// 第二遍重新 load 同一行，直接消费第一遍保留的 vregDupScale/vregDupOffset。
// Cast/Pack 的具体链与 yDtype 绑定，不能把 int8、int4、FP8 的路径互换。
__aicore__ inline void ComputeYVF(
    __ubuf__ xDtype* xAddr, __ubuf__ xDtype* smoothAddr,
    __ubuf__ yCopyDtype* yAddr,
    AscendC::Reg::RegTensor<float>& vregDupScale,
    AscendC::Reg::RegTensor<float>& vregDupOffset, int32_t indexRow)
{
    AscendC::Reg::RegTensor<float> vregInput;
    AscendC::Reg::RegTensor<float> vregXDivScale;
    AscendC::Reg::RegTensor<float> vregYFp32;
    AscendC::Reg::RegTensor<int16_t> vregYInt16;
    AscendC::Reg::RegTensor<half> vregYFp16;
    AscendC::Reg::RegTensor<yCopyDtype> vregY;
    AscendC::Reg::MaskReg preg;

    uint32_t remaining = rowCount;
    uint16_t vfLoop = (rowCount + VL - 1) / VL;
    for (uint16_t j = 0; j < vfLoop; ++j) {
        auto addr = yAddr + indexRow * outAlignLen + j * VL;
        preg = AscendC::Reg::UpdateMask<float>(remaining);
        DataCopyInputVF(xAddr + indexRow * rowCount + j * VL,
            smoothAddr + j * VL, vregInput, preg);
        if constexpr (isSymmetrical) {
            AscendC::Reg::Div(vregYFp32, vregInput, vregDupScale, preg);
        } else {
            AscendC::Reg::Div(vregXDivScale, vregInput, vregDupScale, preg);
            AscendC::Reg::Add(vregYFp32, vregXDivScale, vregDupOffset, preg);
        }

        if constexpr (IsSameType<yDtype, int8_t>::value) {
            // 三级 Cast 是目标 dtype 的真实转换语义，不可压成一个臆造的 CastTrait。
            AscendC::Reg::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregYFp32, preg);
            AscendC::Reg::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg);
            AscendC::Reg::Cast<yDtype, half, castTraitF16ToI8>(vregY, vregYFp16, preg);
        } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value ||
                             IsSameType<yDtype, fp8_e5m2_t>::value) {
            AscendC::Reg::Cast<yDtype, float, castTraitF32tofp8>(vregY, vregYFp32, preg);
        }
        AscendC::Reg::StoreAlign<yCopyDtype, AscendC::Reg::StoreDist::DIST_PACK4_B32>(addr, vregY, preg);
    }
}

// 两遍必须位于同一个 __VEC_SCOPE__，否则寄存器值不能作为跨 pass 的接口。
__aicore__ inline void ComputeVF(
    __ubuf__ xDtype* xAddr, __ubuf__ xDtype* smoothAddr,
    __ubuf__ yCopyDtype* yAddr, __ubuf__ float* scaleAddr,
    __ubuf__ float* offsetAddr, int32_t multiRow)
{
    __VEC_SCOPE__ {
        AscendC::Reg::RegTensor<float> vregDupScale;
        AscendC::Reg::RegTensor<float> vregDupOffset;
        for (uint16_t row = 0; row < static_cast<uint16_t>(multiRow); ++row) {
            ComputeScaleVF(xAddr, smoothAddr, scaleAddr + row, offsetAddr + row,
                vregDupScale, vregDupOffset, row);
            ComputeYVF(xAddr, smoothAddr, yAddr, vregDupScale, vregDupOffset, row);
        }
    }
}

#endif
