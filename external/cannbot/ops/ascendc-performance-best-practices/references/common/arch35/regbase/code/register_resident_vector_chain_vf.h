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
 * \file register_resident_vector_chain_vf.h
 * \brief 通用向量计算链的寄存器驻留 VF 集成骨架
 *
 * 适用架构：A5 / DAV_3510（VL = 256B / sizeof(float) = 64 个 fp32 元素）。
 * 命名空间：源码统一写 AscendC::Reg::；`using namespace AscendC;` 后计算指令裸调用。
 *
 * 面向形态：已确认属于 AIV、且在 UB 内接力的向量计算链
 * （elementwise / activation / 其他向量运算）。Cube MMAD/内嵌 MatMul 不适用。
 * LocalTensor 形态下每个向量 API 各带一次 UB 读 +
 * 一次 UB 写；本骨架把整条链搬进寄存器：入口 LoadAlign 一次、链内 RegTensor
 * 接力、出口 StoreAlign 一次，并消除链内一切 UB staging（含"先 Cast 到 T
 * 再 Cast 回 fp32"一类经 UB 中转的往返）。
 *
 * 核心结构（与 register_resident_vector_chain.md 对应）：
 *   1. 单一 __VEC_SCOPE__ VF 计算域；每迭代处理一个 64-fp32 元素 tile，
 *      uint16_t 循环计数从 0 起步步长 1，体内无 break/continue/运行期 if
 *      （唯一的体内分支是 if constexpr(kRoundToInput)）；
 *   2. UB↔寄存器流量：输入 Reg::LoadAlign<T, DIST_UNPACK_B16>（不带
 *      mask），输出 Reg::StoreAlign<T, DIST_PACK_B32>（带当拍 MaskReg）；
 *      每拍以 Reg::UpdateMask<float>(uint32_t&) 生成/窄化 mask（按宽类型
 *      float 生成、计数递减），尾拍自动缩窄；
 *   3. tile 偏移为 uint32_t（i*64 元素步进）；VF 体内不出现 64 位偏移
 *      变量；不用 UnalignReg/LoadUnAlign/StoreUnAlign 与 CreateAddrReg
 *      （三个 __ubuf__ 指针线性步进，由调用方从 LocalTensor.GetPhyAddr()
 *      派生）；
 *   4. Reg::Cast 的 CastTrait 镜像 kernel 侧 Cast 降低取值
 *      （CastIntrinsicsImplVF：{ZERO, SAT, ZEROING, roundMode}）——SAT 是
 *      kernel 侧基线饱和行为，勿照抄其他骨架的 NO_SAT；
 *   5. tiling 派生标量在循环外计算一次（调用方 hoist 后按值传入），循环
 *      体内不重算标量除法；
 *   6. 不使用 asc_vf_call（其调用期间 MTE2 被阻塞，会切断 CopyIn 预取与
 *      VEC 的重叠）；__VEC_SCOPE__ 直呼形式保持外层流水不变。
 *
 * 拷贝使用：把 VectorChainVF 的示例链体换成目标链的实际 Reg 算子序列；
 * 入口/出口 Cast 的 roundMode 按 kernel 侧原语义选择。源构建契约为
 * device-only（--cce-aicore-only）；编译、精度与性能需在目标工程验证。
 */
#ifndef REGISTER_RESIDENT_VECTOR_CHAIN_VF_H
#define REGISTER_RESIDENT_VECTOR_CHAIN_VF_H

#include "kernel_operator.h"

using namespace AscendC;

namespace vfchain {
// DAV_3510 VL = 256B；fp32 每拍 64 个元素。
constexpr uint32_t VF_TILE_FLOATS = 64U;
} // namespace vfchain

// 示例链（可替换）：两条 sigmoid 型分支相乘的激活链，fp32 全程驻留寄存器。
// kRoundToInput 控制"舍入回输入精度"的寄存器内往返（fp32 -> T CAST_RINT ->
// fp32 CAST_NONE），替代原先经 UB 中转的 staging；该模板参数同时演示把
// 运行期分支 hoist 出 VF 体的做法（两个实例仅差 if constexpr 分支）。
// 入参：__ubuf__ T* 由调用方从 LocalTensor.GetPhyAddr() 派生；六个标量
// （negInvA/aVal/negA/negInvB/bVal/negB）由调用方在循环外 hoist 计算。
template <typename T, bool kRoundToInput>
__aicore__ inline void VectorChainVF(__ubuf__ T* inA, __ubuf__ T* inB, __ubuf__ T* out, uint32_t elements,
                                     float negInvA, float aVal, float negA, float negInvB, float bVal, float negB)
{
    // CastTrait 镜像 kernel 侧 Cast 降低（dav_3510 CastIntrinsicsImplVF:
    // {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, roundMode}）。
    static constexpr Reg::CastTrait kF32ToTCastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait kTToF32CastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

    const uint16_t tileCount =
        static_cast<uint16_t>((elements + vfchain::VF_TILE_FLOATS - 1U) / vfchain::VF_TILE_FLOATS);
    uint32_t remaining = elements;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> aF;
        Reg::RegTensor<float> bF;
        Reg::RegTensor<float> tmpF;
        Reg::RegTensor<T> packedF;
        Reg::MaskReg preg;
        for (uint16_t i = 0; i < tileCount; ++i) {
            preg = Reg::UpdateMask<float>(remaining);
            const uint32_t offset = static_cast<uint32_t>(i) * vfchain::VF_TILE_FLOATS;
            // 入口：unpack load + T->fp32 Cast，fp32 值从此驻留寄存器。
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, inA + offset);
            Reg::Cast<float, T, kTToF32CastTrait>(aF, packedF, preg);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, inB + offset);
            Reg::Cast<float, T, kTToF32CastTrait>(bF, packedF, preg);

            // —— 示例链体：换成目标链的实际 Reg 算子序列 ——
            // 默认模式 Exp/Div（INTRINSIC）与标量形式 Muls/Adds/Duplicate，
            // 与 kernel 侧 LocalTensor 降低一致（位相同的结构性来源）。
            Reg::Muls(tmpF, aF, negInvA, preg);
            Reg::Exp(tmpF, tmpF, preg);
            Reg::Adds(tmpF, tmpF, 1.0f, preg);
            Reg::Duplicate(aF, aVal, preg);
            Reg::Div(aF, aF, tmpF, preg);
            Reg::Muls(aF, aF, 2.0f, preg);
            Reg::Adds(aF, aF, negA, preg);

            Reg::Muls(tmpF, bF, negInvB, preg);
            Reg::Exp(tmpF, tmpF, preg);
            Reg::Adds(tmpF, tmpF, 1.0f, preg);
            Reg::Duplicate(bF, bVal, preg);
            Reg::Div(bF, bF, tmpF, preg);
            Reg::Muls(bF, bF, 2.0f, preg);
            Reg::Adds(bF, bF, negB, preg);

            if constexpr (kRoundToInput) {
                // 寄存器内往返舍入：fp32 -> T (CAST_RINT) -> fp32
                // (CAST_NONE)；原先经 UB 中转的 staging 到此消除。
                Reg::Cast<T, float, kF32ToTCastTrait>(packedF, aF, preg);
                Reg::Cast<float, T, kTToF32CastTrait>(aF, packedF, preg);
                Reg::Cast<T, float, kF32ToTCastTrait>(packedF, bF, preg);
                Reg::Cast<float, T, kTToF32CastTrait>(bF, packedF, preg);
            }
            Reg::Mul(aF, aF, bF, preg);

            // 出口：fp32 -> T（roundMode 按原 kernel 语义）+ packed store。
            Reg::Cast<T, float, kF32ToTCastTrait>(packedF, aF, preg);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(out + offset, packedF, preg);
        }
    }
}

#endif // REGISTER_RESIDENT_VECTOR_CHAIN_VF_H
