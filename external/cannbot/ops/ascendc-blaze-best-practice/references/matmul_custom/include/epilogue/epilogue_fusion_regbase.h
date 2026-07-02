/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_EPILOGUE_FUSION_REGBASE_H
#define EPILOGUE_EPILOGUE_FUSION_REGBASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "kernel_utils/common_utils.h"

using namespace AscendC;

// ============================================================================
// RegBase Epilogue 泛型框架（matmul + vector 后融合）
//
// 与 MemBase 版 epilogue_fusion_membase.h 对称，但 compute 核心使用 __VEC_SCOPE__
// + AscendC::Reg:: API（RegTensor / MaskReg），中间值不写回 UB。
//
// 适用场景：matmul + 复杂 vector 后处理（GELU / SwiGLU / LayerNorm 等），
// 需要减少 UB 中间 buffer 数量或追求更高 AIV 计算效率。
//
// 开发方法详见 matmul_fixpopti_regbase_epilogue.md。
//
// 三接口合约（与 MatmulKernelFused 兼容）：
//   1) struct Params { ... };
//   2) void Init(Params, baseM, baseN, problemShape)
//   3) auto GetTensor()  — 返回 cLocal_
//   4) void operator()(BlockShape, gmOffset, flagId)
//
// 用户定制清单（搜索 [USER] 标记）：
//   [USER T1] Params 结构体 — 额外输入的 GM_ADDR 字段
//   [USER T2] 额外 UB buffer 声明 + Init 中偏移计算
//   [USER T3] operator() 中额外输入的 DataCopyPad（GM→UB）
//   [USER T4] __VEC_SCOPE__ 内的 VF 计算链
// ============================================================================

#ifdef __CCE_AICORE__

class EpilogueFusionRegBase {
public:
    // ---- 类型定义 ----
    // L0CDataType: matmul 累加类型（int8 输入→int32，bf16/fp16 输入→float）
    // ComputeType: VF 内计算类型（通常 float）
    // OutputType:  最终输出类型（通常 bfloat16_t）
    using L0CDataType = int32_t;
    using ComputeType = float;
    using OutputType = bfloat16_t;

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;

    // ---- CastTrait 定义 ----
    // L0CType → ComputeType（如 int32→float，widening，无需 rounding）
    constexpr static AscendC::Reg::CastTrait castTraitL0CToCompute = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // ComputeType → OutputType（如 float→bf16，narrowing，需要 rounding）
    constexpr static AscendC::Reg::CastTrait castTraitComputeToOutput = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // [USER T1] Params 结构体 — 声明额外输入的 GM 地址
    // 示例：
    //   struct Params {
    //       GM_ADDR extraInputAddr{nullptr};   // 额外输入（如 scale、bias）
    //       GM_ADDR outputGmAddr{nullptr};     // 输出
    //   };
    struct Params {
        GM_ADDR extraInputAddr{nullptr};
        GM_ADDR outputGmAddr{nullptr};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // ---- UB buffer 声明 ----
    // cLocal_: matmul 结果（Fixpipe 写入 UB offset 0，大小固定，不可释放）
    AscendC::LocalTensor<L0CDataType> cLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};

    // [USER T2] 额外 UB buffer 声明
    // 示例：1 行 scale 输入 + stageRows 行输出
    //   AscendC::LocalTensor<ComputeType> extraBuf_{...};
    //   AscendC::LocalTensor<OutputType> bf16Out_{...};
    AscendC::LocalTensor<ComputeType> extraBuf_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<OutputType> bf16Out_{AscendC::TPosition::VECIN, 0, UB_SIZE};

    // ---- GM tensor 声明 ----
    AscendC::GlobalTensor<OutputType> outputGlobal_;
    // [USER T2] 额外输入的 GlobalTensor
    AscendC::GlobalTensor<ComputeType> extraInputGlobal_;

    int64_t stageRows_{0};
    ProblemShape problemShape_;

    // ---- Init: UB 布局计算 ----
    __aicore__ inline void Init(
        Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        // 计算 nAlign（int32 对齐）和 nAlignBf16（bf16 对齐）
        constexpr int64_t ALIGN_I32 = 32 / sizeof(L0CDataType);
        int64_t nAlign = ::CeilDiv(l1N, ALIGN_I32) * ALIGN_I32;

        // matmulArea: cLocal_ 占用的 UB 空间（Fixpipe 写入，固定不可释放）
        int64_t splitTaskRation = static_cast<int64_t>(AscendC::GetTaskRation());
        int64_t l1MSplit = ::CeilDiv(l1M, splitTaskRation);
        int64_t matmulAreaBytes = l1MSplit * nAlign * sizeof(L0CDataType);

        // [USER T2] 计算额外 buffer 大小和 stageRows_
        // 示例：extraBuf_ = 1 行 float，bf16Out_ = stageRows 行 bf16
        //   extraBufBytes = nAlign × sizeof(ComputeType)
        //   remainBytes = UB_SIZE - matmulAreaBytes - extraBufBytes
        //   stageRows_ = remainBytes / (nAlign × sizeof(OutputType))
        int64_t extraBufBytes = nAlign * sizeof(ComputeType);
        int64_t remainBytes = UB_SIZE - matmulAreaBytes - extraBufBytes;
        stageRows_ = remainBytes / (nAlign * sizeof(OutputType));

        // [USER T2] 计算 UB 偏移，通过 ReinterpretCast 划分区域
        // cLocal_ 占用 [0, matmulAreaBytes)
        // extraBuf_ 从 matmulArea 元素偏移开始
        int64_t extraBufElemOffset = l1MSplit * nAlign;
        extraBuf_ = cLocal_[extraBufElemOffset].template ReinterpretCast<ComputeType>();

        // bf16Out_ 从 extraBuf_ 之后开始
        int64_t bf16OutByteOffset = matmulAreaBytes + extraBufBytes;
        bf16Out_ = cLocal_[bf16OutByteOffset / sizeof(L0CDataType)].template ReinterpretCast<OutputType>();

        // 绑定 GM 地址
        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(params.outputGmAddr));
        // [USER T2] 绑定额外输入的 GM 地址
        extraInputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ComputeType*>(params.extraInputAddr));
    
        // 在Init中预发射首轮所有反向同步的SetFlag
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

    __aicore__ inline auto GetTensor() { return cLocal_; }

    // ---- operator(): 逐 tile 处理 ----
    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t blockShapeN = Get<1>(blockShape);

        // SPLIT_M: 每个 AIV 处理 halfM 行
        int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                          ? (halfM - AscendC::GetSubBlockIdx()) : halfM;

        int64_t N = Get<MNK_N>(problemShape_);

        if (blockShapeM <= 0) {
            return;
        }

        int64_t curStageRows = AscendC::Std::min(stageRows_, halfM);
        if (curStageRows <= 0) {
            return;
        }

        // [USER T3] 加载额外输入（GM→UB）
        // 示例：加载 1 行 scale 数据
        //   int64_t nPos = dstOffset % N;
        //   DataCopyPad(extraBuf_, extraInputGlobal_[nPos], ...);

        // MTE2反向等待上一轮Vec消费
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        {
            int64_t nPos = dstOffset % N;
            uint16_t nRows = 1;
            uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(ComputeType));
            uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(ComputeType));
            AscendC::DataCopyExtParams cp{nRows, rowBytes, gmRowGap, 0, 0};
            AscendC::DataCopyPadExtParams<ComputeType> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(extraBuf_, extraInputGlobal_[nPos], cp, pp);
        }

        // 等待 MTE2（DataCopyPad）完成，VEC 才能读取 extraBuf_
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

        // 获取 UB 物理地址（__ubuf__ 指针，供 VF 内使用）
        __ubuf__ L0CDataType* srcAddr = (__ubuf__ L0CDataType*)cLocal_.GetPhyAddr();
        __ubuf__ ComputeType* extraAddr = (__ubuf__ ComputeType*)extraBuf_.GetPhyAddr();
        __ubuf__ OutputType* dstAddr = (__ubuf__ OutputType*)bf16Out_.GetPhyAddr();

        // per-call 计算 nAlign（从 blockShapeN，非 Init 时的 l1N，正确处理 tail tile）
        constexpr int64_t ALIGN_I32 = 32 / sizeof(L0CDataType);
        int64_t nAlign = ::CeilDiv(blockShapeN, ALIGN_I32) * ALIGN_I32;
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(ComputeType);
        uint16_t vfLoopNum = (static_cast<uint32_t>(nAlign) + VL - 1) / VL;

        // ---- Stage 循环（按行切分）----
        int64_t stageOffset = 0;
        int64_t totalRows = halfM;

        while (stageOffset < totalRows) {
            int64_t rowsThisStage = AscendC::Std::min(curStageRows, totalRows - stageOffset);
            
            // V反向等待上一轮MTE3搬运结束才能开始
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
            // ---- VF 计算（__VEC_SCOPE__）----
            __VEC_SCOPE__
            {
                // 声明 RegTensor 变量
                AscendC::Reg::RegTensor<L0CDataType> vregL0C;
                AscendC::Reg::RegTensor<ComputeType> vregCompute;
                // [USER T4] 按需声明更多 RegTensor（中间值、额外输入等）
                AscendC::Reg::RegTensor<ComputeType> vregExtra;
                AscendC::Reg::RegTensor<OutputType> vregOutput;
                AscendC::Reg::MaskReg mask;

                // 行循环（必须 uint16_t）
                for (uint16_t row = 0; row < static_cast<uint16_t>(rowsThisStage); row++) {
                    __ubuf__ L0CDataType* rowSrc = srcAddr + (stageOffset + row) * nAlign;
                    __ubuf__ OutputType* rowDst = dstAddr + row * nAlign;

                    // VL 循环（必须 uint16_t）
                    for (uint16_t i = 0; i < vfLoopNum; i++) {
                        uint32_t active = static_cast<uint32_t>(nAlign) - static_cast<uint32_t>(i) * VL;
                        if (active > VL) active = VL;
                        mask = AscendC::Reg::UpdateMask<ComputeType>(active);

                        // Step a: 加载 matmul 结果（UB→Reg, DIST_NORM: 连续 L0CType 块）
                        AscendC::Reg::LoadAlign(vregL0C, rowSrc + i * VL);

                        // Step b: Cast L0CType → ComputeType
                        AscendC::Reg::Cast<ComputeType, L0CDataType, castTraitL0CToCompute>(
                            vregCompute, vregL0C, mask);

                        // [USER T4] 融合计算链 — 在此处实现具体公式
                        //
                        // 可用的 Reg:: API 族（详见 matmul_fixpopti_regbase_epilogue.md §4）：
                        //   二元:   Reg::Add / Sub / Mul / Div(dst, src0, src1, mask)
                        //   标量:   Reg::Adds / Muls / Divs(dst, src, scalar, mask)
                        //   FMA:    Reg::Axpy(dst, src, scalar, mask)  // dst += src × scalar
                        //   数学:   Reg::Exp / Log / Abs / Sqrt(dst, src, mask)
                        //   比较:   Reg::Compare / Compares / Select
                        //
                        // 加载额外输入（按 §4.2.2 选择 LoadDist）:
                        //   连续块: Reg::LoadAlign(vregExtra, extraAddr + i * VL);
                        //   标量广播: Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(vregToken, tokenAddr + row);
                        //
                        // 示例（perchannel scale + pertoken scale + GELU）:
                        //   Reg::LoadAlign(vregScale, scaleAddr + i * VL);           // DIST_NORM
                        //   Reg::Mul(vregCompute, vregCompute, vregScale, mask);
                        //   Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(vregToken, tokenAddr + row);
                        //   Reg::Mul(vregCompute, vregCompute, vregToken, mask);
                        //   // ... GELU chain ...

                        // Step y: Cast ComputeType → OutputType
                        AscendC::Reg::Cast<OutputType, ComputeType, castTraitComputeToOutput>(
                            vregOutput, vregCompute, mask);

                        // Step z: 写回 UB（Reg→UB, DIST_PACK_B32: float→bf16 打包）
                        AscendC::Reg::StoreAlign<OutputType, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            rowDst + i * VL, vregOutput, mask);
                    }
                }
            }

            // 等待 VEC store 完成，MTE3 才能读取 bf16Out_
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            // DataCopyPad: UB bf16Out_ → GM output
            {
                int64_t gmRowOffset = dstOffset + stageOffset * N;
                gmRowOffset += AscendC::GetSubBlockIdx() * halfM * N;
                uint16_t nRows = static_cast<uint16_t>(rowsThisStage);
                uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(OutputType));
                uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(OutputType));
                int64_t ubRowGapBytes = (nAlign - blockShapeN) * static_cast<int64_t>(sizeof(OutputType));
                AscendC::DataCopyExtParams outParams{nRows, rowBytes, ubRowGapBytes, gmRowGap, 0};
                AscendC::DataCopyPad<OutputType>(outputGlobal_[gmRowOffset], bf16Out_, outParams);
            }
            // MTE3搬运完成，通知下一轮V可以开始计算
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);

            stageOffset += rowsThisStage;
        }
        // 通知下一轮MTE2可以开始搬运，非多轮搬运时可删除
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
    }

    __host_aicore__ static Params InitParams(Params const& args) { return args; }

    __aicore__ ~EpilogueFusionRegBase()
    {
        // 析构函数中设置尾所有反向等待的WaitFlag
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }
};

#endif // __CCE_AICORE__

#endif // EPILOGUE_EPILOGUE_FUSION_REGBASE_H
