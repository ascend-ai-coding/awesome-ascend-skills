/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_EPILOGUE_FUSION_MEMBASE_H
#define EPILOGUE_EPILOGUE_FUSION_MEMBASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "../utils/common_utils.h"

using namespace AscendC;

// ============================================================================
// MemBase Epilogue 参考样例 — matmul + 简单 Vector 融合
//
// 展示一个完整的 MemBase Epilogue 骨架，覆盖：
//   - SplitM 偏移计算（UB 读取无偏移 / GM 读写有偏移）
//   - Init 预发射反向依赖 / 析构排空
//   - DataCopyPad stride 参数（UB 侧传 0，GM 侧传 bytes）
//   - 统一的 V_MTE2 / MTE3_V 同步模式（与 RegBase 一致）
//
// 计算段只保留 [USER COMPUTE] 骨架，开发者替换为具体 AscendC API 调用。
// ============================================================================

class MulEpilogue {
public:
    using DataType = float;

    static constexpr uint16_t ZERO_FLAG = 0;

    struct Params {
        GM_ADDR extraInputAddr{nullptr};
        GM_ADDR outputGmAddr{nullptr};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    AscendC::LocalTensor<DataType> cLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::LocalTensor<DataType> dLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::LocalTensor<DataType> cLocalTmp_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};

    AscendC::GlobalTensor<DataType> outputGlobal_;
    AscendC::GlobalTensor<DataType> extraInputGlobal_;

    int64_t stageRows_{0};
    int64_t nAlignL0C_{0};
    ProblemShape problemShape_;

    __aicore__ inline void Init(
        Params const& params, int64_t baseM, int64_t baseN, ProblemShape& problemShape)
    {
        // nAlign: UB 行对齐宽度（32B / sizeof(DataType) 元素一组）
        nAlignL0C_ = ::CeilDiv(baseN, ALIGN_ELEM) * ALIGN_ELEM;

        // matmulArea: cLocal_ 占用空间，行步长是 nAlignL0C_（不是 L0C cube 边长 16）
        int64_t splitTaskRation = static_cast<int64_t>(AscendC::GetTaskRation());
        int64_t splitMRows = ::CeilDiv(baseM, splitTaskRation);
        int64_t matmulArea = splitMRows * nAlignL0C_;
        int64_t matmulAreaBytes = matmulArea * sizeof(DataType);

        // dLocal_ 和 cLocalTmp_ 各占 stageRows 行
        int64_t remainBytes = AscendC::TOTAL_UB_SIZE - matmulAreaBytes;
        int64_t stagePerRowBytes = nAlignL0C_ * sizeof(DataType) * 2;  // dLocal_ + cLocalTmp_
        stageRows_ = remainBytes / stagePerRowBytes;
        if (stageRows_ <= 0) {
            stageRows_ = 1;
        }

        int64_t ubOffset = matmulArea;
        dLocal_ = cLocal_[ubOffset];
        ubOffset += stageRows_;
        cLocalTmp_ = cLocal_[ubOffset];

        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(params.outputGmAddr));
        extraInputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(params.extraInputAddr));

        // 预发射首轮反向依赖
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        (void)flagId;
        int64_t curM = Get<0>(blockShape);
        int64_t curN = Get<1>(blockShape);

        // ---- SplitM 行数计算 ----
        int64_t halfM = ::CeilDiv(curM, AscendC::GetTaskRation());
        int64_t localRows = ((static_cast<uint64_t>(curM) & 1UL) > 0UL)
                              ? (halfM - AscendC::GetSubBlockIdx()) : halfM;
        if (localRows <= 0) {
            return;  // V1 无数据，CV 同步由 kernel 层处理
        }

        int64_t N = Get<MNK_N>(problemShape_);
        int64_t tileM0 = dstOffset / N;
        int64_t tileN0 = dstOffset % N;
        int64_t subM0 = tileM0 + AscendC::GetSubBlockIdx() * halfM;

        // per-call nAlign（从 blockShapeN，非 Init 时的 baseN）
        int64_t nAlign = ::CeilDiv(curN, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t curStageRows = AscendC::Std::min(stageRows_, localRows);

        // ---- Stage 循环 ----
        for (int64_t stageOffset = 0; stageOffset < localRows; stageOffset += curStageRows) {
            int64_t rowsThisStage = AscendC::Std::min(curStageRows, localRows - stageOffset);
            int64_t stageM0 = subM0 + stageOffset;

            // 等 V 读完上一轮 dLocal_
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);

            // GM→UB 加载额外输入（GM offset 含 sub-block 偏移）
            {
                int64_t gmOffset = stageM0 * N + tileN0;
                uint16_t nRows = static_cast<uint16_t>(rowsThisStage);
                uint32_t rowBytes = static_cast<uint32_t>(curN * sizeof(DataType));
                uint32_t gmRowGap = static_cast<uint32_t>((N - curN) * sizeof(DataType));
                // GM→UB: srcStride=GM 侧 bytes, dstStride=UB 侧 32B 单位（nAlign 对齐时传 0）
                AscendC::DataCopyExtParams cp{nRows, rowBytes, gmRowGap, 0, 0};
                AscendC::DataCopyPadExtParams<DataType> pp{false, 0, 0, 0};
                AscendC::DataCopyPad(dLocal_, extraInputGlobal_[gmOffset], cp, pp);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

            // 等上一轮 MTE3 读完 cLocalTmp_
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);

            // [USER] Vector 计算 — 替换为实际 AscendC API
            //   AscendC::Mul(cLocalTmp_, cLocal_[stageOffset * nAlign], dLocal_, rowsThisStage * nAlign);
            //   AscendC::Add / Div / Cast / ...

            // 通知 MTE2 可覆盖 dLocal_
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);

            // 等 V 完成，MTE3 才能读 cLocalTmp_
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            // UB→GM 写回（GM offset 含 sub-block 偏移）
            {
                int64_t gmOffset = stageM0 * N + tileN0;
                uint16_t nRows = static_cast<uint16_t>(rowsThisStage);
                uint32_t rowBytes = static_cast<uint32_t>(curN * sizeof(DataType));
                uint32_t gmRowGap = static_cast<uint32_t>((N - curN) * sizeof(DataType));
                // UB→GM: srcStride=UB 侧 32B 单位（nAlign 对齐时传 0），dstStride=GM 侧 bytes
                AscendC::DataCopyExtParams outParams{nRows, rowBytes, 0, gmRowGap, 0};
                AscendC::DataCopyPad<DataType>(outputGlobal_[gmOffset], cLocalTmp_, outParams);
            }

            // 通知 V 可覆盖 cLocalTmp_
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
        }
    }

    __host_aicore__ static Params InitParams(Params const& args) { return args; }

    __aicore__ ~MulEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

private:
    static constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
};

#endif // EPILOGUE_EPILOGUE_FUSION_MEMBASE_H
