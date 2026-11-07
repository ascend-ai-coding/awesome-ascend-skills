/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file x_all_gather_matmul_base.h
 * \brief Migrated from ops-transformer/mc2/x_all_gather_matmul/op_kernel/x_all_gather_matmul_base.h.
 *        mc2_nd_to_nz.h / mc2_matmul_compute.h include paths adjusted to the vendored
 *        mc2_common/op_kernel copies; the operator's own tiling header renamed to the all_gather_ prefix.
 */
#ifndef ALL_GATHER_MATMUL_BASE_H
#define ALL_GATHER_MATMUL_BASE_H

#include "lib/matmul_intf.h"
#include "mc2_nd_to_nz.h"
#include "mc2_matmul_compute.h"
#include "x_all_gather_matmul_tiling.h"

namespace AscendC {
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;  // 只计算不通信
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class XAllGatherMatmulBase {
public:
    __aicore__ inline XAllGatherMatmulBase() {}
    __aicore__ inline void InitBase(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherGM,
        GM_ADDR workspaceGM, GM_ADDR contextGM, Mc2Tiling::XAllGatherMatmulTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Nd2NzBiasCast();
    __aicore__ inline void MatmulLocalCompute(GM_ADDR aGM, GM_ADDR cGM);
    __aicore__ inline void MatmulLocalComputeL2Cache(GM_ADDR aGM, GM_ADDR cGM);
    __aicore__ inline void MatmulKernelLocal();

protected:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR cGM_;
    GM_ADDR gatherGM_;
    GM_ADDR workspaceGM_;
    Mc2Tiling::XAllGatherMatmulTilingData *tilingData_;
    TPipe *tPipe_;
    uint32_t rankId_{ 0 };
    uint32_t rankDim_{ 8 };
    bool debugOnlyCalc_{ false };
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::InitBase(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    Mc2Tiling::XAllGatherMatmulTilingData *tilingData, TPipe *tPipe)
{
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    cGM_ = cGM;
    gatherGM_ = gatherGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    if (tilingData_->param.gatherLen != 0 || !gatherGM) {
        gatherGM_ = workspaceGM_;
        workspaceGM_ += tilingData_->param.gatherLen;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Nd2NzBiasCast()
{
    TBuf<TPosition::VECCALC> totalUbBuf;
    tPipe_->InitBuffer(totalUbBuf, TOTAL_UB_SIZE);

    // workspace 空间划分:
    GM_ADDR gmNd2NzAddr = workspaceGM_;                                         // step1, ND2NZ 后的地址
    GM_ADDR cToFloat = gmNd2NzAddr + (uint64_t)tilingData_->param.nd2NzWorkLen; // step2, C矩阵float
    GM_ADDR biasToFloat =
        cToFloat + (uint64_t)tilingData_->param.cToFloatLen; // step3, bias矩阵float (当前cToFloatLen都是0)
    if constexpr (Bias2Float) {
        CastBFtoFloat(biasToFloat, biasGM_, tilingData_->param.rankN, totalUbBuf);
        biasGM_ = biasToFloat;
    }
    if constexpr (BNd2Nz) {
        MatrixBtoNZ<typename B_TYPE::T>(gmNd2NzAddr, bGM_, tilingData_->param, totalUbBuf);
        bGM_ = gmNd2NzAddr;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulLocalCompute(
    GM_ADDR aGM, GM_ADDR cGM)
{
    auto &&tiling = tilingData_->localTiling;
    auto &&cfg = tilingData_->param;

    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, SplitType::DEFAULT> mm;
    // Matmul的一次计算流程
    mm.Init(cfg, tiling, tilingData_->localL2Tiling, rankId_);

    // 本卡位置与RankId一致，C矩阵本卡矩阵首地址rankID*cSize
    cGM += (uint64_t)rankId_ * (uint64_t)cfg.rankM * (uint64_t)tiling.N * sizeof(typename C_TYPE::T);
    mm.InitGlobalTensor(bGM_, biasGM_);
    mm.InitGlobalTensor(aGM, (uint64_t)tiling.M * (uint64_t)tiling.Ka, cGM, (uint64_t)tiling.M * (uint64_t)tiling.N);
    mm.Compute(0, GetBlockIdx());
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
XAllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulLocalComputeL2Cache(GM_ADDR aGM,
    GM_ADDR cGM)
{
    auto &&tiling = tilingData_->localTiling;
    auto &&cfg = tilingData_->param;

    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, SplitType::L2CACHE> mm;
    // Matmul的一次计算流程
    mm.Init(cfg, tiling, tilingData_->localL2Tiling, rankId_);

    // 本卡位置与RankId一致，C矩阵本卡矩阵首地址rankID*cSize
    cGM += (uint64_t)rankId_ * (uint64_t)cfg.rankM * (uint64_t)tiling.N * sizeof(typename C_TYPE::T);
    mm.InitGlobalTensor(bGM_, biasGM_);
    mm.InitGlobalTensor(aGM, (uint64_t)tiling.M * (uint64_t)tiling.Ka, cGM, (uint64_t)tiling.M * (uint64_t)tiling.N);
    mm.ComputeWithL2Cache();
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelLocal()
{
    auto &&tiling = tilingData_->localTiling;
    auto &&l2Tiling = tilingData_->localL2Tiling;
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }

    auto &&cfg = tilingData_->param;
    auto aLocalGM = aGM_;
    // Matmul的一次计算流程
    if (l2Tiling.enableL2Tile > 0) {
        MatmulLocalComputeL2Cache(aLocalGM, cGM_);
    } else {
        MatmulLocalCompute(aLocalGM, cGM_);
    }
}
}
#endif
