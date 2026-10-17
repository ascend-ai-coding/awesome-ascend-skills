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
 * \file x_all_gather_matmul_full_mesh.h
 * \brief Migrated verbatim from ops-transformer/mc2/x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h
 *        (arch22, full-mesh AllGather + AscendC::Matmul via the MatmulCompute wrapper). Only the local
 *        base-header include is renamed to the all_gather_-prefixed file.
 */
#ifndef ALL_GATHER_MATMUL_FULL_MESH_H
#define ALL_GATHER_MATMUL_FULL_MESH_H

#include "x_all_gather_matmul_base.h"
#include "adv_api/hccl/hccl.h"

namespace AscendC {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class XAllGatherMatmulFullMesh : public XAllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float> {
public:
    __aicore__ inline XAllGatherMatmulFullMesh() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherGM,
        GM_ADDR workspaceGM, GM_ADDR contextGM, Mc2Tiling::XAllGatherMatmulTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcclPrepare();
    __aicore__ inline void InnerProcess();

    __aicore__ inline void MatmulKernelCompute(GM_ADDR aGM, GM_ADDR cGM, TCubeTiling &tiling,
        Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt);
    __aicore__ inline void MatmulKernelComputeL2Cache(GM_ADDR aGM, GM_ADDR cGM, TCubeTiling &tiling,
        Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt);
    __aicore__ inline void MatmulKernelGather(GM_ADDR aGM, GM_ADDR cGM, TCubeTiling &tiling,
        Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt);

    __aicore__ inline void HcclFinalize();

private:
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    HcclHandle handleId_{ INVALID_HANDLE_ID };
    HcclHandle tailHandleId_{ INVALID_HANDLE_ID };
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Init(GM_ADDR aGM,
    GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    Mc2Tiling::XAllGatherMatmulTilingData *tilingData, TPipe *tPipe)
{
    this->InitBase(aGM, bGM, biasGM, cGM, gatherGM, workspaceGM, contextGM, tilingData, tPipe);
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::XAllGatherMatmulTilingData, mc2CcTiling));
    this->rankId_ = hccl_.GetRankId();
    this->rankDim_ = hccl_.GetRankDim();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Process()
{
    HcclPrepare();
    this->Nd2NzBiasCast();
    InnerProcess();
    HcclFinalize();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::HcclPrepare()
{
    if ASCEND_IS_AIC {
        if (this->debugOnlyCalc_) {
            return;
        }
        auto &&cfg = this->tilingData_->param;
        auto &&tileTiling = this->tilingData_->tileTiling;
        auto &&tailTiling = this->tilingData_->tailTiling;

        const uint64_t aTileCnt = (uint64_t)tileTiling.M * (uint64_t)tileTiling.Ka;
        const uint64_t aTileOffset =
            (uint64_t)cfg.tileCnt * (uint64_t)tileTiling.M * (uint64_t)tileTiling.Ka * sizeof(typename A_TYPE::T);
        const uint64_t aTailCnt = (uint64_t)tailTiling.M * (uint64_t)tailTiling.Ka;
        const uint64_t aRankCnt = (uint64_t)cfg.rankM * (uint64_t)cfg.rankK;

        handleId_ = hccl_.AllGather<true>(this->aGM_, this->gatherGM_, aTileCnt, HcclDataType(cfg.dataType), aRankCnt,
            cfg.tileCnt);
        if (cfg.tailCnt > 0) {
            tailHandleId_ = hccl_.AllGather<true>(this->aGM_ + aTileOffset, this->gatherGM_ + aTileOffset, aTailCnt,
                HcclDataType(cfg.dataType), aRankCnt, cfg.tailCnt);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::InnerProcess()
{
    if ASCEND_IS_AIC {
        auto &&cfg = this->tilingData_->param;
        // 计算本卡数据
        this->MatmulKernelLocal();

        // 获取远端数据计算
        MatmulKernelGather(this->gatherGM_, this->cGM_, this->tilingData_->tileTiling, this->tilingData_->tileL2Tiling,
            handleId_, cfg.tileCnt);

        // 存在尾块
        if (cfg.tailCnt > 0) {
            auto &&tiling = this->tilingData_->tileTiling;
            auto tailCGM = this->cGM_ +
                (uint64_t)cfg.tileCnt * (uint64_t)tiling.M * (uint64_t)tiling.N * sizeof(typename C_TYPE::T);
            auto tailGatherGM = this->gatherGM_ +
                (uint64_t)cfg.tileCnt * (uint64_t)tiling.M * (uint64_t)tiling.Ka * sizeof(typename A_TYPE::T);
            MatmulKernelGather(tailGatherGM, tailCGM, this->tilingData_->tailTiling, this->tilingData_->tailL2Tiling,
                tailHandleId_, cfg.tailCnt);
            // 搬运本地数据放到GM输出操作, 当前由aicpu完成
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelCompute(GM_ADDR aGM,
    GM_ADDR cGM, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt)
{
    using A_T = typename A_TYPE::T;
    using C_T = typename C_TYPE::T;
    auto &&cfg = this->tilingData_->param;
    const uint64_t cOffset = (uint64_t)tiling.M * (uint64_t)tiling.N * sizeof(C_T);
    const uint64_t aOffset = (uint64_t)tiling.M * (uint64_t)tiling.Ka * sizeof(A_T);
    const uint64_t raOffset = (uint64_t)cfg.rankM * (uint64_t)cfg.rankK * sizeof(A_T);
    const uint64_t rcOffset = (uint64_t)cfg.rankM * (uint64_t)cfg.rankN * sizeof(C_T);
    // 归一化Matmul计算类，负责MC2的Matmul计算
    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, SplitType::DEFAULT> mm;
    mm.InitGlobalTensor(this->bGM_, this->biasGM_);
    mm.Init(cfg, tiling, l2Tiling, this->rankId_);
    // 按照切分的片数进行逐片处理
    for (uint32_t i = 0; i < tileCnt; i++) {
        if (!this->debugOnlyCalc_) {
            hccl_.Wait(handleId);
        }

        // 现在需要计算其他卡数据
        for (uint32_t j = 0; j < this->rankDim_; j++) {                       // 第0片不需要去: 8 - 1= 7
            auto rank = (this->rankId_ + j + GetBlockIdx()) % this->rankDim_; // 从自己的rankID开始计算
            if (rank == this->rankId_) {
                continue;
            }
            // 为需要计算的所有矩阵编号，保证多核计算不受计算先后顺序影响
            // 默认从本卡下一张卡矩阵开始计算，按照该顺序为矩阵从0编号
            // 公式：index=(x + y)%rankDim_=0, x = rankId_ + 1, y = rankDim_ - (rankId_ + 1)
            uint32_t index = (rank + this->rankDim_ - (this->rankId_ + 1)) % this->rankDim_ + i * (this->rankDim_ - 1);
            // 计算A和C矩阵的首地址并初始化Buffer
            auto aCalcAddr = aGM + (uint64_t)rank * raOffset;
            auto cCalcAddr = cGM + (uint64_t)rank * rcOffset;
            mm.InitGlobalTensor(aCalcAddr, aOffset, cCalcAddr, cOffset);
            // Matmul计算主流程
            mm.Compute(index);
        }
        aGM += aOffset;
        cGM += cOffset;
    }
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelComputeL2Cache(GM_ADDR aGM,
    GM_ADDR cGM, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt)
{
    using A_T = typename A_TYPE::T;
    using C_T = typename C_TYPE::T;
    auto &&cfg = this->tilingData_->param;
    const uint64_t cOffset = (uint64_t)tiling.M * (uint64_t)tiling.N * sizeof(C_T);
    const uint64_t aOffset = (uint64_t)tiling.M * (uint64_t)tiling.Ka * sizeof(A_T);
    const uint64_t raOffset = (uint64_t)this->rankDim_ * (uint64_t)cfg.rankM * (uint64_t)cfg.rankK * sizeof(A_T);
    const uint64_t rcOffset = (uint64_t)this->rankDim_ * (uint64_t)cfg.rankM * (uint64_t)cfg.rankN * sizeof(C_T);
    // 归一化Matmul计算类，负责MC2的Matmul计算
    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, SplitType::L2CACHE> mm;
    mm.InitGlobalTensor(this->bGM_, this->biasGM_);
    mm.Init(cfg, tiling, l2Tiling, this->rankId_);
    // 按照切分的片数进行逐片处理
    for (uint32_t i = 0; i < tileCnt; i++) {
        if (!this->debugOnlyCalc_) {
            hccl_.Wait(handleId);
        }

        // Matmul计算主流程
        mm.InitGlobalTensor(aGM, raOffset, cGM, rcOffset);
        mm.ComputeWithL2Cache(i);

        aGM += aOffset;
        cGM += cOffset;
    }
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelGather(GM_ADDR aGM,
    GM_ADDR cGM, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt)
{
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }

    if (l2Tiling.enableL2Tile > 0) {
        MatmulKernelComputeL2Cache(aGM, cGM, tiling, l2Tiling, handleId, tileCnt);
    } else {
        MatmulKernelCompute(aGM, cGM, tiling, l2Tiling, handleId, tileCnt);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void XAllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::HcclFinalize()
{
    if ASCEND_IS_AIC {
        if (this->debugOnlyCalc_) {
            return;
        }
        CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);
        CrossCoreWaitFlag(EVENT_ID_6);
        hccl_.Finalize();
    }
}
}
#endif
