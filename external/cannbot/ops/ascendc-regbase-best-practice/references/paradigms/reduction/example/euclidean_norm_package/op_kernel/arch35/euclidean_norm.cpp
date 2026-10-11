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
 * \file euclidean_norm.cpp
 * \brief EuclideanNorm 算子 kernel 入口分发（APT entry）。
 */
#include "euclidean_norm_base.h"
#include "euclidean_norm_group.h"
#include "euclidean_norm_empty.h"

// Kernel 入口：按 TilingKey 双 bool 三路分发到 Empty/Group/Base 模板；
// dtype 由 DTYPE_X 编译期实例化（4 dtype × 3 组合 = 12 份 binary），不进 TilingKey。
template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void euclidean_norm(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        // 空 tensor 快路径（EMPTY_A 零操作 / EMPTY_R 固化值填充）
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormEmptyTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormEmptyKernel<DTYPE_X> op;
        op.Init(y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (isGroup) {
        // Group 2D 分核（Phase1 写 workspace → SyncAll → Phase2 二次归约）
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormGroupKernel<DTYPE_X> op;
        op.InitGroup(x, y, workspace, &tilingData, &pipe);
        op.ProcessGroup();
    } else {
        // Base 主路径（A 多核均分 + R 二分缓存树）
        GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);
        NsEuclideanNorm::EuclideanNormBaseKernel<DTYPE_X> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
