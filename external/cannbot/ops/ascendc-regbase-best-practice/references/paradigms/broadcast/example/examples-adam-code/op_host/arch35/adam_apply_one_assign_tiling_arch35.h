/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AdamApplyOneAssign Tiling — arch35 头文件
// 位置：math/adam_apply_one_assign/op_host/arch35/adam_apply_one_assign_arch35.h
#ifndef ADAM_APPLY_ONE_ASSIGN_TILING_ARCH35_H_
#define ADAM_APPLY_ONE_ASSIGN_TILING_ARCH35_H_
#include <cstdint>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include "math/adam_apply_one_assign/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h"

// ============================================================
// Tiling 函数模块（无状态，与 CANN 框架零耦合）
// 作用域：namespace AdamApplyOneAssign
// ============================================================

namespace AdamApplyOneAssign {

bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                         const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank);

void PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                   const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                   std::vector<std::vector<int64_t>>& normalInputShapes,
                   std::vector<std::vector<int64_t>>& normalOutputShapes);

// BroadcastMergeAxis — 广播合轴: PadAndSqueeze + CheckBroadcastShape 之后调用。
// 从最后一维向前贪心合并相邻维: 组内每个张量(入参+出参)的维度乘积必须为
// 1(整组广播) 或等于坐标系 maximumBroShape 的组乘积(整组稠密);
// 组内部分维广播的混合模式不能合并。失败时封闭当前组、从当前维开新组。
// 合轴后 rank 变小, 调用方需重新读取 maximumBroShape.size()。
void BroadcastMergeAxis(std::vector<int64_t>& maximumBroShape, std::vector<std::vector<int64_t>>& normalInputShapes,
                        std::vector<std::vector<int64_t>>& normalOutputShapes);

void FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t dtypeSize, int64_t ubPerCore, int64_t physNodes,
                   int64_t ubBlockSize, SplitResult& out);

void MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                    MultiCoreResult& out);

void PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides);

} // namespace AdamApplyOneAssign

// ============================================================
// AdamTiling — CANN 主线：单次归一化 → 映射 rank → 模板填充
// ============================================================
namespace optiling {

// 空 CompileInfo：本算子无跨次编译缓存信息（平台参数在 Tiling 阶段直接从系统获取）
struct AdamCompileInfo {};

class AdamTiling {
public:
    explicit AdamTiling(gert::TilingContext* ctx);
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetShapeInfo();
    template <int64_t R>
    ge::graphStatus DoTilingAndSet();

    gert::TilingContext* ctx_;
    std::vector<std::vector<int64_t>> rawInputShapes_;
    std::vector<std::vector<int64_t>> rawOutputShapes_;
    std::vector<int64_t> maxBroShape_;
    std::vector<std::vector<int64_t>> normalInputShapes_;
    std::vector<std::vector<int64_t>> normalOutputShapes_;
    int64_t dtypeSize_ = 0;
    int64_t rank_ = 0;
};

} // namespace optiling
#endif // ADAM_APPLY_ONE_ASSIGN_TILING_ARCH35_H_
