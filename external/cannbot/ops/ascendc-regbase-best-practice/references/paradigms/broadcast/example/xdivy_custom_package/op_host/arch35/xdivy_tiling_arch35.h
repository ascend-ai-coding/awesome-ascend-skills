/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy Tiling 头文件 — arch35 (Ascend950)
// 声明广播 tiling 流水线的 4 个分析函数、平台缓存结构和编排类 XdivyTiling

#ifndef XDIVY_TILING_ARCH35_H_
#define XDIVY_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>
// 共享 TilingData 结构 (XdivyTilingData<R>, SplitResult, MultiCoreResult)
#include "../../op_kernel/arch35/xdivy_tiling_struct.h"

namespace Xdivy {

// CheckBroadcastShape: 广播兼容校验 — 同一维上非 1 的大小必须一致，
// 不一致时 OP_LOGE_FOR_INVALID_VALUES_WITH_REASON 报错并返回 false；
// 必须在 BroadcastMergeAxis 之前调用（合轴的乘积判等依赖此前提）
bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                         const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank);

// PadAndSqueeze: shape 归一化 — 低 rank 前补 1 对齐，删除全 1 哑维，
// 输出最大广播 shape 和压缩后的各输入/输出 shape
bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                   const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                   std::vector<std::vector<int64_t>>& normalInputShapes,
                   std::vector<std::vector<int64_t>>& normalOutputShapes);

// BroadcastMergeAxis: 广播合轴 — 贪心合并相邻维，从内向外分组合并，组内每个张量的维度乘积
// 为 1（全广播）或 maxProd（全稠密）才可合并，混合乘积说明组内有广播边界则封组。
// 降低 rank、拉长连续内层段（更大 NDDMA 突发、更少 kernel 侧拷贝），原地修改各 shape
void BroadcastMergeAxis(std::vector<int64_t>& maximumBroShape, std::vector<std::vector<int64_t>>& normalInputShapes,
                        std::vector<std::vector<int64_t>>& normalOutputShapes);

// FindSplitAxis: 从最内维向外找第一个超出 UB 预算的维作为切分轴，
// 输出主块元素数 ubFactor / 块数 ubOuter / 尾块 ubTail
void FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t dtypeSize, int64_t ubPerCore, int64_t physNodes,
                   int64_t ubBlockSize, SplitResult& out);

// MultiCoreSplit: 把 tile 总数均分到多核 — 前 mainCoreNum 个核各处理
// mainTiles 块，其余核各处理 mainTiles-1 块（母模板核间均衡切分公式）
bool MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                    MultiCoreResult& out);

// PrecomputeStrides: 预计算每维 stride（广播维 stride=0），供 kernel 寻址
bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides);

} // namespace Xdivy

namespace optiling {

// 空 CompileInfo：本算子无跨次编译缓存信息（平台参数在 Tiling 阶段直接从系统获取）
struct XdivyCompileInfo {};

// XdivyTiling: tiling 编排类
// RunTiling: GetShapeInfo 读 shape/dtype → 按 rank 映射 R=4/8 → DoTilingAndSet<R> 填充 TilingData
class XdivyTiling {
public:
    explicit XdivyTiling(gert::TilingContext* ctx);
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
    int64_t physNodes_ = 3; // P: FP32=3, FP16/BF16=4
    int64_t rank_ = 0;
};

} // namespace optiling

#endif // XDIVY_TILING_ARCH35_H_
