/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy Tiling — arch35 (Ascend950) 广播 tiling 实现

#include "xdivy_tiling_arch35.h"
#include "../../op_kernel/arch35/xdivy_struct.h"
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

// ============================================================
// Tiling 函数模块 — namespace Xdivy (pure host-side analysis)
// ============================================================

namespace Xdivy {

// 物理存活 UB 节点数 P（FP32 路径 3 个 TBuf，FP16/BF16 Cast 路径 4 个）
constexpr int64_t PHYS_NODES_FP32 = 3;
constexpr int64_t PHYS_NODES_FP16 = 4;
// NDDMA 预算统一按 4 字节 dtype 计算（FP16 也按 4B 对齐预留）
constexpr int64_t BUF_DTYPE_SIZE = 4;

// CheckBroadcastShape: 广播兼容校验 — 逐维检查所有输入/输出，
// 同维取值非 1 时必须相等，否则报错并返回 false（须在 BroadcastMergeAxis 之前调用）；
// 同时校验输出稠密性 — 输出不允许为广播维
bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                         const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank)
{
    for (int64_t d = 0; d < maxRank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < paddedIn.size(); i++) {
            if (paddedIn[i][d] != 1) {
                if (ref == -1) {
                    ref = paddedIn[i][d];
                } else if (paddedIn[i][d] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "CheckBroadcastShape", "input size and ref size",
                        (std::string("dim ") + std::to_string(d) + " input[" + std::to_string(i) + "] size " +
                         std::to_string(paddedIn[i][d]) + " and " + std::to_string(ref))
                            .c_str(),
                        "broadcast incompatible: input sizes must be equal or 1");
                    return false;
                }
            }
        }
        for (size_t i = 0; i < paddedOut.size(); i++) {
            if (paddedOut[i][d] != 1) {
                if (ref == -1) {
                    ref = paddedOut[i][d];
                } else if (paddedOut[i][d] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) + "] size " +
                         std::to_string(paddedOut[i][d]) + " and " + std::to_string(ref))
                            .c_str(),
                        "broadcast incompatible: output sizes must be equal or 1");
                    return false;
                }
            }
        }
        // 输出稠密性校验: 该维存在非 1 维值 (ref != -1) 时, 输出不允许为广播维 (1)。
        // 广播输出维 stride=0, 多核 tile 映射到同一 GM 地址并发写, 构成写-写数据竞争,
        // 且输出 y 应为 broadcastMax(x1,x2) 的稠密结果 (InferShapeForXdivy 语义),
        // 广播输出无定义, Host 侧直接拒绝 (正常流程由图阶段保证, 此处兜底防御)。
        if (ref != -1) {
            for (size_t i = 0; i < paddedOut.size(); i++) {
                if (paddedOut[i][d] == 1) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) + "] size 1 and " +
                         std::to_string(ref))
                            .c_str(),
                        "broadcast output is not supported: output must be dense (equal to broadcast max dim)");
                    return false;
                }
            }
        }
    }
    return true;
}

// PadShape: 单个 shape 前补 1 对齐到 maxRank（只能在 shape 最前面补 1）。
// 依赖全部显式参数化（无捕获），替代原先的 lambda pad。
static std::vector<int64_t> PadShape(const std::vector<int64_t>& s, int64_t maxRank)
{
    std::vector<int64_t> p;
    p.assign(static_cast<size_t>(maxRank - static_cast<int64_t>(s.size())), 1);
    p.insert(p.end(), s.begin(), s.end());
    return p;
}

// HasNonPositiveDim: 任一维度 <= 0（空 shape / 非法值）时返回 true。
// 0 维会以 maxDim=0 进入 maximumBroShape, 使 totalTiles=0 → usedCoreNum=0,
// 导致 CeilDiv 除零与 SetBlockDim(0) 退化 tiling, 须在 Host 侧拦截。
static bool HasNonPositiveDim(const std::vector<std::vector<int64_t>>& shapes)
{
    for (const auto& s : shapes) {
        for (int64_t d : s) {
            if (d <= 0) {
                return true;
            }
        }
    }
    return false;
}

// PadAndSqueeze: shape 归一化 — 低 rank 前补 1 对齐，删除全 1 哑维，
// 输出最大广播 shape 和压缩后的各输入/输出 shape
bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                   const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                   std::vector<std::vector<int64_t>>& normalInputShapes,
                   std::vector<std::vector<int64_t>>& normalOutputShapes)
{
    int64_t numInputs = static_cast<int64_t>(inputShapes.size());
    int64_t numOutputs = static_cast<int64_t>(outputShapes.size());
    int64_t maxRank = 0;
    for (auto& s : inputShapes) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }
    for (auto& s : outputShapes) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }

    std::vector<std::vector<int64_t>> paddedIn(numInputs);
    std::vector<std::vector<int64_t>> paddedOut(numOutputs);
    for (int64_t i = 0; i < numInputs; i++) {
        paddedIn[i] = PadShape(inputShapes[i], maxRank);
    }
    for (int64_t i = 0; i < numOutputs; i++) {
        paddedOut[i] = PadShape(outputShapes[i], maxRank);
    }

    maximumBroShape.clear();
    normalInputShapes.assign(numInputs, std::vector<int64_t>());
    normalOutputShapes.assign(numOutputs, std::vector<int64_t>());
    for (int64_t d = 0; d < maxRank; d++) {
        bool allOne = true;
        int64_t maxDim = 0;
        for (int64_t i = 0; i < numInputs; i++) {
            if (paddedIn[i][d] != 1) {
                allOne = false;
            }
            maxDim = std::max(maxDim, paddedIn[i][d]);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            if (paddedOut[i][d] != 1) {
                allOne = false;
            }
            maxDim = std::max(maxDim, paddedOut[i][d]);
        }
        if (!allOne) {
            maximumBroShape.push_back(maxDim);
            for (int64_t i = 0; i < numInputs; i++) {
                normalInputShapes[i].push_back(paddedIn[i][d]);
            }
            for (int64_t i = 0; i < numOutputs; i++) {
                normalOutputShapes[i].push_back(paddedOut[i][d]);
            }
        }
    }

    if (maximumBroShape.empty()) {
        maximumBroShape.push_back(1);
        for (int64_t i = 0; i < numInputs; i++) {
            normalInputShapes[i].push_back(1);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            normalOutputShapes[i].push_back(1);
        }
    }
    return true;
}

// MergeShapeByGroups: 按分组闭区间列表把 shape 每组收缩为一维（组内乘积）。
// 依赖全部显式参数化（无捕获），替代原先的 lambda mergeShape。
static std::vector<int64_t> MergeShapeByGroups(const std::vector<int64_t>& s,
                                               const std::vector<std::vector<int64_t>>& groups)
{
    std::vector<int64_t> merged;
    merged.reserve(groups.size());
    for (auto& g : groups) {
        int64_t prod = 1;
        for (int64_t k = g[0]; k <= g[1]; k++) {
            prod *= s[k];
        }
        merged.push_back(prod);
    }
    return merged;
}

// BroadcastMergeAxis — 广播合轴: PadAndSqueeze 之后、FindSplitAxis 之前调用。
// 前置: CheckBroadcastShape 已通过(每维取值 ∈ {1, 该维最大值},
// 正常流程由 InferShapeForXdivy 在图阶段保证, tiling 侧兜底校验),
// 否则乘积判等可能误合并。
// 从最后一维向前贪心扩展: 候选组 [d..groupHighDim] 内每个张量的维度乘积必须为 1(整组广播,
// 合并后该维为 1、stride=0) 或坐标系组乘积(整组稠密连续); 组内混有 1 和非 1
// 说明广播边界落在组内部, 不能合并, 封闭当前组、从 dim d 开新组。

void BroadcastMergeAxis(std::vector<int64_t>& maximumBroShape, std::vector<std::vector<int64_t>>& normalInputShapes,
                        std::vector<std::vector<int64_t>>& normalOutputShapes)
{
    int64_t numInputs = static_cast<int64_t>(normalInputShapes.size());
    int64_t numOutputs = static_cast<int64_t>(normalOutputShapes.size());
    int64_t oldRank = static_cast<int64_t>(maximumBroShape.size());
    if (oldRank <= 1) {
        return; // 单维无可合并
    }

    // 从最后一维向前贪心分组; groups[i] = {groupLowDim, groupHighDim} 闭区间, 按从右到左的发现顺序收集
    std::vector<std::vector<int64_t>> groups;
    int64_t groupLowDim = oldRank - 1;
    int64_t groupHighDim = oldRank - 1;
    for (int64_t d = oldRank - 2; d >= 0; d--) {
        // 坐标系在 [d..groupHighDim] 上的组乘积(合并后该组的维度大小)
        int64_t maxProd = 1;
        for (int64_t k = d; k <= groupHighDim; k++) {
            maxProd *= maximumBroShape[k];
        }
        // 每个张量在 [d..groupHighDim] 上的乘积必须为 1(整组广播) 或 maxProd(整组稠密)
        bool mergeable = true;
        for (int64_t i = 0; i < numInputs && mergeable; i++) {
            int64_t prod = 1;
            for (int64_t k = d; k <= groupHighDim; k++) {
                prod *= normalInputShapes[i][k];
            }
            if (prod != 1 && prod != maxProd) {
                mergeable = false;
            }
        }
        for (int64_t i = 0; i < numOutputs && mergeable; i++) {
            int64_t prod = 1;
            for (int64_t k = d; k <= groupHighDim; k++) {
                prod *= normalOutputShapes[i][k];
            }
            if (prod != 1 && prod != maxProd) {
                mergeable = false;
            }
        }
        if (mergeable) {
            groupLowDim = d; // dim d 并入当前组
        } else {
            groups.push_back({groupLowDim, groupHighDim}); // 封闭当前组, 从 dim d 开新组
            groupLowDim = d;
            groupHighDim = d;
        }
    }
    groups.push_back({groupLowDim, groupHighDim});
    // groups 按从右到左发现, 反转为从左到右
    std::reverse(groups.begin(), groups.end());

    // 每组收缩为一维: 坐标系取组内乘积, 张量取自身组内乘积(整组广播自然收缩为 1)
    maximumBroShape = MergeShapeByGroups(maximumBroShape, groups);
    for (int64_t i = 0; i < numInputs; i++) {
        normalInputShapes[i] = MergeShapeByGroups(normalInputShapes[i], groups);
    }
    for (int64_t i = 0; i < numOutputs; i++) {
        normalOutputShapes[i] = MergeShapeByGroups(normalOutputShapes[i], groups);
    }
}

// FindSplitAxis: 从最内维向外找第一个超出 UB 预算的维作为切分轴，
// 输出主块元素数 ubFactor / 块数 ubOuter / 尾块 ubTail
void FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t dtypeSize, int64_t ubPerCore, int64_t physNodes,
                   int64_t ubBlockSize, SplitResult& out)
{
    int64_t perBufBytes = (ubPerCore / physNodes) & ~(ubBlockSize - 1);
    int64_t perBufElems = perBufBytes / dtypeSize;
    int64_t rank = static_cast<int64_t>(maxBroShape.size());
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (maxBroShape[k] * inner > perBufElems) {
            out.ubFactor = perBufElems / inner;
            out.ubOuter = Ops::Base::CeilDiv(maxBroShape[k], out.ubFactor);
            int64_t rem = maxBroShape[k] % out.ubFactor;
            out.ubTail = (rem == 0) ? out.ubFactor : rem;
            out.ubSplitIdx = k;
            return;
        }
        if (k == 0) {
            out.ubSplitIdx = 0;
            out.ubFactor = maxBroShape[0];
            out.ubOuter = 1;
            out.ubTail = maxBroShape[0];
            return;
        }
        inner *= maxBroShape[k];
    }
}

// MultiCoreSplit: 把 tile 总数均分到多核 — 前 mainCoreNum 个核各处理
// mainTiles 块，其余核各处理 mainTiles-1 块（母模板核间均衡切分公式）
bool MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                    MultiCoreResult& out)
{
    int64_t k = ubSplit.ubSplitIdx;
    int64_t outerProd = 1;
    for (int64_t j = 0; j < k; j++) {
        outerProd *= maxBroShape[j];
    }
    out.totalTiles = outerProd * ubSplit.ubOuter;

    out.usedCoreNum = (out.totalTiles < maxCores) ? out.totalTiles : maxCores;
    out.mainTiles = Ops::Base::CeilDiv(out.totalTiles, out.usedCoreNum);
    out.mainCoreNum = out.totalTiles - (out.mainTiles - 1) * out.usedCoreNum;
    return true;
}

// PrecomputeStrides: 预计算每维 stride（广播维 stride=0），供 kernel 寻址
bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = static_cast<int64_t>(s.size());
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++) {
            prod *= s[j];
        }
        strides[d] = prod;
    }
    return true;
}

} // namespace Xdivy

// ============================================================
// XdivyTiling — CANN-registered tiling orchestrator (namespace optiling)
// ============================================================

namespace optiling {

using namespace Xdivy;
using Ops::Base::GetUbBlockSize;

// 用 dims 前 dimNum 个元素组装 gert::Shape，供 Ops::Base::ToString 统一打印 "[d0, d1, ...]"。
static gert::Shape MakeShapeFromDims(const int64_t* dims, int32_t dimNum)
{
    gert::Shape shape;
    for (int32_t i = 0; i < dimNum; ++i) {
        shape.AppendDim(dims[i]);
    }
    return shape;
}

XdivyTiling::XdivyTiling(gert::TilingContext* ctx)
    : ctx_(ctx)
{}

// GetShapeInfo: 读取输入/输出 shape 与 dtype，做 PadAndSqueeze 归一化并校验 rank ≤ 8
ge::graphStatus XdivyTiling::GetShapeInfo()
{
    auto* computeNodeInfo = ctx_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, computeNodeInfo);
    for (size_t i = 0; i < computeNodeInfo->GetInputsNum(); ++i) {
        auto* shape = ctx_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        const gert::Shape& s = shape->GetStorageShape();
        const size_t rank = s.GetDimNum();
        std::vector<int64_t> dims;
        for (size_t d = 0; d < rank; ++d) {
            dims.push_back(s.GetDim(d));
        }
        rawInputShapes_.push_back(dims);
    }

    for (size_t i = 0; i < computeNodeInfo->GetOutputsNum(); ++i) {
        auto* shape = ctx_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
        const gert::Shape& s = shape->GetStorageShape();
        const size_t rank = s.GetDimNum();
        std::vector<int64_t> dims;
        for (size_t d = 0; d < rank; ++d) {
            dims.push_back(s.GetDim(d));
        }
        rawOutputShapes_.push_back(dims);
    }

    // 空 shape 防御: 拒绝任一维 <= 0, 避免下游 CeilDiv 除零 / SetBlockDim(0)
    OP_CHECK_IF(HasNonPositiveDim(rawInputShapes_) || HasNonPositiveDim(rawOutputShapes_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input/output", "dim size <= 0",
                                                       "all dim sizes must be positive (>= 1)"),
                return ge::GRAPH_FAILED);

    auto* inputDesc = ctx_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supportedDtypes.count(dtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx_->GetNodeName(), "x1", Ops::Base::ToString(dtype).c_str(),
                                                      "Xdivy only supports DT_FLOAT, DT_FLOAT16 and DT_BF16"),
                return GRAPH_FAILED);
    dtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(dtype));
    physNodes_ = (dtype == ge::DT_FLOAT) ? PHYS_NODES_FP32 : PHYS_NODES_FP16;

    PadAndSqueeze(rawInputShapes_, rawOutputShapes_, maxBroShape_, normalInputShapes_, normalOutputShapes_);
    rank_ = static_cast<int64_t>(maxBroShape_.size());

    if (rank_ > XDIVY_RANK_8) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input", "rank exceeds limit", "rank must be <= 8");
        return ge::GRAPH_FAILED;
    }

    // 广播兼容校验: 正常流程已由 InferShapeForXdivy 在图阶段拦截,
    // 此处兜底防御非法 shape 绕过图校验直接进入 tiling
    OP_CHECK_IF(
        !CheckBroadcastShape(normalInputShapes_, normalOutputShapes_, rank_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input/output", "incompatible",
                                               "check broadcast shape failed, shapes must be broadcast-compatible"),
        return ge::GRAPH_FAILED);

    // 广播合轴: broadcast 校验通过后执行; 后续 stride/split/模板映射均基于合轴后 shape
    BroadcastMergeAxis(maxBroShape_, normalInputShapes_, normalOutputShapes_);
    rank_ = static_cast<int64_t>(maxBroShape_.size());

    OP_LOGI(ctx_->GetNodeName(), "Xdivy GetShapeInfo done rank %ld physNodes %ld", rank_, physNodes_);

    return GRAPH_SUCCESS;
}

// DoTilingAndSet<R>: 核心填充 — 执行 UB 切分/多核切分/stride 预计算，
// 按 R 维模板右移补 1 填充 TilingData，并设置 BlockDim
template <int64_t R>
ge::graphStatus XdivyTiling::DoTilingAndSet()
{
    auto* tiling = ctx_->GetTilingData<XdivyTilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);

    // 平台参数 tiling 时直接从系统获取（不经 CompileInfo 缓存），任一为 0 视为环境异常
    fe::PlatFormInfos* platformInfo = ctx_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ctx_->GetNodeName(), "platform.coreNum", "GetCoreNumAiv returned 0"),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ctx_->GetNodeName(), "platform.ubSize",
                                                         "GetCoreMemSize(UB) returned 0"),
                return ge::GRAPH_FAILED);
    int64_t ubPerCore = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = GetUbBlockSize(ctx_);

    int64_t perBufBytes = (ubPerCore / physNodes_) & ~(ubBlockSize - 1);
    tiling->perBufBytes = perBufBytes;

    FindSplitAxis(maxBroShape_, BUF_DTYPE_SIZE, ubPerCore, physNodes_, ubBlockSize, tiling->split);
    MultiCoreSplit(maxBroShape_, tiling->split, coreNum, tiling->multicore);

    int64_t numIn = static_cast<int64_t>(normalInputShapes_.size());
    int64_t numOut = static_cast<int64_t>(normalOutputShapes_.size());
    std::vector<std::vector<int64_t>> inStrides(numIn);
    std::vector<std::vector<int64_t>> outStrides(numOut);
    for (int64_t i = 0; i < numIn; i++) {
        PrecomputeStrides(normalInputShapes_[i], inStrides[i]);
    }
    for (int64_t i = 0; i < numOut; i++) {
        PrecomputeStrides(normalOutputShapes_[i], outStrides[i]);
    }

    tiling->rank = rank_;
    int64_t delta = R - rank_;

    for (int64_t d = 0; d < delta; d++) {
        tiling->maxBroShape[d] = 1;
    }
    for (int64_t d = 0; d < rank_; d++) {
        tiling->maxBroShape[d + delta] = maxBroShape_[d];
    }

    tiling->split.ubSplitIdx += delta;

    tiling->numInputs = numIn;
    tiling->numOutputs = numOut;

    for (int64_t i = 0; i < numIn; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->inputShapes[i][d] = 1;
            tiling->inputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->inputShapes[i][d + delta] = normalInputShapes_[i][d];
            tiling->inputStrides[i][d + delta] = inStrides[i][d];
        }
    }
    for (int64_t i = numIn; i < MAX_INPUT_SLOTS; i++) {
        for (int64_t d = 0; d < R; d++) {
            tiling->inputShapes[i][d] = 1;
            tiling->inputStrides[i][d] = 0;
        }
    }

    for (int64_t i = 0; i < numOut; i++) {
        for (int64_t d = 0; d < delta; d++) {
            tiling->outputShapes[i][d] = 1;
            tiling->outputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank_; d++) {
            tiling->outputShapes[i][d + delta] = normalOutputShapes_[i][d];
            tiling->outputStrides[i][d + delta] = outStrides[i][d];
        }
    }
    for (int64_t i = numOut; i < MAX_OUTPUT_SLOTS; i++) {
        for (int64_t d = 0; d < R; d++) {
            tiling->outputShapes[i][d] = 1;
            tiling->outputStrides[i][d] = 0;
        }
    }

    ctx_->SetBlockDim(tiling->multicore.usedCoreNum);

    // 维测: TilingData 全部字段（shape 经 Ops::Base::ToString 统一打印，maxBroShape 含 padding 前补 1，R 维全量打印）
    OP_LOGI(ctx_->GetNodeName(),
            "Xdivy TilingData: perBufBytes=%ld rank=%ld->R=%d "
            "maxBroShape=%s "
            "split(ubSplitIdx=%ld ubFactor=%ld ubOuter=%ld ubTail=%ld) "
            "multi(usedCoreNum=%ld totalTiles=%ld mainTiles=%ld mainCoreNum=%ld) numIn=%ld numOut=%ld",
            tiling->perBufBytes, rank_, static_cast<int32_t>(R),
            Ops::Base::ToString(MakeShapeFromDims(tiling->maxBroShape, static_cast<int32_t>(R))).c_str(),
            tiling->split.ubSplitIdx, tiling->split.ubFactor, tiling->split.ubOuter, tiling->split.ubTail,
            tiling->multicore.usedCoreNum, tiling->multicore.totalTiles, tiling->multicore.mainTiles,
            tiling->multicore.mainCoreNum, numIn, numOut);

    // input/output 的 shape/stride 经 Ops::Base::ToString 统一打印 "[d0, d1, ...]"
    for (int64_t i = 0; i < numIn; i++) {
        OP_LOGI(ctx_->GetNodeName(), "Xdivy TilingData input[%ld]: shape=%s stride=%s", i,
                Ops::Base::ToString(MakeShapeFromDims(tiling->inputShapes[i], static_cast<int32_t>(R))).c_str(),
                Ops::Base::ToString(MakeShapeFromDims(tiling->inputStrides[i], static_cast<int32_t>(R))).c_str());
    }
    for (int64_t i = 0; i < numOut; i++) {
        OP_LOGI(ctx_->GetNodeName(), "Xdivy TilingData output[%ld]: shape=%s stride=%s", i,
                Ops::Base::ToString(MakeShapeFromDims(tiling->outputShapes[i], static_cast<int32_t>(R))).c_str(),
                Ops::Base::ToString(MakeShapeFromDims(tiling->outputStrides[i], static_cast<int32_t>(R))).c_str());
    }

    return GRAPH_SUCCESS;
}

// RunTiling: 编排入口 — GetShapeInfo 后按 rank 映射 R=4/8 并设置 TilingKey
ge::graphStatus XdivyTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    int64_t mapped = (rank_ <= XDIVY_RANK_4) ? XDIVY_RANK_4 : XDIVY_RANK_8;
    if (mapped == XDIVY_RANK_4) {
        ret = DoTilingAndSet<XDIVY_RANK_4>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_4));
    } else {
        ret = DoTilingAndSet<XDIVY_RANK_8>();
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_8));
    }
    return ret;
}

// TilingFuncXdivy: 框架注册的 tiling 入口 — 跑 RunTiling（本算子不使用 workspace, 置 0）
static ge::graphStatus TilingFuncXdivy(gert::TilingContext* context)
{
    XdivyTiling xdivyTiling(context);
    auto ret = xdivyTiling.RunTiling();
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    return GRAPH_SUCCESS;
}

// TilingParse 空实现：XdivyCompileInfo 为空结构（本算子无跨次编译缓存信息），
static ge::graphStatus TilingPrepareForXdivy([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Xdivy).Tiling(TilingFuncXdivy).TilingParse<XdivyCompileInfo>(TilingPrepareForXdivy);

} // namespace optiling
