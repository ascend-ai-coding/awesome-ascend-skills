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
 * \file euclidean_norm_tiling_arch35.cpp
 * \brief EuclideanNorm 算子 host tiling 实现（新范式）。
 */

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <cstring>

#include "euclidean_norm_tiling_arch35.h"

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling_base_util.h"
#include "../../op_kernel/arch35/euclidean_norm_tiling_data.h"
#include "../../op_kernel/arch35/euclidean_norm_tiling_key.h"

namespace optiling {

// 输入索引：x = 0，axes = 1（与 TilingInputsDataDependency({1}) 的值依赖声明对应）
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_AXES_IDX = 1;

// A/R 规整后轴数合法区间：上限取通用类别 MAX_PATTERN_RANK = 9。
static constexpr int32_t MIN_AXIS_NUM = 2;
static constexpr int32_t MAX_AXIS_NUM = MAX_PATTERN_RANK;

// cacheBuf（kernel 二分树缓存）固定 16KB 预算；FP32 单元素字节数（中间精度换算）。
static constexpr int64_t CACHE_BUF_BYTES = 16 * 1024;
static constexpr int64_t FP32_BYTES = 4;

// UB 预算系数：pre 侧共 3 个 buffer（preIn + preRes + preResTail = P_PRE + P_PRE_EXT），
// post 侧 1 个（outBuf = P_POST）。
static constexpr int64_t P_PRE = 2;
static constexpr int64_t P_PRE_EXT = 1;
static constexpr int64_t P_POST = 1;

// A/R 轴模式化后偶位 A、奇位 R，相邻同类型轴（同为 A 或同为 R）的固定间距。
static constexpr int32_t AXIS_INTERVAL = 2;
// Group 触发阈值分母：A 迭代总数 ≤ 核数/2（A 并行度不足半数核）且 R 迭代 >1 时走 Group 2D 分核。
static constexpr int64_t GROUP_CORE_RATIO = 2;

// 空 tensor 三分类：NORMAL（正常）/ EMPTY_A（非归约维含 0 → 输出 0 元素）/
// EMPTY_R（仅归约维含 0 → 每输出 = sqrt(0) = 0）。
enum class EuclideanNormEmptyKind {
    NORMAL = 0,
    EMPTY_A,
    EMPTY_R,
};

// tiling 全程工作状态：一次 TilingFunc 调用内的全部中间量，按生产阶段分区。
struct EuclideanNormCtx {
    // ── 平台参数（GetPlatformInfo 填充）──
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 0;
    int64_t cacheLineSize = 0;

    // ── 输入信息（GetShapeAndDtype / ParseAxesTensor 填充）──
    ge::DataType xDtype = ge::DT_UNDEFINED;
    int64_t dtypeSize = 0;
    int64_t maxDtypeSize = 0;
    std::vector<int64_t> xShape;
    std::vector<int64_t> reduceAxes;

    // ── 空 tensor 分类（ClassifyEmptyTensor 填充）──
    EuclideanNormEmptyKind emptyKind = EuclideanNormEmptyKind::NORMAL;
    int64_t aTotalEmpty = 0;

    // ── A/R 规整模式（BuildInitialAxisList + PreprocessPattern 填充）──
    std::vector<int64_t> axisShape;
    std::vector<bool> isReduceAxis;
    int32_t axisNum = 0;
    bool isTailR = false;

    // ── UB 切分参数（ComputeAUbFactor / ComputeRUbFactor / ExpandAIfRFullyLoaded 填充）──
    int32_t aSplitIdx = 0;
    int32_t rSplitIdx = 0;
    int64_t aUbFactor = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProdAlign = 0;

    // ── A 方向多核参数（ComputeFusedALoopSplit 填充）──
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // ── R 方向迭代数（ComputeRLoopCnt 填充）──
    int64_t rLoopCntTotal = 0;

    // ── UB buffer 尺寸（ComputeUbSizes 填充）──
    int64_t preBufSize = 0;
    int64_t postBufSize = 0;

    // ── Group 2D 分核参数（ComputeGroupSplit 填充）──
    bool isGroup = false;
    int64_t rGroupCnt = 0;
};

// 获取平台硬件参数：AIV 核数、UB 容量、block 大小、cache line 大小。
// 全走 platform 接口（禁止写死），任一参数为 0 视为环境异常直接失败。
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    // 平台信息句柄（构造器仅存裸指针，须判空后再使用）
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // AIV 核数：多核均分与 Group 触发阈值的基准
    ctx.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ctx.coreNum == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.coreNum",
                                                         "GetCoreNumAiv returned 0"),
                return ge::GRAPH_FAILED);

    // UB 容量：全部切分预算的总盘
    uint64_t ub = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ctx.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(ctx.ubSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.ubSize",
                                                         "GetCoreMemSize(UB) returned 0"),
                return ge::GRAPH_FAILED);

    // block大小 与 cache line 大小（后续A 切分的初始预算基准）
    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    ctx.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context));
    OP_CHECK_IF(ctx.blockSize == 0 || ctx.cacheLineSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.blockSize/cacheLineSize",
                                                         "BlockSize=" + std::to_string(ctx.blockSize) +
                                                             " or cacheLineSize=" + std::to_string(ctx.cacheLineSize) +
                                                             " is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 读取输入 x 的 shape 与 dtype，完成合法性校验并派生字节宽度，供后续 UB 预算使用。
static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    // 标量输入（rank-0）经 EnsureNotScalar 归一为 [1]，下游统一按 ≥1 维处理
    const gert::Shape& xS = Ops::Base::EnsureNotScalar(xShapePtr->GetStorageShape());
    const size_t rank = xS.GetDimNum();
    ctx.xShape.clear();
    for (size_t i = 0; i < rank; ++i) {
        // 负 dim 表示 动态 shape 未实例化，直接拒绝
        OP_CHECK_IF(
            xS.GetDim(i) < 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "x", std::to_string(xS.GetDim(i)),
                "dim[" + std::to_string(i) + "] is negative (dynamic shape must be instantiated before tiling)"),
            return ge::GRAPH_FAILED);
        ctx.xShape.push_back(xS.GetDim(i));
    }

    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ctx.xDtype = xDesc->GetDataType();
    // dtype 白名单（y 跟随 x，int32 不做隐式提升）
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32};
    OP_CHECK_IF(
        supportedDtypes.count(ctx.xDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(ctx.xDtype).c_str(),
                                              "only DT_FLOAT16/DT_BF16/DT_FLOAT/DT_INT32 are supported"),
        return ge::GRAPH_FAILED);
    // b16 中间计算过程需要提升精度到 fp32，UB 统一按 maxDtypeSize（≥4B）预算
    ctx.dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(ctx.xDtype));
    ctx.maxDtypeSize = std::max(ctx.dtypeSize, FP32_BYTES);

    OP_LOGI(context, "input: xShape=%s xDtype=%d dtypeSize=%ld", Ops::Base::ToString(xS).c_str(),
            static_cast<int>(ctx.xDtype), ctx.dtypeSize);
    return ge::GRAPH_SUCCESS;
}

// 归一化axis
static bool PushOneAxis(const char* nodeName, int64_t v, int64_t idx, int64_t xRank, std::set<int64_t>& seen,
                        std::vector<int64_t>& axesOut)
{
    if (v < -xRank || v >= xRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(v),
                                              "axes[" + std::to_string(idx) + "] out of range [-" +
                                                  std::to_string(xRank) + ", " + std::to_string(xRank) + ")");
        return false;
    }
    const int64_t norm = (v < 0) ? (v + xRank) : v; // 负索引：+xRank归一到 [0, xRank)
    if (!seen.insert(norm).second) {                // set 插入失败即重复轴
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(norm),
                                              "duplicate axis in axes (input idx=" + std::to_string(idx) + ")");
        return false;
    }
    axesOut.push_back(norm);
    return true;
}

// 按 T（int32/int64）逐元素读取 axes 并归一化入 axesOut
template <typename T>
static bool LoadAxesLoop(const char* nodeName, const T* data, int64_t axesNum, int64_t xRank, std::set<int64_t>& seen,
                         std::vector<int64_t>& axesOut)
{
    axesOut.reserve(static_cast<size_t>(axesNum));
    for (int64_t i = 0; i < axesNum; ++i) {
        if (!PushOneAxis(nodeName, static_cast<int64_t>(data[i]), i, xRank, seen, axesOut)) {
            return false;
        }
    }
    return true;
}

// 解析值依赖输入 axes 的具体数值到 ctx.reduceAxes：归一化、去重并升序排序。
static ge::graphStatus ParseAxesTensor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const gert::Tensor* axesTensor = context->GetInputTensor(INPUT_AXES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
    const int64_t axesNum = axesTensor->GetShapeSize();
    const int64_t xRank = static_cast<int64_t>(ctx.xShape.size());
    ctx.reduceAxes.clear();

    if (axesNum == 0) {
        // 空 axes = full reduce：展开为 [0, rank)（与 infershape 端语义一致）
        ctx.reduceAxes.resize(static_cast<size_t>(xRank));
        for (int64_t i = 0; i < xRank; ++i) {
            ctx.reduceAxes[static_cast<size_t>(i)] = i;
        }
        return ge::GRAPH_SUCCESS;
    }

    // int32 / int64 两种 dtype 分别读取axes
    const char* nodeName = context->GetNodeName();
    std::set<int64_t> seen;
    const ge::DataType dt = axesTensor->GetDataType();
    if (dt == ge::DT_INT32) {
        const int32_t* data = axesTensor->GetData<int32_t>();
        OP_CHECK_IF(
            data == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "axes", "axes const tensor data ptr is null (int32)"),
            return ge::GRAPH_FAILED);
        if (!LoadAxesLoop<int32_t>(nodeName, data, axesNum, xRank, seen, ctx.reduceAxes)) {
            return ge::GRAPH_FAILED;
        }
    } else if (dt == ge::DT_INT64) {
        const int64_t* data = axesTensor->GetData<int64_t>();
        OP_CHECK_IF(
            data == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "axes", "axes const tensor data ptr is null (int64)"),
            return ge::GRAPH_FAILED);
        if (!LoadAxesLoop<int64_t>(nodeName, data, axesNum, xRank, seen, ctx.reduceAxes)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "axes", Ops::Base::ToString(dt).c_str(),
                                              "only int32/int64 are supported");
        return ge::GRAPH_FAILED;
    }
    // 升序排序保证合轴顺序确定（bitwise reproducible）
    std::sort(ctx.reduceAxes.begin(), ctx.reduceAxes.end());
    return ge::GRAPH_SUCCESS;
}

// 初始化轴列表：拷贝 xShape 并按 reduceAxes 打 A/R 标记（A=保留轴，R=归约轴）。
static void BuildInitialAxisList(EuclideanNormCtx& ctx)
{
    const size_t rank = ctx.xShape.size();
    ctx.axisShape = ctx.xShape;
    ctx.isReduceAxis.assign(rank, false);
    // reduceAxes 命中的轴标 R，其余保持 A
    for (int64_t reduceAxis : ctx.reduceAxes) {
        ctx.isReduceAxis[static_cast<size_t>(reduceAxis)] = true;
    }
}

// A/R 规整第 1 步（共 4 步，顺序不可换）：删除全部 size=1 的轴（对归约结果无贡献）。
static void DropSizeOneAxes(EuclideanNormCtx& ctx)
{
    std::vector<int64_t> newShape;
    std::vector<bool> newIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (ctx.axisShape[i] != 1) {
            newShape.push_back(ctx.axisShape[i]);
            newIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    // 全部轴都是 1 时保留一根 A1 占位（避免空轴表）
    if (newShape.empty()) {
        newShape.push_back(1);
        newIsR.push_back(false);
    }
    ctx.axisShape = std::move(newShape);
    ctx.isReduceAxis = std::move(newIsR);
}

// A/R 规整第 2 步：相邻同类型轴乘积合并，压缩为最短 [A,R] 交替模式（ND 布局下数值等价）。
static void FuseAxis(EuclideanNormCtx& ctx)
{
    std::vector<int64_t> fusedShape;
    std::vector<bool> fusedIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        // 与前一根同类型 → 乘积并入前一根（A 合并=输出维乘积，R 合并=归约长度乘积）
        if (!fusedShape.empty() && fusedIsR.back() == ctx.isReduceAxis[i]) {
            fusedShape.back() *= ctx.axisShape[i];
        } else {
            fusedShape.push_back(ctx.axisShape[i]);
            fusedIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    ctx.axisShape = std::move(fusedShape);
    ctx.isReduceAxis = std::move(fusedIsR);
}

// A/R 规整第 3 步：首轴为 R 时在最前补一根 size=1 的 A 轴，保证模式以 A 起头
// （偶位 A / 奇位 R 下标不变量的前提）。
static void PadLeadingOneA(EuclideanNormCtx& ctx)
{
    if (!ctx.axisShape.empty() && ctx.isReduceAxis.front()) {
        ctx.axisShape.insert(ctx.axisShape.begin(), 1);
        ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), false);
    }
}

// A/R 规整第 4 步：纯 A 退化（无 R 轴）时补一根 R=1，保证模式合法。
static void PadRIfPureA(EuclideanNormCtx& ctx)
{
    bool hasR = false;
    for (bool isReduce : ctx.isReduceAxis) {
        if (isReduce) {
            hasR = true;
            break;
        }
    }
    if (!hasR) {
        if (ctx.axisShape.size() == 1 && ctx.axisShape[0] == 1) {
            // 全 1 退化 [A1]：末尾补 R=1（成 tail-R）
            ctx.axisShape.push_back(1);
            ctx.isReduceAxis.push_back(true);
        } else {
            // 其余：前置 [A1, R1]（成 tail-A）——禁止末尾补 R：最内轴为长度 1 的 R
            // 会使每 A 元素独占 1 元素 burst，搬运效率崩塌
            ctx.axisShape.insert(ctx.axisShape.begin(), {1, 1});
            ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), {false, true});
        }
    }
}

// 空 tensor 三分类（先于规整执行，在原始轴语义上判零维）。
static void ClassifyEmptyTensor(EuclideanNormCtx& ctx)
{
    bool hasZeroA = false;
    bool hasZeroR = false;
    int64_t aTotal = 1;
    // 扫描零维并累计非零 A 轴乘积（= EMPTY_R 的输出元素数）
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        const bool isR = ctx.isReduceAxis[i];
        const int64_t sz = ctx.xShape[i];
        if (!isR) {
            if (sz == 0) {
                hasZeroA = true;
            } else {
                aTotal *= sz;
            }
        } else {
            if (sz == 0) {
                hasZeroR = true;
            }
        }
    }
    if (hasZeroA) {
        // A 轴含 0 → 输出 0 元素
        ctx.emptyKind = EuclideanNormEmptyKind::EMPTY_A;
        ctx.aTotalEmpty = 0;
    } else if (hasZeroR) {
        // 仅 R 轴含 0 → 空归约 sum=0，每输出 = sqrt(0) = 0
        ctx.emptyKind = EuclideanNormEmptyKind::EMPTY_R;
        ctx.aTotalEmpty = aTotal;
    } else {
        ctx.emptyKind = EuclideanNormEmptyKind::NORMAL;
        ctx.aTotalEmpty = 0;
    }
}

// 用 dims 前 dimNum 个元素组装 gert::Shape，供 Ops::Base::ToString 统一打印 "[d0, d1, ...]"。
static gert::Shape MakeShapeFromDims(const int64_t* dims, int32_t dimNum)
{
    gert::Shape shape;
    for (int32_t i = 0; i < dimNum; ++i) {
        shape.AppendDim(dims[i]);
    }
    return shape;
}

// A/R 规整总装（仅 NORMAL 路径）：四步规整后校验轴数与交替不变量，并按轴数奇偶定 isTailR。
static ge::graphStatus PreprocessPattern(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    // 四步顺序不可换：去 1 → 合轴 → 补 leading A → 补 R 增广
    DropSizeOneAxes(ctx);
    FuseAxis(ctx);
    PadLeadingOneA(ctx);
    PadRIfPureA(ctx);

    // 轴数上界 = TilingData 定长数组长度 MAX_PATTERN_RANK
    ctx.axisNum = static_cast<int32_t>(ctx.axisShape.size());
    OP_CHECK_IF(ctx.axisNum < MIN_AXIS_NUM || ctx.axisNum > MAX_AXIS_NUM,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "axisNum",
                    "after A/R pattern regularization, axisNum=" + std::to_string(ctx.axisNum) + " out of [" +
                        std::to_string(MIN_AXIS_NUM) + ", " + std::to_string(MAX_AXIS_NUM) + "]"),
                return ge::GRAPH_FAILED);

    // 偶数轴 → tail-R（最内轴为 R）；奇数轴 → tail-A（最内轴为 A）
    ctx.isTailR = (ctx.axisNum % AXIS_INTERVAL == 0);
    // 规整结果打 DEBUG 日志（shape 经 Ops::Base::ToString 统一打印，轴类型由偶 A / 奇 R 不变量可知）
    OP_LOGD(context->GetNodeName(), "PreprocessPattern: axisNum=%d isTailR=%d axes=%s", ctx.axisNum,
            static_cast<int>(ctx.isTailR),
            Ops::Base::ToString(MakeShapeFromDims(ctx.axisShape.data(), ctx.axisNum)).c_str());
    return ge::GRAPH_SUCCESS;
}

// 计算规整后各轴在输入 GM 上的步长：stride[i] = ∏_{j>i} axisShape[j]
// （ND 连续布局，最内轴 stride=1），供 kernel 侧 Unravel 解码 GM 偏移。
static void ComputeAxisStrides(const EuclideanNormCtx& ctx, int64_t outStride[])
{
    int64_t strideAcc = 1;
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        outStride[i] = strideAcc;
        strideAcc *= ctx.axisShape[static_cast<size_t>(i)];
    }
}

// 返回最后一个 A 轴（最大偶下标）的位置；规整后恒存在，兜底返回 0（防御）。
static int32_t LastAAxisIdx(const EuclideanNormCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

// 返回最后一个 R 轴（最大奇下标）的位置；规整后恒存在，兜底返回 1（防御）。
static int32_t LastRAxisIdx(const EuclideanNormCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

// 计算全部 A 轴长度乘积 = 输出元素总数 aTotal。
static int64_t TotalAProd(const EuclideanNormCtx& ctx)
{
    int64_t total = 1;
    for (int32_t i = 0; i < ctx.axisNum; i += AXIS_INTERVAL) {
        total *= ctx.axisShape[static_cast<size_t>(i)];
    }
    return total;
}

// 计算 aSplitIdx 内侧 A 轴的对齐乘积 innerAProdAlign（aSplit 变化后需重算）。
static void ComputeInnerAProdAlign(EuclideanNormCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerAProdAlign = 1;
    for (int32_t k = ctx.aSplitIdx + AXIS_INTERVAL; k < ctx.axisNum; k += AXIS_INTERVAL) {
        // tail-A 最内 A 轴是搬运 burst 轴，按 32B 元素数（bsElem）向上对齐；其余轴原值累乘
        if (k == ctx.axisNum - 1 && !ctx.isTailR) {
            ctx.innerAProdAlign *= Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(k)], bsElem);
        } else {
            ctx.innerAProdAlign *= ctx.axisShape[static_cast<size_t>(k)];
        }
    }
}

// UB 切分 Step 1：以单条 cache line 为预算确定 A 切分（aSplitIdx / aUbFactor）
static void ComputeAUbFactor(EuclideanNormCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    // 对齐到 block 的 cache line 元素预算
    const int64_t maxInnerAInitElem = ctx.cacheLineSize / ctx.dtypeSize;
    const int64_t cachelineTmp = (maxInnerAInitElem / bsElem) * bsElem;

    // 从最内轴向外逐轴累乘，超预算即停（最内轴按 32B 对齐计入）
    int64_t product = 1;
    int32_t idx = ctx.axisNum - 1;
    while (idx >= 0) {
        int64_t axisSize = (idx == ctx.axisNum - 1) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(idx)], bsElem) :
                               ctx.axisShape[static_cast<size_t>(idx)];
        if (product * axisSize > cachelineTmp) {
            break;
        }
        product *= axisSize;
        idx--;
    }

    if (idx < 0) {
        // 全部装得下 → 不切（aSplitIdx=0）
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else if (idx % AXIS_INTERVAL == 0) {
        // 停在 A 轴 → 按剩余预算切该轴 chunk
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(cachelineTmp / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    } else {
        // 停在 R 轴 → 退到外侧 A 轴，因子 1
        ctx.aSplitIdx = idx - 1;
        ctx.aUbFactor = 1;
    }

    // aSplit 已定，重算内侧 A 对齐乘积
    ComputeInnerAProdAlign(ctx);
}

// 计算单次 R 迭代的可用元素上限 rIMax（UB 预算不等式取逆）；输入非法或预算不足返回 -1。
static int64_t ComputeRiMax(const EuclideanNormCtx& ctx)
{
    // 扣除 cacheBuf 固定预算后的可用 UB
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    const int64_t aUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    // outBuf（post 侧 1 份）按 32B 对齐的占用
    const int64_t postBufSize = Ops::Base::CeilAlign(aUnit * ctx.maxDtypeSize, ctx.blockSize);
    const int64_t aOnlyBytes = P_POST * postBufSize;
    // 每 R 元素的字节成本：3 个 pre buffer × aUnit × maxDtypeSize
    const int64_t bytesPerRElem = (P_PRE + P_PRE_EXT) * aUnit * ctx.maxDtypeSize;
    if (aUnit <= 0 || bytesPerRElem <= 0) {
        return -1;
    }
    const int64_t numer = ubAvailable - aOnlyBytes;
    if (numer <= 0) {
        return -1;
    }
    return numer / bytesPerRElem;
}

// UB 切分 Step 2：在 rIMax 预算内确定 R 切分（rSplitIdx / rUbFactor / rUbFactorAlign）。
static ge::graphStatus ComputeRUbFactor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const int32_t lastR = LastRAxisIdx(ctx);
    const int64_t rIMax = ComputeRiMax(ctx);
    // 连 1 个 R 元素都装不下 → TILING_FAIL
    OP_CHECK_IF(rIMax < 1,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "rIMax",
                    "R_i_max=" + std::to_string(rIMax) + " < 1 (aUbFactor=" + std::to_string(ctx.aUbFactor) +
                        ", innerAProdAlign=" + std::to_string(ctx.innerAProdAlign) + ")"),
                return ge::GRAPH_FAILED);

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerRProdAlign = 1;
    ctx.rSplitIdx = lastR;
    // 从最内 R 轴向外吸收整轴进 innerRProdAlign，超预算即停（tail-R 最内轴按 32B 对齐计入）
    while (ctx.rSplitIdx > 1) {
        int64_t axisSize = (ctx.rSplitIdx == lastR && ctx.isTailR) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)], bsElem) :
                               ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
        if (axisSize * ctx.innerRProdAlign > rIMax) {
            break;
        }
        ctx.innerRProdAlign *= axisSize;
        ctx.rSplitIdx -= AXIS_INTERVAL;
    }

    // 切分轴因子 = min(剩余预算折算元素数, 轴长)，至少 1
    const int64_t rAxisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
    ctx.rUbFactor = std::min(rIMax / ctx.innerRProdAlign, rAxisSize);
    ctx.rUbFactor = std::max<int64_t>(ctx.rUbFactor, 1);

    // 切在最内 R 轴（tail-R 的 burst 轴）时才需要 32B 对齐
    const bool isBurstTailR = (ctx.isTailR && ctx.rSplitIdx == lastR);
    if (isBurstTailR) {
        if (ctx.rUbFactor < rAxisSize) {
            // 切多块：FloorAlign 向下对齐（对齐到 0 表示连 1 个 block 都装不下，报错）
            ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
            OP_CHECK_IF(
                ctx.rUbFactor == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "rUbFactor",
                                                         "rUbFactor floor-aligned to 0 (tail-R burst alignment)"),
                return ge::GRAPH_FAILED);
            ctx.rUbFactorAlign = ctx.rUbFactor;
        } else {
            // 整轴装载：CeilAlign 向上补 pad（actual 与 padded 之差由 kernel 侧清零兜底）；
            // CeilAlign 后超预算则回退 FloorAlign（如果无条件 FloorAlign 会把整轴场景误判成 TILING_FAIL）
            ctx.rUbFactorAlign = Ops::Base::CeilAlign(ctx.rUbFactor, bsElem);
            if (ctx.rUbFactorAlign * ctx.innerRProdAlign > rIMax) {
                ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
                OP_CHECK_IF(ctx.rUbFactor == 0,
                            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                                context->GetNodeName(), "rUbFactor",
                                "rUbFactor floor-aligned to 0 (retry after CeilAlign overflow)"),
                            return ge::GRAPH_FAILED);
                ctx.rUbFactorAlign = ctx.rUbFactor;
            }
        }
    } else {
        // 非 burst 尾轴：padded = actual，无需对齐
        ctx.rUbFactorAlign = ctx.rUbFactor;
    }
    OP_LOGD(context->GetNodeName(),
            "ComputeRUbFactor: rSplitIdx=%d rUbFactor=%ld rUbFactorAlign=%ld innerRProdAlign=%ld", ctx.rSplitIdx,
            ctx.rUbFactor, ctx.rUbFactorAlign, ctx.innerRProdAlign);
    return ge::GRAPH_SUCCESS;
}

// 判定整个 R 空间是否单次迭代全载（即 rLoopCntTotal 将为 1）
// 全载时会把剩余 UB 让给 A。
static bool RIsFullyLoaded(const EuclideanNormCtx& ctx)
{
    // rUbFactor 未占满切分轴 → 有多块
    if (ctx.rUbFactor != ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)]) {
        return false;
    }
    // 切分轴外侧还有 R 轴 → 有外层迭代
    for (int32_t i = ctx.rSplitIdx - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return false;
        }
    }
    return true;
}

// R 全载时把 UB 预算反解为 A 单元上限 aUnitMax；预算非法返回 -1。
static int64_t SolveAUnitMax(const EuclideanNormCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    // 每 A lane 的字节成本：3 × rPadded 个 pre 元素 + 1 个 out 元素
    const int64_t coeff = (P_PRE + P_PRE_EXT) * rPaddedElems * ctx.maxDtypeSize + P_POST * ctx.maxDtypeSize;
    if (coeff <= 0) {
        return -1;
    }
    return ubAvailable / coeff;
}

// UB 切分 Step 3：R 全载时用剩余 UB 扩张 A 单元，减少 A 迭代次数
static ge::graphStatus ExpandAIfRFullyLoaded(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    // R 未全载（还有多轮 R 迭代）时不扩 A
    if (!RIsFullyLoaded(ctx)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    // cacheBuf 行宽硬上限：16KB ÷ fp32 = 4096 lane
    const int64_t cacheLaneLimit = CACHE_BUF_BYTES / FP32_BYTES;
    int64_t totalA = TotalAProd(ctx);

    // 三重约束取 min：UB 反解值 / totalA / cacheBuf 行宽上限；tail-A 再按 32B 向下对齐（最内 A 是 burst 轴）
    int64_t aUnitMax = std::min(SolveAUnitMax(ctx), totalA);
    aUnitMax = std::min(aUnitMax, cacheLaneLimit);
    if (!ctx.isTailR) {
        aUnitMax = Ops::Base::FloorAlign(aUnitMax, bsElem);
    }

    // 扩张目标不大于当前 A 单元则保持不变
    const int64_t curAUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    if (aUnitMax <= curAUnit) {
        return ge::GRAPH_SUCCESS;
    }

    // 从 lastA 向内重新累积，超 aUnitMax 即停（只遍历 A 轴，R 已全驻不参与计算）
    int64_t product = 1;
    int32_t idx = LastAAxisIdx(ctx);
    while (idx >= 0) {
        int64_t axisSize = (idx == ctx.axisNum - 1 && !ctx.isTailR) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(idx)], bsElem) :
                               ctx.axisShape[static_cast<size_t>(idx)];
        if (product * axisSize > aUnitMax) {
            break;
        }
        product *= axisSize;
        idx -= AXIS_INTERVAL;
    }

    if (idx < 0) {
        // 全部 A 轴装得下 → 不切
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else {
        // 停在 A 轴 → 按剩余预算切
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(aUnitMax / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    }

    // aSplit 重定，重算内侧 A 对齐乘积
    ComputeInnerAProdAlign(ctx);
    OP_LOGD(context->GetNodeName(), "ExpandAIfRFullyLoaded: triggered, aSplitIdx=%d aUbFactor=%ld innerAProdAlign=%ld",
            ctx.aSplitIdx, ctx.aUbFactor, ctx.innerAProdAlign);
    return ge::GRAPH_SUCCESS;
}

// A 方向多核切分：外层 A 轴 fused 成一个大的循环后按核均分；迭代-核映射固定；
static void ComputeFusedALoopSplit(EuclideanNormCtx& ctx)
{
    // aSplitIdx 外侧 A 轴乘积（外层迭代维）
    int64_t outerAProd = 1;
    for (int32_t i = 0; i < ctx.aSplitIdx; i += AXIS_INTERVAL) {
        outerAProd *= ctx.axisShape[static_cast<size_t>(i)];
    }
    // 切分轴 chunk 数
    const int64_t aSplitAxisSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitIdx)];
    ctx.aSplitChunkCnt = Ops::Base::CeilDiv(aSplitAxisSize, ctx.aUbFactor);
    ctx.aLoopCntTotal = outerAProd * ctx.aSplitChunkCnt;

    // 大小核均分：前 aLoopCntTotal % coreNum 个核多干 1 轮；
    // usedCoreNum = 全核（小核轮数 > 0）或仅大核数，至少 1
    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = static_cast<int32_t>(ctx.aLoopCntTotal % ctx.coreNum);
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, 1);
}

// R 方向总迭代数：外层 R 轴乘积 × 切分轴 chunk 数；供 kernel 二分树参数与 Group 触发判定。
static void ComputeRLoopCnt(EuclideanNormCtx& ctx)
{
    // rSplitIdx 外侧 R 轴乘积（外层迭代维）
    int64_t outerR = 1;
    for (int32_t i = 1; i < ctx.rSplitIdx; i += AXIS_INTERVAL) {
        outerR *= ctx.axisShape[static_cast<size_t>(i)];
    }
    const int64_t rChunks = Ops::Base::CeilDiv(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)], ctx.rUbFactor);
    ctx.rLoopCntTotal = outerR * rChunks;
}

// 计算 UB buffer 大小
static void ComputeUbSizes(EuclideanNormCtx& ctx)
{
    // A 单元（lane 数）与 R padded 行宽（元素数）
    const int64_t aUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    // preIn / preRes / preResTail 三 buffer 同尺寸，统一按 maxDtypeSize 预算（b16 也按 4B 计）
    ctx.preBufSize = aUnit * rPaddedElems * ctx.maxDtypeSize;
    // outBuf：一维 32B block 对齐
    ctx.postBufSize = Ops::Base::CeilAlign(aUnit * ctx.maxDtypeSize, ctx.blockSize);
}

// Group 2D 分核触发判定：A 方向吃不满半数核且 R 方向有多轮迭代可切。
static bool ShouldUseGroup(const EuclideanNormCtx& ctx)
{
    // A 并行度吃不满半数核（核闲置）
    if (ctx.aLoopCntTotal > ctx.coreNum / GROUP_CORE_RATIO) {
        return false;
    }
    // R 无多轮迭代可切（借不到并行度）
    if (ctx.rLoopCntTotal <= 1) {
        return false;
    }
    return true;
}

// Group 2D 分核参数：按核均分总迭代次数并对齐到 aLoopCntTotal 整数倍，得 usedCoreNum 与 rGroupCnt。
static void ComputeGroupSplit(EuclideanNormCtx& ctx)
{
    // 总迭代数 × 每核负载 → 实际启用的核块数
    int64_t totalOuter = ctx.aLoopCntTotal * ctx.rLoopCntTotal;
    int64_t perCoreNum = Ops::Base::CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks = Ops::Base::CeilDiv(totalOuter, perCoreNum);

    // 对齐到 aLoopCntTotal 整数倍（超核数则向下对齐）——保证每核 = 整数个完整 A chunk × 一段完整 R 区间；
    // Phase1 核index 映射关系：blockIdx = aChunkIdx×rGroupCnt + rChunkIdx
    if (Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal) <= ctx.coreNum) {
        numBlocks = Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal);
    } else {
        numBlocks = Ops::Base::FloorAlign(numBlocks, ctx.aLoopCntTotal);
    }

    ctx.usedCoreNum = static_cast<int32_t>(numBlocks);
    // R 分组数 = Phase2 workspace 行数
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal;
    ctx.isGroup = true;
}

// 申报 workspace 大小：用户部分（仅 Group 需要）+ 系统部分。
static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    size_t usrSize = 0;
    if (ctx.isGroup) {
        // [rGroupCnt 行, aTotal 列] fp32 dense 部分和矩阵：Phase1 写入 / Phase2 RA 归约读出
        int64_t aTotal = TotalAProd(ctx);
        usrSize = static_cast<size_t>(ctx.rGroupCnt) * static_cast<size_t>(aTotal) * sizeof(float);
    }

    // 系统 workspace 由框架接口给出；normal / empty 仅需系统部分
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ws[0] = usrSize + sysWorkspaceSize;
    OP_LOGI(context, "Set ws size:%lu, usrSize:%lu", ws[0], usrSize);
    return ge::GRAPH_SUCCESS;
}

// 把 ctx 全量写入 EuclideanNormTilingData，并打全量 INFO 日志便于问题定位。
static ge::graphStatus FillAndLogTilingData(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    EuclideanNormTilingData* td = context->GetTilingData<EuclideanNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    // 清零兜底：保证未用槽位取值确定
    OP_CHECK_IF(memset_s(td, sizeof(EuclideanNormTilingData), 0, sizeof(EuclideanNormTilingData)) != EOK,
                OP_LOGE(context, "Memset tilingdata error"), return ge::GRAPH_FAILED);

    // GM 步长现场计算（ND 连续布局，最内轴 stride=1）
    int64_t axisStride[MAX_PATTERN_RANK] = {0};
    ComputeAxisStrides(ctx, axisStride);

    // ── pattern 与定长数组：未用轴位置按约定填 axisShape=1 / axisStride=0 ──
    td->axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i] = (i < ctx.axisNum) ? ctx.axisShape[static_cast<size_t>(i)] : 1;
        td->axisStride[i] = (i < ctx.axisNum) ? axisStride[i] : 0;
    }
    // ── A 方向多核参数 ──
    td->aLoopCntTotal = ctx.aLoopCntTotal;
    td->aSplitChunkCnt = ctx.aSplitChunkCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->usedCoreNum = ctx.usedCoreNum;
    // ── UB 切分参数 ──
    td->aSplitIdx = ctx.aSplitIdx;
    td->rSplitIdx = ctx.rSplitIdx;
    td->aUbFactor = ctx.aUbFactor;
    td->rUbFactor = ctx.rUbFactor;
    td->rUbFactorAlign = ctx.rUbFactorAlign;
    td->innerAProdAlign = ctx.innerAProdAlign;
    td->innerRProdAlign = ctx.innerRProdAlign;
    td->rLoopCntTotal = ctx.rLoopCntTotal;
    // ── buffer 尺寸（cacheBuf 固定 16KB 直填）──
    td->preBufSize = ctx.preBufSize;
    td->postBufSize = ctx.postBufSize;
    td->cacheBufUbSize = CACHE_BUF_BYTES;
    td->rGroupCnt = ctx.rGroupCnt;

    // 全量 INFO：pattern / 切分 / 多核 / UB 一次打全，便于问题定位
    OP_LOGI(context, "tiling: dtype=%d axisNum=%d isTailR=%d usedCoreNum=%d isGroup=%d", static_cast<int>(ctx.xDtype),
            ctx.axisNum, static_cast<int>(ctx.isTailR), ctx.usedCoreNum, static_cast<int>(ctx.isGroup));
    OP_LOGI(context, "  aSplitIdx=%d aUbFactor=%ld rSplitIdx=%d rUbFactor=%ld rUbFactorAlign=%ld", td->aSplitIdx,
            td->aUbFactor, td->rSplitIdx, td->rUbFactor, td->rUbFactorAlign);
    OP_LOGI(context, "  innerAProdAlign=%ld innerRProdAlign=%ld rLoopCntTotal=%ld", td->innerAProdAlign,
            td->innerRProdAlign, td->rLoopCntTotal);
    OP_LOGI(context, "  UB: preBuf=%ld postBuf=%ld cache=%ld", td->preBufSize, td->postBufSize, td->cacheBufUbSize);
    OP_LOGI(context, "  aLoopCntTotal=%ld aSplitChunkCnt=%ld aBigCoreLoopCnt=%ld aSmallCoreLoopCnt=%ld aBigCoreCnt=%d",
            td->aLoopCntTotal, td->aSplitChunkCnt, td->aBigCoreLoopCnt, td->aSmallCoreLoopCnt, td->aBigCoreCnt);
    OP_LOGI(context, "  rGroupCnt=%ld", td->rGroupCnt);
    // axisShape / axisStride 经 Ops::Base::ToString 统一打印 "[d0, d1, ...]"
    OP_LOGI(context, "  axisShape=%s", Ops::Base::ToString(MakeShapeFromDims(td->axisShape, td->axisNum)).c_str());
    OP_LOGI(context, "  axisStride=%s", Ops::Base::ToString(MakeShapeFromDims(td->axisStride, td->axisNum)).c_str());
    return ge::GRAPH_SUCCESS;
}

// EMPTY_R 输出填充切分：无需读输入，仅向 y 写 aTotal 个固化值 0（sqrt(0)=0）。
static void ComputeEmptyRTiling(EuclideanNormCtx& ctx)
{
    // 单 buf 上限：64KB 与 UB 预算取小
    constexpr int64_t maxSingleUbBytes = 64 * 1024;
    const int64_t maxUbFactor = std::min(maxSingleUbBytes / ctx.maxDtypeSize, ctx.ubSize / P_POST / ctx.maxDtypeSize);

    // 每核下界 4KB：防输出碎片化（优先铺满多核）
    constexpr int64_t minBytesPerCore = 4096;
    const int64_t minAPerCore = Ops::Base::CeilDiv(minBytesPerCore, ctx.maxDtypeSize);
    const int64_t aTotal = ctx.aTotalEmpty;

    // aUbFactor = clamp(max(下界, CeilDiv(aTotal, coreNum)), 上限, aTotal)，至少 1
    int64_t aUbFactor = std::max(minAPerCore, Ops::Base::CeilDiv(aTotal, ctx.coreNum));
    aUbFactor = std::min(aUbFactor, maxUbFactor);
    aUbFactor = std::min(aUbFactor, aTotal);
    aUbFactor = std::max<int64_t>(aUbFactor, 1);

    // 输出区间按大小核协议均分
    const int64_t aLoopCntTotal = Ops::Base::CeilDiv(aTotal, aUbFactor);
    const int64_t aSmallCoreLoopCnt = aLoopCntTotal / ctx.coreNum;
    const int32_t aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % ctx.coreNum);
    const int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
    const int32_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : std::max(aBigCoreCnt, 1);

    ctx.aUbFactor = aUbFactor;
    ctx.aLoopCntTotal = aLoopCntTotal;
    ctx.aSplitChunkCnt = aLoopCntTotal;
    ctx.aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    ctx.aBigCoreLoopCnt = aBigCoreLoopCnt;
    ctx.aBigCoreCnt = aBigCoreCnt;
    ctx.usedCoreNum = usedCoreNum;

    // outBuf = CeilAlign(max(aUbFactor × maxDtypeSize, 32B), 32B)；不申请任何输入侧 buffer
    int64_t outRaw = aUbFactor * ctx.maxDtypeSize;
    outRaw = std::max(outRaw, ctx.blockSize);
    ctx.postBufSize = Ops::Base::CeilAlign(outRaw, ctx.blockSize);
    ctx.preBufSize = 0;
}

// 填充 EuclideanNormEmptyTilingData（EMPTY_A 全零返回，EMPTY_R 填输出切分字段）。
static ge::graphStatus FillEmptyTilingData(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    EuclideanNormEmptyTilingData* td = context->GetTilingData<EuclideanNormEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    // 清零兜底：EMPTY_A 的 7 字段全零即为合法语义
    OP_CHECK_IF(memset_s(td, sizeof(EuclideanNormEmptyTilingData), 0, sizeof(EuclideanNormEmptyTilingData)) != EOK,
                OP_LOGE(context, "Memset tilingdata error"), return ge::GRAPH_FAILED);

    if (ctx.emptyKind == EuclideanNormEmptyKind::EMPTY_A) {
        // 全零 TilingData：kernel 侧 usedCoreNum=0 全核早退
        OP_LOGI(context, "EMPTY_A: dtype=%d usedCoreNum=0", static_cast<int>(ctx.xDtype));
        return ge::GRAPH_SUCCESS;
    }

    // EMPTY_R 输出切分：aTotal / aUbFactor / 大小核三参 / postBufSize
    td->aTotal = ctx.aTotalEmpty;
    td->usedCoreNum = ctx.usedCoreNum;
    td->aUbFactor = ctx.aUbFactor;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->postBufSize = ctx.postBufSize;

    OP_LOGI(context, "EMPTY_R: dtype=%d aTotal=%ld aUbFactor=%ld usedCoreNum=%d postBuf=%ld",
            static_cast<int>(ctx.xDtype), ctx.aTotalEmpty, td->aUbFactor, td->usedCoreNum, td->postBufSize);
    return ge::GRAPH_SUCCESS;
}

// 空 tensor 快速路径总装：主流程的短路出口，跳过整套 normal 切分。
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    if (ctx.emptyKind == EuclideanNormEmptyKind::EMPTY_R) {
        // EMPTY_R：算输出填充切分
        ComputeEmptyRTiling(ctx);
        OP_LOGD(context->GetNodeName(), "EmptyR: aUbFactor=%ld usedCoreNum=%d postBufSize=%ld", ctx.aUbFactor,
                ctx.usedCoreNum, ctx.postBufSize);
    } else {
        // EMPTY_A：usedCoreNum=0，kernel 全核零操作
        ctx.usedCoreNum = 0;
    }
    OP_CHECK_IF(FillEmptyTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // 固定 tilingKey(isGroup=0, isEmptyTensor=1) 路由到 Empty 模板
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(0U), static_cast<uint64_t>(1U));
    context->SetTilingKey(tilingKey);
    // 框架要求 BlockDim ≥ 1（EMPTY_A 启 1 核即退）
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// TilingFunc 主入口（IMPL_OP_OPTILING 注册）：编排「输入解析 → 空短路 → A/R 规整 →
// UB 三步切分 → 多核/Group 切分 → 填 TilingData → 落 TilingKey/BlockDim/Workspace」全链。
static ge::graphStatus EuclideanNormTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin EuclideanNormTilingFunc");
    EuclideanNormCtx ctx;

    // ── ① 输入解析：平台参数 / x shape-dtype / axes 值 ──
    OP_CHECK_IF(GetPlatformInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAndDtype(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAxesTensor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── ② 空分类（原始轴语义）；非 NORMAL 走空快速路径直接返回 ──
    BuildInitialAxisList(ctx);
    ClassifyEmptyTensor(ctx);

    if (ctx.emptyKind != EuclideanNormEmptyKind::NORMAL) {
        return HandleEmptyTensor(context, ctx);
    }

    // ── ③ A/R 规整 ──
    OP_CHECK_IF(PreprocessPattern(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── ④ UB 三步切分：A 爬坡 → R 反解 → R 全载扩 A ──
    ComputeAUbFactor(ctx);
    OP_LOGD(context->GetNodeName(), "ComputeAUbFactor: aSplitIdx=%d aUbFactor=%ld innerAProdAlign=%ld", ctx.aSplitIdx,
            ctx.aUbFactor, ctx.innerAProdAlign);
    OP_CHECK_IF(ComputeRUbFactor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExpandAIfRFullyLoaded(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── ⑤ 多核切分：A 方向均分 + R 迭代数 ──
    ComputeFusedALoopSplit(ctx);
    ComputeRLoopCnt(ctx);
    OP_LOGD(context->GetNodeName(), "MultiCore: aLoopCntTotal=%ld aSplitChunkCnt=%ld usedCoreNum=%d rLoopCntTotal=%ld",
            ctx.aLoopCntTotal, ctx.aSplitChunkCnt, ctx.usedCoreNum, ctx.rLoopCntTotal);

    // ── ⑥ Group 2D 分核（触发则切 batch 调度模式，kernel SyncAll 依赖）──
    if (ShouldUseGroup(ctx)) {
        ComputeGroupSplit(ctx);
        OP_LOGD(context->GetNodeName(), "Group: rGroupCnt=%ld usedCoreNum=%d", ctx.rGroupCnt, ctx.usedCoreNum);
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to set ScheduleMode!"), return ge::GRAPH_FAILED);
    }

    // ── ⑦ buffer 尺寸 ──
    ComputeUbSizes(ctx);
    OP_LOGD(context->GetNodeName(), "UbSizes: preBufSize=%ld postBufSize=%ld", ctx.preBufSize, ctx.postBufSize);

    // ── ⑧ 填 TilingData（全量 INFO）──
    OP_CHECK_IF(FillAndLogTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // 输出路由：TilingKey(isGroup, isEmptyTensor=0) + BlockDim + Workspace
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(static_cast<uint64_t>(ctx.isGroup ? 1U : 0U), static_cast<uint64_t>(0U));
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct EuclideanNormCompileInfo {};

// TilingParse 空实现：EuclideanNormCompileInfo 为空结构，
// 为了覆盖A2/A3 TilingParse 注册的接口，必须添加这个空实现接口，否则存在不兼容报错的风险
static ge::graphStatus TilingParseForEuclideanNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EuclideanNorm)
    .Tiling(EuclideanNormTilingFunc)
    .TilingParse<EuclideanNormCompileInfo>(TilingParseForEuclideanNorm)
    .TilingInputsDataDependency({1});

} // namespace optiling
