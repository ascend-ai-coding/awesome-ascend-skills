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
 * \file euclidean_norm_infershape.cpp
 * \brief EuclideanNorm 算子的 output shape 推理。
 *        （输出 dtype 推理在 op_graph/euclidean_norm_graph_infer.cpp 注册）
 *
 *   - axes 是 const tensor（IndexNumberType：int32 / int64）；非 const（nullptr）→ GRAPH_FAILED
 *     （与内置 ReduceSum/ReduceMax 严格策略同型：值依赖输入须编译期可 const-fold）
 *   - axes==[]（empty）→ 视为 full reduce，等价 axes=[0..rank-1]（对标 TF / PyTorch；与 NumPy axis=() identity 不同）
 *   - 越界 / 重复 axes → GRAPH_FAILED（与 tiling 端 ParseAxesTensor 同步报错；
 *     重复 axes 在 keep_dims=true 下会让 shape 幂等地正确，但 tiling 仍会拒绝，
 *     infershape 端早拒绝免得后端做无用功）
 *   - keep_dims 是 OPTIONAL attr，默认 false（与 op_def `Attr("keep_dims").AttrType(OPTIONAL).Bool(false)` 对齐）；
 *     GetAttrPointer 返回 nullptr 时按默认值兜底
 *   - 输出 shape 通过 Ops::Base::ReduceDimsWith{Keep,Without}KeepDims helper 推导
 *   - 输入是 unknown rank → 输出标 unknown rank（不再继续推导）
 */

#include <set>
#include <string>
#include <vector>

#include "op_host/infershape_reduce_util.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_AXES_IDX = 1;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t ATTR_KEEP_DIMS_IDX = 0;
constexpr bool KEEP_DIMS_DEFAULT = false; // op_def: Attr("keep_dims").AttrType(OPTIONAL).Bool(false)
} // namespace

// 归一化并追加单个 axis：负索引 +xRank 归一到 [0, xRank)；越界/重复打 ERROR 并返回 false。
static bool PushOneAxis(const char* nodeName, int64_t v, int64_t idx, int64_t xRank, std::set<int64_t>& seen,
                        std::vector<int64_t>& axesOut)
{
    if (v < -xRank || v >= xRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(v),
                                              "axes[" + std::to_string(idx) + "] out of range [-" +
                                                  std::to_string(xRank) + ", " + std::to_string(xRank) + ")");
        return false;
    }
    const int64_t norm = (v < 0) ? (v + xRank) : v; // 负索引归一到 [0, xRank)
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

// 解析 axes tensor 内容并归一化、去重
static bool LoadAxesFromTensor(const char* nodeName, const gert::Tensor* axesTensor, int64_t xRank,
                               std::vector<int64_t>& axesOut)
{
    const int64_t axesNum = axesTensor->GetShapeSize();
    if (axesNum == 0) {
        // 空 axes = full reduce：展开 [0, xRank)
        axesOut.resize(static_cast<size_t>(xRank));
        for (int64_t i = 0; i < xRank; ++i) {
            axesOut[static_cast<size_t>(i)] = i;
        }
        return true;
    }

    // int32 / int64 两种 dtype 分别读取
    std::set<int64_t> seen;
    const ge::DataType dt = axesTensor->GetDataType();
    if (dt == ge::DT_INT32) {
        const int32_t* data = axesTensor->GetData<int32_t>();
        OP_CHECK_IF(
            data == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "axes", "axes const tensor data ptr is null (int32)"),
            return false);
        return LoadAxesLoop<int32_t>(nodeName, data, axesNum, xRank, seen, axesOut);
    }
    if (dt == ge::DT_INT64) {
        const int64_t* data = axesTensor->GetData<int64_t>();
        OP_CHECK_IF(
            data == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "axes", "axes const tensor data ptr is null (int64)"),
            return false);
        return LoadAxesLoop<int64_t>(nodeName, data, axesNum, xRank, seen, axesOut);
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "axes", Ops::Base::ToString(dt).c_str(),
                                          "only int32/int64 are supported");
    return false;
}

// 0-D（标量）输入分支：keep_dims=false 输出 0-D 标量、keep_dims=true 输出 [1]。
static ge::graphStatus HandleScalarInput(gert::InferShapeContext* context, bool keepDims, gert::Shape* yShape)
{
    const gert::Tensor* axesTensor = context->GetInputTensor(INPUT_AXES_IDX);
    // 标量无可归约轴：axes 必须为空（nullptr 或 ShapeSize==0 均合法）
    if (axesTensor != nullptr && axesTensor->GetShapeSize() > 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes",
                                              std::to_string(axesTensor->GetShapeSize()),
                                              "axes must be empty for scalar (0-D) input x");
        return GRAPH_FAILED;
    }

    // 语义对标 TF/PyTorch 标量 reduce：|scalar| 归约结果为 0 维数值
    if (keepDims) {
        *yShape = gert::Shape({1}); // keep_dims 保留被归约轴，标量轴视作长度 1
    } else {
        *yShape = gert::Shape(); // 空 shape 即 0-D 标量
    }
    OP_LOGD(context->GetNodeName(), "Scalar input: keepDims=%d, yShape=%s", static_cast<int>(keepDims),
            ToString(*yShape).c_str());
    return GRAPH_SUCCESS;
}

// 输出 shape 推理主入口（IMPL_OP_INFERSHAPE 注册）：按 axes 与 keep_dims 推导 y 的 shape。
static ge::graphStatus InferShape4EuclideanNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4EuclideanNorm");

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 1) unknown rank 透传：输入未知 → 输出标未知（reduce_var_infershape 同型）
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGI(context->GetNodeName(), "X is unknown rank; set y as unknown rank");
        return GRAPH_SUCCESS;
    }
    const int64_t xRank = static_cast<int64_t>(xShape->GetDimNum());

    // 2) keep_dims 是 OPTIONAL attr：GetAttrPointer 可能返回 nullptr，按 op_def 默认值 false 兜底
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* attrKeepDims = attrs->GetAttrPointer<bool>(ATTR_KEEP_DIMS_IDX);
    const bool keepDims = (attrKeepDims == nullptr) ? KEEP_DIMS_DEFAULT : (*attrKeepDims);

    // 3) scalar (0-D) input：gert::Shape::IsScalar() 即 GetDimNum()==0 的官方封装
    if (xShape->IsScalar()) {
        return HandleScalarInput(context, keepDims, yShape);
    }

    // 4) axes 是 const tensor at compile time：非 const（nullptr）→ 拒绝
    //    与内置 ReduceSum/ReduceMax 同型严格策略：值依赖输入 const-fold 失败即编译期报错，
    //    不把 unknown rank 传播给下游（下游无需支持 unknown shape 语义）。
    const gert::Tensor* axesTensor = context->GetInputTensor(INPUT_AXES_IDX);
    if (axesTensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "axes",
                                                 "axes tensor is not const at infer time; require const axes input "
                                                 "(value-dependent input must be const-foldable at compile time)");
        return GRAPH_FAILED;
    }

    // 5) axes 解析：empty → all reduce (axes=[0..xRank-1])；否则按 int32/int64 读 + 越界/重复检查
    std::vector<int64_t> axes;
    if (!LoadAxesFromTensor(context->GetNodeName(), axesTensor, xRank, axes)) {
        return GRAPH_FAILED;
    }

    // 6) 调 opbase helper 计算输出 shape
    const int32_t axesSize = static_cast<int32_t>(axes.size());
    ge::graphStatus stat = keepDims ?
                               Ops::Base::ReduceDimsWithKeepDims<int64_t>(xShape, axes.data(), axesSize, yShape) :
                               Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(xShape, axes.data(), axesSize, yShape);

    OP_LOGD(context->GetNodeName(), "End InferShape: keepDims=%d, xRank=%ld, axes.size=%zu, outDim=%zu, status=%d",
            static_cast<int>(keepDims), xRank, axes.size(), yShape->GetDimNum(), static_cast<int>(stat));
    return stat;
}

IMPL_OP_INFERSHAPE(EuclideanNorm)
    .InferShape(InferShape4EuclideanNorm)
    .InputsDataDependency({1}); // axes（索引 1）值依赖：infershape 阶段需读取张量内容

} // namespace ops
