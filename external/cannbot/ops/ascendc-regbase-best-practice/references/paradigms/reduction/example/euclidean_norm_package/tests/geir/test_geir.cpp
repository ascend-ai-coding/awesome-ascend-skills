/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * euclidean_norm_package/tests/geir/test_geir.cpp
 * =============================================================================
 * Role: GE (Graph Engine) IR integration test for EuclideanNorm.
 *        Tests graph compilation and execution with a value-dependent axes
 *        Const input.  Verifies both static-shape and dynamic-shape modes.
 *
 * Graph topology:
 *   Data("x", index=0)  ----\
 *                           +--> EuclideanNorm --> y (output)
 *   Const("axes")       ----/
 *
 * Coverage (mirrors add_custom_package/tests/geir):
 *   - fp16 + fp32 via template <typename T>.
 *   - Static shapes: 14 sizes × 2 dtypes.
 *   - Dynamic shapes: 1 graph per dtype, reused for all sizes.
 * =============================================================================
 */

#include <cstdio>
#include <cmath>
#include <type_traits>
#include <vector>
#include <map>
#include <memory>

#include "acl/acl.h"
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api.h"
#include "array_ops.h"
#include "op_proto.h"

using namespace ge;

#define CHECK_GE(x) \
    do { \
        ge::Status __ret = (x); \
        if (__ret != ge::SUCCESS) { \
            fprintf(stderr, "GE error %u at %s:%d: %s\n", __ret, __FILE__, __LINE__, #x); \
            return -1; \
        } \
    } while (0)

template <typename T>
static T FromFloat(float v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloatToFloat16(v);
    } else {
        return v;
    }
}

template <typename T>
static float ToFloat(T v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloat16ToFloat(v);
    } else {
        return v;
    }
}

template <typename T>
static ge::DataType DtypeOf()
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return ge::DT_FLOAT16;
    } else {
        return ge::DT_FLOAT;
    }
}

template <typename T>
static const char* DtypeName()
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return "fp16";
    } else {
        return "fp32";
    }
}

/**
 * BuildGraph: Data(x) + Const(axes={1}) -> EuclideanNorm -> y
 *
 * For static: xShape = {rows, cols}, yShape = {rows}.
 * For dynamic: xShape = {-1, -1}, yShape = {-1}.
 */
static int BuildGraph(ge::Graph& graph, const std::vector<int64_t>& xShape, const std::vector<int64_t>& yShape,
                      const std::vector<int64_t>& axesData, ge::DataType dtype)
{
    auto euclidean_norm = op::EuclideanNorm("euclidean_norm");

    // x: Data placeholder
    ge::TensorDesc xDesc(ge::Shape(xShape), ge::FORMAT_ND, dtype);
    xDesc.SetRealDimCnt(static_cast<int64_t>(xShape.size()));
    auto dataX = op::Data("x").set_attr_index(0);
    dataX.update_input_desc_x(xDesc);
    dataX.update_output_desc_y(xDesc);
    graph.AddOp(dataX);
    euclidean_norm.set_input_x(dataX);

    // axes: Const (value-dependent — must be Const, not Data)
    ge::TensorDesc axesDesc(ge::Shape({static_cast<int64_t>(axesData.size())}), ge::FORMAT_ND, ge::DT_INT64);
    axesDesc.SetPlacement(ge::kPlacementHost);
    axesDesc.SetRealDimCnt(1);
    std::vector<int64_t> axesBuf(axesData);
    ge::Tensor axesTensor(axesDesc, reinterpret_cast<uint8_t*>(axesBuf.data()), axesBuf.size() * sizeof(int64_t));
    auto axesConst = op::Const("axes");
    axesConst.SetAttr("value", axesTensor);
    axesConst.update_output_desc_y(axesDesc);
    graph.AddOp(axesConst);
    euclidean_norm.set_input_axes(axesConst);
    euclidean_norm.update_input_desc_axes(axesDesc);

    // y output
    ge::TensorDesc yDesc(ge::Shape(yShape), ge::FORMAT_ND, dtype);
    yDesc.SetRealDimCnt(static_cast<int64_t>(yShape.size()));
    euclidean_norm.update_output_desc_y(yDesc);
    euclidean_norm.set_attr_keep_dims(false);

    std::vector<ge::Operator> inputs = {dataX};
    std::vector<ge::Operator> outputs = {euclidean_norm};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return 0;
}

/**
 * MakeFeedTensor<T>: host GE tensor filled with index-dependent values.
 * x[i][j] = (linearIdx % 32) * 0.1f — bounded for fp16.
 */
template <typename T>
static ge::Tensor MakeFeedTensor(int64_t rows, int64_t cols)
{
    ge::TensorDesc desc(ge::Shape({rows, cols}), ge::FORMAT_ND, DtypeOf<T>());
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(2);
    size_t num = static_cast<size_t>(rows * cols);
    std::vector<T> buf(num);
    for (size_t i = 0; i < num; ++i) {
        buf[i] = FromFloat<T>(static_cast<float>(i % 32) * 0.1f);
    }
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(buf.data()), num * sizeof(T));
}

/**
 * RunAndVerify<T>: execute graph, verify out[i] = sqrt(sum_j x[i][j]^2).
 */
template <typename T>
static int RunAndVerify(ge::Session* session, uint32_t graph_id, int64_t rows, int64_t cols, const char* mode)
{
    constexpr bool isFp16 = std::is_same_v<T, aclFloat16>;
    std::vector<ge::Tensor> feeds;
    feeds.push_back(MakeFeedTensor<T>(rows, cols));

    std::vector<ge::Tensor> output;
    CHECK_GE(session->RunGraph(graph_id, feeds, output));

    if (output.size() != 1) {
        fprintf(stderr, "[%s/%s] Expected 1 output, got %zu\n", mode, DtypeName<T>(), output.size());
        return -1;
    }

    const T* outData = reinterpret_cast<const T*>(output[0].GetData());
    bool ok = true;
    for (int64_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            float v = ToFloat<T>(FromFloat<T>(static_cast<float>((i * cols + j) % 32) * 0.1f));
            acc += static_cast<double>(v) * v;
        }
        float expected = std::sqrt(static_cast<float>(acc));
        float got = ToFloat<T>(outData[static_cast<size_t>(i)]);
        float tol = isFp16 ? 1.0f : 1e-3f;
        if (std::fabs(got - expected) > tol) {
            fprintf(stderr, "[%s/%s] Mismatch at row %ld: got %f, expected %f\n", mode, DtypeName<T>(), i, got,
                    expected);
            ok = false;
            break;
        }
    }

    fprintf(ok ? stdout : stderr, "Test %s for %s rows=%ld cols=%ld\n", ok ? "PASSED" : "FAILED", DtypeName<T>(), rows,
            cols);
    return ok ? 0 : -1;
}

/**
 * TestStatic<T>: build a new graph per shape, run, remove.
 */
template <typename T>
static int TestStatic(ge::Session* session, uint32_t graph_id, int64_t rows, int64_t cols)
{
    ge::Graph graph("euclidean_norm_static");
    BuildGraph(graph, {rows, cols}, {rows}, {1}, DtypeOf<T>());
    std::map<ge::AscendString, ge::AscendString> options;
    CHECK_GE(session->AddGraph(graph_id, graph, options));
    int ret = RunAndVerify<T>(session, graph_id, rows, cols, "static");
    session->RemoveGraph(graph_id);
    return ret;
}

int main()
{
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    CHECK_GE(ge::GEInitialize(options));

    auto session = std::make_unique<ge::Session>(std::map<ge::AscendString, ge::AscendString>{});

    // Shape list: [rows, cols] — cols is reduced via axes={1}.
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1},    {1, 7},    {1, 16},   {4, 100},   {4, 1023},   {4, 1024},   {4, 1025},
        {4, 4096}, {4, 4097}, {4, 9973}, {8, 16384}, {8, 100000}, {4, 409600}, {4, 1048577},
    };

    int ret = 0;
    int passed = 0;
    uint32_t gid = 0;

    // ===== Static shape tests =====
    fprintf(stdout, "\n==== Static shape tests ====\n");
    for (auto [r, c] : shapes) {
        if (TestStatic<aclFloat16>(session.get(), gid++, r, c) == 0) {
            ++passed;
        } else {
            ret = -1;
        }
        if (TestStatic<float>(session.get(), gid++, r, c) == 0) {
            ++passed;
        } else {
            ret = -1;
        }
    }
    fprintf(stdout, "Static summary: %d/%zu cases PASSED\n", passed, shapes.size() * 2);

    // ===== Dynamic shape tests =====
    int dyn_passed = 0;
    fprintf(stdout, "\n==== Dynamic shape tests ====\n");

    // Dynamic fp16: one graph with shape {-1, -1}, reused for all shapes.
    {
        ge::Graph graph("euclidean_norm_dynamic_fp16");
        BuildGraph(graph, {-1, -1}, {-1}, {1}, ge::DT_FLOAT16);
        std::map<ge::AscendString, ge::AscendString> dynOpts;
        CHECK_GE(session->AddGraph(gid, graph, dynOpts));
        uint32_t dynId = gid++;
        for (auto [r, c] : shapes) {
            if (RunAndVerify<aclFloat16>(session.get(), dynId, r, c, "dynamic") == 0) {
                ++dyn_passed;
            } else {
                ret = -1;
            }
        }
        session->RemoveGraph(dynId);
    }

    // Dynamic fp32
    {
        ge::Graph graph("euclidean_norm_dynamic_fp32");
        BuildGraph(graph, {-1, -1}, {-1}, {1}, ge::DT_FLOAT);
        std::map<ge::AscendString, ge::AscendString> dynOpts;
        CHECK_GE(session->AddGraph(gid, graph, dynOpts));
        uint32_t dynId = gid++;
        for (auto [r, c] : shapes) {
            if (RunAndVerify<float>(session.get(), dynId, r, c, "dynamic") == 0) {
                ++dyn_passed;
            } else {
                ret = -1;
            }
        }
        session->RemoveGraph(dynId);
    }

    passed += dyn_passed;
    fprintf(stdout, "Dynamic summary: %d/%zu cases PASSED\n", dyn_passed, shapes.size() * 2);

    size_t total = shapes.size() * 2 * 2;
    fprintf(stdout, "\n==== Overall summary: %d/%zu cases PASSED ====\n", passed, total);

    ge::GEFinalize();
    return ret;
}
