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
 * xdivy_custom_package/tests/geir/test_geir.cpp
 * =============================================================================
 * Role: GE (Graph Engine) IR integration test for the Xdivy operator.
 *        Tests that Xdivy can be used as a graph node in GE's compilation and
 *        execution pipeline, including broadcast scenarios.
 *
 * Contents:
 *   - CHECK_GE macro: error-checking wrapper for GE API calls.
 *   - FromFloat<T> / ToFloat<T>: fp16 <-> float conversion helpers.
 *   - DtypeOf<T> / DtypeName<T>: runtime dtype helpers for GE types.
 *   - BuildGraph: constructs a GE graph with Data(x1), Data(x2) -> Xdivy -> y.
 *   - MakeFeedTensor: creates host-side input tensors with deterministic data.
 *   - RunAndVerify: executes a graph via ge::Session::RunGraph and checks results.
 *   - Golden: computes (x1==0) ? 0 : (x1/x2) with broadcast on the CPU.
 *   - TestStatic:  tests static shapes (shape known at compile time).
 *   - TestDynamic: tests dynamic shapes (shape = -1, set at runtime).
 *   - main: initialises GE, runs static and dynamic tests for fp16 and fp32.
 *
 * Broadcast verification:
 *   Unlike add_custom (same-shape only), this test exercises broadcast:
 *     - x1 shape (M, K), x2 shape (K) -> y shape (M, K).
 *     - x1 shape (1, K), x2 shape (M, 1) -> y shape (M, K).
 *   The Golden function applies broadcast on the CPU using the same rules.
 *
 * Operator name variants:
 *   - PascalCase:   Xdivy
 *   - snake_case:   xdivy
 *   - UPPER_SNAKE:  XDIVY
 *   - camelCase:    xdivy
 *
 * To create a new operator (e.g. FooBar), replace:
 *   Xdivy -> FooBar
 *   xdivy -> foo_bar
 *   op::Xdivy -> op::FooBar
 *   op_proto.h -> (stays same; content changes)
 * =============================================================================
 */

#include <cstdio>
#include <cmath>
#include <type_traits>
#include <vector>
#include <map>

// acl.h: provides aclFloat16, aclFloatToFloat16, aclFloat16ToFloat.
#include "acl/acl.h"
// GE graph construction headers.
#include "graph.h"
#include "types.h"
#include "tensor.h"
// GE session API for graph compilation/execution.
#include "ge_error_codes.h"
#include "ge_api.h"
// Built-in operator protos (op::Data).
#include "array_ops.h"
// INSTALLED vendor operator proto header (contains REG_OP(Xdivy)).
#include "op_proto.h"

using namespace ge;

/**
 * CHECK_GE(x): wrap a GE API call and check its return status.
 */
#define CHECK_GE(x) \
    do { \
        ge::Status __ret = (x); \
        if (__ret != ge::SUCCESS) { \
            fprintf(stderr, "GE error %u at %s:%d: %s\n", __ret, __FILE__, __LINE__, #x); \
            return -1; \
        } \
    } while (0)

/** FromFloat<T>: convert a C++ float to the tensor element type T. */
template <typename T>
static T FromFloat(float v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloatToFloat16(v);
    } else {
        return v;
    }
}

/** ToFloat<T>: convert a tensor element of type T to C++ float. */
template <typename T>
static float ToFloat(T v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloat16ToFloat(v);
    } else {
        return v;
    }
}

/** DtypeOf<T>: maps the C++ template type T to a GE DataType enum value. */
template <typename T>
static ge::DataType DtypeOf()
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        return ge::DT_FLOAT16;
    else
        return ge::DT_FLOAT;
}

/** DtypeName<T>: human-readable name for log messages. */
template <typename T>
static const char* DtypeName()
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        return "fp16";
    else
        return "fp32";
}

/**
 * BroadcastInfo: describes a broadcast test scenario.
 *   x1Shape, x2Shape — input shapes (must be broadcast-compatible).
 *   yShape            — expected output shape = broadcastMax(x1, x2).
 *
 * The shapes can be multi-dimensional; the test feeds row-major data and
 * computes the broadcast golden on the CPU.
 */
struct BroadcastInfo {
    std::vector<int64_t> x1Shape;
    std::vector<int64_t> x2Shape;
    std::vector<int64_t> yShape;
};

/**
 * ShapeSize: total element count = product of all dims.
 */
static int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s)
        n *= d;
    return n;
}

/**
 * FlatIndexToCoord: convert a flat row-major index to a multi-dim coordinate.
 *   Used by the CPU golden to look up broadcasted input values.
 */
static std::vector<int64_t> FlatIndexToCoord(int64_t flat, const std::vector<int64_t>& shape)
{
    std::vector<int64_t> coord(shape.size());
    for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
        coord[i] = flat % shape[i];
        flat /= shape[i];
    }
    return coord;
}

/**
 * BroadcastLookup: look up the value at output-coord `coord` in an input
 *   tensor whose shape may differ from the output shape (broadcast rules).
 *
 *   For each dim, if the input's dim size is 1 but the output's is N>1, we
 *   read input index 0 (broadcast).  Otherwise we read the same index.
 */
template <typename T>
static float BroadcastLookup(const std::vector<T>& data, const std::vector<int64_t>& inShape,
                             const std::vector<int64_t>& outShape, const std::vector<int64_t>& coord)
{
    // Right-align: input dim at position (i + offset) corresponds to out dim i.
    int64_t inRank = (int64_t)inShape.size();
    int64_t outRank = (int64_t)outShape.size();
    int64_t offset = outRank - inRank;

    int64_t inFlat = 0;
    int64_t stride = 1;
    for (int64_t i = inRank - 1; i >= 0; --i) {
        int64_t outD = i + offset;
        int64_t c = (outD >= 0) ? coord[outD] : 0;
        // Broadcast: if input dim is 1, always read index 0.
        if (inShape[i] == 1)
            c = 0;
        inFlat += c * stride;
        stride *= inShape[i];
    }
    return ToFloat<T>(data[inFlat]);
}

/**
 * BuildGraph: construct a GE graph with one Xdivy node.
 *
 * Graph topology:
 *   Data("x1", index=0) ---\
 *                           +---> Xdivy("xdivy").inputX1
 *   Data("x2", index=1) ---+---> Xdivy("xdivy").inputX2
 *                                |
 *                                +---> output (y)
 *
 * The graph has 2 inputs (x1, x2 feeds) and 1 output (Xdivy result).
 */
static int32_t BuildGraph(ge::Graph& graph, const BroadcastInfo& bc, ge::DataType dtype)
{
    auto xdivy_op = op::Xdivy("xdivy");

    // Helper to create a Data feed node with the given shape.
    auto makeData = [&](const char* name, int64_t index, const std::vector<int64_t>& shape) {
        ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
        desc.SetRealDimCnt(static_cast<int64_t>(shape.size()));
        auto data = op::Data(name).set_attr_index(index);
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        graph.AddOp(data);
        return data;
    };

    auto dataX1 = makeData("x1", 0, bc.x1Shape);
    auto dataX2 = makeData("x2", 1, bc.x2Shape);

    // Connect Data outputs to Xdivy inputs.
    xdivy_op.set_input_x1(dataX1);
    xdivy_op.set_input_x2(dataX2);

    // Output descriptor uses the broadcast output shape.
    ge::TensorDesc outDesc(ge::Shape(bc.yShape), ge::FORMAT_ND, dtype);
    outDesc.SetRealDimCnt(static_cast<int64_t>(bc.yShape.size()));
    xdivy_op.update_output_desc_y(outDesc);

    graph.SetInputs({dataX1, dataX2}).SetOutputs({xdivy_op});
    return 0;
}

/**
 * MakeFeedTensor<T>: create a host-side GE tensor filled with deterministic data.
 *
 *   x1[i] = ((i * 7) % 13) * 0.5f         — includes a zero at index 0, 13, 26, ...
 *   x2[i] = ((i + 1) % 11 + 1) * 1.5f     — never zero (so x1==0 is the only short-circuit)
 *
 * The zero entries in x1 exercise the (x1==0) ? 0 : x1/x2 path.
 */
template <typename T>
static ge::Tensor MakeFeedTensor(const std::vector<int64_t>& shape, bool isX1)
{
    int64_t n = ShapeSize(shape);
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, DtypeOf<T>());
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(static_cast<int64_t>(shape.size()));

    size_t num = static_cast<size_t>(n);
    std::vector<T> buf(num);
    for (size_t i = 0; i < num; ++i) {
        float v = isX1 ? static_cast<float>((i * 7) % 13) * 0.5f : static_cast<float>((i + 1) % 11 + 1) * 1.5f;
        buf[i] = FromFloat<T>(v);
    }
    return ge::Tensor(desc, (uint8_t*)buf.data(), num * sizeof(T));
}

/**
 * RunAndVerify<T>: execute a pre-added graph and verify output correctness.
 *
 *   Builds CPU-side x1/x2 feeds, runs the graph, then computes the broadcast
 *   golden on the CPU and compares element-by-element with tolerance 1e-3.
 */
template <typename T>
static int32_t RunAndVerify(ge::Session* session, uint32_t graphId, const BroadcastInfo& bc, const char* mode)
{
    std::vector<ge::Tensor> feeds;
    feeds.push_back(MakeFeedTensor<T>(bc.x1Shape, true));  // x1 feed
    feeds.push_back(MakeFeedTensor<T>(bc.x2Shape, false)); // x2 feed

    std::vector<ge::Tensor> output;
    CHECK_GE(session->RunGraph(graphId, feeds, output));

    if (output.size() != 1) {
        fprintf(stderr, "[%s/%s] Expected 1 output, got %zu\n", mode, DtypeName<T>(), output.size());
        return -1;
    }

    // Verify output shape matches broadcast result.
    auto outShape = output[0].GetTensorDesc().GetShape();
    if (static_cast<size_t>(outShape.GetDimNum()) != bc.yShape.size()) {
        fprintf(stderr, "[%s/%s] Output rank mismatch: got %zu, expected %zu\n", mode, DtypeName<T>(),
                outShape.GetDimNum(), bc.yShape.size());
        return -1;
    }
    for (size_t d = 0; d < bc.yShape.size(); ++d) {
        if (outShape.GetDim(d) != bc.yShape[d]) {
            fprintf(stderr, "[%s/%s] Output dim %zu mismatch: got %ld, expected %ld\n", mode, DtypeName<T>(), d,
                    outShape.GetDim(d), bc.yShape[d]);
            return -1;
        }
    }

    // CPU-side feeds for golden computation.
    int64_t outN = ShapeSize(bc.yShape);
    size_t num = static_cast<size_t>(outN);
    const T* outData = (const T*)output[0].GetData();

    // Reconstruct input data (must match MakeFeedTensor's formulas).
    std::vector<T> x1Host(ShapeSize(bc.x1Shape));
    std::vector<T> x2Host(ShapeSize(bc.x2Shape));
    for (size_t i = 0; i < x1Host.size(); ++i) {
        x1Host[i] = FromFloat<T>(static_cast<float>((i * 7) % 13) * 0.5f);
    }
    for (size_t i = 0; i < x2Host.size(); ++i) {
        x2Host[i] = FromFloat<T>(static_cast<float>((i + 1) % 11 + 1) * 1.5f);
    }

    // Compute golden with broadcast.
    bool ok = true;
    for (size_t i = 0; i < num; ++i) {
        auto coord = FlatIndexToCoord(static_cast<int64_t>(i), bc.yShape);
        float x1v = BroadcastLookup<T>(x1Host, bc.x1Shape, bc.yShape, coord);
        float x2v = BroadcastLookup<T>(x2Host, bc.x2Shape, bc.yShape, coord);
        float expected = (x1v == 0.0f) ? 0.0f : x1v / x2v;
        float got = ToFloat<T>(outData[i]);
        if (std::fabs(got - expected) > 1e-3f) {
            fprintf(stderr, "[%s/%s] Mismatch at [%zu]: got %f, expected %f (x1=%f x2=%f)\n", mode, DtypeName<T>(), i,
                    got, expected, x1v, x2v);
            ok = false;
            break;
        }
    }

    fprintf(ok ? stdout : stderr, "Test %s for %s shape=%s\n", ok ? "PASSED" : "FAILED", DtypeName<T>(),
            DtypeName<T>());
    return ok ? 0 : -1;
}

/**
 * TestStatic<T>: test static-shape execution for a given broadcast scenario.
 */
template <typename T>
static int32_t TestStatic(ge::Session* session, uint32_t graphId, const BroadcastInfo& bc)
{
    ge::Graph graph("xdivy_static");
    BuildGraph(graph, bc, DtypeOf<T>());
    std::map<ge::AscendString, ge::AscendString> options;
    CHECK_GE(session->AddGraph(graphId, graph, options));
    int32_t ret = RunAndVerify<T>(session, graphId, bc, "static");
    session->RemoveGraph(graphId);
    return ret;
}

/**
 * main: entry point for the GE IR integration test.
 *
 * Workflow:
 *   1. GEInitialize with device 0.
 *   2. Create a ge::Session.
 *   3. Static shape tests: for each broadcast scenario, build a new graph per
 *      shape and dtype, run it, remove the graph.
 *   4. Print summary.
 */
int32_t main()
{
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    CHECK_GE(ge::GEInitialize(options));

    ge::Session* session = new ge::Session(std::map<ge::AscendString, ge::AscendString>{});
    if (session == nullptr) {
        fprintf(stderr, "Failed to create GE session\n");
        ge::GEFinalize();
        return -1;
    }

    // Broadcast scenarios to test:
    //   - same_shape:        no broadcast, 2D.
    //   - row_broadcast:     x1=(M,K), x2=(1,K) -> y=(M,K).
    //   - col_broadcast:     x1=(M,1), x2=(1,K) -> y=(M,K) (full broadcast).
    //   - vector_scalar:     x1=(), x2=(N,) -> y=(N,) (scalar dividend).
    const std::vector<BroadcastInfo> scenarios = {
        {{4, 8}, {4, 8}, {4, 8}},       // same_shape
        {{4, 8}, {1, 8}, {4, 8}},       // row_broadcast
        {{4, 1}, {1, 8}, {4, 8}},       // col_broadcast
        {{}, {16}, {16}},               // vector_scalar (scalar x1)
        {{16}, {16}, {16}},             // 1d same shape
        {{2, 3, 4}, {3, 4}, {2, 3, 4}}, // 3d with 2d
    };

    int32_t ret = 0;
    int32_t passed = 0;
    uint32_t gid = 0;

    fprintf(stdout, "\n==== Static broadcast tests ====\n");
    for (const auto& bc : scenarios) {
        if (TestStatic<aclFloat16>(session, gid++, bc) == 0)
            ++passed;
        else
            ret = -1;
        if (TestStatic<float>(session, gid++, bc) == 0)
            ++passed;
        else
            ret = -1;
    }
    fprintf(stdout, "Static summary: %d/%zu cases PASSED\n", passed, scenarios.size() * 2);

    delete session;
    ge::GEFinalize();
    return ret;
}
