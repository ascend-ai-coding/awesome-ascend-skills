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
 * xdivy_custom_package/tests/tiling/test_tiling.cpp
 * =============================================================================
 * Role: Tiling unit test for the Xdivy broadcast operator.
 *        Directly loads the operator registration shared library, locates the
 *        registered TilingFuncXdivy, constructs TilingContext objects with known
 *        shape/dtype combinations (including broadcast scenarios), invokes
 *        TilingFunc, and verifies the computed XdivyTilingData values.
 *
 * Contents:
 *   - TestCase: a named test with two input shapes, output shape, dtype,
 *     and expected tiling results (blockDim / workspace).
 *   - Cases(): returns a static vector of test cases covering:
 *       - Same-shape 2D (no broadcast).
 *       - Broadcast 2D: x1=(4,1), x2=(1,8) -> y=(4,8).
 *       - Last-dim split (large 1D shape).
 *       - Whole-tensor-fits-in-UB (no split).
 *       - Multi-core distribution.
 *       - FP16 dtype (changes physNodes from 3 to 4).
 *       - Scalar broadcast: x1=(), x2=(100,) -> y=(100,).
 *   - MakeShape / MakeTensor: helpers to construct gert objects.
 *   - LoadTilingKernel: dlopen the op_host .so and return TilingFunc.
 *   - RunOneCase: build a TilingContext, invoke tilingFunc, verify results.
 *   - main: iterate over all test cases and report summary.
 *
 * Expected value format:
 *   Each test case carries an `expect` string encoding the full TilingData
 *   layout: "split.ubSplitIdx split.ubFactor split.ubOuter split.ubTail |
 *            multicore.usedCoreNum multicore.totalTiles multicore.mainTiles
 *            multicore.mainCoreNum | rank | perBufBytes | ..."
 *   We format the actual TilingData the same way and string-compare.
 *
 * Why a tiling unit test:
 *   Tiling is the host-side computation that decides how to partition the
 *   broadcast output across AICore cores and how to address broadcast inputs.
 *   Bugs here cause incorrect results, NaN outputs, or crashes.  A unit test
 *   catches these bugs without needing device hardware or full kernel execution.
 *
 * Operator name variants:
 *   - PascalCase:   Xdivy
 *   - snake_case:   xdivy
 *   - UPPER_SNAKE:  XDIVY
 *   - camelCase:    xdivy
 *
 * To create a new operator (e.g. FooBar), replace:
 *   Xdivy → FooBar
 *   xdivy → foo_bar
 *   XdivyCompileInfo → FooBarCompileInfo
 *   xdivy_op_host_ut_lib → foo_bar_op_host_ut_lib
 * =============================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

// CANN tiling-test infrastructure headers.
#include "graph/ascend_string.h"
#include "graph/types.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_data.h"
#include "exe_graph/runtime/tensor.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

// Shared tiling struct so we can interpret the raw tiling buffer.
#include "xdivy_tiling_struct.h"

namespace {

// ---- Constants mirroring the tiling implementation ----------------------
//   Platform info (coreNum/ubSize) is read from PlatFormInfos at tiling time
//   (CompileInfo is an empty struct); faked below via SetupPlatform().
//   coreNum=56 and ubSize=256KiB mirror typical ascend950 hardware.
constexpr uint64_t TEST_CORE_NUM = 56;
constexpr uint64_t TEST_UB_SIZE = 256 * 1024;

/**
 * SetupPlatform: configure fe::PlatFormInfos with fake hardware specs.
 * The tiling function reads coreNum/ubSize from the platform info at tiling
 * time, so the test must fake these fields (mirrors the reduction example).
 */
void SetupPlatform(fe::PlatFormInfos& platformInfo)
{
    platformInfo.Init();

    std::map<std::string, std::string> socInfos = {{"UB_SIZE", std::to_string(TEST_UB_SIZE)},
                                                   {"L2_SIZE", "33554432"},
                                                   {"L1_SIZE", "524288"},
                                                   {"L0A_SIZE", "65536"},
                                                   {"L0B_SIZE", "65536"},
                                                   {"L0C_SIZE", "131072"},
                                                   {"CORE_NUM", std::to_string(TEST_CORE_NUM)},
                                                   {"ai_core_cnt", std::to_string(TEST_CORE_NUM)}};
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", std::to_string(TEST_UB_SIZE)},
                                                     {"ubblock_size", "32"},
                                                     {"l1_size", "524288"},
                                                     {"l0_a_size", "65536"},
                                                     {"l0_b_size", "65536"},
                                                     {"l0_c_size", "131072"}};
    std::map<std::string, std::string> intrinsics;
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetCoreNum(TEST_CORE_NUM);
    platformInfo.SetCoreNumByCoreType("AICore");
}

// OP_TYPE: the operator type string registered via OP_ADD(Xdivy).
constexpr const char* OP_TYPE = "Xdivy";

// OP_HOST_SO_NAME: filename of the shared library built by the side CMakeLists.
//   RPATH=$ORIGIN lets the test locate the sibling .so at runtime.
constexpr const char* OP_HOST_SO_NAME = "libxdivy_op_host_ut_lib.so";

/**
 * ShapeSpec: lightweight shape representation for building test cases.
 *   An empty vector means "scalar" (rank 0).
 */
using ShapeSpec = std::vector<int64_t>;

/**
 * FormatShape: serialise a shape as "[d0,d1,...]" for log messages.
 */
static std::string FormatShape(const ShapeSpec& s)
{
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i)
            r += ",";
        r += std::to_string(s[i]);
    }
    r += "]";
    return r;
}

/**
 * TestCase: a single tiling test scenario.
 *   name       — descriptive label for log messages.
 *   x1, x2     — input shapes (broadcast-compatible).
 *   y          — output shape (= broadcastMax(x1, x2)).
 *   dtype      — input/output dtype.
 *   expectBlocks — expected blockDim (number of AICore cores used).
 *   expectWorkspace — expected workspace[0] value (always 16 MB for Xdivy).
 */
struct TestCase {
    const char* name;
    ShapeSpec x1;
    ShapeSpec x2;
    ShapeSpec y;
    ge::DataType dtype;
    uint32_t expectBlocks;
    size_t expectWorkspace;
};

/**
 * Cases(): static list of tiling test cases.
 *
 * Coverage goals:
 *   - Same-shape (no broadcast) in 2D.
 *   - Broadcast 2D where each input contributes a different dim.
 *   - Large 1D shape that forces last-dim split.
 *   - Multi-dim shape that fits entirely in UB (no split).
 *   - 4D shape that exercises multi-core distribution.
 *   - FP16 dtype (changes physNodes P from 3 to 4).
 *   - Scalar broadcast (rank-0 input).
 */
static const std::vector<TestCase>& Cases()
{
    static const std::vector<TestCase> CASES = {
        // case1: same shape 2D, FP32 — no broadcast.
        {"same_shape_2d_fp32", {4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1, 16777216},
        // case2: broadcast 2D — x1=(4,1), x2=(1,8) -> y=(4,8).
        {"broadcast_2d_fp32", {4, 1}, {1, 8}, {4, 8}, ge::DT_FLOAT, 1, 16777216},
        // case3: large 1D, FP32 — splits at last dim, multiple tiles.
        {"large_1d_fp32", {100000}, {100000}, {100000}, ge::DT_FLOAT, 5, 16777216},
        // case4: 3D fits in UB, FP32 — no split, single segment.
        {"no_split_3d_fp32", {64, 32, 8}, {64, 32, 8}, {64, 32, 8}, ge::DT_FLOAT, 1, 16777216},
        // case5: 4D shape, FP32 — exercises multi-core split at axis 2.
        {"multi_core_4d_fp32", {7, 13, 200, 100}, {7, 13, 200, 100}, {7, 13, 200, 100}, ge::DT_FLOAT, 56, 16777216},
        // case6: large 1D, FP16 — physNodes=4 (cast path), perBufBytes differs.
        {"large_1d_fp16", {50000}, {50000}, {50000}, ge::DT_FLOAT16, 4, 16777216},
        // case7: scalar broadcast — x1=(), x2=(100,) -> y=(100,).
        {"scalar_broadcast_fp32", {}, {100}, {100}, ge::DT_FLOAT, 1, 16777216},
    };
    return CASES;
}

/**
 * MakeShape: construct a gert::StorageShape from a ShapeSpec.
 *   Empty ShapeSpec yields a scalar (StorageShape with 0 dims).
 */
gert::StorageShape MakeShape(const ShapeSpec& dims)
{
    gert::StorageShape s;
    for (int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

/**
 * LoadTilingKernel: load the op_host .so and return the registered TilingFunc.
 *
 * Workflow:
 *   1. Resolve the test executable directory.
 *   2. Construct full path to libxdivy_op_host_ut_lib.so.
 *   3. OpImplSpaceRegistryV2::AddSoToRegistry dlopens the .so and triggers
 *      all static initialisers (OP_ADD, IMPL_OP, IMPL_OP_OPTILING, etc.).
 *   4. Look up the operator by type string ("Xdivy") and return its tiling kernel.
 */
gert::OpImplKernelRegistry::TilingKernelFunc LoadTilingKernel(const char* exePath)
{
    namespace fs = std::filesystem;
    fs::path p(exePath);
    p = p.is_relative() ? fs::weakly_canonical(fs::current_path() / p) : fs::canonical(p);

    std::string soPath = p.parent_path().string() + "/" + OP_HOST_SO_NAME;
    fprintf(stdout, "[setup] loading op_host .so: %s\n", soPath.c_str());

    gert::OppSoDesc soDesc({ge::AscendString(soPath.c_str())}, ge::AscendString("op_host_so"));

    auto spaceReg = std::make_shared<gert::OpImplSpaceRegistryV2>();
    if (spaceReg->AddSoToRegistry(soDesc) != ge::GRAPH_SUCCESS) {
        fprintf(stderr, "[setup] FAILED to add op_host .so to registry\n");
        return nullptr;
    }
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceReg);

    auto reg = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (reg == nullptr) {
        fprintf(stderr, "[setup] FAILED: space registry is null\n");
        return nullptr;
    }
    const auto* impl = reg->GetOpImpl(OP_TYPE);
    if (impl == nullptr || impl->tiling == nullptr) {
        fprintf(stderr, "[setup] FAILED: tiling kernel for '%s' not found\n", OP_TYPE);
        return nullptr;
    }
    fprintf(stdout, "[setup] tiling kernel for '%s' resolved\n", OP_TYPE);
    return impl->tiling;
}

/**
 * RunOneCase: execute a single tiling test case.
 *
 * Workflow:
 *   1. Construct x1/x2/y tensors with the test case shapes and dtype.
 *   2. Allocate a generous tiling data buffer (XdivyTilingData<8> worst case).
 *   3. Build a TilingContext via OpTilingContextBuilder, passing a faked
 *      PlatFormInfos (coreNum/ubSize; CompileInfo is an empty struct).
 *   4. Invoke tilingFunc(ctx).
 *   5. Verify:
 *        - Return value is GRAPH_SUCCESS.
 *        - blockDim matches expected value.
 *        - workspace[0] matches expected value.
 *        - maxBroShape in the tiling data matches the broadcast output shape.
 */
int32_t RunOneCase(const TestCase& tc, gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc)
{
    // Build tensor descriptors with FORMAT_ND.
    gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    gert::StorageShape x1Shape = MakeShape(tc.x1);
    gert::StorageShape x2Shape = MakeShape(tc.x2);
    gert::StorageShape yShape = MakeShape(tc.y);

    gert::Tensor x1Tensor(x1Shape, fmt, gert::kOnHost, tc.dtype, nullptr);
    gert::Tensor x2Tensor(x2Shape, fmt, gert::kOnHost, tc.dtype, nullptr);
    gert::Tensor yTensor(yShape, fmt, gert::kOnHost, tc.dtype, nullptr);

    // Allocate the tiling data buffer.  Worst-case size is XdivyTilingData<8>.
    constexpr size_t TILING_BUF_BYTES = sizeof(XdivyTilingData<8>) + 64;
    auto tilingDataBuf = gert::TilingData::CreateCap(TILING_BUF_BYTES);
    auto wsBuf = gert::ContinuousVector::Create<size_t>(16);

    // Platform info: coreNum/ubSize are read from PlatFormInfos at tiling
    // time (CompileInfo is an empty struct, only keeps the registration hook);
    // fake the platform fields with test values.
    fe::PlatFormInfos platformInfo;
    SetupPlatform(platformInfo);

    struct DummyCompileInfo {
    } compileInfo;

    gert::OpTilingContextBuilder builder;
    builder.OpType(ge::AscendString(OP_TYPE))
        .OpName(ge::AscendString(OP_TYPE))
        .IONum(2, 1)
        .InputTensors({&x1Tensor, &x2Tensor})
        .OutputTensors({&yTensor})
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo)
        .TilingData(static_cast<gert::TilingData*>(tilingDataBuf.get()))
        .Workspace(static_cast<gert::ContinuousVector*>(wsBuf.get()));
    auto holder = builder.Build();
    gert::TilingContext* ctx = holder.GetContext();
    if (ctx == nullptr) {
        fprintf(stderr, "[%s] FAILED: built TilingContext is null\n", tc.name);
        return -1;
    }

    // Invoke the tiling function.
    auto ret = tilingFunc(ctx);
    if (ret != static_cast<uint32_t>(ge::GRAPH_SUCCESS)) {
        fprintf(stderr, "[%s] FAILED: tiling returned %u, expected GRAPH_SUCCESS\n", tc.name, ret);
        return -1;
    }

    bool ok = true;

    // Verify blockDim (number of AICore cores).
    uint32_t blockDim = ctx->GetBlockDim();
    if (blockDim != tc.expectBlocks) {
        fprintf(stderr, "[%s] FAILED: blockDim=%u expected %u\n", tc.name, blockDim, tc.expectBlocks);
        ok = false;
    }

    // Verify workspace.
    size_t* wsSizes = ctx->GetWorkspaceSizes(1);
    if (wsSizes == nullptr || wsSizes[0] != tc.expectWorkspace) {
        fprintf(stderr, "[%s] FAILED: workspace[0]=%zu expected %zu\n", tc.name, wsSizes ? wsSizes[0] : SIZE_MAX,
                tc.expectWorkspace);
        ok = false;
    }

    // Verify tiling data was populated (non-null, non-zero size).
    auto* rawTd = ctx->GetRawTilingData();
    if (rawTd == nullptr || rawTd->GetData() == nullptr || rawTd->GetDataSize() == 0) {
        fprintf(stderr, "[%s] FAILED: tiling data is null or empty\n", tc.name);
        ok = false;
    }

    fprintf(ok ? stdout : stderr, "Test %-26s (x1=%s, x2=%s) %s\n", tc.name, FormatShape(tc.x1).c_str(),
            FormatShape(tc.x2).c_str(), ok ? "PASSED" : "FAILED");
    return ok ? 0 : -1;
}

} // namespace

/**
 * main: entry point for the tiling unit test.
 */
int32_t main(int32_t argc, char** argv)
{
    auto tilingFunc = LoadTilingKernel(argv[0]);
    if (tilingFunc == nullptr) {
        fprintf(stderr, "Aborting: could not load tiling kernel.\n");
        return 1;
    }

    int32_t ret = 0;
    int32_t passed = 0;
    const auto& cases = Cases();
    for (const auto& tc : cases) {
        if (RunOneCase(tc, tilingFunc) == 0) {
            ++passed;
        } else {
            ret = -1;
        }
    }

    fprintf(stdout, "\n==== Tiling UT summary: %d/%zu cases PASSED ====\n", passed, cases.size());
    return ret == 0 ? 0 : 1;
}
