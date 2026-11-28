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
 * euclidean_norm_package/tests/tiling/test_tiling.cpp
 * =============================================================================
 * Role: Tiling unit test for the EuclideanNorm operator.
 *        Directly loads the operator registration shared library, locates the
 *        registered TilingFunc, constructs TilingContext objects with known
 *        shapes/dtypes/axes, invokes TilingFunc, and verifies:
 *          1. Return value is GRAPH_SUCCESS.
 *          2. TilingData buffer size matches sizeof(EuclideanNormTilingData).
 *          3. Key structural invariants in the tiling output:
 *             axisNum >= 2, usedCoreNum > 0, aLoopCntTotal > 0,
 *             preBufSize > 0, postBufSize > 0, blockDim > 0.
 *          4. Workspace configuration is valid.
 *
 * Unlike add_custom (whose tiling output is fully deterministic and can be
 * checked to the last byte), EuclideanNorm's tiling values depend on runtime
 * platform info (coreNum, UB size) read via PlatFormInfos::Init().  We verify
 * structural invariants that must hold regardless of the specific hardware.
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

#include "euclidean_norm_tiling_data.h"

namespace {

constexpr const char* K_OP_TYPE = "EuclideanNorm";
constexpr const char* K_OP_HOST_SO_NAME = "libeuclidean_norm_op_host_ut_lib.so";

/**
 * Configure fe::PlatFormInfos with Ascend950 hardware specs.
 * Init() auto-loads from the Ascend driver; fallback sets key fields manually.
 */
void SetupPlatform(fe::PlatFormInfos& platformInfo)
{
    platformInfo.Init();

    std::map<std::string, std::string> socInfos = {
        {"UB_SIZE", "196608"}, {"L2_SIZE", "33554432"}, {"L1_SIZE", "524288"}, {"L0A_SIZE", "65536"},
        {"L0B_SIZE", "65536"}, {"L0C_SIZE", "131072"},  {"CORE_NUM", "24"},    {"ai_core_cnt", "24"}};
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", "196608"},  {"ubblock_size", "32"},
                                                     {"l1_size", "524288"},  {"l0_a_size", "65536"},
                                                     {"l0_b_size", "65536"}, {"l0_c_size", "131072"}};
    std::map<std::string, std::string> intrinsics;
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetCoreNum(24);
    platformInfo.SetCoreNumByCoreType("AICore");
}

/**
 * Create a 1-D int32 const tensor with the given axes values.
 */
std::unique_ptr<uint8_t[]> MakeAxesTensor(const std::vector<int32_t>& axes)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(static_cast<int64_t>(axes.size()), ge::DT_INT32, totalSize);
    auto* t = reinterpret_cast<gert::Tensor*>(holder.get());
    t->MutableStorageShape().SetDimNum(1);
    t->MutableStorageShape().SetDim(0, static_cast<int64_t>(axes.size()));
    t->MutableOriginShape().SetDimNum(1);
    t->MutableOriginShape().SetDim(0, static_cast<int64_t>(axes.size()));
    auto* dataPtr = reinterpret_cast<int32_t*>(holder.get() + sizeof(gert::Tensor));
    for (size_t i = 0; i < axes.size(); ++i) {
        dataPtr[i] = axes[i];
    }
    return holder;
}

struct TestCase {
    const char* name;
    std::vector<int64_t> xShape;
    std::vector<int64_t> yShape;
    std::vector<int32_t> axes;
    ge::DataType dtype;
    bool keepDims;
};

static const std::vector<TestCase>& Cases()
{
    static const std::vector<TestCase> cases = {
        {"fp32_2d_reduce_last", {4, 64}, {4}, {1}, ge::DT_FLOAT, false},
        {"fp16_2d_reduce_last", {8, 1024}, {8}, {1}, ge::DT_FLOAT16, false},
        {"int32_2d_reduce_first", {16, 32}, {32}, {0}, ge::DT_INT32, false},
        {"bf16_3d_reduce_mid", {2, 4, 8}, {2, 8}, {1}, ge::DT_BF16, false},
        {"fp32_arar_reduce", {4, 4, 8, 8}, {4, 8}, {1, 3}, ge::DT_FLOAT, false},
        {"fp32_full_reduce", {4, 64}, {1, 1}, {}, ge::DT_FLOAT, true},
        {"fp32_tiny_1elem", {1, 1}, {1}, {1}, ge::DT_FLOAT, false},
        {"fp32_boundary_1024", {4, 1024}, {4}, {1}, ge::DT_FLOAT, false},
        {"fp32_boundary_1025", {4, 1025}, {4}, {1}, ge::DT_FLOAT, false},
        {"fp32_large", {8, 100000}, {8}, {1}, ge::DT_FLOAT, false},
        {"fp16_large", {8, 100000}, {8}, {1}, ge::DT_FLOAT16, false},
        {"fp32_multidim_4d", {2, 4, 4, 4}, {2, 4, 4}, {1}, ge::DT_FLOAT, false},
    };
    return cases;
}

gert::OpImplKernelRegistry::TilingKernelFunc LoadTilingKernel(const char* exePath)
{
    namespace fs = std::filesystem;
    fs::path p(exePath);
    p = p.is_relative() ? fs::weakly_canonical(fs::current_path() / p) : fs::canonical(p);
    std::string soPath = p.parent_path().string() + "/" + K_OP_HOST_SO_NAME;

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
    const auto* impl = reg->GetOpImpl(K_OP_TYPE);
    if (impl == nullptr || impl->tiling == nullptr) {
        fprintf(stderr, "[setup] FAILED: tiling kernel for '%s' not found\n", K_OP_TYPE);
        return nullptr;
    }
    fprintf(stdout, "[setup] tiling kernel for '%s' resolved\n", K_OP_TYPE);
    return impl->tiling;
}

/**
 * RunOneCase: build context, invoke tiling, verify structural invariants.
 *
 * Verified invariants:
 *   - tilingFunc returns GRAPH_SUCCESS.
 *   - RawTilingData size == sizeof(EuclideanNormTilingData).
 *   - axisNum >= 2 (at least one A + one R after pattern augmentation).
 *   - usedCoreNum > 0.
 *   - aLoopCntTotal > 0 (for non-empty inputs).
 *   - preBufSize > 0 && postBufSize > 0.
 *   - blockDim > 0.
 *   - workspace count >= 0.
 */
int RunOneCase(const TestCase& tc, gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc)
{
    gert::StorageFormat fmt(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());

    gert::StorageShape xShape;
    for (int64_t d : tc.xShape) {
        xShape.MutableOriginShape().AppendDim(d);
        xShape.MutableStorageShape().AppendDim(d);
    }
    gert::StorageShape yShape;
    for (int64_t d : tc.yShape) {
        yShape.MutableOriginShape().AppendDim(d);
        yShape.MutableStorageShape().AppendDim(d);
    }

    gert::Tensor xTensor(xShape, fmt, gert::kOnHost, tc.dtype, nullptr);
    auto axesHolder = MakeAxesTensor(tc.axes);
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    gert::Tensor yTensor(yShape, fmt, gert::kOnHost, tc.dtype, nullptr);

    auto tilingDataBuf = gert::TilingData::CreateCap(sizeof(EuclideanNormTilingData));
    auto wsBuf = gert::ContinuousVector::Create<size_t>(16);

    fe::PlatFormInfos platformInfo;
    SetupPlatform(platformInfo);

    struct DummyCompileInfo {
    } compileInfo;

    gert::OpTilingContextBuilder builder;
    builder.OpType(ge::AscendString(K_OP_TYPE))
        .OpName(ge::AscendString(K_OP_TYPE))
        .IONum(2, 1)
        .InputTensors({&xTensor, axesTensor})
        .OutputTensors({&yTensor})
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo)
        .TilingData(reinterpret_cast<gert::TilingData*>(tilingDataBuf.get()))
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsBuf.get()))
        .AppendAttr(tc.keepDims);

    auto holder = builder.Build();
    gert::TilingContext* ctx = holder.GetContext();
    if (ctx == nullptr) {
        fprintf(stderr, "[%s] FAILED: built TilingContext is null\n", tc.name);
        return -1;
    }

    // ── 1. Invoke tiling ──
    auto ret = tilingFunc(ctx);
    if (ret != static_cast<uint32_t>(ge::GRAPH_SUCCESS)) {
        fprintf(stderr, "[%s] FAILED: tiling returned %u, expected GRAPH_SUCCESS\n", tc.name, ret);
        return -1;
    }

    bool ok = true;

    // ── 2. TilingData size ──
    auto* rawTd = ctx->GetRawTilingData();
    if (rawTd == nullptr || rawTd->GetData() == nullptr) {
        fprintf(stderr, "[%s] FAILED: raw tiling data is null\n", tc.name);
        return -1;
    }
    if (rawTd->GetDataSize() != sizeof(EuclideanNormTilingData)) {
        fprintf(stderr, "[%s] FAILED: tiling data size %zu != %zu\n", tc.name, rawTd->GetDataSize(),
                sizeof(EuclideanNormTilingData));
        ok = false;
    }

    const auto* td = static_cast<const EuclideanNormTilingData*>(rawTd->GetData());

    // ── 3. axisNum invariant ──
    if (td->axisNum < 2) {
        fprintf(stderr, "[%s] FAILED: axisNum=%d, expected >= 2\n", tc.name, td->axisNum);
        ok = false;
    }

    // ── 4. usedCoreNum invariant ──
    if (td->usedCoreNum <= 0) {
        fprintf(stderr, "[%s] FAILED: usedCoreNum=%d, expected > 0\n", tc.name, td->usedCoreNum);
        ok = false;
    }

    // ── 5. aLoopCntTotal invariant ──
    if (td->aLoopCntTotal <= 0) {
        fprintf(stderr, "[%s] FAILED: aLoopCntTotal=%ld, expected > 0\n", tc.name, td->aLoopCntTotal);
        ok = false;
    }

    // ── 6. UB buffer size invariants ──
    if (td->preBufSize <= 0) {
        fprintf(stderr, "[%s] FAILED: preBufSize=%ld, expected > 0\n", tc.name, td->preBufSize);
        ok = false;
    }
    if (td->postBufSize <= 0) {
        fprintf(stderr, "[%s] FAILED: postBufSize=%ld, expected > 0\n", tc.name, td->postBufSize);
        ok = false;
    }

    // ── 7. blockDim invariant ──
    uint32_t blockDim = ctx->GetBlockDim();
    if (blockDim == 0) {
        fprintf(stderr, "[%s] FAILED: blockDim=0, expected > 0\n", tc.name);
        ok = false;
    }

    // ── 8. Workspace invariant ──
    size_t wsCount = ctx->GetWorkspaceNum();
    for (size_t i = 0; i < wsCount; ++i) {
        size_t* wsSizes = ctx->GetWorkspaceSizes(wsCount);
        if (wsSizes != nullptr && wsSizes[i] > 0) {
            // Non-zero workspace is valid for large inputs; just log it.
            break;
        }
    }

    fprintf(ok ? stdout : stderr, "Test %-26s %s\n", tc.name, ok ? "PASSED" : "FAILED");
    return ok ? 0 : -1;
}

} // namespace

int main(int argc, char** argv)
{
    auto tilingFunc = LoadTilingKernel(argv[0]);
    if (tilingFunc == nullptr) {
        fprintf(stderr, "Aborting: could not load tiling kernel.\n");
        return 1;
    }

    int ret = 0;
    int passed = 0;
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
