/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_hcom_topo_info.cc
 * \brief
 */

#include <cstdlib>
#include <string>
#include <dlfcn.h>
#include "mc2_hcom_topo_info.h"
#include "op_host/op_tiling/hcom_topo_info.h"
#include "mc2_tiling_utils.h"
// hcom/hcom_topo_info.h (the CANN metadef external/hcom copy) is absent from this CANN install.
// ge::HcomTopoInfo is provided by the vendored op_host/op_tiling/hcom_topo_info.h above (line 20),
// which is the same header (same METADEF_CXX_INC_EXTERNAL_HCOM guard) — so this include is soft-guarded:
// included only when present (harmless: guard dedupes with line 20), skipped otherwise.
#if !defined(BUILD_OPEN_PROJECT) && __has_include("hcom/hcom_topo_info.h")
#include "hcom/hcom_topo_info.h"
#endif

namespace {
    constexpr size_t CCL_BUFFER_RATIO = 2;
}

// Eager-mode HCCL loader: dlsym HcomGetRankSizeEx from libhccl.so (HCCL-direct group rank-size
// query, works in eager mode) — mirrors the shared src/mc2/common version used by x_mega_moe.
namespace {
using FuncGetRankSizeEx = int32_t (*)(const char *, uint32_t *, uint32_t);

std::string GetHcclLibPath() {
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        return "";
    }
#if defined(__x86_64__)
    return std::string(ascendPath) + "/x86_64-linux/lib64/libhccl.so";
#elif defined(__aarch64__)
    return std::string(ascendPath) + "/aarch64-linux/lib64/libhccl.so";
#else

    return "";
#endif
}

class HcclLibLoader {
 public:
    static HcclLibLoader &GetInstance() {
        static HcclLibLoader instance;
        return instance;
    }
    FuncGetRankSizeEx GetRankSizeEx() const { return getRankSizeEx_; }
 private:
    HcclLibLoader() {
        std::string libPath = GetHcclLibPath();
        if (libPath.empty()) { return; }
        handle_ = dlopen(libPath.c_str(), RTLD_NOW);
        if (handle_ == nullptr) {
            return;
        }
        getRankSizeEx_ = reinterpret_cast<FuncGetRankSizeEx>(dlsym(handle_, "HcomGetRankSizeEx"));
    }
    ~HcclLibLoader() = default;
    HcclLibLoader(const HcclLibLoader &) = delete;
    HcclLibLoader &operator=(const HcclLibLoader &) = delete;
    void *handle_{nullptr};
    FuncGetRankSizeEx getRankSizeEx_{nullptr};
};
}  // namespace

namespace Mc2Hcom {
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
const std::string HCOM_GET_COMM_FUNC_NAME = "HcclCommGetHandleWithName";
#else
const std::string HCOM_GET_COMM_FUNC_NAME = "HcomGetCommHandleByGroup";
#endif
#ifdef BUILD_OPEN_PROJECT
const std::string HCCL_GET_RANK_SIZE_NAME = "HcclGetRankSize";
const std::string HCCL_GET_TOPO_TYPE_NAME = "HcclRankGraphGetTopoTypeByLayer";
#else
const std::string HCCL_GET_NET_LAYERS_NAME = "HcclRankGraphGetLayers";
const std::string HCCL_GET_TOPO_TYPE_NAME = "HcclRankGraphGetTopoTypeByLayer";
const std::string HCCL_GET_SIZE_NAME = "HcclRankGraphGetRankSizeByLayer";
#endif
const std::string COMM_GET_HCCL_BUFFER_NAME = "HcclGetHcclBuffer";

static const std::string GetLibPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        return nullptr;
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#else
    return nullptr;
#endif
    std::string fullPath = ascendPath + hcclPathPostfix;

    return fullPath;
}

template <typename T>
T GetHcclLibFunc(void *handle, const std::string &funcName)
{
    return reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
}

MC2HcomTopology::MC2HcomTopology(const char *libPath)
{
    handle_ = dlopen(libPath, RTLD_NOW);
    if (handle_ == nullptr) {
        return;
    }

    getCommHandle_ = GetHcclLibFunc<FuncGetHandle>(handle_, HCOM_GET_COMM_FUNC_NAME);
#ifdef BUILD_OPEN_PROJECT
    getRankSize_ = GetHcclLibFunc<FuncGetRankSize>(handle_, HCCL_GET_RANK_SIZE_NAME);
    getTopoTypeByLayer_ = GetHcclLibFunc<FuncGetTopoTypeByLayer>(handle_, HCCL_GET_TOPO_TYPE_NAME);
#else
    getNetLayers_ = GetHcclLibFunc<FuncGetNetLayers>(handle_, HCCL_GET_NET_LAYERS_NAME);
    getTopoTypeByLayer_ = GetHcclLibFunc<FuncGetTopoTypeByLayer>(handle_, HCCL_GET_TOPO_TYPE_NAME);
    getInstSize_ = GetHcclLibFunc<FuncGetInstSize>(handle_, HCCL_GET_SIZE_NAME);
#endif
    getHcclBuffer_ = GetHcclLibFunc<FuncGetHcclBuffer>(handle_, COMM_GET_HCCL_BUFFER_NAME);

#ifdef BUILD_OPEN_PROJECT
    if (getCommHandle_ == nullptr || getRankSize_ == nullptr || getTopoTypeByLayer_ == nullptr ||
        getHcclBuffer_ == nullptr) {
        dlclose(handle_); // Release dlopen handle to prevent resource leak
        handle_ = nullptr;

        getCommHandle_ = nullptr;
        getRankSize_ = nullptr;
        getTopoTypeByLayer_ = nullptr;
        getHcclBuffer_ = nullptr;
        return;
    }
#else
    if (getCommHandle_ == nullptr || getNetLayers_ == nullptr || getTopoTypeByLayer_ == nullptr
        || getInstSize_ == nullptr || getHcclBuffer_ == nullptr) {
        dlclose(handle_); // Release dlopen handle to prevent resource leak
        handle_ = nullptr;

        getCommHandle_ = nullptr;
        getNetLayers_ = nullptr;
        getTopoTypeByLayer_ = nullptr;
        getInstSize_ = nullptr;
        getHcclBuffer_ = nullptr;
        return;
    }
#endif
}

MC2HcomTopology &MC2HcomTopology::GetInstance()
{
    static const std::string libPath = GetLibPath();
    static MC2HcomTopology loader(libPath.c_str());
    return loader;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const
{
    if (getCommHandle_ == nullptr) {
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCommHandle_(group, commHandle));
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const
{
    if (getRankSize_ == nullptr) {
        return HCCL_E_PTR;
    }
    HcclComm comm;
    HcclResult ret = CallHcomGetCommHandleByGroup(group, &comm);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return static_cast<HcclResult>(getRankSize_(comm, ranksize));
}

HcclResult MC2HcomTopology::CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const
{
    if (getTopoTypeByLayer_ == nullptr) {
        return HCCL_E_PTR;
    }
    HcclComm comm;
    HcclResult ret = CallHcomGetCommHandleByGroup(group, &comm);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return static_cast<HcclResult>(getTopoTypeByLayer_(comm, 0, topoType));
}
#else
HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t **netLayer, uint32_t *netLayerNum) const
{
    if (getNetLayers_ == nullptr) {
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getNetLayers_(comm, netLayer, netLayerNum));
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType) const
{
    if (getTopoTypeByLayer_ == nullptr) {
        return HCCL_E_PTR;
    }
    CommTopo topoRet;
    HcclResult ret = static_cast<HcclResult>(getTopoTypeByLayer_(comm, netLayer, &topoRet));
    if (ret == HCCL_SUCCESS) {
        *topoType = static_cast<uint32_t>(topoRet);
    }
    return ret;
}

HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum) const
{
    if (getInstSize_ == nullptr) {
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getInstSize_(comm, netLayer, rankNum));
}
#endif

HcclResult MC2HcomTopology::CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const
{
    if (getHcclBuffer_ == nullptr) {
        return HCCL_E_PTR;
    }
    void *buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = static_cast<HcclResult>(getHcclBuffer_(comm, &buffer, &size));
    if (ret == HCCL_SUCCESS) {
        *cclBufferSize = size / CCL_BUFFER_RATIO;
    }
    return ret;
}

HcclResult MC2HcomTopology::CallCommGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size) const
{
    if (getHcclBuffer_ == nullptr) {
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getHcclBuffer_(comm, buffer, size));
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm)
{
    if (group == nullptr || cclBufferSize == nullptr || hcclComm == nullptr) {
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetCommHandleByGroup(group, hcclComm);
    if (ret != HCCL_SUCCESS) {
        *hcclComm = nullptr;
        return HCCL_SUCCESS;
    }
    ret = GetInstance().CallCommGetCCLBufSizeCfg(*hcclComm, cclBufferSize);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetHcclBufferByGroup(const char *group, void **buffer, uint64_t *size)
{
    if (group == nullptr || buffer == nullptr || size == nullptr) {
        return HCCL_E_PTR;
    }
    HcclComm hcclComm;
    HcclResult ret = GetInstance().CallHcomGetCommHandleByGroup(group, &hcclComm);
    if (ret != HCCL_SUCCESS) {
        hcclComm = nullptr;
        return ret;
    }
    ret = GetInstance().CallCommGetHcclBuffer(hcclComm, buffer, size);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    return HCCL_SUCCESS;
}

namespace {
// 共用 fallback：topo 信息不可用时回填 max window size 并返回成功。
// BUILD_OPEN_PROJECT / 非 BUILD_OPEN_PROJECT 两支 CommGetGroupLocalWindowSize 均复用，避免实现重复。
HcclResult FillMaxWindowFallback(uint64_t *cclBufferSize)
{
    *cclBufferSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    return HCCL_SUCCESS;
}
} // namespace

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t *cclBufferSize)
{
    if (ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize) != ge::GRAPH_SUCCESS) {
        return FillMaxWindowFallback(cclBufferSize);
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    if (group == nullptr || rankNum == nullptr) {
        return HCCL_E_PTR;
    }
    HcclResult ret = GetInstance().CallHcomGetRankSizeEx(group, rankNum, COMM_IS_NOT_SET_DEVICE);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    if (group == nullptr || topoType == nullptr) {
        return HCCL_E_PTR;
    }
    CommTopo topoRet;
    HcclResult ret = GetInstance().CallHcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    *topoType = static_cast<uint32_t>(topoRet);
    return HCCL_SUCCESS;
}
#else
// Eager-mode adaptation (x_ops leaves BUILD_OPEN_PROJECT empty/OFF, so this #else branch is the
// active one — and the OFF build mode's source staging doesn't propagate -DBUILD_OPEN_PROJECT to
// the compiler, so the `#if` dlsym branch can't be activated without switching the whole build to
// --build-open-project ON). The source-repo #else resolved rank_size/topo via ge::HcomTopoInfo
// — a GE graph-compilation-time query that is EMPTY in torch eager mode, causing
// "GetGroupRankSize: group key [...] has not been added" -> rank_size=0 -> output-dim0 mismatch.
// For eager use (torch binding / test path): rank_size via HCCL-direct HcomGetRankSizeEx
// (libhccl.so, verified on CANN 9.1.0, same as the shared src/mc2/common version used by x_mega_moe);
// topo defaults to COMM_MESH and window size to GetMaxWindowSize() (mirroring the #if branch's
// fallbacks). Correct for single-node A2 (Ascend910B) full-mesh — the only topology this A2-only
// build targets. For multi-node / non-mesh topologies, use graph mode (torchair) instead.

HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t* cclBufferSize)
{
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize);
    if (ret != ge::GRAPH_SUCCESS) {
        // eager: topo info unavailable; fall back to max window size (same as #if branch).
        return FillMaxWindowFallback(cclBufferSize);
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    if (group == nullptr || rankNum == nullptr) {
        return HCCL_E_PTR;
    }
    FuncGetRankSizeEx func = HcclLibLoader::GetInstance().GetRankSizeEx();
    if (func == nullptr) {
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(func(group, rankNum, 0U));
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    ge::HcomTopoInfo::TopoInfo topoInfo;
    if (!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(group, topoInfo)) {
        // eager: topo info not set; default COMM_MESH (A2 single-node full-mesh) -> GetCommSets
        // returns COMM_MESH -> isA3=0 (A2 path). Correct for this op's A2-only build target.

        *topoType = COMM_MESH;
        return HCCL_SUCCESS;
    }
    *topoType = topoInfo.topo_level_descs[static_cast<int32_t>(ge::HcomTopoInfo::TopoLevel::L0)].comm_sets;

    return HCCL_SUCCESS;
}
#endif
}  // namespace Mc2Hcom
