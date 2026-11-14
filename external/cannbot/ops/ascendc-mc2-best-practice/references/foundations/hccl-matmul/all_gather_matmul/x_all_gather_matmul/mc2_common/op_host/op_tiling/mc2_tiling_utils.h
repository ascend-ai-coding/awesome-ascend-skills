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
 * \file mc2_tiling_utils.h
 * \brief Trimmed vendored copy for all_gather_matmul.
 *        Removed the mc2/3rd mat_mul_v3 arch35 include and the two declarations whose signatures
 *        reference its types (UpdateMatmulV3Args / GetMatmulV3PriorityPolicy) — neither is called
 *        on the non-quant A2 all_gather_matmul tiling path.
 */

#ifndef __MC2_TILING_UTILS_H__
#define __MC2_TILING_UTILS_H__

#include <cstdint>
#include <map>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "formulaic_tiling_datatype.h"
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "matmul_formulaic_tiling.h"
#include "platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "soc_spec.h"

namespace mc2tiling {
constexpr uint32_t STANDARD_CARD_4P = 4;
constexpr uint32_t EIGHT_P_8P = 8;  // 8P场景（POD8P/Server8P）
constexpr uint32_t COMM_MESH = 0b1U;
constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
constexpr uint8_t COMM_ALG_FULL_MESH_HOST = 6;
constexpr uint32_t AIC_NUM_950 = 32;
constexpr uint64_t MC2_TILINGKEY_OFFSET =
    uint64_t(1000000000000000000UL);  // 10^18
constexpr size_t RES_LEN = 64;
constexpr size_t MAX_MSG_NUM = 16;
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;  // 只通信不计算
constexpr char HCCL_DETERMINISTIC[] = "HCCL_DETERMINISTIC";
/**
当前通信API未提供枚举，后续会提供
0：默认值 1：HOST_TS（A2/3支持 A5不支持）2：AICPU_TS（A2/3支持 A5不支持）
3：AIV 4：AIV_ONLY（A2/3支持 A5不支持） 5：CCU_MS（A2/3支持 A5不支持）
6：CCU_SCHED（A2/3支持 A5不支持） 7：AICPU_UB/ROCE（A5不支持）
**/
constexpr uint8_t A5_AICPU_TS_ENGINE = 2;
constexpr uint8_t AIV_ENGINE = 3;
constexpr uint8_t A5_CCU_ENGINE = 6;
constexpr uint8_t Y_INDEX = 3;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_VERSION3 = 3;
constexpr double COMM_GROW_RATIO = 1.15;
constexpr uint64_t MTE_STATE_ZONE_SIZE = 1024UL * 1024UL;

constexpr uint64_t LARGE_K = 8192;
constexpr uint64_t LARGE_N = 5120;
constexpr uint64_t SMALL_N_BOUNDARY = 2048;
constexpr uint64_t TINY_M = 512;
constexpr uint64_t SMALL_M = 2048;
constexpr uint64_t MEDIAN_M = 4096;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO1 = 3;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO2 = 1.5;

constexpr uint8_t TIME_LOWER_RATIO = 2;
constexpr double TIME_UPPER_RATIO = 3.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO1 = 1.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO2 = 1.2;
constexpr double CUBE_UTIL_THRESH = 0.85;
constexpr uint32_t AICPU_NUM_BLOCKS_A2 = 6U;

constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;

constexpr auto DEFAULT_KEY_FOR_FITTING_MAP = "0_0";

constexpr static uint64_t ALL_GATHER_HCCL_MEM_LIMIT = 256 * 1024 * 1024;
constexpr static uint64_t ALL_GATHER_HCCL_NUM_LIMIT = 16;

enum class Mc2QuantMode {
  DEFAULT = 0,
  PERTENSOR_MODE,
  PERBLOCK_MODE,
  MXFP_MODE,
  INVALID_MODE,
};

struct HcclAicpuOpParam {
  uint8_t res[RES_LEN];
};

struct Mc2MatmulShapeInfo {
    const gert::StorageShape *x1Shape{nullptr};
    const gert::StorageShape *x2Shape{nullptr};
    const gert::StorageShape *x1ScaleShape{nullptr};
    const gert::StorageShape *x2ScaleShape{nullptr};
    bool isMxfp{false};
    bool isBTrans{false};
    const char* opName{nullptr};
};

struct KFCMsgBody {
  // Rank* aiv * MsgSize * sizeof(消息)
  HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
  HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};
struct KFCNotify {
  // 消息通信
  HcclAicpuOpParam msgSend[MAX_MSG_NUM];  // 填充16个
  HcclAicpuOpParam msgCnt[MAX_MSG_NUM];
};

constexpr std::initializer_list<ge::DataType> FP8DTYPE_SUPPORT_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2,
    ge::DataType::DT_HIFLOAT8};

constexpr std::initializer_list<ge::DataType> MXFP8DTYPE_SUPPORT_LIST = { ge::DataType::DT_FLOAT8_E4M3FN,
    ge::DataType::DT_FLOAT8_E5M2 };

matmul_tiling::DataType ConvertGeTypeToMmType(const std::string &opName,
                                              ge::DataType type);
ge::DataType ConvertMmTypeToGeType(const std::string &opName,
                                   matmul_tiling::DataType type);
uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type);
HcclDataType ConvertGeTypeToHcclType(const std::string &opName,
                                     ge::DataType type);
bool CheckSuppportedFormat(ge::Format format);
bool IsDeterministic();
bool GetRankSize(const std::string &opName, const char *group,
                 int64_t &rankSize);
bool CheckRankSize(const NpuArch npuArch, const uint32_t rankSize);

bool CheckDataTypeVaild(ge::DataType type,
                        std::initializer_list<ge::DataType> supportDtypeList);

// GetSocVersion removed: it was the only caller of fe::PlatFormInfos::GetPlatformResWithLock,
// which needs the full type that the source repo pulled in transitively via the trimmed mc2/3rd
// mat_mul_v3 arch35 header (matmul_v3_base_tiling_advanced.h -> platform/platform_infos_def.h).
// It was unused in this operator (grep found only the definition); soc-version detection here uses
// platform_ascendc::PlatformAscendC (GetNpuArch below), which only holds the fe::PlatFormInfos*
// (forward-declared in tiling_context.h suffices).
inline NpuArch GetNpuArch(const gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    return ascendcPlatform.GetCurNpuArch();
}

class Mc2TilingUtils {
 public:
  static uint8_t GetDebugMode();
  static uint8_t GetDebugCommAlg();
  static uint8_t GetDebugStepSize();
  static ge::graphStatus CommonParamCheck(const gert::TilingContext *context);
  static mc2tiling::HcclDataType GetDataType(ge::DataType type);
  static uint64_t GetMaxWindowSize();
  static bool CheckRankSize(NpuArch npuArch, uint32_t rankSize);
  static HcclDataType ConvertGeTypeToHcclType(const std::string &opName,
                                              ge::DataType type);
  static bool InferGroupSize(Mc2MatmulShapeInfo &mmInfo, uint64_t &groupSizeM,
                             uint64_t &groupSizeN, uint64_t &groupSizeK);
};

const std::map<ge::DataType, matmul_tiling::DataType> D_TYPE_MAP = {
    {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
};

const std::map<matmul_tiling::DataType, ge::DataType> D_TYPE_MATMUL_MAP = {
    {matmul_tiling::DataType::DT_BFLOAT16, ge::DT_BF16},
    {matmul_tiling::DataType::DT_FLOAT16, ge::DT_FLOAT16},
    {matmul_tiling::DataType::DT_FLOAT, ge::DT_FLOAT},
};

const std::map<ge::DataType, int64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
};

const std::map<ge::DataType, mc2tiling::HcclDataType> HCCL_DATA_TYPE = {
    {ge::DataType::DT_INT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT8},
    {ge::DataType::DT_UINT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT8},
    {ge::DataType::DT_INT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT16},
    {ge::DataType::DT_UINT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT16},
    {ge::DataType::DT_INT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT32},
    {ge::DataType::DT_UINT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT32},
    {ge::DataType::DT_FLOAT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP16},
    {ge::DataType::DT_FLOAT, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP32},
    {ge::DataType::DT_BF16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_BFP16},
    {ge::DataType::DT_HIFLOAT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_HIF8}
};

 const std::map<NpuArch, std::set<uint32_t>> supportedRankSizeSet = {
    {NpuArch::DAV_2002, {1, 2, 4}},
    {NpuArch::DAV_2201, {1, 2, 4, 8}},
    {NpuArch::DAV_3510, {1, 2, 4, 8, 16, 32, 64}},
};

const std::set<ge::Format> SUPPORTED_FORMAT = {
    ge::FORMAT_NCL,  ge::FORMAT_NCDHW, ge::FORMAT_DHWCN,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW,  ge::FORMAT_ND};

inline ge::graphStatus GetCclBufferSize(const char* groupStr, uint64_t* cclBufferSize, const char* nodeName)
{
    HcclComm hcclComm;
    if (Mc2Hcom::MC2HcomTopology::CommGetCclBufferSizeByGroup(groupStr, cclBufferSize, &hcclComm) != HCCL_SUCCESS) { return ge::GRAPH_FAILED; }
    if (hcclComm == nullptr) {
        if (Mc2Hcom::MC2HcomTopology::CommGetGroupLocalWindowSize(groupStr, cclBufferSize) != HCCL_SUCCESS) { return ge::GRAPH_FAILED; }

    } else {
    }
    if (*cclBufferSize == 0) { return ge::GRAPH_FAILED; }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetEpWinSize(const gert::TilingContext *context, const char *nodeName,
    uint64_t &hcclBufferSizeEp, uint64_t &maxWindowSizeEp, uint32_t attrGroupEpIndex, bool isLayered)
{
    auto attrs = context->GetAttrs();
    if (mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510) {
        // A5 暂不支持 Hccl CommGetBufSizeCfg 接口，此处暂作规避
        // A5 实际物理分配为 HCCL_BUFFSIZE 的 2 倍
        hcclBufferSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() * 2UL;
        // A5 上前 1MB 作为状态区，剩余空间用作数据区
        maxWindowSizeEp = hcclBufferSizeEp - MTE_STATE_ZONE_SIZE;
    } else {
        if (isLayered) {
            hcclBufferSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
        } else {
            auto groupEpHccl = attrs->GetAttrPointer<char>(static_cast<int>(attrGroupEpIndex));
            if (GetCclBufferSize(groupEpHccl, &hcclBufferSizeEp, nodeName) != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
        }
        maxWindowSizeEp = hcclBufferSizeEp;
    }
    return ge::GRAPH_SUCCESS;
}

// 临时判断是否为标卡4p形态(4卡，950)
inline bool IsStandardCard4P(const uint32_t rankDim, const NpuArch npuArch) {
    return ((rankDim == STANDARD_CARD_4P) && (npuArch == NpuArch::DAV_3510));
}

// 判断是否为8P形态(8卡，950)
inline bool Is8P(const uint32_t rankDim, const NpuArch npuArch)
{
    return ((rankDim == EIGHT_P_8P) && (npuArch == NpuArch::DAV_3510));
}

// 判断是否使用 All2All + Vec Reduce 通路（StandardCard 4P 或 8P）
inline bool IsUseA2APath(const uint32_t rankDim, const NpuArch npuArch)
{
    return ((npuArch == NpuArch::DAV_3510) &&
            (rankDim == STANDARD_CARD_4P || rankDim == EIGHT_P_8P));
}
}  // namespace mc2tiling

#endif
