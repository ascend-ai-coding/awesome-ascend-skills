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
 * \file mc2_tiling_utils.cpp
 * \brief Trimmed vendored copy for all_gather_matmul.
 *        Removed the mc2/3rd mat_mul_v3 arch35 include and the two function definitions whose
 *        signatures reference its types (UpdateMatmulV3Args / GetMatmulV3PriorityPolicy) — neither
 *        is called on the non-quant A2 all_gather_matmul tiling path.
 */

#include <cstdlib>
#include <mutex>
#include <string>

#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_tiling_utils.h"

namespace mc2tiling {
namespace {
constexpr char DEUBG_MODE_ENV[] = "ASCEND_MC2_DEBUG_MODE";
constexpr char DEUBG_COMM_ALG_ENV[] = "ASCEND_MC2_DEBUG_COMM_ALG";
constexpr char DEUBG_STEP_SIZE_ENV[] = "ASCEND_MC2_DEBUG_STEP_SIZE";
constexpr char HCCL_BUFFSIZE[] = "HCCL_BUFFSIZE";

// getenv 在多线程下非可重入（返回进程级静态缓冲区）。此包装在 mutex 内调用并立即复制到
// std::string，使调用点不再出现裸 getenv，规避 G.EXP.22 不可重入告警。
std::mutex g_env_mutex;
std::string SafeGetEnv(const char* name) {
  std::lock_guard<std::mutex> lock(g_env_mutex);
  const char* val = getenv(name);  // NOLINT(concurrency-mt-unsafe) guarded by mutex + copied immediately
  return val != nullptr ? std::string(val) : std::string();
}
}  // namespace
uint8_t Mc2TilingUtils::GetDebugMode() {
  static const uint8_t debugMode = [] {
    std::string env = SafeGetEnv(DEUBG_MODE_ENV);
    return !env.empty() ? static_cast<uint8_t>(std::atoi(env.c_str())) : 0;
  }();
  return debugMode;
}

uint8_t Mc2TilingUtils::GetDebugCommAlg() {
  static const uint8_t debugCommAlg = [] {
    std::string env = SafeGetEnv(DEUBG_COMM_ALG_ENV);
    return !env.empty() ? static_cast<uint8_t>(std::atoi(env.c_str())) : 0;
  }();
  return debugCommAlg;
}

uint8_t Mc2TilingUtils::GetDebugStepSize() {
  static const uint8_t debugStepSize = [] {
    std::string env = SafeGetEnv(DEUBG_STEP_SIZE_ENV);
    return !env.empty() ? static_cast<uint8_t>(std::atoi(env.c_str())) : 0;
  }();
  return debugStepSize;
}

matmul_tiling::DataType ConvertGeTypeToMmType(const std::string &opName,
                                              ge::DataType type) {
  static const std::map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_MAP = {
      {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
      {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
      {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
      {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_HIFLOAT8},
      {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
      {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
  };

  auto iterator = GE_TO_MM_MAP.find(type);
  if (iterator != GE_TO_MM_MAP.end()) {
    return iterator->second;
  }

  return matmul_tiling::DataType::DT_MAX;
}

ge::DataType ConvertMmTypeToGeType(const std::string &opName,
                                   matmul_tiling::DataType type) {
  static const std::map<matmul_tiling::DataType, ge::DataType> MM_TO_GE_MAP = {
      {matmul_tiling::DataType::DT_BFLOAT16, ge::DT_BF16},
      {matmul_tiling::DataType::DT_FLOAT16, ge::DT_FLOAT16},
      {matmul_tiling::DataType::DT_FLOAT, ge::DT_FLOAT},
      {matmul_tiling::DataType::DT_HIFLOAT8, ge::DT_HIFLOAT8},
      {matmul_tiling::DataType::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN},
      {matmul_tiling::DataType::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2},
  };

  auto iterator = MM_TO_GE_MAP.find(type);
  if (iterator != MM_TO_GE_MAP.end()) {
    return iterator->second;
  }

  return ge::DT_MAX;
}

uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type) {
  static const std::map<ge::DataType, int64_t> DATA_TYPE_SIZE_MAP = {
      {ge::DT_BF16, 2},     {ge::DT_FLOAT16, 2},       {ge::DT_FLOAT, 4},
      {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}, {ge::DT_FLOAT8_E5M2, 1},
  };

  auto iterator = DATA_TYPE_SIZE_MAP.find(type);
  if (iterator != DATA_TYPE_SIZE_MAP.end()) {
    return iterator->second;
  }

  return 0;
}

HcclDataType ConvertGeTypeToHcclType(const std::string &opName,
                                     ge::DataType type) {
  static const std::map<ge::DataType, HcclDataType> HCCL_DATA_TYPE_MAP = {
      {ge::DataType::DT_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
      {ge::DataType::DT_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
      {ge::DataType::DT_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
      {ge::DataType::DT_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
      {ge::DataType::DT_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
      {ge::DataType::DT_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
      {ge::DataType::DT_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16},
      {ge::DataType::DT_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32},
      {ge::DataType::DT_BF16, HcclDataType::HCCL_DATA_TYPE_BFP16},
      {ge::DataType::DT_HIFLOAT8, HcclDataType::HCCL_DATA_TYPE_HIF8},
      {ge::DataType::DT_FLOAT8_E4M3FN, HcclDataType::HCCL_DATA_TYPE_FP8E4M3},
      {ge::DataType::DT_FLOAT8_E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E5M2},
  };

  auto iterator = HCCL_DATA_TYPE_MAP.find(type);
  if (iterator != HCCL_DATA_TYPE_MAP.end()) {
    return iterator->second;
  }

  return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

bool CheckSuppportedFormat(ge::Format format) {
  static const std::set<ge::Format> SUPPORT_FORMAT_SET = {ge::FORMAT_ND};

  return (SUPPORT_FORMAT_SET.count(format) != 0);
}

bool IsDeterministic() {
  static const bool deterministic = [] {
    std::string envStr = SafeGetEnv(HCCL_DETERMINISTIC);
    if (envStr.empty()) { return false; }
    std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::toupper);
    return envStr != "FALSE";
  }();
  return deterministic;
}

bool CheckRankSize(const NpuArch npuArch, const uint32_t rankSize) {
  static const std::map<NpuArch, std::set<uint32_t>>
      SUPPORT_RANK_SIZE_SET = {
          {NpuArch::DAV_2002, {1, 2, 4}},
          {NpuArch::DAV_2201, {1, 2, 4, 8}},
          {NpuArch::DAV_3510, {1, 2, 4, 8, 16, 32, 64}},
      };
  auto it = SUPPORT_RANK_SIZE_SET.find(npuArch);
  if (it != SUPPORT_RANK_SIZE_SET.end()) {
    return it->second.count(rankSize) != 0;
  }

  return false;
}

bool CheckDataTypeVaild(ge::DataType type,
                        std::initializer_list<ge::DataType> supportDtypeList) {
  return std::find(supportDtypeList.begin(), supportDtypeList.end(), type) !=
         supportDtypeList.end();
}

ge::graphStatus Mc2TilingUtils::CommonParamCheck(
    const gert::TilingContext *context) {
  const gert::StorageShape *aShape = context->GetInputShape(0);
  const gert::StorageShape *bShape = context->GetInputShape(1);
  if (aShape == nullptr || bShape == nullptr) { return ge::GRAPH_FAILED; }

  uint64_t aShapeDimNum = aShape->GetStorageShape().GetDimNum();
  uint64_t bShapeDimNum = bShape->GetStorageShape().GetDimNum();
  if (aShapeDimNum != 2 || bShapeDimNum != 2) { return ge::GRAPH_FAILED; }

  auto aTensor = context->GetInputDesc(0);
  auto bTensor = context->GetInputDesc(1);
  auto output = context->GetOutputDesc(0);
  if (aTensor == nullptr || bTensor == nullptr || output == nullptr) { return ge::GRAPH_FAILED; }

  auto aShapeFormat = aTensor->GetStorageFormat();
  auto bShapeFormat = bTensor->GetStorageFormat();
  auto outputFormat = output->GetStorageFormat();
  if (aShapeFormat != outputFormat) { return ge::GRAPH_FAILED; }
  if ((mc2tiling::SUPPORTED_FORMAT.count(aShapeFormat) == 0 || mc2tiling::SUPPORTED_FORMAT.count(bShapeFormat) == 0)) { return ge::GRAPH_FAILED; }
  return ge::GRAPH_SUCCESS;
}

mc2tiling::HcclDataType Mc2TilingUtils::GetDataType(ge::DataType type) {
  if (mc2tiling::HCCL_DATA_TYPE.find(type) != mc2tiling::HCCL_DATA_TYPE.end()) {
    return mc2tiling::HCCL_DATA_TYPE.at(type);
  }
  return mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

uint64_t Mc2TilingUtils::GetMaxWindowSize() {
  static const uint64_t maxWindowSize = [] {
    uint16_t defaultWindowSize = 200;
    std::string env = SafeGetEnv(HCCL_BUFFSIZE);
    if (!env.empty()) {
      try {
        defaultWindowSize = static_cast<uint16_t>(std::stoi(env));
      } catch (...) {
      }
    }
    return static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
  }();
  return maxWindowSize;
}

bool GetRankSize(const std::string &opName, const char *group, int64_t &rankSize) {
  uint32_t rankNum = static_cast<uint32_t>(rankSize);
  if (Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(group, &rankNum) != HCCL_SUCCESS) {
      return false;
  }
  rankSize = static_cast<int64_t>(rankNum);
  return true;
};

bool Mc2TilingUtils::CheckRankSize(NpuArch npuArch, uint32_t rankSize) {
  auto it = supportedRankSizeSet.find(npuArch);
  if (it != supportedRankSizeSet.end()) {
    return it->second.count(rankSize) != 0;
  }

  return false;
}

bool Mc2TilingUtils::InferGroupSize(Mc2MatmulShapeInfo &mmInfo, uint64_t &groupSizeM,
                                    uint64_t &groupSizeN, uint64_t &groupSizeK)
{
  // calculate groupSizeM
  if (groupSizeM == 0) {
    // get M of x1
    auto mValue = mmInfo.x1Shape->GetStorageShape().GetDim(mmInfo.x1Shape->GetStorageShape().GetDimNum() - 2);
    int64_t scaleMValue = 0;
    if (mmInfo.isMxfp) {
      // x1Scale is 3 dims in mx scene, 2 dims in perblock scene
      scaleMValue = mmInfo.x1ScaleShape->GetStorageShape().GetDim(
        mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 3);
    } else {
      scaleMValue = mmInfo.x1ScaleShape->GetStorageShape().GetDim(
        mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 2);
    }
    if (scaleMValue == 0) { return false; }
    if ((mValue % scaleMValue) != 0) { return false; }
    groupSizeM = mValue / scaleMValue;
  }

  if (groupSizeN == 0) {
    uint64_t nIdx = 1;
    if (mmInfo.isBTrans) {
      nIdx = 0; // shape is[n, k] when x2 is transposed
    }
    auto nValue = mmInfo.x2Shape->GetStorageShape().GetDim(nIdx);
    auto scaleNValue = mmInfo.x2ScaleShape->GetStorageShape().GetDim(nIdx);
    if (scaleNValue == 0) { return false; }
    if ((nValue % scaleNValue) != 0) { return false; }
    groupSizeN = nValue / scaleNValue;
  }

  if (groupSizeK == 0) {
    // get k of x1 according to x1shape
    auto kValue = mmInfo.x1Shape->GetStorageShape().GetDim(mmInfo.x1Shape->GetStorageShape().GetDimNum() - 1);
    int64_t scaleKValue = 0;
    if (mmInfo.isMxfp) {
      scaleKValue = mmInfo.x1ScaleShape->GetStorageShape().GetDim(
        mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 2) * 2; // in mxfp scene, scaleshape is [m, k/2, 2]
    } else {
      scaleKValue = mmInfo.x1ScaleShape->GetStorageShape().GetDim(
        mmInfo.x1ScaleShape->GetStorageShape().GetDimNum() - 1);
    }
    if (scaleKValue == 0) { return false; }
    if ((kValue % scaleKValue) != 0) { return false; }
    groupSizeK = kValue / scaleKValue;
  }

  return true;
}

}  // namespace mc2tiling
