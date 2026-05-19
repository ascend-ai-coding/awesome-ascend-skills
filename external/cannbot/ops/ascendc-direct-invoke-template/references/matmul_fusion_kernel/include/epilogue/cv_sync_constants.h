// ============================================================
// [PATTERN] CV 同步常量 — AIC ↔ AIV 核间同步
// 融合算子中 AIC 和 AIV 之间的 CrossCoreFlag 配对常量。
// 这些常量经过验证，请勿随意修改数值。
//
// 详见 matmul_fusion_guide.md §3.3 "CV 同步与 Hard Event"
// ============================================================
#ifndef EPILOGUE_CV_SYNC_CONSTANTS_H
#define EPILOGUE_CV_SYNC_CONSTANTS_H

#include <cstdint>

namespace CvSync {

// CrossCoreFlag 模式（AIC ↔ AIV 独立触发）
constexpr uint16_t MODE = 4;

// Flag ID 基址
// Ascend 950PR/950DT: AIV0↔AIC 可用范围 0-10
constexpr int16_t AIC_TO_AIV_FLAG = 8;   // AIC → AIV: L0C 数据就绪
constexpr int16_t AIV_TO_AIC_FLAG = 5;   // AIV → AIC: UB 已消费

// Flag ID 循环分配参数
constexpr int16_t COUNT_ID_MAX = 15;
constexpr int16_t COUNT_FLAG   = 3;      // Flag ID 循环周期

} // namespace CvSync

#endif // EPILOGUE_CV_SYNC_CONSTANTS_H
