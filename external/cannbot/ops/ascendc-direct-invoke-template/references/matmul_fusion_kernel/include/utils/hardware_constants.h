// ============================================================
// [PATTERN] 硬件级常量集中定义（芯片相关，跨算子共享）
// 任何新算子都不应在 Epilogue 内重复定义这些值。
// ============================================================
#ifndef HARDWARE_CONSTANTS_H
#define HARDWARE_CONSTANTS_H

#include <cstdint>

namespace Hardware {

// UB（Unified Buffer）物理大小。
//   - Ascend 3510 (dav-3510) 单 AIV 可见 UB = 248 KB
//   - 若切换芯片架构，需重新确认。通常可从 `kernel_operator_intf.h` 的
//     `TOTAL_UB_SIZE` 或 tikcfw 头文件获取。
// [SAMPLE] 此值对应当前 skill 目标芯片（dav-3510），其他架构必须更新。
constexpr uint32_t UB_SIZE = 248 * 1024;

// 32-byte GM 对齐要求。
// 同架构下通用，一般不会变。
constexpr uint32_t GM_ALIGN_BYTES = 32;

// 根据 dtype 计算对齐元素数（`32 / sizeof(T)`）。
template <typename T>
constexpr int64_t AlignElemOf() {
    return static_cast<int64_t>(GM_ALIGN_BYTES / sizeof(T));
}

}  // namespace Hardware

#endif  // HARDWARE_CONSTANTS_H
