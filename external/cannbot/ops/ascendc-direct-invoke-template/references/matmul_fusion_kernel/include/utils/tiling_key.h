// ============================================================
// [EXTEND] Tiling Key — 转置模式枚举和运行时分发辅助
// 用于将运行时的 transA/transB 参数映射到编译期模板实例。
//
// 修改清单：
// [EXTEND 1] TransposeMode — 新增转置组合时扩展
// ============================================================
#ifndef UTILS_TILING_KEY_H
#define UTILS_TILING_KEY_H

#include <cstdint>

// [EXTEND] 转置模式枚举
// 每种组合对应一个编译期模板实例（见 Host Launcher 的 4 路分发）
enum class TransposeMode : uint8_t {
    NN = 0,  // transA=false, transB=false: C = A @ B
    NT = 1,  // transA=false, transB=true:  C = A @ B^T
    TN = 2,  // transA=true,  transB=false: C = A^T @ B
    TT = 3,  // transA=true,  transB=true:  C = A^T @ B^T
};

// 从命令行参数构造 TransposeMode
inline TransposeMode MakeTransposeMode(uint8_t transA, uint8_t transB) {
    return static_cast<TransposeMode>((transA << 1) | transB);
}

#endif // UTILS_TILING_KEY_H
