# 索引计算 API

## API 列表

| API | 返回类型 | 说明 |
|-----|---------|------|
| `AscendC::Simt::GetThreadIdx()` | uint64_t | 当前线程在核内的索引 (0 ~ threadNum-1) |
| `AscendC::Simt::GetThreadNum()` | uint64_t | 当前核的线程总数 |
| `AscendC::Simt::GetBlockIdx()` | uint64_t | 当前核的全局索引 (0 ~ blockNum-1) |
| `AscendC::Simt::GetBlockNum()` | uint64_t | 总核数 |

## 全局索引计算

```cpp
// 线程在全局数据中的线性索引
uint64_t globalIdx = GetBlockIdx() * GetThreadNum() + GetThreadIdx();

// 步进 stride（所有核所有线程总数）
uint64_t stride = GetThreadNum() * GetBlockNum();
```

## 使用示例

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(uint64_t count, __gm__ float* x, __gm__ float* y) {
    uint64_t idx = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint64_t step = AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum();
    for (uint64_t i = idx; i < count; i += step) {
        y[i] = x[i] * 2.0f;
    }
}
```

## 头文件

```cpp
#include "simt_api/common_functions.h"
```