# 线程排布统一编程模式

## 标准模式

SIMT 算子采用统一的线程排布 for 循环模式，每个线程以 stride 间隔遍历数据：

```cpp
for (uint64_t index = static_cast<uint64_t>(
         AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx());
     index < count;
     index += (AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
    // 每个线程处理 stride 间隔的元素
}
```

## 索引计算 API

| API | 说明 |
|-----|------|
| `AscendC::Simt::GetThreadIdx()` | 当前线程索引 |
| `AscendC::Simt::GetThreadNum()` | 纯线程总数 |
| `AscendC::Simt::GetBlockIdx()` | 当前核索引 |
| `AscendC::Simt::GetBlockNum()` | 核总数 |

## 索引计算分解

```
初始 index = blockIdx * threadNum + threadIdx
步进 stride = threadNum * blockNum
```

- 初始 index：将线程在全局中的起始位置线性化
- 步进 stride：所有核所有线程的总数，确保每个元素只被一个线程处理

## Select 算子示例

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(2048) inline void OpSelectSimt(
    int32_t needCoreNum, int32_t threadNum, int64_t currentCoreElements,
    __gm__ uint8_t* condition, __gm__ T* x1, __gm__ T* x2, __gm__ T* y) {
    for (uint64_t index = static_cast<uint64_t>(
             AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx());
         index < count;
         index += (AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
        y[index] = condition[index] ? x1[index] : x2[index];
    }
}
```