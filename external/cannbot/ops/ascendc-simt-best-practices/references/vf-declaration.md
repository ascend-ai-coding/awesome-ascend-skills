# VF 函数声明模式

## 声明规范

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(thread_num) inline void YourKernel(...);
```

- 必须使用 `__simt_vf__` 标记
- `LAUNCH_BOUND(thread_num)` 可选，默认 1024
- 线程数必须是编译期常量（`constexpr` 或字面量）

## 参数类型支持

| 类型类别 | 支持类型 |
|---------|---------|
| 指针类型 | `__gm__ T*`, `__ubuf__ T*` |
| 标量类型 | bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, half, bfloat16_t |
| 返回值 | 必须是 void |

## 完整声明示例

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpComputeSimt(
    uint64_t count, __gm__ T* x1, __gm__ T* x2, __gm__ T* y) {
    // SIMT kernel 实现
}
```

## 常量线程数声明

```cpp
constexpr uint32_t THREAD_NUM = 512;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...);
```

- `LAUNCH_BOUND(THREAD_NUM)` 与 `Simt::Dim3(THREAD_NUM)` 必须使用同一个 `constexpr` 常量