# 数据搬入搬出方案

## 简单场景：直接在 GM 上计算

SIMT 支持直接操作 GM 数据，通常不需要显式调用 DataCopyPad 等接口：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t count, __gm__ float* x, __gm__ float* y) {
    for (uint64_t i = ...; i < count; i += step) {
        y[i] = x[i] + 1.0f;  // 直接读写 GM
    }
}
```

## 复杂场景：使用 UB 中转

需要在 SIMT 内部使用 UB 时：

1. **tiling 侧**：设置 `SetLocalMemory` 分配动态 UB
2. **kernel 侧**：使用 `TBuf<QuePosition::VECCALC>` 声明 buffer
3. **VF 调用**：将 `__ubuf__` 指针传入 VF

### Tiling 侧

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;
uint64_t ubsize = 256 * 1024;
context->SetLocalMemorySize(ubsize - DCACHE_SIZE);
```

### Kernel 侧

```cpp
TBuf<QuePosition::VECCALC> sharedBuf_;
pipe_->InitBuffer(sharedBuf_, 2048);

LocalTensor<uint32_t> sharedTensor = sharedBuf_.Get<uint32_t>();
__ubuf__ uint32_t* sharedUbPtr = (__ubuf__ uint32_t*)sharedTensor.GetPhyAddr();

// 传入 VF
Simt::VF_CALL<OpSimt<T>>(Simt::Dim3(THREAD_NUM), ..., sharedUbPtr);
```

## 选择指南

| 场景 | 方案 | 原因 |
|------|------|------|
| 逐元素操作，无依赖 | 直接 GM | 避免不必要的 UB 搬运 |
| 需要核内线程间共享数据 | UB 中转 | 共享数据需在核内共享内存 |
| 多次访问同一数据 | UB 中转 | UB 访问延迟低于 GM |