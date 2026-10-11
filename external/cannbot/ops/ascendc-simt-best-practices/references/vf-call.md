# VF_CALL 启动模式

## 调用语法

```cpp
AscendC::Simt::VF_CALL<VFFunction>(AscendC::Simt::Dim3(thread_num), args...);
```

- `VFFunction`：要启动的 SIMT VF 函数（带模板参数）
- `Dim3(thread_num)`：启动线程数，必须是编译期常量
- `args...`：传递给 VF 函数的参数

## 完整示例

### Add 算子

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpComputeSimt(
    uint64_t count, __gm__ T* x1, __gm__ T* x2, __gm__ T* y) {
    for (uint64_t index = static_cast<uint64_t>(
             AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx());
         index < count;
         index += (AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
        y[index] = x1[index] + x2[index];
    }
}

template <typename T>
__aicore__ inline void Process(uint64_t count, GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    __gm__ T* x1_gm = (__gm__ T*) x1;
    __gm__ T* x2_gm = (__gm__ T*) x2;
    __gm__ T* y_gm = (__gm__ T*) y;
    AscendC::Simt::VF_CALL<OpComputeSimt<T>>(
        AscendC::Simt::Dim3(512), count, x1_gm, x2_gm, y_gm);
}
```

### 纯 SIMT 算子 GM 地址传递

纯 SIMT 算子应直接使用传入的 `GM_ADDR` 参数，避免申请 GlobalTensor 后 GetPhyAddr：

```cpp
// ✅ 推荐：直接转换
__gm__ T* x_gm = (__gm__ T*) x;

// ❌ 避免：不必要的 GlobalTensor 中转
GlobalTensor<T> xGlobal;
xGlobal.SetGlobalBuffer((__gm__ T*)x, count);
auto phyAddr = xGlobal.GetPhyAddr();
```