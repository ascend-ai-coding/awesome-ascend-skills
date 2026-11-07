# VF_CALL 启动模式

## 调用语法

```cpp
AscendC::Simt::VF_CALL<VFFunction>(AscendC::Simt::Dim3(thread_num), args...);
```

- `VFFunction`：要启动的 SIMT VF 函数（带模板参数）
- `Dim3(thread_num)`：启动线程数，编译期常量或运行时值（如 tiling 自适应）；运行时值必须 ≤ VF 的 `LAUNCH_BOUND`
- `args...`：传递给 VF 函数的参数

## 完整示例

### Add 算子

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpComputeSimt(
    uint64_t count, __gm__ T* x1, __gm__ T* x2, __gm__ T* y) {
    for (uint64_t index = (uint64_t)(
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

## 参数传递三种方式

```cpp
// ① 按值传（标量/half 位转换）：host 读出转计算类型后按值传
__aicore__ 入口里读 alpha → float alphaVal;   // 半精度手动位展开成 fp32
Simt::VF_CALL<K>(Simt::Dim3(n), alphaVal, ...);

// ② GM 指针 + 索引（scalar_list）：VF 内按 tensor 序号读
__simt_vf__ void Compute(..., __gm__ T1* scalars_gm, uint32_t tensorIdx, ...) {
    float s = (float)scalars_gm[tensorIdx];
}

// ③ UB 数组（__ubuf__）：shape/stride/magic 参数打包进 UB 再传指针
//    host: SetValue 写入 → DataSyncBarrier<MemDsbT::UB>() → asc_vf_call(..., (__ubuf__ uint64_t*)paramBuf.GetPhyAddr());
```
## GM_ADDR 直接转换（纯 SIMT 算子）

纯 SIMT 算子应直接使用传入的 `GM_ADDR` 参数，避免申请 GlobalTensor 后 GetPhyAddr：

```cpp
// ✅ 推荐：直接转换
__gm__ T* x_gm = (__gm__ T*) x;

// ❌ 避免：不必要的 GlobalTensor 中转
GlobalTensor<T> xGlobal;
xGlobal.SetGlobalBuffer((__gm__ T*)x, count);
auto phyAddr = xGlobal.GetPhyAddr();
```