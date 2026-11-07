# VF 函数声明模式

## 声明规范

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(thread_num) inline void YourKernel(...);
```

- 必须使用 `__simt_vf__` 标记
- `LAUNCH_BOUND(thread_num)` 可选，默认 1024，必须是编译期常量（决定寄存器分配上限）
- 启动线程数 `Dim3` 可与 `LAUNCH_BOUND` 同源，也可运行时收缩，但必须 ≤ `LAUNCH_BOUND`

## 三层结构：`__simt_callee__` → `__simt_vf__` → `__aicore__` Process

```cpp
// __simt_callee__: 被 VF 调用的设备函数（只能进入 VF 调用链）
__simt_callee__ __aicore__ inline float ComputeOne(float v) { ... }

// __simt_vf__: SIMT 向量化函数（VF），LAUNCH_BOUND 限定每核线程数以控制寄存器占用
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void OpSimtKernel(int64_t total, __gm__ T* x, __gm__ T* y) { ... }

// __aicore__ Process: 外层标量控制流 + TPipe/UB 管理 + 发射 VF
__aicore__ inline void Process(...) {
    Simt::VF_CALL<OpSimtKernel<T>>(Simt::Dim3(THREAD_NUM), total, x_gm, y_gm);
}
```

- SIMT VF 内**只能调用 `__simt_callee__` 函数或 `__callee__` 函数**——被 VF 调用的 helper 必须带其中之一，否则无法进入 VF 调用链；`__simt_callee__` 函数内同样只能调用这两类（且可使用 SIMT 内置变量如 `threadIdx`/`blockIdx`）。
- 出处：CANN `simd_simt_hybrid_builtin_keywords.md`（SIMT VF 函数约束）。

## 参数类型支持

| 类型类别 | 支持类型 |
|---------|---------|
| 指针类型 | `__gm__ T*`, `__ubuf__ T*`, `GM_ADDR` |
| 标量类型 | bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, half, bfloat16_t |
| 返回值 | 必须是 void |

## 线程数选择

线程数分档（2048/1024/512…）、按索引位宽分档、运行时收缩策略属切分设计，见 [`ascendc-simt-tiling-design`](../../ascendc-simt-tiling-design/SKILL.md)。kernel 侧只须遵守：`LAUNCH_BOUND(N)` 必须是编译期常量（决定寄存器分配上限），`Simt::Dim3` 运行时值不得超过它。

## 相关文档

- VF 调用与参数传递：[vf-call.md](vf-call.md)
- 线程排布与 grid-stride：[thread-stride-pattern.md](thread-stride-pattern.md)
