# SIMT 算子性能优化指南

## 优化规则库

### Rule 001: SIMT VF 内部避免 if-else 分支

- **分类**: 分支优化
- **优先级**: 高
- **适用场景**: `__simt_vf__` 函数内部存在条件分支逻辑（if / else / switch），且分支条件可在编译期确定或通过模板参数消除
- **说明**: SIMT VF（Vector Function）是 SIMT 模式下的向量计算函数，所有线程按 SIMT 方式执行。if-else 分支会导致线程分化（thread divergence），同一 warp/wavefront 内的线程走不同路径，部分线程需要等待另一部分执行完毕，造成流水线气泡，降低有效利用率。通过模板参数将运行时分支提升到编译期，可以彻底消除分支，让所有线程执行统一路径。

**反面示例**（不推荐）:

```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024)
inline void OpFooSimt(GM_ADDR x, GM_ADDR y, int32_t mode, int32_t totalLength) {
    int32_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    int32_t step = Simt::GetThreadNum() * Simt::GetBlockNum();
    for (int32_t i = idx; i < totalLength; i += step) {
        T val = x[i];
        if (mode == 0) {
            y[i] = val + static_cast<T>(1);
        } else {
            y[i] = val * static_cast<T>(2);
        }
    }
}
```

**正面示例**（推荐）:

```cpp
// 通过模板参数消除运行时分支
template <typename T, int32_t Mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024)
inline void OpFooSimt(GM_ADDR x, GM_ADDR y, int32_t totalLength) {
    int32_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    int32_t step = Simt::GetThreadNum() * Simt::GetBlockNum();
    for (int32_t i = idx; i < totalLength; i += step) {
        T val = x[i];
        if constexpr (Mode == 0) {
            y[i] = val + static_cast<T>(1);
        } else {
            y[i] = val * static_cast<T>(2);
        }
    }
}

// 调用处通过模板参数分发
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, int32_t mode, int32_t totalLength) {
    if (mode == 0) {
        AscendC::Simt::VF_CALL<OpFooSimt<T, 0>>(x, y, totalLength);
    } else {
        AscendC::Simt::VF_CALL<OpFooSimt<T, 1>>(x, y, totalLength);
    }
}
```

---

## 线程数选择策略

### 按算子类型选择

| 算子类型 | 建议线程数 | 原因 |
|---------|-----------|------|
| 搬运类算子 | 2048 / 1024 | 内存带宽受限，更多线程隐藏延迟 |
| 计算类算子 | 512 / 1024 | 寄存器压力大，需平衡并行度 |

### 按寄存器压力选择

寄存器压力随 VF 复杂度增加而增大，32 位索引可开更多线程：

| VF 复杂度 | uint32_t 线程数 | uint64_t 线程数 | 寄存器压力说明 |
|-----------|----------------|----------------|-------------|
| NONE / 1D | 1024 | 512 | 极低：仅循环变量 + shape |
| 2D | 1024 | 512 | 中低：1 级 UintDiv + 有限坐标变量 |
| 3D | 1024 | 512 | 中等：2 级 UintDiv + 更多坐标变量 |
| 4D | 1024 | 512 | 较高：3 级 UintDiv + 4 组坐标/偏移 |
| ND (5~8D) | 256 | 128 | 最高：运行时循环 + 可变维度 |

### 寄存器与线程数关系

| 线程数范围 | 每线程可用寄存器数 |
|-----------|-----------------|
| 1025~2048 | 16 |
| 513~1024 | 32 |
| 257~512 | 64 |
| 1~256 | 127 |

### 调优步骤

1. 初始值：uint32_t = 1024，uint64_t = 256（保守起点）
2. 编译验证：逐步提高 uint64_t 线程数（256->512），若编译通过说明寄存器充足
3. 性能验证：通过仿真或 NPU 实测确认无 register spill 导致的性能回退
4. 最终定值：取编译通过 + 性能不退化的最大值

### LAUNCH_BOUND 模板化

对于已通过 `IDX_T` 模板参数支持 32/64 位索引的 VF，LAUNCH_BOUND 也应按位宽模板化：

```cpp
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? 1024 : 512;

template <typename T, typename IDX_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS<IDX_T>) inline void OpSimt(...);
```