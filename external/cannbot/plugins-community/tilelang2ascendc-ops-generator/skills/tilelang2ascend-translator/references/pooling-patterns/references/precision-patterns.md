# Precision Patterns for Pooling Operators

> **定位**：通用（fp32 累加 + divisor 三级策略前向反向共享；反向须逐输出位置除 divisor，见 `ops/tilelang-op-design/references/pooling/backward-patterns.md` §4）。

## 核心策略

### fp32 累加（强制）

**所有 pooling 累加必须在 fp32 中进行**，即使输入/输出是 fp16。

```cpp
// 累加器永远是 fp32
AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;  // fp32 accumulator
AscendC::LocalTensor<float> accLocal_;

// fp16 路径: Cast up on load
PrepareInputTensorHelper(inLocalFp32_, inLocal_, inCastBuf_, W_ * C_);
// → fp16→fp32 Cast (CAST_NONE = round-half-to-even)

// 在 fp32 中累加
AscendC::Add(accLocal_[ow * C_], accLocal_[ow * C_], inLocalFp32_[iw * C_], C_);

// Cast down on store
FinalizeOutputTensorHelper(outLocal_, outLocalFp32_, OW_ * C_);
// → fp32→fp16 Cast (CAST_NONE)
```

### Cast 辅助函数

```cpp
// Load: fp32 → ReinterpretCast (zero-cost), fp16 → Cast to fp32
template <typename T>
__aicore__ inline void PrepareInputTensorHelper(
    AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<T> &src,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf, int32_t count)
{
    if constexpr (std::is_same_v<T, float>) {
        dst = src.template ReinterpretCast<float>();  // 零开销
    } else {
        dst = castBuf.Get<float>();
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// Store: fp32 → ReinterpretCast (zero-cost), fp16 → allocate cast buffer
template <typename T>
__aicore__ inline void FinalizeOutputTensorHelper(
    AscendC::LocalTensor<T> &out, AscendC::LocalTensor<float> &src, int32_t count)
{
    if constexpr (!std::is_same_v<T, float>) {
        AscendC::Cast(out, src, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
```

## Divisor 三级策略

Pooling 的归一化有三种 divisor 策略，按优先级排列：

### 策略 1: divisor_override（最高优先级）

```cpp
if (divisorOverride_ > 0) {
    const float invDiv = 1.0f / static_cast<float>(divisorOverride_);
    AscendC::Muls(outLocalFp32_, accLocal_, invDiv, OW_ * C_);
}
```

用户显式指定 divisor 时使用。单指令 `Muls(C)` 覆盖整行，最快。

### 策略 2: count_include_pad && !ceil_mode → KD*KH*KW（快路径）

```cpp
else if (countIncludePad_ > 0) {
    if (ceilMode_ == 0) {
        // ceil_mode=False 时窗口永不右截断，divisor 恒为 KD*KH*KW
        const float invDiv = 1.0f / static_cast<float>(KD_ * KH_ * KW_);
        AscendC::Muls(outLocalFp32_, accLocal_, invDiv, OW_ * C_);
    }
```

**关键事实**: ceil_mode=False 时，最后一个输出位置计算为 `(size + 2*pad - k) / stride + 1`，其左边界一定 ≤ size+pad-k，窗口不会超出右边界，因此 divisor 恒等于 KD*KH*KW。只有 ceil_mode=True 才可能出现右截断。

### 策略 3: ceil_mode + count_include_pad → 逐位置 ComputePaddedDivisor

```cpp
    else {
        for (int ow = 0; ow < OW_; ++ow) {
            const int div = ComputePaddedDivisor(od, oh, ow);
            const float invDiv = 1.0f / static_cast<float>(div);
            AscendC::Muls(outLocalFp32_[ow * C_], accLocal_[ow * C_], invDiv, C_);
        }
    }
}
```

### 策略 4: count_include_pad=False → 逐位置 ComputeDivisor

```cpp
else {
    for (int ow = 0; ow < OW_; ++ow) {
        const int div = ComputeDivisor(od, oh, ow);
        const float invDiv = (div > 0) ? (1.0f / static_cast<float>(div)) : 0.0f;
        AscendC::Muls(outLocalFp32_[ow * C_], accLocal_[ow * C_], invDiv, C_);
    }
}
```

## ComputeDivisor vs ComputePaddedDivisor

### ComputeDivisor: 有效位置计数（count_include_pad=False）

```cpp
// 窗口与有效输入区域的交集 → valid count
dStart = max(0, od*sD - pD)        // clamp start at 0
dEnd   = min(D, dStart + kD)       // clamp end at D
valid  = dEnd - dStart
return validD * validH * validW
```

### ComputePaddedDivisor: Padded-Clamped 窗口大小（count_include_pad=True, ceil_mode=True）

```cpp
// 窗口与 PADDED 输入区域的交集 → pool_size
dStart = od*sD - pD                // NO clamp at 0 (padding counts)
dEnd   = min(D+pD, dStart + kD)    // clamp at padded boundary
poolD  = dEnd - dStart
return poolD * poolH * poolW
```

**差异**: ComputePaddedDivisor 对起始位置**不**做 0-clamp（pad 区域计入 divisor），但对超出 padded input 的右侧做截断。这匹配 PyTorch 的 `avg_pool3d_out_frame` 语义。

## ceil_mode divisor bug 复现与修复

### Bug 表现
```
场景: ceil_mode=True, count_include_pad=True, shape=(2,16,9,10,11), k=3, s=2, p=1
错误: 输出右边界元素 max_abs=0.209，1/6 元素（最后一维输出边界）不匹配
根因: kernel 用固定 KD*KH*KW=27 做 divisor，但 ceil_mode 下右边界窗口被截断
      如 H 边界 pool_h=2 而非 3 → 正确 divisor=18 而非 27 → 输出偏小 27/18=1.5x
```

### 修复
```cpp
// Host: 传入 ceilMode flag
int32_t ceilMode = ceil_mode ? 1 : 0;

// Kernel: cip=True 时按 ceilMode 分支
if (countIncludePad_ > 0) {
    if (ceilMode_ == 0) {
        // 单 Muls 快路径（divisor=KD*KH*KW）
    } else {
        // 逐 ow 用 ComputePaddedDivisor
    }
}
```

### 验证方法
```
标准 6 场景 → 均 ceil_mode=False，无法暴露此 bug
必须追加 branch 测试:
  ceil_mode=True  + count_include_pad=True
  ceil_mode=True  + count_include_pad=False
  ceil_mode=False + count_include_pad=True  (已覆盖)
  ceil_mode=False + count_include_pad=False (已覆盖)
  + 非对称 kernel (KD≠KH≠KW) + 非对称 stride + padding
```
