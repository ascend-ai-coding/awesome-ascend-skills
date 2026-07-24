---
id: P100
bottlenecks: [scalar_compute, scalar_loading, compute_bound]
op_families: [elementwise, special, index_scatter, attention]
complexity: L1
conflicts_with: []
synergizes_with: [P97, P95]
requires: []
has_preconditions: true
has_playbook: false
---

# P100: GatherMask 替代逐元素 SetValue/GetValue (GatherMask Replacement)

## 核心思想
当需要从连续 tensor 中按固定间隔取元素重组数据时（如 RoPE 奇偶分拆、解交织、矩阵转置等），优化前用 `for` 循环逐元素 `src.GetValue(i)` → `dst.SetValue(j, val)` 操作，每条调用仅处理 1 个元素（O(n) 次 Scalar ↔ Vector 数据通路切换）。优化后用 AscendC `GatherMask` API，在 Vector 单元上一次完成间隔采集，O(1) 条向量指令替代 O(n) 次标量操作，且零拷贝（直接从源 tensor 的偏移视图上操作）。

## 代码骨架

```cpp
// === 改造前（基线）：逐元素 SetValue/GetValue 解交织 ===
// 从 kvF32[rms_size_:] 中分离偶数位和奇数位
int half = rope_size_ / 2;
LocalTensor<float> kF32 = ropeBuf_.Get<float>();

for (int i = 0; i < half; ++i) {
    kF32.SetValue(i, kvF32.GetValue(rms_size_ + 2 * i));        // 偶数位
}
for (int i = 0; i < half; ++i) {
    kF32.SetValue(half + i, kvF32.GetValue(rms_size_ + 2 * i + 1)); // 奇数位
}
// 2 × half 次 SetValue + GetValue Scalar 调用，每次仅处理 1 个元素

// === 改造后：GatherMask 解交织（2 条指令完成） ===
uint64_t rsvdCnt = 0;
LocalTensor<float> ropeSrc = kvF32[rms_size_];  // 零拷贝偏移视图

// deinterleave=1: 从 offset=1 开始，步长 1 → 取偶数索引位
GatherMask(kF32, ropeSrc,
    static_cast<uint8_t>(1), true,               // deinterleave=1, isReverse=true
    static_cast<uint32_t>(rope_size_),
    {1, 1, 8, 0}, rsvdCnt);

// deinterleave=2: 从 offset=2 开始，步长 2 → 取奇数索引位  
GatherMask(kF32[half], ropeSrc,
    static_cast<uint8_t>(2), true,               // deinterleave=2, isReverse=true
    static_cast<uint32_t>(rope_size_),
    {1, 1, 8, 0}, rsvdCnt);
PipeBarrier<PIPE_V>();
// 2 条向量指令替代 2×half 次标量操作
```

## GatherMask 参数说明

```cpp
GatherMask(
    dst,              // 输出 tensor
    src,              // 源 tensor（可使用零拷贝偏移视图）
    deinterleave,     // uint8_t: 间隔步长。1=连续, 2=隔1取1, N=隔N-1取1
    isReverse,        // bool: 是否反向序
    count,            // uint32_t: 要收集的元素总数
    repeatMaskParams, // AscendC::RepeatMaskParams: {repeatTimes, dstBlkStride, srcBlkStride, dstRepStride, srcRepStride}
    rsvdCnt           // uint64_t&: 保留参数
);
```

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `deinterleave` | `1` | 连续取（偶数位用 deinterleave=1 + offset=1） |
| | `2` | 隔 1 取 1（奇数位用 deinterleave=2 + offset=2） |
| `isReverse` | `true` | 默认传 true |
| `repeatMaskParams` | `{1, 1, 8, 0}` | 常用默认值：单 repeat，dst/src block stride=1/8 |

## 关键修改点

1. **零拷贝偏移**：`LocalTensor<T> ropeSrc = src[offset]` 创建视图，GatherMask 直接在原 tensor 上操作
2. **GatherMask 后加 PipeBarrier**：`PipeBarrier<PIPE_V>()` 确保 Vector 写入完成后再被后续指令读取
3. **dst 拆分**：利用 `dst[half]` 切片将两段 GatherMask 结果写入同一 tensor 的不同区域
4. **预期收益**：元素数 ≥ 128 时 5-20x 加速；< 64 时收益递减但通常仍优于标量

## 适用性检测 (grep)

```bash
# 检测逐元素 SetValue/GetValue 模式（主反模式）
grep -nE "for.*SetValue.*GetValue|for.*GetValue.*SetValue" op_kernel/*.cpp

# 检测解交织/奇偶拆分模式
grep -nE "even.*odd|odd.*even|interleave|deinterleave|2\s*\*\s*i\s*\+" op_kernel/*.cpp

# 确认 GatherMask 是否已存在
grep -nE "GatherMask" op_kernel/*.cpp
```

## 常见陷阱

⚠️ GatherMask 后必须加 `PipeBarrier<PIPE_V>()`，否则 dst 数据可能被后续指令读为旧值
⚠️ `deinterleave` 是 uint8_t 类型，传入 int 需显式 cast：`static_cast<uint8_t>(1)`
⚠️ `count` 参数应传元素总数而非 half；两段 GatherMask 的 count 通常相同（各取一半元素）
⚠️ GatherMask 从 src 的当前位置开始取，需确保 `ropeSrc = src[offset]` 偏移正确
⚠️ `rsvdCnt` 参数必须声明为 `uint64_t` 并初始化为 0

## 代码搜索关键词

```bash
grep -nE "GatherMask|deinterleave|rsvdCnt|GatherMaskParams" op_kernel/*.cpp
```

## 来源

- tmp.md 优化点2：GatherMask 替代 SetValue/GetValue 逐元素解交织（RoPE 场景），2 条向量指令替代 2×half 次标量操作
