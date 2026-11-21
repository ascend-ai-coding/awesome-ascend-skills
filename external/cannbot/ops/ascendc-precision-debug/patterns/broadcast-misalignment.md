# broadcast / 维度对齐错误

## 形态指纹

- **特定 shape 组合全错或大比例错**：如 [8,1]+[1,1024] 错、[8,1024]+[8,1024] 对。
- 误差不是"差一点"而是"值完全对不上"，且常呈规律的错位/重复图案。
- 多发于：多输入 broadcast、转置/reshape 后下标重映射、多维 stride 手算。

## 根因机理

广播或维度展开时，下标→内存偏移的映射算错：stride 顺序搞反、axis 选错、把 NCHW 当 NHWC 展平、reshape 后按错位的线性下标取数。取错数据当然"全错"。

## 验证方法

1. 构造可解码输入：如 input[i]=i（等差序列），看输出错位的数值能否反推出错误的映射关系。
2. 最小用例二分：[1,N]、[N,1]、[2,2] 逐个试，定位出错的具体维度组合。
3. 与 numpy 的 broadcast 结果逐 shape 对比（numpy 是广播语义的金标准）。

## 修复 pattern

```cpp
// 原则：下标映射集中到一个函数，单元测试覆盖各 shape 组合
inline uint32_t GetOffset(uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    return ((n * C + c) * H + h) * W + w;   // NCHW，别凭记忆散写
}

// 广播时显式按小维度取模，而不是默认连续读取
uint32_t srcIdx = dstIdx % broadcastLen;
```

reshape/transpose 场景：把"逻辑下标 → 物理偏移"画成表格核对，不要脑算。

## 反例（像但不是）

- 误差是"差一点"而非"对不上" → 不是映射错，查精度类模式。
- 只有全局尾部错 → 更像 tail-block-misaligned。
