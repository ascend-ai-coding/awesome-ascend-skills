# 尾块未处理 / 非对齐 shape

## 形态指纹

- 误差集中在输出尾部（bad_index_pattern = tail）。
- 对齐 shape（如 32B 整数倍）比对通过，非对齐 shape（如 1000、1023）才错。
- first_bad_index 常出现在最后一个 tile/尾段起点附近。

## 根因机理

DataCopy 搬运和向量计算通常按 32B 对齐块处理，元素总数不是块整数倍时产生尾块。常见错误：

- 尾块直接按整块长度搬运/计算，越界读到脏数据或写越界；
- 尾块被直接丢弃不处理；
- 多核切分时最后一个核的尾段长度算错。

## 验证方法

1. 对齐 shape 跑一遍（如 1024）：通过 + 非对齐（如 1000）失败 → 高度疑似。
2. dump 输出看首个错点位置是否 = floor(N/block)*block（尾段起点）。
3. 输入全 1、输出应为常数的算子：尾部值异常一目了然。

## 修复 pattern

```cpp
// 尾块单独计算真实长度
uint32_t tailLen = totalLen - blockNum * blockLen;
if (tailLen > 0) {
    // 按 tailLen 搬运与计算，不要按 blockLen
    DataCopy(tailGm, tailUb, tailLen);
}

// 或用 DataCopyPad / 带 mask 的计算接口处理非对齐尾段
DataCopyPad(dstGm, srcUb, copyParams, padParams);
```

多核场景：逐核打印/检查每个核的 [start, len)，确认末核 len 未越界、无重叠。

## 反例（像但不是）

- 对齐 shape 也错 → 不是尾块，回 Step 4 重新匹配。
- 误差周期性出现在每块末尾（不只全局尾部）→ 更像块内处理或 broadcast 步长问题。
