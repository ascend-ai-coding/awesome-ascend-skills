# 维度合并与大 BLOCK 累加（归一化算子专用）

> 适用于 BatchNorm / LayerNorm / GroupNorm / InstanceNorm / RMSNorm / Softmax 等归一化算子。

## 适用条件

- 算子类型为 BatchNorm / LayerNorm / GroupNorm / InstanceNorm / RMSNorm / Softmax。
- 代码中存在对 stats unit（group / row / channel）内元素的归约操作。
- 当前实现使用嵌套循环或多通道分块累加。

## 典型代码特征（问题模式）

```python
# 特征 1：嵌套循环处理连续维度
for c in range(c_start, c_end):
    for hw_block in range(0, L, BLOCK_HW):
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals)  # 小量多次标量累加

# 特征 2：mask 覆盖率过低
BLOCK_HW = 256
L = H * W  # 若 L=16，mask 覆盖率仅 6.25%

# 特征 3：标量累加次数远大于向量化加载次数
# 如：3584 次标量累加 vs 14 次向量化加载
```

## 判断逻辑

1. 检查 stats kernel 中是否存在嵌套循环处理连续维度。
2. 检查 `tl.load` 的 mask 覆盖率是否 < 50%。
3. 检查标量累加次数是否远大于向量化加载次数。

若以上任一条件满足，则命中本优化点。

## 优化方向

- 将连续维度（如 `H × W`）合并为一个长维度 `L`，减少嵌套循环层数。
- 使用更大的 `BLOCK` 一次性覆盖 stats unit 内的大部分或全部元素，提高 mask 覆盖率。
- 用向量化累加器替代标量累加器，把多次小量 `tl.sum` 合并为少量大向量求和。

## 预期收益

- 减少标量循环开销和冗余的 `tl.sum` 调用。
- 提高 Vector 单元利用率，降低归一化算子的总体延迟。
