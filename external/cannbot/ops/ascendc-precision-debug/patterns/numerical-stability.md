# 数值稳定性（exp/softmax/log/div 特殊值）

## 形态指纹

- 输出出现 `nan` / `inf`，而 golden 中对应位置是正常值。
- 大输入才触发（exp 溢出）、输入含 0 或相等元素才触发（log(0)、除零、x-x 后开方）。
- 多发于：softmax/layernorm 手写实现、exp/log/sqrt/div 组合算子。
- 误差只出现在极端输入、常规值全对的**系统性**失败 → 专卡 `patterns/ftz-subnormal-boundary.md`（FTZ/subnormal 边界）。

## 根因机理

- `exp(x)`：x 较大（fp16 下 x>11 即溢出）直接算必炸，softmax 必须先减 max。
- `log(x)`：x=0 → -inf；x<0 → nan。
- 除法：除数为 0 或近 0 未加 eps 保护。
- 方差开方：浮点误差使方差出现极小负值，sqrt(负数) → nan。

## 验证方法

1. 构造触发输入单独验证：全大值（如全 100）→ 查 exp 溢出；含 0 → 查 log/div；相等行 → 查方差。
2. 对可疑中间结果 dump，定位首个出现 nan/inf 的 stage（nan 会传播，首个 nan 点才是根因点）。

## 修复 pattern

```cpp
// softmax：先减 max
// 反例：Exp(x); Sum; Div
// 正例：
Max(rowMax, x, ...);        // 每行最大值
Sub(x, x, rowMax, ...);     // x - max，指数输入压到 <=0
Exp(x, x, ...);
Sum(rowSum, x, ...);
Div(y, x, rowSum, ...);

// 除法/开方加 eps
Adds(denom, denom, 1e-6f, len);   // 防除零
// 方差非负保护
Maxs(var, var, 0.0f, len);        // 或 adds eps 后再 Sqrt
Sqrt(std, var, len);
```

优先用官方高阶 API（AscendC::SoftMax、LayerNorm 等），其内部已做稳定性处理。

## 反例（像但不是）

- golden 里也含 nan → 输入本身带 nan，先查输入构造，不是算子问题。
- nan 出现在多次运行结果不同的场景 → 更像 uninitialized-memory（脏数据）。
