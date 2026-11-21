# FTZ / subnormal 边界（极端输入系统性失败）

## 形态指纹

- 误差**只出现在极端输入区间、常规值全对**：fp32 下 `exp(z)` 在 z∈(-104,-87.3) 产出 subnormal、z≤-104 为真 0、z>88.7 溢出 inf。
- 失败呈**系统性**：窗口内成百上千元素、MARE 接近 1（actual=0 vs golden 有限值）——单点 1 ulp 抖动不属本卡（见反例）。
- 典型触发：大数（±FLT_MAX 级）× subnormal 中间量 ≈ 有限正常值——golden 能给出，kernel 被刷 0。

## 根因机理（指令级 FTZ）

- **NPU 的 FTZ 是指令级的，不是向量单元整体的**：Exp 指令对 subnormal 输出刷 0；**Mul 对 subnormal 输入/输出均正确保留**（真机实测：exp(-87.4) 经 Mul 路径输出 1.1036e-38，与 CPU 逐位一致；探针可复验）。
- 推论：subnormal 结果可以经 Mul 从 normal 中间量产出——"NPU 物理上无法保留 subnormal"**不成立**。
- CPU golden（SLEEF/libm）保留 subnormal，故窗口内出现系统性偏差。

## 验证程序（定性硬件豁免前必须走完）

1. 先排除公式/顺序错误：已按 golden 的运算顺序/语义重算仍失败，且常规值全对，才进入 FTZ 定性。
2. 探针确认指令级行为：构造 z 使 exp(z) 落 subnormal 窗口，分别验证 Exp 输出（预期刷 0）与 Mul 对 subnormal 的保留性。
3. 查官方精度模式：FTZ_FALSE 类（如 ExpAlgo/DivAlgo 的 `PRECISION_*_FTZ_FALSE` 枚举）——详见 `references/hardware-precision-api.md` 的 Algo 枚举值汇总表，列出了所有指令的 FTZ_TRUE/FTZ_FALSE 可选值。存在即属可对齐手段，设计文档场景下为设计内首选。
4. 1-3 皆不成立，才允许按"剩余失败定性"红线流程定性硬件豁免——**"确实无法对齐"的验证清单即 1-3，缺项即 verdict 不成立**。

## 修复手段

- **首选：官方 FTZ_FALSE 类精度模式**（设计文档场景下为设计内手段；其内部即利用 Mul 保留 subnormal 的特性）。
- 避让重构：改公式使中间量不落入 subnormal 窗口（如 x·sigmoid(z) 形态：`e=exp(|z|)` 恒 ≥1，z≥0 取 `x·e/(1+e)`，z<0 取 `x/(1+e)`，数学等价、全程无 subnormal）。设计文档场景下须确认重构不偏离设计（"修复不偏离设计"红线）。
- **禁止**：凭"NPU 整体 FTZ、物理上无法对齐"的笼统假设直接定性硬件豁免；官方模式存在时手写其等价算法（属偏离设计）。

## 反例（像但不是）

- 偶发单元素、golden 与 actual 仅差 1 ulp → 刀锋点 golden 噪声，按"完成线非逐比特一致"红线处理，不走本卡。
- 同一输入多次运行结果不一致 → 更像 pipe-sync / uninitialized-memory，不是 FTZ。
