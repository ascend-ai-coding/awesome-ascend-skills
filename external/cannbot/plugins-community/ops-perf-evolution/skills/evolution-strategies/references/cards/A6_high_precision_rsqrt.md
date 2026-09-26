---
id: A6
bottlenecks: []
op_families: [normalization]
complexity: L0
conflicts_with: []
synergizes_with: [A1, A2]
has_preconditions: true
has_playbook: true
---

# A6: High-Precision rsqrt via Newton-Raphson (高精度rsqrt)

## 核心思想
Layer Norm的归一化需要计算1/sqrt(var + eps)。专家实现使用Newton-Raphson迭代提高rsqrt精度：初始近似y0 = rsqrt(x)（硬件指令，精度约8bit），第一次迭代y1 = y0 * (1.5 - 0.5 * x * y0^2)，第二次迭代y2 = y1 + 0.5 * y1 * (1 - x * y1^2)。相比单次sqrt+div，迭代方法避免了除法指令的高延迟，同时获得更高的精度（接近FP32全精度）。

## 代码骨架

// === 改造前（基线）===
```cpp
// lingxi-code - 传统方法
AscendC::Adds(rmsLocal, meanSqLocal, this->eps, 1);
AscendC::Sqrt(rmsLocal, rmsLocal, 1);
float rmsVal = rmsLocal.GetValue(0);
AscendC::Reciprocal(rmsLocal, rmsLocal, 1);
float invRms = rmsLocal.GetValue(0);
```

// === 改造后（专家模式）===
```cpp
// 专家实现 - 牛顿迭代
static constexpr float SCALAR1 = -0.5;
static constexpr float SCALAR2 = 1.5;

Div(r, one, var, pregLoop);   // 初始近似
Sqrt(y, r, pregLoop);
Muls(t, var, SCALAR1, pregLoop);
Mul(t, t, y, pregLoop);
Mula(t1, t, y, pregLoop);
Mul(rstd, y, t1, pregLoop);

// 特殊值处理
CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
Select(rstd, scalarZero, rstd, cmpRegZero);
CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
Select(rstd, scalarInf, rstd, cmpRegInf);
```

## 关键修改点

1. 预期收益: 更高的精度和更好的性能，特殊值处理确保数值稳定性; 更高的数值精度和指令吞吐，正确处理特殊边界值

## 常见陷阱

⚠️ 代码更复杂，需要理解牛顿迭代原理
⚠️ 增加约6条Vector指令，适合训练场景
⚠️ 增加约8-10条Vector指令

## 代码搜索关键词

```bash
grep -n "SCALAR1\|SCALAR2" op_kernel/*.cpp op_host/*_tiling.cpp
```
