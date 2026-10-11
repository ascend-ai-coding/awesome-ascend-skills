# 数值稳定性核心知识

> 数值稳定性的理论基础、问题根源与技术原理

---

## 1 核心问题

### 1.1 四大数值风险

| 风险 | 表现 | 典型场景 | 后果 |
|------|------|---------|------|
| **上溢** | 中间结果 > 浮点表示上限 | `exp(100)` 在 fp32 | Inf |
| **下溢** | 中间结果 < 浮点表示下限 | `exp(-100)` → 0 | 精度丢失 |
| **灾难性相消** | 两个接近的大数相减 | `E[x²] - E[x]²` 当 x 紧密分布 | 有效位损失 |
| **累积误差** | 长序列求和舍入累积 | 10⁶ 个 fp16 求和 | 结果不可靠 |

### 1.2 浮点精度约束

| 类型 | 有效位 | Machine Epsilon | 安全累加长度 |
|------|-------|----------------|------------|
| fp32 | ~7 位十进制 | 1.2e-7 | < 10⁶ |
| fp16 | ~3 位十进制 | 9.8e-4 | < 10³ |
| bf16 | ~2 位十进制 | 7.8e-3 | < 10³ |

**推论**: fp16/bf16 输入的归约算子，累加器必须 ≥ fp32

---

## 2 六大稳定化技术

### T1: Max-Shift + LogSumExp

**解决问题**: `log(Σexp(x))` 或 `Softmax = exp(x)/Σexp(x)` 上溢

**数学原理**:
```
log(Σexp(xᵢ)) = m + log(Σexp(xᵢ - m)),  其中 m = max(x)
```

**为什么有效**:
- 右侧 `exp(xᵢ - m) ∈ (0, 1]`（至少一项 = 1）
- 避免 exp 参数过大
- Softmax 分母 ≥ 1，**不需要 eps**（常见误区）

**必要代价**: 两遍归约（或 online 单遍维护 running-max/sum）

**Backward 稳定性**（训练算子）:

LogSoftmax Backward:
```
Forward: y = x - m - log(sum(exp(x-m)))
Backward: grad_x = grad_y - exp(y) * sum(grad_y)
```

Softmax Backward:
```
Forward: y = exp(x-m) / sum(exp(x-m))
Backward: grad_x = y * (grad_y - sum(grad_y * y))
```

**稳定实现关键**:
- **复用 forward 输出**: LogSoftmax 用 `exp(y)` 而非重算；Softmax 直接用 `y`
- **sum 使用 fp32 累加器**（T3）: `sum(grad_y)` 或 `sum(grad_y * y)` 必须 fp32
- **避免重复计算**: 预先计算 sum，再广播到每个元素

---

### T2: Eps 保护分母

**解决问题**: 归一化分母趋 0，除法放大误差或产生 NaN

**数学形式**:
```
1/sqrt(var)       →  rsqrt(var + eps)
x/||x||           →  x / max(||x||, eps)
(k + α·Σa²)^(-β)  →  保持 k > 0
```

**Eps 取值原则**:
- fp32: 1e-12 ~ 1e-5（LayerNorm 常用 1e-5）
- fp16/bf16: 1e-7 ~ 1e-4（精度有限，需更大 eps）
- 来源: spec.attribute 或算子固定值，禁止硬编码

---

### T3: 高精度累加器

**解决问题**: fp16/bf16 累加误差快速累积

**原理**: 累加器提升到 fp32（或更高）
```
sum_fp16 += x_fp16[i]              // ❌ 误差累积
sum_fp32 += Cast<fp32>(x_fp16[i])  // ✅ 稳定
```

**分级策略**:

| 长度 | 跨度 | 策略 | 开销 |
|------|------|------|------|
| < 10⁴ | 任意 | fp32 累加 | 无 |
| 10⁴ ~ 10⁶ | < 10¹⁰ | fp32（可选 pairwise） | 无 |
| 10⁴ ~ 10⁶ | ≥ 10¹⁰ | Kahan 补偿 | +1 fp32 buffer |
| > 10⁶ | ≥ 10¹⁰ | 强烈推荐 Kahan | +1 fp32 buffer |

**Kahan 补偿求和**: 维护 compensation 追踪舍入误差

---

### T4: Welford / 两遍均值方差

**解决问题**: `var = E[x²] - E[x]²` 的灾难性相消

**Welford 单遍递推**:
```
δ = x - mean_old
mean_new = mean_old + δ/n
M2_new = M2_old + δ·(x - mean_new)
var = M2 / n
```

**两遍法**（更简单）:
```
Pass 1: mean = Σx / n
Pass 2: var = Σ(x - mean)² / n
```

**何时不需要**: RMSNorm 只计算 `E[x²]`，无相消风险，单遍 fp32 累加即可

**判据**: `needs_welford = (需要方差) AND (非纯均方)`

---

### T5: 值域保护

**解决问题**: exp/log/pow 定义域错误或上下溢

| 操作 | fp32 | fp16 | bf16 | 原因 |
|------|------|------|------|------|
| log | `log(max(x, 1e-12))` | `log(max(x, 1e-7))` | `log(max(x, 1e-7))` | x ≤ 0 → NaN |
| exp | `exp(clamp(x, -88, 88))` | `exp(clamp(x, -10, 10))` | `exp(clamp(x, -10, 10))` | fp32/fp16: 上下溢；bf16: 精度约束 |
| sqrt | `sqrt(max(x, 0))` | 同左 | 同左 | x < 0 → NaN |
| rsqrt | `rsqrt(max(x, 1e-12))` | `rsqrt(max(x, 1e-7))` | `rsqrt(max(x, 1e-7))` | 定义域+除零 |
| log(0) | `Select(x==0, 0, log(x))` | 同左 | 同左 | 特定位置 mask |

**Dtype 差异**: fp16/bf16 精度低，eps 需更大才有效

---

### T6: 分段公式

**解决问题**: 单一公式在不同值域有不同数值风险

**典型案例**:

| 算子 | 原公式 | 稳定公式 | 原因 |
|------|-------|---------|------|
| Sigmoid | `1/(1+exp(-x))` | `x≥0: 1/(1+exp(-x))`<br>`x<0: exp(x)/(1+exp(x))` | 确保 exp 参数 ≤ 0 |
| log1p_exp | `log(1+exp(x))` | `x≥0: x+log(1+exp(-x))`<br>`x<0: log(1+exp(x))` | 等价 logsumexp(0,x) |

**实现**: Select/Mux 避免分支开销，预计算公共项

---

## 3 技术组合规律

### 3.1 依赖关系

- **T1 几乎必带 T3**: log 域归约涉及求和 → fp32 累加器
- **T2 几乎必带 T3**: 归一化需要求和/范数 → fp32 累加器
- **T4 是 T2 的可选增强**: 仅当需要均值+方差（非纯均方）
- **T5/T6 跨场景**: 任何场景的 Compute 遇到 exp/log/pow 都可能需要

### 3.2 场景决策树

```
formula 含 log/exp/softmax/logsumexp？
├─ YES → logdomain 场景 (T1+T3，可能+T5)
└─ NO  → formula 含归一化分母？
          ├─ YES → normdenom 场景 (T2+T3，可能+T4)
          └─ NO  → 仅 T5/T6，叠加主范式
```

---

## 4 常见误区

### ❌ 误区 1: Softmax 分母需要 eps
**错误**: `y = exp(x-m) / (sum(exp(x-m)) + eps)`  
**正确**: 分母已 ≥ 1（含 exp(0)=1），无需 eps

### ❌ 误区 2: RMSNorm 需要 Welford
**错误**: 使用 Welford 算法  
**正确**: 只计算 `E[x²]`，无相消风险，单遍 fp32 即可

### ❌ 误区 3: Elementwise 需要 fp32 累加器
**错误**: Sigmoid/Log/Sqrt 需要 fp32 累加器  
**正确**: 无归约，仅需 T5 值域保护

### ❌ 误区 4: 所有 fp16 都升 fp32
**错误**: 所有 fp16 算子 Cast→fp32  
**正确**: 仅涉及**归约累加**的算子需要 fp32 累加器

---

## 5 理论基础

### 5.1 浮点运算误差界

对浮点运算 `⊕`，存在相对误差 ε ≤ machine epsilon:
```
fl(a ⊕ b) = (a ⊕ b)(1 + ε)
```

推论: n 次加法累积误差约 `n·ε`

### 5.2 条件数

```
κ(x) = |x·f'(x) / f(x)|
```

κ >> 1 时问题病态，需稳定化。例: `f(x)=exp(x)`, `κ=|x|` → x 大时需 max-shift

### 5.3 数学等价 ≠ 数值等价

| 不稳定形式 | 稳定形式 | 变换依据 |
|-----------|---------|---------|
| `log(Σexp(x))` | `m + log(Σexp(x-m))` | 对数恒等式 |
| `E[x²] - E[x]²` | Welford 递推 | 重排求和 |
| `1/sqrt(var)` | `rsqrt(var + eps)` | 分母保护 |

**黄金法则**: 理论等价不保证数值等价
