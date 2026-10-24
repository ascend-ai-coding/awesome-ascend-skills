# 数值稳定性实现模式

> 场景识别、数据流结构、Buffer 配置与 Compute 实现要点
>
> 本文是 NumericalStable 范式入口，只讲「怎么做」。技术原理、各 dtype 的 eps/clamp
> 取值、常见误区见 [knowledge.md](knowledge.md)——**落地 T1–T6 前必读对应条目**，
> 不得凭记忆自创稳定化公式或 eps 取值。

---

## 1 三步决策流程

```
输入: spec.yaml 的 formula + numerical_stability + dtype_policy

Step 1 — 技术识别（可多选）
  formula 含 exp/log/softmax/logsumexp？           → T1
  formula 含归一化分母 (rsqrt, x/‖x‖)？            → T2
  长轴求和 + fp16/bf16？                          → T3
  formula 含 E[x²]-E[x]² 或需均值+方差？          → T4
  exp/log/pow 有定义域/溢出风险？                 → T5
  单一公式跨值域有数值风险？                       → T6

Step 2 — 场景归属
  T1 存在 → logdomain
  T2 存在 → normdenom
  仅 T5/T6 → 无独立场景，叠加主范式 Compute

Step 3 — 参数决策
  accumulator_dtype = fp32 if (T3 且 input∈{fp16,bf16})
  needs_welford = (T4 且 需方差 且 非纯均方)
  logsumexp_pass = online if 归约轴 > UB容量 else two_pass
  needs_dbuf = (T1 且 末尾消费 d=x-m)
```

---

## 2 Logdomain 场景（T1+T3）

### 2.1 数据流

**Two-Pass**:
```
Pass 1: m = ReduceMax(x)              // fp32 或输入精度
Pass 2: d = x - m                     // fp32，可选存 dBuf
        e = Exp(d)                    // fp16/输入精度
        s = ReduceSum<fp32>(e)        // T3: fp32 累加器
        y = 末尾表达式(d, e, s)
```

**Online（长轴优化）**:
```
running_max = -Inf; running_sum = 0
for chunk:
    chunk_max = ReduceMax(chunk)
    running_sum *= Exp(running_max - chunk_max)  // 修正旧 sum
    running_max = Max(running_max, chunk_max)
    running_sum += ReduceSum(Exp(chunk - running_max))
```

### 2.2 Buffer 配置

| Buffer | 精度 | 用途 | 必需性 |
|--------|------|------|--------|
| mBuf | fp32 | 存 max | ✅ 必需 |
| sBuf | fp32 | 存 sum(exp(x-m)) | ✅ 必需 |
| dBuf | fp32 | 存 d=x-m | ⚠️ 末尾消费 d 时需要 |
| expBuf | fp16 | 存 exp(d) | ⚠️ 末尾多次使用时需要 |

**dBuf 判定**:
- LogSoftmax: `y = d - log(s)` → **需要 dBuf**（末尾用 d）
- Softmax: `y = e / s` → **不需要 dBuf**（末尾用 e）

### 2.3 Compute 伪码

```cpp
// [T1] Pass 1: max-shift
LocalTensor<fp32> mBuf = ReduceMax<fp32>(xBuf);

// [T1+T3] Pass 2: LogSumExp
if constexpr (NeedsDBuf) {
    LocalTensor<fp32> dBuf = Sub<fp32>(xBuf, mBuf);
    LocalTensor<T> expBuf = Exp(dBuf);
    LocalTensor<fp32> sBuf = ReduceSum<fp32>(expBuf);  // [T3] fp32 累加

    // LogSoftmax: y = d - log(s)
    yBuf = Sub(dBuf, Log(sBuf));
} else {
    LocalTensor<T> tmpBuf = Sub(xBuf, mBuf);
    LocalTensor<T> expBuf = Exp(tmpBuf);
    LocalTensor<fp32> sBuf = ReduceSum<fp32>(expBuf);  // [T3] fp32 累加

    // Softmax: y = exp(d) / s
    yBuf = Div(expBuf, sBuf);
}
```

**关键点**:
- Softmax 分母 ≥ 1，**不加 eps**
- s 必须 fp32 累加器（T3）
- 选择 two_pass 或 online 看归约轴长度 vs UB 容量

### 2.4 Backward 实现（训练算子）

**LogSoftmax Backward**:
```cpp
// Input: grad_y (梯度输入), y (forward 输出)
// Output: grad_x

// [T3] Pass 1: fp32 累加 sum(grad_y)
LocalTensor<fp32> sum_grad_y = ReduceSum<fp32>(grad_yBuf);

// Pass 2: 计算梯度
LocalTensor<T> exp_y = Exp(yBuf);  // 复用 forward 的 y
grad_xBuf = Sub(grad_yBuf, Mul(exp_y, sum_grad_y));  // grad_y - exp(y)*sum
```

**Softmax Backward**:
```cpp
// Input: grad_y (梯度输入), y (forward 输出)
// Output: grad_x

// [T3] Pass 1: fp32 累加 sum(grad_y * y)
LocalTensor<fp32> sum_grad_y_mul_y = ReduceSum<fp32>(Mul(grad_yBuf, yBuf));

// Pass 2: 计算梯度
grad_xBuf = Mul(yBuf, Sub(grad_yBuf, sum_grad_y_mul_y));  // y*(grad_y - sum)
```

**Buffer 配置**:

| Buffer | 精度 | 用途 | 说明 |
|--------|------|------|------|
| yBuf | fp16 | 存 forward 输出 | 必须保留，backward 复用 |
| sum_buf | fp32 | 存归约结果 | T3: fp32 累加器 |
| grad_xBuf | fp16 | 输出梯度 | 与输入精度一致 |

**关键点**:
- **必须复用 forward 输出 y**，避免重复计算 exp
- **归约必须 fp32 累加器**（T3），sum(grad_y) 或 sum(grad_y*y)
- LogSoftmax backward 需要计算 `exp(y)`，Softmax backward 直接用 `y`

---

## 3 Normdenom 场景（T2+T3）

### 3.1 数据流（三种子场景）

**场景 A: 仅均方（RMSNorm, L2Normalize）**
```
sum_x2 = ReduceSum<fp32>(x²)       // [T3] fp32 累加
rms = Sqrt(sum_x2 / N + eps)       // [T2] eps 保护
y = x / rms  或  x * rsqrt(...)
```

**场景 B: 两遍均值+方差（LayerNorm）**
```
Pass 1: mean = ReduceSum<fp32>(x) / N
Pass 2: var = ReduceSum<fp32>((x-mean)²) / N
rstd = Rsqrt(var + eps)            // [T2] eps 保护
y = (x - mean) * rstd * γ + β
```

**场景 C: Welford 单遍（LayerNorm 优化）**
```
foreach x[i]:
    δ = x[i] - mean_old
    mean_new = mean_old + δ / n
    M2_new = M2_old + δ * (x[i] - mean_new)
var = M2 / n
rstd = Rsqrt(var + eps)
y = (x - mean) * rstd * γ + β
```

### 3.2 Buffer 配置

| 场景 | 新增 Buffer | 精度 | 用途 |
|------|------------|------|------|
| 仅均方 | 无 | - | 累加器融入主范式 sum |
| Welford | meanBuf, m2Buf | fp32 | 运行均值和 M2 |
| 两遍 | meanBuf | fp32 | 存 Pass1 的 mean |

### 3.3 Compute 伪码

```cpp
// ---- 场景 A: 仅均方（RMSNorm）----
if constexpr (!NeedsWelford) {
    LocalTensor<fp32> sum_x2 = ReduceSum<fp32>(Mul(xBuf, xBuf));  // [T3]
    fp32 rms = Sqrt(sum_x2 / N + eps);  // [T2] eps 保护
    yBuf = Mul(xBuf, Rsqrt(sum_x2 / N + eps));
}

// ---- 场景 B: 两遍（LayerNorm）----
else if constexpr (UseTwoPass) {
    // Pass 1
    LocalTensor<fp32> meanBuf = ReduceSum<fp32>(xBuf) / N;  // [T3]

    // Pass 2
    LocalTensor<fp32> centered = Sub<fp32>(xBuf, meanBuf);
    LocalTensor<fp32> varBuf = ReduceSum<fp32>(Mul(centered, centered)) / N;

    // [T2] eps 保护
    fp32 rstd = Rsqrt(varBuf + eps);
    yBuf = Mul(Mul(centered, rstd), gammaBuf);  // 简化，实际含 beta
}

// ---- 场景 C: Welford 单遍 ----
else {
    LocalTensor<fp32> meanBuf = 0, m2Buf = 0;
    for (int i = 0; i < N; i++) {
        fp32 delta = xBuf[i] - meanBuf;
        meanBuf += delta / (i + 1);
        m2Buf += delta * (xBuf[i] - meanBuf);
    }
    fp32 var = m2Buf / N;
    fp32 rstd = Rsqrt(var + eps);  // [T2]
    yBuf = Mul(Mul(Sub(xBuf, meanBuf), rstd), gammaBuf);
}
```

**关键点**:
- eps 来自 TilingData 或编译期常量，禁止硬编码
- RMSNorm 不需要 Welford（常见误区）
- 所有累加器必须 fp32（T3）

---

## 4 值域保护模式（T5/T6）

### 4.1 T5: 逐点保护

```cpp
// [T5 log_eps]
yBuf = Log(Max(xBuf, eps));  // eps: fp32=1e-12, fp16=1e-7

// [T5 exp_clamp]
yBuf = Exp(Clamp(xBuf, lo, hi));  // fp32:(-88,88), fp16:(-10,10)

// [T5 sqrt_eps]
yBuf = Sqrt(Max(xBuf, 0.0f));

// [T5 zero_mask] 特定位置
yBuf = Select(xBuf == 0.0f, 0.0f, Log(xBuf));
```

### 4.2 T6: 分段公式

**Sigmoid**:
```cpp
// x≥0: 1/(1+exp(-x)), x<0: exp(x)/(1+exp(x))
// 两个分支都复用 exp_neg_abs，确保全值域 exp 参数 ≤ 0
LocalTensor<T> exp_neg_abs = Exp(Neg(Abs(xBuf)));
yBuf = Select(
    xBuf >= 0,
    Reciprocal(Add(1.0f, exp_neg_abs)),
    Mul(exp_neg_abs, Reciprocal(Add(1.0f, exp_neg_abs)))
);
```

**SoftMarginLoss**:
```cpp
// log(1+exp(-t*x)) → max(0,-t*x) + log(1+exp(-|t*x|))
LocalTensor<T> neg_tx = Neg(Mul(tBuf, xBuf));
yBuf = Add(Max(0, neg_tx), Log1p(Exp(Neg(Abs(neg_tx)))));
```

---

## 5 与主范式融合

NumericalStable 是修饰范式，与主范式共同声明：

```yaml
paradigms: [Elementwise, NumericalStable]   # 主范式 + 修饰范式
```

**默认主范式**：`op.paradigms` 只有 `NumericalStable` 一项时，按 `Elementwise` 处理。

> 该默认只是兜底。归约类算子（Softmax / LayerNorm 等）必须显式写出 `Reduction` /
> `ReductionComposite`——默认值不会替你补回漏标的归约轴，pass 结构和累加器精度都会跟着错。

### 5.1 修改清单

| 章节 | 修改 | 说明 |
|------|------|------|
| Kernel 入口 / CopyIn / CopyOut / 地址偏移 | ❌ 不改 | 写"跟主范式保持一致" |
| 跨核分核 / 轴依赖 / TilingKey / 模板类 | ❌ 不改 | 稳定化不改并行策略 |
| **Compute** | ✅ 改 | 叠加 T1-T6 公式 |
| **Buffer 清单** | ✅ 改 | +mBuf/sBuf/dBuf/meanBuf/m2Buf |
| **UB 预算** | ✅ 改 | 加入稳定化 buffer 开销 |
| **Pass 结构** | ✅ 改 | two_pass/online/Welford |
| **TilingData** | ⚠️ 可选 | 至多 +eps 字段 |

### 5.2 编译期常量 vs TilingData

| 参数 | 位置 | 原因 |
|------|------|------|
| needs_welford | `if constexpr` | formula 静态决定 |
| needs_dbuf | `if constexpr` | 末尾表达式静态决定 |
| logsumexp_pass | `if constexpr` | UB/轴长静态决定 |
| range_guard | `if constexpr` | formula 静态决定 |
| eps | TilingData 或 `constexpr` | attribute→TilingData；固定值→编译期 |
| accumulator_dtype | 模板参数 | dtype 静态决定 |
| **禁止** | **TilingKey** | 稳定化不膨胀 key |

### 5.3 文档书写风格

**不改的章节**:
```markdown
## 3 CopyIn
跟主范式保持一致。
```

**改的章节（diff 风格）**:
```markdown
## 5 Compute
**主范式**: y = exp(x) / sum(exp(x))

**稳定化差异**:
1. [T1 max-shift] m = max(x)
2. [T1 LogSumExp] y = exp(x-m) / sum(exp(x-m))
3. [T3 fp32累加] sum 用 fp32 累加器

**新增 Buffer**: mBuf(fp32), sBuf(fp32)
**Pass**: 从单遍改为 two_pass
```

---

## 6 UB 预算计算

```cpp
// 主范式
size_t ub_main = sizeof(T) * (in + out + tmp);

// + 稳定化
size_t ub_stable = 0;
if (logdomain) {
    ub_stable += sizeof(fp32) * tile_size;  // mBuf
    ub_stable += sizeof(fp32) * tile_size;  // sBuf
    if (needs_dbuf) ub_stable += sizeof(fp32) * tile_size;
    ub_stable += sizeof(T) * tile_size;     // expBuf
} else if (normdenom && needs_welford) {
    ub_stable += 2 * sizeof(fp32) * tile_size;  // meanBuf + m2Buf
}

size_t ub_total = ub_main + ub_stable;
```

---

## 7 Golden 校验清单

- [ ] Compute 是否标注 T1-T6 应用点
- [ ] Buffer 清单是否含稳定化新增 buffer
- [ ] UB 预算是否含稳定化开销
- [ ] Pass 结构是否说明遍数变化
- [ ] 稳定化分支是否用 `if constexpr`（不进 TilingKey）
- [ ] 不改的章节是否写"跟主范式保持一致"
- [ ] eps 来源是否明确（禁止硬编码）
- [ ] 是否避免常见误区（Softmax 加 eps、RMSNorm 用 Welford 等）

---

## 8 快速参考

### 8.1 算子决策速查

| 算子 | 技术 | 场景 | needs_dbuf | needs_welford | 关键点 |
|------|------|------|-----------|--------------|--------|
| Softmax | T1+T3 | logdomain | false | - | 分母≥1，不需 eps |
| LogSoftmax | T1+T3 | logdomain | true | - | 需要 dBuf |
| **SoftmaxGrad** | T3 | logdomain | - | - | 复用 forward 的 y |
| **LogSoftmaxGrad** | T3 | logdomain | - | - | 需计算 exp(y) |
| LayerNorm | T2+T3+T4 | normdenom | - | true | Welford 或两遍 |
| RMSNorm | T2+T3 | normdenom | - | false | 单遍均方 |
| L2Normalize | T2+T3 | normdenom | - | false | max(norm, eps) |
| Log | T5 | NONE | - | - | log(max(x, eps)) |
| Sigmoid | T6 | NONE | - | - | 分段公式 |

### 8.2 技术触发条件

| 技术 | 触发条件 | 关键实现 |
|------|---------|---------|
| T1 | formula 含 log/exp/softmax | max-shift + two_pass |
| T2 | formula 含归一化分母 | rsqrt(var + eps) |
| T3 | 长轴求和 + fp16/bf16 | fp32 累加器 |
| T4 | 需均值+方差（非纯均方） | Welford 或两遍 |
| T5 | exp/log/pow 定义域/溢出风险 | Max/Clamp/Select |
| T6 | 单一公式跨值域风险 | Select 分段 |
