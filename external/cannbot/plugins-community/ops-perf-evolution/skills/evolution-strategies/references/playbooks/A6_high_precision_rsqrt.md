# A6 Playbook: 高精度 rsqrt Newton-Raphson 迭代

> 本 Playbook 为**强制流程**。采纳 A6 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A6 的核心是**在 LayerNorm/RMSNorm 的 1/sqrt(x) 计算中，用 Newton-Raphson 迭代替代 naive sqrt+div 或单次 rsqrt，提高精度并降低延迟**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a6_locations.txt`：

```bash
# 1. rsqrt / sqrt / reciprocal 计算
grep -n "rsqrt|Rsqrt|Sqrt|sqrt|Reciprocal|reciprocal|inv.*sqrt|1\\.0f.*sqrt" \
    shared/original/op_kernel/*.cpp > /tmp/a6_locations.txt
# 2. 归一化场景
grep -n "LayerNorm|RMSNorm|layernorm|rmsnorm|Normalize|norm|var.*eps" \
    shared/original/op_kernel/*.cpp >> /tmp/a6_locations.txt
# 3. 当前精度处理方式
grep -n "SCALAR|Newton|newton|NR_ITER|iter.*rsqrt|y0.*y1" \
    shared/original/op_kernel/*.cpp >> /tmp/a6_locations.txt
# 4. 特殊值处理
grep -n "POS_INF|inf|nan|NaN|Compare.*var|Select.*zero" \
    shared/original/op_kernel/*.cpp >> /tmp/a6_locations.txt
# 5. 数据精度
grep -n "float|half|bf16|fp16|fp32" \
    shared/original/op_kernel/*.cpp >> /tmp/a6_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **rsqrt/sqrt 位置**：所有 `rsqrt`/`Sqrt`/`Reciprocal` 调用位置
- **归一化类型**：LayerNorm/RMSNorm/其他
- **当前实现方式**：sqrt+div / 单次 rsqrt / Newton 迭代
- **特殊值处理**：inf/0/NaN 处理
- **数据精度**：FP32 / FP16 / BF16

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 归一化类型 | `?` (LN/RMSN/其他) | 不变 | `?_kernel.cpp:L?` |
| 精度 | `?` (FP32/FP16/BF16) | 不变 | `?_kernel.cpp:L?` |
| 当前 rsqrt 实现 | `?` (sqrt+div/单次rsqrt) | Newton-Raphson | `?_kernel.cpp:L?` |
| 迭代次数 | `?` (无/1/2) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| 特殊值处理 | `?` (无/有) | Compare+Select | `?_kernel.cpp:L?` |
| eps 处理 | `?` (无/有) | 有（var+eps） | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的精度要求和延迟约束，判断你的代码属于以下哪种形态：

- **形态 α — 单次 Newton-Raphson 迭代（推理场景，最常见）**：硬件 rsqrt 初始近似 + 1 次 Newton 迭代，精度从 ~8bit 提升到接近 FP32，增加约 3-4 条 Vector 指令。
- **形态 β — 双次 Newton-Raphson 迭代（训练场景，精度敏感）**：硬件 rsqrt + 2 次 Newton 迭代，精度最高，增加约 6-8 条 Vector 指令。适合训练时的梯度计算。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 单次 Newton-Raphson）

```cpp
// === 改造前（sqrt+div，高延迟）===
__aicore__ inline void RsqrtNaive(LocalTensor<float> var,
                                   LocalTensor<float> output,
                                   float eps, uint32_t count) {
    // var + eps
    Adds(output, var, eps, count);
    // sqrt
    Sqrt(output, output, count);
    // 1/x — 除法指令延迟高
    Reciprocal(output, output, count);
}

// === 改造后（Newton-Raphson 单次迭代）===
__aicore__ inline void RsqrtNewton(LocalTensor<float> var,
                                    LocalTensor<float> output,
                                    LocalTensor<float> tmp1,
                                    LocalTensor<float> tmp2,
                                    float eps, uint32_t count) {
    static constexpr float SCALAR1 = -0.5f;
    static constexpr float SCALAR2 = 1.5f;
    
    // Step 1: x = var + eps
    Adds(output, var, eps, count);
    
    // Step 2: y0 = rsqrt(x) — 硬件初始近似
    // 若硬件无 rsqrt，用 Sqrt + Reciprocal 作为初始值
    // Rsqrt(y0, output, count);  // 或：Sqrt(tmp1, output, count); Reciprocal(y0, tmp1, count);
    
    // Step 3: Newton-Raphson 迭代 y1 = y0 * (1.5 - 0.5 * x * y0^2)
    // t = x * (-0.5)
    Muls(tmp1, output, SCALAR1, count);
    // t = t * y0
    Mul(tmp1, tmp1, output, count);
    // t1 = t * y0 + 1.5  (即 Mula)
    Mula(tmp2, tmp1, output, count);  // tmp2 = tmp1 * output + 1.5? 不对，重新组织
    
    // 正确展开：
    // y0^2 = y0 * y0
    Mul(tmp1, output, output, count);   // tmp1 = y0^2
    // x * y0^2 * (-0.5)
    Muls(tmp1, tmp1, SCALAR1, count);
    Mul(tmp1, tmp1, output, count);      // tmp1 = x * y0^2 * (-0.5)
    // 1.5 + tmp1
    Adds(tmp2, tmp1, SCALAR2, count);    // tmp2 = 1.5 - 0.5 * x * y0^2
    // y1 = y0 * tmp2
    Mul(output, output, tmp2, count);    // output = y0 * (1.5 - 0.5 * x * y0^2)
    
    // Step 4: 特殊值处理
    CompareScalar(tmp1, var, POS_INF, count);
    Select(output, tmp1, output, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（双次 Newton-Raphson 迭代）**：
  在形态 α 基础上再做一次迭代：
  ```cpp
  // 第二次迭代 y2 = y1 * (1.5 - 0.5 * x * y1^2)
  Mul(tmp1, output, output, count);     // y1^2
  Muls(tmp1, tmp1, SCALAR1, count);
  Mul(tmp1, tmp1, output, count);        // x * y1^2 * (-0.5)
  Adds(tmp2, tmp1, SCALAR2, count);
  Mul(output, output, tmp2, count);      // y2
  ```
  形态 β 适合训练场景，精度接近 FP32 全精度。

- **与 A1 的协同**：A1（FP32 Intermediate）要求中间计算用 FP32，Newton-Raphson 在 FP32 下收敛最快。若算子已有 A1，A6 只需在 FP32 精度链内替换 rsqrt 实现。
- **与 A2 的协同**：A2（Welford）提供稳定的 var 计算，A6 提供高精度的 1/sqrt(var+eps)。两者常一起出现：A2 算稳定 var，A6 算高精度 rsqrt。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: SCALAR1 必须是 -0.5f，SCALAR2 必须是 1.5f，不能交换
约束 2: 输入 x = var + eps 必须 > 0，若 x <= 0 需兜底（输出 0 或极大值）
约束 3: 初始近似 y0 的精度决定迭代次数。若硬件 rsqrt 精度低（<6bit），需用形态 β
约束 4: 特殊值处理：var = inf → 输出 0；var = 0 → 输出 1/sqrt(eps)
约束 5: FP16/BF16 下 Newton 迭代可能因精度不足而发散，建议用 FP32 中间计算
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `输入精度 = ?`, `计算精度 = ?`, `输出精度 = ?`
- `迭代次数 = ?`, `eps = ?`
- `特殊值覆盖率 = ?%`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 Newton-Raphson 迭代结构（1.5 - 0.5 * x * y^2）
grep -cE "1\.5|SCALAR2|SCALAR1|-0\.5" modified_files/op_kernel/*.cpp
# 期望: >= 2

# 检查 2: 有 rsqrt 或 sqrt+reciprocal
grep -cE "Rsqrt|rsqrt|Sqrt.*Reciprocal|Reciprocal.*Sqrt" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 无 naive sqrt+div（无 Reciprocal 在 Sqrt 之后且没有迭代）
grep -cE "Sqrt\s*\([^)]*\)[^;]*;\s*Reciprocal\s*\(" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "初始近似步骤"）

# 检查 4: 有 eps 相加步骤
grep -cE "var.*eps|eps.*var|Adds.*eps" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 有特殊值处理（Compare+Select 或边界检查）
grep -cE "Compare.*inf|Select.*zero|CMPMODE|SELMODE" modified_files/op_kernel/*.cpp
# 期望: >= 1（或 note 中说明 "无 inf/0 风险，跳过"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：Rsqrt 不存在 | 确认 CANN 版本。旧版本用 `Sqrt` + `Reciprocal` 组合 |
| 运行时：结果精度仍低 | 检查迭代次数。单次迭代精度约 16bit，双次约 24bit。若仍低，用 FP32 中间计算 |
| 运行时：var=0 时输出异常 | 检查 eps 是否已加。`Adds(var, var, eps)` 必须在 rsqrt 之前 |
| 运行时：结果发散 | 检查 SCALAR1/SCALAR2 符号。公式是 `y * (1.5 - 0.5 * x * y^2)`，不是 `y * (0.5 - 1.5 * x * y^2)` |
| FP16 下迭代发散 | FP16 尾数只有 10bit，Newton 迭代可能不收敛。改用 FP32 中间 buffer |
| 与 A1 的 FP32 链冲突 | Newton 迭代应在 FP32 域执行。若输入是 FP16，先 Cast 到 FP32 |
| 性能下降明显 | 单次 Newton 增加 3-4 条指令，双次增加 6-8 条。若下降 >20%，考虑用形态 α |
| 特殊值遗漏 inf | `Compare(var, POS_INF)` 对 inf 返回 true。若需处理，用 Select 将输出置 0 |
| eps 太小导致 underflow | 若 eps < 1e-7 且 var 也小，x = var + eps 可能下溢。检查 eps 量级 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A6 Playbook Completion]
Step 1: done (/tmp/a6_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: precision=? iter=? eps=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
