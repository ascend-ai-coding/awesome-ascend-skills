# A2 Playbook: Welford 数值稳定均值/方差

> 本 Playbook 为**强制流程**。采纳 A2 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A2 的核心是**在 BatchNorm/LayerNorm 等归一化算子的均值/方差计算中，使用 Welford 在线算法替代朴素的 sum-of-squares，避免大数吃小数和 catastrophic cancellation**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a2_locations.txt`：

```bash
# 1. 均值/方差/归约计算
grep -n "mean|Mean|var|Var|sum|Sum|Reduce|reduce|normali|Normali" \
    shared/original/op_kernel/*.cpp > /tmp/a2_locations.txt
# 2. 归一化类型
grep -n "BatchNorm|LayerNorm|RMSNorm|InstanceNorm|GroupNorm|norm|Norm" \
    shared/original/op_kernel/*.cpp >> /tmp/a2_locations.txt
# 3. 当前方差计算方式
grep -n "sum.*sq|sq.*sum|mean.*sq|sq.*mean|N-1|N.*1|unbiased|batchVar" \
    shared/original/op_kernel/*.cpp >> /tmp/a2_locations.txt
# 4. 数据类型与精度
grep -n "float|half|bf16|fp16|fp32|float32" \
    shared/original/op_kernel/*.cpp >> /tmp/a2_locations.txt
# 5. 样本数量处理
grep -n "batchSize|N.*count|patternR|numElements|elementCount" \
    shared/original/op_kernel/*.cpp >> /tmp/a2_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **均值/方差计算位置**：所有 `Mean`/`Var`/`Reduce` 调用位置
- **归一化类型**：BatchNorm/LayerNorm/RMSNorm/InstanceNorm
- **当前方差算法**：sum-of-squares / 两趟算法 / Welford
- **数据精度**：FP32 / FP16 / BF16
- **样本数量处理**：单样本（N=1）/ 多样本（N>1）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 归一化类型 | `?` (BN/LN/RMSN/IN) | 不变 | `?_kernel.cpp:L?` |
| 精度 | `?` (FP32/FP16/BF16) | 不变 | `?_kernel.cpp:L?` |
| 当前方差算法 | `?` (sum-of-squares/两趟/Welford) | Welford | `?_kernel.cpp:L?` |
| 样本方差校正 | `?` (无/N-1/N) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| 单样本处理 | `?` (无/有) | 有（避免除以0） | `?_kernel.cpp:L?` |
| EMA 更新 | `?` (无/有) | 不变 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的归一化类型和方差计算方式，判断你的代码属于以下哪种形态：

- **形态 α — Welford 在线算法（训练模式，最常见）**：在线更新均值和 M2（平方差累加），单趟完成，避免大数吃小数。适合 BatchNorm 训练统计量计算。
- **形态 β — 样本方差校正（N-1 vs N）**：训练时总体方差需转换为样本方差（N/(N-1)），推理时直接用总体方差。需处理 N=1 的单样本场景。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — Welford 在线算法）

```cpp
// === 改造前（sum-of-squares，数值不稳定）===
__aicore__ inline void MeanVarNaive(LocalTensor<float> input,
                                     uint32_t count, uint32_t N) {
    // 先算均值
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += input.GetValue(i);
    }
    float mean = sum / N;
    
    // 再算方差：sum((x - mean)^2)
    float var = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float diff = input.GetValue(i) - mean;
        var += diff * diff;
    }
    // ❌ 风险：大数减 mean 导致 catastrophic cancellation
}

// === 改造后（Welford 在线算法，数值稳定）===
__aicore__ inline void MeanVarWelford(LocalTensor<float> input,
                                       uint32_t count, uint32_t N) {
    float mean = 0.0f;
    float M2 = 0.0f;  // 平方差累加
    
    for (uint32_t i = 0; i < count; i++) {
        float x = input.GetValue(i);
        float delta = x - mean;
        mean += delta / (i + 1);
        float delta2 = x - mean;
        M2 += delta * delta2;
    }
    
    // 总体方差 = M2 / N
    float populationVar = M2 / N;
    
    // 样本方差 = M2 / (N - 1)，N=1 时避免除以0
    float batchVarScale = (N > 1) ? (float)N / (float)(N - 1) : 1.0f;
    float sampleVar = populationVar * batchVarScale;
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（样本方差校正 + 单样本兜底）**：
  当算子需要同时输出训练统计量和推理 running 统计量时：
  ```cpp
  // 训练统计量（样本方差，无偏估计）
  float batchVarScale = 1.0f;
  if (patternR0 * patternR1 > 1) {
      batchVarScale = (float)N / (float)(N - 1);
  }
  // patternR0 * patternR1 == 1 时（单样本），batchVarScale = 1.0f，避免除以0
  float batchVar = populationVar * batchVarScale;
  
  // Running 统计量（EMA 更新，不涉及 N-1 校正）
  runningMean = momentum * runningMean + (1.0f - momentum) * mean;
  runningVar = momentum * runningVar + (1.0f - momentum) * populationVar;
  ```

- **与 A1 的协同**：A1（FP32 Intermediate）要求中间计算用 FP32，Welford 算法在 FP32 下效果最佳。若算子已有 A1，A2 只需在 FP32 精度链内替换方差算法。
- **与 A6 的协同**：A6（高精度 rsqrt）用于 LayerNorm 的 1/sqrt(var+eps)，A2 提供稳定的 var 计算。两者常一起出现：A2 算稳定 var，A6 算高精度 rsqrt。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: Welford 的 mean 初始值必须为 0.0f，M2 初始值必须为 0.0f
约束 2: N=1 时 batchVarScale 必须为 1.0f，不能是 N/(N-1)（避免除以0）
约束 3: 循环索引从 0 开始，delta / (i + 1) 的分母不能为 0
约束 4: M2 可能为负数（浮点误差），需 clamp 到 >= 0
约束 5: EMA 更新 running 统计量时不做 N-1 校正，只用总体方差
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `输入精度 = ?`, `计算精度 = ?`, `输出精度 = ?`
- `N 范围 = [?, ?]`, `batchVarScale = ?`（N=1 时 / N>1 时）
- `M2 范围 = [?, ?]`, `是否 clamp 到 >=0` = yes/no
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 Welford 在线更新（delta = x - mean）
grep -cE "delta.*=.*x.*-.*mean|mean.*+=.*delta" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 M2（平方差累加）
grep -cE "M2|m2|delta.*delta2|delta2.*delta" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 有样本方差校正（N-1 或 batchVarScale）
grep -cE "N-1|N.*-.*1|batchVarScale|sample.*var|unbiased" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有单样本兜底（N > 1 或 N == 1）
grep -cE "N\s*>\s*1|N\s*==\s*1|patternR.*>\s*1|patternR.*==\s*1" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 无 sum-of-squares（朴素两趟算法）
grep -cE "sum.*sq|sq.*sum|for.*mean.*for.*var" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "形态 β 保留两趟算法用于 EMA"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：M2 未定义 | 确认 M2 变量声明为 `float M2 = 0.0f;` |
| 运行时：N=1 时 NaN | 检查 batchVarScale 在 N=1 时是否为 1.0f，不能是 N/(N-1) |
| 运行时：方差为负数 | M2 因浮点误差可能略负，添加 `M2 = max(M2, 0.0f);` |
| 运行时：精度仍不如 baseline | 确认 mean 和 M2 的更新顺序正确：`delta = x - mean; mean += delta / n; delta2 = x - mean; M2 += delta * delta2;` |
| 与 A1 的 FP32 链冲突 | Welford 应在 FP32 域内计算。若输入是 FP16，先 Cast 到 FP32 再 Welford |
| 大规模 N 时性能下降 | Welford 是 O(N) 单趟算法，性能通常优于两趟。若下降明显，检查是否有额外内存访问 |
| EMA 更新用错方差类型 | Running 统计量必须用总体方差（M2/N），不能用样本方差（M2/(N-1)） |
| 循环索引从 1 开始 | Welford 分母是 `(i + 1)`，若循环从 1 开始则分母应为 `i` |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A2 Playbook Completion]
Step 1: done (/tmp/a2_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: precision=? N_range=? batchVarScale=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
