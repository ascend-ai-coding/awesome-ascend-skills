# A5 Playbook: Softmax 数值安全与特殊值处理

> 本 Playbook 为**强制流程**。采纳 A5 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A5 的核心是**在 softmax/exp 计算中引入数值稳定技巧（max-subtraction 或 online softmax），防止 exp 溢出和 NaN 传播**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a5_locations.txt`：

```bash
# 1. Softmax / exp / log 计算
grep -n "softmax\|Softmax\|exp\|Exp\|log\|Log" \
    shared/original/op_kernel/*.cpp > /tmp/a5_locations.txt
# 2. 当前数值处理（是否有 max 减法、NaN 处理）
grep -n "max.*sub\|subtract.*max\|maxVal\|maxLocal\|online.*softmax\|NaN\|nan\|CMPMODE" \
    shared/original/op_kernel/*.cpp >> /tmp/a5_locations.txt
# 3. 精度相关（BF16/FP16）
grep -n "bf16\|bfloat16\|half\|fp16" \
    shared/original/op_kernel/*.cpp >> /tmp/a5_locations.txt
# 4. Reduce / sum 操作
grep -n "Reduce\|Sum\|sum\|Add\|accum" \
    shared/original/op_kernel/*.cpp >> /tmp/a5_locations.txt
# 5. 已有数值安全处理
grep -n "Compare\|Select\|CMPMODE\|SELMODE\|special.*value\|numerical" \
    shared/original/op_kernel/*.cpp >> /tmp/a5_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Softmax/Exp 位置**：所有 `softmax` / `exp` / `log` 调用位置
- **当前数值处理**：是否已有 max 减法、NaN 处理
- **精度类型**：FP32 / FP16 / BF16
- **Reduce 操作**：sum / max reduce 的实现方式
- **已有安全**：Compare+Select / special value 处理

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 计算类型 | `?` (softmax/exp/log) | 不变 | `?_kernel.cpp:L?` |
| 精度 | `?` (FP32/FP16/BF16) | 不变 | `?_kernel.cpp:L?` |
| 当前数值处理 | `?` (无/max/online) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| max 计算 | `?` (无/有) | 有 | `?_kernel.cpp:L?` |
| NaN 处理 | `?` (无/有) | Compare+Select | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的计算类型和精度，判断你的代码属于以下哪种形态：

- **形态 α — Softmax max-trick（标准数值稳定）**：计算 softmax 时，先减 max 再 exp，避免溢出。
- **形态 β — Online softmax（两阶段稳定）**：先求 max，再求 exp-sum，最后归一化。适合大规模 reduce。
- **形态 γ — NaN/inf 特殊值处理（Compare+Select）**：用 `Compare` + `Select` 将 NaN 替换为 0，防止 NaN 传播。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — Softmax max-trick，最常见）

```cpp
// === 改造前（直接 exp，数值不稳定）===
__aicore__ inline void SoftmaxNaive(LocalTensor<float> input,
                                     LocalTensor<float> output,
                                     uint32_t count) {
    // 直接 exp：当 input[i] > 88（FP32）或 > 11（FP16）时 exp 溢出为 inf
    Exp(output, input, count);
    
    // 求和
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += output.GetValue(i);
    }
    
    // 归一化：若 sum = inf，结果全 NaN
    float invSum = 1.0f / sum;
    Mul(output, output, invSum, count);
}

// === 改造后（max-subtraction 数值稳定）===
__aicore__ inline void SoftmaxStable(LocalTensor<float> input,
                                      LocalTensor<float> output,
                                      LocalTensor<float> maxLocal,
                                      LocalTensor<float> expLocal,
                                      uint32_t count) {
    // Step 1: 求 max
    Reduce(maxLocal, input, count, ReduceMode::REDUCE_MAX);
    float maxVal = maxLocal.GetValue(0);
    
    // Step 2: x - max（防止 exp 溢出）
    Sub(output, input, maxVal, count);
    
    // Step 3: exp(x - max) — 最大值变为 exp(0) = 1，其他 < 1，不会溢出
    Exp(expLocal, output, count);
    
    // Step 4: 求和
    Reduce(sumLocal, expLocal, count, ReduceMode::REDUCE_SUM);
    float sumVal = sumLocal.GetValue(0);
    
    // Step 5: 归一化
    float invSum = 1.0f / sumVal;
    Mul(output, expLocal, invSum, count);
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（Online softmax，大规模 reduce）**：
  当数据量极大（如长序列 attention），分 tile 计算时需要 online 更新：
  ```cpp
  float onlineMax = -FLT_MAX;
  float onlineSum = 0.0f;
  
  for (uint32_t tile = 0; tile < nTiles; tile++) {
      // 当前 tile 的 max
      Reduce(tileMaxLocal, tileInput, tileSize, REDUCE_MAX);
      float tileMax = tileMaxLocal.GetValue(0);
      
      // 更新 online max 和 sum
      float newMax = max(onlineMax, tileMax);
      float scale = exp(onlineMax - newMax);  // 旧数据缩放
      
      // 当前 tile 的 exp-sum
      Sub(tileInput, tileInput, newMax, tileSize);
      Exp(tileExp, tileInput, tileSize);
      Reduce(tileSumLocal, tileExp, tileSize, REDUCE_SUM);
      float tileSum = tileSumLocal.GetValue(0);
      
      onlineSum = onlineSum * scale + tileSum;
      onlineMax = newMax;
  }
  ```
  形态 β 适合 Flash Attention 等分 tile softmax 场景，避免全量 reduce。

- **形态 γ（NaN/inf 特殊值处理）**：
  用 `Compare` + `Select` 处理 NaN：
  ```cpp
  // Compare: mask = (x == x) ? true : false
  // NaN 的特性：NaN != NaN，所以 NaN 的比较结果为 false
  Compare(maskLocal, xLocal, xLocal, CMPMODE::EQ, count);
  
  // Select: mask == true ? x : 0.0f
  // NaN 位置输出 0.0f，阻止 NaN 传播
  Select(outLocal, maskLocal, xLocal, 0.0f,
         SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
  ```
  形态 γ 可与形态 α/β 叠加：先做 NaN 清理，再做 softmax。

- **与 A1 的协同**：A1（FP32 Intermediate）要求 compute 用 FP32，A5 的 max-trick 在 FP32 下效果最佳。若算子已有 A1，A5 只需在 FP32 精度链内添加 max-subtraction。

- **与 A3 的边界**：A3（Rounding Mode）处理 Cast 的 rounding，A5 处理 exp 的数值稳定。两者独立但常一起出现：A5 做 softmax 稳定，A3 做精度转换 rounding。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: max 的初始值必须足够小（如 -FLT_MAX），不能是 0
约束 2: exp(x - max) 后，最大值为 1.0，需确保 1.0 在目标精度下可表示
约束 3: sum 不能为 0（全 -inf 输入时）。若 sum == 0，输出应全为 0 或均匀分布
约束 4: NaN 处理时 Compare 的 CMPMODE 必须是 EQ（x == x 为 false 当 x 是 NaN）
约束 5: Online softmax 的 scale = exp(onlineMax - newMax) 在 newMax >> onlineMax 时可能下溢为 0，需兜底
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `输入精度 = ?`, `计算精度 = ?`, `输出精度 = ?`
- `max 初始值 = ?`, `exp 溢出阈值 = ?`
- `sum 范围 = [?, ?]`
- `NaN 处理覆盖率 = ?%`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 max reduce 或 max 计算
grep -cE "REDUCE_MAX|Reduce.*MAX|maxVal|maxLocal|FindMax" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 subtract max 步骤
grep -cE "Sub.*max|sub.*max|x.*-.*max|input.*-.*maxVal" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: exp 在 subtract 之后（不是直接 exp input）
grep -cE "Exp.*input|Exp.*xLocal" modified_files/op_kernel/*.cpp
# 期望: == 0（或仅在 note 中说明有 NaN 清理路径）

grep -cE "Exp.*sub|Exp.*output|Exp.*after" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有 sum reduce 和归一化
grep -cE "REDUCE_SUM|Reduce.*SUM|invSum|1\.0f.*\/|Mul.*inv" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: NaN 处理（形态 gamma）或注释说明不需要
grep -cE "Compare.*EQ.*x.*x|Select.*NaN|SELMODE|CMPMODE::EQ" modified_files/op_kernel/*.cpp
# 期望: >= 1（或 note 中说明 "无 NaN 风险，跳过"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：REDUCE_MAX 不存在 | 确认 CANN 版本支持 `Reduce` API。旧版本可能用 `Max` 或循环实现 |
| 运行时：exp 仍溢出 | 检查 max 是否确实减去了。常见错误：`Sub(output, input, maxVal, count)` 的 maxVal 是标量还是 tensor。标量减法需用 `Duplicate` 广播 |
| 运行时：结果全为 0 | max 初始值设成了 0 而不是 -FLT_MAX。若输入全负，max=0 会导致所有 exp(x-0) < 1，可能下溢 |
| 运行时：精度对不上 baseline | online softmax 的 scale 累积有精度损失。调试时先与形态 α 对比，确认 online 版本无误再使用 |
| sum = inf | 输入值过大，即使减 max 后仍有值 > 88（FP32 exp 溢出阈值）。检查输入是否有异常大值 |
| sum = 0 | 输入全为 -inf 或极小值。添加兜底：`if (sumVal == 0) { Duplicate(output, 0.0f, count); return; }` |
| NaN 处理遗漏 inf | `Compare(x, x, EQ)` 对 inf 返回 true（inf == inf）。若需处理 inf，额外用 `Compare(x, maxVal, CMPMODE::GT)` 检测 |
| 多 tile online softmax 精度漂移 | 每 tile 的 max 和 sum 独立，跨 tile 累积时浮点误差累积。长序列（>4096）误差明显，建议用 FP32 累积 |
| BF16 下 max-trick 仍溢出 | BF16 的 exp 阈值约为 11.5。减 max 后若仍有值 > 11.5，需进一步缩放或 tile 拆分 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A5 Playbook Completion]
Step 1: done (/tmp/a5_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: precision=? max_init=? sum_range=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
