# A8 Playbook: 量化专用精度处理

> 本 Playbook 为**强制流程**。采纳 A8 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A8 的核心是**在量化/反量化计算中，用 Maxs+Mins 指令将值严格限制在 [quant_min, quant_max] 范围内，并根据计算阶段选择正确的 RoundMode（CAST_RINT 四舍五入 / CAST_ROUND 向最近偶数舍入）**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a8_locations.txt`：

```bash
# 1. 量化/反量化操作
grep -n "quant|Quant|dequant|Dequant|scale|zero.*point|quant_min|quant_max|fake_quant|FakeQuant" \
    shared/original/op_kernel/*.cpp > /tmp/a8_locations.txt
# 2. Cast 与 RoundMode
grep -n "Cast|cast|RoundMode|ROUND|RINT|quant_min|quant_max" \
    shared/original/op_kernel/*.cpp >> /tmp/a8_locations.txt
# 3. 当前边界处理
grep -n "Maxs|maxs|Mins|mins|clamp|Clip|clip|boundary" \
    shared/original/op_kernel/*.cpp >> /tmp/a8_locations.txt
# 4. 数据类型
grep -n "INT8|int8|INT32|int32|float|half|bf16" \
    shared/original/op_kernel/*.cpp >> /tmp/a8_locations.txt
# 5. 计算序列
grep -n "Adds|Muls|Mul|Add|Sub|Div" \
    shared/original/op_kernel/*.cpp >> /tmp/a8_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **量化位置**：所有 `quant`/`dequant`/`FakeQuant` 调用位置
- **Cast 位置**：所有 `Cast` 调用位置、当前 RoundMode 类型
- **当前边界处理**：Maxs/Mins/clamp/无
- **数据类型**：INT8/INT32/FP16/BF16/FP32
- **计算序列**：scale→zero_point→round→clamp 的完整链路

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 量化类型 | `?` (INT8/INT4/FP8) | 不变 | `?_kernel.cpp:L?` |
| 当前 RoundMode | `?` (全部 NONE/RINT) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| 当前边界处理 | `?` (无/Maxs/Mins) | Maxs + Mins | `?_kernel.cpp:L?` |
| quant_min/quant_max | `?` (无/有) | 有 | `?_kernel.cpp:L?` |
| 计算序列 | `?` (scale→round→cast) | scale→round→clamp→cast | `?_kernel.cpp:L?` |
| 精度 | `?` (FP32/FP16/BF16) | 不变 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的量化类型和计算阶段，判断你的代码属于以下哪种形态：

- **形态 α — 标准量化（INT8/INT4，最常见）**：反量化后计算，再量化输出。反量化用 `CAST_NONE`，量化输出用 `CAST_RINT`，最后用 `Maxs+Mins` clamp 到 [quant_min, quant_max]。
- **形态 β — 量化感知训练（QAT，FakeQuant）**：FakeQuant 需要模拟量化误差，用 `CAST_ROUND`（向最近偶数舍入）以匹配硬件量化行为。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 标准量化）

```cpp
// === 改造前（无边界限制，精度损失）===
__aicore__ inline void QuantNaive(LocalTensor<float> input,
                                   LocalTensor<int8_t> output,
                                   float scale, int32_t zeroPoint,
                                   uint32_t count) {
    // x / scale + zero_point
    Muls(tmp, input, 1.0f / scale, count);
    Adds(tmp, tmp, zeroPoint, count);
    // ❌ 风险：值可能超出 [quant_min, quant_max]
    // ❌ 风险：舍入模式不明确
    Cast(output, tmp, RoundMode::CAST_NONE, count);
}

// === 改造后（边界限制 + 正确 RoundMode）===
__aicore__ inline void QuantSafe(LocalTensor<float> input,
                                  LocalTensor<int8_t> output,
                                  LocalTensor<float> tmp,
                                  float scale, int32_t zeroPoint,
                                  int32_t quantMin, int32_t quantMax,
                                  uint32_t count) {
    // Step 1: x / scale + zero_point
    Muls(tmp, input, 1.0f / scale, count);
    Adds(tmp, tmp, static_cast<float>(zeroPoint), count);
    
    // Step 2: 四舍五入（CAST_RINT）
    Cast(tmp, tmp, RoundMode::CAST_RINT, count);
    
    // Step 3: 边界限制 [quant_min, quant_max]
    // Maxs: tmp = max(tmp, quantMin)
    Maxs(tmp, tmp, quantMin, count);
    // Mins: tmp = min(tmp, quantMax)
    Mins(tmp, tmp, quantMax, count);
    
    // Step 4: 输出
    Cast(output, tmp, RoundMode::CAST_NONE, count);
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（FakeQuant / QAT）**：
  量化感知训练需要模拟硬件量化行为，通常用 `CAST_ROUND`（向最近偶数舍入）：
  ```cpp
  // FakeQuant: 模拟量化误差
  Cast(tmp, input, RoundMode::CAST_ROUND, count);  // 向最近偶数舍入
  Maxs(tmp, tmp, quantMin, count);
  Mins(tmp, tmp, quantMax, count);
  // 反量化回 FP32
  Subs(tmp, tmp, zeroPoint, count);
  Muls(tmp, tmp, scale, count);
  ```

- **与 A3 的协同**：A3（Rounding Mode）处理通用的 Cast 精度，A8 专门针对量化场景的边界 + RoundMode 组合。两者可叠加：A3 处理非量化 Cast，A8 处理量化 Cast。
- **与 A7 的边界**：A7 处理索引边界 [0, size-1]，A8 处理量化值边界 [quant_min, quant_max]。两者独立但常一起出现。
- **与 D4 的协同**：D4（FP8/INT4 量化）定义低比特类型，A8 确保量化值的边界和舍入正确。两者必须同时使用。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: Maxs/Mins 的顺序不能交换：先 Maxs（下界），再 Mins（上界）
约束 2: quantMin/quantMax 必须是 int32_t 标量，不能是浮点
约束 3: RoundMode 选择：标准量化用 CAST_RINT，QAT 用 CAST_ROUND
约束 4: INT8 的 quantMin = -128, quantMax = 127；UINT8 的 quantMin = 0, quantMax = 255
约束 5: 若输入值在 [quantMin, quantMax] 范围内，Maxs+Mins 不应改变值（验证边界正确性）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `量化类型 = ?`, `quantMin = ?`, `quantMax = ?`
- `RoundMode = ?`, `scale = ?`, `zeroPoint = ?`
- `Maxs+Mins 覆盖率 = ?%`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 Maxs 操作
grep -cE "Maxs|maxs" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 Mins 操作
grep -cE "Mins|mins" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 有 quant_min/quant_max 或边界常量
grep -cE "quantMin|quantMax|quant_min|quant_max|-128|127|255" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有 CAST_RINT 或 CAST_ROUND
grep -cE "CAST_RINT|CAST_ROUND" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 无全部 CAST_NONE（至少一个量化 Cast 不是 NONE）
grep -cE "Cast\s*\([^,]+,\s*[^,]+,\s*RoundMode::CAST_NONE[^)]*\)[^;]*;[^C]*Maxs" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "初始 upcast 用 NONE"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：Maxs/Mins 不存在 | 确认 CANN 版本。旧版本用 `Compare` + `Select` 组合 |
| 运行时：量化值仍越界 | 检查 Maxs/Mins 顺序：必须先 Maxs（下界）再 Mins（上界）。顺序反了结果错误 |
| 运行时：精度不如 baseline | 检查 RoundMode。标准量化用 `CAST_RINT`，QAT 用 `CAST_ROUND`。不要混用 |
| INT8 值被截断为 0 | 检查 quantMin 是否为 -128（不是 0）。有符号 INT8 下界是 -128 |
| quantMax 设错 | UINT8: 255；INT8: 127。FP8_E4M3: ~448；FP8_E5M2: ~57344。需与量化规范对齐 |
| 与 A3 的 RoundMode 冲突 | A3 处理通用 Cast（upcast NONE, downcast RINT），A8 处理量化 Cast（RINT + clamp）。不冲突，但需确认各自作用域 |
| 边界处理遗漏负数 | `Maxs(tmp, tmp, quantMin)` 必须存在。若遗漏，负值可能下溢 |
| 性能下降 | Maxs+Mins 各增加 1 条 Vector 指令，开销极小。若下降明显，检查是否引入了额外同步 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A8 Playbook Completion]
Step 1: done (/tmp/a8_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: quantMin=? quantMax=? RoundMode=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
