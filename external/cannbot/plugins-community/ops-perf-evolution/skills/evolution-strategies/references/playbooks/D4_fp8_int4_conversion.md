# D4 Playbook: FP8/INT4 量化输出类型

> 本 Playbook 为**强制流程**。采纳 D4 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> D4 的核心是**在 FP8/INT4 低比特量化中，支持多种 rounding mode（RINT/ROUND/FLOOR/TRUNC/CEIL），根据训练/推理场景选择最合适的舍入方式，避免系统性偏差**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/d4_locations.txt`：

```bash
# 1. FP8/INT4 类型
grep -n "FP8|fp8|INT4|int4|FLOAT8|E4M3|E5M2|HIFLOAT8|DT_FLOAT8|DT_INT4" \
    shared/original/op_kernel/*.cpp > /tmp/d4_locations.txt
# 2. Cast 与 RoundMode
grep -n "Cast|cast|RoundMode|RINT|ROUND|FLOOR|TRUNC|CEIL|MODE_RINT|MODE_ROUND" \
    shared/original/op_kernel/*.cpp >> /tmp/d4_locations.txt
# 3. 量化参数
grep -n "quant_min|quant_max|scale|zero.*point|amax|min.*max" \
    shared/original/op_kernel/*.cpp >> /tmp/d4_locations.txt
# 4. 当前 rounding 方式
grep -n "CAST_NONE|CAST_RINT|CAST_ROUND|CAST_FLOOR|CAST_TRUNC|CAST_CEIL" \
    shared/original/op_kernel/*.cpp >> /tmp/d4_locations.txt
# 5. 数据类型配置
grep -n "DT_FLOAT16|DT_BF16|DT_INT8|DT_FLOAT|half|bf16|float" \
    shared/original/op_kernel/*.cpp >> /tmp/d4_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **FP8/INT4 位置**：所有 FP8/INT4 类型引用位置
- **Cast 位置**：所有 `Cast` 调用位置、当前 RoundMode
- **量化参数**：quant_min/quant_max/scale/amax
- **当前 rounding 方式**：NONE/RINT/ROUND/FLOOR/TRUNC/CEIL
- **输入输出类型**：FP16/BF16/FP32 → FP8/INT4

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 输出类型 | `?` (FP8/INT4/INT8) | 不变 | `?_kernel.cpp:L?` |
| 当前 rounding | `?` (NONE/RINT/ROUND) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| 量化参数 | `?` (无/scale/amax) | 有 | `?_kernel.cpp:L?` |
| 输入精度 | `?` (FP16/BF16/FP32) | 不变 | `?_kernel.cpp:L?` |
| 场景 | `?` (训练/推理) | 匹配 rounding mode | `?_kernel.cpp:L?` |
| FP8 子类型 | `?` (E4M3/E5M2/无) | 明确指定 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 FP8 子类型和场景，判断你的代码属于以下哪种形态：

- **形态 α — FP8 E4M3/E5M2（推理场景，最常见）**：FP8 输出强制使用 `MODE_RINT`（Banker's rounding），符合标准。E4M3 用于权重/激活，E5M2 用于梯度。
- **形态 β — INT4 量化（存储压缩）**：INT4 范围 [-8, 7]，需要 FLOOR/TRUNC 舍入，根据硬件要求选择。
- **形态 γ — 训练量化（QAT）**：训练时通常用 `MODE_RINT` 减少累积误差，推理时可能切换。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — FP8 E4M3/E5M2）

```cpp
// === 改造前（固定 CAST_ROUND）===
__aicore__ inline void QuantFp8Naive(LocalTensor<float> input,
                                      LocalTensor<float8_t> output,
                                      float amax, uint32_t count) {
    // x / amax
    Muls(tmp, input, 1.0f / amax, count);
    // ❌ 固定 ROUND，不符合 FP8 标准
    Cast(output, tmp, RoundMode::CAST_ROUND, count);
}

// === 改造后（FP8 标准 RINT + 子类型区分）===
__aicore__ inline void QuantFp8Safe(LocalTensor<float> input,
                                     LocalTensor<float8_t> output,
                                     LocalTensor<float> tmp,
                                     float amax, uint32_t count) {
    // Step 1: x / amax
    Muls(tmp, input, 1.0f / amax, count);
    
    // Step 2: FP8 标准舍入 — MODE_RINT（Banker's rounding）
    Cast(output, tmp, RoundMode::CAST_RINT, count);
    
    // Step 3: clamp 到 FP8 有效范围（根据子类型）
    // E4M3: max ≈ 448.0, E5M2: max ≈ 57344.0
    // 具体范围由硬件/规范决定
}

// Host 侧：配置多输出类型
static const std::vector<ge::DataType> yDataType = {
    ge::DT_INT8,
    ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E5M2,
    ge::DT_HIFLOAT8
};
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（INT4 量化）**：
  INT4 范围 [-8, 7]，需要显式 clamp 和特定舍入：
  ```cpp
  // INT4 量化
  Cast(tmp, input, RoundMode::CAST_FLOOR, count);  // 或 CAST_TRUNC
  Maxs(tmp, tmp, -8, count);
  Mins(tmp, tmp, 7, count);
  Cast(output, tmp, RoundMode::CAST_NONE, count);
  ```

- **形态 γ（训练量化，多 mode 切换）**：
  根据场景参数选择 rounding mode：
  ```cpp
  switch (roundMode) {
      case MODE_RINT:
          Cast(output, tmp, RoundMode::CAST_RINT, count); break;
      case MODE_ROUND:
          Cast(output, tmp, RoundMode::CAST_ROUND, count); break;
      case MODE_FLOOR:
          Cast(output, tmp, RoundMode::CAST_FLOOR, count); break;
      case MODE_TRUNC:
          Cast(output, tmp, RoundMode::CAST_TRUNC, count); break;
      case MODE_CEIL:
          Cast(output, tmp, RoundMode::CAST_CEIL, count); break;
  }
  ```

- **与 A8 的协同**：A8（量化专用精度）处理标准量化边界 [quant_min, quant_max]，D4 处理低比特类型（FP8/INT4）的特殊舍入。两者必须同时使用：A8 做边界限制，D4 做类型特化舍入。
- **与 A3 的边界**：A3（Rounding Mode）处理通用 Cast 精度，D4 处理低比特特化舍入。D4 的 rounding mode 选择应优先于 A3 的通用规则。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: FP8 E4M3 输出必须用 MODE_RINT（标准强制）
约束 2: INT4 范围 [-8, 7]，Maxs/Mins 的边界不能错
约束 3: amax 必须 > 0，若 amax <= 0 需兜底（输出全 0 或报错）
约束 4: 不同 FP8 子类型（E4M3/E5M2/HIFLOAT8）的范围不同，不能混用
约束 5: 训练时推荐 MODE_RINT，推理时根据硬件要求选择
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `输出类型 = ?`, `FP8 子类型 = ?`
- `RoundMode = ?`, `amax = ?`
- `clamp 范围 = [?, ?]`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 FP8 或 INT4 类型引用
grep -cE "FP8|fp8|INT4|int4|FLOAT8|E4M3|E5M2|HIFLOAT8" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 CAST_RINT 或 CAST_ROUND 或 CAST_FLOOR 或 CAST_TRUNC 或 CAST_CEIL
grep -cE "CAST_RINT|CAST_ROUND|CAST_FLOOR|CAST_TRUNC|CAST_CEIL" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 无全部 CAST_NONE
grep -cE "Cast\s*\([^,]+,\s*[^,]+,\s*RoundMode::CAST_NONE" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "输入 upcast 用 NONE"）

# 检查 4: 有量化参数（amax/scale/quant_min）
grep -cE "amax|scale|quant_min|quant_max" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 有多输出类型配置
grep -cE "DT_FLOAT8|DT_INT4|yDataType|output.*DataType" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：DT_FLOAT8_E4M3FN 不存在 | 确认 CANN 版本。旧版本可能用不同的 FP8 类型名 |
| 运行时：FP8 输出不符合标准 | FP8 强制用 `MODE_RINT`。若用 `MODE_ROUND` 或其他，可能不符合硬件标准 |
| INT4 值溢出 | INT4 范围 [-8, 7]。检查 Maxs/Mins 边界是否正确。不要设成 [0, 15]（那是 UINT4） |
| amax = 0 导致除零 | 添加 `if (amax <= 0) { Duplicate(output, 0.0f, count); return; }` |
| E4M3/E5M2 混用 | E4M3 适合权重/激活（精度高），E5M2 适合梯度（范围大）。不要反过来 |
| 训练累积误差大 | 训练量化用 `MODE_RINT`（Banker's rounding），比 `MODE_ROUND` 更稳定 |
| 与 A8 的边界冲突 | A8 处理 [quant_min, quant_max] 的 Maxs/Mins，D4 处理 FP8/INT4 的类型特化。两者不冲突，但需确认 clamp 范围与类型匹配 |
| HIFLOAT8 范围错误 | HIFLOAT8 的范围与 E4M3/E5M2 不同，不能共用同一套 clamp 值 |
| 推理性能下降 | 多 rounding mode 只在 host 侧选择一次，kernel 内是固定的，不影响性能 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[D4 Playbook Completion]
Step 1: done (/tmp/d4_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: output_type=? RoundMode=? amax=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
