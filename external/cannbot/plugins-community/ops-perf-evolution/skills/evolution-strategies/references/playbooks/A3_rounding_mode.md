# A3 Playbook: Rounding Mode 精度控制

> 本 Playbook 为**强制流程**。采纳 A3 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A3 的核心是**在精度转换（Cast）的关键节点选择正确的 RoundMode，确保 upcast 不引入额外误差、downcast 按场景选择舍入方式**，减少累计精度损失。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a3_locations.txt`：

```bash
# 1. Cast 操作与 RoundMode
grep -n "Cast\s*(\|RoundMode\|CAST_NONE\|CAST_RINT\|CAST_FLOOR\|CAST_CEIL\|CAST_TRUNC\|CAST_ROUND" \
    shared/original/op_kernel/*.cpp > /tmp/a3_locations.txt
# 2. 精度类型
grep -n "bf16\|bfloat16\|half\|fp16\|float32\|fp32" \
    shared/original/op_kernel/*.cpp >> /tmp/a3_locations.txt
# 3. 当前同步机制（与 A4 协同）
grep -n "PipeBarrier\|SyncAll\|SetFlag\|WaitFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/a3_locations.txt
# 4. 计算序列（Cast→Compute→Cast）
grep -n "Add\s*(\|Mul\s*(\|Sub\s*(\|Div\s*(\|Exp\s*(\|Sqrt\s*(" \
    shared/original/op_kernel/*.cpp >> /tmp/a3_locations.txt
# 5. 输入输出精度
grep -n "inDataType\|outDataType\|input.*Type\|output.*Type\|dtype\|dataType" \
    shared/original/op_kernel/*.cpp >> /tmp/a3_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Cast 位置**：所有 `Cast` 调用位置、当前 RoundMode 类型
- **精度类型**：BF16/FP16/FP32 的分布（输入、中间、输出）
- **同步机制**：是否有 PipeBarrier/SyncAll（A4 协同点）
- **Compute 序列**：Cast→Compute→Cast 的完整链路
- **输入输出精度**：数据流向和精度转换节点

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Cast 次数 | `?` | 不变 | `?_kernel.cpp:L?` |
| 当前 RoundMode | `?` (全部 CAST_NONE / 混合) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| 精度链 | `?` (BF16→FP32→BF16 / FP16→FP32→FP16) | 不变 | `?_kernel.cpp:L?` |
| Upcast RoundMode | `?` (NONE/RINT) | CAST_NONE | `?_kernel.cpp:L?` |
| Downcast RoundMode | `?` (NONE/RINT) | CAST_RINT | `?_kernel.cpp:L?` |
| 同步点 | `?` (无/SyncAll) | PipeBarrier<PIPE_V>（若 A4 适用） | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的精度链和舍入场景，判断你的代码属于以下哪种形态：

- **形态 α — BF16/FP16 计算链（最常见）**：低精度输入 → FP32 中间计算 → 低精度输出。Upcast 用 `CAST_NONE`，downcast 用 `CAST_RINT`。
- **形态 β — 量化场景（整数舍入）**：FP32 结果需要量化为 INT8/INT32，用 `CAST_ROUND` 或 `CAST_FLOOR`（视量化策略而定）。
- **形态 γ — 多阶段混合精度链**：链很长（如 BF16→FP32→BF16→FP32→BF16），每段 upcast/downcast 需独立判断。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — BF16/FP16 计算链，最常见）

```cpp
// === 改造前（全部 CAST_NONE，精度损失）===
__aicore__ inline void ComputeBf16(...) {
    // Upcast: BF16 → FP32（NONE 没问题，但需确认）
    Cast(fp32Buf, bf16In, RoundMode::CAST_NONE, count);
    // Compute in FP32
    Add(fp32Buf, fp32Buf, bias, count);
    // Downcast: FP32 → BF16（❌ NONE 会截断，累积误差大）
    Cast(bf16Out, fp32Buf, RoundMode::CAST_NONE, count);
}

// === 改造后（upcast NONE，downcast RINT）===
__aicore__ inline void ComputeBf16(...) {
    // Step 1: Upcast BF16 → FP32（CAST_NONE：保留原始值，不引入舍入）
    Cast(fp32Buf, bf16In, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();  // A4 协同：确保 Cast 完成
    
    // Step 2: Compute in FP32
    Add(fp32Buf, fp32Buf, bias, count);
    PipeBarrier<PIPE_V>();
    
    // Step 3: Downcast FP32 → BF16（CAST_RINT：四舍五入，减少累积误差）
    Cast(bf16Out, fp32Buf, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（量化场景）**：
  当算子涉及量化（如 FakeQuant、Quantize/Dequantize）时，舍入模式需对齐量化规范：
  ```cpp
  // 对称量化：四舍五入到最近整数
  Cast(int32Buf, fp32Buf, RoundMode::CAST_ROUND, count);
  
  // 非对称量化/取整：向下取整（floor）
  Cast(int32Buf, fp32Buf, RoundMode::CAST_FLOOR, count);
  
  // 截断量化：直接截断小数
  Cast(int32Buf, fp32Buf, RoundMode::CAST_TRUNC, count);
  ```
  形态 β 必须与量化规范文档对齐，不能随意选择 RoundMode。

- **形态 γ（多级精度链）**：
  多个 Cast-Compute 阶段串联时，每段的 downcast 都需用 `CAST_RINT`：
  ```cpp
  for (uint32_t i = 0; i < stages; i++) {
      // Upcast: 低精度 → FP32
      Cast(fp32Buf, lowPrecIn[i], RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      
      // Compute
      Compute(fp32Buf, ...);
      PipeBarrier<PIPE_V>();
      
      // Downcast: FP32 → 低精度（每段 downcast 都用 RINT）
      Cast(lowPrecOut[i], fp32Buf, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
  }
  ```

- **与 A1 的协同**：A1（FP32 Intermediate）要求中间计算用 FP32，A3 要求 Cast 到 FP32 时用 `CAST_NONE`、Cast 回低精度时用 `CAST_RINT`。两者必须同时使用：A1 决定精度链结构，A3 决定舍入模式。
- **与 A4 的协同**：A4（PipeBarrier）在 Cast→Compute→Cast 之间插入同步。A3 修改 RoundMode 时，若算子已有 A4 的 barrier，只需在已有 barrier 基础上调整 RoundMode 参数；若无 A4，建议同时引入（精度链 + 同步 + 舍入 = 完整方案）。
- **与 A5 的边界**：A5（Softmax 数值稳定）处理 exp 溢出，A3 处理 Cast 精度。两者独立但常一起出现：A5 做 softmax 稳定，A3 做精度转换 rounding。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: Upcast（低精度→FP32）必须用 CAST_NONE，不能用 RINT（RINT 会在 upcast 时引入虚假舍入）
约束 2: Downcast（FP32→低精度）必须用 CAST_RINT 或 CAST_ROUND，不能用 NONE（NONE 会截断，误差大）
约束 3: 量化场景的 RoundMode 必须与量化规范对齐（对称/非对称/截断）
约束 4: RoundMode 修改不改变 buffer 大小和数据类型，只改变舍入行为
约束 5: 若算子同时有 A4 的 PipeBarrier，RoundMode 修改后需确认 barrier 位置未变
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `Cast 总数 = ?`, `Upcast 次数 = ?`, `Downcast 次数 = ?`
- `RoundMode 修改: NONE→? = ? 处, RINT→? = ? 处`
- `精度链 = ?`（如 BF16→FP32→BF16）
- `同步点数量 = ?`（A4 协同）
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 Cast 操作
grep -cE "Cast\s*\(" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: Upcast 用 CAST_NONE
grep -cE "Cast\s*\([^,]+,\s*[^,]+,\s*RoundMode::CAST_NONE" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: Downcast 用 CAST_RINT / CAST_ROUND / CAST_FLOOR / CAST_CEIL / CAST_TRUNC
grep -cE "Cast\s*\([^,]+,\s*[^,]+,\s*RoundMode::CAST_RINT|RoundMode::CAST_ROUND|RoundMode::CAST_FLOOR|RoundMode::CAST_CEIL|RoundMode::CAST_TRUNC" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 无全部 CAST_NONE（至少一个 downcast 不是 NONE）
grep -cE "Cast\s*\([^,]+,\s*[^,]+,\s*RoundMode::CAST_NONE[^)]*\)[^;]*;[^C]*Cast\s*\([^,]+,\s*[^,]+,\s*RoundMode::CAST_NONE" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "纯 FP32 算子，无 downcast"）

# 检查 5: 精度类型匹配（BF16/FP16 与 float 互转）
grep -cE "bf16.*float|float.*bf16|half.*float|float.*half|bfloat16.*float|float.*bfloat16" modified_files/op_kernel/*.cpp
# 期望: >= 1（或 note 中说明 "纯 FP32 算子，无精度转换"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：RoundMode 不存在 | 确认 CANN 版本支持 `RoundMode` 枚举。旧版本可能用 `RoundMode::CAST_NONE` 的等价写法或不同命名空间 |
| 运行时：精度仍不如 baseline | 检查是否所有 downcast 点都改了 RoundMode。常见遗漏：最后一个输出 Cast 仍用 NONE |
| 运行时：upcast 用 RINT 导致精度下降 | Upcast 必须用 `CAST_NONE`。若误用 RINT，低精度→FP32 时会引入舍入，损失原始信息 |
| 量化场景精度不对 | 确认量化规范要求的舍入方式。对称量化通常用 `CAST_ROUND`，非对称可能用 `CAST_FLOOR`。不要用错 |
| BF16 输出出现大范围偏差 | BF16 只有 7 位尾数，`CAST_RINT` 和 `CAST_NONE` 差异在特定值域才明显。若偏差大，检查是否有中间 Cast 遗漏 |
| 与 A4 的 PipeBarrier 冲突 | A3 只改 RoundMode 参数，不改 barrier 位置。若 barrier 被误删，重新加回 |
| 多级链中某段 downcast 漏改 | 每段 downcast 独立检查。建议用 grep 统计所有 `Cast(.*,.*, RoundMode::` 的分布 |
| Cast 目标类型与 RoundMode 不匹配 | `CAST_RINT` 用于浮点→浮点或浮点→整数的 downcast。整数→浮点的 upcast 用 `CAST_NONE` |
| 旧代码用 `Round()` 指令而非 `Cast()` | `AscendC::Round()` 是独立指令，不通过 RoundMode 控制。若基线用 `Round()`，需评估是否替换为 `Cast+CAST_RINT` |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A3 Playbook Completion]
Step 1: done (/tmp/a3_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: cast_count=? upcast_none=? downcast_rint=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
