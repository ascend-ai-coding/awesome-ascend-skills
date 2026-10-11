# A1 Playbook: FP32 Intermediate Computation (FP32 中间计算实操流程)

> 本 Playbook 为**强制流程**。采纳 A1 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步，禁止"看起来改了"就声明完成。
>
> A1 的核心是**在 FP16/BF16 输入的算子中，将关键中间计算提升到 FP32，避免低精度累加/除法/开方的数值误差**。它与 D1 协同：D1 负责 dtype 支持范围，A1 负责精度链管理。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a1_locations.txt`：

```bash
# 1. 所有计算指令（累加、乘、除、开方等精度敏感操作）
grep -n "Add\|Mul\|Div\|Sqrt\|Exp\|Log\|ReduceSum\|ReduceMax\|ReduceMin" \
    shared/original/op_kernel/*.cpp > /tmp/a1_locations.txt
# 2. 当前 Cast 调用（已有精度转换点）
grep -n "Cast\|RoundMode::CAST" shared/original/op_kernel/*.cpp >> /tmp/a1_locations.txt
# 3. 模板参数 T / calcType / 硬编码类型
grep -n "template\|typename T\|calcType\|using ComputeType" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/a1_locations.txt
# 4. 累加器 / 中间结果变量（norm、sum、mean、var 等）
grep -n "sum\|accum\|mean\|var\|norm\|rms\|std" \
    shared/original/op_kernel/*.cpp >> /tmp/a1_locations.txt
# 5. InitBuffer / UB 分配（临时 FP32 buffer 需要空间）
grep -n "InitBuffer\|LocalTensor" shared/original/op_kernel/*.cpp >> /tmp/a1_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **精度敏感计算点**：所有 Add/Mul/Div/Sqrt/Reduce 的文件 + 行号
- **当前类型**：kernel 是模板参数 `T` 还是硬编码 `half`/`bfloat16_t`
- **累加器位置**：sum/accum/mean/var 等变量的定义和使用位置
- **现有 Cast**：已有的精度转换点
- **UB 余量**：当前 InitBuffer 后是否还有空间容纳 FP32 临时 buffer

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 输入 dtype | `?` (half / bfloat16_t / float) | 不变 | — |
| 计算类型 | `T` 或 `half` | `calcType = conditional<T==half, float, T>` | `?_kernel.cpp:L?` |
| 累加器 dtype | `?` | `float`（FP16 场景必须提升） | `?_kernel.cpp:L?` |
| Cast 上提点 | 无 | FP16→FP32（计算前） | `?_kernel.cpp:L?` |
| Cast 下降点 | 无 | FP32→FP16（输出前） | `?_kernel.cpp:L?` |
| 临时 FP32 buffer | 无 | `tileSize * sizeof(float)` | `?_kernel.cpp:L?` |
| RoundMode | — | `CAST_NONE`（上提）/ `CAST_RINT`（下降） | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 dtype 和计算模式，判断你的代码属于以下哪种形态：

- **形态 α — 纯 FP32**：输入已是 `float` → A1 无意义（已是 FP32 计算）
- **形态 β — FP16 计算未提升**：输入是 `half`，计算直接用 `half`（如 `Add(half, half)`）→ **必须添加 Cast 到 FP32**
- **形态 γ — 已有 D1 多 dtype 支持**：有模板参数 `T`，但 `calcType == T`（未提升）→ 修改 `calcType` 定义，使 `T==half` 时 `calcType=float`

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 最常见）

```cpp
// === 改造前（FP16 直接计算，精度损失）===
__aicore__ inline void Compute(LocalTensor<half> input, LocalTensor<half> output) {
    LocalTensor<half> sumLocal = sumBuf.Get<half>();
    LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
    
    // 精度问题：half 累加 1000 次后误差显著
    Add(sumLocal, sumLocal, input, tileSize);
    Sqrt(tmpLocal, sumLocal, tileSize);
    Div(output, input, tmpLocal, tileSize);
}

// === 改造后（FP16 → FP32 计算 → FP16 输出）===
__aicore__ inline void Compute(LocalTensor<half> input, LocalTensor<half> output) {
    // 临时 FP32 buffer（生命周期：本次 Compute 内）
    LocalTensor<float> inputFp32 = castBuf.Get<float>();
    LocalTensor<float> sumFp32   = sumBufFp32.Get<float>();
    LocalTensor<float> tmpFp32   = tmpBufFp32.Get<float>();
    
    // Step 1: 上提 Cast（FP16 → FP32）
    Cast(inputFp32, input, RoundMode::CAST_NONE, tileSize);
    
    // Step 2: FP32 计算（精度安全区）
    Add(sumFp32, sumFp32, inputFp32, tileSize);
    Sqrt(tmpFp32, sumFp32, tileSize);
    
    // Step 3: 若后续还需 FP32 则保留；若需输出则下降 Cast
    Cast(output, tmpFp32, RoundMode::CAST_RINT, tileSize);
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（纯 FP32）**：A1 不适用。检查瓶颈是否为 compute_bound → 换 P46/P84 优化计算效率。
- **形态 γ（已有 D1 模板）**：
  ```cpp
  // 修改 calcType 定义，使 half 自动提升为 float
  using calcType = std::conditional_t<std::is_same_v<T, half>, float, T>;
  // BF16 保持原类型（尾数 7 位 + 指数 8 位，通常不需要 FP32 提升，除非归约累加）
  ```
  若算子含 **归约/累加**（LayerNorm/RMSNorm/Softmax），即使 BF16 也建议 `calcType = float`。
- **如果 UB 紧张**：FP32 临时 buffer 占用是 FP16 的 2 倍。若溢出，优先保证累加器为 FP32，其他中间量可保持 FP16。
- **多阶段计算链**：若算子有 3+ 阶段连续计算（如 `Mul → Add → Sqrt → Div`），**只在链头 Cast 一次上提，链尾 Cast 一次下降**，中间全程 FP32。禁止每阶段都 Cast，开销会吞噬精度收益。
- **与 A2（Welford）协同**：若算子用 Welford 算法，mean/var 的更新必须在 FP32 下进行；A1 的 `sumFp32` 就是 Welford 的累加器。

## Step 4: 约束复核（防崩溃）

**精度链约束**：
```
FP16 输入 → Cast(UP, CAST_NONE) → FP32 计算 → Cast(DOWN, CAST_RINT) → FP16 输出
BF16 输入 → 可选 Cast(UP)       → FP32/BF16   → 可选 Cast(DOWN)    → BF16 输出
FP32 输入 → A1 不适用
```

**RoundMode 选择**：
| 方向 | RoundMode | 理由 |
|---|---|---|
| FP16→FP32 上提 | `CAST_NONE` | 无损扩展，无需舍入 |
| FP32→FP16 下降 | `CAST_RINT` | 四舍五入，误差最小 |
| FP32→BF16 下降 | `CAST_NONE` | BF16 范围大，直接截断 |

**UB 额外占用**：
```
临时 FP32 buffer 大小 = tileSize × sizeof(float) = tileSize × 4
vs 原 FP16 buffer      = tileSize × sizeof(half)  = tileSize × 2
额外占用              = tileSize × 2（比原方案多一倍）
```

**约束**：`原 UB 占用 + 额外 FP32 buffer ≤ UB_TOTAL × 0.8`
- 若不满足，优先只提升**累加器**为 FP32，其他中间量保持原类型。
- 若仍不满足，回到 D1 的 tileSize 调整或 P8 UB 分区。

**在 implementation_note.txt "Playbook Step 4" 中报告具体计算**（每种 dtype 的占用 + 是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: 存在 Cast 调用（上提或下降至少一处）
grep -cE "Cast\s*\(" modified_files/op_kernel/*.cpp
# 期望: >= 1（形态 beta/gamma 必须有）

# 检查 2: RoundMode 显式指定（不能省略，默认行为不确定）
grep -cE "RoundMode::CAST_NONE|RoundMode::CAST_RINT|RoundMode::CAST_ROUND" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: FP32 中间 tensor 存在（LocalTensor<float> 或 calcType = float）
grep -cE "LocalTensor\s*<\s*float\s*>|calcType.*float|sumFp32|accumFp32" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 无裸 half/bfloat16_t 的精度敏感计算（Add/Mul/Div/Sqrt 等）
# （允许 <= 2 处非精度敏感的计算如 Max/Min；但累加/除法/开方必须 FP32）
grep -En "Add\s*\([^,]+half|Div\s*\([^,]+half|Sqrt\s*\([^)]*half|ReduceSum.*half" \
    modified_files/op_kernel/*.cpp | wc -l
# 期望: == 0（若 >0 说明还有 half 直接参与精度敏感计算）

# 检查 5: 计算链不是每阶段都 Cast（中间结果反复 Cast 开销太大）
# 统计 Cast 调用次数；合理范围：2（上提+下降）~ 4（多分支各一次）
grep -cE "Cast\s*\(" modified_files/op_kernel/*.cpp
# 期望: <= 4（超过说明中间阶段多余 Cast，应合并为全程 FP32）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：UB overflow | FP32 临时 buffer 翻倍；回 Step 4 复算，优先只提升累加器 |
| 编译失败：Cast 模板不匹配 | `Cast(dst, src, mode, count)` 的 dst/src 类型必须对应（float↔half 配对）。检查是否误传了 bfloat16_t 到 half 的 Cast |
| 运行时：FP16 精度仍不达标 | 检查是否遗漏了某条计算路径（如 tail block 仍用 half 计算）。所有路径必须统一走 FP32 |
| 运行时：BF16 结果与 FP32 baseline 差异大 | BF16 尾数 7 位，允许 1e-2 相对误差；若超出，即使 BF16 也建议 `calcType = float`（形态 γ） |
| 性能下降 > 10% | Cast 开销 + FP32 计算量增加。若算子是 memory_bound，A1 帮助有限；若是 compute_bound，FP32 vs FP16 计算量翻倍可能反而更慢 → 权衡精度 vs 性能，或仅提升累加器 |
| 多阶段链每阶段都 Cast，性能极差 | 合并为链头一次上提、链尾一次下降，中间全程 FP32（见 3C 多阶段计算链） |
| Cast 方向反了（FP32→FP16 用了 CAST_NONE） | `CAST_NONE` 截断无舍入，误差大。下降必须用 `CAST_RINT` |
| 与 A2 Welford 同时使用时 mean 精度仍差 | Welford 的 delta 计算也必须在 FP32 下执行，不能只提升累加器 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A1 Playbook Completion]
Step 1: done (/tmp/a1_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB calc: FP32 buffer=elements×4 ≤ UB_TOTAL×0.8: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
