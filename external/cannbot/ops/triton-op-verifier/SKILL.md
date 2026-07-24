---
name: external-cannbot-ops-triton-op-verifier
description: '算子代码验证 Skill — 按照标准验证流程验证生成的内核代码。 创建验证项目文件，调用 scripts/verify.py 运行验证，验证通过后
  调用 scripts/benchmark.py 进行性能测试并收集结果。 触发：当用户需要验证 Triton 算子代码功能正确性或采集其性能数据时使用。

  '
argument-hint: 输入：generated-code-path、task-file-path、op-name、warmup、repeats。 输出：验证结果（成功/失败）、错误信息、性能数据。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
original-name: triton-op-verifier
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Kernel Verifier Skill

<role>
你是一个内核代码验证专家。你的任务是按照标准验证流程，创建验证项目并运行，检查生成的算子代码是否能正确编译运行且与参考实现的输出一致。验证通过后，执行性能测试并收集性能数据。
</role>

## 精度判定规则

> 本节是 verify.py 精度判定的**唯一权威说明**。所有阈值、决策矩阵、前置检查只在此处定义；下游章节（Step 2/3）只引用、不重复。

verify.py 按"`--non-compute` 开关 + 输入 dtype + 输出 dtype"分流到 **5 类判定路径**。

### 1. 输入类型推断（KernelBench / NPUKernelBench 统一）

从实际传入对象推断（不依赖 task 文件结构化 spec）：

1. 存在 `torch.Tensor` 输入 → 取所有 tensor 中**最高精度 dtype**
   （例：输入为 `[fp16 tensor, fp32 tensor, int64 tensor]` → 取 fp32）
2. 否则存在 `list/tuple of Tensor`（tensor_list）→ 取首个 tensor_list 首元素 dtype
3. 否则视为**无 tensor 输入**（`no_tensor`，最严路径）

**dtype 优先级**：

精度从高到低排列：float64 > float32 > float16 > bfloat16 > float8_e4m3 / float8_e5m2 > int64 > int32 > int16 > int8 / uint8 > bool

**`input_type` 二分类**（用于 §2 决策矩阵分流）：

- `input_type = float`：输入最高精度 dtype 属于浮点族（float64/32/16、bfloat16、float8_e4m3/e5m2、复数 complex64/128）
- `input_type = int`：输入最高精度 dtype 属于整型族（int64/32/16/8、uint8、bool）
- `input_type = no_tensor`：无任何 tensor 输入（走最严判定）

> **关于 bool 的两处特殊性**（避免混淆）：
> - bool 作**输入**：归入 `int`，与 int 系输入等同；分流时只看输出 dtype（如输出 fp 走浮点判定）
> - bool 作**输出**：不进入 input_type 分流，直接走 §2 "bool 输出"路径（`torch.equal` 严格相等）

### 2. 五类判定决策矩阵

| 类别 | 输入 type | 输出 dtype | `--non-compute` | 误差要求 |
|---|---|---|---|---|
| **非计算类** | 任意 | 任意 | **是** | 二进制完全一致（view-as-int 比对，含 NaN bit pattern） |
| **bool 输出** | 任意 | bool | 否 | `torch.equal` 严格相等 |
| **整数计算类** | int / no_tensor | int | 否 | `\|actual − golden\| == 0` |
| **量化计算类 fp→int** | float | int | 否 | `\|actual − golden\| <= 1` |
| **浮点计算类** | 任意 | float | 否 | 三项 AND（见 §4） |

### 3. 比对前置检查（按顺序，任一失败即判 fail）

1. 形状必须一致
2. NaN 位置必须完全一致（mask 按位相等）
3. Inf 位置和符号必须完全一致
4. `bool` dtype：要求 `torch.equal` 完全相等，不进入精度判定
5. 仅在 `finite_mask`（双方都 finite）上做精度计算；dtype 不一致时 impl 会被 cast 到 golden 的 dtype

### 4. 浮点计算类：三项 AND 整体判定

#### 4.1 元素级 matched 定义（分桶）

对每个 finite 元素 `i`，按 `|golden[i]|` 落入的类别分别判定：

- **小值域** `|golden[i]| < small_value_threshold`：
  `matched[i] = (|actual[i] - golden[i]| <= small_value_error)`
- **正常域** `|golden[i]| >= small_value_threshold`：
  `matched[i] = (|actual[i] - golden[i]| / (|golden[i]| + 1e-7) <= rel_threshold)`

> 计算前两侧统一升 float32，避免低精度 dtype 自身误差污染。
> 分母 `+1e-7` 仅为保险——正常域里 `|golden| >= sv_thr ≫ 1e-7`。

#### 4.2 三项通过条件（AND，全部满足才算通过）

1. **`max_error_cap`**：所有 finite 元素满足 `|diff| <= atol + rtol * |golden|`（dtype-aware，要求 100% 通过）
2. **`required_matched_ratio`**：`sum(matched) / total_finite >= 0.9`
3. **`MERE`**：对所有 finite 元素计算 `rel_err = |diff| / (|golden| + 1e-7)` 再取均值，要求 `MERE < rel_threshold`。当 `total_finite == 0` 时本项自动通过。

#### 4.3 阈值表

**matched_mask 与 MERE 阈值**（沿用 NPU Benchmark 标准）：

| 数据类型 | small_value_threshold | small_value_error | rel_threshold (= MERE 上限) |
|---|---|---|---|
| `float16` | 2⁻¹¹ ≈ 4.88e-4 | 2⁻¹⁶ ≈ 1.53e-5 | 2⁻¹⁰ ≈ 9.77e-4 |
| `bfloat16` | 2⁻⁸ ≈ 3.91e-3 | 2⁻¹⁶ ≈ 1.53e-5 | 2⁻⁷ ≈ 7.81e-3 |
| `float32` | 2⁻¹⁴ ≈ 6.10e-5 | 2⁻³⁰ ≈ 9.31e-10 | 2⁻¹³ ≈ 1.22e-4 |
| `hifloat32` | 2⁻¹² ≈ 2.44e-4 | 2⁻²⁸ ≈ 3.73e-9 | 2⁻¹¹ ≈ 4.88e-4 |
| `float8_e4m3` | 2⁻⁴ = 0.0625 | 2⁻⁶ ≈ 0.015625 | 2⁻³ = 0.125 |
| `float8_e5m2` | 2⁻³ = 0.125 | 2⁻⁵ = 0.03125 | 2⁻² = 0.25 |
| 其他 dtype（fallback） | 2⁻¹⁴ | 2⁻³⁰ | 2⁻¹³ |

**max_error_cap 阈值**（`|diff| <= atol + rtol * |golden|`）：

| 数据类型 | atol | rtol |
|---|---|---|
| `float16` | 9e-2 | 2⁻¹⁰ ≈ 9.77e-4 |
| `bfloat16` | 1e-1 | 2⁻⁷ ≈ 7.81e-3 |
| `float32` | 1e-3 | 2⁻¹³ ≈ 1.22e-4 |
| 其他 dtype（fallback） | 1e-3 | 2⁻¹³ |

### 5. 运行时诊断输出

verify.py 在每个 case 会向日志输出：

- `[输入类型判定] 来源=...，候选 dtypes=...，最高精度=...，input_type=...`
- `[评测模式] 模式=...，输入 dtype=...，输出 dtype=...，误差要求=...`

便于上游 agent 立即看到当前 case 落入了哪一类、用了哪些阈值。

---

## 验证流程

```
输入：generated_code.py + task_file.py
    ↓
[0. Triton 退化预检查] → scripts/validate_triton_impl.py (AST 静态分析)
    ↓ (通过)
[1. 创建验证项目] → 两个文件
    ↓
[2. 执行验证脚本] → scripts/verify.py --op_name ...
    ↓
[3. 收集验证结果]
    ↓
[验证通过] → [4. 执行性能测试] → scripts/benchmark.py --op_name ...
    ↓
[5. 收集性能结果]
    ↓
输出：验证结果 + 性能数据
```

---

## Step 0: Triton 退化预检查（AST 静态分析）

在创建验证项目之前，先使用 `validate_triton_impl.py` 对生成代码进行退化检测。此检查为纯 AST 静态分析，无需 NPU/torch 运行时，毫秒级完成。

**命令模板**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/validate_triton_impl.py \
    <生成代码文件路径> --json
```

**检测三种退化类型**：

| 类型 | 含义 | 检测方式 |
|------|------|---------|
| Type 1 | 完全无 `@triton.jit` kernel | AST 中无 `triton.jit` 装饰的函数定义 |
| Type 2 | 有 kernel 但 `forward()` 未调用 | kernel 定义存在但 `ModelNew.forward()` 未引用（含 wrapper 函数追踪） |
| Type 3 | 部分计算使用 PyTorch | `forward()` 中存在禁止的 `torch.*` / `F.*` 计算操作（精确到行号） |

**结果判断**：
- exit code == 0 → 通过，继续 Step 1
- exit code != 0 → 退化检测到，解析 JSON 中的 `regression_type` 和 `suggestion`，直接返回失败

**JSON 输出格式**：

```json
{
  "valid": false,
  "regression_type": 3,
  "checks": {
    "triton_kernel_exists": {"passed": true, "kernels": [...]},
    "kernel_called_from_forward": {"passed": true, "called": [...]},
    "no_forbidden_torch_ops": {"passed": false, "violations": [{"line": 45, "call": "F.softmax", "reason": "..."}]}
  },
  "suggestion": "..."
}
```

---

## Step 1: 创建验证项目

在当前迭代的验证目录（如 `{output-path}/iter_{iteration}/verify/`）下创建两个文件：

### 文件 1: `{op_name}_torch.py`

直接复制任务文件的完整内容。此文件包含 `Model`、`get_inputs()`、`get_init_inputs()`。

### 文件 2: `{op_name}_triton_ascend_impl.py`

直接复制生成代码的完整内容。此文件包含 `ModelNew` 类。

---

## Step 2: 执行验证（⚠️ 必须使用本脚本，禁止自创测试方法）

**必须使用** `bash` 工具调用本 skill 自带的 `scripts/verify.py` 脚本。

**命令模板**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/verify.py \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --triton_impl_name <triton实现模块名> \
    --timeout 900
```

**实际调用示例**（假设验证目录为 `/tmp/workspace/softmax/verify`，算子名为 `softmax`）：

```bash
python3 /path/to/triton-op-verifier/scripts/verify.py \
    --op_name softmax \
    --verify_dir /tmp/workspace/softmax/verify \
    --triton_impl_name triton_ascend_impl \
    --timeout 900
```

**参数说明**：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--op_name` | 是 | 算子名称，与文件名前缀对应 |
| `--verify_dir` | 否 | 验证目录路径，默认当前目录 |
| `--triton_impl_name` | 否 | Triton 实现模块名（不含 `{op_name}_` 前缀），默认 `triton_ascend_impl` |
| `--timeout` | 否 | 超时秒数，默认 900 |
| `--output` | 否 | 验证结果 JSON 输出路径，默认 `{verify_dir}/verify_result.json` |
| `--non-compute` | 否 | 适用于非计算类算子（不做数值运算、只对张量进行形状变换、维度重排、切分拼接、索引、类型转换等数据重组操作的算子，常见如 Reshape、Transpose、Concat、Split、Gather、Cast、Pad 等），强制走二进制完全一致判定 |

**超时设置**：默认 900 秒，复杂算子可适当增加。

**注意事项**：
- 禁止自己编写 Python 代码来测试算子（如手动 import 并 forward 比较）
- 禁止使用 `torch.allclose` 或其他自创方法替代 `scripts/verify.py`
- 禁止跳过此步骤直接报告验证结果
- 禁止对计算类算子（含数值运算）传 `--non-compute`；该开关仅适用于不做数值运算、只做形状变换 / 维度重排 / 切分拼接 / 索引 / 类型转换等数据重组操作的算子（如 Reshape、Transpose、Concat、Split、Gather、Cast、Pad）。误用会强制走二进制完全一致判定，把正常的浮点舍入差异判为失败
- 对非计算类算子（例如形状变换、维度重排、切分拼接、索引、类型转换等数据重组操作的算子）**一定要**传 `--non-compute`；漏传会让此类算子按浮点三项判定走，容许超出预期的差异，无法识别真正的位级不一致

---

## Step 3: 收集验证结果

verify.py 会在 `verify_dir` 下生成 `verify_result.json`（或 `--output` 指定路径），包含：

```json
{
  "op_name": "softmax",
  "total_cases": 5,
  "passed_cases": 4,
  "failed_cases": 1,
  "failures": [
    {
      "case_idx": 2,
      "input_desc": [
        {"type": "tensor", "shape": [128, 256], "dtype": "torch.float16"}
      ],
      "error_type": "CompilationError",
      "error_msg": "..."
    }
  ]
}
```

**精度失败时的 `metrics` 字段**：当 `error_type == "AccuracyError"`（浮点三项判定未通过）时，`failures[*]` 会带上结构化 `metrics`，便于下游分类失败原因（max_error_cap 违例 / 离群点过多 / 平均误差偏大）：

```json
{
  "case_idx": 1,
  "input_desc": [...],
  "error_type": "AccuracyError",
  "error_msg": "...",
  "metrics": {
    "matched_ratio": 0.95,
    "max_abs_diff": 0.2,
    "MERE": 2.0e-4,
    "rel_threshold": 1.22e-4,
    "small_value_threshold": 6.10e-5,
    "small_value_error": 9.31e-10,
    "atol": 1.0e-3,
    "rtol": 1.22e-4,
    "max_error_cap_violation_count": 12,
    "required_matched_ratio": 0.9,
    "total_finite": 1000,
    "matched_count": 950,
    "small_count": 0,
    "normal_count": 1000,
    "checks": {
      "max_error_cap": false,
      "required_matched_ratio": false,
      "MERE": false
    }
  }
}
```

`checks` 三个布尔位标记每项判定是否独立通过。阈值定义见上文 §精度判定规则 §4。

非浮点类失败（`non_compute` / `bool_output` / `integer_compute` / `quant_fp_to_int`）的 `metrics` 字段较简单，含 `category` / `violation_count` / `total_*` 等基本计数。

**多 shape 行为**：每个 shape 独立 try/except，失败不中止后续 shape；全部跑完才落盘并退出。

**退出码语义（策略 A：严格）**：
- `passed_cases == total_cases` 且 `total_cases > 0` → exit 0，`verifier_result = true`
- 否则（`passed_cases < total_cases`，或 `total_cases == 0`）→ exit 1，`verifier_result = false`，`verifier_error` 应读取 `verify_result.json.failures` 的**全部条目**（不是第一个），汇总后提交给 Conductor。

**超时**：脚本输出 `"验证超时"` 且退出码为 1 → `verifier_error = "验证超时（{timeout}秒）"`。

---

## Step 4: 执行性能测试（验证通过后执行）

**前置条件（L1 脚本层强制）**：benchmark.py 启动时会自动按 `--triton_impl_name` 推导对应的 `verify_result` 文件并校验 `passed_cases == total_cases`；不通过时直接 **exit 2**，禁止运行 benchmark。详见下方"L1 verify 闸门"小节。

仅在 verify.py 的 `passed_cases == total_cases` 时执行（策略 A）。verify 有任何失败 → 禁止执行 benchmark.py。

使用 `bash` 工具调用本 skill 自带的 `scripts/benchmark.py` 脚本。

**命令模板**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/benchmark.py \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --triton_impl_name <triton实现模块名> \
    --warmup <warmup次数> \
    --repeats <测试次数> \
    --output <输出文件路径>
```

**实际调用示例**：

```bash
python3 /path/to/triton-op-verifier/scripts/benchmark.py \
    --op_name softmax \
    --verify_dir /tmp/workspace/softmax/verify \
    --triton_impl_name triton_ascend_impl \
    --warmup 5 \
    --repeats 50 \
    --output /tmp/workspace/softmax/iter_0/perf_result.json
```

> **注意**：`--output` 路径由调用方指定，性能报告将写入该路径。通常由 `kernelgen-workflow` SubAgent 指定为 `{output-path}/iter_{iteration}/perf_result.json`。

**参数说明**：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--op_name` | 是 | 算子名称 |
| `--verify_dir` | 否 | 验证目录路径，默认当前目录 |
| `--triton_impl_name` | 否 | Triton 实现模块名（不含 `{op_name}_` 前缀），默认 `triton_ascend_impl` |
| `--warmup` | 否 | warmup 次数，默认 5 |
| `--repeats` | 否 | 正式测试次数，默认 50 |
| `--output` | 否 | 性能报告输出路径（JSON 格式）|
| `--verify_not_required` | 否 | 跳过 L1 verify 闸门（默认强制要求 verify_result 全过）|

---

### L1 verify 闸门

benchmark.py 启动时按 `--triton_impl_name` 推导对应的 verify_result 文件名：

| triton_impl_name | 对应 verify json |
|-----------------|----------------|
| `triton_ascend_impl`（默认，Phase 3）| `verify_result.json` |
| `triton_baseline`（Phase 4 baseline）| `verify_result_baseline.json` |
| `triton_optimized`（Phase 4 optimized）| `verify_result_optimized.json` |
| 其他 `triton_xxx` | `verify_result_xxx.json` |

**判定规则**（默认开启，传 `--verify_not_required` 可跳过）：

| 情况 | 退出码 | 说明 |
|------|-------|------|
| 文件不存在 | exit 2 | 必须先跑 verify.py |
| 文件读取失败 | exit 2 | JSON 损坏 |
| `total_cases == 0` | exit 2 | verify 未实际跑任何 shape |
| `passed_cases < total_cases` | exit 2 | 精度未全过，benchmark 无意义且会传染下游 |
| `passed_cases == total_cases > 0` | 继续执行 benchmark | — |

**exit 2 时 stderr 会打印**：verify_json 路径 / passed/total / 前 5 条 failures，便于上游 agent 把错误等价映射到 verify 失败处理路径。

---

## Step 5: 收集性能结果

性能测试完成后，从 `--output` 指定的 JSON 文件中读取结果。

### 性能报告格式

```json
{
  "op_name": "softmax",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 3,
  "passed_cases": 3,
  "failed_cases": 0,
  "nan_indices": [],
  "inf_indices": [],
  "zero_indices": [],
  "negative_indices": [],
  "none_indices": [],
  "framework": {
    "avg_latency_ms": 1.2345,
    "peak_memory_mb": 256.00,
    "operators": {"...": 0.0}
  },
  "implementation": {
    "avg_latency_ms": 0.5678,
    "peak_memory_mb": 128.00,
    "operators": {"...": 0.0}
  },
  "speedup_vs_torch": 2.1746,
  "per_shape_results": [
    {
      "case_idx": 1,
      "input_desc": [{"type":"tensor","shape":[128,256],"dtype":"torch.float16"}],
      "status": "pass",
      "framework": {"avg_latency_ms": 1.23, "peak_memory_mb": 64.0},
      "implementation": {"avg_latency_ms": 0.56, "peak_memory_mb": 32.0},
      "speedup_vs_torch": 2.19,
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

**字段说明**：

| 指标 | 说明 |
|------|------|
| `avg_latency_ms` | 各 shape 延时的算术平均（兼容语义）|
| `peak_memory_mb` | 峰值内存占用（MB）|
| `speedup_vs_torch` | **几何平均加速比** = `(∏ s_i)^(1/n)`，仅对 status==pass 且 `s_i` 为有限正数的 shape 取几何平均；全部异常时为 `null` |
| `passed_cases` / `failed_cases` | 多 shape 通过 / 失败计数（异常 shape 仍计入 `passed_cases`，因为算子功能正常）|
| `nan_indices` / `inf_indices` / `zero_indices` / `negative_indices` / `none_indices` | 各类异常 `s_i` 的 case_idx 列表（从 1 开始），不进入几何平均；无异常时为 `[]` |
| `per_shape_results[].status` | `"pass"` 或 `"fail"` |
| `per_shape_results[].speedup_vs_torch` | 该 shape 的加速比；fail 或异常时为 `null` |

**边界值处理**：

`s_i = framework_latency_ms / impl_latency_ms` 可能因 profiler 故障、极小延时等出现异常值。`compute_overall` 对每个 `s_i` 按以下优先级分类：

| 类别 | 判定 | 落盘行为 |
|------|------|---------|
| `none` | `s_i is None` | `per_shape.speedup_vs_torch = null`，case_idx 入 `none_indices` |
| `nan` | `math.isnan(s_i)` | 同上，入 `nan_indices` |
| `inf` | `math.isinf(s_i)` | 同上，入 `inf_indices` |
| `negative` | `s_i < 0` | 同上，入 `negative_indices` |
| `zero` | `s_i == 0` | 同上，入 `zero_indices` |
| `valid` | 有限正数 | 进入几何平均 |

异常 shape **仍计入 `passed_cases`**（算子功能正常，仅测量数据不可信），但 `s_i` 不参与整体几何平均。全部 shape 都异常时 `speedup_vs_torch = null`。

**退出码**：
- exit 0：benchmark 正常完成（按 shape 内部 try/except，pass/fail 写在 per_shape_results）
- exit 1：脚本本身崩溃
- exit 2：L1 verify 闸门拒绝（precondition 未满足，benchmark 未实际运行）

调用方通过读 JSON 判断 `passed_cases == total_cases`；exit 2 时无 JSON 产出，应等价于"对应 verify 失败"处理。

**返回**：
- `perf_result`：dict（完整性能数据）
- `perf_report_path`：str（性能报告文件路径）

---

## 脚本位置

验证脚本位于本 skill 的 `scripts/` 目录：

| 脚本 | 用途 |
|------|------|
| `scripts/validate_triton_impl.py` | 退化预检查（AST 静态分析） |
| `scripts/verify.py` | 验证正确性 |
| `scripts/benchmark.py` | 测试性能 |

**CLI 参数**：
- `validate_triton_impl.py`: `<file_path>`, `[--json]`
- `verify.py`: `--op_name`, `--verify_dir`, `--triton_impl_name`, `--timeout`, `--output`, `--non-compute`
- `benchmark.py`: `--op_name`, `--verify_dir`, `--triton_impl_name`, `--warmup`, `--repeats`, `--output`, `--skip_framework`, `--framework_latency_ms`, `--verify_not_required`