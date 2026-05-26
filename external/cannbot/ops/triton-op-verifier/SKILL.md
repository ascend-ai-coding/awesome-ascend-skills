---
name: external-cannbot-ops-triton-op-verifier
description: '算子代码验证 Skill — 按照标准验证流程验证生成的内核代码。 创建验证项目文件，调用 scripts/verify.py 运行验证，验证通过后
  调用 scripts/benchmark.py 进行性能测试并收集结果。 触发：当用户需要验证 Triton 算子代码功能正确性或采集其性能数据时使用。

  '
argument-hint: 输入：generated-code-path、task-file-path、op-name、warmup、repeats。 输出：验证结果（成功/失败）、错误信息、性能数据。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
original-name: triton-op-verifier
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# Kernel Verifier Skill

<role>
你是一个内核代码验证专家。你的任务是按照标准验证流程，创建验证项目并运行，检查生成的算子代码是否能正确编译运行且与参考实现的输出一致。验证通过后，执行性能测试并收集性能数据。
</role>

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

**超时设置**：默认 900 秒，复杂算子可适当增加。

**⛔ 禁止事项**：
- 禁止自己编写 Python 代码来测试算子（如手动 import 并 forward 比较）
- 禁止使用 `torch.allclose` 或其他自创方法替代 `scripts/verify.py`
- 禁止跳过此步骤直接报告验证结果

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

**多 shape 行为**：每个 shape 独立 try/except，失败不中止后续 shape；全部跑完才落盘并退出。

**退出码语义（策略 A：严格）**：
- `passed_cases == total_cases` → exit 0，`verifier_result = true`
- `passed_cases <  total_cases` → exit 1，`verifier_result = false`，`verifier_error` 应读取 `verify_result.json.failures` 的**全部条目**（不是第一个），汇总后提交给 Conductor。

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

## 精度阈值说明

验证使用基于数据类型的 **MERE/MARE 双门限相对误差**判定（NPU Benchmark 标准），与 `torch.allclose` 不同。

**判定公式**（必须同时满足）：

```
MERE < threshold        且        MARE < 10 × threshold
```

其中：
- `MERE` = mean(|actual - golden| / max(|golden|, threshold))，平均相对误差
- `MARE` = max(|actual - golden| / max(|golden|, threshold))，最大相对误差
- 计算前两侧统一升 float32，避免低精度 dtype 自身误差污染
- 分母用 `clamp(min=threshold)` 而非 `+epsilon`：当 `|golden| < threshold`（参考值已小到 dtype 精度极限）时，rel_err 退化为 `|diff| / threshold`，等价于按绝对误差归一化，避免零值/极小值附近误报

**dtype 阈值表**（2 的幂次方）：

| 数据类型 | threshold | MERE 上限 | MARE 上限 (10×t) |
|---------|-----------|-----------|------------------|
| `float16` | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| `bfloat16` | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| `float32` | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| `hifloat32` | 2⁻¹¹ ≈ 4.88e-4 | 4.88e-4 | 4.88e-3 |
| `float8_e4m3` | 2⁻³ = 0.125 | 0.125 | 1.25 |
| `float8_e5m2` | 2⁻² = 0.25 | 0.25 | 2.5 |
| 其他 dtype（fallback） | 2⁻¹³ | 1.22e-4 | 1.22e-3 |

**比对前置检查**（按顺序，任一失败即判 fail）：
1. 形状必须一致
2. NaN 位置必须完全一致（mask 按位相等）
3. Inf 位置和符号必须完全一致
4. `bool` dtype：要求 `torch.equal` 完全相等，不进入 MERE/MARE 判定
5. 仅在 `finite_mask` 上做 MERE/MARE 计算；当 dtype 不一致时 impl 会被 cast 到 golden 的 dtype

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
- `verify.py`: `--op_name`, `--verify_dir`, `--triton_impl_name`, `--timeout`, `--output`
- `benchmark.py`: `--op_name`, `--verify_dir`, `--triton_impl_name`, `--warmup`, `--repeats`, `--output`, `--skip_framework`, `--framework_latency_ms`, `--verify_not_required`