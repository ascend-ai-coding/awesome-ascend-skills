# NPU 性能 Profiling（验证通过后必须执行）

## 说明

**⛔ 强制步骤**：验证通过后，**必须**使用通用脚本 `scripts/profile_golden.py` 调用 `{op}_golden.py`，并通过 `torch_npu.profiler` 采集 golden 算子在 NPU 上的性能数据。golden 文件本身不包含 profiling 代码。

**核心目标**：对所有算子的 golden 都必须成功采集到正确的性能数据。无论算子输入多复杂，都要找到一种方式让 profiling 成功运行。

## 输入模式选择（决策树）

`profile_golden.py` 支持两种输入模式。**必须在执行前根据算子特征选择正确的模式**，避免盲目尝试后崩溃：

```
算子的所有 tensor 输入是否都接受任意随机值？
│
├─ 是 → 模式 A：--input / --arg（随机值模式）
│        适用：逐元素算子、标准 matmul、softmax 等
│
└─ 否 → 是否有部分 tensor 有值域约束但可单独控制？
         │
         ├─ 是，且约束简单（≤3 个 tensor 有约束）
         │    → 模式 A + 对受限 tensor 使用 --arg 传入合法值
         │      或 模式 A + 在 _make_inputs() 中对受限 tensor
         │      使用 abs/randint 等保证值域
         │
         └─ 否（tensor 间有结构依赖、状态缓存、间接索引等）
              → 模式 B：--factory（工厂函数模式）
                适用：有状态算子、paged attention 类、
                压缩器、带 block table 的算子等
```

## 模式 A：`--input` / `--arg`（随机值模式）

**适用条件**：所有 tensor 接受任意随机值，或仅少量 tensor 有简单值域约束。

```bash
python3 skills/pypto-golden-generate/scripts/profile_golden.py \
  custom/{op}/{op}_golden.py \
  --function {op}_golden \
  --input <tensor_param_name>:<SHAPE>:<DTYPE> \
  --input <tensor_param_name2>:<SHAPE>:<DTYPE> \
  --arg <scalar_param_name>=<VALUE> \
  --device <DEVICE_ID>
```

参数说明：

- `--function`：golden 函数名；缺省时脚本会优先使用文件名同名函数，或自动寻找唯一的 `*_golden` 函数
- `--input NAME:SHAPE[:DTYPE]`：tensor 输入规格，可重复；**多输入算子必须为每个 tensor 参数传入一条**。脚本使用 `torch.randn`（浮点）或 `torch.randint(-3, 4, ...)`（整数）生成值
- `--arg NAME=JSON`：非 tensor 参数，可重复，如 `--arg eps=1e-5`、`--arg d=512`
- `--device DEVICE_ID`：NPU 卡号（整数），默认为 0
- `--iters`：迭代次数，默认 1

**构造输入参数步骤**：

1. **读取 golden 函数签名**：确定所有 tensor 参数名、scalar 参数名及其默认值
2. **读取 SPEC.md `p0_shapes`**：获取 P0 典型 shape（按参数顺序对应 tensor 输入）
3. **读取 SPEC.md `default_params`**：获取 scalar 参数的值
4. **确定 dtype**：从 SPEC.md `supported_dtypes` 和 golden 函数注释中推断每个 tensor 的 dtype

**示例**（多输入算子 `ds_v4_hc_pre`）：

```bash
# golden 签名: ds_v4_hc_pre_golden(x, hc_fn, hc_scale, hc_base, hc=4, d=4096, ...)
# SPEC p0_shapes: [[1024, 16384]]  (只有 x 的 shape，其余是 weights)

python3 skills/pypto-golden-generate/scripts/profile_golden.py \
  custom/pto_case/ds_v4_hc_pre/ds_v4_hc_pre_golden.py \
  --function ds_v4_hc_pre_golden \
  --input x:1024x16384:bfloat16 \
  --input hc_fn:24x16384:float32 \
  --input hc_scale:3:float32 \
  --input hc_base:1x24:float32 \
  --arg hc=4 \
  --arg d=4096 \
  --device 15
```

## 模式 B：`--factory`（工厂函数模式）

**适用条件**：算子输入有语义约束，随机值会导致 golden 函数崩溃。典型场景：

| 约束类型 | 示例 | 崩溃原因 |
|----------|------|----------|
| **间接索引** | `block_table[b, idx]` → 用作 state 的行索引 | 随机索引越界或指向未初始化区域 |
| **状态依赖** | `kv_state[block_idx, pos, :]` 需要先写入再读取 | 随机 state 导致后续计算异常 |
| **整除/模约束** | `(start_pos + i + 1) % ratio == 0` 触发压缩 | 随机 start_pos 可能永远不触发 |
| **位置编码** | `ape[pos]` 中 pos 由 `global_pos % ratio` 计算 | 随机 pos 可能越界 |
| **序列长度** | `seqused_kv` 必须 ≤ 实际序列维度 | 随机长度导致 gather 越界 |
| **多 tensor 关联** | `x.shape[1]` 必须与 `state.shape[1]` 匹配 | 随机 shape 之间不一致 |

**执行方式**：

```bash
python3 skills/pypto-golden-generate/scripts/profile_golden.py \
  custom/{op}/{op}_golden.py \
  --function {op}_golden \
  --factory _make_inputs \
  --device <DEVICE_ID>
```

`--factory` 指定 golden 模块中的一个函数名（通常为 `_make_inputs`），该函数接受 `device` 参数。脚本调用工厂函数获取合法输入，**忽略所有 `--input` 和 `--arg` 参数**。

**多 P0 shape 支持**：`_make_inputs()` 可以返回多组 case，脚本会自动对每组分别 profiling 并在报告中生成对比表格：

```python
# 单 case（向后兼容）
def _make_inputs(device):
    return [x], {}

# 多 case（每个性能 P0 shape 一组）
def _make_inputs(device):
    cases = []
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device=device)
    cases.append(("perf_p0_small", [x], {}))
    x = torch.randn(16, 2048, dtype=torch.bfloat16, device=device)
    cases.append(("perf_p0_large", [x], {}))
    return cases
```

脚本自动检测返回格式：`list[(str, list, dict)]` 视为多 case，`(list, dict)` 视为单 case。

**示例**（有状态算子 `ds_v4_compressor`）：

```bash
# golden 签名: ds_v4_compressor_golden(x, sin, cos, wkv, wgate, ape, weight,
#   kv_state, score_state, kv_block_table, score_block_table, hadamard,
#   start_pos_dy, d=32, rope_head_dim=16, ratio=8, rotate=False, eps=1e-6)
#
# 约束: block_table 需要合法索引、start_pos_dy 需要 ratio-2、
#       state 需要 zeros 初始化、shape 间有依赖关系

python3 skills/pypto-golden-generate/scripts/profile_golden.py \
  custom/pto_case/ds_v4_compressor/ds_v4_compressor_golden.py \
  --function ds_v4_compressor_golden \
  --factory _make_inputs \
  --device 0
```

**`_make_inputs()` 函数要求**（详见 SKILL.md §4）：

```python
def _make_inputs(device):
    """返回 (args_list, kwargs_dict) 或 [(case_name, args_list, kwargs_dict), ...]。"""
    # ... 构造所有 tensor，确保语义合法 ...
    args = [x, sin, cos, wkv, wgate, ape, weight,
            kv_state, score_state, block_table, block_table,
            hadamard, start_pos_dy]
    kwargs = {"d": 32, "rope_head_dim": 16, "ratio": 8,
              "rotate": False, "eps": 1e-6}
    return args, kwargs
```

## Profiling 流程

1. **由 `profile_golden.py` 预分配输入 tensor**（避免 `randn` 开销混入 kernel 计时）
   - 模式 A：根据 `--input` 规格用 `torch.randn` / `torch.randint` 生成
   - 模式 B：调用 `_make_inputs(device)` 获取合法输入
2. **噪声注入 profiling 循环**：每次迭代先执行 `torch.randn(480MB).to(float32).npu()` + `torch.max(a)` 制造迭代边界标记，再调用 golden 函数，最后 `torch_npu.npu.synchronize()` + `prof.step()`
3. 用 `torch_npu.profiler` 记录 NPU + CPU 事件；Profiler 配置 `ProfilerLevel.Level1` + `AiCMetrics.PipeUtilization`
4. CANN profiler 生成 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`：每个设备侧 kernel 的 Name、Type、Duration(us) 等
5. **E2E 双路径提取**：
   - **Path A（主路径，减法）**：E2E = 所有 kernel Duration 之和 − 噪声 kernel（ReduceMax）Duration 之和
   - **Path B（校验路径，加法）**：E2E = Σ 每个非噪声 op type 的 total Duration
   - 两条路径结果必须一致，不一致时输出 warning
6. **Per-op 统计**：按 `Type` 列分组，计算每个 op type 的 `count`（调用次数）、`mean_duration`（单次均值）、`total`（总耗时）
7. 写入 GOLDEN_PERF_REPORT.md

## 产物

| 产物 | 位置 | 说明 |
|------|------|------|
| `GOLDEN_PERF_REPORT.md` | `custom/{op}/` | 可直接阅读的性能报告（E2E 双路径、op 级 count + mean_duration + total） |
| `kernel_details.csv` | `custom/{op}/prof/{op}_golden/.../ASCEND_PROFILER_OUTPUT/` | 每个设备侧 kernel 的 Name、Type、Duration(us) 等 |

## GOLDEN_PERF_REPORT.md 示例

### 单 case（向后兼容）

```markdown
# ds_v4_hc_pre Golden NPU Performance Report

- **Device**: npu:15
- **Timestamp**: 2026-06-08T07:35:02.270396+00:00
- **Iterations**: 1
- **Input Shape**: {x: (1024, 16384), hc_fn: (24, 16384), hc_scale: (3,), hc_base: (1, 24)}
- **dtype**: {x: torch.bfloat16, hc_fn: torch.float32, hc_scale: torch.float32, hc_base: torch.float32}
- **Profiling Data**: `prof/ds_v4_hc_pre_golden/`

### E2E Performance

**Total kernel duration**: 1186.5us (total - noise)
- Cross-check (Σ per-op total): 1186.5us

### Op Performance

| op | count | mean_duration | total |
|----|-------|--------------|-------|
| ReduceSum | 41 | 11.0us | 451.3us |
| RealDiv | 41 | 7.82us | 320.5us |
| MatMulV2 | 1 | 111.7us | 111.7us |
| Mul | 6 | 17.8us | 106.5us |
| ...

## Notes
- Data source: `ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- Noise ops (ReduceMax) filtered out
- Each iteration: noise injection (480MB randn + max) → golden call → sync
```

### 多 case（多 P0 shape）

```markdown
# attention Golden NPU Performance Report

- **Device**: npu:0
- **Timestamp**: 2026-06-09T10:00:00.000000+00:00
- **Iterations**: 1

## Performance Summary

| case | Input Shape | dtype | E2E (us) |
|------|------------|-------|----------|
| perf_p0_small | {q: (2, 8, 512, 64), k: (2, 8, 512, 64), v: (2, 8, 512, 64)} | {q: torch.bfloat16, k: torch.bfloat16, v: torch.bfloat16} | 245.3us |
| perf_p0_large | {q: (4, 16, 1024, 128), k: (4, 16, 1024, 128), v: (4, 16, 1024, 128)} | {q: torch.bfloat16, k: torch.bfloat16, v: torch.bfloat16} | 1832.7us |

## Case: perf_p0_small

- **Input Shape**: {q: (2, 8, 512, 64), k: (2, 8, 512, 64), v: (2, 8, 512, 64)}
- **dtype**: {q: torch.bfloat16, k: torch.bfloat16, v: torch.bfloat16}
- **Profiling Data**: `prof/attention_golden/perf_p0_small/`

### E2E Performance

**Total kernel duration**: 245.3us (total - noise)
- Cross-check (Σ per-op total): 245.3us

### Op Performance

| op | count | mean_duration | total |
|----|-------|--------------|-------|
| MatMulV2 | 2 | 45.2us | 90.4us |
| SoftmaxV2 | 1 | 32.1us | 32.1us |
| ...

## Case: perf_p0_large

- **Input Shape**: {q: (4, 16, 1024, 128), k: (4, 16, 1024, 128), v: (4, 16, 1024, 128)}
- **dtype**: {q: torch.bfloat16, k: torch.bfloat16, v: torch.bfloat16}
- **Profiling Data**: `prof/attention_golden/perf_p0_large/`

### E2E Performance

**Total kernel duration**: 1832.7us (total - noise)
- Cross-check (Σ per-op total): 1832.7us

### Op Performance

| op | count | mean_duration | total |
|----|-------|--------------|-------|
| MatMulV2 | 2 | 380.5us | 761.0us |
| SoftmaxV2 | 1 | 215.3us | 215.3us |
| ...

## Notes
- Data source: `ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- Noise ops (ReduceMax) filtered out
- Each iteration: noise injection (480MB randn + max) → golden call → sync
```

## 故障排查

**⛔ 核心原则：profiling 必须成功。遇到崩溃时按以下流程排查，不得跳过或留空报告。**

### Golden 函数崩溃（exit code 2）

**症状**：`Golden function crashed on iteration 0. Random tensor values may violate semantic constraints...`

**排查流程**：

```
崩溃 → 读取错误信息中的异常类型
│
├─ IndexError / out of range
│    → 某个 tensor 用作索引，随机值越界
│    → 修复：该 tensor 必须用 randint(low=0, high=max_valid) 或手动构造
│    → 推荐：切换到 --factory 模式
│
├─ RuntimeError (shape mismatch / size mismatch)
│    → 多个 tensor 的 shape 之间有依赖关系，随机 shape 不匹配
│    → 修复：在 _make_inputs() 中从基础参数推导所有关联 shape
│    → 推荐：切换到 --factory 模式
│
├─ RuntimeError (CUDA/NPU error)
│    → 可能是 dtype 不匹配或 device 不一致
│    → 修复：确保所有 tensor 使用相同 device，dtype 符合 golden 函数预期
│
├─ ValueError / AssertionError（golden 内部校验失败）
│    → 输入值不满足算子的前置条件
│    → 修复：阅读 golden 函数的前置检查逻辑，构造满足条件的输入
│
└─ 其他异常
     → 阅读 golden 函数源码，理解输入约束
     → 在 _make_inputs() 中构造合法输入
     → 切换到 --factory 模式
```

### kernel_details.csv 未生成

**症状**：`kernel_details.csv not found — report will have no data`

**排查流程**：

```
无 kernel_details.csv
│
├─ prof/ 目录下有多个 ASCEND_PROFILER_OUTPUT/
│    → _find_ascend_output 已自动选择最新的目录
│    → 检查最新目录中是否有 kernel_details.csv
│
├─ ASCEND_PROFILER_OUTPUT/ 中只有 operator_details.csv
│    → CANN profiler 未生成设备侧 kernel 数据
│    → 可能原因：golden 函数在 CPU 上执行（无 NPU 硬件）
│    → 检查：torch.npu.is_available() 和 device_count()
│
└─ prof/ 目录为空
     → profiler 未正常启动
     → 检查 torch_npu 版本和 CANN 版本兼容性
```

### E2E 为 0 或异常小

**症状**：报告生成但 E2E = 0.000us 或仅几个 us

**排查流程**：

```
E2E 异常
│
├─ E2E = 0.000us
│    → kernel_details.csv 中无有效数据
│    → 检查 CSV 的 Type 和 Duration(us) 列是否有值
│    → 可能所有 kernel 被归类为噪声（ReduceMax）
│
├─ E2E 异常小（远小于预期）
│    → golden 函数可能走了短路分支（如 should_compress 始终为 False）
│    → 修复：调整 _make_inputs() 中的参数，确保触发核心计算路径
│    → 例：start_pos_dy = [ratio - 2] 确保 (start_pos + i + 1) % ratio == 0
│
└─ E2E 双路径不匹配
     → 检查是否有非 ReduceMax 的噪声 kernel
     → 在 profile_golden.py 的 NOISE_OPS 中添加额外噪声 op type
```

## 注意事项

- 仅 NPU 环境下生效；无 NPU 硬件时 `profile_golden.py` 跳过采集
- golden 文件只负责 `{op}_golden()`、`_make_inputs()` 和 `_validate()`，不包含 profiling 逻辑
- **模式 A 输入必须显式指定**：不提供 `--input` 时脚本会报错 `missing required arguments`，不会自动生成默认 tensor
- **模式 B 优先**：当算子有任何语义约束时，**始终优先使用 `--factory _make_inputs`**，不要尝试用 `--input` + `--arg` 拼凑
- **E2E 双路径校验**：Path A（减法：total − noise）为主路径，Path B（加法：Σ per-op total）为校验路径。两者数学等价，不一致时触发 warning
- **profiling 前 warmup**：脚本在正式 profiling 循环前会先执行一次 golden 调用（warmup），确保 JIT 编译等一次性开销不计入性能数据
- **`_make_inputs()` 与 `_validate()` 共用**：`_validate()` 内部也应调用 `_make_inputs()` 获取输入，确保验证和 profiling 使用完全一致的数据
