# `msprof application` Runner 契约

`collect_perf_application.py` 负责选择 warmup、启动一次 profiler、解析 `op_summary.csv` 和取中位数。工程内的持久 runner 负责构造输入并按固定顺序发射 kernel。runner 不得依赖外部评测框架。

内容：命令行接口、选择优先级、两种 warmup、execution manifest schema v5、最小循环骨架、CANN 版本与会话上限。

采集命令另需 `--identity-manifest`，由工作流在采集前写入目标源码/二进制 SHA、Device、CANN 和频率状态。`--ops-warmup-probe-cmd` 先验证 Max/ArgMax 可运行；只有探测失败才执行 `--warmup-build-cmd`。`--compiled-warmup-binary` 告知采集器此时应产出的 `_C.so`，构建成功后由采集器独立哈希并写入 warmup 身份。

## CANN 版本差异与不完整安装（实测：Ascend950PR，CANN 9.0.0 build 20260422 / 不完整 9.1.0 build 20260610）

采集器已内置以下适配，runner 与调用方无需处理，但必须知晓：

1. **`--aic-mode=task-based` 是版本相关参数**：CANN 9.0.0 只有显式传它才导出 `op_summary`；9.1.0 传不传都导出（带参实测无害）。采集器固定传该参数并在 `collection_manifest.json` 记录 `aic_mode` 与 `aic_mode_rationale`。
2. **不完整安装预检（ops 包缺失）**：采集前 `inspect_cann_environment()` 盘点实际解析的 `$ASCEND_HOME_PATH`、version.info 版本/时间戳、opapi 库族（`libopapi*.so`）与 `aclnnArgMax/aclnnMax` host 符号存在性，结果写入 manifest 的 `cann_environment`。不完整安装（如裁剪版 9.1.0 只有 `libopapi_oam.so`）没有系统 warmup 算子符号，framework_ops warmup 必然失败；此时唯一回退是 compiled warmup，若 torch_npu 与该 CANN 不匹配（实测 9.0.0 编译的 torch_npu 在 9.1.0 下 `std::bad_alloc`），采集以明确原因失败而非 561103 之类的隐晦错误。**不要**用链接另一版本 opapi 库的方式混用（实测 9.0.0 libopapi + 9.1.0 runtime → 调用期 ACLNN_ERR_INNER_NULLPTR 561103）。
3. **单会话 relevant launch 上限（默认 540，`--max-launches-per-session` 可调）**：CANN 9.0.0 的分析管线在单会话 relevant op 记录数过大时**静默导出空 `op_summary`，msprof 仍 exit 0**。实测（同一命令重复执行）：540 条从未失败（冻结协议规模，10+ 次运行）；720 条 2 成 1 败（aging/unaging 导出路径抖动）；840 条全败。默认取全量验证过的 540，并对 op_summary 为空的会话**自动重试一次**（重试前清除失败 PROF 数据，`empty_summary_retries` 记入 manifest）。采集器按 warmup/target kernel 数估算每 case launches，自动把大 case 集拆成多个连续会话（`session-001/...`，各自独立 manifest 与 msprof 日志），合并解析并整体校验 launch 拓扑；单会话可容纳时保持原有根目录布局。**msprof 的 `--application` 参数还有 2048 字符硬限制**，规划时同步保证渲染后命令长度低于上限。
4. **exit 0 ≠ 数据有效**：`op_summary` 缺失或 0 条记录时 msprof 照常退出并打印 "Data is saved"。采集器在解析阶段以 `expected == parsed == mapped` 校验兜底，拓扑为空时报错并提示会话上限/不完整安装两个可能原因。

## 现成 runner 模板（assets/app_runner/）

按目标工程的调用层选模板拷入工程，只改标记的 ADAPTER 段；两者命令行与 manifest 完全符合本契约：

| 条件 | 模板 | 说明 |
|------|------|------|
| 有 torch/框架接口 | `assets/app_runner/torch/torch_app_runner.py` | 只改 TARGET ADAPTER 段（算子名、kernel 序列、输入构造、调用）；示例为 CANN 内置 Add，任何 torch_npu 环境可独立跑通 |
| 只有 aclnn（算子已注册并编出 vendor 包） | `assets/app_runner/aclnn/`（C++，`bash build.sh [--vendor-dir <vendor>]`） | 只改 `csrc/target_adapter_<op>.cpp` 四个符号；warmup 用 aclnnArgMax/aclnnMax，无 torch 依赖。随附示例为 CANN 内置 Add（`bash build.sh` 即可构建验证）；适配自定义算子时传 `--vendor-dir` 链接其 `libcust_opapi.so` |

aclnn 模板适配自定义算子的强制要求（实测踩坑，缺一即失败）：

1. **运行前必须 source vendor 包的 `bin/set_env.bash`**（把 vendor `op_api/lib` 加入 `LD_LIBRARY_PATH` 并把 `ASCEND_CUSTOM_OPP_PATH` 指向 vendor 目录本身）；只设 opp 根目录时 `GetWorkspaceSize` 返回 161001 PARAM_NULLPTR。
2. **输出 buffer 必须零初始化**：torch 路径用 `at::zeros` 分配输出，kernel 可能依赖零初值；裸 `aclrtMalloc` 脏内存会挂死。在 `target_make_inputs` 里用 `aclrtMemset` 补零（host 侧操作，不产生 kernel、不污染 target 序列）。
3. **口径差异**：aclnn 模板的 target 序列不含 torch 路径的输出清零 kernel（如 `aclnnInplaceZero_*`）；同一对比组内 baseline/candidate 必须用同一 runner，跨 runner 数据不可比。

## probe 声明协议（可选）

warmup 走非 torch 栈的 runner（如 aclnn 模板）在 probe 成功时向 stdout 打印一行：

```
OPS_WARMUP_PROBE_RESULT: {"logical_op": "ArgMax", "implementation": "aclnn_argmax_256mib_fp16_v1"}
```

采集器解析该行并采纳 `implementation` 作为 framework_ops warmup 身份；torch runner 不打印时默认 `torch_{op}_256mib_fp16_v1`。manifest 的 warmup identity 必须与采集器选择一致（照旧校验）。

## 命令行接口

runner 至少接受：

```text
--case-ids <逗号分隔 ID>
--samples 10
--execution-manifest <输出 JSON>
--warmup-mode framework_ops|compiled
--warmup-op ArgMax|Max|OpsProfilingWarmup
```

`--samples` 不是调参入口，值不等于 `10` 时 runner 必须失败。

## 选择优先级

1. 普通目标算子优先探测并使用 `ArgMax`。
2. 目标属于 ArgMax/ArgMaxV2 家族时，优先探测并使用 `Max`，避免 warmup 与 target 同名。
3. 只有 `--ops-warmup-probe-cmd` 非零、超时或无法启动时，才执行内置包的构建命令并选择 `OpsProfilingWarmup`。
4. ops 探测与内置包构建都失败时，整次采集失败。

探测命令必须包含 `{warmup_op}`，例如：

```bash
python3 ./run_cases.py --probe-warmup-op {warmup_op}
```

runner 的 probe 入口必须在目标 Device 上分配精确 256 MiB FP16 tensor，实际执行所选 Max/ArgMax 并 synchronize；仅 import 成功不能视为 ops 环境可用。

## 执行顺序

1. 启动进程、import 本地目标算子包并设置 Device。
2. compiled 模式从工程内 asset import `ops_profiling_warmup`；禁止运行时生成或下载 warmup。framework_ops 模式直接使用 probe 已验证的 Max/ArgMax。
3. 对每个 case 构造输入；构造阶段不得调用与目标或 warmup 同名的算子。
4. 两种模式都只分配一次精确256 MiB的 FP16 NPU tensor：`torch.empty((128, 1024, 1024), dtype=torch.float16, device=...)`。
5. 每个 case 连续执行10组：`warmup → synchronize → target → synchronize`。
6. 全部执行完再原子写 `execution-manifest.json`。

不得把 warmup 另跑成一段，也不得执行“1次 warmup + 10次 target”。每一个 target 前必须紧邻一次 warmup。

## 默认 Max/ArgMax 实现

```python
OPS_WARMUP_SHAPE = (128, 1024, 1024)
OPS_WARMUP_BYTES = 256 * 1024 * 1024


def make_ops_warmup_tensor(torch, device):
    value = torch.empty(OPS_WARMUP_SHAPE, dtype=torch.float16, device=device)
    assert value.numel() * value.element_size() == OPS_WARMUP_BYTES
    return value


def run_ops_warmup(torch, value, warmup_op):
    if warmup_op == "ArgMax":
        return torch.argmax(value)
    if warmup_op == "Max":
        return torch.max(value)
    raise ValueError(f"unexpected ops warmup: {warmup_op}")
```

execution manifest 的 `implementation` 确定性填写：`ArgMax` 对应 `torch_argmax_256mib_fp16_v1`，`Max` 对应 `torch_max_256mib_fp16_v1`。

## 无 ops 环境时的 compiled 实现

先按 [`compiled_warmup.md`](compiled_warmup.md) 将 skill asset 落地；probe 失败后才构建。runner 只使用其稳定接口：

```python
from ops_profiling_warmup import compiled_warmup, make_warmup_tensor

warmup_tensor = make_warmup_tensor(device)
compiled_warmup(warmup_tensor)
```

该调用对固定 tensor 做一次全量 `x += 1`，对应 kernel `OpsProfilingWarmup` 和实现身份 `ops_profiling_warmup_v1`。只有 ops probe 失败时才构建。构建失败，或构建成功后的 import、校验、运行失败，都会让采集失败；不再切回 Max/ArgMax。

## execution manifest schema v5

```json
{
  "schema_version": 5,
  "samples_per_case": 10,
  "target_operator": "Exp",
  "warmup": {
    "mode": "framework_ops",
    "logical_op": "ArgMax",
    "implementation": "torch_argmax_256mib_fp16_v1",
    "tensor_bytes": 268435456,
    "tensor_allocation": {
      "shape": [128, 1024, 1024],
      "dtype": "float16",
      "elements": 134217728
    }
  },
  "case_runs": [
    {
      "case_id": "1",
      "warmup_kernel_sequence": ["ArgMax"],
      "target_kernel_sequence": ["cast", "exp_main"],
      "iteration_kernel_sequence": ["ArgMax", "cast", "exp_main"]
    }
  ]
}
```

约束：

- `warmup` 的 mode、logical_op、implementation、tensor_bytes 和 tensor_allocation 必须与采集器选择完全一致。
- `case_runs` 顺序必须与 `cases.csv`/`--case-ids` 一致。
- 三个 kernel sequence 填 profiler 中的精确 `Op Name`，不是 Python API 名。
- `iteration_kernel_sequence` 必须严格等于 warmup sequence 与 target sequence 的拼接，且两者 kernel 名集合不相交。
- 同一 case 的10组 sequence 必须完全相同；动态分支产生不同拓扑时拆分 case。
- 输入构造和清理只能产生不同名 kernel；相关 launch 多一个、少一个或乱序都会拒绝数据。

## 最小循环骨架

```python
warmup_tensor = (
    make_warmup_tensor(device)
    if args.warmup_mode == "compiled"
    else make_ops_warmup_tensor(torch, device)
)

for case_id in selected_case_ids:
    target_args = make_case_inputs(cases[case_id])
    torch.npu.synchronize(device)
    for _ in range(10):
        if args.warmup_mode == "compiled":
            compiled_warmup(warmup_tensor)
        else:
            run_ops_warmup(torch, warmup_tensor, args.warmup_op)
        torch.npu.synchronize(device)
        local_extension.target_op(*target_args)
        torch.npu.synchronize(device)
```

runner 不做 Python 计时。采集器从 `op_summary.csv` 按 task start time 重建10组映射；每组只加总 target sequence，最后用 `statistics.median` 聚合。
