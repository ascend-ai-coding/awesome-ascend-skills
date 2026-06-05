# Layer Norm Profiler 性能基准说明

采集必须使用 **`torch_npu.profiler`**（`profile`、`schedule`、`tensorboard_trace_handler`、`prof.step()`，且单次 `with` 内迭代步数与 schedule 一致）。

通用流程与目录约定见技能 **ascendc-operator-performance-eval**（`.cursor/skills/ascendc-operator-performance-eval/SKILL.md`）。

## 本算子约定

| 项 | 说明 |
|----|------|
| 工程路径 | `torch.ops.npu.layer_norm`（需 `import ascend_kernel` 或 `load_library`） |
| 原生路径 | `torch.nn.functional.layer_norm`（与输入同 NPU device） |
| 用例文件 | 仅 **`layer_norm_perf_cases.jsonl`** |
| Schedule | **`warmup=5`、`active=5` 固定**（由 `layer_norm_profiler_common.PROFILER_SCHEDULE_*` 定义） |
| 输出 | **`layer_norm_torch_npu_profiler_report.md`**；不生成 `*_profiler_results.json` |
| Trace 根 | 默认 `test/profiler_trace/layer_norm_torch_npu/{custom,native}/case_XXX/` |

## 运行示例

```bash
conda activate atk_py39
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/path/to/ascend-kernel/python:${PYTHONPATH}"
cd /path/to/ascend-kernel/csrc/ops/layer_norm/test
python benchmark_layer_norm_torch_npu_profiler.py
```

可选参数：`--case-file`、`--trace-root`、`--device`、`--wait`、`--repeat`、`--divisor-mode`、`--only-case`、`--report-md`。**不可**修改 warmup/active（已自脚本内固定为 5/5）。
