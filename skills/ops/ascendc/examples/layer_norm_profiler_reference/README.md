# Layer Norm profiler 参考目录

本目录为 **`ascend-kernel/csrc/ops/layer_norm/test/`** 中与 `torch_npu.profiler` 性能对比相关文件的镜像，供技能 **ascendc-operator-performance-eval**（`.cursor/skills/ascendc-operator-performance-eval/SKILL.md`）查阅与拷贝模板。

## 约定

- 用例仅 **`layer_norm_perf_cases.jsonl`**，不生成 `.json`。
- 基准脚本 **`warmup=5`、`active=5` 固定**，不写 `layer_norm_torch_npu_profiler_results.json`，只写 **`layer_norm_torch_npu_profiler_report.md`**。

## 同步方式

若仓内 `layer_norm/test/` 中上述脚本或用例有更新，请将变更复制回本目录，保持与技能示例一致。
