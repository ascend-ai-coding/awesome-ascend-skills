# torch_npu.profiler 采集与 op_statistic 指标

与算子 `test/` 下基准脚本配套使用。本技能 **`ascendc-operator-performance-eval`** 要求 **`warmup=5`、`active=5` 固定**，不得通过 CLI 修改。

---

## 1. 必须使用的能力

- **`torch_npu.profiler.profile`**：`activities` 含 CPU、NPU（按项目惯例）。
- **`torch_npu.profiler.tensorboard_trace_handler(dir)`**：每次 `with` 在 `dir` 下写出导出树。
- **`torch_npu.profiler.schedule`**：`wait` / `warmup` / `active` / `repeat` / `skip_first` 与业务约定一致；**本技能强制 `warmup=5`、`active=5`**。
- **`prof.step()`**：每个循环迭代末尾调用一次。
- **循环次数**：`repeat * (wait + warmup + active)`，与 schedule 一致。

可选：`torch_npu.profiler._ExperimentalConfig` 与 `ProfilerLevel`（与仓库内已有脚本对齐）。

---

## 2. 导出目录命名

- 在 `tensorboard_trace_handler` 指定目录下会出现 **以 `_ascend_pt` 为后缀** 的子目录（前缀由工具链生成，文档与技能中 **只描述后缀**）。
- 性能表相对路径：**`*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv`**。

---

## 3. 指标计算

1. 对单次 `with profile` 生成的 `op_statistic.csv`，对 **每一行** 的 **Total Time(us)** **求和**（所有 OP 类型行计入，除非脚本明确排除）。
2. **归一化**：  
   `per_step_metric = sum_total_us / divisor`  
   推荐 **`divisor = active × repeat`**；本技能 **`active` 固定为 5**，通常 **`repeat = 1`**，故 **`divisor = 5`**。
3. 表头可能带 BOM 或列名微调；解析时应兼容 **`Total Time(us)`** 或等价列名。

---

## 4. 目录约定（避免混读）

- **每个用例、每种实现**（如 `custom` / `native`）使用 **一次独立** `with profile`。
- Handler 根路径建议：`{trace_root}/{op_trace_tag}/{custom|native}/case_XXX/`，每次运行前清空该 `case_XXX`，避免读到旧 `*_ascend_pt`。
- **`repeat > 1`** 可能产生多份导出；若按「最新修改时间」选取 CSV，须在报告或注释中说明语义；简单场景固定 **`repeat = 1`**。

---

## 5. Markdown 报告（与技能主文档一致）

- 按 **输入主张量 dtype** 分节（如 `float16`、`bfloat16`）。
- 对比报告：每节表含 Case、Shape、工程 per-step、原生 per-step、比值；节尾 dtype 小结；文末全量汇总 + **按数据类型汇总** 表。
- 报告正文注明 **`warmup=5`、`active=5` 为技能固定约定**。
- **不生成** `*_torch_npu_profiler_results.json`；仅以 Markdown 为对外结果文件。
