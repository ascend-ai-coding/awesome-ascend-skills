# IR 分析优化器

## 概述

IR（中间表示）分析优化器通过提取编译器的最终阶段 IR（`last_pass.mlir`），对 triton 脚本进行分析优化。

当 IR 分析给出优化建议时，智能体（Agent）必须首先检查对应的优化点是否已在之前的迭代中执行过：

- **已执行** → 重新根据IR分析结果执行优化。并诊断为何之前的优化未能产生预期的效果。
- **之前未命中** → 根据IR分析结果执行优化并分析之前未命中原因。
- **无对应的优化点** → 直接应用 IR 分析优化器的优化建议。

## IR 多轮迭代模式

在 Phase 4 调用方（triton-op-generator AGENTS.md）开启 IR 多轮迭代后，IR（优化点 25）允许多轮重复命中，其他优化点单轮即过。每轮必须遵守：

- **每轮重新提取 IR**：必须重新执行 `run_and_extract.sh` 提取基于"上一轮优化后代码"的最新 `last_pass.mlir`，禁止复用历史 IR 快照。
- **聚焦增量分析**：本轮 IR 分析应聚焦两件事 —— (a) 上一轮 IR 优化是否带来预期效果（profiling 或 speedup 对比）；(b) 是否还有**新的**优化建议。
- **返回字段**：latency-optimizer 在返回信息中给出 `ir_has_more_suggestions: bool`。仅当本轮还能给出新建议时为 `true`；若 IR 分析已无新建议（即上一轮已是相同结论、无新增优化空间），返回 `false`，Phase 4 调用方据此退出 IR 多轮迭代。
- **退出条件**：达到 `ir_max_iterations`（默认 20）或 `ir_has_more_suggestions == false` 任一成立即结束 IR 多轮迭代。

## IR 提取

使用 `<skill_dir>/scripts/run_and_extract.sh <script_name>.py` 来提取 IR 文件,并将提取到的IR文件保存到 `<workspace>/ir_output/<kernel_name>`

### 环境变量

| 变量 | 值 | 用途 |
|----------|-------|---------|
| `TRITON_DEBUG` | `1` | 启用 IR 转储目录 |
| `TRITON_ALWAYS_COMPILE` | `1` | 强制重新编译，即使存在缓存 |
| `TRITON_DISABLE_LINE_INFO` | `0` | 保留源代码行信息 |
| `TRITON_DISABLE_FFTS` | `1` | 禁用 FFTS（910_95 不支持此功能） |
| `ASCEND_RT_VISIBLE_DEVICES` | `${NPU_DEVICE:-2}` | 选择 NPU 设备 |

### 提取流程

1. 使用上述环境变量运行 Python 脚本，以触发编译并转储 IR。
2. 解析 stdout 中的 `Dumping intermediate results to <dir>` 行。
3. 对于每个唯一的内核（通过 `kernel.ttadapter.mlir` 中的内核名称去重），运行 `bishengir-compile`。
   - 脚本会自动探测当前编译器支持的 IR dump 参数（优先级：`--mlir-print-ir-after-all` > `--bishengir-print-ir-after` > `--print-after-all`），也可通过环境变量 `BISHENGIR_PRINT_IR_FLAG` 强制指定。
   - 若使用 `--mlir-print-ir-after-all` / `--print-after-all`，输出中会有多个 `IR Dump After` 段落，脚本提取最后一个。
   - 若使用 `--bishengir-print-ir-after=<pass>`，输出直接就是该 pass 后的 IR。
4. 从编译器输出中提取最后一个阶段的 IR。
5. 保存至 `<IR_OUTPUT_DIR>/<kernel_name>_last_pass.mlir`。

**重要提示：** `bishengir-compile` 会调用 `hivmc-a5` 作为子进程。bishengir 的 bin 目录必须位于 `$PATH` 中。如果缺失，流程将在 `InjectIR` 处停止，而不会继续到 LLVM IR。

### 输出文件

| 文件 | 描述 |
|------|-------------|
| `<kernel_name>_ttir.mlir` | Triton 前端 IR（可选） |
| `<kernel_name>_ttadapter.mlir` | 适配器 IR（可选） |
| `<kernel_name>_last_pass.mlir` | BishengIR 最终阶段 IR（**必需**） |

中间文件的保存由 `IR_SAVE_TTIR`、`IR_SAVE_TTADAPTER` 和 `IR_SAVE_ALL` 控制。

### 处理自动调优 (Autotune)

当使用 `@triton.autotune` 时，会生成多个转储目录（每个配置变体一个）。脚本通过内核名称进行去重，仅保留每个唯一内核的第一个配置变体。无论有多少个 autotune 配置，每个内核最终只生成一个 IR 文件。

### Triton 缓存损坏

编译阶段 `bishengir-compile` 的段错误通常是由 Triton 缓存中不兼容的预编译头引起的。在进一步调查之前，请清除缓存：

rm -rf ~/.triton/cache/*

## IR 分析决策流程

在做任何分析策略之前，先仔细阅读以下文件获取 IR 分析优化相关指南：
  - **主要参考流程** →`./docs_triton_IR/AGENTS.md`
  - **支撑文件目录** ：
  `./docs_triton_IR/AscendNPU-IR/README.md`
  `./docs_triton_IR/docs_ascendnpu_ir/README.md`
  `./docs_triton_IR/docs_for_triton_agent/README.md`
  `./docs_triton_IR/docs_triton_ascend/README.md`
  `./docs_triton_IR/triton-ascend/README.md`

### 重要约束
主要参考流程`./docs_triton_IR/AGENTS.md`中 性能分析指南的 步骤1：IR分析（主要）和 知识库层级文档指引中的IR分析部分，对生成的IR文件进行分析优化。
严禁在优化完成前使用msprof分析工具
