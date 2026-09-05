---
name: external-cannbot-ops-ops-profiling-msopprof-visualization
description: 面向算子的 msOpProf 数据采集与可视化技能。用于执行环境检查、最小非重复采集、已有采集结果重绘，以及生成包含 Details、Roofline、Timeline、Cache、Source、片上内存、Warp
  Stall、指令时间线和原始数据页面的独立 HTML 报告。当用户要求将上板采集的算子性能数据生成可视化报告、查看某个性能模块页面、或基于已有采集目录重建报告时触发。
original-name: msopprof-visualization
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# msOpProf 采集与可视化

## 职责

将算子工程或已有 msOpProf 采集目录转换为可验证的独立 HTML 性能报告。

本技能负责：

1. 识别 CANN、msprof、设备、可执行文件和应用工作目录；
2. 按请求模块规划最少且不重复的 profiler replay；
3. 对命令结果和产物进行语义校验，安全复用已完成采集；
4. 构造统一报告载荷；
5. 生成交互式 HTML，并明确区分“可用、缺失、不可支持”；
6. 输出机器可读的验证结果和性能计时。

不得推测或伪造 profiler 未提供的指标。缺少证据时，必须显示不可用原因，或在非定向报告中明确省略。

## 输入模式

### 1. 完整采集并生成报告

需要：

- `--operator-path`：算子工程根目录；
- `--output`：采集和报告输出目录；
- 可自动发现或显式指定的 `--app`；
- 应用必需参数，使用可重复的 `--app-arg`；
- 可选的 `--kernel-name`。

```bash
python scripts/run_pipeline.py \
  --operator-path /path/to/operator \
  --output /path/to/output \
  --preset complete
```

### 2. 只使用已有采集结果生成报告

已有目录必须包含 `collection_manifest.json`。

```bash
python scripts/run_pipeline.py \
  --mode visualize \
  --output /path/to/collection \
  --preset complete
```

也可直接运行：

```bash
python scripts/visualize.py \
  --input /path/to/collection \
  --output /path/to/report \
  --preset complete
```

### 3. 定向生成模块

```bash
python scripts/visualize.py \
  --input /path/to/collection \
  --output /path/to/report \
  --feature roofline \
  --feature timeline \
  --feature cache
```

查看可用特性：

```bash
python scripts/visualize.py --list-features
python scripts/visualize.py --explain source
```

不得自行创造特性名称。

## 标准工作流

### 步骤 1：判断执行模式

- 用户提供算子工程：默认 `full`；
- 用户只需要采集：使用 `collect`；
- 用户已有 `collection_manifest.json`：优先 `visualize`，不得重复采集；
- 只需要检查命令规划：先使用 `--dry-run`。

### 步骤 2：解析环境

稳定主机可建立可复用环境档案：

```bash
python scripts/environment_context.py init \
  --profile /stable/path/msopprof_environment.json \
  --base-dir /path/to/operator \
  --source 'source /usr/local/Ascend/ascend-toolkit/set_env.sh'
```

环境档案只允许复用稳定的 CANN 环境变量、`msprof` 路径和 CLI 能力信息。它不得跳过算子相关的可执行文件、Kernel、调试符号、插桩和产物校验。

涉及环境复用或失效时，读取 `references/environment-and-failure.md`。

### 步骤 3：执行短时预检

正式采集前默认执行一次 `BasicInfo` canary：

- `--launch-count=1`；
- 使用解析出的应用工作目录；
- 使用短时超时；
- 校验返回码和实际产物。

超时、非零返回或语义空产物均视为基础设施失败。立即中止，不得继续执行 Occupancy、Roofline、Timeline、Source 等昂贵采集。

所有 profiler 命令必须在独立进程组中运行。超时后先终止、再强制清理完整进程树，并保存日志和诊断信息。

### 步骤 4：选择最小采集计划

根据请求页面反推采集块，并执行语义去重：

- `Roofline` 已包含标准 Default 指标时，不额外执行相同 Default replay；
- `MemoryDetail` 缺失时，可在 Roofline 产物通过内存模块语义校验后复用；
- Raw Data 优先复用 MemoryDetail，其次复用 Roofline；
- 单次精确 Kernel、单次 launch 的成功 preflight 可复用为 Discovery；
- Timeline、Source、Occupancy 等语义独立模块不得仅因 CLI 支持逗号组合而合并。

详细规则读取 `references/collection-workflow.md`。

### 步骤 5：语义校验产物

命令返回码为 0 不等于采集成功。必须校验：

- profiler 输出目录存在；
- 必需文件存在；
- 产物中包含可解析、非空且符合模块语义的数据；
- alias/reuse 的来源和原因已写入 manifest；
- 顶层 `_internal/run_state.json` 为 `completed`，才允许后续复用。

Source 需要精确构建树中的 `.debug_line`。`-g` 是编译参数，不是 msprof 参数。不得把二进制 printable strings 当作完整源码映射。

### 步骤 6：独立构建报告模块

支持页面：

- `details`：基本信息、核占用率、计算、内存和建议；
- `roofline`：实测点、带宽线、算力上限和交互坐标；
- `timeline`：原生事件、lane、范围选择和切片详情；
- `cache`：Hit/Miss、Block、Cache family 和统计；
- `onchip-memory`：仅使用编译器 `memory_info.json`；
- `source`：源码文件、行、地址、指令和显式关联；
- `warp-stall`：仅使用 SIMT/PCSampling 证据；
- `instruction-timeline`：仅使用受支持的指令时间线产物；
- `raw-data`：原始 CSV 表格。

模块数据合同读取 `references/visualization-contract.md`。

### 步骤 7：生成完整特性报告

要在同一报告中展示所有支持页面，并对缺失模块保留诊断页，使用：

```bash
python scripts/visualize.py \
  --input /path/to/collection \
  --output /path/to/report \
  --feature details \
  --feature roofline \
  --feature timeline \
  --feature cache \
  --feature onchip-memory \
  --feature source \
  --feature warp-stall \
  --feature instruction-timeline \
  --feature raw-data \
  --unavailable-policy explain \
  --report-name msopprof_complete_report.html
```

报告较大时可用 `--compress-payload on` 将载荷以 gzip 内嵌（体积约降低一个数量级，需现代浏览器解压）；默认 `off` 为纯 JSON 内嵌，兼容性最好。两种模式渲染内容一致。

需要预估各 preset 的采集耗时（不重跑 profiler）时：

```bash
python scripts/estimate_runtime.py \
  --timing /path/to/output/_internal/timing_summary.json
```

### 步骤 8：验证与交付

运行结构和真实数据验证：

```bash
python scripts/self_check.py \
  --collection /path/to/collection \
  --output /path/to/validation-output
```

交付时至少说明：

- HTML 报告路径；
- `report_index.json`；
- 可用模块；
- 不可用模块及原因；
- 运行耗时和峰值内存；
- 验证是否通过。

不得让用户打开 `templates/report_template.html`。应打开由 `--report-name` 指定的最终报告，并返回其绝对路径。

## 输出合同

默认输出包括：

```text
OUTPUT/
├── collection_manifest.json
├── feature_catalog.json
├── 00_discovery/
├── 01_details/
├── 02_roofline/
├── 03_timeline/
├── 04_source/
├── 05_warp_stall/
├── 06_instruction_timeline/
├── 07_memory_detail/
├── 08_raw_data/
├── 09_timeline_detail/
├── 10_kernel_scale/
├── 11_onchip_memory/
├── _internal/
├── report_payload.json
├── report_index.json
└── report.html
```

`report_index.json` 是机器可读交付回执。`collection_manifest.json` 是采集真源。

## 强制规则

1. 不得伪造缺失指标或从不等价数据推导模块。
2. `memory_info.json` 缺失时，不得用 MemoryDetail 计数伪造片上内存分配寿命和地址。
3. 不得复用 `running`、`aborted`、超时、非零返回或语义空的采集结果。
4. 采集目录不得具有 group/other write 权限；发现后应修正并记录。
5. `--app` 一旦指定即为权威；未指定时必须记录候选排序和选择理由。
6. 应用工作目录默认是 `--operator-path`，不是调用命令时的 shell 目录。
7. Timeline 和 Source 成本较高；只有请求其独有证据时才执行。
8. 报告页面必须使用同一 canonical payload，不得为不同页面重复解析并产生不一致数据。
9. 完整报告允许模块不可用，但必须提供证据和下一步动作。
10. 所有正式提交不得包含 `.pytest_cache`、`__pycache__`、`*.pyc`、运行日志或历史验证结果。

## 参考资料

按需读取，避免一次性加载全部内容：

- `references/collection-workflow.md`：采集计划、参数和 replay 去重；
- `references/environment-and-failure.md`：环境档案、预检、超时和诊断；
- `references/data-contract.md`：manifest、alias、可用性和输出真源；
- `references/visualization-contract.md`：各可视化模块的数据与交互合同；
- `references/validation-and-performance.md`：验证、浏览器检查和性能门禁。
