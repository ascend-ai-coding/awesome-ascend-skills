# Source Analysis — 执行总纲

## 角色

分析算子源码（tiling + kernel + 接口），提取分支路径和参数维度，产出代码路径清单、算子接口模型和参数定义，供下游 Step 5（case mapper）和 TTK 模块消费。

## 输入

| 数据 | 来源 | 用途 |
|------|------|------|
| 算子源码路径 | Step 1.2 | tiling / kernel / 接口分析 |
| 平台参数（npu_arch / soc_version / core_count / ub_size） | Step 1.0-1.3 | 平台分支判断 |

## 输出文件

| 文件 | Phase | 说明 |
|------|-------|------|
| `S2P0_scout_t.md` | Phase 0 | tiling 侦察报告（入口函数 + P0/P1/P2 文件清单） |
| `S2P0_scout_t.json` | Phase 0 | tiling 侦察结构化数据（供 verify 消费） |
| `S2P0_scout_k.md` | Phase 0 | kernel 侦察报告（P0 当前 kernel 文件 + op.Init 可达模板 + P1/P2 include 文件 + key 列表） |
| `S2P0_scout_k.json` | Phase 0 | kernel 侦察结构化数据（供 verify 消费，含 `reachable_kernel_templates`） |
| `S2P0_file_manifest.json` | Phase 0 | 源码文件清单（tiling/kernel 优先级 + 排除列表） |
| `S2P0_source_scope.md` | Phase 0 | 源码读取最大边界（后续任务按自身规则读取；P2 禁止读取） |
| `S2P1_path_list.json` | Phase 1 | 代码路径清单（path graph：tiling_paths + kernel_branches + paths） |
| `S2P1_route_dispatch_model.md` | Phase 1 | evaluator 指导书 |
| `S2P1_route_extract.py` | Phase 1 | route evaluator 脚本 |
| `S2P1_case_generator.py` | Phase 1 | route case 生成器 |
| `S2P1_route_cases.json` | Phase 1 | route cases（case generator 输出） |
| `S2P1_cases.json` | Phase 1 | 用例文件（evaluator 附带输出，flat array，仅含 reachable 用例） |
| `S2P1_tiling_glossary.md` | Phase 1 | tiling 变量含义表（tiling 源码变量名 → 语义名映射） |
| `S2P1_operator_model.json` | Phase 1 | 算子接口模型（inputs/outputs/attributes，以 `_def.cpp` 为权威约束源） |
| `S2P1_low_configs.json` | Phase 1 | 常见网络 shape 配置（语义参数名） |
| `S2P1_postcheck_report.json` | Phase 1 | Phase 1 统一 postcheck 报告 |

## Step 2：源码分析任务

Step 2 要从算子源码中回答三个问题：

- 可执行路径：tiling 分支、kernel dispatch、tiling key 到 kernel 模板的映射是什么，route seed 的 `case_values` 字段多样性目标是什么？
- 接口模型：inputs / outputs / attributes / dtype / shape / 约束是什么？
- 常见配置：low 档应覆盖哪些典型网络 shape？

Step 2 分两阶段执行：Phase 0 生成源码读取边界；Phase 1 并行回答上述三个问题。

### Phase 0：生成源码读取边界

Phase 0 只做一件事：主 Agent 同步调用一次 `scripts/run_s2p0.py`，生成 Phase 1 的源码读取边界。

主 Agent 只根据退出码判断是否进入 Phase 1。成功后，Phase 1 使用 `S2P0_source_scope.md` 作为源码读取边界。

不要做：

- 不拆分执行 `scout_t.py`、`scout_k.py`、`s2p0_verify.py`。
- 不读取或分析 Phase 0 相关脚本源码。
- 不派发子 agent 分析 Phase 0。
- 不读取 scout 报告，不总结 P0/P1/P2。

完成条件：`run_s2p0.py` 退出码为 0。

### Phase 1：并行回答三个问题

Phase 1 必须在 Phase 0 成功后启动。Task A/B/C 互不依赖，主 Agent 必须一次性启动三个独立子 agent：一个执行 Task A、一个执行 Task B、一个执行 Task C；禁止串行执行，禁止等待任一 Task 完成、校验或重试后再派发其他 Task。

| Task | 要回答的问题 | 产物 |
|------|--------------|------|
| Task A | 源码有哪些可执行 tiling/kernel 路径？tiling key 如何映射到 kernel 模板？`case_values` 字段应满足什么多样性目标？ | `S2P1_route_dispatch_model.md`、`S2P1_tiling_glossary.md`、`S2P1_route_extract.py`、`S2P1_case_generator.py`、`S2P1_route_cases.json`、`S2P1_path_list.json`、`S2P1_cases.json` |
| Task B | 算子的对外接口、输入输出、属性和约束是什么？ | `S2P1_operator_model.json` |
| Task C | low 档应覆盖哪些常见网络 shape 配置？ | `S2P1_low_configs.json` |

校验规则：

- A/B/C 全部返回后，主 Agent 只调用 `scripts/check_s2p1.py` 做 Phase 1 产出的文件级粗验收（存在性 + JSON 可解析 + 顶层结构），并给出 `owner_task` 失败归属。
- postcheck PASS：Step 2 完成，进入 Step 3。
- postcheck FAIL：按报告中的 `owner_task` 只重试对应 Task；`owner_task=multiple` 时只重试失败 owners；重试后再次运行 postcheck。
- 主 Agent 不逐个手写 JSON 校验，不根据子 agent 摘要判断 Step 2 是否完成。

### 源码读取策略

Phase 1 不做全仓源码阅读，只按 Phase 0 生成的边界读取源码。后续任务不得自行扩大边界。

| 范围 | 用途 | 读取策略 |
|------|------|----------|
| P0 | kernel dispatch 入口，建立 tiling key 与 kernel 模板关系 | 默认读取 |
| P1 | 当前 kernel 实际可达模板，分析实现内部约束 | 按需读取 |
| P2 | include 可达但当前 kernel 未使用的文件 | 默认不读 |

P0/P1 是最大可读边界，不代表所有任务都必须读取；Task A 默认仅读 P0 dispatch 入口，只有分析具体 kernel 内部约束时才读取 P1。

Scout-T/K 只做文件级筛选；后续任务读取 P0/P1 时，仍需按目标平台过滤源码内 `#if/#elif/#else/#endif`、`GetCurNpuArch()`、SocVersion 等平台条件。

`S2P0_scout_k.json.reachable_kernel_templates` 记录已解析 `op.Init(...)` 分支；`unresolved_init_calls` 仅作诊断，不单独阻塞 Phase 0。`S2P0_file_manifest.json.kernel.file_list` 只能包含 P0/P1；`kernel.excluded` 与 `S2P0_source_scope.md` 排除区必须包含 P2。

### P2 最小片段例外

仅当必填产物字段依赖某个已知 P2 符号，且 P0/P1 无法获得时，允许读取最小必要片段。

约束：

- 仅限已知符号、宏、枚举或常量定义。
- 只读定义附近，不展开整个文件，不跟随 include。
- 不读取 P2 kernel 模板实现分析算法。
- 必须在产物中记录读取理由和源码证据。

### Phase 0 命令

```bash
python3 {skill_base}/scripts/run_s2p0.py \
  --op-name {op_name} \
  --op-path {op_path} \
  --npu-arch {npu_arch} \
  --soc-version {soc_version}
```

如需记录耗时，可在命令前加 `/usr/bin/time -p`。耗时记录只用于观察，不参与流程判定。

失败处理：

- 退出码非 0：按 wrapper 输出的失败阶段重跑一次；仍失败则触发轮次耗尽协议。
- 退出码为 0：进入 Phase 1。

### Phase 1 Postcheck 命令

```bash
python3 {skill_base}/scripts/check_s2p1.py \
  --whitebox-dir {output_dir}
```

失败处理：postcheck 失败后按 `owner_task` 重试对应 Task 一次；仍失败则触发轮次耗尽协议。

Step 2 完成标志：`check_s2p1.py` 输出 `PASS S2P1`。

## 子 Agent Prompt 模板

主 Agent 派发 Task A/B/C 子 agent 时，仅替换占位符，不改写模板结构。

### 占位符定义

| 占位符 | 含义 | 填入来源 |
|--------|------|---------|
| `{op_name}` | 算子名称 | Step 1.1 |
| `{op_path}` | 算子源码路径 | Step 1.2 |
| `{npu_arch}` | NPU 架构 | Step 1.0 |
| `{soc_version}` | SOC 版本 | Step 1.0 |
| `{chip_model}` | 芯片型号 | Step 1.0 |
| `{core_count}` | VectorCore 核数 | Step 1.3 |
| `{ub_size}` | UB 大小(KB) | Step 1.3 |
| `{output_dir}` | 产物写入路径 | `{op_path}/tests/whitebox/` |
| `{skill_base}` | 技能根目录绝对路径 | skill 自身路径 |
| `{task_a_entry_doc}` | Task A 入口参考文档路径 | `{skill_base}/references/task-a/00-overview.md` |
| `{source_scope_path}` | S2P0_source_scope.md 路径 | Phase 0 产出 |

---

### Task A：Route Evaluator 系统

主 Agent 派发 Task A 时，先补充调用侧说明：Task A 的参考文档根目录是 `{skill_base}/references/task-a/`，不要从当前文件所在目录推导相对路径。

```
你是代码路径分析专家。请回答：该算子源码中有哪些可执行 tiling/kernel 路径，tiling key 如何映射到 kernel 模板，`case_values` 字段应满足什么多样性目标？

请先 Read 参考文档 `{task_a_entry_doc}`，然后按 01 → 02 → 03 顺序逐步执行。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- Task A 入口文档：{task_a_entry_doc}
- 产出写入路径：{output_dir}
- S2P0_source_scope.md 路径：{source_scope_path}

源码读取范围：Read `{source_scope_path}` 获取文件清单，严格按该文件列出的范围读取，不得自行添加。

完成后只返回 `PASS Task A`、产物路径列表和必要计数（tiling_paths / kernel_branches / paths / orphans / glossary）。
```

---

### Task B：接口分析

```
你是接口分析专家。请回答：该算子的对外接口是什么，inputs/outputs/attributes 及约束分别是什么？

请先 Read 参考文档 `{skill_base}/references/source-analysis/01-interface-analyzer.md`，然后按照文档中的规则逐步执行接口分析。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- 产出写入路径：{output_dir}

请生成 `S2P1_operator_model.json` 到产出路径。完成后只返回 `PASS Task B`、产物路径和 inputs/outputs/attributes 数量。
```

---

### Task C：网络搜索

```
你是网络配置搜索专家。请回答：该算子 low 档应覆盖哪些常见网络 shape 配置？

请先 Read 参考文档 `{skill_base}/references/source-analysis/02-network-search.md`，然后按照文档中的规则逐步执行网络搜索。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- 产出写入路径：{output_dir}

请生成 `S2P1_low_configs.json` 到产出路径。完成后只返回 `PASS Task C`、产物路径和典型配置数量。
```
