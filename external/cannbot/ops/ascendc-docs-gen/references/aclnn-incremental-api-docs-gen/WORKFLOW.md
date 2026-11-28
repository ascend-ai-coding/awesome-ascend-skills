# WORKFLOW：ACLNN 增量算子 API 文档生成组件

本组件在目标仓中的位置：`ops/ascendc-docs-gen/references/aclnn-incremental-api-docs-gen/`。本文件是组件入口文档，覆盖定位、目录布局、环境变量、用法、输出契约、架构声明与安装。

## 1. 定位

增量式 ACLNN 算子 API 文档生成组件。输入是同一算子的两个不同版本（original / modified 双仓检出，如不同 commit / tag / 分支的检出）。组件对比两版算子代码差异，基于新版本代码仅重写受影响的文档章节，产出修改后的文档，而非整篇重新生成。主入口为 `src/run_all_cli.py`。

处理链路为四阶段管道：

| 阶段 | 模块 | 职责 |
| --- | --- | --- |
| 快照构建 | `src/snapshot_construction/` | 为 original 版本构建基线：doc_json、code_map、tree_id、锚点与证据元组 |
| 变更检测 | `src/change_detection/` | 4 级锚点重定向（文件、函数、章节、注释）+ doc-part 变更影响分析 |
| 生成与校验 | `src/generation_validation/` | 4 阶段 LLM 流水线：Evidence、Style、Generation、Validation |
| 文档回填 | `src/md_fill/` | 按标题把生成内容回填进既有 markdown |

## 2. 目录布局

```
├── src/
│   ├── run_all_cli.py                     # 主编排入口（非交互全参 + TTY 交互 TUI）
│   ├── main_snapshot.py                   # 阶段入口：快照构建（Stage 1）
│   ├── run_change_detection_pipeline.py   # 阶段入口：变更检测 + 生成（Stage 2-6）
│   ├── run_consistency_check_pipeline.py  # 阶段入口：validate_only 一致性校验
│   ├── settings.py                        # 环境变量读取唯一站点（load_settings / apply_settings）
│   ├── logger.py                          # 全局日志
│   ├── snapshot_construction/             # 阶段 1：快照构建
│   ├── change_detection/                  # 阶段 2：变更检测
│   ├── generation_validation/             # 阶段 3：LLM 生成与校验
│   ├── llm/                               # LLMClient（OpenAI 兼容，重试/退避）
│   └── md_fill/                           # 阶段 4：markdown 回填
├── resources/            # 运行时资源（stages.py 加载 soul.json/templates.yaml；config.py 定义 chip_map.yaml）
│   ├── templates.yaml    # 品类模板
│   ├── chip_map.yaml     # 芯片代号映射
│   └── soul.json         # 风格与禁词配置
├── tests/                # pytest 测试套件
├── gitcode_pr_file_get.py        # 可选外围取件工具（见附录 8 节，非主流程）
├── run_pr_batch.py               # 可选外围批量工具（见附录 8 节，非主流程）
├── requirements.txt              # 唯一依赖源（宽松约束）
└── pytest.ini                    # pytest 配置（pythonpath 与 norecursedirs）
```

几点说明：`gitcode_pr_file_get.py` 与 `run_pr_batch.py` 为可选外围工具，不属于版本对比主流程，其用法与数据清理注意事项见附录 8 节。

## 3. 环境变量

全部环境变量集中由 `src/settings.py` 的 `load_settings()` 读取与校验（import 期零副作用，校验发生在各入口启动期）。

### 必填

| 变量 | 用途 |
| --- | --- |
| `INCREMENTAL_LLM_KEY` | LLM 网关 API key |
| `INCREMENTAL_LLM_URL` | LLM 网关 base_url（OpenAI 兼容） |

任一缺失时，入口启动期（参数解析之后、管道执行之前）以 exit 2 退出，报错含变量名与本文件指引：

```
[settings] 环境变量 INCREMENTAL_LLM_KEY 未设置——请参考 WORKFLOW.md 配置
```

不存在静默回退 api.openai.com 的行为。

### 可选带默认

| 变量 | 用途 | 默认 |
| --- | --- | --- |
| `INCREMENTAL_LLM_MODEL` | 模型名 | `qwen3.8-max`（`src/settings.py` 单点定义） |
| `CANN_PLUGIN_LOG_DIR` | 日志目录 | `<repo>/data/logs`（首次写日志时惰性创建，import 无副作用） |

### 场景型（按需设置）

| 变量 | 用途 |
| --- | --- |
| `CANN_OPS_ROOT` | 本地 ops 仓根目录。`main_snapshot.py` 的 `--workspace_dir` 默认值；两者皆缺时该入口明确报错。也是测试夹具与 dry-run 脚本的回退项 |
| `CANN_OPS_WORKSPACE` | cann-document-generation 工作目录（变更检测自测入口 `run_change_detection_cli.py` 的 `__main__` 自测块使用） |
| `CANN_EVIDENCE_DIR` | 证据输出目录（dry-run 命名验证脚本 `tests/dry_run_naming_validation.py` 使用） |
| `TEST_OPERATOR_PATH` | 测试夹具的算子路径，优先于 `CANN_OPS_ROOT`（`tests/anchor_construction/conftest.py`：两者皆缺时相关测试自动 skip） |

### Legacy（旧 CLI 自测用途，日常使用无需设置）

`CANN_ORIGIN_OP_PATH`、`CANN_DATA_PATH`、`CANN_OUTPUT_PATH` 是历史自测脚本遗留：现行代码中仅 `CANN_DATA_PATH` 仍被一个自测脚本读取（`src/change_detection/test_change_detection/test_path.py`），`CANN_ORIGIN_OP_PATH` 与 `CANN_OUTPUT_PATH` 已无任何读取点。

### 代理

本组件不再代设 `no_proxy`（历史版本曾把它覆写为 localhost，已移除）。LLM 网关需要绕过代理时，请自行在运行环境设置 `no_proxy` / `NO_PROXY`。

## 4. 用法

以下命令均从仓库根执行（`python` 需 3.11+；仓内虚拟环境为 `.venv/bin/python`）。

### 4.1 主入口：run_all_cli.py

非交互完整模式（全部 flag）：

```bash
python src/run_all_cli.py \
  --original_op_repo <original-repo-root> \
  --original_op_path <rel/op/path> \
  --modified_op_repo <modified-repo-root> \
  --modified_op_path <rel/op/path> \
  --output_data_path <output-dir> \
  [--validate-only] [--skip-fix] \
  [--md-stem aclnnXxx ...] \
  [--workspace_dir <dir>] \
  [--dashboard]
```

| flag | 说明 |
| --- | --- |
| `--original_op_repo` / `--original_op_path` | 修改前算子仓根路径 + 仓内相对路径（如 `math/add`） |
| `--modified_op_repo` / `--modified_op_path` | 修改后算子仓根路径 + 仓内相对路径 |
| `--output_data_path` | 输出数据根目录（非交互模式必填） |
| `--validate-only` | 仅预校验，跳过变更检测与生成 |
| `--skip-fix` | 跳过校验失败的修复步骤（仅影响 `--validate-only`） |
| `--md-stem` | 一个或多个 markdown 文档 stem（如 `aclnnAbs aclnnAdd`），省略时自动选择 |
| `--workspace_dir` | 工作区根目录（默认：仓库根） |
| `--dashboard` | 并行 doc_part 处理的实时面板 |

行为模式：

- 非 TTY（管道 / CI）：4 个路径 flag（`--original_op_repo`、`--original_op_path`、`--modified_op_repo`、`--modified_op_path`）与 `--output_data_path` 必填；缺失时 exit 2 报错并列出缺失 flag 名，不回退交互界面。
- TTY：省略参数进入交互式 Rich/questionary 引导界面；`--output_data_path` 缺省推导为 `<workspace>/data/outputs/snapshot`。

模块模式等价可用：`python -m src.run_all_cli --help`。

### 4.2 阶段入口（VSCode 扩展 / 脚本对接）

快照构建（Stage 1）：

```bash
python src/main_snapshot.py \
  --operator_path <path> \
  --data_path <path> \
  --workspace_dir <ops-root>
```

`--workspace_dir` 缺省读 `CANN_OPS_ROOT`；两者皆缺时明确报错退出。`--operator_path` 默认 `data/ops-math/math/abs`，`--data_path` 默认 `out`。

变更检测 + 生成（Stage 2-6）：

```bash
python src/run_change_detection_pipeline.py \
  --snapshot-id <id> \
  --workspace-dir <dir> \
  --original-op-dir <rel> \
  --modified-op-dir <rel> \
  --data-path <rel> \
  [--operator-version base] \
  [--verbose]
```

一致性校验（validate_only 路径）：

```bash
python src/run_consistency_check_pipeline.py \
  --snapshot-id <id> \
  --workspace-dir <dir> \
  --original-op-dir <rel> \
  --data-path <rel> \
  [--operator-version base] \
  [--verbose]
```

注意 flag 风格的历史差异：`main_snapshot.py` 用下划线（`--workspace_dir`），两个 stage 入口用连字符（`--workspace-dir`），均如实保留，请照抄。

### 4.3 校验 UI

```bash
python -m src.generation_validation.ui.cli --help
python -m src.generation_validation.ui.cli -l
python -m src.generation_validation.ui.cli -f <evidence-file> -d <doc-part> -m full|validate_only
```

- `-f/--file`：证据文件（相对 `data/evidences/`）
- `-d/--doc-part`：文档部件（support / summary / function_proto / sub_function_hash / constraint / example）
- `-m/--mode`：full 或 validate_only（默认 full）
- `-l/--list`：列出可用证据文件后退出
- `-o/--ops-dir`：算子源码目录（example doc_part 需要）

`data/evidences/` 下无证据文件时报错退出（exit 1）。

## 5. 输出与契约

### 5.1 最终交付物（Markdown）

组件**不原地修改原文档**：回填基底优先取 modified（新版本）仓 `docs/` 下的同名 md（保留人工修改但未被生成覆盖的章节），缺失时回退 original 版本的选中文件；产物是带 `[new]` 前缀的新文件，是否以其替换原文档由使用者决定。

| 入口 | 最终 Markdown 位置 | 命名 |
| --- | --- | --- |
| `run_all_cli.py`（主入口） | `<output_data_path>/<算子名>/<md_stem>/results/` | `[new]<md_stem>_<YYYYmmdd_HHMMSS>.md` |
| `run_change_detection_pipeline.py`（阶段入口） | `<original-op-dir>/docs/`（算子仓内） | `[new]<md_stem>.md`（无时间戳）；路径同时出现在 `==result==` JSON 的 `markdownFiles` 字段（`originalPath` / `generatedPath`） |

### 5.2 完成判据

以 **`[new]` Markdown 文件存在且校验通过**为完成条件：

- 同目录 `results/detailed_results.json` 中相应 doc_part 的 `success` 为 `true`；
- 相应 doc_part 未出现在阻止回填清单中——校验失败的 doc_part 不会回填进最终文档（阻止原因见 `results/evaluation_reasons.json` 的 `gen_val_result`）。

### 5.3 中间产物与清理

- `--output_data_path` 是**运行中间产物目录**（非交互必填；TTY 模式缺省推导 `<workspace>/data/outputs/snapshot`），其下累积快照、`anchor/`（证据）、`style/`、`re_detection/`、`results/` 等中间数据，仅用于问题排查与结果追溯。
- 日志：写入 `CANN_PLUGIN_LOG_DIR` 指定目录，缺省 `<repo>/data/logs`。
- 组件**不提供清理手段**：建议将 `--output_data_path` 置于临时目录，或在确认交付物后择机主动清理（外围批量工具的数据累积注意事项同此，见附录 8 节）。

### stdout 标记与哑文件（VSCode 扩展对接契约）

三个阶段入口在 stdout 发射机器可读标记，标记后输出 JSON 载荷；两个 stage 入口另以哑文件表达成败。扩展侧按字节级匹配这些字符串，修改任何入口时不得改动。哑文件保证：两个 stage 入口在**运行开始时清理旧 DONE/FAILED**，且写入终态前删除相反状态文件（写 DONE 删 FAILED、写 FAILED 删 DONE）——运行期间无标记 = 进行中，运行结束后存在的标记必属本次运行：

| 标记 / 文件 | main_snapshot.py | run_change_detection_pipeline.py | run_consistency_check_pipeline.py | run_all_cli.py |
| --- | --- | --- | --- | --- |
| `===snapshot-result===` | 发射（快照 JSON） | 不发射 | 不发射 | 不发射 |
| `===change-detection-result===` / `===/change-detection-result===` | 不发射 | 发射（透传自 change_deteciton_cli） | 不发射 | 不发射到真实 stdout（内部调用 change_deteciton_cli 时经 redirect_stdout_to_log 重定向，标记与 JSON 仅进入日志文件） |
| `==result==` / `==/result==` | 不发射 | 发射（最终 JSON） | 发射（最终 JSON） | 不发射 |
| `<workspace-dir>/<data-path>/DONE` / `FAILED` 哑文件 | 不写 | 成功写 DONE，失败写 FAILED | 同左 | 不写 |

两点如实说明：

1. `main_snapshot.py` 只发射 stdout 标记，不写 DONE/FAILED 哑文件。这是历史设计的不对称性，保持原状。
2. `run_all_cli.py` 是面向人类终端的编排入口，不发射 `===snapshot-result===` 与 `==result==`，也不写哑文件；其完整管道内部虽会调用 `change_deteciton_cli()`（该函数自身发射 `===change-detection-result===` 标记），但调用整体包裹在 `redirect_stdout_to_log()` 内，标记与 JSON **只写入日志**（`CANN_PLUGIN_LOG_DIR`，缺省 `<repo>/data/logs`），不会出现在真实 stdout（`--validate-only` 模式则完全跳过该阶段）。需要机器可读标记的对接方请使用阶段入口 `run_change_detection_pipeline.py`。

## 6. 架构声明

本组件是自带 LLM 编排器的独立 Python 管道，不是一组交给宿主 agent 推理的确定性工具：

- `LLMClient`（`src/llm/client.py`）直连 `INCREMENTAL_LLM_URL` 网关（OpenAI 兼容协议；重试、退避与超时由组件内部管理）。
- 四阶段管道内部自行调度（含 doc_part 级并行），不依赖外部 agent 驱动。
- 与宿主 agent 推理模式（skill 提供知识 + 确定性工具、推理交给宿主模型）的分工不同：本组件把"读代码、比对变更、写文档"的推理过程内化为确定性管道加 LLM 调用。

接入方需要知晓的资源需求：

- `INCREMENTAL_LLM_*` 环境变量就绪（LLM 网关可达且额度充足）。
- 运行期产生并发 LLM 调用（doc_part 级并行，费用与速率需按上游网关评估）。
- 单次完整运行耗时以分钟计。

## 7. 安装

- Python 3.11+（仓内 `.python-version` 固定 3.11.9）。
- 依赖唯一来源为 `requirements.txt`，宽松约束共 10 个依赖（9 个运行时 + 1 个开发测试 pytest）；其中 `clang` / `libclang` 用于代码图解析，缺失会 import 失败：

```bash
pip install -r requirements.txt
```

- 测试：`python -m pytest`（配置在仓根 `pytest.ini`）。
- 更详细的安装说明见 `README.md`。

## 8. 附录：可选外围批量工具 run_pr_batch.py

`run_pr_batch.py` 为可选外围工具，不属于版本对比主流程：读取仓根任务清单 `gitcode_pr.txt`（两列空白分隔：`<pr_url>  <op_path>`，`#` 注释行忽略；清单不随仓分发，由使用者按此格式自建），通过 `gitcode_pr_file_get.py` 为每条任务自动检出前后两个版本（merge_base 与 head）到 `.temp_git/`，再逐条调用主入口 `run_all_cli.py`（`--md-stem` 自动选择），日志与批次报告写入 `data/outputs/pr_batch_logs/`，任一任务失败则以非零码退出。

```bash
python run_pr_batch.py [--dashboard]
python gitcode_pr_file_get.py <pr_url> [-o <dir>] [-t base head merge_base commit] [--sha <sha>]
```

`gitcode_pr_file_get.py` 也可单独用作取件工具：公开 PR 无需 token；默认输出到仓库内 `.temp_git`（相对脚本所在目录解析，与 cwd 无关）；`-t/--targets` 默认 `head`，含 `commit` 时必须提供 `--sha`。

**数据清理警示**：上述脚本不提供数据清理能力，`.temp_git/` 检出与 `data/outputs/` 产物、日志会随运行持续累积。建议使用临时目录运行，或在确认交付物后择机主动清理相关数据。
