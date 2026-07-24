---
name: external-cannbot-plugins-community-ops-perf-evolution-skills-evolution-report
description: 进化优化完成后自动生成标准化 HTML 可视化报告（全自动脚本生成，含自检，支持 ops-evo 和 lingxi-evo 两种 pipeline）。当进化轮次结束或需要汇总基线与各轮变体的性能对比、决策树与资源消耗时使用。
original-name: evolution-report
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

## What I do

从进化输出目录中解析数据，通过 **全自动脚本** 生成标准化 HTML 可视化报告。脚本自动检测 pipeline 类型（`ops-evo` 或 `lingxi-evo`），自动完成所有内容填充（测试用例、最优策略分析、图表、diff、决策树），并在生成后执行自检，输出警告到 stderr。

**不再依赖 LLM 后处理** — 脚本内部通过 `normalize_eval_json()` 自动处理 evaluation_results.json 的多种格式（完整格式、扁平格式、lingxi 格式），从 `call_spec.json` 和 `world_model.json` 自动提取测试用例和策略分析，无需手动填充占位符。

## When to use

ops-evo 或 lingxi-evo 进化流程完成后（所有轮次执行完毕），作为最后一步自动调用。

**自动检测**：脚本通过目录名（`_ops-evo_` vs `_evo_`）和目录结构自动判断 pipeline 类型，无需手动指定。

## 完整工作流

### Step 1: 调用脚本生成报告

```bash
python3 <skill_path>/scripts/generate_report.py <output_dir> [--baseline-source <path>]
```

脚本执行以下操作：
1. **Pipeline 检测**：通过目录名和结构自动检测 `ops-evo` 或 `lingxi-evo`
2. **数据加载**：读取 `baseline_evaluation.json`、`world_model_final.json`、各轮次 `evaluation_results.json`
3. **格式统一**：通过 `normalize_eval_json()` 自动处理多种 eval 格式（完整格式、扁平格式、lingxi 格式）
4. **内容填充**：
   - 测试用例：从 `shared/call_spec.json` 自动提取输入参数、标量参数
   - 最优策略分析：从 world model 节点和 `implementation_note.txt` 自动构建
   - 图表、diff、决策树、资源统计：脚本自动生成
5. **自检**：生成完成后自动执行 `self_check_report()`，验证数据一致性

脚本输出报告路径到 stdout，自检结果输出到 stderr。

### Step 2: 查看自检结果

脚本 stderr 输出自检结果：
- `[报告自检通过] 无警告` — 数据完整，可直接使用
- `[报告自检警告]` + 具体警告列表 — 需人工复核

**常见警告及处理**：

| 警告 | 含义 | 处理方式 |
|------|------|----------|
| `所有 N 个变体均标记为编译失败` | `evaluation_results.json` 格式异常或 ops-partial 子agent 未正确输出 | 检查子agent 是否调用 `evaluate_ops_direct.py`，或 `compilation_success` 字段是否为 `None` |
| `最优变体 compilation_success=False` | 最佳变体实际编译失败 | 检查 world_model 中的 score 是否与 eval 一致 |
| `最优变体 time_us 无效` | 评估未返回有效时间 | 检查评估子进程是否崩溃 |
| `speedup 不一致` | world_model score 与 eval 计算结果偏差 >5% | 检查 baseline_time 是否被正确记录 |
| `报告仍含未填充的 LLM_FILL 占位符` | 脚本自动填充逻辑有遗漏 | 检查 `call_spec.json` 和 `world_model.json` 是否存在 |
| `代码修改部分文件数异常多 (N 个)` | `modified_files/` 包含了未实际修改的文件（如 docs、tests）或 ops-partial 复制了整个目录 | 检查 `_should_include_modified_file()` 过滤规则，或确认 ops-partial 是否只保存了有差异的文件 |
| `modified_files 路径前缀可能未正确处理` | modified_files 目录多了一层算子名前缀（如 `modified_files/op_name/op_kernel/...`），与 `shared/original/`（`original/op_kernel/...`）路径不对齐 | 脚本会自动检测并剥离前缀，若仍报警请检查 `_detect_modified_files_prefix()` 逻辑 |
| `应用最优变体命令数量过多` | apply-cmd 中 cp 命令超过 20 条，通常是路径前缀未剥离导致每个文件单独一条命令 | 检查 `build_apply_cmd()` 中的路径归一化和批量合并逻辑 |
| `apply-cmd 含冗余指令` | apply-cmd 包含 shebang、注释或 `set -e` 等非核心内容 | `build_apply_cmd()` 已精简为仅变量定义 + cp 命令 |
| `未包含模型信息` | 无法从 `~/.claude/config.json` 或会话 JSONL 中解析模型名称 | 检查 `get_model_from_config()` 或 session 文件中的 `model` 字段 |
| `无源代码修改（所有文件与基线一致）` | **优化过程问题**：ops-partial 未产生实际代码修改，或修改后未正确保存到 `modified_files/`（仅 ops-evo） | 检查子agent 是否按规范执行了 diff 校验和保存；若持续出现，需加强修改验证指令 |
| `代码修改第 N 段 diff 为新增文件` | diff 中出现了 `--- /dev/null` 标记的全新文件，不应出现在"代码修改"章节 | 检查 `build_code_diff_sections()` 是否正确跳过了无基线对应的新文件 |
| `apply-cmd 存在 N 个重复目标路径` | cp 命令中同一目标文件出现多次，通常是路径去重逻辑失效 | 检查 `build_apply_cmd()` 中的 `seen` 集合去重逻辑 |
| `轮次耗时过短` | 轮次持续时间明显低于子 agent 实际运行时间（如 <5 分钟） | 检查 `_get_dir_birthtime()` 是否正确获取了目录创建时间（Linux 上 `st_ctime` 不是创建时间，已改用 `stat -c '%W'`） |
| `总耗时远超各轮次之和` | 总耗时 > 轮次之和 × 3，说明包含了空闲等待时间 | 检查 `calc_active_duration_ms()` 是否正确排除了大间隔空闲 |
| `总耗时为 0 或负值` | 资源统计模块未能计算有效时长 | 检查会话 JSONL 文件是否存在、时间戳字段是否可解析 |
| `主会话耗时异常偏低` | 主会话显示时间远小于 wall-clock 跨度（如 2 分钟 vs 2 小时） | 正常现象：当主会话大部分时间在等待后台 agent 时，活跃间隔会被过滤。若活跃占比 <15%，自动回退到 wall-clock 跨度 |
| `词元统计量与预期偏差大` | 自动匹配的会话 JSONL 可能不是 ops-evo 主会话（如选中了后续修复报告的会话） | 检查 `.claude/projects/` 下是否有多个候选 JSONL；必要时使用 `--session-jsonl` 显式指定正确的会话文件 |

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | 是 | — | 进化输出目录路径 |
| `--baseline-source` | 否 | 自动检测 `shared/original/` (ops-evo) 或 `shared/kernel/` (lingxi-evo) | Baseline 源码目录，用于生成 diff |
| `--title` | 否 | 自动从算子名生成 | 自定义报告标题 |
| `--pipeline` | 否 | 自动从目录名推断 | 显式指定 pipeline 类型 (`ops-evo` 或 `lingxi-evo`)，覆盖自动检测 |
| `--session-jsonl` | 否 | 自动检测 | 显式指定主会话 JSONL 文件路径，用于提取 token/耗时统计。若未指定，脚本自动从 `.claude/projects/` 中匹配，并用 `round_1` 创建时间过滤以避免选错 |

## 输出

```
<output_dir>/evolution-report_<op_name>_<YYYYMMDD_HHMMSS>.html
```

## 数据来源

| 数据项 | 来源文件 | 处理方 |
|--------|----------|--------|
| 算子名、时间戳 | output 目录名 | 脚本 |
| 硬件信息 | `world_model_final.json` → `hw_params` | 脚本 |
| 测试用例参数 | `shared/call_spec.json` → inputs / scalar_args / tensor_kwargs | 脚本 |
| Baseline 耗时 | `baseline_evaluation.json` → `baseline.time_us` / `gen_time_ms_avg` / `baseline_time_us` | 脚本 |
| 每轮变体结果 | `round_N/parallel_M/evaluation_results.json` | 脚本（含格式归一化） |
| 代码修改 diff | `ops-evo`: `modified_files/` vs `shared/original/`<br>`lingxi-evo`: `round_N/parallel_M/kernel/` vs `shared/kernel/` | 脚本 |
| 决策树 | `world_model_final.json` → `decision_tree` | 脚本 |
| 最优策略分析 | `world_model_final.json` + `implementation_note.txt` | 脚本 |
| 耗时统计 | `.claude/projects/` 会话记录 + `round_*` 目录时间戳 | 脚本（自动用 `round_1` 创建时间锚定正确会话） |
| Token 用量 | `.claude/projects/` 会话记录（assistant 消息的 usage 字段） | 脚本（自动用 `round_1` 创建时间锚定正确会话） |

## 关键实现：`normalize_eval_json()`

子agent 可能保存多种 eval 格式，`normalize_eval_json()` 自动归一化为完整格式：

**1. 扁平格式**（ops-partial 输出）：
```json
{
  "tag": "evolved",
  "time_us": 50.641,
  "precision_passed": true,
  "pipeline": {...}
}
```

**2. Lingxi 格式**（lingxi-partial 输出，有 `comparison` 和 `evolved` 但无 `baseline`）：
```json
{
  "comparison": {"compilation_success": true, "precision_passed": true, "speedup": 1.5},
  "evolved": {"time_us": 40.0, "pipeline": null}
}
```

**3. 完整格式**（已标准化）：
```json
{
  "baseline": {},
  "evolved": {"time_us": 50.641, "precision_passed": true, ...},
  "comparison": {"compilation_success": true, "precision_passed": true, ...}
}
```

**关键规则**：扁平格式中若 `precision_passed=true` 且 `time_us > 0`，则自动推断 `compilation_success=true`。

## 自检机制（`self_check_report`）

脚本生成报告后自动执行以下 15 项检查：

1. **占位符残留**：确保 HTML 中不含 `LLM_FILL`
2. **最优变体有效性**：检查 best variant 的 `compilation_success`、`precision_passed`、`time_us`
3. **全失败异常检测**：若所有变体均标记为编译失败，触发警告（历史上这是扁平格式未归一化的典型症状）
4. **World Model 数据一致性**：比较 world_model score 与 eval 计算的 speedup，偏差 >5% 时警告
5. **测试用例可用性**：确认测试用例参数已从 `call_spec.json` 提取
6. **模型信息可用性**：确认报告包含已知模型名称
7. **副标题质量**：拒绝占位符式副标题（test/report/default 等）
8. **apply-cmd 合理性**：检查 cp 命令数量是否过多（>20）或路径前缀未正确处理
9. **耗时统计合理性**：检查各轮次持续时间是否为正值且不低于 0.5 分钟
10. **Token 统计完整性**：确认资源统计表包含 Cache Creation 列
11. **决策树拓扑验证**：检查所有变体是否有对应树节点，检测孤立节点（无父引用）
12. **代码 diff 新文件泄漏检测**：若 diff 段仅有新增行且含 `/dev/null` 标记，说明新文件错误出现在报告中
13. **apply-cmd 重复目标路径检测**：检查 cp 命令是否存在重复的目标路径
14. **总耗时合理性**：总耗时不应超过各轮次之和的 3 倍（否则包含了空闲等待时间）
15. **无源代码修改检测**：最优变体有有效耗时但 modified_files 为空，说明 ops-partial 未正确保存修改

## 资源消耗统计

报告包含 **资源消耗统计** 部分，自动从会话记录和输出目录中提取：
- **耗时统计**：各阶段（主会话、代理、轮次）的时间范围和持续时间（仅计活跃时间，排除空闲间隔）
- **Token 用量**：各组件（主会话、进化代理、子代理）的 input/output/cache token 消耗

### 耗时计算规则

- `calc_active_duration_ms()` 使用间隔累加法：仅累加 ≤600s 的相邻消息间隔，完全排除大间隔空闲
- 总耗时优先取 evo_agent 持续时间（而非 main_session），避免包含任务结束后的等待时间
- 轮次耗时使用目录 birth time（`stat -c '%W'`）而非 `st_ctime` 或 `st_mtime`

### 数据来源

1. **会话记录文件**：`~/.claude/projects/<project-id>/*.jsonl`
   - 解析 `type=assistant` 消息的 `usage` 字段获取 token 用量
   - 解析 `timestamp` 字段获取时间范围

2. **输出目录时间戳**：`output/<op_name>_{ops-evo|evo}_<timestamp>/round_*/`
   - 从目录的创建/修改时间提取各轮次耗时

## 容错

- 文件缺失：跳过对应章节，不中断报告生成
- evaluation_results.json 格式异常：`normalize_eval_json()` 自动归一化
- diff 生成失败：显示"无基线对比"标签

## 退出码

| 码 | 含义 |
|----|------|
| 0 | 成功（自检可能有警告，但不影响退出码） |
| 1 | output 目录无效 |
| 2 | 关键数据文件缺失 |

## 依赖

Python 3.10+ 标准库，无外部依赖。
