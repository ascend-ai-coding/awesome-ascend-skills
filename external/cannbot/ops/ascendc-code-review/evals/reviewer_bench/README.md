# reviewer-bench

AscendC 代码检视评测框架。脚本驱动，非 CLI 包。

## 项目结构

```
scripts/                 评测流水线脚本（见下方）
src/batch_review/        批量并行检视调度器（opencode / claude 双引擎）
prompts/                 检视提示词模板
benchmark_tasks/            ground-truth 评测数据集（真值基准，不可再生）
history_real_comment/    历史真实评论数据
reports/                 输出目录
```

## 数据集分档

`benchmark_tasks/` 下三档，用途与规模各不相同：

| 目录 | 用途 | 实测规模 | 筛选口径 |
|------|------|---------|---------|
| `L1-human-supervised-review-task/` | 高频、易命中的人工检视意见子集 | 9 份任务 / 8 个 PR（#4699 的两个 commit 各一份），均 ops-transformer，每份 3–27 条 | 未在仓内定义（见下方说明） |
| `L2-human-supervised-review-tasks/` | 单 PR 全量人工监督集（当前仅 #4046，非 L1 的超集） | 1 个 PR（ops-transformer #4046），11 条 | 未在仓内定义 |
| `top20_redline_and_topk_filtered/` | 红线 + TOPK 高频问题集 | 16 个 PR，跨 ops-math 9 / ops-nn 2 / ops-transformer 5 | 按问题类型过滤（红线 / TOPK） |
| `ascendc-log-tasks/` | LOG-1 / LOG-2 规则专项集（**每条 GT 含 `关联条款` 字段，唯一支持条款级归因切分的档**） | 10 个 PR / 21 条 GT（LOG-1 ×16 / LOG-2 ×5），跨 ops-math 4 / ops-nn 4 / ops-transformer 2 | **已定义**（见下方「ascendc-log-tasks 筛选口径」） |

**ascendc-log-tasks 筛选口径**（2026-09-17 定义，为首个口径明确定义的档）：

- **收录**：人工检视评论中明确指向 LOG-1 违规（裸/旧式 `OP_LOGE` 应替换为 log.h 预定义宏 `OP_LOGE_FOR_*/WITH_*`）或 LOG-2 违规（日志算子名硬编码/来源不对，应为 `context->GetNodeName()` 或直接传 context）的条目
- **排除**：① 宏间精修类（代码已用预定义宏，属宏选择优化，如 #3775/#2189/#5052 系列）；② 缺日志问题（"return 前没打日志"，#5616/#5546）；③ 作者采纳回复（#1994）；④ AI 自动检视报告产出（#7272）；⑤ 与现行规则冲突的历史意见（#3799 人工建议硬编码算子名字符串，与 LOG-2 直接冲突）；⑥ 归因模糊项（#2625 主诉为错误消息缺实际值）；⑦ 日志文案维度（中文日志 #5910、首字母大写 #6200——对应规则不存在或已取消）
- **数据源**：refined 层优先（含 commit_id），refined 未收录的高价值条目从 raw 层补充（raw 无 commit 字段，文内标"待查"，可经 `gen_proposer_sample.py` 按 PR ref + 评论时间定位补齐；文件名暂不含 commit 哈希段）
- **格式**：沿用现有 GT 结构（run_eval 四正则可解析），每条新增 `关联条款: LOG-1/LOG-2` 字段；代码片段标"待补"（评测仅依赖文件/行号/问题描述）

**档间关系**（同 PR 样本对比，勿按档名推断包含关系）：

- L2 仅含 #4046 一个 PR。在该 PR 上 L1 保留 3 条（报告内编号 `[4] [7] [8]`）、L2 保留全部
  11 条，即**在该 PR 上** L1 是 L2 的子集；L1 的其余 7 个 PR 均不在 L2 内，**两档整体不构成包含关系**。
- L1 与 top20 亦有重叠样本（ops-transformer #4099，commit `b68e9745d63a`，两档各一份）。
  同报告内编号的条目在文件/行号/描述上**并不一致**（如 `[7]` 在两档分别指向
  `infershape.cpp:836` 与 `tiling_arch35.cpp:674`）。跨档比对请以
  `(仓库, PR, commit, 行号)` 为键，**勿以报告内编号为准**。
- `ascendc-log-tasks/` 与既有三档**无 PR 重叠**（已核对）。该档 GT 条目带
  `关联条款` 字段，评测后可按 LOG-1/LOG-2 切分条款级召回；其文件名不含 commit
  哈希段（部分 PR 来自 raw 层无 commit 字段，补查后可改名对齐其余三档命名惯例）。

> ⚠️ **筛选口径未定义**：仓内无这三档的筛选判据定义——`human-supervised` 仅出现在本节的
> 目录名引用中，生成脚本与筛选规则均不在仓内（`gen_ground_truth.py` 产出的是
> `benchmark_tasks/{repo}-ground-truth-reports`，与三档不同源）。上表 L1/L2 的用途描述据数据反推，
> 未经生成方确认。补齐判据前，请勿据目录名推断筛选规则。

数据生成链路见 `scripts/` 下的 `fetch_review_comments.py` → `clean_dataset.py` →
`refine_dataset.py` → `gen_ground_truth.py`（该链路产出 `benchmark_tasks/{repo}-ground-truth-reports`，
与上述三档不同源）。`benchmark_tasks/` 声明为真值基准，**不可再生**。

## scripts 脚本一览

按用途分四组，流水线核心步骤见下方「流水线」。

**数据采集与清洗**（`history_real_comment/` 三层：raw → clean → refined）
- `fetch_review_comments.py` — GitCode API 拉取全部 PR 人工检视评论 → raw jsonl
- `clean_dataset.py` — DeepSeek LLM 语义清洗，每条评论加 quality/reason → clean jsonl
- `refine_dataset.py` — 按 (pr, commit) 分组二筛，组级聚合 → refined jsonl

**真值与代码准备**
- `gen_ground_truth.py` — 从 refined jsonl 生成真值检视报告 md（每条评论 → 一条发现问题）
- `restore_pr_repo.py` — 恢复到 PR 评论发生时的代码状态（bare repo + commit 定位 + amend 检测）
- `gen_proposer_sample.py` — 推断评论提出人 + commit 定位，生成带提出人的样例报告
- `prepare_bench_data.py` — 扫描 GT 报告 → 恢复代码 + 生成 diff + 代码清单 → manifest.json
- `quick_prepare_redline.py` — 红线 PR 测试数据快速准备（绕过 bare repo PR ref）

**配置生成**
- `gen_bench_config.py` — 读 manifest → 组装文件检视 prompt → review-config-bench.json
- `gen_eval_config.py` — 扫描检视报告 + 匹配 GT → 组装评测 prompt → config

**评测核心**
- `deterministic_routing.py` — 确定性条例路由：扫 references `<适用>` 头做声明式匹配 → 路由计划
- `normalize_report.py` — AI 报告归一化：提取 findings（yaml/markdown）+ 行号 re-tracking + 保守过滤
- `run_eval.py` — 端到端评测：4 阶段匹配（path→side→line→semantic）→ 召回率/精确率/F1
- `gen_leaderboard.py` — 聚合 eval 结果 → leaderboard.json

## 流水线

```bash
# 1. 准备数据（耗时）：恢复代码 + 生成 diff → manifest.json
python scripts/prepare_bench_data.py --reports-dir benchmark_tasks/top20_redline_and_topk_filtered

# 2. 组装评测配置（秒级）：读 manifest → review-config-bench.json
python scripts/gen_bench_config.py --manifest manifest.json

# 3. 批量检视：调度多个 Agent 并行跑（详见 src/batch_review/README.md）
PYTHONPATH=src python -m batch_review run --config review-config-bench.json

# 4. 端到端评测：4 阶段匹配 + 归一化
python scripts/run_eval.py \
  --ai-dir reports/e2e_redline \
  --gt-dir benchmark_tasks/top20_redline_and_topk_filtered \
  --diff-dir bench_data_redline \
  --output reports/e2e_redline/eval_results.json

# 5. 生成排行榜
python scripts/gen_leaderboard.py
```

## 扩充 benchmark_tasks

给一个检视问题类型（如"除零保护"），从历史评论捞存量同类问题生成 benchmark_task。提示词产出 md，现成脚本管线接手准备代码。

```bash
# 1. 提示词扫 4 仓 refined，按类型捞同类评论 → 产出 benchmark_task md
#    模板：prompts/gen_benchmark_by_type_prompt.md（把 {type} 替换为目标类型，喂 LLM）
#    读 history_real_comment/refined/*.jsonl，产出 md 到 benchmark_tasks/{type}-tasks/

# 2. 用现成管线准备代码（扫这批 md → 恢复评论时代码 + 生成 diff + 代码清单 → manifest）
python scripts/prepare_bench_data.py --reports-dir benchmark_tasks/{type}-tasks

# 3. 后续同流水线（gen_bench_config → batch_review → run_eval）
```

说明：评测（run_eval）只取 GT 的 文件/行号/问题描述，不依赖代码片段，故提示词产出的 md 代码片段可标"待补"或省略，不影响评测。

## 依赖

```bash
pip install -e .
```

## 文档

- 批量检视调度器：`src/batch_review/README.md`
