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
python scripts/prepare_bench_data.py --reports-dir benchmark_tasks/top20_transformer_filtered

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
