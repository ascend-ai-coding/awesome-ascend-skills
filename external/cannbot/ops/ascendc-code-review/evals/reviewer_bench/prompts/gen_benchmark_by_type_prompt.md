# 按类型扩充 benchmark_task 提示词

## 输入
- 检视问题类型：{type}（如"除零保护""指针判空""二进制膨胀"）
- refined 数据：history_real_comment/refined/*.jsonl（4 仓：ops-cv/ops-math/ops-nn/ops-transformer）

## 任务
扫 4 仓 refined，按 {type} 语义匹配每条 comment 的 body + reason，筛出同类评论。

## 筛选规则
- quality=accepted 才纳入
- 同 PR 同行号去重，只留一条
- 每类型取 ≤10 条，优先 reason 明确的
- 命中评论所在 PR 的全部同类评论可同组（一个 PR → 一份 benchmark_task）

## 输出
对每个命中 PR，产出一份 benchmark_task md，格式对齐 benchmark_tasks/top20_redline_and_topk_filtered/ 现有样本：

```markdown
# 代码检视报告

## 检视概览
- **仓库**: {repo}
- **PR编号**: {pr_number}
- **代码文件**: {diff_file}
- **代码侧别**: {Kernel/Tiling/混合，从文件路径判：op_kernel/→Kernel，op_host/→Tiling}
- **检视来源**: 人工检视评论（GitCode PR）
- **总评论数**: {命中条数}
- **检视时间**: {当天日期}

## 发现问题

### 文件: {diff_file}（{侧别}侧）

#### [1] 人工检视意见
- **提出人**: {author}
- **文件**: {diff_file}
- **行号**: {new_line}
- **评论时间**: {created_at 取日期}
- **Commit**: {commit_id}
- **问题描述**:

  > {body}

- **代码片段**（行{new_line}）:

```cpp
  待补（评测不依赖，可省略）
```
```

## 代码片段
评测（run_eval）只取 文件/行号/问题描述，不读代码片段，故代码片段可标"待补"或省略，不影响评测。代码恢复由后续 `prepare_bench_data.py` 管线完成（扫 md → bare repo 恢复评论时代码 + 生成 diff）。

## 约束
- 只产出 md 文件，不修改 refined 原数据
- 文件名：`{序号}_{repo}_pr_{pr_number}_{commit_id前12位}.md`
- 放到 benchmark_tasks/{type}-tasks/ 目录（新建）
- 严格按上述格式，不得自创章节
