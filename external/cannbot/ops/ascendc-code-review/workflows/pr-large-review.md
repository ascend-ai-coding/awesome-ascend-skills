# 大型 PR 检视场景

## 触发
由 `workflows/pr-review.md` Stage 0 自动检测（文件数 >10）后跳转进入。不单独暴露给用户。

## 编排

### 任务清单

启动时创建 6 个固定任务（全部 pending，若从 pr-review 跳转则先清理旧任务）：

| 任务 | 阶段 | 执行者 |
|------|------|--------|
| 任务0 | 文件分组 + 预扫描 | file-split（子Agent）→ global-pre-scan（子Agent × N 并行） |
| 任务1 | 摘要 + 分组 + API 预研 | summarize（子Agent × N）∥ clause-grouping（子Agent × 1）∥ api-prestudy（子Agent × 1，仅 Kernel 侧） |
| 任务2 | 负载感知波次检视 | 逐波派发检视子 Agent |
| 任务3 | 共享文件检视 + 综合研判 | shared 检视（子Agent）→ synthesize（主Agent） |
| 任务4 | 合并结果 | merge（主Agent） |
| 任务5 | 行号校验 + 报告 | line-verify → report-write（主Agent） |

### 阶段0：文件分组 + 预扫描

1. 将任务0 标记为 in_progress
2. 若 diff_path 和 repo_path 已由上游传入 → 跳过 code-fetch
3. 主 Agent Read diff 前 200 行，提取变更文件路径列表
4. 派发 **1 个子 Agent** 执行 `steps/pr-large-review.file-split.md`，传入文件路径列表，产出 file_groups
5. 对每个 file_group **并行派发子 Agent** 执行 `steps/pr-large-review.global-pre-scan.md`：
   - 传入：group_file_list + repo_path
   - 产出：该组的 matched_rules（条例级匹配清单）
   - 每波 ≤10 Agent，超过 10 组分批
6. 收集 per-group matched_rules，将任务0 标记为 done

### 阶段1：摘要 + 分组 + API 预研（并行派发）

1. 将任务1 标记为 in_progress
2. 在单个消息中并行派发子 Agent：
   - **summarize × N**：对每个 file_group 派发，Read `steps/pr-large-review.code-summarize.md`，每波 ≤10 Agent
   - **clause-grouping × 1**：派发 1 个子 Agent，Read `steps/pr-large-review.clause-grouping.md`，传入 per-group matched_rules
   - **api-prestudy × 1**（条件派发：仅当 diff 含 `op_kernel/` 路径或代码特征判定为 Kernel/混合侧时）：Read `steps/common.api-prestudy.md`，传入 Kernel 侧文件列表 + 预研报告路径 `./operators/pr-{pr_number}/api_prestudy.md`
3. 收集 per-group summary_path + 全局波次规划表 + API 预研路径（若已派发），将任务1 标记为 done

### 阶段2：负载感知波次逐条检视

1. 将任务2 标记为 in_progress
2. Read `steps/pr-large-review.clause-review.md` 获取 prompt 模板
3. 使用波次规划表逐波派发：每波 ≤10 组，每组 2-3 条例 + ≤5 文件，波内并行波间串行
4. 收集全部结果，将任务2 标记为 done

### 阶段3：共享文件检视 + 综合研判

1. 将任务3 标记为 in_progress
2. 若 shared_bucket 非空，派发 shared 检视（≤1 波）
3. 主 Agent Read + 执行 `steps/pr-large-review.synthesize.md`：跨文件组模式识别、冲突解决、置信度过滤
4. 将任务3 标记为 done

### 阶段4：合并结果

1. 将任务4 标记为 in_progress
2. 主 Agent Read + 执行 `steps/pr-large-review.merge.md`
3. 将任务4 标记为 done

### 阶段5：行号校验 + 报告

1. 将任务5 标记为 in_progress
2. 主 Agent Read + 执行 `steps/pr-review.line-verify.md`（新上下文）
3. 主 Agent Read + 执行 `steps/common.report-write.md`
4. 输出 `./operators/pr-{N}/{N}_review_summary.md`，将任务5 标记为 done

---

## 约束

- 严格按阶段顺序执行，禁止跳步
- code-fetch 失败则终止流程
- 禁止提前 Read 未执行阶段的 step 文件
- 每波 ≤10 Agent，>4 文件组分批
- **主 Agent 只做编排派发**——file-split、global-pre-scan、summarize、clause-grouping 全部由子 Agent 执行
