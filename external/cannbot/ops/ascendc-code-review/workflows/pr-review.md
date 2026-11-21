# PR 检视场景
## 触发
检视 PR、审核 PR、帮我检视这个 PR

核心流程：从用户输入提取 PR 链接（解析 repo_url 与 pr_number）并判断托管平台（仅支持 GitCode）→ 派发 code-fetch 子 Agent 获取 diff 与完整源码（脚本由子 Agent 执行，主 Agent 不直接执行）→ 主 Agent 直接执行检视计划设计（侧别识别 + 文档级分组）→ 单波派发通用检视子 agent（每文件一个）逐条检视 → 行号校对 → 撰写报告 → 可选行内评论。

---

## 编排
### 任务清单

启动时创建 5 个固定任务（全部 pending）：

| 任务 | 阶段 | 内容 |
|------|------|------|
| 任务0 | 获取 diff + 检视计划设计 | 派发 code-fetch 子 Agent 获取 diff 和完整源码 → 主 Agent 直接执行检视计划（侧别识别 + 文档级分组） |
| 任务1 | 逐条检视 | 单波派发通用检视子 agent（每文件一个） |
| 任务2 | 行号校对 | steps/pr-review.line-verify.md |
| 任务3 | 撰写报告 | steps/common.report-write.md |
| 任务4 | 可选行内评论 | 检查 gitcode-toolkit 可用性后执行 steps/pr-review.inline-pr-comment.md |

### 阶段0：获取 diff + 检视计划设计

1. 将任务0 标记为 in_progress
2. Read + 执行 `steps/common.env-check.md`（依赖自检，已装即走）。失败则中止检视
3. 从用户输入中提取 PR 链接，解析出 repo_url 与 pr_number，判断托管平台（仅支持 GitCode，非 GitCode PR 终止）
4. Read + 执行 `steps/pr-review.code-fetch.md` 的派发指令，派发 code-fetch 子 Agent 获取 diff 和完整源码（获取/克隆脚本由子 Agent 执行，主 Agent 不直接执行）
5. 等待 code-fetch 子 Agent 返回（产出 diff_path + repo_path + oversized + 变更文件数 + 变更行数——后两者记录备用，供阶段3 报告头部元信息）
6. **超大 PR 提示**（oversized 由 code-fetch 调 review_mode.py 双高判定，阈值见 `{skill_base}/scripts/workflow.review-thresholds.yaml`）：
   - `oversized=true` → **先向用户输出下方提示，再继续标准流程**（提示不阻断、不等待用户回复，编排不变）
   - `oversized=false` → 直接继续

   **用户可见提示（仅 `oversized=true` 时输出一次）**：

   > 检测到超大 PR（变更文件数与变更行数双高）。**强烈建议先在本地手工拆分**：按算子或子系统边界把它拆成几个小 PR，再分批走文件检视流程（`workflows/file-review.md`）逐个检视——超大 diff 单次全量检视容易因上下文超载而漏检，失败重跑代价也高。本次全量 PR 检视仍将继续；若想改为分批检视，请中断并告知我。
7. Read + 执行 `steps/common.plan-design.md`（主 Agent 直接执行，不派发子 Agent）：做侧别识别 + 文档级检视分组。传入 diff 路径 + repo_path + scope_hint + 检视类型 `pr` + 检视标识 `{pr_number}`。**plan-design 负责调用 `{skill_base}/scripts/workflow.create_review_dir.py --type pr --id {pr_number}` 创建 yaml 输出目录**，返回值含 `yaml_dir` 路径
8. 将任务0 标记为 done

### 阶段1：逐条检视

1. 将任务1 标记为 in_progress
2. **启动 yaml collector 服务**（子 Agent 通过 HTTP 提交 yaml，不接触 yaml 输出目录路径）：
   - 选可用端口：`PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")`
   - 后台启动（分离 stdio 避免阻塞 shell）：`setsid python3 {skill_base}/scripts/workflow.submit_server.py {yaml_dir} {PORT} > /tmp/collector_{PORT}.log 2>&1 < /dev/null &`（保存 PID）
   - 就绪检查（最多重试 2 次）：`sleep 1 && curl -s http://127.0.0.1:{PORT}/health || (sleep 2 && curl -s http://127.0.0.1:{PORT}/health)`；若仍失败，检查 `/tmp/collector_{PORT}.log` 排查原因并尝试重启
3. Read `steps/pr-review.clause-review.md` 获取 prompt 模板
4. 按阶段0 检视计划的分组**单波派发**：在单个消息中并行调用全部组的 `Agent` 工具（每组 = 一个规则文件的全部条款），`subagent_type` 使用 `"general"`
   - 每组用 prompt 模板填入：**{skill_base}（本 skill 根目录绝对路径）** + 侧别 + 条例ID + diff路径 + repo_path + **collector_port（=步骤2 选定的 PORT）**，**不传 yaml_output_dir**
   - 子 Agent 通过 `curl -X POST http://127.0.0.1:{PORT}/submit` 提交 yaml 结果，不再以文本消息返回结果
5. 所有子 Agent 返回后汇总（子 Agent 结果已通过 collector 落盘 yaml，无需收集文本返回值）
6. **完整性校验（按条款清单）**：主 Agent 维护「期望条款清单」= 检视计划中所有已派发组的条例 ID，并记录每个子 Agent 的会话 ID 与其承接的组名对应关系。所有子 Agent 返回后比对：
   - 已落盘清单：`ls {yaml_dir}/ | sed 's/\.yaml$//; s/_dup[0-9]*$//'`
   - 缺失清单 = 期望条款清单 − 已落盘清单
   - 缺失非空 → 按三级补派重试：
     ① **从原子 Agent 捞**：向缺失条款所属的子 Agent（会话 ID 已记录）发消息，要求补交缺失的 yaml（该 Agent 上下文里已有检视过程，通常只是漏提交）
     ② **拼组重派**：捞不回的缺失条款，无论来源文件，合并成一个额外的子 Agent 重跑（该 Agent 按各条例所属规则文件 Read 对应章节，只补缺失条例）
     ③ 仍缺失则标注「该条款结果缺失」，记录失败组名与子 Agent 会话 ID
     落盘文件名为 `{条例ID}.yaml`。校验与补派照常执行但**不再写入报告**（静默保证覆盖完整）
7. **关闭 collector**：`kill {COLLECTOR_PID}`（所有子 Agent 返回后）
8. 将任务1 标记为 done

### 阶段2：行号校对
1. 将任务2 标记为 in_progress
2. Read + 执行 `steps/pr-review.line-verify.md` 薄壳，传入 yaml_dir + diff路径 + repo_path。薄壳内部调用 `{skill_base}/scripts/workflow.line_verify.py` 扫描 yaml 目录、原地修正行号 + diff 红线校验
3. 将任务2 标记为 done

### 阶段3：撰写报告
1. 将任务3 标记为 in_progress
2. Read + 执行 `steps/common.report-write.md` 薄壳：传入 yaml_dir + 报告输出路径 + 头部元信息（检视文件数/代码行数 = code-fetch 返回的变更文件数/变更行数）。薄壳内部调用 `{skill_base}/scripts/workflow.assemble_report.py` 组装报告正文，主 Agent 替换头部占位符。**替换完成后检查报告行数（`wc -l`），若 >2000 行则派发 `steps/common.report-filter.md` 过滤子 Agent 原地精简报告**
3. 报告输出路径 `./operators/pr-{pr_number}/{pr_number}_review_summary.md`
4. 将任务3 标记为 done

### 阶段4：可选行内评论（最后执行）
1. 将任务4 标记为 in_progress，先向用户展示最终报告或报告链接；阶段3 的头部补齐与超长过滤必须已完成。
2. 检查能否找到 `gitcode-toolkit` skill 及其行内评论说明；找不到则跳过本阶段，不读取或执行下一步文件。
3. 仅当可用时，Read + 执行 `steps/pr-review.inline-pr-comment.md`，传入 PR URL + repo_path + diff_path + yaml_dir + 最终报告路径 + 用户对本次 PR 评论的明确要求（保留原话及后续调整，未指定则留空）；检视 head SHA 取已获取源码的 `git rev-parse HEAD`，不得用当前远端 SHA 替代。
4. 发布完成、用户取消或跳过后将任务4 标记为 done；等待问卷回答时保持 in_progress。本阶段不改变已完成的报告。

---

## 与文件检视的关键差异

| 差异点 | 说明 |
|--------|------|
| 阶段0 多一步 code-fetch | 先派发 code-fetch 子 Agent 获取 diff + clone 源码，再主 Agent 直接执行检视计划 |
| 阶段1 传 diff + 完整源码 | 每组额外传入 diff 路径、完整源码路径 |
| 阶段2 走 pr-review.line-verify | 薄壳调 `{skill_base}/scripts/workflow.line_verify.py --diff`，带 diff 红线校验 |
| 报告路径 | `./operators/pr-{pr_number}/` |

## 约束

- 严格按阶段顺序执行，禁止跳步
- plan-design 由主 Agent 直接执行，不派发子 Agent
- code-fetch 返回的变更文件数/变更行数必须传递到阶段3（报告头部元信息 {{FILE_COUNT}}/{{TOTAL_LINES}}）
- 阶段2 走 `steps/pr-review.line-verify.md` 薄壳：脚本内部对 yaml 做 diff 红线校验与行号校对
- PR 检视模式下 code-fetch 失败则终止流程
- `oversized=true` 的拆分建议只提示、不阻断：不等待用户回复、不标记任务为 deleted、不改变后续阶段顺序
- 禁止提前 Read 未执行阶段的 step 文件
