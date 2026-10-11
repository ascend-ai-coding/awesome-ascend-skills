# TileLang 文件检视场景

## 触发

检视代码、审核代码、检查规范、代码审查、帮我检视指定文件。

---

## 编排

### 任务清单

启动时创建 3 个固定任务（全部标记为 `pending`）：

| 任务 | 阶段 | 内容 |
|---|---|---|
| 任务0 | 代码概要 + 检视计划设计 | 先执行 code-summarize，再执行 plan-design |
| 任务1 | 格式检查 + 逐条检视 | 格式检查与首批条例并行；任务完成并确认结果后滚动补位 |
| 任务2 | 撰写报告 | 执行 `steps/common.report-write.md` |

### 输入解析

从用户输入提取待检视文件，统一为 `file_input`。支持单个文件路径、多个文件路径，以及通过目录枚举得到的文件列表。Python、TileLang DSL 和 Markdown 均可作为输入。

只处理用户输入能够确定的检查对象，不自行扩展文件集合。

### 阶段0：代码概要 + 检视计划设计

1. 将任务0标记为 `in_progress`。
2. 从 `file_input` 提取 `operator_name`，作为本 workflow 的检视标识。依照 `instructions.md` 的“产物与资源路径”规则，在切换 shell 目录或派发子 Agent 前固定绝对路径 `review_output_dir`；用户指定相对输出目录时，以调用时的用户工作目录解析。同时根据本 `instructions.md` 的实际位置确定绝对路径 `skill_base`，仅用于访问 skill 资源。
3. 使用 `spawn_agent` 派发代码概要子 Agent，读取并执行 `steps/file-review.code-summarize.md`。传入 `file_input` 和概要输出路径 `{review_output_dir}/code_summary.md`；返回文件类型、代码侧别、概要路径和跨文件关系。Python和TileLang代码的侧别按实际输入识别为 Kernel、Host或混合；Markdown的侧别为 `N/A`。
4. 等待代码概要子 Agent 完成，收集文件类型、代码侧别、代码概要路径和跨文件关系。
5. 读取并执行 `steps/common.plan-design.md`，单独派发计划设计子 Agent，产出通用分组、波次、预期条例集合和跳过清单。传入：
   - `file_input`
   - 文件类型
   - 代码侧别
   - 代码概要路径
   - `scope_hint`
   - 检视类型 `file`
   - 空的检视标识（文件检视没有 PR 号）
   - 阶段1可用子 Agent槽位数
   - `skill_base`（计划设计子 Agent 须调用 skill 脚本并读取 references）

   `plan-design` 负责调用 `scripts/workflow.create_review_dir.py` 创建 YAML 输出目录，并返回 `yaml_dir`。
6. 将任务0标记为 `done`。

### 阶段1：格式检查 + 逐条检视

1. 将任务1标记为 `in_progress`。
2. 启动 YAML collector 服务。子 Agent 只通过 HTTP 提交 YAML，不接触 `yaml_dir`：
   - collector只监听本机回环地址。若环境设置了 `all_proxy`、`ALL_PROXY`、`http_proxy` 或 `https_proxy`，不得清除或覆盖这些外网代理；应在启动collector前将本地地址加入代理绕过列表：

     ```bash
     export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
     export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
     ```

     健康检查以及所有子 Agent向collector提交YAML时，仍须在 `curl` 中显式使用 `--noproxy 127.0.0.1,localhost`，避免类似 `all_proxy=socks5://172.17.0.1:33334` 的设置劫持本地HTTP请求。该绕过仅适用于collector，不影响Ruff等外部资源下载继续使用现有代理。
   - 在同一次 shell 调用中选端口、启动 collector、立即记录 `$!` 并执行至多两次就绪检查。不要把末尾的 `&` 接在一串 `&&` 命令后面，否则整串命令可能被后台化。启动调用结束后，主 Agent 必须保留打印出的端口和精确 PID；后续 shell 调用不能假设这些变量仍然存在：

     ```bash
     REVIEW_COLLECTOR_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
     setsid python3 "{skill_base}/scripts/workflow.submit_server.py" "{yaml_dir}" "$REVIEW_COLLECTOR_PORT" > "/tmp/collector_${REVIEW_COLLECTOR_PORT}.log" 2>&1 < /dev/null &
     REVIEW_COLLECTOR_PID=$!
     printf 'collector_port=%s collector_pid=%s\n' "$REVIEW_COLLECTOR_PORT" "$REVIEW_COLLECTOR_PID"
     sleep 1
     if ! curl -sS --noproxy 127.0.0.1,localhost --max-time 3 "http://127.0.0.1:${REVIEW_COLLECTOR_PORT}/health"; then
       sleep 2
       if ! curl -sS --noproxy 127.0.0.1,localhost --max-time 3 "http://127.0.0.1:${REVIEW_COLLECTOR_PORT}/health"; then
         sed -n '1,160p' "/tmp/collector_${REVIEW_COLLECTOR_PORT}.log"
         kill "$REVIEW_COLLECTOR_PID" 2>/dev/null || true
         exit 1
       fi
     fi
     ```

   - 两次检查仍失败时，已按精确 PID 关闭本次启动进程；根据本次日志排查可恢复原因后，重新执行启动块并记录新的端口和 PID。不得派发子 Agent 连接未通过健康检查的 collector。
3. 读取 `steps/common.format-check.md` 和 `steps/file-review.clause-review.md`，取得格式检查与逐条检视的执行模板。
4. 保持检视计划中的分组内容和稳定优先级不变，将计划波次按原顺序展平为待派发队列；计划波次只表示初始容量安排，不是文件检视的等待屏障。按完成顺序滚动调度：
   - 以阶段1开始时可供本次检视使用的槽位数确定并发上限 `min(本次检视并发槽位数, 6)`；运行中的格式与条例子 Agent总数不得超过该上限。每次派发前还要确认环境当前确有空闲子 Agent槽位，不能仅凭阶段0的槽位数超额派发。
   - 首批并行派发一个格式检查子 Agent，以及最多 `min(尚未派发的条例分组数, 本次检视并发槽位数 - 1, 5)` 个条例子 Agent；每次派发仍须确认实际空闲槽位。只有一个可用槽位时先单独执行格式检查；没有条例分组时也必须执行格式检查。
   - 格式检查子 Agent读取并执行 `steps/common.format-check.md`，传入 `file_input`、`collector_port` 和 `skill_base`，通过 `/submit?type=format` 提交一份汇总YAML。
   - 首批未容纳的条例分组保留在队列中。一个任务完成并通过第6步对应的结果检查后，若有待派发分组且槽位可用，立即按队列顺序派发下一组，不等待同时运行的其他任务；已运行的低优先级任务不因后续任务完成而中断。
   - 每个格式或条例任务都调用一次 `spawn_agent`，`task_name` 在本次检视中保持唯一；每次派发使用新的子 Agent，已完成的子 Agent 不复用于其他分组。
   - 每个条例子 Agent按照 message 模板接收：分组标识、文件类型、代码侧别、reference文件名、条例原文ID和标题、`file_input`、代码概要路径及 `collector_port`。
   - 禁止向任何子 Agent传递 `yaml_dir`，也禁止子 Agent读取其中已有结果。
   - 条例子 Agent对每条条例独立检视，并通过以下端点提交 YAML，其中 `rule` 使用reference文件名或去掉 `.md` 的文件名：

     ```text
     http://127.0.0.1:{记录的collector_port}/submit?group={group_id}&rule={reference_file}&clause={clause_id}
     ```

     collector 会强制写入 `group_name`、`rule_file`、`canonical_id` 和 `submission_key`。同一身份重复提交时保留已有结果，并生成带 `_dup1`、`_dup2` 后缀的新文件。

5. 每个任务完成并检查结果后输出已完成/总数和当前运行数。队列派发完且所有运行中任务均返回后汇总完成情况；具体结果以collector已落盘的YAML为准，不从子 Agent文本回复中重复收集详情。
6. 每个子 Agent返回后，先核对该任务的预期结果，再释放槽位并补派下一组：
   - 以该条例分组及其条例ID构造预期 YAML 集合，不使用“YAML 总数等于 Agent 数量”作为完整性依据；检查该组每条预期条例的结果是否存在。
   - 格式任务完成时确认至少存在一份可解析且顶层 `type: format` 的汇总YAML。`ERROR` 或 `PARTIAL` 是有效格式结果，不因状态重派。
   - 对确认缺失的条例分组或格式结果，将对应完整任务优先用空出的槽位自动补派一次；补派任务也使用新子 Agent和唯一 `task_name`，同一分组不得同时运行两份任务。补派后仍缺失则停止派发新分组，等待已运行任务返回，关闭本次collector并报告缺失项，不得无限重试。
   - 所有任务返回后，再以计划中的完整预期条例集合和格式结果做一次全局完整性检查；任何缺失都按上述一次补派上限处理，全部确认后才关闭collector并进入报告阶段。
7. 全部任务完成并通过全局完整性检查后，使用记录的精确 PID 关闭本次 collector：

   ```bash
   kill {记录的精确COLLECTOR_PID}
   ```

   使用启动调用打印出的精确 PID，并先核实该 PID 对应本次 collector；无论阶段成功、失败还是中断，退出前都必须关闭本次 collector，不得按进程名批量结束其他实例。
8. 确认所有预期条例和格式YAML均已落盘且collector已关闭后，将任务1标记为 `done`。

### 阶段2：撰写报告

1. 将任务2标记为 `in_progress`。
2. 读取并执行 `steps/common.report-write.md`。传入阶段0固定的 `review_output_dir`、`yaml_dir`、报告输出路径，以及文件类型、代码侧别和时间戳等头部元信息。该步骤调用 `scripts/workflow.assemble_report.py` 组装报告正文，由主 Agent 补全剩余头部信息。
3. 报告输出路径为 `{review_output_dir}/{source_file}_review_summary.md`，传给 assembler 时使用该绝对路径；不得从后来切换的 shell 目录、输入仓库或 skill 安装位置重新推导产物位置。
4. 替换头部信息后检查报告行数；若超过5000行，读取并执行 `steps/common.report-filter.md`，按严重程度删除不严重条例发现、压缩格式详情并更新统计。
5. 将任务2标记为 `done`。

---

## 上下文传递链

```text
阶段0 → code-summarize → 文件类型 + 代码侧别 + 概要路径 + 跨文件关系
                              ↓
         plan-design → 检视计划 + yaml_dir
                              ↓
阶段1 → 格式检查子 Agent + 首批条例子 Agent 并行提交 YAML
                              ↓
         后续通用检视子 Agent 按优先级滚动补位并提交 YAML
                              ↓
阶段2 → assemble_report.py 读取 yaml_dir 并组装最终报告
```

## 约束

- 严格按阶段顺序执行，禁止跳步。
- 阶段0的 plan-design 必须等待 code-summarize 返回后再单独派发。
- 本 workflow 不执行 API 预研，也不传递 API 预研报告路径。
- 本 workflow 不统计代码行数，不调用 `workflow.review_mode.py`，也不产生或传递 `mode`、`guidance`。
- 格式检查子 Agent计入任一时刻最多6个和当前可用槽位的并发上限；格式检查只生成结果，不执行格式修复。
- collector 启动后，无论成功、失败还是中断，退出阶段1前都必须关闭记录的精确 PID。
- 禁止提前读取尚未执行阶段的 step 文件。
