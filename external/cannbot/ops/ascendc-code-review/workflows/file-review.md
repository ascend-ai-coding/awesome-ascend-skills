# 文件检视场景

## 触发
检视代码、审核代码、检查规范、代码审查、帮我检视 xxx

---

## 编排

### 任务清单

启动时创建 4 个固定任务（全部 pending）：

| 任务 | 阶段 | 内容 |
|------|------|------|
| 任务0 | 代码概要 + 条例分组 + API 预研 | 并行派发 code-summarize + clause-routing + api-prestudy（仅 Kernel 侧） |
| 任务1 | 逐条检视 | 按波次派发检视子 Agent |
| 任务2 | 行号校对 | steps/common.line-verify.md |
| 任务3 | 撰写报告 | steps/common.report-write.md |

### 输入解析

从用户输入提取代码文件（支持：单文件路径、多文件路径、目录路径 find 枚举），统一为 `file_input`（可以是单个路径，也可以是多个路径）。

### 阶段0：代码概要 + 条例分组 + API 预研（并行）

1. 将任务0 标记为 in_progress
2. 从 file_input 提取算子名，确认文件存在
3. **在单个消息中并行派发子 Agent**（A 和 B 总是派发，C 仅当侧别包含 Kernel 时派发）：

**子 Agent A — 代码概要**：
```
Read + 执行 steps/file-review.code-summarize.md 的派发指令
传入：file_input + 概要输出路径 ./operators/{operator_name}/code_summary.md
```

**子 Agent B — 条例分组**：
```
Read + 执行 steps/common.clause-routing.md 的派发指令
传入：file_input + 用户意图范围（如用户指定了检视范围，如"只检查数值安全"，传入对应的类别名；否则传空表示全量）
```

**子 Agent C — API 文档预研**（条件派发：仅当 file_input 含 `op_kernel/` 路径或代码特征判定为 Kernel/混合侧时）：
```
Read + 执行 steps/common.api-prestudy.md 的派发指令
传入：file_input（仅 Kernel 侧文件）+ 预研报告输出路径 ./operators/{operator_name}/api_prestudy.md
```

4. 等待所有子 Agent 返回，收集：
   - 子 Agent A → 侧别 + 概要路径
   - 子 Agent B → 分组规划表（波次、每组条例ID列表）
   - 子 Agent C → API 预研报告路径（若已派发）
5. **侧别回填**：若子 Agent C 未派发（纯 Tiling 侧），跳过 API 预研路径
6. 将任务0 标记为 done

### 阶段1：逐条检视

1. 将任务1 标记为 in_progress
2. Read `steps/file-review.clause-review.md` 获取 prompt 模板
3. 按阶段0 的分组规划表，逐波派发：
   - 每波在单个消息中并行调用 ≤10 个 `Agent` 工具
   - `subagent_type` 使用 `"general"`
   - 每组用 prompt 模板填入：侧别 + 条例ID和标题 + file_input + 代码概要路径 + API 预研路径（若存在）
   - 波次内并行，波次间串行
4. 每波完成后输出进度，所有波次完成后汇总
5. 将任务1 标记为 done

### 阶段2：行号校对

1. 将任务2 标记为 in_progress
2. 传入阶段1 的 FAIL/SUSPICIOUS 列表，Read + 执行 `steps/common.line-verify.md`
3. 将任务2 标记为 done

### 阶段3：撰写报告

1. 将任务3 标记为 in_progress
2. 传入阶段1+2 的结果，Read + 执行 `steps/common.report-write.md`
3. 报告输出路径 `./operators/{operator_name}/{source_file}_review_summary.md`
4. 将任务3 标记为 done

---

## 上下文传递链

```
                 ┌─ code-summarize → 侧别 + 概要路径 + 跨文件关系
阶段0（并行） ───┤─ clause-routing → 分组规划表（含文件范围）
                 └─ api-prestudy → API 预研报告路径（仅 Kernel 侧）
                         ↓
阶段1 → 逐条结果 (PASS/FAIL/SUSPICIOUS)
         ↓
阶段2 → 校对后的 FAIL/SUSPICIOUS
         ↓
阶段3 → 报告文件
```

## 约束

- 严格按阶段顺序执行，禁止跳步
- 阶段0 的子 Agent 必须在单个消息中并行派发（A + B 总是，C 仅 Kernel 侧）
- 禁止提前 Read 未执行阶段的 step 文件
