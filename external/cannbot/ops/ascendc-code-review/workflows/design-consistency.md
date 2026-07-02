# 设计一致性检视场景

## 触发
设计实现一致性、设计一致性检查、对照设计文档、验证设计实现、设计一致性

## 说明
此场景不检视编码规范条例，只执行设计一致性检查。

---

## 编排

### 任务清单

启动时创建 4 个固定任务（全部 pending，不含条例提取）：

| 任务 | 阶段 | 执行文件 |
|------|------|---------|
| 任务0 | 获取代码 + 概要（含设计映射） | steps/design-consistency.code-summarize.md |
| 任务1 | 跳过条例提取 | — |
| 任务2 | 设计一致性检查 | steps/design-consistency.clause-review.md |
| 任务3 | 行号校对 | steps/common.line-verify.md |
| 任务4 | 撰写报告 | steps/design-consistency.report-write.md |

### 阶段0：获取代码 + 概要（含设计映射）

1. 将任务0 标记为 in_progress
2. 从代码文件路径提取算子名，确认代码文件存在；若 docs_input 为目录，确认目录非空且含 .md 或 .yaml 文件
3. Read + 执行 `steps/design-consistency.code-summarize.md`
4. 传入参数：
   - 代码文件路径
   - 文档输入（docs_input）
   - 概要输出路径 `./operators/{operator_name}/code_summary.md`
5. 执行完毕确认概要末尾包含「设计映射」表
6. 将任务0 标记为 done

### 阶段1：跳过条例提取

将任务1 直接标记为 done。不读取规范文档。

### 阶段2：设计一致性检查

1. 将任务2 标记为 in_progress
2. Read + 执行 `steps/design-consistency.clause-review.md`
3. 传入参数：代码文件路径、文档输入（docs_input）、代码概要路径
4. 收集 S1-S7 判定结果
5. 将任务2 标记为 done

### 阶段3：行号校对

1. 将任务3 标记为 in_progress
2. 传入阶段2 的 ❌ 项列表，Read + 执行 `steps/common.line-verify.md`
3. 传入参数：代码文件路径
4. 将任务3 标记为 done

### 阶段4：撰写报告

1. 将任务4 标记为 in_progress
2. 传入阶段2+3 的结果，Read + 执行 `steps/design-consistency.report-write.md`
3. 传入参数：
   - S1-S7 判定表 + 校对后行号
   - 报告输出路径 `./operators/{operator_name}/{source_file}_design_consistency_review.md`
4. 将任务4 标记为 done

---

## 与文件检视的关键差异

| 差异点 | 说明 |
|--------|------|
| 少一个任务 | 4 个任务，不含条例提取 |
| 阶段0 多传文档输入 | code-summarize 追加设计映射 |
| 阶段1 跳过 | 直接 done |
| 阶段2 不走条例检视 | 走 S1-S7 设计一致性检查 |
| 阶段4 报告格式不同 | 只含设计一致性章节 |

## 约束

- 不读取 `references/` 下的规范文档
- 设计映射表每个判定需来源文档原文引用（标注文件名+章节）
- ❌ 项附具体代码位置和偏差描述
- 行号校对不可跳过
