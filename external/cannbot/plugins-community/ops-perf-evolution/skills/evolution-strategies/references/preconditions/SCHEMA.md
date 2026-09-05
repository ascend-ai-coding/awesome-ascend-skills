# 策略 Preconditions Schema

**用途**：定义策略的**硬性适用性检查**。子 agent 在投入代码修改前必须逐项执行，任一失败则该策略被跳过。

**为什么需要**：策略卡片描述"做什么"，但不是所有算子都适用。例如"CV 流水预发射"（P14）只对 Cube+Vector 融合算子有意义，对 pure vector 算子（如 FastGELU）强行应用反而有害。Preconditions 把这类适用性判断从 agent 的模糊直觉变为确定性的门控。

---

## 文件格式

**位置**：`plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/{ID}.yaml`（ID 与 frontmatter `id` 一致）

**最小示例**：

```yaml
strategy_id: P1
description: 双缓冲机制的适用性检查
checks:
  - id: has_compute_loop
    type: grep_count
    pattern: "for\\s*\\(|while\\s*\\("
    files: ["kernel"]
    expected: ">= 1"
    fail_msg: "算子无主计算循环，不适用双缓冲（如 Reduce 单次计算）"

  - id: buffer_num_is_one
    type: grep_value
    pattern: "BUFFER_NUM\\s*=\\s*(\\d+)"
    files: ["kernel", "header"]
    expected_value: "1"
    fail_msg: "BUFFER_NUM 已 ≥2，双缓冲已应用。检查是否可以用 P20 三缓冲升级"
```

---

## 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `strategy_id` | string | ✅ | 策略 ID，必须与 frontmatter 一致 |
| `description` | string | ✅ | 一句话说明这个 Preconditions 的用途 |
| `checks` | list | ✅ | 检查项列表（至少 1 条） |

---

## Check 字段

每条 check 必填以下字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✅ | check 的唯一标识（snake_case），用于错误报告 |
| `type` | enum | ✅ | check 类型：`grep_count` / `grep_value` / `profiling_metric` |
| `fail_msg` | string | ✅ | 失败时给子 agent 的引导（应说明为什么不适用 + 建议用什么策略替代） |

其余字段按 `type` 不同而不同。

### type: grep_count

按 grep 命中的行数判断。

| 字段 | 必填 | 说明 |
|---|---|---|
| `pattern` | ✅ | 正则（扩展 grep `-E` 格式） |
| `files` | ✅ | 文件集合关键字列表。枚举：`kernel` / `header` / `tiling` / `all` |
| `expected` | ✅ | 行数断言：`== N` / `>= N` / `> N` / `<= N` / `< N` |

### type: grep_value

按 grep 捕获组的第一个匹配值判断。

| 字段 | 必填 | 说明 |
|---|---|---|
| `pattern` | ✅ | 正则，必须含 `(...)` 捕获组 |
| `files` | ✅ | 同上 |
| `expected_value` | ✅ | 精确匹配的目标值（字符串比较） |

若无匹配，该 check 直接视为失败。

### type: profiling_metric

读取 `baseline_evaluation.json` 的字段做数值断言。

| 字段 | 必填 | 说明 |
|---|---|---|
| `metric` | ✅ | JSON 路径（点分隔），如 `evolved.bottleneck` 或 `baseline.ub_usage_pct` |
| `expected` | ✅ | 断言表达式：`== X` / `>= X` / `> X` / `<= X` / `< X` / `in [a, b, c]` |

若 metric 不存在，check 默认视为通过（避免在缺少 profiling 数据时误杀）。

---

## `files` 关键字的文件集映射

脚本 `check_preconditions.py` 接收子 agent 传入的文件集路径：

| 关键字 | 映射 |
|---|---|
| `kernel` | `shared/original/op_kernel/*.cpp`, `shared/original/op_kernel/*.h` |
| `header` | `shared/original/op_kernel/*.h`, `shared/original/op_host/*.h` |
| `tiling` | `shared/original/op_host/*_tiling.cpp`, `*_tiling.h` |
| `all` | 所有上述文件 |

若 `parent_solution_ref` 非空，检查对象改为父变体的 `modified_files/` 下的同路径文件。

---

## 严格度规则

- **任一 check 失败 → 整个策略被跳过**（agent 不得强行继续）
- **无 profiling 数据** → 仅 `profiling_metric` 类型的 check 视为通过；`grep_count` 和 `grep_value` 正常执行
- **无 Preconditions 文件** → 策略不受限（与当前行为一致）

---

## `fail_msg` 编写规范

失败消息必须对 agent 有引导价值：

✅ 好的写法：
```yaml
fail_msg: "BUFFER_NUM 已 ≥2，双缓冲已应用。检查是否可以用 P20 三缓冲升级"
```

❌ 不够的写法：
```yaml
fail_msg: "不适用"
fail_msg: "check failed"
```

格式建议：`{具体现象} → {建议的替代方向}`。

---

## 示例：P14 CV 预发射的 Preconditions

```yaml
strategy_id: P14
description: CV 流水预发射仅对 Cube+Vector 融合算子有意义
checks:
  - id: has_cube_vector_fusion
    type: grep_count
    pattern: "ASCEND_IS_AIC|ASCEND_IS_AIV"
    files: ["kernel"]
    expected: ">= 2"
    fail_msg: "算子无 Cube+Vector 双核结构（缺 AIC/AIV 分支）。换 P1 双缓冲或 P20 三缓冲（同核内流水）"

  - id: has_syncall_to_remove
    type: grep_count
    pattern: "SyncAll\\(\\)"
    files: ["kernel"]
    expected: ">= 1"
    fail_msg: "算子无 SyncAll 等粗粒度同步（可能已用细粒度 flag）。P14 优化效果有限"
```

---

## 新增 Preconditions 的流程

1. 读源策略文件的 "When to Use" / "Overview" / "Trade-off" 段落
2. 把适用性条件翻译成 grep 模式或 profiling metric
3. 每条 check 写一个 `fail_msg`，包含建议替代策略
4. 在本目录写 `{ID}.yaml`
5. 测试：
   ```bash
   python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/check_preconditions.py \
       --strategy-ids {ID} \
       --kernel-files /path/to/kernel/*.cpp
   ```
6. 提交前确保 Playbook 和 Preconditions 的 ID 一致

---

## 关联文档

- Playbook schema：`../strategy_playbooks/SCHEMA.md`
- 卡片 frontmatter schema：`../strategy_cards/SCHEMA.md`
- 策略贡献指南：`../strategy_cards/CONTRIBUTING.md`
