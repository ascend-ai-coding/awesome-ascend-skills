# 策略 Playbook Schema

**用途**：定义策略的**可执行的分步操作流程**（SOP）。子 agent 采纳策略后必须逐步执行，避免"只改常量不重构代码"的退化行为。

**为什么需要**：策略卡片的"代码骨架"只是抽象示例，与真实算子代码形态差距大，agent 常常不知道怎么对号入座。Playbook 通过强制的分步流程（定位 → 填表 → 形态识别 → 改造 → 验证）把抽象策略转为可执行 SOP。

---

## 文件格式

**位置**：`plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/{ID}_{slug}.md`

命名要求：与对应的 `strategy_cards/{ID}_{slug}.md` 文件名完全一致（利于加载器通过 ID 定位）。

---

## 必备章节（Step 1-6）

每个 Playbook 必须包含以下章节，顺序不能换。

### Step 1: 定位关键结构

用 `grep -n` 命令定位当前算子代码中与策略相关的结构（循环 / 缓冲 / 同步点 / DMA 调用等）。

**交付物**：行号列表（文件 + 行范围），写入 `/tmp/{strategy_id}_locations.txt` 备查。

### Step 2: 改造计划表（强制填写）

子 agent 必须填完此表才能进入 Step 3。表格列出所有要修改的元素的当前值 / 目标值 / 修改位置。

模板：
```markdown
| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| ... | ... | ... | ... |
```

**强制性**：Playbook 的 Step 2 必须明确说"**不填完此表不得进入 Step 3**"。

### Step 3: 代码改造（核心）

此步骤分三个子步骤：

**3A. 形态识别**：列出该策略下的典型代码形态（1 个 canonical + 2-3 个变体 note），子 agent 必须回答"我属于哪种形态"。

**3B. Canonical Template**：一份完整的"改造前 → 改造后"对比代码（≤30 行）。

**3C. Variant Notes**：简短的 "if 代码是 X → adjust Y" 指导（2-3 条），覆盖常见变体。

### Step 4: 约束复核（防崩溃）

策略相关的硬性约束：UB 容量、核数限制、workspace 预算等。提供计算公式，子 agent 必须验算。

### Step 5: 编码并自检（5 条 grep，全部必须过）

**这是防偷懒的核心机制**。列出 3-5 条可执行的 grep 命令，每条都有明确的期望结果。

模板：
```bash
grep -c "PATTERN" FILE  # 期望值：>= N
```

**严格度**：
- 任一检查未过 = 回到 Step 3 重做
- 禁止在 `implementation_note.txt` 中把失败检查标记为"不适用"
- 若确实有合法的另类实现不匹配 grep，必须在 `implementation_note.txt` 中说明具体 grep 项、实际代码、为什么另类实现等价

### Step 6: Known Pitfalls

列出 3-5 条常见失败模式 + 具体修复建议。格式：`{问题现象} → {修复步骤}`。

---

## Playbook 长度

- **目标**：每个 Playbook 120-150 行（含代码块）
- **上限**：200 行（再多就拆分）
- **加载策略**：仅在 `node.strategy_combination` 命中的 Playbook 才加载到子 agent prompt，避免上下文爆炸

---

## 与卡片的关系

| 维度 | 卡片（Card） | Playbook |
|---|---|---|
| 目的 | 让 agent **理解策略意图** | 让 agent **可执行地落地** |
| 长度 | ~40 行 | ~120-150 行 |
| 加载时机 | 筛选阶段（SELECT） | 采纳阶段（EXECUTE） |
| 必须有 | 所有 103 个策略 | 仅 Top 策略（初期 3 个） |
| 正文格式 | 核心思想 / 代码骨架 / 关键修改点 / 常见陷阱 / Grep 关键词 | Step 1-6 |

**重要**：Playbook 和 Card 是**互补**而非替代。命中 Playbook 的策略，子 agent 看到 Card（理解意图）+ Preconditions（确认适用）+ Playbook（可执行 SOP）三者。

---

## 新增 Playbook 的流程

1. 确认策略有足够落地价值（Top 5-20 最常用的）
2. 读源策略文件的 "Overview" / "Variant" / "Pitfalls" 段落
3. 按 Step 1-6 结构起草 Playbook
4. 填入 canonical template（简化源文件的第一个 Variant）+ 2-3 条 variant notes
5. 把"Pitfalls"段落整理成 Step 6 的 `{现象} → {修复}` 格式
6. 提取/设计 Step 5 的 grep 验证命令（每条对应 Step 3 的一个关键修改）
7. 测试：
   ```bash
   python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/load_playbook.py \
       --strategy-ids {ID} --output /tmp/playbook.txt
   cat /tmp/playbook.txt
   ```
8. 提交前确保 Playbook 的 ID 与 Card、Preconditions 的 ID 都对齐

---

## 示例骨架（P1）

```markdown
# P1 Playbook: 双缓冲实操流程

## Step 1: 定位关键结构
```bash
grep -n "BUFFER_NUM\|bufferNum" op_kernel/*.h op_host/*.h
grep -n "for.*Process\|Init.*Buffer" op_kernel/*.cpp
```

## Step 2: 改造计划表（不填完不许进入 Step 3）
| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| BUFFER_NUM | ? | 2 | ?.h:L?? |
| tile_size | ? | 减半为 ? | tiling.cpp:L?? |
| 循环结构 | 串行 3 行 | Prologue+Steady+Epilogue | ... |

## Step 3: 循环结构转换

### 3A. 形态识别
...

### 3B. Canonical Template
```cpp
...
```

### 3C. Variant Notes
- 如果 outer+inner 嵌套：只对 inner 做变换
- 如果 UB 紧张：减半 tile_size 同步
- 如果已有细粒度 PipeBarrier：保留现有同步

## Step 4: UB 容量复核
新 UB = 2 × tile_size × dtype_size × buffer_count

## Step 5: 编码并自检
```bash
grep -c "BUFFER_NUM.*=.*2" foo.h          # >= 1
grep -c "CopyIn.*i\s*+\s*1" foo.cpp       # >= 1
```

## Step 6: Known Pitfalls
- UB overflow → 减半 tile_size
- First iter miscompute → 检查 Prologue 顺序
```

---

## 关联文档

- Preconditions schema：`../strategy_preconditions/SCHEMA.md`
- 卡片 schema：`../strategy_cards/SCHEMA.md`
- 策略贡献指南：`../strategy_cards/CONTRIBUTING.md`
