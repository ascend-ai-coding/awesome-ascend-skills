# 策略卡片贡献指南

本文档说明**如何新增、修改、维护策略卡片**，供团队成员共同参与策略库的完善。

## 快速概览

```
plugins-community/ops-perf-evolution/skills/evolution-strategies/references/
├── strategies/                    # 完整策略文件（500-1000 行，面向专家）
├── strategy_cards/                # 精简卡片（~40 行，面向子 agent 注入）
│   ├── SCHEMA.md                  # Frontmatter 规范（必读）
│   ├── CONTRIBUTING.md            # 本文档
│   └── {ID}_{slug}.md             # 各策略卡片
├── strategy_index.md              # 人读索引（按瓶颈 / 按算子类型查表）
└── strategy_cards_frontmatter_draft.yaml  # 草稿缓存（可删）
```

## 常见场景速查

| 我要做什么 | 跳到章节 |
|------|------|
| 新增一个策略（从完整文件到卡片） | [场景 A](#场景-a) |
| 修改已有策略的卡片内容 | [场景 B](#场景-b) |
| 修改某个卡片的 frontmatter（瓶颈/算子族等） | [场景 C](#场景-c) |
| 补充策略间的冲突/协同关系 | [场景 D](#场景-d) |
| 删除一个废弃策略 | [场景 E](#场景-e) |
| 验证整个策略库的一致性 | [场景 F](#场景-f) |
| 为高价值策略添加 Playbook + Preconditions | [场景 G](#场景-g) |

---

## 命名规范

**文件名格式**：`{ID}_{snake_case_english_slug}.md`

- `ID`：必须与 frontmatter `id` 一致。类别前缀：
  - `P` = 性能策略（P1-P88）
  - `D` = 数据类型策略（D1-D5）
  - `A` = 精度策略（A1-A8）
  - `X` = 进化发现的新策略（X1+）
- `slug`：**仅小写英文字母 + 数字 + 下划线**，禁用中文/特殊字符/空格
- slug 长度 ≤ 35 字符
- 源：优先用 `plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/` 源文件名的英文部分

**示例**：
- ✅ `P1_double_buffering.md`
- ✅ `P14_cv_pipeline_preload.md`
- ❌ `P1_双缓冲机制.md`（含中文）
- ❌ `P14 Pipeline Preload.md`（含空格）
- ❌ `P14_CV_Pipeline_Preload.md`（大小写混合）

---

## 场景 A：新增一个策略 <a id="场景-a"></a>

### A.1 先写完整策略文件

在 `plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/` 下创建 `{prefix}_{num}_{slug}.md`，格式参考已有文件（含 Overview / Variant / Expert implementation / Trade-off）。

### A.2 生成卡片（自动）

```bash
# 重新运行生成器，增量模式（已有卡片不覆盖）
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/generate_strategy_cards.py
```

### A.3 生成 frontmatter 草稿并审校

```bash
# 从 strategy_index.md + 源文件推导 frontmatter
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/generate_frontmatter_draft.py \
    --output /tmp/new_strategy_draft.yaml

# 编辑草稿：只保留你新增的策略，补充 conflicts_with / synergizes_with
vim /tmp/new_strategy_draft.yaml

# 注入到卡片
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/inject_frontmatter.py \
    --draft /tmp/new_strategy_draft.yaml \
    --only X1
```

### A.4 校验

```bash
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py --validate-all
```

### A.5 更新 strategy_index.md（若需）

若新策略需要被 `--bottleneck` 或 `--op-family` 反查命中，确保在 `strategy_index.md` 的三张表中也加入：
- 主表（策略 ID + Layer + Tags）
- 按瓶颈类型查表（Primary/Secondary 列）
- 按算子类型查表（L0/L1 列）

### A.6 提交

```bash
git add plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/X1_xxx.md
git add plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/X1_xxx.md
git add plugins-community/ops-perf-evolution/skills/evolution-strategies/references/INDEX.json
git commit -m "feat: 新增策略 X1 {name}"
```

---

## 场景 B：修改已有策略的卡片内容 <a id="场景-b"></a>

**直接编辑对应卡片文件**（保留 frontmatter 头部不动）：

```bash
vim plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P14_cv_pipeline_preload.md
# 修改正文（## 核心思想 / ## 代码骨架 / ## 关键修改点 等）
```

**注意**：
- 不要修改 frontmatter 的 `id`（违反 schema 一致性）
- 若修改了核心模式，同步更新源文件 `plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/perf_14_*.md`

---

## 场景 C：修改卡片的 frontmatter <a id="场景-c"></a>

### 方法 1：直接编辑（单个策略修改）

```bash
vim plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P14_cv_pipeline_preload.md
# 修改 frontmatter 字段（--- 和 --- 之间）
# 校验
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py --validate-all
```

### 方法 2：批量修改（多个策略修改）

1. 生成当前草稿
   ```bash
   python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/generate_frontmatter_draft.py \
       --output /tmp/current_draft.yaml
   ```
2. 编辑草稿（只改需要修改的策略条目）
3. 用 `--force` 覆盖注入
   ```bash
   python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/inject_frontmatter.py \
       --draft /tmp/current_draft.yaml \
       --only P14,P18,P53 \
       --force
   ```

---

## 场景 D：补充冲突/协同关系 <a id="场景-d"></a>

### 方式 1：修改脚本 `patch_relations.py` 里的预定义表

打开 `plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/patch_relations.py`，在 `EXTRA_CONFLICTS` 或 `EXTRA_SYNERGIES` 里增加条目：

```python
EXTRA_SYNERGIES = {
    "P14": ["P17", "P29", "P38", "P60", "P80"],
    # 新增：
    "P90": ["P91", "P92"],  # 你的新关系
}
```

### 方式 2：直接编辑卡片 frontmatter

```yaml
---
id: P90
...
conflicts_with: [P91]
synergizes_with: [P92, P93]
---
```

### 运行校验

```bash
# 关系脚本自动对称化（A→B 会补 B→A）
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/patch_relations.py

# 校验所有卡片
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py --validate-all
```

---

## 场景 E：删除废弃策略 <a id="场景-e"></a>

### E.1 删除卡片和源文件

```bash
git rm plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P90_xxx.md
git rm plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/perf_90_xxx.md
```

### E.2 清理引用关系

其他卡片的 `conflicts_with` / `synergizes_with` 若引用了 P90，需要手动移除：

```bash
# 查找所有引用
grep -l "P90" plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/*.md

# 手动编辑每个匹配的文件，从列表中移除 P90
```

### E.3 清理 strategy_index.md 中的引用

```bash
grep -n "P90" plugins-community/ops-perf-evolution/skills/evolution-strategies/references/INDEX.json
# 手动删除对应行
```

### E.4 校验

```bash
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py --validate-all
# 应 0 errors（若有 conflicts_with_unknown_id 或 synergizes_with_unknown_id 错误，说明还有遗漏引用）
```

---

## 场景 F：验证整个策略库的一致性 <a id="场景-f"></a>

```bash
# 1. Schema 校验（必填字段 + 枚举值 + 关系一致性）
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py --validate-all

# 2. 文件名规范校验（dry-run 应该显示 0 to rename）
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/rename_cards_to_canonical.py --dry-run

# 3. 端到端查询测试
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py \
    --bottleneck mte2_stall --limit 5
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/query_strategies.py \
    --op-family flash_attention --limit 5
```

**预期**：
- Schema 校验报告 "0 errors"
- 重命名 dry-run 报告 "0 to rename"
- 查询返回合理的策略 ID 列表

---

## 场景 G：为高价值策略添加 Playbook + Preconditions <a id="场景-g"></a>

**什么时候需要**：策略卡片只能描述"做什么"，无法保证子 agent 真正落地。对于使用频率高、落地价值大的策略（如 P1 / P5 / P14），应补齐 Playbook + Preconditions 填平鸿沟：

- **Preconditions**（`../strategy_preconditions/{ID}.yaml`）：硬性适用性闸门。不满足的策略在主 agent 阶段被拦截，不进入子 agent prompt
- **Playbook**（`../strategy_playbooks/{ID}_{slug}.md`）：分步 SOP。子 agent 采纳后必须按 Step 1-6 逐步执行，附带修改后 grep 自检

### G.1 Preconditions 编写（~30 行 / YAML）

参考 `../strategy_preconditions/SCHEMA.md` 的 schema。一个 check 覆盖一个适用性条件：

```yaml
strategy_id: P1
description: 双缓冲机制的适用性检查
checks:
  - id: has_compute_loop
    type: grep_count
    pattern: "for\\s*\\(.*;.*;.*\\)|while\\s*\\(.*\\)"
    files: [kernel]
    expected: ">= 1"
    fail_msg: "算子无主计算循环 → 双缓冲无从下手。换 D1 混合精度或 P7 数据对齐"
```

**要点**：
- `pattern`：用**双引号**字符串，正则 escape 如 `\\s` `\\(` 必须双反斜杠（YAML parser 会处理 escape 还原为 `\s` `\(`）
- `fail_msg`：格式 `{具体现象} → {建议替代策略}`，避免空泛的"不适用"
- `type` 当前支持：`grep_count` / `grep_value` / `profiling_metric`

### G.2 Playbook 编写（~120-150 行 / Markdown）

参考 `../strategy_playbooks/SCHEMA.md`。必备章节 Step 1-6：

| Step | 内容 |
|---|---|
| Step 1 | 定位关键结构（grep 命令 + 交付物列表） |
| Step 2 | 改造计划表（必须填满才进入 Step 3） |
| Step 3 | 代码改造：3A 形态识别 / 3B canonical template / 3C variant notes |
| Step 4 | 约束复核（UB / workspace / event_id 等） |
| Step 5 | 5 条 grep 自检命令（每条对应一个关键修改） |
| Step 6 | Known pitfalls + 修复建议表格 |

**关键要求**：
- Step 5 每条 grep 应指向 `modified_files/` 下的文件
- Step 3 的 canonical template 应简化自源策略文件的第一个 Variant（≤30 行）
- 禁止超过 200 行

### G.3 端到端测试（每新增一个必须做）

```bash
# 1. 构造一个 mock kernel 目录（模拟策略适用的场景）
mkdir -p /tmp/mock_op/op_kernel /tmp/mock_op/op_host
# ... 写少量代码模拟场景 ...

# 2. 运行 Preconditions（应全过）
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/check_preconditions.py \
    --strategy-ids {NEW_ID} --kernel-dir /tmp/mock_op --summary

# 3. 构造一个不适用场景（例如已经应用过该策略）
# ... 修改 mock kernel 的某处违反 Preconditions ...

# 4. 运行 Preconditions（应拦截 + fail_msg 有意义）
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/check_preconditions.py \
    --strategy-ids {NEW_ID} --kernel-dir /tmp/mock_op --summary

# 5. Playbook 加载
python3 plugins-community/ops-perf-evolution/skills/ops-evaluation/scripts/load_playbook.py \
    --strategy-ids {NEW_ID} --list-only
# 期望看到 loaded: [{NEW_ID}]

# 6. 清理
rm -rf /tmp/mock_op
```

### G.4 常见问题

**Q: Preconditions 应该写多严格？**
- 宁严勿宽。失败 → agent 换用其他策略，比强行应用不适用策略造成劣化更好。
- 但 `fail_msg` 必须给出"替代策略"建议，让 agent 知道怎么办。

**Q: Playbook 的 Canonical Template 与算子差异大时怎么办？**
- 在 Step 3C 变形 Note 中覆盖 2-3 个常见变形。
- 超出变形覆盖的，显式授权 agent 在 `implementation_note.txt` 中说明"属于形态 X，不匹配 3B/3C 模板，基于 Step 3A 结构描述自行推导"。

**Q: 现有 103 张卡片是否都要补 Playbook？**
- 不。只补 Top 使用频率 + Top 落地价值的 5-20 个。其他卡片保持现状（走现有 `ops_partial.md` 阶段 1-2 流程），无 Playbook 不影响功能。

**Q: 新加一个策略需要同时加 Playbook 吗？**
- 不强制。先按场景 A 创建卡片 + frontmatter 即可。后续若发现该策略 agent 落地质量差（只改常量不重构），再补 Playbook。

### G.5 命名对齐

三处必须 ID 一致：
- `strategy_cards/{ID}_{slug}.md`
- `strategy_playbooks/{ID}_{slug}.md`（slug 与 card 一致，利于检索）
- `strategy_preconditions/{ID}.yaml`（无 slug，ID 唯一）

---

## 重要不变量（必须遵守）

| 不变量 | 违反后果 | 保证方式 |
|--------|---------|---------|
| frontmatter 的 `id` 必须与文件名前缀一致 | `query_strategies.py --validate-all` 报错 | rename 脚本保证 |
| `bottlenecks` 所有值来自 SCHEMA.md 的 14 种枚举 | 查询无法匹配 | `--validate-all` 拦截 |
| `op_families` 所有值来自 SCHEMA.md 的 16 种枚举 | 查询无法匹配 | `--validate-all` 拦截 |
| `complexity` 必须是 L0/L1/L2 | 查询无法按复杂度筛选 | `--validate-all` 拦截 |
| `conflicts_with` / `synergizes_with` 引用的 ID 必须存在 | 删除策略后残留引用 | `--validate-all` 拦截 |
| 冲突关系必须对称（A↔B） | query `--exclude-conflicts-of` 不对称 | `patch_relations.py` 自动对称化 |
| 卡片内容必须包含 `## 核心思想` / `## 代码骨架` / `## 关键修改点` / `## 代码搜索关键词` 四个主要段落 | 子 agent 无法按统一流程使用 | 人工审校 |

---

## 脚本参考

### 运行时脚本（agent 持续使用）

| 脚本 | 用途 |
|------|------|
| `query_strategies.py` | 按瓶颈/算子族/复杂度/冲突排除筛选策略（**核心工具**） |
| `extract_strategy_reference.py` | 主 agent 注入卡片到子 agent prompt（自动剥除 frontmatter） |
| `infer_op_family.py` | 从算子名启发式推断算子族 |
| `check_preconditions.py` | 主 agent 筛选策略前执行适用性硬门控 |
| `load_playbook.py` | 主 agent 按 ID 加载 Playbook markdown 注入 prompt |

### 维护脚本（一次性使用）

| 脚本 | 用途 | 何时用 |
|------|------|-------|
| `generate_strategy_cards.py` | 从 `strategies/` 批量生成卡片 | 新增策略、从头重建 |
| `generate_frontmatter_draft.py` | 从 strategy_index.md 推导 frontmatter 草稿 | 新增策略、批量修改 |
| `inject_frontmatter.py` | 将草稿注入到卡片顶部 | 新增/批量修改后 |
| `enhance_frontmatter_from_source.py` | 从源策略文件推断瓶颈关键词 | 草稿中 bottlenecks 为空时 |
| `patch_relations.py` | 补充对称化的冲突/协同关系 | 扩展 EXTRA_CONFLICTS / EXTRA_SYNERGIES 后 |
| `patch_p_series_bottlenecks.py` | 手工审校后的瓶颈批量注入 | 参考此脚本补充新瓶颈 |
| `rename_cards_to_canonical.py` | 规范化文件名（小写英文） | 新增策略/重构后 |

---

## 提交前 checklist

- [ ] 文件名符合 `{ID}_{snake_case_english}.md` 规范（`rename_cards_to_canonical.py --dry-run` 显示 0 to rename）
- [ ] frontmatter 完整且所有枚举值合法（`query_strategies.py --validate-all` 报告 0 errors）
- [ ] 关系字段对称（运行 `patch_relations.py` 自动对称化）
- [ ] 卡片正文含四个必要段落（核心思想/代码骨架/关键修改点/代码搜索关键词）
- [ ] 若新增策略，已在 `strategy_index.md` 的三张表中更新
- [ ] 若删除策略，已清理其他卡片的引用
- [ ] 若新增 Playbook/Preconditions，已过 G.3 的端到端测试
- [ ] 端到端查询测试通过
