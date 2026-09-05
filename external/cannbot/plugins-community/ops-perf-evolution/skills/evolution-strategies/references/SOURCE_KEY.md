# Source Key 命名规范

> 适用范围：`evolution-knowledge` / `evolution-strategies` / `evolution-world-model` 三个 skill
> 版本：v1.0
> 关联设计：[`docs/design/knowledge-strategy-architecture-v3.2.md`](../../../../docs/design/knowledge-strategy-architecture-v3.2.md)

## 1. 为什么需要 source_key

进化系统中所有"知识/策略引用"都必须用结构化 ID 而不是自由文本，这样才能：

- **ledger 可追溯**：`lineage.jsonl` 里的 `source_keys_used` 直接指向具体文件
- **do-not-reread 可去重**：`state.json` 的 `read_keys[]` 用同一套 namespace
- **诊断→检索可机器化**：`bottleneck_to_sources.py` 输出 `candidate_source_keys[]` 而不是策略 ID 列表
- **重命名可检测**：CI 检查每个 source_key 都能解析到文件

## 2. 格式

```
{skill}#{kind}/{id}_{slug}
```

| 段 | 含义 | 示例 |
|---|---|---|
| `skill` | skill 名（即 `ops/` 下的目录名） | `evolution-strategies` |
| `kind` | 资源类型，决定加载器走哪条路径 | `card` / `preconditions` / `playbook` / `family` / `a3` 等 |
| `id_slug` | 该资源在 kind 内的唯一标识，对应文件名（不含 `.md` / `.yaml` 后缀） | `P1_double_buffer` |

完整示例：

```
evolution-strategies#card/P1_double_buffer
evolution-strategies#preconditions/P1
evolution-strategies#playbook/P1_double_buffering
evolution-strategies#family/matmul_guide
evolution-knowledge#a3/optimization_patterns/double_buffering
evolution-knowledge#a3/profiling_reference/optimization_quickref
evolution-world-model#schema
ascendc-dev-knowledge#api_reference_docs/DataCopy
```

## 3. 各 skill 的 kind 集合

### evolution-strategies

| kind | 路径 | 示例文件 |
|---|---|---|
| `card` | `references/cards/{id}_{slug}.md` | `P1_double_buffer.md` |
| `preconditions` | `references/preconditions/{id}.yaml` | `P1.yaml` |
| `playbook` | `references/playbooks/{id}_{slug}.md` | `P1_double_buffering.md` |
| `family` | `references/families/{family}_guide.md` | `matmul_guide.md` |
| `discovered` | `references/discovered/disc_{Xn}_{slug}.md` | `disc_X1_xyz.md` |

### evolution-knowledge

| kind | 路径 | 示例 |
|---|---|---|
| `a3` | `references/a3/{category}/{name}.md` | `a3/optimization_patterns/double_buffering` |

子类别（category）取自 `references/a3/INDEX.md`：
- `hardware/`、`algorithm_insights/`、`ascendc_api/`、`optimization_patterns/`、`proven_solutions/`、`profiling_reference/`（Phase A7 新增）

### evolution-world-model

| kind | 路径 |
|---|---|
| `schema` | `references/schema.md` |
| `operations` | `references/operations.md` |
| `state_schema` | `references/state_schema.md` |

## 4. 解析规则

source_key 反向解析到文件路径：

```python
def resolve(source_key: str) -> Path:
    skill, rest = source_key.split("#", 1)
    return Path(f"ops/{skill}/references/{rest}")
    # 注意：实际文件可能是 .md / .yaml，由 kind 决定后缀
```

后缀决定规则：
- `preconditions/` 后缀为 `.yaml`
- 其余所有 kind 后缀为 `.md`

## 5. 命名约束

| 段 | 约束 |
|---|---|
| `skill` | 必须存在于 `ops/` 目录 |
| `kind` | 必须是该 skill 已注册的 kind（见 §3） |
| `id_slug` | 仅含 `[a-zA-Z0-9_]`，不含空格、`/`、`#` |
| 整体长度 | ≤ 120 字符 |

## 6. 重命名兜底机制

策略卡或 playbook 重命名后旧 source_key 仍可能被 ledger 引用。维护 `aliases.yaml` 兜底表：

```yaml
# plugins-community/ops-perf-evolution/skills/evolution-strategies/references/aliases.yaml
"evolution-strategies#card/P1_old_name": "evolution-strategies#card/P1_double_buffer"
```

CI 脚本 `tests/strategy/check_source_key_validity.sh` 会：
1. 扫所有 ledger / lineage.jsonl 引用
2. 检查每个 source_key 能否直接解析或通过 aliases 解析
3. 任一失败则 CI 红

## 7. 与 INDEX.json 的关系

每个 skill 的 `references/INDEX.json` 是 source_key 的程序化目录。`INDEX.json` 的 `entries[].source_key` 必须可解析为本规范定义的格式。

INDEX.json 既是 machine-parseable 查询入口，也是 source_key 合法性的"权威清单"——CI 检查源就是 INDEX.json。

## 8. 使用示例

### 在 partial-prompt 中引用

```markdown
## Recommended Strategies

### Primary: P1 Double Buffer
- card: evolution-strategies#card/P1_double_buffer
- playbook: evolution-strategies#playbook/P1_double_buffering  ★ 必须按 SOP 执行
- preconditions_passed: P1.buffer_num_is_one (=1)
```

### 在 ledger 中记录

```jsonl
{"node_id": "n123", "source_keys": ["evolution-strategies#card/P1_double_buffer", "evolution-strategies#playbook/P1_double_buffering"], "speedup": 1.04}
```

### 在 state.json 中作为去重 key

```json
{
  "read_keys": [
    "evolution-strategies#card/P1_double_buffer",
    "evolution-knowledge#a3/optimization_patterns/double_buffering"
  ]
}
```
