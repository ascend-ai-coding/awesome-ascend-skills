# Task A Step 3：Kernel Dispatch 读取

本文件对应 `01-code-analyzer.md` 的步骤 3：读取 Scout-K 和 kernel P0 dispatch 块，提取 key 到 kernel class 的映射。

---

## 读取目标

读取 kernel 源码的最终目的是为 `S2P1_path_config.json` 的 `paths.kernel_class`、`orphan_explanations.key_instructions` 提供源码依据。

| 信息 | 提取重点 | 分析深度 |
|------|---------|---------|
| tiling key | dispatch 条目中的 key 值 | 覆盖全部 active key |
| kernel 类名 | 每个 key 的 dispatch 模板类名 | 提取类名和模板参数，不需理解 kernel 实现 |

**LLM 只提 key + 模板类名。** dispatch 行号、`source` 字段组装、orphan 集合运算均交脚本，LLM 不做（见下方规则）。

---

## 读取 Scout-K

读取 Phase 0 产出的 `S2P0_scout_k.md` 和 `S2P0_scout_k.json`。

Scout-K 报告已经过 Scout-Verify 校验，可直接信任其 dispatch 类型、key 值列表、P0 文件清单、arch 守卫状态和行号索引。

`kernel.total_key_count` 是 dispatch 覆盖完整性校验基准。

---

## Kernel P0 读取规则

- 只读取 `S2P0_source_scope.md` 列出的 kernel P0 文件（dispatch 入口）。
- 提取每个 dispatch 条目的 key 值、模板类名、模板参数。
- 不展开 kernel 类内部实现；Task A 只需要 dispatch 到 kernel class 的映射。
- 将提取到的模板类名写入 `paths.kernel_class` 或 `orphan_explanations.key_instructions`。
- **行号交脚本**：`source` 字段和 kernel 行号由 `build_path_list.py` 结合 `S2P0_scout_k.json` 自动组装，LLM 不手写。同一 key 内若存在按属性/尺寸的二次 dispatch，每条分支作为独立路径（其 K 序号在 `04-path-config-schema.md` §路径 ID 命名规则中体现），conditions 需含 kernel 侧区分条件。

---

## Dispatch 覆盖规则（集合运算交脚本）

**LLM 不做集合运算。** orphan 检测由脚本 `build_path_list.py` 依据以下集合关系自动执行：

- `declared_keys`：`paths` 和 `degradations` 中声明的 tiling_key。
- `active_keys`：Scout-K 中所有有效 dispatch key。
- `orphan_keys = active_keys - declared_keys`（脚本计算）。

LLM 的职责仅限于：

- 对脚本可能报出的 orphan key，**提供解释文本**（`orphan_explanations`，见 `04-path-config-schema.md` §orphan_explanations）——即说明该 key 为何 tiling 侧无法触达，并附 kernel 类名。LLM 不亲自做 `active_keys - declared_keys` 的集合检测。
- 若 tiling 公式能产生 key 但 kernel 侧无 dispatch，按 `04-path-config-schema.md` §tiling_no_kernel_keys 声明，不写入 `paths`。
