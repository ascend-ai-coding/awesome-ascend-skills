# 经验库预检 — 流程指南

> **何时**：Stage 5 写 impl 之前（由 Coder 执行，不属 Stage 1-4）。
> **做什么**：从经验库提取与该算子相关的已知陷阱，生成 checklist 写入 `MEMORY.md` → `## Experience Preflight`。
> **谁消费**：Coder（只关注无自动兜底的规则）、自动扫描（机器解析 checklist 格式 + AST 扫描代码）。
> **自动扫描独立运行**：检查 `[-]` 消除 + AST scan（impl 阶段，stage 5/6）。

---

## 1. 注入固定规则

读取 [fixed_rules.md](fixed_rules.md) 中 F1-F8（跨算子高频反模式，≥2 个算子踩过坑）。

这 8 条**无条件注入**每轮预检输出，全部标记 `[x]`。自动化标记：F1/F2/F4/F8 标记 `🤖`（AST 扫描），F6 标记 `🔧`（其他自动扫描），F3/F5/F7 无标记（Coder 人工检查）。

---

## 2. 提取算子特征

从 **SPEC.md + task_desc.py + API_REPORT.md** 提取 3 个维度：

| 维度 | 提取来源 | 用途 |
|------|---------|------|
| **操作集** | task_desc.py forward() + API_REPORT.md pypto.* 调用 | §3 搜索入口 |
| **dtype 路径** | SPEC.md 输入输出 dtype + API_REPORT.md 精度路由 | cast 约束匹配 |
| **结构特征** | 输出数量、动态轴、loop/loop_unroll、算法类别 | 模式约束匹配 |

操作集必须以实际 API 调用为准，不能仅凭文档声明。

---

## 3. 扫描经验库

对操作集中每个 API 名，按 **3a → 3b → 3c** 顺序搜索。每路统一结构：**搜索 → 提取 → 与已有结果去重 → 保留或丢弃**。

### 全局规则

- **N/A 过滤**：规则涉及的 API 不在本算子操作集中 → 不输出。
- **与固定清单去重**：与 F1-F8 语义等价的规则 → 不重复输出。
- **S2/S3 项**：不输出。
- **同一 API 的规则合并为一条**，不论来源。

### 3a. experience_classified（主规则集）

**source**：`skills/pypto-op-knowledge/references/experience_classified/` 下 8 个 `.md` 文件。

**extract**：API 驱动——对操作集中每个 API 名搜索。命中即提取解决方案，转为禁止/必须项。

**output**：
- S0 项（有自动扫描兜底）：直接标记 `[x]` + `🤖`。
- S0 项（无自动扫描兜底，且不在 F1-F8 中）：标记 `[-]` 待验证，上限 **3 条**。
- S1 项：上限 **5 条**，仅保留该算子特有的精度陷阱。

### 3b. API 文档 + DEBUG_GUIDEBOOK（补缺口，去重 3a）

**source**：
- 操作集中每个 API 对应 `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/operation/pypto-{name}.md`
- `skills/pypto-general-debug/references/DEBUG_GUIDEBOOK.md` fingerprint 索引 → 命中的 leaf 文件

| fingerprint | leaf 文件 |
|-------------|----------|
| `@pypto.frontend.jit` | jit-signature.md |
| `pypto.view` / `pypto.assemble` | pypto-view.md |
| `pypto.matmul` / `pypto.sum` | matmul.md |
| 动态轴 / `pypto.loop` | dynamic-shapes.md |
| tile 配置 | tile-shapes.md |
| Python 运算符在 JIT 内 | python-operators.md |
| `pypto.cumsum` / scan | scan-and-reduction.md |
| NPU launch 失败 | npu-launch-failures.md |

**extract**：API 文档仅提取参数硬约束；DEBUG_GUIDEBOOK 仅提取 `## Issue:` 标题 + 一句话方案。

**output**：上限 **5 条**。API 文档仅保留有自动扫描兜底的 S0 项；DEBUG_GUIDEBOOK 允许无自动扫描兜底的 S0 项（标记 `[-]`）。

### 3c. 测试仓库（补数值）

**source**：仅对前两层中缺数值的规则，搜 `python/tests/st/` 和 `tests/ops/`（仓库根相对路径），精读 ≤5 个文件。

**extract**：只提取 tile shapes 数字、atol/rtol 数值。

**output**：写入 `### References` 区（非 checklist 条目，不参与 checklist 解析）。

---

## 4. 输出 checklist

写入 `MEMORY.md` → `## Experience Preflight`。

### 约束

- **总条数上限 20 条**（含 F1-F8）。超出时裁剪：S0（有自动扫描兜底）> S1 > S0（无自动扫描兜底）。
- 每行必须是 `- [x]` / `- [-]` / `- [ ]` 开头（按正则 `^\s*-\s+\[([x\- ])\]` 解析）。
- **禁止表格**（无法承载 `[-]` 的子注释，且无法被机器解析）。
- `[-]` 条目下方必须紧跟 `> ⚠️ 待验证：{具体待确认项}`。
- N/A 规则直接删除，不写 "N/A" 条目。

### 标记语义

| 标记 | 含义 | 示例 |
|------|------|------|
| `- [x]` | 规则适用，已确认 | `- [x] [S0] pypto.cast: INT8→FP32 禁止直转 🤖` |
| `- [-]` | 不可判定，Coder 在 Stage 5 确认 | `- [-] [S0] pypto.view: 3D→2D tail axis 对齐` |
| `- [ ]` | 规则与 API 需求冲突，需规避方案 | `- [ ] [S0] pypto.X: 本算子需要...` |

**自动化标记后缀**：

| 标记 | 含义 | Coder 动作 |
|------|------|-----------|
| `🤖` | AST 自动扫描（F1/F2/F4/F8 + §3 中有 AST 兜底的条目） | 跳过，`submit_for_verify` 时自动拦截 |
| `🔧` | 其他自动扫描（如 F6） | 跳过，门禁时自动拦截 |
| 无标记 | 无自动兜底 | Coder 必须人工检查 |

### `[-]` 消除时机

Stage 5 编码时，Coder 确定具体参数后：
- 确认合规 → 改为 `- [x]`
- 确认风险后接受 → 将 `> ⚠️ 待验证` 改为 `> ✅ 已知风险，接受`

`submit_for_verify` 门禁会检查残留 `[-]` 项，有残留则不通过。

### 输出模板

```markdown
## Experience Preflight

操作集：{api_list} | 总计 {N} 条（固定 8 + 扫描 {M}）

- [x] [S0] `pypto.div/mul/add/sub`: 首参必须是 Tensor，不能是 Python 标量 (F1) 🤖
- [x] [S0] `pypto.mul/add/sub/div`: 禁止显式构造 `pypto.Element()` 作为参数 (F2) 🤖
- [x] [S0] 跨调用链 rank 变化点必须紧邻 `set_vec_tile_shapes(target_rank)` (F3)
- [x] [S0] `pypto.cast`: A2A3 cast 路径受限，INT8/UINT8/INT4 必经 FP16 (F4) 🤖
- [x] [S0] `pypto.rms_norm`: epsilon 必须与 golden 严格一致，参数名是 `epsilon` (F5)
- [x] [S0] wrapper 签名 init_params 必须放 `**kwargs`，不能用 `*` 分隔符 (F6) 🔧
- [x] [S0] `pypto.loop`: 禁止 SSA 重赋值 `x = f(x)`，用 persistent buffer + `[:]` (F7)
- [x] [S0] `pypto.zeros/ones`: dtype 必须用关键字参数 `dtype=` 传入 (F8) 🤖
- [-] [S0] `pypto.matmul`: out_dtype 是第 3 个位置参数，trans 是 kwargs（来源：matmul.md）
  > ⚠️ 待验证：本算子 matmul 调用是否使用了正确的参数顺序
- [-] [S0] `pypto.view`: 3D→2D 时 tail axis 必须 16 对齐（来源：vector.md）
  > ⚠️ 待验证：本算子 view 是否涉及 3D→2D 降维
- [x] [S1] `pypto.matmul`: n_tiles 边界精度问题（来源：matmul.md）

### References

tile baseline: cube [128,128], vec 按 normal rule
atol/rtol: 1e-3 (from tests/st/test_{op}.py)
```

### 模板要点

1. **所有条目统一 `- [x]`/`[-]`/`[ ]` 格式**——F1-F8 也标 `[x]`，不存在"无条件输出但不标 checkbox"的例外。
2. **按 API 分组排列**——同一 API 的规则紧邻，不按来源分区。
3. **`🤖`/`🔧` 标记 = 自动扫描已覆盖**——Coder 跳过有标记的规则，只人工检查无标记的条目。
4. **References 独立于 checklist**——tile shapes、atol/rtol 等参考值放 `### References` 子节，**不使用 `-` 前缀**（避免被 checklist 正则误匹配为条目）。
5. **无"行动区/规则区"分区**——`[-]` 本身就是"需要行动"，不需要额外分区。
6. **每条一行规则描述 + 来源标注**——`(F1)` 或 `（来源：{文件}）`，便于溯源。
