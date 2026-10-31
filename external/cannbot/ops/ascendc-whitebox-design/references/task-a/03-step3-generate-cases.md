# Task A Step 3：生成 Route Cases 并执行

## 职责

运行 `S2P1_case_generator.py` 自动生成 route cases，然后执行 `S2P1_route_extract.py` 生成 `S2P1_path_list.json` 和 `S2P1_cases.json`。

Step 3 只运行 Step 2 已在白盒目录中基于模板补全的脚本副本，不重新生成脚本结构。

## 输入文件

| 文件 | 用途 |
|---|---|
| `S2P1_case_generator.py` | case 生成器（Step 2 产出） |
| `S2P1_route_extract.py` | evaluator 脚本（Step 2 产出） |
| `S2P1_route_dispatch_model.md` | 变量域、覆盖目标和分支条件（参考） |

## 输出文件

```text
S2P1_route_cases.json
S2P1_path_list.json
S2P1_cases.json
```

## Case Generator 预期

Case generator 的实现方式由当前算子的 `S2P1_route_dispatch_model.md` 决定，具体组织方式以 Step 2 生成的脚本为准。若脚本输出包含 `branch_group` / `sub_branch`，Step 3 应按主分支与子分支聚合统计和审计覆盖情况。

最终 route cases 必须满足：

- 覆盖 `S2P1_route_dispatch_model.md` 声明的 route 分支和 kernel dispatch 目标。
- `case_values` 只包含 `01` 定义的外部字段，不包含可由这些字段机械推导出的内部变量。
- `case_values` 的字段和取值必须满足 `S2P1_route_dispatch_model.md` 定义的分支条件和合法范围。
- `case_values` 的数值字段不能因固定 witness 实现而大量重复；除非分支约束或合法域不足，generator 必须优先生成非重复合法值。
- `expected` 必须足以验证目标 route 稳定性，不能只验证 `reachable` / `tiling_only` 分类。

生成逻辑：

1. 使用当前 generator 的动态区组织方式生成 route case objects。
2. 调用当前生成器的 case_values 组装入口生成 `case_values`。
3. 回调 evaluator 验证 route：
   - 匹配 → 通过，记录 case。
   - 不匹配 → 返回动态区重新在目标分支约束域内随机采样的 candidate，重试（最多 10 次或当前 generator 定义的更高上限）；不得对同一个已返回 candidate 原样重复验证。
   - 10 次仍失败 → 标记该组合为 `failed`，跳过继续。
4. 回调 `eval_kernel_route` 记录 kernel 可达性。
5. 调用 `build_case_object(...)` 生成 case record（只含 `case_values` 及 witness 字段）。

`S2P1_route_cases.json` 中每条 case 结构：

```json
{
  "id": "RC0",
  "case_values": {},
  "seed_constraints": {},
  "purpose": "...",
  "evidence": []
}
```

`case_values` 保存 `01` 定义的外部字段，不保存可由这些字段机械推导出的内部变量。Task A 只生成 `case_values`，不构造真实 tensor shape，完整算子实例（inputs/outputs/attrs、可选输出、空 tensor、TensorList、shape/rank 覆盖）由 Step 5 Mapper 承接。

### 设计规则

| 规则 | 说明 |
|---|---|
| 产物优先 | 以 route 覆盖、字段边界和合法性为验收，不以固定 seed pool 或 variants 数量为验收 |
| branch-aware | 字段取值必须绑定当前 route branch 合法域 |
| constraint-driven sampling | 生成器应把 Step 1 的分支条件翻译为约束判断代码，并在约束域内随机采样，不得使用固定 witness 表代替约束采样 |
| field-level uniqueness | 对 `case_values` 每个字段全局跟踪已使用值，优先生成未重复的合法取值；仅在合法域或分支约束不足时允许重复 |
| evaluator 兜底 | route 不匹配说明候选值不适合当前分支，应替换、重采样或标记失败 |
| 超限标记跳过 | 10 次重试仍失败标记 failed，不阻断其余生成 |
| 固定种子 | `--seed 42`（默认），保证可复现 |
| 控制数量 | case 组织应覆盖分支范围，避免无意义的全量笛卡尔积 |
| 字段边界 | `case_values` 只包含 `01` 允许的外部字段；接口实例与 shape 派生由 Step 5 Mapper 承接 |

## 执行命令

```bash
python3 S2P1_case_generator.py --operator <op_name> [--seed 42]
python3 S2P1_route_extract.py --operator <op_name> --cases S2P1_route_cases.json
python3 -m json.tool S2P1_path_list.json >/dev/null
python3 -m json.tool S2P1_cases.json >/dev/null
```

## 完成标准

1. case generator 输出 0 个 failed 组合。
2. 已生成 `S2P1_route_cases.json`（`generated_by: "case_generator"`）。
3. 已执行 evaluator 生成 `S2P1_path_list.json`。
4. evaluator 附带生成 `S2P1_cases.json`（flat array，每条含 path_id + case_values）。
5. JSON 语法和脚本内置 schema 校验通过（不读取 schema 文件）。
6. 每个声明的 route / mode 目标至少有 1 条 case 覆盖，且必要时可按主分支/子分支审计覆盖。
7. `S2P1_cases.json` 的字段和取值满足 `S2P1_route_dispatch_model.md` 声明的字段边界、分支条件和合法范围。
8. 每个 kernel catalog branch 有 witness 或标记 orphan。
9. `kernel_orphans` 和 `unresolved` 已 review。
