# Task D Contract Gate

> **路径约定**：`{skill_base}` = 技能根目录绝对路径，`{output_dir}` = `{op_path}/tests/whitebox`。

Step 3 只检查最终用例 JSON 的覆盖契约：`S2P2_cases.json` 是否完整、合法地覆盖 Task D 声明的 path / tilingkey / param_def entry。

Step 3 不检查：

- `S2P2_gen_cases.py` 的实现细节、随机种子、cap 公式、函数名、seed 生成方式、采样压缩算法。
- `S2P2_traceability.md` 的小节格式、源码行号或写作质量。
- Task A/B/C 的源码事实。
- case 数量是否足够多、case 分布是否均匀、采样值是否最优。
- 第二层 LLM soft review。

## 输入文件

Step 3 只读取以下文件：

- `S2P1_path_list.json`
- `S2P2_param_def.json`
- `S2P2_cases.json`

## 执行命令

```bash
python3 {skill_base}/scripts/s3_task_d_gate.py \
  --output-dir {output_dir}
```

脚本输出：

- `S3_verification_report.json`
- `S3_verification_report.md`

## 检查项

Step 3 只有一个检查项：`D1 cases_path_key_coverage`。

遍历 `S2P2_cases.json` 中每条 case，检查：

1. case 必须是 object。
2. case 必须包含 `path`，且为 string。
3. case 必须包含 `key`，且为 int。
4. case 必须包含 `_group`，且为 string。
5. case 必须包含 dtype 参数字段；字段名来自 `S2P2_param_def.json.dtype_tensors[0].param`。
6. `case.path` 必须存在于 `S2P1_path_list.json.paths[*].id`。
7. `case.key` 必须存在于 `S2P2_param_def.json.tiling_keys`。
8. `(case._group, case.dtype, case.path, case.key)` 必须能回连到 `S2P2_param_def.json.groups[*].per_dtype[*]` 中某个 entry。

## 覆盖率要求

计算以下集合：

```text
reachable_paths = S2P1_path_list.paths[*].id where reachability == "reachable"
param_def_paths = S2P2_param_def.groups[*].per_dtype[*].path
expected_tilingkeys = S2P2_param_def.tiling_keys
param_def_entries = (group_id, dtype, path, key) from S2P2_param_def
case_paths = S2P2_cases[*].path
case_tilingkeys = S2P2_cases[*].key
case_entries = (case._group, case.dtype, case.path, case.key)
```

必须满足：

1. `reachable_paths ⊆ case_paths`，reachable path 覆盖率为 100%。
2. `param_def_paths ⊆ case_paths`，param_def path 覆盖率为 100%。
3. `expected_tilingkeys ⊆ case_tilingkeys`，tilingkey 覆盖率为 100%。
4. `param_def_entries ⊆ case_entries`，param_def entry 覆盖率为 100%。
5. `case_paths ⊆ S2P1_path_list.paths[*].id`，不允许未知 path。
6. `case_tilingkeys ⊆ expected_tilingkeys`，不允许未知 tilingkey。

任一项不满足 → **fail**。

## 报告字段

`S3_verification_report.json` 必须包含：

- `status`: `pass` / `fail`
- `scope`: 固定为 `cases_path_key_coverage`
- `check`: 固定为 `cases_path_key_coverage`
- `checks_total`: 固定为 1
- `checks_pass` / `checks_fail` / `checks_warn`
- `metrics`
- `issues`
- `inputs`
- `limitations`

`metrics` 至少包含：

- `case_count`
- `reachable_path_count`
- `covered_reachable_path_count`
- `reachable_path_coverage`
- `param_def_path_count`
- `covered_param_def_path_count`
- `param_def_path_coverage`
- `expected_tilingkey_count`
- `covered_tilingkey_count`
- `tilingkey_coverage`
- `param_def_entry_count`
- `covered_param_def_entry_count`
- `param_def_entry_coverage`
- `missing_reachable_paths`
- `missing_param_def_paths`
- `unknown_paths`
- `missing_tilingkeys`
- `unknown_tilingkeys`
- `missing_param_def_entries`

## 失败处理

- `status == "fail"` → 回 Step 2 Task D 修正 `S2P2_param_def.json` 或重新生成 `S2P2_cases.json`。
- `status == "pass"` → 进入 Step 4 用户确认。
