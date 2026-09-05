# Task D 执行总纲

## 读取规则（强制）

1. 读完本文件后，按「执行顺序约束」表逐文件 Read 并执行。**每完成一个 Step 的状态判断后，才能读取下一个 Step 文件。**
2. 禁止在 Step 1 之前读取 Step 2-5 的子文件，禁止一次性 Read 所有子文件。

## 执行顺序约束（强制）

以下 Steps 必须按编号顺序逐步骤执行，禁止跳步或抢跑。

| Step | 文件 | 前置条件 | 状态判断 |
|------|------|---------|---------|
| Step 1：可达性标注 | `references/task-d/01-step1-merge.md` | 已 Read 完整的 S2P1_path_list.json 和 S2P1_operator_model.json | 每条路径的 reachability 已判定 |
| Step 2：路径分组 | `references/task-d/02-step2-group.md` | Step 1 已完成，仅对 reachable 路径分组 | 每条 reachable 路径已分配 group，顶层 groups 列表已构建 |
| Step 3：参数推导 | `references/task-d/03-step3-derive.md` | Step 2 已完成，groups 已确定 | S2P2_analysis_data.json 已写入，assemble_dim_spec.py 已运行生成 S2P2_dim_spec.json，pick_dims.py 已运行生成 S2P2_param_def_groups.json，format_json.py 已格式化 |
| Step 4：输出 | `references/task-d/04-step4-output.md` | Step 1-3 全部完成 | S2P2_param_def.json 已由 builder 生成，S2P1_path_list.json 已由 update_path_list.py 更新且 6 项校验通过，S2P2_traceability.md 已生成 |
| Step 5：生成 S2P2_gen_cases.py | `references/task-d/05-step5-gen-cases.md` | Step 4 已完成，S2P2_param_def.json 已就绪 | S2P2_gen_cases.py 已从模板复制，python3 执行产出 S2P2_cases.json |

**完成标志**：S2P2_analysis_data.json 已写入，assemble_dim_spec.py 已运行生成 S2P2_dim_spec.json，pick_dims.py 已生成 S2P2_param_def_groups.json 并格式化，build_param_def.py 已运行产出 S2P2_param_def.json，S2P2_cases.json 已写入，6 项校验全部通过

**通用规则**：

- 前置条件表中标明的条件未全部满足时，禁止启动该 Step 的任何操作
- 完成当前 Step 的全部子步骤并确认状态判断满足后，才能进入下一 Step
- 自检失败 → 回到对应 Step 修正，修正完成后方可继续
