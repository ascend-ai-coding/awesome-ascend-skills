# 执行工作流

本文给出唯一推荐顺序：**算子级理论分析 → 当前用例诊断与决策 → 候选切分参数 → 调优方案 → 真机实测**。`<SKILL>` 是技能根目录，`<SCRIPTS>` 是其 `scripts/`，`<RUN_DIR>` 是本次单算子运行的独占目录。

## 0. 创建独占产物目录

```text
<RUN_DIR>/
  compatibility/compatibility_report.json
  operator_record/validation.json
  build/package_gate.json
  baseline/baseline_info_all.json
  anchors/anchors_<case>.json
  decisions/decision_<case>.json
  candidates/candidates_<case>.json
  candidates/passed_<case>.json
  plans/tuning_plan_<case>.json
  results/scan_results_<case>.json
  reports/rankings.md
  reports/final_report.md
```

目录名包含算子名和时间或其他唯一标识。不覆盖旧运行目录，不把两次运行的产物混在一起。多算子任务一次只完成一个算子的整条链路。

## 1. 兼容性审计与算子档案

先按 [operator-guide.md](operator-guide.md) 完成兼容性审计，写入 `compatibility_report.json`。每项只允许 `compatible`、`needs_engine_change`、`not_applicable`，并包含理由和源码位置。任一项需要改引擎时，记录结论并停止本轮调优。

没有有效算子档案时创建结构骨架：

```bash
python3 <SKILL>/scripts/create_profile_scaffold.py \
  --op <op_name> --repo-rel <repo_relative_op_dir> \
  --repo-commit <OP_REPO_HEAD>
```

骨架不携带算子事实，只提供固定结构。按 [operator-guide.md](operator-guide.md) 从当前源码填写四件套，完成必填字段、源码证据和单测后删除 `_scaffold`：

```python
import sys
sys.path.insert(0, "<SCRIPTS>")
from ttk_runner import load_profile

operator_record = load_profile("<op_name>")
```

已有算子档案只有在 `version_anchor.op_repo_commit` 与当前仓一致、相关差异已对账时可复用。验证摘要写入 `<RUN_DIR>/operator_record/validation.json`。

## 2. 分析当前调优算子

这一步分析的对象是算子。它必须发生在读取用例形状和基线性能之前，且不生成单独的理论分析文件。
可以参考[qbmm-guide.md](qbmm-guide.md)
从算子定义、主机侧切分代码、内核源码、算子档案和目标 SoC 中理清：

1. 算子的数学语义、输入输出、数据类型和主要计算量级；
2. 数据从外存到 L2、L1、L0 再到计算和搬出的通路；
3. 主机侧如何选模板、生成切分参数并处理 batch、broadcast、transpose 和尾块；
4. 各内核模板的实现差异、适用条件和不可混用的字段；
5. 哪些字段可调，哪些由主机侧派生，哪些与启动参数绑定；
6. 容量、对齐、并行、流水和模板路由之间的关键约束与耦合关系。

此时只建立后续决策需要的算子机制模型，不估算某个用例的瓶颈，不生成候选，也不把经验判断写成硬约束。这一步不生成独立产物。

## 3. 构建并验收安装包

强制模板模式：

```python
from worktree_setup import run_build_all
run_build_all("<OP_REPO>", jobs=32, op="<op_name>")
```

安装侧注入模式：

```bash
python3 <SCRIPTS>/worktree_setup.py install-stub --op <op_name> --mode <mode>
python3 <SCRIPTS>/worktree_setup.py stub-status --op <op_name>
```

构建返回成功后仍须验证算子发现与注册链路：

```python
from ttk_runner import validate_vendor_package_gate

validate_vendor_package_gate(
    output_path="<RUN_DIR>/build/package_gate.json"
)
```

动态实现文件、注册头或其他关键产物缺失时，本次构建验收失败；修复打包规则后重建，禁止手工向安装目录补文件继续。

## 4. 获取当前用例事实：基线、健康度和锚点

到这一步才解析当前用例：

```python
from ttk_runner import configure_paths, load_profile, validate_case_csv, extract_all_params, split_csv

configure_paths(
    op="<op_name>", op_base="<算子目录>", ttk_dir="<TTK_DIR>",
    run_dir="<RUN_DIR>", case_dir="<CASE_DIR>",
)
load_profile("<op_name>")
assert validate_case_csv(csv_path)
all_params = extract_all_params(csv_path)
case_csvs = split_csv(csv_path)
```

然后批量运行基线，并让程序生成定量健康度：

```python
from ttk_runner import (
    start_baseline_batch, check_task_done, collect_baseline_batch,
    output_baseline_info_batch, get_soc_version, get_aic_num,
    verify_layout_coverage,
)

start_baseline_batch(csv_path, all_params)
# 按返回的预计时长低频检查完成标记
baselines = collect_baseline_batch(csv_path, all_params)
result_csv = csv_path.replace(".csv", "_result.csv")
aic_num = get_aic_num(get_soc_version(result_csv))
output_baseline_info_batch(
    baselines, all_params,
    "<RUN_DIR>/baseline/baseline_info_all.json",
    aic_num=aic_num,
)
coverage = verify_layout_coverage(result_csv)
assert not coverage["undeclared"]
assert not coverage["declared_unparseable"]
assert not coverage["kt_undeclared"]
```

程序输出的健康度包括多核利用率、尾部负载、片上各级缓存利用率、K 循环和复用等定量事实。单个用例精度失败时保留证据并排除；发现链路错误、全部失败或覆盖门禁失败时停止。

随后逐用例获取可注入计算模板的主机侧锚点：

```python
from ttk_runner import (
    injectable_templates, start_anchor_probe, anchor_paths,
    check_task_done, collect_anchor_probe,
)

params = all_params[case_idx]
baseline = baselines[case_idx]
templates = injectable_templates(
    params["case_class"], params["quant_mode"], baseline.template,
)
tag = f"case{case_idx + 1}"
start_anchor_probe(
    case_csvs[case_idx], templates,
    params["M"], params["N"], params["K"],
    log_tag=tag, case_class=params["case_class"],
    batch_count=params["batch_count"], quant_mode=params["quant_mode"],
    baseline_template=baseline.template,
)
anchors, problems = collect_anchor_probe(
    templates, params, log_tag=tag,
    save_path=f"<RUN_DIR>/anchors/anchors_{tag}.json",
)
```

默认严格门禁：缺模板、实际路由不符、布局或核数缺失、约束自检异常都会阻断后续步骤。

## 5. 结合当前用例做诊断和决策

现在才把算子级分析中与当前用例相关的机制，与用例形状、`baseline_info_all.json`、当前锚点和算子约束放在一起。每个用例写 `decision_<case>.json`：

```json
{
  "case_name": "<case>",
  "operator_mechanisms": [
    {"mechanism": "...", "source": "...", "relevance": "..."}
  ],
  "observed": {
    "baseline_template": "...",
    "perf_us": 0,
    "health": {},
    "anchors": []
  },
  "comparison": {
    "mechanism_vs_observed": [],
    "confirmed_facts": [],
    "risk_signals": [],
    "unconfirmed_bottlenecks": []
  },
  "decision": {
    "action": "continue",
    "focus": [],
    "templates": [],
    "candidate_budget": {},
    "excluded_directions": []
  },
  "evidence": []
}
```

决策规则：

- 健康度数值是程序事实；“计算受限”“搬运受限”等是推断，没有性能计数器时写入风险或未确认瓶颈；
- 理论可行但当前模板不可注入的方向必须排除并说明原因；
- 只有源码机制或锚点证明必然非法时，才可标记不可行；
- `action=stop` 时给出明确停止理由，不生成候选；继续时必须先确定重点方向、模板范围和预算。

全部用例决策落盘后，再进入候选生成。

## 6. 生成并筛选候选切分参数

候选只能从已落盘的用例决策生成。每个候选包含计算模板、该模板的完整可注入字段、非空 `reason` 和 `expected`，并能追溯到一个决策方向。至少覆盖：

- 基线或锚点回放；
- 锚点附近的成组扰动；
- 针对风险信号的多字段组合；
- 可行的跨模板迁移；
- 少量容量边界或少见形态探索。

```python
from ttk_runner import load_candidates, filter_candidates, validate_candidate_count

candidates, generated = load_candidates(
    "<RUN_DIR>/candidates/candidates_<tag>.json"
)
passed, passed_by_template, failed = filter_candidates(
    candidates, params, anchors=anchors
)
validate_candidate_count(generated, passed_by_template)
```

把通过项和每个拒绝原因落盘。不得通过改说明文字或死字段制造重复候选；未经用户同意不得降低算子档案声明的候选下限。

## 7. 候选确定后形成调优方案

候选筛选完成后，写 `tuning_plan_<case>.json`，至少包含：

```json
{
  "case_name": "<case>",
  "objective": "...",
  "candidate_groups": [],
  "execution_order": [],
  "baseline_replay": {},
  "estimated_seconds": 0,
  "precision_gate": "PASS",
  "ranking_rule": "只比较精度通过且路由可验证的性能",
  "stop_conditions": [],
  "failure_collection": [],
  "restore_actions": []
}
```

调优方案必须回答：为什么按这个顺序跑、何时停止、失败收集什么、如何恢复。方案形成后才开始真机扫描。

## 8. 按方案串行实测

```python
from ttk_runner import start_ttk_scan, check_task_done, collect_ttk_scan, scan_paths

start_ttk_scan(
    passed, case_csvs[case_idx],
    params["M"], params["N"], params["K"],
    log_tag=tag, case_class=params["case_class"],
    batch_count=params["batch_count"], quant_mode=params["quant_mode"],
    baseline_template=baseline.template,
)
results = collect_ttk_scan(
    passed, log_tag=tag,
    save_path=f"<RUN_DIR>/results/scan_results_{tag}.json",
)
```

上一个用例收集并落盘后才能开始下一个。用例标签只隔离日志，不解除安装状态、编译缓存和设备的串行约束。路由不一致、无法解码、精度非通过或无性能值的结果不进榜。

## 9. 排名、恢复和终检

```python
from ttk_runner import generate_ranking_block, generate_rankings_md

block = generate_ranking_block(
    case_idx + 1, case_name, params, baseline, results,
    generated, passed_by_template, top_n=10,
)
generate_rankings_md(all_blocks, "<RUN_DIR>/reports/rankings.md")
```

所有候选都慢于基线时推荐基线，不强行给新配置。安装侧注入模式无论成功失败都恢复，并记录恢复证据：

```bash
python3 <SCRIPTS>/worktree_setup.py restore-stub --op <op_name> --mode <mode>
python3 <SCRIPTS>/worktree_setup.py stub-status --op <op_name>
```

交付报告必须串起完整因果链：相关算子机制、当前用例事实、调优决策、候选覆盖、调优方案、实测结果和恢复状态。交付前逐项确认必要产物存在且可读、安装侧无打桩残留；否则只能报告未完成及原因。
