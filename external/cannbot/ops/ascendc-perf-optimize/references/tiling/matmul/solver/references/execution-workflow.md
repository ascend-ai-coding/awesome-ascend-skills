# 执行工作流

本文的全部步骤、门禁和产物要求仅适用于用户已明确要求执行的实测模式，进入条件见 [实测求解器入口](../../solver.md#支持边界)。默认理论分析按 [MatMul 入口](../../index.md) 给出几个可能的优化方向即可，无需执行本文流程。

本文给出唯一推荐顺序：**算子级理论分析 → 当前用例诊断与决策 → 候选切分参数 → 完整真机扫描**。`<SKILL>` 是技能根目录，`<SCRIPTS>` 是其 `scripts/`，`<RUN_DIR>` 是本次单算子运行的独占目录。

## 0. 创建独占产物目录

```text
<RUN_DIR>/
  compatibility/compatibility_report.json
  operator_record/validation.json
  build/package_gate.json
  build/environment.json
  build/worktree_patches/
  baseline/baseline_info_all.json
  anchors/anchors_<case>.json
  decisions/decision_<case>.json
  candidates/candidates_<case>.json
  candidates/passed_<case>.json
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
切分原理与参数影响参考 [matmul-tiling-theory.md](matmul-tiling-theory.md)。
从算子定义、主机侧切分代码、内核源码、算子档案和目标 SoC 中理清：

1. 算子的数学语义、输入输出、数据类型和主要计算量级；
2. 数据从外存到 L2、L1、L0 再到计算和搬出的通路；
3. 主机侧如何选模板、生成切分参数并处理 batch、broadcast、transpose 和尾块；
4. 各内核模板的实现差异、适用条件和不可混用的字段；
5. 哪些字段可调，哪些由主机侧派生，哪些与启动参数绑定；
6. 容量、对齐、并行、流水和模板路由之间的关键约束与耦合关系。

此时只建立后续决策需要的算子机制模型，不估算某个用例的瓶颈，不生成候选，也不把经验判断写成硬约束。这一步不生成独立产物。

## 3. 构建并验收安装包

在首次构建、打桩或切换 vendor 前保存原环境（两种执行模式都需要）：

```bash
python3 <SCRIPTS>/worktree_setup.py prepare-run \
  --op <op_name> --ops-nn-dir <OP_REPO> --run-dir <RUN_DIR>
```

原安装包、主仓 `build_out` 和 `config.ini` 备份到本轮目录，重复准备沿用首次快照。临时工作树使用独立路径和 detached HEAD，用户已有工作树不复用、不改写。备份资源本身为符号链接时，准备阶段会拒绝修改链接目标。运行目录应位于这些资源之外。

强制模板模式：

```python
from worktree_setup import run_build_all
run_build_all("<OP_REPO>", jobs=32, op="<op_name>", run_dir="<RUN_DIR>")
```

调用 `start_build()` 或 `run_build_with_retry()` 时同样传入本轮 `run_dir`，确保构建使用本轮工作树。

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

`start_baseline_batch()` 用各用例事实计算超时：单次 NPU 超时取最大值（毫秒），覆盖整个 CSV 的进程超时与预计时长分别按各用例预算求和（秒）。直接对多行 CSV 调用 `start_baseline()` 时也应传入 `all_params`。

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
coverage = verify_layout_coverage(result_csv, all_params=all_params)
assert not coverage["undeclared"]
assert not coverage["declared_unparseable"]
assert not coverage["kt_undeclared"]
```

`collect_baseline_batch()` 返回 `list[BaselineResult]`，后续取 `baseline = baselines[case_idx]` 传给 `generate_ranking()`。`BaselineResult.tiling_data` 内部是带 `to_dict()` 方法的 `TilingData`；构造 `BaselineResult` 或 `CandidateResult` 时也可传平铺字段字典，引擎会自动转换。

若从基线 JSON 恢复结果，先取对应用例的 `baseline` 节点，再显式构造结果对象；整个 JSON 字典不能直接作为 `baseline` 参数：

```python
from ttk_runner import BaselineResult

# saved 是对应用例的 baseline 节点，含 tiling_data、perf_us 等字段。
baseline = BaselineResult(
    template=saved["template"], perf_us=saved["perf_us"],
    tiling_data=saved["tiling_data"] or {},
    precision_status=saved["precision_status"],
    tiling_parse_error=saved.get("tiling_parse_error") or "",
)
```

程序输出已声明模型可计算的健康度：容量指标沿用现有模型，多核利用率、尾部负载和 K 循环由当前算子的调度适配器推导；未声明模型时明确标记无法估算。单个用例精度失败时保留证据并排除；发现链路错误、全部失败或覆盖门禁失败时停止。

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

`collect_anchor_probe()` 返回的第二项只包含阻断问题。约束适配器的 `warnings` 会打印提示，严格模式也允许继续；需要修正的错误写入 `errors`。

默认严格门禁：缺模板、实际路由不符、布局或核数缺失、约束自检异常都会阻断后续步骤。

若 forced 包仍命中另一 host 策略，先回查兼容性审计和档案的选择层补丁；不能按原候选标签继续扫参，也不能用 `not_applicable` 掩盖已承诺的强制路由失效。

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

## 7. 串行完整扫描

候选筛选完成后，将通过的候选总数、各模板数量和预计扫描时长回填到已有的 `decision_<case>.json`。预算或范围变化需用户确认。扫描全部通过静态筛选的候选，不因中间性能结果提前结束；执行顺序由引擎按模板分组，各模板内部保留候选列表顺序。

```python
from ttk_runner import estimate_scan_eta, start_ttk_scan, check_task_done, collect_ttk_scan, scan_paths

estimated_seconds = estimate_scan_eta(
    params["M"], params["N"], params["K"], len(passed),
    batch_count=params["batch_count"],
)
# 将候选数量、各模板数量与 estimated_seconds 回填到本用例决策。

start_ttk_scan(
    passed, case_csvs[case_idx],
    params["M"], params["N"], params["K"],
    log_tag=tag, case_class=params["case_class"],
    batch_count=params["batch_count"], quant_mode=params["quant_mode"],
    baseline_template=baseline.template,
)
# 按返回的预计时长低频检查 scan_paths(tag)["done"]；确认完成后再收集。
results = collect_ttk_scan(
    passed, log_tag=tag, params=params,
    save_path=f"<RUN_DIR>/results/scan_results_{tag}.json",
)
```

上一个用例收集并落盘后才能开始下一个。用例标签只隔离日志，不解除安装状态、编译缓存和设备的串行约束。路由不一致、无法解码、精度非通过或无性能值的结果不进榜。

## 8. 排名、恢复和终检

```python
from ttk_runner import generate_ranking_block, generate_rankings_md

block = generate_ranking_block(
    case_idx + 1, case_name, params, baseline, results,
    generated, passed_by_template, top_n=10,
)
generate_rankings_md(all_blocks, "<RUN_DIR>/reports/rankings.md")
```

所有候选都慢于基线时推荐基线，不强行给新配置。安装侧注入模式无论成功失败都恢复，并记录恢复证据：

默认恢复与状态查询只处理档案声明的安装侧打桩模式。worktree 模式的调参宏属于构建内容；纯 worktree 档案的 `stub-status` 会提示无需检查。

```bash
python3 <SCRIPTS>/worktree_setup.py restore-stub --op <op_name> --mode <mode>
python3 <SCRIPTS>/worktree_setup.py stub-status --op <op_name>
```

所有后台任务结束后，先保存本轮实际使用的档案四件套，再执行统一收尾；失败或提前停止时也执行相同恢复：

```python
import shutil
from worktree_setup import cleanup_run_environment

try:
    shutil.copytree("<SKILL>/references/profiles/<op_name>",
                    "<RUN_DIR>/operator_record/profile", dirs_exist_ok=True)
finally:
    assert cleanup_run_environment("<RUN_DIR>"), "环境未完全恢复，保留备份并处理失败项"
```

也可使用 `python3 <SCRIPTS>/worktree_setup.py cleanup-run --run-dir <RUN_DIR>`。用单个控制脚本串联流程时，将收尾放在 `finally` 中，并先停止、等待本轮后台任务退出。清理器保存工作树源码补丁后通过 Git 回收本轮实际创建的工作树；原有安装包和主仓构建产物恢复，本轮新增的对应资源删除，最后恢复 `config.ini` 原始内容。成功后删除环境备份，报告和复现材料保留。单项失败保留快照与进度，重试只处理未完成项；`environment.json` 的 `cleaned` 为真才表示恢复成功。

运行异常时保留日志、已完成候选的结果和失败原因，并执行恢复；未完成全量扫描时明确记录未扫描项，不宣称调优完成。

交付报告必须串起完整因果链：相关算子机制、当前用例事实、调优决策、候选覆盖、实测结果和恢复状态。交付前逐项确认必要产物存在且可读、本轮修改已恢复且临时工作树已回收；否则只能报告未完成及原因。仅切回 `custom_normal_nn` 或将宏置零，不代表已恢复用户原环境。无首次快照的旧运行不执行自动删除。
