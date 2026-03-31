---
name: external-mindstudio-cluster-fast-slow-rank-detector
description: 专门用于 Ascend 集群 Profiling 性能数据的“快慢卡”诊断专家技能。当用户提供【集群性能数据目录/路径】并要求分析【快慢卡】、【慢节点】、【负载不均衡】或【集群瓶颈】时，必须触发此技能。该技能会自动接收集群路径，调度相关工具输出快慢卡的宏观定性与微观根因（如
  Host 下发瓶颈、算子计算劣化）。
original-name: cluster-fast-slow-rank-detector
synced-from: https://github.com/kali20gakki/mindstudio-skills
synced-date: '2026-03-26'
synced-commit: 266c7821de7b51b683d4605960d0d86f7d631e03
license: UNKNOWN
---


# 集群快慢卡诊断

## 1. 技能目标
在 Ascend 多卡/集群场景下，利用 Advisor 工具结合专家规则，自动识别因计算、通信或 Host 下发导致的性能瓶颈卡（慢卡），并下钻定位微观根因。

## 2. 诊断先验知识库 (Expert Rules)
禁止仅凭单项指标字面意思下结论，必须严格遵守以下华为官方诊断逻辑：

* **【Host 下发瓶颈 (伪快卡)】**
    * **现象**：某卡（Rank X）的 `Free Time` 极长（占比 > 10% 或远超均值），且 `Compute` 和 `Communication` 时间异常偏短。
    * **定性**：**Rank X 绝非快卡，而是导致集群阻塞的“慢卡”。** CPU 下发慢导致其 NPU 饿死（产生巨大 Free Time）。当它终于发起通信时，其他卡已等待多时（其他卡 Wait 长），故其通信瞬间完成。
    * **动作**：调用 `scripts/compare_api_stats.py`，重点观察 `launch`、`aclrtSynchronizeDevice` 等下发/同步 API 的耗时与间隙差异。
* **【纯计算快慢卡】**
    * **现象**：各卡 `Free Time` 普遍较短且均匀，但某卡 `Compute Time` 显著大于均值。
    * **定性**：计算型慢卡。若单算子调用次数 (`count`) 不同，为负载切分不均；若次数相同但平均耗时 (`avg_time`) 激增，为算子硬件劣化或动态 Shape 导致。
    * **动作**：调用 `scripts/compare_op_stats.py` 对比算子执行差异。
* **【通信/慢链路瓶颈】**
    * **现象**：各卡通信带宽远低于理论值（如 SDMA < 2GB/s）。
    * **定性**：通常为小包通信（ZeRO3 切分过细）、SDMA 地址未对齐或硬件问题。

## 3. 硬性约束 (MUST DO)

1. **执行已有脚本，严禁造轮子**：微观下钻环节（Step 3）**必须且只能**通过在终端执行 `scripts/` 下的对比脚本获取差异数据。严禁在未执行脚本前自行读取 CSV/DB 进行 Diff 分析。
2. **禁止 Trace 分析**：本技能流程不包含 Timeline 级分析，Step 4 输出报告后立即结束，禁止主动读取 `trace_view.json`。

## 4. 标准操作流程 (SOP)

1. **宏观体检**：强制调用 `msprof-analyze-advisor` 工具输入集群路径，获取总体耗时极差与“慢卡分析”矩阵。
2. **瓶颈定性**：对照【先验知识库】，判定核心瓶颈属于 `Host下发慢`、`计算慢` 还是 `纯通信慢`，并锁定真正的“慢卡 RankID”。
3. **微观下钻**：根据瓶颈类型，在终端直接执行下方对应的对比脚本（计算慢用 OP 脚本，下发慢用 API 脚本）。仅当路径自动发现失败报错时，才可通过补充 `--slow-path` 参数重试。
4. **输出报告**：按以下结构输出最终回复：
   * **诊断结论**：瓶颈类型及真正慢卡的 RankID。
   * **宏观证据**：引用 Advisor 报告中的极差数据（如 Free Time 对比）。
   * **微观根因**：结合脚本输出的 Top 差异数据（特定 API 或算子的耗时比对），解释物理原因。
   * **优化建议**：给出针对性建议（如绑核、切分策略、排查算子Shape差异等）。

## 5. 脚本调用手册

对比脚本统一存放在本技能目录的 `scripts/` 文件夹中，支持自动发现集群目录或手动指定文件（优先 CSV，次选 DB）。

**核心命令模板：**
```bash
# 【计算类瓶颈】调用算子对比脚本（将 <本技能目录> 替换为 get_skill 返回中的路径）
python <本技能目录>/scripts/compare_op_stats.py <集群数据根目录> <慢卡RankID> <快卡RankID> [--top N]

# 【下发类瓶颈】调用 API 对比脚本
python <本技能目录>/scripts/compare_api_stats.py <集群数据根目录> <慢卡RankID> <快卡RankID> [--top N]
```
参数说明：
* cluster_pah: 集群数据根目录（包含 profiler_info_{rank}.json）。 
* slow_rank / fast_rank: 慢卡与快卡（基准）的 Rank ID。 
* --top N: （可选）输出差异最大的前 N 条，默认 20。 
* --slow-path / --fast-path: （可选）当集群自动发现机制报错时，用于手动指定慢/快卡的 *.csv 或 *.db 绝对路径。 
* --json: （可选）以 JSON 格式结构化输出。