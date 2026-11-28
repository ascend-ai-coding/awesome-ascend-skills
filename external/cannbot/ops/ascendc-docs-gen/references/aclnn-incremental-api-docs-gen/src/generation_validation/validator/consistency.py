# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""一致性检查器 - 检查生成文档与证据的一致性"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from rich.console import Console
from src.logger import debug
from src.llm.client import LLMClient
from ..validator.evidence_batch import EvidenceBatchManager
from ..dashboard import dashboard_progress, dashboard_log
from .. import config

console = Console()


class ConsistencyChecker:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.batch_manager = EvidenceBatchManager(max_tokens=config.MAX_TOKENS)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算文本的 token 数量"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_chars = len(re.findall(r"[a-zA-Z0-9]", text))
        other_chars = len(text) - chinese_chars - english_chars
        tokens = chinese_chars * 2 + (english_chars + other_chars) // 4
        return tokens

    def check(self, document: str, evidences=None, doc_part: str = "") -> dict:
        results = {
            "total_evidence": 0,
            "consistency_scores": {},
            "accuracy_rate": 1.0,
            "passed": True,
            "issues": [],
        }

        all_evidences = [ev for ev in (evidences or [])]
        results["total_evidence"] = len(all_evidences)

        if not all_evidences:
            results["reason"] = "无证据数据"
            return results

        batches = self._split_into_batches(all_evidences)

        prefix = (
            f"[dim]\\[{config.format_doc_part(doc_part)}][/dim]" if doc_part else ""
        )
        if len(batches) > 1:
            dashboard_log(
                f"  > {prefix} 验证: {len(all_evidences)} 个证据分为 {len(batches)} 个批次"
            )

        with ThreadPoolExecutor(max_workers=config.VALIDATION_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._evaluate_single_batch, document, batch): i
                for i, batch in enumerate(batches)
            }
            batch_results = [None] * len(batches)
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    batch_results[idx] = {"batch_scores": {}, "issues": []}
                    dashboard_log(
                        f"  > {prefix} 批次 {idx + 1}/{len(batches)} 失败: {e}"
                    )
                completed += 1
                dashboard_progress(completed, len(batches))
                if len(batches) > 1:
                    dashboard_log(f"  > {prefix} 批次 {idx + 1}/{len(batches)} 完成")

        all_scores = []
        all_issues = []
        issue_id = 1

        for batch_result in batch_results:
            batch_scores = batch_result.get("batch_scores", {})
            all_scores.append(batch_scores)

            batch_issues = batch_result.get("issues", [])
            for issue in batch_issues:
                issue_copy = issue.copy()
                issue_copy["id"] = issue_id
                all_issues.append(issue_copy)
                issue_id += 1

            cross_source_findings = batch_result.get("cross_source_findings", [])
            for finding in cross_source_findings:
                # 仅将 LLM 明确标记为 definite 的跨源冲突提升为 unresolvable issue；
                # 不能完全确定的差异（definite 缺失/false）不告警，避免误报
                if not finding.get("consistent", True) and finding.get(
                    "definite", False
                ):
                    fact = finding.get("fact", "未知事实")
                    source1 = finding.get("source1", "未知来源1")
                    source2 = finding.get("source2", "未知来源2")
                    unresolvable_issue = {
                        "id": issue_id,
                        "location": f"证据间一致性: {fact}",
                        "reason": f"不同证据来源对同一事实给出不同的值: {source1} vs {source2}",
                        "evidence": f"{source1}; {source2}",
                        "suggestion": "代码修改不完整，无法通过重新生成文档解决",
                        "unresolvable": True,
                    }
                    all_issues.append(unresolvable_issue)
                    issue_id += 1

        if all_scores:
            avg_scores = {
                "factual_accuracy": sum(
                    s.get("factual_accuracy", 0) for s in all_scores
                )
                / len(all_scores),
                "formula_correctness": sum(
                    s.get("formula_correctness", 0) for s in all_scores
                )
                / len(all_scores),
                "numerical_accuracy": sum(
                    s.get("numerical_accuracy", 0) for s in all_scores
                )
                / len(all_scores),
                "description_consistency": sum(
                    s.get("description_consistency", 0) for s in all_scores
                )
                / len(all_scores),
                "logical_coherence": sum(
                    s.get("logical_coherence", 0) for s in all_scores
                )
                / len(all_scores),
            }

            avg_scores = {k: round(v, 2) for k, v in avg_scores.items()}

            overall_accuracy = (
                avg_scores["factual_accuracy"] * 0.3
                + avg_scores["formula_correctness"] * 0.2
                + avg_scores["numerical_accuracy"] * 0.2
                + avg_scores["description_consistency"] * 0.15
                + avg_scores["logical_coherence"] * 0.15
            )

            results["consistency_scores"] = avg_scores
            results["accuracy_rate"] = round(overall_accuracy, 2)
            results["issues"] = all_issues
            results["passed"] = (
                results["accuracy_rate"] >= config.CONSISTENCY_PASS_THRESHOLD
            )

        return results

    def _split_into_batches(self, evidence_list: list[dict]) -> list[list[dict]]:
        """将证据列表按 token 阈值分成多个批次"""
        return self.batch_manager.split_into_batches(evidence_list)

    def _evaluate_single_batch(
        self, document: str, batch_evidences: list[dict]
    ) -> dict:
        """对单个批次的证据进行评估"""
        from pathlib import Path

        evidence_texts = []
        for i, ev in enumerate(batch_evidences):
            content = ev.get("content", "")
            if isinstance(content, dict):
                label = (
                    content.get("file_name", "")
                    or Path(content.get("file_path", "")).name
                    or ev.get("evidence_id", f"ev_{i}")
                )
            else:
                label = ev.get("evidence_id", f"ev_{i}")
            evidence_texts.append(f"【证据 {label}】\n{content}")

        evidence_combined = "\n\n".join(evidence_texts)

        prompt = f"""请作为**昇腾算子文档严格一致性审核专家**，基于提供的**证据内容**评估**文档内容**的准确性与一致性。

## 核心裁决原则
1.  **事实准确性为最高优先级**：
    *   由于审核阈值较高，**任何与代码实现或头文件明确矛盾的事实性描述均视为严重错误**。
    *   **运行时属性（数据类型、芯片型号、算法逻辑）**：**代码实现 > 头文件注释**。若代码不支持某类型，文档声称支持即为**事实错误**。
    *   **静态描述（API 用法、参数含义）**：**头文件注释 > 代码注释/日志**。若文档描述与头文件定义的核心语义冲突，视为**事实错误**。
    *   **校验宏语义（方向极易读反）**：`OP_TILING_CHECK(cond, log, ret)` / `OPS_CHECK` / `CHECK` 的第一个参数是**报错条件**——cond 成立才报错返回。如 `OP_TILING_CHECK(format == FORMAT_FRACTAL_NZ, "xxFormat is invalid")` 表示 FRACTAL_NZ **被禁止**，文档写 ND 是正确的，不得判为错误；反之若文档声称支持 NZ 才是**事实错误**。同理 `CHECK(dtype != DT_FLOAT, ...)` 表示必须为 FLOAT。
    *   **版本与命名错配豁免（共享代码常态，极易误判为"算子名矛盾"）**：昇腾算子仓库常用同一目录的共享代码实现多个 API 版本——证据文件名、结构体名、注册宏、内部函数名中的版本号与文档描述的 API 版本不一致（如 `moe_distribute_combine_v2_tiling.cpp` / `MoeDistributeCombineV2TilingFunc` / `IMPL_OP(MoeDistributeCombineV2)` 实现 `aclnnMoeDistributeCombineV3`）属**共享实现常态，单独不构成事实错误**，禁止据此报"算子名称矛盾/版本不符/名称不一致"。判定基准是**约束语义**（取值范围、数据类型、shape、必传条件等）是否与代码一致，而非名称中的版本号。同理，kernel 内部变量名/形参名与文档参数名不同（如 kernel 的 `epSendCountsOut` vs 文档的 `epRecvCountsOut`）属两套命名体系，不构成参数名错误，仅当语义本身矛盾时才标记。suggestion 中禁止建议将文档中的算子名/参数名改为证据中的旧版本名或 kernel 内部名。**该豁免仅适用于名称层面：取值范围、上下限、边界条件、模式归属等语义写反/写错（如 fullmesh 与 hierarchy 的 BS 上限对调）的判定标准不变，不得因该豁免放宽。**
2.  **严格区分"错误"与"瑕疵"**：
    *   **事实错误（必须标记）**：文档描述与证据**直接矛盾**，会导致用户调用失败、编译错误或逻辑误解。例如：支持类型列表错误、芯片型号张冠李戴、返回值类型错误等。
    *   **非错误（不标记）**：
        *   表述不够优美、冗余或风格不统一。
        *   非关键信息的遗漏（除非遗漏导致严重误导）。
        *   证据本身模糊不清导致的判断困难（此时不标记错误，但评分需酌情降低，若无法判断则保守评分）。
3.  **冲突消解**：
    *   当证据内部存在冲突时，依据"上下文感知的证据优先级"裁决。
    *   若无法确定哪个证据更权威，**不标记为错误**，但在评分时需考虑不确定性。
    *   **重要**：仅当存在**明确且不可调和**的矛盾（差异无法用表述详略、注释滞后、作用域/关注点不同等合理原因解释）时，才在 issues 中标记，并设置 `"unresolvable": true`。这类问题无法通过重新生成解决：
        - 两个同等权威的证据对同一事实给出明确相反的描述（如一个明确支持 INT8，另一个明确排除了 INT8）
        - 部分证据已更新（如头文件中新增了 BF16 数据类型支持），但其他证据仍引用旧值（如参数校验代码中未包含 BF16），导致无法同时满足所有证据，且差异无法用注释/文档滞后解释
        - 证据要求的修改与文档中其他引用不一致，重新生成无法同时满足所有证据要求
        - 不同来源的证据对同一事实给出不同的值（如参数名、数据类型支持列表、返回值等），只要差异明确且不可调和，即使依据优先级规则可以选出"更权威"的一方，也必须标记为 unresolvable。因为这种不一致通常意味着代码修改不完整，无法通过重新生成文档解决
    *   **若差异不能完全确定为真实冲突**（如仅因描述详略不同、注释未同步、不同文件描述粒度不同、或仅个别字段写法不同），**不要标记为 unresolvable**，视为正常证据差异，按优先级规则选用值即可，也不列入 cross_source_findings。

## 文档内容（待检查）
```markdown
{document}
```

## 证据内容（作为评判标准）
{evidence_combined}

## 任务要求

**重要**：cross_source_findings 和 issues 中的每一条描述都必须注明对应的源文件名（如 aclnn_abs.h），让读者能直接定位到具体文件，禁止使用 ev_xxx 等内部代号。

### 第一步：证据间一致性检查（必须首先执行）

在检查文档之前，先对比各证据来源之间的一致性：

1. 提取每条证据中的关键事实（函数签名、参数名、参数类型、数据类型列表、返回值等）。
2. 对比不同证据文件中对同一事实的描述：
   - 例：头文件中的函数签名 vs 实现文件中的函数签名
   - 例：不同文件中的数据类型支持列表
3. 仅当不同证据对同一事实给出**明确且不可调和**的不同值，且差异无法用表述详略、注释滞后、作用域/关注点不同等合理原因解释时，才标记为 unresolvable（并列入 cross_source_findings，`"definite": true`）；即使依据优先级规则可以选出"更权威"的一方，此类明确冲突也必须标记。
   - 原因：这种不一致通常意味着代码修改不完整，无法通过重新生成文档解决。
   - **若差异不能完全确定为真实冲突**（如仅描述详略不同、注释未同步、不同文件关注点不同、仅个别字段写法不同），**不要标记** unresolvable，也不列入 cross_source_findings，按优先级规则选用值即可。

### 第二步：文档与证据一致性检查

1. 识别文档类型与上下文：
    - 若文档涉及芯片架构、数据类型支持、算子兼容性，严格依据代码实现逻辑进行校验。
    - 若文档涉及通用功能描述、API 用法，依据头文件注释和代码逻辑的一致性进行校验。

2. 代码块精确校验：
    - 文档中的每个代码块（函数签名、调用示例、类型声明等），必须在证据中找到对应的源码。
    - 逐一核对代码块中的：函数名、参数名、参数类型、参数顺序、返回值类型。
    - 任何不一致均视为事实错误；**但"版本与命名错配"除外**（见核心裁决原则）：名称中的版本号后缀差异（V2 实现文件 vs V3 API 文档）、kernel 内部变量名与文档参数名的差异，不构成事实错误。

3. 逐项核查：
    - 检查文档中的每一个事实性陈述（如"支持 BFLOAT16"、"适用于 910B"）。
    - 在证据中查找对应的代码逻辑（如 if (arch == DAV_2201) 或 support_list.contains(DT_BF16)）。
    - 特别关注：数据类型支持列表、芯片架构条件、参数约束条件、错误码定义。

4. 标记问题：
    - 仅标记明确的事实性冲突。
    - 不标记：表述风格问题、非关键信息的遗漏、证据本身模糊不清导致的判断困难。

## 评分维度（每项 0.0-1.0）
- factual_accuracy: 事实准确性 - 文档描述是否符合证据中的事实
- formula_correctness: 公式正确性 - 数学公式、计算逻辑是否与证据一致
- numerical_accuracy: 数值正确性 - 参数值、返回值、数据类型等是否准确
- description_consistency: 描述一致性 - 术语、命名、接口名等是否一致
- logical_coherence: 逻辑通顺 - 描述是否有歧义、前后矛盾或逻辑错误

## 输出格式要求
请返回以下 JSON 格式（必须是有效的 JSON，不要有额外文本）：
{{
    "cross_source_findings": [
        {{
            "fact": "对比的事实（如：函数签名、参数名、数据类型列表等）",
            "source1": "证据1 源文件名及其值",
            "source2": "证据2 源文件名及其值",
            "consistent": false,
            "definite": true
        }}
    ],
    "batch_scores": {{
        "factual_accuracy": 0.0-1.0,
        "formula_correctness": 0.0-1.0,
        "numerical_accuracy": 0.0-1.0,
        "description_consistency": 0.0-1.0,
        "logical_coherence": 0.0-1.0
    }},
    "issues": [
        {{
            "id": 1,
            "location": "问题所在的具体位置（如：第X段/第X句）",
            "reason": "错误原因",
            "evidence_id": "引用的证据编号（如 ev_001、ev_001_chunk_1）",
            "evidence": "从该证据中【逐字复制】的原文片段，必须与证据完全一致",
            "suggestion": "修改建议",
            "unresolvable": false
        }}
    ]
}}

**cross_source_findings 填写规则**：
- 若所有证据对同一事实的描述一致，填写空数组 []
- 仅当发现**明确且不可调和**的不一致（差异无法用表述详略、注释滞后等合理原因解释）时，才列出，并设置 `"definite": true`，同时将 corresponding issues 标记为 unresolvable: true
- 若不一致**不能完全确定**为真实冲突（如仅表述详略、注释未同步、关注点不同等），**不列入** cross_source_findings，也不标记 unresolvable

## 注意
- 如果没有事实性错误，issues 为空数组 []。
- 评分标准（保留 2 位小数，如 0.85、0.92）：
    - 1.0：完全符合代码逻辑及头文件描述，无任何事实错误，表述清晰。
    - [0.9,1.0)：核心事实正确，但存在极轻微的非关键性瑕疵（如术语不统一、个别非核心参数描述不够详尽），不影响用户使用。
    - [0.0,0.9)：存在事实性错误或严重遗漏，会导致用户误解或调用失败。
- 禁止模糊推断：reason 和 suggestion 必须基于证据中的具体代码逻辑或头文件描述。如果代码逻辑清晰显示不支持，直接判定为错误。
- **禁止编造证据（严格执行）：**
  - `evidence` 字段必须是从证据中**逐字复制**的原文片段，禁止基于其他证据的模式推断、补全或合成
  - **反例说明**：不得因其他函数签名相似，就推断目标函数的签名也相同，必须以实际证据为准
  - 若无法在证据中找到能直接支撑你判定的原文，`evidence` 字段必须填 `UNVERIFIABLE`，且该问题不计入 issues（直接删除该 issue 条目，不要输出）
  - `evidence_id` 字段必填，必须填写证据的真实编号（如 `ev_001`、`ev_001_chunk_2`），不得凭空编造编号
  - 若你声称"证据中不存在某内容"，必须先在证据中**完整搜索**确认，不得仅凭印象判定；搜索时以目标函数的精确名称为准（如 `asc_abs`），不得用近似名称替代
"""

        debug(f"一致性检查 Prompt：\n{prompt}")
        result = self.llm.chat_with_json([{"role": "user", "content": prompt}])

        if isinstance(result, str):
            debug(f"一致性检查 LLM 返回 str, 尝试提取 JSON: {result[:200]}")
            m = re.search(r"\{[\s\S]*\}", result)
            if m:
                try:
                    result = json.loads(m.group())
                except Exception:
                    debug(f"一致性检查 JSON 提取失败: {m.group()[:200]}")

        if result and isinstance(result, dict):
            return result

        return {
            "batch_scores": {
                "factual_accuracy": 0.0,
                "formula_correctness": 0.0,
                "numerical_accuracy": 0.0,
                "description_consistency": 0.0,
                "logical_coherence": 0.0,
            },
            "issues": [
                {
                    "id": 1,
                    "location": "评估失败",
                    "reason": "LLM 调用失败，无法获取评估结果",
                    "evidence": "无",
                    "suggestion": "请检查 LLM 服务是否正常",
                }
            ],
        }
