# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Evidence extractor - extract structured data from evidences by doc_part"""

import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.logger import log_result

from .. import config
from ..core.models import (
    DocPart,
    RawEvidence,
    RawEvidenceChainInput,
)
from ..dashboard import (
    dashboard_log,
    get_current_doc_part,
    get_dashboard,
    set_dashboard_context,
)
from ..evidence.chip_mapper import ChipMapper
from ..evidence.ranker import TARGET_PREFERENCE
from ..evidence.support_verifier import SupportVerifier

console = Console()


class EvidenceExtractor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._console_lock = threading.Lock()

    @staticmethod
    def _deduplicate_anchors_by_file(anchors: list) -> list:
        file_paths = set()
        for anchor in anchors:
            anchor_info = anchor.anchor_information or {}
            if anchor_info.get("type") == "file":
                fp = anchor_info.get("level1", {}).get("file_path")
                if fp:
                    file_paths.add(fp)
        filtered = []
        for anchor in anchors:
            anchor_info = anchor.anchor_information or {}
            anchor_type = anchor_info.get("type", "unknown")
            fp = anchor_info.get("level1", {}).get("file_path")
            if fp and fp in file_paths and anchor_type != "file":
                continue
            filtered.append(anchor)
        return filtered

    @staticmethod
    def extract_raw_evidences(raw_input: RawEvidenceChainInput, deduplicate: bool = False) -> list[RawEvidence]:
        """Extract raw evidences from anchors"""
        evidences = []

        anchors = raw_input.anchor_tuple
        if deduplicate:
            anchors = EvidenceExtractor._deduplicate_anchors_by_file(anchors)

        for anchor in anchors:
            anchor_info = anchor.anchor_information or {}
            anchor_type = anchor_info.get("type", "unknown")
            dep_class = anchor_info.get("dep_class", "original")
            for ev in anchor.evidence:
                content = (ev.evidence_information or {}).copy()
                evidences.append(
                    RawEvidence(
                        evidence_id=ev.get_evidence_id(),
                        source=anchor_type,
                        content=content,
                        gene_rule=ev.gene_rule,
                        support_chips=ev.support_chips,
                        dep_class=dep_class,
                    )
                )

        return evidences

    @staticmethod
    def _merge_and_deduplicate(list1: list, list2: list) -> list:
        """Merge two lists and deduplicate by evidence_id"""
        seen = set()
        result = []
        for item in list1 + list2:
            refs = item.get("evidence_id", [])
            ref_key = tuple(sorted(refs)) if isinstance(refs, list) else tuple(refs) if refs else ()
            if ref_key not in seen:
                seen.add(ref_key)
                result.append(item)
        return result

    @staticmethod
    def _prepare_evidences_content(ev_list: list[dict]) -> list[dict]:
        """Prepare evidences content by filtering and adding source_file"""
        from pathlib import Path

        contents = []
        for ev in ev_list:
            evidence_id = ev.get("evidence_id", "")
            content = ev.get("content", {})
            ev_type = ev.get("type", "")
            dep_class = ev.get("dep_class", "original")

            if isinstance(content, dict):
                source_file = content.get("file_name", "") or Path(content.get("file_path", "")).name or evidence_id
                if ev_type == "function":
                    content = {k: v for k, v in content.items() if k not in ["call", "callee", "return"]}
                    if "content" in content and isinstance(content["content"], list):
                        # fmt: off
                        content["content"] = [
                            {k: v for k, v in item.items() if k != "full_function"}
                            for item in content["content"]
                        ]
                        # fmt: on
                elif ev_type in ("note", "section"):
                    content = {k: v for k, v in content.items() if k not in ["start_line", "end_line", "token_hash"]}
                content["source_file"] = source_file
                if dep_class == "repo_other":
                    content["_source_tag"] = "外部依赖（仓内公共文件/其他算子引用）"
                contents.append(content)
            else:
                contents.append({"source_file": evidence_id, "content": content})
        return contents

    @staticmethod
    def _get_target_category(target: str) -> str:
        """Get category for target from ranker config"""
        return TARGET_PREFERENCE.get(target, {}).get("category", "fact")

    @staticmethod
    def _group_evidences_by_confidence(evidences_by_target: dict, modified_file_basenames: set = None) -> dict:
        """Group evidences by confidence level.

        Modified files get priority layer (1.0) to ensure they're extracted first.
        """
        from pathlib import Path

        grouped = {}
        for target_key, ev_list in evidences_by_target.items():
            groups = {1.0: [], 0.9: [], 0.8: [], 0.7: [], 0.6: [], 0.5: []}
            for ev in ev_list:
                conf = ev.get("confidence", 0.5)

                # Check if this evidence is from a modified file
                is_modified = False
                if modified_file_basenames:
                    file_path = ""
                    content = ev.get("content", {})
                    if isinstance(content, dict):
                        file_path = content.get("file_path", "")
                    elif isinstance(content, list) and content:
                        if isinstance(content[0], dict):
                            file_path = content[0].get("file_path", "")

                    if file_path and Path(file_path).name in modified_file_basenames:
                        is_modified = True

                if is_modified:
                    groups[1.0].append(ev)
                elif conf >= 0.85:
                    groups[0.9].append(ev)
                elif conf >= 0.75:
                    groups[0.8].append(ev)
                elif conf >= 0.65:
                    groups[0.7].append(ev)
                elif conf >= 0.55:
                    groups[0.6].append(ev)
                else:
                    groups[0.5].append(ev)
            grouped[target_key] = [
                groups[1.0],
                groups[0.9],
                groups[0.8],
                groups[0.7],
                groups[0.6],
                groups[0.5],
            ]
        return grouped

    @staticmethod
    def _add_metadata_to_extraction(extraction: dict, category: str, confidence: float) -> dict:
        """Add category, confidence and source to each item in extraction"""
        for _, items in extraction.items():
            for item in items:
                if isinstance(item, dict):
                    item["category"] = category
                    item["confidence"] = confidence
        return extraction

    @staticmethod
    def _base_progressive_prompt(
        task_content: str,
        content: str,
        doc_original: str,
        category_instruction: str,
        return_fields: str,
    ) -> str:
        """Base prompt template for progressive extraction"""
        repo_other_section = ""
        if "_source_tag" in content:
            repo_other_section = """
# 证据来源说明
标记为"_source_tag": "外部依赖（仓内公共文件/其他算子引用）"的证据来自算子引用的仓内公共文件或其他算子代码，非算子自身代码。
这些文件定义了算子依赖的公共类型、宏定义、共享约束等。
提取时请结合算子自身代码判断这些外部依赖对文档内容的影响。
"""
        return f"""# 任务
从以下证据中提取新版资料书写所需内容，并对自己现有证据的解析结果进行自评打分。

# Category说明
{category_instruction}

# 资料书写所需内容
{task_content}

# 现有证据
{content}
{repo_other_section}
# 旧版文档信息（仅做参考）
{doc_original}

# 打分依据
自评时请从以下维度打分：
- completeness: 字段覆盖完整性，是否提取了所有必要字段
- accuracy: 内容准确度，提取的内容是否准确反映证据信息
- confidence: 置信度，对当前结果的自信程度
- 严重遗漏情况最高分不超过 0.8

# 注意
1. extraction的内容必须严格来自`# 现有证据`，禁止修改证据。
2. extraction中的"其他"字段仅存放资料书写所需的必须内容。
3. 仅当`# 现有证据`中不同代码文件对同一事实给出**明确且不可调和**的不同值（如头文件与实现文件对同一函数给出不同签名、
A 文件明确支持某类型而 B 文件明确排除），且差异无法用表述详略、注释滞后、作用域/关注点不同等合理原因解释时，
才在 source_conflicts 中报告（并设置 `"definite": true`）；即使你已依据优先级选择了其中一个值，此类明确冲突仍须报告。
4. 若差异**不能完全确定**为真实冲突（如仅因描述详略不同、注释未同步、不同文件关注点不同、或仅个别字段写法差异），
**不要**报告到 source_conflicts，视为正常证据差异，按优先级选用值即可。
source_conflicts 仅关注`# 现有证据`中不同代码文件之间的不一致，`# 旧版文档信息`仅供参考，不作为冲突检测的来源。
5. source_conflicts 的 values 中每个值必须附带来源文件名，格式为"值 (文件名)"，如 "out_new (aclnn_abs.h)"。
6. **否定式校验不得反转（注意宏参数顺序）**：昇腾校验宏 `OP_TILING_CHECK(cond, log, ret)` / `OPS_CHECK` / `CHECK` 的第一个参数是**报错条件**——cond 成立才报错返回。
因此 `OP_TILING_CHECK(format == FORMAT_FRACTAL_NZ, "xxFormat is invalid")` 表示 FRACTAL_NZ **被禁止**（支持的是 ND 等其他格式）；
`OP_TILING_CHECK(dtype != DT_FLOAT, ...)` 表示数据类型**必须**为 FLOAT。绝不能把报错条件里的取值写成该字段的支持取值；
某字段在证据中没有明确校验时，保持`# 旧版文档信息`中的原值，不得臆测
7. **参数可选性与默认值**：代码中仅当参数非空才校验（如 `if (param != nullptr) {...}`）或未传入时按默认值处理
（如未传入时按 1.0 处理/跳过该逻辑）的参数，须提取为可选参数并在"使用说明"中写明**默认行为**
（如"可选参数，传空指针表示不使用，未传入时默认为1.0"）
8. **参数归属**：只提取目标算子接口（`aclnnXxx` 与 `aclnnXxxGetWorkspaceSize` 的函数签名）实际包含的参数；
多个算子变体共享的文件（如共用 tiling/公共校验文件）中出现的**其他变体的参数**（本接口签名中不存在的参数，如 RMSNorm 融合系参数）
不得提取为本文档的参数行；字段取值也不得从其他参数类比推测——证据中没有该参数的明确校验时保持旧版文档原值

请返回JSON格式：
{{
  "extraction": {{
    {return_fields}
  }},
  "self_score": {{
    "score": 0.0-1.0,
    "reason": "一句话自评理由"
  }},
  "source_conflicts": [
    {{
      "fact": "冲突的事实（如：函数签名参数名）",
      "values": ["来源1的值", "来源2的值"],
      "source_files": ["aclnn_abs.h", "aclnn_abs.cpp"],
      "definite": true
    }}
  ]
}}
"""

    def extract_by_doc_part(
        self,
        doc_part: str,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        modified_file_basenames: set = None,
    ) -> tuple[dict, list]:
        """Extract structured data based on doc_part and target groups.
        Returns (extracted_dict, source_conflicts).
        """
        if doc_part == DocPart.SUPPORT:
            return self._extract_support(evidences_by_target)
        elif doc_part == DocPart.SUMMARY:
            return self._extract_summary(
                evidences_by_target,
                doc_original,
                progressive,
                threshold,
                modified_file_basenames,
            )
        elif doc_part == DocPart.FUNCTION_PROTO:
            return self._extract_function_proto(
                evidences_by_target,
                doc_original,
                progressive,
                threshold,
                modified_file_basenames,
            )
        elif doc_part == DocPart.SUB_FUNCTION_HASH:
            return self._extract_sub_function_hash(
                evidences_by_target,
                doc_original,
                progressive,
                threshold,
                modified_file_basenames,
            )
        elif doc_part == DocPart.CONSTRAINT:
            return self._extract_constraint(
                evidences_by_target,
                doc_original,
                progressive,
                threshold,
                modified_file_basenames,
            )
        else:
            return {}, []

    @staticmethod
    def _extract_chips_from_file_content(file_content: str, file_path: str = "") -> tuple[set, set]:
        """Rule-based extraction of chip identifiers and arch values from file content.

        Uses regex to find:
        - AICore().AddConfig("chip_name") calls
        - SocVersion::ChipName enum references
        - __CCE_AICORE__ == value / __NPU_ARCH__ == value preprocessor directives
        - DAV_xxxx architecture identifiers

        Returns (chips_set, archs_set).
        """
        chips = set()
        archs = set()

        if not file_content or not isinstance(file_content, str):
            return chips, archs

        # Pattern 1: AICore().AddConfig("chip_name", ...)
        for m in re.finditer(r'AICore\s*\(\s*\)\s*\.AddConfig\s*\(\s*"([^"]+)"', file_content):
            chips.add(m.group(1))

        # Pattern 2: SocVersion::ChipName enum references
        for m in re.finditer(r"SocVersion\s*::\s*(\w+)", file_content):
            chips.add(m.group(1))

        # Pattern 3: #if __CCE_AICORE__ == value or #elif __CCE_AICORE__ == value
        for m in re.finditer(r"#\s*(?:if|elif).*__CCE_AICORE__\s*==\s*(\w+)", file_content):
            archs.add(m.group(1))

        # Pattern 4: #if __NPU_ARCH__ == value
        for m in re.finditer(r"#\s*if.*__NPU_ARCH__\s*==\s*(\w+)", file_content):
            archs.add(m.group(1))

        # Pattern 5: DAV_xxxx identifiers (e.g., DAV_3510, DAV_220)
        for m in re.finditer(r"DAV_(\d+)", file_content):
            archs.add(m.group(1))

        # Pattern 6: arch directory in file path (e.g., arch22, arch35)
        if file_path:
            arch_dir_match = re.search(r"arch(\d+)", file_path)
            if arch_dir_match:
                archs.add(arch_dir_match.group(1))

        return chips, archs

    @staticmethod
    def _extract_chips_from_file_info(content: dict) -> tuple[set, set]:
        """Extract chip hints from file path and file_content field.

        Looks at file_name and file_path for architecture directory hints
        and scans file_content for chip patterns.
        """
        chips = set()
        archs = set()

        file_path = content.get("file_path", "") or ""
        file_name = content.get("file_name", "") or ""
        file_content = content.get("file_content", "") or ""

        # Scan full file content for chip patterns
        content_chips, content_archs = EvidenceExtractor._extract_chips_from_file_content(file_content, file_path)
        chips.update(content_chips)
        archs.update(content_archs)

        # Detect architecture from tiling file names (e.g., tiling_a3.cpp, tiling_a5.cpp)
        for name in [file_name, file_path]:
            if not name:
                continue
            m = re.search(r"tiling_([aA]3|[aA]5)\b", name)
            if m:
                suffix = m.group(1)
                if suffix.lower() == "a3":
                    chips.add("Ascend 910_93")
                elif suffix.lower() == "a5":
                    chips.add("Ascend 950")
            # arch22 / arch35 directory pattern
            m = re.search(r"arch(\d+)", name)
            if m:
                archs.add(m.group(1))

        return chips, archs

    def _extract_support(self, evidences_by_target: dict) -> dict:
        all_chips = set([])
        all_archs = set([])
        source_map = defaultdict(list)

        # Step 1: 提取 section 的 support_chips 结果 + 从 file 证据中规则提取芯片
        for _, ev_list in evidences_by_target.items():
            for ev in ev_list:
                if ev.get("type") == "section":
                    support_chips = ev.get("support_chips", {})
                    all_chips.update(support_chips.get("chips", []) + support_chips.get("socs", []))
                    all_archs.update(support_chips.get("archs", []))
                elif ev.get("type") == "file":
                    content = ev.get("content", {})
                    source_map[content.get("file_path")].append(
                        {
                            "content": content.get("file_content"),
                            "evidence_id": ev.get("evidence_id"),
                        }
                    )
                    # Rule-based chip extraction from file content and file info
                    rule_chips, rule_archs = self._extract_chips_from_file_info(content)
                    all_chips.update(rule_chips)
                    all_archs.update(rule_archs)

        # Step 2: LLM 基于 file 内容验证（增加遗漏） support_chips
        verifier = SupportVerifier(self.llm_client)
        verifier_result = verifier.verify(dict(source_map), all_chips, all_archs)
        if verifier_result:
            all_chips.update(verifier_result.get("support_chips", []))
            all_archs.update(verifier_result.get("support_archs", []))

        # Step 3: 映射产品型号信息
        mapper = ChipMapper()
        supported, stats = mapper.collect_supported_products(list(all_chips), list(all_archs))
        mapper.print_mapping_summary(stats)
        mapper.print_mapping_warnings(stats)
        md_header = "## 产品支持情况\n\n"
        md_content = mapper.generate_markdown(supported)
        result = {
            "support_matrix": {
                "support_chips": list(all_chips),
                "support_archs": list(all_archs),
                "supported": list(supported),
                "md_content": md_header + md_content,
            }
        }
        return result, []

    def _extract_summary(
        self,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        modified_file_basenames: set = None,
    ) -> tuple[dict, list]:
        """Extract summary from evidences using LLM"""
        if not self.llm_client:
            return {}, []

        grouped = self._group_evidences_by_confidence(evidences_by_target, modified_file_basenames)
        targets = [(k, v) for k, v in grouped.items() if k in ["summary_clarify", "summary_formula"]]
        if progressive:
            return self._parallel_extract_targets(targets, doc_original, "summary", threshold)

        result = {}
        all_conflicts = []
        for group_key, extraction in self._parallel_extract_targets(targets, doc_original, "summary", 0.0)[0].items():
            result[group_key] = extraction.get("extraction", extraction)
        return result, all_conflicts

    def _extract_function_proto(
        self,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        modified_file_basenames: set = None,
    ) -> tuple[dict, list]:
        """Extract function prototype info"""
        if not self.llm_client:
            return {}, []

        grouped = self._group_evidences_by_confidence(evidences_by_target, modified_file_basenames)
        targets = list(grouped.items())
        if progressive:
            return self._parallel_extract_targets(targets, doc_original, "function_proto", threshold)

        result = {}
        all_conflicts = []
        for group_key, extraction in self._parallel_extract_targets(targets, doc_original, "function_proto", 0.0)[
            0
        ].items():
            result[group_key] = extraction.get("extraction", extraction)
        return result, all_conflicts

    def _extract_sub_function_hash(
        self,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        modified_file_basenames: set = None,
    ) -> tuple[dict, list]:
        """Extract sub_function_hash param and errorcode info"""
        if not self.llm_client:
            return {}, []

        grouped = self._group_evidences_by_confidence(evidences_by_target, modified_file_basenames)
        targets = [(k, v) for k, v in grouped.items() if "param_matrix" in k or "errorcode" in k]
        if progressive:
            return self._parallel_extract_targets(targets, doc_original, "sub_function_hash", threshold)

        result = {}
        all_conflicts = []
        for group_key, extraction in self._parallel_extract_targets(targets, doc_original, "sub_function_hash", 0.0)[
            0
        ].items():
            result[group_key] = extraction.get("extraction", extraction)
        return result, all_conflicts

    def _extract_constraint(
        self,
        evidences_by_target: dict,
        doc_original: dict,
        progressive: bool = False,
        threshold: float = 0.8,
        modified_file_basenames: set = None,
    ) -> tuple[dict, list]:
        """Extract constraint info"""
        if not self.llm_client:
            return {}, []

        grouped = self._group_evidences_by_confidence(evidences_by_target, modified_file_basenames)
        targets = list(grouped.items())
        if progressive:
            return self._parallel_extract_targets(targets, doc_original, "constraint", threshold)

        result = {}
        all_conflicts = []
        for group_key, extraction in self._parallel_extract_targets(targets, doc_original, "constraint", 0.0)[
            0
        ].items():
            result[group_key] = extraction.get("extraction", extraction)
        return result, all_conflicts

    def _call_llm(self, prompt: str, system_prompt: str) -> dict:
        """Call LLM client with prompt"""
        return self.llm_client.chat_with_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    def _print_extraction_result(
        self, extraction, target_key, status_icon, batch_idx, reason, score, threshold
    ) -> None:
        if get_dashboard():
            dashboard_log(
                f"  {status_icon} [bold cyan]{target_key}[/bold cyan] Level {batch_idx + 1} score={score:.2f}"
            )
            return
        with self._console_lock:
            header_table = Table(show_header=False, border_style="cyan", box=None)
            header_table.add_column(style="cyan")
            # header_table.add_row(f"[bold cyan]{target_key}[/bold cyan] - [cyan]Level {batch_idx + 1}[/cyan]")

            info_table = Table(show_header=True, border_style="cyan")
            info_table.add_column("Batch", style="cyan")
            info_table.add_column("Score", justify="center")
            info_table.add_column("Status", justify="center")
            info_table.add_row(str(batch_idx + 1), f"[bold]{score:.2f}[/bold]", status_icon)

            reason_text = f"[dim]{reason}[/dim]"

            extraction_table = Table(
                show_header=True,
                header_style="bold cyan",
                border_style="cyan",
                expand=True,
            )
            extraction_table.add_column("Target", style="cyan", no_wrap=True, ratio=1)
            extraction_table.add_column("Confidence", justify="center", no_wrap=True, ratio=1)
            extraction_table.add_column("Content", style="white", overflow="fold", ratio=4)
            extraction_table.add_column("Evidence", style="dim", ratio=2)
            for target, items in extraction.items():
                if items:
                    for item in items:
                        conf = item.get("confidence", 0)
                        content = item.get("content", "")
                        if not isinstance(content, str):
                            content = json.dumps(content, ensure_ascii=False)
                        evidence_ids = ", ".join(item.get("source_file", item.get("evidence_id", [])))
                        conf_icon = "[green]✓[/green]" if conf >= threshold else "[yellow]~[/yellow]"
                        extraction_table.add_row(
                            target,
                            f"{conf:.2f} {conf_icon}",
                            content,
                            f"[dim]{evidence_ids}[/dim]",
                        )

            content_panel = Panel(
                extraction_table if extraction else "[dim]No extraction results[/dim]",
                title="Extraction",
                border_style="cyan",
            )

            result_layout = Table(show_header=False, border_style="cyan", box=None)
            result_layout.add_column(style="cyan", overflow="fold")
            result_layout.add_row(header_table)
            result_layout.add_row(info_table)
            result_layout.add_row(reason_text)
            result_layout.add_row(content_panel)

            console.print()
            console.print(
                Panel(
                    result_layout,
                    title=f"[bold cyan]{target_key}[/bold cyan] - [cyan]Level {batch_idx + 1}[/cyan]",
                    border_style="cyan",
                )
            )

    def _parallel_extract_targets(
        self,
        targets: list[tuple[str, list[list[dict]]]],
        doc_original: dict,
        doc_part: str,
        threshold: float,
    ) -> tuple[dict, list]:
        """Parallel extraction across independent targets. Returns (result_dict, all_conflicts)."""
        result = {}
        all_conflicts = []
        current_dashboard = get_dashboard()
        current_doc_part = get_current_doc_part()

        def _run_with_context(target_key, ev_groups):
            if current_dashboard:
                tokens = set_dashboard_context(current_dashboard, current_doc_part)
                try:
                    return self._progressive_extract_single_target(
                        target_key=target_key,
                        ev_groups=ev_groups,
                        doc_original=doc_original,
                        doc_part=doc_part,
                        threshold=threshold,
                    )
                finally:
                    for t in reversed(tokens):
                        t.var.reset(t)
            else:
                return self._progressive_extract_single_target(
                    target_key=target_key,
                    ev_groups=ev_groups,
                    doc_original=doc_original,
                    doc_part=doc_part,
                    threshold=threshold,
                )

        with ThreadPoolExecutor(max_workers=config.EXTRACT_MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _run_with_context,
                    target_key=group_key,
                    ev_groups=ev_groups,
                ): group_key
                for group_key, ev_groups in targets
            }
            _is_tty = sys.stdout.isatty()
            _has_dashboard = current_dashboard is not None
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                console=console,
                disable=not _is_tty or _has_dashboard,
            ) as progress:
                task = progress.add_task(f"正在提取 {config.format_doc_part(doc_part)}", total=len(futures))
                for future in as_completed(futures):
                    group_key = futures[future]
                    extraction, conflicts = future.result()
                    result[group_key] = extraction
                    all_conflicts.extend(conflicts)
                    progress.update(task, advance=1)
        return result, all_conflicts

    def _progressive_extract_single_target(
        self,
        target_key: str,
        ev_groups: list[list[dict]],
        doc_original: dict,
        doc_part: str,
        threshold: float = 0.8,
    ) -> tuple[dict, list]:
        """Progressive extraction for single target with self-scoring"""
        if not self.llm_client:
            return {}, []

        final_result = {}
        all_source_conflicts = []
        batch_idx = 0
        round_num = 0

        while batch_idx < len(ev_groups):
            current_evs = list(ev_groups[batch_idx])

            # First round: merge 1.0 (modified files) + 0.9 (highest confidence)
            if batch_idx == 0 and len(ev_groups) > 1:
                layer_1_0_count = len(current_evs)
                layer_0_9_count = len(ev_groups[batch_idx + 1])
                current_evs.extend(ev_groups[batch_idx + 1])
                batch_idx = 2  # Skip 0.9 layer, next round starts from 0.8
                log_result(
                    f"[{target_key}] Merged 1.0 layer ({layer_1_0_count}) + 0.9 layer ({layer_0_9_count}) "
                    f"= {len(current_evs)} evidences"
                )
            else:
                batch_idx += 1

            if not current_evs:
                continue

            round_num += 1

            prompt = self._build_progressive_prompt(
                target=target_key,
                current_evidences=current_evs,
                doc_original=doc_original,
                doc_part=doc_part,
            )

            try:
                response = self._call_llm(prompt, "你是一个技术文档助手，擅长从代码和注释中提取信息。")
            except Exception as e:
                log_result(f"[{target_key}] LLM call failed: {e}")
                return {}, []

            if not isinstance(response, dict):
                log_result(f"[{target_key}] LLM returned {type(response).__name__} instead of dict: {response}")
                return {}, []
            extraction = response.get("extraction", {})
            self_score = response.get("self_score", {})
            if not extraction and isinstance(self_score, dict):
                log_result(f"[{target_key}] extraction is empty, attempting to recover from top-level keys")
                # fmt: off
                extraction = {
                    k: v
                    for k, v in response.items()
                    if k not in ("self_score", "extraction") and isinstance(v, list)
                }
                # fmt: on
            if not isinstance(extraction, dict):
                log_result(f"[{target_key}] extraction is {type(extraction).__name__}, expected dict")
                extraction = {}
            else:
                non_list_keys = [k for k, v in extraction.items() if not isinstance(v, list)]
                if non_list_keys:
                    log_result(f"[{target_key}] filtering non-list values for keys: {non_list_keys}")
                    extraction = {k: v for k, v in extraction.items() if isinstance(v, list)}
            if not isinstance(self_score, dict):
                log_result(f"[{target_key}] self_score is {type(self_score).__name__}, expected dict")
                self_score = {}
            score = self_score.get("score", 0.0)
            reason = self_score.get("reason", "N/A")
            category = self._get_target_category(target_key)
            extraction = self._add_metadata_to_extraction(extraction, category, score)

            source_conflicts = response.get("source_conflicts", [])
            if source_conflicts:
                log_result(f"[{target_key}] Detected {len(source_conflicts)} source conflict(s)")
                for conflict in source_conflicts:
                    log_result(
                        f"  - {conflict.get('fact', '?')}: {conflict.get('values', [])} "
                        f"(evidences: {conflict.get('evidence_ids', [])})"
                    )
                all_source_conflicts.extend(source_conflicts)

            status_icon = "[green]✓[/green]" if score >= threshold else "[yellow]✗[/yellow]"

            log_result(
                f"[{target_key} - Level {round_num}] batch = {round_num}, "
                f"score = {score}, reason = {reason}, extraction = {json.dumps(extraction, ensure_ascii=False)}"
            )

            self._print_extraction_result(
                extraction,
                target_key,
                status_icon,
                round_num - 1,
                reason,
                score,
                threshold,
            )

            if score >= threshold:
                final_result = extraction
                break
            else:
                final_result = self._merge_results(final_result, extraction)
                batch_idx += 1

        return final_result, all_source_conflicts

    def _build_progressive_prompt(
        self,
        target: str,
        current_evidences: list[dict],
        doc_original: dict,
        doc_part: str,
    ) -> str:
        """Build prompt for progressive extraction with self-scoring"""
        category = self._get_target_category(target)
        category_instruction = (
            "请严格遵循证据中的事实型内容" if category == "fact" else "可基于证据中的描述型内容进行适当发挥和补充"
        )

        contents = self._prepare_evidences_content(current_evidences)
        content_json = json.dumps(contents, ensure_ascii=False, indent=2)
        doc_original_json = json.dumps(doc_original, ensure_ascii=False, indent=2)

        if doc_part == "summary":
            return self._build_summary_progressive_prompt(target, content_json, doc_original_json, category_instruction)
        elif doc_part == "function_proto":
            return self._build_function_proto_progressive_prompt(content_json, doc_original_json, category_instruction)
        elif doc_part == "sub_function_hash":
            return self._build_sub_function_progressive_prompt(target, content_json, doc_original, category_instruction)
        elif doc_part == "constraint":
            return self._build_constraint_progressive_prompt(content_json, doc_original_json, category_instruction)
        else:
            return f"""从以下内容中提取信息并自评：
{content_json}

返回格式：{{"extraction": {{...}}, "self_score": {{"score": 0.0-1.0, "reason": "..."}}}}"""

    def _build_summary_progressive_prompt(
        self, target: str, content: str, doc_original: str, category_instruction: str
    ) -> str:
        """Build progressive prompt for summary"""
        if target == "summary_clarify":
            task_content = """1、简要介绍接口功能，增加V版本之间功能对比
2、多个aclnn接口写在一个文档内的情况，需要简要介绍多个aclnn接口间的区别。Inplace和非Inplace的接口区别默认已知不写。

目标：指导用户快速理解算子使用场景、计算逻辑和作用。
①算子接口功能（必选）：讲清楚接口功能、计算逻辑、使用场景、与其他接口关系（若有请提供跳转链接）等，可参考友商表述，但不能使用友商术语。
②区分芯片：若功能在不同芯片上有差异，请分别描述，无序列表形式。"""
            return_fields = (
                '"算子接口功能": [{"content": "描述内容", "source_file": ["源文件名"]}],\n'
                '    "区分芯片": [{"content": "描述内容", "source_file": ["源文件名"]}]'
            )
        elif target == "summary_formula":
            task_content = """1、罗列计算公式

目标：指导用户快速理解算子使用场景、计算逻辑和作用。
①计算公式：复杂算子建议有公式，公式变量需给出解释。
②示例：复杂算子，根据实际需要，可通过示例阐述计算逻辑。
③公式差异项：证据代码中的计算逻辑（计算顺序、参与项、常量）与旧文档公式存在差异时，必须单独成条显式提取差异点（如"bias 在反量化之后累加，原文公式为反量化之前累加"），供下游更新公式使用；计算逻辑未变时不得臆造差异。"""
            return_fields = (
                '"计算公式": [{"content": "描述内容", "source_file": ["源文件名"]}],\n'
                '    "示例": [{"content": "描述内容", "source_file": ["源文件名"]}]'
            )
        else:
            task_content = ""
            return_fields = '"内容": [{"content": "描述内容", "source_file": ["源文件名"]}]'

        return self._base_progressive_prompt(task_content, content, doc_original, category_instruction, return_fields)

    def _build_function_proto_progressive_prompt(
        self, content: str, doc_original: str, category_instruction: str
    ) -> str:
        """Build progressive prompt for function_proto"""
        task_content = """1、介绍算子两段式接口调用流程，eg:每个算子分为两段式接口，
必须先调用"aclnnAddGetWorkspaceSize"或者"aclnnInplaceAddGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，
再调用"aclnnAdd"或者"aclnnInplaceAdd"接口执行计算。
2、每个接口一个单独的代码块，标明Cpp语言类型，每行一个参数，eg:
```cpp
aclnnStatus aclnnAddGetWorkspaceSize(
  const aclTensor  *self,
  const aclTensor  *other,
  const aclScalar  *alpha,
  char             *beta
  uint64_t          theta
  double            x
  bool              flag
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnAdd(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```"""
        return_fields = (
            '"调用流程": [{"content": "描述内容", "source_file": ["源文件名"]}],\n'
            '    "代码块": [{"content": "描述内容", "source_file": ["源文件名"]}]'
        )
        return self._base_progressive_prompt(task_content, content, doc_original, category_instruction, return_fields)

    def _build_sub_function_progressive_prompt(
        self,
        target: str,
        content: str,
        doc_original: dict,
        category_instruction: str,
    ) -> str:
        """Build progressive prompt for sub_function"""
        if "param_matrix" in target:
            task_content = """1、表格固定呈现8个固定字段。
1) 参数名：例如：self（aclTensor*）,other（aclTensor*）,alpha（aclScalar*）,beta（char*）,theta（uint64_t）,
x（double）,out（xx）,workspaceSize（xx）,executor(xx).
2) 输入/输出: 二选一
3) 描述：给出参数含义+公式变量映射，例如：表示xxx含义，对应公式中self。
4) 使用说明（其他字段无法承载的剩余内容）：空Tensor支持度（必选，支持和不支持讲清楚）+参数带Optional后缀需显式讲传空指针场景+其他约束（取值/范围/场景等）
备注：实际软件要求"空Tensor"默认支持，但是资料公共地方没说明过，因此建议所有接口显式讲清楚，避免混淆。
例如：- 支持空Tensor。- 数据类型与other的数据类型需满足数据类型推导规则（参见互推导关系）。- shape需要与other满足broadcast关系。
；或：预留参数，当前暂不支持，传入空指针即可。 ；或：取值范围为[xx, xx]
5) 数据类型：仅aclXxx类型参数（如aclTensor/aclTensorList/aclScalar/aclIntArray等）需要填写；
非此类对象，属于C本身自带的参数类型（如char*/bool/uint64/float/double等），不需要填写。
6) 数据格式：不涉及的打-，涉及的写具体内容，如ND、FRACTAL_NZ（NZ）等。
7) 维度(shape)：不涉及的打-，涉及的写范围，例如 0-8 或 1-8。
8) 非连续Tensor：不涉及的打-，涉及的参数若支持非连续打√、不支持的打×。

2、区分芯片写作：
数据类型、shape、format、取值等信息，表里提供所有产品支持的并集，差异描述放在表下方。* 注意：不要在表格里区分芯片描述，差异化描述在表格外，以"产品1、产品2：xx描述"形式组织。"""
            # 注入本小节参数白名单（旧文档参数表首列）：约束提取的参数行范围，
            # 防止其他小节/其他算子变体的参数串入（如执行接口小节只应有执行参数）
            doc_original = doc_original.get("param_matrix", {}).get(target, {})
            if isinstance(doc_original, str) and doc_original.strip():
                from ..pipeline.deterministic_checks import _table_params

                allowed = _table_params(doc_original)
                if allowed:
                    task_content += (
                        f"\n\n# 本小节参数白名单（旧文档参数表首列，可选参数可能带 Optional 后缀）\n"
                        f"{', '.join(allowed)}\n"
                        f"提取的参数行的参数名必须来自上述白名单（忽略 Optional 后缀差异）；"
                        f"白名单之外的参数仅当其出现在本小节标题对应 aclnn 接口的新版函数签名中才允许提取，"
                        f"否则一律不得作为本小节的参数行输出。"
                    )
            return_fields = (
                '"表格固定字段内容": [{"content": "描述内容", "source_file": ["源文件名"]}],\n'
                '    "区分芯片写作": [{"content": "描述内容", "source_file": ["源文件名"]}]'
            )
        elif "errorcode" in target:
            task_content = """1、表格固定呈现3个固定字段。
1) 返回值：string名称，例如：ACLNN_ERR_PARAM_NULLPTR
2) 错误码: 例如：161001
3) 描述：致错误可能原因描述，例如：selfRef、other、alpha、out存在空指针。
校验表里原因不要区分芯片，最好模糊和笼统，因为本表不支持按芯片过滤，
尽量找大颗粒原因，不要写细节，提供定位方向即可。原因笼统些，例如下方罗列的：
xx、xx存在空指针
xx、xx或xx的数据类型不在支持的范围内
xx、xx或xx的数据类型不匹配
xx、xx或xx的shape维度不在支持的范围内
xx、xx的shape维度不匹配
xx、xx的shape不满足broadcast规则
xx的shape不满足输出shape推导公式"""
            return_fields = '"表格固定字段内容": [{"content": "描述内容", "source_file": ["源文件名"]}]'
            doc_original = doc_original.get("errorcode_matrix", {}).get(target, {})
        else:
            task_content = ""
            return_fields = '"内容": [{"content": "描述内容", "source_file": ["源文件名"]}]'
        doc_original_json = json.dumps(doc_original, ensure_ascii=False, indent=2)
        return self._base_progressive_prompt(
            task_content,
            content,
            doc_original_json,
            category_instruction,
            return_fields,
        )

    def _build_constraint_progressive_prompt(self, content: str, doc_original: str, category_instruction: str) -> str:
        """Build progressive prompt for constraint"""
        task_content = "# 资料书写所需内容\n从代码中提取约束说明信息：\n- 确定性\n- 输入约束\n- 其他约束"
        return_fields = (
            '"确定性": [{"content": "描述内容", "source_file": ["源文件名"]}],\n'
            '    "输入约束": [{"content": "描述内容", "source_file": ["源文件名"]}],\n'
            '    "其他约束": [{"content": "描述内容", "source_file": ["源文件名"]}]'
        )
        return self._base_progressive_prompt(task_content, content, doc_original, category_instruction, return_fields)

    def _merge_results(self, prev: dict, new: dict) -> dict:
        """Merge previous and new extraction results"""
        if not prev:
            return new
        if not new:
            return prev

        merged = {}
        all_keys = set(prev.keys()) | set(new.keys())
        for key in all_keys:
            prev_items = prev.get(key, [])
            new_items = new.get(key, [])
            merged[key] = self._merge_and_deduplicate(prev_items, new_items)
        return merged
