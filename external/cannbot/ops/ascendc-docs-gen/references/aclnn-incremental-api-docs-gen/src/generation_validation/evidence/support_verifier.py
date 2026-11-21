# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Support verifier - uses LLM to verify chips/archs extracted from code evidences (batched)"""

from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from src.llm.client import LLMClient
from src.logger import highlight
from ..validator.evidence_batch import EvidenceBatchManager
from ..dashboard import dashboard_progress, dashboard_log
from .. import config

console = Console()


class SupportVerifier:
    """LLM-based verifier for support chips/archs extraction (batched + parallel)"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.batch_manager = EvidenceBatchManager(max_tokens=config.MAX_TOKENS)

    @staticmethod
    def build_batch_verification_prompt(
        file_segments: List[dict],
    ) -> str:
        """Build LLM verification prompt for a batch of file segments"""
        code_blocks = []
        for i, seg in enumerate(file_segments, 1):
            source_file = seg.get("source_file", seg.get("evidence_id", f"file_{i}"))
            content = seg.get("content", "")
            code_blocks.append(f"【源文件 {i}: {source_file}】\n```cpp\n{content}\n```")

        combined_code = "\n\n".join(code_blocks)

        prompt = f"""你是代码分析专家，专精芯片型号和架构标识符识别。

【源代码片段（共 {len(file_segments)} 个文件）】
{combined_code}

【你的任务】
仔细审查上述所有代码文件，提取其中包含的芯片型号和架构标识符：

1-查找代码中是否包含 AICore().AddConfig() 中的芯片？
2-查找形如 .AddConfig("chip_name", ...) 的调用
3-注意注释中的芯片说明
4-查找代码中是否包含 curShortSocName == platform_ascendc::SocVersion::<SOC>
5-if/switch 中的 SocVersion::<SOC> 代码中的 SoC/NpuArch 标识就是支持芯片
6-代码中是否包含架构标识符（如 NPU_ARCH 值）？
7-查找 #if NPU_ARCH == ... 的预处理指令
8-查找文件路径中的 arch35, arch30 等目录标识

【重要】返回的芯片名称必须是代码中的完整标识符，不要缩写或改写：
- 如果 AddConfig("Ascend910_93") 则返回 "Ascend910_93"，不要返回 "A3" 或 "910_93"
- 如果 SocVersion::Ascend910C 则返回 "Ascend910C"
- 如果 SocVersion::Ascend950 则返回 "Ascend950"
- 如果文件路径含 arch22 目录，archs 中返回 "2201"（不是 "22"）
- 如果文件路径含 arch35 目录，archs 中返回 "3510"（不是 "35"）
- 如果代码含 DAV_3510，archs 中返回 "3510"
- 如果代码含 __CCE_AICORE__ == 200，archs 中返回 "200"

如果提取到了新架构和新芯片，返回的应该是对应的完整名字，例如：
    - chips = ["Ascend950", "Ascend910_93"]
    - archs = ["3510", "2201"]

【输出格式】请返回JSON格式：
{{
  "chips": ["Ascend950"],
  "archs": ["3510"],
  "explanation": "一句话简要说明"
}}
"""
        return prompt

    def verify_batch_with_llm(
        self,
        batch_segments: List[dict],
    ) -> tuple[Set[str], Set[str]]:
        """Verify a batch of file segments with one LLM call"""
        prompt = self.build_batch_verification_prompt(batch_segments)
        try:
            response = self.llm_client.chat_with_json(
                [
                    {
                        "role": "system",
                        "content": "你是一个昇腾算子代码分析专家，精通芯片型号和架构标识符识别。",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:
            return set(), set()

        chips = set()
        archs = set()

        for chip in response.get("chips", []):
            if chip and chip not in ["无", "(无)"]:
                chips.add(chip.strip())

        for arch in response.get("archs", []):
            if arch and arch not in ["无", "(无)"]:
                archs.add(arch.strip())

        return chips, archs

    def verify(
        self, source_map: dict, all_chips: Set[str], all_archs: Set[str]
    ) -> Dict[str, List[str]]:
        # 展平 source_map 为 evidence 列表，复用 EvidenceBatchManager
        evidence_list = []
        for source_file, code_segments in source_map.items():
            for seg in code_segments:
                evidence_list.append(
                    {
                        "evidence_id": seg.get("evidence_id", source_file),
                        "source_file": source_file,
                        "content": seg.get("content", ""),
                        "type": "file",
                    }
                )

        if not evidence_list:
            return {
                "support_chips": list(all_chips),
                "support_archs": list(all_archs),
            }

        batches = self.batch_manager.split_into_batches(evidence_list)

        if len(batches) > 1:
            msg = f"> [dim]\\[## 产品支持情况][/dim] 验证: {len(evidence_list)} 个证据分为 {len(batches)} 个批次"
            highlight(msg)
            dashboard_log(f"  {msg}")

        batch_results = [None] * len(batches)
        completed = 0
        with ThreadPoolExecutor(max_workers=config.SUPPORT_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.verify_batch_with_llm, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    batch_results[idx] = (set(), set())
                    dashboard_log(
                        f"  > [dim]\\[## 产品支持情况][/dim] Batch {idx + 1}/{len(batches)} failed: {e}"
                    )
                completed += 1
                dashboard_progress(completed, len(batches))
                if len(batches) > 1:
                    dashboard_log(
                        f"  > [dim]\\[## 产品支持情况][/dim] Batch {idx + 1}/{len(batches)} done"
                    )

        final_chips = all_chips.copy()
        final_archs = all_archs.copy()
        for chips, archs in batch_results:
            final_chips.update(chips)
            final_archs.update(archs)

        msg = f"  > [dim]\\[## 产品支持情况][/dim]\n\t建议芯片: {final_chips}\n\t建议架构: {final_archs}"
        highlight(msg)
        dashboard_log(msg)

        return {
            "support_chips": list(final_chips),
            "support_archs": list(final_archs),
        }
