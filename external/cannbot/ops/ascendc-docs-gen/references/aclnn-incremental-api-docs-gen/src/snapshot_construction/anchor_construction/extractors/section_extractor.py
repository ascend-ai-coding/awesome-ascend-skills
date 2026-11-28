# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
SectionExtractor - Unified section extraction for all doc_parts.

Consolidates section extraction patterns from:
- constraint/data_parsing.py: _get_code_block() (adjacent non-blank-line blocks)
- constraint/data_parsing.py: SocVersion, __NPU_ARCH__, __CCE_AICORE__ pattern matching
- support/extract_new_0331.py: extract_type1() (AICore().AddConfig pattern)
- support/extract_new_0331.py: extract_type2() (curShortSocName == SocVersion pattern)
- support/extract_new_0331.py: extract_type3() (__NPU_ARCH__/__CCE_AICORE__ #if pattern)
- sub_function/extract/extract_section.py: keyword-based section extraction
- sub_function/merge/merge_section_extract.py: SECTION_MAPPING

Uses unified infrastructure from common/:
- brace_utils.py: line-based and content-based brace functions
- anchor_builder.py: build_section_anchor()
- data_loader.py: DataLoader
- id_generator.py: IdGenerator
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.snapshot_construction.anchor_construction.common.anchor_builder import (
    build_section_anchor,
)
from src.snapshot_construction.anchor_construction.common.brace_utils import (
    find_enclosing_function,
    find_endif_line,
    find_full_if_else_block,
    find_if_block_start,
    find_matching_brace,
)
from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader
from src.snapshot_construction.anchor_construction.common.id_generator import (
    IdConfig,
    IdGenerator,
)
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig

logger = logging.getLogger(__name__)


# Pattern constants (from support/extract_new_0331.py)
PATTERN_ADD_CONFIG = re.compile(r'AICore\s*\(\s*\)\s*\.AddConfig\s*\(\s*"([^"]+)"')

PATTERN_SOC_VERSION = re.compile(r"curShortSocName\s*==\s*platform_ascendc::SocVersion::(\w+)")

PATTERN_NPU_ARCH = re.compile(r"#\s*if.*__NPU_ARCH__\s*==\s*(\w+)")

PATTERN_CCE_AICORE = re.compile(r"#\s*(if|elif).*__CCE_AICORE__\s*==\s*(\w+)")

# Keyword mapping (from sub_function/merge/merge_section_extract.py)
SECTION_MAPPING = {
    "SUPPORT_LIST": "dtype",
    "check_dtype": "dtype",
    "CheckFormat": "dmat",
    "CheckShape": "shape",
    "continuous": "tensor",
    "CheckParams": "errorcode",
}

# Keywords for keyword-based extraction (from sub_function/extract/extract_section.py)
CPP_KEYWORDS = [
    "SUPPORT_LIST",
    "check_dtype",
    "CheckFormat",
    "CheckShape",
    "continuous",
    "CheckParams",
]


class SectionExtractor:
    """Unified section extractor for all doc_parts.

    Extracts section anchors from operator source files using multiple pattern
    detection strategies. Returns in-memory data (no file writing).

    The extractor does NOT assign gene_rule - that's the pipeline's job via
    GeneRuleRegistry. gene_deter and gene_target are placeholders.

    Attributes:
        path_config: PathConfig instance for path resolution
        data_loader: DataLoader instance for loading file content
        id_generator: IdGenerator instance for anchor_id and evidence_id
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ):
        """Initialize SectionExtractor with required dependencies.

        Args:
            path_config: PathConfig instance containing operator and output paths
            data_loader: DataLoader instance for loading file content
            id_generator: IdGenerator instance for generating anchor and evidence IDs
        """
        self.path_config = path_config
        self.data_loader = data_loader
        self.id_generator = id_generator
        logger.debug(f"SectionExtractor initialized for operator: {path_config.operator_path}")

    def extract_generic_code_block(self, lines: list[str], target_line_idx: int) -> dict[str, Any]:
        """Extract adjacent non-blank-line code block around a target line.

        This is the simplest extraction pattern from constraint/data_parsing.py _get_code_block().
        Extracts all contiguous non-blank lines around the target line.

        Args:
            lines: List of file lines (0-indexed)
            target_line_idx: Target line index (0-based)

        Returns:
            Dict with:
                - content: Code block content
                - start_line: Start line number (1-based)
                - end_line: End line number (1-based)
                - token_hash: Same as content (placeholder)

        Example:
            >>> lines = ["", "int x = 1;", "int y = 2;", "", "int z = 3;"]
            >>> result = extractor.extract_generic_code_block(lines, 1)
            >>> result['start_line'] == 2  # 1-based
            >>> result['end_line'] == 3    # 1-based
        """
        left, right = target_line_idx - 1, target_line_idx + 1

        # Expand left to include all non-blank lines
        while left >= 0 and lines[left].strip():
            left -= 1

        # Expand right to include all non-blank lines
        while right < len(lines) and lines[right].strip():
            right += 1

        # Extract block
        start = left + 1
        block_lines = lines[start:right]
        content = "".join(block_lines)
        start_line = left + 2  # Convert 0-based slice start to 1-based line number
        end_line = right  # Convert 0-based slice end to 1-based line number

        return {
            "content": content,
            "start_line": start_line,
            "end_line": end_line,
            "token_hash": content,
        }

    def extract_soc_version_sections(self, lines: list[str]) -> list[dict[str, Any]]:
        """Extract sections with curShortSocName == SocVersion pattern.

        Pattern from support/extract_new_0331.py extract_type2().
        Finds if/else blocks containing SocVersion comparisons.

        Args:
            lines: List of file lines

        Returns:
            List of dicts with:
                - content: Full if-else block content
                - start_line: Start line (1-based)
                - end_line: End line (1-based)
                - token_hash: Same as content
                - socs: List of SOC names matched
        """
        sections = []
        match_indices = [i for i, line in enumerate(lines) if PATTERN_SOC_VERSION.search(line)]

        if not match_indices:
            return sections

        # Deduplicate blocks (same if-else block may match multiple times)
        extracted_blocks = {}
        for idx in match_indices:
            m = PATTERN_SOC_VERSION.search(lines[idx])
            soc = m.group(1) if m else "UNKNOWN"
            if_start = find_if_block_start(lines, idx)
            block_end = find_full_if_else_block(lines, if_start)
            extracted_blocks.setdefault((if_start, block_end), []).append(soc)

        # Build section dicts
        for (if_start, block_end), socs in extracted_blocks.items():
            end = block_end + 1
            content = "".join(lines[if_start:end])
            sections.append(
                {
                    "content": content,
                    "start_line": if_start + 1,  # 1-based
                    "end_line": block_end + 1,  # 1-based
                    "token_hash": content,
                    "socs": socs,
                }
            )

        return sections

    def extract_npu_arch_sections(self, lines: list[str]) -> list[dict[str, Any]]:
        """Extract sections with __NPU_ARCH__/__CCE_AICORE__ #if pattern.

        Pattern from support/extract_new_0331.py extract_type3() and
        constraint/data_parsing.py lines 262-280.
        Finds preprocessor #if/#elif blocks with architecture macros.

        Args:
            lines: List of file lines

        Returns:
            List of dicts with:
                - content: Full #if/#endif block content
                - start_line: Start line (1-based)
                - end_line: End line (1-based)
                - token_hash: Same as content
                - arch_name: Architecture name matched
                - arch_macro: Macro name (__NPU_ARCH__ or __CCE_AICORE__)
        """
        sections = []
        extracted_blocks = {}

        for i, line in enumerate(lines):
            m_npu = PATTERN_NPU_ARCH.search(line)
            m_cce = PATTERN_CCE_AICORE.search(line)

            if m_npu:
                arch_name, arch_macro = m_npu.group(1), "__NPU_ARCH__"
            elif m_cce:
                arch_name, arch_macro = m_cce.group(2), "__CCE_AICORE__"
            else:
                continue

            endif_idx = find_endif_line(lines, i)
            key = (i, endif_idx)

            if key not in extracted_blocks:
                extracted_blocks[key] = (arch_name, arch_macro)

        # Build section dicts
        for (if_start, endif_idx), (arch_name, arch_macro) in extracted_blocks.items():
            end = endif_idx + 1
            content = "".join(lines[if_start:end])
            sections.append(
                {
                    "content": content,
                    "start_line": if_start + 1,  # 1-based
                    "end_line": endif_idx + 1,  # 1-based
                    "token_hash": content,
                    "arch_name": arch_name,
                    "arch_macro": arch_macro,
                }
            )

        return sections

    def extract_add_config_sections(self, lines: list[str]) -> list[dict[str, Any]]:
        """Extract sections with AICore().AddConfig pattern.

        Pattern from support/extract_new_0331.py extract_type1().
        Finds enclosing functions containing AICore().AddConfig calls.

        Args:
            lines: List of file lines

        Returns:
            List of dicts with:
                - content: Full function content
                - start_line: Start line (1-based)
                - end_line: End line (1-based)
                - token_hash: Same as content
                - chips: List of chip names matched
        """
        sections = []
        match_indices = [i for i, line in enumerate(lines) if PATTERN_ADD_CONFIG.search(line)]

        if not match_indices:
            return sections

        # Deduplicate functions (same function may have multiple AddConfig calls)
        extracted_functions = {}
        for idx in match_indices:
            m = PATTERN_ADD_CONFIG.search(lines[idx])
            chip = m.group(1) if m else "UNKNOWN"
            sig_start, func_end = find_enclosing_function(lines, idx)
            extracted_functions.setdefault((sig_start, func_end), []).append(chip)

        # Build section dicts
        for (sig_start, func_end), chips in extracted_functions.items():
            end = func_end + 1
            content = "".join(lines[sig_start:end])
            sections.append(
                {
                    "content": content,
                    "start_line": sig_start + 1,  # 1-based
                    "end_line": func_end + 1,  # 1-based
                    "token_hash": content,
                    "chips": chips,
                }
            )

        return sections

    def extract_keyword_sections(self, lines: list[str], keywords: list[str] = None) -> list[dict[str, Any]]:
        """Extract sections containing specific keywords.

        Pattern from sub_function/extract/extract_section.py.
        Finds code blocks containing keywords like SUPPORT_LIST, check_dtype, etc.

        Args:
            lines: List of file lines
            keywords: List of keywords to search (default: CPP_KEYWORDS)

        Returns:
            List of dicts with:
                - content: Code block content
                - start_line: Start line (1-based)
                - end_line: End line (1-based)
                - token_hash: Same as content
                - source: Keyword that matched
        """
        if keywords is None:
            keywords = CPP_KEYWORDS

        sections = []
        content = "".join(lines)

        # Normalize for case-insensitive matching
        def normalize_text(text: str) -> str:
            return re.sub(r"\s+", "", text.lower())

        normalized_content = normalize_text(content)
        found_sources = set()

        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)

            pos = 0
            while True:
                match = re.search(re.escape(normalized_keyword), normalized_content[pos:])
                if not match:
                    break

                actual_pos = pos + match.start()

                # Find line number from position
                content[:actual_pos].count("\n") + 1

                # Extract section around keyword
                section = self._extract_section_around_keyword(content, keyword, actual_pos, lines)
                if section:
                    section["source"] = keyword
                    sections.append(section)
                    found_sources.add(keyword)

                pos = actual_pos + 1

        return sections

    def _extract_section_around_keyword(
        self, content: str, keyword: str, start_pos: int, lines: list[str]
    ) -> dict[str, Any] | None:
        """Extract a code section around a keyword position.

        Helper method for extract_keyword_sections().
        Attempts to find enclosing braces or statement boundaries.

        Args:
            content: Full file content as string
            keyword: Keyword that was matched
            start_pos: Position in content where keyword was found
            lines: List of file lines

        Returns:
            Dict with content, start_line, end_line, token_hash, or None if extraction fails
        """
        line_num = content[:start_pos].count("\n") + 1

        section_start_pos = start_pos
        search_pos = start_pos - 1

        # Search backwards for boundaries
        while search_pos > 0:
            if content[search_pos] == ";":
                section_start_pos = search_pos + 1
                break
            if content[search_pos] == "{":
                # Found opening brace - find matching closing brace
                end_brace_pos = find_matching_brace(content, search_pos)
                if end_brace_pos != -1:
                    section_start_pos = search_pos
                    section_end_pos = end_brace_pos + 1

                    start_line = content[:section_start_pos].count("\n") + 1
                    end_line = content[:section_end_pos].count("\n")

                    section_content = content[section_start_pos:section_end_pos]

                    return {
                        "content": section_content,
                        "start_line": start_line,
                        "end_line": end_line,
                        "token_hash": section_content,
                    }
                break
            if content[search_pos] == "\n" and search_pos < start_pos - 1:
                next_char = content[search_pos + 1]
                if next_char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    section_start_pos = search_pos + 1
                    break
            search_pos -= 1

        # Fallback: extract surrounding lines
        start_line = max(1, line_num - 5)
        end_line = min(len(lines), line_num + 10)

        start = start_line - 1
        section_content = "\n".join(lines[start:end_line])

        return {
            "content": section_content,
            "start_line": start_line,
            "end_line": end_line,
            "token_hash": section_content,
        }

    def extract(
        self,
        operator_path: Path,
        section_patterns: list[str] = None,
        source_files: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Extract section anchors from operator directory.

        Main extraction method that orchestrates all pattern-specific extraction
        methods based on section_patterns configuration.

        Args:
            operator_path: Path to operator source directory
            section_patterns: List of pattern types to extract:
                - "soc_version": curShortSocName == SocVersion pattern
                - "npu_arch": __NPU_ARCH__/__CCE_AICORE__ #if pattern
                - "add_config": AICore().AddConfig pattern
                - "keyword": keyword-based extraction (SUPPORT_LIST, etc.)
                - "generic": adjacent non-blank-line blocks
                Default: all patterns
            source_files: Optional list of source file paths/names to filter by.
                When provided, only files matching (by path string or basename) are kept.
                When None, all discovered files are returned (backward compatible).

        Returns:
            List of anchor dicts with structure:
            {
                "anchor_id": "anchor_section_001",
                "anchor_information": {
                    "type": "section",
                    "level1": {"file_path": ..., "start_line": ..., "end_line": ..., "token_hash": ...},
                    "level2": {"token_hash": ...}
                },
                "evidence": [
                    {
                        "evidence_id": "ev_section_001",
                        "envidence_id": "ev_section_001",  # compat requirement
                        "evidence_information": {...},
                        "gene_rule": {"deter": "暂空", "target": []}  # placeholder
                    }
                ],
                "note": "以下为section型标准结构"
            }

        Note:
            gene_rule is placeholder (deter="暂空", target=[]).
            Pipeline assigns actual gene_rule via GeneRuleRegistry.
        """
        if section_patterns is None:
            section_patterns = [
                "soc_version",
                "npu_arch",
                "add_config",
                "keyword",
                "generic",
            ]

        logger.info(f"Extracting section anchors from: {operator_path}")
        logger.debug(f"  Patterns: {section_patterns}")

        anchors = []

        # Find all cpp files in operator directory
        cpp_files = []
        for pattern in ["**/*.cpp", "**/*.h"]:
            cpp_files.extend(operator_path.glob(pattern))

        logger.debug(f"Found {len(cpp_files)} C++ files")

        if source_files is not None:
            source_names = {Path(s).name for s in source_files}
            source_strs = set(source_files)
            cpp_files = [fp for fp in cpp_files if str(fp) in source_strs or fp.name in source_names]
            logger.debug(f"Filtered to {len(cpp_files)} C++ files matching source_files")

        for file_path in cpp_files:
            lines = self.data_loader.read_lines(file_path)
            if not lines:
                continue

            file_sections = []

            # Extract based on patterns
            if "soc_version" in section_patterns:
                soc_sections = self.extract_soc_version_sections(lines)
                for section in soc_sections:
                    section["file_path"] = str(file_path)
                    file_sections.append(section)

            if "npu_arch" in section_patterns:
                arch_sections = self.extract_npu_arch_sections(lines)
                for section in arch_sections:
                    section["file_path"] = str(file_path)
                    file_sections.append(section)

            if "add_config" in section_patterns:
                config_sections = self.extract_add_config_sections(lines)
                for section in config_sections:
                    section["file_path"] = str(file_path)
                    file_sections.append(section)

            if "keyword" in section_patterns:
                keyword_sections = self.extract_keyword_sections(lines)
                for section in keyword_sections:
                    section["file_path"] = str(file_path)
                    file_sections.append(section)

            # Build anchors from sections
            for section in file_sections:
                anchor_id, evidence_id = self.id_generator.next_pair(anchor_type="section")

                anchor = build_section_anchor(
                    anchor_id=anchor_id,
                    file_path=section["file_path"],
                    content=section["content"],
                    start_line=section["start_line"],
                    end_line=section["end_line"],
                    token_hash=section["token_hash"],
                    gene_deter="暂空",  # placeholder - pipeline assigns later
                    gene_target=[],  # placeholder - pipeline assigns later
                    evidence_id=evidence_id,
                    note="以下为section型标准结构",
                )

                # Add envidence_id for compatibility
                anchor["evidence"][0]["envidence_id"] = evidence_id

                # Add pattern-specific metadata to evidence (for pipeline use)
                if "socs" in section:
                    anchor["evidence"][0]["support_chips"] = {
                        "source": "platform_ascendc::SocVersion",
                        "socs": section["socs"],
                    }
                elif "arch_name" in section:
                    anchor["evidence"][0]["support_chips"] = {
                        "source": section["arch_macro"],
                        "arch": section["arch_name"],
                    }
                elif "chips" in section:
                    anchor["evidence"][0]["support_chips"] = {
                        "source": "AICore().AddConfig",
                        "chips": section["chips"],
                    }
                elif "source" in section:
                    # Keyword-based - store original keyword for registry matching
                    anchor["evidence"][0]["keyword_source"] = section["source"]

                anchors.append(anchor)
                logger.debug(f"  Created anchor: {anchor_id} from {file_path.name}")

        logger.info(f"Extracted {len(anchors)} section anchors")
        return anchors


if __name__ == "__main__":
    """Self-test for SectionExtractor."""
    import tempfile

    print("Testing SectionExtractor...")

    # Create mock operator directory
    with tempfile.TemporaryDirectory() as tmpdir:
        operator_path = Path(tmpdir) / "test_operator"
        operator_path.mkdir()

        # Create op_host directory with mock tiling file (SocVersion pattern)
        op_host_dir = operator_path / "op_host"
        op_host_dir.mkdir()

        tiling_file = op_host_dir / "test_tiling.cpp"
        tiling_code = """
if (curShortSocName == platform_ascendc::SocVersion::Ascend910B1) {
    // SOC-specific code
    int x = 1;
} else if (curShortSocName == platform_ascendc::SocVersion::Ascend910B2) {
    // Another SOC
    int y = 2;
}
"""
        tiling_file.write_text(tiling_code)

        # Create op_kernel directory with mock kernel file (NPU_ARCH pattern)
        op_kernel_dir = operator_path / "op_kernel"
        op_kernel_dir.mkdir()

        kernel_file = op_kernel_dir / "test_kernel.cpp"
        kernel_code = """
#if __NPU_ARCH__ == Ascend910B
    // Architecture-specific code
    int a = 1;
    int b = 2;
#endif
"""
        kernel_file.write_text(kernel_code)

        # Create def.cpp file (AddConfig pattern)
        def_file = op_host_dir / "test_def.cpp"
        def_code = """
void ConfigureChip() {
    AICore().AddConfig("Ascend910B1");
    AICore().AddConfig("Ascend910B2");
}
"""
        def_file.write_text(def_code)

        # Create file with keywords
        keyword_file = op_host_dir / "test_check.cpp"
        keyword_code = """
void CheckInputs() {
    SUPPORT_LIST(DT_FLOAT, DT_INT32);
    check_dtype(input_tensor);
    CheckFormat(input_tensor);
}
"""
        keyword_file.write_text(keyword_code)

        # Create output directory
        out_path = Path(tmpdir) / "output"
        out_path.mkdir()

        # Create PathConfig
        path_config = PathConfig(operator_path=operator_path, out_path=out_path)

        # Create DataLoader
        data_loader = DataLoader(path_config)

        # Create IdGenerator with default config
        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        # Create SectionExtractor
        extractor = SectionExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        # Test individual extraction methods
        print("\nTesting extract_generic_code_block...")
        lines = ["", "int x = 1;", "int y = 2;", "", "int z = 3;"]
        result = extractor.extract_generic_code_block(lines, 1)
        print(f"  start_line: {result['start_line']}, end_line: {result['end_line']}")
        assert result["start_line"] == 2
        assert result["end_line"] == 3
        print("  OK")

        # Test SocVersion extraction
        print("\nTesting extract_soc_version_sections...")
        lines = tiling_code.split("\n")
        soc_sections = extractor.extract_soc_version_sections(lines)
        print(f"  Found {len(soc_sections)} sections")
        assert len(soc_sections) == 1
        assert "Ascend910B1" in soc_sections[0]["socs"]
        assert "Ascend910B2" in soc_sections[0]["socs"]
        print("  OK")

        # Test NPU_ARCH extraction
        print("\nTesting extract_npu_arch_sections...")
        lines = kernel_code.split("\n")
        arch_sections = extractor.extract_npu_arch_sections(lines)
        print(f"  Found {len(arch_sections)} sections")
        assert len(arch_sections) == 1
        assert arch_sections[0]["arch_name"] == "Ascend910B"
        assert arch_sections[0]["arch_macro"] == "__NPU_ARCH__"
        print("  OK")

        # Test AddConfig extraction
        print("\nTesting extract_add_config_sections...")
        lines = def_code.split("\n")
        config_sections = extractor.extract_add_config_sections(lines)
        print(f"  Found {len(config_sections)} sections")
        assert len(config_sections) == 1
        assert "Ascend910B1" in config_sections[0]["chips"]
        assert "Ascend910B2" in config_sections[0]["chips"]
        print("  OK")

        # Test keyword extraction
        print("\nTesting extract_keyword_sections...")
        lines = keyword_code.split("\n")
        keyword_sections = extractor.extract_keyword_sections(lines)
        print(f"  Found {len(keyword_sections)} sections")
        assert len(keyword_sections) >= 1
        sources = [s["source"] for s in keyword_sections]
        assert "SUPPORT_LIST" in sources or "check_dtype" in sources or "CheckFormat" in sources
        print("  OK")

        # Test full extraction
        print("\nTesting full extract() method...")
        anchors = extractor.extract(operator_path)
        print(f"  Found {len(anchors)} anchors")
        for anchor in anchors:
            print(f"    - {anchor['anchor_id']}: {anchor['anchor_information']['level1']['file_path']}")
            assert "anchor_id" in anchor
            assert "anchor_information" in anchor
            assert "evidence" in anchor
            assert anchor["anchor_information"]["type"] == "section"
            assert len(anchor["evidence"]) > 0
            assert "evidence_id" in anchor["evidence"][0]
            assert "envidence_id" in anchor["evidence"][0]
            assert anchor["evidence"][0]["evidence_id"] == anchor["evidence"][0]["envidence_id"]
        print("  OK")

    print("\nSectionExtractor self-test passed!")
