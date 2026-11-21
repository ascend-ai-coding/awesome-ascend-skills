# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
NoteExtractor - Unified note extraction for all doc_parts.

Consolidates note extraction patterns from:
- summary_function_proto/get_ancor2envidence.py: process_operator_notes()
- sub_function/extract/extract_notes.py: extract_comment_blocks()
- sub_function/merge/merge_note_extract.py: convert_notes_to_anchor_tuple()

Uses unified infrastructure from common/:
- file_discovery.py: find_header_files()
- anchor_builder.py: build_note_anchor()
- data_loader.py: DataLoader
- id_generator.py: IdGenerator
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.snapshot_construction.anchor_construction.common.anchor_builder import (
    build_note_anchor,
)
from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader
from src.snapshot_construction.anchor_construction.common.file_discovery import (
    find_header_files,
)
from src.snapshot_construction.anchor_construction.common.id_generator import (
    IdGenerator,
    IdConfig,
)
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig

logger = logging.getLogger(__name__)


class NoteExtractor:
    """Unified note extractor for all doc_parts.

    Extracts note anchors from Doxygen-style /** */ comment blocks in header files,
    using unified discovery and anchor building functions. Returns in-memory data
    (no file writing).

    The extractor does NOT assign gene_rule - that's the pipeline's job via
    GeneRuleRegistry. gene_deter and gene_target are placeholders.

    Attributes:
        path_config: PathConfig instance for path resolution
        data_loader: DataLoader instance for loading metadata
        id_generator: IdGenerator instance for anchor_id and evidence_id
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ):
        """Initialize NoteExtractor with required dependencies.

        Args:
            path_config: PathConfig instance containing operator and output paths
            data_loader: DataLoader instance for loading metadata
            id_generator: IdGenerator instance for generating anchor and evidence IDs
        """
        self.path_config = path_config
        self.data_loader = data_loader
        self.id_generator = id_generator
        logger.debug(
            f"NoteExtractor initialized for operator: {path_config.operator_path}"
        )

    def extract(
        self,
        operator_path: Path,
        source_files: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Extract note anchors from operator directory.

        Discovers header files using unified file_discovery.find_header_files(),
        extracts /** */ Doxygen comment blocks with @param/@return annotations,
        finds behind_code (function signature after comment), builds note anchors,
        and returns list of anchor dicts.

        Args:
            operator_path: Path to operator source directory
            source_files: Optional list of source file paths/names to filter by.
                When provided, only files matching (by path string or basename) are kept.
                When None, all discovered files are returned (backward compatible).

        Returns:
            List of anchor dicts with structure:
            {
                "anchor_id": "anchor_note_001",
                "anchor_information": {
                    "type": "note",
                    "level1": {"file_path": ..., "behind_code": ...},
                    "level2": {"token_hash": ..., "note_hash": ...}
                },
                "evidence": [
                    {
                        "evidence_id": "ev_note_001",
                        "envidence_id": "ev_note_001",  # compat requirement
                        "evidence_information": {...},
                        "gene_rule": {"deter": "暂空", "target": []}  # placeholder
                    }
                ],
                "note": "以下为note型标准结构"
            }

        Note:
            gene_rule is placeholder (deter="暂空", target=[]).
            Pipeline assigns actual gene_rule via GeneRuleRegistry.
        """
        logger.info(f"Extracting note anchors from: {operator_path}")

        # Discover header files using unified function
        header_files = find_header_files(operator_path)
        logger.debug(f"Found {len(header_files)} header files")

        if source_files is not None:
            source_names = {Path(s).name for s in source_files}
            source_strs = set(source_files)
            header_files = [
                fp
                for fp in header_files
                if str(fp) in source_strs or fp.name in source_names
            ]
            logger.debug(
                f"Filtered to {len(header_files)} header files matching source_files"
            )

        anchors = []

        for header_file in header_files:
            # Extract comment blocks from this file
            comment_blocks = self._extract_comment_blocks(header_file)
            logger.debug(
                f"  Found {len(comment_blocks)} comment blocks in {header_file.name}"
            )

            for block in comment_blocks:
                # Generate IDs
                anchor_id, evidence_id = self.id_generator.next_pair(anchor_type="note")

                # Build anchor with placeholder gene_rule
                anchor = build_note_anchor(
                    anchor_id=anchor_id,
                    file_path=str(header_file),
                    behind_code=block["behind_code"],
                    note_hash=block["note_content"],
                    token_hash=block["behind_code"],
                    gene_deter="暂空",  # placeholder - pipeline assigns later
                    gene_target=[],  # placeholder - pipeline assigns later
                    evidence_id=evidence_id,
                    note="以下为note型标准结构",
                )

                # Add envidence_id for compatibility (duplicate of evidence_id)
                # Required by downstream change_detection code
                anchor["evidence"][0]["envidence_id"] = evidence_id

                anchors.append(anchor)
                logger.debug(f"    Created anchor: {anchor_id}")

        logger.info(f"Extracted {len(anchors)} note anchors")
        return anchors

    def _extract_comment_blocks(self, file_path: Path) -> list[dict[str, str]]:
        """Parse a single header file and extract valid documentation blocks.

        Args:
            file_path: Path to header file

        Returns:
            List of dicts with keys:
            - file_path: str
            - start_line: int
            - end_line: int
            - behind_code: str (function signature after comment)
            - note_content: str (cleaned comment content)
            - function_signature: str (extracted signature)
        """
        result = []

        try:
            lines = self.data_loader.read_lines(file_path)
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return result

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for /** comment start
            if "/**" in line:
                start_line = i + 1

                comment_content = []
                comment_lines = [line]

                i += 1
                while i < len(lines):
                    comment_lines.append(lines[i])
                    if "*/" in lines[i]:
                        end_line = i + 1
                        i += 1
                        break
                    comment_content.append(lines[i])
                    i += 1
                else:
                    i += 1
                    continue

                # Check for @param or @return
                full_comment = "".join(comment_lines)
                has_param = "@param" in full_comment
                has_return = "@return" in full_comment

                if not (has_param or has_return):
                    logger.debug(
                        f"    Skipping comment block at lines {start_line}-{end_line}: no @param/@return"
                    )
                    continue

                # Extract behind_code (function signature after comment)
                behind_code = []

                # Skip empty lines and preprocessor directives
                while i < len(lines):
                    stripped = lines[i].strip()
                    if stripped and not stripped.startswith("#"):
                        break
                    i += 1

                # Collect function signature
                while i < len(lines):
                    line_content = lines[i]
                    behind_code.append(line_content)

                    stripped = line_content.strip()
                    if stripped.endswith(";") or (stripped and stripped[-1] == "}"):
                        i += 1
                        break

                    # Stop if we hit another comment
                    if "/**" in line_content or "/*" in line_content:
                        if behind_code:
                            behind_code = behind_code[:-1]
                        break

                    i += 1

                behind_code_str = "".join(behind_code).strip()

                # Extract function signature
                function_signature = self._extract_function_signature(behind_code_str)

                # Clean comment content
                note_content = self._clean_comment(comment_lines)

                result.append(
                    {
                        "file_path": str(file_path),
                        "start_line": start_line,
                        "end_line": end_line,
                        "behind_code": behind_code_str,
                        "note_content": note_content,
                        "function_signature": function_signature or "",
                    }
                )
            else:
                i += 1

        return result

    def _extract_function_signature(self, code: str) -> str | None:
        """Extract function signature from code block.

        Args:
            code: Code string to parse

        Returns:
            Extracted function signature or None
        """
        if not code:
            return None

        # Remove ACLNN_API and extern "C" markers
        code = re.sub(r"ACLNN_API\s*", "", code)
        code = re.sub(r'extern\s+"C"\s*', "", code)

        # Normalize whitespace
        code = re.sub(r"\s+", " ", code.strip())

        # Match function signature pattern
        pattern = r"(aclnnStatus|void|int|uint64_t)\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*;?"

        match = re.search(pattern, code)
        if match:
            signature = match.group(0).rstrip(";").strip()
            return signature

        # Fallback: simple pattern
        simple_pattern = r"(aclnnStatus|void|int|uint64_t)\s+[a-zA-Z0-9_]+"
        simple_match = re.search(simple_pattern, code)
        if simple_match:
            return simple_match.group(0)

        return None

    def _clean_comment(self, comment_lines: list[str]) -> str:
        """Remove comment markers and preserve content.

        Args:
            comment_lines: List of comment lines

        Returns:
            Cleaned comment content
        """
        content = []

        for line in comment_lines:
            cleaned = line
            cleaned = re.sub(r"/\*\*", "", cleaned)
            cleaned = re.sub(r"\*/", "", cleaned)
            cleaned = re.sub(r"^\s*\*", "", cleaned)
            cleaned = re.sub(r"^\s*\*?\s*", "", cleaned)
            content.append(cleaned)

        return "\n".join(content).strip()


if __name__ == "__main__":
    """Self-test for NoteExtractor."""
    import tempfile
    from pathlib import Path

    print("Testing NoteExtractor...")

    # Create mock operator directory
    with tempfile.TemporaryDirectory() as tmpdir:
        operator_path = Path(tmpdir) / "test_operator"
        operator_path.mkdir()

        # Create op_api directory with mock header file
        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        # Create mock header file with Doxygen comment
        header_file = op_api_dir / "aclnn_test.h"
        header_content = """
/**
 * @brief Test function
 * @param x Input parameter
 * @param y Output parameter
 * @return Status code
 */
aclnnStatus aclnnTest(void* x, void* y);

/**
 * @brief Another function
 * @param data Data pointer
 * @return void
 */
void anotherFunction(int* data);
"""
        header_file.write_text(header_content)

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

        # Create NoteExtractor
        extractor = NoteExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        # Extract anchors
        anchors = extractor.extract(operator_path)

        print(f"  Found {len(anchors)} anchors")
        for anchor in anchors:
            print(f"    - {anchor['anchor_id']}")
            print(f"      type: {anchor['anchor_information']['type']}")
            print(
                f"      behind_code: {anchor['anchor_information']['level1']['behind_code'][:50]}..."
            )
            assert "anchor_id" in anchor
            assert "anchor_information" in anchor
            assert "evidence" in anchor
            assert anchor["anchor_information"]["type"] == "note"
            assert len(anchor["evidence"]) > 0
            assert "evidence_id" in anchor["evidence"][0]
            assert "envidence_id" in anchor["evidence"][0]
            assert (
                anchor["evidence"][0]["evidence_id"]
                == anchor["evidence"][0]["envidence_id"]
            )

        print("  All assertions passed!")

    print("\nNoteExtractor self-test passed!")
