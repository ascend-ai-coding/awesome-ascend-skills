# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
FileExtractor - Unified file extraction for all doc_parts.

Consolidates file extraction patterns from:
- constraint/data_parsing.py: create_file_anchor()
- support/extract_new_0331.py: extract_file_records()
- summary_function_proto/get_ancor2envidence.py: process_operator_files()
- sub_function/extract/extract_files.py: find_operator_files() + extract_file_info()
- sub_function/merge/merge_operator_files.py: convert_files_to_anchor_tuple()

Uses unified infrastructure from common/:
- file_discovery.py: find_operator_files()
- anchor_builder.py: build_file_anchor()
- data_loader.py: DataLoader
- id_generator.py: IdGenerator
"""

import logging
from pathlib import Path
from typing import Any, Optional

from src.snapshot_construction.anchor_construction.common.anchor_builder import (
    build_file_anchor,
)
from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader
from src.snapshot_construction.anchor_construction.common.file_discovery import (
    find_operator_files,
)
from src.snapshot_construction.anchor_construction.common.id_generator import (
    IdGenerator,
    IdConfig,
)
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig

logger = logging.getLogger(__name__)


class FileExtractor:
    """Unified file extractor for all doc_parts.

    Extracts file anchors from operator source files, using unified discovery
    and anchor building functions. Returns in-memory data (no file writing).

    The extractor does NOT assign gene_rule - that's the pipeline's job via
    GeneRuleRegistry. gene_deter and gene_target are placeholders.

    Attributes:
        path_config: PathConfig instance for path resolution
        data_loader: DataLoader instance for loading code_map, file_func, tree_id
        id_generator: IdGenerator instance for anchor_id and evidence_id
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ):
        """Initialize FileExtractor with required dependencies.

        Args:
            path_config: PathConfig instance containing operator and output paths
            data_loader: DataLoader instance for loading metadata (code_map, file_func, tree_id)
            id_generator: IdGenerator instance for generating anchor and evidence IDs
        """
        self.path_config = path_config
        self.data_loader = data_loader
        self.id_generator = id_generator
        logger.debug(
            f"FileExtractor initialized for operator: {path_config.operator_path}"
        )

    def extract(
        self,
        operator_path: Path,
        source_files: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Extract file anchors from operator directory.

        Discovers all operator source files using unified file_discovery functions,
        builds file anchors using build_file_anchor(), and returns list of anchor dicts.

        Args:
            operator_path: Path to operator source directory
            source_files: Optional list of source file paths/names to filter by.
                When provided, only files matching (by path string or basename) are kept.
                When None, all discovered files are returned (backward compatible).

        Returns:
            List of anchor dicts with structure:
            {
                "anchor_id": "anchor_file_001",
                "anchor_information": {
                    "type": "file",
                    "level1": {"file_path": ..., "file_name": ...},
                    "level2": {"code_map": ..., "token_hash": ..., "file_功能": ...},
                    "level3": {"tree_id": ...}
                },
                "evidence": [
                    {
                        "evidence_id": "ev_file_001",
                        "envidence_id": "ev_file_001",  # compat requirement
                        "evidence_information": {...},
                        "gene_rule": {"deter": "暂空", "target": []}  # placeholder
                    }
                ],
                "note": "以下为file型标准结构"
            }

        Note:
            gene_rule is placeholder (deter="暂空", target=[]).
            Pipeline assigns actual gene_rule via GeneRuleRegistry.
        """
        logger.info(f"Extracting file anchors from: {operator_path}")

        # Discover files using unified function
        files_with_source = find_operator_files(operator_path)
        logger.debug(f"Found {len(files_with_source)} operator files")

        if source_files is not None:
            source_names = {Path(s).name for s in source_files}
            source_strs = set(source_files)
            files_with_source = [
                (fp, tag)
                for fp, tag in files_with_source
                if str(fp) in source_strs or fp.name in source_names
            ]
            logger.debug(
                f"Filtered to {len(files_with_source)} files matching source_files"
            )

        anchors = []

        for file_path, source_pattern in files_with_source:
            file_name = file_path.name

            # Load metadata
            code_map = self.data_loader.load_code_map(file_name)
            file_func = self.data_loader.load_file_func(file_name)
            tree_id = self.data_loader.load_tree_id()
            file_content = self.data_loader.read_file_content(file_path)

            # Generate IDs
            anchor_id, evidence_id = self.id_generator.next_pair(anchor_type="file")

            # Build anchor with placeholder gene_rule
            anchor = build_file_anchor(
                anchor_id=anchor_id,
                file_path=str(file_path),
                file_name=file_name,
                code_map=code_map,
                file_content=file_content,
                file_func=file_func,
                tree_id=tree_id,
                gene_deter="暂空",  # placeholder - pipeline assigns later
                gene_target=[],  # placeholder - pipeline assigns later
                evidence_id=evidence_id,
                note="以下为file型标准结构",
            )

            # Add envidence_id for compatibility (duplicate of evidence_id)
            # Required by downstream change_detection code
            anchor["evidence"][0]["envidence_id"] = evidence_id

            anchors.append(anchor)
            logger.debug(f"  Created anchor: {anchor_id} for {file_name}")

        logger.info(f"Extracted {len(anchors)} file anchors")
        return anchors


if __name__ == "__main__":
    """Self-test for FileExtractor."""
    import tempfile
    from pathlib import Path

    print("Testing FileExtractor...")

    # Create mock operator directory
    with tempfile.TemporaryDirectory() as tmpdir:
        operator_path = Path(tmpdir) / "test_operator"
        operator_path.mkdir()

        # Create op_api directory with mock files
        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        # Create mock header file
        header_file = op_api_dir / "aclnn_test.h"
        header_file.write_text("// Mock header file\nint aclnnTest();\n")

        # Create mock cpp file
        cpp_file = op_api_dir / "aclnn_test.cpp"
        cpp_file.write_text("// Mock cpp file\nint aclnnTest() { return 0; }\n")

        # Create mock def.h
        def_h = op_api_dir / "def.h"
        def_h.write_text("// Mock def.h\n")

        # Create output directory
        out_path = Path(tmpdir) / "output"
        out_path.mkdir()

        # Create required subdirectories for DataLoader
        code_map_dir = out_path / "code_map" / "files"
        code_map_dir.mkdir(parents=True)

        # Create mock code_map file
        code_map_file = code_map_dir / "aclnn_test.h_map.json"
        code_map_file.write_text('{"nodes": [], "edges": []}')

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

        # Create FileExtractor
        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        # Extract anchors
        anchors = extractor.extract(operator_path)

        print(f"  Found {len(anchors)} anchors")
        for anchor in anchors:
            print(
                f"    - {anchor['anchor_id']}: {anchor['anchor_information']['level1']['file_name']}"
            )
            assert "anchor_id" in anchor
            assert "anchor_information" in anchor
            assert "evidence" in anchor
            assert anchor["anchor_information"]["type"] == "file"
            assert len(anchor["evidence"]) > 0
            assert "evidence_id" in anchor["evidence"][0]
            assert "envidence_id" in anchor["evidence"][0]
            assert (
                anchor["evidence"][0]["evidence_id"]
                == anchor["evidence"][0]["envidence_id"]
            )

        print("  All assertions passed!")

    print("\nFileExtractor self-test passed!")
