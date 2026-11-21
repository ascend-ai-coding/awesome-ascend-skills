# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Tests for extractor classes.

Tests cover:
- FileExtractor: File discovery, anchor structure, evidence IDs
- FunctionExtractor: Function extraction, anchor structure, call/callee/return data
- NoteExtractor: Note extraction from Doxygen comments, @param/@return parsing
"""

import tempfile
from pathlib import Path

import pytest

from src.snapshot_construction.anchor_construction.common.data_loader import DataLoader
from src.snapshot_construction.anchor_construction.common.id_generator import (
    IdConfig,
    IdGenerator,
)
from src.snapshot_construction.anchor_construction.common.path_config import PathConfig
from src.snapshot_construction.anchor_construction.extractors.file_extractor import (
    FileExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.function_extractor import (
    FunctionExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.note_extractor import (
    NoteExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.section_extractor import (
    CPP_KEYWORDS,
    PATTERN_ADD_CONFIG,
    PATTERN_CCE_AICORE,
    PATTERN_NPU_ARCH,
    PATTERN_SOC_VERSION,
    SECTION_MAPPING,
    SectionExtractor,
)


class TestFileExtractor:
    """Test suite for FileExtractor class."""

    @pytest.fixture
    def mock_operator_dir(self, tmp_path: Path) -> Path:
        """Create mock operator directory with standard structure.

        Creates:
        - op_api/aclnn_test.h
        - op_api/aclnn_test.cpp
        - op_api/def.h
        - op_graph/test_proto.h
        """
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        # Create op_api directory
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

        # Create op_graph directory
        op_graph_dir = operator_path / "op_graph"
        op_graph_dir.mkdir()

        # Create mock proto file
        proto_file = op_graph_dir / "test_proto.h"
        proto_file.write_text("// Mock proto file\n")

        return operator_path

    @pytest.fixture
    def mock_output_dir(self, tmp_path: Path) -> Path:
        """Create mock output directory with required subdirectories.

        Creates:
        - code_map/files/
        - tree_id/
        - organized_docs_with_descriptions/原始文档/
        - organized_docs_with_descriptions/算子说明/
        """
        out_path = tmp_path / "output"
        out_path.mkdir()

        # Create code_map directory
        code_map_dir = out_path / "code_map" / "files"
        code_map_dir.mkdir(parents=True)

        # Create tree_id directory
        tree_id_dir = out_path / "tree_id"
        tree_id_dir.mkdir()

        # Create original docs directory
        original_docs_dir = out_path / "tree_id" / "organized_docs_with_descriptions" / "原始文档"
        original_docs_dir.mkdir(parents=True)

        # Create operator note directory
        operator_note_dir = out_path / "tree_id" / "organized_docs_with_descriptions" / "算子说明"
        operator_note_dir.mkdir(parents=True)

        return out_path

    @pytest.fixture
    def path_config(self, mock_operator_dir: Path, mock_output_dir: Path) -> PathConfig:
        """Create PathConfig instance."""
        return PathConfig(operator_path=mock_operator_dir, out_path=mock_output_dir)

    @pytest.fixture
    def data_loader(self, path_config: PathConfig) -> DataLoader:
        """Create DataLoader instance."""
        return DataLoader(path_config)

    @pytest.fixture
    def id_generator(self) -> IdGenerator:
        """Create IdGenerator with default config."""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        return IdGenerator(config)

    @pytest.fixture
    def file_extractor(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ) -> FileExtractor:
        """Create FileExtractor instance."""
        return FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

    def test_file_extractor_initialization(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ):
        """Test FileExtractor initialization."""
        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        assert extractor.path_config == path_config
        assert extractor.data_loader == data_loader
        assert extractor.id_generator == id_generator

    def test_file_discovery_finds_correct_patterns(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test file discovery finds all expected file patterns."""
        anchors = file_extractor.extract(mock_operator_dir)

        # Should find 3 files: aclnn_test.h, aclnn_test.cpp, test_proto.h
        assert len(anchors) == 3

        # Check file names are present
        file_names = [a["anchor_information"]["level1"]["file_name"] for a in anchors]
        assert "aclnn_test.h" in file_names
        assert "aclnn_test.cpp" in file_names
        assert "test_proto.h" in file_names

    def test_anchor_structure_has_required_fields(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test anchor structure has anchor_id, anchor_information, evidence."""
        anchors = file_extractor.extract(mock_operator_dir)

        for anchor in anchors:
            # Check required top-level fields
            assert "anchor_id" in anchor
            assert "anchor_information" in anchor
            assert "evidence" in anchor

            # Check anchor_information structure
            anchor_info = anchor["anchor_information"]
            assert "type" in anchor_info
            assert anchor_info["type"] == "file"
            assert "level1" in anchor_info
            assert "level2" in anchor_info
            assert "level3" in anchor_info

            # Check level1 fields
            level1 = anchor_info["level1"]
            assert "file_path" in level1
            assert "file_name" in level1

            # Check level2 fields
            level2 = anchor_info["level2"]
            assert "code_map" in level2
            assert "token_hash" in level2
            assert "file_功能" in level2

            # Check level3 fields
            level3 = anchor_info["level3"]
            assert "tree_id" in level3

    def test_evidence_has_both_ids(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test evidence has both evidence_id and envidence_id (compat requirement)."""
        anchors = file_extractor.extract(mock_operator_dir)

        for anchor in anchors:
            evidence_list = anchor["evidence"]
            assert len(evidence_list) > 0

            for evidence in evidence_list:
                # Check both IDs are present
                assert "evidence_id" in evidence
                assert "envidence_id" in evidence

                # Check they are equal (compat requirement)
                assert evidence["evidence_id"] == evidence["envidence_id"]

                # Check ID format
                assert evidence["evidence_id"].startswith("ev_file_")

    def test_anchor_id_format(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test anchor_id follows expected format: anchor_file_{seq:03d}."""
        anchors = file_extractor.extract(mock_operator_dir)

        for i, anchor in enumerate(anchors, start=1):
            anchor_id = anchor["anchor_id"]
            expected_id = f"anchor_file_{i:03d}"
            assert anchor_id == expected_id

    def test_gene_rule_is_placeholder(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test gene_rule is placeholder (pipeline assigns later)."""
        anchors = file_extractor.extract(mock_operator_dir)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            gene_rule = evidence["gene_rule"]

            # Check gene_rule structure
            assert "deter" in gene_rule
            assert "target" in gene_rule

            # Check placeholder values
            assert gene_rule["deter"] == "暂空"
            assert gene_rule["target"] == []

    def test_note_field_present(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test note field is present with standard message."""
        anchors = file_extractor.extract(mock_operator_dir)

        for anchor in anchors:
            assert "note" in anchor
            assert anchor["note"] == "以下为file型标准结构"

    def test_empty_operator_directory(
        self,
        file_extractor: FileExtractor,
        tmp_path: Path,
    ):
        """Test extraction from empty operator directory."""
        empty_operator = tmp_path / "empty_operator"
        empty_operator.mkdir()

        anchors = file_extractor.extract(empty_operator)

        # Should return empty list
        assert len(anchors) == 0

    def test_with_mock_code_map(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
        mock_output_dir: Path,
    ):
        """Test extraction with mock code_map data."""
        # Create mock code_map file for aclnn_test.h
        code_map_dir = mock_output_dir / "code_map" / "files"
        code_map_file = code_map_dir / "aclnn_test.h_map.json"
        code_map_data = {
            "nodes": [
                {"id": "func1", "type": "function"},
                {"id": "class1", "type": "class"},
            ],
            "edges": [
                {"source": "func1", "type": "calls", "target": "func2"},
            ],
        }
        import json

        code_map_file.write_text(json.dumps(code_map_data))

        # Extract anchors
        anchors = file_extractor.extract(mock_operator_dir)

        # Find anchor for aclnn_test.h
        aclnn_anchor = None
        for anchor in anchors:
            if anchor["anchor_information"]["level1"]["file_name"] == "aclnn_test.h":
                aclnn_anchor = anchor
                break

        assert aclnn_anchor is not None

        # Check code_map is loaded
        code_map = aclnn_anchor["anchor_information"]["level2"]["code_map"]
        assert "nodes" in code_map
        assert "edges" in code_map
        assert len(code_map["nodes"]) > 0

    def test_file_content_loaded(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test file content is loaded into evidence."""
        anchors = file_extractor.extract(mock_operator_dir)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            evidence_info = evidence["evidence_information"]

            # Check file_content is present
            assert "file_content" in evidence_info
            assert len(evidence_info["file_content"]) > 0

    def test_id_generator_shared_counters(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test ID generator uses shared counters (anchor and evidence IDs match)."""
        anchors = file_extractor.extract(mock_operator_dir)

        for anchor in anchors:
            anchor_id = anchor["anchor_id"]
            evidence_id = anchor["evidence"][0]["evidence_id"]

            # Extract sequence numbers
            anchor_seq = anchor_id.split("_")[-1]
            evidence_seq = evidence_id.split("_")[-1]

            # Should match (shared counters)
            assert anchor_seq == evidence_seq

    def test_multiple_extractions_reset_counter(
        self,
        file_extractor: FileExtractor,
        mock_operator_dir: Path,
    ):
        """Test multiple extractions reset counter each time."""
        # First extraction
        anchors1 = file_extractor.extract(mock_operator_dir)
        first_anchor_id = anchors1[0]["anchor_id"]

        # Reset counter
        file_extractor.id_generator.reset()

        # Second extraction
        anchors2 = file_extractor.extract(mock_operator_dir)
        second_anchor_id = anchors2[0]["anchor_id"]

        # Should start from 001 again after reset
        assert first_anchor_id == second_anchor_id
        assert first_anchor_id == "anchor_file_001"


class TestFileExtractorIntegration:
    """Integration tests for FileExtractor with real file patterns."""

    def test_op_host_op_api_alternative_location(self, tmp_path: Path):
        """Test file discovery finds op_api under op_host directory."""
        # Create operator with op_host/op_api structure
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        op_host_dir = operator_path / "op_host"
        op_host_dir.mkdir()

        op_api_dir = op_host_dir / "op_api"
        op_api_dir.mkdir()

        # Create mock file
        header_file = op_api_dir / "aclnn_test.h"
        header_file.write_text("// Mock header\n")

        # Create output directory
        out_path = tmp_path / "output"
        out_path.mkdir()

        code_map_dir = out_path / "code_map" / "files"
        code_map_dir.mkdir(parents=True)

        # Create PathConfig and dependencies
        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        # Create extractor
        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        # Extract anchors
        anchors = extractor.extract(operator_path)

        # Should find the file
        assert len(anchors) == 1
        assert anchors[0]["anchor_information"]["level1"]["file_name"] == "aclnn_test.h"

    def test_tiling_h_file_discovery(self, tmp_path: Path):
        """Test file discovery finds tiling.h files in op_host directory."""
        # Create operator with tiling.h
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        op_host_dir = operator_path / "op_host"
        op_host_dir.mkdir()

        tiling_h = op_host_dir / "tiling.h"
        tiling_h.write_text("// Mock tiling header\n")

        # Create output directory
        out_path = tmp_path / "output"
        out_path.mkdir()

        code_map_dir = out_path / "code_map" / "files"
        code_map_dir.mkdir(parents=True)

        # Create dependencies
        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        anchors = extractor.extract(operator_path)

        # Should find tiling.h
        assert len(anchors) == 1
        assert anchors[0]["anchor_information"]["level1"]["file_name"] == "tiling.h"


class TestFileExtractorEdgeCases:
    """Edge case tests for FileExtractor."""

    def test_missing_code_map_file(self, tmp_path: Path):
        """Test extraction when code_map file is missing."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        header_file = op_api_dir / "aclnn_test.h"
        header_file.write_text("// Mock header\n")

        out_path = tmp_path / "output"
        out_path.mkdir()

        # Don't create code_map directory - should handle gracefully

        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        anchors = extractor.extract(operator_path)

        # Should still extract anchor with empty code_map
        assert len(anchors) == 1
        code_map = anchors[0]["anchor_information"]["level2"]["code_map"]
        assert code_map == {"nodes": [], "edges": []}

    def test_missing_file_func(self, tmp_path: Path):
        """Test extraction when file_func is missing."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        header_file = op_api_dir / "aclnn_test.h"
        header_file.write_text("// Mock header\n")

        out_path = tmp_path / "output"
        out_path.mkdir()

        code_map_dir = out_path / "code_map" / "files"
        code_map_dir.mkdir(parents=True)

        # Don't create original docs directory - should handle gracefully

        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        anchors = extractor.extract(operator_path)

        # Should still extract anchor with empty file_func
        assert len(anchors) == 1
        file_func = anchors[0]["anchor_information"]["level2"]["file_功能"]
        assert file_func == ""

    def test_missing_tree_id(self, tmp_path: Path):
        """Test extraction when tree_id is missing."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        header_file = op_api_dir / "aclnn_test.h"
        header_file.write_text("// Mock header\n")

        out_path = tmp_path / "output"
        out_path.mkdir()

        code_map_dir = out_path / "code_map" / "files"
        code_map_dir.mkdir(parents=True)

        # Don't create tree_id directory - should handle gracefully

        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        extractor = FileExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        anchors = extractor.extract(operator_path)

        # Should still extract anchor with empty tree_id
        assert len(anchors) == 1
        tree_id = anchors[0]["anchor_information"]["level3"]["tree_id"]
        assert tree_id == ""


class TestNoteExtractor:
    """Test suite for NoteExtractor class."""

    @pytest.fixture
    def mock_operator_with_notes(self, tmp_path: Path) -> Path:
        """Create mock operator directory with header files containing Doxygen comments."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        # Create op_api directory
        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        # Create mock header file with Doxygen comments
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

        return operator_path

    @pytest.fixture
    def mock_output_dir(self, tmp_path: Path) -> Path:
        """Create mock output directory."""
        out_path = tmp_path / "output"
        out_path.mkdir()
        return out_path

    @pytest.fixture
    def path_config(self, mock_operator_with_notes: Path, mock_output_dir: Path) -> PathConfig:
        """Create PathConfig instance."""
        return PathConfig(operator_path=mock_operator_with_notes, out_path=mock_output_dir)

    @pytest.fixture
    def data_loader(self, path_config: PathConfig) -> DataLoader:
        """Create DataLoader instance."""
        return DataLoader(path_config)

    @pytest.fixture
    def id_generator(self) -> IdGenerator:
        """Create IdGenerator with default config."""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        return IdGenerator(config)

    @pytest.fixture
    def note_extractor(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ) -> NoteExtractor:
        """Create NoteExtractor instance."""
        return NoteExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

    def test_note_extractor_initialization(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ):
        """Test NoteExtractor initialization."""
        extractor = NoteExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        assert extractor.path_config == path_config
        assert extractor.data_loader == data_loader
        assert extractor.id_generator == id_generator

    def test_note_extractor_with_mock_header(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test NoteExtractor extracts anchors from mock header file."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        # Should find 2 comment blocks with @param/@return
        assert len(anchors) == 2

        # Verify first anchor structure
        anchor1 = anchors[0]
        assert "anchor_id" in anchor1
        assert "anchor_information" in anchor1
        assert "evidence" in anchor1
        assert anchor1["anchor_information"]["type"] == "note"
        assert anchor1["anchor_id"] == "anchor_note_001"

        # Verify evidence structure
        assert len(anchor1["evidence"]) > 0
        assert "evidence_id" in anchor1["evidence"][0]
        assert "envidence_id" in anchor1["evidence"][0]
        assert anchor1["evidence"][0]["evidence_id"] == "ev_note_001"
        assert anchor1["evidence"][0]["evidence_id"] == anchor1["evidence"][0]["envidence_id"]

        # Verify gene_rule is placeholder
        assert "gene_rule" in anchor1["evidence"][0]
        assert anchor1["evidence"][0]["gene_rule"]["deter"] == "暂空"
        assert anchor1["evidence"][0]["gene_rule"]["target"] == []

    def test_note_anchor_has_type_note(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test note anchor has anchor_information with type='note'."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        # Verify all anchors have type="note"
        assert len(anchors) > 0
        for anchor in anchors:
            assert anchor["anchor_information"]["type"] == "note"

    def test_note_behind_code_extraction(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test behind_code extraction (function signature after comment)."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        # Verify behind_code contains function signature
        assert len(anchors) > 0

        # First anchor should have aclnnTest signature
        anchor1 = anchors[0]
        behind_code1 = anchor1["anchor_information"]["level1"]["behind_code"]
        assert "aclnnTest" in behind_code1
        assert "aclnnStatus" in behind_code1

        # Second anchor should have anotherFunction signature
        anchor2 = anchors[1]
        behind_code2 = anchor2["anchor_information"]["level1"]["behind_code"]
        assert "anotherFunction" in behind_code2

    def test_note_param_return_parsing(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test @param/@return parsing."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        # Verify note_content contains @param and @return
        assert len(anchors) > 0

        # First anchor should have @param x, @param y, @return
        anchor1 = anchors[0]
        note_hash1 = anchor1["anchor_information"]["level2"]["note_hash"]
        assert "@param" in note_hash1
        assert "@return" in note_hash1

        # Second anchor should have @param data, @return
        anchor2 = anchors[1]
        note_hash2 = anchor2["anchor_information"]["level2"]["note_hash"]
        assert "@param" in note_hash2
        assert "@return" in note_hash2

    def test_note_evidence_id_and_envidence_id_present(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test evidence_id and envidence_id both present."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        # Verify both evidence_id and envidence_id are present
        assert len(anchors) > 0
        for anchor in anchors:
            evidence = anchor["evidence"][0]
            assert "evidence_id" in evidence
            assert "envidence_id" in evidence
            assert evidence["evidence_id"] == evidence["envidence_id"]

    def test_note_extractor_skips_comments_without_param_or_return(
        self,
        tmp_path: Path,
    ):
        """Test NoteExtractor skips comment blocks without @param or @return."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        # Create op_api directory with mock header file
        op_api_dir = operator_path / "op_api"
        op_api_dir.mkdir()

        # Create mock header file with Doxygen comment (no @param/@return)
        header_file = op_api_dir / "aclnn_test.h"
        header_content = """
/**
 * @brief Test function
 * This is a brief description only.
 */
aclnnStatus aclnnTest(void* x);

/**
 * @brief Valid function
 * @param x Input parameter
 * @return Status code
 */
aclnnStatus aclnnValid(void* x);
"""
        header_file.write_text(header_content)

        # Create output directory
        out_path = tmp_path / "output"
        out_path.mkdir()

        # Create dependencies
        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
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

        # Verify only one anchor (the one with @param/@return)
        assert len(anchors) == 1
        assert "aclnnValid" in anchors[0]["anchor_information"]["level1"]["behind_code"]

    def test_note_extractor_empty_operator(
        self,
        tmp_path: Path,
    ):
        """Test NoteExtractor returns empty list for operator with no header files."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        # Create output directory
        out_path = tmp_path / "output"
        out_path.mkdir()

        # Create dependencies
        path_config = PathConfig(operator_path=operator_path, out_path=out_path)
        data_loader = DataLoader(path_config)
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

        # Verify empty list
        assert len(anchors) == 0

    def test_note_anchor_id_format(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test anchor_id follows expected format: anchor_note_{seq:03d}."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        for i, anchor in enumerate(anchors, start=1):
            anchor_id = anchor["anchor_id"]
            expected_id = f"anchor_note_{i:03d}"
            assert anchor_id == expected_id

    def test_note_gene_rule_is_placeholder(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test gene_rule is placeholder (pipeline assigns later)."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            gene_rule = evidence["gene_rule"]

            # Check gene_rule structure
            assert "deter" in gene_rule
            assert "target" in gene_rule

            # Check placeholder values
            assert gene_rule["deter"] == "暂空"
            assert gene_rule["target"] == []

    def test_note_field_present(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test note field is present with standard message."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        for anchor in anchors:
            assert "note" in anchor
            assert anchor["note"] == "以下为note型标准结构"

    def test_note_id_generator_shared_counters(
        self,
        note_extractor: NoteExtractor,
        mock_operator_with_notes: Path,
    ):
        """Test ID generator uses shared counters (anchor and evidence IDs match)."""
        anchors = note_extractor.extract(mock_operator_with_notes)

        for anchor in anchors:
            anchor_id = anchor["anchor_id"]
            evidence_id = anchor["evidence"][0]["evidence_id"]

            # Extract sequence numbers
            anchor_seq = anchor_id.split("_")[-1]
            evidence_seq = evidence_id.split("_")[-1]

            # Should match (shared counters)
            assert anchor_seq == evidence_seq


# =============================================================================
# FunctionExtractor Tests
# =============================================================================


@pytest.fixture
def mock_cpp_file(tmp_path: Path) -> Path:
    """Create a mock C++ file with aclnnStatus function.

    Args:
        tmp_path: Temporary path fixture from pytest

    Returns:
        Path to the created mock operator directory
    """
    # Create operator directory structure
    operator_dir = tmp_path / "ops" / "math" / "abs"
    operator_dir.mkdir(parents=True)

    # Create aclnn_abs.cpp file
    cpp_file = operator_dir / "aclnn_abs.cpp"

    cpp_content = """
#include "aclnn_abs.h"

aclnnStatus aclnnAbs(void* workspace, uint64_t workspaceSize, aclTensor* tensor) {
    // Implementation
    if (workspace == nullptr) {
        return ACLNN_STATUS_ERROR;
    }

    // Process tensor
    return ACLNN_STATUS_OK;
}

aclnnStatus aclnnAbsGetWorkspaceSize(aclTensor* tensor, uint64_t* workspaceSize) {
    // Calculate workspace size
    *workspaceSize = 1024;
    return ACLNN_STATUS_OK;
}
"""

    cpp_file.write_text(cpp_content, encoding="utf-8")

    return operator_dir


@pytest.fixture
def mock_functions_json(tmp_path: Path) -> Path:
    """Create mock functions.json file.

    Args:
        tmp_path: Temporary path fixture from pytest

    Returns:
        Path to the output directory containing functions.json
    """
    # Create code_map directory
    code_map_dir = tmp_path / "code_map"
    code_map_dir.mkdir(parents=True)

    # Create functions.json
    functions_json = code_map_dir / "functions.json"

    functions_data = [
        {
            "id": "aclnnAbs",
            "calls": ["helper1", "helper2"],
            "called_by": ["main"],
            "returns": ["ACLNN_STATUS_OK", "ACLNN_STATUS_ERROR"],
        },
        {
            "id": "aclnnAbsGetWorkspaceSize",
            "calls": ["calculateSize"],
            "called_by": ["aclnnAbs"],
            "returns": ["ACLNN_STATUS_OK"],
        },
    ]

    import json

    functions_json.write_text(json.dumps(functions_data, indent=2), encoding="utf-8")

    return tmp_path


@pytest.fixture
def function_extractor(tmp_path: Path, mock_cpp_file: Path, mock_functions_json: Path) -> FunctionExtractor:
    """Create FunctionExtractor instance with mock data.

    Args:
        tmp_path: Temporary path fixture
        mock_cpp_file: Mock operator directory fixture
        mock_functions_json: Mock functions.json fixture

    Returns:
        FunctionExtractor instance
    """
    path_config = PathConfig(mock_cpp_file, mock_functions_json)
    data_loader = DataLoader(path_config)

    id_config = IdConfig(
        type_prefix_in_anchor=True,
        type_prefix_in_evidence=True,
        shared_counters=True,
    )
    id_generator = IdGenerator(id_config)

    return FunctionExtractor(path_config, data_loader, id_generator)


class TestFunctionExtractor:
    """Test suite for FunctionExtractor."""

    def test_extract_returns_list(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that extract() returns a list."""
        anchors = function_extractor.extract(mock_cpp_file)

        assert isinstance(anchors, list)
        assert len(anchors) > 0

    def test_anchor_has_function_type(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that anchor_information has type='function'."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            assert "anchor_information" in anchor
            assert anchor["anchor_information"]["type"] == "function"

    def test_anchor_has_level1_level2_level3(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that anchor has level1, level2, level3 structure."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            anchor_info = anchor["anchor_information"]

            assert "level1" in anchor_info
            assert "level2" in anchor_info
            assert "level3" in anchor_info

            # Check level1 structure
            level1 = anchor_info["level1"]
            assert "file_path" in level1
            assert "function_name" in level1

            # Check level2 structure
            level2 = anchor_info["level2"]
            assert "call" in level2
            assert "callee" in level2
            assert "return" in level2

            # Check level3 structure
            level3 = anchor_info["level3"]
            assert "function_sign" in level3

    def test_evidence_has_calls_called_by_returns(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that evidence has calls, called_by, returns populated."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            assert "evidence" in anchor
            assert len(anchor["evidence"]) > 0

            evidence = anchor["evidence"][0]
            assert "evidence_information" in evidence

            evidence_info = evidence["evidence_information"]
            assert "call" in evidence_info
            assert "callee" in evidence_info
            assert "return" in evidence_info

    def test_evidence_id_present(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that evidence_id is present in evidence."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            assert "evidence_id" in evidence
            assert evidence["evidence_id"].startswith("ev_function_")

    def test_envidence_id_present(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that envidence_id is present in evidence (legacy compatibility)."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            assert "envidence_id" in evidence
            assert evidence["envidence_id"].startswith("ev_function_")

    def test_anchor_id_format(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that anchor_id follows format: anchor_function_{seq:03d}."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            assert "anchor_id" in anchor
            assert anchor["anchor_id"].startswith("anchor_function_")
            # Check sequence format (3 digits)
            seq_part = anchor["anchor_id"].split("_")[-1]
            assert len(seq_part) == 3
            assert seq_part.isdigit()

    def test_gene_rule_empty(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that gene_rule is empty (assigned by downstream)."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            assert "gene_rule" in evidence

            gene_rule = evidence["gene_rule"]
            assert gene_rule["deter"] == ""
            assert gene_rule["target"] == []

    def test_function_name_extraction(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that function names are correctly extracted."""
        anchors = function_extractor.extract(mock_cpp_file)

        function_names = [anchor["anchor_information"]["level1"]["function_name"] for anchor in anchors]

        # Should extract aclnnAbs and aclnnAbsGetWorkspaceSize
        assert "aclnnAbs" in function_names
        assert "aclnnAbsGetWorkspaceSize" in function_names

    def test_calls_populated_from_functions_json(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that calls are populated from functions.json."""
        anchors = function_extractor.extract(mock_cpp_file)

        # Find aclnnAbs anchor
        aclnn_abs_anchor = None
        for anchor in anchors:
            if anchor["anchor_information"]["level1"]["function_name"] == "aclnnAbs":
                aclnn_abs_anchor = anchor
                break

        assert aclnn_abs_anchor is not None

        # Check calls from functions.json
        calls = aclnn_abs_anchor["anchor_information"]["level2"]["call"]
        assert "helper1" in calls
        assert "helper2" in calls

    def test_called_by_populated_from_functions_json(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that called_by is populated from functions.json."""
        anchors = function_extractor.extract(mock_cpp_file)

        # Find aclnnAbs anchor
        aclnn_abs_anchor = None
        for anchor in anchors:
            if anchor["anchor_information"]["level1"]["function_name"] == "aclnnAbs":
                aclnn_abs_anchor = anchor
                break

        assert aclnn_abs_anchor is not None

        # Check called_by from functions.json
        called_by = aclnn_abs_anchor["anchor_information"]["level2"]["callee"]
        assert "main" in called_by

    def test_returns_populated_from_functions_json(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that returns are populated from functions.json."""
        anchors = function_extractor.extract(mock_cpp_file)

        # Find aclnnAbs anchor
        aclnn_abs_anchor = None
        for anchor in anchors:
            if anchor["anchor_information"]["level1"]["function_name"] == "aclnnAbs":
                aclnn_abs_anchor = anchor
                break

        assert aclnn_abs_anchor is not None

        # Check returns from functions.json
        returns = aclnn_abs_anchor["anchor_information"]["level2"]["return"]
        assert "ACLNN_STATUS_OK" in returns
        assert "ACLNN_STATUS_ERROR" in returns

    def test_relative_path_format(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that file_path is relative from ops/ onwards."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            file_path = anchor["anchor_information"]["level1"]["file_path"]
            # Should start with ops/
            assert file_path.startswith("ops/")
            # Should contain math/abs
            assert "math/abs" in file_path

    def test_note_present(self, function_extractor: FunctionExtractor, mock_cpp_file: Path):
        """Test that note is present in anchor."""
        anchors = function_extractor.extract(mock_cpp_file)

        for anchor in anchors:
            assert "note" in anchor
            assert anchor["note"] == "以下为function型标准结构"

    def test_empty_functions_json(self, tmp_path: Path):
        """Test extraction when functions.json is empty."""
        # Create operator directory
        operator_dir = tmp_path / "ops" / "math" / "abs"
        operator_dir.mkdir(parents=True)

        # Create C++ file
        cpp_file = operator_dir / "aclnn_abs.cpp"
        cpp_file.write_text("""
aclnnStatus aclnnAbs() {
    return ACLNN_STATUS_OK;
}
""")

        # Create empty code_map directory (no functions.json)
        out_path = tmp_path / "output"
        out_path.mkdir()
        code_map_dir = out_path / "code_map"
        code_map_dir.mkdir()

        path_config = PathConfig(operator_dir, out_path)
        data_loader = DataLoader(path_config)

        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        extractor = FunctionExtractor(path_config, data_loader, id_generator)
        anchors = extractor.extract(operator_dir)

        # Should still extract functions, but with empty call/callee/return
        assert len(anchors) > 0

        for anchor in anchors:
            level2 = anchor["anchor_information"]["level2"]
            assert level2["call"] == []
            assert level2["callee"] == []
            assert level2["return"] == []

    def test_no_cpp_files(self, tmp_path: Path):
        """Test extraction when no aclnn_*.cpp files exist."""
        # Create operator directory without C++ files
        operator_dir = tmp_path / "ops" / "math" / "abs"
        operator_dir.mkdir(parents=True)

        # Create empty code_map directory
        out_path = tmp_path / "output"
        out_path.mkdir()
        code_map_dir = out_path / "code_map"
        code_map_dir.mkdir()

        path_config = PathConfig(operator_dir, out_path)
        data_loader = DataLoader(path_config)

        id_config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        id_generator = IdGenerator(id_config)

        extractor = FunctionExtractor(path_config, data_loader, id_generator)
        anchors = extractor.extract(operator_dir)

        # Should return empty list
        assert anchors == []


# =============================================================================
# SectionExtractor Tests
# =============================================================================


class TestSectionExtractorPatterns:
    """Test pattern constants and mappings."""

    def test_pattern_add_config(self):
        """测试 AICore().AddConfig 正则模式"""
        # Valid patterns
        assert PATTERN_ADD_CONFIG.search('AICore().AddConfig("Ascend910B1")')
        assert PATTERN_ADD_CONFIG.search('AICore().AddConfig("Ascend910B2")')
        assert PATTERN_ADD_CONFIG.search('AICore ().AddConfig ("chip_name")')

        # Extract chip name
        m = PATTERN_ADD_CONFIG.search('AICore().AddConfig("Ascend910B1")')
        assert m.group(1) == "Ascend910B1"

        # Invalid patterns
        assert not PATTERN_ADD_CONFIG.search('AddConfig("chip")')  # Missing AICore()
        assert not PATTERN_ADD_CONFIG.search("AICore().AddConfig(chip)")  # Missing quotes

    def test_pattern_soc_version(self):
        """测试 SocVersion 正则模式"""
        # Valid patterns
        assert PATTERN_SOC_VERSION.search("curShortSocName == platform_ascendc::SocVersion::Ascend910B1")
        assert PATTERN_SOC_VERSION.search("curShortSocName == platform_ascendc::SocVersion::Ascend910B2")

        # Extract SOC name
        m = PATTERN_SOC_VERSION.search("curShortSocName == platform_ascendc::SocVersion::Ascend910B1")
        assert m.group(1) == "Ascend910B1"

        # Invalid patterns
        assert not PATTERN_SOC_VERSION.search("SocVersion::Ascend910B1")  # Missing curShortSocName
        assert not PATTERN_SOC_VERSION.search("curShortSocName == Ascend910B1")  # Missing SocVersion::

    def test_pattern_npu_arch(self):
        """测试 __NPU_ARCH__ 正则模式"""
        # Valid patterns
        assert PATTERN_NPU_ARCH.search("#if __NPU_ARCH__ == Ascend910B")
        assert PATTERN_NPU_ARCH.search("# if __NPU_ARCH__ == Ascend910B2")

        # Extract arch name
        m = PATTERN_NPU_ARCH.search("#if __NPU_ARCH__ == Ascend910B")
        assert m.group(1) == "Ascend910B"

        # Invalid patterns
        assert not PATTERN_NPU_ARCH.search("__NPU_ARCH__ == Ascend910B")  # Missing #if
        assert not PATTERN_NPU_ARCH.search("#if __NPU_ARCH__")  # Missing comparison

    def test_pattern_cce_aicore(self):
        """测试 __CCE_AICORE__ 正则模式"""
        # Valid patterns
        assert PATTERN_CCE_AICORE.search("#if __CCE_AICORE__ == Ascend910B")
        assert PATTERN_CCE_AICORE.search("#elif __CCE_AICORE__ == Ascend910B2")

        # Extract arch name
        m = PATTERN_CCE_AICORE.search("#if __CCE_AICORE__ == Ascend910B")
        assert m.group(2) == "Ascend910B"

        # Invalid patterns
        assert not PATTERN_CCE_AICORE.search("__CCE_AICORE__ == Ascend910B")  # Missing #if/#elif

    def test_section_mapping(self):
        """测试 SECTION_MAPPING 关键词映射"""
        assert SECTION_MAPPING["SUPPORT_LIST"] == "dtype"
        assert SECTION_MAPPING["check_dtype"] == "dtype"
        assert SECTION_MAPPING["CheckFormat"] == "dmat"
        assert SECTION_MAPPING["CheckShape"] == "shape"
        assert SECTION_MAPPING["continuous"] == "tensor"
        assert SECTION_MAPPING["CheckParams"] == "errorcode"

    def test_cpp_keywords(self):
        """测试 CPP_KEYWORDS 列表"""
        assert "SUPPORT_LIST" in CPP_KEYWORDS
        assert "check_dtype" in CPP_KEYWORDS
        assert "CheckFormat" in CPP_KEYWORDS
        assert "CheckShape" in CPP_KEYWORDS
        assert "continuous" in CPP_KEYWORDS
        assert "CheckParams" in CPP_KEYWORDS
        assert len(CPP_KEYWORDS) == 6


class TestSectionExtractor:
    """Test suite for SectionExtractor class."""

    @pytest.fixture
    def mock_operator_with_sections(self, tmp_path: Path) -> Path:
        """Create mock operator directory with various section patterns."""
        operator_path = tmp_path / "test_operator"
        operator_path.mkdir()

        # Create op_host directory
        op_host_dir = operator_path / "op_host"
        op_host_dir.mkdir()

        # SocVersion pattern
        tiling_file = op_host_dir / "test_tiling.cpp"
        tiling_file.write_text("""
if (curShortSocName == platform_ascendc::SocVersion::Ascend910B1) {
    int x = 1;
} else if (curShortSocName == platform_ascendc::SocVersion::Ascend910B2) {
    int y = 2;
}
""")

        # AddConfig pattern
        def_file = op_host_dir / "test_def.cpp"
        def_file.write_text("""
void ConfigureChip() {
    AICore().AddConfig("Ascend910B1");
    AICore().AddConfig("Ascend910B2");
}
""")

        # Keyword pattern
        check_file = op_host_dir / "test_check.cpp"
        check_file.write_text("""
void CheckInputs() {
    SUPPORT_LIST(DT_FLOAT, DT_INT32);
    check_dtype(input_tensor);
    CheckFormat(input_tensor);
}
""")

        # Create op_kernel directory
        op_kernel_dir = operator_path / "op_kernel"
        op_kernel_dir.mkdir()

        # NPU_ARCH pattern
        kernel_file = op_kernel_dir / "test_kernel.cpp"
        kernel_file.write_text("""
#if __NPU_ARCH__ == Ascend910B
    int a = 1;
    int b = 2;
#endif
""")

        return operator_path

    @pytest.fixture
    def mock_output_dir(self, tmp_path: Path) -> Path:
        """Create mock output directory."""
        out_path = tmp_path / "output"
        out_path.mkdir()
        return out_path

    @pytest.fixture
    def path_config(self, mock_operator_with_sections: Path, mock_output_dir: Path) -> PathConfig:
        """Create PathConfig instance."""
        return PathConfig(operator_path=mock_operator_with_sections, out_path=mock_output_dir)

    @pytest.fixture
    def data_loader(self, path_config: PathConfig) -> DataLoader:
        """Create DataLoader instance."""
        return DataLoader(path_config)

    @pytest.fixture
    def id_generator(self) -> IdGenerator:
        """Create IdGenerator with default config."""
        config = IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            shared_counters=True,
        )
        return IdGenerator(config)

    @pytest.fixture
    def section_extractor(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ) -> SectionExtractor:
        """Create SectionExtractor instance."""
        return SectionExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

    def test_section_extractor_initialization(
        self,
        path_config: PathConfig,
        data_loader: DataLoader,
        id_generator: IdGenerator,
    ):
        """Test SectionExtractor initialization."""
        extractor = SectionExtractor(
            path_config=path_config,
            data_loader=data_loader,
            id_generator=id_generator,
        )

        assert extractor.path_config == path_config
        assert extractor.data_loader == data_loader
        assert extractor.id_generator == id_generator

    def test_extract_generic_code_block_basic(self, section_extractor: SectionExtractor):
        """测试基本代码块提取 - 相邻非空行"""
        lines = ["", "int x = 1;", "int y = 2;", "", "int z = 3;"]
        result = section_extractor.extract_generic_code_block(lines, 1)

        assert result["start_line"] == 2  # 1-based
        assert result["end_line"] == 3  # 1-based
        assert "int x = 1;" in result["content"]
        assert "int y = 2;" in result["content"]
        assert "int z = 3;" not in result["content"]

    def test_extract_generic_code_block_single_line(self, section_extractor: SectionExtractor):
        """测试单行代码块提取"""
        lines = ["", "int x = 1;", ""]
        result = section_extractor.extract_generic_code_block(lines, 1)

        assert result["start_line"] == 2
        assert result["end_line"] == 2
        assert result["content"] == "int x = 1;"

    def test_extract_generic_code_block_expansion(self, section_extractor: SectionExtractor):
        """测试代码块向两侧扩展"""
        lines = [
            "",
            "void func1() {",
            "    int a = 1;",
            "    int b = 2;",  # target line
            "}",
            "",
            "void func2() {",
            "    int c = 3;",
            "}",
        ]
        result = section_extractor.extract_generic_code_block(lines, 3)

        # Should expand to include entire func1
        assert result["start_line"] == 2
        assert result["end_line"] == 5
        assert "void func1()" in result["content"]
        assert "void func2()" not in result["content"]

    def test_extract_soc_version_sections_single(self, section_extractor: SectionExtractor):
        """测试单个 SocVersion 匹配"""
        lines = [
            "if (curShortSocName == platform_ascendc::SocVersion::Ascend910B1) {",
            "    int x = 1;",
            "}",
        ]
        sections = section_extractor.extract_soc_version_sections(lines)

        assert len(sections) == 1
        assert "Ascend910B1" in sections[0]["socs"]
        assert sections[0]["start_line"] == 1
        assert sections[0]["end_line"] == 3

    def test_extract_soc_version_sections_multiple(self, section_extractor: SectionExtractor):
        """测试多个 SocVersion 匹配 (if-else 链)"""
        lines = [
            "if (curShortSocName == platform_ascendc::SocVersion::Ascend910B1) {",
            "    int x = 1;",
            "} else if (curShortSocName == platform_ascendc::SocVersion::Ascend910B2) {",
            "    int y = 2;",
            "} else {",
            "    int z = 3;",
            "}",
        ]
        sections = section_extractor.extract_soc_version_sections(lines)

        # Should deduplicate into single block
        assert len(sections) == 1
        assert "Ascend910B1" in sections[0]["socs"]
        assert "Ascend910B2" in sections[0]["socs"]
        assert sections[0]["start_line"] == 1
        assert sections[0]["end_line"] == 7

    def test_extract_soc_version_sections_no_match(self, section_extractor: SectionExtractor):
        """测试无匹配情况"""
        lines = [
            "int x = 1;",
            "int y = 2;",
        ]
        sections = section_extractor.extract_soc_version_sections(lines)

        assert len(sections) == 0

    def test_extract_npu_arch_sections_single(self, section_extractor: SectionExtractor):
        """测试单个 __NPU_ARCH__ 匹配"""
        lines = [
            "#if __NPU_ARCH__ == Ascend910B",
            "    int a = 1;",
            "    int b = 2;",
            "#endif",
        ]
        sections = section_extractor.extract_npu_arch_sections(lines)

        assert len(sections) == 1
        assert sections[0]["arch_name"] == "Ascend910B"
        assert sections[0]["arch_macro"] == "__NPU_ARCH__"
        assert sections[0]["start_line"] == 1
        assert sections[0]["end_line"] == 4

    def test_extract_npu_arch_sections_cce_aicore(self, section_extractor: SectionExtractor):
        """测试 __CCE_AICORE__ 匹配"""
        lines = [
            "#if __CCE_AICORE__ == Ascend910B",
            "    int a = 1;",
            "#endif",
        ]
        sections = section_extractor.extract_npu_arch_sections(lines)

        assert len(sections) == 1
        assert sections[0]["arch_name"] == "Ascend910B"
        assert sections[0]["arch_macro"] == "__CCE_AICORE__"

    def test_extract_npu_arch_sections_multiple(self, section_extractor: SectionExtractor):
        """测试多个架构宏匹配"""
        lines = [
            "#if __NPU_ARCH__ == Ascend910B",
            "    int a = 1;",
            "#elif __NPU_ARCH__ == Ascend910B2",
            "    int b = 2;",
            "#endif",
        ]
        sections = section_extractor.extract_npu_arch_sections(lines)

        # Should deduplicate into single block
        assert len(sections) == 1
        assert sections[0]["start_line"] == 1
        assert sections[0]["end_line"] == 5

    def test_extract_add_config_sections_single(self, section_extractor: SectionExtractor):
        """测试单个 AICore().AddConfig 匹配"""
        lines = [
            "void ConfigureChip() {",
            '    AICore().AddConfig("Ascend910B1");',
            "}",
        ]
        sections = section_extractor.extract_add_config_sections(lines)

        assert len(sections) == 1
        assert "Ascend910B1" in sections[0]["chips"]
        assert sections[0]["start_line"] == 1
        assert sections[0]["end_line"] == 3

    def test_extract_add_config_sections_multiple(self, section_extractor: SectionExtractor):
        """测试多个 AddConfig 调用"""
        lines = [
            "void ConfigureChip() {",
            '    AICore().AddConfig("Ascend910B1");',
            '    AICore().AddConfig("Ascend910B2");',
            "}",
        ]
        sections = section_extractor.extract_add_config_sections(lines)

        # Should deduplicate into single function
        assert len(sections) == 1
        assert "Ascend910B1" in sections[0]["chips"]
        assert "Ascend910B2" in sections[0]["chips"]

    def test_extract_add_config_sections_multiple_functions(self, section_extractor: SectionExtractor):
        """测试多个函数中的 AddConfig"""
        lines = [
            "void ConfigureChip1() {",
            '    AICore().AddConfig("Ascend910B1");',
            "}",
            "",
            "void ConfigureChip2() {",
            '    AICore().AddConfig("Ascend910B2");',
            "}",
        ]
        sections = section_extractor.extract_add_config_sections(lines)

        # Should extract both functions
        assert len(sections) == 2
        assert "Ascend910B1" in sections[0]["chips"]
        assert "Ascend910B2" in sections[1]["chips"]

    def test_extract_keyword_sections_support_list(self, section_extractor: SectionExtractor):
        """测试 SUPPORT_LIST 关键词提取"""
        lines = [
            "void CheckInputs() {",
            "    SUPPORT_LIST(DT_FLOAT, DT_INT32);",
            "}",
        ]
        sections = section_extractor.extract_keyword_sections(lines)

        assert len(sections) >= 1
        sources = [s["source"] for s in sections]
        assert "SUPPORT_LIST" in sources

    def test_extract_keyword_sections_check_dtype(self, section_extractor: SectionExtractor):
        """测试 check_dtype 关键词提取"""
        lines = [
            "void ValidateInput() {",
            "    check_dtype(input_tensor);",
            "}",
        ]
        sections = section_extractor.extract_keyword_sections(lines)

        assert len(sections) >= 1
        sources = [s["source"] for s in sections]
        assert "check_dtype" in sources

    def test_extract_keyword_sections_multiple_keywords(self, section_extractor: SectionExtractor):
        """测试多个关键词提取"""
        lines = [
            "void ValidateAll() {",
            "    SUPPORT_LIST(DT_FLOAT);",
            "    check_dtype(input);",
            "    CheckFormat(input);",
            "    CheckShape(input);",
            "}",
        ]
        sections = section_extractor.extract_keyword_sections(lines)

        assert len(sections) >= 1
        sources = [s["source"] for s in sections]
        # Should find at least some keywords
        assert any(kw in sources for kw in ["SUPPORT_LIST", "check_dtype", "CheckFormat", "CheckShape"])

    def test_extract_keyword_sections_custom_keywords(self, section_extractor: SectionExtractor):
        """测试自定义关键词列表"""
        lines = [
            "void CustomCheck() {",
            "    CUSTOM_KEYWORD();",
            "}",
        ]
        sections = section_extractor.extract_keyword_sections(lines, keywords=["CUSTOM_KEYWORD"])

        assert len(sections) >= 1
        assert sections[0]["source"] == "CUSTOM_KEYWORD"

    def test_extract_keyword_sections_no_match(self, section_extractor: SectionExtractor):
        """测试无关键词匹配"""
        lines = [
            "int x = 1;",
            "int y = 2;",
        ]
        sections = section_extractor.extract_keyword_sections(lines)

        assert len(sections) == 0

    def test_extract_full_integration(
        self,
        section_extractor: SectionExtractor,
        mock_operator_with_sections: Path,
    ):
        """测试完整提取流程集成"""
        anchors = section_extractor.extract(mock_operator_with_sections)

        # Should find anchors from all patterns
        assert len(anchors) >= 1

        # Verify anchor structure
        for anchor in anchors:
            assert "anchor_id" in anchor
            assert "anchor_information" in anchor
            assert "evidence" in anchor
            assert anchor["anchor_information"]["type"] == "section"
            assert "evidence_id" in anchor["evidence"][0]
            assert "envidence_id" in anchor["evidence"][0]
            assert anchor["evidence"][0]["evidence_id"] == anchor["evidence"][0]["envidence_id"]

    def test_extract_with_pattern_filter(
        self,
        section_extractor: SectionExtractor,
        mock_operator_with_sections: Path,
    ):
        """测试按模式类型过滤提取"""
        # Extract only soc_version pattern
        anchors = section_extractor.extract(mock_operator_with_sections, section_patterns=["soc_version"])

        assert len(anchors) >= 1

        # Verify it's a SocVersion anchor
        anchor = anchors[0]
        assert "support_chips" in anchor["evidence"][0]
        assert "socs" in anchor["evidence"][0]["support_chips"]

    def test_anchor_id_format(
        self,
        section_extractor: SectionExtractor,
        mock_operator_with_sections: Path,
    ):
        """Test anchor_id follows expected format: anchor_section_{seq:03d}."""
        anchors = section_extractor.extract(mock_operator_with_sections)

        for i, anchor in enumerate(anchors, start=1):
            anchor_id = anchor["anchor_id"]
            expected_id = f"anchor_section_{i:03d}"
            assert anchor_id == expected_id

    def test_gene_rule_is_placeholder(
        self,
        section_extractor: SectionExtractor,
        mock_operator_with_sections: Path,
    ):
        """Test gene_rule is placeholder (pipeline assigns later)."""
        anchors = section_extractor.extract(mock_operator_with_sections)

        for anchor in anchors:
            evidence = anchor["evidence"][0]
            gene_rule = evidence["gene_rule"]

            # Check gene_rule structure
            assert "deter" in gene_rule
            assert "target" in gene_rule

            # Check placeholder values
            assert gene_rule["deter"] == "暂空"
            assert gene_rule["target"] == []

    def test_note_field_present(
        self,
        section_extractor: SectionExtractor,
        mock_operator_with_sections: Path,
    ):
        """Test note field is present with standard message."""
        anchors = section_extractor.extract(mock_operator_with_sections)

        for anchor in anchors:
            assert "note" in anchor
            assert anchor["note"] == "以下为section型标准结构"

    def test_soc_version_metadata_preserved(self, section_extractor: SectionExtractor):
        """测试 SocVersion 元数据保留"""
        lines = [
            "if (curShortSocName == platform_ascendc::SocVersion::Ascend910B1) {",
            "    int x = 1;",
            "}",
        ]
        sections = section_extractor.extract_soc_version_sections(lines)

        assert len(sections) == 1
        assert "socs" in sections[0]
        assert sections[0]["socs"] == ["Ascend910B1"]

    def test_npu_arch_metadata_preserved(self, section_extractor: SectionExtractor):
        """测试 NPU_ARCH 元数据保留"""
        lines = [
            "#if __NPU_ARCH__ == Ascend910B",
            "    int a = 1;",
            "#endif",
        ]
        sections = section_extractor.extract_npu_arch_sections(lines)

        assert len(sections) == 1
        assert "arch_name" in sections[0]
        assert "arch_macro" in sections[0]
        assert sections[0]["arch_name"] == "Ascend910B"
        assert sections[0]["arch_macro"] == "__NPU_ARCH__"

    def test_add_config_metadata_preserved(self, section_extractor: SectionExtractor):
        """测试 AddConfig 元数据保留"""
        lines = [
            "void Configure() {",
            '    AICore().AddConfig("Ascend910B1");',
            '    AICore().AddConfig("Ascend910B2");',
            "}",
        ]
        sections = section_extractor.extract_add_config_sections(lines)

        assert len(sections) == 1
        assert "chips" in sections[0]
        assert "Ascend910B1" in sections[0]["chips"]
        assert "Ascend910B2" in sections[0]["chips"]

    def test_keyword_metadata_preserved(self, section_extractor: SectionExtractor):
        """测试关键词元数据保留"""
        lines = [
            "void Check() {",
            "    SUPPORT_LIST(DT_FLOAT);",
            "}",
        ]
        sections = section_extractor.extract_keyword_sections(lines)

        assert len(sections) >= 1
        assert "source" in sections[0]
        assert sections[0]["source"] == "SUPPORT_LIST"

    def test_empty_operator_directory(
        self,
        section_extractor: SectionExtractor,
        tmp_path: Path,
    ):
        """Test extraction from empty operator directory."""
        empty_operator = tmp_path / "empty_operator"
        empty_operator.mkdir()

        anchors = section_extractor.extract(empty_operator)

        # Should return empty list
        assert len(anchors) == 0

    def test_id_generator_shared_counters(
        self,
        section_extractor: SectionExtractor,
        mock_operator_with_sections: Path,
    ):
        """Test ID generator uses shared counters (anchor and evidence IDs match)."""
        anchors = section_extractor.extract(mock_operator_with_sections)

        for anchor in anchors:
            anchor_id = anchor["anchor_id"]
            evidence_id = anchor["evidence"][0]["evidence_id"]

            # Extract sequence numbers
            anchor_seq = anchor_id.split("_")[-1]
            evidence_seq = evidence_id.split("_")[-1]

            # Should match (shared counters)
            assert anchor_seq == evidence_seq


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
