# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Regression tests for _expand_template_targets and _get_sub_function_count.

_expand_template_targets expands template variables ({sub_function_hash},
{sub_error_code}) in gene_rule targets based on sub_function_count.

_get_sub_function_count reads doc_json to determine how many sub_functions
exist for an operator.
"""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

from src.snapshot_construction.anchor_construction.pipeline import (
    _expand_template_targets,
    _PARAM_FIELDS,
    AnchorPipeline,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_anchor(
    anchor_type: str, targets: list[list[str]], keyword_source: str = None
) -> dict:
    """Create a minimal mock anchor for testing."""
    anchor = {
        "anchor_id": f"anchor_{anchor_type}_001",
        "anchor_information": {
            "type": anchor_type,
            "level1": {"file_path": "test/path", "file_name": "test.h"},
        },
        "evidence": [
            {
                "evidence_id": f"ev_{anchor_type}_001",
                "gene_rule": {"deter": "", "target": targets},
            }
        ],
    }
    if keyword_source:
        anchor["evidence"][0]["keyword_source"] = keyword_source
    return anchor


def _get_targets(anchor: dict) -> list[list[str]]:
    """Extract targets from an anchor."""
    return anchor["evidence"][0]["gene_rule"]["target"]


# ===========================================================================
# TestExpandTemplateTargets
# ===========================================================================


class TestExpandTemplateTargets:
    """Tests for _expand_template_targets module-level function."""

    # ---- File anchor tests ----

    def test_file_anchor_aclnn_h_expansion(self):
        """aclnn_h file anchor with template targets → N sub_functions × 6 fields."""
        # .h file: 6 param fields (no tensor, no errorcode)
        tmpl_targets = [
            ["{sub_function_hash}", f]
            for f in ["des", "inout", "usage", "dtype", "dmat", "shape"]
        ]
        anchor = _make_anchor("file", tmpl_targets)
        result = _expand_template_targets([anchor], 2)

        assert len(result) == 1
        targets = _get_targets(result[0])
        # 6 fields × 2 sub_functions = 12
        assert len(targets) == 12

        # Verify sub_function_1 fields
        sf1_targets = targets[:6]
        for i, field in enumerate(["des", "inout", "usage", "dtype", "dmat", "shape"]):
            assert sf1_targets[i] == ["sub_function_1_param_matrix", field]

        # Verify sub_function_2 fields
        sf2_targets = targets[6:]
        for i, field in enumerate(["des", "inout", "usage", "dtype", "dmat", "shape"]):
            assert sf2_targets[i] == ["sub_function_2_param_matrix", field]

    def test_file_anchor_aclnn_cpp_expansion(self):
        """aclnn_cpp file anchor with template targets → N sub_functions × (7 fields + errorcode)."""
        # .cpp file: 7 param fields + errorcode
        tmpl_targets = [["{sub_function_hash}", f] for f in _PARAM_FIELDS]
        tmpl_targets.append(["{sub_error_code}", "errorcode"])

        anchor = _make_anchor("file", tmpl_targets)
        result = _expand_template_targets([anchor], 2)

        assert len(result) == 1
        targets = _get_targets(result[0])
        # (7 fields + 1 errorcode) × 2 sub_functions = 16
        assert len(targets) == 16

        # Verify sub_function_1: 7 param fields + errorcode
        sf1 = targets[:8]
        for i, field in enumerate(_PARAM_FIELDS):
            assert sf1[i] == ["sub_function_1_param_matrix", field]
        assert sf1[7] == ["sub_function_1_errorcode", "errorcode"]

        # Verify sub_function_2: 7 param fields + errorcode
        sf2 = targets[8:]
        for i, field in enumerate(_PARAM_FIELDS):
            assert sf2[i] == ["sub_function_2_param_matrix", field]
        assert sf2[7] == ["sub_function_2_errorcode", "errorcode"]

    def test_file_anchor_no_template_passthrough(self):
        """File anchor without template vars → unchanged."""
        targets = [["some_key", "des"], ["other_key", "inout"]]
        anchor = _make_anchor("file", targets)
        original_targets = deepcopy(targets)

        result = _expand_template_targets([anchor], 3)

        assert len(result) == 1
        assert _get_targets(result[0]) == original_targets

    # ---- Function anchor tests ----

    def test_function_anchor_expansion(self):
        """Function anchor → expanded to N sub_functions × (7 fields + errorcode)."""
        # Function anchors get rebuilt from scratch, input targets don't matter
        tmpl_targets = [["{sub_function_hash}", "des"]]
        anchor = _make_anchor("function", tmpl_targets)

        result = _expand_template_targets([anchor], 2)

        assert len(result) == 1
        targets = _get_targets(result[0])
        # (7 fields + 1 errorcode) × 2 = 16
        assert len(targets) == 16

        # Verify structure: for each sub_function, 7 param fields then errorcode
        for sf_idx in range(2):
            offset = sf_idx * 8
            for i, field in enumerate(_PARAM_FIELDS):
                assert targets[offset + i] == [
                    f"sub_function_{sf_idx + 1}_param_matrix",
                    field,
                ]
            assert targets[offset + 7] == [
                f"sub_function_{sf_idx + 1}_errorcode",
                "errorcode",
            ]

    def test_function_anchor_count_1(self):
        """sub_function_count=1 → 8 targets (7 fields + 1 errorcode)."""
        anchor = _make_anchor("function", [["{sub_function_hash}", "des"]])

        result = _expand_template_targets([anchor], 1)

        targets = _get_targets(result[0])
        assert len(targets) == 8

        for i, field in enumerate(_PARAM_FIELDS):
            assert targets[i] == ["sub_function_1_param_matrix", field]
        assert targets[7] == ["sub_function_1_errorcode", "errorcode"]

    def test_function_anchor_count_3(self):
        """sub_function_count=3 → 24 targets (8 × 3)."""
        anchor = _make_anchor("function", [["{sub_function_hash}", "des"]])

        result = _expand_template_targets([anchor], 3)

        targets = _get_targets(result[0])
        assert len(targets) == 24

        for sf in range(1, 4):
            offset = (sf - 1) * 8
            for i, field in enumerate(_PARAM_FIELDS):
                assert targets[offset + i] == [f"sub_function_{sf}_param_matrix", field]
            assert targets[offset + 7] == [f"sub_function_{sf}_errorcode", "errorcode"]

    # ---- Note anchor tests ----

    def test_note_anchor_expansion(self):
        """Note anchor → expanded to N sub_functions × 7 fields (NO errorcode)."""
        tmpl_targets = [["{sub_function_hash}", "des"]]
        anchor = _make_anchor("note", tmpl_targets)

        result = _expand_template_targets([anchor], 2)

        assert len(result) == 1
        targets = _get_targets(result[0])
        # 7 fields × 2 = 14 (no errorcode for note)
        assert len(targets) == 14

        for sf_idx in range(2):
            offset = sf_idx * 7
            for i, field in enumerate(_PARAM_FIELDS):
                assert targets[offset + i] == [
                    f"sub_function_{sf_idx + 1}_param_matrix",
                    field,
                ]

    def test_note_anchor_no_errorcode(self):
        """Verify note targets don't contain 'errorcode'."""
        anchor = _make_anchor("note", [["{sub_function_hash}", "des"]])

        result = _expand_template_targets([anchor], 3)

        targets = _get_targets(result[0])
        for t in targets:
            assert "errorcode" not in t
            assert len(t) == 2
            assert t[1] in _PARAM_FIELDS

    # ---- Section anchor tests ----

    def test_section_normal_copies_n_times(self):
        """Normal section (e.g., SUPPORT_LIST) → copied N times, each with single target."""
        tmpl_targets = [["{sub_function_hash}", "dtype"]]
        anchor = _make_anchor("section", tmpl_targets)

        result = _expand_template_targets([anchor], 3)

        # Normal section: copied N times
        assert len(result) == 3
        for n in range(1, 4):
            targets = _get_targets(result[n - 1])
            assert len(targets) == 1
            assert targets[0] == [f"sub_function_{n}_param_matrix", "dtype"]

    def test_section_checkparams_single_anchor(self):
        """CheckParams section → single anchor with N errorcode targets."""
        # CheckParams: targets have "errorcode" as second element
        tmpl_targets = [["{sub_error_code}", "errorcode"]]
        anchor = _make_anchor("section", tmpl_targets)

        result = _expand_template_targets([anchor], 2)

        # CheckParams: single anchor
        assert len(result) == 1
        targets = _get_targets(result[0])
        assert len(targets) == 2
        assert targets[0] == ["sub_function_1_errorcode", "errorcode"]
        assert targets[1] == ["sub_function_2_errorcode", "errorcode"]

    def test_section_normal_target_format(self):
        """Each copied section has target like [['sub_function_N_param_matrix', 'dtype']]."""
        tmpl_targets = [["{sub_function_hash}", "dtype"]]
        anchor = _make_anchor("section", tmpl_targets)

        result = _expand_template_targets([anchor], 2)

        for n, expanded in enumerate(result, start=1):
            targets = _get_targets(expanded)
            assert targets == [[f"sub_function_{n}_param_matrix", "dtype"]]

    # ---- General tests ----

    def test_no_template_placeholders_in_output(self):
        """After expansion, no target contains '{' or '}'."""
        anchors = [
            _make_anchor(
                "file",
                [["{sub_function_hash}", "des"], ["{sub_error_code}", "errorcode"]],
            ),
            _make_anchor("function", [["{sub_function_hash}", "des"]]),
            _make_anchor("note", [["{sub_function_hash}", "inout"]]),
            _make_anchor("section", [["{sub_function_hash}", "dtype"]]),
            _make_anchor("section", [["{sub_error_code}", "errorcode"]]),
        ]

        result = _expand_template_targets(anchors, 2)

        for anchor in result:
            for t in _get_targets(anchor):
                for elem in t:
                    assert "{" not in elem, f"Found '{{' in: {elem}"
                    assert "}" not in elem, f"Found '}}' in: {elem}"

    def test_empty_anchor_tuple(self):
        """Empty input → empty output."""
        result = _expand_template_targets([], 3)
        assert result == []

    def test_mixed_anchor_types(self):
        """Mix of file/function/note/section → all expanded correctly."""
        file_anchor = _make_anchor(
            "file",
            [
                ["{sub_function_hash}", "des"],
                ["{sub_error_code}", "errorcode"],
            ],
        )
        func_anchor = _make_anchor("function", [["{sub_function_hash}", "des"]])
        note_anchor = _make_anchor("note", [["{sub_function_hash}", "des"]])
        section_normal = _make_anchor("section", [["{sub_function_hash}", "dtype"]])
        section_cp = _make_anchor("section", [["{sub_error_code}", "errorcode"]])
        section_cp["anchor_id"] = "anchor_section_002"  # unique id

        anchors = [file_anchor, func_anchor, note_anchor, section_normal, section_cp]
        result = _expand_template_targets(anchors, 2)

        # file: 1 anchor (mutated)
        # function: 1 anchor (mutated)
        # note: 1 anchor (mutated)
        # section normal: 2 copies
        # section checkparams: 1 anchor
        # Total: 1 + 1 + 1 + 2 + 1 = 6
        assert len(result) == 6

        # Verify file anchor (index 0)
        file_targets = _get_targets(result[0])
        assert len(file_targets) == 4  # (1 param + 1 errorcode) × 2

        # Verify function anchor (index 1)
        func_targets = _get_targets(result[1])
        assert len(func_targets) == 16  # (7 + 1) × 2

        # Verify note anchor (index 2)
        note_targets = _get_targets(result[2])
        assert len(note_targets) == 14  # 7 × 2

        # Verify normal section (indices 3, 4)
        assert len(_get_targets(result[3])) == 1
        assert len(_get_targets(result[4])) == 1

        # Verify checkparams section (index 5)
        cp_targets = _get_targets(result[5])
        assert len(cp_targets) == 2


# ===========================================================================
# TestGetSubFunctionCount
# ===========================================================================


class TestGetSubFunctionCount:
    """Tests for AnchorPipeline._get_sub_function_count method."""

    def _make_pipeline_mock(self, operator_path: Path, doc_json_dir: Path):
        """Create a minimal AnchorPipeline mock with path_config."""
        pipeline = object.__new__(AnchorPipeline)
        pipeline.path_config = MagicMock()
        pipeline.path_config.operator_path = operator_path
        pipeline.path_config.doc_json = doc_json_dir
        pipeline.path_config.md_file = None
        pipeline.path_config.md_stem = None
        return pipeline

    def test_get_sub_function_count_with_valid_doc_json(self, tmp_path):
        """Mock doc_json with 2 sub_function_hash_N keys → returns 2."""
        # Create operator dir with aclnn*.md file
        op_dir = tmp_path / "operator"
        op_dir.mkdir()
        api_dir = op_dir / "op_api"
        api_dir.mkdir()
        md_file = api_dir / "aclnn_abs.md"
        md_file.write_text("# Abs", encoding="utf-8")

        # Create doc_json dir with matching JSON
        doc_json_dir = tmp_path / "doc_json"
        doc_json_dir.mkdir()
        doc_json_file = doc_json_dir / "aclnn_abs.json"
        doc_json_data = {
            "sub_function_hash_1": {"param_matrix": {}},
            "sub_function_hash_2": {"param_matrix": {}},
        }
        doc_json_file.write_text(json.dumps(doc_json_data), encoding="utf-8")

        pipeline = self._make_pipeline_mock(op_dir, doc_json_dir)
        count = pipeline._get_sub_function_count()

        assert count == 2

    def test_get_sub_function_count_missing_doc_json(self, tmp_path):
        """No doc_json file → returns 1."""
        op_dir = tmp_path / "operator"
        op_dir.mkdir()
        api_dir = op_dir / "op_api"
        api_dir.mkdir()
        md_file = api_dir / "aclnn_abs.md"
        md_file.write_text("# Abs", encoding="utf-8")

        # doc_json dir exists but no matching JSON
        doc_json_dir = tmp_path / "doc_json"
        doc_json_dir.mkdir()

        pipeline = self._make_pipeline_mock(op_dir, doc_json_dir)
        count = pipeline._get_sub_function_count()

        assert count == 1

    def test_get_sub_function_count_zero_keys(self, tmp_path):
        """doc_json exists but no sub_function_hash_ keys → returns 1."""
        op_dir = tmp_path / "operator"
        op_dir.mkdir()
        api_dir = op_dir / "op_api"
        api_dir.mkdir()
        md_file = api_dir / "aclnn_abs.md"
        md_file.write_text("# Abs", encoding="utf-8")

        doc_json_dir = tmp_path / "doc_json"
        doc_json_dir.mkdir()
        doc_json_file = doc_json_dir / "aclnn_abs.json"
        # JSON with no sub_function_hash_ keys
        doc_json_data = {"some_other_key": "value"}
        doc_json_file.write_text(json.dumps(doc_json_data), encoding="utf-8")

        pipeline = self._make_pipeline_mock(op_dir, doc_json_dir)
        count = pipeline._get_sub_function_count()

        assert count == 1
