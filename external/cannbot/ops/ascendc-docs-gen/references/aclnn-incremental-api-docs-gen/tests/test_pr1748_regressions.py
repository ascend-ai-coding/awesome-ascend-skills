# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Regression tests for PR 1748 review findings.

Locks four fixed defects:
1. run_evaluation 返回 4 元组 (含 repo_other_changed_files), 且所有调用点按 4 元组解包
   (run_change_detection_pipeline.py 曾按 3 元组解包 → 必然 ValueError)。
2. run_consistency_check_pipeline._vget dict/对象双态取值
   (Step 3 曾用 getattr 读 dict → 修复结果统计恒 False)。
3. cli._is_validation_passed dict/对象双态判定
   (run_full_pipeline/run_validate_only 尾部曾用 getattr(v, "passed", True) 读 dict → 恒 True)。
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────
# 1. run_evaluation 4 元组契约
# ─────────────────────────────────────────────────────────────


class TestRunEvaluationFourTupleContract:
    def _call_run_evaluation(self, run_detection_return):
        from src.change_detection.doc_part_change_detection.change_evaluator import (
            run_evaluation,
        )

        with (
            patch(
                "src.change_detection.doc_part_change_detection.subpart_mapper.run_detection",
                return_value=run_detection_return,
            ),
            patch(
                "src.change_detection.doc_part_change_detection.docpart_file_extractor"
                ".DocpartFileExtractor.extract_fine_grained_mappings",
                return_value={},
            ),
        ):
            return run_evaluation(
                evidence_dir="fake_evidence",
                before_dir="fake_before",
                after_dir="fake_after",
                doc_content="",
                fine_grained=True,
                console=Console(quiet=True),
            )

    def test_empty_branch_returns_four_tuple(self):
        """无受影响 doc_part 且无 tiling 变更 → 短路分支也必须返回 4 元组。"""
        result = self._call_run_evaluation(
            (set(), {}, [], [], ["repo_other/changed.cpp"]),
        )
        assert isinstance(result, tuple) and len(result) == 4
        verified_todo_list, all_reasons, modified_file_dir_list, repo_other = result
        assert verified_todo_list == []
        assert all_reasons == {}
        assert modified_file_dir_list == []
        assert repo_other == ["repo_other/changed.cpp"]

    def test_empty_branch_unpacks_like_call_sites(self):
        """与 run_change_detection_pipeline.py / run_all_cli.py 的解包方式一致。"""
        (
            verified_todo_list,
            _all_reasons,
            _modified_file_dir_list,
            _repo_other_changed_files,
        ) = self._call_run_evaluation((set(), {}, ["a.cpp"], ["b.h"], ["repo/c.cpp"]))
        assert verified_todo_list == []


class TestCallSitesUnpackFourValues:
    """AST 锁定: 所有 run_evaluation(...) 调用点必须按 4 目标解包。"""

    CALL_SITE_FILES = [
        REPO_ROOT / "src" / "run_change_detection_pipeline.py",
        REPO_ROOT / "src" / "run_all_cli.py",
        REPO_ROOT / "src" / "change_detection" / "doc_part_change_detection" / "change_evaluator.py",
    ]

    @staticmethod
    def _is_run_evaluation_call(node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id == "run_evaluation"
        if isinstance(func, ast.Attribute):
            return func.attr == "run_evaluation"
        return False

    def test_all_unpack_sites_have_four_targets(self):
        checked = 0
        for path in self.CALL_SITE_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not (
                    isinstance(node.value, ast.Call)
                    and self._is_run_evaluation_call(node.value)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Tuple)
                ):
                    continue
                checked += 1
                n_elts = len(node.targets[0].elts)
                assert n_elts == 4, (
                    f"{path.name}:{node.lineno} 按 {n_elts} 元组解包 run_evaluation, "
                    f"但其返回 4 元组 (verified_todo_list, all_reasons, "
                    f"modified_file_dir_list, repo_other_changed_files)"
                )
        assert checked >= 3, "未找到预期的 run_evaluation 解包调用点, AST 锁定失效"


# ─────────────────────────────────────────────────────────────
# 2. _vget dict/对象双态 (run_consistency_check_pipeline)
# ─────────────────────────────────────────────────────────────


@dataclass
class _FakeValidationModel:
    passed: bool = False
    consistency_passed: bool = False


class TestVgetDualState:
    def test_dict_reads_keys(self):
        from src.run_consistency_check_pipeline import _vget

        v = {"passed": True, "consistency_passed": True}
        assert _vget(v, "passed", False) is True
        assert _vget(v, "consistency_passed", False) is True

    def test_dict_missing_key_returns_default(self):
        from src.run_consistency_check_pipeline import _vget

        assert _vget({}, "passed", False) is False

    def test_object_reads_attributes(self):
        from src.run_consistency_check_pipeline import _vget

        v = _FakeValidationModel(passed=True, consistency_passed=True)
        assert _vget(v, "passed", False) is True
        assert _vget(v, "consistency_passed", False) is True

    def test_step3_all_passed_logic_with_dict_validation(self):
        """回归锁定: dict 型 validation 元素旧写法 getattr 恒 False, 修复永不回计通过。"""
        from src.run_consistency_check_pipeline import _vget

        validation = [
            {"passed": True, "consistency_passed": True},
            {"passed": True, "consistency_passed": True},
        ]
        all_passed = all(
            _vget(v, "passed", False) or _vget(v, "consistency_passed", False)
            for v in validation
        )
        assert all_passed is True
        # 旧写法确实是坏的 (证明该回归曾经存在)
        assert all(getattr(v, "passed", False) for v in validation) is False

    def test_step3_all_passed_logic_false_when_failed(self):
        from src.run_consistency_check_pipeline import _vget

        validation = [
            {"passed": True, "consistency_passed": True},
            {"passed": False, "consistency_passed": False},
        ]
        all_passed = all(
            _vget(v, "passed", False) or _vget(v, "consistency_passed", False)
            for v in validation
        )
        assert all_passed is False


# ─────────────────────────────────────────────────────────────
# 3. _is_validation_passed dict/对象双态 (generation_validation/ui/cli)
# ─────────────────────────────────────────────────────────────


class TestIsValidationPassedDualState:
    def test_dict_consistency_and_metric_pass(self):
        from src.generation_validation.ui.cli import _is_validation_passed

        assert _is_validation_passed({"consistency_passed": True, "metric_passed": "PASS"}) is True
        assert _is_validation_passed({"consistency_passed": True, "metric_passed": None}) is True
        assert _is_validation_passed({"consistency_passed": True, "metric_passed": "N/A"}) is True

    def test_dict_failures(self):
        from src.generation_validation.ui.cli import _is_validation_passed

        assert _is_validation_passed({"consistency_passed": False, "metric_passed": "PASS"}) is False
        assert _is_validation_passed({"consistency_passed": True, "metric_passed": "FAIL"}) is False
        assert _is_validation_passed({}) is False

    def test_object_state(self):
        from src.generation_validation.ui.cli import _is_validation_passed

        assert _is_validation_passed(
            _FakeValidationModel(passed=True, consistency_passed=True)
        ) is True
        assert _is_validation_passed(_FakeValidationModel()) is False

    def test_old_getattr_default_true_was_broken_for_dict(self):
        """回归锁定: 旧写法 getattr(v, "passed", True) 对 dict 恒 True (假通过)。"""
        failing = {"passed": False, "consistency_passed": False, "metric_passed": "FAIL"}
        assert getattr(failing, "passed", True) is True  # 旧写法的错误行为
        from src.generation_validation.ui.cli import _is_validation_passed

        assert _is_validation_passed(failing) is False  # 修复后正确判 FAIL
