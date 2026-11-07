# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from retrieval.cards import _parse_front
from retrieval.config import FIELD_WEIGHTS
from retrieval.eval import evaluate
from retrieval.errors import ModelRuntimeError
from retrieval.embedding import APIEmbedding
from retrieval.graph import _load_graph, cmd_neighbors
from retrieval.grep import grep_results
from retrieval.index import INDEX_VERSION, _read_index, cmd_verify
from retrieval.llm import LLMClient
from retrieval.platforms import (
    filter_candidate_ids,
    is_950_only,
    platform_context,
    platform_filter_output,
)
from retrieval.plan import brief_preflight_output, plan_task, preflight_task
from retrieval.rerank.llm_judge import LLMJudge
from retrieval.scope import facet_filter
from retrieval.search import search_results


def _search_index():
    docs = []
    specs = (
        ("a-950.md", ["950"]),
        ("b-no-metadata.md", None),
        ("c-950-a2.md", ["950", "a2"]),
    )
    for path, platforms in specs:
        doc = {
            "path": path,
            "bundle": "fixture",
            "section": "",
            "category": "",
            "reldir": "",
            "base": path[:-3],
            "title": "needle",
            "description": "",
            "kind": "api",
            "tags": [],
            "names": [],
            "status": "verified",
            "field_lengths": {field: (1 if field == "title" else 0) for field in FIELD_WEIGHTS},
        }
        if platforms is not None:
            doc["platforms"] = platforms
        docs.append(doc)
    postings = {doc_id: {"title": 1} for doc_id in range(len(docs))}
    return {
        "meta": {"card_count": len(docs), "global_tags": []},
        "docs": docs,
        "df": {"needle": len(docs)},
        "avg_field_len": {field: (1.0 if field == "title" else 0.0) for field in FIELD_WEIGHTS},
        "_pt": {"needle": postings},
        "_docs_with": {"needle": set(range(len(docs)))},
    }


class FrontmatterPlatformParsingTest(unittest.TestCase):
    def test_parses_inline_platforms_as_list(self):
        frontmatter, _ = _parse_front("---\nplatforms: [950, a3]\n---\nbody\n")

        self.assertEqual(["950", "a3"], frontmatter["platforms"])

    def test_parses_block_platforms_as_list(self):
        frontmatter, _ = _parse_front("---\nplatforms:\n  - 950\n  - a3\n---\nbody\n")

        self.assertEqual(["950", "a3"], frontmatter["platforms"])


class PlatformContextTest(unittest.TestCase):
    def test_detects_a3_aliases(self):
        for task in (
            "在 910C 平台实现算子",
            "在 Atlas A3 平台调试 kernel",
            "A3 算子性能优化",
        ):
            with self.subTest(task=task):
                context = platform_context(task)
                self.assertTrue(context["enabled"])
                self.assertEqual("a3", context["target"])
                self.assertEqual("task", context["source"])

    def test_disables_automatic_filter_for_a3_and_950_comparison(self):
        for task in (
            "对比 A3 和 Ascend 950 的 DataCopy API",
            "对比 A3 和 Ascend 950PR 的 DataCopy API",
            "对比 A3 和 950DT 的 DataCopy API",
            "对比 A3 和 A5 的 DataCopy API",
            "对比 A3 和 arch35 的 DataCopy API",
        ):
            with self.subTest(task=task):
                context = platform_context(task)
                self.assertFalse(context["enabled"])
                self.assertIsNone(context["target"])
                self.assertEqual("multi_platform_task", context["reason"])

    def test_explicit_a3_enables_filter_without_task_platform(self):
        context = platform_context("DataCopyPad 怎么用", explicit="A3")

        self.assertTrue(context["enabled"])
        self.assertEqual("a3", context["target"])
        self.assertEqual("explicit", context["source"])


class Strict950OnlyPolicyTest(unittest.TestCase):
    def test_matches_only_exact_single_950_platform(self):
        self.assertTrue(is_950_only({"platforms": ["950"]}))

        for platforms in (
            [],
            ["a3"],
            ["950", "a2"],
            ["950", "a3"],
            "950",
            None,
        ):
            with self.subTest(platforms=platforms):
                self.assertFalse(is_950_only({"platforms": platforms}))

    def test_filters_exact_950_only_and_preserves_other_candidates(self):
        idx = {
            "docs": [
                {"path": "only-950.md", "platforms": ["950"]},
                {"path": "no-metadata.md"},
                {"path": "a3.md", "platforms": ["a3"]},
                {"path": "950-a2.md", "platforms": ["950", "a2"]},
                {"path": "950-a3.md", "platforms": ["950", "a3"]},
            ]
        }
        context = platform_context(explicit="a3")

        self.assertEqual({1, 2, 3, 4}, filter_candidate_ids(idx, None, context))
        self.assertEqual({2, 3}, filter_candidate_ids(idx, {0, 2, 3}, context))

    def test_disabled_filter_leaves_candidate_sentinel_unchanged(self):
        idx = {"docs": [{"path": "only-950.md", "platforms": ["950"]}]}

        self.assertIsNone(filter_candidate_ids(idx, None, platform_context("generic query")))

    def test_output_warns_when_enabled_filter_has_no_results(self):
        output = platform_filter_output(platform_context(explicit="a3"), result_count=0)

        self.assertEqual("no_results_after_filter", output["warning"])


class RetrievalRoutePlatformFilterTest(unittest.TestCase):
    def test_search_filters_before_top_k_and_keeps_retained_scores(self):
        idx = _search_index()
        unfiltered = search_results(["needle"], k=3, idx=idx)

        filtered = search_results(["needle"], k=2, idx=idx, platform="a3")

        self.assertEqual(["b-no-metadata.md", "c-950-a2.md"], [item["path"] for item in filtered])
        unfiltered_scores = {item["path"]: item["score"] for item in unfiltered}
        self.assertEqual(
            [unfiltered_scores[item["path"]] for item in filtered],
            [item["score"] for item in filtered],
        )

    def test_facet_filter_applies_shared_platform_policy(self):
        idx = _search_index()

        self.assertEqual({1, 2}, facet_filter(idx, status="all", platform="a3"))

    def test_grep_filters_before_match_limit(self):
        cards = {
            "a-950.md": "---\ntitle: A\nplatforms: [950]\n---\nneedle\n",
            "b-no-metadata.md": "---\ntitle: B\n---\nneedle\n",
            "c-950-a2.md": "---\ntitle: C\nplatforms: [950, a2]\n---\nneedle\n",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for name, content in cards.items():
                path = Path(tmpdir) / name
                path.write_text(content, encoding="utf-8")
                paths[name] = str(path)
            with patch("retrieval.grep.concept_paths", return_value=list(cards)), patch(
                "retrieval.grep.id_to_path", side_effect=lambda path: paths.get(path, path)
            ):
                matches = grep_results("needle", k=2, platform="a3")

        self.assertEqual(["b-no-metadata.md", "c-950-a2.md"], [item["path"] for item in matches])

    def test_neighbors_filters_before_top_k(self):
        source = "source.md"
        neighbors = ["a-950.md", "b-no-metadata.md", "c-950-a2.md"]
        edges = [
            {"source": source, "target": path, "type": "related", "weight": 1.0, "why": "fixture"}
            for path in neighbors
        ]
        graph = {
            "nodes": {path: {"title": path, "kind": "api", "bundle": "fixture"} for path in [source] + neighbors},
            "adj": {source: edges},
        }
        idx = _search_index()
        idx["docs"].insert(0, {"path": source})
        stdout = io.StringIO()
        with patch("retrieval.graph._load_graph", return_value=graph), patch(
            "retrieval.graph.load_index", return_value=idx
        ), redirect_stdout(stdout):
            cmd_neighbors(source, 1, None, None, 2, platform="a3")

        output = json.loads(stdout.getvalue())
        self.assertEqual(["b-no-metadata.md", "c-950-a2.md"], [item["path"] for item in output["neighbors"]])
        self.assertTrue(output["platform_filter"]["enabled"])


class PreflightPlatformPropagationTest(unittest.TestCase):
    def test_plan_detects_a3_and_propagates_platform_to_supported_commands(self):
        plan = plan_task("在 910C 平台使用 needle API", _search_index())

        self.assertTrue(plan["platform_filter"]["enabled"])
        supported = [command for command in plan["commands"] if " search " in command or " grep " in command]
        self.assertTrue(supported)
        self.assertTrue(all("--platform a3" in command for command in supported))

    def test_a3_alias_is_not_reported_as_missing_soc(self):
        plan = plan_task("A3 平台使用 needle API", _search_index())

        self.assertNotIn("未识别到芯片/SoC/硬件型号", plan["missing_signals"])

    def test_plan_disables_filter_for_a3_and_950_comparison(self):
        plan = plan_task("对比 A3 和 950 平台的 needle API", _search_index())

        self.assertFalse(plan["platform_filter"]["enabled"])
        self.assertEqual("multi_platform_task", plan["platform_filter"]["reason"])
        self.assertTrue(all("--platform" not in command for command in plan["commands"]))

    def test_preflight_filters_results_and_follow_up_commands(self):
        with patch("retrieval.plan.grep_results", return_value=[]):
            output = preflight_task("在 Atlas A3 平台使用 needle API", _search_index(), k=2)

        selected_paths = set(output["suggested_get"])
        selected_paths.update(item["path"] for item in output["results"])
        selected_paths.update(item["path"] for item in output["read_first"])
        self.assertNotIn("a-950.md", selected_paths)
        self.assertTrue(output["platform_filter"]["enabled"])
        follow_commands = [item["command"] for item in output["follow_candidates"] if "command" in item]
        self.assertTrue(follow_commands)
        self.assertTrue(all(
            "--platform a3" in command
            for command in follow_commands
            if " neighbors " in command or " pipeline " in command
        ))

    def test_explicit_platform_enables_preflight_without_task_alias(self):
        with patch("retrieval.plan.grep_results", return_value=[]):
            output = preflight_task("needle API", _search_index(), k=2, platform="a3")

        self.assertTrue(output["platform_filter"]["enabled"])
        self.assertNotIn("a-950.md", [item["path"] for item in output["results"]])

    def test_brief_output_keeps_platform_filter_metadata(self):
        with patch("retrieval.plan.grep_results", return_value=[]):
            output = preflight_task("A3 needle API", _search_index(), k=2)

        self.assertEqual(output["platform_filter"], brief_preflight_output(output)["platform_filter"])


class RetrievalEvalPlatformGuardTest(unittest.TestCase):
    def test_forbidden_path_passes_when_platform_filter_removes_it(self):
        cases = [{
            "id": "a3-platform-guard",
            "task": "A3 needle API",
            "expected": ["b-no-metadata.md"],
            "forbidden": ["a-950.md"],
        }]
        with patch("retrieval.plan.grep_results", return_value=[]):
            result = evaluate(cases, _search_index(), k=2)

        self.assertEqual(1.0, result["platform_guard_pass_rate"])
        self.assertTrue(result["cases"][0]["platform_guard_pass"])
        self.assertEqual([], result["cases"][0]["forbidden_hits"])

    def test_forbidden_path_marks_case_failed_when_it_is_selected(self):
        cases = [{
            "id": "unfiltered-platform-guard",
            "task": "needle API",
            "expected": ["b-no-metadata.md"],
            "forbidden": ["a-950.md"],
        }]
        with patch("retrieval.plan.grep_results", return_value=[]):
            result = evaluate(cases, _search_index(), k=2)

        self.assertEqual(0.0, result["platform_guard_pass_rate"])
        self.assertFalse(result["cases"][0]["platform_guard_pass"])
        self.assertFalse(result["cases"][0]["case_pass"])
        self.assertEqual(["a-950.md"], result["cases"][0]["forbidden_hits"])


class ResourceAndExceptionContractTest(unittest.TestCase):
    def test_index_reader_closes_its_file(self):
        reader = io.StringIO('{"docs": []}')

        with patch("builtins.open", return_value=reader):
            self.assertEqual({"docs": []}, _read_index())

        self.assertTrue(reader.closed)

    def test_index_verify_closes_its_file(self):
        payload = {
            "meta": {
                "index_version": INDEX_VERSION,
                "content_fingerprint": "fixture",
                "bundles": [],
            },
            "docs": [],
        }
        reader = io.StringIO(json.dumps(payload))
        stdout = io.StringIO()

        with patch("retrieval.index.os.path.exists", return_value=True), patch(
            "retrieval.index.concept_paths", return_value=[]
        ), patch("retrieval.index._recompute_fingerprint", return_value="fixture"), patch(
            "builtins.open", return_value=reader
        ), redirect_stdout(stdout):
            cmd_verify(emit_json=True)

        self.assertTrue(reader.closed)

    def test_graph_artifact_reader_closes_its_file(self):
        broken_graph_module = SimpleNamespace(load_nodes=lambda: (_ for _ in ()).throw(RuntimeError("bad graph")))
        reader = io.StringIO('{"nodes": [], "edges": []}')

        with patch.dict(sys.modules, {"okf_graph": broken_graph_module}), patch(
            "retrieval.graph.os.path.exists", return_value=True
        ), patch("builtins.open", return_value=reader):
            self.assertEqual({"nodes": {}, "adj": {}}, _load_graph())

        self.assertTrue(reader.closed)

    def test_verdict_reader_closes_its_file(self):
        reader = io.StringIO('{"scores": {"card.md": 7}}')
        hits = [{"path": "card.md", "score": 1.0, "method": "bm25"}]

        with patch("builtins.open", return_value=reader):
            result = LLMJudge().rerank({"docs": []}, "query", hits, {"verdicts": "fixture.json"})

        self.assertEqual(7.0, result[0]["score"])
        self.assertTrue(reader.closed)

    def test_embedding_runtime_error_keeps_original_cause(self):
        original = RuntimeError("transport failed")
        backend = APIEmbedding()
        client = SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(original))
        )

        with patch.object(backend, "_client", client):
            with self.assertRaises(ModelRuntimeError) as raised:
                backend.encode(["query"])

        self.assertIs(original, raised.exception.__cause__)

    def test_llm_runtime_error_keeps_original_cause(self):
        original = RuntimeError("sdk failed")

        class BrokenMessages:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise original

        broken_sdk = SimpleNamespace(
            AssistantMessage=object,
            TextBlock=object,
            ClaudeAgentOptions=lambda **_kwargs: object(),
            query=lambda **_kwargs: BrokenMessages(),
        )

        client = LLMClient(timeout_s=1)
        with patch.object(client, "_sdk", return_value=broken_sdk):
            with self.assertRaises(ModelRuntimeError) as raised:
                client.complete_text("query")

        self.assertIs(original, raised.exception.__cause__)


class SkillGuidancePlatformPolicyTest(unittest.TestCase):
    def test_skill_documents_a3_filter_and_follow_up_parameter(self):
        text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertIn("`--platform a3`", text)
        self.assertIn("`platforms: [950]`", text)
        self.assertIn("910C", text)
        self.assertIn("后续检索", text)


if __name__ == "__main__":
    unittest.main()
