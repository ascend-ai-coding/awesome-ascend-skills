# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
QUERY_SCRIPT = SKILL_ROOT / "scripts" / "knowledge_query.py"


class SearchScopeCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.knowledge_root = Path(cls._tmpdir.name)
        cards = {
            "ops/needle_op.md": """---
title: Needle operator
type: Operator
status: verified
---
needle operator implementation
""",
            "reference/fixture/needle_api.md": """---
title: Needle API
type: ASC-DevKit API Reference
status: verified
---
needle API reference
""",
            "runbooks/needle_debug.md": """---
title: Needle debugging
type: Runbook
status: verified
---
needle debugging notes
""",
        }
        for relative_path, content in cards.items():
            path = cls.knowledge_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        (cls.knowledge_root / "index.md").write_text("# Fixture knowledge base\n", encoding="utf-8")

        result = cls._run("build")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    @classmethod
    def _run(cls, *args):
        return subprocess.run(
            [sys.executable, str(QUERY_SCRIPT), "--knowledge-root", str(cls.knowledge_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_search_accepts_scope_and_filters_by_doc_id_prefix(self):
        result = self._run("search", "--query", "needle", "--scope", "ops/")

        self.assertEqual(["ops/needle_op.md"], self._result_paths(result))

    def test_search_accepts_dir_as_scope_alias(self):
        result = self._run("search", "--query", "needle", "--dir", "runbooks/")

        self.assertEqual(["runbooks/needle_debug.md"], self._result_paths(result))

    def test_search_normalizes_reference_scope(self):
        result = self._run("search", "--query", "needle", "--scope", "reference/")

        self.assertEqual(["fixture/needle_api.md"], self._result_paths(result))

    def test_search_treats_all_scope_as_unfiltered(self):
        result = self._run("search", "--query", "needle", "--scope", "all")

        self.assertEqual(
            {"fixture/needle_api.md", "ops/needle_op.md", "runbooks/needle_debug.md"},
            set(self._result_paths(result)),
        )

    def test_grep_treats_all_scope_as_unfiltered(self):
        result = self._run("grep", "needle", "--scope", "all", "--only", "body")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {"fixture/needle_api.md", "ops/needle_op.md", "runbooks/needle_debug.md"},
            {item["path"] for item in json.loads(result.stdout)["matches"]},
        )

    def _result_paths(self, result):
        self.assertEqual(0, result.returncode, result.stderr)
        return [item["path"] for item in json.loads(result.stdout)["results"]]


class SearchScopeSkillContractTest(unittest.TestCase):
    def test_skill_documents_search_scope_without_claiming_all_routes_share_facets(self):
        text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertIn("search   --query Q [--scope/--dir S]", text)
        self.assertIn("`all` 表示不限制路径", text)
        self.assertNotIn("facets（任意路线", text)


if __name__ == "__main__":
    unittest.main()
