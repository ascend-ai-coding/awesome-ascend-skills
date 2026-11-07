# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Root trust rule tests: installer dependency checkouts must stay usable when linked.

The cannbot installer provisions cannbot-knowledge under
``<project>/.cannbot/dependencies/<plugin>/cannbot-knowledge`` and such a checkout only
carries ``ops/`` + ``index.md``, so its retrieval index is built lazily on first query.
Root discovery resolves every candidate path, therefore a checkout reached through a
symlink (a shared cache linked into several projects, or an installer that links instead
of cloning) no longer contains the ``dependencies`` / ``cannbot-knowledge`` segments.
The readiness gate must key on the discovery fact instead of the path literal, otherwise
discovery trusts the root while the index build rejects it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
QUERY_SCRIPT = SKILL_ROOT / "scripts" / "knowledge_query.py"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from retrieval import root as root_module  # noqa: E402
from retrieval.root import (  # noqa: E402
    DEPENDENCY_DISCOVERY_SOURCE,
    is_cannbot_dependency_root,
    is_trusted_knowledge_root,
)


INDEX_MD = """---
okf_version: "0.2"
---

# 知识正文

- [算子知识](ops/index.md)
"""

CARD = """---
title: Needle operator
type: Operator
status: verified
platforms: [a2]
---
needle operator implementation
"""


def _write_knowledge_base(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "index.md").write_text(INDEX_MD, encoding="utf-8")
    (root / "ops" / "index.md").write_text("# 算子\n\n- [Needle](needle_op.md)\n", encoding="utf-8")
    (root / "ops" / "needle_op.md").write_text(CARD, encoding="utf-8")


class RootTrustTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        base = Path(cls._tmpdir.name)
        # Shared cache: the knowledge repo checked out once, linked into projects.
        cls.cache = base / "cache" / "cannbot-knowledge"
        _write_knowledge_base(cls.cache / "knowledge")
        # Project whose installer used a symlink instead of a clone.
        cls.linked_project = base / "proj_linked"
        deps = cls.linked_project / ".cannbot" / "dependencies" / "triton-op-generator"
        deps.mkdir(parents=True)
        (deps / "cannbot-knowledge").symlink_to(cls.cache, target_is_directory=True)
        # Project with the plain (non-linked) installer layout.
        cls.literal_project = base / "proj_literal"
        cls.literal_kb = cls.literal_project / ".cannbot" / "dependencies" / "triton-op-generator" / "cannbot-knowledge"
        _write_knowledge_base(cls.literal_kb / "knowledge")
        # A directory that is not a knowledge root at all.
        cls.not_a_root = base / "plain"
        (cls.not_a_root / "ops").mkdir(parents=True)
        # Looks like an installer checkout by directory names, but sits outside any
        # `.cannbot` tree, so it must not be trusted by the literal fallback.
        cls.bogus_dependency = base / "elsewhere" / "dependencies" / "plugin" / "cannbot-knowledge" / "knowledge"
        (cls.bogus_dependency / "ops").mkdir(parents=True)
        (cls.bogus_dependency / "index.md").write_text(INDEX_MD, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run(self, cwd, *args):
        env = dict(os.environ)
        for name in ("CANNBOT_KNOWLEDGE_ROOT", "OKF_KNOWLEDGE_ROOT", "KNOWLEDGE_ROOT",
                     "CANNBOT_KNOWLEDGE_ROOTS", "OKF_KNOWLEDGE_ROOTS", "KNOWLEDGE_ROOTS"):
            env.pop(name, None)
        return subprocess.run(
            [sys.executable, str(QUERY_SCRIPT), *args],
            cwd=str(cwd), capture_output=True, text=True, check=False, env=env,
        )

    def test_literal_dependency_layout_is_trusted(self):
        knowledge = self.literal_kb / "knowledge"
        self.assertTrue(is_cannbot_dependency_root(knowledge))
        self.assertTrue(is_trusted_knowledge_root(knowledge))

    def test_symlinked_dependency_resolves_away_from_layout_but_stays_trusted(self):
        # Document the effect the gate must survive: resolution drops the segments.
        probed = root_module._cannbot_dependency_paths(self.linked_project.resolve())
        self.assertEqual(1, len(probed), "dependency probe should find the linked checkout")
        resolved = str(probed[0].resolve())
        self.assertNotIn("dependencies", Path(resolved).parts)
        # The probe is cwd-relative, exactly like discovery, so evaluate it from the project.
        previous = os.getcwd()
        try:
            os.chdir(self.linked_project)
            self.assertTrue(is_cannbot_dependency_root(resolved))
            self.assertTrue(is_trusted_knowledge_root(resolved))
        finally:
            os.chdir(previous)

    def test_symlinked_dependency_is_discovered_from_project(self):
        result = self._run(self.linked_project, "discover")
        self.assertEqual(0, result.returncode, result.stderr)
        selected = json.loads(result.stdout)["selected"]
        self.assertIsNotNone(selected)
        self.assertEqual(DEPENDENCY_DISCOVERY_SOURCE, selected["source"])
        self.assertNotIn("dependencies", Path(selected["path"]).parts)

    def test_preflight_builds_index_instead_of_rejecting_symlinked_checkout(self):
        cache_index = self.cache / "knowledge" / "search" / "okf.index.json"
        if cache_index.exists():
            cache_index.unlink()
        result = self._run(self.linked_project, "preflight", "--brief", "--task", "needle operator")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("does not look ready", result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("route", payload)
        # The index was built inside the linked checkout, i.e. in the shared cache.
        self.assertTrue(cache_index.is_file())

    def test_plain_directory_is_not_trusted(self):
        self.assertFalse(is_cannbot_dependency_root(self.not_a_root))
        self.assertFalse(is_trusted_knowledge_root(self.not_a_root))

    def test_dependency_named_path_outside_cannbot_is_not_trusted(self):
        # `dependencies/<plugin>/cannbot-knowledge/knowledge` without a `.cannbot`
        # ancestor scores exactly MIN_KNOWLEDGE_SCORE (ops/ + index.md) but has no strong
        # marker, so neither gate may accept it.
        self.assertFalse(is_cannbot_dependency_root(self.bogus_dependency))
        self.assertFalse(is_trusted_knowledge_root(self.bogus_dependency))


if __name__ == "__main__":
    unittest.main()
