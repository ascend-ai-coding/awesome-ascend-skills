# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The skill finds its own files, wherever it was installed.

Quickstart installs the plugin into any project. The skill arrives there as
`.opencode/skills/pypto-op-perf-panko-manual` -- a symlink, so the files are reachable --
and that project has no `cannbot-skills/ops/` tree at all. A path spelled from
the repo root resolves only when the working directory happens to be a clone,
so the agent could discover the skill and then fail on its first harness call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import CHIP_ARGS, HARNESS, SCRIPTS_DIR, init_argv  # noqa: E402

SKILL_DIR = SCRIPTS_DIR.parent

KERNEL = """import pypto


@pypto.frontend.jit()
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


class SkillPathResolutionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(self._tmp.name) / "some-project"
        (self.project / ".opencode" / "skills").mkdir(parents=True)
        self.installed = (
            self.project / ".opencode" / "skills" / "pypto-op-perf-panko-manual"
        )
        os.symlink(SKILL_DIR, self.installed)
        self.op_dir = self.project / "custom" / "demo"
        self.op_dir.mkdir(parents=True)
        (self.op_dir / "demo_impl.py").write_text(KERNEL, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def _installed_init(self, *extra):
        """Run `init` through the installed symlink and return its payload."""
        res = subprocess.run(
            [
                sys.executable,
                str(self.installed / "scripts" / "panko_harness.py"),
                *init_argv(
                    "custom/demo",
                    "custom/demo/demo_impl.py",
                    *CHIP_ARGS,
                    *extra,
                    catalog=None,
                ),
            ],
            cwd=self.project,
            capture_output=True,
            text=True,
        )
        return json.loads(res.stdout)

    def test_the_project_has_no_repo_tree(self):
        """The premise: a path spelled from the repo root cannot resolve here."""
        self.assertFalse((self.project / "cannbot-skills").exists())

    def test_init_through_the_installed_symlink_finds_its_catalogue(self):
        """No --catalog, cwd is the project, invoked through .opencode/skills.

        `open` counts the seeded frontier. Zero would mean the catalogue was not
        found -- which is what the hardcoded path produced here.
        """
        payload = self._installed_init()
        self.assertTrue(payload["initialized"])
        self.assertGreater(payload["open"], 0)

    def test_init_echoes_what_it_resolved(self):
        payload = self._installed_init()
        self.assertTrue(os.path.isdir(payload["skill_root"]))
        self.assertTrue(os.path.isfile(payload["catalog"]))

    def test_an_explicit_catalog_still_wins(self):
        """The default must not take the argument away from a caller who has one."""
        other = Path(self._tmp.name) / "other_catalog.json"
        other.write_text("[]", encoding="utf-8")
        payload = self._installed_init("--catalog", str(other))
        self.assertEqual(payload["catalog"], str(other))
        self.assertEqual(payload["open"], 0)

    def test_skill_root_follows_the_symlink_to_the_real_layout(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        import importlib.util

        spec = importlib.util.spec_from_file_location("panko_harness", HARNESS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(Path(mod.SKILL_ROOT).resolve(), SKILL_DIR.resolve())
        self.assertTrue(Path(mod.DEFAULT_CATALOG).is_file())


if __name__ == "__main__":
    unittest.main()
