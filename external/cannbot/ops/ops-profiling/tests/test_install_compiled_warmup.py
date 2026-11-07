# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_compiled_warmup.py"
SPEC = importlib.util.spec_from_file_location("install_compiled_warmup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class InstallCompiledWarmupTest(unittest.TestCase):
    def test_installs_complete_asset_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            first = MODULE.install(project, "local_warmup")
            second = MODULE.install(project, "local_warmup")
            target = project / "local_warmup"

            self.assertEqual(first["status"], "installed")
            self.assertEqual(second["status"], "already_current")
            self.assertEqual(first["asset_sha256"], second["asset_sha256"])
            self.assertTrue((target / "build.sh").is_file())
            self.assertTrue(bool((target / "build.sh").stat().st_mode & 0o100))
            self.assertTrue((target / "csrc" / "op_kernel" / "compiled_warmup_kernel.cpp").is_file())
            self.assertTrue((target / "ops_profiling_warmup" / "__init__.py").is_file())

            (target / "build" / "ascend950").mkdir(parents=True)
            (target / "ops_profiling_warmup" / "_C.so").write_bytes(b"generated")
            third = MODULE.install(project, "local_warmup")
            self.assertEqual(third["status"], "already_current")

    def test_refuses_to_overwrite_local_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            MODULE.install(project, "local_warmup")
            (project / "local_warmup" / "build.sh").write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "local changes"):
                MODULE.install(project, "local_warmup")

    def test_destination_cannot_escape_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "inside"):
                MODULE.install(Path(temporary), "../outside")


if __name__ == "__main__":
    unittest.main()
