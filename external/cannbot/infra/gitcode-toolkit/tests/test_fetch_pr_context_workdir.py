# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fetch_pr_context  # noqa: E402


class FetchPrContextWorkdirTest(unittest.TestCase):
    def test_existing_work_dir_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            work_dir = Path(parent) / "existing-work-dir"
            work_dir.mkdir()
            marker = work_dir / "keep.txt"
            marker.write_text("user data\n", encoding="utf-8")

            with (
                patch.object(
                    fetch_pr_context,
                    "ERROR_LOGGER",
                    logging.getLogger("test"),
                    create=True,
                ),
                patch.object(
                    fetch_pr_context,
                    "clone_repo",
                    side_effect=RuntimeError("clone failed"),
                ) as clone,
            ):
                result = fetch_pr_context.main(
                    ["--repo", "cann/example", "--pr", "1", "--work-dir", str(work_dir)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data\n")
            clone.assert_not_called()

    def test_failed_new_work_dir_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            work_dir = Path(parent) / "nested" / "new-work-dir"
            clone_paths: list[Path] = []
            clone_states: list[bool] = []

            def fail_clone(_owner: str, _repo: str, path: str) -> None:
                clone_paths.append(Path(path))
                work_dir.mkdir(exist_ok=True)
                clone_states.append(work_dir.is_dir())
                (work_dir / "partial-clone").write_text("temporary\n", encoding="utf-8")
                raise RuntimeError("clone failed")

            with (
                patch.object(
                    fetch_pr_context,
                    "ERROR_LOGGER",
                    logging.getLogger("test"),
                    create=True,
                ),
                patch.object(fetch_pr_context, "clone_repo", side_effect=fail_clone),
            ):
                result = fetch_pr_context.main(
                    ["--repo", "cann/example", "--pr", "1", "--work-dir", str(work_dir)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(clone_paths, [work_dir])
            self.assertEqual(clone_states, [True])
            self.assertFalse(work_dir.exists())


if __name__ == "__main__":
    unittest.main()
