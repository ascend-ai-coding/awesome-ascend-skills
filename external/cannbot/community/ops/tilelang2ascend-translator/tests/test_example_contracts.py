# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the actual example sources against a CPU API contract model.

This catches layout/count mistakes; it does not validate the CANN compiler,
device instructions, synchronization timing or performance.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ExampleContractTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required for CPU contract test")
    def test_topk_and_even_odd_sources(self):
        tests = Path(__file__).resolve().parent
        assets = tests.parent / "assets/scalar_vector_examples"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "contracts"
            command = [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(tests / "stubs"),
                "-I",
                str(assets),
                str(tests / "example_contracts.cpp"),
                str(assets / "topk64.cpp"),
                str(assets / "split_even_odd64.cpp"),
                "-o",
                str(output),
            ]
            build = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=60
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run(
                [str(output)], capture_output=True, text=True, check=False, timeout=30
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
