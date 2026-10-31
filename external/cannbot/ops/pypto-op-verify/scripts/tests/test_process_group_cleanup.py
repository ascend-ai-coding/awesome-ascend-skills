# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real process-tree cleanup after the test leader exits on SIGTERM."""

import importlib.util
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_kernel_test.py"
SPEC = importlib.util.spec_from_file_location("kernel_test_process_cleanup", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def alive(pid):
    stat = runner._read_proc_stat(str(pid))
    return stat is not None and stat[0] != "Z"


class ProcessCleanupTest(unittest.TestCase):
    def test_term_resistant_descendant_is_killed_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "descendant.pid"
            child_source = (
                "import os, pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(60)"
            )
            leader_source = (
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
                "time.sleep(60)"
            )
            child = runner.MonitoredProcess(
                [sys.executable, "-c", leader_source, child_source, str(pid_file)],
                Path(directory) / "test.log",
            )
            try:
                deadline = time.monotonic() + 10
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_file.exists(), "descendant did not become ready")
                pid = int(pid_file.read_text())
                self.assertTrue(alive(pid))
                child.kill_tree()
                deadline = time.monotonic() + 3
                while alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(
                    alive(pid), "descendant survived process-group cleanup"
                )
                self.assertIsNotNone(child.returncode)
            finally:
                try:
                    os.killpg(child.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.proc.wait(timeout=5)
                child.close()

    def test_already_reaped_process_is_safe_to_clean_up(self):
        with tempfile.TemporaryDirectory() as directory:
            child = runner.MonitoredProcess(
                [sys.executable, "-c", "pass"], Path(directory) / "test.log"
            )
            try:
                child.proc.wait(timeout=5)
                child.kill_tree()
                self.assertEqual(child.returncode, 0)
            finally:
                child.close()


if __name__ == "__main__":
    unittest.main()
