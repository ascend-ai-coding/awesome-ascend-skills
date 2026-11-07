# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The correctness command runs without a shell.

`test_command` reaches the harness as a string in a dispatch prompt and is the
one input to E(x) that is not a number. Handing it to `bash -c` made every
character in it executable, so it is parsed here instead: the documented form is
`cd custom/<op> && python3 test_<op>.py`, with optional `NAME=value` prefixes,
and anything that genuinely needs a shell is refused by name rather than
silently mis-run.

What these pin is the parse and the `&&` semantics -- the exit code is `s`, so
"stops at the first failure" and "124 on timeout" are part of the contract.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import SCRIPTS_DIR  # noqa: E402

sys.path.insert(0, str(SCRIPTS_DIR))
from bayesian_optimization import evaluator  # noqa: E402


class CommandParseTest(unittest.TestCase):
    """What the documented forms parse to."""

    def test_the_documented_form(self):
        steps, cwd = evaluator.command_steps("cd custom/demo && python3 test_demo.py")
        self.assertEqual(cwd, "custom/demo")
        self.assertEqual(steps, [(["python3", "test_demo.py"], {})])

    def test_a_bare_command_has_no_cwd(self):
        steps, cwd = evaluator.command_steps("python3 test_demo.py")
        self.assertIsNone(cwd)
        self.assertEqual(steps, [(["python3", "test_demo.py"], {})])

    def test_env_prefixes_become_the_step_environment(self):
        steps, _ = evaluator.command_steps("TILE_FWK_DEVICE_ID=3 python3 t.py")
        self.assertEqual(steps, [(["python3", "t.py"], {"TILE_FWK_DEVICE_ID": "3"})])

    def test_quoting_is_honoured(self):
        steps, _ = evaluator.command_steps('python3 -c "print(1)"')
        self.assertEqual(steps, [(["python3", "-c", "print(1)"], {})])

    def test_successive_cd_segments_compose(self):
        _, cwd = evaluator.command_steps("cd a && cd b && python3 t.py")
        self.assertEqual(cwd, os.path.join("a", "b"))

    def test_an_absolute_cd_replaces_what_came_before(self):
        _, cwd = evaluator.command_steps("cd a && cd /tmp && python3 t.py")
        self.assertEqual(cwd, "/tmp")

    def test_two_commands_are_two_steps(self):
        steps, _ = evaluator.command_steps("python3 a.py && python3 b.py")
        self.assertEqual(len(steps), 2)


class RefusedCommandTest(unittest.TestCase):
    """What is refused, and why refusing beats running it wrong.

    Every one of these used to reach `bash -c` and do something; run without a
    shell they would silently do something ELSE -- a pipe would become two
    arguments, a redirect a filename -- and the wrong answer would be scored as
    a correctness result.
    """

    def test_a_pipe_is_refused(self):
        self._refuses("python3 t.py | tee log")

    def test_a_redirect_is_refused(self):
        self._refuses("python3 t.py > log")

    def test_a_subshell_is_refused(self):
        self._refuses("python3 $(which t.py)")

    def test_a_variable_expansion_is_refused(self):
        self._refuses("python3 ${T}.py")

    def test_a_glob_is_refused(self):
        self._refuses("python3 test_*.py")

    def test_backgrounding_is_refused(self):
        self._refuses("python3 t.py &")

    def test_a_semicolon_is_refused(self):
        self._refuses("cd x ; python3 t.py")

    def test_an_empty_segment_is_refused(self):
        self._refuses("python3 t.py &&")

    def test_a_cd_with_two_arguments_is_refused(self):
        self._refuses("cd a b && python3 t.py")

    def test_a_command_that_only_sets_variables_is_refused(self):
        self._refuses("DEVICE=0")

    def _refuses(self, cmd):
        with self.assertRaises(evaluator.UnsupportedCommand):
            evaluator.command_steps(cmd)


class RunCommandTest(unittest.TestCase):
    """The runner itself, against real processes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_it_runs_and_captures_output(self):
        self._script("ok.py", "print('hello')\n")
        out, code = evaluator.run_command(
            f"cd {self.dir} && {Path(sys.executable).name} ok.py", 30
        )
        self.assertEqual(code, 0)
        self.assertIn("hello", out)

    def test_stderr_is_captured_with_stdout(self):
        self._script("err.py", "import sys; sys.stderr.write('boom'); sys.exit(2)\n")
        out, code = evaluator.run_command(
            f"cd {self.dir} && {Path(sys.executable).name} err.py", 30
        )
        self.assertEqual(code, 2)
        self.assertIn("boom", out)

    def test_it_stops_at_the_first_failure(self):
        self._script("bad.py", "raise SystemExit(3)\n")
        self._script("after.py", "print('SHOULD NOT RUN')\n")
        exe = Path(sys.executable).name
        out, code = evaluator.run_command(
            f"cd {self.dir} && {exe} bad.py && {exe} after.py", 30
        )
        self.assertEqual(code, 3)
        self.assertNotIn("SHOULD NOT RUN", out)

    def test_a_missing_program_is_127_rather_than_a_traceback(self):
        out, code = evaluator.run_command("no_such_program_xyz --version", 30)
        self.assertEqual(code, 127)
        self.assertIn("not found on PATH", out)

    def test_the_step_environment_reaches_the_process(self):
        self._script("env.py", "import os; print(os.environ.get('PANKO_T', 'unset'))\n")
        out, _ = evaluator.run_command(
            f"cd {self.dir} && PANKO_T=42 {Path(sys.executable).name} env.py", 30
        )
        self.assertIn("42", out)

    def test_a_timeout_is_124(self):
        self._script("slow.py", "import time; time.sleep(30)\n")
        _, code = evaluator.run_command(
            f"cd {self.dir} && {Path(sys.executable).name} slow.py", 1
        )
        self.assertEqual(code, 124)

    def _script(self, name, body):
        p = self.dir / name
        p.write_text(body, encoding="utf-8")
        return p


class DefaultRunTest(unittest.TestCase):
    """E(x)'s runner speaks exit codes, including for a command it will not run."""

    def test_a_refused_command_is_126_and_says_why(self):
        code, out = evaluator.default_run("python3 t.py | tee log", 5)
        self.assertEqual(code, 126)
        self.assertIn("needs a shell", out)

    def test_a_good_command_carries_its_code_and_output(self):
        code, out = evaluator.default_run(
            f'{Path(sys.executable).name} -c "print(7)"', 30
        )
        self.assertEqual(code, 0)
        self.assertIn("7", out)


if __name__ == "__main__":
    unittest.main()
