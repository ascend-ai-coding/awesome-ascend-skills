#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
# A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------
"""Regression tests use disposable repositories only; never commit in a user's repo."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SCANNER = SCRIPTS / "clause.check_error_codes.py"
INSTALLER = SCRIPTS / "install_error_code_hooks.py"
CATALOG = (
    SCRIPTS.parents[1]
    / "cann-error-code-standard/references/documented-error-codes.json"
)
spec = importlib.util.spec_from_file_location("error_code_gate", SCANNER)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
BANNED = gate.load_policy(CATALOG)


class LexerTest(unittest.TestCase):
    def codes(self, text):
        return {f.code for f in gate.scan_text(text, "sample.cpp", BANNED)}

    def test_each_deprecated_literal_and_macro(self):
        for name, code in gate.LEGACY_MACROS.items():
            with self.subTest(code=code):
                self.assertEqual({code}, self.codes(f'REPORT("{code}");'))
                self.assertEqual(
                    {code}, self.codes(f"{name}\n(op, 0, actual, expected);")
                )

    def test_current_codes_and_modern_macros(self):
        self.assertEqual(
            set(),
            self.codes(
                'REPORT("EZ0004"); OP_LOGE_WITH_INVALID_INPUT(op, "x"); OP_LOGE_FOR_INVALID_SHAPE(op, "x", a, b);'
            ),
        )
        self.assertEqual(set(), self.codes('"EZ00010 XEZ0001 EZ0001X";'))

    def test_comments_and_quoted_comment_delimiters(self):
        self.assertEqual(
            set(),
            self.codes(
                '// EZ0001\n/* OP_LOGE_WITH_INVALID_ATTR(op, x); EZ0002 */\n"EZ0004";'
            ),
        )
        self.assertEqual({"EZ0001"}, self.codes('"http://server/EZ0001"; // comment'))
        self.assertEqual({"EZ0002"}, self.codes('R"tag(/* EZ0002 // */)tag";'))

    def test_multiline_comments_splicing_and_line_numbers(self):
        text = '/*\nEZ0001\n*/\nREPORT(\n"EZ0002");\n'
        self.assertEqual(
            [(5, "EZ0002")],
            [(f.line, f.code) for f in gate.scan_text(text, "x.cpp", BANNED)],
        )
        self.assertEqual(set(), self.codes("// ignored \\\nEZ0001\n"))
        self.assertEqual({"EZ0003"}, self.codes('"EZ00\\\n03";'))

    def test_concatenation_and_alias_chain(self):
        self.assertEqual({"EZ0005"}, self.codes('"EZ00" /* join */\n"05";'))
        text = "#define OLD OP_LOGE_WITH_INVALID_ATTR\n#define WRAP(x) OLD(x)\nWRAP(1);"
        matches = gate.scan_text(text, "x.cpp", BANNED)
        self.assertTrue(
            any(
                f.entry == "WRAP" and f.line == 3 and f.code == "EZ0002"
                for f in matches
            )
        )


class GitGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Gate Test",
            "GIT_AUTHOR_EMAIL": "gate@example.test",
            "GIT_COMMITTER_NAME": "Gate Test",
            "GIT_COMMITTER_EMAIL": "gate@example.test",
        }
        self.git("init", "-q")
        self.write("sample.cpp", "int initial = 0;\n")
        self.baseline = self.snapshot()
        self.git("update-ref", "HEAD", self.baseline)
        self.git("update-ref", "refs/remotes/origin/main", self.baseline)

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            text=True,
            capture_output=True,
            env=self.env,
            check=True,
        )
        return result.stdout.strip()

    def write(self, name, text, stage=True):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if stage:
            self.git("add", "--", name)

    def snapshot(self, parents=()):
        tree = self.git("write-tree")
        args = ["commit-tree", tree, "-m", "synthetic hook test object"]
        for parent in parents:
            args += ["-p", parent]
        return self.git(*args)

    def run_gate(self, *args, data=None):
        return subprocess.run(
            [
                sys.executable,
                str(SCANNER),
                "--root",
                str(self.repo),
                "--catalog",
                str(CATALOG),
                *args,
            ],
            input=data,
            text=True,
            capture_output=True,
            env=self.env,
        )

    def test_index_not_working_tree_controls_result(self):
        self.write("sample.cpp", 'REPORT("EZ0001");\n')
        self.write("sample.cpp", 'REPORT("EZ0004");\n', stage=False)
        result = self.run_gate("--staged")
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("EZ0001", result.stdout)
        self.git("add", "sample.cpp")
        self.write("sample.cpp", 'REPORT("EZ0002");\n', stage=False)
        self.assertEqual(0, self.run_gate("--staged").returncode)

    def test_legacy_lines_and_deletion_do_not_block(self):
        self.write("sample.cpp", 'REPORT("EZ0001");\nint a = 1;\n')
        old = self.snapshot([self.baseline])
        self.git("update-ref", "HEAD", old)
        self.write("sample.cpp", 'REPORT("EZ0001");\nint a = 2;\n')
        self.assertEqual(0, self.run_gate("--staged").returncode)
        self.git("rm", "-f", "sample.cpp")
        self.assertEqual(0, self.run_gate("--staged").returncode)

    def test_new_literal_suffix_in_concat_is_blocked(self):
        self.write("sample.cpp", 'auto code = "EZ00"\n"04";\n')
        old = self.snapshot([self.baseline])
        self.git("update-ref", "HEAD", old)
        self.write("sample.cpp", 'auto code = "EZ00"\n"01";\n')
        self.assertEqual(1, self.run_gate("--staged").returncode)

    def test_uncommenting_existing_line_is_a_new_reference(self):
        self.write("sample.cpp", '/*\nREPORT("EZ0002");\n*/\n')
        old = self.snapshot([self.baseline])
        self.git("update-ref", "HEAD", old)
        self.write("sample.cpp", 'REPORT("EZ0002");\n')
        self.assertEqual(1, self.run_gate("--staged").returncode)

    def test_inserted_lines_do_not_turn_unchanged_legacy_into_findings(self):
        self.write("sample.cpp", 'REPORT("EZ0001");\n')
        old = self.snapshot([self.baseline])
        self.git("update-ref", "HEAD", old)
        self.write("sample.cpp", 'int x = 0;\n\nREPORT("EZ0001");\n')
        self.assertEqual(0, self.run_gate("--staged").returncode)

    def test_new_alias_definition_activates_unchanged_call(self):
        self.write("sample.cpp", "#define OLD ModernMacro\nOLD(op, x);\n")
        old = self.snapshot([self.baseline])
        self.git("update-ref", "HEAD", old)
        self.write("sample.cpp", "#define OLD OP_LOGE_WITH_INVALID_ATTR\nOLD(op, x);\n")
        result = self.run_gate("--staged")
        self.assertEqual(1, result.returncode)
        self.assertIn("sample.cpp:2:", result.stdout)

    def test_unusual_filename_and_nested_alias(self):
        self.write(
            "dir with spaces/[x] file.cpp",
            "#define OLD OP_LOGE_WITH_INVALID_ATTR\nOLD(op, v);\n",
        )
        self.assertEqual(1, self.run_gate("--staged").returncode)
        self.write("dir with spaces/[x] file.cpp", '// EZ0001\nREPORT("EZ0004");\n')
        self.assertEqual(0, self.run_gate("--staged").returncode)

    def test_non_sources_and_excluded_tree_are_not_scanned(self):
        self.write("readme.md", "EZ0001")
        self.write("third_party/old.cpp", 'REPORT("EZ0001");')
        self.write("build/old.cpp", 'REPORT("EZ0001");')
        self.assertEqual(0, self.run_gate("--staged").returncode)
        self.write("experimental/new.cpp", 'REPORT("EZ0007");')
        self.assertEqual(1, self.run_gate("--staged").returncode)

    def test_each_outgoing_commit_scanned_even_if_later_fixed(self):
        self.write("sample.cpp", 'REPORT("EZ0001");\n')
        bad = self.snapshot([self.baseline])
        self.write("sample.cpp", 'REPORT("EZ0004");\n')
        fixed = self.snapshot([bad])
        result = self.run_gate(
            "--pre-push",
            "origin",
            "unused",
            data=f"refs/heads/main {fixed} refs/heads/main {self.baseline}\n",
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn(bad, result.stdout)

    def test_new_branch_and_delete(self):
        self.write("sample.cpp", 'REPORT("EZ0006");\n')
        bad = self.snapshot([self.baseline])
        zeros = "0" * len(bad)
        result = self.run_gate(
            "--pre-push",
            "origin",
            "unused",
            data=f"refs/heads/topic {bad} refs/heads/topic {zeros}\n",
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertEqual(
            0,
            self.run_gate(
                "--pre-push",
                "origin",
                "unused",
                data=f"(delete) {zeros} refs/heads/topic {bad}\n",
            ).returncode,
        )
        self.assertEqual(
            0,
            self.run_gate(
                "--pre-push",
                "origin",
                "unused",
                data=f"refs/heads/new {self.baseline} refs/heads/new {zeros}\n",
            ).returncode,
        )

    def test_merge_commit_and_multiple_refs(self):
        self.write("sample.cpp", 'REPORT("EZ0003");\n')
        side = self.snapshot([self.baseline])
        merge = self.snapshot([self.baseline, side])
        data = (
            f"refs/heads/good {self.baseline} refs/heads/good {self.baseline}\n"
            f"refs/heads/topic {merge} refs/heads/topic {self.baseline}\n"
        )
        self.assertEqual(
            1, self.run_gate("--pre-push", "origin", "unused", data=data).returncode
        )

    def test_scan_errors_fail_closed(self):
        result = self.run_gate("--catalog", str(self.repo / "missing.json"), "--staged")
        self.assertEqual(2, result.returncode)
        self.assertEqual(
            2,
            self.run_gate(
                "--pre-push", "origin", "unused", data="malformed\n"
            ).returncode,
        )
        self.assertEqual(2, self.run_gate("--base", "missing-ref").returncode)

    def install(self):
        return subprocess.run(
            [
                sys.executable,
                str(INSTALLER),
                "--repo",
                str(self.repo),
                "--catalog",
                str(CATALOG),
            ],
            text=True,
            capture_output=True,
            env=self.env,
        )

    def test_installation_hooks_and_index_preservation(self):
        if os.name == "nt":
            self.skipTest("POSIX Git hook execution verified on the target Linux host")
        self.assertEqual(0, self.install().returncode)
        self.assertEqual(0, self.install().returncode)  # idempotent
        self.write("sample.cpp", 'REPORT("EZ0005");\n')
        tree = self.git("write-tree")
        hook = self.repo / ".git/hooks/pre-commit"
        result = subprocess.run(
            [str(hook)], cwd=self.repo, env=self.env, text=True, capture_output=True
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertEqual(tree, self.git("write-tree"))
        self.assertEqual(self.baseline, self.git("rev-parse", "HEAD"))
        bad = self.snapshot([self.baseline])
        push_hook = hook.parent / "pre-push"
        payload = f"refs/heads/topic {bad} refs/heads/topic {self.baseline}\n"
        push_result = subprocess.run(
            [str(push_hook), "origin", "unused"],
            cwd=self.repo,
            input=payload,
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(1, push_result.returncode, push_result.stderr)
        self.write("sample.cpp", 'REPORT("EZ0004");\n')
        self.assertEqual(
            0,
            subprocess.run(
                [str(hook)], cwd=self.repo, env=self.env, capture_output=True
            ).returncode,
        )
        good = self.snapshot([self.baseline])
        payload = f"refs/heads/topic {good} refs/heads/topic {self.baseline}\n"
        self.assertEqual(
            0,
            subprocess.run(
                [str(push_hook), "origin", "unused"],
                cwd=self.repo,
                input=payload,
                env=self.env,
                text=True,
                capture_output=True,
            ).returncode,
        )
        manifest = json.loads(
            (hook.parent / "cann-error-code/manifest.json").read_text()
        )
        self.assertEqual("cann-error-code-standard", manifest["source_skill"])

    def test_custom_hook_is_preserved(self):
        hook = self.repo / ".git/hooks/pre-commit"
        original = "#!/bin/sh\nexit 17\n"
        hook.write_text(original)
        self.assertEqual(2, self.install().returncode)
        self.assertEqual(original, hook.read_text())
        self.assertFalse((hook.parent / "pre-push").exists())

    def test_respects_effective_hooks_path(self):
        self.git("config", "core.hooksPath", ".git/custom-hooks")
        result = self.install()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue((self.repo / ".git/custom-hooks/pre-commit").is_file())
        self.assertTrue((self.repo / ".git/custom-hooks/pre-push").is_file())
        self.assertFalse((self.repo / ".git/hooks/pre-commit").exists())

    def test_hook_path_metacharacters_are_literal(self):
        hooks_dir = self.repo / ".git" / "hooks with spaces ' ; & literal"
        self.git("config", "core.hooksPath", str(hooks_dir))
        result = self.install()
        self.assertEqual(0, result.returncode, result.stderr)
        self.write("sample.cpp", 'REPORT("EZ0005");\n')
        result = subprocess.run(
            [str(hooks_dir / "pre-commit")],
            cwd=self.repo, env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("EZ0005", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
