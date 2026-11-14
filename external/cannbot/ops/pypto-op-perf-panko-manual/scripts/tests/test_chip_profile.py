# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The chip envelope is derived from the platform ini, not transcribed.

What is under test is the DERIVATION: that each figure is read where the
platform puts it, that bytes come out as KB, that two SKUs sharing a family are
still read apart, and that a field the ini does not carry is reported rather
than guessed.

None of that needs a real device's figures, so none are stored here. Every ini
is written by `_support.write_platform_ini` into the test's own temporary
directory, with numbers chosen to make the assertion legible: `AscendTestA` and
`AscendTestB` share a family and differ in core count, and `AscendTestC` is
another family that also carries a `[VectorCoreSpec]`.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import (
    CHIP_ENVELOPE_OFFLINE,
    SCRIPTS_DIR,
    init_argv,
    read_state,  # noqa: E402
    run_cli,
    write_platform_ini,
)

import chip_profile  # noqa: E402

HARNESS = SCRIPTS_DIR / "panko_harness.py"

KB = 1024

SOC_A = "AscendTestA"
SOC_B = "AscendTestB"
SOC_C = "AscendTestC"


def _write_all(directory):
    """The three synthetic platforms every test in this file reads."""
    for soc, cube, vector in ((SOC_A, 4, 8), (SOC_B, 6, 12)):
        write_platform_ini(
            directory,
            soc,
            cube=cube,
            vector=vector,
            ub=32 * KB,
            l0a=8 * KB,
            l0b=8 * KB,
            l0c=16 * KB,
            l1=64 * KB,
            arch="1000",
            short="AscendTest",
        )
    # Same L0A / L0B / L1 as the first family and a different UB and L0C -- the
    # shape that lets a transcription survive a generation change unnoticed. Its
    # [VectorCoreSpec] carries a DIFFERENT ub_size, so reading the wrong section
    # cannot pass.
    write_platform_ini(
        directory,
        SOC_C,
        cube=2,
        vector=4,
        ub=64 * KB,
        l0a=8 * KB,
        l0b=8 * KB,
        l0c=32 * KB,
        l1=64 * KB,
        arch="2000",
        short="AscendTestZ",
        vector_ub=128 * KB,
    )
    return Path(directory)


KERNEL = """import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b


def demo(x):
    return k(x, x)
"""


class ReadIniTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ini_dir = _write_all(Path(self._tmp.name))
        self.dirs = [str(self.ini_dir)]
        self.addCleanup(self._tmp.cleanup)

    def test_sizes_are_bytes_in_the_ini_and_kb_out(self):
        """The ini holds bytes; every consumer of the envelope expects KB."""
        text = (self.ini_dir / f"{SOC_A}.ini").read_text(encoding="utf-8")
        self.assertIn(f"ub_size={32 * KB}", text)
        env = self._env(SOC_A)
        self.assertEqual(env["ub_kb"], 32)
        self.assertEqual(env["l1_kb"], 64)
        self.assertEqual(env["l0c_kb"], 16)

    def test_core_counts_are_read_as_counts(self):
        env = self._env(SOC_A)
        self.assertEqual((env["cube_cores"], env["vector_cores"]), (4, 8))

    def test_two_skus_of_one_family_differ_in_core_count(self):
        """The whole argument for deriving rather than transcribing: a count
        taken from one SKU is wrong on its sibling.
        """
        a, b = self._env(SOC_A), self._env(SOC_B)
        for same in ("ub_kb", "l1_kb", "l0a_kb", "l0b_kb", "l0c_kb"):
            self.assertEqual(a[same], b[same], same)
        self.assertNotEqual(a["cube_cores"], b["cube_cores"])
        self.assertNotEqual(a["vector_cores"], b["vector_cores"])

    def test_the_family_name_does_not_determine_the_figures(self):
        """`Short_SoC_version` and `NpuArch` agree across those two SKUs, so
        neither is a key the envelope could be looked up by. The full SoC name --
        the ini's basename -- is the only thing that identifies it, and that is
        exactly what `acl.get_soc_name()` returns.
        """
        seen = []
        for soc in (SOC_A, SOC_B):
            cp = configparser.ConfigParser(strict=False)
            cp.optionxform = str
            cp.read(str(self.ini_dir / f"{soc}.ini"), encoding="utf-8")
            seen.append(
                (cp.get("version", "NpuArch"), cp.get("version", "Short_SoC_version"))
            )
        self.assertEqual(seen[0], seen[1])
        self.assertNotEqual(
            self._env(SOC_A)["cube_cores"], self._env(SOC_B)["cube_cores"]
        )

    def test_only_some_of_the_figures_move_between_families(self):
        """L0A, L0B and L1 are unchanged here, which is how a transcription
        survives a generation change without anyone noticing: most of what it
        says stays true.
        """
        a, c = self._env(SOC_A), self._env(SOC_C)
        for same in ("l0a_kb", "l0b_kb", "l1_kb"):
            self.assertEqual(a[same], c[same], same)
        for moves in ("ub_kb", "l0c_kb", "cube_cores"):
            self.assertNotEqual(a[moves], c[moves], moves)

    def test_ub_size_is_read_section_scoped(self):
        """A platform may carry `ub_size` in [AICoreSpec] AND [VectorCoreSpec].
        A flat key scan gets two hits, so the fixture gives them different
        values and reading the wrong one cannot pass.
        """
        text = (self.ini_dir / f"{SOC_C}.ini").read_text(encoding="utf-8")
        self.assertIn("[VectorCoreSpec]", text)
        self.assertEqual(chip_profile.FIELDS["ub_kb"], ("AICoreSpec", "ub_size"))
        self.assertEqual(self._env(SOC_C)["ub_kb"], 64)

    def test_a_missing_field_is_reported_not_guessed(self):
        bad = self.ini_dir / "AscendTestBroken.ini"
        bad.write_text("[SoCInfo]\ncube_core_cnt=4\n", encoding="utf-8")
        env, why = chip_profile.read_ini(str(bad))
        self.assertIsNone(env)
        self.assertIn("ub_size", why)

    def test_a_non_integer_value_is_reported_not_guessed(self):
        odd = write_platform_ini(self.ini_dir, "AscendTestOdd", ub=32 * KB)
        odd.write_text(
            odd.read_text(encoding="utf-8").replace(
                f"ub_size={32 * KB}", "ub_size=plenty"
            ),
            encoding="utf-8",
        )
        env, why = chip_profile.read_ini(str(odd))
        self.assertIsNone(env)
        self.assertIn("ub_size", why)

    def test_an_unknown_soc_finds_no_ini(self):
        self.assertEqual(chip_profile.find_ini("AscendNope", self.dirs), "")

    def test_the_source_path_is_carried_out(self):
        """a5-roofline-and-levers.md: record which ini was read alongside the
        number.
        """
        self.assertTrue(self._env(SOC_A)["source"].endswith(f"{SOC_A}.ini"))

    def _env(self, soc):
        env, why = chip_profile.read_ini(chip_profile.find_ini(soc, self.dirs))
        self.assertIsNotNone(env, why)
        return env


class InitResolutionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.op_dir = self.root / "custom" / "demo"
        self.op_dir.mkdir(parents=True)
        (self.op_dir / "demo_impl.py").write_text(KERNEL, encoding="utf-8")
        self.cann = self.root / "cann" / "aarch64-linux" / "data" / "platform_config"
        _write_all(self.cann)
        self.addCleanup(self._tmp.cleanup)

    def test_it_refuses_rather_than_falling_back_to_a_transcribed_default(self):
        """The review's words: silently applying one SKU's parameters to another
        is the failure.
        """
        res, rc = self._init()
        self.assertEqual(rc, 3)
        self.assertEqual(res["error"], "chip_envelope_unresolved")
        self.assertIn("ask_user", res)

    def test_the_refusal_asks_the_user_and_names_the_ways_out(self):
        res, _ = self._init()
        self.assertEqual(
            [opt["id"] for opt in res["ask_user"]["options"]],
            ["cann_env", "name_soc", "explicit"],
        )
        self.assertIn("do not choose for them", res["hint"])

    def test_it_derives_from_the_ini_when_the_soc_is_named(self):
        res, rc = self._init(
            {"ASCEND_TOOLKIT_HOME": str(self.root / "cann"), "PANKO_SOC_NAME": SOC_C}
        )
        self.assertEqual(rc, 0)
        env = res["chip_envelope"]
        self.assertEqual((env["cube_cores"], env["vector_cores"]), (2, 4))
        self.assertEqual(env["ub_kb"], 64)
        self.assertTrue(env["source"].endswith(f"{SOC_C}.ini"))

    def test_the_run_records_which_ini_it_read(self):
        self._init(
            {"ASCEND_TOOLKIT_HOME": str(self.root / "cann"), "PANKO_SOC_NAME": SOC_A}
        )
        st = read_state(self.op_dir)
        self.assertTrue(st["chip_envelope"]["source"].endswith(f"{SOC_A}.ini"))
        self.assertEqual(st["chip_envelope"]["soc"], SOC_A)

    def test_an_explicit_envelope_is_recorded_as_explicit(self):
        """So a report cannot read as derived when the numbers were supplied."""
        res, rc = self._init(
            extra=["--chip-envelope", json.dumps(CHIP_ENVELOPE_OFFLINE)]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(res["chip_envelope"]["source"], "explicit:cli")

    def test_an_incomplete_explicit_envelope_is_refused(self):
        res, rc = self._init(extra=["--chip-envelope", '{"ub_kb": 192}'])
        self.assertEqual(rc, 3)
        self.assertIn("missing", res["reason"])

    def _init(self, env=None, extra=()):
        # The real environment minus anything that would resolve a chip for us,
        # so the CANN variables are the only thing under test. A stripped env
        # would also hide optuna and trip the earlier preflight instead.
        e = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("ASCEND") and k != "PANKO_SOC_NAME"
        }
        e.update(env or {})
        return run_cli(
            *init_argv("custom/demo", "custom/demo/demo_impl.py", *extra),
            cwd=self.root,
            env=e,
        )


if __name__ == "__main__":
    unittest.main()
