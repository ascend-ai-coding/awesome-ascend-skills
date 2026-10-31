# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The name the envelope was derived from, checked against the chip in the slot.

The review's sentence begins with the device id: PANKO takes
TILE_FWK_DEVICE_ID and never asks what is behind it. Deriving the envelope from
`acl.get_soc_name()` answers "what chip is in this box", which is the same
question as "what chip is device N" only while the box is homogeneous -- and
says nothing at all when the SoC was named by hand.

So the name is resolved FOR the device, and checked again before the first
evaluation. `acl` is stubbed here: the point under test is which call is made
and what is done with the answer, neither of which needs silicon.
"""

from __future__ import annotations

import sys
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import harness  # noqa: E402

import chip_profile  # noqa: E402


class FakeAcl:
    """`acl` as the two pyACL shapes we have to survive.

    `scoped=False` is a build whose get_soc_name takes no device argument. It is
    not an error case -- it is most of them -- so a device-scoped call that
    raises must fall back rather than lose the name.
    """

    def __init__(self, per_device, box=None, scoped=True):
        self.per_device, self.box, self.scoped = per_device, box, scoped
        self.calls = []

    def get_soc_name(self, device=None):
        self.calls.append(device)
        if device is None:
            if self.box is None:
                raise RuntimeError("no default name")
            return self.box
        if not self.scoped:
            raise TypeError("get_soc_name() takes no arguments")
        return self.per_device[device]


class _Stubbed(unittest.TestCase):
    def stub(self, acl):
        mod = types.ModuleType("acl")
        mod.get_soc_name = acl.get_soc_name
        self._saved = sys.modules.get("acl")
        sys.modules["acl"] = mod
        self.addCleanup(self._restore)
        return acl

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("acl", None)
        else:
            sys.modules["acl"] = self._saved


class LiveSocTest(_Stubbed):
    def test_the_soc_is_resolved_for_the_device_being_measured(self):
        acl = self.stub(
            FakeAcl({0: "AscendTestA", 1: "AscendTestB"}, box="AscendTestA")
        )
        self.assertEqual(chip_profile.live_soc(1), ("AscendTestB", "acl(device=1)"))
        self.assertEqual(acl.calls, [1])

    def test_a_build_that_rejects_the_device_falls_back_and_says_so(self):
        """`how` has to distinguish them: a state file must not read as "device
        1 was identified" when what answered was "this box is a 910B3".
        """
        acl = self.stub(FakeAcl({}, box="AscendTestA", scoped=False))
        self.assertEqual(chip_profile.live_soc(1), ("AscendTestA", "acl"))
        self.assertEqual(acl.calls, [1, None])

    def test_live_soc_ignores_the_environment_variable(self):
        """The whole point. PANKO_SOC_NAME is what gets checked; a name checked
        against itself is not a check.
        """
        self.stub(FakeAcl({0: "AscendTestB"}))
        with unittest.mock.patch.dict("os.environ", {"PANKO_SOC_NAME": "AscendTestA"}):
            self.assertEqual(chip_profile.soc_name(0)[1], "PANKO_SOC_NAME")
            self.assertEqual(chip_profile.live_soc(0)[0], "AscendTestB")

    def test_no_acl_is_no_answer_rather_than_a_guess(self):
        self._saved = sys.modules.get("acl")
        sys.modules["acl"] = None
        self.addCleanup(self._restore)
        self.assertEqual(chip_profile.live_soc(0), ("", ""))


class ConfirmTest(_Stubbed):
    def test_the_names_agreeing_is_ok(self):
        self.stub(FakeAcl({0: "AscendTestA"}))
        self.assertEqual(chip_profile.confirm("AscendTestA", 0)[0], "ok")

    def test_the_names_differing_is_a_mismatch(self):
        self.stub(FakeAcl({0: "AscendTestB"}))
        verdict, live, _ = chip_profile.confirm("AscendTestA", 0)
        self.assertEqual((verdict, live), ("mismatch", "AscendTestB"))

    def test_nothing_to_ask_is_unknown_and_not_a_mismatch(self):
        """A box without pyACL is still a box PANKO can search. What it must not
        do is report that the check passed.
        """
        self._saved = sys.modules.get("acl")
        sys.modules["acl"] = None
        self.addCleanup(self._restore)
        self.assertEqual(chip_profile.confirm("AscendTestA", 0)[0], "unknown")


class GateTest(unittest.TestCase):
    """`confirm_chip` is the gate itself: what it writes, and what it stops."""

    def test_a_match_records_the_confirmation_and_does_not_stop(self):
        st = self._st()
        bad = harness.confirm_chip(
            st, "0", probe=lambda e, d: ("ok", "AscendTestA", "acl(device=0)")
        )
        self.assertIsNone(bad)
        c = st["chip_envelope"]["confirmed"]
        self.assertEqual(c["verdict"], "ok")
        self.assertEqual(c["via"], "acl(device=0)")

    def test_unknown_proceeds_but_is_recorded_as_unconfirmed(self):
        """So a report can say whether the numbers were checked against silicon
        or only against a name.
        """
        st = self._st()
        self.assertIsNone(
            harness.confirm_chip(st, "0", probe=lambda e, d: ("unknown", "", ""))
        )
        self.assertEqual(st["chip_envelope"]["confirmed"]["verdict"], "unknown")

    def test_a_mismatch_halts_and_names_both_chips(self):
        st = self._st()
        bad = harness.confirm_chip(
            st, "0", probe=lambda e, d: ("mismatch", "AscendTestB", "acl(device=0)")
        )
        self.assertIsNotNone(bad)
        self.assertEqual(bad["error"], "chip_mismatch")
        self.assertEqual(bad["expected_soc"], "AscendTestA")
        self.assertEqual(bad["live_soc"], "AscendTestB")
        self.assertIn("AscendTestB", bad["reason"])

    def test_the_mismatch_asks_the_user_rather_than_choosing(self):
        """Re-initialising throws the search away and switching devices may not
        be possible. Neither is the harness's call to make.
        """
        st = self._st()
        bad = harness.confirm_chip(
            st, "0", probe=lambda e, d: ("mismatch", "AscendTestC", "acl")
        )
        self.assertEqual(
            [opt["id"] for opt in bad["ask_user"]["options"]],
            ["re_init", "other_device"],
        )
        self.assertIn("do not choose for them", bad["hint"])

    def test_it_reports_where_the_envelope_came_from(self):
        """A mismatch on a hand-typed PANKO_SOC_NAME is a typo; a mismatch on a
        derived envelope means the device moved under the run. Different
        conversations, so the source is in the refusal.
        """
        st = self._st(source="/p/AscendTestA.ini", via="PANKO_SOC_NAME")
        bad = harness.confirm_chip(
            st, "0", probe=lambda e, d: ("mismatch", "AscendTestB", "acl")
        )
        self.assertEqual(bad["envelope_source"], "/p/AscendTestA.ini")
        self.assertEqual(st["chip_envelope"]["soc_via"], "PANKO_SOC_NAME")

    def test_an_explicit_envelope_names_no_chip_so_there_is_nothing_to_check(self):
        """`--chip-envelope` supplies NUMBERS, not a name, and its documented use
        is measuring a chip deliberately -- including reading one box's figures
        on another. Failing it against the live SoC would close the escape hatch
        that exists for when derivation is wrong. It is recorded as unconfirmed,
        which is what it is.
        """
        st = {"chip_envelope": {"source": "explicit:cli", "cube_cores": 28}}
        self.assertIsNone(
            harness.confirm_chip(
                st,
                "0",
                probe=lambda e, d: ("mismatch", "AscendTestA", "acl")
                if e
                else ("unknown", "AscendTestA", "acl"),
            )
        )
        self.assertEqual(st["chip_envelope"]["confirmed"]["verdict"], "unknown")

    def test_a_state_with_no_recorded_soc_is_unknown_not_a_mismatch(self):
        """Resumed from before the field existed. There is nothing to check
        against, and inventing a failure there would strand the campaign.
        """
        st = {"chip_envelope": {"source": "transcribed:910B3"}}
        self.assertIsNone(harness.confirm_chip(st, "0", probe=chip_profile.confirm))
        self.assertEqual(st["chip_envelope"]["confirmed"]["verdict"], "unknown")

    def _st(self, soc="AscendTestA", source="/p/AscendTestA.ini", via="acl"):
        return {
            "chip_envelope": {
                "soc": soc,
                "source": source,
                "soc_via": via,
                "cube_cores": 20,
                "vector_cores": 40,
            }
        }


if __name__ == "__main__":
    unittest.main()
