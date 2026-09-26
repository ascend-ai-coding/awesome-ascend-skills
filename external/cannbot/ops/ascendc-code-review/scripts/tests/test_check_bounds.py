# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for clause.check_bounds.py — C truncation semantics.

Validates that division/modulo use C/C++ truncation-toward-zero semantics
(not Python floor semantics), preventing false-negative wraparound detection
on negative operands (issue #527).
"""

import importlib.util
import os
import sys

import pytest


def _load_check_bounds():
    mod_path = os.path.join(os.path.dirname(__file__), "..", "clause.check_bounds.py")
    spec = importlib.util.spec_from_file_location("clause_check_bounds", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clause_check_bounds"] = mod
    spec.loader.exec_module(mod)
    return mod


cb = _load_check_bounds()


# ─── c_div: C truncation division ─────────────────────────────────

@pytest.mark.parametrize("l,r,expected", [
    (-5, 3, -1),
    (5, -3, -1),
    (-5, -3, 1),
    (5, 3, 1),
    (7, 3, 2),
    (-7, 3, -2),
    (7, -3, -2),
    (-7, -3, 2),
    (0, 3, 0),
    (0, -3, 0),
    (3, 5, 0),
    (-3, 5, 0),
    (1, 1, 1),
    (-1, -1, 1),
])
def test_c_div_truncation(l, r, expected):
    assert cb.c_div(l, r) == expected


def test_c_div_by_zero():
    assert cb.c_div(5, 0) is None
    assert cb.c_div(-5, 0) is None
    assert cb.c_div(0, 0) is None


def test_c_div_large_integer():
    big = 10 ** 20
    assert cb.c_div(-big, 3) == -(big // 3)
    assert cb.c_div(big, -3) == -(big // 3)


# ─── c_mod: C truncation modulo ──────────────────────────────────

@pytest.mark.parametrize("l,r,expected", [
    (-5, 3, -2),
    (5, -3, 2),
    (-5, -3, -2),
    (5, 3, 2),
    (7, 3, 1),
    (-7, 3, -1),
    (7, -3, 1),
    (-7, -3, -1),
    (0, 3, 0),
    (0, -3, 0),
    (3, 5, 3),
    (-3, 5, -3),
])
def test_c_mod_truncation(l, r, expected):
    assert cb.c_mod(l, r) == expected


def test_c_mod_by_zero():
    assert cb.c_mod(5, 0) is None
    assert cb.c_mod(-5, 0) is None
    assert cb.c_mod(0, 0) is None


# ─── check(): end-to-end wraparound / overflow detection ─────────

def test_check_negative_modulo_triggers_wraparound():
    """Issue #527: a % b with a=-5, b=3 → C gives -2 → wraparound risk."""
    rc = cb.check("a % b", ["a=int32_t:-5:-5", "b=int32_t:3:3"], "wraparound")
    assert rc == 1


def test_check_positive_modulo_safe():
    """No false positive: 5 % 3 == 2, non-negative → SAFE."""
    rc = cb.check("a % b", ["a=int32_t:5:5", "b=int32_t:3:3"], "wraparound")
    assert rc == 0


def test_check_negative_division_triggers_wraparound():
    """-5 / 3 == -1 (C truncation) → negative → wraparound for unsigned."""
    rc = cb.check("a / b", ["a=int32_t:-5:-5", "b=int32_t:3:3"], "wraparound")
    assert rc == 1


def test_check_cross_zero_modulo_triggers_wraparound():
    """a % b with a in [-3, 5], b=3: min < 0 via near-zero sampling."""
    rc = cb.check("a % b", ["a=int32_t:-3:5", "b=int32_t:3:3"], "wraparound")
    assert rc == 1


def test_check_pure_negative_modulo_triggers_wraparound():
    """a % b with a in [-9, -3], b=3: near-endpoint sampling catches -2 (at a=-8)."""
    rc = cb.check("a % b", ["a=int32_t:-9:-3", "b=int32_t:3:3"], "wraparound")
    assert rc == 1


def test_counter_example_cross_zero_modulo_reproducible():
    """Counter-example for a % b (a in [-3,5], b=3) must actually trigger wraparound."""
    var_info = cb.parse_variables(["a=int32_t:-3:5", "b=int32_t:3:3"])
    ast = cb.Parser(cb.tokenize("a % b")).parse()
    counter = cb.pick_values(ast, var_info, want_min=True)
    result = cb.c_mod(counter["a"], counter["b"])
    assert result < 0, f"counter a={counter['a']}, b={counter['b']} gives {result}, should be < 0"


def test_counter_example_pure_negative_modulo_reproducible():
    """Counter-example for a % b (a in [-9,-3], b=3) must trigger wraparound."""
    var_info = cb.parse_variables(["a=int32_t:-9:-3", "b=int32_t:3:3"])
    ast = cb.Parser(cb.tokenize("a % b")).parse()
    counter = cb.pick_values(ast, var_info, want_min=True)
    result = cb.c_mod(counter["a"], counter["b"])
    assert result < 0, f"counter a={counter['a']}, b={counter['b']} gives {result}, should be < 0"


def test_check_positive_division_safe():
    """No false positive: 5 / 3 == 1, non-negative → SAFE."""
    rc = cb.check("a / b", ["a=int32_t:5:5", "b=int32_t:3:3"], "wraparound")
    assert rc == 0


def test_check_divzero():
    rc = cb.check("a / b", ["a=int32_t:1:10", "b=int32_t:0:5"], "divzero")
    assert rc == 1


def test_check_modulo_divzero():
    rc = cb.check("a % b", ["a=int32_t:1:10", "b=int32_t:0:5"], "divzero")
    assert rc == 1
