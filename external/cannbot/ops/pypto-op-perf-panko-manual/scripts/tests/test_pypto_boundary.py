# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The kernel and its host wrapper live in one file and must not be confused.

`<op>_impl.py` holds the PyPTO kernel and the torch wrapper that launches it.
Anything that reads the file to decide what the KERNEL is or holds has to stop
at that boundary: a `torch.matmul` in the wrapper is not a cube op in the
kernel, and a `torch.cat` there holds no unified buffer.

Both directions are errors. Counting the wrapper opens actions whose delta has
nowhere to land and inflates the UB estimate; a boundary test hard-coded to the
literal `pypto` would answer "not pypto" for an aliased kernel and hold every
cube action it really has.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _support import SCRIPTS_DIR  # noqa: F401, E402

import predicates  # noqa: E402
import bayesian_optimization as bayesian  # noqa: E402

WRAPPER_DOES_CUBE = """
import torch
import pypto


@pypto.frontend.jit()
def demo_kernel(x, y):
    pypto.set_vec_tile_shapes(64, 128)
    t = pypto.view(x, [64, 128])
    o = pypto.sigmoid(t)
    pypto.assemble(o, y)


def demo(x):
    w = torch.matmul(x, x)
    b = torch.expand(w, [4, 4])
    out = torch.empty_like(b)
    demo_kernel(b, out)
    return out
"""

KERNEL_DOES_CUBE = """
import torch
import pypto


@pypto.frontend.jit()
def demo_kernel(x, y):
    pypto.set_cube_tile_shapes([128, 128], [64, 64], [128, 128])
    a = pypto.matmul(x, x)
    pypto.assemble(a, y)


def demo(x):
    out = torch.empty_like(x)
    demo_kernel(x, out)
    return out
"""

WRAPPER_DOES_ELEMENTWISE = """
import torch
import pypto


@pypto.frontend.jit()
def demo_kernel(x, y):
    pypto.set_vec_tile_shapes(64, 128)
    t = pypto.view(x, [64, 128])
    o = pypto.sigmoid(t)
    pypto.assemble(o, y)


def demo(x, w):
    a = torch.cat([x, w], dim=0)
    b = torch.add(a, w)
    c = b + a
    d = torch.mul(c, b)
    out = torch.empty_like(d)
    demo_kernel(d, out)
    return out
"""


class PredicateBoundaryTest(unittest.TestCase):
    def test_wrapper_cube_does_not_open_a_cube_action(self):
        calls = self.pypto_calls(WRAPPER_DOES_CUBE)
        self.assertFalse(calls & predicates.CUBE_CALLS)

    def test_wrapper_expand_is_not_an_in_kernel_broadcast(self):
        calls = self.pypto_calls(WRAPPER_DOES_CUBE)
        self.assertFalse(calls & predicates.BROADCAST_CALLS)

    def test_a_real_kernel_cube_op_still_opens_the_action(self):
        """The guard must not close actions the kernel genuinely has."""
        calls = self.pypto_calls(KERNEL_DOES_CUBE)
        self.assertTrue(calls & predicates.CUBE_CALLS)

    def test_an_aliased_import_is_still_pypto(self):
        aliased = KERNEL_DOES_CUBE.replace(
            "import pypto\n", "import pypto as pp\n"
        ).replace("pypto.", "pp.")
        tree = ast.parse(aliased)
        self.assertIn("pp", predicates.pypto_roots(tree))
        self.assertTrue(predicates.pypto_calls(tree) & predicates.CUBE_CALLS)

    def test_torch_is_never_a_pypto_root(self):
        tree = ast.parse(WRAPPER_DOES_CUBE)
        self.assertEqual(set(predicates.pypto_roots(tree)), {"pypto"})

    def pypto_calls(self, src):
        return predicates.pypto_calls(ast.parse(src))


class ResidencyScopeTest(unittest.TestCase):
    def test_wrapper_statements_are_not_charged_to_a_tile_site(self):
        """The site holds `t` and `o`: operands + 1 = 2.

        The whole-module walk charged the wrapper's cat/add/BinOp to vec#0 and
        returned 3 -- a 50% over-estimate of the buffer the site needs, which
        shrinks the per-site cap and rejects legal tiles before the device.
        """
        self.assertEqual(
            bayesian.domain.vec_residency(WRAPPER_DOES_ELEMENTWISE), {"vec#0": 2}
        )

    def test_kernel_elementwise_chain_is_still_counted(self):
        src = """
import pypto


@pypto.frontend.jit()
def demo_kernel(x, w, y):
    pypto.set_vec_tile_shapes(64, 128)
    a = pypto.view(x, [64, 128])
    b = pypto.view(w, [64, 128])
    c = pypto.add(a, b)
    pypto.assemble(c, y)
"""
        res = bayesian.domain.vec_residency(src)
        self.assertEqual(res, {"vec#0": 3})  # a, b, c

    def test_a_file_with_no_jit_function_keeps_the_whole_module_walk(self):
        """Not a PyPTO kernel at all; returning nothing would be a worse answer."""
        src = """
import pypto

pypto.set_vec_tile_shapes(64, 128)
a = pypto.view(x, [64, 128])
b = pypto.sigmoid(a)
"""
        self.assertEqual(bayesian.domain.vec_residency(src), {"vec#0": 2})


if __name__ == "__main__":
    unittest.main()
