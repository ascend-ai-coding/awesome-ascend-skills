# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Which names in a source file mean `pypto`.

`<op>_impl.py` carries the kernel and the host wrapper together, and an alias is
legal, so every reader that has to tell a pypto call from a torch one needs this
same answer. It lived in two copies -- `predicates` and `bayesian_optimization.domain`
-- which is one copy too many for a test that decides whether a candidate's
statements count as kernel work.

Nothing here imports anything but `ast`, so both readers can have it without
taking on each other's dependencies.
"""

import ast


def _pypto_aliases(node):
    """The names one `import pypto ...` statement binds to the pypto module."""
    out = set()
    for alias in node.names:
        if alias.name == "pypto" or alias.name.startswith("pypto."):
            out.add((alias.asname or alias.name).split(".")[0])
    return out


def pypto_roots(tree):
    """Names bound to the pypto module in this file, `pypto` always among them.

    `import pypto as pp` is legal Python and kernels are written that way, so a
    boundary test hard-coded to the literal `pypto` would answer "not pypto" for
    an aliased kernel. That is the opposite error from counting the wrapper and
    just as wrong: it would hold every cube and broadcast action on a kernel that
    has them.
    """
    roots = {"pypto"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= _pypto_aliases(node)
    return frozenset(roots)
