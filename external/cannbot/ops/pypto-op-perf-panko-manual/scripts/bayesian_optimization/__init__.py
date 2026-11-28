# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bayesian optimization of the numeric tile space, for Stage 7.

Imported as a package, so a call site reads `bayesian.space.HW` rather than a
`bo_` prefix repeated on every module:

    import bayesian_optimization as bayesian

    apply      AST discovery of tile call sites, byte-exact argument rewriting
    normalize  is a proposed normalisation admissible? (static, INIT)
    space      discrete domains and the static feasibility gate
    lever      ask / tell over one action's tile sites, persisted in search_state
    driver     a closed TPE loop over a fixed program
    evaluator  the frozen E(x) wired as that loop's objective
    block      the two composed: one request in, one configuration out
    domain     the derived tile domain, and its fingerprint
    seed       space-covering start points
    semantic   is this candidate the same program as the incumbent?
    const      integer constants that set loop trip counts

`driver` and `evaluator` used to live only on `improve/bo-config-search`,
excluded from this package on the grounds that owning an evaluation loop "is the
one thing a per-action lever cannot do". That was true of a lever asked for one
candidate at a time and it was the wrong constraint: measured over two operators,
the per-candidate design produced three BO trials on one kernel and zero on the
other, and every tile that actually won was found by the language model. A block
owns the loop and is still fired at the lever's moment -- the INIT-time driver's
mechanism, at the per-action lever's timing.
"""

from . import (
    apply,  # noqa: F401
    block,  # noqa: F401
    const,  # noqa: F401
    domain,  # noqa: F401
    driver,  # noqa: F401
    evaluator,  # noqa: F401
    lever,  # noqa: F401
    normalize,  # noqa: F401
    seed,  # noqa: F401
    semantic,  # noqa: F401
    space,  # noqa: F401
)
