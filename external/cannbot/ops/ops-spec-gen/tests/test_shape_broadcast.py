# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy_broadcast_two / numpy_broadcast_n 单测。

shape_eval 的 broadcast 逻辑实际上调用本模块，单测放在这里独立验证 SymbolicShape
广播行为，与 shape_eval 解耦。
"""
from __future__ import annotations

import pytest

from evaluators.parser import parse_shape_literal
from evaluators.shape_eval import numpy_broadcast_two, numpy_broadcast_n
from evaluators.types import DslError


def _sh(lst):
    return parse_shape_literal(lst)


class TestNumpyBroadcastTwo:
    def test_two_folded(self):
        out = numpy_broadcast_two(_sh(["...d"]), _sh(["...d"]))
        # 同名折叠维 → folded_name 保留
        assert out.folded_name is not None
        assert out.explicit == []

    def test_explicit_equal(self):
        out = numpy_broadcast_two(_sh(["M", "K"]), _sh(["M", "K"]))
        assert [d.name for d in out.explicit] == ["M", "K"]

    def test_one_against_dim(self):
        # const 1 广播
        out = numpy_broadcast_two(_sh([1, "K"]), _sh(["M", "K"]))
        # const 1 让步给 M
        assert out.explicit[0].name == "M"

    def test_incompatible_consts(self):
        with pytest.raises(DslError, match="incompatible_dims|无法广播"):
            numpy_broadcast_two(_sh([3]), _sh([4]))


class TestNumpyBroadcastN:
    def test_three_shapes(self):
        out = numpy_broadcast_n([_sh(["M"]), _sh([1]), _sh(["M"])])
        assert out.explicit[0].name == "M"

    def test_empty_raises(self):
        with pytest.raises(DslError):
            numpy_broadcast_n([])


    def test_multi_operand_numpy_conflict_is_independent_of_input_order(self):
        from itertools import permutations
        shapes = [_sh([3, 4]), _sh([3, "N"]), _sh([3, 5])]
        for order in permutations(shapes):
            with pytest.raises(DslError, match="incompatible_dims"):
                numpy_broadcast_n(list(order))
