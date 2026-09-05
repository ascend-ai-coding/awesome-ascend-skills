# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""numpy_broadcast_two / numpy_broadcast_n / simulate 单测。

shape_eval 的 broadcast 逻辑实际上调用本模块，单测放在这里独立验证 SymbolicShape
广播行为，与 shape_eval 解耦。
"""
from __future__ import annotations

import pytest

from evaluators.parser import parse_shape_literal
from evaluators.broadcast import simulate, numpy_broadcast_two, numpy_broadcast_n
from evaluators.types import DslError, SymbolicShape


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


class TestSimulate:
    def test_kind_none_equal(self):
        out = simulate([_sh(["M", "K"]), _sh(["M", "K"])], {"kind": "none"})
        assert isinstance(out.output_shape, SymbolicShape)

    def test_kind_none_violation(self):
        # B 路线：kind=none 只静态校核 rank + const 维；symbol 名差异（M vs N）不再触发
        # 静态失败，放过到 stage 8 / 运行时。下面用 const 维不等触发违例。
        with pytest.raises(DslError, match="numpy_violation|不一致|不等"):
            simulate([_sh(["M", 4]), _sh(["M", 8])], {"kind": "none"})

    def test_kind_none_symbol_name_diff_ok(self):
        # B 路线：symbol 名仅为 owner 命名，跨 input 不参与判等
        out = simulate([_sh(["M", "K"]), _sh(["N", "K"])], {"kind": "none"})
        assert isinstance(out.output_shape, SymbolicShape)

    def test_kind_none_rank_mismatch(self):
        with pytest.raises(DslError, match="numpy_violation|rank|不一致"):
            simulate([_sh(["M", "K"]), _sh(["M"])], {"kind": "none"})

    def test_kind_numpy(self):
        out = simulate([_sh([1, "K"]), _sh(["M", "K"])], {"kind": "numpy"})
        assert out.output_shape.explicit[0].name == "M"

    def test_unknown_kind(self):
        with pytest.raises(DslError, match="未知 broadcast.kind|dsl_parse_error"):
            simulate([_sh(["M"])], {"kind": "wat"})
