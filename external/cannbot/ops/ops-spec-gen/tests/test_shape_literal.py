# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""shape_literal parser 单测：解析 inputs[].shape.symbolic 列表为 SymbolicShape。

不涉及 shape_rule 求值（那是 shape_eval 的事），只测列表 → SymbolicShape 的解析行为。
"""
from __future__ import annotations

import pytest

from evaluators.parser import parse_shape_literal
from evaluators.types import Dim, SymbolicShape, DslError


class TestShapeLiteral:
    def test_pure_folded(self):
        sh = parse_shape_literal(["...d"])
        assert sh.folded_name == "d"
        assert sh.explicit == []

    def test_folded_plus_explicit(self):
        sh = parse_shape_literal(["...batch", "M", "K"])
        assert sh.folded_name == "batch"
        assert [d.name for d in sh.explicit] == ["M", "K"]

    def test_pure_explicit(self):
        sh = parse_shape_literal(["M", "K"])
        assert sh.folded_name is None
        assert [d.name for d in sh.explicit] == ["M", "K"]

    def test_const_dims(self):
        sh = parse_shape_literal([2, 3, 4])
        assert all(d.kind == "const" for d in sh.explicit)
        assert [d.value for d in sh.explicit] == [2, 3, 4]

    def test_empty(self):
        sh = parse_shape_literal([])
        assert sh.folded_name is None and sh.explicit == []

    def test_folded_must_be_first(self):
        with pytest.raises(DslError, match="折叠维"):
            parse_shape_literal(["M", "...d"])

    def test_at_most_one_folded(self):
        with pytest.raises(DslError, match="折叠维"):
            parse_shape_literal(["...a", "...b"])

    def test_lowercase_explicit_rejected(self):
        with pytest.raises(DslError, match="大写起始"):
            parse_shape_literal(["m"])

    def test_negative_const_rejected(self):
        with pytest.raises(DslError, match="非负"):
            parse_shape_literal([-1])

    def test_bool_rejected(self):
        with pytest.raises(DslError, match="bool"):
            parse_shape_literal([True])


class TestSymbolicShape:
    def test_rank_min_folded(self):
        sh = parse_shape_literal(["...d", "M"])
        assert sh.rank_min == 1   # folded 不计入

    def test_rank_min_explicit(self):
        sh = parse_shape_literal(["M", "K"])
        assert sh.rank_min == 2

    def test_is_fully_explicit(self):
        assert parse_shape_literal(["M", "K"]).is_fully_explicit
        assert not parse_shape_literal(["...d", "M"]).is_fully_explicit

    def test_str(self):
        sh = parse_shape_literal(["...batch", "M", "K"])
        assert str(sh) == "[...batch, M, K]"
