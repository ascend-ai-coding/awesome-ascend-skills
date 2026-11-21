# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Data types for shape / dtype DSL.

Design:
  * `Dim` — a single dimension. Literal dimensions retain their three kinds;
    shape-rule arithmetic also carries an immutable expression:
      - 'expr': multiplication / floor division without invented constant values
      - 'const':  fixed integer (e.g. literal `2`, `4`)
      - 'symbol': named explicit dim (e.g. `M`, `K`, `N`); requires registration in
                  spec.yaml's shape_constraints.symbols
      - 'folded': named placeholder for "any number of dims" (e.g. `...batch_a`);
                  may appear at most once and only as the leading element

  * `SymbolicShape` — a shape with optional folded prefix + explicit tail.
      "...d"           → folded='d',     explicit=[]
      "...batch", M, K → folded='batch', explicit=[M, K]
      [M, K]           → folded=None,    explicit=[M, K]

  * `DslError` — raised by parser/solver; caught by stage_* and converted to Finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Dim:
    kind: str                # 'const' | 'symbol' | 'folded' | 'expr'
    name: Optional[str] = None
    value: Optional[int] = None
    expression: Optional[tuple[str, "Dim", int]] = None

    def __post_init__(self):
        if self.kind == "const":
            assert self.value is not None and self.name is None
        elif self.kind in ("symbol", "folded"):
            assert self.name is not None and self.value is None
        elif self.kind == "expr":
            assert self.name is None and self.value is None and self.expression is not None
            op, left, right = self.expression
            assert op in ("*", "//") and isinstance(left, Dim) and type(right) is int
            assert left.kind != "folded" and right >= (1 if op == "//" else 0)
        else:
            raise ValueError(f"Dim.kind must be const/symbol/folded/expr, got {self.kind!r}")

    def _integer_operation(self, op, other):
        if type(other) is not int:
            return NotImplemented
        if self.kind == "folded":
            raise DslError("folded_dim_misuse", "不能对未知数量的折叠维做整数运算")
        if other < (1 if op == "//" else 0):
            raise DslError("incompatible_dims", "维度乘数须非负，整除除数须为正整数")
        if self.kind == "const":
            value = self.value * other if op == "*" else self.value // other
            return Dim(kind="const", value=value)
        if other == 1:
            return self
        if op == "*" and other == 0:
            return Dim(kind="const", value=0)
        return Dim(kind="expr", expression=(op, self, other))

    def __mul__(self, other):
        return self._integer_operation("*", other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        return self._integer_operation("//", other)

    @property
    def symbols(self) -> frozenset[str]:
        if self.kind == "symbol":
            return frozenset([self.name])
        if self.kind == "expr":
            return self.expression[1].symbols
        return frozenset()

    def resolve(self, values: dict[str, int]) -> int:
        """Evaluate preserved arithmetic only after the caller binds symbols."""
        if self.kind == "const":
            return self.value
        if self.kind == "symbol":
            value = values[self.name]
            if type(value) is not int or value < 0:
                raise DslError("incompatible_dims", "符号维度须绑定非负整数")
            return value
        if self.kind == "expr":
            op, left, right = self.expression
            value = left.resolve(values)
            return value * right if op == "*" else value // right
        raise DslError("folded_dim_misuse", "未知数量的折叠维不能绑定成单一维度")

    def __str__(self) -> str:
        if self.kind == "const":
            return str(self.value)
        if self.kind == "expr":
            op, left, right = self.expression
            return f"({left} {op} {right})"
        if self.kind == "folded":
            return f"...{self.name}"
        return self.name  # type: ignore[return-value]


@dataclass
class SymbolicShape:
    """Shape with optional folded prefix + explicit-dim tail."""
    folded_name: Optional[str]   # None means fully explicit
    explicit: list[Dim]

    @classmethod
    def from_dims(cls, dims: list[Dim]) -> "SymbolicShape":
        """Build from a positional Dim list. Folded must be at index 0 if present."""
        if not dims:
            return cls(folded_name=None, explicit=[])
        if dims[0].kind == "folded":
            for d in dims[1:]:
                if d.kind == "folded":
                    raise DslError(
                        code="folded_dim_misuse",
                        message="折叠维 '...x' 在 symbolic 列表中至多出现一次",
                    )
            return cls(folded_name=dims[0].name, explicit=list(dims[1:]))
        # No folded prefix: assert no folded anywhere
        for d in dims:
            if d.kind == "folded":
                raise DslError(
                    code="folded_dim_misuse",
                    message="折叠维 '...x' 必须是 symbolic 列表的首元素",
                )
        return cls(folded_name=None, explicit=list(dims))

    @property
    def rank_min(self) -> int:
        """Lower bound on rank (folded contributes ≥ 0)."""
        return len(self.explicit)

    @property
    def is_fully_explicit(self) -> bool:
        return self.folded_name is None

    def __str__(self) -> str:
        parts: list[str] = []
        if self.folded_name is not None:
            parts.append(f"...{self.folded_name}")
        parts.extend(str(d) for d in self.explicit)
        return "[" + ", ".join(parts) + "]"


@dataclass
class BroadcastTrace:
    """Output of broadcast simulator: per-axis alignment record."""
    output_shape: SymbolicShape
    notes: list[str] = field(default_factory=list)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)



class DslError(Exception):
    """Raised by parser/solver/broadcast. Caught by stage_* and turned into Finding."""

    def __init__(self, code: str, message: str, field_path: str = ""):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.field_path = field_path
