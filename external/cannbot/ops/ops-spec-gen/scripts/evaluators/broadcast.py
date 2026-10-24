# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Broadcast simulator — 校验算子计算中的 broadcast 语义。

broadcast 描述的是算子计算是否需要对数据进行 broadcast（数据复制/扩展），
而非仅描述 shape 变换。给定 SymbolicShape，按 spec.yaml `broadcast.kind` 模拟。

Three modes:
  * 'numpy'    — 计算涉及数据广播（full numpy ellipsis broadcast；right-align,
                 dims either equal or one of them is 1 or missing）
  * 'none'     — 计算不涉及任何 broadcast（输入 shape 严格一致，无需数据扩展）
  * 'explicit' — 部分维度 broadcast + 部分维度不 broadcast
                 （apply per-axis policy from broadcast.rules, trailing/leading scope）

Only the 2-input case is fully simulated (sufficient for our examples).  N-input
broadcasts can be handled by chaining numpy_broadcast_two pairwise.
"""

from __future__ import annotations

from .types import Dim, SymbolicShape, BroadcastTrace, DslError


# ---------- numpy broadcast (used by add, complex, plus matmul batch dims) -


def _dims_broadcastable(a: Dim, b: Dim) -> tuple[bool, Dim]:
    """Return (compatible, resulting_dim).

    Rules:
      - const k == const k → const k
      - const 1, X        → X (X is broadcast over)
      - X, const 1        → X
      - symbol N, symbol N → symbol N
      - symbol N, const k → symbol N (assume symbol can take k; downstream may refine)
      - symbol N, symbol M (different names) → not directly compatible
    """
    if a.kind == "const" and b.kind == "const":
        if a.value == b.value:
            return True, a
        if a.value == 1:
            return True, b
        if b.value == 1:
            return True, a
        return False, a
    if a.kind == "const" and a.value == 1:
        return True, b
    if b.kind == "const" and b.value == 1:
        return True, a
    if a.kind == "symbol" and b.kind == "symbol":
        if a.name == b.name:
            return True, a
        # different symbol names — cannot prove compatibility statically
        return False, a
    if a.kind == "symbol" and b.kind == "const":
        return True, a
    if b.kind == "symbol" and a.kind == "const":
        return True, b
    return False, a


def numpy_broadcast_two(a: SymbolicShape, b: SymbolicShape) -> SymbolicShape:
    """Right-align two shapes per numpy ellipsis rule.

    If either side has a folded prefix of unknown length, the result also has a
    folded prefix (we cannot resolve it statically without per-case shape).
    """
    # Right-align explicit tails
    a_exp = a.explicit
    b_exp = b.explicit

    out_explicit_rev: list[Dim] = []
    n = max(len(a_exp), len(b_exp))
    for i in range(1, n + 1):
        ai = a_exp[-i] if i <= len(a_exp) else None
        bi = b_exp[-i] if i <= len(b_exp) else None
        if ai is None:
            out_explicit_rev.append(bi)  # type: ignore[arg-type]
            continue
        if bi is None:
            out_explicit_rev.append(ai)
            continue
        ok, dim = _dims_broadcastable(ai, bi)
        if not ok:
            raise DslError(
                "incompatible_dims",
                f"无法广播：{ai} ↔ {bi}（位置末尾第 {i} 维）",
            )
        out_explicit_rev.append(dim)
    out_explicit = list(reversed(out_explicit_rev))

    # Folded prefix:
    # - if both sides have folded → output also folded (cannot resolve without case shapes)
    # - if exactly one side has folded → output folded (the other side's missing
    #   prefix dims are absorbed)
    # - if neither side has folded → output fully explicit
    if a.folded_name is not None and b.folded_name is not None:
        out_folded = f"bcast_{a.folded_name}_{b.folded_name}"
    elif a.folded_name is not None:
        out_folded = a.folded_name
    elif b.folded_name is not None:
        out_folded = b.folded_name
    else:
        out_folded = None

    return SymbolicShape(folded_name=out_folded, explicit=out_explicit)


def numpy_broadcast_n(shapes: list[SymbolicShape]) -> SymbolicShape:
    if not shapes:
        raise DslError("dsl_eval_error", "broadcast 至少需要 1 个输入")
    if len(shapes) == 1:
        return shapes[0]
    out = shapes[0]
    for s in shapes[1:]:
        out = numpy_broadcast_two(out, s)
    return out


# ---------- explicit-rules broadcast (used by matmul) ----------------------


def _explicit_check_two(
    a: SymbolicShape,
    b: SymbolicShape,
    rules: list[dict],
) -> SymbolicShape:
    """Apply explicit rules to two shapes. Each rule is a dict with keys:
    `scope` (trailing/leading/axis), `count` (int; -1 = remainder), `policy`.

    For matmul-style rules:
        - {scope: trailing, count: 2, policy: no_broadcast} → end 2 dims must equal
        - {scope: leading,  count: -1, policy: numpy}      → leading dims numpy bcast

    Output shape: result of the rules; for the matmul case, leading numpy +
    trailing no_broadcast yields broadcast(batch_a, batch_b) ++ [M, N].
    """
    # Validate trailing rule covers correct count then split a/b/output similarly
    trailing_rule = next((r for r in rules if r.get("scope") == "trailing"), None)
    leading_rule = next((r for r in rules if r.get("scope") == "leading"), None)

    if trailing_rule is None or leading_rule is None:
        raise DslError(
            "explicit_rules_uncovered",
            "explicit broadcast 至少需要 'trailing' 与 'leading' 各一条规则",
        )

    t_count = int(trailing_rule.get("count", 0))
    if t_count <= 0 or t_count > a.rank_min or t_count > b.rank_min:
        raise DslError(
            "explicit_rules_uncovered",
            f"trailing.count={t_count} 与输入 rank ({a.rank_min}/{b.rank_min}) 不匹配",
        )

    a_trail = a.explicit[-t_count:]
    b_trail = b.explicit[-t_count:]
    a_lead = SymbolicShape(folded_name=a.folded_name, explicit=a.explicit[:-t_count])
    b_lead = SymbolicShape(folded_name=b.folded_name, explicit=b.explicit[:-t_count])

    # trailing policy
    t_policy = trailing_rule.get("policy")
    if t_policy == "no_broadcast":
        if len(a_trail) != len(b_trail):
            raise DslError("incompatible_dims",
                           f"trailing 维数不一致: {a_trail} vs {b_trail}")
        out_trail = list(a_trail)  # both must equal at runtime; symbolically use a's
    elif t_policy == "numpy":
        out_trail = numpy_broadcast_two(
            SymbolicShape(folded_name=None, explicit=a_trail),
            SymbolicShape(folded_name=None, explicit=b_trail),
        ).explicit
    else:
        raise DslError("explicit_rules_uncovered",
                       f"trailing.policy 未支持: {t_policy!r}")

    # leading policy
    l_policy = leading_rule.get("policy")
    if l_policy == "numpy":
        out_lead = numpy_broadcast_two(a_lead, b_lead)
    elif l_policy == "no_broadcast":
        # B 路线：symbolic symbol 名（如 matmul 的 K）仅为 owner 命名，不再用于跨 input
        # 判等；leading no_broadcast 只静态校核 rank 一致与 const 维相等，symbol 名是否
        # 配对放过到 stage 8 / 运行时（numpy 跑公式时不一致会抛错）。
        if len(a_lead.explicit) != len(b_lead.explicit):
            raise DslError("incompatible_dims",
                           f"leading 显式维 rank 不一致: {a_lead} vs {b_lead}")
        for i, (da, db) in enumerate(zip(a_lead.explicit, b_lead.explicit)):
            if da.kind == "const" and db.kind == "const" and da.value != db.value:
                raise DslError("incompatible_dims",
                               f"leading 第 {i} 维 const 不等: {da} vs {db}")
        # 折叠维只要"都是折叠 / 都不是折叠"即可
        a_has_folded = a_lead.folded_name is not None
        b_has_folded = b_lead.folded_name is not None
        if a_has_folded != b_has_folded:
            raise DslError("incompatible_dims",
                           f"leading 一侧有折叠维另一侧没有: {a_lead} vs {b_lead}")
        if a_has_folded and b_has_folded and a_lead.folded_name != b_lead.folded_name:
            out_folded = f"bcast_{a_lead.folded_name}_{b_lead.folded_name}"
        else:
            out_folded = a_lead.folded_name
        out_lead = SymbolicShape(folded_name=out_folded, explicit=list(a_lead.explicit))
    else:
        raise DslError("explicit_rules_uncovered",
                       f"leading.policy 未支持: {l_policy!r}")

    return SymbolicShape(
        folded_name=out_lead.folded_name,
        explicit=list(out_lead.explicit) + list(out_trail),
    )


# ---------- entry: simulate per spec.broadcast.kind ------------------------


def simulate(
    inputs: list[SymbolicShape],
    broadcast_spec: dict,
) -> BroadcastTrace:
    """Drive the right backend per `broadcast.kind`."""
    kind = (broadcast_spec or {}).get("kind", "none")
    rules = (broadcast_spec or {}).get("rules") or []

    if kind == "none":
        if len(inputs) <= 1:
            return BroadcastTrace(output_shape=inputs[0] if inputs else
                                  SymbolicShape(folded_name=None, explicit=[]))
        first = inputs[0]
        for i, s in enumerate(inputs[1:], 1):
            # B 路线：symbolic 折叠维名 / symbol 名仅为 owner 命名，不参与跨 input 判等。
            # 跨 input 关系由 outputs.shape_rule 表达；stage 5 在 kind=none 下只静态校核
            # rank 结构（折叠维是否同时存在 + explicit 长度）与 const 维数值是否一致，
            # 其余（symbol 名是否匹配）放过到 stage 8 跑公式时由 numpy 自身判定。
            a_has_folded = first.folded_name is not None
            b_has_folded = s.folded_name is not None
            if a_has_folded != b_has_folded or len(s.explicit) != len(first.explicit):
                raise DslError(
                    "numpy_violation",
                    f"broadcast.kind=none 但输入 rank 结构不一致: {first} vs {s}",
                )
            for j, (da, db) in enumerate(zip(first.explicit, s.explicit)):
                if da.kind == "const" and db.kind == "const" and da.value != db.value:
                    raise DslError(
                        "numpy_violation",
                        f"broadcast.kind=none 第 {j} 维 const 不等: {da} vs {db}",
                    )
        return BroadcastTrace(output_shape=first)

    if kind == "numpy":
        out = numpy_broadcast_n(inputs)
        return BroadcastTrace(output_shape=out)

    if kind == "explicit":
        if len(inputs) != 2:
            raise DslError(
                "explicit_rules_uncovered",
                f"explicit broadcast v1 仅支持 2 输入，得到 {len(inputs)}",
            )
        if not rules:
            raise DslError("explicit_rules_uncovered",
                           "explicit broadcast 缺 rules 字段")
        out = _explicit_check_two(inputs[0], inputs[1], rules)
        return BroadcastTrace(output_shape=out)

    raise DslError("dsl_parse_error", f"未知 broadcast.kind: {kind!r}")
