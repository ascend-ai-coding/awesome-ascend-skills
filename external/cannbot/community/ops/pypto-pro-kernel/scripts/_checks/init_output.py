# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Rule: a bare `pl.make_tile` next to `pl.init_output` needs a drain.

| Rule | Evidence | What happens if missed |
| --- | --- | --- |
| `init-output-needs-a-drain` | Board measurement + generated C++ (tables below). | **Silently** outputs input data. |

## What `init_output` Hides

It is not a single instruction. Codegen in `backend_cce_block_out_ops.cpp:548` expands it as:

```cpp
Tile<Vec, T, 1, (256*32)/sizeof(T)> tmp(...);
TASSIGN(tmp, 0x0);                   // ← temporary tile is fixed at UB address 0x0
get_buf(PIPE_V, 0, 0);
TEXPANDS(tmp, value);                // ← before SetValidShape; fills the entire chunk (8 KB for b32)
rls_buf(PIPE_V, 0, 0);
get_buf(PIPE_MTE3, 0, 0);
for (...) { tmp.SetValidShape(1, chunk); TSTORE(gm, tmp); }
rls_buf(PIPE_MTE3, 0, 0);
// no barrier
```

`TSTORE` reads UB `[0, 8 KB)` on MTE3, and the immediately following `TLOAD` writes the same UB region on MTE2—a
cross-pipeline WAR race that **neither side reports**.

## Evidence 1: Board Runs in Groups of 12 After Widening the Window with `size=65536`

`size=65536` → 32 TSTORE iterations, 8 KB each; the immediately following `pl.load` lands in the same UB region:

| Kernel | Failures in 12 runs | Wrong values |
| --- | ---: | --- |
| `@pl.jit()` + bare `make_tile(addr=0x0)` | **6** | 128 or 384 values, **all elements from input `x`**. |
| Same + `pl.system.bar_all()` | 0 | — |
| Same + `pl.system.bar_mte3()` | **6** | Same. |
| **`auto_mutex=True` + bare `make_tile(addr=0x0)`** | **6** | Same, identical on each run. |
| `@pl.jit()` + `make_tile_group(addrs=0x0, mutex_ids=[0])` | 0 | — |
| `auto_mutex=True` + same group | 0 | — |
| `@pl.jit()` + bare `make_tile(addr=0x2000)` | 0 | — |

**The decisive factors are the address and tile declaration form, not call order**: all rows above use the legal order
required by the docstring (`init_output` before the tile declaration), yet still show 6/12 failures. The ineffective
`bar_mte3()` confirms that `bar_*` is a barrier **within** a pipeline (see #8 in
`references/diagnostics/validation-traps.md`).

## Evidence 2: Where `get_buf` Appears in Generated C++

Dumped `build/<kernel>__a5__d0/tk_none/kernel.cpp` from the same run:

| Kernel | Around the user's `TLOAD` |
| --- | --- |
| Bare `make_tile` + `@pl.jit()` | `... rls_buf(PIPE_MTE3,0,0) TLOAD` — **no `get_buf` at all**. |
| Bare `make_tile` + `auto_mutex=True` | **Exactly the same as above.** |
| `make_tile_group` + `@pl.jit()` | `... rls_buf(PIPE_MTE3,0,0) get_buf(PIPE_MTE2,0,0) TLOAD rls_buf(PIPE_MTE2,0,0)` |
| `make_tile_group` + `auto_mutex=True` | **Exactly the same as above.** |

`get_buf`/`rls_buf` are ownership tokens for buffers across pipelines; `init_output`'s `TSTORE` takes the same buffer ID
`(0, 0)`, so the two operations are serialized.

**The identical generated pairs reveal the mechanism: `make_tile_group` emits `get_buf`; `auto_mutex` has no effect
here** (each C++ pair is textually identical, and board results also match: 6/12 vs 6/12, 0/12 vs 0/12). I previously
attributed the safety to `auto_mutex`; that was wrong (see #19 in `references/verification/evidence.md`). Therefore,
this rule accepts only `bar_all()` and `make_tile_group` as waivers; **`auto_mutex=True` is not a waiver**.

## Criterion

Report only when all of these hold:

1. The function has `@pl.jit`, whether or not `auto_mutex` is enabled.
2. The function body contains `pl.init_output(...)`.
3. The function body contains a bare `pl.make_tile(...)`—**this tile is outside the buffer-ID protocol**. Follow one
layer into a helper in the same file (such as `def _tile(shape, addr)`).
4. The function body **does not** contain `pl.system.bar_all()`.

Item 3 separates this rule from `auto-mutex-needs-tile-group`: that rule detects “`auto_mutex` is declared but a bare
tile is used”; this one detects “`init_output` is used without a UB-side token.” **The same kernel can trigger both**,
because they describe different conditions and have different fixes (the former needs mutex IDs; this one needs a drain
or group).

## Scope Limits

- **Do not judge addresses.** `make_tile(addr=0x2000)` measured safe, but `addr` can be any expression, and the safety
boundary (`(256*32)/sizeof(T)` bytes) depends on tile dtype; it cannot be determined from the `make_tile` line alone.
**A kernel with a sufficiently distant address may be falsely reported**; waive it with a `反例：` declaration and include
the computed byte distance in the reason—it is more useful than “I know this is safe.”
- **Recognize only `bar_all`.** Measurement found `bar_mte3()` ineffective; other `bar_*` have not been tried.
Hand-written `sync_src`/`sync_dst` could theoretically block the race if the correct pipeline pair is chosen, but they
are **untested** and are not accepted as waivers. This rule prefers a false positive over silently passing the input
data.
- **Do not judge placement.** `bar_all()` before `init_output` also passes. Checking order requires control-flow
analysis, which belongs in the parser ([skill constraints](../../SKILL.md#constraints-and-workspace-policy)).
- **Do not judge the target Tensor of `init_output`.** The overwrite happens in UB and does not depend on `offset`,
`size`, or target identity—`TEXPANDS` runs before `SetValidShape`, so `size=128` still fills 8 KB.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .source import (
    Call,
    Finding,
    Source,
    apply_claim,
    calls,
    counterexample_claims,
    has_decorator,
    read,
)

RULE = "init-output-needs-a-drain"

INIT_OUTPUT = "init_output"
MAKE_TILE = "make_tile"
MAKE_TILE_GROUP = "make_tile_group"
BAR_ALL = "bar_all"


def check(path: Path) -> list[Finding]:
    source = read(path)
    findings: list[Finding] = []
    for function in _jit_kernels(source):
        found = _unprotected_init_output(source, function)
        findings += apply_claim(
            source, function, RULE, found, counterexample_claims(function)
        )
    return findings


def _jit_kernels(source: Source) -> list[ast.FunctionDef]:
    """Return functions decorated with `@pl.jit`, sorted by line number; both parenthesized and bare forms count.

    Include nested definitions: example kernels are often created inside factories (the address probe in
    `init_and_insert` does this), so looking only at module scope would miss them all.
    """
    found = [
        node for node in ast.walk(source.tree) if has_decorator(node, source.pl, "jit")
    ]
    return sorted(found, key=lambda fn: fn.lineno)


def _tile_makers(source: Source) -> set[str]:
    """Return helper names in this module that produce bare Tiles.

    Few examples write `pl.make_tile(pl.TileType(...), addr=...)` directly because the line is long; they wrap it in a
    helper such as `def _tile(shape, addr)`. **Recognizing only literal `pl.make_tile` calls would miss them all**,
    including the original `init_and_insert` case for this rule (all its Tiles go through `_tile()`). A rule that misses
    its motivating case is useless.

    Follow one helper layer up, within the same file only; cross-file lookup requires a call graph, a different class of
    tool. A function decorated with `@pl.jit` is not a helper because one kernel does not call another kernel.
    """
    makers = set()
    for node in ast.walk(source.tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if has_decorator(node, source.pl, "jit"):
            continue
        if any(call.name == MAKE_TILE for call in calls(node, source.pl)):
            makers.add(node.name)
    return makers


def _bare_tiles(source: Source, function: ast.FunctionDef) -> list[Call]:
    """Return bare Tiles in this kernel: direct `pl.make_tile` calls plus Tiles returned by helpers.

    Exclude `make_tile_group`: measurement shows that it emits `get_buf` and is the safe form.
    """
    direct = [call for call in calls(function, source.pl) if call.name == MAKE_TILE]
    makers = _tile_makers(source)
    indirect = [
        Call(node.lineno, node.col_offset, node.func.id, node)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in makers
    ]
    return sorted(direct + indirect, key=lambda call: (call.lineno, call.col))


def _unprotected_init_output(
    source: Source, function: ast.FunctionDef
) -> list[Finding]:
    inits = [call for call in calls(function, source.pl) if call.name == INIT_OUTPUT]
    bare = _bare_tiles(source, function)
    if not inits or not bare:
        return []
    system = None if source.pl is None else f"{source.pl}.system"
    if any(call.name == BAR_ALL for call in calls(function, system)):
        return []
    where = "、".join(
        f"第 {call.lineno} 行"
        + ("" if call.name == MAKE_TILE else f"（经 `{call.name}`）")
        for call in bare
    )
    return [
        Finding(
            source.path,
            call.lineno,
            RULE,
            f"`{source.pl}.{INIT_OUTPUT}` 的临时 tile 写死在 UB `0x0`、占 "
            f"`(256*32)/sizeof(T)` 字节（b32 是 8 KB），而它的 `TSTORE` 之后不发屏障；"
            f"而{where}的裸 `{source.pl}.{MAKE_TILE}` 生成的 cpp 里一个 `get_buf` 都没有，"
            f"撞上那一段时静默把输入数据写进 GM（实测 12 次错 6 次）。"
            f"加 `{source.pl}.system.{BAR_ALL}()`，或改用 `{MAKE_TILE_GROUP}`"
            f"（它会发 `get_buf`，实测 12 次 0 次）",
        )
        for call in inits
    ]
