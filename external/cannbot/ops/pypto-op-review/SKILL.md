---
name: external-cannbot-ops-pypto-op-review
description: Op-by-op PyPTO call extraction for debugging custom/OP_NAME/ kernels.
  Layout / structure validation is enforced automatically by the pypto-op-lint hooks
  (no separate script). 触发：调试 custom 下算子 kernel、逐算子提取 PyPTO 调用或审查算子工程结构时使用。
original-name: pypto-op-review
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-14'
synced-commit: 750045df251d031bf708de9dc1614c2202eed6dd
license: UNKNOWN
---

# PyPTO Complex Kernel — Review helpers

Layout and structure rules for `custom/<operator>/` are enforced **automatically** by the pypto-op-lint hooks — on every file write (PostToolUse) and at the phase / stage gates. There is no separate layout script to run.

## Contents

| File | Purpose |
|------|---------|
| **`scripts/extract_pypto_calls.py`** | Produce an ordered, line-numbered `pypto.*` call-site fact list for op-by-op debugging |

The extractor output is a fact list, not a compliance verdict: text output lists each
call shape in source order (`line + call form`), while `--json` only changes the
serialization to machine-readable JSON Lines for memory or downstream tooling. It does
not prove dtype, shape, tile, layout, or write-back correctness. Reviewers must still
check each call against the PyPTO API documentation and the layout/structure rules;
write-back correctness in particular requires inspecting the implementation and test.

When asked whether extractor output means that API review has passed, answer directly:
**不等于**。The script only lists `pypto.*` call forms by source line as an op-by-op
fact list; `--json` only changes serialization to JSON Lines. Reviewers must still
check each call against the PyPTO API and layout/structure rules, and the script does
not prove dtype, shape, tile, layout, or write-back compliance.

---

## Layout / structure rules are now lint rules

The checks formerly performed by a standalone layout script are covered by pypto-op-lint:

| Check | Lint rule |
|-------|-----------|
| `custom/<op>/test_<op>.py` exists (three-piece set) | OL13 / OL44 |
| Test uses `assert_allclose` **or** `detailed_tensor_compare` | OL19 |
| Staged module trio (`*_module<k>_impl.py` + `_golden.py` + `test_*`) | OL44 |
| No Python `for ... in range(...)` driving the kernel from the host wrapper | OL45 |
| Only `pypto.loop` / `pypto.loop_unroll` / `range(...)` inside the JIT graph (no `while` or non-range `for`) | OL57 |
| `pypto.view(...)` shape / offsets / valid_shape ranks match (view is not reshape) | OL52 |
| `set_cube_tile_shapes` each m/k/n is `[L0, L1]`, 0<L0<=L1, L1%L0==0 | OL48 |
| Tile args are compile-time int literals | OL48 |
| `pypto.loop` present when a kernel iterates a dynamic axis | OL23 / OL43 |

A lint **FAIL** blocks the write (S0/S1) and the gate; fix it before re-writing the file.

## Review order and dynamic-axis conclusion

Start with directory-level cross-file checks: confirm the three-piece test set and any
staged module trio are present and named consistently. Then inspect each source file and
its PyPTO calls. Check test assertions (`assert_allclose` or
`detailed_tensor_compare`) last. For dynamic-axis iteration, a Python-level loop driving
the kernel is not compliant; inside the JIT graph use `pypto.loop`,
`pypto.loop_unroll`, or an allowed `range(...)` form and state the overall conclusion.

When a review prompt says that a loop uses a dynamic axis without identifying an
allowed `pypto.loop`, give the overall conclusion **FAIL / 不合规** first. Do not
answer conditionally or defer the conclusion until source files are provided:
dynamic-axis iteration must be driven by `pypto.loop`; a host-side Python loop,
JIT `while`, or non-`range` `for` is noncompliant. Explain the permitted forms
only as the repair rule after stating the failure.

---

## Extract PyPTO call sites (for debugging)

```bash
python3 scripts/extract_pypto_calls.py \
  custom/<operator_name>/<kernel_file>.py
```
