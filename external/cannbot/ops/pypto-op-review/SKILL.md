---
name: external-cannbot-ops-pypto-op-review
description: Op-by-op PyPTO call extraction for debugging custom/OP_NAME/ kernels.
  Layout / structure validation is enforced automatically by the pypto-op-lint hooks
  (no separate script). 触发：调试 custom 下算子 kernel、逐算子提取 PyPTO 调用或审查算子工程结构时使用。
original-name: pypto-op-review
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# PyPTO Complex Kernel — Review helpers

Layout and structure rules for `custom/<operator>/` are enforced **automatically** by the pypto-op-lint hooks — on every file write (PostToolUse) and at the phase / stage gates. There is no separate layout script to run.

## Contents

| File | Purpose |
|------|---------|
| **`scripts/extract_pypto_calls.py`** | List every `pypto.*` call site by line number — used for op-by-op debugging |

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

---

## Extract PyPTO call sites (for debugging)

```bash
python3 scripts/extract_pypto_calls.py \
  custom/<operator_name>/<kernel_file>.py
```
